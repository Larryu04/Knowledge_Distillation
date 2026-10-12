"""
judge_eval.py -- pairwise, reference-guided LLM-as-judge: v8 vs each opponent, gpt-4o-mini judge.

For every gold question and every opponent, the judge sees the question, the gold answer and
two anonymous answers ("A" / "B") and picks the better one or a tie. Each pair is judged
TWICE with A/B swapped. Per item, v8 scores +1 / 0 / -1 per order; the sum decides:
  > 0 -> v8 win,  < 0 -> v8 loss,  = 0 -> tie  (so a verdict that flips with position = tie).
An empty answer (e.g. a thinking model that ran out of budget) loses without a judge call.

Inputs : eval_results_judge/answers/<model>.jsonl   (from generate_answers.py)
Outputs: eval_results_judge/judgments/<opponent>.jsonl  raw verdicts, cached -> resumable
         eval_results_judge/judge_items.csv            one row per (opponent, item)
         eval_results_judge/judge_summary.csv          -> plot_metrics.py chart 3
         eval_results_judge/human_blind_20.csv (+ _key) your 20-question blind sheet (--make-blind)

  $env:OPENAI_API_KEY="sk-..."
  .venv\\Scripts\\python.exe judge_eval.py                      # every opponent with answers on disk
  .venv\\Scripts\\python.exe judge_eval.py --opponents base7b   # sanity check first
  .venv\\Scripts\\python.exe judge_eval.py --make-blind         # write the human grading sheet
  .venv\\Scripts\\python.exe judge_eval.py --agreement          # after you fill in Human_Winner
"""

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

GOLD_PATH = "hebrew_cs_pilot_dataset_92.json"
ROOT = "eval_results_judge"
ANSWERS_DIR = os.path.join(ROOT, "answers")
JUDGMENTS_DIR = os.path.join(ROOT, "judgments")
SUMMARY_CSV = os.path.join(ROOT, "judge_summary.csv")
ITEMS_CSV = os.path.join(ROOT, "judge_items.csv")
BLIND_CSV = os.path.join(ROOT, "human_blind_20.csv")
BLIND_KEY = os.path.join(ROOT, "human_blind_20_key.json")

JUDGE_MODEL = "gpt-4o-mini"   # project rule: the judge is always mini
WORKERS = 4                   # mini's TPM limit is far above what 4 threads use
SUBJECT = "v8"

# Display order = row order in the chart (closest peer first, frontier last).
OPPONENTS = {
    "base7b":   ("DictaLM-2.0, no fine-tune", "7B"),
    "dicta12b": ("DictaLM-3.0 Instruct", "12B"),
    "dicta24b": ("DictaLM-3.0 Thinking", "24B"),
    "gpt4o":    ("GPT-4o", "proprietary"),
}

JUDGE_SYSTEM = """You are a strict university teaching assistant grading answers to Hebrew computer-science exam questions.
You get the question, a GOLD reference answer written by the course staff, and two candidate answers, A and B.

Decide which candidate is the better answer FOR A STUDENT, judged in this order:
1. Correctness: no factual or technical errors, consistent with the gold answer. A wrong final answer is a serious flaw.
2. Completeness: covers the key points of the gold answer and actually answers what was asked.
3. Clarity: a clear step-by-step explanation in correct Hebrew. Answers that drift into another language are worse.

Ignore: length (longer is NOT better), formatting, markdown, and the exact wording of the final-answer line
(e.g. "תשובה:" vs "תשובה סופית:"). Do not prefer an answer because it resembles the gold answer's phrasing --
only its content matters. The order A/B is random and means nothing.

Reply with JSON only: {"reasoning": "<2-3 sentences, English>", "winner": "A" | "B" | "tie"}
Use "tie" only when the answers are genuinely equal in quality."""


def judge_user(q, gold, a, b):
    return (f"## Question\n{q}\n\n## Gold reference answer\n{gold}\n\n"
            f"## Answer A\n{a}\n\n## Answer B\n{b}")


# ---------------------------------------------------------------- io

def load_gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        r["gold"] = f"{r['rationale'].strip()}\nתשובה סופית: {r['output'].strip()}"
    return rows


def load_answers(model):
    path = os.path.join(ANSWERS_DIR, f"{model}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["idx"]] = rec["answer"]
    return out


def load_judgments(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    out[(rec["idx"], rec["order"])] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


_write_lock = threading.Lock()


def append(path, rec):
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------- judging

def call_judge(client, user_msg):
    from openai import APIConnectionError, APIStatusError, RateLimitError

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_msg}],
                temperature=0, max_tokens=300,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            winner = str(data.get("winner", "")).strip().upper()
            winner = "tie" if winner == "TIE" else winner
            if winner in ("A", "B", "tie"):
                return winner, data.get("reasoning", "")
            raise ValueError(f"bad winner {winner!r}")
        except (RateLimitError, APIConnectionError, json.JSONDecodeError, ValueError) as e:
            wait = min(5 * 2 ** attempt, 60)
            print(f"  [retry {attempt + 1}/6] {type(e).__name__}: {e}; sleeping {wait}s")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            time.sleep(min(5 * 2 ** attempt, 60))
    raise RuntimeError("judge failed 6 times")


def judge_opponent(client, opp, rows, subj, opp_ans):
    """Fill the judgment cache for one opponent; returns {(idx, order): rec}."""
    path = os.path.join(JUDGMENTS_DIR, f"{opp}.jsonl")
    cache = load_judgments(path)
    jobs = []
    for idx in sorted(set(subj) & set(opp_ans)):
        if not subj[idx] or not opp_ans[idx]:
            continue  # decided without the judge in score_item
        for order in ("v8_first", "opp_first"):
            if (idx, order) not in cache:
                jobs.append((idx, order))
    print(f"[{opp}] {len(cache)} cached verdicts, {len(jobs)} to request")

    def work(idx, order):
        r = rows[idx]
        a, b = (subj[idx], opp_ans[idx]) if order == "v8_first" else (opp_ans[idx], subj[idx])
        winner, why = call_judge(client, judge_user(r["instruction"], r["gold"], a, b))
        if winner == "tie":
            v8_pts = 0
        else:
            v8_won = (winner == "A") == (order == "v8_first")
            v8_pts = 1 if v8_won else -1
        rec = {"idx": idx, "order": order, "winner_pos": winner, "v8_points": v8_pts,
               "reasoning": why, "judge": JUDGE_MODEL}
        append(path, rec)
        return rec

    with ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(work, *j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            cache[(rec["idx"], rec["order"])] = rec
            if k % 20 == 0 or k == len(futs):
                print(f"[{opp}] {k}/{len(futs)}")
    return cache


def score_item(idx, subj, opp_ans, cache):
    """-> (outcome, flipped) with outcome in win/tie/loss from v8's side, or None if not judged."""
    s, o = subj.get(idx), opp_ans.get(idx)
    if s is None or o is None:
        return None
    if not s and not o:
        return "tie", False
    if not s:
        return "loss", False
    if not o:
        return "win", False
    v1, v2 = cache.get((idx, "v8_first")), cache.get((idx, "opp_first"))
    if not v1 or not v2:
        return None
    total = v1["v8_points"] + v2["v8_points"]
    flipped = v1["v8_points"] * v2["v8_points"] < 0
    return ("win" if total > 0 else "loss" if total < 0 else "tie"), flipped


# ---------------------------------------------------------------- human blind sheet

def make_blind(rows, subj, n=20, seed=7):
    """Stratified 20-item sheet vs the strongest opponent available, A/B randomized."""
    opp = next((o for o in reversed(list(OPPONENTS)) if load_answers(o)), None)
    if not opp:
        sys.exit("[error] no opponent answers yet")
    opp_ans = load_answers(opp)
    rng = random.Random(seed)
    pool = [i for i in sorted(set(subj) & set(opp_ans)) if subj[i] and opp_ans[i]]
    by_domain = {}
    for i in pool:
        by_domain.setdefault(rows[i]["domain"], []).append(i)
    domains = list(by_domain)
    rng.shuffle(domains)
    picks = [rng.choice(by_domain[d]) for d in domains[:n]]
    picks += rng.sample([i for i in pool if i not in picks], max(0, n - len(picks)))

    key = {}
    with open(BLIND_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sheet_ID", "Domain", "Question", "Gold_Answer", "Answer_A", "Answer_B",
                    "Human_Winner (A/B/tie)", "Notes"])
        for sid, idx in enumerate(sorted(picks), 1):
            v8_is_a = rng.random() < 0.5
            a, b = (subj[idx], opp_ans[idx]) if v8_is_a else (opp_ans[idx], subj[idx])
            w.writerow([sid, rows[idx]["domain"], rows[idx]["instruction"], rows[idx]["gold"], a, b, "", ""])
            key[sid] = {"idx": idx, "opponent": opp, "v8_is": "A" if v8_is_a else "B"}
    with open(BLIND_KEY, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)
    print(f"wrote {BLIND_CSV} (v8 vs {opp}, {len(picks)} items) -- don't open {BLIND_KEY} until graded")


def agreement(rows, subj):
    with open(BLIND_KEY, encoding="utf-8") as f:
        key = json.load(f)
    with open(BLIND_CSV, encoding="utf-8-sig", newline="") as f:
        graded = {r["Sheet_ID"]: r["Human_Winner (A/B/tie)"].strip() for r in csv.DictReader(f)}
    opp = next(iter(key.values()))["opponent"]
    cache = load_judgments(os.path.join(JUDGMENTS_DIR, f"{opp}.jsonl"))
    opp_ans = load_answers(opp)
    agree = n = 0
    for sid, k in key.items():
        h = graded.get(sid, "").upper()
        if h not in ("A", "B", "TIE"):
            continue
        human = "tie" if h == "TIE" else ("win" if h == k["v8_is"] else "loss")
        judged = score_item(k["idx"], subj, opp_ans, cache)
        if judged is None:
            continue
        n += 1
        agree += human == judged[0]
        if human != judged[0]:
            print(f"  disagree: sheet {sid} ({rows[k['idx']]['domain']}): you={human}, judge={judged[0]}")
    if n:
        print(f"human vs {JUDGE_MODEL} agreement: {agree}/{n} = {agree / n:.0%} (v8 vs {opp})")
    else:
        print("no graded rows found -- fill the Human_Winner column with A, B or tie")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponents", nargs="+", choices=OPPONENTS, help="default: all with answers on disk")
    ap.add_argument("--make-blind", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()

    rows = load_gold()
    subj = load_answers(SUBJECT)
    if not subj:
        sys.exit(f"[error] no {SUBJECT} answers: run generate_answers.py --model {SUBJECT}")
    if len(subj) < len(rows):
        print(f"[warn] v8 has only {len(subj)}/{len(rows)} answers")

    if args.make_blind:
        return make_blind(rows, subj)
    if args.agreement:
        return agreement(rows, subj)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[error] set OPENAI_API_KEY first")
    from openai import OpenAI

    client = OpenAI(timeout=120)
    os.makedirs(JUDGMENTS_DIR, exist_ok=True)

    wanted = args.opponents or list(OPPONENTS)
    summary, items = [], []
    for opp in OPPONENTS:  # always iterate in display order
        opp_ans = load_answers(opp)
        if not opp_ans:
            if opp in wanted:
                print(f"[{opp}] no answers on disk, skipped")
            continue
        if opp in wanted:
            cache = judge_opponent(client, opp, rows, subj, opp_ans)
        else:  # not re-judged this run, but keep its cached result in the summary
            cache = load_judgments(os.path.join(JUDGMENTS_DIR, f"{opp}.jsonl"))

        counts = {"win": 0, "tie": 0, "loss": 0}
        flips = 0
        for idx in range(len(rows)):
            res = score_item(idx, subj, opp_ans, cache)
            if res is None:
                continue
            outcome, flipped = res
            counts[outcome] += 1
            flips += flipped
            items.append({"Opponent": opp, "idx": idx, "Domain": rows[idx]["domain"],
                          "Outcome_for_v8": outcome, "Position_Flip": int(flipped),
                          "v8_chars": len(subj[idx]), "Opp_chars": len(opp_ans[idx])})
        n = sum(counts.values())
        if not n:
            continue
        name, params = OPPONENTS[opp]
        summary.append({"Opponent": name, "Params": params, "Wins": counts["win"],
                        "Ties": counts["tie"], "Losses": counts["loss"], "N": n,
                        "Win_Rate": round(counts["win"] / n, 3),
                        "Win_or_Tie_Rate": round((counts["win"] + counts["tie"]) / n, 3),
                        "Position_Flips": flips, "Judge": JUDGE_MODEL})
        print(f"[{opp}] v8 W/T/L = {counts['win']}/{counts['tie']}/{counts['loss']} "
              f"(n={n}, position flips={flips})")

    for path, data in ((SUMMARY_CSV, summary), (ITEMS_CSV, items)):
        if data:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
            print(f"wrote {path}")
    if summary:
        print("next: .venv\\Scripts\\python.exe plot_metrics.py")


if __name__ == "__main__":
    main()
