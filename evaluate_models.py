"""
evaluate_models.py — objective comparison of the fine-tuned QLoRA adapter against the
raw base model (dicta-il/dictalm2.0-instruct) on the 24-item Hebrew CS gold set.

Pipeline
--------
1. Load hebrew_cs_pilot_dataset_24.json (lowercase course_name/domain/instruction/
   rationale/output schema).
2. Quantitative — masked teacher-forcing cross-entropy loss over ("rationale" + "output"),
   with the prompt ("PREAMBLE" + "instruction") masked to -100. Tokenization / masking is
   byte-for-byte the `make_example_builder` closure from train.py.
3. Qualitative — generate a fresh answer for each instruction from both models, shuffle the
   two answers per row into Answer_A / Answer_B, and export (all under eval_results_v6/):
       blind_evaluation.csv   (for human grading; Score_A / Score_B left blank)
       answer_key.csv         (secret: which model is A / B per row)
4. Report — per-domain average loss (domain read directly from each row's "domain" key)
   + overall average, printed and written to eval_results_v6/loss_results.csv.

Memory: the base model is fully released (del + gc + torch.cuda.empty_cache) before the
fine-tuned model is loaded, so a 12 GB card only ever holds one 7B/4-bit model at a time.

Run:  python evaluate_models.py
"""

import argparse
import csv
import gc
import json
import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

# ------------------------------------------------------------------------------------------
# Config — kept identical to train.py / inference.py where it matters for a fair comparison.
# ------------------------------------------------------------------------------------------
BASE_MODEL_ID = "dicta-il/dictalm2.0-instruct"
ADAPTER_DIR = "out/dictalm2-tutor-qlora-v6/adapter"
EVAL_PATH = "hebrew_cs_pilot_dataset_24.json"   # syllabus-aligned 24-item gold set, lowercase 5-key schema
OUT_DIR = "eval_results_v6/"

SEED = 42
MAX_SEQ_LEN = 1024

# MUST match train.py exactly (this is what the prompt-mask boundary depends on).
PREAMBLE = (
    "אתה מתרגל אקדמי מומחה בתחומי רשתות מחשבים, אבטחת סייבר, מערכות הפעלה, ושפות תכנות C++, Java וPython. "
    "ענה בעברית בלבד. תחילה הסבר את ההיגיון שלב אחר שלב, ולאחר מכן כתוב שורה "
    "שמתחילה ב'תשובה סופית:' ובה תשובה קצרה ותמציתית."
)
FINAL_ANSWER_PREFIX = "תשובה סופית:"

# Generation config — identical to inference.py GEN_KWARGS.
GEN_KWARGS = dict(
    max_new_tokens=512,
    min_new_tokens=8,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
)

DOMAINS = [
    "Computer Networks",
    "Cybersecurity",
    "Operating Systems",
    "C++",
    "Java",
    "Python",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ------------------------------------------------------------------------------------------
# Tokenizer — same PAD handling as train.py (PAD = <unk>, never EOS).
# ------------------------------------------------------------------------------------------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    assert tok.pad_token_id != tok.eos_token_id, "PAD must differ from EOS"
    tok.model_max_length = MAX_SEQ_LEN
    return tok


# ------------------------------------------------------------------------------------------
# Example construction — verbatim port of train.py::make_example_builder.
# Prompt tokens -> -100, loss only over ("Rationale" + "Output" + EOS).
# ------------------------------------------------------------------------------------------
def make_example_builder(tok):
    eos = tok.eos_token

    def build(instruction: str, rationale: str, output: str):
        user = f"{PREAMBLE}\n\n{instruction.strip()}"
        assistant = f"{rationale.strip()}\n\n{FINAL_ANSWER_PREFIX} {output.strip()}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

        prompt_text = tok.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        ).rstrip()
        if not full_text.endswith(eos):
            full_text += eos

        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        n = min(len(prompt_ids), len(full_ids))
        while n > 0 and full_ids[:n] != prompt_ids[:n]:
            n -= 1

        labels = [-100] * n + full_ids[n:]
        keep = any(t != -100 for t in labels) and eos_id_in(labels, tok.eos_token_id)
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "_len": len(full_ids),
            "_keep": keep,
        }

    return build


def eos_id_in(labels, eos_id):
    return any(t == eos_id for t in labels)


# ------------------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------------------
def load_eval_rows():
    with open(EVAL_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    out = []
    for i, r in enumerate(rows):
        ins = r.get("instruction", "").strip()
        rat = r.get("rationale", "").strip()
        ans = r.get("output", "").strip()
        dom = r.get("domain", "").strip()
        assert ins and rat and ans and dom, f"row {i} missing a field"
        assert dom in DOMAINS, f"row {i} has unknown domain {dom!r}"
        out.append(
            {
                "qid": f"Q{i + 1:02d}",
                "instruction": ins,
                "rationale": rat,
                "output": ans,
                "domain": dom,
                "gold_reference": f"{rat}\n\n{FINAL_ANSWER_PREFIX} {ans}",
            }
        )

    print(f"[data] loaded {len(out)} eval rows from {EVAL_PATH}")
    counts = {d: 0 for d in DOMAINS}
    for r in out:
        counts[r["domain"]] += 1
    print("[data] domain distribution:")
    for d in DOMAINS:
        if counts[d]:
            print(f"        {d:<20} {counts[d]}")
    return out


# ------------------------------------------------------------------------------------------
# Model builders
# ------------------------------------------------------------------------------------------
def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_base_model(tok):
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if DEVICE == "cuda":
        kwargs.update(quantization_config=_bnb_config(), device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    if DEVICE != "cuda":
        model.to(DEVICE)
    return model


def build_finetuned_model(tok, adapter_dir):
    from peft import PeftModel

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"adapter dir not found: {adapter_dir} (train.py must have run and saved it)"
        )
    base = build_base_model(tok)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    return model


def free_model(model):
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ------------------------------------------------------------------------------------------
# Quantitative — per-example masked teacher-forcing loss (batch size 1 == exact per-row CE).
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def compute_losses(model, tok, rows, tag):
    build = make_example_builder(tok)
    losses = []
    for r in rows:
        ex = build(r["instruction"], r["rationale"], r["output"])
        assert ex["_keep"], f"{r['qid']}: example failed the train.py keep-check"
        if ex["_len"] > MAX_SEQ_LEN:
            print(f"[warn] {r['qid']} exceeds MAX_SEQ_LEN ({ex['_len']}), skipping loss")
            losses.append(float("nan"))
            continue

        input_ids = torch.tensor([ex["input_ids"]], device=model.device)
        attention_mask = torch.tensor([ex["attention_mask"]], device=model.device)
        labels = torch.tensor([ex["labels"]], device=model.device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(out.loss.item())  # mean CE over this row's unmasked target tokens
        losses.append(loss)
        n_target = int((labels != -100).sum().item())
        print(f"[loss:{tag}] {r['qid']} ({r['domain']:<17}) loss={loss:.4f}  target_tokens={n_target}")
    return losses


# ------------------------------------------------------------------------------------------
# Qualitative — one generation per instruction (batch size 1).
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def generate_answers(model, tok, rows, tag):
    answers = []
    for r in rows:
        msgs = [{"role": "user", "content": f"{PREAMBLE}\n\n{r['instruction'].strip()}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            **GEN_KWARGS,
        )
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answers.append(text)
        preview = text.replace("\n", " ")[:90]
        print(f"[gen:{tag}] {r['qid']}  {preview}...")
    return answers


# ------------------------------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------------------------------
def write_blind_csv(rows, base_gens, ft_gens, blind_path, key_path, rng):
    blind_cols = [
        "Question_ID", "Domain", "Instruction", "Gold_Reference",
        "Answer_A", "Answer_B", "Score_A", "Score_B",
    ]
    key_cols = ["Question_ID", "Answer_A_Model", "Answer_B_Model"]

    with open(blind_path, "w", newline="", encoding="utf-8-sig") as bf, \
         open(key_path, "w", newline="", encoding="utf-8-sig") as kf:
        bw = csv.DictWriter(bf, fieldnames=blind_cols)
        kw = csv.DictWriter(kf, fieldnames=key_cols)
        bw.writeheader()
        kw.writeheader()

        for r, b_gen, f_gen in zip(rows, base_gens, ft_gens):
            # Independent coin flip per row -> fully shuffled assignment.
            if rng.random() < 0.5:
                answer_a, answer_b = b_gen, f_gen
                model_a, model_b = "BASE", "FINE_TUNED"
            else:
                answer_a, answer_b = f_gen, b_gen
                model_a, model_b = "FINE_TUNED", "BASE"

            bw.writerow({
                "Question_ID": r["qid"],
                "Domain": r["domain"],
                "Instruction": r["instruction"],
                "Gold_Reference": r["gold_reference"],
                "Answer_A": answer_a,
                "Answer_B": answer_b,
                "Score_A": "",
                "Score_B": "",
            })
            kw.writerow({
                "Question_ID": r["qid"],
                "Answer_A_Model": model_a,
                "Answer_B_Model": model_b,
            })

    print(f"[csv] wrote {blind_path}  (blind, for human grading)")
    print(f"[csv] wrote {key_path}  (secret answer key — open only after grading)")


def _avg(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def loss_report(rows, base_losses, ft_losses, out_path):
    by_domain = {d: {"base": [], "ft": []} for d in DOMAINS}
    for r, bl, fl in zip(rows, base_losses, ft_losses):
        bucket = by_domain.get(r["domain"])
        if bucket is None:  # Uncategorized -> still counts toward overall, not a domain row
            continue
        bucket["base"].append(bl)
        bucket["ft"].append(fl)

    header = f"{'Domain':<20} {'N':>3}  {'Base Loss':>11}  {'FT Loss':>11}  {'Delta (B-F)':>12}  {'Base PPL':>10}  {'FT PPL':>10}"
    sep = "-" * len(header)
    lines = ["", "=" * len(header), "  CATEGORICAL MASKED TEACHER-FORCING LOSS  (loss over target only)", "=" * len(header), header, sep]

    csv_rows = []
    for d in DOMAINS:
        b = _avg(by_domain[d]["base"])
        f = _avg(by_domain[d]["ft"])
        n = len(by_domain[d]["base"])
        b_ppl = float(np.exp(b)) if b == b else float("nan")
        f_ppl = float(np.exp(f)) if f == f else float("nan")
        lines.append(f"{d:<20} {n:>3}  {b:>11.4f}  {f:>11.4f}  {b - f:>+12.4f}  {b_ppl:>10.2f}  {f_ppl:>10.2f}")
        csv_rows.append({
            "Domain": d, "N": n,
            "Base_Avg_Loss": round(b, 6), "FineTuned_Avg_Loss": round(f, 6),
            "Loss_Delta_Base_minus_FT": round(b - f, 6),
            "Base_Perplexity": round(b_ppl, 4), "FineTuned_Perplexity": round(f_ppl, 4),
        })

    lines.append(sep)
    ob = _avg(base_losses)
    of = _avg(ft_losses)
    ob_ppl = float(np.exp(ob)) if ob == ob else float("nan")
    of_ppl = float(np.exp(of)) if of == of else float("nan")
    lines.append(f"{'OVERALL (all 24)':<20} {len(rows):>3}  {ob:>11.4f}  {of:>11.4f}  {ob - of:>+12.4f}  {ob_ppl:>10.2f}  {of_ppl:>10.2f}")
    lines.append("=" * len(header))
    lines.append("")
    lines.append("Delta > 0  => fine-tuned model assigns higher likelihood to the gold target (better).")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    csv_rows.append({
        "Domain": "OVERALL", "N": len(rows),
        "Base_Avg_Loss": round(ob, 6), "FineTuned_Avg_Loss": round(of, 6),
        "Loss_Delta_Base_minus_FT": round(ob - of, 6),
        "Base_Perplexity": round(ob_ppl, 4), "FineTuned_Perplexity": round(of_ppl, 4),
    })
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[csv] wrote {out_path}")


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
def main():
    global EVAL_PATH, ADAPTER_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", default=ADAPTER_DIR)
    ap.add_argument("--eval-path", default=EVAL_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--skip-generation", action="store_true",
                    help="only compute the loss report (no blind CSVs)")
    ap.add_argument("--blind-csv", default="blind_evaluation.csv")
    ap.add_argument("--key-csv", default="answer_key.csv")
    ap.add_argument("--loss-csv", default="loss_results.csv")
    ap.add_argument("--raw-json", default="eval_raw_results.json")
    args = ap.parse_args()

    EVAL_PATH = args.eval_path
    ADAPTER_DIR = args.adapter_dir

    os.makedirs(args.out_dir, exist_ok=True)
    blind_csv_path = os.path.join(args.out_dir, args.blind_csv)
    key_csv_path = os.path.join(args.out_dir, args.key_csv)
    loss_csv_path = os.path.join(args.out_dir, args.loss_csv)
    raw_json_path = os.path.join(args.out_dir, args.raw_json)

    seed_everything(SEED)
    print(f"[env] device={DEVICE}  torch={torch.__version__}  base={BASE_MODEL_ID}")
    if DEVICE != "cuda":
        print("[env] WARNING: no CUDA device — 4-bit quantization is skipped, this will be slow / RAM-heavy.")

    tok = build_tokenizer()
    rows = load_eval_rows()

    # ---- Base model -------------------------------------------------------------------
    print("\n" + "#" * 90 + "\n# BASE MODEL\n" + "#" * 90)
    base_model = build_base_model(tok)
    base_losses = compute_losses(base_model, tok, rows, tag="base")
    base_gens = [] if args.skip_generation else generate_answers(base_model, tok, rows, tag="base")
    free_model(base_model)
    print("[mem] base model released")
    if torch.cuda.is_available():
        print(f"[mem] cuda allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---- Fine-tuned model -------------------------------------------------------------
    print("\n" + "#" * 90 + "\n# FINE-TUNED MODEL (base + QLoRA adapter)\n" + "#" * 90)
    ft_model = build_finetuned_model(tok, ADAPTER_DIR)
    ft_losses = compute_losses(ft_model, tok, rows, tag="ft")
    ft_gens = [] if args.skip_generation else generate_answers(ft_model, tok, rows, tag="ft")
    free_model(ft_model)
    print("[mem] fine-tuned model released")

    # ---- Reports --------------------------------------------------------------------
    if not args.skip_generation:
        rng = random.Random(SEED)  # deterministic blind assignment
        write_blind_csv(rows, base_gens, ft_gens, blind_csv_path, key_csv_path, rng)

    loss_report(rows, base_losses, ft_losses, loss_csv_path)

    # ---- Raw dump for later inspection ---------------------------------------------
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "qid": r["qid"], "domain": r["domain"], "instruction": r["instruction"],
                    "base_loss": base_losses[i], "ft_loss": ft_losses[i],
                    "base_generation": base_gens[i] if base_gens else None,
                    "ft_generation": ft_gens[i] if ft_gens else None,
                }
                for i, r in enumerate(rows)
            ],
            f, ensure_ascii=False, indent=2,
        )
    print(f"[json] wrote {raw_json_path}")
    print("\n[done]")


if __name__ == "__main__":
    main()
