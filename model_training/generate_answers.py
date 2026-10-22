"""
generate_answers.py -- answer the 92 gold questions with every model in the judge benchmark.

One JSONL per model under eval_results_judge/answers/, appended after EVERY item and flushed,
so an interrupted run (crash, reboot, Ctrl-C) resumes where it stopped: already-answered
items are skipped.

Models (--model):
  v8         DictaLM-2.0-instruct 7B + v8 QLoRA adapter   (local, transformers, 4-bit)
  base7b     DictaLM-2.0-instruct 7B, no adapter          (local, transformers, 4-bit)
  dicta12b   DictaLM-3.0-Nemotron-12B-Instruct Q4_K_M     (local, llama-server, all on GPU)
  dicta24b   DictaLM-3.0-24B-Thinking Q4_K_M              (local, llama-server, GPU+RAM split)
  gpt4o      gpt-4o via the OpenAI API                    (throttled under the 30k TPM cap)

Every model gets the identical prompt: train.py's 23-subject PREAMBLE + the question, as one
user turn (same as inference.py), temperature 0.3 / top_p 0.9.

  .venv\\Scripts\\python.exe generate_answers.py --model v8
  .venv\\Scripts\\python.exe generate_answers.py --model base7b
  .venv\\Scripts\\python.exe generate_answers.py --model dicta12b --gguf models\\DictaLM-3.0-Nemotron-12B-Instruct-Q4_K_M.gguf
  .venv\\Scripts\\python.exe generate_answers.py --model dicta24b --gguf models\\DictaLM-3.0-24B-Thinking-Q4_K_M.gguf
  $env:OPENAI_API_KEY="sk-..."; .venv\\Scripts\\python.exe generate_answers.py --model gpt4o

Close games / other GPU apps before the local runs (a 2nd GPU workload segfaulted an eval
before). The llama-server models are started and stopped by this script.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque

sys.stdout.reconfigure(encoding="utf-8")

GOLD_PATH = "hebrew_cs_pilot_dataset_92.json"
ANSWERS_DIR = "eval_results_judge/answers"
LLAMA_SERVER = r"C:\Users\Larry\corecs\tools\llama.cpp-bin\llama-server.exe"
LLAMA_PORT = 8091
TEMPERATURE, TOP_P = 0.3, 0.9
ANSWER_MAX_TOKENS = 1024        # API / llama-server models (local HF models use inference.GEN_KWARGS)

# gpt-4o throttle: the org cap is 30k tokens/min, shared with anything else running.
GPT4O_MODEL = "gpt-4o"          # user-approved exception (2026-10-11), benchmark answers ONLY
TPM_BUDGET = 22_000             # stay well under 30k so a judge/other run can't tip it over
MIN_SLEEP_S = 2.0               # floor between gpt-4o calls even when the window is empty

MODELS = {
    "v8":       {"kind": "hf", "adapter": True},
    "base7b":   {"kind": "hf", "adapter": False},
    "dicta12b": {"kind": "llama", "ngl": 99, "ctx": 4096, "reasoning_budget": None},
    # 24B Q4_K_M = 14.3 GB > 12 GB VRAM: ~24 of 40 layers (~8.6 GB) on the GPU, the rest on
    # CPU/RAM; leaves room for the KV cache + desktop. Always-on thinking, capped so one item
    # can't run for an hour; thoughts are discarded.
    "dicta24b": {"kind": "llama", "ngl": 24, "ctx": 8192, "reasoning_budget": 2048},
    "gpt4o":    {"kind": "openai"},
}


# ---------------------------------------------------------------- shared

def load_gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


def user_prompt(preamble, question):
    return f"{preamble}\n\n{question.strip()}"


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass  # half-written last line from a crash: that item is redone
    return done


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


THINK_RE = re.compile(r"^.*</think>\s*", re.S)


def strip_thinking(text):
    """Keep only the final answer. An unclosed <think> means no answer was produced."""
    if "</think>" in text:
        return THINK_RE.sub("", text).strip()
    if "<think>" in text:
        return ""
    return text.strip()


# ---------------------------------------------------------------- local HF (v8 / base7b)

def run_hf(name, cfg, rows, todo, out_path, preamble):
    import torch
    from transformers import AutoModelForCausalLM

    import inference

    tok = inference.load_tokenizer()
    if cfg["adapter"]:
        model = inference.load_model(tok)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            inference.BASE_MODEL_ID,
            quantization_config=inference.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        model.eval()
        model.config.pad_token_id = tok.pad_token_id

    for idx in todo:
        torch.manual_seed(1234 + idx)  # sampling is on (inference.GEN_KWARGS): make it repeatable
        t0 = time.time()
        msgs = [{"role": "user", "content": user_prompt(preamble, rows[idx]["instruction"])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id, **inference.GEN_KWARGS)
        new = out[0][enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        hit_cap = len(new) >= inference.GEN_KWARGS["max_new_tokens"]
        yield idx, {"answer": text, "completion_tokens": int(len(new)),
                    "finish_reason": "length" if hit_cap else "stop",
                    "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- llama-server (dicta12b / dicta24b)

def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def start_llama_server(cfg, gguf, ngl):
    if not os.path.exists(gguf):
        sys.exit(f"[error] GGUF not found: {gguf}")
    cmd = [LLAMA_SERVER, "-m", gguf, "--port", str(LLAMA_PORT), "--host", "127.0.0.1",
           "-dev", "Vulkan0",           # the RTX; Vulkan1 is the AMD iGPU
           "-ngl", str(ngl), "-c", str(cfg["ctx"]), "-np", "1", "--jinja"]
    if cfg["reasoning_budget"] is not None:
        cmd += ["--reasoning-format", "deepseek", "--reasoning-budget", str(cfg["reasoning_budget"])]
    log_path = os.path.join(ANSWERS_DIR, "llama_server.log")
    log = open(log_path, "a", encoding="utf-8")
    print(f"[llama] starting: {' '.join(cmd)}\n[llama] log -> {log_path}")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):  # model load from disk can take a few minutes
        if proc.poll() is not None:
            sys.exit(f"[error] llama-server exited with code {proc.returncode}; see {log_path} "
                     "(VRAM full? lower --ngl)")
        try:
            if _get(f"http://127.0.0.1:{LLAMA_PORT}/health")[0] == 200:
                print("[llama] ready")
                return proc
        except OSError:
            pass
        time.sleep(1)
    proc.kill()
    sys.exit("[error] llama-server did not become ready within 10 minutes")


def run_llama(name, cfg, rows, todo, out_path, preamble, gguf, ngl):
    from openai import OpenAI

    proc = start_llama_server(cfg, gguf, ngl)
    client = OpenAI(base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1", api_key="local", timeout=3600)
    budget = cfg["reasoning_budget"] or 0
    try:
        for idx in todo:
            t0 = time.time()
            resp = client.chat.completions.create(
                model="local",
                messages=[{"role": "user", "content": user_prompt(preamble, rows[idx]["instruction"])}],
                temperature=TEMPERATURE, top_p=TOP_P, seed=1234 + idx,
                max_tokens=ANSWER_MAX_TOKENS + budget,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            yield idx, {"answer": strip_thinking(raw),
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                        "reasoning_chars": len(reasoning),
                        "finish_reason": choice.finish_reason,
                        "seconds": round(time.time() - t0, 1)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------- gpt-4o (OpenAI API)

class TokenWindow:
    """Sliding 60 s token count; block until the next call fits under the TPM budget."""

    def __init__(self, budget):
        self.budget = budget
        self.events = deque()  # (timestamp, tokens)

    def used(self):
        now = time.time()
        while self.events and now - self.events[0][0] > 60:
            self.events.popleft()
        return sum(t for _, t in self.events)

    def wait_for(self, estimate):
        while self.used() + estimate > self.budget and self.events:
            wait = 60 - (time.time() - self.events[0][0]) + 0.5
            print(f"  [tpm] {self.used()} tokens in last 60 s, sleeping {wait:.0f}s")
            time.sleep(max(wait, 1))

    def record(self, tokens):
        self.events.append((time.time(), tokens))


def run_openai(name, cfg, rows, todo, out_path, preamble):
    from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[error] set OPENAI_API_KEY first")
    client = OpenAI(timeout=120)
    window = TokenWindow(TPM_BUDGET)

    for idx in todo:
        content = user_prompt(preamble, rows[idx]["instruction"])
        # Conservative: Hebrew is ~2 chars/token or worse; reserve the full completion budget.
        window.wait_for(len(content) // 2 + ANSWER_MAX_TOKENS)
        for attempt in range(6):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=GPT4O_MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=TEMPERATURE, top_p=TOP_P, seed=1234 + idx,
                    max_tokens=ANSWER_MAX_TOKENS,
                )
                break
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code < 500 and e.status_code != 429:
                    raise
                wait = min(10 * 2 ** attempt, 120)
                print(f"  [retry {attempt + 1}/6] {type(e).__name__}: sleeping {wait}s")
                window.record(ANSWER_MAX_TOKENS)  # assume the failed call cost something
                time.sleep(wait)
        else:
            sys.exit(f"[error] gpt-4o failed 6 times on item {idx}; rerun to resume")

        window.record(resp.usage.total_tokens)
        choice = resp.choices[0]
        yield idx, {"answer": (choice.message.content or "").strip(),
                    "completion_tokens": resp.usage.completion_tokens,
                    "finish_reason": choice.finish_reason,
                    "seconds": round(time.time() - t0, 1)}
        time.sleep(MIN_SLEEP_S)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--gguf", help="GGUF path (dicta12b / dicta24b)")
    ap.add_argument("--ngl", type=int, help="override GPU layer count for llama-server models")
    ap.add_argument("--limit", type=int, help="only the first N items (smoke test)")
    args = ap.parse_args()

    from inference import PREAMBLE  # single source of truth for the prompt

    cfg = MODELS[args.model]
    rows = load_gold()
    os.makedirs(ANSWERS_DIR, exist_ok=True)
    out_path = os.path.join(ANSWERS_DIR, f"{args.model}.jsonl")
    done = load_done(out_path)
    n = min(args.limit or len(rows), len(rows))
    todo = [i for i in range(n) if i not in done]
    print(f"[{args.model}] {len(done)} already done, {len(todo)} to go -> {out_path}")
    if not todo:
        return

    if cfg["kind"] == "hf":
        gen = run_hf(args.model, cfg, rows, todo, out_path, PREAMBLE)
    elif cfg["kind"] == "llama":
        if not args.gguf:
            sys.exit("[error] --gguf is required for llama-server models")
        gen = run_llama(args.model, cfg, rows, todo, out_path, PREAMBLE, args.gguf, args.ngl or cfg["ngl"])
    else:
        gen = run_openai(args.model, cfg, rows, todo, out_path, PREAMBLE)

    t_start = time.time()
    for k, (idx, res) in enumerate(gen, 1):
        rec = {"idx": idx, "model": args.model, "domain": rows[idx]["domain"], **res}
        append(out_path, rec)
        eta = (time.time() - t_start) / k * (len(todo) - k) / 60
        flag = "  EMPTY" if not res["answer"] else ""
        print(f"[{args.model}] {k}/{len(todo)} item {idx} ({rec['domain']}) "
              f"{res['seconds']}s, {res['finish_reason']}{flag}  ETA {eta:.0f} min")

    empties = sum(1 for line in open(out_path, encoding="utf-8")
                  if line.strip() and not json.loads(line)["answer"])
    if empties:
        print(f"[warn] {empties} item(s) have an empty answer (thinking ran past the budget?). "
              "The judge scores an empty answer as a loss for that model.")


if __name__ == "__main__":
    main()
