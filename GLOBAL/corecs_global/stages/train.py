"""
Stage 3 — QLoRA fine-tuning (runs under GLOBAL_TRAIN_PYTHON: torch + CUDA + bitsandbytes).

Ported from the original single-program trainer, keeping the choices that kept a small
non-English fine-tune from collapsing or looping:
  * 4-bit NF4 QLoRA + gradient checkpointing      -> a 7B model fits a 12 GB card.
  * embed_tokens / lm_head FROZEN (asserted)       -> the base vocabulary stays intact.
  * prompt tokens masked to -100                   -> loss only on the tutor's answer.
  * explicit EOS in every label, PAD != EOS        -> no never-ending generations.
  * stratified per-course validation split, early stopping on the MACRO per-course loss
    -> a pooled loss can't hide one course regressing while the big ones improve.
  * inverse-sqrt course oversampling when course sizes differ.
  * VRAM cap + a worst-case-batch preflight        -> a clean OOM in seconds, not a driver
    crash hours in; on OOM the preflight retries with micro-batch 1 automatically.
  * resume from the newest COMPLETE checkpoint after a crash / worker restart.
"""

import glob
import json
import math
import os
import random
import re
import sys
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from ..jobs import DEMO_TRAIN_FRACTION
from ..languages import build_preamble, preset
from .base import StageContext, StageError, atomic_write_json, read_json

SEED = 42
VAL_FRACTION = 0.05
MIN_VAL_PER_COURSE = 20
EFFECTIVE_BATCH = 16
DEFAULT_MICRO_BATCH = 2
VRAM_FRACTION = float(os.environ.get("GLOBAL_VRAM_FRACTION", 0.85))
LORA_DROPOUT = 0.05
WARMUP_FRAC = 0.03
MAX_GRAD_NORM = 0.3
NEFTUNE_ALPHA = 5
EARLY_STOPPING_PATIENCE = 5
CANDIDATE_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Generation defaults written into the model profile and used for the post-training sample.
GEN_KWARGS = dict(max_new_tokens=1024, temperature=0.3, top_p=0.9, repetition_penalty=1.15)


# --------------------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------------------
def configure_cuda(ctx: StageContext) -> None:
    if not torch.cuda.is_available():
        raise StageError("No CUDA GPU is available to the training worker. Fine-tuning needs an "
                         "NVIDIA GPU (12 GB+ recommended for 7B models).")
    # Less fragmentation with variable-length batches. (expandable_segments is unsupported on Windows.)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    props = torch.cuda.get_device_properties(0)
    ctx.log(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB (capped at {VRAM_FRACTION:.0%})")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
def build_tokenizer(model_id: str, max_len: int):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not tok.chat_template:
        raise StageError(f"{model_id} has no chat template. Pick an *-Instruct / chat model.")
    # PAD must differ from EOS: if PAD == EOS, label masking can teach the model that EOS is
    # "just padding" -> generations that never stop. Prefer an existing pad, then <unk>, then
    # any other special token; never add a token (that would need a trainable embedding).
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        candidates = [tok.unk_token] if tok.unk_token else []
        candidates += [t for t in tok.all_special_tokens if t not in (tok.eos_token, tok.bos_token)]
        candidates += sorted(t for t in tok.get_vocab() if re.fullmatch(r"<\|?(reserved|pad|fim_pad)[^>]*\|?>", t))
        for cand in candidates:
            tid = tok.convert_tokens_to_ids(cand)
            if tid is not None and tid != tok.eos_token_id:
                tok.pad_token = cand
                break
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        raise StageError(f"Could not find a padding token distinct from EOS for {model_id}.")
    tok.padding_side = "right"
    tok.model_max_length = max_len
    return tok


# --------------------------------------------------------------------------------------
# Examples: rendered with the model's OWN chat template, prompt masked
# --------------------------------------------------------------------------------------
def make_example_builder(tok, preamble: str, final_prefix: str):
    eos = tok.eos_token

    def build(instruction: str, rationale: str, output: str):
        # Preamble goes in the USER turn, not a system message: several chat templates
        # (Mistral family) reject the system role, and the runtime folds it in the same way.
        user = f"{preamble}\n\n{instruction.strip()}"
        assistant = f"{rationale.strip()}\n\n{final_prefix} {output.strip()}"
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        prompt_text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        # rstrip(): some templates append whitespace after EOS; the sequence must end at EOS.
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False).rstrip()
        if not full_text.endswith(eos):
            full_text += eos
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        # Longest shared prefix (guards against a token straddling the prompt/answer boundary).
        n = min(len(prompt_ids), len(full_ids))
        while n > 0 and full_ids[:n] != prompt_ids[:n]:
            n -= 1
        labels = [-100] * n + full_ids[n:]
        keep = any(t != -100 for t in labels) and tok.eos_token_id in labels
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels,
                "_len": len(full_ids), "_keep": keep}

    return build


def load_dataset(ctx: StageContext, tok, preamble: str, final_prefix: str, max_len: int):
    rows = read_json(ctx.ws.merged, default=[])
    if not rows:
        raise StageError("The merged dataset is empty; the generation stage must run first.")
    build = make_example_builder(tok, preamble, final_prefix)
    examples, domains, questions, too_long, bad = [], [], [], 0, 0
    for r in rows:
        ins, rat, out, dom = (r.get(k, "") for k in ("instruction", "rationale", "output", "domain"))
        if not (ins.strip() and rat.strip() and out.strip() and dom.strip()):
            bad += 1
            continue
        ex = build(ins, rat, out)
        if not ex["_keep"]:
            bad += 1
            continue
        if ex["_len"] > max_len:
            too_long += 1
            continue
        examples.append({k: v for k, v in ex.items() if not k.startswith("_")})
        domains.append(dom)
        questions.append(ins.strip())
    if len(examples) < 20:
        raise StageError(f"Only {len(examples)} usable training rows; need at least 20.")
    lens = sorted(len(e["input_ids"]) for e in examples)
    ctx.log(f"Rows kept {len(examples)}, too long {too_long}, malformed {bad}; tokens "
            f"p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}")
    if too_long > 0.05 * len(rows):
        ctx.log(f"Warning: {too_long} rows exceed max_seq_len={max_len}; consider raising it.")

    rng = random.Random(SEED)
    by_domain: dict[str, list[int]] = {}
    for i, d in enumerate(domains):
        by_domain.setdefault(d, []).append(i)
    train_idx, val_idx = [], []
    for d, idxs in sorted(by_domain.items(), key=lambda kv: -len(kv[1])):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(min(MIN_VAL_PER_COURSE, len(idxs) // 5), round(len(idxs) * VAL_FRACTION))
        n_val = max(1, min(n_val, len(idxs) // 2))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
        ctx.log(f"  {d}: train={len(idxs) - n_val} val={n_val}")
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (Dataset.from_list([examples[i] for i in train_idx]),
            Dataset.from_list([examples[i] for i in val_idx]),
            [domains[i] for i in train_idx], [domains[i] for i in val_idx],
            [questions[i] for i in val_idx], lens[-1])


def domain_sample_weights(domains):
    """Soft inverse-sqrt-frequency weights: small courses get ~2-4x more draws, not the
    10x+ that plain inverse frequency would give (which overfits the smallest course)."""
    counts = Counter(domains)
    top = max(counts.values())
    w = {d: (top / c) ** 0.5 for d, c in counts.items()}
    return torch.tensor([w[d] for d in domains], dtype=torch.double)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(model_id: str, lora_r: int):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [m for m in CANDIDATE_TARGET_MODULES if m in present] or "all-linear"
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=LORA_DROPOUT, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    for name, p in model.named_parameters():
        if p.requires_grad and ("embed_tokens" in name or "lm_head" in name):
            raise RuntimeError(f"Vocabulary parameter is trainable: {name}")
    return model


class WeightedDomainTrainer(Trainer):
    def __init__(self, *args, domain_sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_sample_weights = domain_sample_weights

    def _get_train_sampler(self, train_dataset=None):
        if self.domain_sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        # num_samples = sum(weights), not len(weights): majority rows still get ~1 draw/epoch
        # and minority rows get extra draws on top, instead of renormalizing the budget down.
        n = int(round(self.domain_sample_weights.sum().item()))
        return WeightedRandomSampler(self.domain_sample_weights, num_samples=n, replacement=True)


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to a per-example masked loss right away, so eval memory stays flat."""
    if isinstance(logits, tuple):
        logits = logits[0]
    shift_logits, shift_labels = logits[:, :-1, :].float(), labels[:, 1:]
    token_loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
    mask = shift_labels != -100
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def make_compute_metrics(val_domains):
    def compute_metrics(eval_pred):
        losses = np.asarray(eval_pred.predictions).reshape(-1)
        by_domain: dict[str, list[float]] = {}
        for loss, dom in zip(losses, val_domains):
            by_domain.setdefault(dom, []).append(loss)
        metrics = {}
        for i, (dom, vals) in enumerate(sorted(by_domain.items())):
            metrics[f"loss_course_{i}"] = float(np.mean(vals))
        metrics["macro_domain_loss"] = float(np.mean(list(metrics.values()))) if metrics else float("nan")
        return metrics

    return compute_metrics


class ProgressCallback(TrainerCallback):
    def __init__(self, ctx: StageContext):
        self.ctx = ctx
        self.last_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_loss = logs["loss"]

    def on_step_end(self, args, state, control, **kwargs):
        loss = f", loss {self.last_loss:.3f}" if self.last_loss is not None else ""
        self.ctx.reporter.update(state.global_step, state.max_steps,
                                 f"Step {state.global_step:,} / {state.max_steps:,}{loss}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_macro_domain_loss" in metrics:
            self.ctx.log(f"Eval at step {state.global_step}: macro per-course loss "
                         f"{metrics['eval_macro_domain_loss']:.4f}")


def find_resumable_checkpoint(output_dir: str, micro_batch: int):
    """Newest COMPLETE checkpoint written with the same micro-batch (a crash mid-save leaves
    a partial folder that transformers' get_last_checkpoint would happily resume from)."""
    required = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    ckpts = [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*")) if re.search(r"checkpoint-\d+$", d)]
    for d in sorted(ckpts, key=lambda p: int(p.rsplit("-", 1)[1]), reverse=True):
        if any(not os.path.isfile(os.path.join(d, f)) for f in required):
            continue
        if not zipfile.is_zipfile(os.path.join(d, "optimizer.pt")):
            continue
        with open(os.path.join(d, "trainer_state.json"), encoding="utf-8") as f:
            if json.load(f).get("train_batch_size") != micro_batch:
                continue
        return d
    return None


def preflight(model, train_ds, collator, micro_batch: int) -> bool:
    """Forward+backward on the longest possible micro-batch. True if it fits under the cap."""
    lens = [len(x) for x in train_ds["input_ids"]]
    longest = sorted(range(len(train_ds)), key=lens.__getitem__, reverse=True)[:micro_batch]
    batch = collator([train_ds[i] for i in longest])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model.train()
    try:
        model(**batch).loss.backward()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        model.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Stage entrypoint
# --------------------------------------------------------------------------------------
def run(ctx: StageContext) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = ctx.config
    configure_cuda(ctx)
    seed_everything(SEED)
    out_dir = str(ctx.ws.train_dir)
    os.makedirs(out_dir, exist_ok=True)

    syllabi = read_json(ctx.ws.syllabi)
    course_names = [c["course_name"] for c in syllabi["courses"]]
    preamble = build_preamble(cfg.language, cfg.program_name, course_names)
    final_prefix = preset(cfg.language)["final_answer"]

    ctx.reporter.start(f"Loading {cfg.base_model}")
    tok = build_tokenizer(cfg.base_model, cfg.max_seq_len)
    train_ds, val_ds, train_domains, val_domains, val_questions, longest = load_dataset(ctx, tok, preamble, final_prefix, cfg.max_seq_len)
    weights = domain_sample_weights(train_domains)
    use_weighted = not torch.all(weights == weights[0]).item()

    ctx.reporter.update(0, 1, f"Loading {cfg.base_model} in 4-bit (first run downloads the weights)", force=True)
    model = build_model(cfg.base_model, cfg.lora_r)
    model.config.pad_token_id = tok.pad_token_id
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx.log(f"LoRA r={cfg.lora_r}: {trainable / 1e6:.1f}M trainable parameters")

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest",
                                      label_pad_token_id=-100, pad_to_multiple_of=8)

    micro = DEFAULT_MICRO_BATCH
    if not preflight(model, train_ds, collator, micro):
        ctx.log(f"Worst-case batch ({micro} x {longest} tokens) does not fit; retrying with micro-batch 1")
        micro = 1
        if not preflight(model, train_ds, collator, micro):
            raise StageError(f"Out of GPU memory even with micro-batch 1 at {longest} tokens. Lower "
                             "max_seq_len or choose a smaller base model.")
    accum = EFFECTIVE_BATCH // micro

    samples_per_epoch = int(round(weights.sum().item())) if use_weighted else len(train_ds)
    steps_per_epoch = math.ceil(samples_per_epoch / EFFECTIVE_BATCH)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.demo:
        total_steps = max(1, math.ceil(total_steps * DEMO_TRAIN_FRACTION))
        ctx.log(f"Demo run: training for {total_steps} steps ({DEMO_TRAIN_FRACTION:.0%} of the normal run)")
    warmup = max(min(20, total_steps // 10), round(WARMUP_FRAC * total_steps))
    # ~4 evals/epoch on real runs; small runs still get at least two evals.
    eval_every = max(1, min(max(50, steps_per_epoch // 4), total_steps // 2))
    ctx.log(f"{len(train_ds)} train rows, {total_steps} steps (micro-batch {micro} x accum {accum}), "
            f"eval every {eval_every}")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        max_steps=total_steps if cfg.demo else -1,   # overrides num_train_epochs when set
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        weight_decay=0.0,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_bnb_8bit",          # non-paged: paging relies on unified memory WDDM can't oversubscribe
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=NEFTUNE_ALPHA,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_domain_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        seed=SEED,
    )
    trainer = WeightedDomainTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tok,
        compute_metrics=make_compute_metrics(val_domains),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        domain_sample_weights=weights if use_weighted else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE), ProgressCallback(ctx)],
    )

    last = find_resumable_checkpoint(out_dir, micro)
    if last:
        ctx.log(f"Resuming from {os.path.basename(last)}")
    try:
        trainer.train(resume_from_checkpoint=last)
    except torch.cuda.OutOfMemoryError as e:
        raise StageError("Ran out of GPU memory during training. Lower max_seq_len or use a smaller "
                         "base model, then retry (training resumes from the last checkpoint).") from e

    trainer.model.save_pretrained(str(ctx.ws.adapter_dir))
    tok.save_pretrained(str(ctx.ws.adapter_dir))
    best = trainer.state.best_metric
    ctx.log(f"Adapter saved. Best macro per-course validation loss: {best:.4f}" if best is not None else "Adapter saved.")

    # Smoke test on held-out questions, logged so the lecturer sees real answers right away.
    model.config.use_cache = True
    model.eval()
    tok.padding_side = "left"
    samples = []
    for question in val_questions[:2]:
        msgs = [{"role": "user", "content": f"{preamble}\n\n{question}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=0.3, top_p=0.9,
                                 repetition_penalty=1.15, no_repeat_ngram_size=4,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        samples.append({"question": question, "answer": answer})
        ctx.log(f"Sample Q: {question[:200]}\nSample A: {answer[:600]}")

    atomic_write_json(ctx.ws.train_meta, {
        "base_model": cfg.base_model,
        "preamble": preamble,
        "final_answer_prefix": final_prefix,
        "gen_kwargs": GEN_KWARGS,
        "best_macro_loss": best,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "steps": trainer.state.global_step,
        "max_seq_len": cfg.max_seq_len,
        "samples": samples,
    })
    ctx.reporter.finish(f"Adapter trained ({trainer.state.global_step:,} steps)",
                        best_macro_loss=best)
