"""
train.py — QLoRA knowledge-distillation fine-tune of a Hebrew academic-tutor "Student".

Teacher: GPT-4o (already used, offline, to generate merged_training_dataset_V1.json)
Student: dicta-il/dictalm2.0-instruct  (Mistral-7B base, Hebrew-extended tokenizer)

Design goals (see the write-up that accompanied this file):
  * 4-bit NF4 QLoRA + gradient checkpointing  -> fits an RTX 4070 Super (12 GB).
  * embed_tokens / lm_head kept FROZEN         -> native Hebrew stays intact.
  * low-rank, low-LR, few-epoch adapter        -> gentle format/domain teaching.
  * prompt tokens masked to -100               -> loss only on the assistant answer.
  * explicit EOS in every label + distinct PAD -> no infinite-generation loops.

Tested with:
  torch 2.6.0+cu124  transformers 5.16.1  peft 0.20  bitsandbytes 0.50.2
  accelerate 1.14  datasets 5.0  numpy 2.5   (Python 3.13, Windows)
"""

import glob
import json
import os
import random
import re
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

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
MODEL_ID = "dicta-il/dictalm2.0-instruct"
DATA_PATH = "merged_6_courses_new.json"   # schema: course_name/domain/instruction/rationale/output
OUTPUT_DIR = "out/dictalm2-tutor-qlora-v6"   # v6: trains on merged_6_courses_new.json, domain read
                                              # straight from the JSON (no keyword classifier)

SEED = 42
MAX_SEQ_LEN = 832           # merged_6_courses_new real token max = 822 (p50 565, p99 707), so nothing
                            # is dropped. Bounds the worst-case micro-batch at PER_DEVICE_BATCH x 832
                            # tokens. Don't go lower: 640 would drop ~5% of rows, the long
                            # rich-rationale ones that match the eval style.
VAL_FRACTION = 0.05
MIN_VAL_PER_DOMAIN = 20     # floor so a small eval slice (e.g. Cybersecurity) doesn't make
                            # its per-domain loss -- and therefore early stopping -- noisy

# Hebrew tutor preamble — prepended to every user turn, IDENTICALLY in train.py and
# inference.py. It conditions the model into the "explain then conclude" behaviour.
PREAMBLE = (
    "אתה מתרגל אקדמי מומחה בתחומי רשתות מחשבים, אבטחת סייבר, מערכות הפעלה, ושפות תכנות C++, Java וPython. "
    "ענה בעברית בלבד. תחילה הסבר את ההיגיון שלב אחר שלב, ולאחר מכן כתוב שורה "
    "שמתחילה ב'תשובה סופית:' ובה תשובה קצרה ותמציתית."
)
FINAL_ANSWER_PREFIX = "תשובה סופית:"

# LoRA. NOTE: NOT included here on purpose -> embed_tokens, lm_head (frozen vocabulary).
# V4 spans 6 courses incl. a syntax-heavy shift to Java/C++/Python, so r was raised
# 8 -> 16 to give the adapter room for the extra diversity. Still ~0.6% of params.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP  (drop these 3 for an even gentler run)
]

# Optimisation — conservative on purpose.
# EPOCHS 3 -> 2: with ~7.4k rows (was ~4k), 2 passes give MORE weight updates than the
# old 3x4k run; a 3rd pass mostly adds forgetting risk. EarlyStopping is the safety net.
EPOCHS = 2
LR = 1e-4                     # tracks batch size / stability, NOT dataset size -> unchanged
# Memory / hardware safety. The new dataset is ~2.3x longer per row than V4 (p50 565 vs 241
# tokens), so the old batch of 8 became ~5-6.6k tokens per micro-batch and pushed the 12 GB
# card into overcommit; on Windows (WDDM) that shows up as a freeze / nvlddmkm.sys BSOD
# rather than a clean CUDA OOM. Keep EFFECTIVE batch = 16 so LR / schedule / step counts
# stay comparable to v4/v5. (Resume skips checkpoints written with a different micro-batch
# size, so changing PER_DEVICE_BATCH between runs is safe.)
PER_DEVICE_BATCH = 2
EVAL_BATCH = 2
GRAD_ACCUM = 8                # effective batch = 16
VRAM_FRACTION = 0.85          # hard cap on this process (~10.4 GB of 12.3 GB) -> a Python OOM you
                              # can catch, instead of the driver overcommitting; leaves headroom
                              # for the Windows compositor / browser
RESUME_FROM_LAST = True       # pick up from the newest checkpoint in OUTPUT_DIR after a crash
WARMUP_FRAC = 0.03           # warmup_steps computed from real step count (see main())
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.3
NEFTUNE_ALPHA = 5


def configure_cuda_safety() -> None:
    """Must run before the first CUDA allocation."""
    # Less fragmentation with variable-length batches, and free cached blocks before the cap
    # below is hit. (expandable_segments is not supported on Windows, so not used.)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128"
    )
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[vram] cap {VRAM_FRACTION:.0%} of {total:.1f} GB = {VRAM_FRACTION * total:.1f} GB")


def find_resumable_checkpoint(output_dir):
    """Newest checkpoint that is COMPLETE and was written with the current micro-batch size.

    Trainer writes checkpoint-N in place, so a crash mid-save leaves a partial folder (the v6
    run died writing optimizer.pt at step 107: 0.3 MB instead of ~86 MB, no scheduler /
    rng / trainer_state). transformers' get_last_checkpoint only looks at folder names and
    would resume from it and crash."""
    required = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "trainer_state.json")
    ckpts = [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
             if re.search(r"checkpoint-\d+$", d)]
    for d in sorted(ckpts, key=lambda p: int(p.rsplit("-", 1)[1]), reverse=True):
        missing = [f for f in required if not os.path.isfile(os.path.join(d, f))]
        if missing:
            print(f"[resume] skipping {d}: incomplete, missing {missing}")
            continue
        if not zipfile.is_zipfile(os.path.join(d, "optimizer.pt")):
            print(f"[resume] skipping {d}: optimizer.pt is truncated")
            continue
        with open(os.path.join(d, "trainer_state.json"), encoding="utf-8") as f:
            saved_bs = json.load(f).get("train_batch_size")
        if saved_bs != PER_DEVICE_BATCH:
            print(f"[resume] skipping {d}: written with micro-batch {saved_bs}, now {PER_DEVICE_BATCH}")
            continue
        return d
    return None


class VramLogCallback(TrainerCallback):
    """Prints peak VRAM at every logging step so a creeping footprint is visible early."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            gb = 1024**3
            print(f"[vram] step {state.global_step}: peak allocated "
                  f"{torch.cuda.max_memory_allocated() / gb:.2f} GB | peak reserved "
                  f"{torch.cuda.max_memory_reserved() / gb:.2f} GB")


def preflight_worst_case(model, train_ds, collator) -> None:
    """One forward+backward on the LONGEST possible micro-batch (the PER_DEVICE_BATCH longest
    rows) before training starts. Batches are padded to their longest row, so if this fits
    under the VRAM cap every real batch does -- an OOM here happens in seconds, in a
    controlled way, instead of hours in when the sampler finally draws two long rows together."""
    lens = [len(x) for x in train_ds["input_ids"]]
    longest = sorted(range(len(train_ds)), key=lens.__getitem__, reverse=True)[:PER_DEVICE_BATCH]
    batch = collator([train_ds[i] for i in longest])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    try:
        model(**batch).loss.backward()
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"[preflight] OOM on the worst-case micro-batch ({PER_DEVICE_BATCH} x "
            f"{batch['input_ids'].shape[1]} tokens). Set PER_DEVICE_BATCH = 1 and "
            f"GRAD_ACCUM = {PER_DEVICE_BATCH * GRAD_ACCUM}."
        )
    finally:
        model.zero_grad(set_to_none=True)
    gb = 1024**3
    print(f"[preflight] worst-case micro-batch {tuple(batch['input_ids'].shape)}: peak allocated "
          f"{torch.cuda.max_memory_allocated() / gb:.2f} GB | peak reserved "
          f"{torch.cuda.max_memory_reserved() / gb:.2f} GB")
    torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# --------------------------------------------------------------------------------------
# Tokenizer + EOS/PAD configuration
# --------------------------------------------------------------------------------------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # DictaLM/Mistral ship no pad token. Do NOT reuse EOS as PAD — if PAD==EOS the
    # collator's label masking can teach the model that EOS is "just padding", which
    # is exactly how you get runs that never stop. Use the (otherwise unused) <unk>.
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    assert tok.pad_token_id != tok.eos_token_id, "PAD must differ from EOS"

    tok.padding_side = "right"          # right-pad for training
    tok.model_max_length = MAX_SEQ_LEN
    return tok


# --------------------------------------------------------------------------------------
# Example construction: render with the model's OWN chat template, mask the prompt.
# --------------------------------------------------------------------------------------
def make_example_builder(tok):
    eos = tok.eos_token

    def build(instruction: str, rationale: str, output: str):
        user = f"{PREAMBLE}\n\n{instruction.strip()}"
        assistant = f"{rationale.strip()}\n\n{FINAL_ANSWER_PREFIX} {output.strip()}"
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

        # Prompt prefix (everything the model should NOT be trained to produce).
        prompt_text = tok.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True
        )
        # Full conversation. rstrip() removes the trailing space some templates
        # (incl. DictaLM's) append AFTER eos_token, so the sequence ends exactly at EOS.
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        ).rstrip()
        if not full_text.endswith(eos):
            full_text += eos

        # add_special_tokens=False: the template already emits <s> / </s>.
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]

        # Largest n <= len(prompt_ids) with full_ids[:n] == prompt_ids[:n]
        # (guards against a token straddling the prompt/response boundary).
        n = min(len(prompt_ids), len(full_ids))
        while n > 0 and full_ids[:n] != prompt_ids[:n]:
            n -= 1

        labels = [-100] * n + full_ids[n:]
        keep = any(t != -100 for t in labels) and (eos_id_in(labels, tok.eos_token_id))
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


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def load_dataset(tok):
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    build = make_example_builder(tok)
    examples, domains, dropped_len, dropped_bad = [], [], 0, 0
    for r in rows:
        # course_name/domain are metadata only -- never fed into build(), which formats
        # just instruction+rationale+output into the model's prompt/labels.
        ins, rat, out = r.get("instruction", ""), r.get("rationale", ""), r.get("output", "")
        domain = r.get("domain", "")
        if not (ins.strip() and rat.strip() and out.strip() and domain.strip()):
            dropped_bad += 1
            continue
        ex = build(ins, rat, out)
        if not ex["_keep"]:
            dropped_bad += 1
            continue
        if ex["_len"] > MAX_SEQ_LEN:
            dropped_len += 1
            continue
        examples.append({k: v for k, v in ex.items() if not k.startswith("_")})
        domains.append(domain)

    print(f"[data] kept={len(examples)}  dropped_too_long={dropped_len}  dropped_bad={dropped_bad}")
    lens = sorted(len(e["input_ids"]) for e in examples)
    print(f"[data] token length  p50={lens[len(lens)//2]}  p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")

    # Stratified split: every domain gets a guaranteed eval slice sized off ITS OWN count,
    # not the pooled 5%. A plain random split starves the smallest domain (Cybersecurity)
    # of eval rows, which is exactly why a pooled eval_loss can look fine while 5/6 domains
    # quietly regress (see summary.md) -- the metric never had enough signal from them.
    rng = random.Random(SEED)
    by_domain = {}
    for i, d in enumerate(domains):
        by_domain.setdefault(d, []).append(i)

    train_idx, val_idx = [], []
    print("[data] per-domain split (train / val):")
    for d, idxs in sorted(by_domain.items(), key=lambda kv: -len(kv[1])):
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(MIN_VAL_PER_DOMAIN, round(len(idxs) * VAL_FRACTION))
        n_val = min(n_val, len(idxs) // 2)   # never eat more than half a tiny domain's rows
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
        print(f"         {d:20s}  train={len(idxs) - n_val:5d}  val={n_val:4d}")
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_domains = [domains[i] for i in train_idx]
    val_domains = [domains[i] for i in val_idx]
    train_ds = Dataset.from_list([examples[i] for i in train_idx])
    val_ds = Dataset.from_list([examples[i] for i in val_idx])

    # Print one fully-decoded training target so masking is visually verifiable.
    sample = train_ds[0]
    supervised = [t for t in sample["labels"] if t != -100]
    print("\n[check] supervised span for example 0:\n" + tok.decode(supervised) + "\n")

    return train_ds, val_ds, train_domains, val_domains


def build_domain_sample_weights(domains):
    """Soft inverse-sqrt-frequency oversampling weight per training row.

    Plain shuffling draws each row with equal probability, so within a fixed number of
    epochs/steps the majority domains (Networks/OS) get proportionally far more gradient
    updates than Cybersecurity/C++/Java/Python -- the likely cause of the per-domain
    regression in summary.md. sqrt(max_count / count) softens this (minority domains get
    ~2-4x more draws, not a hard 10-15x that plain inverse-frequency would give the
    smallest class), which limits overfitting risk on the smallest domain.
    """
    counts = Counter(domains)
    max_count = max(counts.values())
    weight_by_domain = {d: (max_count / c) ** 0.5 for d, c in counts.items()}
    print("[sampler] domain oversampling weights:", {d: round(w, 2) for d, w in weight_by_domain.items()})
    return torch.tensor([weight_by_domain[d] for d in domains], dtype=torch.double)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,                # transformers 5.x: `dtype` (was `torch_dtype`)
        attn_implementation="sdpa",          # portable; FA2 is painful on Windows
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        # modules_to_save intentionally unset -> embed_tokens & lm_head stay frozen.
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Hard assertion: nothing in the embedding / head is trainable.
    for name, p in model.named_parameters():
        if p.requires_grad and ("embed_tokens" in name or "lm_head" in name):
            raise RuntimeError(f"Vocabulary parameter is trainable: {name}")
    return model


class WeightedDomainTrainer(Trainer):
    """Trainer whose train sampler draws minority-domain rows more often (see
    build_domain_sample_weights), so the same EPOCHS/step budget gives Cybersecurity /
    C++ / Java / Python proportionally more updates instead of being drowned out by
    Networks/OS. Eval is untouched (still sequential, no oversampling)."""

    def __init__(self, *args, domain_sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_sample_weights = domain_sample_weights

    def _get_train_sampler(self, train_dataset=None):
        if self.domain_sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        # num_samples = sum(weights), NOT len(weights). A weight-1.0 (majority-domain) row
        # then still gets ~1 expected draw/epoch; oversampled rows get MORE draws on top of
        # that. Using len(weights) here previously renormalized the whole draw budget down,
        # so majority rows were drawn <1x/epoch -- less total training, not more balanced
        # training (see summary.md / [[hebrew-distillation-pitfalls]]).
        num_samples = int(round(self.domain_sample_weights.sum().item()))
        return WeightedRandomSampler(
            self.domain_sample_weights,
            num_samples=num_samples,
            replacement=True,
        )


def preprocess_logits_for_metrics(logits, labels):
    """Reduce (batch, seq, vocab) logits to a per-example scalar loss immediately, before
    Trainer accumulates predictions across the whole eval set -- keeps eval memory the
    same as a normal forward pass instead of holding full-vocab logits for every row."""
    if isinstance(logits, tuple):
        logits = logits[0]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
    )
    mask = shift_labels != -100
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def make_compute_metrics(val_domains):
    """Per-domain masked loss + a macro average across all domains present in val_domains.
    Used as metric_for_best_model so checkpoint selection / early stopping reflects every
    target domain equally -- not the pooled eval_loss, which is dominated by whichever
    domain has the most eval rows (Networks/OS) and can look fine while others regress.

    val_domains comes straight from each row's "domain" key (see load_dataset), so every
    row is accurately attributed -- no keyword-classifier miss bucket to exclude anymore."""

    def compute_metrics(eval_pred):
        per_example_loss = np.asarray(eval_pred.predictions).reshape(-1)
        by_domain = {}
        for loss_val, dom in zip(per_example_loss, val_domains):
            by_domain.setdefault(dom, []).append(loss_val)
        metrics, domain_means = {}, []
        for dom, vals in by_domain.items():
            m = float(np.mean(vals))
            key = "loss_" + dom.lower().replace(" ", "_").replace("++", "pp")
            metrics[key] = m
            domain_means.append(m)
        metrics["macro_domain_loss"] = float(np.mean(domain_means)) if domain_means else float("nan")
        return metrics

    return compute_metrics


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def main():
    configure_cuda_safety()
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tok = build_tokenizer()
    train_ds, val_ds, train_domains, val_domains = load_dataset(tok)
    domain_weights = build_domain_sample_weights(train_domains)
    model = build_model()

    # PAD the model config too (used by generate / some internals).
    model.config.pad_token_id = tok.pad_token_id

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding="longest",
        label_pad_token_id=-100,     # padded label positions -> ignored by the loss
        pad_to_multiple_of=8,
    )

    # Schedule params scale with the ACTUAL per-epoch sample count. With domain oversampling
    # the dataloader draws sum(domain_weights) samples per epoch, not len(train_ds) -- using
    # len(train_ds) here would undercount steps_per_epoch and hand Trainer a stale warmup /
    # eval cadence relative to what num_train_epochs=EPOCHS actually iterates.
    import math
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    samples_per_epoch = int(round(domain_weights.sum().item()))
    steps_per_epoch = math.ceil(samples_per_epoch / eff_batch)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(20, round(WARMUP_FRAC * total_steps))
    eval_every = max(50, steps_per_epoch // 4)
    print(f"[sched] {len(train_ds)} unique train rows | {samples_per_epoch} draws/epoch (oversampled) | "
          f"{steps_per_epoch} steps/epoch | {total_steps} total steps | "
          f"warmup {warmup_steps} | eval/save every {eval_every}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        # Non-paged 8-bit AdamW: the LoRA optimizer state is only ~tens of MB, so paging buys
        # nothing, and paged optimizers rely on CUDA unified memory, which Windows/WDDM cannot
        # oversubscribe -- an avoidable extra path through the driver that keeps crashing.
        optim="adamw_bnb_8bit",
        bf16=True,
        fp16=False,
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
        metric_for_best_model="macro_domain_loss",   # balanced across domains, not pooled
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,     # 0 is safest on Windows
        seed=SEED,
    )

    trainer = WeightedDomainTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,        # transformers 5.x: replaces the old `tokenizer=` arg
        compute_metrics=make_compute_metrics(val_domains),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        domain_sample_weights=domain_weights,
        # patience raised 3 -> 5: macro_domain_loss is a noisier composite metric (6
        # domain means, some off eval slices as small as MIN_VAL_PER_DOMAIN=20 rows) than
        # a single pooled eval_loss, and the v4-balanced run tripped patience=3 on what
        # was mostly noise after only 37% of the planned steps.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), VramLogCallback()],
    )

    preflight_worst_case(model, train_ds, collator)

    last_ckpt = find_resumable_checkpoint(OUTPUT_DIR) if RESUME_FROM_LAST else None
    if last_ckpt:
        print(f"[resume] continuing from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save the adapter + tokenizer (NOT the merged model).
    final_dir = os.path.join(OUTPUT_DIR, "adapter")
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"\n[done] adapter saved to {final_dir}")

    # ---- quick smoke test -----------------------------------------------------------
    model.config.use_cache = True
    model.eval()
    q = "מהו ההבדל המרכזי בין TCP ל-UDP?"
    msgs = [{"role": "user", "content": f"{PREAMBLE}\n\n{q}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    print("\n[sample]\n" + tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
