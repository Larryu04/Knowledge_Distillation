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

import json
import os
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
MODEL_ID = "dicta-il/dictalm2.0-instruct"
DATA_PATH = "merged_training_dataset_V4_clean.json"   # run clean_dataset.py first
OUTPUT_DIR = "out/dictalm2-tutor-qlora-v4"

SEED = 42
MAX_SEQ_LEN = 1024          # V4 real token max = 444 (incl. Java code blocks); never truncates
VAL_FRACTION = 0.05

# Hebrew tutor preamble — prepended to every user turn, IDENTICALLY in train.py and
# inference.py. It conditions the model into the "explain then conclude" behaviour.
PREAMBLE = (
    "אתה מתרגל אקדמי מומחה בתחומי רשתות מחשבים, אבטחת סייבר ומערכות הפעלה. "
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
PER_DEVICE_BATCH = 8          # if you OOM: set to 4 and GRAD_ACCUM to 4
GRAD_ACCUM = 2                # effective batch = 16
WARMUP_FRAC = 0.03           # warmup_steps computed from real step count (see main())
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.3
NEFTUNE_ALPHA = 5


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
    examples, dropped_len, dropped_bad = [], 0, 0
    for r in rows:
        ins, rat, out = r.get("Instruction", ""), r.get("Rationale", ""), r.get("Output", "")
        if not (ins.strip() and rat.strip() and out.strip()):
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

    print(f"[data] kept={len(examples)}  dropped_too_long={dropped_len}  dropped_bad={dropped_bad}")
    lens = sorted(len(e["input_ids"]) for e in examples)
    print(f"[data] token length  p50={lens[len(lens)//2]}  p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")

    ds = Dataset.from_list(examples).shuffle(seed=SEED)
    split = ds.train_test_split(test_size=VAL_FRACTION, seed=SEED)

    # Print one fully-decoded training target so masking is visually verifiable.
    sample = split["train"][0]
    supervised = [t for t in sample["labels"] if t != -100]
    print("\n[check] supervised span for example 0:\n" + tok.decode(supervised) + "\n")

    return split["train"], split["test"]


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


# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tok = build_tokenizer()
    train_ds, val_ds = load_dataset(tok)
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

    # Schedule params scale with dataset size so you never hand-tune them again.
    import math
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(20, round(WARMUP_FRAC * total_steps))
    eval_every = max(50, steps_per_epoch // 4)
    print(f"[sched] {len(train_ds)} train rows | {steps_per_epoch} steps/epoch | "
          f"{total_steps} total steps | warmup {warmup_steps} | eval/save every {eval_every}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="paged_adamw_8bit",
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,     # 0 is safest on Windows
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,        # transformers 5.x: replaces the old `tokenizer=` arg
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

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
