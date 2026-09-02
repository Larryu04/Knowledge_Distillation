"""
inference.py — run the fine-tuned Hebrew tutor (base in 4-bit + LoRA adapter).

Usage:
  python inference.py                                  # interactive REPL
  python inference.py -q "מהו three-way handshake?"    # single question
  python inference.py --merge out/merged               # write a merged fp16 model and exit

The prompt construction here is byte-for-byte identical to train.py.
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "dicta-il/dictalm2.0-instruct"
ADAPTER_DIR = "out/dictalm2-tutor-qlora-v4/adapter"

# MUST match train.py exactly.
PREAMBLE = (
    "אתה מתרגל אקדמי מומחה בתחומי רשתות מחשבים, אבטחת סייבר ומערכות הפעלה. "
    "ענה בעברית בלבד. תחילה הסבר את ההיגיון שלב אחר שלב, ולאחר מכן כתוב שורה "
    "שמתחילה ב'תשובה סופית:' ובה תשובה קצרה ותמציתית."
)

GEN_KWARGS = dict(
    max_new_tokens=512,
    min_new_tokens=8,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.15,   # guardrail vs. degenerate repetition
    no_repeat_ngram_size=4,    # guardrail vs. n-gram loops
)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    tok.padding_side = "left"   # left-pad for generation
    return tok


def load_model(tok, merge_to=None):
    if merge_to:
        # Merge on CPU/fp16 for a standalone model (needs ~15 GB RAM, not VRAM).
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.float16, device_map="cpu"
        )
        merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
        merged.config.pad_token_id = tok.pad_token_id
        merged.save_pretrained(merge_to)
        tok.save_pretrained(merge_to)
        print(f"[merge] saved merged model to {merge_to}")
        return None

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    return model


@torch.no_grad()
def answer(model, tok, question: str) -> str:
    msgs = [{"role": "user", "content": f"{PREAMBLE}\n\n{question.strip()}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        **GEN_KWARGS,
    )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--merge", metavar="DIR", help="merge adapter into base, save to DIR, exit")
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_model(tok, merge_to=args.merge)
    if model is None:
        return

    if args.question:
        print(answer(model, tok, args.question))
        return

    print("Hebrew tutor ready. Empty line or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("שאלה> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print("\n" + answer(model, tok, q) + "\n" + "-" * 60)


if __name__ == "__main__":
    sys.exit(main())
