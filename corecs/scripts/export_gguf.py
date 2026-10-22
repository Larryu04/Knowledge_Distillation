"""
export_gguf.py — merge a LoRA adapter into its base model and export a CoreCS-ready GGUF.

Run from the TRAINING repo (Knowledge_Dis), with its venv active, so this script can read
the exact prompt preamble + sampling settings from that repo's inference.py:

    cd C:\\Users\\Larry\\Knowledge_Dis
    .venv\\Scripts\\activate
    python ..\\corecs\\scripts\\export_gguf.py --adapter out/dictalm2-tutor-qlora-v7/adapter --name corecs-v7

Produces, in the CoreCS models folder (default: %APPDATA%\\com.corecs.app\\models):
    corecs-v7-q4_k_m.gguf           the model (chat template embedded in its metadata)
    corecs-v7-q4_k_m.corecs.json    optional "model profile": preamble + sampling defaults

CoreCS itself is model-agnostic: it loads ANY valid .gguf in that folder. The profile file
is how a checkpoint tells the app how it was trained to be prompted; without one, the app
sends plain chat messages with default sampling.

Disk strategy (merging a 7B model is disk-hungry):
    merged HF model (bf16, ~14.5 GB)  ->  intermediate GGUF  ->  Q4_K_M (~4.4 GB)
The intermediate is bf16 (~14.5 GB, best quality) when there is room, otherwise q8_0
(~7.7 GB, near-lossless). Every intermediate is deleted as soon as it is no longer needed.
"""

import argparse
import datetime as dt
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GB = 1024**3
CORECS_ROOT = Path(__file__).resolve().parent.parent
LLAMA_SRC = CORECS_ROOT / "tools" / "llama.cpp-src"
LLAMA_BIN = CORECS_ROOT / "tools" / "llama.cpp-bin"
DEFAULT_MODELS_DIR = Path(os.environ.get("APPDATA", Path.home())) / "com.corecs.app" / "models"

# Free space needed at peak for each intermediate type (merged model + intermediate + margin).
NEED_BF16 = 34 * GB
NEED_Q8 = 25 * GB


def log(msg):
    print(f"[export] {msg}", flush=True)


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def load_training_config(module_name: str):
    """Import PREAMBLE / GEN_KWARGS / BASE_MODEL_ID from the training repo's inference module."""
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(module_name)
    return mod.BASE_MODEL_ID, getattr(mod, "PREAMBLE", None), getattr(mod, "GEN_KWARGS", {})


def merge(base_id: str, adapter: Path, out_dir: Path):
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"merging {adapter} into {base_id} on CPU (bf16, needs ~16 GB RAM)...")
    tok = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(out_dir)  # writes chat_template.jinja -> embedded into the GGUF

    # The SentencePiece model gives llama.cpp the exact original vocab (Hebrew-extended).
    spm = hf_hub_download(base_id, "tokenizer.model")
    shutil.copy(spm, out_dir / "tokenizer.model")
    del merged, base


def run(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def verify(gguf_path: Path):
    sys.path.insert(0, str(LLAMA_SRC / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    keys = {f.name for f in reader.fields.values()}
    for required in ("tokenizer.chat_template", "tokenizer.ggml.tokens", "general.architecture"):
        if required not in keys:
            raise SystemExit(f"[export] FAILED: {required} missing from {gguf_path.name}")
    n_vocab = len(reader.fields["tokenizer.ggml.tokens"].data)
    log(f"verified: architecture + chat template present, vocab size {n_vocab}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, type=Path, help="LoRA adapter dir, e.g. out/dictalm2-tutor-qlora-v7/adapter")
    ap.add_argument("--name", required=True, help="output base name, e.g. corecs-v7")
    ap.add_argument("--display-name", help='name shown in the app (default: derived from --name, e.g. "CoreCS v7")')
    ap.add_argument("--quant", default="Q4_K_M", help="llama.cpp quant type (default Q4_K_M)")
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="where CoreCS looks for models")
    ap.add_argument("--work-dir", type=Path, default=Path("out/_gguf_work"), help="scratch space for intermediates")
    ap.add_argument("--training-module", default="inference", help="module in CWD defining PREAMBLE/GEN_KWARGS/BASE_MODEL_ID")
    ap.add_argument("--context", type=int, default=4096, help="context length the app should run this model with")
    ap.add_argument("--keep-work", action="store_true", help="don't delete intermediates (debugging)")
    args = ap.parse_args()

    for p in (LLAMA_SRC / "convert_hf_to_gguf.py", LLAMA_BIN / "llama-quantize.exe"):
        if not p.exists():
            raise SystemExit(f"[export] missing toolchain file: {p}")
    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"[export] not a LoRA adapter dir: {args.adapter}")

    base_id, preamble, gen_kwargs = load_training_config(args.training_module)
    stem = f"{args.name}-{args.quant.lower()}"
    final_gguf = args.models_dir / f"{stem}.gguf"
    profile_path = args.models_dir / f"{stem}.corecs.json"
    merged_dir = args.work_dir / f"{args.name}-merged"
    args.models_dir.mkdir(parents=True, exist_ok=True)

    free = free_bytes(args.work_dir)
    if free >= NEED_BF16:
        inter_type = "bf16"
    elif free >= NEED_Q8:
        inter_type = "q8_0"
    else:
        raise SystemExit(f"[export] only {free / GB:.1f} GB free; need at least {NEED_Q8 / GB:.0f} GB")
    inter_gguf = args.work_dir / f"{args.name}-{inter_type}.gguf"
    log(f"{free / GB:.1f} GB free -> intermediate type {inter_type}")

    # 1. merge
    if not (merged_dir / "config.json").exists():
        merge(base_id, args.adapter, merged_dir)
    else:
        log(f"reusing existing merged model in {merged_dir}")

    # 2. HF -> GGUF (intermediate)
    display = args.display_name or args.name.replace("-", " ").replace("corecs", "CoreCS")
    run([sys.executable, LLAMA_SRC / "convert_hf_to_gguf.py", merged_dir,
         "--outfile", inter_gguf, "--outtype", inter_type, "--model-name", display])
    if not args.keep_work:
        shutil.rmtree(merged_dir)
        log("deleted merged HF model")

    # 3. quantize
    quant_cmd = [LLAMA_BIN / "llama-quantize.exe"]
    if inter_type == "q8_0":
        quant_cmd.append("--allow-requantize")
    # Written as .part and renamed at the end, so a running CoreCS never sees a half-written model.
    part_gguf = final_gguf.with_name(final_gguf.name + ".part")
    run(quant_cmd + [inter_gguf, part_gguf, args.quant])
    if not args.keep_work:
        inter_gguf.unlink()
        log(f"deleted intermediate {inter_gguf.name}")

    verify(part_gguf)

    # 4. model profile — how this checkpoint expects to be prompted (see docs/project_memory.md)
    sampling = {
        "temperature": gen_kwargs.get("temperature", 0.7),
        "top_p": gen_kwargs.get("top_p", 0.9),
        "repeat_penalty": gen_kwargs.get("repetition_penalty", 1.1),
        "max_tokens": gen_kwargs.get("max_new_tokens", 1024),
    }
    profile = {
        "schema": 1,
        "display_name": display,
        "context_length": args.context,
        "prompt": {"preamble": preamble, "separator": "\n\n"} if preamble else None,
        "sampling": sampling,
        "source": {
            "base_model": base_id,
            "adapter": str(args.adapter.resolve()),
            "quant": args.quant,
            "intermediate": inter_type,
            "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part_gguf, final_gguf)

    log(f"done: {final_gguf} ({final_gguf.stat().st_size / GB:.2f} GB)")
    log(f"profile: {profile_path}")


if __name__ == "__main__":
    main()
