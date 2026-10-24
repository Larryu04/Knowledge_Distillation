"""
Stage 4 — Model packaging & GGUF quantization (runs under GLOBAL_TRAIN_PYTHON).

    LoRA adapter + base model --merge (CPU, bf16)--> merged HF model
        --convert_hf_to_gguf.py--> intermediate GGUF (bf16, or q8_0 when disk is tight)
        --llama-quantize--> <name>-<quant>.gguf  (+ <name>-<quant>.corecs.json profile)
        --> <name>-bundle.zip  (model, profile, adapter, portable runtime, desktop installer,
                                Hugging Face model card + Space template, deployment guide)

Every intermediate is deleted as soon as it is no longer needed: merging a 7B model needs
~35 GB of scratch space at peak. The GGUF and the ZIP are written as *.part and renamed at
the end, so nothing downstream ever sees a half-written file.
"""

import datetime as dt
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .. import db
from ..config import get_settings
from ..jobs import model_size_b
from . import deploy
from .base import StageContext, StageError, atomic_write_json, read_json, slugify

GB = 1024**3
STORED_SUFFIXES = {".gguf", ".exe", ".dll", ".safetensors", ".bin", ".zip", ".so", ".dylib"}
RUNTIME_BIN_PATTERNS = ("llama-server", "llama-server.exe", "llama-server-impl.dll", "ggml*.dll", "llama.dll",
                        "llama-common.dll", "mtmd.dll", "libomp.dll", "LICENSE*", "*.so", "*.dylib")
CONTEXT_LENGTH = 4096

# Weighted sub-steps, so the stage progress bar moves sensibly.
STEPS = {"merge": 0.35, "convert": 0.30, "quantize": 0.20, "bundle": 0.15}


class Steps:
    def __init__(self, ctx: StageContext):
        self.ctx, self.done = ctx, 0.0

    def begin(self, name: str, message: str):
        self.ctx.reporter.update(self.done, 1.0, message, force=True)
        self.ctx.log(message)

    def end(self, name: str):
        self.done += STEPS[name]
        self.ctx.reporter.update(self.done, 1.0, f"{name} done", force=True)


def exe(path: Path) -> Path:
    return path.with_suffix(".exe") if os.name == "nt" else path


def run_logged(ctx: StageContext, cmd: list) -> None:
    ctx.log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-30:]
    if proc.wait() != 0:
        tool = next((Path(str(c)).name for c in cmd[:2] if str(c).endswith(".py")), Path(str(cmd[0])).name)
        raise StageError(f"{tool} failed:\n" + "\n".join(tail[-12:]))


def merge(ctx: StageContext, base_id: str, out_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ctx.ws.adapter_dir), use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu",
                                                low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, str(ctx.ws.adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(out_dir)       # writes the chat template -> embedded into the GGUF
    # SentencePiece models: the original .model file gives llama.cpp the exact vocab.
    try:
        from huggingface_hub import hf_hub_download

        shutil.copy(hf_hub_download(base_id, "tokenizer.model"), out_dir / "tokenizer.model")
    except Exception:  # noqa: BLE001 - BPE tokenizers (Llama 3, Qwen) have no tokenizer.model
        pass
    del merged, base


def verify_gguf(ctx: StageContext, path: Path) -> None:
    sys.path.insert(0, str(get_settings().llama_src / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    keys = {f.name for f in reader.fields.values()}
    for required in ("tokenizer.chat_template", "tokenizer.ggml.tokens", "general.architecture"):
        if required not in keys:
            raise StageError(f"GGUF verification failed: {required} missing from {path.name}")
    ctx.log(f"Verified {path.name}: architecture, vocabulary ({len(reader.fields['tokenizer.ggml.tokens'].data):,} "
            "tokens) and chat template present")


def build_profile(ctx: StageContext, meta: dict, display: str, quant: str, inter: str) -> dict:
    g = meta["gen_kwargs"]
    rows = read_json(ctx.ws.merged, default=[])
    by_course = {}
    for r in rows:
        by_course.setdefault(r["course_name"], r["instruction"])
    return {
        "schema": 1,
        "display_name": display,
        "context_length": CONTEXT_LENGTH,
        "prompt": {"preamble": meta["preamble"], "separator": "\n\n"},
        "sampling": {
            "temperature": g["temperature"],
            "top_p": g["top_p"],
            "repeat_penalty": g["repetition_penalty"],
            "max_tokens": g["max_new_tokens"],
        },
        "example_questions": list(by_course.values())[:4],
        "language": ctx.config.language,
        "courses": list(by_course),
        "source": {
            "base_model": meta["base_model"],
            "quant": quant,
            "intermediate": inter,
            "program": ctx.config.program_name,
            "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
            "generator": "CoreCS Global",
        },
    }


def model_card(ctx: StageContext, meta: dict, profile: dict, gguf_name: str) -> str:
    courses = "\n".join(f"- {c}" for c in profile["courses"])
    lang_code = {"English": "en", "Hebrew": "he", "Arabic": "ar", "Russian": "ru", "Spanish": "es",
                 "French": "fr", "German": "de"}.get(ctx.config.language)
    lang = f"language:\n- {lang_code}\n" if lang_code else ""
    return f"""---
base_model: {meta['base_model']}
{lang}tags:
- gguf
- education
- tutor
- corecs
library_name: gguf
---

# {profile['display_name']}

An offline course tutor for the **{ctx.config.program_name}** program, fine-tuned (QLoRA) from
[{meta['base_model']}](https://huggingface.co/{meta['base_model']}) on synthetic, syllabus-grounded
Q&A generated with CoreCS Global.

## Courses

{courses}

## Files

- `{gguf_name}`: the model ({profile['source']['quant']}), runs with llama.cpp / LM Studio / Ollama / CoreCS.
- `{Path(gguf_name).with_suffix('.corecs.json').name}`: prompt preamble + sampling defaults used in training.

## Prompting

The model was trained with this preamble prepended to the first user message (not a system
message):

> {meta['preamble']}

## Limitations

Trained on synthetic data derived from course syllabi. It can be wrong; students should
verify important answers. Usage is subject to the base model's license.
"""


def copy_runtime_bins(dest: Path) -> int:
    src = get_settings().llama_bin
    if not src.is_dir():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in RUNTIME_BIN_PATTERNS:
        for f in src.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dest / f.name)
                n += 1
    return n


def zip_dir(src: Path, zip_path: Path, root_name: str) -> None:
    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w", allowZip64=True) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                comp = zipfile.ZIP_STORED if f.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                z.write(f, f"{root_name}/{f.relative_to(src).as_posix()}", compress_type=comp)
    os.replace(part, zip_path)


def run(ctx: StageContext) -> None:
    s = get_settings()
    cfg = ctx.config
    meta = read_json(ctx.ws.train_meta)
    if not meta or not (ctx.ws.adapter_dir / "adapter_config.json").exists():
        raise StageError("No trained adapter found; the fine-tuning stage must run first.")
    convert_py = s.llama_src / "convert_hf_to_gguf.py"
    quantize = exe(s.llama_bin / "llama-quantize")
    for p in (convert_py, quantize):
        if not p.exists():
            raise StageError(f"llama.cpp toolchain file missing: {p}. Run scripts/setup_toolchain.py "
                             "or set GLOBAL_TOOLCHAIN_DIR.")

    ctx.reporter.start("Preparing export")
    steps = Steps(ctx)
    slug = slugify(cfg.program_name, fallback="tutor")
    stem = f"{slug}-tutor-{cfg.quant.lower()}"
    work = ctx.ws.package_dir / "work"
    stage_dir = ctx.ws.package_dir / "bundle"          # becomes the ZIP's content
    model_dir = stage_dir / "model"
    final_gguf = model_dir / f"{stem}.gguf"
    work.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    size_b = model_size_b(cfg.base_model)
    need_bf16, need_q8 = (4.8 * size_b + 2) * GB, (3.5 * size_b + 2) * GB
    free = shutil.disk_usage(work).free
    if final_gguf.exists():
        inter_type = read_json(ctx.ws.package_dir / "export.json", default={}).get("intermediate", "bf16")
        ctx.log(f"Reusing existing {final_gguf.name}")
        steps.done = STEPS["merge"] + STEPS["convert"] + STEPS["quantize"]
    else:
        if free >= need_bf16:
            inter_type = "bf16"
        elif free >= need_q8:
            inter_type = "q8_0"
        else:
            raise StageError(f"Only {free / GB:.1f} GB of free disk; exporting this model needs at least "
                             f"{need_q8 / GB:.0f} GB of scratch space.")
        merged_dir = work / "merged"
        inter = work / f"{stem}-{inter_type}.gguf"

        steps.begin("merge", f"Merging adapter into {cfg.base_model} on CPU (bf16)")
        if not (merged_dir / "config.json").exists():
            merge(ctx, cfg.base_model, merged_dir)
        steps.end("merge")

        steps.begin("convert", f"Converting to GGUF ({inter_type} intermediate, {free / GB:.0f} GB free)")
        if not inter.exists():
            run_logged(ctx, [s.train_python, convert_py, merged_dir, "--outfile", inter,
                             "--outtype", inter_type, "--model-name", cfg.display_name])
        shutil.rmtree(merged_dir, ignore_errors=True)
        steps.end("convert")

        steps.begin("quantize", f"Quantizing to {cfg.quant}")
        part = final_gguf.with_name(final_gguf.name + ".part")
        cmd = [quantize] + (["--allow-requantize"] if inter_type == "q8_0" else []) + [inter, part, cfg.quant]
        run_logged(ctx, cmd)
        verify_gguf(ctx, part)
        os.replace(part, final_gguf)
        inter.unlink(missing_ok=True)
        atomic_write_json(ctx.ws.package_dir / "export.json", {"intermediate": inter_type})
        steps.end("quantize")

    steps.begin("bundle", "Assembling the download bundle")
    profile = build_profile(ctx, meta, cfg.display_name, cfg.quant, inter_type)
    profile_path = final_gguf.with_suffix(".corecs.json")
    atomic_write_json(profile_path, profile)

    # Portable runtime: llama-server + a static chat UI served by llama-server itself.
    portable_src = s.runtime_dir / "portable"
    portable = stage_dir / "runtime" / "portable"
    shutil.rmtree(portable, ignore_errors=True)
    shutil.copytree(portable_src, portable, ignore=shutil.ignore_patterns("bin", "README.md"))
    atomic_write_json(portable / "webui" / "profile.json", profile)
    n_bins = copy_runtime_bins(portable / "bin")
    if n_bins == 0:
        ctx.log("No llama-server binaries in the toolchain; the portable runtime will use llama-server from PATH.")

    if s.desktop_installer:
        desktop = stage_dir / "runtime" / "desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.desktop_installer, desktop / s.desktop_installer.name)
        ctx.log(f"Included desktop installer {s.desktop_installer.name}")

    if cfg.include_adapter:
        shutil.copytree(ctx.ws.adapter_dir, stage_dir / "adapter", dirs_exist_ok=True)

    hf = stage_dir / "huggingface"
    (hf / "space").mkdir(parents=True, exist_ok=True)
    (hf / "README.md").write_text(model_card(ctx, meta, profile, final_gguf.name), encoding="utf-8")
    shutil.copytree(portable / "webui", hf / "space" / "webui", dirs_exist_ok=True)
    model_repo, _ = deploy.repo_ids(cfg.hf_repo, slug)
    for name, text in deploy.space_files(profile, final_gguf.name, model_repo).items():
        (hf / "space" / name).write_text(text, encoding="utf-8")

    data_dir = stage_dir / "dataset"
    data_dir.mkdir(exist_ok=True)
    for f in (ctx.ws.merged, ctx.ws.syllabi, ctx.ws.dataset_stats):
        if f.exists():
            shutil.copy2(f, data_dir / f.name)

    guide = deploy.build_guide(ctx, profile, final_gguf.name, f"{slug}-bundle.zip", slug,
                               has_desktop=bool(s.desktop_installer), has_bins=n_bins > 0)
    (stage_dir / "DEPLOY.md").write_text(deploy.render_markdown(guide), encoding="utf-8")
    (stage_dir / "README.txt").write_text(deploy.bundle_readme(profile, final_gguf.name), encoding="utf-8")

    zip_path = ctx.ws.package_dir / f"{slug}-bundle.zip"
    zip_dir(stage_dir, zip_path, f"{slug}-tutor")
    shutil.rmtree(work, ignore_errors=True)
    steps.end("bundle")

    db.update_job(ctx.job_id, bundle_path=str(zip_path))
    ctx.reporter.finish(f"Bundle ready ({zip_path.stat().st_size / GB:.2f} GB)",
                        bundle=zip_path.name, gguf=final_gguf.name,
                        gguf_gb=round(final_gguf.stat().st_size / GB, 2))
