"""
Pipeline stages. Each module exposes run(ctx: StageContext) and must be idempotent /
resumable: a stage can be killed at any point (cancel, crash, worker restart) and re-run.

    ingest   -> syllabi.json            (files/pasted text -> structured courses + topics)
    generate -> dataset/merged.json     (teacher LLM -> validated Q&A rows per course)
    train    -> train/adapter/          (QLoRA on the base model)
    package  -> package/<name>.zip      (merge -> GGUF -> quantize -> runtime bundle)
    deploy   -> deploy.json             (filled-in hosting & sharing instructions)
"""

import importlib


def load_stage(name: str):
    return importlib.import_module(f"{__name__}.{name}")
