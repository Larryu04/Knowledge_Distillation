"""
Job model: the lecturer-facing configuration, the ordered pipeline stages, and the up-front
time / cost estimates shown before (and, for stages not yet started, during) a run.
"""

import math
import re
from dataclasses import asdict, dataclass, fields

from .config import ALLOWED_TEACHER_MODELS, DEFAULT_TEACHER_MODEL, QUANT_TYPES, get_settings

# (stage name, lane). A worker only claims jobs whose next stage is in one of its lanes, so
# API-bound synthesis for one lecturer never waits behind hours of GPU training for another.
STAGES = (
    ("ingest", "io"),
    ("generate", "io"),
    ("train", "gpu"),
    ("package", "gpu"),
    ("deploy", "io"),
)
STAGE_NAMES = tuple(s for s, _ in STAGES)
STAGE_LANE = dict(STAGES)
STAGE_LABELS = {
    "ingest": "Syllabus ingestion",
    "generate": "Dataset generation",
    "train": "Fine-tuning",
    "package": "Packaging & GGUF quantization",
    "deploy": "Deployment guide",
}

TERMINAL = ("succeeded", "failed", "cancelled")


class ConfigError(ValueError):
    pass


@dataclass
class JobConfig:
    program_name: str
    language: str = "English"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    teacher_model: str = DEFAULT_TEACHER_MODEL
    examples_per_course: int = 400
    review_syllabi: bool = True
    epochs: int = 2
    lora_r: int = 16
    learning_rate: float = 1e-4
    max_seq_len: int = 1024
    quant: str = "Q4_K_M"
    include_adapter: bool = True
    generation_concurrency: int = 4
    display_name: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "JobConfig":
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        self.program_name = (self.program_name or "").strip()
        if not self.program_name:
            raise ConfigError("Program name is required.")
        if self.teacher_model not in ALLOWED_TEACHER_MODELS:
            raise ConfigError(f"Teacher model must be one of {ALLOWED_TEACHER_MODELS}.")
        if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", self.base_model or ""):
            raise ConfigError("Base model must be a Hugging Face id like 'org/model'.")
        if self.quant not in QUANT_TYPES:
            raise ConfigError(f"Quantization must be one of {QUANT_TYPES}.")
        checks = [
            ("examples_per_course", 20, 3000),
            ("epochs", 1, 5),
            ("lora_r", 4, 64),
            ("max_seq_len", 256, 4096),
            ("generation_concurrency", 1, 16),
        ]
        for name, lo, hi in checks:
            val = int(getattr(self, name))
            if not lo <= val <= hi:
                raise ConfigError(f"{name} must be between {lo} and {hi}.")
            setattr(self, name, val)
        self.learning_rate = float(self.learning_rate)
        if not 1e-6 <= self.learning_rate <= 1e-3:
            raise ConfigError("learning_rate must be between 1e-6 and 1e-3.")
        self.language = (self.language or "English").strip()
        self.display_name = (self.display_name or "").strip() or f"{self.program_name} Tutor"


# --------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------
TOKENS_IN_PER_EXAMPLE = 1400       # system prompt + topic + recent-instruction sample
TOKENS_OUT_PER_EXAMPLE = 700
RETRY_FACTOR = 1.35                # content-validation retries, measured on the 23-course run
SECONDS_PER_EXAMPLE = 7.0          # one gpt-4o-mini call incl. pacing, per concurrent lane
INGEST_TOKENS_IN_PER_SOURCE = 15000
INGEST_TOKENS_OUT_PER_SOURCE = 3000
EFFECTIVE_BATCH = 16


def model_size_b(model_id: str) -> float:
    """Best-effort parameter count (billions) from the model id, default 7."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id)
    return float(m.group(1)) if m else 7.0


def estimate(cfg: JobConfig, n_courses: int, n_sources: int) -> dict:
    s = get_settings()
    n_courses = max(1, n_courses)
    rows = n_courses * cfg.examples_per_course
    size = model_size_b(cfg.base_model)

    gen_lanes = min(cfg.generation_concurrency, n_courses)
    steps = math.ceil(rows * 0.95 / EFFECTIVE_BATCH) * cfg.epochs
    sec_per_step = 1.2 * size + 0.5            # ~9 s/step for 7B on an RTX 4070-class GPU
    seconds = {
        "ingest": 20 + 15 * max(1, n_sources),
        "generate": rows * SECONDS_PER_EXAMPLE * RETRY_FACTOR / gen_lanes,
        "train": 300 + steps * sec_per_step,
        "package": 120 + 160 * size,
        "deploy": 2,
    }

    tok_in = rows * TOKENS_IN_PER_EXAMPLE * RETRY_FACTOR + n_sources * INGEST_TOKENS_IN_PER_SOURCE
    tok_out = rows * TOKENS_OUT_PER_EXAMPLE * RETRY_FACTOR + n_sources * INGEST_TOKENS_OUT_PER_SOURCE
    cost = tok_in / 1e6 * s.price_in_per_m + tok_out / 1e6 * s.price_out_per_m
    return {
        "rows": rows,
        "train_steps": steps,
        "stage_seconds": {k: round(v) for k, v in seconds.items()},
        "total_seconds": round(sum(seconds.values())),
        "api_cost_usd": round(cost, 2),
        "gguf_gb": round(size * 0.62, 1) if cfg.quant == "Q4_K_M" else round(size * (0.72 if cfg.quant == "Q5_K_M" else 1.07), 1),
    }
