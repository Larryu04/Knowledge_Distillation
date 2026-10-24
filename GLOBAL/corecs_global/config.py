"""
Platform settings, read once from environment variables (see .env.example).

Everything a lecturer can choose per job lives in JobConfig (jobs.py); this module only holds
operator-level settings: where data lives, which Python runs the GPU stages, where the
llama.cpp toolchain is, and which teacher models are allowed.
"""

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The teacher (synthetic-data) model is pinned. Larger models are deliberately NOT offered:
# they cost ~15x more per token and hit much lower tokens-per-minute limits on typical
# accounts, which turns a 1-hour synthesis into a multi-hour crawl of 429 retries.
ALLOWED_TEACHER_MODELS = ("gpt-4o-mini",)
DEFAULT_TEACHER_MODEL = "gpt-4o-mini"

# Approximate list prices (USD per 1M tokens) used ONLY for the cost estimate shown in the
# UI before a job starts. Override with GLOBAL_PRICE_IN / GLOBAL_PRICE_OUT if they change.
DEFAULT_PRICE_IN_PER_M = 0.15
DEFAULT_PRICE_OUT_PER_M = 0.60

# Base models offered in the UI. Any Hugging Face causal-LM with a chat template also works
# via the "custom" field; these are the ones known to fit QLoRA on a 12 GB card.
BASE_MODELS = (
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen 2.5 7B Instruct (multilingual)", "gated": False},
    {"id": "dicta-il/dictalm2.0-instruct", "label": "DictaLM 2.0 Instruct 7B (Hebrew-first)", "gated": False},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B Instruct v0.3", "gated": True},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B Instruct", "gated": True},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen 2.5 1.5B Instruct (fast test runs)", "gated": False},
)

QUANT_TYPES = ("Q4_K_M", "Q5_K_M", "Q8_0")


def _path_env(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser().resolve() if val else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    toolchain_dir: Path            # contains llama.cpp-src/ and llama.cpp-bin/ (scripts/setup_toolchain.py)
    runtime_dir: Path              # portable runtime + desktop app sources shipped in bundles
    train_python: str              # interpreter for train/package stages (needs torch + CUDA)
    platform_python: str           # interpreter for ingest/generate stages
    access_token: str | None       # if set, every /api call must present it
    teacher_backend: str           # "openai" (default) or "fake" (offline tests / demos)
    price_in_per_m: float
    price_out_per_m: float
    max_upload_mb: int
    allowed_teacher_models: tuple = field(default=ALLOWED_TEACHER_MODELS)

    @property
    def desktop_installer(self) -> Path | None:
        """Looked up on every use, so a newly dropped installer is served without a restart."""
        return _find_desktop_installer(self.runtime_dir)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "global.db"

    @property
    def secret_key_path(self) -> Path:
        return self.data_dir / "secret.key"

    @property
    def llama_src(self) -> Path:
        return self.toolchain_dir / "llama.cpp-src"

    @property
    def llama_bin(self) -> Path:
        return self.toolchain_dir / "llama.cpp-bin"


def _find_desktop_installer(runtime_dir: Path) -> Path | None:
    """GLOBAL_DESKTOP_INSTALLER, else the newest *-setup.exe in runtime/installer/ (the drop
    folder for a prebuilt installer), else the newest one from a local runtime/desktop build."""
    explicit = os.environ.get("GLOBAL_DESKTOP_INSTALLER")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    nsis = runtime_dir / "desktop" / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    for folder in (runtime_dir / "installer", nsis):
        found = sorted(folder.glob("*-setup.exe"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
        if found:
            return found[-1]
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime_dir = _path_env("GLOBAL_RUNTIME_DIR", ROOT / "runtime")
    data_dir = _path_env("GLOBAL_DATA_DIR", ROOT / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        toolchain_dir=_path_env("GLOBAL_TOOLCHAIN_DIR", ROOT / "tools"),
        runtime_dir=runtime_dir,
        train_python=os.environ.get("GLOBAL_TRAIN_PYTHON") or sys.executable,
        platform_python=os.environ.get("GLOBAL_PLATFORM_PYTHON") or sys.executable,
        access_token=os.environ.get("GLOBAL_ACCESS_TOKEN") or None,
        teacher_backend=os.environ.get("GLOBAL_TEACHER_BACKEND", "openai"),
        price_in_per_m=float(os.environ.get("GLOBAL_PRICE_IN", DEFAULT_PRICE_IN_PER_M)),
        price_out_per_m=float(os.environ.get("GLOBAL_PRICE_OUT", DEFAULT_PRICE_OUT_PER_M)),
        max_upload_mb=int(os.environ.get("GLOBAL_MAX_UPLOAD_MB", 50)),
    )
