"""Shared stage plumbing: the context object, job workspace layout, and small file helpers."""

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .. import db
from ..jobs import JobConfig
from ..progress import Reporter


class StageError(RuntimeError):
    """A failure with a message meant for the lecturer (shown in the UI as-is)."""


@dataclass
class Workspace:
    root: Path

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def sources(self) -> Path:
        return self.root / "sources.json"

    @property
    def pasted(self) -> Path:
        """The pasted syllabus text exactly as submitted, so "Edit & re-run" can show it again."""
        return self.root / "pasted.txt"

    @property
    def publish(self) -> Path:
        return self.root / "publish.json"

    @property
    def syllabi(self) -> Path:
        return self.root / "syllabi.json"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def course_data_dir(self) -> Path:
        return self.dataset_dir / "courses"

    @property
    def merged(self) -> Path:
        return self.dataset_dir / "merged.json"

    @property
    def dataset_stats(self) -> Path:
        return self.dataset_dir / "stats.json"

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def adapter_dir(self) -> Path:
        return self.train_dir / "adapter"

    @property
    def train_meta(self) -> Path:
        return self.train_dir / "train_meta.json"

    @property
    def package_dir(self) -> Path:
        return self.root / "package"

    @property
    def deploy(self) -> Path:
        return self.root / "deploy.json"


@dataclass
class StageContext:
    job_id: str
    stage: str
    config: JobConfig
    ws: Workspace
    reporter: Reporter

    @classmethod
    def for_job(cls, job_id: str, stage: str) -> "StageContext":
        job = db.get_job(job_id)
        if job is None:
            raise StageError(f"job {job_id} not found")
        return cls(
            job_id=job_id,
            stage=stage,
            config=JobConfig.from_dict(job["config"]),
            ws=Workspace(db.job_dir(job_id)),
            reporter=Reporter(job_id, stage),
        )

    def log(self, message: str) -> None:
        self.reporter.log(message)


def slugify(text: str, fallback: str = "item") -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return slug[:60] or fallback


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data) -> None:
    """Write via temp file + os.replace, so a kill mid-write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
