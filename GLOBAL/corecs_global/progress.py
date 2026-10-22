"""
Stage progress reporting and ETA.

A stage calls reporter.update(done, total, message) as often as it likes; writes to SQLite
are throttled. The stage ETA is a smoothed rate (EMA of seconds per unit) so a slow first
API call or a checkpoint save doesn't make the estimate jump around.

The overall ETA (pipeline_eta) adds, for stages not started yet, the up-front estimate from
jobs.estimate(), which is refreshed with the real course count after ingestion.
"""

import time

from . import db
from .jobs import STAGE_NAMES

WRITE_INTERVAL_S = 1.0
EMA_ALPHA = 0.15


class Reporter:
    def __init__(self, job_id: str, stage: str):
        self.job_id = job_id
        self.stage = stage
        self._started = time.time()
        self._last_write = 0.0
        self._last_done = None
        self._last_t = None
        self._sec_per_unit = None
        self._resume_base = None

    # ------------------------------------------------------------------
    def start(self, message: str = "") -> None:
        db.update_stage_progress(self.job_id, self.stage, {
            "status": "running", "started_at": self._started, "done": 0, "total": None,
            "fraction": 0.0, "eta_s": None, "message": message,
        })
        db.add_event(self.job_id, "stage", {"stage": self.stage, "status": "running", "message": message})

    def update(self, done: float, total: float, message: str = "", force: bool = False) -> None:
        now = time.time()
        if self._resume_base is None:
            self._resume_base = done          # resumed work doesn't count toward the rate
        if self._last_done is not None and done > self._last_done:
            inst = (now - self._last_t) / (done - self._last_done)
            self._sec_per_unit = inst if self._sec_per_unit is None else (
                EMA_ALPHA * inst + (1 - EMA_ALPHA) * self._sec_per_unit)
        if self._last_done is None or done != self._last_done:
            self._last_done, self._last_t = done, now

        if not force and now - self._last_write < WRITE_INTERVAL_S:
            return
        self._last_write = now
        eta = None
        if self._sec_per_unit is not None and total:
            eta = max(0.0, (total - done) * self._sec_per_unit)
        elif total and done > self._resume_base:
            eta = (now - self._started) / (done - self._resume_base) * (total - done)
        fraction = min(1.0, done / total) if total else 0.0
        db.update_stage_progress(self.job_id, self.stage, {
            "done": done, "total": total, "fraction": round(fraction, 4),
            "eta_s": round(eta) if eta is not None else None, "message": message,
        })
        db.heartbeat(self.job_id)

    def log(self, message: str) -> None:
        print(f"[{self.stage}] {message}", flush=True)
        db.add_event(self.job_id, "log", {"stage": self.stage, "message": message})

    def finish(self, message: str = "done", **extra) -> None:
        db.update_stage_progress(self.job_id, self.stage, {
            "status": "done", "fraction": 1.0, "eta_s": 0, "message": message,
            "finished_at": time.time(), **extra,
        })
        db.add_event(self.job_id, "stage", {"stage": self.stage, "status": "done", "message": message})


def pipeline_eta(job: dict) -> int | None:
    """Seconds until the whole pipeline is done, or None if it can't be estimated."""
    if job["status"] in ("succeeded", "failed", "cancelled"):
        return 0 if job["status"] == "succeeded" else None
    if job["status"] == "awaiting_review":
        return None
    est = (job.get("estimate") or {}).get("stage_seconds", {})
    total = 0.0
    for stage in STAGE_NAMES:
        p = job["progress"].get(stage, {})
        status = p.get("status", "pending")
        if status == "done":
            continue
        if status == "running":
            if p.get("eta_s") is not None:
                total += p["eta_s"]
            else:
                elapsed = time.time() - p.get("started_at", time.time())
                total += max(0.0, est.get(stage, 0) - elapsed)
        else:
            total += est.get(stage, 0)
    return round(total)


def overall_fraction(job: dict) -> float:
    """Time-weighted completion across all stages (0..1)."""
    est = (job.get("estimate") or {}).get("stage_seconds", {})
    weights = {s: max(1, est.get(s, 1)) for s in STAGE_NAMES}
    total = sum(weights.values())
    done = 0.0
    for s in STAGE_NAMES:
        p = job["progress"].get(s, {})
        if p.get("status") == "done":
            done += weights[s]
        elif p.get("status") == "running":
            done += weights[s] * float(p.get("fraction") or 0)
    return round(done / total, 4)
