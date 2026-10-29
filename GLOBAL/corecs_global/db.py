"""
SQLite persistence + job queue.

One file, WAL mode, safe for one API process and several worker processes on the same host.
A job moves through the stages in jobs.STAGES; between stages it goes back to
status='queued' with next_stage set, so a worker serving that stage's lane can claim it.

    queued(next_stage=ingest) -> running -> [awaiting_review] -> queued(generate) -> running
    -> queued(train) -> ... -> succeeded | failed | cancelled
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .config import get_settings
from .jobs import STAGE_LANE, STAGE_NAMES

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    status           TEXT NOT NULL,          -- queued|running|awaiting_review|succeeded|failed|cancelled
    next_stage       TEXT,                   -- stage to run when status='queued'
    current_stage    TEXT,
    config           TEXT NOT NULL,          -- JobConfig JSON
    estimate         TEXT NOT NULL,          -- jobs.estimate() JSON, refreshed after ingestion
    progress         TEXT NOT NULL DEFAULT '{}',  -- {stage: {status, done, total, eta_s, message, ...}}
    error            TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_id        TEXT,
    heartbeat        REAL,
    bundle_path      TEXT,
    replaced_by      TEXT                    -- set when the lecturer edited this job and re-ran it as a new one
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, next_stage, created_at);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,                  -- log|stage|status|progress
    payload  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, id);

CREATE TABLE IF NOT EXISTS secrets (
    job_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    ciphertext BLOB NOT NULL,
    PRIMARY KEY (job_id, name)
);
"""

JSON_COLS = ("config", "estimate", "progress")


def _connect(path=None) -> sqlite3.Connection:
    path = path or get_settings().db_path
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; explicit BEGIN when needed
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connect():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
        if "replaced_by" not in cols:                               # databases created before edit & re-run
            c.execute("ALTER TABLE jobs ADD COLUMN replaced_by TEXT")


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    return d


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------
def create_job(config: dict, estimate: dict) -> str:
    """Created as 'staging' (invisible to workers) so uploads can be written first; call
    enqueue() once the job's inputs are complete."""
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    progress = {s: {"status": "pending"} for s in STAGE_NAMES}
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id, created_at, updated_at, status, next_stage, config, estimate, progress) "
            "VALUES (?, ?, ?, 'staging', ?, ?, ?, ?)",
            (job_id, now, now, STAGE_NAMES[0], json.dumps(config, ensure_ascii=False),
             json.dumps(estimate), json.dumps(progress)),
        )
    return job_id


def enqueue(job_id: str) -> None:
    update_job(job_id, status="queued")
    add_event(job_id, "status", {"status": "queued"})


def delete_job(job_id: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM secrets WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def get_job(job_id: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def list_jobs(limit: int = 50, include_replaced: bool = False) -> list[dict]:
    where = "" if include_replaced else "WHERE replaced_by IS NULL "
    with connect() as c:
        rows = c.execute(f"SELECT * FROM jobs {where}ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def active_jobs() -> list[dict]:
    """Jobs not yet in a terminal state (staging, queued, running, awaiting review)."""
    with connect() as c:
        rows = c.execute("SELECT * FROM jobs WHERE status NOT IN ('succeeded', 'failed', 'cancelled')").fetchall()
    return [_row(r) for r in rows]


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    for k in JSON_COLS:
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k], ensure_ascii=False)
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def update_stage_progress(job_id: str, stage: str, patch: dict) -> None:
    """Merge `patch` into progress[stage] atomically (workers and the API both write here)."""
    with connect() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT progress FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                c.execute("ROLLBACK")
                return
            progress = json.loads(row["progress"])
            progress.setdefault(stage, {}).update(patch)
            c.execute("UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
                      (json.dumps(progress, ensure_ascii=False), time.time(), job_id))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise


def claim_next(worker_id: str, lanes: tuple[str, ...]) -> dict | None:
    """Atomically move the oldest queued job whose next stage is in `lanes` to running."""
    stages = [s for s in STAGE_NAMES if STAGE_LANE[s] in lanes]
    if not stages:
        return None
    marks = ",".join("?" * len(stages))
    now = time.time()
    with connect() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                f"SELECT id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 "
                f"AND next_stage IN ({marks}) ORDER BY created_at LIMIT 1",
                stages,
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            c.execute(
                "UPDATE jobs SET status = 'running', current_stage = next_stage, worker_id = ?, "
                "heartbeat = ?, updated_at = ?, error = NULL WHERE id = ?",
                (worker_id, now, now, row["id"]),
            )
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    return get_job(row["id"])


def heartbeat(job_id: str) -> None:
    with connect() as c:
        c.execute("UPDATE jobs SET heartbeat = ? WHERE id = ?", (time.time(), job_id))


def requeue_stale(max_age_s: float = 120) -> list[str]:
    """Jobs left 'running' by a worker that died (no heartbeat) go back to the queue at the
    same stage. Every stage checkpoints, so the retry resumes instead of starting over."""
    cutoff = time.time() - max_age_s
    with connect() as c:
        rows = c.execute(
            "SELECT id FROM jobs WHERE status = 'running' AND (heartbeat IS NULL OR heartbeat < ?)",
            (cutoff,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        for job_id in ids:
            c.execute(
                "UPDATE jobs SET status = 'queued', next_stage = current_stage, worker_id = NULL "
                "WHERE id = ? AND status = 'running'",
                (job_id,),
            )
    for job_id in ids:
        add_event(job_id, "log", {"message": "Worker restarted; resuming from the last checkpoint."})
    return ids


# --------------------------------------------------------------------------------------
# Events (progress feed for the UI's live view)
# --------------------------------------------------------------------------------------
def add_event(job_id: str, kind: str, payload: dict) -> None:
    with connect() as c:
        c.execute("INSERT INTO events (job_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
                  (job_id, time.time(), kind, json.dumps(payload, ensure_ascii=False)))


def events_after(job_id: str, after_id: int = 0, limit: int = 500) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, ts, kind, payload FROM events WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
            (job_id, after_id, limit),
        ).fetchall()
    return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"], **json.loads(r["payload"])} for r in rows]


def tail_events(job_id: str, limit: int = 200) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, ts, kind, payload FROM events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"], **json.loads(r["payload"])} for r in reversed(rows)]


# --------------------------------------------------------------------------------------
# Secrets (ciphertext only; see vault.py)
# --------------------------------------------------------------------------------------
def put_secret(job_id: str, name: str, ciphertext: bytes) -> None:
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO secrets (job_id, name, ciphertext) VALUES (?, ?, ?)",
                  (job_id, name, ciphertext))


def get_secret(job_id: str, name: str) -> bytes | None:
    with connect() as c:
        row = c.execute("SELECT ciphertext FROM secrets WHERE job_id = ? AND name = ?",
                        (job_id, name)).fetchone()
    return row["ciphertext"] if row else None


def delete_secrets(job_id: str, name: str | None = None) -> None:
    with connect() as c:
        if name:
            c.execute("DELETE FROM secrets WHERE job_id = ? AND name = ?", (job_id, name))
        else:
            c.execute("DELETE FROM secrets WHERE job_id = ?", (job_id,))


def job_dir(job_id: str):
    d = get_settings().jobs_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def worker_identity() -> str:
    return f"{os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'host')}:{os.getpid()}"
