"""
Job worker. Claims queued jobs whose next stage belongs to one of its lanes, runs that stage
in a subprocess, and advances the job.

    python -m corecs_global.worker                  # all lanes (single-machine default)
    python -m corecs_global.worker --lanes io       # API-bound stages: ingest, generate, publish, deploy
    python -m corecs_global.worker --lanes gpu      # GPU stages: train, package

Run one gpu-lane worker per GPU and as many io-lane workers as your API rate limits allow.

Secrets: the worker decrypts the lecturer's credentials from the vault and passes them to the
stage subprocess as environment variables, only for the stages that need them. The teacher
key is deleted once dataset generation succeeds, the Hugging Face token once the model is
published; everything is deleted at a terminal state.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

from . import db, vault
from .config import ROOT, get_settings
from .jobs import STAGE_LANE, STAGE_NAMES, JobConfig

POLL_S = 2.0
CHECK_S = 1.0
TEACHER_STAGES = ("ingest", "generate")
HF_STAGES = ("train", "package", "publish")


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the stage and its children (packaging spawns convert / quantize processes)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def stage_env(job_id: str, stage: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("GLOBAL_TEACHER_API_KEY", None)
    if stage in TEACHER_STAGES:
        key = vault.load(job_id, vault.TEACHER_KEY)
        if key:
            env["GLOBAL_TEACHER_API_KEY"] = key
    if stage in HF_STAGES:
        token = vault.load(job_id, vault.HF_TOKEN)
        if token:
            env["HF_TOKEN"] = token
    return env


def advance(job: dict, stage: str) -> None:
    job_id = job["id"]
    cfg = JobConfig.from_dict(job["config"])
    if stage == "generate":
        vault.forget(job_id, vault.TEACHER_KEY)
    if stage == "publish":
        vault.forget(job_id, vault.HF_TOKEN)
    idx = STAGE_NAMES.index(stage)
    if idx == len(STAGE_NAMES) - 1:
        db.update_job(job_id, status="succeeded", next_stage=None, current_stage=None, worker_id=None)
        vault.forget(job_id)
        db.add_event(job_id, "status", {"status": "succeeded"})
        return
    nxt = STAGE_NAMES[idx + 1]
    if stage == "ingest" and cfg.review_syllabi:
        db.update_job(job_id, status="awaiting_review", next_stage=nxt, worker_id=None)
        db.add_event(job_id, "status", {"status": "awaiting_review",
                                        "message": "Review the extracted courses and topics, then approve."})
        return
    db.update_job(job_id, status="queued", next_stage=nxt, worker_id=None)
    db.add_event(job_id, "status", {"status": "queued", "next_stage": nxt})


def fail(job_id: str, stage: str, status: str, error: str | None) -> None:
    db.update_stage_progress(job_id, stage, {"status": status, "eta_s": None})
    db.update_job(job_id, status=status, worker_id=None, error=error)
    vault.forget(job_id)
    db.add_event(job_id, "status", {"status": status, "stage": stage, "error": error})


def run_stage(job: dict, worker_id: str) -> None:
    s = get_settings()
    job_id, stage = job["id"], job["current_stage"]
    python = s.train_python if STAGE_LANE[stage] == "gpu" else s.platform_python
    log_dir = db.job_dir(job_id) / "logs"
    log_dir.mkdir(exist_ok=True)
    db.add_event(job_id, "stage", {"stage": stage, "status": "starting"})
    print(f"[worker] job {job_id}: {stage} ({python})", flush=True)

    with open(log_dir / f"{stage}.log", "a", encoding="utf-8") as log:
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        proc = subprocess.Popen(
            [python, "-m", "corecs_global.stages.run_stage", job_id, stage],
            cwd=str(ROOT), env=stage_env(job_id, stage), stdout=log, stderr=subprocess.STDOUT, **kwargs,
        )
        try:
            while proc.poll() is None:
                time.sleep(CHECK_S)
                db.heartbeat(job_id)
                current = db.get_job(job_id)
                if current is None or current["cancel_requested"]:
                    kill_tree(proc)
                    fail(job_id, stage, "cancelled", None)
                    print(f"[worker] job {job_id}: cancelled", flush=True)
                    return
        except KeyboardInterrupt:
            kill_tree(proc)
            db.update_job(job_id, status="queued", next_stage=stage, worker_id=None)
            db.add_event(job_id, "log", {"message": "Worker stopped; the job will resume from its last checkpoint."})
            raise

    code = proc.returncode
    if code == 0:
        advance(db.get_job(job_id), stage)
        print(f"[worker] job {job_id}: {stage} done", flush=True)
    else:
        err = (db.get_job(job_id) or {}).get("error") or f"The {stage} stage crashed (exit code {code}); see logs/{stage}.log."
        fail(job_id, stage, "failed", err)
        print(f"[worker] job {job_id}: {stage} FAILED: {err}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="io,gpu", help="comma-separated: io, gpu (default both)")
    ap.add_argument("--once", action="store_true", help="exit when the queue is empty (tests / batch use)")
    args = ap.parse_args()
    lanes = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    if not set(lanes) <= {"io", "gpu"}:
        sys.exit("--lanes must be a subset of io,gpu")
    serve(lanes, once=args.once)


def serve(lanes: tuple[str, ...], once: bool = False) -> None:
    db.init_db()
    worker_id = db.worker_identity()
    stale = db.requeue_stale()
    if stale:
        print(f"[worker] requeued {len(stale)} interrupted job(s): {', '.join(stale)}", flush=True)
    print(f"[worker] {worker_id} serving lanes {lanes}", flush=True)
    while True:
        job = db.claim_next(worker_id, lanes)
        if job is None:
            if once:
                return
            time.sleep(POLL_S)
            continue
        run_stage(job, worker_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] stopped", flush=True)
