"""
Subprocess entrypoint for one stage of one job:

    python -m corecs_global.stages.run_stage <job_id> <stage>

Each stage runs in its own process so that a cancel can kill it outright, a crash (or a CUDA
driver fault) can't take the worker down, and GPU memory is fully released between stages.
Exit codes: 0 success, 1 expected failure (message stored on the job), 2 unexpected crash.
"""

import sys
import traceback

from .. import db
from . import load_stage
from .base import StageContext, StageError


def main(job_id: str, stage: str) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except AttributeError:
        pass
    try:
        ctx = StageContext.for_job(job_id, stage)
        load_stage(stage).run(ctx)
        return 0
    except StageError as e:
        db.update_job(job_id, error=str(e))
        db.add_event(job_id, "log", {"stage": stage, "level": "error", "message": str(e)})
        print(f"[{stage}] ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - report anything else, with the traceback in the log
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        msg = f"Unexpected error in {stage}: {type(e).__name__}: {e}"
        db.update_job(job_id, error=msg)
        db.add_event(job_id, "log", {"stage": stage, "level": "error", "message": msg, "detail": tb[-4000:]})
        return 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m corecs_global.stages.run_stage <job_id> <stage>")
    sys.exit(main(sys.argv[1], sys.argv[2]))
