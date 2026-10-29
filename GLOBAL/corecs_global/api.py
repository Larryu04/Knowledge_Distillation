"""
HTTP API + static web UI.

    uvicorn corecs_global.api:app --host 127.0.0.1 --port 8000

The API never runs pipeline work itself: it validates input, stores uploads and encrypted
credentials, enqueues the job, and serves progress from SQLite. Workers do the rest.
"""

import asyncio
import json
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, db, hf_publish, student_app, vault
from .config import BASE_MODELS, QUANT_TYPES, get_settings
from .jobs import STAGE_LABELS, STAGE_LANE, STAGE_NAMES, TERMINAL, ConfigError, JobConfig, estimate
from .languages import PRESETS
from .llm.teacher import FatalAPIError, make_teacher
from .progress import overall_fraction, pipeline_eta
from .stages.base import Workspace, atomic_write_json, read_json, slugify
from .stages.ingest import MIN_TOPICS, dedupe_courses, normalize_courses
from .worker import TEACHER_STAGES

WEB_DIR = Path(__file__).resolve().parent / "web"
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
PASTE_SEPARATOR = re.compile(r"^\s*-{3,}\s*$", re.M)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="CoreCS Global", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_ORIGIN = re.compile(r"https?://[^/\s,\[\]()<>\"']+", re.I)


def cors_origins(raw: str) -> list[str]:
    """Parse GLOBAL_CORS_ORIGINS. Browsers send a bare origin (scheme://host[:port]), and the match
    is exact, so reduce each entry to that: drops paths, trailing slashes, quotes, and recovers a
    URL pasted as a Markdown link. Entries with no URL in them are reported and skipped."""
    origins = []
    for item in filter(None, (s.strip() for s in raw.split(","))):
        if item == "*":
            origins.append(item)
        elif m := _ORIGIN.search(item):
            origins.append(m.group(0).lower())
        else:
            print(f"WARNING: GLOBAL_CORS_ORIGINS entry {item!r} is not a URL; ignored.", flush=True)
    return list(dict.fromkeys(origins))


# The web UI may be hosted elsewhere (e.g. Vercel) and call this API cross-origin.
# GLOBAL_CORS_ORIGINS: comma-separated allowed origins; "*" (default) allows any.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(os.environ.get("GLOBAL_CORS_ORIGINS", "*")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Access control (optional, single shared token). EventSource can't send headers, so the
# token is also accepted as a cookie or ?token= query parameter.
# --------------------------------------------------------------------------------------
def require_access(request: Request) -> None:
    token = get_settings().access_token
    if not token:
        return
    auth = request.headers.get("authorization", "")
    supplied = (auth[7:] if auth.lower().startswith("bearer ") else None) \
        or request.cookies.get("global_token") or request.query_params.get("token")
    if supplied != token:
        raise HTTPException(401, "Access token required.")


def _job_or_404(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "Job not found.")
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


def job_view(job: dict) -> dict:
    stages = []
    for name in STAGE_NAMES:
        p = dict(job["progress"].get(name, {}))
        if not p and job["status"] == "succeeded":
            p["status"] = "skipped"        # a stage added after this job ran (e.g. publish)
        p.update(name=name, label=STAGE_LABELS[name], lane=STAGE_LANE[name])
        stages.append(p)
    bundle = Path(job["bundle_path"]) if job.get("bundle_path") else None
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "current_stage": job["current_stage"],
        "next_stage": job["next_stage"],
        "error": job["error"],
        "cancel_requested": bool(job["cancel_requested"]),
        "config": job["config"],
        "estimate": job["estimate"],
        "stages": stages,
        "eta_s": pipeline_eta(job),
        "fraction": overall_fraction(job),
        "replaced_by": job.get("replaced_by"),
        "bundle": {"name": bundle.name, "bytes": bundle.stat().st_size} if bundle and bundle.exists() else None,
    }


# --------------------------------------------------------------------------------------
# UI + metadata
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/config.js")
def web_config():
    """Served alongside the UI: tells app.js the API is same-origin (Vercel builds its own)."""
    return Response('window.GLOBAL_API_URL = "";\n', media_type="text/javascript",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/options", dependencies=[Depends(require_access)])
def options():
    s = get_settings()
    return {
        "version": __version__,
        "base_models": BASE_MODELS,
        "languages": list(PRESETS),
        "quant_types": QUANT_TYPES,
        "teacher_models": s.allowed_teacher_models,
        "defaults": JobConfig(program_name="-").to_dict(),
        "max_upload_mb": s.max_upload_mb,
        # Forced demo (GLOBAL_TEACHER_BACKEND=fake); otherwise each job chooses with JobConfig.demo.
        "demo_mode": s.teacher_backend == "fake",
        "desktop_installer": s.desktop_installer.name if s.desktop_installer else None,
        "stages": [{"name": n, "label": STAGE_LABELS[n]} for n in STAGE_NAMES],
    }


@app.post("/api/estimate", dependencies=[Depends(require_access)])
async def estimate_endpoint(request: Request):
    body = await request.json()
    try:
        cfg = JobConfig.from_dict({**body.get("config", {}), "program_name": body.get("config", {}).get("program_name") or "-"})
    except (ConfigError, TypeError, ValueError) as e:
        raise HTTPException(400, str(e))
    n_sources = max(1, int(body.get("n_sources", 1)))
    return estimate(cfg, int(body.get("n_courses") or n_sources), n_sources)


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------
# One GPU: a single real job may be in flight at a time (a second one would sit in the queue for
# hours behind the first), plus a few short demo jobs. Checked and claimed under this lock so two
# simultaneous submissions can't both slip past the check.
_admission = threading.Lock()


def _check_capacity(demo: bool, exclude: str | None = None) -> None:
    active = [j for j in db.active_jobs() if j["id"] != exclude]
    if demo:
        limit = get_settings().max_demo_jobs
        if sum(1 for j in active if j["config"].get("demo")) >= limit:
            raise HTTPException(429, f"{limit} demo runs are already in progress on this server. "
                                     "Try again when one of them finishes.")
    elif any(not j["config"].get("demo") for j in active):
        raise HTTPException(409, "Another lecturer's tutor is already being built on this server, and only one "
                                 "real run can use the GPU at a time. Try again when it finishes, or turn on "
                                 "Demo mode to try the pipeline now.")


def _verify_key(api_key: str, model: str) -> None:
    try:
        make_teacher(api_key, model).verify()
    except FatalAPIError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 - network trouble etc.; don't enqueue a doomed job
        raise HTTPException(502, f"Could not verify the API key with the provider: {type(e).__name__}.")


def _hf_user(token: str) -> str:
    try:
        return hf_publish.check_token(token)
    except hf_publish.PublishError as e:
        raise HTTPException(400, str(e))


def _save_sources(ws: Workspace, files: list[UploadFile], pasted: str, max_bytes: int,
                  kept: list[tuple[str, Path]] = ()) -> list[dict]:
    """`kept` are (original name, path) pairs of files carried over from the job being re-run."""
    ws.uploads.mkdir(parents=True, exist_ok=True)
    sources = []
    for i, (name, path) in enumerate(kept):
        stored = f"{i:02d}-{slugify(Path(name).stem, 'file')}{path.suffix.lower()}"
        shutil.copyfile(path, ws.uploads / stored)
        sources.append({"name": name, "file": stored})
    for i, f in enumerate(files or [], len(kept)):
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"{f.filename}: unsupported type (use PDF, DOCX, TXT or MD).")
        stored = f"{i:02d}-{slugify(Path(f.filename).stem, 'file')}{suffix}"
        with open(ws.uploads / stored, "wb") as out:
            size = 0
            while chunk := f.file.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"{f.filename} is larger than {max_bytes >> 20} MB.")
                out.write(chunk)
        sources.append({"name": f.filename, "file": stored})
    blocks = [b.strip() for b in PASTE_SEPARATOR.split(pasted or "") if b.strip()]
    for i, block in enumerate(blocks, 1):
        stored = f"pasted-{i:02d}.txt"
        (ws.uploads / stored).write_text(block, encoding="utf-8")
        sources.append({"name": f"Pasted text {i}", "file": stored})
    if pasted.strip():
        ws.pasted.write_text(pasted, encoding="utf-8")
    return sources


def _uploaded_files(job_id: str) -> list[dict]:
    """Uploaded (not pasted) files of a job, as {name, file, bytes}."""
    ws = Workspace(db.job_dir(job_id))
    out = []
    for src in read_json(ws.sources, default=[]):
        path = ws.uploads / src["file"]
        if not src["file"].startswith("pasted-") and path.exists():
            out.append({"name": src["name"], "file": src["file"], "bytes": path.stat().st_size})
    return out


def _pasted_text(job_id: str) -> str:
    ws = Workspace(db.job_dir(job_id))
    if ws.pasted.exists():
        return ws.pasted.read_text(encoding="utf-8")
    # Jobs created before pasted.txt existed: rebuild from the stored blocks.
    blocks = sorted(ws.uploads.glob("pasted-*.txt")) if ws.uploads.exists() else []
    return "\n---\n".join(b.read_text(encoding="utf-8") for b in blocks)


@app.post("/api/jobs", dependencies=[Depends(require_access)])
def create_job(
    config: str = Form(...),
    api_key: str = Form(""),
    hf_token: str = Form(""),
    pasted_text: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    retry_of: str = Form(""),
    keep_files: str = Form("[]"),
):
    """`retry_of` makes this an "Edit & re-run" of a failed/cancelled job: the new job may reuse
    that job's uploaded files listed in `keep_files` (JSON list of stored names), and the old job
    is hidden from the job list once the new one is queued."""
    s = get_settings()
    try:
        cfg = JobConfig.from_dict(json.loads(config))
    except (ConfigError, TypeError, ValueError) as e:
        raise HTTPException(400, str(e))
    kept: list[tuple[str, Path]] = []
    if retry_of:
        old = _job_or_404(retry_of)
        if old["status"] not in ("failed", "cancelled"):
            raise HTTPException(409, "Only failed or cancelled jobs can be edited and re-run.")
        if old.get("replaced_by"):
            raise HTTPException(409, f"This job was already re-run as job {old['replaced_by']}.")
        try:
            wanted = set(json.loads(keep_files))
        except (TypeError, ValueError):
            raise HTTPException(400, "keep_files must be a JSON list.")
        old_uploads = Workspace(db.job_dir(retry_of)).uploads
        kept = [(f["name"], old_uploads / f["file"]) for f in _uploaded_files(retry_of) if f["file"] in wanted]
    cfg.demo = cfg.demo or s.teacher_backend == "fake"
    _check_capacity(cfg.demo, exclude=retry_of or None)   # fail fast, before verifying keys
    api_key = api_key.strip()
    if not cfg.demo:
        if not api_key:
            raise HTTPException(400, "An API key for the teacher model is required.")
        _verify_key(api_key, cfg.teacher_model)
    hf_token = hf_token.strip()
    cfg.hf_repo = ""                       # server-assigned, from the token's account
    if hf_token:
        cfg.hf_repo = hf_publish.repo_for(_hf_user(hf_token), slugify(cfg.program_name, fallback="tutor"))
    elif not cfg.demo:
        raise HTTPException(400, "A Hugging Face token with Write access is required: the server publishes "
                                 f"your model to your account when training finishes. {hf_publish.TOKEN_HELP}")

    n_inputs = len(kept) + len([f for f in files if f.filename]) + len([b for b in PASTE_SEPARATOR.split(pasted_text) if b.strip()])
    if n_inputs == 0:
        raise HTTPException(400, "Upload at least one syllabus file or paste syllabus text.")

    with _admission:
        _check_capacity(cfg.demo, exclude=retry_of or None)
        job_id = db.create_job(cfg.to_dict(), estimate(cfg, n_inputs, n_inputs))
    ws = Workspace(db.job_dir(job_id))
    try:
        sources = _save_sources(ws, files, pasted_text, s.max_upload_mb << 20, kept)
    except HTTPException:
        db.delete_job(job_id)
        shutil.rmtree(ws.root, ignore_errors=True)
        raise
    atomic_write_json(ws.sources, sources)
    if api_key:
        vault.store(job_id, vault.TEACHER_KEY, api_key)
    if hf_token:
        vault.store(job_id, vault.HF_TOKEN, hf_token)
    if retry_of:
        db.add_event(job_id, "log", {"message": f"Re-run of job {retry_of} with edited inputs."})
        db.update_job(retry_of, replaced_by=job_id)
        db.add_event(retry_of, "log", {"message": f"Edited and re-run as job {job_id}."})
    db.enqueue(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}/inputs", dependencies=[Depends(require_access)])
def get_inputs(job_id: str):
    """Everything needed to pre-fill the setup form for "Edit & re-run" (credentials excluded)."""
    job = _job_or_404(job_id)
    return {"config": job["config"], "pasted_text": _pasted_text(job_id), "files": _uploaded_files(job_id),
            "status": job["status"], "error": job["error"], "replaced_by": job.get("replaced_by")}


@app.get("/api/jobs", dependencies=[Depends(require_access)])
def list_jobs(ids: str = "", admin: str = ""):
    """No accounts: each browser remembers the ids of the jobs it created and asks only for those
    (`ids`, comma-separated). The admin key (GLOBAL_ADMIN_KEY) lists every job on the server."""
    if admin and admin == get_settings().admin_key:
        return [job_view(j) for j in db.list_jobs()]
    wanted = list(dict.fromkeys(i for i in ids.split(",") if re.fullmatch(r"[0-9a-f]{12}", i)))[:200]
    jobs = [j for j in map(db.get_job, wanted) if j and not j.get("replaced_by")]
    return [job_view(j) for j in sorted(jobs, key=lambda j: j["created_at"], reverse=True)]


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_access)])
def get_job(job_id: str):
    return job_view(_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/events", dependencies=[Depends(require_access)])
def get_events(job_id: str, after: int = 0):
    _job_or_404(job_id)
    return db.events_after(job_id, after) if after else db.tail_events(job_id)


@app.get("/api/jobs/{job_id}/stream", dependencies=[Depends(require_access)])
async def stream(job_id: str, request: Request, after: int = 0):
    """Server-Sent Events: `snapshot` (full job view, when it changes) and `event` (log lines)."""
    _job_or_404(job_id)

    async def gen():
        last_id, last_update, last_sent = after, None, 0.0
        if not after:
            tail = db.tail_events(job_id, 200)
            for ev in tail:
                yield f"event: event\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            last_id = tail[-1]["id"] if tail else 0
        while not await request.is_disconnected():
            job = db.get_job(job_id)
            if job is None:
                return
            for ev in db.events_after(job_id, last_id):
                last_id = ev["id"]
                yield f"event: event\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            now = time.time()
            # Re-send at least every 5 s so the ETA countdown stays fresh between DB writes.
            if job["updated_at"] != last_update or now - last_sent > 5:
                last_update, last_sent = job["updated_at"], now
                yield f"event: snapshot\ndata: {json.dumps(job_view(job), ensure_ascii=False)}\n\n"
            if job["status"] in TERMINAL:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/syllabi", dependencies=[Depends(require_access)])
def get_syllabi(job_id: str):
    _job_or_404(job_id)
    data = read_json(Workspace(db.job_dir(job_id)).syllabi)
    if data is None:
        raise HTTPException(404, "Syllabi have not been extracted yet.")
    return data


@app.put("/api/jobs/{job_id}/syllabi", dependencies=[Depends(require_access)])
async def put_syllabi(job_id: str, request: Request):
    job = _job_or_404(job_id)
    if job["status"] != "awaiting_review":
        raise HTTPException(409, "Courses can only be edited while the job is awaiting review.")
    body = await request.json()
    courses = dedupe_courses(normalize_courses(body.get("courses"), "edited"))
    if not courses:
        raise HTTPException(400, f"Keep at least one course with {MIN_TOPICS}+ topics.")
    ws = Workspace(db.job_dir(job_id))
    old = {c["course_name"]: c.get("source") for c in (read_json(ws.syllabi) or {}).get("courses", [])}
    for c in courses:
        c["source"] = old.get(c["course_name"], "edited")
    atomic_write_json(ws.syllabi, {"program": job["config"]["program_name"], "courses": courses})
    return {"courses": courses}


@app.post("/api/jobs/{job_id}/approve", dependencies=[Depends(require_access)])
def approve(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] != "awaiting_review":
        raise HTTPException(409, "This job is not awaiting review.")
    ws = Workspace(db.job_dir(job_id))
    courses = (read_json(ws.syllabi) or {}).get("courses", [])
    cfg = JobConfig.from_dict(job["config"])
    db.update_job(job_id, status="queued", estimate=estimate(cfg, len(courses), len(read_json(ws.sources, []))))
    db.add_event(job_id, "status", {"status": "queued", "message": f"Approved {len(courses)} course(s)."})
    return job_view(db.get_job(job_id))


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_access)])
def cancel(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] in TERMINAL:
        raise HTTPException(409, f"Job already {job['status']}.")
    if job["status"] == "running":
        db.update_job(job_id, cancel_requested=1)          # the worker kills the stage process
        db.add_event(job_id, "log", {"message": "Cancelling..."})
    else:
        db.update_job(job_id, status="cancelled", cancel_requested=1)
        vault.forget(job_id)
        db.add_event(job_id, "status", {"status": "cancelled"})
    return job_view(db.get_job(job_id))


@app.post("/api/jobs/{job_id}/retry", dependencies=[Depends(require_access)])
async def retry(job_id: str, request: Request):
    job = _job_or_404(job_id)
    if job["status"] not in ("failed", "cancelled"):
        raise HTTPException(409, "Only failed or cancelled jobs can be retried.")
    if job.get("replaced_by"):
        raise HTTPException(409, f"This job was already re-run as job {job['replaced_by']}.")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    stage = next((s for s in STAGE_NAMES if job["progress"].get(s, {}).get("status") != "done"), None)
    if stage is None:
        raise HTTPException(409, "Every stage already finished.")
    api_key = (body.get("api_key") or "").strip()
    remaining = STAGE_NAMES[STAGE_NAMES.index(stage):]
    demo = job["config"].get("demo") or get_settings().teacher_backend == "fake"
    if not demo and any(s in TEACHER_STAGES for s in remaining):
        if not api_key:
            raise HTTPException(400, "This retry re-runs a stage that calls the teacher model; provide the API key again.")
        _verify_key(api_key, job["config"]["teacher_model"])
        vault.store(job_id, vault.TEACHER_KEY, api_key)
    hf_token = (body.get("hf_token") or "").strip()
    hf_repo = job["config"].get("hf_repo") or ""
    if hf_repo and "publish" in remaining:
        if not hf_token:
            raise HTTPException(400, "This retry publishes the model to Hugging Face; enter your Write token again.")
        if _hf_user(hf_token) != hf_repo.split("/")[0]:
            raise HTTPException(400, f"Use a token for the Hugging Face account {hf_repo.split('/')[0]}, "
                                     "which this job publishes to.")
    if hf_token:
        vault.store(job_id, vault.HF_TOKEN, hf_token)
    with _admission:
        _check_capacity(bool(demo), exclude=job_id)
        db.update_stage_progress(job_id, stage, {"status": "pending", "eta_s": None})
        db.update_job(job_id, status="queued", next_stage=stage, current_stage=None, error=None, cancel_requested=0)
    db.add_event(job_id, "status", {"status": "queued", "message": f"Retrying from {STAGE_LABELS[stage]}."})
    return job_view(db.get_job(job_id))


DELETE_WAIT_S = 45          # the worker polls cancel every second; killing the stage tree can take a while
STALE_HEARTBEAT_S = 30      # a "running" job this quiet has no live worker behind it


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_access)])
async def delete_job(job_id: str):
    """Deletes a job in any state. A running stage is cancelled first (so its process lets go of
    the job's files and the GPU); a job whose worker died is deleted straight away."""
    job = _job_or_404(job_id)
    if job["status"] == "running" and time.time() - (job.get("heartbeat") or 0) < STALE_HEARTBEAT_S:
        db.update_job(job_id, cancel_requested=1)
        deadline = time.time() + DELETE_WAIT_S
        while (job := db.get_job(job_id)) and job["status"] == "running":
            if time.time() > deadline:
                raise HTTPException(409, "The job is still stopping. Try deleting it again in a moment.")
            await asyncio.sleep(1.0)
    elif job["status"] not in TERMINAL:
        db.update_job(job_id, status="cancelled", cancel_requested=1)   # nothing may claim it meanwhile
    vault.forget(job_id)
    shutil.rmtree(get_settings().jobs_dir / job_id, ignore_errors=True)
    db.delete_job(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_access)])
def download(job_id: str):
    job = _job_or_404(job_id)
    path = Path(job["bundle_path"]) if job.get("bundle_path") else None
    if not path or not path.exists():
        raise HTTPException(404, "The bundle is not ready yet.")
    return FileResponse(path, filename=path.name, media_type="application/zip")


@app.get("/api/jobs/{job_id}/dataset", dependencies=[Depends(require_access)])
def dataset(job_id: str):
    _job_or_404(job_id)
    path = Workspace(db.job_dir(job_id)).merged
    if not path.exists():
        raise HTTPException(404, "The dataset is not ready yet.")
    return FileResponse(path, filename=f"dataset-{job_id}.json", media_type="application/json")


@app.get("/api/jobs/{job_id}/deploy", dependencies=[Depends(require_access)])
def deploy_guide(job_id: str):
    _job_or_404(job_id)
    guide = read_json(Workspace(db.job_dir(job_id)).deploy)
    if guide is None:
        raise HTTPException(404, "The deployment guide is not ready yet.")
    return JSONResponse(guide)


def _manifest(job_id: str, repo: str) -> tuple[dict, dict]:
    _job_or_404(job_id)
    guide = read_json(Workspace(db.job_dir(job_id)).deploy)
    if guide is None:
        raise HTTPException(404, "The deployment guide is not ready yet.")
    try:
        return guide, student_app.build_manifest(guide, repo)
    except student_app.ManifestError as e:
        raise HTTPException(400, str(e))


@app.get("/api/jobs/{job_id}/model-manifest", dependencies=[Depends(require_access)])
def model_manifest(job_id: str, repo: str = ""):
    """corecs-model.json alone: tells a CoreCS desktop app which model to pull on first launch."""
    _, manifest = _manifest(job_id, repo)
    return JSONResponse(manifest, headers={
        "Content-Disposition": f'attachment; filename="{student_app.MANIFEST_FILE}"'})


@app.get("/api/installer", dependencies=[Depends(require_access)])
def installer_download():
    """The plain desktop installer for the "Download App" page: with no corecs-model.json next
    to it, the app pulls its built-in default model on first launch."""
    installer = get_settings().desktop_installer
    if not installer or not installer.exists():
        raise HTTPException(404, "No desktop installer is configured on this server "
                                 "(build runtime/desktop or set GLOBAL_DESKTOP_INSTALLER).")
    return FileResponse(installer, filename=installer.name, media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/student-app", dependencies=[Depends(require_access)])
def student_app_download(job_id: str, repo: str = ""):
    """The generic desktop installer + this job's model manifest, as one ZIP for students."""
    guide, manifest = _manifest(job_id, repo)
    installer = get_settings().desktop_installer
    if not installer or not installer.exists():
        raise HTTPException(404, "No desktop installer is configured on this server "
                                 "(build runtime/desktop or set GLOBAL_DESKTOP_INSTALLER).")
    root = f"{slugify(guide['display_name'], fallback='tutor')}-app"
    return Response(student_app.build_zip(installer, manifest, root), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{root}.zip"'})
