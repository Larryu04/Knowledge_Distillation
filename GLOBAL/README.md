# CoreCS Global

Turn a degree program's syllabi into a private, offline AI tutor. A lecturer uploads syllabi
and their own LLM API key in a web UI. The platform writes a syllabus-grounded synthetic
training set, QLoRA-fine-tunes an open model on it, quantizes it to GGUF, and hands back a
downloadable bundle that students run on their own computers, plus a step-by-step guide for
publishing it on Hugging Face.

```
Syllabus Ingestion -> Dataset Generation -> Fine-Tuning -> Packaging & GGUF -> Deployment Guide
   (teacher LLM)        (teacher LLM)         (GPU)          (GPU / CPU)         (filled-in)
```

## Quick start (single machine, Windows)

```powershell
# 1. Platform (API, UI, ingestion/generation)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. GPU stages: a Python with torch + CUDA (can be a separate venv; point GLOBAL_TRAIN_PYTHON at it)
#    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#    pip install -r requirements-train.txt

# 3. llama.cpp toolchain for GGUF conversion + the bundled student runtime
.venv\Scripts\python scripts\setup_toolchain.py

# 4. Configure and run
copy .env.example .env      # set GLOBAL_TRAIN_PYTHON etc.
powershell -ExecutionPolicy Bypass -File scripts\run_platform.ps1
# -> http://127.0.0.1:8000
```

Try it without an API key or GPU time: set `GLOBAL_TEACHER_BACKEND=fake` (simulated teacher)
and pick "Qwen 2.5 1.5B Instruct (fast test runs)". A full run takes about 4 minutes on an
RTX 4070 Super.

## Architecture

```
 Browser (vanilla JS, no build)                         Lecturer
   │  multipart upload, REST, Server-Sent Events
   ▼
 FastAPI  (corecs_global/api.py)
   │  validates config, verifies the API key, stores uploads,
   │  encrypts credentials (vault.py), enqueues; never runs pipeline work
   ▼
 SQLite, WAL  (db.py)  jobs · events · secrets(ciphertext)
   ▲  claim (atomic), heartbeat, progress, events
   │
 Worker(s)  (worker.py)  --lanes io | gpu | io,gpu
   │  one subprocess per stage: killable on cancel, crash-isolated, frees VRAM
   ▼
 stages/run_stage.py ──► ingest · generate · deploy      (platform Python, io lane)
                     └─► train · package                (GLOBAL_TRAIN_PYTHON, gpu lane)
```

**Queue and lanes.** A job is a row whose `next_stage` advances as stages finish; between
stages it goes back to `queued`. Workers only claim jobs whose next stage is in their lane.
That way, API-bound synthesis for one lecturer never waits behind hours of GPU training for
another. Run one `--lanes gpu` worker per GPU and as many `--lanes io` workers as your rate
limits allow. SQLite is the queue, so a single host needs no Redis or Celery.

**Resilience.** Every stage checkpoints and is idempotent. Generation writes each accepted
row atomically, training resumes from the newest *complete* checkpoint, and packaging reuses
finished artifacts. A dead worker's job is requeued at the same stage on the next worker
start (heartbeat timeout). Failed and cancelled jobs can be retried from the UI.

**Progress and ETA.** Stages report `done/total`. The stage ETA is an EMA of seconds per
unit (generated row, training step). The overall ETA adds up-front estimates for stages not
yet started; those estimates are refreshed with the real course count after ingestion.

**Credentials.** API keys are Fernet-encrypted at rest and never returned by the API. The
worker decrypts them into the environment of only the stage subprocess that needs them. The
teacher key is deleted when generation succeeds, and all credentials are deleted when the
job ends. The teacher model is locked to `gpt-4o-mini` (`config.ALLOWED_TEACHER_MODELS`).

**Syllabus review.** By default the job pauses after ingestion (`awaiting_review`) so the
lecturer can fix course names, topic lists and out-of-scope terms before any money is spent
on generation.

## Stages

| Stage | Module | Output |
|---|---|---|
| Syllabus ingestion | `stages/ingest.py` | PDF/DOCX/TXT/MD/pasted text → `syllabi.json`: courses, granular topics, out-of-scope terms |
| Dataset generation | `stages/generate.py` | `dataset/merged.json`: `{course_name, domain, instruction, rationale, output}` rows, validated (3-5 numbered steps, worked example, synthesis, scope, language, near-duplicates) |
| Fine-tuning | `stages/train.py` | `train/adapter/`: 4-bit QLoRA, frozen vocabulary, prompt masking, PAD≠EOS, macro per-course early stopping, VRAM preflight with automatic micro-batch fallback |
| Packaging & GGUF | `stages/package.py` | merge → GGUF (bf16 or q8_0 intermediate by free disk) → quantize → verify → `<program>-bundle.zip` |
| Deployment guide | `stages/deploy.py` | `deploy.json` + `DEPLOY.md`: local test, `hf upload`, student download link, optional Docker Space |

## The bundle

```
<program>-tutor/
  README.txt, DEPLOY.md
  model/<name>-q4_k_m.gguf            the tutor
  model/<name>-q4_k_m.corecs.json     preamble + sampling profile (read by both runtimes)
  runtime/portable/                   llama-server + DLLs, start scripts, offline chat UI
  runtime/desktop/                    CoreCS desktop installer (if one is built/configured)
  adapter/                            LoRA adapter (optional)
  huggingface/                        model card + Docker Space template
  dataset/                            generated data + structured syllabi, for review
```

## Repository layout

```
corecs_global/        platform package (api, worker, db, vault, progress, jobs, languages, llm/, stages/, web/)
runtime/portable/     zero-install student runtime (copied into every bundle)
runtime/desktop/      CoreCS desktop app sources (Tauri 2 + React + Rust), generic, model-agnostic
scripts/              setup_toolchain.py, run_platform.ps1
tests/                pytest: queue/lanes, vault, ETA, validation, ingestion, API + worker end to end
```

## Tests

```powershell
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The suite runs fully offline (fake teacher, no GPU). It drives the real API and real worker
subprocesses through ingestion, review, generation, resume, cancel/retry and failure paths.

## Production notes

- Put the server behind HTTPS and set `GLOBAL_ACCESS_TOKEN`: lecturers submit API keys. The
  token is a single shared secret. Per-lecturer accounts are the natural next step and would
  scope `jobs` by owner.
- Bundles are 1-6 GB each. Delete finished jobs from the UI, or point `GLOBAL_DATA_DIR` at a
  large disk. Exporting a 7B model needs about 35 GB of scratch space at peak.
- Fine-tuned models inherit their base model's license. The deployment guide reminds the lecturer.
