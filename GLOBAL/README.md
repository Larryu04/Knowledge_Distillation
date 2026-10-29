# CoreCS Global

Turn a degree program's syllabi into a private, offline AI tutor. A lecturer uploads syllabi,
their own LLM API key and a Hugging Face Write token in a web UI. The platform writes a
syllabus-grounded synthetic training set, QLoRA-fine-tunes an open model on it, quantizes it to
GGUF, publishes the model to the lecturer's Hugging Face account, and hands back a bundle that
students run on their own computers, plus a student app ZIP and a sharing guide.

```
Syllabus Ingestion -> Dataset Generation -> Fine-Tuning -> Packaging & GGUF -> Publishing -> Deployment Guide
   (teacher LLM)        (teacher LLM)         (GPU)          (GPU / CPU)      (HF Hub)       (filled-in)
```

Part of the [CoreCS monorepo](../README.md). All commands below run from this `GLOBAL/`
directory.

## Quick start (single machine, Windows)

```powershell
cd GLOBAL

# 1. Platform (API, UI, ingestion/generation)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. GPU stages: a Python with torch + CUDA (can be a separate venv; point GLOBAL_TRAIN_PYTHON at it)
#    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#    pip install -r requirements-train.txt

# 3. llama.cpp toolchain for GGUF conversion + the bundled student runtime (-> GLOBAL\tools\)
.venv\Scripts\python scripts\setup_toolchain.py

# 4. Configure
copy .env.example .env

# 5. Run the API/UI and one worker for both lanes
powershell -ExecutionPolicy Bypass -File scripts\run_platform.ps1
# -> http://127.0.0.1:8000
```

### Configuration (`.env`)

`.env.example` is the template and is committed. `.env` is your copy and is git-ignored.
`run_platform.ps1` loads it at startup. Every setting is optional; uncomment a line to use it.

| Variable | Purpose |
|---|---|
| `GLOBAL_DATA_DIR` | Where jobs, uploads, bundles and the SQLite DB live (default `./data`). Bundles are GBs each. |
| `GLOBAL_TRAIN_PYTHON` | Interpreter with torch + CUDA + `requirements-train.txt`, used for the train/package stages |
| `GLOBAL_TOOLCHAIN_DIR` | llama.cpp toolchain location (default `./tools`) |
| `GLOBAL_DESKTOP_INSTALLER` | Full path to the desktop installer to serve. Default: the newest `*-setup.exe` in `runtime/installer/`, else in the `runtime/desktop` build output |
| `GLOBAL_ACCESS_TOKEN` | Shared secret required by the UI/API. **Set it whenever the server is reachable by others** (e.g. via ngrok). |
| `GLOBAL_DISABLE_BUNDLE_DOWNLOAD` | `1` stops serving the multi-GB bundle ZIP (it contains the GGUF), e.g. behind a metered ngrok tunnel. The student app ZIP still works. |
| `GLOBAL_SECRET_KEY` | Fernet key for credentials at rest (default: generated into `data/secret.key`) |
| `GLOBAL_TEACHER_BACKEND` | `fake` simulates the teacher LLM: no API key, no cost |
| `GLOBAL_VRAM_FRACTION` | Fraction of GPU memory training may use (default 0.85) |
| `GLOBAL_CORS_ORIGINS` | Comma-separated origins allowed to call the API (default `*`). Set it to your Vercel URL in production. |

Try it without an API key or GPU time: set `GLOBAL_TEACHER_BACKEND=fake` (simulated teacher)
and pick "Qwen 2.5 1.5B Instruct (fast test runs)". A full run takes about 4 minutes on an
RTX 4070 Super.

## Deployment: Vercel frontend + local backend (via ngrok)

The platform can run decoupled. The static UI (`corecs_global/web/`) is hosted on Vercel, and
the API, worker and GPU stay on your own machine, exposed through an ngrok HTTPS tunnel:

```
 Lecturer's browser ──► https://<app>.vercel.app      (static: index.html, app.js, config.js)
        │
        │  fetch / SSE / downloads, to window.GLOBAL_API_URL
        ▼
 https://<name>.ngrok-free.app ──► ngrok agent ──► http://127.0.0.1:8000  (FastAPI + worker + GPU)
```

At build time `build-config.mjs` writes `config.js` with `window.GLOBAL_API_URL` from the
`GLOBAL_API_URL` env var, and `app.js` sends every request there. If the backend serves the
UI itself (local mode), its own `/config.js` sets the value to `""` (same origin).

### 1. Backend (your machine)

```powershell
cd GLOBAL
# In .env, set at least:
#   GLOBAL_ACCESS_TOKEN=<a long random string>      # the tunnel is public; lecturers send API keys
#   GLOBAL_CORS_ORIGINS=https://<app>.vercel.app    # optional, tightens the default "*"
powershell -ExecutionPolicy Bypass -File scripts\run_platform.ps1
```

### 2. Tunnel

```powershell
ngrok config add-authtoken <your-ngrok-token>      # once
ngrok http 8000 --url=<name>.ngrok-free.app       # free static domain from the ngrok dashboard
```

Use the free static domain: a random URL changes on every restart, and Vercel would need a
redeploy each time. `app.js` sends the `ngrok-skip-browser-warning` header, so ngrok's
interstitial page doesn't intercept API calls.

### 3. Frontend (Vercel)

1. Import the GitHub repo into Vercel.
2. **Root Directory:** `GLOBAL/corecs_global/web`. `vercel.json` there sets the build
   (`node build-config.mjs`), the output directory and the SPA rewrites. Framework preset: *Other*.
3. **Environment variable:** `GLOBAL_API_URL` = `https://<name>.ngrok-free.app` (no trailing
   slash needed; `https://` is added if you leave it out).
4. Deploy. After changing `GLOBAL_API_URL`, **redeploy**, because the value is baked into `config.js` at build time.

On first load the dashboard asks for the access token, stores it in the browser, and sends it
as a `Bearer` header (or as `?token=` for the event stream and download links).

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Could not reach the CoreCS Global server" | Backend or ngrok isn't running, or `GLOBAL_API_URL` is wrong. Open `https://<name>.ngrok-free.app/api/options` directly; it should return JSON. |
| "The backend … did not return JSON" | The request hit an HTML page instead of the API: the ngrok tunnel is offline, or `GLOBAL_API_URL` points at the Vercel site itself. Fix the variable and redeploy. |
| Repeated access-token prompt | The token doesn't match `GLOBAL_ACCESS_TOKEN`. Clear the site's local storage and enter the right one. |
| CORS error in the browser console | `GLOBAL_CORS_ORIGINS` doesn't list the exact Vercel origin (scheme + host, no trailing slash). |

## Architecture

```
 Browser (vanilla JS, no build; served by FastAPI or Vercel)        Lecturer
   │  multipart upload, REST, Server-Sent Events
   ▼
 FastAPI  (corecs_global/api.py)
   │  validates config, verifies the API key and HF Write token, stores uploads,
   │  encrypts credentials (vault.py), enqueues; never runs pipeline work
   ▼
 SQLite, WAL  (db.py)  jobs · events · secrets(ciphertext)
   ▲  claim (atomic), heartbeat, progress, events
   │
 Worker(s)  (worker.py)  --lanes io | gpu | io,gpu
   │  one subprocess per stage: killable on cancel, crash-isolated, frees VRAM
   ▼
 stages/run_stage.py ──► ingest · generate · publish · deploy   (platform Python, io lane)
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
| Publishing | `stages/publish.py` | Creates `<hf-user>/<program>-tutor-gguf` (public) if needed and pushes the GGUF, its `.corecs.json` and the model card in one commit → `publish.json` |
| Deployment guide | `stages/deploy.py` | `deploy.json` + `DEPLOY.md`: local test, the published model, sharing with students, optional Docker Space |

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

### Student desktop app

The desktop installer is one generic build, so nothing is compiled per lecturer. Once the
publish stage has pushed the model, the deploy page offers a **student app** ZIP
(`GET /api/jobs/<id>/student-app`): the installer plus a generated `corecs-model.json` naming
the published repository, file, size and SHA-256 (`student_app.py`). On first
launch the app pulls that model. Without a manifest it pulls the default CoreCS v8 model from
[`Larryu04/corecs-v8-gguf`](https://huggingface.co/Larryu04/corecs-v8-gguf). See
[`runtime/desktop/README.md`](runtime/desktop/README.md#first-launch-which-model-gets-pulled).

**Serving the installer.** Build it from `runtime/desktop` (`npm run tauri build -- --bundles nsis`)
and copy `CoreCS_<version>_x64-setup.exe` into **`GLOBAL/runtime/installer/`**. The server
uses the newest `*-setup.exe` there and re-checks on every request, so no restart is needed.
The deploy page hides the ZIP button while no installer is found. Only a `runtime/desktop`
build reads `corecs-model.json`. A build of the standalone `corecs/` app ignores it.

**What lecturers do** (also shown on the deploy page):

1. Download the student app ZIP.
2. Share the ZIP with the students.
3. Tell the students to extract the ZIP completely before running the installer. The
   `README.txt` inside says the same. Run from inside the ZIP, the installer can't see
   `corecs-model.json`, and the app installs the default v8 model instead.

## Repository layout

```
GLOBAL/
  corecs_global/        platform package (api, worker, db, vault, progress, jobs, languages, llm/, stages/)
    web/                UI: index.html, app.js, styles.css, fonts/ + vercel.json, build-config.mjs (Vercel)
  runtime/portable/     zero-install student runtime (copied into every bundle)
  runtime/desktop/      CoreCS desktop app sources (Tauri 2 + React + Rust), generic, model-agnostic
  runtime/installer/    drop folder for the prebuilt desktop installer the server hands out (*.exe git-ignored)
  scripts/              setup_toolchain.py, run_platform.ps1
  tests/                pytest: queue/lanes, vault, ETA, validation, ingestion, API + worker end to end
  .env.example          configuration template (copy to .env)
  requirements*.txt     platform / GPU-stage / dev dependencies
  data/, tools/         git-ignored: job data, llama.cpp toolchain
```

`runtime/desktop/` is a generic, model-agnostic copy of the Hebrew-specific app in
[`../corecs/`](../corecs/). Changes to one are not synced to the other automatically.

## Tests

```powershell
cd GLOBAL
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The suite runs fully offline (fake teacher, no GPU). It drives the real API and real worker
subprocesses through ingestion, review, generation, resume, cancel/retry and failure paths.

### Publishing to Hugging Face

The lecturer's Hugging Face token must have **Write** access. It is checked when the job is
submitted (`hf_publish.check_token`), so a read-only or mistyped token is rejected in seconds,
not after hours of training. The check also gives the account name, which fixes the repository
(`<hf-user>/<program-slug>-tutor-gguf`) up front. The model card, `DEPLOY.md` and the Space
template in the bundle all link to it. Right after packaging, the publish stage creates the
repository as a public model repository (students' apps download without a token). It then
pushes `<name>.gguf`, `<name>.corecs.json` and `README.md` in one commit. Re-running is safe:
files the Hub already has aren't uploaded again. If the repository already existed as a private
one, the deploy page warns that students can't download it.

The token is encrypted in the vault and deleted as soon as publishing succeeds, or when the
job fails or is cancelled. A retry that still has to publish asks for it again, and it must
belong to the same account. In demo mode (`GLOBAL_TEACHER_BACKEND=fake`) the token is optional:
without one the stage records "not published", and the deploy page offers the bundle only.

## Production notes

- Put the server behind HTTPS and set `GLOBAL_ACCESS_TOKEN`: lecturers submit API keys and Hugging Face Write tokens. The
  token is a single shared secret. Per-lecturer accounts are the natural next step and would
  scope `jobs` by owner.
- Bundles are 1-6 GB each. Delete finished jobs from the UI, or point `GLOBAL_DATA_DIR` at a
  large disk. Exporting a 7B model needs about 35 GB of scratch space at peak.
- Fine-tuned models inherit their base model's license. The deployment guide reminds the lecturer.
