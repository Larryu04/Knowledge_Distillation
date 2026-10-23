# CoreCS

An offline, privacy-first AI tutor for Computer Science undergraduates. CoreCS runs a
fine-tuned 7B model entirely on the student's own computer via a bundled llama.cpp engine;
no account, no cloud, and nothing leaves the machine.

Part of the [CoreCS monorepo](../README.md). It ships the v8 model trained in
[`../model_training/`](../model_training/). A generic, model-agnostic copy of this app is bundled by
CoreCS Global in [`../GLOBAL/runtime/desktop/`](../GLOBAL/runtime/desktop/). All commands below
run from this `corecs/` directory.

- Architecture, stack and data locations: [`docs/summary.md`](docs/summary.md)
- Progress, decisions and gotchas (living log): [`docs/project_memory.md`](docs/project_memory.md)

## Prerequisites (Windows)

- Node.js 20+
- Rust (stable, MSVC toolchain) via `rustup`
- Visual Studio Build Tools with "Desktop development with C++"

## Setup

```powershell
cd corecs
npm install
powershell -ExecutionPolicy Bypass -File scripts\fetch-llama.ps1   # llama-server sidecar (git-ignored)
```

Put a model (`*.gguf`, optionally with its `*.corecs.json` profile) in
`%APPDATA%\com.corecs.app\models`, or create one from a LoRA adapter with
[`scripts/export_gguf.py`](scripts/export_gguf.py). Run the export from `model_training/` with its
venv active, so it reads the exact preamble and sampling settings from `inference.py`:

```powershell
cd model_training
.venv\Scripts\activate
python ..\corecs\scripts\export_gguf.py --adapter out/dictalm2-tutor-qlora-v8/adapter --name corecs-v8
```

## Development

```powershell
npm run tauri dev                                                  # run the app
$env:CORECS_MODEL = "C:\path\to\model.gguf"; npm run tauri dev     # force a specific model
cd src-tauri; cargo test --lib                                     # Rust unit tests
cd src-tauri; cargo run --example smoke -- C:\path\to\model.gguf   # end-to-end core test
npx tsc --noEmit                                                   # type-check the UI
```

## Build an installer

```powershell
npm run tauri build -- --bundles nsis   # -> src-tauri\target\release\bundle\nsis\*-setup.exe
```

The installer contains the app and the inference engine (~18.5 MB). The ~4.4 GB model file is
distributed separately.
