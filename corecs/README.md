# CoreCS

An offline, privacy-first AI tutor for Computer Science undergraduates. CoreCS runs a
fine-tuned 7B model entirely on the student's own computer via a bundled llama.cpp engine;
no account, no cloud, and nothing leaves the machine.

- Architecture, stack and data locations: [`docs/summary.md`](docs/summary.md)
- Progress, decisions and gotchas (living log): [`docs/project_memory.md`](docs/project_memory.md)

## Prerequisites (Windows)

- Node.js 20+
- Rust (stable, MSVC toolchain) via `rustup`
- Visual Studio Build Tools with "Desktop development with C++"

## Setup

```powershell
npm install
powershell -ExecutionPolicy Bypass -File scripts\fetch-llama.ps1   # llama-server sidecar (git-ignored)
```

Put a model (`*.gguf`, optionally with its `*.corecs.json` profile) in
`%APPDATA%\com.corecs.app\models`, or create one from a LoRA adapter with
[`scripts/export_gguf.py`](scripts/export_gguf.py).

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
npm run tauri build
```
