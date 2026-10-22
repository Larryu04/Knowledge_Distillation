# CoreCS Desktop (runtime)

An offline, privacy-first desktop tutor. It runs any fine-tuned GGUF model entirely on the
student's own computer through a bundled llama.cpp engine: no account, no cloud, nothing
leaves the machine.

This is the generic desktop runtime shipped by CoreCS Global. It is model-agnostic: it loads
any valid `*.gguf` from its models folder, plus an optional `<stem>.corecs.json` profile that
CoreCS Global writes next to every model it produces (training preamble + sampling defaults).

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
`%APPDATA%\com.corecs.app\models`. Models are produced by the CoreCS Global pipeline.

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
npm run tauri build -- --bundles nsis
```

The installer lands in `src-tauri/target/release/bundle/nsis/`. CoreCS Global picks it up
automatically from there (or from `GLOBAL_DESKTOP_INSTALLER`) and includes it in every
lecturer bundle.

## Architecture

| Layer | Technology |
|---|---|
| App shell & installer | Tauri 2 (OS webview, ~10 MB shell) |
| UI | React 19 + TypeScript + Vite + Tailwind CSS v4; Markdown, KaTeX, Shiki |
| Core | Rust (tokio, reqwest): engine, chat streaming, persistence |
| Persistence | SQLite (rusqlite, bundled) + FTS5 |
| Inference | llama.cpp `llama-server` sidecar on 127.0.0.1 with a per-launch API key |

Gotchas worth knowing:

- The profile preamble is folded into the **first user turn**, never sent as a `system`
  message (Mistral-family chat templates reject the `system` role).
- `llama-server.exe` is a thin launcher for `llama-server-impl.dll`; the DLLs must sit next
  to it.
- llama.cpp's `repeat_penalty` only looks at the last 64 tokens; if a model loops, add
  `"repeat_last_n": -1` and/or `"dry_multiplier": 0.8` to the profile's `sampling`.
