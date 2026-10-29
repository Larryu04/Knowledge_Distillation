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

## First launch: which model gets pulled

The installer is one generic build. Which model it pulls from the Hugging Face Hub on first
launch is decided by an optional `corecs-model.json` manifest (`src-tauri/src/provision.rs`):

1. **Lecturer's model.** CoreCS Global publishes the lecturer's model to their Hugging Face
   account, and its deploy page offers a "student app" ZIP: this installer plus a
   `corecs-model.json` naming that repository. The NSIS hook
   (`src-tauri/windows/hooks.nsh`) copies the manifest from the installer's folder into the
   install folder, and the app pulls that model once (also if other models exist), then makes it
   the active model. A manifest dropped into `%APPDATA%\com.corecs.app\` works too.
2. **Default model.** No manifest and an empty models folder: the app pulls CoreCS v8,
   [`Larryu04/corecs-v8-gguf`](https://huggingface.co/Larryu04/corecs-v8-gguf) /
   `corecs-v8-q4_k_m.gguf` (4.37 GB, SHA-256 pinned in `provision.rs`) plus its
   `.corecs.json` profile. To ship a different default, set `CORECS_DEFAULT_REPO` /
   `CORECS_DEFAULT_GGUF` at build time. A different default isn't checksum-verified.

Downloads resume after an interruption ("Retry" in the banner) and are checked against the
manifest's size and SHA-256 when it has them. `CORECS_MODEL` skips all of this.

## Development

```powershell
npm run tauri dev                                                  # run the app
$env:CORECS_MODEL = "C:\path\to\model.gguf"; npm run tauri dev     # force a specific model
cd src-tauri; cargo test --lib                                     # Rust unit tests
cd src-tauri; cargo test --lib -- --ignored hub_download           # Hugging Face download + resume (network)
cd src-tauri; cargo run --example smoke -- C:\path\to\model.gguf   # end-to-end core test
npx tsc --noEmit                                                   # type-check the UI
```

## Build an installer

```powershell
npm run tauri build -- --bundles nsis
```

The installer lands in `src-tauri/target/release/bundle/nsis/`. To serve it from CoreCS Global,
copy it into `GLOBAL/runtime/installer/`. The server also finds it in the build folder, and
`GLOBAL_DESKTOP_INSTALLER` overrides both. It goes into the student app ZIP and into every
lecturer bundle.

Build from this folder, not from the standalone `corecs/` app. Only this build has the
first-launch download and the NSIS hook that reads `corecs-model.json`.

`fetch-llama.ps1` also copies the VC++ runtime (`vcruntime140.dll`, `vcruntime140_1.dll`,
`msvcp140.dll`) from the Visual Studio redist folder into `src-tauri/binaries/`, and the installer
ships it next to `llama-server`. The app therefore runs on a clean Windows without the
VC++ Redistributable.

### Testing a first install

`scripts/reset-test-machine.ps1` returns a machine to "never installed": it uninstalls CoreCS
silently and removes `%APPDATA%\com.corecs.app` and `%LOCALAPPDATA%\com.corecs.app`. It is
standalone, so it can be copied to a test laptop on its own. `-BackupModels` moves the
downloaded models aside instead of deleting them, and `-RestoreModels` puts them back.

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
