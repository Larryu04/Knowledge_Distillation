# CoreCS — Project Memory

Living document: progress, decisions and gotchas. Update it whenever something lands.
Newest entries go on top within each section.

_Last updated: 2026-10-01 — v8 model exported and loaded in the app_

---

## Status at a glance

| Area | State |
|---|---|
| Scaffold (Tauri 2 + React + Tailwind) | ✅ done |
| llama-server sidecar bundled (b11269, Vulkan) | ✅ done |
| Rust core: engine / SQLite / streaming chat | ✅ done, unit + end-to-end tested |
| GGUF export pipeline (`scripts/export_gguf.py`) | ✅ done, v7 and **v8** (final, 23 subjects) converted by the user |
| Chat UI (sidebar + search, markdown/math/code, stop, engine banner, model picker) | ✅ done |
| **Full integration (UI ↔ Rust core ↔ llama-server ↔ v7 GGUF)** | ✅ **tested end-to-end by the user in `npm run tauri dev`: ~93 tok/s on Vulkan (RTX 4070 SUPER), RTL, data binding and model picker confirmed working** |
| Installer build (NSIS) | ✅ builds (18.5 MB setup.exe); installed-app run not yet verified — see "Building the installer" |
| Code signing (Windows SmartScreen) | ⏳ not started |
| First-run model setup (in-app download / import) | ⏳ not started (models folder + Refresh works today) |
| macOS build | ⏳ needs a Mac; sidecar name `llama-server-aarch64-apple-darwin` |

**Phase status:** v8 (the final model) is in the models folder alongside v7, no code change was
needed. Next: verify the installed app, then installer polish (see "Next steps").

---

## Progress log

### 2026-10-01 — v8 model dropped in
- User exported the v8 adapter with `scripts/export_gguf.py --name corecs-v8` →
  `corecs-v8-q4_k_m.gguf` (Q4_K_M, 4.37 GB) + `corecs-v8-q4_k_m.corecs.json` (display name
  "CoreCS v8", 23-subject preamble, same sampling as v7) and loaded it in the app.
- Weights confirmed distinct from the v7 file. Speed not re-measured; same architecture and
  quant as v7 (~93 tok/s).

### 2026-09-30 — End-to-end integration verified (phase 1 complete)
- User ran the full app (`npm run tauri dev`) with the converted v7 model:
  ~93 tokens/s on Vulkan, Hebrew RTL rendering correct, data binding and model picker correct.
- Phase 1 closed; development paused until v8 is trained.

### 2026-09-30 — Chat UI
- Layout: `Sidebar` (new chat, FTS search with highlighted snippets, conversations grouped
  Today / Yesterday / 7 days / 30 days / Older, inline rename, delete with native confirm),
  header with conversation title + `ModelPicker`, `EngineBanner`, `MessageList`, `Composer`.
- State: zustand store (`src/store.ts`). The streaming reply's text lives in `stream`, not in
  `messages`, so switching conversations mid-answer is safe; optimistic placeholders get real
  ids from the `started` event; a fallback resyncs from SQLite if `done` never arrives.
- Rendering: react-markdown + GFM + KaTeX (LaTeX `\(..\)` / `\[..\]` delimiters are converted
  to dollar delimiters for remark-math), Shiki highlighting (21 CS languages, JS regex engine,
  applied once streaming finishes), copy buttons, links open in the system browser.
- Composer: auto-grow, Enter to send / Shift+Enter newline, Stop button while streaming,
  disabled until the engine is ready. Ctrl+K focuses search.
- Engine banner states: loading / failed (with log details + Retry) / stopped / no model
  (Open models folder + Refresh). Picker lists models with size and a TUNED badge (has profile).
- Rust: `EngineStatus` now carries `modelId`; search snippets mark matches with the control
  characters U+0002 / U+0003 (brackets are common in code, so they can't be used as markers).
- `export_gguf.py` now writes `.gguf.part` and renames at the end (app never sees half a file).
- Visual QA: rendered every state in headless Edge with mocked Tauri IPC. Found and fixed two
  RTL bugs (see gotchas: bidi heuristic).

### 2026-09-30 — Rust core
- `engine.rs` SidecarManager: random loopback port + random API key per launch; `/health`
  polling (180 s timeout); stderr tail captured and surfaced on failure; status events
  (`engine://status`); crash detection; clean stop on app exit.
- Windows **Job Object** (`KILL_ON_JOB_CLOSE`): llama-server dies with CoreCS even on hard
  kill / `tauri dev` restart. Verified: no orphaned server after `Stop-Process -Force`.
- Automatic GPU choice: this dev machine exposes `Vulkan0` = RTX 4070 SUPER and
  `Vulkan1` = AMD iGPU; the heuristic prefers the discrete GPU. Override: setting `engine.device`.
- `db.rs`: SQLite with `PRAGMA user_version` migrations, WAL, FK cascade, FTS5 search,
  settings table. Replies left `streaming` by a crash are marked `partial` on next launch.
- `chat.rs`: streaming `send_message` over a Tauri Channel; cancel (`cancel_generation`) keeps
  text as `partial`; one generation at a time; auto-title from first message.
- `models.rs`: scans models folder for GGUF (magic-byte check) + optional `.corecs.json` profile.
- `src/lib/api.ts`: typed TS wrapper for every command/event.
- Tests: 10 unit tests (`cargo test --lib`), `cargo run --example smoke -- <model.gguf>`
  end-to-end (verified with Qwen2.5-0.5B: ~250–400 tok/s on the 4070, Hebrew intact,
  mid-stream cancel persisted as partial).
- Replaced `tauri-plugin-shell` with `tokio::process` (testable outside Tauri; kill_on_drop;
  smaller permission surface).

### 2026-09-30 — Model export pipeline
- `scripts/export_gguf.py` merges a LoRA adapter into `dicta-il/dictalm2.0-instruct`, converts
  to GGUF and quantizes to Q4_K_M, writing into the CoreCS models folder along with a profile
  JSON (preamble + sampling read from the training repo's `inference.py`).
- Toolchain pinned to llama.cpp **b11269** (same build as the sidecar) in `tools/`.

### 2026-09-30 — Scaffold
- Tauri 2 + React 19 + TS + Vite + Tailwind v4; Heebo font bundled; CSP locked to local.
- llama.cpp b11269 win-vulkan-x64 in `src-tauri/binaries` (git-ignored, `scripts/fetch-llama.ps1`).

---

## Next steps (when development resumes)

1. ~~**Drop in v8**~~ — ✅ done 2026-10-01. Next: install `setup.exe` and verify the installed
   app loads v8 (see "Building the installer").
2. **First-run model setup**: in-app "Import model file…" (dialog plugin is already installed)
   and optional download with SHA-256 verification.
3. **Installer polish**: app icon (current icons are the Tauri defaults — generate with
   `npm run tauri icon path\to\logo.png`), code signing certificate, bump `version` in
   `tauri.conf.json` + `package.json` + `Cargo.toml`.
4. **Before merging into the ML monorepo**: delete `tools/llama.cpp-src` (a cloned repo with
   its own `.git`), or keep it git-ignored; `corecs/.gitignore` already excludes `tools/`,
   `src-tauri/binaries/*`, `src-tauri/target` and `*.gguf`.
5. Nice-to-have: regenerate answer, export chat, settings screen (device, context size),
   crash log file, light theme.

---

## Building the installer

```powershell
cd C:\Users\Larry\corecs
npm run tauri build -- --bundles nsis
```
Output: `src-tauri\target\release\bundle\nsis\CoreCS_<version>_x64-setup.exe`.
(`npm run tauri build` without `--bundles` also builds an `.msi` via WiX.)

First release build, 2026-09-30: succeeded in ~3 min, **installer 18.5 MB** (app + llama-server
+ 23 DLLs, LZMA-compressed; the model is NOT included). `target\release` contains
`llama-server.exe` and all 23 DLLs next to `corecs.exe`.
**Not yet verified:** installing that setup.exe and running the installed app (check that the
DLLs land next to `CoreCS.exe` and that the model loads). Do this once before distributing.

Notes:
- The installer is unsigned → Windows SmartScreen warns on other machines until it is signed.
- The installed app uses the same models folder, `%APPDATA%\com.corecs.app\models`.
- Tauri warns that the identifier `com.corecs.app` ends in `.app` (clashes with macOS bundle
  naming). Harmless on Windows. Changing it before a macOS build would also move the data
  folder (`%APPDATA%\<identifier>`), so existing models/chats would need to be moved.

---

## Key design decisions

| Decision | Why |
|---|---|
| **No git repo in `corecs/`** (user decision) | The folder will be moved into the main ML monorepo later; do not `git init`. |
| **English only** for all code comments, docs and READMEs (user rule) | Hebrew appears only as content (UI example prompts, test data). |
| Word-count text direction instead of `dir="auto"` | See gotchas; Hebrew CS sentences often start with an English term. |
| Streaming text kept outside `messages` in the store | Tokens don't rewrite the message array; switching chats mid-answer is safe. |
| Shiki with the JS regex engine, only after streaming ends | No WASM (CSP stays strict); no re-highlighting on every token. |
| Tauri over Electron | ~10 MB vs ~150 MB shell; Rust core; good sidecar support. |
| llama.cpp `llama-server` sidecar over Ollama / Python | No second install for users (Ollama); no multi-GB PyInstaller+torch bundle (Python). |
| Model **not** inside the installer | NSIS/MSI struggle >2 GB, and app updates would re-ship 4.4 GB. First-run download or import instead. |
| UI never calls llama-server directly | Rust owns port/key/cancel/persistence; strict CSP; other local processes can't use the model. |
| Loopback + random port + random API key | No fixed port conflicts; nothing else on the machine can drive the model. |
| Vulkan build on Windows | Works on NVIDIA/AMD/Intel without shipping ~500 MB CUDA runtime. |
| **Model-agnostic via profile JSON** | Any valid GGUF loads. `<stem>.corecs.json` optionally carries the training preamble + sampling. v8/v9 = new files, no code change. |
| Preamble folded into the **first user turn**, never a `system` message | DictaLM/Mistral's template raises on the `system` role; training put the preamble inside the user turn. |
| Chat template taken from the GGUF (`--jinja`) | The model file is the source of truth for its own format. |
| `tokio::process` instead of tauri-plugin-shell | Testable without Tauri, Job Object assignment, kill_on_drop. |
| SQLite in Rust (rusqlite) rather than tauri-plugin-sql | Business logic stays in the core; UI can't run arbitrary SQL. |
| Stream persistence every 750 ms | A crash mid-answer loses <1 s of text. |

## Model profile contract (`<stem>.corecs.json`)

```json
{
  "schema": 1,
  "display_name": "CoreCS v7",
  "context_length": 4096,
  "prompt": { "preamble": "…", "separator": "\n\n" },
  "sampling": { "temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.15, "max_tokens": 512 },
  "source": { "base_model": "…", "adapter": "…", "quant": "Q4_K_M", "exported_at": "…" }
}
```
All fields optional. `sampling` is passed verbatim into the llama-server request, so any
llama-server sampling key works (e.g. `top_k`, `min_p`, `dry_multiplier`).

---

## Gotchas / lessons learned

- **Bidi heuristic (`src/lib/bidi.ts`)**: `dir="auto"` picks direction from the FIRST strong
  character, so "**TCP** הוא פרוטוקול…" rendered LTR, and table header cells went RTL while data
  cells stayed LTR (misaligned columns). Now: count Hebrew vs Latin words (code and KaTeX
  excluded); RTL if hebrew*3 >= latin. Applied per paragraph/heading/list/blockquote and per
  whole table; list items and cells inherit.
- UI visual testing without the model: serve the UI with `mockIPC` from
  `@tauri-apps/api/mocks` and screenshot with headless Edge
  (`msedge --headless=new --screenshot=… --virtual-time-budget=8000 URL`). Channel events can be
  faked with `__TAURI_INTERNALS__.runCallback(channel.id, { index, message })`.
- **llama-server.exe is a 9 KB launcher** for `llama-server-impl.dll`; the DLLs (ggml*, llama,
  llama-common, mtmd, libomp) must sit next to it. `tauri.conf.json` bundles them as resources.
- **Relative model paths break** because llama-server runs with its own folder as cwd →
  `models::load` makes paths absolute. Don't use `fs::canonicalize` (Windows `\\?\` prefix
  confuses llama.cpp).
- **Split UTF-8 in SSE**: network chunks can cut Hebrew characters in half; the SSE parser
  buffers bytes and decodes only complete lines.
- **Sampling parity with training eval**: `inference.py` uses HF `repetition_penalty=1.15`
  (whole sequence) + `no_repeat_ngram_size=4`. llama.cpp's `repeat_penalty` only looks at the
  last 64 tokens by default and has no n-gram blocker; if a model loops, add
  `"repeat_last_n": -1` and/or `"dry_multiplier": 0.8` to the profile's `sampling`.
- **Training was single-turn** (preamble + one question). Multi-turn follow-ups are
  out-of-distribution for v7; worth including multi-turn samples in v8/v9 data.
- **Disk space** on the dev machine is tight (~31 GB free): `export_gguf.py` auto-selects a
  q8_0 intermediate instead of bf16 when <34 GB free.
- **Never run `pip install -r llama.cpp/requirements.txt`** in the training venv: it pins its
  own torch and would replace the CUDA build. The converter only needs its bundled `gguf-py`.
- PowerShell 5.1 `Get-Content`/`Set-Content` default to ANSI and corrupt UTF-8 Hebrew in
  source files. Use explicit `-Encoding utf8` or an editor.
- Scripted doc edits with JavaScript `String.replace`: `$` sequences in the replacement string
  are special (`` $` `` inserts the text before the match). Use a replacer function instead.

## Dev commands

```powershell
npm run tauri dev                                   # run the app
$env:CORECS_MODEL="C:\path\model.gguf"; npm run tauri dev   # force a specific model
cd src-tauri; cargo test --lib                      # unit tests
cd src-tauri; cargo run --example smoke -- ..\tools\test-models\qwen2.5-0.5b-instruct-q4_k_m.gguf
powershell -ExecutionPolicy Bypass -File scripts\fetch-llama.ps1   # re-fetch sidecar
npm run tauri build -- --bundles nsis               # installer (see "Building the installer")
```
