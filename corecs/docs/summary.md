# CoreCS — Summary

CoreCS is a desktop AI tutor for Computer Science undergraduates. It runs a fine-tuned 7B
language model **entirely on the student's own computer**: no account, no cloud, no internet
connection after setup. It is distributed as a single installer for Windows and macOS.

## Selling point: offline and private by design

- **100% local inference.** The model runs on the user's GPU (or CPU) through a bundled
  llama.cpp engine. Questions, answers and chat history never leave the machine.
- **Specialized beats big.** A 7B model distilled for specific CS courses (networks, security,
  OS, data structures, algorithms, databases, automata, ...) aims to outperform much larger
  general-purpose models (e.g. 70B) in those domains, while fitting in ~4.5 GB.
- **Works anywhere.** No subscription, no rate limits, no outage, and it works in the
  library, on a plane, or in an exam-prep room with no Wi-Fi.
- **Hebrew-first.** The current tutor model answers in Hebrew; the UI is RTL-aware.

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| App shell & installer | **Tauri 2** | ~10 MB shell using the OS webview (WebView2 / WKWebView) instead of bundling Chromium; first-class sidecar bundling; NSIS/MSI/DMG installers. |
| UI | **React 19 + TypeScript + Vite + Tailwind CSS v4** | Modern, fast iteration. Markdown, KaTeX (math) and Shiki (code highlighting) for tutor answers. Fonts bundled locally (Heebo) — no CDN. |
| Core / backend | **Rust** (tokio, reqwest) | Owns the model engine, persistence and streaming. The UI only talks to it over Tauri IPC. |
| Persistence | **SQLite** (rusqlite, bundled) + FTS5 | One file in the app-data folder; transactional; full-text search over past chats. |
| Inference | **llama.cpp `llama-server`** sidecar | Small native binary, OpenAI-compatible streaming API, GGUF models, GPU via Vulkan (Windows) / Metal (macOS), CPU fallback. |
| Model format | **GGUF, Q4_K_M** (~4.4 GB for 7B) | Best quality-per-GB at this size; the chat template is embedded in the file. |

## Architecture

```
┌──────────────────────── User's machine (nothing leaves it) ────────────────────────┐
│                                                                                     │
│  React UI (WebView)                                                                 │
│   └─ talks ONLY via Tauri IPC: invoke("send_message", …) + streaming Channel        │
│                                  │                                                  │
│                                  ▼                                                  │
│  Rust core                                                                          │
│   ├─ engine.rs   SidecarManager: launches llama-server on 127.0.0.1:<random port>   │
│   │              with a random per-launch API key; health-checks; picks the         │
│   │              discrete GPU; stops it on exit (Job Object kills it even on crash) │
│   ├─ chat.rs     builds the prompt, streams tokens (SSE) to the UI, cancel,         │
│   │              persists partial replies every 750 ms                              │
│   ├─ db.rs       SQLite: conversations, messages, settings, FTS5 search             │
│   └─ models.rs   discovers *.gguf in the models folder (+ optional profile JSON)    │
│                                  │ HTTP, loopback only                              │
│                                  ▼                                                  │
│  llama-server (sidecar)  ←  %APPDATA%\com.corecs.app\models\*.gguf                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Model-agnostic.** The app loads *any* valid GGUF from its models folder. A checkpoint can
ship an optional profile (`<name>.corecs.json`) with its training-time prompt preamble and
sampling defaults. Swapping v7 → v8 → v9 means dropping in a new file, with no code change.

## Data locations (Windows)

| What | Where |
|---|---|
| Chat database | `%APPDATA%\com.corecs.app\corecs.db` |
| Models | `%APPDATA%\com.corecs.app\models\*.gguf` (+ `*.corecs.json`) |
| Engine binaries | installed next to `CoreCS.exe` (`llama-server.exe` + DLLs) |

## Repository layout

```
corecs/
├─ src/                    React UI: App.tsx, store.ts (zustand), components/,
│                          lib/api.ts (typed IPC), lib/bidi.ts (Hebrew/English direction)
├─ src-tauri/
│  ├─ src/                 Rust core: lib.rs, commands.rs, engine.rs, chat.rs, db.rs, models.rs
│  ├─ examples/smoke.rs    end-to-end test of the core without the UI
│  ├─ binaries/            llama-server sidecar + DLLs (git-ignored; scripts/fetch-llama.ps1)
│  └─ tauri.conf.json
├─ scripts/
│  ├─ fetch-llama.ps1      downloads the pinned llama.cpp build into src-tauri/binaries
│  └─ export_gguf.py       LoRA adapter → merged → GGUF Q4_K_M + profile (run from training repo)
├─ tools/                  git-ignored: llama.cpp source + binaries for conversion, test models
└─ docs/                   this file + project_memory.md
```

The model training pipeline lives in a separate repository (`Knowledge_Dis`) and is never
modified by this app.
