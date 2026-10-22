# Portable runtime

Zero-install student runtime shipped in every bundle: llama.cpp's `llama-server` serves both
the model (OpenAI-compatible API) and this folder's `webui/` chat page, bound to 127.0.0.1.

```
portable/
  start-windows.bat        finds model/*.gguf (or ../../model/*.gguf), starts the server, opens the browser
  start-macos-linux.sh     same, using bin/llama-server or llama-server from PATH (brew install llama.cpp)
  webui/                   static chat UI; reads profile.json (copied from <model>.corecs.json at start)
  bin/                     added at packaging time from the toolchain (llama-server + DLLs); not in git
```

The chat page applies the model profile exactly like the desktop app: the training preamble
is folded into the first user turn (never a `system` message), sampling settings are passed
through to llama-server, and older turns are trimmed to fit the context window.
