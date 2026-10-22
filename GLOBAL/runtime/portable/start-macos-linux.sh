#!/bin/sh
# Starts the offline tutor: llama-server serves the model AND the chat UI on 127.0.0.1 only.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8088
CTX=4096
URL="http://127.0.0.1:$PORT"

MODEL="$(ls "$DIR"/model/*.gguf 2>/dev/null | head -n 1 || true)"
[ -z "$MODEL" ] && MODEL="$(ls "$DIR"/../../model/*.gguf 2>/dev/null | head -n 1 || true)"
if [ -z "$MODEL" ]; then
  echo "No .gguf model found. Put the model file in $DIR/model and run this again."
  exit 1
fi

# The model's profile (training preamble + sampling) travels next to it as <name>.corecs.json.
PROFILE="${MODEL%.gguf}.corecs.json"
[ -f "$PROFILE" ] && cp "$PROFILE" "$DIR/webui/profile.json"

if [ -x "$DIR/bin/llama-server" ]; then
  SERVER="$DIR/bin/llama-server"
elif command -v llama-server >/dev/null 2>&1; then
  SERVER="llama-server"
else
  echo "llama-server was not found. Install llama.cpp (macOS: brew install llama.cpp) and run this again."
  exit 1
fi

echo
echo "  Model: $MODEL"
echo "  Your browser opens at $URL once the model is loaded."
echo "  Everything runs on this computer. Press Ctrl+C to stop the tutor."
echo

(
  i=0
  while [ $i -lt 300 ]; do
    if curl -fs "$URL/health" >/dev/null 2>&1; then
      (open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true)
      break
    fi
    i=$((i + 1))
    sleep 1
  done
) &

exec "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" --parallel 1 --jinja --path "$DIR/webui"
