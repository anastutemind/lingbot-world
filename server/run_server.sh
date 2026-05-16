#!/usr/bin/env bash
# Launch the LingBot-World interactive browser server.
# Run from the repo root (so `import wan` and examples/ resolve).
#
#   bash server/run_server.sh
#
# Env (optional): LBW_SIZE, LBW_T5_CPU, LBW_PLAY_FPS, LBW_PORT, LBW_CKPT_DIR
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/server${PYTHONPATH:+:$PYTHONPATH}"
PORT="${LBW_PORT:-8000}"
CKPT="${LBW_CKPT_DIR:-lingbot-world-base-cam}"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint dir '$CKPT' missing. Run: bash runpod_download.sh --fast"
  exit 1
fi

echo ">>> Serving on 0.0.0.0:${PORT}  (ckpt=$CKPT size=${LBW_SIZE:-480*832})"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" \
     --app-dir "$REPO/server" --ws-ping-interval 20 --ws-ping-timeout 20
