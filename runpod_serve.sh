#!/usr/bin/env bash
# One-shot: ensure the interactive browser server is fully set up and running
# on a RunPod pod. Idempotent-ish; safe to re-run.
#
#   bash runpod_serve.sh
#
# Assumes runpod_setup.sh already ran (torch + flash-attn + deps). Downloads
# the Fast weights if missing, installs the web layer, launches the server
# bound to 0.0.0.0:${LBW_PORT:-8000}. Expose that port on the pod.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${LBW_PORT:-8000}"
export LBW_SIZE="${LBW_SIZE:-480*832}"
export LBW_T5_CPU="${LBW_T5_CPU:-1}"
export LBW_PLAY_FPS="${LBW_PLAY_FPS:-16}"

if [ ! -d lingbot-world-base-cam ] || [ ! -d lingbot-world-base-cam/lingbot_world_fast ]; then
  echo ">>> Fetching weights (base Cam + Fast) …"
  bash runpod_download.sh --fast
fi

echo ">>> Installing web layer …"
pip install -q -r server/requirements-server.txt

echo ">>> Starting interactive server on :$PORT"
exec env LBW_PORT="$PORT" bash server/run_server.sh
