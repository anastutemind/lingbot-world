#!/usr/bin/env bash
# Download LingBot-World model weights from HuggingFace into ./lingbot-world-base-cam
#
#   bash runpod_download.sh            # base (Cam) only
#   bash runpod_download.sh --fast     # base (Cam) + Fast weights
#
# Weights are large (tens of GB). Run from inside the persistent volume
# (e.g. /workspace/lingbot-world) so they survive pod restarts.
set -euo pipefail

WANT_FAST=0
[ "${1:-}" = "--fast" ] && WANT_FAST=1

# Faster, more reliable transfers
pip install -q "huggingface_hub[hf_transfer]" >/dev/null 2>&1 || true
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ">>> Downloading LingBot-World-Base (Cam) -> ./lingbot-world-base-cam"
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam

if [ "$WANT_FAST" = "1" ]; then
  echo ">>> Downloading LingBot-World-Fast -> ./lingbot-world-base-cam/lingbot_world_fast"
  huggingface-cli download robbyant/lingbot-world-fast \
    --local-dir ./lingbot-world-base-cam/lingbot_world_fast
fi

echo ">>> Download complete."
du -sh ./lingbot-world-base-cam 2>/dev/null || true
