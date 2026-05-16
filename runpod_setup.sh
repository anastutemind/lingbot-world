#!/usr/bin/env bash
# RunPod environment setup for LingBot-World.
# Run this ONCE on a fresh RunPod CUDA pod (recommended template:
# "RunPod PyTorch 2.4+" on Ubuntu 22.04, CUDA 12.x, >=1 NVIDIA GPU).
#
#   bash runpod_setup.sh
#
# Notes:
# - flash-attn is compiled against the installed torch/CUDA; this can take
#   10-20 min. A prebuilt wheel matching your torch/cuda/python is much faster
#   if available (see RUNPOD.md).
# - Use the persistent /workspace volume so deps/weights survive pod restarts.
set -euo pipefail

echo ">>> System packages (ffmpeg for imageio, build tools for flash-attn)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y --no-install-recommends ffmpeg git ninja-build build-essential
fi

echo ">>> Python: $(python --version 2>&1)  pip: $(pip --version 2>&1)"

echo ">>> Upgrading pip tooling"
pip install --upgrade pip setuptools wheel ninja packaging

echo ">>> Installing project dependencies (excluding flash_attn; built separately)"
# requirements.txt pins flash_attn, which must be built with --no-build-isolation
# AFTER torch is present. Install everything else first.
grep -v -i '^flash[_-]attn' requirements.txt > /tmp/req_no_fa.txt
pip install -r /tmp/req_no_fa.txt

echo ">>> Verifying torch sees CUDA"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available — use a GPU pod / CUDA template"
print("GPUs:", torch.cuda.device_count())
PY

echo ">>> Building flash-attn (this is slow; be patient)"
pip install flash-attn --no-build-isolation

echo ">>> Installing huggingface-cli for weight downloads"
pip install "huggingface_hub[cli]"

echo ">>> Setup complete. Next: bash runpod_download.sh   then   bash runpod_run.sh"
