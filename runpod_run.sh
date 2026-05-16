#!/usr/bin/env bash
# Convenience inference wrapper with automatic GPU-count detection.
#
# Usage:
#   bash runpod_run.sh [MODE] [EXAMPLE] [FRAME_NUM] [SIZE]
#
#   MODE      cam | act | fast      (default: cam)
#   EXAMPLE   examples/ subdir name (default: 00 for cam, 05 for act, 03 for fast)
#   FRAME_NUM number of frames      (default: 161 cam / 121 act / 81 fast)
#   SIZE      480*832 | 720*1280    (default: 480*832)
#
# Examples:
#   bash runpod_run.sh                       # cam, example 00, 480p
#   bash runpod_run.sh act 05 121            # act-controlled, 480p
#   bash runpod_run.sh fast 03 81            # fast causal inference
#   bash runpod_run.sh cam 00 161 720*1280   # 720p
#
# GPU count is auto-detected and used for --nproc_per_node and --ulysses_size.
set -euo pipefail

MODE="${1:-cam}"
SIZE_DEFAULT="480*832"

case "$MODE" in
  cam)  SCRIPT=generate.py;      EXAMPLE="${2:-00}"; FRAME="${3:-161}"; EXTRA="" ;;
  act)  SCRIPT=generate.py;      EXAMPLE="${2:-05}"; FRAME="${3:-121}"; EXTRA="--allow_act2cam --sample_steps 20" ;;
  fast) SCRIPT=generate_fast.py; EXAMPLE="${2:-03}"; FRAME="${3:-81}";  EXTRA="" ;;
  *) echo "Unknown MODE '$MODE' (expected: cam|act|fast)"; exit 1 ;;
esac
SIZE="${4:-$SIZE_DEFAULT}"

CKPT_DIR="lingbot-world-base-cam"
if [ ! -d "$CKPT_DIR" ]; then
  echo "ERROR: '$CKPT_DIR' not found. Run: bash runpod_download.sh"; exit 1
fi
if [ ! -d "examples/$EXAMPLE" ]; then
  echo "ERROR: examples/$EXAMPLE not found. Available:"; ls examples/; exit 1
fi

# Auto-detect GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  NGPU=1
fi
[ "$NGPU" -lt 1 ] && NGPU=1
echo ">>> Detected $NGPU GPU(s) | mode=$MODE script=$SCRIPT example=$EXAMPLE frames=$FRAME size=$SIZE"

PROMPT="The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."

CMD=(torchrun --nproc_per_node="$NGPU" "$SCRIPT"
  --task i2v-A14B
  --size "$SIZE"
  --ckpt_dir "$CKPT_DIR"
  --image "examples/$EXAMPLE/image.jpg"
  --action_path "examples/$EXAMPLE"
  --dit_fsdp --t5_fsdp
  --ulysses_size "$NGPU"
  --frame_num "$FRAME"
  --prompt "$PROMPT")

# Single-GPU / tight-memory: drop FSDP sharding, offload T5 to CPU
if [ "$NGPU" -eq 1 ]; then
  CMD=(torchrun --nproc_per_node=1 "$SCRIPT"
    --task i2v-A14B
    --size "$SIZE"
    --ckpt_dir "$CKPT_DIR"
    --image "examples/$EXAMPLE/image.jpg"
    --action_path "examples/$EXAMPLE"
    --ulysses_size 1
    --t5_cpu
    --frame_num "$FRAME"
    --prompt "$PROMPT")
fi

# shellcheck disable=SC2206
[ -n "$EXTRA" ] && CMD+=($EXTRA)

set -x
"${CMD[@]}"
