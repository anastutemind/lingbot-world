# Running LingBot-World on RunPod

This repo's inference is CUDA/NVIDIA-GPU only (built on Wan2.2, needs
`flash-attn`). It cannot run on the macOS machine it was cloned to — use a
RunPod GPU pod. These helper scripts automate the whole flow.

## 1. Create the pod

- **Template:** RunPod PyTorch 2.4+ (Ubuntu 22.04, CUDA 12.x)
- **GPU:** The A14B model is large. Recommended:
  - **8× A100/H100 80GB** for the README-default 480p/720p multi-GPU runs
  - **1× A100/H100 80GB** works for smaller `frame_num` at 480p
    (the run wrapper auto-switches to a single-GPU, `--t5_cpu` config)
  - Limited VRAM: see the community 4-bit NF4 build linked in `README.md`
- **Volume:** attach a persistent volume mounted at `/workspace`
  (weights are tens of GB — you don't want to re-download every restart)
- **Disk:** allow ~120 GB for base (Cam) + Fast weights

## 2. On the pod

```sh
cd /workspace
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world

bash runpod_setup.sh            # system + python deps + flash-attn (slow, ~10-20m)
bash runpod_download.sh         # base (Cam) weights   (add --fast for Fast weights)
bash runpod_run.sh              # auto-detects GPU count and runs example 00
```

Copy the three `runpod_*.sh` files and this guide onto the pod if you didn't
push them upstream — e.g. `scp` them, paste via the RunPod web terminal, or
keep them in your own fork.

## 3. Run modes

`runpod_run.sh [MODE] [EXAMPLE] [FRAME_NUM] [SIZE]`

| Command | What it does |
| --- | --- |
| `bash runpod_run.sh` | Cam-controlled, example 00, 161 frames, 480p |
| `bash runpod_run.sh cam 00 961 480*832` | ~1-min video @16fps (needs lots of VRAM) |
| `bash runpod_run.sh act 05 121` | Action-controlled |
| `bash runpod_run.sh fast 03 81` | Fast causal inference (needs `--fast` weights) |
| `bash runpod_run.sh cam 00 161 720*1280` | 720p |

GPU count is detected via `nvidia-smi` and wired into `--nproc_per_node`
and `--ulysses_size`. On a single GPU it drops FSDP sharding and adds
`--t5_cpu` to fit memory.

## 4. flash-attn build is slow

`pip install flash-attn --no-build-isolation` compiles from source against the
installed torch/CUDA. To skip the wait, grab a prebuilt wheel matching your
`python` / `torch` / `cuda` from the
[flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases),
then `pip install <wheel>.whl` instead of running the build step.

## 5. Outputs

Generated videos are written by `generate.py` / `generate_fast.py` into the
working directory (see their `--save_file` / default output handling). Pull
them off the pod via the RunPod file browser or `scp`.

## Troubleshooting

- **`CUDA not available`** in setup → you booted a CPU pod; recreate with a GPU.
- **OOM** → lower `FRAME_NUM`, use 480p, add more GPUs, or use the NF4 build.
- **flash-attn build fails** → ensure `ninja` + `build-essential` installed
  (setup script does this) and torch was installed *before* flash-attn.
- **Slow HF download** → scripts enable `hf_transfer`; ensure good pod network.
