# OpenPI LIBERO Setup Guide (Eval + Training)

AI-executable setup guide. Follow steps sequentially on a fresh machine with NVIDIA GPUs and Ubuntu 22.04+.

## Prerequisites

- Ubuntu 22.04+ with NVIDIA GPUs (RTX 4090 24GB or better)
- `uv` package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git with submodules
- Data already downloaded at `/workspace/libero` (HuggingFace dataset `physical-intelligence/libero`)
- PyTorch base model at `/workspace/models/pi05_base_pytorch` (for LoRA training)
- wandb API key (set via `WANDB_API_KEY`)

## 1. System Dependencies

```bash
apt-get update && apt-get install -y \
  software-properties-common make g++ clang \
  libosmesa6-dev libglew-dev libglfw3-dev \
  libglib2.0-0 libsm6 libxrender1 libxext6 \
  libegl1 libgles2-mesa-dev tmux

# Python 3.8 (needed for client-server eval only, skip if using standalone eval)
add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
apt-get install -y python3.8 python3.8-venv python3.8-dev python3.8-distutils
```

## 2. Clone and Initialize Repo

```bash
cd /workspace
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or if already cloned:
cd /workspace/openpi && git submodule update --init --recursive
```

## 3. Install OpenPI Main Environment (Python 3.11)

```bash
cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 3a. Install LIBERO Simulation Dependencies in Main Env

Required for standalone eval (single-process, no websocket).

```bash
cd /workspace/openpi

# Install mujoco, robosuite, and LIBERO deps into the main .venv
uv pip install --python .venv/bin/python mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1
uv pip install --python .venv/bin/python future easydict "gym==0.25.2" PyOpenGL==3.1.7

# CRITICAL: Downgrade numba to avoid segfault in Python 3.11
uv pip install --python .venv/bin/python "numba==0.60.0" "llvmlite==0.43.0"

# Install LIBERO package
uv pip install --python .venv/bin/python -e third_party/libero
```

### 3b. Apply Transformers Patches (for PyTorch training only)

```bash
cd /workspace/openpi
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## 4. Create LIBERO Config File

Prevents interactive prompt from LIBERO's init script.

```bash
mkdir -p /tmp/libero && cat > /tmp/libero/config.yaml << 'EOF'
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
datasets: /workspace/openpi/third_party/libero/libero/datasets
assets: /workspace/openpi/third_party/libero/libero/libero/assets
EOF
```

## 5. Symlink Local Dataset for LeRobot

LeRobot looks for data under `~/.cache/huggingface/lerobot/{repo_id}`.

```bash
mkdir -p ~/.cache/huggingface/lerobot/physical-intelligence
ln -sfn /workspace/libero ~/.cache/huggingface/lerobot/physical-intelligence/libero
```

---

## Eval: Standalone (Recommended)

Single-process evaluation: loads model + runs LIBERO sim together. No websocket.

Script: `examples/libero/eval_libero_standalone.py`

```bash
cd /workspace/openpi
mkdir -p logs

# Example: libero_10, 3 trials/task, GPU 1
tmux new-session -d -s eval_standalone \
  "CUDA_VISIBLE_DEVICES=1 \
   MUJOCO_GL=osmesa \
   LIBERO_CONFIG_PATH=/tmp/libero \
   PYTHONPATH=\$PWD/third_party/libero:\$PWD/src:\$PYTHONPATH \
   .venv/bin/python examples/libero/eval_libero_standalone.py \
     --task-suite-name libero_10 \
     --num-trials-per-task 3 \
   2>&1 | tee logs/eval_standalone_libero10_\$(date +%Y%m%d_%H%M%S).log"
```

Available task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task-suite-name` | `libero_10` | Which LIBERO benchmark |
| `--num-trials-per-task` | `3` | Rollouts per task |
| `--config-name` | `pi05_libero` | OpenPI model config |
| `--checkpoint-dir` | auto (GCS) | Override checkpoint path |
| `--video-out-path` | `data/libero/videos` | Where to save replay videos |

## Eval: Client-Server (Alternative)

Two separate processes communicating via websocket. Useful if environments need Python 3.8.

### Setup Python 3.8 env (one-time)

```bash
cd /workspace/openpi
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate
```

### Run

Terminal 1 — Policy Server (Python 3.11, GPU 0):
```bash
cd /workspace/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --env LIBERO
```

Terminal 2 — Eval Client (Python 3.8):
```bash
cd /workspace/openpi
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
MUJOCO_GL=osmesa python examples/libero/main.py \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 3
```

---

## Training: PyTorch LoRA Fine-tune

### Step 1: Compute Norm Stats (fast, no image decoding)

```bash
cd /workspace/openpi
uv run scripts/compute_norm_stats_parquet.py \
  --config-name pi05_libero_lora_pytorch \
  --local-parquet-dir /workspace/libero/data
```

Takes ~30 seconds. Saves to `assets/pi05_libero_lora_pytorch/physical-intelligence/libero/norm_stats.json`.

### Step 2: Update Config

In `src/openpi/training/config.py`, set `pytorch_weight_path` in the `pi05_libero_lora_pytorch` config:

```python
pytorch_weight_path="/workspace/models/pi05_base_pytorch",
```

### Step 3: Start Training

```bash
cd /workspace/openpi
mkdir -p logs

tmux new-session -d -s training \
  "CUDA_VISIBLE_DEVICES=2 \
   OPENPI_DATASET_LOCAL_DIR=/workspace/libero \
   WANDB_API_KEY=<your_wandb_key> \
   WANDB_RUN_NAME=pi05_libero_lora_r16_bs16_30k \
   .venv/bin/python scripts/train_pytorch.py pi05_libero_lora_pytorch \
     --exp_name pi05_libero_lora_r16_bs16_30k --overwrite \
   2>&1 | tee logs/train_pi05_libero_lora_\$(date +%Y%m%d_%H%M%S).log"
```

### Key environment variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) |
| `OPENPI_DATASET_LOCAL_DIR` | Local path to LeRobot dataset (skips HuggingFace download) |
| `WANDB_API_KEY` | Weights & Biases API key |
| `WANDB_RUN_NAME` | Human-readable run name on wandb dashboard |
| `WANDB_MODE=disabled` | Set this to skip wandb entirely |

### Training config details (pi05_libero_lora_pytorch)

- Model: pi0.5, LoRA rank=16, alpha=16
- Batch size: 16
- LR: cosine decay, peak 1e-4, end 1e-5, warmup 1000 steps
- Total steps: 30,000
- Trainable: ~441M params (12.1%), frozen: ~3.2B params
- SigLIP vision encoder: trainable
- Precision: bfloat16
- GPU memory: ~12GB (fits on RTX 4090)

### Monitor

```bash
# Attach to tmux session
tmux attach -t training

# Or tail the log
tail -f /workspace/openpi/logs/train_pi05_libero_lora_*.log

# Check tmux sessions
tmux ls
```

### Checkpoints

Saved to: `checkpoints/pi05_libero_lora_pytorch/<exp_name>/`

---

## Eval a Fine-tuned Checkpoint

### Standalone eval

```bash
cd /workspace/openpi
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/tmp/libero \
  PYTHONPATH=$PWD/third_party/libero:$PWD/src \
  .venv/bin/python examples/libero/eval_libero_standalone.py \
    --task-suite-name libero_10 \
    --num-trials-per-task 50 \
    --config-name pi05_libero_lora_pytorch \
    --checkpoint-dir checkpoints/pi05_libero_lora_pytorch/pi05_libero_lora_r16_bs16_30k/30000 \
  2>&1 | tee logs/eval_finetuned_libero10.log
```

### Client-server eval

```bash
# Server
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero_lora_pytorch \
  --policy.dir checkpoints/pi05_libero_lora_pytorch/pi05_libero_lora_r16_bs16_30k/30000
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `uv run` reverts mujoco version | Use `.venv/bin/python` directly instead of `uv run` |
| numba segfault in Python 3.11 | `uv pip install --python .venv/bin/python "numba==0.60.0" "llvmlite==0.43.0"` |
| EGL rendering fails (headless) | Use `MUJOCO_GL=osmesa` |
| HuggingFace rate limit (429) | Set `OPENPI_DATASET_LOCAL_DIR=/workspace/libero` and symlink cache |
| `torch.load` weights_only error | Script already patches via `functools.partial(torch.load, weights_only=False)` |
| wandb not configured | Set `WANDB_API_KEY=...` or `WANDB_MODE=disabled` |
| Missing `future`/`easydict`/`gym` | Install: `uv pip install --python .venv/bin/python future easydict "gym==0.25.2"` |

## Reference: Benchmark Results (pi0.5 official checkpoint)

| Suite | Success Rate |
|-------|-------------|
| libero_spatial | 98.8% |
| libero_object | 98.2% |
| libero_goal | 98.0% |
| libero_10 | 92.4% |
| **Average** | **96.85%** |

Our eval (pi0.5 official, libero_10, 3 trials/task): **93.3%** (28/30)
