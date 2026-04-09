# LIBERO Evaluation Guide

Standalone evaluation of pi0.5 (JAX or PyTorch) on LIBERO benchmarks.

## Environment Setup (one-time)

### 1. System Dependencies

```bash
apt-get update && apt-get install -y \
  libosmesa6-dev libglew-dev libglfw3-dev \
  libglib2.0-0 libsm6 libxrender1 libxext6 \
  libegl1 libgles2-mesa-dev
```

### 2. OpenPI Main Environment

```bash
cd /workspace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 3. LIBERO Simulation Dependencies

```bash
cd /workspace/openpi
git submodule update --init --recursive

uv pip install --python .venv/bin/python mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1
uv pip install --python .venv/bin/python future easydict "gym==0.25.2" PyOpenGL==3.1.7

# CRITICAL: Downgrade numba to avoid segfault in Python 3.11
uv pip install --python .venv/bin/python "numba==0.60.0" "llvmlite==0.43.0"

# Install LIBERO package
uv pip install --python .venv/bin/python -e third_party/libero
```

### 4. LIBERO Config File

```bash
mkdir -p /tmp/libero && cat > /tmp/libero/config.yaml << 'EOF'
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
datasets: /workspace/openpi/third_party/libero/libero/datasets
assets: /workspace/openpi/third_party/libero/libero/libero/assets
EOF
```

---

## Quick Start

```bash
cd /workspace/openpi

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  LIBERO_CONFIG_PATH=/tmp/libero \
  PYTHONPATH=$PWD/third_party/libero:$PWD/src:$PYTHONPATH \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
  .venv/bin/python examples/libero/eval_libero_standalone.py \
    --config-name pi05_libero \
    --task-suite-name libero_10 \
    --num-trials-per-task 3 \
  2>&1 | tee logs/eval_libero10_$(date +%Y%m%d_%H%M%S).log
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config-name` | `pi05_libero` | Model config (`pi05_libero` for JAX, `pi05_libero_lora_pytorch` for PyTorch LoRA) |
| `--checkpoint-dir` | auto (GCS) | Override checkpoint path. If omitted, uses the default for the config |
| `--task-suite-name` | `libero_10` | Benchmark: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--num-trials-per-task` | `3` | Rollouts per task |
| `--seed` | `7` | Global seed for full reproducibility |
| `--video-out-path` | `data/libero/videos` | Where to save episode replay videos |
| `--max-tasks` | `0` | Limit to first N tasks (0 = all) |
| `--dump-actions` | `None` | Save action chunks to `.npz` for reproducibility verification |
| `--log-action-every` | `20` | Log action details every N steps |

## Run All 4 Suites in Parallel

Use the provided script to evaluate all 4 benchmarks on 4 GPUs simultaneously:

```bash
cd /workspace/openpi
bash scripts/eval_libero_all.sh
```

The script assigns one GPU per task suite and runs each with 3 trials/task. Logs and videos are saved per-suite. Monitor with:

```bash
# Watch all logs
tail -f logs/eval_libero_*.log

# Check results
cat data/libero/eval_*/eval_summary.json | python -m json.tool
```

## Evaluate a Fine-tuned Checkpoint

```bash
cd /workspace/openpi

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa \
  LIBERO_CONFIG_PATH=/tmp/libero \
  PYTHONPATH=$PWD/third_party/libero:$PWD/src:$PYTHONPATH \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
  .venv/bin/python examples/libero/eval_libero_standalone.py \
    --config-name pi05_libero_lora_pytorch \
    --checkpoint-dir checkpoints/pi05_libero_lora_pytorch/my_exp/30000 \
    --task-suite-name libero_10 \
    --num-trials-per-task 50 \
  2>&1 | tee logs/eval_finetuned_libero10.log
```

## Reproducibility

All random seeds are derived from `--seed`. With `XLA_FLAGS='--xla_gpu_deterministic_ops=true'`, two runs with identical arguments produce **bit-for-bit identical** action chunks.

Verify with:

```bash
# Run 1
... --seed 42 --dump-actions /tmp/run1.npz ...

# Run 2 (identical args)
... --seed 42 --dump-actions /tmp/run2.npz ...

# Compare
python -c "
import numpy as np
r1, r2 = np.load('/tmp/run1.npz'), np.load('/tmp/run2.npz')
assert sorted(r1.files) == sorted(r2.files)
diffs = [np.max(np.abs(r1[k]-r2[k])) for k in r1.files]
print(f'{sum(d==0 for d in diffs)}/{len(diffs)} exact, max_diff={max(diffs)}')
"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `uv run` reverts mujoco version | Use `.venv/bin/python` directly instead of `uv run` |
| numba segfault in Python 3.11 | `uv pip install --python .venv/bin/python "numba==0.60.0" "llvmlite==0.43.0"` |
| EGL rendering fails (headless) | Use `MUJOCO_GL=osmesa` |
| `torch.load` weights_only error | Script patches this automatically |
| Missing `future`/`easydict`/`gym` | `uv pip install --python .venv/bin/python future easydict "gym==0.25.2"` |

## Reference Results (pi0.5 official checkpoint)

| Suite | Tasks | Success Rate |
|-------|-------|-------------|
| libero_spatial | 10 | 98.8% |
| libero_object | 10 | 98.2% |
| libero_goal | 10 | 98.0% |
| libero_10 | 10 | 92.4% |
| **Average** | | **96.85%** |
