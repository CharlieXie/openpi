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

## Eval Results

### Our eval (pi0.5 JAX official checkpoint, 3 trials/task, seed=7, 2026-04-09)

| Suite | Success | Rate | Failed Tasks |
|-------|---------|------|--------------|
| libero_spatial | 30/30 | **100.0%** | -- |
| libero_object | 30/30 | **100.0%** | -- |
| libero_goal | 29/30 | **96.7%** | open the middle drawer of the cabinet (2/3) |
| libero_10 | 29/30 | **96.7%** | put both moka pots on the stove (2/3) |
| **Average** | **118/120** | **98.3%** | |

<details>
<summary>Per-task breakdown (click to expand)</summary>

**libero_spatial** (100.0%)

| Task | Rate |
|------|------|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 3/3 |
| pick up the black bowl next to the ramekin and place it on the plate | 3/3 |
| pick up the black bowl from table center and place it on the plate | 3/3 |
| pick up the black bowl on the cookie box and place it on the plate | 3/3 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 3/3 |
| pick up the black bowl on the ramekin and place it on the plate | 3/3 |
| pick up the black bowl next to the cookie box and place it on the plate | 3/3 |
| pick up the black bowl on the stove and place it on the plate | 3/3 |
| pick up the black bowl next to the plate and place it on the plate | 3/3 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 3/3 |

**libero_object** (100.0%)

| Task | Rate |
|------|------|
| pick up the alphabet soup and place it in the basket | 3/3 |
| pick up the cream cheese and place it in the basket | 3/3 |
| pick up the salad dressing and place it in the basket | 3/3 |
| pick up the bbq sauce and place it in the basket | 3/3 |
| pick up the ketchup and place it in the basket | 3/3 |
| pick up the tomato sauce and place it in the basket | 3/3 |
| pick up the butter and place it in the basket | 3/3 |
| pick up the milk and place it in the basket | 3/3 |
| pick up the chocolate pudding and place it in the basket | 3/3 |
| pick up the orange juice and place it in the basket | 3/3 |

**libero_goal** (96.7%)

| Task | Rate |
|------|------|
| open the middle drawer of the cabinet | **2/3** |
| put the bowl on the stove | 3/3 |
| put the wine bottle on top of the cabinet | 3/3 |
| open the top drawer and put the bowl inside | 3/3 |
| put the bowl on top of the cabinet | 3/3 |
| push the plate to the front of the stove | 3/3 |
| put the cream cheese in the bowl | 3/3 |
| turn on the stove | 3/3 |
| put the bowl on the plate | 3/3 |
| put the wine bottle on the rack | 3/3 |

**libero_10** (96.7%)

| Task | Rate |
|------|------|
| put both the alphabet soup and the tomato sauce in the basket | 3/3 |
| put both the cream cheese box and the butter in the basket | 3/3 |
| turn on the stove and put the moka pot on it | 3/3 |
| put the black bowl in the bottom drawer of the cabinet and close it | 3/3 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 3/3 |
| pick up the book and place it in the back compartment of the caddy | 3/3 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 3/3 |
| put both the alphabet soup and the cream cheese box in the basket | 3/3 |
| put both moka pots on the stove | **2/3** |
| put the yellow and white mug in the microwave and close it | 3/3 |

</details>

### Reference results (pi0.5 official, from paper, 50 trials/task)

| Suite | Success Rate |
|-------|-------------|
| libero_spatial | 98.8% |
| libero_object | 98.2% |
| libero_goal | 98.0% |
| libero_10 | 92.4% |
| **Average** | **96.85%** |
