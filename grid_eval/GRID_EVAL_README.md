# Waypoint Grid Evaluation -- Handoff Notes

## What Was Done

### 1. Modified eval script to accept CLI overrides
**File**: `src/openpi/waypoint/eval_libero.py`

Added `--vlm-checkpoint`, `--ae-checkpoint`, `--num-trials`, `--video-out-path` CLI args that override YAML config values. This avoids creating dozens of config files.

### 2. Created base eval config
**File**: `configs/eval_grid_base.yaml`

Key settings: `task_suite: libero_spatial`, `num_trials_per_task: 3`, `dataset_statistics_path: /workspace/data/dataset_statistics.json`.

### 3. Created orchestrator script
**File**: `run_grid_eval.py`

Python script managing parallel evaluation. Supports 3 phases:
```bash
cd /workspace/openpi

# Phase 1: sweep all 14 VLMs with anchor AE (ae03/1300)
.venv/bin/python run_grid_eval.py --phase 1

# Phase 2: sweep all 13 AEs for top N VLMs from Phase 1
.venv/bin/python run_grid_eval.py --phase 2 --top-vlms 3

# Phase 3: validate top combos with more trials
.venv/bin/python run_grid_eval.py --phase 3 --top-n 5 --trials 20

# View results from any completed phase
.venv/bin/python run_grid_eval.py --report
```

### 4. Fixed environment issues
- Created `~/.libero/config.yaml` (LIBERO prompts interactively without it, causing EOFError in subprocess)
- Installed eval dependencies: `robosuite==1.4.1 transforms3d bddl easydict gym==0.26.2` + `libero` (editable)
- transformers patch, tensorflow 2.15.0, torch.load fix were already applied

## Current Status (as of handoff)

**Phase 1 is running** -- orchestrator is alive, 10 of 14 jobs executing, 4 queued.

The orchestrator will:
1. Wait for all 14 jobs to finish
2. Print leaderboard to stdout
3. Save results to `logs/grid/results_phase1.json`
4. Exit

**It does NOT auto-start Phase 2.** You must run Phase 2 manually after Phase 1 finishes.

## What Remains To Do

### Phase 1: Wait for completion (~15-20 min remaining)
```bash
# Check if Phase 1 is done:
cat /workspace/openpi/logs/grid/results_phase1.json 2>/dev/null && echo "Phase 1 DONE" || echo "Phase 1 still running"

# Or check orchestrator:
tail -30 /workspace/openpi/logs/grid/phase1_orchestrator.log

# Or check individual job completion:
grep "Overall success rate" /workspace/openpi/logs/grid/vlm_*.log

# Check process status:
ps aux | grep run_grid_eval | grep -v grep
```

### Phase 2: Run AE sweep for top VLMs
```bash
cd /workspace/openpi
.venv/bin/python run_grid_eval.py --phase 2 --top-vlms 3
```
This reads `logs/grid/results_phase1.json`, takes top 3 VLMs, sweeps all 13 AEs for each (39 jobs). Takes ~1.5 hours.

### Phase 3: Validate top combinations
```bash
cd /workspace/openpi
.venv/bin/python run_grid_eval.py --phase 3 --top-n 5 --trials 20
```
This reads `logs/grid/results_phase2.json`, validates top 5 combos with 20 trials each. Takes ~1 hour.

## Architecture

```
Checkpoints:
  VLM t10: steps 800, 1000, 1200, 1400  (4 checkpoints)
  VLM t11: steps 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000  (10 checkpoints)
  AE 03:   steps 800, 900, 1000, 1100, 1300, 1800, 1900, 2000  (8 checkpoints)
  AE 04:   steps 100, 200, 300, 400, 500  (5 checkpoints)

Strategy: Anchor-based search
  Phase 1: Fix AE=ae03/1300, sweep 14 VLMs    -> find top 3
  Phase 2: Top 3 VLMs x 13 AEs = 39 jobs      -> find top 5 combos
  Phase 3: Top 5 combos x 20 trials            -> final ranking

Resources: 2 GPUs x 96GB, ~15GB per eval, max 5 per GPU = 10 parallel
```

## Key Files
| File | Purpose |
|------|---------|
| `run_grid_eval.py` | Orchestrator script |
| `configs/eval_grid_base.yaml` | Base eval config |
| `src/openpi/waypoint/eval_libero.py` | Eval script (modified with CLI args) |
| `logs/grid/*.log` | Individual job logs |
| `logs/grid/results_phase{1,2,3}.json` | Phase results (auto-generated) |
| `logs/grid/phase1_orchestrator.log` | Orchestrator output |
| `~/.libero/config.yaml` | LIBERO config (prevents interactive prompt) |

## Known Issues
1. **Model loading is slow with high parallelism** -- 10 concurrent processes cause ~5x slowdown on model init (291s vs 63s single). This is CPU/IO contention during weight allocation. Acceptable since eval time (30 min) dominates.
2. **Orchestrator skip logic** -- if a job's log already exists with results, it's skipped. To re-run, delete the log file first.
3. **max-per-gpu** -- default is 5. Can try `--max-per-gpu 6` if GPU memory allows (each job ~15GB, 6x15=90GB < 96GB).
