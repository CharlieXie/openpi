#!/bin/bash
#SBATCH --job-name=eval-joint
#SBATCH --gpus=pro6000:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/chuanlia001ssd/repos/pi_train/logs/eval_%x-%j.out
#SBATCH --error=/projects/chuanlia001ssd/repos/pi_train/logs/eval_%x-%j.err

CONFIG="${1:-configs/eval_waypoint_joint_libero.yaml}"
EVAL_SUBDIR="${2:-eval_libero}"

PIDIR="/projects/chuanlia001ssd/repos/pi_train"
ENVDIR="/home/chuanlia001/envs/openpi"
PYTHON="$ENVDIR/bin/python"

cd "$PIDIR"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

CHECKPOINT=$(grep '^joint_checkpoint:' "$CONFIG" | awk '{print $2}')
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: joint_checkpoint not found in $CONFIG"
    exit 1
fi

EVAL_DIR="$CHECKPOINT/$EVAL_SUBDIR"
EVAL_VIDEOS="$EVAL_DIR/videos"
mkdir -p "$EVAL_DIR" "$EVAL_VIDEOS" logs

RUNTIME_CONFIG="$EVAL_DIR/eval_config_${SLURM_JOB_ID}.yaml"
sed "s|^video_out_path:.*|video_out_path: $EVAL_VIDEOS|" "$CONFIG" > "$RUNTIME_CONFIG"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Eval output dir: $EVAL_DIR"
echo "Date: $(date)"
echo "================"

export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH="$ENVDIR/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$PIDIR/.torch_cache"
export PYTHONPATH="$PIDIR/src:$PIDIR/third_party/libero:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RESULTS_GPU0="$EVAL_DIR/results_gpu0_${SLURM_JOB_ID}.json"
RESULTS_GPU1="$EVAL_DIR/results_gpu1_${SLURM_JOB_ID}.json"
LOG_GPU0="$EVAL_DIR/eval_gpu0_${SLURM_JOB_ID}.log"
LOG_GPU1="$EVAL_DIR/eval_gpu1_${SLURM_JOB_ID}.log"

echo "=== Launching 2 GPU processes: tasks 0-4 on GPU 0, tasks 5-9 on GPU 1 ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON -u -m openpi.waypoint.eval_libero \
    --config "$RUNTIME_CONFIG" --task-start 0 --task-end 5 \
    --results-file "$RESULTS_GPU0" \
    > "$LOG_GPU0" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON -u -m openpi.waypoint.eval_libero \
    --config "$RUNTIME_CONFIG" --task-start 5 --task-end 10 \
    --results-file "$RESULTS_GPU1" \
    > "$LOG_GPU1" 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (tasks 0-4)"
echo "GPU 1 PID: $PID1 (tasks 5-9)"

wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

echo ""
echo "GPU 0 exit code: $EXIT0"
echo "GPU 1 exit code: $EXIT1"
echo ""

MERGED_FILE="$EVAL_DIR/results_merged_${SLURM_JOB_ID}.json"

$PYTHON -c "
import json, sys

files = ['$RESULTS_GPU0', '$RESULTS_GPU1']
merged = {}
for f in files:
    try:
        with open(f) as fh:
            merged.update(json.load(fh))
    except Exception as e:
        print(f'Warning: cannot read {f}: {e}', file=sys.stderr)

if not merged:
    print('ERROR: No results to merge')
    sys.exit(1)

total_success = sum(r['successes'] for r in merged.values())
total_trials = sum(r['trials'] for r in merged.values())
overall_rate = total_success / total_trials if total_trials else 0

print()
print('=' * 60)
print(f'Overall success rate: {overall_rate:.2%} ({total_success}/{total_trials})')
for name, r in merged.items():
    print(f\"  {name}: {r['success_rate']:.2%} ({r['successes']}/{r['trials']})\")
print('=' * 60)

out = '$MERGED_FILE'
with open(out, 'w') as fh:
    json.dump({'overall': {'success_rate': overall_rate, 'successes': total_success, 'trials': total_trials}, 'tasks': merged}, fh, indent=2)
print(f'Merged results saved to {out}')
"

echo "=== Done at $(date) ==="
