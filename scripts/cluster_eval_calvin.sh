#!/bin/bash
#SBATCH --job-name=eval-calvin
#SBATCH --gpus=6000ada:2
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/chuanlia001ssd/repos/pi_train/logs/eval_calvin_%x-%j.out
#SBATCH --error=/projects/chuanlia001ssd/repos/pi_train/logs/eval_calvin_%x-%j.err

CONFIG="${1:-configs/eval_cluster_calvin.yaml}"

PIDIR="/projects/chuanlia001ssd/repos/pi_train"
ENVDIR="/home/chuanlia001/envs/openpi"
CALVIN="/projects/chuanlia001ssd/repos/calvin"
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

EVAL_DIR="$CHECKPOINT/eval_calvin"
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
export PYTHONPATH="$PIDIR/src:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export NUMBA_DISABLE_JIT=1
export CALVIN_ROOT="$CALVIN"

RESULTS_GPU0="$EVAL_DIR/results_gpu0_${SLURM_JOB_ID}.json"
RESULTS_GPU1="$EVAL_DIR/results_gpu1_${SLURM_JOB_ID}.json"
LOG_GPU0="$EVAL_DIR/eval_gpu0_${SLURM_JOB_ID}.log"
LOG_GPU1="$EVAL_DIR/eval_gpu1_${SLURM_JOB_ID}.log"

echo "=== Launching 2 GPU processes: seq 0-249 on GPU 0, seq 250-499 on GPU 1 ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON -u -m openpi.waypoint.eval_calvin \
    --config "$RUNTIME_CONFIG" --seq-start 0 --seq-end 250 \
    --results-file "$RESULTS_GPU0" \
    > "$LOG_GPU0" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON -u -m openpi.waypoint.eval_calvin \
    --config "$RUNTIME_CONFIG" --seq-start 250 --seq-end 500 \
    --results-file "$RESULTS_GPU1" \
    > "$LOG_GPU1" 2>&1 &
PID1=$!

echo "GPU 0 PID: $PID0 (seq 0-249)"
echo "GPU 1 PID: $PID1 (seq 250-499)"

wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

echo ""
echo "GPU 0 exit: $EXIT0"
echo "GPU 1 exit: $EXIT1"
echo ""

MERGED_FILE="$EVAL_DIR/results_merged_${SLURM_JOB_ID}.json"

$PYTHON -c "
import json, sys, numpy as np

files = ['$RESULTS_GPU0', '$RESULTS_GPU1']
all_results = []
for f in files:
    try:
        with open(f) as fh:
            data = json.load(fh)
            all_results.extend(data['per_sequence_results'])
            print(f'{f}: {data[\"num_sequences\"]} seq, avg_len={data[\"avg_seq_len\"]:.3f}')
    except Exception as e:
        print(f'Warning: cannot read {f}: {e}', file=sys.stderr)

if not all_results:
    print('ERROR: No results to merge')
    sys.exit(1)

n = len(all_results)
avg = np.mean(all_results)
print()
print('=' * 60)
print(f'CALVIN Merged Results ({n} sequences)')
print(f'Average successful sequence length: {avg:.3f}')
print('Chain success rates:')
for i in range(1, 6):
    s = sum(r >= i for r in all_results)
    print(f'  {i}/5: {s/n:.1%} ({s}/{n})')
print('=' * 60)

out = '$MERGED_FILE'
merged = {
    'avg_seq_len': float(avg),
    'chain_sr': {str(i): float(sum(r >= i for r in all_results)/n) for i in range(1,6)},
    'num_sequences': n,
    'per_sequence_results': all_results,
}
with open(out, 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Merged results saved to {out}')
"

echo "=== Done at $(date) ==="
