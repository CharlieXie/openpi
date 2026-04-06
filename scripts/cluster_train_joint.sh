#!/bin/bash
#SBATCH --job-name=joint-train
#SBATCH --gpus=pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/chuanlia001ssd/repos/pi_train/logs/train_%x-%j.out
#SBATCH --error=/projects/chuanlia001ssd/repos/pi_train/logs/train_%x-%j.err

# Usage: sbatch scripts/cluster_train_joint.sh [config]
# Default config: configs/waypoint_joint_libero.yaml

CONFIG="${1:-configs/waypoint_joint_libero.yaml}"

PIDIR="/projects/chuanlia001ssd/repos/pi_train"
ENVDIR="/home/chuanlia001/envs/openpi"
PYTHON="$ENVDIR/bin/python"

cd "$PIDIR"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

mkdir -p logs checkpoints

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Num GPUs: $SLURM_GPUS_ON_NODE"
echo "Config: $CONFIG"
echo "Date: $(date)"
echo "================"

export PYTHONPATH="$PIDIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
export PYTHONFAULTHANDLER=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

LOGFILE="logs/train_joint-${SLURM_JOB_ID}.log"

echo "Starting joint training with $NGPU GPUs..."
echo "Log file: $LOGFILE"
$ENVDIR/bin/torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
    scripts/train_waypoint_joint.py --config "$CONFIG" 2>&1 | tee "$LOGFILE"

echo "=== Done at $(date) ==="
