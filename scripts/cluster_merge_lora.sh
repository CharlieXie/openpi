#!/bin/bash
#SBATCH --job-name=merge-lora
#SBATCH --gpus=a5000:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/projects/chuanlia001ssd/repos/pi_train/logs/merge_%x-%j.out
#SBATCH --error=/projects/chuanlia001ssd/repos/pi_train/logs/merge_%x-%j.err

# Usage: sbatch scripts/cluster_merge_lora.sh [base] [lora_dir] [config]

PIDIR="/projects/chuanlia001ssd/repos/pi_train"
PYTHON="/home/chuanlia001/envs/openpi/bin/python"

BASE="${1:-/projects/chuanlia001ssd/models/900/model_merged.safetensors}"
LORA_DIR="${2:-/projects/chuanlia001/checkpoints/waypoint_joint_libero_10_lora_sg_0.3_reproduce_continue1/700}"
CONFIG="${3:-configs/waypoint_joint_libero.yaml}"
OUTPUT="${LORA_DIR}/model_merged.safetensors"

cd "$PIDIR"

echo "=== Merge LoRA Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Base: $BASE"
echo "LoRA: ${LORA_DIR}/lora.safetensors"
echo "Output: $OUTPUT"
echo "Date: $(date)"
echo "======================"

export PYTHONPATH="$PIDIR/src:${PYTHONPATH:-}"

$PYTHON scripts/merge_lora.py \
    --base "$BASE" \
    --lora "${LORA_DIR}/lora.safetensors" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --device cuda

echo "=== Done at $(date) ==="
