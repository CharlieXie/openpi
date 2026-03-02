#!/bin/bash
set -eo pipefail

cd /workspace/openpi

AE=/workspace/openpi/checkpoints/waypoint_ae_libero_spatial_03/1300
LOG_DIR=/workspace/openpi/logs/grid
VIDEO_BASE=data/libero/videos_wp_spatial_soup
BASE_CONFIG=configs/eval_waypoint_libero.yaml

mkdir -p "$LOG_DIR"

SOUP_DIRS=(
    /workspace/openpi/checkpoints/waypoint_vlm_libero_spatial_soup_t10_800_t11_1800
    /workspace/openpi/checkpoints/waypoint_vlm_libero_spatial_soup_t10_800w06_t11_1800w04
    /workspace/openpi/checkpoints/waypoint_vlm_libero_spatial_soup_t10_800w04_t11_1800w06
)
LOG_NAMES=(
    soup_t10w05_t11w05__ae_ae03_1300
    soup_t10w06_t11w04__ae_ae03_1300
    soup_t10w04_t11w06__ae_ae03_1300
)
VIDEO_DIRS=(
    soup_t10w05_t11w05
    soup_t10w06_t11w04
    soup_t10w04_t11w06
)
GPUS=(0 1 0)

PIDS=()

for i in 0 1 2; do
    log="${LOG_DIR}/${LOG_NAMES[$i]}.log"
    video="${VIDEO_BASE}/${VIDEO_DIRS[$i]}"
    gpu=${GPUS[$i]}

    echo "[START] GPU:$gpu  ${LOG_NAMES[$i]}  ->  $log"

    MUJOCO_GL=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH=$PWD/src:$PWD/third_party/libero:${PYTHONPATH:-} \
    PYTHONFAULTHANDLER=1 \
    .venv/bin/python -u src/openpi/waypoint/eval_libero.py \
        --config "$BASE_CONFIG" \
        --vlm-checkpoint "${SOUP_DIRS[$i]}" \
        --ae-checkpoint "$AE" \
        --video-out-path "$video" \
        > "$log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All 3 launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion ..."

for idx in "${!PIDS[@]}"; do
    wait "${PIDS[$idx]}" && rc=$? || rc=$?
    echo "  ${LOG_NAMES[$idx]} (PID ${PIDS[$idx]}) finished  exit=$rc"
done

echo ""
echo "=== Done ==="
