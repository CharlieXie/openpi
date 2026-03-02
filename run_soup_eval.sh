#!/bin/bash
set -eo pipefail

cd /workspace/openpi

VLM_T10=/workspace/openpi/checkpoints/waypoint_vlm_libero_spatial_t10/800
VLM_T11=/workspace/openpi/checkpoints/waypoint_vlm_libero_spatial_t11/1800
AE=/workspace/openpi/checkpoints/waypoint_ae_libero_spatial_03/1300

CKPT_BASE=/workspace/openpi/checkpoints
LOG_DIR=/workspace/openpi/logs/grid
VIDEO_BASE=data/libero/videos_wp_spatial_soup
BASE_CONFIG=configs/eval_waypoint_libero.yaml

mkdir -p "$LOG_DIR"

# 6 weight combinations for t10/t11 (complementing existing 0.4/0.6, 0.5/0.5, 0.6/0.4)
W10_VALS=(0.1  0.2  0.3  0.7  0.8  0.9)
W11_VALS=(0.9  0.8  0.7  0.3  0.2  0.1)

# GPU assignment: distribute 3 per GPU
#   Batch 1 (idx 0,1,2): GPU 0, 1, 0
#   Batch 2 (idx 3,4,5): GPU 1, 0, 1
GPU_MAP=(0 1 0 1 0 1)

run_eval() {
    local vlm_ckpt=$1
    local gpu=$2
    local log_file=$3
    local video_path=$4

    MUJOCO_GL=osmesa \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH=$PWD/src:$PWD/third_party/libero:${PYTHONPATH:-} \
    PYTHONFAULTHANDLER=1 \
    .venv/bin/python -u src/openpi/waypoint/eval_libero.py \
        --config "$BASE_CONFIG" \
        --vlm-checkpoint "$vlm_ckpt" \
        --ae-checkpoint "$AE" \
        --video-out-path "$video_path" \
        > "$log_file" 2>&1 &

    echo $!
}

wait_for_inference() {
    local log_file=$1
    local label=$2
    local timeout=600
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if [ -f "$log_file" ] && grep -q "Total model loading" "$log_file" 2>/dev/null; then
            echo "    ✓ $label — model loaded, inference started"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "    ⚠ $label — timeout waiting for model loading (${timeout}s)"
    return 0
}

# ── Step 1: Create soup checkpoints ──────────────────────────────────────────
echo "============================================================"
echo "  Step 1: Creating model soup checkpoints"
echo "============================================================"

SOUP_DIRS=()
SOUP_LABELS=()

for i in "${!W10_VALS[@]}"; do
    w10=${W10_VALS[$i]}
    w11=${W11_VALS[$i]}
    w10i=$(printf "%02d" "$(echo "$w10 * 10 / 1" | bc)")
    w11i=$(printf "%02d" "$(echo "$w11 * 10 / 1" | bc)")

    label="t10w${w10i}_t11w${w11i}"
    soup_dir="${CKPT_BASE}/waypoint_vlm_libero_spatial_soup_t10_800w${w10i}_t11_1800w${w11i}"

    if [ -f "$soup_dir/model.safetensors" ]; then
        echo "  [SKIP] $label  already exists"
    else
        echo "  [CREATE] $label  (t10=${w10}, t11=${w11})"
        .venv/bin/python /workspace/model_soup.py \
            "$VLM_T10" "$VLM_T11" -w "$w10" "$w11" -o "$soup_dir"
    fi

    SOUP_DIRS+=("$soup_dir")
    SOUP_LABELS+=("$label")
done

# ── Step 2: Launch evaluations ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 2: Running evaluations (2 GPUs, max 3 concurrent starts)"
echo "============================================================"

BATCH1_PIDS=()
BATCH1_LOGS=()
BATCH1_LABELS=()

# Batch 1: indices 0..2
for i in 0 1 2; do
    w10=${W10_VALS[$i]}
    w11=${W11_VALS[$i]}
    w10i=$(printf "%02d" "$(echo "$w10 * 10 / 1" | bc)")
    w11i=$(printf "%02d" "$(echo "$w11 * 10 / 1" | bc)")

    gpu=${GPU_MAP[$i]}
    log="${LOG_DIR}/soup_t10w${w10i}_t11w${w11i}__ae_ae03_1300.log"
    video="${VIDEO_BASE}/soup_t10w${w10i}_t11w${w11i}"

    echo "  [BATCH-1] GPU:$gpu  ${SOUP_LABELS[$i]}  ->  $log"
    pid=$(run_eval "${SOUP_DIRS[$i]}" "$gpu" "$log" "$video")
    BATCH1_PIDS+=("$pid")
    BATCH1_LOGS+=("$log")
    BATCH1_LABELS+=("${SOUP_LABELS[$i]}")
done

echo ""
echo "  Waiting for batch-1 model loading to finish ..."
for i in "${!BATCH1_LOGS[@]}"; do
    wait_for_inference "${BATCH1_LOGS[$i]}" "${BATCH1_LABELS[$i]}"
done

# Batch 2: indices 3..5
BATCH2_PIDS=()

echo ""
echo "  Launching batch-2 ..."
for i in 3 4 5; do
    w10=${W10_VALS[$i]}
    w11=${W11_VALS[$i]}
    w10i=$(printf "%02d" "$(echo "$w10 * 10 / 1" | bc)")
    w11i=$(printf "%02d" "$(echo "$w11 * 10 / 1" | bc)")

    gpu=${GPU_MAP[$i]}
    log="${LOG_DIR}/soup_t10w${w10i}_t11w${w11i}__ae_ae03_1300.log"
    video="${VIDEO_BASE}/soup_t10w${w10i}_t11w${w11i}"

    echo "  [BATCH-2] GPU:$gpu  ${SOUP_LABELS[$i]}  ->  $log"
    pid=$(run_eval "${SOUP_DIRS[$i]}" "$gpu" "$log" "$video")
    BATCH2_PIDS+=("$pid")
done

# ── Step 3: Wait for all ─────────────────────────────────────────────────────
echo ""
echo "  All 6 evaluations launched. Waiting for completion ..."
ALL_PIDS=("${BATCH1_PIDS[@]}" "${BATCH2_PIDS[@]}")

for idx in "${!ALL_PIDS[@]}"; do
    pid=${ALL_PIDS[$idx]}
    wait "$pid" 2>/dev/null && rc=$? || rc=$?
    echo "    PID $pid (${SOUP_LABELS[$idx]}) finished  exit=$rc"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All evaluations complete. Log files:"
echo "============================================================"
for i in "${!W10_VALS[@]}"; do
    w10=${W10_VALS[$i]}
    w11=${W11_VALS[$i]}
    w10i=$(printf "%02d" "$(echo "$w10 * 10 / 1" | bc)")
    w11i=$(printf "%02d" "$(echo "$w11 * 10 / 1" | bc)")
    log="${LOG_DIR}/soup_t10w${w10i}_t11w${w11i}__ae_ae03_1300.log"
    echo "  t10:${w10} + t11:${w11}  ->  $log"
done
