#!/usr/bin/env bash
# Run LIBERO eval on all 4 task suites in parallel, one GPU per suite.
#
# Usage:
#   bash scripts/eval_libero_all.sh                          # defaults
#   bash scripts/eval_libero_all.sh --config pi05_libero     # JAX official
#   bash scripts/eval_libero_all.sh --trials 50              # 50 trials/task
#   bash scripts/eval_libero_all.sh --config pi05_libero_lora_pytorch \
#       --ckpt checkpoints/pi05_libero_lora_pytorch/my_exp/30000
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
CONFIG="pi05_libero"
CKPT=""
TRIALS=3
SEED=7
GPUS="0,1,2,3"

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)  CONFIG="$2"; shift 2 ;;
    --ckpt)    CKPT="$2";   shift 2 ;;
    --trials)  TRIALS="$2"; shift 2 ;;
    --seed)    SEED="$2";   shift 2 ;;
    --gpus)    GPUS="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -lt 4 ]]; then
  echo "ERROR: Need 4 GPUs, got ${#GPU_LIST[@]} ($GPUS)"
  exit 1
fi

SUITES=(libero_spatial libero_object libero_goal libero_10)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$SCRIPT_DIR"
mkdir -p logs

echo "============================================================"
echo "  LIBERO Full Benchmark Eval"
echo "  Config     : $CONFIG"
echo "  Checkpoint : ${CKPT:-auto}"
echo "  Trials/task: $TRIALS"
echo "  Seed       : $SEED"
echo "  GPUs       : $GPUS"
echo "  Timestamp  : $TIMESTAMP"
echo "============================================================"

CKPT_ARG=""
if [[ -n "$CKPT" ]]; then
  CKPT_ARG="--checkpoint-dir $CKPT"
fi

PIDS=()
for i in "${!SUITES[@]}"; do
  SUITE="${SUITES[$i]}"
  GPU="${GPU_LIST[$i]}"
  LOG="logs/eval_${SUITE}_${TIMESTAMP}.log"
  VIDEO_DIR="data/libero/eval_${SUITE}_${TIMESTAMP}"

  echo "  [GPU $GPU]  $SUITE  ->  $LOG"

  CUDA_VISIBLE_DEVICES="$GPU" \
  MUJOCO_GL=osmesa \
  LIBERO_CONFIG_PATH=/tmp/libero \
  PYTHONPATH="$SCRIPT_DIR/third_party/libero:$SCRIPT_DIR/src:${PYTHONPATH:-}" \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
  "$SCRIPT_DIR/.venv/bin/python" examples/libero/eval_libero_standalone.py \
    --config-name "$CONFIG" \
    $CKPT_ARG \
    --task-suite-name "$SUITE" \
    --num-trials-per-task "$TRIALS" \
    --seed "$SEED" \
    --video-out-path "$VIDEO_DIR" \
    > "$LOG" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "All 4 suites launched.  PIDs: ${PIDS[*]}"
echo "Monitor:  tail -f logs/eval_*_${TIMESTAMP}.log"
echo ""

# ── Wait for all to finish ──────────────────────────────────────────────
FAILED=0
for i in "${!SUITES[@]}"; do
  SUITE="${SUITES[$i]}"
  PID="${PIDS[$i]}"
  if wait "$PID"; then
    echo "[DONE]  $SUITE  (pid $PID)  OK"
  else
    echo "[FAIL]  $SUITE  (pid $PID)  exit=$?"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
  echo "  All 4 suites completed successfully."
else
  echo "  $FAILED suite(s) failed. Check logs."
fi
echo "============================================================"

# ── Print summary table ────────────────────────────────────────────────
echo ""
echo "Suite              | Success Rate"
echo "-------------------+-------------"
for SUITE in "${SUITES[@]}"; do
  JSON="data/libero/eval_${SUITE}_${TIMESTAMP}/eval_summary.json"
  if [[ -f "$JSON" ]]; then
    RATE=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['total_successes']}/{d['total_episodes']}  ({d['overall_success_rate']}%)\")")
    printf "%-18s | %s\n" "$SUITE" "$RATE"
  else
    printf "%-18s | (no result)\n" "$SUITE"
  fi
done
echo ""
