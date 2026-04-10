#!/usr/bin/env bash
# Run LIBERO eval on all 4 task suites in parallel, one GPU per suite, each in its own tmux session.
#
# Usage:
#   bash scripts/eval_libero_all.sh                          # defaults
#   bash scripts/eval_libero_all.sh --config pi05_libero     # JAX official
#   bash scripts/eval_libero_all.sh --trials 50              # 50 trials/task
#   bash scripts/eval_libero_all.sh --config pi05_libero_lora_pytorch_4_gpu \
#       --ckpt checkpoints/pi05_libero_lora_pytorch_4_gpu/pi05_libero_lora_pytorch_4_gpu/8000
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

CKPT_ARG=""
if [[ -n "$CKPT" ]]; then
  CKPT_ARG="--checkpoint-dir $CKPT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              LIBERO Full Benchmark Eval                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config     : $CONFIG"
echo "║  Checkpoint : ${CKPT:-auto}"
echo "║  Trials/task: $TRIALS"
echo "║  Seed       : $SEED"
echo "║  GPUs       : $GPUS"
echo "║  Timestamp  : $TIMESTAMP"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Launch each suite in its own tmux session ──────────────────────────
for i in "${!SUITES[@]}"; do
  SUITE="${SUITES[$i]}"
  GPU="${GPU_LIST[$i]}"
  LOG="logs/eval_${CONFIG}_${SUITE}_${TIMESTAMP}.log"
  VIDEO_DIR="data/libero/eval_${CONFIG}_${SUITE}_${TIMESTAMP}"
  SESSION="eval_${SUITE}"

  # Kill existing session with same name if present
  tmux kill-session -t "$SESSION" 2>/dev/null || true

  tmux new-session -d -s "$SESSION" \
    "cd $SCRIPT_DIR && \
     CUDA_VISIBLE_DEVICES=$GPU \
     MUJOCO_GL=osmesa \
     LIBERO_CONFIG_PATH=/tmp/libero \
     PYTHONPATH=$SCRIPT_DIR/third_party/libero:$SCRIPT_DIR/src:\${PYTHONPATH:-} \
     XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
     $SCRIPT_DIR/.venv/bin/python examples/libero/eval_libero_standalone.py \
       --config-name $CONFIG \
       $CKPT_ARG \
       --task-suite-name $SUITE \
       --num-trials-per-task $TRIALS \
       --seed $SEED \
       --video-out-path $VIDEO_DIR \
       2>&1 | tee $LOG; \
     echo ''; echo '══════ $SUITE FINISHED ══════'; exec bash"

  echo "  [GPU $GPU]  tmux: $SESSION  ->  $LOG"
done

echo ""
echo "All 4 suites launched in tmux sessions."
echo ""
echo "Monitor:"
echo "  tmux attach -t eval_libero_spatial"
echo "  tmux attach -t eval_libero_object"
echo "  tmux attach -t eval_libero_goal"
echo "  tmux attach -t eval_libero_10"
echo ""
echo "Or tail all logs:"
echo "  tail -f logs/eval_${CONFIG}_*_${TIMESTAMP}.log"
echo ""
echo "List sessions:  tmux ls"
echo ""

# ── Wait for all sessions to finish, then print summary ───────────────
echo "Waiting for all evaluations to complete..."
echo "(this script will poll every 60s; press Ctrl-C to detach -- evals keep running)"
echo ""

while true; do
  ALL_DONE=true
  for SUITE in "${SUITES[@]}"; do
    SESSION="eval_${SUITE}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      LOG="logs/eval_${CONFIG}_${SUITE}_${TIMESTAMP}.log"
      # Check if the eval finished (summary JSON is written at the end)
      JSON="data/libero/eval_${CONFIG}_${SUITE}_${TIMESTAMP}/eval_summary.json"
      if [[ -f "$JSON" ]]; then
        RATE=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['total_successes']}/{d['total_episodes']} ({d['overall_success_rate']}%)\")" 2>/dev/null || echo "parsing...")
        echo "  [DONE]  $SUITE : $RATE"
      else
        ALL_DONE=false
        LAST_LINE=$(tail -1 "$LOG" 2>/dev/null || echo "starting...")
        echo "  [RUN]   $SUITE : $LAST_LINE"
      fi
    else
      echo "  [EXIT]  $SUITE : session ended"
    fi
  done
  echo ""

  if $ALL_DONE; then
    break
  fi
  sleep 60
done

# ── Print final summary table ──────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              FINAL RESULTS SUMMARY                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-18s │ %s\n" "Suite" "Success Rate"
echo "║──────────────────────────────────────────────────────────────"
TOTAL_S=0
TOTAL_E=0
for SUITE in "${SUITES[@]}"; do
  JSON="data/libero/eval_${CONFIG}_${SUITE}_${TIMESTAMP}/eval_summary.json"
  if [[ -f "$JSON" ]]; then
    RATE=$(python3 -c "
import json
d=json.load(open('$JSON'))
print(f\"{d['total_successes']}/{d['total_episodes']}  ({d['overall_success_rate']}%)\")
" 2>/dev/null || echo "(error)")
    S=$(python3 -c "import json; print(json.load(open('$JSON'))['total_successes'])" 2>/dev/null || echo "0")
    E=$(python3 -c "import json; print(json.load(open('$JSON'))['total_episodes'])" 2>/dev/null || echo "0")
    TOTAL_S=$((TOTAL_S + S))
    TOTAL_E=$((TOTAL_E + E))
    printf "║  %-18s │ %s\n" "$SUITE" "$RATE"
  else
    printf "║  %-18s │ (no result)\n" "$SUITE"
  fi
done
echo "║──────────────────────────────────────────────────────────────"
if [[ $TOTAL_E -gt 0 ]]; then
  AVG=$(python3 -c "print(f'{$TOTAL_S/$TOTAL_E*100:.1f}')")
  printf "║  %-18s │ %s\n" "AVERAGE" "$TOTAL_S/$TOTAL_E  ($AVG%)"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
