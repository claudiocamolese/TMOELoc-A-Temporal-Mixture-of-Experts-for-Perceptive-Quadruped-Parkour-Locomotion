#!/usr/bin/env bash
# Wait for the GRU baseline to finish, then score both students the same way.
#
#   setsid nohup ./scripts/unattended_compare.sh > /dev/null 2>&1 < /dev/null &
#
# The comparison is the point of the whole project, so it is run once, on the
# fixed EVAL config, with identical rollout length and environment count for
# both arms. The teacher was already scored under the same protocol during the
# MoE battery and is reused.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/launch_env.sh
mkdir -p logs/run_console

LOG=logs/compare_unattended.log
ENVS=128
STEPS=2000
TARGET=15000
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== waiting for the baseline to reach $TARGET ==="
while true; do
  it=$(count_training_iterations logs/run_console/baseline.log)
  [ "$it" -ge "$TARGET" ] && { log "baseline reached $it"; break; }
  pgrep -f "python.*train\.py.*Parkour-Student-Unitree" >/dev/null 2>&1 || {
    log "baseline process gone at $it (short of $TARGET) - scoring what exists"; break; }
  sleep 180
done
sleep 120   # let the last checkpoint finish writing

BASE_CKPT=$(select_checkpoint "logs/rsl_rl/unitree_go2_parkour_baseline/*/model_*.pt" newest)
MOE_CKPT=$(select_checkpoint "logs/rsl_rl/unitree_go2_parkour_moe/*/model_*.pt" newest)
[ -f "$BASE_CKPT" ] || { log "FATAL: no baseline checkpoint"; exit 1; }
OUT=$(dirname "$BASE_CKPT")/analysis_final
mkdir -p "$OUT"
log "baseline: $BASE_CKPT"
log "moe:      $MOE_CKPT"

# The baseline is the GRU/dense student, so it needs its own gym id; the MoE
# eval task would build the wrong agent config.
BASE_TASK=Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0

log "START baseline eval"
if nice -n 19 timeout -k 60 7200 "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py \
  --task "$BASE_TASK" --checkpoint "$BASE_CKPT" --num_envs "$ENVS" --steps "$STEPS" \
  --tag baseline --out_dir "$OUT" --headless >> "$LOG" 2>&1; then
  log "OK baseline eval"
else
  log "FAIL baseline eval"
fi

# perception ablations still apply to the GRU student; the gate ones are skipped
# by the sweep itself
log "START baseline sweep"
if nice -n 19 timeout -k 60 7200 "$PYTHON_BIN" scripts/rsl_rl/intervention_sweep.py \
  --task "$BASE_TASK" --checkpoint "$BASE_CKPT" --num_envs "$ENVS" --steps 1500 \
  --conditions none,zero_context,no_depth --headless \
  --out_dir "$(dirname "$BASE_CKPT")/sweep_final" >> "$LOG" 2>&1; then
  log "OK baseline sweep"
else
  log "FAIL baseline sweep"
fi

MOE_DIR=$(dirname "$MOE_CKPT")/analysis_final
log "=== head-to-head, matched horizon ==="
if nice -n 19 "$PYTHON_BIN" scripts/analysis/episode_metrics.py --horizon "$STEPS" --boot 2000 \
  --per-terrain --out "$OUT/head_to_head.json" \
  "$OUT/trace_baseline.npz" "$MOE_DIR/trace_main.npz" "$MOE_DIR/trace_teacher.npz" >> "$LOG" 2>&1; then
  log "OK head-to-head"
else
  log "FAIL head-to-head"
fi

log "=== comparison finished ==="
