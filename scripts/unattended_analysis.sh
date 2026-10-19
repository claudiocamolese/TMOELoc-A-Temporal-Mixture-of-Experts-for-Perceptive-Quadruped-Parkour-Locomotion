#!/usr/bin/env bash
# Run the full test battery on the final MoE checkpoint, unattended.
#
#   setsid nohup ./scripts/unattended_analysis.sh > /dev/null 2>&1 < /dev/null &
#
# It waits for the MoE run to reach its last checkpoint, lets the baseline get
# going, and only then starts, because the baseline is the priority: it has a
# 24 h run to finish and this has all the time in the world.
#
# Every GPU stage is gated on free VRAM. Two 192-environment simulators fit in
# 32.6 GB only just (31.9 GB measured), and the process that loses an allocation
# race is not necessarily this one -- it could be the training run. So each
# stage waits until there is real headroom, and the rollouts run at fewer
# environments than the training does.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/launch_env.sh
mkdir -p logs/run_console

LOG=logs/analysis_unattended.log
TARGET=15000
ENVS=128                # below the training run's 192, to leave headroom
STEPS=2000              # the report's own conclusion: 1200 cannot resolve a 30% change
NEED_MB=7000            # free VRAM required before starting a GPU stage
MAX_WAIT_GPU=$((6*3600))

mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

free_mb() {
  local total used
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
  echo $(( total - used ))
}

wait_for_gpu() {
  local waited=0
  while [ "$(free_mb)" -lt "$NEED_MB" ]; do
    if [ "$waited" -ge "$MAX_WAIT_GPU" ]; then
      log "gave up waiting for $NEED_MB MiB (only $(free_mb) free after $((waited/3600))h)"
      return 1
    fi
    sleep 300; waited=$((waited+300))
  done
  return 0
}

# $1 label, rest: command
stage() {
  local label=$1; shift
  if ! wait_for_gpu; then log "SKIPPED $label - no VRAM"; return 1; fi
  log "START $label  (free $(free_mb) MiB)"
  if nice -n 19 timeout -k 60 7200 "$@" >> "$LOG" 2>&1; then
    log "OK    $label"
  else
    log "FAIL  $label (exit $?)"
  fi
}

log "=== unattended analysis started (pid $$) ==="

# ---- 1. wait for the MoE run to finish -------------------------------------
while true; do
  it=$(tr '\r' '\n' < logs/run_console/moe.log 2>/dev/null | grep -c "Learning iteration")
  [ "$it" -ge "$TARGET" ] && { log "moe reached $it iterations"; break; }
  pgrep -f "StudentMoE-Unitree-Go2-v0" >/dev/null 2>&1 || {
    log "moe process gone at iteration $it (short of $TARGET) - analysing what exists"; break; }
  sleep 120
done

# let the baseline claim its memory first, and never analyse a half-written file
sleep 900

CKPT=$(select_checkpoint "logs/rsl_rl/unitree_go2_parkour_moe/*/model_*.pt" newest)
[ -f "$CKPT" ] || { log "FATAL: no MoE checkpoint found"; exit 1; }
RUN_DIR=$(dirname "$CKPT")
OUT="$RUN_DIR/analysis_final"
mkdir -p "$OUT"
log "checkpoint: $CKPT"
log "output:     $OUT"

EVAL_TASK=Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0
TRAIN_TASK=Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0
EV=("$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py --num_envs "$ENVS" --headless --out_dir "$OUT")

# ---- 2. weight-space inspection: no simulator, so it can always run --------
log "START weights"
if nice -n 19 "$PYTHON_BIN" scripts/analysis/inspect_moe_weights.py "$CKPT" >> "$LOG" 2>&1; then
  log "OK    weights"
else
  log "FAIL  weights"
fi

# ---- 3. main rollout: per-terrain metrics, routing, attention --------------
stage "eval main" "${EV[@]}" --task "$EVAL_TASK" --checkpoint "$CKPT" \
  --steps "$STEPS" --context_every 40 --attention --tag main

# ---- 4. dynamics probe: needs the TRAINING config, EVAL freezes mass/COM ---
stage "eval probe" "${EV[@]}" --task "$TRAIN_TASK" --checkpoint "$CKPT" \
  --steps 900 --context_every 3 --tag probe

# ---- 5. teacher on the identical config: the distillation ceiling ----------
stage "eval teacher" "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py \
  --task "$EVAL_TASK" --checkpoint "$TEACHER_CKPT" --policy teacher \
  --num_envs "$ENVS" --steps "$STEPS" --tag teacher --out_dir "$OUT" --headless

# ---- 6. the eleven-condition ablation --------------------------------------
stage "intervention sweep" "$PYTHON_BIN" scripts/rsl_rl/intervention_sweep.py \
  --checkpoint "$CKPT" --num_envs "$ENVS" --steps 1500 --headless \
  --out_dir "$RUN_DIR/sweep_final"

# ---- 7. offline analyses over whatever the rollouts produced --------------
log "--- offline stage ---"
run_offline() {
  local label=$1; shift
  if nice -n 19 "$@" >> "$LOG" 2>&1; then log "OK    $label"; else log "FAIL  $label"; fi
}
[ -f "$OUT/trace_main.npz" ] && {
  run_offline "routing"   "$PYTHON_BIN" scripts/analysis/analyze_student.py "$OUT/trace_main.npz" --out "$OUT/routing.json"
  run_offline "attention" "$PYTHON_BIN" scripts/analysis/attention_probe.py "$OUT/trace_main.npz" --out "$OUT/attention.json"
  run_offline "attn plot" "$PYTHON_BIN" scripts/analysis/plot_attention.py "$OUT/trace_main.npz"
}
[ -f "$OUT/trace_probe.npz" ] && {
  run_offline "dyn probe"  "$PYTHON_BIN" scripts/analysis/fit_context_probe.py "$OUT/trace_probe.npz" --out "$OUT/probe_directions.npz"
  run_offline "fall probe" "$PYTHON_BIN" scripts/analysis/fall_predictability.py "$OUT/trace_probe.npz"
}

# ---- 8. causal test: remove the decodable dynamics directions -------------
[ -f "$OUT/probe_directions.npz" ] && stage "eval ablated" "${EV[@]}" \
  --task "$EVAL_TASK" --checkpoint "$CKPT" --steps "$STEPS" --context_every 200 \
  --ablate_context "$OUT/probe_directions.npz" --tag ablated

# ---- 9. the honest headline numbers, matched horizon ----------------------
# Arrays preserve filenames containing spaces; nullglob removes unmatched globs.
shopt -s nullglob
TRACES=("$OUT"/trace_*.npz "$RUN_DIR/sweep_final"/trace_*.npz)
shopt -u nullglob
if (( ${#TRACES[@]} )); then
  run_offline "episode metrics" "$PYTHON_BIN" scripts/analysis/episode_metrics.py \
    --horizon "$STEPS" --boot 2000 --per-terrain --out "$OUT/episode_metrics.json" "${TRACES[@]}"
fi

log "=== analysis finished ==="
log "results under $OUT and $RUN_DIR/sweep_final"
