#!/usr/bin/env bash
# The remaining plan, chained: dense distillation, then RL on both arms.
#
#   setsid nohup ./scripts/run_pipeline.sh > /dev/null 2>&1 < /dev/null &
#
# Order matters and is not arbitrary:
#
#   1  dense distillation from scratch, 15000 iters  (~26 h)
#        Soft routing is worth +2.77 +- 0.67 goals/env-min measured at inference
#        on top-2 weights, and costs no compute here since every expert is
#        evaluated anyway. Trained from the start rather than switched at the end,
#        because weights shaped under top-2 carry its imprint.
#   2  RL fine-tune the dense student, 5000 iters    (~9 h)
#   3  RL fine-tune the GRU baseline, 5000 iters     (~9 h)
#        The control. Without it, "our arm got an extra training stage" answers
#        the whole comparison, and the result is worth nothing.
#
# Waits for whatever is training now to finish before starting.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh
LOG=logs/pipeline.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== pipeline armed; waiting for the current run to finish ==="
while pgrep -f "python.*train\.py" >/dev/null 2>&1; do sleep 300; done
log "GPU free"
sleep 60

# ---- 1. dense distillation ------------------------------------------------
log "STAGE 1: dense distillation (15000 iters)"
"$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-v0 \
  --seed 1 --headless --num_envs 192 --max_iterations 15000 \
  --teacher_checkpoint "$TEACHER_CKPT" >> logs/run_console/moe_dense.log 2>&1
log "STAGE 1 done (exit $?)"

DENSE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_moe_dense/*/model_*.pt 2>/dev/null | head -1)
[ -f "$DENSE" ] || { log "FATAL: no dense checkpoint; stopping"; exit 1; }
log "dense checkpoint: $DENSE"

# ---- 2. RL on the dense mixture -------------------------------------------
log "STAGE 2: PPO fine-tuning of the dense mixture"
"$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-FT-v0 \
  --checkpoint "$DENSE" --num_envs 192 --max_iterations 5000 \
  --run_name dense_ft --headless >> logs/run_console/ft_dense.log 2>&1
log "STAGE 2 done (exit $?)"

# ---- 3. the control: same treatment for the baseline ----------------------
BASE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_baseline/*/model_14999.pt 2>/dev/null | head -1)
if [ -f "$BASE" ]; then
  log "STAGE 3: PPO fine-tuning of the GRU baseline (control)"
  "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-FT-v0 \
    --checkpoint "$BASE" --num_envs 192 --max_iterations 5000 \
    --run_name base_ft --headless >> logs/run_console/ft_base.log 2>&1
  log "STAGE 3 done (exit $?)"
else
  log "ALERT: no baseline checkpoint, so the control cannot run"
fi
log "=== pipeline finished ==="
