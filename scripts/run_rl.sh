#!/usr/bin/env bash
# The two RL fine-tunings, at a budget proportionate to what the stage costs.
#
#   setsid nohup ./scripts/run_rl.sh > /dev/null 2>&1 < /dev/null &
#
# 20000 iterations rather than 5000. The stage is far cheaper than it looked:
# steps_per_env is 24 against distillation's 120, and the encoder is frozen so no
# CNN or Transformer backward runs, which together put it at roughly 45 iterations
# a minute. At 5000 it collected 23 M environment steps against distillation's
# 346 M -- fifteen times less -- and the adaptive learning rate sat at its floor
# throughout, so the policy was still moving when it would have stopped.
#
#   20000 x 24 x 192 = 92 M steps, about seven hours per arm.
#
# Both arms get the identical budget. The control is the whole point: without it
# "you gave one arm an extra training stage" answers the comparison by itself.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh
LOG=logs/rl.log
ITERS=20000
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

MOE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_moe_dense/*/model_14999.pt 2>/dev/null | head -1)
BASE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_baseline/*/model_14999.pt 2>/dev/null | head -1)
[ -f "$MOE" ] || { log "FATAL: no dense checkpoint"; exit 1; }
[ -f "$BASE" ] || { log "FATAL: no baseline checkpoint"; exit 1; }
log "=== RL stage, $ITERS iterations per arm ==="
log "moe : $MOE"
log "base: $BASE"

log "ARM 1: dense mixture, soft routing"
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-FT-v0 \
  --checkpoint "$MOE" --top_k 4 --num_envs 192 --max_iterations "$ITERS" \
  --run_name moe_ft --headless >> logs/run_console/ft_moe.log 2>&1
log "  exit $?"

log "ARM 2: GRU baseline (control), identical budget"
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-FT-v0 \
  --checkpoint "$BASE" --num_envs 192 --max_iterations "$ITERS" \
  --run_name base_ft --headless >> logs/run_console/ft_base.log 2>&1
log "  exit $?"
log "=== done ==="
