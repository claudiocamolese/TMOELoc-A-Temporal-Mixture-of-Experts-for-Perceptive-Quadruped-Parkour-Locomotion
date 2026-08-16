#!/usr/bin/env bash
# Both mixtures fine-tuned in parallel, then the control once the GPU frees up.
#
# A fine-tuning costs about 11 GB -- far less than the 19 GB of a distillation,
# because the encoder is frozen and a rollout is 24 steps rather than 120 -- so
# two fit in 32.6 GB with room. Three do not: 33 GB would not fit, which is why
# the control waits rather than being chained blindly behind the first arm.
#
#   arm A  dense mixture      (already running when this starts)
#   arm B  top-2 mixture      launched here, same treatment
#   arm C  GRU baseline       the control, after A and B have freed the GPU
#
# Arm B keeps its native top-2 routing. Converting it to soft routing would have
# isolated the distillation regime as the single difference between A and B, but
# it would also have left no top-2 arm anywhere -- and top-2 is the architecture
# this project set out to build. That comparison is in any case largely settled:
# at 15000 iterations the two distillations were within 1.5 sigma of each other
# (2.54 against 2.46 courses per env-minute), so spending a twelve-hour arm to
# sharpen it rather than to measure top-2 end to end is the worse trade.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh
LOG=logs/rl2.log
ITERS=20000
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
free_mb(){ echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
                  - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }

TOP2=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe/*/model_14999.pt 2>/dev/null | head -1)
BASE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_baseline/*/model_14999.pt 2>/dev/null | head -1)
[ -f "$TOP2" ] || { log "FATAL: no top-2 checkpoint"; exit 1; }

log "=== arm B: top-2 mixture, native top-2 routing, $ITERS iterations ==="
log "checkpoint: $TOP2  (free $(free_mb) MiB)"
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-FT-v0 \
  --checkpoint "$TOP2" --num_envs 192 --max_iterations "$ITERS" \
  --run_name top2_ft --headless >> logs/run_console/ft_top2.log 2>&1
log "arm B exit $?"

log "waiting for the GPU before the control"
while pgrep -f "python.*finetune_rl\.py" >/dev/null 2>&1; do sleep 120; done
while [ "$(free_mb)" -lt 15000 ]; do sleep 60; done

if [ -f "$BASE" ]; then
  log "=== arm C: GRU baseline (control), identical budget ==="
  setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-FT-v0 \
    --checkpoint "$BASE" --num_envs 192 --max_iterations "$ITERS" \
    --run_name base_ft --headless >> logs/run_console/ft_base.log 2>&1
  log "arm C exit $?"
else
  log "ALERT: no baseline checkpoint; the control cannot run"
fi
log "=== done ==="
