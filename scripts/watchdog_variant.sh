#!/usr/bin/env bash
# Keep the failure-aware MoE run alive unattended.
#
# Same restart contract as watchdog_train.sh, including the trap that matters:
# train.py resolves --teacher_checkpoint before get_checkpoint_path, so passing
# it alongside --resume silently reloads the phase-1 teacher into a fresh
# directory and restarts from zero while looking healthy. A resume therefore
# passes only --resume/--load_run/--checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

LOG=${WLOG:-logs/watchdog_warm.log}
TASK=${TASK:-Isaac-Extreme-Parkour-StudentMoEWarm-Unitree-Go2-v0}
EXP=${EXP:-unitree_go2_parkour_moe_warm}
CONSOLE=${CONSOLE:-logs/run_console/moe_warm.log}
TARGET=15000; ENVS=192; SEED=1; POLL=120; MAX_RESTARTS=6; HEARTBEAT=1800
n_restart=0
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
running(){ pgrep -f -- "python.*train\.py.*${TASK}" >/dev/null 2>&1; }
iters(){ grep -c "Learning iteration" "$CONSOLE" 2>/dev/null || echo 0; }

log "=== watchdog started for $EXP (pid $$) ==="
last=0
while true; do
  it=$(iters)
  if [ "$it" -ge "$TARGET" ]; then log "reached $it iterations; exiting."; exit 0; fi
  if running; then
    now=$(date +%s)
    if [ $(( now - last )) -ge "$HEARTBEAT" ]; then log "alive - $it/$TARGET"; last=$now; fi
    sleep "$POLL"; continue
  fi
  if [ "$n_restart" -ge "$MAX_RESTARTS" ]; then
    log "GIVING UP after $n_restart restarts at iteration $it."; exit 1
  fi
  n_restart=$((n_restart+1))
  ckpt=$(ls -1t "logs/rsl_rl/$EXP"/*/model_*.pt 2>/dev/null | head -1)
  args=(--task "$TASK" --seed "$SEED" --headless --num_envs "$ENVS" --max_iterations "$TARGET")
  if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    args+=(--resume --load_run "$(basename "$(dirname "$ckpt")")" --checkpoint "$(basename "$ckpt")")
    log "restart $n_restart: RESUMING from $(basename "$ckpt")"
  else
    args+=(--teacher_checkpoint "$TEACHER_CKPT")
    log "restart $n_restart: FROM THE TEACHER (no checkpoint yet)"
  fi
  setsid nohup "$PYTHON_BIN" scripts/rsl_rl/train.py "${args[@]}" >> "$CONSOLE" 2>&1 < /dev/null &
  log "  launched pid $!"
  sleep 180
done
