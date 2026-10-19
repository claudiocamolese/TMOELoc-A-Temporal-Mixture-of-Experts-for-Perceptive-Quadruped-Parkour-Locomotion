#!/usr/bin/env bash
# Keep the training chain alive across an unattended multi-day run.
#
#   setsid nohup ./scripts/watchdog_train.sh > /dev/null 2>&1 < /dev/null &
#
# run_experiments.sh already chains MoE -> baseline. While that script is alive
# it owns the chain and this one only takes a heartbeat. It intervenes when the
# chain script itself is gone but the work is not finished, which is what a
# crash looks like from the outside.
#
# THE RESTART DETAIL THAT MATTERS: train.py resolves the checkpoint as
#
#     if args_cli.teacher_checkpoint is not None:  resume_path = teacher
#     else:                                        resume_path = get_checkpoint_path(...)
#
# so --teacher_checkpoint *overrides* --resume. Passing both on a restart would
# silently reload the phase-1 teacher into a fresh log directory and start from
# iteration 0 while looking perfectly healthy. A resume therefore passes
# --resume/--load_run/--checkpoint and NO --teacher_checkpoint; only a run with
# no checkpoint of its own is (correctly) started from the teacher.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/launch_env.sh
mkdir -p logs/run_console

LOG=logs/watchdog.log
STATE=logs/watchdog_state
TARGET_ITERS=15000
NUM_ENVS=192
SEED=1
POLL=120
MAX_RESTARTS=6          # a crashloop must not burn two days of GPU
HEARTBEAT=1800          # seconds between "still fine" lines

MOE_TASK=Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0
BASE_TASK=Isaac-Extreme-Parkour-Student-Unitree-Go2-v0
MOE_EXP=unitree_go2_parkour_moe
BASE_EXP=unitree_go2_parkour_baseline

mkdir -p logs "$STATE"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Matching the bare task name would also match any shell whose command line
# happens to mention it -- an interactive grep is enough to fake a running
# trainer and make the watchdog sit still while nothing trains. Anchoring on the
# interpreter and the script leaves only real training processes.
running()      { pgrep -f -- "python.*train\.py.*$1" >/dev/null 2>&1; }
running_chain(){ pgrep -f -- "bash.*run_experiments\.sh" >/dev/null 2>&1; }
iters_done()   { count_training_iterations "$1"; }
latest_ckpt()  { select_checkpoint "logs/rsl_rl/$1/*/model_*.pt"; }

restart_count() { cat "$STATE/restarts_$1" 2>/dev/null || echo 0; }
bump_restart()  { echo $(( $(restart_count "$1") + 1 )) > "$STATE/restarts_$1"; }

# $1 exp name  $2 task  $3 console log
relaunch() {
  local exp=$1 task=$2 console=$3
  local n; n=$(restart_count "$exp")
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    log "GIVING UP on $exp: already restarted $n times. Manual attention needed."
    return 1
  fi
  bump_restart "$exp"

  local ckpt run_dir args
  ckpt=$(latest_ckpt "$exp")
  args=(--task "$task" --seed "$SEED" --headless --num_envs "$NUM_ENVS"
        --max_iterations "$TARGET_ITERS")

  if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
    run_dir=$(basename "$(dirname "$ckpt")")
    # deliberately no --teacher_checkpoint here: it would win over --resume
    args+=(--resume --load_run "$run_dir" --checkpoint "$(basename "$ckpt")")
    log "restarting $exp (attempt $((n+1))) RESUMING from $run_dir/$(basename "$ckpt")"
  else
    args+=(--teacher_checkpoint "$TEACHER_CKPT")
    log "restarting $exp (attempt $((n+1))) FROM THE TEACHER - no checkpoint of its own yet"
  fi

  setsid nohup "$PYTHON_BIN" scripts/rsl_rl/train.py "${args[@]}" \
    >> "$console" 2>&1 < /dev/null &
  log "  launched pid $!, console $console"
  sleep 180   # let it get past Isaac Sim startup before the next health check
}

log "=== watchdog started (pid $$), target $TARGET_ITERS iters, poll ${POLL}s ==="
last_beat=0

while true; do
  now=$(date +%s)
  moe_it=$(iters_done logs/run_console/moe.log)
  base_it=$(iters_done logs/run_console/baseline.log)

  if running_chain; then
    # the chain script is in charge; only report
    if [ $(( now - last_beat )) -ge "$HEARTBEAT" ]; then
      log "chain alive - moe $moe_it/$TARGET_ITERS, baseline $base_it/$TARGET_ITERS"
      last_beat=$now
    fi
    sleep "$POLL"; continue
  fi

  # chain script gone: finished, or something died
  if [ "$base_it" -ge "$TARGET_ITERS" ]; then
    log "baseline reached $base_it iterations. Nothing left to supervise; exiting."
    exit 0
  fi

  if running "$MOE_TASK"; then
    if [ $(( now - last_beat )) -ge "$HEARTBEAT" ]; then
      log "moe running standalone - $moe_it/$TARGET_ITERS"; last_beat=$now
    fi
    sleep "$POLL"; continue
  fi

  if running "$BASE_TASK"; then
    if [ $(( now - last_beat )) -ge "$HEARTBEAT" ]; then
      log "baseline running standalone - $base_it/$TARGET_ITERS"; last_beat=$now
    fi
    sleep "$POLL"; continue
  fi

  # nothing is running and the baseline is not done
  if [ "$moe_it" -lt "$TARGET_ITERS" ]; then
    log "NOTHING RUNNING and moe is only at $moe_it - restarting moe"
    relaunch "$MOE_EXP" "$MOE_TASK" logs/run_console/moe.log || exit 1
  else
    log "NOTHING RUNNING, moe done at $moe_it, baseline at $base_it - restarting baseline"
    relaunch "$BASE_EXP" "$BASE_TASK" logs/run_console/baseline.log || exit 1
  fi
  last_beat=$(date +%s)
done
