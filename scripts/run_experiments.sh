#!/usr/bin/env bash
# Sequentially train the two distillation students that form the main comparison.
#
#   C        : unified temporal Transformer context + top-2 Mixture-of-Experts actor
#   Original : GRU depth backbone + Conv1D state history encoder + dense actor
#
# Both start from the same frozen phase-1 teacher, see the same environment and
# get the same optimisation budget. Running them concurrently was measured to be
# a large net loss on this machine (peak VRAM 31.9/32.6 GB and ~6x slower per
# iteration), so they are chained instead.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

ITERS="${ITERS:-15000}"
NUM_ENVS="${NUM_ENVS:-192}"
SEED="${SEED:-1}"
OUT=logs/run_console
mkdir -p "$OUT"

run () {
  local tag=$1 task=$2
  echo "[$(date '+%F %T')] START $tag  (task=$task iters=$ITERS envs=$NUM_ENVS seed=$SEED)"
  $PYTHON_BIN scripts/rsl_rl/train.py \
    --task "$task" \
    --seed "$SEED" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS" \
    --teacher_checkpoint "$TEACHER_CKPT" \
    > "$OUT/$tag.log" 2>&1
  echo "[$(date '+%F %T')] END   $tag  exit=$?"
}

run moe      Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0
run baseline Isaac-Extreme-Parkour-Student-Unitree-Go2-v0

echo "[$(date '+%F %T')] ALL RUNS FINISHED"
