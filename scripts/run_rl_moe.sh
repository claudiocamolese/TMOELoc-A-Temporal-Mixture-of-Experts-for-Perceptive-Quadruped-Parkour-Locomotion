#!/usr/bin/env bash
# Arm A on its own, so the two arms can be launched independently.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/launch_env.sh
mkdir -p logs/run_console
MOE=$(select_checkpoint "logs/rsl_rl/unitree_go2_parkour_moe_dense/*/model_14999.pt" newest)
echo "[$(date '+%F %T')] arm A: dense mixture, epochs 2, noise 0.25" >> logs/rl2.log
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-FT-v0 \
  --checkpoint "$MOE" --top_k 4 --num_envs 192 --max_iterations 20000 \
  --epochs 2 --init_noise_std 0.25 \
  --run_name moe_ft --headless >> logs/run_console/ft_moe.log 2>&1
echo "[$(date '+%F %T')] arm A exit $?" >> logs/rl2.log
