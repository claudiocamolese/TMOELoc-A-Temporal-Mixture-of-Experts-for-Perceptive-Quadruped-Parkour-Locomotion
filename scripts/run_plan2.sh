#!/usr/bin/env bash
# Pick the best mixture we have at 15000 iterations, then RL both arms.
#
#   setsid nohup ./scripts/run_plan2.sh > /dev/null 2>&1 < /dev/null &
#
# Replaces run_plan.sh from its stage 3 on. The dense distillation it started is
# left running -- it has its own session -- and this waits for it.
#
#   1  wait for the dense run to reach 15000
#   2  score dense@15k against the original top-2 mixture at 14999, same
#      protocol, same rollout length, both in one sitting
#   3  RL fine-tune whichever wins
#   4  RL fine-tune the GRU baseline on the same budget. The control: without it
#      "you gave one arm an extra training stage" answers the whole comparison
#
# Selection is on courses completed per environment-minute. Pace alone rewards a
# policy that sprints and falls, hazard alone rewards one that barely moves; a
# completed course requires both, so it is the one number that cannot be gamed by
# trading one against the other. Pace and hazard are reported alongside it.
#
# Fine-tuning always runs with soft routing (--top_k 4). On the top-2 weights that
# is measured at +2.77 +- 0.67 goals per env-minute for a fall cost inside the
# noise floor, and it costs nothing here because PPO retrains the actor anyway.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

LOG=logs/plan2.log
ENVS=128
STEPS=2000
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
free_mb(){ echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
                  - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }

log "=== waiting for the dense run to reach 15000 ==="
while true; do
  it=$(grep -c "Learning iteration" logs/run_console/moe_dense.log 2>/dev/null || echo 0)
  [ "$it" -ge 15000 ] && { log "dense reached $it"; break; }
  pgrep -f "python.*train\.py.*StudentMoEDense" >/dev/null 2>&1 || {
    log "dense run gone at $it; selecting from what exists"; break; }
  sleep 300
done
sleep 120
while [ "$(free_mb)" -lt 12000 ]; do sleep 60; done

DENSE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_moe_dense/*/model_*.pt 2>/dev/null | head -1)
TOP2=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe/*/model_14999.pt 2>/dev/null | head -1)
OUT=logs/select_15k; mkdir -p "$OUT"
log "dense: ${DENSE:-none}"
log "top-2: ${TOP2:-none}"

# ---- score both, same protocol, each under the config it was trained with ----
for pair in "dense15k:$DENSE:Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-Eval-v0" \
            "top2_15k:$TOP2:Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0"; do
  tag=${pair%%:*}; rest=${pair#*:}; ck=${rest%%:*}; task=${rest#*:}
  [ -f "$ck" ] || { log "skip $tag: no checkpoint"; continue; }
  log "eval $tag"
  nice -n 10 timeout -k 60 5400 "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py \
    --task "$task" --checkpoint "$ck" --num_envs "$ENVS" --steps "$STEPS" \
    --tag "$tag" --out_dir "$OUT" --headless >> "$LOG" 2>&1 \
    && log "  ok" || log "  FAILED"
done

WINNER=$("$PYTHON_BIN" - "$OUT" <<'PY' 2>>"$LOG"
import os, sys
import numpy as np
sys.path.insert(0, "scripts/analysis")
from episode_metrics import episode_table, summarise
o = sys.argv[1]
res = {}
for tag in ("dense15k", "top2_15k"):
    p = os.path.join(o, f"trace_{tag}.npz")
    if not os.path.exists(p):
        continue
    d = np.load(p, allow_pickle=True)
    rows = episode_table(d["dones"].astype(bool), d["goal_idx"].astype(int))
    res[tag] = summarise(rows, tag, boot=1000)
for k, v in res.items():
    print(f"{k:9s} courses/min {v['success_per_min']:5.2f}  pace {v['goals_per_min']:6.2f}"
          f"+-{v['goals_per_min_se']:.2f}  hazard {v['fall_hazard_per_1k_steps']:.3f}"
          f"+-{v['fall_hazard_se']:.3f}", file=sys.stderr)
if not res:
    print("NONE")
else:
    print(max(res, key=lambda k: res[k]["success_per_min"]))
PY
)
log "WINNER: $WINNER"

case "$WINNER" in
  dense15k) FT_CKPT="$DENSE"; FT_TASK=Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2-FT-v0 ;;
  top2_15k) FT_CKPT="$TOP2";  FT_TASK=Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-FT-v0 ;;
  *) log "FATAL: no candidate could be scored; stopping rather than guessing"; exit 1 ;;
esac
log "fine-tuning $FT_CKPT under $FT_TASK"

# ---- RL on ours -------------------------------------------------------------
log "STAGE: PPO fine-tuning of the selected mixture"
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task "$FT_TASK" --checkpoint "$FT_CKPT" --top_k 4 \
  --num_envs 192 --max_iterations 5000 --run_name moe_ft --headless \
  >> logs/run_console/ft_moe.log 2>&1
log "  exit $?"

# ---- RL on the control ------------------------------------------------------
BASE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_baseline/*/model_14999.pt 2>/dev/null | head -1)
if [ -f "$BASE" ]; then
  log "STAGE: PPO fine-tuning of the GRU baseline (control)"
  setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-FT-v0 --checkpoint "$BASE" \
    --num_envs 192 --max_iterations 5000 --run_name base_ft --headless \
    >> logs/run_console/ft_base.log 2>&1
  log "  exit $?"
else
  log "ALERT: no baseline checkpoint; the control cannot run and the comparison is incomplete"
fi
log "=== done ==="
