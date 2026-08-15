#!/usr/bin/env bash
# The remaining plan, start to finish, unattended.
#
#   setsid nohup ./scripts/run_plan.sh > /dev/null 2>&1 < /dev/null &
#
#   1  judge the warm-started run at iteration 4000 against the original MoE at
#      the same iteration. Its job is not to finish -- it is to answer whether
#      the warm start is worth carrying into the dense run. Stopping it there
#      saves about twenty hours.
#   2  retrain the mixture from scratch with dense routing, with or without the
#      warm start according to that verdict.
#   3  PPO fine-tune the dense mixture on the task reward.
#   4  PPO fine-tune the GRU baseline on the same budget. This is the control:
#      without it, "you gave one arm an extra training stage" answers the entire
#      comparison.
#
# Three things this script does deliberately, each because the first version of
# the 4k decision got them wrong:
#
#   - every checkpoint is evaluated under the gym id it was trained with. Loading
#     the risk checkpoint into the plain MoE config failed outright (risk_head an
#     unexpected key, dyn_head [3,64] against [5,64]) and produced an empty
#     verdict.
#   - the verdict must be an explicit word. Treating "anything that is not KEEP"
#     as licence to act killed a healthy run on no evidence.
#   - kills are -9. Isaac Sim ignores SIGTERM, and a plain pkill left two
#     192-environment simulators sharing 32.6 GB with 0.7 GB spare.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

LOG=logs/plan.log
ENVS=128
STEPS=1200
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
free_mb(){ echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
                  - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )); }

# ---------------------------------------------------------------- 1. verdict
log "=== waiting for the warm run to reach 4000 ==="
while true; do
  it=$(grep -c "Learning iteration" logs/run_console/moe_warm.log 2>/dev/null || echo 0)
  [ "$it" -ge 4000 ] && { log "warm run reached $it"; break; }
  pgrep -f "python.*train\.py.*StudentMoEWarm" >/dev/null 2>&1 || {
    log "warm run gone at $it; judging on what exists"; break; }
  sleep 180
done
sleep 60

WARM=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe_warm/*/model_4000.pt 2>/dev/null | head -1)
[ -f "$WARM" ] || WARM=$(ls -1t logs/rsl_rl/unitree_go2_parkour_moe_warm/*/model_*.pt 2>/dev/null | head -1)
MOE=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe/*/model_4000.pt 2>/dev/null | head -1)
OUT=logs/decide_warm; mkdir -p "$OUT"
log "warm: $WARM"
log "moe : $MOE"

VERDICT=UNRESOLVED
if [ -f "$WARM" ] && [ -f "$MOE" ]; then
  for pair in "warm4k:$WARM:Isaac-Extreme-Parkour-StudentMoEWarm-Unitree-Go2-Eval-v0" \
              "moe4k:$MOE:Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0"; do
    tag=${pair%%:*}; rest=${pair#*:}; ck=${rest%%:*}; task=${rest#*:}
    log "eval $tag"
    nice -n 10 timeout -k 60 3600 "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py \
      --task "$task" --checkpoint "$ck" --num_envs "$ENVS" --steps "$STEPS" \
      --tag "$tag" --out_dir "$OUT" --headless >> "$LOG" 2>&1 \
      && log "  ok" || log "  FAILED"
  done
  VERDICT=$("$PYTHON_BIN" - "$OUT" <<'PY' 2>>"$LOG"
import math, os, sys
import numpy as np
sys.path.insert(0, "scripts/analysis")
from episode_metrics import episode_table, summarise
o = sys.argv[1]
try:
    r = {}
    for tag in ("warm4k", "moe4k"):
        d = np.load(os.path.join(o, f"trace_{tag}.npz"), allow_pickle=True)
        rows = episode_table(d["dones"].astype(bool), d["goal_idx"].astype(int))
        r[tag] = summarise(rows, tag, boot=1000)
except Exception as e:
    print("UNRESOLVED"); print(f"comparison failed: {e}", file=sys.stderr); sys.exit()
a, b = r["moe4k"], r["warm4k"]
dp = b["goals_per_min"] - a["goals_per_min"]
sp = math.hypot(b["goals_per_min_se"], a["goals_per_min_se"])
dh = b["fall_hazard_per_1k_steps"] - a["fall_hazard_per_1k_steps"]
sh = math.hypot(b["fall_hazard_se"], a["fall_hazard_se"])
for k, v in (("moe4k", a), ("warm4k", b)):
    print(f"{k:7s} pace {v['goals_per_min']:6.2f}+-{v['goals_per_min_se']:.2f}  "
          f"hazard {v['fall_hazard_per_1k_steps']:.3f}+-{v['fall_hazard_se']:.3f}", file=sys.stderr)
print(f"delta   pace {dp:+.2f}+-{sp:.2f}  hazard {dh:+.3f}+-{sh:.3f}", file=sys.stderr)
better = (dp >= sp) or (dh <= -sh)
worse = (dp <= -sp) or (dh >= sh)
print("WARM_HELPS" if (better and not worse) else "WARM_NEUTRAL")
PY
)
fi
log "VERDICT: $VERDICT"

# The warm run has answered its question either way; it does not need to finish.
log "stopping the warm run"
pkill -9 -f "python.*train\.py.*StudentMoEWarm" 2>/dev/null
pkill -9 -f "bash.*watchdog_variant.sh" 2>/dev/null
sleep 45
while [ "$(free_mb)" -lt 20000 ]; do sleep 30; done

if [ "$VERDICT" = "WARM_HELPS" ]; then
  DENSE_TASK=Isaac-Extreme-Parkour-StudentMoEDenseWarm-Unitree-Go2
  DENSE_EXP=unitree_go2_parkour_moe_dense_warm
else
  # UNRESOLVED lands here too: without evidence that the warm start helps, the
  # dense run stays the simpler of the two.
  DENSE_TASK=Isaac-Extreme-Parkour-StudentMoEDense-Unitree-Go2
  DENSE_EXP=unitree_go2_parkour_moe_dense
fi
log "dense variant chosen: $DENSE_TASK"

# ------------------------------------------------------- 2. dense distillation
log "STAGE 2: dense distillation, 15000 iterations"
setsid "$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task "${DENSE_TASK}-v0" --seed 1 --headless --num_envs 192 \
  --max_iterations 15000 --teacher_checkpoint "$TEACHER_CKPT" \
  >> logs/run_console/moe_dense.log 2>&1
log "STAGE 2 exit $?"

DENSE_CKPT=$(ls -1t "logs/rsl_rl/$DENSE_EXP"/*/model_*.pt 2>/dev/null | head -1)
[ -f "$DENSE_CKPT" ] || { log "FATAL: no dense checkpoint"; exit 1; }
log "dense checkpoint: $DENSE_CKPT"

# ------------------------------------------------------------ 3. RL on ours
log "STAGE 3: PPO fine-tuning of the dense mixture"
setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
  --task "${DENSE_TASK}-FT-v0" --checkpoint "$DENSE_CKPT" \
  --num_envs 192 --max_iterations 5000 --run_name dense_ft --headless \
  >> logs/run_console/ft_dense.log 2>&1
log "STAGE 3 exit $?"

# --------------------------------------------------------- 4. RL on the control
BASE=$(ls -1t logs/rsl_rl/unitree_go2_parkour_baseline/*/model_14999.pt 2>/dev/null | head -1)
if [ -f "$BASE" ]; then
  log "STAGE 4: PPO fine-tuning of the GRU baseline (control)"
  setsid "$PYTHON_BIN" scripts/rsl_rl/finetune_rl.py \
    --task Isaac-Extreme-Parkour-Student-Unitree-Go2-FT-v0 --checkpoint "$BASE" \
    --num_envs 192 --max_iterations 5000 --run_name base_ft --headless \
    >> logs/run_console/ft_base.log 2>&1
  log "STAGE 4 exit $?"
else
  log "ALERT: no baseline checkpoint; the control cannot run and the comparison is incomplete"
fi
log "=== plan finished ==="
