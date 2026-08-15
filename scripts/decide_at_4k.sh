#!/usr/bin/env bash
# At iteration 4000 of the failure-aware run, decide whether to keep it.
#
#   setsid nohup ./scripts/decide_at_4k.sh > /dev/null 2>&1 < /dev/null &
#
# The criterion is fixed here, before the data exist, so it cannot be moved
# afterwards. terrain_levels is deliberately NOT the judge: it is a curriculum
# quantity, and iterations 12000 and 14999 of the first run were statistically
# identical in evaluation while it appeared to decline. The judge is the
# evaluation itself, on the metrics the project is scored by.
#
# Both checkpoints are taken at iteration 4000 and scored under one protocol.
#
#   KEEP   the risk run is better on at least one axis by >= 1 sigma and not
#          worse on the other by >= 1 sigma
#   SWITCH otherwise -- the objective-split hypothesis has not paid, and the
#          remaining candidate is the warm-started actor, which is what explains
#          the baseline's early lead
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

LOG=logs/decide_4k.log
TARGET=4000
ENVS=128
STEPS=1200
RISK_CONSOLE=logs/run_console/moe_risk.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== waiting for the risk run to reach $TARGET ==="
while true; do
  it=$(grep -c "Learning iteration" "$RISK_CONSOLE" 2>/dev/null || echo 0)
  [ "$it" -ge "$TARGET" ] && { log "risk run reached $it"; break; }
  pgrep -f "python.*train\.py.*StudentMoERisk" >/dev/null 2>&1 || {
    log "ALERT: risk run died at $it; deciding on what exists"; break; }
  sleep 180
done
sleep 60

RISK_CKPT=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe_risk/*/model_4000.pt 2>/dev/null | head -1)
MOE_CKPT=$(ls -1 logs/rsl_rl/unitree_go2_parkour_moe/*/model_4000.pt 2>/dev/null | head -1)
[ -f "$RISK_CKPT" ] || RISK_CKPT=$(ls -1t logs/rsl_rl/unitree_go2_parkour_moe_risk/*/model_*.pt 2>/dev/null | head -1)
[ -f "$RISK_CKPT" ] && [ -f "$MOE_CKPT" ] || { log "FATAL: missing checkpoint (risk=$RISK_CKPT moe=$MOE_CKPT)"; exit 1; }
OUT=logs/decide_4k; mkdir -p "$OUT"
log "risk: $RISK_CKPT"
log "moe : $MOE_CKPT"

# Each checkpoint must be evaluated under the config it was trained with.
# Loading the risk checkpoint into the plain MoE config fails outright -- the
# risk head is an unexpected key and dyn_head is [3,64] against [5,64] -- and
# that failure is what produced the empty verdict the first time.
for pair in "risk4k:$RISK_CKPT:Isaac-Extreme-Parkour-StudentMoERisk-Unitree-Go2-Eval-v0" \
            "moe4k:$MOE_CKPT:Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0"; do
  tag=${pair%%:*}; rest=${pair#*:}; ck=${rest%%:*}; TASK=${rest#*:}
  log "eval $tag"
  nice -n 10 timeout -k 60 3600 "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py \
    --task "$TASK" --checkpoint "$ck" --num_envs "$ENVS" --steps "$STEPS" \
    --tag "$tag" --out_dir "$OUT" --headless >> "$LOG" 2>&1 \
    && log "  ok" || log "  FAILED"
done

VERDICT=$("$PYTHON_BIN" - "$OUT" <<'PY'
import json, math, sys, os
o=sys.argv[1]
try:
    import numpy as np
    sys.path.insert(0,"scripts/analysis")
    from episode_metrics import episode_table, summarise
    r={}
    for tag in ("risk4k","moe4k"):
        d=np.load(os.path.join(o,f"trace_{tag}.npz"),allow_pickle=True)
        rows=episode_table(d["dones"].astype(bool), d["goal_idx"].astype(int))
        r[tag]=summarise(rows, tag, boot=1000)
except Exception as e:
    print("ERROR"); sys.exit()
a,b=r["moe4k"],r["risk4k"]
dp=b["goals_per_min"]-a["goals_per_min"]; sp=math.hypot(b["goals_per_min_se"],a["goals_per_min_se"])
dh=b["fall_hazard_per_1k_steps"]-a["fall_hazard_per_1k_steps"]; sh=math.hypot(b["fall_hazard_se"],a["fall_hazard_se"])
print(f"moe4k  pace {a['goals_per_min']:.2f}+-{a['goals_per_min_se']:.2f}  hazard {a['fall_hazard_per_1k_steps']:.3f}+-{a['fall_hazard_se']:.3f}", file=sys.stderr)
print(f"risk4k pace {b['goals_per_min']:.2f}+-{b['goals_per_min_se']:.2f}  hazard {b['fall_hazard_per_1k_steps']:.3f}+-{b['fall_hazard_se']:.3f}", file=sys.stderr)
print(f"delta  pace {dp:+.2f}+-{sp:.2f}  hazard {dh:+.3f}+-{sh:.3f}", file=sys.stderr)
better = (dp >= sp) or (dh <= -sh)
worse  = (dp <= -sp) or (dh >= sh)
print("KEEP" if (better and not worse) else "SWITCH")
PY
2>>"$LOG")
log "VERDICT: $VERDICT"

# Fail SAFE, not open. The first version treated anything that was not
# literally KEEP as a licence to switch, so when the comparison errored it
# killed a healthy run on no evidence at all. Only an explicit SWITCH may
# terminate anything.
if [ "$VERDICT" != "SWITCH" ]; then
  log "verdict is '$VERDICT' -- not an explicit SWITCH, so nothing is killed."
  [ "$VERDICT" = "KEEP" ] && log "keeping the failure-aware run." \
                          || log "ALERT: the comparison did not produce a verdict; needs a human."
  exit 0
fi

log "switching to the warm-started variant"
# -9 on purpose: Isaac Sim ignores SIGTERM, so a plain pkill leaves the run
# alive, and two 192-environment simulators then share 32.6 GB with 0.7 GB to
# spare. Measured: that is exactly what happened on the first attempt.
pkill -9 -f "python.*train\.py.*StudentMoERisk" 2>/dev/null
pkill -9 -f "bash.*watchdog_risk.sh" 2>/dev/null
sleep 45
setsid nohup "$PYTHON_BIN" scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-StudentMoEWarm-Unitree-Go2-v0 \
  --seed 1 --headless --num_envs 192 --max_iterations 15000 \
  --teacher_checkpoint "$TEACHER_CKPT" \
  >> logs/run_console/moe_warm.log 2>&1 < /dev/null &
log "launched warm-start run, pid $!"
sleep 120
grep -c "warm-started" logs/run_console/moe_warm.log >> "$LOG" 2>&1
log "=== done ==="
