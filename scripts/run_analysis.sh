#!/usr/bin/env bash
# Full analysis of a trained student, in one command.
#
#   ./scripts/run_analysis.sh                       # newest checkpoint of the newest run
#   ./scripts/run_analysis.sh -c logs/.../model_15000.pt
#   ./scripts/run_analysis.sh -s 1500               # longer rollouts, tighter error bars
#   ./scripts/run_analysis.sh --skip-ablation       # faster: no causal test
#
# Five stages, each writing under <checkpoint_dir>/analysis/:
#
#   1  performance   MXD and MEV per terrain, on the fixed EVAL config
#   2  routing       expert x terrain, expert x gait phase, effective expert count
#   3  attention     does the newest token use the 0.8 s window, and does the
#                    lookback change with the gait phase
#   4  probe         is mass / COM / friction linearly decodable from the context
#                    (needs the TRAINING config: EVAL disables that randomisation)
#   5  ablation      project the decodable directions out of the context and
#                    re-measure MXD. Separates "the context stores it" from
#                    "the actor uses it"
#
# Safe to run while training is in progress: nice'd, read-only, and it skips any
# checkpoint written in the last few seconds. Expect training to slow down while
# the simulator is up.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

CKPT=""; RUN_DIR=""; STEPS=800; NUM_ENVS=192; SKIP_ABL=0; MIN_AGE=30
EVAL_TASK="Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Eval-v0"
TRAIN_TASK="Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-v0"

usage() {
  sed -n '2,23p' "$0" | sed 's/^# \?//'
  cat <<'EOF'

  -c FILE   checkpoint (default: newest of the newest run)
  -r DIR    run directory
  -s N      rollout length in control steps (default 800)
  -n N      environments (default 192)
  --skip-ablation      skip stage 5
  --eval-task  TASK    gym id for the fixed-config stages
  --train-task TASK    gym id for the probe stage
EOF
  exit 0
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CKPT="$2"; shift 2 ;;
    -r) RUN_DIR="$2"; shift 2 ;;
    -s) STEPS="$2"; shift 2 ;;
    -n) NUM_ENVS="$2"; shift 2 ;;
    --eval-task) EVAL_TASK="$2"; shift 2 ;;
    --train-task) TRAIN_TASK="$2"; shift 2 ;;
    --skip-ablation) SKIP_ABL=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  [[ -z "$RUN_DIR" ]] && RUN_DIR=$(find logs/rsl_rl -mindepth 2 -maxdepth 2 -type d \
      -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [[ -d "$RUN_DIR" ]] || { echo "no run directory under logs/rsl_rl" >&2; exit 1; }
  CUTOFF=$(date -d "-${MIN_AGE} seconds" '+%Y-%m-%d %H:%M:%S')
  CKPT=$(find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' ! -newermt "$CUTOFF" \
         -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
[[ -f "$CKPT" ]] || { echo "checkpoint not found: ${CKPT:-<none>}" >&2; exit 1; }

ITER=$(basename "$CKPT" .pt | sed 's/^model_//')
OUT="$(dirname "$CKPT")/analysis"
mkdir -p "$OUT"
EV=(nice -n 15 timeout -k 30 3600 "$PYTHON_BIN" scripts/rsl_rl/evaluate_students.py
    --num_envs "$NUM_ENVS" --headless --out_dir "$OUT")

echo "checkpoint : $CKPT  (iteration $ITER)"
echo "output     : $OUT"
echo "rollouts   : $NUM_ENVS envs x $STEPS steps"
echo

stage () { echo; echo "──────── $1 ────────"; }

stage "1+2+3  performance, routing and attention (EVAL config)"
"${EV[@]}" --task "$EVAL_TASK" --checkpoint "$CKPT" --steps "$STEPS" \
  --context_every 40 --attention --tag main > "$OUT/eval_main.log" 2>&1
if [[ ! -f "$OUT/trace_main.npz" ]]; then
  echo "failed; tail of $OUT/eval_main.log:" >&2; tail -20 "$OUT/eval_main.log" >&2; exit 1
fi
sed -n '/^terrain /,/^TOTAL/p' "$OUT/eval_main.log"
"$PYTHON_BIN" scripts/analysis/analyze_student.py "$OUT/trace_main.npz" --out "$OUT/routing.json" \
  | sed -n '/=== 1/,/=== 3/p'
"$PYTHON_BIN" scripts/analysis/attention_probe.py "$OUT/trace_main.npz" --out "$OUT/attention.json" \
  | sed -n '/=== attention/,$p'

stage "4  dynamics probe (TRAINING config: mass and COM must vary)"
"${EV[@]}" --task "$TRAIN_TASK" --checkpoint "$CKPT" --steps $((STEPS * 3 / 4)) \
  --context_every 3 --tag probe > "$OUT/eval_probe.log" 2>&1
if [[ -f "$OUT/trace_probe.npz" ]]; then
  "$PYTHON_BIN" scripts/analysis/fit_context_probe.py "$OUT/trace_probe.npz" \
    --out "$OUT/probe_directions.npz" | sed -n '/target/,$p'
else
  echo "probe rollout failed; see $OUT/eval_probe.log" >&2
fi

if [[ $SKIP_ABL -eq 0 && -f "$OUT/probe_directions.npz" ]]; then
  stage "5  ablation: project the decodable directions out of the context"
  "${EV[@]}" --task "$EVAL_TASK" --checkpoint "$CKPT" --steps "$STEPS" \
    --context_every 200 --ablate_context "$OUT/probe_directions.npz" \
    --tag ablated > "$OUT/eval_ablated.log" 2>&1
  sed -n '/^terrain /,/^TOTAL/p' "$OUT/eval_ablated.log"
  "$PYTHON_BIN" - "$OUT" <<'PY'
import json, sys, os
o = sys.argv[1]
try:
    a = json.load(open(os.path.join(o, "metrics_main.json")))["total"]
    b = json.load(open(os.path.join(o, "metrics_ablated.json")))["total"]
except FileNotFoundError as e:
    sys.exit(f"cannot compare: {e}")
import math
sa = a["mxd_std"] / math.sqrt(max(a["episodes"], 1))
sb = b["mxd_std"] / math.sqrt(max(b["episodes"], 1))
d = b["mxd_mean"] - a["mxd_mean"]; se = math.sqrt(sa**2 + sb**2)
print(f"\nMXD intact  {a['mxd_mean']:.3f} +- {sa:.3f}  ({a['episodes']} episodes)")
print(f"MXD ablated {b['mxd_mean']:.3f} +- {sb:.3f}  ({b['episodes']} episodes)")
print(f"change      {d:+.3f} +- {se:.3f}   ->  {abs(d)/se if se else 0:.1f} sigma")
print("verdict     " + ("the actor uses the dynamics information"
                        if d < -2 * se else
                        "no measurable effect: the context stores it, the actor does not rely on it"))
PY
fi

echo
echo "done. everything under $OUT:"
ls -1sh "$OUT" | sed 's/^/  /'
