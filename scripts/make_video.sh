#!/usr/bin/env bash
# Record and render a rollout from the most recently saved checkpoint.
#
#   ./scripts/make_video.sh                  # newest checkpoint of the newest run
#   ./scripts/make_video.sh -s 900 -n 8      # longer rollout, more environments
#   ./scripts/make_video.sh -e 2             # bypass the router, run expert 2 alone
#   ./scripts/make_video.sh -c path/to/model_5000.pt
#
# Safe to run while training is in progress: it skips any checkpoint written in
# the last few seconds, which would otherwise be a half-flushed torch.save, and
# it only reads. Expect the training run to slow down for the couple of minutes
# the simulator is up.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/launch_env.sh

NUM_ENVS=4
STEPS=600
EXPERT=""
CKPT=""
RUN_DIR=""
TASK="Isaac-Extreme-Parkour-StudentMoE-Unitree-Go2-Play-v0"
STRIDE=2
ELEV=18
AZIM=-62
MIN_AGE=20   # seconds a checkpoint must have existed before we trust it

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  echo
  echo "  -c FILE   checkpoint to use (default: newest under the run directory)"
  echo "  -r DIR    run directory     (default: most recently modified under logs/rsl_rl)"
  echo "  -n N      environments      (default $NUM_ENVS)"
  echo "  -s N      control steps     (default $STEPS)"
  echo "  -e N      force expert N, MoE only"
  echo "  -t TASK   gym task id       (default $TASK)"
  echo "  --stride/--elev/--azim  rendering options"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CKPT="$2"; shift 2 ;;
    -r) RUN_DIR="$2"; shift 2 ;;
    -n) NUM_ENVS="$2"; shift 2 ;;
    -s) STEPS="$2"; shift 2 ;;
    -e) EXPERT="$2"; shift 2 ;;
    -t) TASK="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --elev) ELEV="$2"; shift 2 ;;
    --azim) AZIM="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- checkpoint
if [[ -z "$CKPT" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(find logs/rsl_rl -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
  fi
  if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "no run directory found under logs/rsl_rl" >&2; exit 1
  fi
  # Newest by mtime, skipping anything still being written: a torch.save caught
  # mid-flush loads as a truncated archive. NB: -mmin only takes whole minutes,
  # and a relative -newermt argument is not portable, so pass a real timestamp.
  CUTOFF=$(date -d "-${MIN_AGE} seconds" '+%Y-%m-%d %H:%M:%S')
  CKPT=$(find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' ! -newermt "$CUTOFF" \
         -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [[ -z "$CKPT" ]]; then
    CKPT=$(find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -printf '%T@ %p\n' \
           | sort -rn | head -1 | cut -d' ' -f2-)
    [[ -n "$CKPT" ]] && echo "warning: only a just-written checkpoint is available, it may be incomplete"
  fi
fi
[[ -f "$CKPT" ]] || { echo "checkpoint not found: ${CKPT:-<none>}" >&2; exit 1; }

RUN_DIR=$(dirname "$CKPT")
ITER=$(basename "$CKPT" .pt | sed 's/^model_//')
TAG="router"; EXPERT_ARG=()
if [[ -n "$EXPERT" ]]; then TAG="expert$EXPERT"; EXPERT_ARG=(--force_expert "$EXPERT"); fi
OUT="$RUN_DIR/playback_it${ITER}"

echo "checkpoint : $CKPT  (iteration $ITER)"
echo "task       : $TASK"
echo "output     : $OUT"
echo "rollout    : $NUM_ENVS envs x $STEPS steps, routing=$TAG"
echo

# ------------------------------------------------------------------- record
"$PYTHON_BIN" scripts/rsl_rl/play_student.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs "$NUM_ENVS" --steps "$STEPS" \
  --out_dir "$OUT" --headless "${EXPERT_ARG[@]}" \
  > "$OUT.log" 2>&1
STATUS=$?
if [[ $STATUS -ne 0 || ! -f "$OUT/rollout_$TAG.npz" ]]; then
  echo "rollout failed (exit $STATUS); last lines of $OUT.log:" >&2
  tail -25 "$OUT.log" >&2
  exit 1
fi
mv -f "$OUT.log" "$OUT/record.log"

# ------------------------------------------------------------------- render
for ((e = 0; e < NUM_ENVS; e++)); do
  "$PYTHON_BIN" scripts/analysis/render_rollout.py "$OUT/rollout_$TAG.npz" \
    --env "$e" --stride "$STRIDE" --elev "$ELEV" --azim "$AZIM" | tail -1
done

echo
echo "done. videos in $OUT:"
ls -1sh "$OUT"/*.mp4 | sed 's/^/  /'
