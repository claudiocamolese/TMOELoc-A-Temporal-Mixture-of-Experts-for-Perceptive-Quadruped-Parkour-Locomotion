#!/usr/bin/env bash
# Shared environment for headless Isaac Sim runs. Activate Isaac Lab first.
#
#   source scripts/launch_env.sh
#
# An empty CUDA_VISIBLE_DEVICES is interpreted by carb/PhysX as "no devices",
# which makes Isaac Sim fall back to CPU and then abort, even though torch is
# perfectly happy. Pin it explicitly, and make sure the WSL GPU stubs are on
# the loader path before any virtualenv-local copies of libcuda.
EXPERTPARKOUR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
export TEACHER_CKPT="${TEACHER_CKPT:-$EXPERTPARKOUR_ROOT/pretrained/teacher/2025-08-13_05-15-58/model_49999.pt}"

# Preserve caller-provided libraries. Only prepend optional WSL/CUDA paths when
# they exist; Linux installations can use the active environment's loader path.
for parkour_library_dir in /usr/local/cuda-12.8/lib64 /usr/lib/wsl/lib; do
  if [[ -d "$parkour_library_dir" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$parkour_library_dir:"*) ;;
      *) export LD_LIBRARY_PATH="$parkour_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done
unset parkour_library_dir

# shellcheck source=scripts/lib/experiment_helpers.sh
source "$EXPERTPARKOUR_ROOT/scripts/lib/experiment_helpers.sh"
