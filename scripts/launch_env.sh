#!/usr/bin/env bash
# Environment for headless Isaac Sim runs on this machine (WSL2 + RTX 5090).
#
#   source scripts/launch_env.sh
#
# An empty CUDA_VISIBLE_DEVICES is interpreted by carb/PhysX as "no devices",
# which makes Isaac Sim fall back to CPU and then abort, even though torch is
# perfectly happy. Pin it explicitly, and make sure the WSL GPU stubs are on
# the loader path before any virtualenv-local copies of libcuda.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN=/home/olaf/miniconda3/envs/parkour/bin/python
export TEACHER_CKPT=/home/olaf/Claudio/Isaaclab_Parkour/pretrained/teacher/2025-08-13_05-15-58/model_49999.pt
