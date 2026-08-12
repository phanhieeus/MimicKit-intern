#!/usr/bin/env bash
# Install MimicKit dependencies on a Kaggle notebook (Python 3.11, CUDA GPU).
#
# Isaac Gym is not installable on Kaggle (it needs a manual NVIDIA download and
# Python <= 3.8), so we use the Newton engine, which installs from PyPI.
#
# Usage (from the repo root):
#   bash kaggle/setup.sh

set -euo pipefail

echo "=== apt packages (GL + virtual display, needed only for rendering/video) ==="
apt-get update -qq
# ffmpeg is what tools/render_robot_video.py pipes raw frames into to make the mp4.
apt-get install -y -qq xvfb libgl1 libglx-mesa0 libegl1 libglu1-mesa libosmesa6 ffmpeg > /dev/null

echo "=== python packages ==="
# torch is already installed on Kaggle with the right CUDA build -- never let pip
# replace it, otherwise the GPU build gets swapped for a CPU one.
#
# warp-lang is pinned to 1.15.0 on purpose. mujoco-warp 3.5.0.2 declares
# warp-lang>=1.11.0, but its update_gradient_JTDAJ_sparse_tiled kernel
# (_src/solver.py:2274) reads J_kj in an else-branch where only the if-branch
# assigns it. Warp <= 1.15 hoisted such variables to function scope; 1.16.0
# tightened symbol resolution, so that kernel now fails to compile with
# "Referencing undefined symbol: J_kj" the first time the solver steps.
pip install -q \
    "newton==1.0.0" \
    "mujoco==3.5.0" \
    "mujoco-warp==3.5.0.2" \
    "warp-lang==1.15.0" \
    "pyglet>=2.1.6,<3" \
    "usd-core>=25.5" \
    "trimesh>=4.6.8" \
    gymnasium \
    "diffusers>=0.36.0" \
    moviepy \
    matplotlib \
    pyyaml \
    tensorboardX \
    "wandb>=0.17.4"

echo "=== versions ==="
python - <<'PY'
import torch, warp, newton
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("warp    :", warp.config.version)
print("newton  :", newton.__version__)
PY

echo "=== done ==="
