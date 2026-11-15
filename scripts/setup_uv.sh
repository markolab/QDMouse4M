#!/usr/bin/env bash
set -euo pipefail

uv sync --extra sleap --extra lp3d --extra dev

if [[ "${INSTALL_LIGHTNING_POSE:-0}" == "1" ]]; then
  uv pip install --extra-index-url https://pypi.nvidia.com/ lightning-pose
fi

cat <<'MSG'
Environment ready.

For LP3D training/inference on a CUDA Linux machine, rerun with:

  INSTALL_LIGHTNING_POSE=1 bash scripts/setup_uv.sh

or install lightning-pose manually into this uv environment following your local
CUDA/PyTorch instructions.
MSG
