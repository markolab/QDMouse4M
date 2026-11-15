#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"

MODEL_DIR="${1:?Usage: run_sleap_eval.sh MODEL_DIR EVAL_ROOT SPLIT OUT_DIR}"
EVAL_ROOT="${2:?Usage: run_sleap_eval.sh MODEL_DIR EVAL_ROOT SPLIT OUT_DIR}"
SPLIT="${3:?Usage: run_sleap_eval.sh MODEL_DIR EVAL_ROOT SPLIT OUT_DIR}"
OUT_DIR="${4:?Usage: run_sleap_eval.sh MODEL_DIR EVAL_ROOT SPLIT OUT_DIR}"

qdmouse4m infer-sleap \
  --release-root "${QDMOUSE4M_RELEASE_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --split "${SPLIT}" \
  --model-dir "${MODEL_DIR}" \
  --out-dir "${OUT_DIR}"

