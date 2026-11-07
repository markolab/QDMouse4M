#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"

USAGE="Usage: bash scripts/run_gait.sh <kpms_syllable_bundle_dir> [out_dir] [mouse_map_csv]"
BUNDLE_DIR="${1:?${USAGE}}"
OUT_DIR="${2:-outputs/gait}"
MOUSE_MAP="${3:-}"
CONFIG="${CONFIG:-configs/gait.yaml}"
KPMS_CONFIG="${KPMS_CONFIG:-configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml}"

mkdir -p "${OUT_DIR}"

qdmouse4m gait strides \
  --config "${CONFIG}" \
  --release-root "${QDMOUSE4M_RELEASE_ROOT}" \
  --output-dir "${OUT_DIR}"

ANN_ARGS=()
if [[ -f "${BUNDLE_DIR}/syllable_annotations.json" ]]; then
  ANN_ARGS+=(--annotations-json "${BUNDLE_DIR}/syllable_annotations.json")
elif [[ -f "${KPMS_CONFIG}" ]]; then
  ANN_ARGS+=(--config "${KPMS_CONFIG}")
fi

MOUSE_ARGS=()
if [[ -n "${MOUSE_MAP}" && -f "${MOUSE_MAP}" ]]; then
  MOUSE_ARGS+=(--mouse-map "${MOUSE_MAP}")
fi

qdmouse4m gait figures \
  --config "${CONFIG}" \
  --release-root "${QDMOUSE4M_RELEASE_ROOT}" \
  --per-stride-csv "${OUT_DIR}/per_stride.csv" \
  --syllable-bundle-dir "${BUNDLE_DIR}" \
  "${ANN_ARGS[@]}" \
  "${MOUSE_ARGS[@]}" \
  --output-dir "${OUT_DIR}/figures" \
  --csv-dir "${OUT_DIR}"
