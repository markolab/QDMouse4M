#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"

CONFIG="${CONFIG:-configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml}"
USAGE="Usage: bash scripts/run_kpms_inference.sh <kpms_project_dir> [model_name] [out_dir]"
PROJECT_DIR="${1:?${USAGE}}"
MODEL_NAME="${2:-nst50_ds2_ld15_k1e06_seed1_seed1}"
OUT_DIR="${3:-outputs/kpms/${MODEL_NAME}}"
DATASET_DIR="${OUT_DIR}/dataset"
RESULTS_H5="${OUT_DIR}/results_full.h5"
BUNDLE_DIR="${OUT_DIR}/syllable_bundle"

qdmouse4m kpms prepare \
  --config "${CONFIG}" \
  --release-root "${QDMOUSE4M_RELEASE_ROOT}" \
  --output-dir "${DATASET_DIR}"

qdmouse4m kpms apply \
  --config "${CONFIG}" \
  --dataset-pkl "${DATASET_DIR}/kpms_dataset_clean.pkl" \
  --project-dir "${PROJECT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --output-h5 "${RESULTS_H5}"

qdmouse4m kpms export \
  --config "${CONFIG}" \
  --dataset-pkl "${DATASET_DIR}/kpms_dataset_clean.pkl" \
  --results-h5 "${RESULTS_H5}" \
  --output-dir "${BUNDLE_DIR}" \
  --fill-downsample-neighbors
