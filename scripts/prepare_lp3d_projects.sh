#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT to the release root.}"

DATASET_ROOT="${1:-training_subsets/refonly_pose_k50_v1_valid3dfix_20260503}"
shift || true
SPLITS=("$@")
if [[ ${#SPLITS[@]} -eq 0 ]]; then
  SPLITS=(train_0050 train_0100 train_0500 train_1000 train_5000 train_10000)
fi

if [[ "${MATERIALIZE_IMAGES:-0}" == "1" ]]; then
  IMG_ARGS=(materialize-images --dataset-root "${DATASET_ROOT}" --workers "${MATERIALIZE_WORKERS:-6}")
  for split in "${SPLITS[@]}"; do
    IMG_ARGS+=(--split "${split}")
  done
  qdmouse4m "${IMG_ARGS[@]}"
fi

ARGS=(prepare-projects lp3d --dataset-root "${DATASET_ROOT}")
for split in "${SPLITS[@]}"; do
  ARGS+=(--split "${split}")
done

qdmouse4m "${ARGS[@]}"
