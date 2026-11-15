#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"
WORK_DIR="${WORK_DIR:-outputs/lp3d}"
DATASET_ROOT="${DATASET_ROOT:-${QDMOUSE4M_RELEASE_ROOT}/training_subsets/refonly_pose_k50_v1_valid3dfix_20260503}"
SPLITS="${SPLITS:-train_0050 train_0100 train_0500 train_1000 train_5000 train_10000}"
BACKBONES="${BACKBONES:-dino dinov2 dinov3}"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "${WORK_DIR}"
for split in ${SPLITS}; do
  for backbone in ${BACKBONES}; do
    for seed in ${SEEDS}; do
      out="${WORK_DIR}/${split}/${backbone}/seed${seed}"
      qdmouse4m train-lp3d \
        --project-dir "${DATASET_ROOT}/lp3d/${split}" \
        --output-dir "${out}" \
        --backbone "${backbone}" \
        --seed "${seed}" \
        --resume
    done
  done
done

