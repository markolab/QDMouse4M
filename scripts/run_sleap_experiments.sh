#!/usr/bin/env bash
set -euo pipefail

: "${QDMOUSE4M_RELEASE_ROOT:?Set QDMOUSE4M_RELEASE_ROOT.}"
: "${SLEAP_TEMPLATE:?Set SLEAP_TEMPLATE to a SLEAP-NN single-instance YAML template.}"

WORK_DIR="${WORK_DIR:-outputs/sleap}"
DATASET_ROOT="${DATASET_ROOT:-${QDMOUSE4M_RELEASE_ROOT}/training_subsets/refonly_pose_k50_v1_valid3dfix_20260503}"
SPLITS="${SPLITS:-train_0050 train_0100 train_0500 train_1000 train_5000 train_10000}"
BACKBONES="${BACKBONES:-unet convnext swint}"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "${WORK_DIR}"
for split in ${SPLITS}; do
  for backbone in ${BACKBONES}; do
    for seed in ${SEEDS}; do
      out="${WORK_DIR}/${split}/${backbone}/seed${seed}"
      qdmouse4m train-sleap \
        --sleap-dir "${DATASET_ROOT}/sleap/${split}" \
        --template "${SLEAP_TEMPLATE}" \
        --output-dir "${out}" \
        --backbone "${backbone}" \
        --seed "${seed}"
    done
  done
done

