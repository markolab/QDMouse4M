# Keypoint-MoSeq Inference

This repository includes public-release utilities for applying a trained
Keypoint-MoSeq model to the released QDMouse4M 3D trajectories and exporting
full-frame per-session syllable predictions.

All commands are path-relative to `$QDMOUSE4M_RELEASE_ROOT` or to paths supplied
by the user. No cluster-specific paths are assumed.

## Data Source

KPMS prep reads:

```text
$QDMOUSE4M_RELEASE_ROOT/sessions/session_<session_id>/labels/keypoint_annotations.npz
```

It uses:

```text
keypoints_3d
keypoint_confidence_3d
valid_3d
invalid_jump_timepoint, if present
keypoint_names
frame_index, if present
```

The prep step masks invalid 3D cells as missing observations before KPMS
inference. Downstream code should trust `valid_3d` and
`invalid_jump_timepoint` rather than relying only on coordinate finiteness.

## Recommended Release Config

The release config for the current public KPMS model is:

```text
configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml
```

Model settings:

```text
num_states: 50
latent_dim: 15
downsample_rate: 2
kappa_ar: 1e6
kappa_full: 1e5
seed: 1
model_name: nst50_ds2_ld15_k1e06_seed1_seed1
```

The config also stores the bodypart order, KPMS skeleton/anterior/posterior
bodypart definitions, and the final 23-name ethogram annotations used for
release summaries.

## One-Step Wrapper

After installing `keypoint-moseq`, `jax`, `jax-moseq`, and `h5py` in a
JAX-compatible environment, run:

```bash
export QDMOUSE4M_RELEASE_ROOT=/path/to/qdmouse4m_release_neurips_v1

bash scripts/run_kpms_inference.sh \
  /path/to/kpms_project_dir \
  nst50_ds2_ld15_k1e06_seed1_seed1 \
  outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1
```

`/path/to/kpms_project_dir` should be the Keypoint-MoSeq project directory that
contains the trained model checkpoint folder
`nst50_ds2_ld15_k1e06_seed1_seed1/`.

## Manual Steps

Prepare a KPMS dataset pickle from the release:

```bash
qdmouse4m kpms prepare \
  --config configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml \
  --release-root "$QDMOUSE4M_RELEASE_ROOT" \
  --output-dir outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/dataset
```

Apply a trained KPMS checkpoint to all release sessions:

```bash
qdmouse4m kpms apply \
  --config configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml \
  --dataset-pkl outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/dataset/kpms_dataset_clean.pkl \
  --project-dir /path/to/kpms_project_dir \
  --model-name nst50_ds2_ld15_k1e06_seed1_seed1 \
  --output-h5 outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/results_full.h5
```

Export compact per-session syllable files:

```bash
qdmouse4m kpms export \
  --config configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml \
  --dataset-pkl outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/dataset/kpms_dataset_clean.pkl \
  --results-h5 outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/results_full.h5 \
  --output-dir outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/syllable_bundle \
  --fill-downsample-neighbors
```

Each exported session NPZ contains:

```text
syllable       # [T] int32 syllable id per frame; -1 means unknown
known          # [T] bool, true where syllable is filled
coordinates    # [T, 12, 3] release 3D coordinates used for KPMS
confidences    # [T, 12] KPMS confidence/missingness weights
bodyparts
fps
downsample_rate
```

Use frames where `known == True` and `syllable >= 0` for syllable-level
analysis.
