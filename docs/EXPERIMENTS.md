# Experiment Matrix

This repository provides scheduler-free scripts. On a cluster, wrap these
commands with your own scheduler outside the repository.

## LP3D

Backbones:

```text
dino
dinov2
dinov3
```

Small splits:

```text
train_0050, train_0100, train_0500, train_1000, train_5000, train_10000
```

Large LP3D-only splits:

```text
train_15000_in_session_extra5k
train_20000_in_session_extra10k
train_15000_out_20251030
train_15000_out_20251109_11
train_20000_out_20251030_20251109_11
```

Default seeds:

```text
0, 1, 2
```

Default LP3D settings:

```text
resize: 256 x 256
train/val: 80/20
max_epochs: 160
min_epochs: 80
early_stop_patience: 8 validation checks
check_val_every_n_epoch: 5
temporal loss: off
PCA multiview loss: off
supervised reprojection loss: off
imgaug_3d: off
```

## SLEAP

Backbones:

```text
unet
convnext
swint
```

SLEAP uses all six views in one single-instance model. The default validation
fraction is 0.2. Large splits above 10k are LP3D-only by default.

## Keypoint-MoSeq

The public KPMS inference config is:

```text
configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml
```

It applies a trained `nst50_ds2_ld15_k1e06_seed1_seed1` model to the fixed
release 3D trajectories with:

```text
num_states: 50
latent_dim: 15
downsample_rate: 2
kappa_full: 1e5
apply_num_iters: 100
```

See `docs/KPMS.md` for the inference/export commands and syllable annotation
map.
