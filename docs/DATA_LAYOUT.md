# Data Layout

All paths in this document are relative to `$QDMOUSE4M_RELEASE_ROOT`.

## Sessions

Each session is stored as:

```text
sessions/session_<session_id>/
  labels/keypoint_annotations.npz
  sam3_dynamic_bboxes/bboxes_<view>.csv
  calibrations/camera_calibration.toml
  videos/refonly/<view>.mp4 or <view>.avi
  videos/fluoonly/<view>.mp4 or <view>.avi
  kpms/
```

`keypoint_annotations.npz` contains the canonical release labels. Consumers
should trust `valid_3d` rather than testing only whether a 3D coordinate is
finite, because some finite coordinates are intentionally marked invalid after
release-cleaning.

Keypoint-MoSeq utilities in this repository read the 3D trajectories from the
same `keypoint_annotations.npz` files and write derived syllable predictions to
user-selected output directories. If exported back into a release-like tree, the
expected session-level KPMS file pattern is:

```text
sessions/session_<session_id>/kpms/<model_name>_kpms_syllables.npz
```

## Training Splits

Small/medium refonly K50 training data:

```text
training_subsets/refonly_pose_k50_v1_valid3dfix_20260503/
```

Available splits:

```text
train_0050
train_0100
train_0500
train_1000
train_5000
train_10000
```

Large LP3D-only splits:

```text
training_subsets/refonly_pose_k50_v1_valid3dfix_large_20260503/
```

Available split names:

```text
train_15000_in_session_extra5k
train_20000_in_session_extra10k
train_15000_out_20251030
train_15000_out_20251109_11
train_20000_out_20251030_20251109_11
```

## Evaluation Splits

In-session evaluation:

```text
evaluation_sets/refonly_pose_k50_in_session_v1_valid3dfix_20260503/
```

Common split names:

```text
test_100000_stratified
test_10000_in
```

Out-of-session evaluation:

```text
evaluation_sets/refonly_pose_k50_outsession_v1_valid3dfix_20260503/
```

Common split names:

```text
test_100000_out
test_10000_out
```

The 10k out split is a stratified subset of the 100k out split.
