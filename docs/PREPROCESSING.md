# Preprocessing

The preprocessing code has two stages:

1. Materialize refonly dynamic-crop PNGs from released videos and SAM3 dynamic
   bounding boxes.
2. Convert split manifests and canonical labels into LP3D/SLEAP project folders.

It does not create new labels; it reads `keypoint_annotations.npz`, applies the
SAM3 dynamic bounding boxes, and writes crop-space labels.

## Image Pool Materialization

Command:

```bash
qdmouse4m materialize-images \
  --dataset-root training_subsets/refonly_pose_k50_v1_valid3dfix_20260503 \
  --split train_0050 \
  --workers 6
```

Inputs:

```text
sessions/session_<session_id>/videos/refonly/<view>.mp4 or <view>.avi
sessions/session_<session_id>/sam3_dynamic_bboxes/bboxes_<view>.csv
<dataset_root>/subsets/<split>/selected_timepoints.csv
```

Output:

```text
<dataset_root>/image_pool/labeled-data/qdmouse_release_refonly_k50_pool_<view>/img########.png
```

The cropper uses black padding for out-of-frame regions and resizes each dynamic
crop to the configured tile size, 280 px by default.

## LP3D

Command:

```bash
qdmouse4m prepare-projects lp3d \
  --dataset-root training_subsets/refonly_pose_k50_v1_valid3dfix_20260503 \
  --split train_0050 \
  --split train_0100
```

Output per split:

```text
lp3d/<split>/
  project.yaml
  project_summary.json
  CollectedData_<view>.csv
  bboxes_CollectedData_<view>.csv
  calibrations.csv
  calibrations/qdmouse_release_camera_calibration.toml
  labeled-data/qdmouse_release_refonly_k50_<split>_<view>/img########.png
```

The `CollectedData_<view>.csv` files contain crop-space 2D labels. The bbox
CSVs contain the corresponding full-frame SAM3 dynamic crop boxes so Lightning
Pose can map between crop-space labels and calibrated full-frame coordinates.

## SLEAP

Command:

```bash
qdmouse4m prepare-projects sleap \
  --dataset-root training_subsets/refonly_pose_k50_v1_valid3dfix_20260503 \
  --split train_0050
```

Output per split:

```text
sleap/<split>/
  labels_all_views.slp
  labels_<view>.slp
  points_<view>.npy
  confidence_<view>.npy
  image_paths_<view>.txt
  sleap_frame_index.csv
  images/<view>/img########.png
```

The all-view SLEAP file trains one 2D model over all six camera views.

## Label Filtering

The default threshold is `confidence >= 0.5`. Keypoints outside the dynamic crop
or marked invalid by the canonical 3D validity mask are written as missing.
