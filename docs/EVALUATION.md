# Evaluation

Both SLEAP and LP3D predictions are converted to a common crop-space NPZ:

```text
predictions_crop_space.npz
  pool_index
  keypoints_crop_xy        # [T, 6, 12, 2]
  keypoint_confidence      # [T, 6, 12]
  view_names
  keypoint_names
```

The metric script compares that NPZ against the canonical release labels and
SAM3 dynamic bboxes.

## Metrics

Reported metrics include:

```text
2D crop-space error mean/P50/P90/P95/P99 in pixels
PCK at 2, 5, 10, and 20 pixels
mOKS
mAP over OKS thresholds 0.50:0.05:0.95
mAR over OKS thresholds 0.50:0.05:0.95
reprojection error in pixels when calibration tooling is available
3D jump-rate summaries when triangulation is available
```

The OKS defaults are `stddev=0.05` and `scale=280`, matching the dynamic crop
label coordinate system. Adjust them with `qdmouse4m evaluate --oks-stddev` and
`--oks-scale` if your analysis needs a different normalization.

