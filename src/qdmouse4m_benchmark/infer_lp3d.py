"""Run LP3D inference on a prepared LP3D evaluation project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import KEYPOINTS, VIEWS
from .lp3d_tools import allow_truncated_pngs, install_dinov3_forward_patch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--metrics-only", action="store_true")
    return p.parse_args()


def extract_pool_index(image_path: str) -> int:
    stem = Path(str(image_path)).stem
    if not stem.startswith("img"):
        raise ValueError(f"Unexpected image path from LP: {image_path}")
    return int(stem[3:])


def dataframe_to_arrays(df: pd.DataFrame, n_pool: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.full((n_pool, len(KEYPOINTS), 2), np.nan, dtype=np.float32)
    conf = np.zeros((n_pool, len(KEYPOINTS)), dtype=np.float32)
    for image_path, row in df.iterrows():
        pool_i = extract_pool_index(str(image_path))
        for kp_i, kp in enumerate(KEYPOINTS):
            x = y = np.nan
            likelihood = 1.0
            for col in df.columns:
                if len(col) >= 3 and str(col[1]) == kp:
                    if str(col[2]) == "x":
                        x = float(row[col])
                    elif str(col[2]) == "y":
                        y = float(row[col])
                    elif str(col[2]) == "likelihood":
                        likelihood = float(row[col])
            if np.isfinite(x) and np.isfinite(y):
                xy[pool_i, kp_i] = (x, y)
                conf[pool_i, kp_i] = likelihood
    return xy, conf


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    project_dir = eval_root / "lp3d" / args.split
    split_csv = eval_root / "subsets" / args.split / "selected_timepoints.csv"
    split_df = pd.read_csv(split_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_npz = args.out_dir / "predictions_crop_space.npz"

    if not args.metrics_only and not pred_npz.exists():
        allow_truncated_pngs()
        install_dinov3_forward_patch()
        from lightning_pose.api.model import Model

        model = Model.from_dir(args.model_dir)
        try:
            model.cfg.eval.batch_size = int(args.batch_size)
        except Exception:
            pass
        result = model.predict_on_label_csv_multiview(
            csv_file_per_view=[project_dir / f"CollectedData_{view}.csv" for view in VIEWS],
            bbox_file_per_view=None,
            camera_params_file=project_dir / "calibrations.csv",
            data_dir=project_dir,
            compute_metrics=False,
            add_train_val_test_set=False,
        )
        n_pool = len(split_df)
        pred_xy = np.full((n_pool, len(VIEWS), len(KEYPOINTS), 2), np.nan, dtype=np.float32)
        pred_conf = np.zeros((n_pool, len(VIEWS), len(KEYPOINTS)), dtype=np.float32)
        for view_i, view in enumerate(VIEWS):
            df = result.predictions[view]
            df.to_csv(args.out_dir / f"predictions_{view}.csv")
            view_xy, view_conf = dataframe_to_arrays(df, n_pool)
            pred_xy[:, view_i] = view_xy
            pred_conf[:, view_i] = view_conf
        np.savez_compressed(
            pred_npz,
            pool_index=split_df["pool_index"].to_numpy(dtype=np.int64),
            keypoints_crop_xy=pred_xy,
            keypoint_confidence=pred_conf,
            view_names=np.asarray(VIEWS),
            keypoint_names=np.asarray(KEYPOINTS),
        )
    elif not pred_npz.exists():
        raise FileNotFoundError(pred_npz)

    (args.out_dir / "lp3d_inference_summary.json").write_text(
        json.dumps(
            {
                "eval_root": str(eval_root),
                "split": args.split,
                "model_dir": str(args.model_dir.resolve()),
                "predictions": str(pred_npz),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(pred_npz)


if __name__ == "__main__":
    main()

