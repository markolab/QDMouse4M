"""Convert SLEAP-NN predictions to the benchmark crop-space NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import KEYPOINTS, VIEWS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--pred-slp", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    return p.parse_args()


def view_from_video(video) -> str:
    text = str(getattr(video, "filename", "") or getattr(video, "source_filename", ""))
    for view in VIEWS:
        if view in text:
            return view
    raise ValueError(f"Could not infer view from SLEAP video filename: {text}")


def main() -> None:
    import sleap_io as sio

    args = parse_args()
    split_df = pd.read_csv(args.eval_root / "subsets" / args.split / "selected_timepoints.csv")
    pred_xy = np.full((len(split_df), len(VIEWS), len(KEYPOINTS), 2), np.nan, dtype=np.float32)
    pred_conf = np.zeros((len(split_df), len(VIEWS), len(KEYPOINTS)), dtype=np.float32)

    labels = sio.load_slp(args.pred_slp)
    for lf in labels.labeled_frames:
        view = view_from_video(lf.video)
        view_i = VIEWS.index(view)
        pool_i = int(lf.frame_idx)
        if pool_i < 0 or pool_i >= len(split_df) or not lf.instances:
            continue
        inst = lf.instances[0]
        points = inst.numpy()
        pred_xy[pool_i, view_i, :, :] = points[:, :2]
        if points.shape[1] >= 3:
            pred_conf[pool_i, view_i, :] = np.nan_to_num(points[:, 2], nan=0.0)
        else:
            pred_conf[pool_i, view_i, :] = np.isfinite(points[:, :2]).all(axis=1).astype(np.float32)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        pool_index=split_df["pool_index"].to_numpy(dtype=np.int64),
        keypoints_crop_xy=pred_xy,
        keypoint_confidence=pred_conf,
        view_names=np.asarray(VIEWS),
        keypoint_names=np.asarray(KEYPOINTS),
    )
    print(args.out_npz)


if __name__ == "__main__":
    main()

