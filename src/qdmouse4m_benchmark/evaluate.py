"""Compute benchmark metrics from crop-space prediction NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import KEYPOINTS, VIEWS
from .paths import release_root
from .release_io import crop_labels_for_row, load_bboxes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, default=None)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--pred-npz", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, default=None)
    p.add_argument("--model-kind", default="")
    p.add_argument("--model-name", default="")
    p.add_argument("--tile-size", type=float, default=280.0)
    p.add_argument("--gt-confidence-threshold", type=float, default=0.5)
    p.add_argument("--pred-confidence-threshold", type=float, default=0.5)
    p.add_argument("--oks-stddev", type=float, default=0.05)
    p.add_argument("--oks-scale", type=float, default=280.0)
    p.add_argument("--fps", type=float, default=60.0)
    return p.parse_args()


def finite_percentiles(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out: dict[str, float | int | None] = {f"{prefix}_n": int(values.size)}
    if values.size == 0:
        for key in ("mean", "p50", "p90", "p95", "p99"):
            out[f"{prefix}_{key}"] = None
        return out
    out[f"{prefix}_mean"] = float(np.mean(values))
    for q in (50, 90, 95, 99):
        out[f"{prefix}_p{q}"] = float(np.percentile(values, q))
    return out


def build_ground_truth(
    root: Path,
    split_df: pd.DataFrame,
    tile_size: float,
    conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt_crop = np.full((len(split_df), len(VIEWS), len(KEYPOINTS), 2), np.nan, dtype=np.float32)
    gt_conf = np.zeros((len(split_df), len(VIEWS), len(KEYPOINTS)), dtype=np.float32)
    for row_i, row in enumerate(split_df.itertuples(index=False)):
        for view_i, view in enumerate(VIEWS):
            xy, conf, _bbox = crop_labels_for_row(
                release_root=root,
                session_id=str(row.session_id),
                frame_idx=int(row.frame_idx),
                view=view,
                view_index=view_i,
                tile_size=tile_size,
                confidence_threshold=conf_thr,
            )
            gt_crop[row_i, view_i] = xy
            gt_conf[row_i, view_i] = conf
    return gt_crop, gt_conf


def crop_to_full(root: Path, split_df: pd.DataFrame, pred_crop: np.ndarray, tile_size: float) -> np.ndarray:
    full = np.full_like(pred_crop, np.nan)
    for row_i, row in enumerate(split_df.itertuples(index=False)):
        session_id = str(row.session_id)
        frame_idx = int(row.frame_idx)
        for view_i, view in enumerate(VIEWS):
            bbox = load_bboxes(str(root), session_id, view).loc[frame_idx]
            full[row_i, view_i, :, 0] = pred_crop[row_i, view_i, :, 0] * float(bbox["w"]) / tile_size + float(
                bbox["x"]
            )
            full[row_i, view_i, :, 1] = pred_crop[row_i, view_i, :, 1] * float(bbox["h"]) / tile_size + float(
                bbox["y"]
            )
    return full


def try_triangulate(root: Path, split_df: pd.DataFrame, pred_full: np.ndarray, pred_conf: np.ndarray, conf_thr: float):
    try:
        from lightning_pose.data.cameras import CameraGroup
    except Exception:
        return None, None
    first_session = str(split_df.iloc[0].session_id)
    cal = root / "sessions" / f"session_{first_session}" / "calibrations" / "camera_calibration.toml"
    camgroup = CameraGroup.load(cal)
    pts = pred_full.copy()
    pts[pred_conf < conf_thr] = np.nan
    n, v, k, _ = pts.shape
    flat = np.transpose(pts, (1, 0, 2, 3)).reshape(v, n * k, 2)
    pred_3d = camgroup.triangulate_fast(flat).reshape(n, k, 3).astype(np.float32)
    reproj = camgroup.project(pred_3d.reshape(n * k, 3)).reshape(v, n, k, 2)
    reproj = np.transpose(reproj, (1, 0, 2, 3)).astype(np.float32)
    reproj_err = np.linalg.norm(reproj - pts, axis=-1)
    reproj_err[~np.isfinite(pts).all(axis=-1)] = np.nan
    return pred_3d, reproj_err


def oks_metrics(errors: np.ndarray, gt_valid: np.ndarray, pred_valid: np.ndarray, stddev: float, scale: float):
    denom = 2.0 * (stddev * scale) ** 2
    oks = np.exp(-(errors**2) / max(denom, 1e-9))
    oks[~np.isfinite(errors)] = 0.0
    matched = gt_valid & pred_valid & np.isfinite(errors)
    out = {
        "mOKS": float(np.mean(oks[matched])) if np.any(matched) else None,
        "oks_n_gt": int(np.count_nonzero(gt_valid)),
        "oks_n_pred": int(np.count_nonzero(pred_valid)),
        "oks_n_matched": int(np.count_nonzero(matched)),
    }
    ap, ar = [], []
    for thr in np.arange(0.50, 0.951, 0.05):
        tp = np.count_nonzero(matched & (oks >= thr))
        ap.append(tp / max(1, np.count_nonzero(pred_valid)))
        ar.append(tp / max(1, np.count_nonzero(gt_valid)))
    out["mAP"] = float(np.mean(ap))
    out["mAR"] = float(np.mean(ar))
    return out


def jump_metrics(split_df: pd.DataFrame, pred_3d: np.ndarray, fps: float):
    if pred_3d is None:
        return {}
    speeds = []
    rows = split_df[["session_id", "frame_idx"]].copy()
    rows["_i"] = np.arange(len(rows))
    for _sid, group in rows.sort_values(["session_id", "frame_idx"]).groupby("session_id"):
        idx = group["_i"].to_numpy()
        frame = group["frame_idx"].to_numpy()
        if len(idx) < 2:
            continue
        dt = np.diff(frame) / fps
        dist = np.linalg.norm(pred_3d[idx[1:]] - pred_3d[idx[:-1]], axis=-1)
        sp = dist / dt[:, None] / 10.0
        sp[dt <= 0] = np.nan
        speeds.append(sp)
    if not speeds:
        return {}
    speed = np.concatenate(speeds, axis=0)
    vals = speed[np.isfinite(speed)]
    out = finite_percentiles(vals, "speed_cm_s")
    for thr in (50, 100, 200):
        out[f"jump_rate_keypoint_gt_{thr}cm_s"] = float(np.mean(vals > thr)) if vals.size else None
    return out


def main() -> None:
    args = parse_args()
    root = release_root(args.release_root)
    split_df = pd.read_csv(args.eval_root / "subsets" / args.split / "selected_timepoints.csv")
    pred = np.load(args.pred_npz, allow_pickle=False)
    pred_xy = pred["keypoints_crop_xy"].astype(np.float32)
    pred_conf = pred["keypoint_confidence"].astype(np.float32)
    gt_xy, gt_conf = build_ground_truth(root, split_df, args.tile_size, args.gt_confidence_threshold)

    gt_valid = np.isfinite(gt_xy).all(axis=-1)
    pred_valid = np.isfinite(pred_xy).all(axis=-1) & (pred_conf >= args.pred_confidence_threshold)
    err2d = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    err2d[~(gt_valid & pred_valid)] = np.nan
    out = {
        "model_kind": args.model_kind,
        "model_name": args.model_name,
        "eval_root": str(args.eval_root),
        "split": args.split,
        "n_timepoints": int(len(split_df)),
        "n_views": len(VIEWS),
        "n_keypoints": len(KEYPOINTS),
    }
    out.update(finite_percentiles(err2d, "error2d_px"))
    for thr in (2, 5, 10, 20):
        vals = err2d[gt_valid & pred_valid]
        out[f"pck_{thr}px"] = float(np.mean(vals <= thr)) if vals.size else None
    out.update(oks_metrics(err2d, gt_valid, pred_valid, args.oks_stddev, args.oks_scale))

    pred_full = crop_to_full(root, split_df, pred_xy, args.tile_size)
    pred_3d, reproj_err = try_triangulate(root, split_df, pred_full, pred_conf, args.pred_confidence_threshold)
    if reproj_err is not None:
        out.update(finite_percentiles(reproj_err, "reproj_px"))
    if pred_3d is not None:
        out.update(jump_metrics(split_df, pred_3d, args.fps))
        if args.out_npz:
            args.out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out_npz, pred_3d=pred_3d, reproj_error_px=reproj_err, error2d_px=err2d)
    elif args.out_npz:
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, error2d_px=err2d)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
