"""Load release annotations and convert between full-frame and crop coordinates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import KEYPOINTS, VIEWS


@lru_cache(maxsize=64)
def load_annotations(release_root: str, session_id: str) -> dict[str, np.ndarray]:
    path = (
        Path(release_root)
        / "sessions"
        / f"session_{session_id}"
        / "labels"
        / "keypoint_annotations.npz"
    )
    z = np.load(path, allow_pickle=False)
    data = {k: np.array(z[k]) for k in z.files}
    names = [str(x) for x in data["keypoint_names"].astype(str).tolist()]
    if names != KEYPOINTS:
        raise RuntimeError(f"Unexpected keypoint order in {path}: {names}")
    return data


@lru_cache(maxsize=512)
def load_bboxes(release_root: str, session_id: str, view: str) -> pd.DataFrame:
    path = (
        Path(release_root)
        / "sessions"
        / f"session_{session_id}"
        / "sam3_dynamic_bboxes"
        / f"bboxes_{view}.csv"
    )
    return pd.read_csv(path).set_index("frame_idx", drop=False)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def image_pool_path(dataset_root: Path, view: str, pool_index: int) -> Path:
    return (
        dataset_root
        / "image_pool"
        / "labeled-data"
        / f"qdmouse_release_refonly_k50_pool_{view}"
        / f"img{pool_index:08d}.png"
    )


def crop_labels_for_row(
    *,
    release_root: Path,
    session_id: str,
    frame_idx: int,
    view: str,
    view_index: int,
    tile_size: float,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    ann = load_annotations(str(release_root), session_id)
    bboxes = load_bboxes(str(release_root), session_id, view)
    bbox = bboxes.loc[int(frame_idx)]

    xy_full = ann["keypoints_2d_px"][frame_idx, view_index].astype(np.float32)
    conf = ann["keypoint_confidence_2d"][frame_idx, view_index].astype(np.float32)
    valid3d = ann.get("valid_3d")

    xy_crop = np.full((len(KEYPOINTS), 2), np.nan, dtype=np.float32)
    x0 = float(bbox["x"])
    y0 = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    xy_crop[:, 0] = (xy_full[:, 0] - x0) * tile_size / max(w, 1e-6)
    xy_crop[:, 1] = (xy_full[:, 1] - y0) * tile_size / max(h, 1e-6)
    inside = (
        np.isfinite(xy_crop).all(axis=1)
        & (conf >= confidence_threshold)
        & (xy_crop[:, 0] >= 0)
        & (xy_crop[:, 0] < tile_size)
        & (xy_crop[:, 1] >= 0)
        & (xy_crop[:, 1] < tile_size)
    )
    if valid3d is not None:
        inside &= valid3d[frame_idx].astype(bool)
    xy_crop[~inside] = np.nan
    return xy_crop, conf, {"x": x0, "y": y0, "h": h, "w": w}


def multilevel_lp_dataframe(rows: list[tuple[str, np.ndarray]], scorer: str = "qdmouse") -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [(scorer, keypoint, coord) for keypoint in KEYPOINTS for coord in ("x", "y")]
    )
    data = []
    index = []
    for image_path, xy in rows:
        index.append(image_path)
        data.append(xy.reshape(-1))
    return pd.DataFrame(data, index=index, columns=columns)

