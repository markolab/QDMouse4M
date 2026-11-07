"""Materialize dynamic-crop image pools from release videos."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .constants import VIEWS
from .paths import release_root, resolve_under


@dataclass(frozen=True)
class MaterializeTask:
    release_root: str
    dataset_root: str
    session_id: str
    view: str
    rows_csv: str
    tile_size: int
    video_kind: str
    overwrite: bool


def crop_frame(frame: np.ndarray, bbox: pd.Series, tile_size: int) -> np.ndarray:
    """Crop a full frame with black padding and resize to the benchmark tile."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = max(1, int(round(float(bbox["w"]))))
    h = max(1, int(round(float(bbox["h"]))))
    transform = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y]], dtype=np.float32)
    crop = cv2.warpAffine(
        rgb,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def split_manifest(dataset_root: Path, split: str) -> pd.DataFrame:
    path = dataset_root / "subsets" / split / "selected_timepoints.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_task_csvs(dataset_root: Path, split_df: pd.DataFrame) -> list[Path]:
    task_dir = dataset_root / "_materialize_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for session_id, group in split_df.groupby("session_id", sort=True):
        path = task_dir / f"{session_id}.csv"
        group[["pool_index", "session_id", "frame_idx"]].sort_values(
            "frame_idx", kind="stable"
        ).to_csv(path, index=False)
        paths.append(path)
    return paths


def video_path(root: Path, session_id: str, view: str, video_kind: str) -> Path:
    session_video_dir = root / "sessions" / f"session_{session_id}" / "videos" / video_kind
    candidates = [
        session_video_dir / f"{view}.mp4",
        session_video_dir / f"{view}.avi",
        session_video_dir / f"{view}.mov",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {video_kind} video found for session={session_id}, view={view}")


def materialize_one(task: MaterializeTask) -> dict[str, object]:
    root = Path(task.release_root)
    dataset_root = Path(task.dataset_root)
    rows = pd.read_csv(task.rows_csv).sort_values("frame_idx", kind="stable")
    if rows.empty:
        return {"session_id": task.session_id, "view": task.view, "written": 0}

    out_dir = (
        dataset_root
        / "image_pool"
        / "labeled-data"
        / f"qdmouse_release_refonly_k50_pool_{task.view}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bboxes = pd.read_csv(
        root
        / "sessions"
        / f"session_{task.session_id}"
        / "sam3_dynamic_bboxes"
        / f"bboxes_{task.view}.csv"
    ).set_index("frame_idx", drop=False)

    cap = cv2.VideoCapture(str(video_path(root, task.session_id, task.view, task.video_kind)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for session={task.session_id}, view={task.view}")

    wanted = {int(row.frame_idx): int(row.pool_index) for row in rows.itertuples(index=False)}
    max_frame = max(wanted)
    frame_idx = 0
    written = 0
    try:
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not read frame {frame_idx} for session={task.session_id}, view={task.view}"
                )
            pool_index = wanted.get(frame_idx)
            if pool_index is not None:
                out_path = out_dir / f"img{pool_index:08d}.png"
                if task.overwrite or not out_path.exists():
                    crop = crop_frame(frame, bboxes.loc[frame_idx], task.tile_size)
                    ok_write = cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if not ok_write:
                        raise RuntimeError(f"Could not write {out_path}")
                written += 1
            frame_idx += 1
    finally:
        cap.release()
    return {
        "session_id": task.session_id,
        "view": task.view,
        "written": int(written),
        "max_frame": int(max_frame),
    }


def materialize_split(
    *,
    root: Path,
    dataset_root: Path,
    split: str,
    tile_size: int,
    workers: int,
    video_kind: str,
    overwrite: bool,
) -> list[dict[str, object]]:
    split_df = split_manifest(dataset_root, split)
    task_csvs = write_task_csvs(dataset_root, split_df)
    tasks = [
        MaterializeTask(
            release_root=str(root),
            dataset_root=str(dataset_root),
            session_id=path.stem,
            view=view,
            rows_csv=str(path),
            tile_size=tile_size,
            video_kind=video_kind,
            overwrite=overwrite,
        )
        for path in task_csvs
        for view in VIEWS
    ]

    results = []
    if workers <= 1:
        for task in tasks:
            results.append(materialize_one(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(materialize_one, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result), flush=True)
                results.append(result)
    results.sort(key=lambda row: (str(row["session_id"]), str(row["view"])))
    summary_path = dataset_root / "image_pool" / f"materialize_{split}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, default=None)
    p.add_argument("--dataset-root", required=True, help="Relative to release root or absolute.")
    p.add_argument("--split", action="append", required=True)
    p.add_argument("--tile-size", type=int, default=280)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--video-kind", default="refonly", choices=["refonly", "fluoonly"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = release_root(args.release_root)
    dataset_root = resolve_under(root, args.dataset_root)
    output = {}
    for split in args.split:
        output[split] = materialize_split(
            root=root,
            dataset_root=dataset_root,
            split=split,
            tile_size=args.tile_size,
            workers=args.workers,
            video_kind=args.video_kind,
            overwrite=args.overwrite,
        )
    print(json.dumps({"materialized": {k: len(v) for k, v in output.items()}}, indent=2))


if __name__ == "__main__":
    main()
