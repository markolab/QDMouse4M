"""Prepare SLEAP and Lightning Pose 3D project folders from release manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .constants import KEYPOINTS, SKELETON_EDGES, VIEWS
from .paths import release_root, resolve_under, symlink_or_copy
from .release_io import crop_labels_for_row, image_pool_path, image_size, multilevel_lp_dataframe


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def split_manifest(dataset_root: Path, split: str) -> pd.DataFrame:
    path = dataset_root / "subsets" / split / "selected_timepoints.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_available_calibration(root: Path, split_df: pd.DataFrame) -> Path:
    for session_id in split_df["session_id"].astype(str).drop_duplicates():
        path = root / "sessions" / f"session_{session_id}" / "calibrations" / "camera_calibration.toml"
        if path.exists():
            return path
    raise FileNotFoundError("No camera_calibration.toml found for split sessions.")


def prepare_lp3d(
    *,
    root: Path,
    dataset_root: Path,
    split: str,
    out_dir: Path | None = None,
    confidence_threshold: float = 0.5,
    copy_images: bool = False,
) -> Path:
    split_df = split_manifest(dataset_root, split)
    project_dir = out_dir or (dataset_root / "lp3d" / split)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "videos").mkdir(exist_ok=True)
    (project_dir / "labeled-data").mkdir(exist_ok=True)

    session_name = f"qdmouse_release_refonly_k50_{split}"
    first_image = image_pool_path(dataset_root, VIEWS[0], int(split_df.iloc[0].pool_index))
    tile_w, tile_h = image_size(first_image)
    if tile_w != tile_h:
        raise ValueError(f"Expected square dynamic crop images; got {tile_w}x{tile_h}: {first_image}")
    tile_size = float(tile_w)

    for view_i, view in enumerate(VIEWS):
        labeled_rel_dir = Path("labeled-data") / f"{session_name}_{view}"
        labeled_abs_dir = project_dir / labeled_rel_dir
        labeled_abs_dir.mkdir(parents=True, exist_ok=True)
        label_rows: list[tuple[str, np.ndarray]] = []
        bbox_rows = []
        for row in split_df.itertuples(index=False):
            pool_index = int(row.pool_index)
            src_img = image_pool_path(dataset_root, view, pool_index)
            dst_img = labeled_abs_dir / f"img{pool_index:08d}.png"
            symlink_or_copy(src_img, dst_img, copy=copy_images)
            img_rel = (labeled_rel_dir / dst_img.name).as_posix()
            xy, _conf, bbox = crop_labels_for_row(
                release_root=root,
                session_id=str(row.session_id),
                frame_idx=int(row.frame_idx),
                view=view,
                view_index=view_i,
                tile_size=tile_size,
                confidence_threshold=confidence_threshold,
            )
            label_rows.append((img_rel, xy))
            bbox_rows.append({"image_path": img_rel, **bbox})

        labels = multilevel_lp_dataframe(label_rows)
        labels.to_csv(project_dir / f"CollectedData_{view}.csv")
        pd.DataFrame(bbox_rows).set_index("image_path").to_csv(
            project_dir / f"bboxes_CollectedData_{view}.csv"
        )

    cal_src = first_available_calibration(root, split_df)
    cal_dst = project_dir / "calibrations" / "qdmouse_release_camera_calibration.toml"
    cal_dst.parent.mkdir(exist_ok=True)
    if not cal_dst.exists():
        cal_dst.write_bytes(cal_src.read_bytes())
    cal_rows = [
        {
            "image_path": f"labeled-data/{session_name}/img{int(row.pool_index):08d}.png",
            "file": "calibrations/qdmouse_release_camera_calibration.toml",
        }
        for row in split_df.itertuples(index=False)
    ]
    pd.DataFrame(cal_rows).set_index("image_path").to_csv(project_dir / "calibrations.csv")

    project_yaml = {
        "project_name": f"qdmouse4m_release_refonly_{split}",
        "project_type": "multiview",
        "label_file_key": "CollectedData",
        "labeled_data_dir": "labeled-data",
        "video_dir": "videos",
        "session_name": session_name,
        "views": VIEWS,
    }
    (project_dir / "project.yaml").write_text(yaml.safe_dump(project_yaml, sort_keys=False))
    write_json(
        project_dir / "project_summary.json",
        {
            "source": "qdmouse4m_benchmark.prepare_lp3d",
            "subset_name": split,
            "session_name": session_name,
            "view_names": VIEWS,
            "keypoints": KEYPOINTS,
            "num_cameras": len(VIEWS),
            "num_keypoints": len(KEYPOINTS),
            "total_timepoints": int(len(split_df)),
            "total_rows_per_view": int(len(split_df)),
            "tile_size": int(tile_size),
            "image_source_mode": "SAM3 dynamic-bbox refonly crops; grayscale replicated to RGB",
            "label_policy": (
                f"2D labels with confidence >= {confidence_threshold}; "
                "outside-crop or invalid-3D keypoints set missing"
            ),
            "csv_files": {"train_eval": [f"CollectedData_{view}.csv" for view in VIEWS]},
            "bbox_files": {"train_eval": [f"bboxes_CollectedData_{view}.csv" for view in VIEWS]},
            "calibration_file": "calibrations/qdmouse_release_camera_calibration.toml",
            "calibration_index_files": {"train_eval": "calibrations.csv"},
        },
    )
    return project_dir


def skeleton():
    import sleap_io as sio

    sk = sio.Skeleton(KEYPOINTS, name="qdmouse")
    for a, b in SKELETON_EDGES:
        sk.add_edge(a, b)
    return sk


def export_sleap_labels(sleap_dir: Path, split: str) -> None:
    import sleap_io as sio

    sk = skeleton()
    per_view_labels = {}
    frame_index = pd.read_csv(sleap_dir / "sleap_frame_index.csv")
    for view in VIEWS:
        points = np.load(sleap_dir / f"points_{view}.npy").astype(np.float32)
        rows = frame_index[frame_index["view"].astype(str) == view].reset_index(drop=True)
        video = sio.Video.from_filename(str((sleap_dir / "images" / view).resolve()), keep_open=False)
        labeled_frames = []
        for row_i, row in rows.iterrows():
            inst = sio.Instance.from_numpy(points[row_i], skeleton=sk)
            labeled_frames.append(
                sio.LabeledFrame(video=video, frame_idx=int(row.pool_index), instances=[inst])
            )
        labels = sio.Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[sk])
        labels.provenance.update({"source": "qdmouse4m_benchmark", "split": split, "view": view})
        out_path = sleap_dir / f"labels_{view}.slp"
        labels.save(out_path)
        per_view_labels[view] = labels

    all_labels = sio.Labels(
        labeled_frames=[lf for view in VIEWS for lf in per_view_labels[view].labeled_frames],
        videos=[video for view in VIEWS for video in per_view_labels[view].videos],
        skeletons=[sk],
    )
    all_labels.provenance.update({"source": "qdmouse4m_benchmark", "split": split, "views": VIEWS})
    all_labels.save(sleap_dir / "labels_all_views.slp")


def prepare_sleap(
    *,
    root: Path,
    dataset_root: Path,
    split: str,
    out_dir: Path | None = None,
    confidence_threshold: float = 0.5,
    copy_images: bool = False,
    write_slp: bool = True,
) -> Path:
    split_df = split_manifest(dataset_root, split)
    sleap_dir = out_dir or (dataset_root / "sleap" / split)
    sleap_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    first_image = image_pool_path(dataset_root, VIEWS[0], int(split_df.iloc[0].pool_index))
    tile_w, tile_h = image_size(first_image)
    if tile_w != tile_h:
        raise ValueError(f"Expected square dynamic crop images; got {tile_w}x{tile_h}: {first_image}")
    tile_size = float(tile_w)

    for view_i, view in enumerate(VIEWS):
        img_dir = sleap_dir / "images" / view
        img_dir.mkdir(parents=True, exist_ok=True)
        points = np.full((len(split_df), len(KEYPOINTS), 2), np.nan, dtype=np.float32)
        confs = np.zeros((len(split_df), len(KEYPOINTS)), dtype=np.float32)
        image_paths = []
        for row_i, row in enumerate(split_df.itertuples(index=False)):
            pool_index = int(row.pool_index)
            src_img = image_pool_path(dataset_root, view, pool_index)
            dst_img = img_dir / f"img{pool_index:08d}.png"
            symlink_or_copy(src_img, dst_img, copy=copy_images)
            rel_img = Path("images") / view / dst_img.name
            xy, conf, _bbox = crop_labels_for_row(
                release_root=root,
                session_id=str(row.session_id),
                frame_idx=int(row.frame_idx),
                view=view,
                view_index=view_i,
                tile_size=tile_size,
                confidence_threshold=confidence_threshold,
            )
            points[row_i] = xy
            confs[row_i] = conf
            image_paths.append(rel_img.as_posix())
            frame_rows.append(
                {
                    "subset": split,
                    "view": view,
                    "pool_index": pool_index,
                    "session_id": str(row.session_id),
                    "frame_idx": int(row.frame_idx),
                    "image_path": rel_img.as_posix(),
                }
            )
        np.save(sleap_dir / f"points_{view}.npy", points)
        np.save(sleap_dir / f"confidence_{view}.npy", confs)
        (sleap_dir / f"image_paths_{view}.txt").write_text("\n".join(image_paths) + "\n")

    pd.DataFrame(frame_rows).to_csv(sleap_dir / "sleap_frame_index.csv", index=False)
    write_json(
        sleap_dir / "sleap_export_manifest.json",
        {
            "subset_name": split,
            "views": VIEWS,
            "keypoints": KEYPOINTS,
            "skeleton_edges": SKELETON_EDGES,
            "n_timepoints": int(len(split_df)),
            "n_labeled_frames_all_views": int(len(split_df) * len(VIEWS)),
            "image_mode": "SAM3 dynamic-bbox refonly crops; grayscale replicated to RGB",
            "tile_size": int(tile_size),
            "label_policy": f"confidence >= {confidence_threshold}, outside-crop labels missing",
        },
    )
    if write_slp:
        export_sleap_labels(sleap_dir, split)
    return sleap_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("lp3d", "sleap"):
        s = sub.add_parser(name)
        s.add_argument("--release-root", type=Path, default=None)
        s.add_argument("--dataset-root", required=True, help="Relative to release root or absolute.")
        s.add_argument("--split", action="append", required=True)
        s.add_argument("--out-root", type=Path, default=None)
        s.add_argument("--confidence-threshold", type=float, default=0.5)
        s.add_argument("--copy-images", action="store_true")
    sub.choices["sleap"].add_argument("--no-slp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = release_root(args.release_root)
    dataset_root = resolve_under(root, args.dataset_root)
    outputs = []
    for split in args.split:
        out_dir = None if args.out_root is None else args.out_root / split
        if args.command == "lp3d":
            outputs.append(
                str(
                    prepare_lp3d(
                        root=root,
                        dataset_root=dataset_root,
                        split=split,
                        out_dir=out_dir,
                        confidence_threshold=args.confidence_threshold,
                        copy_images=args.copy_images,
                    )
                )
            )
        else:
            outputs.append(
                str(
                    prepare_sleap(
                        root=root,
                        dataset_root=dataset_root,
                        split=split,
                        out_dir=out_dir,
                        confidence_threshold=args.confidence_threshold,
                        copy_images=args.copy_images,
                        write_slp=not args.no_slp,
                    )
                )
            )
    print(json.dumps({"prepared": outputs}, indent=2))


if __name__ == "__main__":
    main()

