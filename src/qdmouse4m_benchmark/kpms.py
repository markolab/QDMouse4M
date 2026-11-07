"""Keypoint-MoSeq dataset prep, model application, and syllable export.

This module is intentionally scheduler- and cluster-agnostic. It assumes the
public QDMouse4M release layout and takes trained Keypoint-MoSeq model paths as
user-provided arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from .constants import KEYPOINTS

FPS_DEFAULT = 40.724

KPMS_SKELETON = [
    ["ear_L", "ear_R"],
    ["ear_L", "back_top"],
    ["ear_R", "back_top"],
    ["back_top", "back_middle"],
    ["back_middle", "back_bottom"],
    ["back_bottom", "tail_base"],
    ["tail_base", "tail_middle"],
    ["tail_middle", "tail_tip"],
    ["forepaw_L", "back_middle"],
    ["forepaw_R", "back_middle"],
    ["hindpaw_L", "back_bottom"],
    ["hindpaw_R", "back_bottom"],
]

ANTERIOR_BODYPARTS = ["ear_L", "ear_R", "forepaw_L", "forepaw_R", "back_top", "back_middle"]
POSTERIOR_BODYPARTS = [
    "back_bottom",
    "tail_base",
    "tail_middle",
    "tail_tip",
    "hindpaw_L",
    "hindpaw_R",
]


def release_root_from_arg(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    env_value = os.environ.get("QDMOUSE4M_RELEASE_ROOT")
    if not env_value:
        raise SystemExit("Set QDMOUSE4M_RELEASE_ROOT or pass --release-root.")
    return Path(env_value).expanduser().resolve()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Install PyYAML or run in the project environment to load KPMS YAML configs."
        ) from exc
    with path.expanduser().open() as f:
        return yaml.safe_load(f) or {}


def model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("model") or {})


def dataset_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("dataset") or {})


def scalar_string(value: np.ndarray | Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr)


def session_id_from_key(key: str) -> str:
    match = re.search(r"(\d{12,14}-\d+)", key)
    return match.group(1) if match else key.replace("session_", "")


def load_session_order(release_root: Path, session_ids_file: Path | None) -> list[str]:
    if session_ids_file:
        ids = []
        for raw in session_ids_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                ids.append(line.replace("session_", ""))
        return ids

    sessions_csv = release_root / "sessions.csv"
    if sessions_csv.is_file():
        with sessions_csv.open(newline="") as f:
            return [row["session_id"] for row in csv.DictReader(f)]

    return sorted(
        p.name.replace("session_", "") for p in (release_root / "sessions").glob("session_*")
    )


def keypoint_annotations_path(release_root: Path, session_id: str) -> Path:
    return (
        release_root
        / "sessions"
        / f"session_{session_id}"
        / "labels"
        / "keypoint_annotations.npz"
    )


def invalid_jump_mask(z: Any, shape: tuple[int, int]) -> np.ndarray:
    if "invalid_jump_timepoint" not in z.files:
        return np.zeros(shape, dtype=bool)
    raw = np.asarray(z["invalid_jump_timepoint"]).astype(bool)
    if raw.shape == shape:
        return raw
    if raw.shape == (shape[0],):
        return np.repeat(raw[:, None], shape[1], axis=1)
    raise RuntimeError(f"Unexpected invalid_jump_timepoint shape {raw.shape}; expected {shape}.")


def fill_nan_time(coords: np.ndarray) -> np.ndarray:
    out = coords.astype(np.float64, copy=True)
    t = np.arange(out.shape[0])
    flat = out.reshape(out.shape[0], -1)
    for j in range(flat.shape[1]):
        finite = np.isfinite(flat[:, j])
        if finite.all():
            continue
        if finite.any():
            flat[:, j] = np.interp(t, t[finite], flat[finite, j])
    return out


def peak_body_speed_cm_s(coords: np.ndarray, keypoints: list[str], fps: float) -> float:
    if "back_middle" not in keypoints or coords.shape[0] < 2:
        return float("nan")
    filled = fill_nan_time(coords)
    bm = keypoints.index("back_middle")
    speed = np.linalg.norm(np.diff(filled[:, bm, :], axis=0), axis=1) * float(fps) / 10.0
    return float(np.nanmax(np.r_[0.0, speed]))


def downsample_dict(data: dict[str, np.ndarray], rate: int) -> dict[str, np.ndarray]:
    if rate <= 1:
        return {k: np.asarray(v) for k, v in data.items()}
    return {k: np.asarray(v)[::rate].copy() for k, v in data.items()}


def prepare_dataset(args: argparse.Namespace, config: dict[str, Any]) -> None:
    release_root = release_root_from_arg(args.release_root)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_session_dir = out_dir / "per_session"
    per_session_dir.mkdir(parents=True, exist_ok=True)

    dcfg = dataset_cfg(config)
    expected_sessions = args.expected_sessions
    if expected_sessions is None:
        expected_sessions = int(dcfg.get("expected_sessions") or 0)
    fps = float(args.fps if args.fps is not None else dcfg.get("fps", FPS_DEFAULT))
    mask_invalid = bool(args.mask_invalid)
    mask_invalid_jump = bool(args.mask_invalid_jump_timepoint)

    session_ids = load_session_order(release_root, args.session_ids_file)
    if expected_sessions and len(session_ids) != expected_sessions:
        raise RuntimeError(f"Expected {expected_sessions} sessions, found {len(session_ids)}.")

    coordinates: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    bodyparts_ref: list[str] | None = None

    for session_id in session_ids:
        session_key = f"session_{session_id}"
        npz_path = keypoint_annotations_path(release_root, session_id)
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path, allow_pickle=False)
        keypoints = [str(x) for x in z["keypoint_names"].astype(str).tolist()]
        if keypoints != KEYPOINTS:
            raise RuntimeError(f"Unexpected keypoint order in {npz_path}: {keypoints}")
        if bodyparts_ref is None:
            bodyparts_ref = keypoints
        elif keypoints != bodyparts_ref:
            raise RuntimeError(f"Keypoint mismatch in {session_id}: {keypoints} vs {bodyparts_ref}")

        coords = z["keypoints_3d"].astype(np.float32, copy=True)
        if "keypoint_confidence_3d" in z.files:
            conf = np.nan_to_num(z["keypoint_confidence_3d"].astype(np.float32, copy=True), nan=0.0)
        else:
            conf = np.isfinite(coords).all(axis=2).astype(np.float32)
        conf = np.clip(conf, 0.0, 1.0)
        if "valid_3d" in z.files:
            valid_3d = z["valid_3d"].astype(bool, copy=False)
        else:
            valid_3d = np.isfinite(coords).all(axis=2)
        if mask_invalid_jump:
            valid_3d = valid_3d & ~invalid_jump_mask(z, valid_3d.shape)
        if args.max_frames_per_session > 0:
            n = int(args.max_frames_per_session)
            coords = coords[:n]
            conf = conf[:n]
            valid_3d = valid_3d[:n]

        finite_before_mask = np.isfinite(coords).all(axis=2)
        if mask_invalid:
            coords[~valid_3d] = np.nan
            conf[~valid_3d] = 0.0
        conf[~np.isfinite(coords).all(axis=2)] = 0.0

        finite_after = np.isfinite(coords).all(axis=2)
        by_keypoint = {}
        for i, kp in enumerate(keypoints):
            kp_conf = conf[:, i]
            by_keypoint[kp] = {
                "finite_fraction": float(np.mean(finite_after[:, i])),
                "valid_3d_fraction": float(np.mean(valid_3d[:, i])),
                "median_confidence": (
                    float(np.nanmedian(kp_conf[kp_conf > 0])) if np.any(kp_conf > 0) else 0.0
                ),
            }
            rows.append({"session_id": session_id, "keypoint": kp, **by_keypoint[kp]})

        if "frame_index" in z.files:
            frame_index = z["frame_index"][: coords.shape[0]]
        else:
            frame_index = np.arange(coords.shape[0])
        np.savez_compressed(
            per_session_dir / f"{session_key}_release3d_kpms_input.npz",
            frame_index=frame_index,
            keypoints=np.asarray(keypoints),
            coordinates=coords,
            confidences=conf,
            valid_3d=valid_3d,
            finite_before_mask=finite_before_mask[: coords.shape[0]],
            release_npz=np.asarray(str(npz_path.relative_to(release_root))),
        )
        coordinates[session_key] = coords
        confidences[session_key] = conf
        summaries.append(
            {
                "session": session_key,
                "session_id": session_id,
                "frames": int(coords.shape[0]),
                "finite_fraction_all": float(np.mean(finite_after)),
                "valid_3d_fraction_all": float(np.mean(valid_3d)),
                "frame_fraction_all_keypoints_finite": float(np.mean(np.all(finite_after, axis=1))),
                "peak_body_speed_cm_s": peak_body_speed_cm_s(coords, keypoints, fps),
                "by_keypoint": by_keypoint,
            }
        )

    assert bodyparts_ref is not None
    common = {
        "source": "qdmouse4m kpms prepare",
        "release_root": str(release_root),
        "dataset_variant": "release3d_maskinvalid" if mask_invalid else "release3d_rawvalidity",
        "mask_invalid": mask_invalid,
        "mask_invalid_jump_timepoint": mask_invalid_jump,
        "bodyparts": bodyparts_ref,
        "sessions": list(coordinates.keys()),
        "fps": fps,
        "config": config,
    }
    for name in ("kpms_dataset_clean.pkl", "kpms_dataset_release3d.pkl"):
        with (out_dir / name).open("wb") as f:
            pickle.dump({**common, "coordinates": coordinates, "confidences": confidences}, f)

    qa = {
        **common,
        "n_sessions": len(summaries),
        "total_frames": int(sum(item["frames"] for item in summaries)),
        "peak_body_speed_cm_s_max": float(max(item["peak_body_speed_cm_s"] for item in summaries)),
        "summaries": summaries,
    }
    (out_dir / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n")
    with (out_dir / "qa_by_keypoint.csv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["session_id", "keypoint"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "dataset_pkl": str(out_dir / "kpms_dataset_clean.pkl"),
                "qa_summary": str(out_dir / "qa_summary.json"),
            },
            indent=2,
        )
    )


def import_kpms_runtime() -> tuple[Any, Any, Any]:
    try:
        import jax  # type: ignore
        import keypoint_moseq as kpms  # type: ignore
        from jax_moseq.utils import convert_data_precision, set_mixed_map_iters  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Keypoint-MoSeq runtime is not installed. Install keypoint-moseq, "
            "jax, and jax-moseq in a CUDA/JAX-compatible environment, then retry."
        ) from exc
    set_mixed_map_iters(32)
    return jax, kpms, convert_data_precision


def infer_downsample_rate(
    project_dir: Path,
    model_name: str,
    explicit: int,
    config: dict[str, Any],
) -> int:
    if explicit > 0:
        return int(explicit)
    summary_path = project_dir / model_name / "run_summary.json"
    if summary_path.exists():
        return max(1, int(json.loads(summary_path.read_text()).get("downsample_rate", 1)))
    return max(1, int(model_cfg(config).get("downsample_rate", 1)))


def apply_model(args: argparse.Namespace, config: dict[str, Any]) -> None:
    jax, kpms, convert_data_precision = import_kpms_runtime()
    dataset = pickle.loads(args.dataset_pkl.expanduser().read_bytes())
    project_dir = args.project_dir.expanduser().resolve()
    output_h5 = args.output_h5.expanduser().resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    downsample_rate = infer_downsample_rate(
        project_dir,
        args.model_name,
        args.downsample_rate,
        config,
    )
    coordinates = downsample_dict(dataset["coordinates"], downsample_rate)
    confidences = downsample_dict(dataset["confidences"], downsample_rate)
    cfg = kpms.load_config(str(project_dir))
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    data = convert_data_precision(data)
    model, _train_data, _train_metadata, iteration = kpms.load_checkpoint(
        project_dir=str(project_dir),
        model_name=args.model_name,
    )
    apply_kwargs: dict[str, Any] = {"error_estimator": cfg.get("error_estimator")}
    for key in ("anterior_idxs", "posterior_idxs"):
        if key in cfg:
            apply_kwargs[key] = cfg[key]
    print("[kpms] devices:", jax.devices(), flush=True)
    print(
        json.dumps(
            {
                "project_dir": str(project_dir),
                "model_name": args.model_name,
                "checkpoint_iteration": int(iteration),
                "dataset_pkl": str(args.dataset_pkl),
                "downsample_rate": int(downsample_rate),
                "num_iters": int(args.num_iters),
                "output_h5": str(output_h5),
            },
            indent=2,
        ),
        flush=True,
    )
    kpms.apply_model(
        model,
        data,
        metadata,
        project_dir=str(project_dir),
        model_name=args.model_name,
        num_iters=int(args.num_iters),
        save_results=True,
        results_path=str(output_h5),
        parallel_message_passing=args.parallel_message_passing,
        **apply_kwargs,
    )
    print(f"[done] wrote {output_h5}", flush=True)


def load_run_metadata(results_h5: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"results_h5": str(results_h5)}
    project_dir = results_h5.parent.parent if results_h5.name == "results.h5" else results_h5.parent
    split_path = project_dir / "train_test_split.json"
    if split_path.exists():
        metadata["train_test_split"] = json.loads(split_path.read_text())
    run_summary_path = results_h5.parent / "run_summary.json"
    if run_summary_path.exists():
        metadata["run_summary"] = json.loads(run_summary_path.read_text())
    return metadata


def expand_syllables(
    z_downsampled: np.ndarray,
    total_frames: int,
    downsample_rate: int,
    fill_neighbors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    syllable = np.full((total_frames,), -1, dtype=np.int32)
    known = np.zeros((total_frames,), dtype=bool)
    for i, value in enumerate(np.asarray(z_downsampled, dtype=np.int32)):
        start = i * downsample_rate
        if start >= total_frames:
            break
        stop = min(total_frames, start + (downsample_rate if fill_neighbors else 1))
        syllable[start:stop] = int(value)
        known[start:stop] = True
    return syllable, known


def export_syllables(args: argparse.Namespace, config: dict[str, Any]) -> None:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install h5py to export Keypoint-MoSeq syllable bundles.") from exc

    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    dataset = pickle.loads(args.dataset_pkl.expanduser().read_bytes())
    coordinates: dict[str, np.ndarray] = dataset["coordinates"]
    confidences: dict[str, np.ndarray] = dataset["confidences"]
    bodyparts = np.asarray([str(x) for x in dataset["bodyparts"]])
    fps = float(dataset.get("fps", FPS_DEFAULT))
    metadata = load_run_metadata(args.results_h5.expanduser())
    run_info = metadata.get("run_summary") or model_cfg(config)
    downsample_rate = args.downsample_rate
    if downsample_rate <= 0:
        downsample_rate = max(
            1,
            int(run_info.get("downsample_rate", model_cfg(config).get("downsample_rate", 1))),
        )
    requested = {s if s.startswith("session_") else f"session_{s}" for s in args.session}
    annotations = {str(k): str(v) for k, v in (config.get("syllable_annotations") or {}).items()}

    manifest = {
        "results_h5": str(args.results_h5),
        "dataset_pkl": str(args.dataset_pkl),
        "output_dir": str(args.output_dir),
        "downsample_rate": int(downsample_rate),
        "fps": fps,
        "fill_downsample_neighbors": bool(args.fill_downsample_neighbors),
        "syllable_annotations": annotations,
        "exports": [],
        **metadata,
    }
    with h5py.File(args.results_h5.expanduser(), "r") as h5:
        for session_key in sorted(h5.keys()):
            if requested and session_key not in requested:
                continue
            if session_key not in coordinates:
                print(f"[skip] {session_key}: missing from dataset pickle", flush=True)
                continue
            z = np.asarray(h5[f"{session_key}/syllable"], dtype=np.int32)
            coords = np.asarray(coordinates[session_key], dtype=np.float32)
            conf = np.asarray(confidences[session_key], dtype=np.float32)
            syllable, known = expand_syllables(
                z,
                total_frames=int(coords.shape[0]),
                downsample_rate=int(downsample_rate),
                fill_neighbors=bool(args.fill_downsample_neighbors),
            )
            out_path = args.output_dir / f"{session_key}_kpms_syllables.npz"
            np.savez_compressed(
                out_path,
                session_key=np.asarray(session_key),
                session_id=np.asarray(session_id_from_key(session_key)),
                syllable=syllable,
                known=known,
                coordinates=coords,
                confidences=conf,
                bodyparts=bodyparts,
                fps=np.asarray(fps, dtype=np.float32),
                downsample_rate=np.asarray(downsample_rate, dtype=np.int32),
                source_results_h5=np.asarray(str(args.results_h5)),
                run_name=np.asarray(run_info.get("run_name", "")),
                model_name=np.asarray(run_info.get("model_name", "")),
                num_states=np.asarray(run_info.get("num_states", -1), dtype=np.int32),
                latent_dim=np.asarray(run_info.get("latent_dim", -1), dtype=np.int32),
                kappa_full=np.asarray(run_info.get("kappa_full", np.nan), dtype=np.float64),
            )
            item = {
                "session_key": session_key,
                "session_id": session_id_from_key(session_key),
                "path": str(out_path),
                "total_frames": int(coords.shape[0]),
                "known_frames": int(known.sum()),
                "unique_syllables": [int(x) for x in sorted(np.unique(syllable[known]).tolist())],
            }
            manifest["exports"].append(item)
            print(json.dumps(item), flush=True)

    if annotations:
        (args.output_dir / "syllable_annotations.json").write_text(
            json.dumps(annotations, indent=2) + "\n"
        )
    manifest_path = args.output_dir / "kpms_syllable_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote {manifest_path}", flush=True)


def add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional KPMS YAML config, e.g. configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    prep = subparsers.add_parser("prepare", help="Prepare release 3D trajectories for KPMS.")
    add_common_config_arg(prep)
    prep.add_argument("--release-root", type=Path)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--session-ids-file", type=Path)
    prep.add_argument("--expected-sessions", type=int)
    prep.add_argument("--max-frames-per-session", type=int, default=0)
    prep.add_argument("--fps", type=float)
    prep.add_argument("--mask-invalid", action="store_true", default=True)
    prep.add_argument("--no-mask-invalid", dest="mask_invalid", action="store_false")
    prep.add_argument("--mask-invalid-jump-timepoint", action="store_true", default=True)
    prep.add_argument(
        "--no-mask-invalid-jump-timepoint",
        dest="mask_invalid_jump_timepoint",
        action="store_false",
    )

    apply = subparsers.add_parser(
        "apply",
        help="Apply a trained KPMS checkpoint to a prepared dataset.",
    )
    add_common_config_arg(apply)
    apply.add_argument("--dataset-pkl", type=Path, required=True)
    apply.add_argument("--project-dir", type=Path, required=True)
    apply.add_argument("--model-name", required=True)
    apply.add_argument("--output-h5", type=Path, required=True)
    apply.add_argument("--num-iters", type=int, default=100)
    apply.add_argument("--downsample-rate", type=int, default=0)
    apply.add_argument("--parallel-message-passing", default=None)

    export = subparsers.add_parser(
        "export",
        help="Export per-session full-frame KPMS syllable NPZs.",
    )
    add_common_config_arg(export)
    export.add_argument("--results-h5", type=Path, required=True)
    export.add_argument("--dataset-pkl", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--session", action="append", default=[])
    export.add_argument("--downsample-rate", type=int, default=0)
    export.add_argument("--fill-downsample-neighbors", action="store_true", default=True)
    export.add_argument(
        "--no-fill-downsample-neighbors",
        dest="fill_downsample_neighbors",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.subcommand == "prepare":
        prepare_dataset(args, config)
    elif args.subcommand == "apply":
        apply_model(args, config)
    elif args.subcommand == "export":
        export_syllables(args, config)
    else:
        raise SystemExit(f"Unknown KPMS subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
