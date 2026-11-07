"""Run SLEAP-NN inference and benchmark evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, default=None)
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--model-dir", type=Path, required=True, help="SLEAP model parent or leaf.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics-only", action="store_true")
    return p.parse_args()


def model_leaf(model_dir: Path) -> Path:
    if (model_dir / "best.ckpt").exists():
        return model_dir
    leaves = sorted(p for p in model_dir.glob("*.single_instance.n=*") if (p / "best.ckpt").exists())
    if not leaves:
        raise FileNotFoundError(f"Could not find SLEAP best.ckpt under {model_dir}")
    return leaves[-1]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_slp = args.eval_root / "sleap" / args.split / "labels_all_views.slp"
    pred_slp = args.out_dir / "predictions.slp"
    pred_npz = args.out_dir / "predictions_crop_space.npz"
    metrics_json = args.out_dir / "metrics_summary.json"
    metrics_npz = args.out_dir / "metrics_arrays.npz"

    if not args.metrics_only and not pred_slp.exists():
        subprocess.check_call(
            [
                "sleap-nn",
                "track",
                "-i",
                str(test_slp),
                "-m",
                str(model_leaf(args.model_dir)),
                "-o",
                str(pred_slp),
                "--only_labeled_frames",
                "-d",
                args.device,
                "-b",
                str(args.batch_size),
            ]
        )
    if not pred_npz.exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "qdmouse4m_benchmark.convert_sleap_predictions",
                "--eval-root",
                str(args.eval_root),
                "--split",
                args.split,
                "--pred-slp",
                str(pred_slp),
                "--out-npz",
                str(pred_npz),
            ]
        )
    cmd = [
        sys.executable,
        "-m",
        "qdmouse4m_benchmark.evaluate",
        "--eval-root",
        str(args.eval_root),
        "--split",
        args.split,
        "--pred-npz",
        str(pred_npz),
        "--out-json",
        str(metrics_json),
        "--out-npz",
        str(metrics_npz),
        "--model-kind",
        "sleap",
        "--model-name",
        args.model_dir.name,
    ]
    if args.release_root is not None:
        cmd += ["--release-root", str(args.release_root)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

