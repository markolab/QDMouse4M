"""Run one SLEAP-NN all-view single-instance training job."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sleap-dir", type=Path, required=True, help="Folder containing labels_all_views.slp.")
    p.add_argument("--template", type=Path, required=True, help="SLEAP-NN YAML template.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=["unet", "convnext", "swint"], default="unet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.sleap_dir / "labels_all_views.slp"
    if not labels.exists():
        raise FileNotFoundError(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    template_stem = args.template.stem
    run_dir = args.output_dir / "run"
    run_dir.mkdir(exist_ok=True)
    config_dst = run_dir / args.template.name
    if not config_dst.exists():
        config_dst.write_bytes(args.template.read_bytes())

    # Backbone edits are intentionally expressed as CLI overrides to avoid locking
    # the repository to one SLEAP-NN YAML schema.
    overrides = [
        f"data_config.train_labels_path=['{labels}']",
        f"data_config.validation_fraction={args.val_fraction}",
        f"data_config.preprocessing.max_height={args.image_size}",
        f"data_config.preprocessing.max_width={args.image_size}",
        "data_config.preprocessing.crop_size=null",
        f"trainer_config.ckpt_dir={ckpt_dir}",
        f"+trainer_config.seed={args.seed}",
        "model_config.head_configs.single_instance.confmaps.sigma=3.5",
        "model_config.head_configs.single_instance.confmaps.output_stride=4",
    ]
    if args.backbone == "convnext":
        overrides += [
            "model_config.backbone_config.unet=null",
            "model_config.backbone_config.swint=null",
            "model_config.backbone_config.convnext.model_type=tiny",
            "model_config.backbone_config.convnext.pre_trained_weights=null",
            "model_config.backbone_config.convnext.in_channels=1",
            "model_config.backbone_config.convnext.output_stride=4",
        ]
    elif args.backbone == "swint":
        overrides += [
            "model_config.backbone_config.unet=null",
            "model_config.backbone_config.convnext=null",
            "model_config.backbone_config.swint.model_type=tiny",
            "model_config.backbone_config.swint.pre_trained_weights=null",
            "model_config.backbone_config.swint.in_channels=1",
            "model_config.backbone_config.swint.output_stride=4",
        ]

    cmd = ["sleap-nn", "train", "-c", template_stem, "-d", str(run_dir), *overrides]
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.check_call(cmd)
        (args.output_dir / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()

