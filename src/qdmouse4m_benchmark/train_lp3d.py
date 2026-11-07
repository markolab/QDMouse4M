"""Train a Lightning Pose 3D multiview transformer on a prepared LP3D project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from .constants import LP3D_BACKBONES
from .lp3d_tools import allow_truncated_pngs, install_dinov3_forward_patch, install_resume_patch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=sorted(LP3D_BACKBONES), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=160)
    p.add_argument("--min-epochs", type=int, default=80)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--check-val-every-n-epoch", type=int, default=5)
    p.add_argument("--train-prob", type=float, default=0.8)
    p.add_argument("--val-prob", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--unfreezing-epoch", type=int, default=20)
    p.add_argument("--patch-mask-init-epoch", type=int, default=20)
    p.add_argument("--patch-mask-final-epoch", type=int, default=120)
    p.add_argument("--patch-mask-final-ratio", type=float, default=0.5)
    p.add_argument("--imgaug-3d", action="store_true", help="Enable LP 3D augmentation.")
    p.add_argument("--reprojection-loss-weight", type=float, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size % 128 != 0:
        raise ValueError("Use LP image sizes that are multiples of 128, e.g. 256.")
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_truncated_pngs()
    install_dinov3_forward_patch()
    if args.resume:
        install_resume_patch(output_dir)

    from lightning_pose.train import train

    summary = json.loads((project_dir / "project_summary.json").read_text())
    views = list(summary["view_names"])
    keypoints = list(summary["keypoints"])
    csv_files = summary.get("csv_files", {}).get("train_eval") or [
        f"CollectedData_{view}.csv" for view in views
    ]
    bbox_files = summary.get("bbox_files", {}).get("train_eval") or [
        f"bboxes_CollectedData_{view}.csv" for view in views
    ]
    calibration = summary.get("calibration_index_files", {}).get("train_eval") or "calibrations.csv"
    model_name = args.model_name or f"qdmouse4m_refonly_{args.backbone}"

    cfg = OmegaConf.create(
        {
            "data": {
                "image_resize_dims": {"height": args.image_size, "width": args.image_size},
                "data_dir": str(project_dir),
                "video_dir": str(project_dir / "videos"),
                "csv_file": csv_files,
                "bbox_file": bbox_files,
                "camera_params_file": str(project_dir / calibration),
                "view_names": views,
                "num_keypoints": len(keypoints),
                "keypoint_names": keypoints,
                "mirrored_column_matches": None,
                "columns_for_singleview_pca": None,
                "downsample_factor": 2,
            },
            "training": {
                "imgaug": "dlc",
                "imgaug_3d": bool(args.imgaug_3d),
                "train_batch_size": args.batch_size,
                "val_batch_size": args.batch_size,
                "test_batch_size": args.batch_size,
                "train_prob": args.train_prob,
                "val_prob": args.val_prob,
                "train_frames": 1.0,
                "num_gpus": 1,
                "num_workers": args.num_workers,
                "early_stop_patience": args.early_stop_patience,
                "unfreezing_epoch": args.unfreezing_epoch,
                "min_epochs": args.min_epochs,
                "max_epochs": args.max_epochs,
                "check_val_every_n_epoch": args.check_val_every_n_epoch,
                "log_every_n_steps": 10,
                "rng_seed_data_pt": args.seed,
                "rng_seed_model_pt": args.seed,
                "optimizer": "AdamW",
                "optimizer_params": {"learning_rate": args.learning_rate},
                "lr_scheduler": "multisteplr",
                "lr_scheduler_params": {
                    "multisteplr": {"milestones": [80, 120, 140], "gamma": 0.5}
                },
                "uniform_heatmaps_for_nan_keypoints": True,
                "patch_mask": {
                    "init_epoch": args.patch_mask_init_epoch,
                    "final_epoch": min(args.patch_mask_final_epoch, args.max_epochs),
                    "init_ratio": 0.0,
                    "final_ratio": args.patch_mask_final_ratio,
                },
            },
            "model": {
                "losses_to_use": [],
                "backbone": LP3D_BACKBONES[args.backbone],
                "model_type": "heatmap_multiview_transformer",
                "heatmap_loss_type": "mse",
                "model_name": model_name,
                "checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            },
            "losses": {
                "pca_multiview": {"log_weight": 5.0, "components_to_keep": 3, "epsilon": None},
                "pca_singleview": {"log_weight": 5.0, "components_to_keep": 0.99, "epsilon": None},
                "temporal": {"log_weight": 5.0, "epsilon": 20.0, "prob_threshold": 0.05},
                "supervised_reprojection_heatmap_mse": {
                    "log_weight": args.reprojection_loss_weight,
                },
            },
            "eval": {
                "predict_vids_after_training": False,
                "test_videos_directory": str(project_dir / "videos"),
                "save_vids_after_training": False,
                "confidence_thresh_for_vid": 0.9,
            },
            "callbacks": {
                "anneal_weight": {
                    "attr_name": "total_unsupervised_importance",
                    "init_val": 0.0,
                    "increase_factor": 0.02,
                    "final_val": 1.0,
                    "freeze_until_epoch": 40,
                }
            },
        }
    )
    OmegaConf.save(cfg, output_dir / "config.yaml")
    train(cfg, model_dir=output_dir, skip_evaluation=True)


if __name__ == "__main__":
    main()

