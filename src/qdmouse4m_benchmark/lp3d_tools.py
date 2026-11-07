"""LP3D helper patches for robust training and inference."""

from __future__ import annotations

import json
import math
from pathlib import Path


def install_dinov3_forward_patch() -> None:
    """Handle minor DINOv3/transformers API drift seen across LP environments."""
    try:
        from lightning_pose.models.heatmap_tracker_multiview import (
            HeatmapTrackerMultiviewTransformer,
        )
    except Exception:
        return

    if getattr(HeatmapTrackerMultiviewTransformer.forward_vit, "_qdmouse4m_dinov3_patch", False):
        return
    original = HeatmapTrackerMultiviewTransformer.forward_vit

    def forward_vit_universal(self, images):
        import torch

        ve = self.backbone.vision_encoder
        try:
            return original(self, images)
        except (TypeError, AttributeError) as exc:
            msg = str(exc)
            expected = ("output_attentions", "output_hidden_states", "return_dict", "encoder")
            if not any(token in msg for token in expected):
                raise

        try:
            embeddings = ve.embeddings(images, bool_masked_pos=None, interpolate_pos_encoding=True)
        except TypeError:
            embeddings = ve.embeddings(images, bool_masked_pos=None)

        n_prefix = 1 + int(getattr(ve.config, "num_register_tokens", 0) or 0)
        embeddings = embeddings[:, n_prefix:]
        view_batch, n_patches, dim = embeddings.shape
        batch = view_batch // self.num_views
        view_indices = torch.arange(self.num_views, device=embeddings.device).repeat(batch)
        view_embeddings = self.view_embeddings[view_indices].unsqueeze(1).expand(-1, n_patches, -1)
        hidden = (embeddings + view_embeddings).reshape(batch, self.num_views * n_patches, dim)

        if hasattr(ve, "encoder"):
            try:
                output = ve.encoder(
                    hidden,
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=None,
                )
            except TypeError:
                output = ve.encoder(hidden, head_mask=None)
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            norm = getattr(ve, "layernorm", None) or getattr(ve, "norm", None)
            if norm is not None:
                hidden = norm(hidden)
        else:
            pos = ve.rope_embeddings(images) if hasattr(ve, "rope_embeddings") else None
            for layer in ve.layer:
                hidden = layer(hidden, attention_mask=None, position_embeddings=pos)
            hidden = ve.norm(hidden)

        patches_per_view = hidden.shape[1] // self.num_views
        side = math.isqrt(patches_per_view)
        out = hidden.reshape(batch, self.num_views, patches_per_view, dim)
        out = out.reshape(batch, self.num_views, side, side, dim).permute(0, 1, 4, 2, 3)
        return out.reshape(view_batch, dim, side, side)

    forward_vit_universal._qdmouse4m_dinov3_patch = True
    HeatmapTrackerMultiviewTransformer.forward_vit = forward_vit_universal
    print(json.dumps({"step": "installed_dinov3_forward_patch"}), flush=True)


def install_resume_patch(output_dir: Path) -> None:
    """Resume LP training from the newest last.ckpt when present."""
    try:
        import lightning.pytorch as pl
        import lightning_pose.train as lp_train_module
    except Exception:
        return

    if getattr(pl.Trainer.fit, "_qdmouse4m_resume_patch", False):
        return

    original_callbacks = lp_train_module.get_callbacks

    def callbacks_with_last(*args, **kwargs):
        callbacks = list(original_callbacks(*args, **kwargs))
        callbacks.append(
            pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=0, filename="step_{step}")
        )
        return callbacks

    lp_train_module.get_callbacks = callbacks_with_last
    original_fit = pl.Trainer.fit

    def fit_with_resume(self, *args, ckpt_path=None, **kwargs):
        if ckpt_path is None:
            candidates = sorted(output_dir.rglob("last.ckpt"))
            if candidates:
                ckpt_path = str(candidates[-1])
                print(json.dumps({"step": "resume_from_last_ckpt", "ckpt_path": ckpt_path}), flush=True)
        return original_fit(self, *args, ckpt_path=ckpt_path, **kwargs)

    fit_with_resume._qdmouse4m_resume_patch = True
    pl.Trainer.fit = fit_with_resume


def allow_truncated_pngs() -> None:
    try:
        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        pass
