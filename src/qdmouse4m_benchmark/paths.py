"""Path helpers for locating the release root."""

from __future__ import annotations

import os
from pathlib import Path


def release_root(raw: str | Path | None = None) -> Path:
    """Return the release root from a CLI argument or environment variable."""
    value = raw or os.environ.get("QDMOUSE4M_RELEASE_ROOT")
    if not value:
        raise RuntimeError(
            "Set QDMOUSE4M_RELEASE_ROOT or pass --release-root. "
            "This package expects the dataset root to be provided at runtime."
        )
    return Path(value).expanduser().resolve()


def resolve_under(root: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else root / path


def relpath_for_csv(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def symlink_or_copy(src: Path, dst: Path, *, copy: bool = False) -> None:
    """Create a relative symlink by default; copy only when explicitly requested."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        import shutil

        shutil.copy2(src, dst)
        return
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    os.symlink(rel, dst)
