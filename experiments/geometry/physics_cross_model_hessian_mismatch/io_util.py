"""Atomic IO; refuse writes into preserved trees."""

from __future__ import annotations

from pathlib import Path

from geometry.physics_task_aligned_curvature.io_util import (  # noqa: F401
    file_meta,
    p_mc,
    peak_rss_mb,
    platonic_root,
    resolve_path,
    write_df,
    write_json,
    write_text,
)

from .config import PRESERVED


def assert_not_preserved(out: Path, root: Path) -> None:
    resolved = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if resolved == pres or pres in resolved.parents:
            raise RuntimeError(f"refusing to write into preserved tree {rel}")
