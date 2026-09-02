"""IO helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def platonic_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "experiments" / "geometry").is_dir():
        return cwd
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "experiments" / "geometry").is_dir() and (
            (p / "outputs" / "geometry").is_dir() or (p / "pyproject.toml").exists()
        ):
            return p
    home = Path.home() / "platonic-universe"
    if home.is_dir():
        return home
    return cwd


def resolve_path(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def assert_not_preserved(out: Path, root: Path) -> None:
    from .config import PRESERVED

    rel = str(out.relative_to(root)) if out.is_relative_to(root) else str(out)
    for p in PRESERVED:
        if rel.rstrip("/") == p.rstrip("/") or rel.startswith(p.rstrip("/") + "/"):
            raise RuntimeError(f"refusing to write into preserved path {p}")


def file_sha256(path: Path, maxlen: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if maxlen <= 0:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(maxlen))
    return h.hexdigest()[:16]


def write_json(path: Path, obj: Any, *, force: bool = False) -> None:
    if path.exists() and not force:
        bak = path.with_suffix(path.suffix + f".superseded")
        path.replace(bak) if False else None  # keep latest
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_df(path: Path, df: pd.DataFrame, *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def p_mc(b_count: int, B: int) -> float:
    return (b_count + 1) / (B + 1)
