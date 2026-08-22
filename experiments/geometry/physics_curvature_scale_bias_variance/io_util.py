"""Atomic IO. Refuses writes into preserved trees."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PRESERVED, ExpConfig


def platonic_root() -> Path:
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home() / "platonic-universe"
    if home.is_dir():
        return home.resolve()
    here = Path(__file__).resolve()
    for cand in [here.parents[i] for i in range(2, min(8, len(here.parents)))]:
        if (cand / "data_hf").is_dir() or (cand / "outputs" / "geometry").is_dir():
            return cand.resolve()
    return Path.cwd().resolve()


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def assert_not_preserved(out: Path, root: Path) -> None:
    resolved = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if resolved == pres or pres in resolved.parents:
            raise RuntimeError(f"refusing to write into preserved tree {rel}")


def atomic_replace(tmp: Path, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        ts = time.strftime("%Y%m%dT%H%M%S")
        dest.rename(dest.with_name(f"{dest.stem}.superseded.{ts}{dest.suffix}"))
    tmp.replace(dest)


def write_json(path: Path, obj: Any, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    atomic_replace(tmp, path, force=force)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    atomic_replace(tmp, path, force=force)


def p_mc(b: int, B: int) -> float:
    return float(b + 1) / float(B + 1)


def fisher_z(rho: float) -> float:
    r = float(np.clip(rho, -0.999999, 0.999999))
    return float(np.arctanh(r))


platonic_root = platonic_root
resolve_path = resolve_path
assert_not_preserved = assert_not_preserved
