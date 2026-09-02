"""IO helpers; refuse writes into preserved trees."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PRESERVED


def platonic_root() -> Path:
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "experiments" / "geometry").is_dir():
        return cwd.resolve()
    home = Path.home() / "platonic-universe"
    if home.is_dir():
        return home.resolve()
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "experiments" / "geometry").is_dir():
            return cand.resolve()
    return cwd.resolve()


def resolve_path(root: Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p).resolve()


def assert_not_preserved(out: Path, root: Path) -> None:
    r = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if r == pres or pres in r.parents:
            raise RuntimeError(f"refusing write into preserved tree {rel}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_json(path: Path, obj: Any, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    if path.exists() and force:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path.rename(path.with_name(f"{path.stem}.superseded.{stamp}{path.suffix}"))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    if path.exists() and force:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path.rename(path.with_name(f"{path.stem}.superseded.{stamp}{path.suffix}"))
    tmp.replace(path)


def p_mc(b_count: int, B: int) -> float:
    return float(b_count + 1) / float(B + 1)


def find_qlca_outputs(root: Path) -> Path | None:
    candidates = [
        resolve_path(root, SOURCE_REL),
        root / "paper" / "curvature_neurreps" / "audit_outputs" / "quadratic_label_chart_alignment",
        Path.home() / "platonic-universe" / "outputs" / "geometry" / "physics_quadratic_label_chart_alignment",
    ]
    for p in candidates:
        if (p / "decision.json").is_file() and (p / "anchor_risks.csv").is_file():
            return p
    return None


SOURCE_REL = "outputs/geometry/physics_quadratic_label_chart_alignment"
