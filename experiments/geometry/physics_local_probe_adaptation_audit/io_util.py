"""IO helpers; refuse writes into preserved trees."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PRESERVED


def platonic_root() -> Path:
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home() / "platonic-universe"
    if home.is_dir():
        return home.resolve()
    here = Path(__file__).resolve()
    for cand in [here.parents[i] for i in range(2, min(8, len(here.parents)))]:
        if (cand / "outputs" / "geometry").is_dir():
            return cand.resolve()
    return Path.cwd().resolve()


def resolve_path(root: Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def assert_not_preserved(out: Path, root: Path) -> None:
    r = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if r == pres or pres in r.parents:
            raise RuntimeError(f"refusing write into preserved tree {rel}")


def write_json(path: Path, obj: Any, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and force:
        path.rename(path.with_name(f"{path.stem}.superseded.{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}"))
    tmp.replace(path)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and force:
        path.rename(path.with_name(f"{path.stem}.superseded.{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}"))
    tmp.replace(path)


def p_mc(b: int, B: int) -> float:
    return float(b + 1) / float(B + 1)
