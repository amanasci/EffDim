"""Config, atomic IO, hashes. Never writes into preserved geometry trees."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import N_BOOT, N_PERM, PRESERVED, SEED, SOURCE_AUDIT, SOURCE_CPRS, SOURCE_MM, SOURCE_NDC, SOURCE_QPD


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


@dataclass
class ValConfig:
    output_dir: str = "outputs/geometry/physics_curvature_probe_submission_validation"
    cprs_dir: str = SOURCE_CPRS
    qpd_dir: str = SOURCE_QPD
    mm_dir: str = SOURCE_MM
    audit_dir: str = SOURCE_AUDIT
    ndc_dir: str = SOURCE_NDC
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    seed: int = SEED
    force: bool = False
    smoke: bool = False
    stage: str = "all"

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def perm_boot(self) -> tuple[int, int]:
        if self.smoke:
            return min(self.n_perm, 80), min(self.n_boot, 40)
        return int(self.n_perm), int(self.n_boot)


def assert_not_preserved(out: Path, root: Path) -> None:
    resolved = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if resolved == pres or pres in resolved.parents:
            raise RuntimeError(f"refusing to write into preserved geometry dir {rel}")


def atomic_replace(tmp: Path, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        ts = time.strftime("%Y%m%dT%H%M%S")
        dest.rename(dest.with_name(f"{dest.stem}.superseded.{ts}{dest.suffix}"))
    tmp.replace(dest)


def write_json(path: Path, obj: Any, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    atomic_replace(tmp, path, force=force)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    atomic_replace(tmp, path, force=force)


def file_sha_full(p: Path) -> str:
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def p_report(p: float, n: int) -> str:
    if not np.isfinite(p):
        return "nan"
    floor = 1.0 / float(n + 1)
    if p <= 0.0:
        return f"<{floor:.2e}"
    return f"{p:.4g}"


def hash_select_cprs(sids: list[int], n: int, *, seed: int) -> list[int]:
    scored = [(hashlib.sha256(f"cprs:{seed}:{int(s)}".encode()).hexdigest(), int(s)) for s in sids]
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]
