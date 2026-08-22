"""Config, atomic IO, hashes, and association primitives. Never writes into preserved trees."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .config import (
    DEFAULT_THRESHOLDS,
    N_BOOT,
    N_PERM,
    PRESERVED,
    SEED,
    SHARED_CORE_CONTROLS,
    SOURCE_ADCP,
    SOURCE_CPRS,
    SOURCE_EDM,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QPD,
)


def platonic_root() -> Path:
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home() / "platonic-universe"
    if home.is_dir():
        return home.resolve()
    here = Path(__file__).resolve()
    for cand in [here.parents[i] for i in range(2, min(6, len(here.parents)))]:
        if (cand / "data_hf").is_dir() or (cand / "outputs" / "geometry").is_dir():
            return cand.resolve()
    return Path.cwd().resolve()


def resolve_path(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


@dataclass
class AuditConfig:
    output_dir: str = "outputs/geometry/physics_adaptive_dataset_curvature_probe_audit"
    adaptive_dir: str = SOURCE_ADCP
    rank_sweep_dir: str = SOURCE_CPRS
    nested_dir: str = SOURCE_NDC
    multimodel_dir: str = SOURCE_MM
    qpd_dir: str = SOURCE_QPD
    freeze_dir: str = SOURCE_EDM
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    seed: int = SEED
    force: bool = False
    stage: str = "all"
    smoke: bool = False
    max_seconds: float = 36000.0

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def adcp(self, root: Path) -> Path:
        return resolve_path(root, self.adaptive_dir)

    def cprs(self, root: Path) -> Path:
        return resolve_path(root, self.rank_sweep_dir)

    def ndc(self, root: Path) -> Path:
        return resolve_path(root, self.nested_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)

    def perm_boot(self) -> tuple[int, int]:
        if self.smoke:
            return min(self.n_perm, 200), min(self.n_boot, 80)
        return int(self.n_perm), int(self.n_boot)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def sha16(payload: Any) -> str:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    if not p.exists():
        return "missing"
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


def file_sha_full(p: Path) -> str:
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_select(sids: list[int], n: int, *, seed: int, prefix: str) -> list[int]:
    scored = [(hashlib.sha256(f"{prefix}:{seed}:{int(s)}".encode()).hexdigest(), int(s)) for s in sids]
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]


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


def p_value(exceed: int, n: int) -> float:
    return float((exceed + 1) / (n + 1))


def p_report(exceed: int, n: int) -> str:
    if exceed <= 0:
        return f"<{1.0 / (n + 1):.6g}"
    return f"{(exceed + 1) / (n + 1):.6g}"


def p_monte_carlo_ci(exceed: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper–Pearson interval on the raw exceedance rate, then +1/(n+1) shift."""
    from scipy.stats import beta

    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if exceed <= 0 else float(beta.ppf(alpha / 2.0, exceed, n - exceed + 1))
    hi = 1.0 if exceed >= n else float(beta.ppf(1.0 - alpha / 2.0, exceed + 1, n - exceed))
    return float((lo * n + 1) / (n + 1)), float((hi * n + 1) / (n + 1))


def spearman_dict(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": n}
    rho, p = spearmanr(x[m], y[m])
    return {"rho": float(rho), "pvalue": float(p), "n": n}


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    if np.std(x[m]) < 1e-15 or np.std(y[m]) < 1e-15:
        return float("nan")
    return float(pearsonr(x[m], y[m])[0])


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    n = int(m.sum())
    if n < 12:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": n}
    xr = rankdata(x[m]).astype(np.float64)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(n), Zr])
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    rho, p = spearmanr(xr - A @ bx, yr - A @ by)
    return {"rho": float(rho), "pvalue": float(p), "n": n}


def associate(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> dict[str, float]:
    raw = spearman_dict(x, y)
    if Z is None:
        return {"raw": raw["rho"], "controlled": float("nan"), "n": raw["n"], "p_raw": raw["pvalue"]}
    ctl = partial_spearman(x, y, Z)
    return {"raw": raw["rho"], "controlled": ctl["rho"], "n": raw["n"], "p_raw": raw["pvalue"], "p_ctl": ctl["pvalue"]}


def control_matrix(df: pd.DataFrame, cols=SHARED_CORE_CONTROLS) -> np.ndarray | None:
    if not all(c in df.columns for c in cols):
        return None
    return np.column_stack([df[c].fillna(0).to_numpy(float) for c in cols])


def delta_85_80(rho_by_d: dict[int, float], d80: int, d85: int) -> float:
    a, b = rho_by_d.get(int(d80)), rho_by_d.get(int(d85))
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(b - a)


def peak_abs(rho_by_d: dict[int, float]) -> tuple[int | None, float]:
    items = [(int(d), float(v)) for d, v in rho_by_d.items() if np.isfinite(v)]
    if not items:
        return None, float("nan")
    d, v = max(items, key=lambda kv: abs(kv[1]))
    return d, v


def linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan"), float("nan")
    b, a = np.polyfit(x[m], y[m], 1)
    return float(b), float(a)


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))


def sources_available(root: Path, cfg: AuditConfig) -> bool:
    return cfg.adcp(root).exists() and cfg.cprs(root).exists() and cfg.mm(root).exists()
