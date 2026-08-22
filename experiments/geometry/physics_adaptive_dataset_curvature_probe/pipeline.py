"""Config, atomic IO, hash helpers. Never writes into completed geometry trees."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geometry.physics_activation_atlas.paths import platonic_root, resolve_path

from .config import (
    DEFAULT_THRESHOLDS,
    HASH_PREFIX,
    K_FRAC_OF_N,
    K_PRESET,
    N_BOOT,
    N_PERM,
    N_SCALE_ANCHORS,
    PRESERVED,
    SEED,
    SOURCE_MM,
    TARGET_ANCHORS,
)


@dataclass
class AdaptiveProbeConfig:
    output_dir: str = "outputs/geometry/physics_adaptive_dataset_curvature_probe"
    multimodel_dir: str = SOURCE_MM
    encoder_family: str = "vit_base"
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    n_anchors: int | None = None
    n_scale_anchors: int = N_SCALE_ANCHORS
    seed: int = SEED
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 400.0
    smoke: bool = False
    skip_desi: bool = False

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)

    def target_anchors(self) -> int:
        if self.n_anchors is not None:
            return int(self.n_anchors)
        if self.smoke:
            return 8
        return TARGET_ANCHORS

    def perm_boot(self) -> tuple[int, int]:
        if self.smoke:
            return min(self.n_perm, 200), min(self.n_boot, 80)
        return int(self.n_perm), int(self.n_boot)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: AdaptiveProbeConfig, reserve: bool = False) -> bool:
    return (cfg.max_seconds - (time.time() - t0)) > (cfg.analyze_reserve_seconds if reserve else 20.0)


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
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


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


def hash_select(sids: list[int], n: int, *, seed: int) -> list[int]:
    scored = [(hashlib.sha256(f"{HASH_PREFIX}:{seed}:{int(s)}".encode()).hexdigest(), int(s)) for s in sids]
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]


def primary_k(n_obs: int) -> int | None:
    """Largest preset k with k <= K_FRAC_OF_N * n. None if even 256 is too large."""
    cap = float(K_FRAC_OF_N) * float(n_obs)
    ok = [k for k in K_PRESET if k <= cap]
    return int(max(ok)) if ok else None


def scale_list(n_obs: int) -> list[int]:
    cap = float(K_FRAC_OF_N) * float(n_obs)
    return [k for k in K_PRESET if k <= cap]


def device_of(cfg: AdaptiveProbeConfig) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def p_report(exceed: int, n: int) -> str:
    """Zero exceedances as p<1/(B+1), never p=0."""
    if exceed <= 0:
        return f"<{1.0 / (n + 1):.6g}"
    return f"{(exceed + 1) / (n + 1):.6g}"


def p_value(exceed: int, n: int) -> float:
    return float((exceed + 1) / (n + 1))


def crossing_d(ds: np.ndarray, r2: np.ndarray, tau: float):
    hit = [int(d) for d, v in zip(ds, r2) if np.isfinite(v) and v >= tau]
    return int(min(hit)) if hit else "not_reached"


def existing_min(*vals) -> float:
    nums = [float(v) for v in vals if v is not None and v != "not_reached" and np.isfinite(float(v) if not isinstance(v, str) else np.nan)]
    return float(min(nums)) if nums else float("nan")


def existing_max(*vals) -> float:
    nums = []
    for v in vals:
        if v is None or v == "not_reached":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            nums.append(x)
    return float(max(nums)) if nums else float("nan")
