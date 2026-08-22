"""Lightweight synthetic checks. Seeds frozen before empirical inspection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .inference import associate, control_matrix, permutation_curves


SYNTH_SEEDS = {"calibration": 8100, "evaluation": 8300}


def make_wide(
    n: int,
    ds: list[int],
    *,
    seed: int,
    kind: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = {"sample_id": np.arange(n)}
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z3 = rng.normal(size=n)
    rows["log_knn_radius"] = z1
    rows["local_label_variance"] = z2
    rows["local_evaluation_count"] = z3
    if kind == "null":
        y = rng.normal(size=n)
        for d in ds:
            rows[f"KH{d}"] = rng.normal(size=n) * (0.5 + 0.05 * (d - 8))
    elif kind == "var_only":
        y = rng.normal(size=n)
        for d in ds:
            rows[f"KH{d}"] = rng.normal(size=n) * (0.2 + 0.15 * (d - 8))
    elif kind == "confound":
        y = 0.8 * z1 + 0.2 * rng.normal(size=n)
        for d in ds:
            rows[f"KH{d}"] = 0.7 * z1 + rng.normal(size=n) * 0.4
    elif kind == "planted16":
        signal = rng.normal(size=n)
        y = -0.6 * signal + 0.2 * rng.normal(size=n)
        for d in ds:
            w = 1.0 if d == 16 else (0.15 if abs(d - 16) <= 1 else 0.0)
            rows[f"KH{d}"] = w * signal + rng.normal(size=n) * 0.5
    else:
        raise ValueError(kind)
    rows["local_r2"] = y
    return pd.DataFrame(rows)


def eval_family(kind: str, ds: list[int], *, seed: int, n: int = 256, n_perm: int = 400) -> dict[str, Any]:
    wide = make_wide(n, ds, seed=seed, kind=kind)
    perm = permutation_curves(wide, ds, ycol="local_r2", x_prefix="KH", n_perm=n_perm, seed=seed + 1, controlled=False)
    ctl = permutation_curves(wide, ds, ycol="local_r2", x_prefix="KH", n_perm=n_perm, seed=seed + 2, controlled=True)
    peak = int(ds[int(np.nanargmax(np.abs([perm["obs"][d]["raw"] for d in ds])))])
    return {
        "kind": kind,
        "p_global_raw": perm["p_global"],
        "p_global_ctl": ctl["p_global"],
        "peak_raw": peak,
        "rho16_raw": perm["obs"].get(16, {}).get("raw", float("nan")),
        "rho16_ctl": ctl["obs"].get(16, {}).get("controlled", float("nan")),
        "rho12_ctl": ctl["obs"].get(12, {}).get("controlled", float("nan")),
    }
