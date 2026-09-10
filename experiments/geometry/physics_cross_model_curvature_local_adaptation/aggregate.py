"""Equal-model-weight aggregate with synchronized anchor resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import control_matrix, freedman_lane_y

from .inference import _assoc
from .io_util import p_mc


def _aligned_tables(by_model: dict[str, pd.DataFrame]) -> tuple[list[str], list[int], dict[str, pd.DataFrame]]:
    models = sorted(by_model)
    common = None
    for m, df in by_model.items():
        s = set(df.sample_id.astype(int).tolist())
        common = s if common is None else (common & s)
    sids = sorted(common)
    aligned = {m: by_model[m].set_index("sample_id").loc[sids].reset_index() for m in models}
    return models, sids, aligned


def equal_weight_stats(aligned: dict[str, pd.DataFrame]) -> dict[str, float]:
    cgs, cas, As = [], [], []
    for df in aligned.values():
        cg = float(_assoc(df, "mse_G")["controlled"])
        cp = float(_assoc(df, "mse_P")["controlled"])
        ca = float(_assoc(df, "delta_adapt")["controlled"])
        cgs.append(cg)
        cas.append(ca)
        As.append(cg - cp)
    return {
        "C_G_bar": float(np.mean(cgs)),
        "C_A_bar": float(np.mean(cas)),
        "A_bar": float(np.mean(As)),
        "sign_C_G": [float(x) for x in cgs],
        "sign_C_A": [float(x) for x in cas],
        "sign_A": [float(x) for x in As],
    }


def synchronized_inference(
    by_model: dict[str, pd.DataFrame],
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    models, sids, aligned = _aligned_tables(by_model)
    obs = equal_weight_stats(aligned)
    rng = np.random.default_rng(seed)
    n = len(sids)
    boot = {k: np.empty(n_boot) for k in ("C_G_bar", "C_A_bar", "A_bar")}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub = {m: df.iloc[idx].reset_index(drop=True) for m, df in aligned.items()}
        st = equal_weight_stats(sub)
        for k in boot:
            boot[k][b] = st[k]
    # synchronized Freedman–Lane: residualize KH within model, apply the same permutation
    null = {k: np.empty(n_perm) for k in ("C_G_bar", "C_A_bar", "A_bar")}
    Zs = {m: control_matrix(df.reset_index(drop=True)) for m, df in aligned.items()}
    khs = {m: df.K_H_cross.to_numpy(float) for m, df in aligned.items()}
    for b in range(n_perm):
        # one permutation applied to every model's rank-residual vector via shared rng stream
        # Use a shared permutation of indices on the residualized KH of the first model,
        # then apply the same index perm to each model's residual reconstruction.
        perm = rng.permutation(n)
        sub = {}
        for m, df in aligned.items():
            khp = freedman_lane_y(khs[m], Zs[m], rng)
            # overwrite with index-synchronized permutation of the already-residualized draw:
            # re-apply a common permutation to the residualized vector
            tmp = df.copy()
            tmp["K_H_cross"] = khp[perm]
            sub[m] = tmp
        st = equal_weight_stats(sub)
        for k in null:
            null[k][b] = st[k]

    out: dict[str, Any] = {
        "models": models,
        "n_common_anchors": n,
        "observed": {k: obs[k] for k in ("C_G_bar", "C_A_bar", "A_bar")},
        "per_model": {
            "C_G": obs["sign_C_G"],
            "C_A": obs["sign_C_A"],
            "A": obs["sign_A"],
        },
        "n_positive_C_G": int(sum(x > 0 for x in obs["sign_C_G"])),
        "n_positive_C_A": int(sum(x > 0 for x in obs["sign_C_A"])),
        "n_positive_A": int(sum(x > 0 for x in obs["sign_A"])),
        "n_perm": n_perm,
        "n_boot": n_boot,
    }
    for k, expected_pos in (("C_G_bar", True), ("C_A_bar", True), ("A_bar", True)):
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        obs_v = float(obs[k])
        b_count = int(np.sum(null[k] >= obs_v)) if expected_pos else int(np.sum(np.abs(null[k]) >= abs(obs_v)))
        out[k] = {
            "observed": obs_v,
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b_count, n_perm),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
    return out
