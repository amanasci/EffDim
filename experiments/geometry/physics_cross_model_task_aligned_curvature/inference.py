"""Per-model P2 and synchronized cross-model R1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate
from geometry.physics_task_aligned_curvature.io_util import p_mc

from .config import INFER_SEED, REPLICATION


def synchronized_cross_model(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    xcol: str,
    ycol: str,
    models: tuple[str, ...],
    *,
    n_perm: int,
    n_boot: int,
    seed: int = INFER_SEED,
) -> dict[str, Any]:
    """Equal-model-weight mean of equal-target-weight ρ_ctl. Shared anchor permutation."""
    models = tuple(m for m in models if m in model_frames)
    ref = next(iter(model_frames[models[0]].values()))
    sids = ref.sample_id.to_numpy(int)
    n = len(ref)
    for m in models:
        for df in model_frames[m].values():
            if not np.array_equal(df.sample_id.to_numpy(int), sids):
                raise RuntimeError(f"{m}: sample_id order mismatch")

    def mean_of_means(idx=None, yperm=None):
        bars = []
        per = {}
        for m in models:
            rhos = []
            for t, df in model_frames[m].items():
                sub = df if idx is None else df.iloc[idx].reset_index(drop=True)
                x = sub[xcol].to_numpy(float)
                y = sub[ycol].to_numpy(float)
                Z = np.column_stack(
                    [
                        sub["log_knn_radius"].to_numpy(float),
                        sub["local_eval_label_variance"].to_numpy(float),
                        sub["local_evaluation_count"].to_numpy(float),
                    ]
                )
                if yperm is not None:
                    y = yperm[m][t]
                    if idx is not None:
                        y = y[idx]
                rhos.append(associate(x, y, Z)["controlled"])
            bar = float(np.mean(rhos))
            bars.append(bar)
            per[m] = {"bar": bar, "per_target": {t: float(r) for t, r in zip(model_frames[m], rhos)}}
        return float(np.mean(bars)), per

    obs, per_m = mean_of_means()
    rng = np.random.default_rng(seed + 17)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[b], _ = mean_of_means(idx=idx)
    # Freedman–Lane on each frame; same residual shuffle order across models/targets
    null = np.empty(n_perm)
    from scipy.stats import rankdata

    for b in range(n_perm):
        perm = rng.permutation(n)
        yperm: dict[str, dict[str, np.ndarray]] = {}
        for m in models:
            yperm[m] = {}
            for t, df in model_frames[m].items():
                y = df[ycol].to_numpy(float)
                Z = np.column_stack(
                    [
                        df["log_knn_radius"].to_numpy(float),
                        df["local_eval_label_variance"].to_numpy(float),
                        df["local_evaluation_count"].to_numpy(float),
                    ]
                )
                msk = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
                y2 = y.copy()
                yr = rankdata(y[msk]).astype(np.float64)
                Zr = np.column_stack([rankdata(Z[msk, j]) for j in range(Z.shape[1])])
                A = np.column_stack([np.ones(int(msk.sum())), Zr])
                bhat, *_ = np.linalg.lstsq(A, yr, rcond=None)
                fit = A @ bhat
                resid = yr - fit
                idx = np.where(msk)[0]
                y2[idx] = fit + resid[np.argsort(perm[idx])]
                yperm[m][t] = y2
        null[b], _ = mean_of_means(yperm=yperm)
    b_count = int(np.sum(null >= obs)) if np.isfinite(obs) else n_perm
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    n_pos = int(sum(per_m[m]["bar"] > 0 for m in models))
    return {
        "observed": float(obs),
        "per_model": {m: per_m[m]["bar"] for m in models},
        "per_model_targets": {m: per_m[m]["per_target"] for m in models},
        "n_models_positive": n_pos,
        "n_models": int(len(models)),
        "all_predicted_sign": bool(all(per_m[m]["bar"] > 0 for m in models)),
        "ci95": [float(lo), float(hi)],
        "p_mc": p_mc(b_count, n_perm),
        "side": "greater",
        "equal_model_weight": True,
        "equal_target_weight": True,
        "models": list(models),
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "replication_excludes_vit_base": list(models) == list(REPLICATION),
    }
