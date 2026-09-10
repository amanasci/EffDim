"""End-to-end label shuffles of G, P, Δ_adapt, and controlled associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict

from .config import MIN_TEST_PER_FOLD, MIN_TRAIN_PER_FOLD, PRIMARY_K, PROBE_ALPHA
from .inference import _assoc


def _cache_anchor_folds(X, fold, neigh, sid_to_ai, sids) -> list[dict]:
    cache = []
    for sid in sids:
        ai = sid_to_ai[int(sid)]
        N = neigh[ai, :PRIMARY_K]
        entries = []
        for f in sorted(set(fold[N].tolist())):
            te_local = np.where(fold[N] == f)[0]
            tr_local = np.where(fold[N] != f)[0]
            tr, te = N[tr_local], N[te_local]
            if len(tr) < MIN_TRAIN_PER_FOLD or len(te) < MIN_TEST_PER_FOLD:
                continue
            entries.append({"tr": tr, "te": te, "N": N})
        cache.append({"sid": int(sid), "N": N, "entries": entries})
    return cache


def run_shuffles(
    *,
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    neigh: np.ndarray,
    sid_to_ai: dict[int, int],
    sids: list[int],
    kh: pd.Series,
    controls: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Permute labels globally; refit G (per outer fold) and P (per anchor/fold)."""
    rng = np.random.default_rng(seed)
    cache = _cache_anchor_folds(X, fold, neigh, sid_to_ai, sids)
    rows = []
    for b in range(n_perm):
        yb = rng.permutation(y)
        # global OOF under shuffle
        yhat = np.full(len(y), np.nan)
        for f in range(5):
            tr = fold != f
            te = fold == f
            w, b0, info = ridge_fit_intercept(X[tr], yb[tr], alpha=PROBE_ALPHA)
            if info["ok"]:
                yhat[te] = ridge_predict(X[te], w, b0)
            else:
                yhat[te] = float(np.nanmean(yb[tr]))
        recs = []
        for item in cache:
            N = item["N"]
            pred_p = np.full(len(N), np.nan)
            pos = {int(ix): j for j, ix in enumerate(N)}
            for e in item["entries"]:
                w, b0, info = ridge_fit_intercept(X[e["tr"]], yb[e["tr"]], alpha=PROBE_ALPHA)
                if not info["ok"]:
                    continue
                for ix, val in zip(e["te"], ridge_predict(X[e["te"]], w, b0)):
                    pred_p[pos[int(ix)]] = val
            mse_g = float(np.nanmean((yb[N] - yhat[N]) ** 2))
            mse_p = float(np.nanmean((yb[N] - pred_p) ** 2))
            recs.append(
                {
                    "sample_id": item["sid"],
                    "mse_G": mse_g,
                    "mse_P": mse_p,
                    "delta_adapt": mse_g - mse_p if np.isfinite(mse_g) and np.isfinite(mse_p) else float("nan"),
                    "K_H_cross": float(kh.loc[item["sid"]]) if item["sid"] in kh.index else float("nan"),
                }
            )
        df = pd.DataFrame(recs).merge(controls, on="sample_id", how="left")
        a_g = _assoc(df, "mse_G")
        a_a = _assoc(df, "delta_adapt")
        a_p = _assoc(df, "mse_P")
        rows.append(
            {
                "perm": b,
                "C_G": a_g["controlled"],
                "C_A": a_a["controlled"],
                "A": float(a_g["controlled"] - a_p["controlled"]) if np.isfinite(a_g["controlled"]) else float("nan"),
                "mean_delta_adapt": float(np.nanmean(df.delta_adapt)),
            }
        )
        if (b + 1) % 8 == 0:
            print(f"[cmcla][shuffle] {b+1}/{n_perm}", flush=True)
    tab = pd.DataFrame(rows)
    summary = {
        "n": int(len(tab)),
        "median_C_G": float(tab.C_G.median()),
        "median_C_A": float(tab.C_A.median()),
        "median_A": float(tab.A.median()),
        "frac_C_G_positive": float((tab.C_G > 0).mean()),
        "frac_C_A_positive": float((tab.C_A > 0).mean()),
        "false_positive_safe": bool(abs(float(tab.C_A.median())) < 0.15 and abs(float(tab.C_G.median())) < 0.15),
        "note": "Safety gate is absence of a systematic positive association, not exact null equality.",
    }
    return {"rows": rows, **summary}
