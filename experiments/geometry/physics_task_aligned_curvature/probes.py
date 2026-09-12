"""Historical (descriptive) and confirmatory leakage-safe probes."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import MIN_EVAL, PROBE_ALPHA, SPLIT_SALT, TARGETS, TRAIN_FRAC


def hash_u01(target: str, sample_id: int) -> float:
    msg = f"{SPLIT_SALT}:{target}:{int(sample_id)}".encode()
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:8], "little") / float(2**64)


def split_mask(sample_ids: np.ndarray, finite: np.ndarray, target: str) -> dict[str, np.ndarray]:
    n = len(sample_ids)
    train = np.zeros(n, dtype=bool)
    eval_ = np.zeros(n, dtype=bool)
    for i, sid in enumerate(sample_ids):
        if not finite[i]:
            continue
        u = hash_u01(target, int(sid))
        if u < TRAIN_FRAC:
            train[i] = True
        else:
            eval_[i] = True
    return {"train": train, "eval": eval_}


def fit_ridge(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Train-only Ridge, α=100, unpenalized intercept. No standardization (historical)."""
    m = mask & np.isfinite(y)
    ridge = Ridge(alpha=PROBE_ALPHA, fit_intercept=True)
    ridge.fit(X[m], y[m])
    return {
        "w": ridge.coef_.astype(np.float64),
        "b": float(ridge.intercept_),
        "n_train": int(m.sum()),
        "alpha": PROBE_ALPHA,
        "standardized": False,
        "train_only": True,
    }


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def local_risk(
    y: np.ndarray,
    yhat: np.ndarray,
    neigh: np.ndarray,
    eval_mask: np.ndarray,
    *,
    min_eval: int = MIN_EVAL,
) -> dict[str, np.ndarray]:
    n_a = len(neigh)
    mse = np.full(n_a, np.nan)
    r2 = np.full(n_a, np.nan)
    n_eval = np.zeros(n_a, dtype=np.int64)
    var = np.full(n_a, np.nan)
    for i in range(n_a):
        idx = neigh[i]
        take = eval_mask[idx] & np.isfinite(y[idx]) & np.isfinite(yhat[idx])
        jj = idx[take]
        n_eval[i] = int(len(jj))
        if n_eval[i] < min_eval:
            continue
        yi = y[jj]
        yh = yhat[jj]
        mse[i] = float(np.mean((yi - yh) ** 2))
        ym = float(yi.mean())
        denom = float(np.sum((yi - ym) ** 2))
        r2[i] = float("nan") if denom < 1e-15 else 1.0 - float(np.sum((yi - yh) ** 2)) / denom
        var[i] = float(np.var(yi, ddof=1)) if n_eval[i] > 1 else float("nan")
    return {"mse_G": mse, "r2_G": r2, "n_eval": n_eval, "local_eval_label_variance": var}


def coverage_ok(n_eval: np.ndarray, *, min_eval: int = MIN_EVAL, fail_frac: float = 0.10) -> dict[str, Any]:
    below = n_eval < min_eval
    frac = float(below.mean()) if len(n_eval) else 1.0
    return {
        "min_eval": min_eval,
        "frac_below": frac,
        "n_below": int(below.sum()),
        "n": int(len(n_eval)),
        "ok": frac <= fail_frac,
    }


def confirmatory_for_target(shared: dict, target: str) -> dict[str, Any]:
    finite = shared["finite"][target]
    sids_all = shared["sample_id_row"]
    split = split_mask(sids_all, finite, target)
    rec = fit_ridge(shared["X"], shared["Y"][target], split["train"])
    yhat = predict(shared["X"], rec["w"], rec["b"])
    # leakage: train labels must not enter eval stats — local_risk uses eval_mask only
    risk = local_risk(shared["Y"][target], yhat, shared["neigh"], split["eval"])
    cov = coverage_ok(risk["n_eval"])
    rec.update(risk)
    rec["split"] = split
    rec["yhat"] = yhat
    rec["coverage"] = cov
    rec["target"] = target
    rec["analysis"] = "C_confirmatory"
    rec["label_leakage"] = False
    return rec


def historical_for_target(shared: dict, target: str, hist_w: dict, hist_r2: pd.DataFrame | None) -> dict[str, Any]:
    w = hist_w[target]["w"]
    b = hist_w[target]["b"]
    yhat = predict(shared["X"], w, b)
    # descriptive: score on all labelled neighbours (historical definition)
    all_lab = shared["finite"][target]
    risk = local_risk(shared["Y"][target], yhat, shared["neigh"], all_lab, min_eval=MIN_EVAL)
    out = {
        "w": w,
        "b": b,
        "yhat": yhat,
        "analysis": "H_historical_descriptive",
        "label_leakage": True,
        "potentially_label_dependent": True,
        "target": target,
        **risk,
    }
    if hist_r2 is not None:
        sub = hist_r2[hist_r2.target == target]
        mp = {int(s): float(r) for s, r in zip(sub.sample_id, sub.local_r2)}
        out["historical_local_r2"] = np.array([mp.get(int(s), np.nan) for s in shared["sids"]], dtype=float)
    return out
