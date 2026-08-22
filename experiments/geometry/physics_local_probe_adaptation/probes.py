"""Strict global-fold OOF patch probes: G, I, C, P, T."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import MIN_TEST_PER_FOLD, MIN_TRAIN_PER_FOLD, PRIMARY_D, PROBE_ALPHA, ALPHA_GRID
from .metrics import metrics_from_preds
from .ridge import nested_alpha_1se, ridge_fit_intercept, ridge_predict


def _tangent_coords(Xloc: np.ndarray, d: int = PRIMARY_D) -> np.ndarray:
    """Transductive PCA chart on the full patch (unsupervised)."""
    x0 = Xloc.mean(axis=0)
    Xc = Xloc - x0
    # economy SVD
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((len(Xloc), d), np.nan)
    J = Vt[:d].T  # (D, d)
    if J.shape[1] < d:
        return np.full((len(Xloc), d), np.nan)
    return Xc @ J


def fit_anchor_oof(
    *,
    X: np.ndarray,
    y: np.ndarray,
    yhat_g: np.ndarray,
    fold: np.ndarray,
    neigh_idx: np.ndarray,
    sample_ids_row: np.ndarray,
    alpha: float = PROBE_ALPHA,
    do_tangent: bool = True,
    do_nested_alpha: bool = False,
    do_insample: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Return OOF predictions and fold logs for one patch."""
    idx = np.asarray(neigh_idx, dtype=np.int64)
    folds_present = sorted(set(fold[idx].tolist()))
    n = len(idx)
    pred = {
        "G": np.full(n, np.nan),
        "I": np.full(n, np.nan),
        "C": np.full(n, np.nan),
        "P": np.full(n, np.nan),
        "T": np.full(n, np.nan),
        "P_insample": np.full(n, np.nan),
    }
    fold_logs = []
    weight_rows = []
    U = _tangent_coords(X[idx], PRIMARY_D) if do_tangent else None
    selected_alpha = float(alpha)
    alpha_info = {}

    # optional nested α on union of all outer-train folds is wrong; select per outer fold below
    for f in folds_present:
        te_local = np.where(fold[idx] == f)[0]
        tr_local = np.where(fold[idx] != f)[0]
        te = idx[te_local]
        tr = idx[tr_local]
        overlap = set(sample_ids_row[tr].tolist()) & set(sample_ids_row[te].tolist())
        n_tr, n_te = int(len(tr)), int(len(te))
        log = {
            "fold": int(f),
            "n_train": n_tr,
            "n_test": n_te,
            "train_test_overlap": int(len(overlap)),
            "ok": True,
            "reason": "",
        }
        if overlap:
            log["ok"] = False
            log["reason"] = "overlap"
            fold_logs.append(log)
            continue
        if n_tr < MIN_TRAIN_PER_FOLD or n_te < MIN_TEST_PER_FOLD:
            log["ok"] = False
            log["reason"] = "counts"
            pred["G"][te_local] = yhat_g[te]
            fold_logs.append(log)
            continue

        pred["G"][te_local] = yhat_g[te]

        # I: patch mean of training labels
        y_tr = y[tr]
        mtr = np.isfinite(y_tr)
        mu = float(np.mean(y_tr[mtr])) if mtr.sum() else float("nan")
        pred["I"][te_local] = mu

        # C: calibrate global OOF on train → a + b * yhat_g
        g_tr = yhat_g[tr]
        m = np.isfinite(y_tr) & np.isfinite(g_tr)
        if m.sum() >= 8:
            g0 = g_tr[m]
            y0 = y_tr[m]
            # simple least squares with intercept
            G = np.column_stack([np.ones(m.sum()), g0])
            coef, *_ = np.linalg.lstsq(G, y0, rcond=None)
            a_c, b_c = float(coef[0]), float(coef[1])
            pred["C"][te_local] = a_c + b_c * yhat_g[te]
        else:
            pred["C"][te_local] = yhat_g[te]

        # P: ambient ridge
        a_use = float(alpha)
        if do_nested_alpha:
            a_use, alpha_info = nested_alpha_1se(X[tr], y[tr], fold[tr], ALPHA_GRID, seed=seed + f)
            selected_alpha = a_use
        w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=a_use)
        if info["ok"]:
            pred["P"][te_local] = ridge_predict(X[te], w, b)
            weight_rows.append(
                {
                    "fold": int(f),
                    "model": "P",
                    "w": w,
                    "b": b,
                    "edf": info["edf"],
                    "cond": info["cond"],
                    "alpha": a_use,
                }
            )
        if do_insample:
            # invalid: train on all patch members (leakage comparison only)
            w_all, b_all, info_all = ridge_fit_intercept(X[idx], y[idx], alpha=a_use)
            if info_all["ok"]:
                pred["P_insample"][te_local] = ridge_predict(X[te], w_all, b_all)

        # T: tangent ridge
        if do_tangent and U is not None and np.all(np.isfinite(U)):
            wT, bT, infoT = ridge_fit_intercept(U[tr_local], y[tr], alpha=a_use)
            if infoT["ok"]:
                pred["T"][te_local] = ridge_predict(U[te_local], wT, bT)
                weight_rows.append(
                    {
                        "fold": int(f),
                        "model": "T",
                        "w": wT,
                        "b": bT,
                        "edf": infoT["edf"],
                        "cond": infoT["cond"],
                        "alpha": a_use,
                    }
                )

        log["alpha"] = a_use
        fold_logs.append(log)

    y_patch = y[idx]
    out_metrics = {}
    for name, yh in pred.items():
        out_metrics[name] = metrics_from_preds(y_patch, yh)

    # direction diagnostics from P weights
    dir_diag = {"selected_alpha": selected_alpha, "alpha_info": alpha_info}
    p_ws = [r for r in weight_rows if r["model"] == "P"]
    if len(p_ws) >= 2:
        cos = []
        for i in range(len(p_ws)):
            for j in range(i + 1, len(p_ws)):
                u, v = p_ws[i]["w"], p_ws[j]["w"]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu > 0 and nv > 0:
                    cos.append(float(np.dot(u, v) / (nu * nv)))
        dir_diag["P_fold_cosine_med"] = float(np.median(cos)) if cos else float("nan")
        dir_diag["P_edf_med"] = float(np.median([r["edf"] for r in p_ws]))
        dir_diag["P_cond_med"] = float(np.median([r["cond"] for r in p_ws]))
        w_mean = np.mean([r["w"] for r in p_ws], axis=0)
        dir_diag["P_w_norm"] = float(np.linalg.norm(w_mean))
    else:
        dir_diag["P_fold_cosine_med"] = float("nan")

    return {
        "metrics": out_metrics,
        "fold_logs": fold_logs,
        "n_eval": int(np.isfinite(y_patch).sum()),
        "overlap_any": any(int(l.get("train_test_overlap", 0)) > 0 for l in fold_logs),
        "dir_diag": dir_diag,
        "pred": pred,
        "idx": idx,
    }
