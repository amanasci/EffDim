"""G / G-cal / G-affine-cal / P / T with strict outer-fold OOF isolation."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_local_probe_adaptation.config import MIN_TEST_PER_FOLD, MIN_TRAIN_PER_FOLD
from geometry.physics_local_probe_adaptation.metrics import metrics_from_preds
from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict

from .config import PRIMARY_D, PROBE_ALPHA


def _coords(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    return (np.asarray(Xloc, dtype=np.float64) - np.asarray(x0, dtype=np.float64)) @ np.asarray(J, dtype=np.float64)


def fit_anchor_oof(
    *,
    X: np.ndarray,
    y: np.ndarray,
    yhat_g: np.ndarray,
    fold: np.ndarray,
    neigh_idx: np.ndarray,
    sample_ids_row: np.ndarray,
    w_G_by_fold: dict[int, np.ndarray] | None = None,
    b_G_by_fold: dict[int, float] | None = None,
    x0: np.ndarray | None = None,
    J: np.ndarray | None = None,
    alpha: float = PROBE_ALPHA,
) -> dict[str, Any]:
    idx = np.asarray(neigh_idx, dtype=np.int64)
    folds_present = sorted(set(fold[idx].tolist()))
    n = len(idx)
    pred = {k: np.full(n, np.nan) for k in ("G", "Gcal", "C", "P", "T", "P_dir_Gcal", "G_dir_Pcal")}
    fold_logs = []
    weights = []
    U = None
    if x0 is not None and J is not None:
        U = _coords(X[idx], x0, J)
        if U.shape[1] != PRIMARY_D:
            U = U[:, :PRIMARY_D]

    for f in folds_present:
        te_local = np.where(fold[idx] == f)[0]
        tr_local = np.where(fold[idx] != f)[0]
        te, tr = idx[te_local], idx[tr_local]
        overlap = set(sample_ids_row[tr].tolist()) & set(sample_ids_row[te].tolist())
        log = {
            "fold": int(f),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_test_overlap": int(len(overlap)),
            "ok": True,
            "reason": "",
        }
        if overlap:
            log["ok"] = False
            log["reason"] = "overlap"
            fold_logs.append(log)
            continue
        if len(tr) < MIN_TRAIN_PER_FOLD or len(te) < MIN_TEST_PER_FOLD:
            log["ok"] = False
            log["reason"] = "counts"
            pred["G"][te_local] = yhat_g[te]
            fold_logs.append(log)
            continue

        pred["G"][te_local] = yhat_g[te]
        y_tr, g_tr = y[tr], yhat_g[tr]
        m = np.isfinite(y_tr) & np.isfinite(g_tr)
        if int(m.sum()) >= 8:
            resid = y_tr[m] - g_tr[m]
            a_i = float(np.mean(resid))
            pred["Gcal"][te_local] = yhat_g[te] + a_i
            Gmat = np.column_stack([np.ones(int(m.sum())), g_tr[m]])
            coef, *_ = np.linalg.lstsq(Gmat, y_tr[m], rcond=None)
            a_c, b_c = float(coef[0]), float(coef[1])
            pred["C"][te_local] = a_c + b_c * yhat_g[te]
        else:
            pred["Gcal"][te_local] = yhat_g[te]
            pred["C"][te_local] = yhat_g[te]

        wP, bP, infoP = ridge_fit_intercept(X[tr], y[tr], alpha=alpha)
        if infoP["ok"]:
            pred["P"][te_local] = ridge_predict(X[te], wP, bP)
            weights.append({"fold": int(f), "model": "P", "w": wP, "b": bP, "edf": infoP["edf"], "cond": infoP["cond"]})
            # local direction, global-style intercept from train G residual mean on P direction
            # P_dir + Gcal: freeze w_P, replace intercept so train mean matches y (already in bP).
            # Hybrid: scale P predictions by affine onto G on train, then... user asked:
            # local tangent/ambient direction with global calibration.
            # Use w_P with intercept chosen so mean(P_dir) matches mean(G) on train.
            p_tr = ridge_predict(X[tr], wP, bP)
            g_ok = np.isfinite(g_tr) & np.isfinite(p_tr)
            if int(g_ok.sum()) >= 8:
                # map P_train to G_train affine, apply to test P
                A = np.column_stack([np.ones(int(g_ok.sum())), p_tr[g_ok]])
                cf, *_ = np.linalg.lstsq(A, g_tr[g_ok], rcond=None)
                pred["P_dir_Gcal"][te_local] = float(cf[0]) + float(cf[1]) * ridge_predict(X[te], wP, bP)
            # global direction, local P calibration: affine from G to y is C; also G_dir with P's affine
            pred["G_dir_Pcal"][te_local] = pred["C"][te_local]

        if w_G_by_fold is not None and int(f) in w_G_by_fold:
            weights.append(
                {
                    "fold": int(f),
                    "model": "G",
                    "w": np.asarray(w_G_by_fold[int(f)], dtype=np.float64),
                    "b": float(b_G_by_fold[int(f)]) if b_G_by_fold else 0.0,
                    "edf": float("nan"),
                    "cond": float("nan"),
                }
            )

        if U is not None and np.all(np.isfinite(U)):
            wT, bT, infoT = ridge_fit_intercept(U[tr_local], y[tr], alpha=alpha)
            if infoT["ok"]:
                pred["T"][te_local] = ridge_predict(U[te_local], wT, bT)
                weights.append({"fold": int(f), "model": "T", "w": wT, "b": bT, "edf": infoT["edf"], "cond": infoT["cond"]})
        log["alpha"] = float(alpha)
        fold_logs.append(log)

    y_patch = y[idx]
    metrics = {name: metrics_from_preds(y_patch, yh) for name, yh in pred.items()}
    return {
        "metrics": metrics,
        "fold_logs": fold_logs,
        "n_eval": int(np.isfinite(y_patch).sum() & np.isfinite(pred["G"]).sum()),
        "n_eval_G": int((np.isfinite(y_patch) & np.isfinite(pred["G"])).sum()),
        "n_eval_P": int((np.isfinite(y_patch) & np.isfinite(pred["P"])).sum()),
        "identical_GP_eval": bool(
            np.array_equal(np.isfinite(pred["G"]), np.isfinite(pred["P"]))
        ),
        "overlap_any": any(int(l.get("train_test_overlap", 0)) > 0 for l in fold_logs),
        "pred": pred,
        "idx": idx,
        "weights": weights,
    }


def refit_global_fold_weights(X: np.ndarray, y: np.ndarray, fold: np.ndarray, *, alpha: float = PROBE_ALPHA):
    w_by, b_by = {}, {}
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=alpha)
        if info["ok"]:
            w_by[int(f)] = w
            b_by[int(f)] = b
    return w_by, b_by
