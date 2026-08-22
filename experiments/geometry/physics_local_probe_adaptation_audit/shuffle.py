"""End-to-end label shuffle on audit anchors (G and P only)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .config import PROBE_ALPHA, SEED
from .io_util import p_mc


def _local_var_neigh(y: np.ndarray, neigh_idx: np.ndarray) -> float:
    idx = np.asarray(neigh_idx, dtype=int)
    yy = y[idx]
    m = np.isfinite(yy)
    return float(np.var(yy[m])) if int(m.sum()) >= 2 else float("nan")


def _controls_for_anchor(
    sid: int,
    ai: int,
    *,
    y: np.ndarray,
    neigh: np.ndarray,
    frozen: pd.Series,
    frozen_eval: pd.Series,
) -> dict:
    idx = np.asarray(neigh[ai], dtype=int)
    return {
        "log_knn_radius": float(frozen.loc[int(sid)]),
        "local_label_variance": _local_var_neigh(y, idx),
        "local_evaluation_count": float(frozen_eval.loc[int(sid)]),
    }


def audit_subset(sids: list[int], n: int, seed: int) -> list[int]:
    scored = [(hashlib.sha256(f"lpa_audit:{seed}:{int(s)}".encode()).hexdigest(), int(s)) for s in sids]
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]


def _fold_ops(X: np.ndarray, fold: np.ndarray, alpha: float) -> list[dict]:
    ops = []
    for f in sorted(set(fold.tolist())):
        tr = np.where(fold != f)[0]
        Xtr = np.asarray(X[tr], dtype=np.float64)
        xm = Xtr.mean(axis=0)
        Xc = Xtr - xm
        XtX = Xc.T @ Xc
        np.fill_diagonal(XtX, np.diag(XtX) + float(alpha))
        try:
            L = np.linalg.cholesky(XtX)
        except np.linalg.LinAlgError:
            L = None
        ops.append({"f": int(f), "tr": tr, "te": np.where(fold == f)[0], "L": L, "Xc": Xc, "xm": xm})
    return ops


def global_oof_from_ops(X: np.ndarray, y: np.ndarray, ops: list[dict]) -> np.ndarray:
    yhat = np.full(len(y), np.nan)
    for op in ops:
        if op["L"] is None:
            continue
        ytr = y[op["tr"]]
        m = np.isfinite(ytr)
        yc = ytr[m] - float(ytr[m].mean())
        Xc = op["Xc"][m]
        w = np.linalg.solve(op["L"].T, np.linalg.solve(op["L"], Xc.T @ yc))
        b = float(ytr[m].mean() - op["xm"] @ w)
        pred = X[op["te"]] @ w + b
        yhat[op["te"]] = pred
    return yhat


def patch_dMSE_anchor(
    X: np.ndarray,
    y: np.ndarray,
    yhat_g: np.ndarray,
    fold: np.ndarray,
    neigh_idx: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict

    idx = np.asarray(neigh_idx, dtype=int)
    yp = np.full(len(idx), np.nan)
    for f in sorted(set(fold[idx].tolist())):
        te = idx[fold[idx] == f]
        tr = idx[fold[idx] != f]
        if len(tr) < 32 or len(te) < 8:
            continue
        w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=alpha)
        if info["ok"]:
            yp[np.isin(idx, te)] = ridge_predict(X[te], w, b)
    m = np.isfinite(y[idx]) & np.isfinite(yhat_g[idx]) & np.isfinite(yp)
    if m.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    mse_g = float(np.mean((y[idx][m] - yhat_g[idx][m]) ** 2))
    mse_p = float(np.mean((y[idx][m] - yp[m]) ** 2))
    return mse_g - mse_p, mse_g, mse_p


def run_shuffle(
    *,
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    neigh: np.ndarray,
    sid_to_ai: dict[int, int],
    audit_sids: list[int],
    kh: pd.Series,
    log_radius: pd.Series,
    eval_count: pd.Series,
    n_perm: int,
    seed: int,
    alpha: float = PROBE_ALPHA,
) -> dict:
    ops = _fold_ops(X, fold, alpha)
    rng = np.random.default_rng(seed)
    rows = []
    obs_dm = []
    obs_kh = []
    obs_ctrl = []
    for sid in audit_sids:
        ai = sid_to_ai[int(sid)]
        dm, _, _ = patch_dMSE_anchor(X, y, global_oof_from_ops(X, y, ops), fold, neigh[ai], alpha=alpha)
        obs_dm.append(dm)
        obs_kh.append(float(kh.loc[int(sid)]))
        obs_ctrl.append(_controls_for_anchor(sid, ai, y=y, neigh=neigh, frozen=log_radius, frozen_eval=eval_count))
    obs_tab = pd.DataFrame({"sample_id": audit_sids, "dMSE_GP": obs_dm, "K_H_cross": obs_kh})
    for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count"):
        obs_tab[c] = [obs_ctrl[i][c] for i in range(len(audit_sids))]
    obs_rho = float(
        associate(
            obs_tab.K_H_cross.to_numpy(float),
            obs_tab.dMSE_GP.to_numpy(float),
            control_matrix(obs_tab),
        )["controlled"]
    )

    null = []
    for b in range(n_perm):
        yp = rng.permutation(y)
        yhat_g = global_oof_from_ops(X, yp, ops)
        dms = []
        khs = []
        ctrls = []
        for sid in audit_sids:
            ai = sid_to_ai[int(sid)]
            dm, _, _ = patch_dMSE_anchor(X, yp, yhat_g, fold, neigh[ai], alpha=alpha)
            dms.append(dm)
            khs.append(float(kh.loc[int(sid)]))
            ctrls.append(_controls_for_anchor(sid, ai, y=yp, neigh=neigh, frozen=log_radius, frozen_eval=eval_count))
        tab = pd.DataFrame({"dMSE_GP": dms, "K_H_cross": khs})
        for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count"):
            tab[c] = [ctrls[i][c] for i in range(len(audit_sids))]
        null.append(
            float(
                associate(
                    tab.K_H_cross.to_numpy(float),
                    tab.dMSE_GP.to_numpy(float),
                    control_matrix(tab),
                )["controlled"]
            )
        )
        rows.append({"perm": b, "rho_ctl": null[-1]})
        if (b + 1) % 50 == 0:
            print(f"[lpa-audit] shuffle {b+1}/{n_perm}", flush=True)
    arr = np.asarray(null, float)
    if np.isfinite(obs_rho) and obs_rho > 0:
        b_count = int(np.sum(arr >= obs_rho))
    elif np.isfinite(obs_rho) and obs_rho < 0:
        b_count = int(np.sum(arr <= obs_rho))
    else:
        b_count = n_perm
    p_one = p_mc(b_count, n_perm)
    return {
        "rows": rows,
        "obs_rho_ctl": obs_rho,
        "n_audit": len(audit_sids),
        "n_perm": n_perm,
        "p_mc": p_one,
        "null_mean": float(np.nanmean(arr)),
        "pass": bool(np.isfinite(obs_rho) and (p_one > 0.05 or abs(obs_rho) <= 0.02)),
        "inconclusive": bool(abs(obs_rho) <= 0.02),
        "skipped": False,
    }
