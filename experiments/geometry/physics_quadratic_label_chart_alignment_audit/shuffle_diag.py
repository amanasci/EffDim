"""Shuffle-label diagnostics: false-positive safety vs null calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_quadratic_label_chart_alignment.config import LIN_GRID, QUAD_GRID
from geometry.physics_quadratic_label_chart_alignment.features import phi2_frob
from geometry.physics_quadratic_label_chart_alignment.models import (
    _design_L,
    _design_UQ,
    _ridge_block,
    mse,
    oof_predict_model,
)
from geometry.physics_quadratic_label_chart_alignment.synthetic import _make_chart

from .truncated_bs import uq_contains_L


def _oof_fixed_with_train(U, y, fold, kind, *, al=10.0, aq=100.0) -> dict[str, Any]:
    n_lin = U.shape[1]
    yhat = np.full(len(y), np.nan)
    train_mses, val_mses, gnorms = [], [], []
    strongest = float(max(QUAD_GRID))
    for f in sorted(set(np.asarray(fold).tolist())):
        te = fold == f
        tr = ~te
        if int(tr.sum()) < 16 or int(te.sum()) < 4:
            continue
        Xtr = _design_UQ(U[tr]) if kind == "UQ" else _design_L(U[tr])
        Xte = _design_UQ(U[te]) if kind == "UQ" else _design_L(U[te])
        aq_ = aq if kind == "UQ" else 1.0
        w, b, info = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=aq_, compute_edf=True)
        if not info.get("ok"):
            continue
        train_mses.append(float(np.mean((y[tr] - (Xtr @ w + b)) ** 2)))
        yhat[te] = Xte @ w + b
        if kind == "UQ":
            gnorms.append(float(np.linalg.norm(w[n_lin:])))
        uniq = sorted(set(fold[tr].tolist()))
        if len(uniq) >= 2:
            vf = uniq[0]
            t2 = tr & (fold != vf)
            vmask = fold == vf
            if int(t2.sum()) >= 8 and int(vmask.sum()) >= 4:
                X2 = _design_UQ(U[t2]) if kind == "UQ" else _design_L(U[t2])
                Xv = _design_UQ(U[vmask]) if kind == "UQ" else _design_L(U[vmask])
                w2, b2, inf2 = _ridge_block(X2, y[t2], n_lin=n_lin, alpha_lin=al, alpha_quad=aq_)
                if inf2.get("ok"):
                    val_mses.append(float(np.mean((y[vmask] - (Xv @ w2 + b2)) ** 2)))
    return {
        "mse_train": float(np.mean(train_mses)) if train_mses else float("nan"),
        "mse_val": float(np.mean(val_mses)) if val_mses else float("nan"),
        "mse_test": mse(y, yhat),
        "alpha_quad": float(aq if kind == "UQ" else 1.0),
        "gamma_norm": float(np.median(gnorms)) if gnorms else float("nan"),
        "edf_note": "fixed-alpha synthetic path (not nested CV)",
        "selected_strongest_aq": kind == "UQ" and abs(float(aq) - strongest) < 1e-12,
    }


def diagnose_one_shuffle(U, y_src, fold, rng) -> dict[str, Any]:
    y = rng.permutation(y_src)
    L = _oof_fixed_with_train(U, y, fold, "L")
    UQ = _oof_fixed_with_train(U, y, fold, "UQ")
    dQ = float(L["mse_test"] - UQ["mse_test"])
    var_y = float(np.var(y))
    return {
        "delta_Q": dQ,
        "g_Q": dQ / max(abs(L["mse_test"]), 1e-12),
        "var_y": var_y,
        "delta_over_var": dQ / max(var_y, 1e-12),
        "mse_L_train": L["mse_train"],
        "mse_UQ_train": UQ["mse_train"],
        "mse_L_val": L["mse_val"],
        "mse_UQ_val": UQ["mse_val"],
        "mse_L_test": L["mse_test"],
        "mse_UQ_test": UQ["mse_test"],
        "coord_rms": float(np.sqrt(np.mean(U * U))),
        "alpha_quad": UQ["alpha_quad"],
        "gamma_norm": UQ["gamma_norm"],
        "uq_selected_max_aq": UQ["selected_strongest_aq"],
        "uq_contains_L": uq_contains_L(),
        "finite_quad_penalty": True,
        "outer_test_in_tuning": False,
    }


def synth_shuffle_battery(n_seeds: int, seed: int) -> pd.DataFrame:
    rng0 = np.random.default_rng(seed)
    U, BS, fold = _make_chart(rng0)
    n, d = U.shape
    a1 = rng0.normal(size=d)
    c = rng0.normal(size=BS.shape[0])
    y_al = U @ a1 + phi2_frob(U) @ (BS.T @ c) + rng0.normal(scale=0.05, size=n)
    rows = []
    for i in range(n_seeds):
        rec = diagnose_one_shuffle(U, y_al, fold, np.random.default_rng(seed + 1000 + i))
        rec["seed"] = int(seed + 1000 + i)
        rec["design"] = "synthetic_fixed_alpha"
        rec["var_unshuffled"] = float(np.var(y_al))
        rows.append(rec)
    # fill unshuffled L mse once
    mse_L_u = _oof_fixed_with_train(U, y_al, fold, "L")["mse_test"]
    for rec in rows:
        rec["mse_L_unshuffled"] = mse_L_u
    return pd.DataFrame(rows)


def real_design_shuffle_one(U, y, fold, rng) -> dict[str, Any]:
    ysh = rng.permutation(np.asarray(y, dtype=np.float64))
    yL, _dL = oof_predict_model(U, ysh, fold, kind="L")
    yUQ, dUQ = oof_predict_model(U, ysh, fold, kind="UQ")
    mse_L, mse_UQ = mse(ysh, yL), mse(ysh, yUQ)
    dQ = float(mse_L - mse_UQ)
    aqs = [f["alpha_quad"] for f in dUQ.get("folds", []) if "alpha_quad" in f]
    return {
        "delta_Q": dQ,
        "g_Q": dQ / max(abs(mse_L), 1e-12),
        "var_y": float(np.var(ysh)),
        "delta_over_var": dQ / max(float(np.var(ysh)), 1e-12),
        "mse_L_test": mse_L,
        "mse_UQ_test": mse_UQ,
        "coord_rms": float(np.sqrt(np.mean(U * U))),
        "alpha_quad_median": float(np.median(aqs)) if aqs else float("nan"),
        "uq_selected_max_aq": bool(aqs) and abs(float(np.median(aqs)) - float(max(QUAD_GRID))) < 1e-12,
        "uq_contains_L": uq_contains_L(),
        "design": "real_nested_cv",
        "outer_test_in_tuning": False,
        "lin_grid_matches_L": True,
    }


def gates_from_deltas(deltas: np.ndarray) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    med = float(np.median(d)) if d.size else float("nan")
    mad = float(np.median(np.abs(d - med))) if d.size else float("nan")
    return {
        "shuffle_no_positive_gain": bool(np.isfinite(med) and med <= 0),
        "shuffle_well_calibrated": bool(np.isfinite(med) and abs(med) < max(0.05, 0.5 * mad if np.isfinite(mad) else 0.05)),
        "median_delta_Q": med,
        "frac_positive": float(np.mean(d > 0)) if d.size else float("nan"),
        "n": int(d.size),
        "mad": mad,
    }
