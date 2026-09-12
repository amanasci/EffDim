"""Q-residual task-aligned cross energy from reconstructed split-half B^S."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric
from geometry.physics_activation_atlas.effdim_curvature_metrics import cross_metric_pair
from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank

from .algebra import contract_w_B, cross_energy_g, energy_g, projectors, sphere_normal_w, trace_g
from .audit import load_q_frame
from .config import D_LAT, K, N_Q_SPLITS, Q_PRODUCTION_SEED


def q_task_aligned_anchor(
    X: np.ndarray,
    neigh: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    w: np.ndarray,
    *,
    ai: int,
    n_splits: int = N_Q_SPLITS,
) -> dict[str, Any]:
    Xloc = np.asarray(X[neigh], dtype=np.float64)
    Jd = np.asarray(J, dtype=np.float64)[:, :D_LAT]
    x0 = np.asarray(x0, dtype=np.float64)
    proj = projectors(x0, Jd)
    wQ = sphere_normal_w(w, proj)
    fits = _fit_rank(Xloc, x0, Jd, D_LAT, K, n_splits, Q_PRODUCTION_SEED, int(ai))
    if not fits:
        return {"ok": False, "E_Q_cross": float("nan"), "T_Q_cross": float("nan"), "K_H_cross": float("nan")}
    e_cross = []
    t_cross = []
    kh = []
    eA, eB = [], []
    for f in fits:
        BA = unpack_BS_symmetric(f["BS_flat_A"], D_LAT)
        BB = unpack_BS_symmetric(f["BS_flat_B"], D_LAT)
        bA = contract_w_B(wQ, BA)
        bB = contract_w_B(wQ, BB)
        e_cross.append(cross_energy_g(bA, bB, proj["ginv"]))
        t_cross.append(trace_g(bA, proj["ginv"]) * trace_g(bB, proj["ginv"]))
        kh.append(float(f["K_H_cross"]))
        eA.append(energy_g(bA, proj["ginv"]))
        eB.append(energy_g(bB, proj["ginv"]))
        f.pop("BS_flat_A", None)
        f.pop("BS_flat_B", None)
        f.pop("H_mean", None)
    # Mean of split-wise cross energies. Not (mean B)^2.
    return {
        "ok": True,
        "E_Q_cross": float(np.mean(e_cross)),
        "T_Q_cross": float(np.mean(t_cross)),
        "K_H_cross_recon": float(np.mean(kh)),
        "split_E_reliability": float(np.corrcoef(eA, eB)[0, 1]) if len(eA) > 1 else float("nan"),
        "n_splits": int(len(fits)),
        "averaged_then_squared": False,
        "clamped": False,
        "kind": "Q_residual_sphere_normal_cross",
        "wQ_norm": float(np.linalg.norm(wQ)),
    }


def q_for_weights(
    X: np.ndarray,
    neigh: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    weights: dict[str, np.ndarray],
    *,
    ai: int,
    n_splits: int = N_Q_SPLITS,
) -> dict[str, Any]:
    """One Q reconstruction, contracted against every probe weight. Tensors discarded."""
    Xloc = np.asarray(X[neigh], dtype=np.float64)
    Jd = np.asarray(J, dtype=np.float64)[:, :D_LAT]
    x0 = np.asarray(x0, dtype=np.float64)
    proj = projectors(x0, Jd)
    wQ = {t: sphere_normal_w(w, proj) for t, w in weights.items()}
    fits = _fit_rank(Xloc, x0, Jd, D_LAT, K, n_splits, Q_PRODUCTION_SEED, int(ai))
    acc = {t: {"e": [], "t": []} for t in weights}
    kh = []
    if not fits:
        return {"ok": False, "K_H_cross_recon": float("nan"), "per_target": {t: {"E_Q_cross": float("nan")} for t in weights}}
    for f in fits:
        BA = unpack_BS_symmetric(f["BS_flat_A"], D_LAT)
        BB = unpack_BS_symmetric(f["BS_flat_B"], D_LAT)
        kh.append(float(f["K_H_cross"]))
        for t, wq in wQ.items():
            bA = contract_w_B(wq, BA)
            bB = contract_w_B(wq, BB)
            acc[t]["e"].append(cross_energy_g(bA, bB, proj["ginv"]))
            acc[t]["t"].append(trace_g(bA, proj["ginv"]) * trace_g(bB, proj["ginv"]))
        f.pop("BS_flat_A", None)
        f.pop("BS_flat_B", None)
        f.pop("H_mean", None)
    per = {
        t: {
            "E_Q_cross": float(np.mean(v["e"])),
            "T_Q_cross": float(np.mean(v["t"])),
            "averaged_then_squared": False,
            "clamped": False,
        }
        for t, v in acc.items()
    }
    return {"ok": True, "K_H_cross_recon": float(np.mean(kh)), "n_splits": int(len(fits)), "per_target": per}


def eval_q_all_weights(shared: dict, weights: dict[str, np.ndarray], *, n_splits: int, n_use: int | None = None) -> dict[str, Any]:
    sids = shared["sids"][: n_use or len(shared["sids"])]
    n = len(sids)
    out = {t: {"E_Q_cross": np.full(n, np.nan), "T_Q_cross": np.full(n, np.nan)} for t in weights}
    KH = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    for i, sid in enumerate(sids):
        x0, J = load_q_frame(shared, int(sid))
        ai = shared["sid_to_ai"][int(sid)]
        rec = q_for_weights(shared["X"], shared["neigh"][i], x0, J, weights, ai=ai, n_splits=n_splits)
        KH[i] = rec.get("K_H_cross_recon", np.nan)
        ok[i] = bool(rec.get("ok"))
        for t, vals in rec.get("per_target", {}).items():
            out[t]["E_Q_cross"][i] = vals.get("E_Q_cross", np.nan)
            out[t]["T_Q_cross"][i] = vals.get("T_Q_cross", np.nan)
    return {"per_target": out, "K_H_cross_recon": KH, "ok": ok}
