"""Regenerate T2/T3 scalar oracles on the frozen dual-estimator clouds. No decoder fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.known_curvature_dual_estimator_robustness.config import CELLS, DATA_SEED, D_LAT, K_PRIMARY
from geometry.known_curvature_dual_estimator_robustness.fixtures import ambient_rotation_d28, truth_at
from geometry.known_curvature_dual_estimator_robustness.quadratic import oracle_T2_matched, oracle_T3_uniform
from geometry.known_curvature_dual_estimator_robustness.sampling import apply_noise, make_anchors, sample_training
from geometry.known_curvature_point_patch_fixture_audit.estimator_q import knn_indices
from geometry.known_curvature_point_patch_fixture_audit.oracle import _quad_phi

from .scalar_oracles import matched_q_scalars


def _design_cond(U: np.ndarray) -> float:
    Phi = _quad_phi(np.asarray(U, dtype=np.float64))
    gram = Phi.T @ Phi
    w = np.linalg.eigvalsh(gram)
    w = np.clip(w, 0.0, None)
    return float(w.max() / max(w.min(), 1e-18))


def regenerate_oracles(n_workers_unused: int = 0) -> pd.DataFrame:
    """Deterministic clouds + knn + T2/T3 scalars aligned by fixture/condition/sample_id."""
    del n_workers_unused
    Qrot = ambient_rotation_d28()
    anchors = {nm: make_anchors(nm, Qrot) for nm in ("F0", "F1", "F2", "F4")}
    rows = []
    for fi, sa, no in CELLS:
        anc = anchors[fi]
        tr = sample_training(fi, sa, Qrot, seed=DATA_SEED)
        nz = apply_noise(fi, no, tr, Qrot, seed=DATA_SEED)
        X_obs = nz["X_obs"]
        print(f"[oc] oracles {fi} {sa} {no}", flush=True)
        idx = knn_indices(X_obs, anc["X"], K_PRIMARY, device=None)
        sx = float(nz["s_x"])
        rms = float(nz.get("rms_eps") or 0.0)
        for i in range(len(anc["z"])):
            nidx = idx[i]
            z_nb = tr["z"][nidx]
            U = z_nb - anc["z"][i][None, :]
            t2 = oracle_T2_matched(fi, Qrot, tr["z"], nidx, anc["z"][i])
            rlat = float(np.median(np.linalg.norm(U, axis=1)))
            t3 = oracle_T3_uniform(fi, Qrot, anc["z"][i], max(rlat, 1e-4), seed=1000 + i)
            s2 = matched_q_scalars(t2["B_S"], t2["g"])
            s3 = matched_q_scalars(t3["B_S"], t3["g"])
            pw = truth_at(fi, anc["z"][i], Qrot)
            sp = matched_q_scalars(pw["B_S"], pw["g"])
            rows.append(
                {
                    "cell": f"{fi}_{sa}_{no}",
                    "fixture": fi,
                    "sampling": sa,
                    "noise": no,
                    "anchor_i": i,
                    "sample_id": int(anc["sample_id"][i]),
                    "K_H_T2": s2["K_H_star"],
                    "K_dir_T2": s2["K_dir_star"],
                    "K_H_T3": s3["K_H_star"],
                    "K_dir_T3": s3["K_dir_star"],
                    "K_H_pw": sp["K_H_star"],
                    "K_dir_pw": sp["K_dir_star"],
                    "q_n_eff": int(len(nidx)),
                    "q_design_cond": _design_cond(U),
                    "radius_lat": rlat,
                    "s_x": sx,
                    "rms_eps": rms,
                }
            )
    return pd.DataFrame(rows)
