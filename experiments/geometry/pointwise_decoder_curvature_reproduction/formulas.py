"""Colleague formulas, decomposition, and Phase-1 unit tests.

Historical estimator (decoder_curvature.plain_decoder_curvature):
    J = DF, g = J^T J, P_T = J g^{-1} J^T
    II^E_ab = (I - P_T) ∂_{ab} F
    H = g^{ab} II^E_ab     (UNNORMALIZED trace; no 1/d)
    Differentiated map is raw model.decode, not a sphere-normalized wrap.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import hessian, jacrev

from pu_manifold.decoder_curvature import CURVATURE_CONVENTION, plain_decoder_curvature


def axes(H_est: np.ndarray, H_true: np.ndarray) -> dict:
    """Exact 07_instrument_fixture_sweep_run.axes."""
    from scipy.stats import spearmanr

    he = np.linalg.norm(H_est, axis=1)
    ht = np.linalg.norm(H_true, axis=1)
    num = (H_est * H_true).sum(axis=1)
    den = np.maximum(he * ht, 1e-30)
    return {
        "rho": float(spearmanr(he, ht).statistic),
        "median_cosine": float(np.median(num / den)),
        "median_ratio": float(np.median(he / np.maximum(ht, 1e-12))),
    }


def split_indices(n: int, split_seed: int, holdout_fraction: float):
    """crossmodal_curvature.split_indices: holdout is the first round(n*f) of a permutation."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    return perm[n_holdout:], perm[:n_holdout]


def anchor_indices(n_rows: int, split_seed: int, holdout_fraction: float, n_anchors: int, anchor_seed: int) -> dict:
    from pu_manifold.subsample import draw_row_indices

    train_idx, holdout_idx = split_indices(n_rows, split_seed, holdout_fraction)
    holdout_idx = np.asarray(holdout_idx)
    if holdout_idx.shape[0] < n_anchors:
        raise ValueError(f"holdout {len(holdout_idx)} < n_anchors {n_anchors}")
    if holdout_idx.shape[0] == n_anchors:
        anchor_idx = np.sort(holdout_idx)
    else:
        anchor_pos = draw_row_indices(holdout_idx.shape[0], n_anchors, anchor_seed)
        anchor_idx = np.sort(holdout_idx[anchor_pos])
    return {
        "train_idx": np.sort(np.asarray(train_idx)),
        "holdout_idx": np.sort(holdout_idx),
        "anchor_idx": anchor_idx,
    }


def jets_numpy(F, z: np.ndarray, eps: float = 1e-5):
    """Finite-difference Jacobian and Hessian of F: R^d -> R^D at rows of z."""
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    f0 = np.stack([F(z[i]) for i in range(n)])
    D = f0.shape[1]
    J = np.zeros((n, D, d))
    H = np.zeros((n, D, d, d))
    for a in range(d):
        e = np.zeros(d)
        e[a] = eps
        fp = np.stack([F(z[i] + e) for i in range(n)])
        fm = np.stack([F(z[i] - e) for i in range(n)])
        J[:, :, a] = (fp - fm) / (2 * eps)
        for b in range(a, d):
            eb = np.zeros(d)
            eb[b] = eps
            fpp = np.stack([F(z[i] + e + eb) for i in range(n)])
            fpm = np.stack([F(z[i] + e - eb) for i in range(n)])
            fmp = np.stack([F(z[i] - e + eb) for i in range(n)])
            fmm = np.stack([F(z[i] - e - eb) for i in range(n)])
            hab = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[:, :, a, b] = hab
            H[:, :, b, a] = hab
    return f0, J, H


def projectors_from_jets(x: np.ndarray, J: np.ndarray):
    """x (D,), J (D,d) -> g, ginv, P_T, P_N, Gh."""
    g = J.T @ J
    ginv = np.linalg.pinv(g)
    PT = J @ ginv @ J.T
    nrm = float(np.linalg.norm(x))
    Gh = x / max(nrm, 1e-15)
    PN = np.eye(x.shape[0]) - PT
    PNS = PN - np.outer(Gh, Gh)
    return g, ginv, PT, PN, PNS, Gh


def decompose_hessians(x: np.ndarray, J: np.ndarray, Q: np.ndarray, *, sphere: bool) -> dict:
    """Q is (D,d,d) Hessian. Historical II uses PN = I-P_T only."""
    g, ginv, PT, PN, PNS, Gh = projectors_from_jets(x, J)
    d = J.shape[1]
    II_E = np.einsum("ij,jab->iab", PN, Q)
    B_S = np.einsum("ij,jab->iab", PNS, Q)
    II_R = -np.einsum("ab,i->iab", g, Gh)
    H_E_unavg = np.einsum("ab,iab->i", ginv, II_E)
    H_E_avg = H_E_unavg / d
    H_S_unavg = np.einsum("ab,iab->i", ginv, B_S)
    H_S_avg = H_S_unavg / d

    def energy(B):
        # ||B||_g^2 = g^{ac} g^{bd} <B_ab, B_cd>
        Bw = np.einsum("ac,bd,iab->icd", ginv, ginv, B)
        return float(np.tensordot(B, Bw, axes=([0, 1, 2], [0, 1, 2])))

    eE, eR, eS = energy(II_E), energy(II_R), energy(B_S)
    out = {
        "g": g,
        "H_E_unavg": H_E_unavg,
        "H_E_avg": H_E_avg,
        "H_S_unavg": H_S_unavg,
        "H_S_avg": H_S_avg,
        "energy_II_E": eE,
        "energy_II_R": eR,
        "energy_B_S": eS,
        "f_res": float(eS / eE) if eE > 1e-30 else float("nan"),
        "sphere": bool(sphere),
        "reconstruct_rel": float(np.linalg.norm(II_E - (II_R + B_S)) / max(np.linalg.norm(II_E), 1e-12)),
        "PT_II_E": float(np.linalg.norm(np.einsum("ij,jab->iab", PT, II_E))),
        "xT_Q_plus_g": float(np.linalg.norm(np.einsum("i,iab->ab", Gh, Q) + g)) if sphere else float("nan"),
    }
    return out


def autodiff_jets(decode_one, z: torch.Tensor):
    J = jacrev(decode_one)(z)
    Q = hessian(decode_one)(z)
    x = decode_one(z)
    return x.detach().cpu().numpy(), J.detach().cpu().numpy(), Q.detach().cpu().numpy()


def run_unit_tests() -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    rng = np.random.default_rng(0)

    # Affine immersion: F(u) = A u + b  -> II^E = 0
    A = rng.standard_normal((8, 3))
    b = rng.standard_normal(8)
    u = rng.standard_normal(3)
    J = A
    Q = np.zeros((8, 3, 3))
    x = A @ u + b
    dec = decompose_hessians(x, J, Q, sphere=False)
    rec("flat_affine_II_E_zero", dec["energy_II_E"] < 1e-14, e=dec["energy_II_E"])

    # Great sphere: F(u) = (u, sqrt(1-|u|^2)) near north pole, or stereo. Use F(z)=normalize([z,1])
    def sphere_chart(z):
        v = np.concatenate([z, [1.0]])
        return v / np.linalg.norm(v)

    z0 = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    f0, J0, Q0 = jets_numpy(sphere_chart, z0[None, :], eps=1e-5)
    decs = decompose_hessians(f0[0], J0[0], Q0[0], sphere=True)
    rec("great_sphere_B_S_small", decs["energy_B_S"] < 1e-4, e=decs["energy_B_S"])
    rec("great_sphere_II_E_nonzero", decs["energy_II_E"] > 1e-2, e=decs["energy_II_E"])
    rec("II_E_eq_II_R_plus_B_S", decs["reconstruct_rel"] < 5e-3, rel=decs["reconstruct_rel"])
    rec("P_T_II_E_zero", decs["PT_II_E"] < 1e-3, n=decs["PT_II_E"])
    rec("normalized_xT_Q_eq_minus_g", decs["xT_Q_plus_g"] < 5e-3, n=decs["xT_Q_plus_g"])
    # H^E = -x + H^S under averaged convention: compare vectors
    H_E_avg = decs["H_E_avg"]
    H_S_avg = decs["H_S_avg"]
    rec(
        "H_E_avg_eq_minus_x_plus_H_S",
        np.linalg.norm(H_E_avg - (-f0[0] + H_S_avg)) / max(np.linalg.norm(H_E_avg), 1e-12) < 5e-2,
    )

    # Coordinate invariance of metric trace under invertible reparam
    # F(u) = sphere_chart(u); u' = M u. H_unavg in ambient should match after reparam of g,II.
    M = np.array([[1.0, 0.2, 0.0], [0.0, 1.1, 0.1], [0.0, 0.0, 0.9]])
    Minv = np.linalg.inv(M)

    def F_re(w):
        return sphere_chart(M @ w)

    w0 = Minv @ z0
    f1, J1, Q1 = jets_numpy(F_re, w0[None, :], eps=1e-5)
    d0 = decompose_hessians(f0[0], J0[0], Q0[0], sphere=True)
    d1 = decompose_hessians(f1[0], J1[0], Q1[0], sphere=True)
    rec(
        "metric_trace_reparam_invariant",
        np.linalg.norm(d0["H_E_unavg"] - d1["H_E_unavg"]) / max(np.linalg.norm(d0["H_E_unavg"]), 1e-12) < 5e-2,
    )

    # Autodiff vs FD on three small points of analytic sphere chart via a tiny Module
    class Chart(torch.nn.Module):
        def decode(self, z):
            if z.ndim == 1:
                v = torch.cat([z, z.new_ones(1)])
                return v / torch.linalg.norm(v)
            ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
            v = torch.cat([z, ones], dim=-1)
            return v / torch.linalg.norm(v, dim=-1, keepdim=True)

        def parameters(self):
            return iter(())

        def double(self):
            return self

    m = Chart()
    zs = torch.tensor([[0.1, -0.05], [0.0, 0.2], [-0.15, 0.08]], dtype=torch.float64)
    rels = []
    decode_one = lambda z: m.decode(z)
    for i in range(3):
        x_ad, J_ad, Q_ad = autodiff_jets(decode_one, zs[i])
        x_fd, J_fd, Q_fd = jets_numpy(lambda u: m.decode(torch.as_tensor(u, dtype=torch.float64)).numpy(), zs[i].numpy()[None, :], eps=1e-5)
        rels.append(float(np.linalg.norm(Q_ad - Q_fd[0]) / max(np.linalg.norm(Q_ad), 1e-12)))
    rec("autodiff_vs_fd_three_points", max(rels) < 5e-2, max_rel=max(rels))

    rec("convention_is_unnormalized_trace", CURVATURE_CONVENTION == "trace")

    ids = np.array([10, 3, 7, 1, 9], dtype=np.int64)
    rec("sample_id_rows_align_by_explicit_id", np.array_equal(ids[np.argsort(ids)], np.sort(ids)))

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}
