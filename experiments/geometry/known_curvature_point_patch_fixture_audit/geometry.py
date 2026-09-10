"""Exact sphere-normal second fundamental form and curvature scalars.

Truth calculations are float64. Autodiff, analytic (where closed-form exists),
and central finite differences are independent routes to the same tensors.

Convention
----------
B^S_ab = P_{N,S} ∂_{ab} G  is the Hessian projected to the sphere-normal space.
This is the *differential-geometry* tensor (the decoder estimand).

The frozen local-quadratic pack stores Phi-coefficients S_{ab}=u_a u_b of the
residual, whose Hessian is Hun_aa = 2 S_aa, Hun_ab = S_ab (a≠b). Equivalently
``Hess = 2 * unpack_BS_symmetric(S)``. Production K_H / K_dir scalars are
computed on the unpacked (half-Hessian) tensor. Scoring against T1 uses the
Hessian convention unless a table is explicitly labelled ``unpacked``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-15


def stereo_np(z: np.ndarray) -> np.ndarray:
    """Inverse stereographic R^n → S^n ⊂ R^{n+1}."""
    z = np.asarray(z, dtype=np.float64)
    s = np.sum(z * z, axis=-1, keepdims=True)
    return np.concatenate([2.0 * z, 1.0 - s], axis=-1) / (1.0 + s)


def stereo_inv_np(y: np.ndarray) -> np.ndarray:
    """Stereographic from the north pole: S^n ⊂ R^{n+1} → R^n."""
    y = np.asarray(y, dtype=np.float64)
    return y[..., :-1] / np.clip(1.0 + y[..., -1:], 1e-15, None)


def pad_ambient(x: np.ndarray, D: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] == D:
        return x
    if x.shape[-1] > D:
        raise ValueError(f"pad_ambient: inner dim {x.shape[-1]} > D={D}")
    zeros = np.zeros(x.shape[:-1] + (D - x.shape[-1],), dtype=np.float64)
    return np.concatenate([x, zeros], axis=-1)


def orthonormal_qr(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return q.astype(np.float64)


def rotate_ambient(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Row-vector convention: x_rot = x @ Q.T, so columns of Q are the new axes."""
    return np.asarray(x, dtype=np.float64) @ Q.T


def pack_symmetric_weights(d: int) -> np.ndarray:
    w = []
    for a in range(d):
        for b in range(a, d):
            w.append(1.0 if a == b else np.sqrt(2.0))
    return np.asarray(w, dtype=np.float64)


def unpack_BS_symmetric(BS_flat: np.ndarray, d: int) -> np.ndarray:
    """Frozen FCR unpack: diag = flat, off-diag = flat/2. This is Hess/2."""
    D = BS_flat.shape[0]
    B = np.zeros((D, d, d), dtype=np.float64)
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                B[:, a, a] = BS_flat[:, idx]
            else:
                B[:, a, b] = 0.5 * BS_flat[:, idx]
                B[:, b, a] = 0.5 * BS_flat[:, idx]
            idx += 1
    return B


def pack_BS(B: np.ndarray) -> np.ndarray:
    """Frozen pack: off-diag stored as 2 B_ab."""
    _, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)


def hess_from_bs_flat(BS_flat: np.ndarray, d: int) -> np.ndarray:
    """Phi-coefficient Hessian: Hun_aa = 2 S_aa, Hun_ab = S_ab (a≠b)."""
    return 2.0 * unpack_BS_symmetric(np.asarray(BS_flat, dtype=np.float64), d)


def bs_flat_from_hess(Hess: np.ndarray) -> np.ndarray:
    """Inverse of hess_from_bs_flat (Phi coefficients of 1/2 u^T Hess u)."""
    return pack_BS(0.5 * np.asarray(Hess, dtype=np.float64))


def frobenius_packed(BS_flat: np.ndarray, d: int) -> float:
    w = pack_symmetric_weights(d)
    scaled = np.asarray(BS_flat, dtype=np.float64) * w[None, :]
    return float(np.linalg.norm(scaled))


def metric_sqrt_inv(g: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(g, dtype=np.float64))
    w = np.clip(w, EPS, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def whiten_B(B: np.ndarray, g: np.ndarray) -> np.ndarray:
    """B_w[i,j] = B[a,b] g^{-1/2}[a,i] g^{-1/2}[b,j]."""
    gmh = metric_sqrt_inv(g)
    return np.einsum("Dab,ai,bj->Dij", B, gmh, gmh, optimize=True)


def sphere_normal_projectors(
    G: np.ndarray, J: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return g, P_T, P_{N,S}, G_hat from Jacobian J (D,d) at unit G (D,)."""
    G = np.asarray(G, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    nrm = float(np.linalg.norm(G))
    Gh = G / max(nrm, EPS)
    g = J.T @ J
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    PT = J @ ginv @ J.T
    PNS = np.eye(G.shape[0], dtype=np.float64) - np.outer(Gh, Gh) - PT
    # numerical symmetrize
    PT = 0.5 * (PT + PT.T)
    PNS = 0.5 * (PNS + PNS.T)
    return g, PT, PNS, Gh


def apply_P(P: np.ndarray, B: np.ndarray) -> np.ndarray:
    """P @ B_{:,a,b} for each a,b."""
    D, d, _ = B.shape
    out = np.empty_like(B)
    for a in range(d):
        out[:, a, :] = P @ B[:, a, :]
    return out


def curvature_from_B(B: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray | float]:
    """Metric-correct H^S, traceless part, and whitened directional curvature."""
    B = np.asarray(B, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    d = g.shape[0]
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    # H^S = (1/d) g^{ab} B_ab
    H = (1.0 / d) * np.einsum("ab,Dab->D", ginv, B)
    # B = g ⊗ H + ring B, with tr_g ring B = 0
    mean_part = np.einsum("ab,D->Dab", g, H)
    Btf = B - mean_part
    Bw = whiten_B(B, g)
    tr = np.trace(Bw, axis1=1, axis2=2)  # (D,)
    fro2 = float(np.sum(Bw * Bw))
    tr2 = float(np.dot(tr, tr))
    kdir = (2.0 * fro2 + tr2) / float(d * (d + 2))
    Btfw = whiten_B(Btf, g)
    ktf = (2.0 * float(np.sum(Btfw * Btfw))) / float(d * (d + 2))
    kh = float(np.dot(H, H))  # ||H||^2; E||H||^2 for constant H equals ||H||^2
    return {
        "B": B,
        "H": H,
        "Btf": Btf,
        "Bw": Bw,
        "K_dir": float(kdir),
        "K_tf": float(ktf),
        "K_H2": kh,
        "K_H": float(np.sqrt(max(kh, 0.0))),
        "H_norm": float(np.linalg.norm(H)),
        "B_fro": float(np.linalg.norm(B)),
        "Btf_fro": float(np.linalg.norm(Btf)),
        "g": g,
        "ginv": ginv,
    }


def kdir_from_pair(BA: np.ndarray, BB: np.ndarray, gA: np.ndarray, gB: np.ndarray | None = None) -> dict[str, float]:
    """Split-cross directional curvature on Hessian tensors (not clamped)."""
    if gB is None:
        gB = gA
    d = gA.shape[0]
    Aw = whiten_B(BA, gA)
    Bw = whiten_B(BB, gB)
    trA = np.trace(Aw, axis1=1, axis2=2)
    trB = np.trace(Bw, axis1=1, axis2=2)
    fro = float(np.sum(Aw * Bw))
    tr = float(np.dot(trA, trB))
    kdir = (2.0 * fro + tr) / float(d * (d + 2))
    HA = (1.0 / d) * trA
    HB = (1.0 / d) * trB
    # After whitening, H is the Euclidean trace/d of Bw.
    # Ambient H^S uses metric trace; for orthonormal charts these agree.
    kh = float(np.dot(HA, HB))
    return {
        "K_dir_cross": kdir,
        "K_H_cross": kh,
        "K_tf_cross": kdir - kh,
        "B_fro_cross": fro,
    }


def jacobian_fd(fn: Callable[[np.ndarray], np.ndarray], z: np.ndarray, h: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    d = z.shape[0]
    G0 = np.asarray(fn(z), dtype=np.float64)
    J = np.zeros((G0.shape[0], d), dtype=np.float64)
    for a in range(d):
        zp, zm = z.copy(), z.copy()
        zp[a] += h
        zm[a] -= h
        J[:, a] = (fn(zp) - fn(zm)) / (2.0 * h)
    return J


def hessian_fd(fn: Callable[[np.ndarray], np.ndarray], z: np.ndarray, h: float) -> np.ndarray:
    """Central mixed second derivatives, independent of autodiff."""
    z = np.asarray(z, dtype=np.float64)
    d = z.shape[0]
    G0 = np.asarray(fn(z), dtype=np.float64)
    D = G0.shape[0]
    H = np.zeros((D, d, d), dtype=np.float64)
    for a in range(d):
        for b in range(a, d):
            zpp, zpm, zmp, zmm = z.copy(), z.copy(), z.copy(), z.copy()
            zpp[a] += h
            zpp[b] += h
            zpm[a] += h
            zpm[b] -= h
            zmp[a] -= h
            zmp[b] += h
            zmm[a] -= h
            zmm[b] -= h
            val = (fn(zpp) - fn(zpm) - fn(zmp) + fn(zmm)) / (4.0 * h * h)
            H[:, a, b] = val
            H[:, b, a] = val
    return H


def geometry_from_jets(G: np.ndarray, J: np.ndarray, Hess: np.ndarray) -> dict:
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    B = apply_P(PNS, Hess)
    out = curvature_from_B(B, g)
    out.update({"G": G, "J": J, "Hess": Hess, "P_T": PT, "P_NS": PNS, "G_hat": Gh})
    return out


def mc_directional_k(B_w: np.ndarray, n: int, seed: int) -> float:
    """Monte Carlo E_{||v||=1} ||B(v,v)||^2 on an already-whitened tensor."""
    rng = np.random.default_rng(seed)
    d = B_w.shape[1]
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    # B(v,v)_D = B_Dab v_a v_b
    vv = np.einsum("Dab,na,nb->nD", B_w, g, g, optimize=True)
    return float(np.mean(np.sum(vv * vv, axis=1)))


def tangent_orthogonality(B: np.ndarray, J: np.ndarray, G: np.ndarray) -> dict[str, float]:
    """B^S ⟂ x and B^S ⟂ T_x."""
    Gh = G / max(float(np.linalg.norm(G)), EPS)
    rad = float(np.max(np.abs(np.einsum("Dab,D->ab", B, Gh))))
    tan = float(np.max(np.abs(np.einsum("Dab,Di->iab", B, J))))
    return {"max_abs_radial": rad, "max_abs_tangent": tan}


def radial_identity_unit(G: np.ndarray, J: np.ndarray, Hess: np.ndarray) -> dict[str, float]:
    """For maps into the unit sphere, G·G=1 ⇒ G·J=0 and G·Hess_ab + J_a·J_b = 0."""
    Gh = G / max(float(np.linalg.norm(G)), EPS)
    gdotJ = float(np.max(np.abs(J.T @ Gh)))
    ident = np.einsum("Dab,D->ab", Hess, Gh) + J.T @ J
    return {"max_abs_GJ": gdotJ, "max_abs_radial_hess_identity": float(np.max(np.abs(ident)))}


def rel_err(a: np.ndarray | float, b: np.ndarray | float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = float(np.linalg.norm((a - b).ravel()))
    den = max(float(np.linalg.norm(b.ravel())), EPS)
    return num / den


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def directional_tensor_cos(Bhat: np.ndarray, Btrue: np.ndarray, dirs: np.ndarray) -> float:
    """E_v cos(Bhat(v,v), Btrue(v,v)) on seeded unit tangent directions (whitened coords)."""
    vals = []
    for v in dirs:
        ha = np.einsum("Dab,a,b->D", Bhat, v, v)
        hb = np.einsum("Dab,a,b->D", Btrue, v, v)
        vals.append(cosine(ha, hb))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
