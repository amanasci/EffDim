"""Known generators F0–F5 with analytic, autodiff, and finite-difference truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import (
    AMBIENT_ROTATION_HASH16,
    AMBIENT_ROTATION_SEED,
    D_AMB,
    D_LAT,
    F1_C,
    F3_R2,
    F3_S2,
    F4_AMPLITUDES,
    F4_BUMP_SCALE,
    F4_CENTERS,
    F4_WIDTHS,
    F5_WIDTHS,
    FD_STEP,
    TOL_ANALYTIC_REL,
    TOL_AUTODIFF_FD_REL,
    TOL_F0_BS,
    TOL_INVARIANT,
    TOL_MC_KDIR,
    TOL_ORTH,
    TOL_RADIAL,
)
from .geometry import (
    apply_P,
    curvature_from_B,
    geometry_from_jets,
    hessian_fd,
    jacobian_fd,
    mc_directional_k,
    orthonormal_qr,
    pad_ambient,
    radial_identity_unit,
    rel_err,
    rotate_ambient,
    sphere_normal_projectors,
    stereo_np,
    tangent_orthogonality,
)
from .io_util import sha256_bytes

CLIFFORD_P = 8
CLIFFORD_Q = 8


_Q_PATH = Path(__file__).resolve().parent / "ambient_rotation_Q.npy"


def ambient_rotation(D: int = D_AMB, seed: int = AMBIENT_ROTATION_SEED) -> np.ndarray:
    """Load the frozen ambient rotation (QR is not bit-stable across LAPACK builds)."""
    if not _Q_PATH.is_file():
        raise RuntimeError(f"missing frozen ambient rotation {_Q_PATH}")
    Q = np.load(_Q_PATH).astype(np.float64)
    if Q.shape != (D, D):
        raise RuntimeError(f"ambient rotation shape {Q.shape} != {(D, D)}")
    h = sha256_bytes(np.ascontiguousarray(Q).tobytes())[:16]
    if h != AMBIENT_ROTATION_HASH16:
        raise RuntimeError(f"ambient rotation hash {h} != frozen {AMBIENT_ROTATION_HASH16}")
    return Q


def _as2d(z: np.ndarray, d: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        if z.shape[0] != d:
            raise ValueError(f"expected latent dim {d}, got {z.shape}")
        return z[None, :], True
    return z, False


def _unbatch(x: np.ndarray, squeezed: bool) -> np.ndarray:
    return x[0] if squeezed else x


# -------------------- numpy generators --------------------


def f0_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    z2, sq = _as2d(z, D_LAT)
    y = pad_ambient(stereo_np(z2), D_AMB)
    return _unbatch(rotate_ambient(y, Q), sq)


def f1_np(z: np.ndarray, Q: np.ndarray, c: float = F1_C) -> np.ndarray:
    r = float(np.sqrt(max(1.0 - c * c, 0.0)))
    z2, sq = _as2d(z, D_LAT)
    phi = stereo_np(z2)
    y = np.concatenate([r * phi, np.full((z2.shape[0], 1), c, dtype=np.float64)], axis=1)
    y = pad_ambient(y, D_AMB)
    return _unbatch(rotate_ambient(y, Q), sq)


def _clifford_np(z: np.ndarray, Q: np.ndarray, r2: float, s2: float) -> np.ndarray:
    r, s = float(np.sqrt(r2)), float(np.sqrt(s2))
    z2, sq = _as2d(z, D_LAT)
    u, v = z2[:, :CLIFFORD_P], z2[:, CLIFFORD_P:]
    xu, yv = stereo_np(u), stereo_np(v)
    y = np.concatenate([r * xu, s * yv], axis=1)
    y = pad_ambient(y, D_AMB)
    return _unbatch(rotate_ambient(y, Q), sq)


def f2_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return _clifford_np(z, Q, 0.5, 0.5)


def f3_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return _clifford_np(z, Q, F3_R2, F3_S2)


def bump_field(z: np.ndarray, widths: tuple[float, ...]) -> np.ndarray:
    z2, sq = _as2d(z, D_LAT)
    h = np.zeros(z2.shape[0], dtype=np.float64)
    C = np.asarray(F4_CENTERS, dtype=np.float64)
    A = np.asarray(F4_AMPLITUDES, dtype=np.float64)
    W = np.asarray(widths, dtype=np.float64)
    for i in range(len(A)):
        d2 = np.sum((z2 - C[i]) ** 2, axis=1)
        h += A[i] * np.exp(-d2 / (2.0 * W[i] ** 2))
    return h[0] if sq else h


def _bumped_np(z: np.ndarray, Q: np.ndarray, widths: tuple[float, ...]) -> np.ndarray:
    z2, sq = _as2d(z, D_LAT)
    phi = stereo_np(z2)
    h = bump_field(z2, widths)
    extra = (F4_BUMP_SCALE * h)[:, None]
    y = np.concatenate([phi, extra], axis=1)
    n = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.clip(n, 1e-15, None)
    y = pad_ambient(y, D_AMB)
    return _unbatch(rotate_ambient(y, Q), sq)


def f4_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return _bumped_np(z, Q, F4_WIDTHS)


def f5_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return _bumped_np(z, Q, F5_WIDTHS)


GENERATORS_NP: dict[str, Callable] = {
    "F0": f0_np,
    "F1": f1_np,
    "F2": f2_np,
    "F3": f3_np,
    "F4": f4_np,
    "F5": f5_np,
}


# -------------------- torch generators (autodiff truth) --------------------


def stereo_jacobian_np(z: np.ndarray) -> np.ndarray:
    """Exact Jacobian of inverse stereographic φ: R^n → S^n ⊂ R^{n+1}."""
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    s = float(np.dot(z, z))
    den = 1.0 + s
    J = np.zeros((n + 1, n), dtype=np.float64)
    J[:n, :] = (2.0 / den) * np.eye(n) - (4.0 / (den * den)) * np.outer(z, z)
    J[n, :] = -4.0 * z / (den * den)
    return J


def _embed_J(J_low: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Pad a low-dimensional Jacobian into R^D and apply the ambient rotation."""
    D = Q.shape[0]
    Jpad = np.zeros((D, J_low.shape[1]), dtype=np.float64)
    Jpad[: J_low.shape[0], :] = J_low
    return Q @ Jpad


def stereo_t(z: torch.Tensor) -> torch.Tensor:
    s = (z * z).sum(dim=-1, keepdim=True)
    return torch.cat([2.0 * z, 1.0 - s], dim=-1) / (1.0 + s)


def pad_t(x: torch.Tensor, D: int) -> torch.Tensor:
    if x.shape[-1] == D:
        return x
    z = x.new_zeros(x.shape[:-1] + (D - x.shape[-1],))
    return torch.cat([x, z], dim=-1)


class FixtureMap(torch.nn.Module):
    def __init__(self, name: str, Q: np.ndarray):
        super().__init__()
        self.name = name
        self.register_buffer("Q", torch.as_tensor(Q, dtype=torch.float64))
        self.d = D_LAT

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = z.ndim == 1
        if squeeze:
            z = z.unsqueeze(0)
        y = self._raw(z)
        y = pad_t(y, D_AMB)
        out = y @ self.Q.T
        return out[0] if squeeze else out

    def _raw(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class F0Map(FixtureMap):
    def _raw(self, z):
        return stereo_t(z)


class F1Map(FixtureMap):
    def __init__(self, Q, c: float = F1_C):
        super().__init__("F1", Q)
        self.c = float(c)
        self.r = float(np.sqrt(max(1.0 - c * c, 0.0)))

    def _raw(self, z):
        phi = stereo_t(z)
        ccol = z.new_full((z.shape[0], 1), self.c)
        return torch.cat([self.r * phi, ccol], dim=1)


class CliffordMap(FixtureMap):
    def __init__(self, name, Q, r2, s2):
        super().__init__(name, Q)
        self.r = float(np.sqrt(r2))
        self.s = float(np.sqrt(s2))

    def _raw(self, z):
        u, v = z[:, :CLIFFORD_P], z[:, CLIFFORD_P:]
        return torch.cat([self.r * stereo_t(u), self.s * stereo_t(v)], dim=1)


class BumpedMap(FixtureMap):
    def __init__(self, name, Q, widths):
        super().__init__(name, Q)
        self.register_buffer("centers", torch.as_tensor(F4_CENTERS, dtype=torch.float64))
        self.register_buffer("amps", torch.as_tensor(F4_AMPLITUDES, dtype=torch.float64))
        self.register_buffer("widths", torch.as_tensor(widths, dtype=torch.float64))
        self.a = float(F4_BUMP_SCALE)

    def _raw(self, z):
        phi = stereo_t(z)
        h = z.new_zeros(z.shape[0])
        for i in range(self.amps.numel()):
            d2 = ((z - self.centers[i]) ** 2).sum(dim=-1)
            h = h + self.amps[i] * torch.exp(-d2 / (2.0 * self.widths[i] ** 2))
        extra = (self.a * h).unsqueeze(-1)
        y = torch.cat([phi, extra], dim=1)
        return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=1e-15)


def make_torch_map(name: str, Q: np.ndarray) -> FixtureMap:
    if name == "F0":
        return F0Map("F0", Q)
    if name == "F1":
        return F1Map(Q)
    if name == "F2":
        return CliffordMap("F2", Q, 0.5, 0.5)
    if name == "F3":
        return CliffordMap("F3", Q, F3_R2, F3_S2)
    if name == "F4":
        return BumpedMap("F4", Q, F4_WIDTHS)
    if name == "F5":
        return BumpedMap("F5", Q, F5_WIDTHS)
    raise KeyError(name)


def autodiff_jets(model: torch.nn.Module, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobian and Hessian of a unit-sphere map at one latent point, float64.

    Uses ``torch.func`` (vector-valued hessian), not ``autograd.functional.hessian``,
    which is defined only for scalar maps.
    """
    from torch.func import hessian, jacrev

    zt = torch.as_tensor(z, dtype=torch.float64)
    model = model.double()

    def f(u: torch.Tensor) -> torch.Tensor:
        return model(u)

    G = f(zt).detach().cpu().numpy().astype(np.float64)
    J = jacrev(f)(zt)
    Hess = hessian(f)(zt)
    return (
        G,
        J.detach().cpu().numpy().astype(np.float64),
        Hess.detach().cpu().numpy().astype(np.float64),
    )


def autodiff_jacobian(model: torch.nn.Module, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G, J, g without Hessian (oracle volume weights)."""
    from torch.func import jacrev

    zt = torch.as_tensor(z, dtype=torch.float64)
    model = model.double()

    def f(u: torch.Tensor) -> torch.Tensor:
        return model(u)

    G = f(zt).detach().cpu().numpy().astype(np.float64)
    J = jacrev(f)(zt).detach().cpu().numpy().astype(np.float64)
    g = J.T @ J
    return G, J, g


def batch_autodiff_geometry(name: str, Z: np.ndarray, Q: np.ndarray, chunk: int = 16) -> list[dict]:
    """vmap Hessian of the exact generator (truth, not the learned decoder)."""
    from torch.func import hessian, jacrev, vmap

    model = make_torch_map(name, Q).double()
    Zt = torch.as_tensor(Z, dtype=torch.float64)
    if Zt.ndim == 1:
        Zt = Zt.unsqueeze(0)

    def f_one(u: torch.Tensor) -> torch.Tensor:
        return model(u)

    out = []
    n = Zt.shape[0]
    for s in range(0, n, chunk):
        sl = Zt[s : s + chunk]
        need = chunk - sl.shape[0]
        if need > 0:
            sl = torch.cat([sl, sl[-1:].repeat(need, 1)], 0)
        J = vmap(jacrev(f_one))(sl)
        Hess = vmap(hessian(f_one))(sl)
        G = vmap(f_one)(sl)
        m = Zt[s : s + chunk].shape[0]
        for i in range(m):
            out.append(
                geometry_from_jets(
                    G[i].detach().cpu().numpy(),
                    J[i].detach().cpu().numpy(),
                    Hess[i].detach().cpu().numpy(),
                )
            )
    return out


# -------------------- analytic tensors --------------------


def _analytic_f0(z: np.ndarray, Q: np.ndarray) -> dict:
    G = f0_np(z, Q)
    J = _embed_J(stereo_jacobian_np(z), Q)
    D = G.shape[0]
    d = z.shape[0]
    B = np.zeros((D, d, d), dtype=np.float64)
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    out = curvature_from_B(B, g)
    out.update({"G": G, "J": J, "Hess": np.zeros((D, d, d)), "P_T": PT, "P_NS": PNS, "G_hat": Gh, "source": "analytic"})
    return out


def _analytic_f1(z: np.ndarray, Q: np.ndarray) -> dict:
    """Latitude sphere: all principal curvatures = cot(α) = c/r, ring B = 0."""
    c = float(F1_C)
    r = float(np.sqrt(1.0 - c * c))
    kappa = c / r
    G = f1_np(z, Q)
    J_phi = stereo_jacobian_np(z)
    J_low = np.concatenate([r * J_phi, np.zeros((1, z.shape[0]), dtype=np.float64)], axis=0)
    J = _embed_J(J_low, Q)
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    phi = stereo_np(z)
    nu_low = np.concatenate([-c * phi, np.array([r], dtype=np.float64)])
    nu = rotate_ambient(pad_ambient(nu_low[None, :], D_AMB), Q)[0]
    nu = PNS @ nu
    nn = float(np.linalg.norm(nu))
    if nn < 1e-15:
        raise RuntimeError("F1 analytic normal vanished")
    nu = nu / nn
    # B_ab = κ g_ab ν  (Hessian convention)
    B = np.einsum("ab,D->Dab", g, kappa * nu)
    out = curvature_from_B(B, g)
    out.update({"G": G, "J": J, "P_T": PT, "P_NS": PNS, "G_hat": Gh, "nu": nu, "kappa": kappa, "source": "analytic"})
    return out


def _clifford_normal(z: np.ndarray, Q: np.ndarray, r2: float, s2: float) -> np.ndarray:
    r, s = float(np.sqrt(r2)), float(np.sqrt(s2))
    u, v = z[:CLIFFORD_P], z[CLIFFORD_P:]
    xu, yv = stereo_np(u), stereo_np(v)
    nu_low = np.concatenate([-s * xu, r * yv])
    nu = rotate_ambient(pad_ambient(nu_low[None, :], D_AMB), Q)[0]
    return nu


def _analytic_clifford(z: np.ndarray, Q: np.ndarray, r2: float, s2: float, name: str) -> dict:
    """S^8(r)×S^8(s) ⊂ S^{17}: principal curvatures s/r (8×) and -r/s (8×)."""
    r, s = float(np.sqrt(r2)), float(np.sqrt(s2))
    ku, kv = s / r, -r / s
    G = _clifford_np(z, Q, r2, s2)
    u, v = z[:CLIFFORD_P], z[CLIFFORD_P:]
    Ju = r * stereo_jacobian_np(u)
    Jv = s * stereo_jacobian_np(v)
    J_low = np.zeros((Ju.shape[0] + Jv.shape[0], D_LAT), dtype=np.float64)
    J_low[: Ju.shape[0], :CLIFFORD_P] = Ju
    J_low[Ju.shape[0] :, CLIFFORD_P:] = Jv
    J = _embed_J(J_low, Q)
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    nu = PNS @ _clifford_normal(z, Q, r2, s2)
    nu = nu / max(float(np.linalg.norm(nu)), 1e-15)
    # Product embedding: II is umbilic on each factor and vanishes on mixed pairs.
    B = np.zeros((G.shape[0], D_LAT, D_LAT), dtype=np.float64)
    gu = J[:, :CLIFFORD_P].T @ J[:, :CLIFFORD_P]
    gv = J[:, CLIFFORD_P:].T @ J[:, CLIFFORD_P:]
    B[:, :CLIFFORD_P, :CLIFFORD_P] = np.einsum("ij,D->Dij", gu, ku * nu)
    B[:, CLIFFORD_P:, CLIFFORD_P:] = np.einsum("ij,D->Dij", gv, kv * nu)
    kappas = np.array([ku] * CLIFFORD_P + [kv] * CLIFFORD_Q, dtype=np.float64)
    out = curvature_from_B(B, g)
    out.update(
        {
            "G": G,
            "J": J,
            "P_T": PT,
            "P_NS": PNS,
            "G_hat": Gh,
            "nu": nu,
            "kappas": kappas,
            "source": "analytic",
            "ku": ku,
            "kv": kv,
        }
    )
    return out


def analytic_geometry(name: str, z: np.ndarray, Q: np.ndarray) -> dict | None:
    z = np.asarray(z, dtype=np.float64)
    if name == "F0":
        return _analytic_f0(z, Q)
    if name == "F1":
        return _analytic_f1(z, Q)
    if name == "F2":
        return _analytic_clifford(z, Q, 0.5, 0.5, "F2")
    if name == "F3":
        return _analytic_clifford(z, Q, F3_R2, F3_S2, "F3")
    return None


def autodiff_geometry(name: str, z: np.ndarray, Q: np.ndarray, model: FixtureMap | None = None) -> dict:
    if model is None:
        model = make_torch_map(name, Q)
    G, J, Hess = autodiff_jets(model, z)
    out = geometry_from_jets(G, J, Hess)
    out["source"] = "autodiff"
    return out


def fd_geometry(name: str, z: np.ndarray, Q: np.ndarray, h: float = FD_STEP) -> dict:
    fn = lambda u, n=name: GENERATORS_NP[n](u, Q)
    G = fn(z)
    J = jacobian_fd(fn, z, h)
    Hess = hessian_fd(fn, z, h)
    out = geometry_from_jets(G, J, Hess)
    out["source"] = "finite_difference"
    return out


def validate_point(name: str, z: np.ndarray, Q: np.ndarray) -> dict:
    """Independent analytic / autodiff / FD agreement plus geometric identities."""
    z = np.asarray(z, dtype=np.float64)
    ad = autodiff_geometry(name, z, Q)
    fd = fd_geometry(name, z, Q)
    an = analytic_geometry(name, z, Q)
    checks = {}
    def _rel_or_abs(a, b) -> float:
        na, nb = float(np.linalg.norm(np.ravel(a))), float(np.linalg.norm(np.ravel(b)))
        if max(na, nb) < 1e-10:
            return float(np.linalg.norm(np.ravel(np.asarray(a) - np.asarray(b))))
        return rel_err(a, b)

    checks["ad_fd_B_rel"] = _rel_or_abs(ad["B"], fd["B"])
    checks["ad_fd_H_rel"] = _rel_or_abs(ad["H"], fd["H"])
    checks["ad_fd_Kdir_rel"] = _rel_or_abs(ad["K_dir"], fd["K_dir"])
    if an is not None:
        checks["an_ad_B_rel"] = rel_err(an["B"], ad["B"])
        checks["an_ad_H_rel"] = rel_err(an["H"], ad["H"])
        checks["an_ad_Kdir_rel"] = rel_err(an["K_dir"], ad["K_dir"])
        checks["an_ad_Ktf_rel"] = rel_err(an["K_tf"], ad["K_tf"])
    orth = tangent_orthogonality(ad["B"], ad["J"], ad["G"])
    rad = radial_identity_unit(ad["G"], ad["J"], ad["Hess"])
    checks.update({f"orth_{k}": v for k, v in orth.items()})
    checks.update({f"rad_{k}": v for k, v in rad.items()})
    k_mc = mc_directional_k(ad["Bw"], n=4096, seed=0)
    checks["kdir_mc"] = k_mc
    checks["kdir_mc_rel"] = rel_err(k_mc, ad["K_dir"])
    checks["H_norm"] = ad["H_norm"]
    checks["K_dir"] = ad["K_dir"]
    checks["K_tf"] = ad["K_tf"]
    checks["Btf_fro"] = ad["Btf_fro"]
    ad_zero = float(np.linalg.norm(ad["B"])) <= 1e-8 and ad["K_dir"] <= TOL_F0_BS
    fd_k_ok = fd["K_dir"] <= 1e-8 if ad_zero else checks["ad_fd_B_rel"] <= TOL_AUTODIFF_FD_REL
    ok = (
        fd_k_ok
        and orth["max_abs_radial"] <= TOL_ORTH
        and orth["max_abs_tangent"] <= TOL_ORTH
        and rad["max_abs_GJ"] <= TOL_RADIAL
        and rad["max_abs_radial_hess_identity"] <= TOL_RADIAL
        and checks["kdir_mc_rel"] <= TOL_MC_KDIR
    )
    if an is not None:
        if name == "F0":
            ok = ok and ad_zero
        else:
            ok = ok and checks["an_ad_B_rel"] <= TOL_ANALYTIC_REL
    checks["ok"] = bool(ok)
    return {"checks": checks, "autodiff": {k: ad[k] for k in ("K_dir", "K_tf", "H_norm", "K_H")}}


def invariance_checks(name: str, z: np.ndarray, Q: np.ndarray) -> dict:
    """Latent-coordinate rotation and extra ambient rotation leave scalars invariant."""
    ad0 = autodiff_geometry(name, z, Q)
    rng = np.random.default_rng(123)
    Rlat, _ = np.linalg.qr(rng.standard_normal((D_LAT, D_LAT)))
    Rlat = Rlat * np.sign(np.diag(np.ones(D_LAT)))
    # G(z) = G0(R^T R z); new coords w = R^T z so G_new(w)=G_old(R w)? 
    # If we rotate latent labels z' = R z, the image point is G(R^{-1} z') which is a reparametrization.
    # Scalars at the same manifold point: evaluate at z and at R z using the *same* generator
    # is NOT the same point. Correct test: pull back the coordinate rotation through G.
    # Let G'(w) = G(R w). Then G'(R^T z) = G(z) same point, curvature scalars must match.
    Q_extra = orthonormal_qr(D_AMB, seed=999)
    Q2 = Q_extra @ Q
    ad_amb = autodiff_geometry(name, z, Q2)
    model = make_torch_map(name, Q)

    class RotLat(torch.nn.Module):
        def __init__(self, base, R):
            super().__init__()
            self.base = base
            self.register_buffer("R", torch.as_tensor(R, dtype=torch.float64))

        def forward(self, w):
            return self.base(self.R @ w)

    G, J, Hess = autodiff_jets(RotLat(model, Rlat), Rlat.T @ z)
    ad_lat = geometry_from_jets(G, J, Hess)
    return {
        "K_dir_base": ad0["K_dir"],
        "K_dir_lat": ad_lat["K_dir"],
        "K_dir_amb": ad_amb["K_dir"],
        "H_norm_base": ad0["H_norm"],
        "H_norm_lat": ad_lat["H_norm"],
        "H_norm_amb": ad_amb["H_norm"],
        "rel_K_lat": rel_err(ad_lat["K_dir"], ad0["K_dir"]),
        "rel_K_amb": rel_err(ad_amb["K_dir"], ad0["K_dir"]),
        "rel_H_lat": rel_err(ad_lat["H_norm"], ad0["H_norm"]),
        "rel_H_amb": rel_err(ad_amb["H_norm"], ad0["H_norm"]),
        "ok": bool(
            rel_err(ad_lat["K_dir"], ad0["K_dir"]) <= TOL_INVARIANT
            and rel_err(ad_amb["K_dir"], ad0["K_dir"]) <= TOL_INVARIANT
        ),
    }


@dataclass
class FixtureSpec:
    name: str
    family: str
    closed_form: bool
    expected_H: str
    expected_tf: str
    notes: str


SPECS = {
    "F0": FixtureSpec("F0", "great_subsphere", True, "zero", "zero", "totally geodesic S^16 in S^767; B^S=0"),
    "F1": FixtureSpec("F1", "latitude_subsphere", True, "nonzero_const", "zero", f"c={F1_C}, kappa=c/r"),
    "F2": FixtureSpec("F2", "minimal_clifford", True, "zero", "nonzero_const", "r^2=s^2=1/2; H=0, TF≠0"),
    "F3": FixtureSpec("F3", "nonminimal_clifford", True, "nonzero_const", "nonzero_const", f"r^2={F3_R2}, s^2={F3_S2}"),
    "F4": FixtureSpec("F4", "low_freq_bumped_sphere", False, "variable", "variable", "frozen wide Gaussian bumps"),
    "F5": FixtureSpec("F5", "high_freq_bumped_sphere", False, "variable", "variable", "frozen narrow Gaussian bumps"),
}


def fixture_definitions() -> dict:
    return {
        "d": D_LAT,
        "D": D_AMB,
        "ambient_rotation_seed": AMBIENT_ROTATION_SEED,
        "ambient_rotation_hash16": AMBIENT_ROTATION_HASH16,
        "F1_c": F1_C,
        "F1_r": float(np.sqrt(1.0 - F1_C**2)),
        "F3_r2": F3_R2,
        "F3_s2": F3_S2,
        "F4_bump_scale": F4_BUMP_SCALE,
        "F4_amplitudes": list(F4_AMPLITUDES),
        "F4_widths": list(F4_WIDTHS),
        "F5_widths": list(F5_WIDTHS),
        "F4_centers": [list(c) for c in F4_CENTERS],
        "clifford_p": CLIFFORD_P,
        "clifford_q": CLIFFORD_Q,
        "stereo": "inverse_stereographic R^n -> S^n, north-pole",
        "unit_normalized": True,
        "specs": {k: vars(v) for k, v in SPECS.items()},
        "packing_note": (
            "T1 B^S is the Hessian P_NS ∂_{ab}G. Frozen quadratic BS_flat stores "
            "Phi=u_a u_b coefficients; Hess = 2 * unpack_BS_symmetric(BS_flat)."
        ),
    }
