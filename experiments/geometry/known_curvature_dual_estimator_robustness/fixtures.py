"""F0/F1/F2/F4 sphere immersions at D=28. Analytic maps copied from the
known-curvature fixture audit; only pad/rotation dimension changed (768→28).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import hessian, jacrev

from geometry.known_curvature_point_patch_fixture_audit.geometry import (
    apply_P,
    hessian_fd,
    jacobian_fd,
    orthonormal_qr,
    pad_ambient,
    rotate_ambient,
    sphere_normal_projectors,
    stereo_np,
    stereo_inv_np,
)

from .config import (
    AMBIENT_ROTATION_SEED,
    D_AMB,
    D_LAT,
    F1_C,
    F4_AMPLITUDES,
    F4_BUMP_SCALE,
    F4_CENTERS,
    F4_WIDTHS,
)

CLIFFORD_P = 8


def ambient_rotation_d28(seed: int = AMBIENT_ROTATION_SEED) -> np.ndarray:
    """Seeded QR rotation in R^{28}. The frozen 768×768 Q.npy cannot be used at D=28."""
    return orthonormal_qr(D_AMB, seed)


def bump_field(z: np.ndarray) -> np.ndarray:
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    sq = np.asarray(z, dtype=np.float64).ndim == 1
    h = np.zeros(z2.shape[0], dtype=np.float64)
    C = np.asarray(F4_CENTERS, dtype=np.float64)
    A = np.asarray(F4_AMPLITUDES, dtype=np.float64)
    W = np.asarray(F4_WIDTHS, dtype=np.float64)
    for i in range(len(A)):
        d2 = np.sum((z2 - C[i]) ** 2, axis=1)
        h += A[i] * np.exp(-d2 / (2.0 * W[i] ** 2))
    return h[0] if sq else h


def f0_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    y = pad_ambient(stereo_np(z2), D_AMB)
    out = rotate_ambient(y, Q)
    return out[0] if np.asarray(z).ndim == 1 else out


def f1_np(z: np.ndarray, Q: np.ndarray, c: float = F1_C) -> np.ndarray:
    r = float(np.sqrt(max(1.0 - c * c, 0.0)))
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    phi = stereo_np(z2)
    y = np.concatenate([r * phi, np.full((z2.shape[0], 1), c, dtype=np.float64)], axis=1)
    out = rotate_ambient(pad_ambient(y, D_AMB), Q)
    return out[0] if np.asarray(z).ndim == 1 else out


def f2_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r = s = float(np.sqrt(0.5))
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    xu, yv = stereo_np(z2[:, :CLIFFORD_P]), stereo_np(z2[:, CLIFFORD_P:])
    y = np.concatenate([r * xu, s * yv], axis=1)
    out = rotate_ambient(pad_ambient(y, D_AMB), Q)
    return out[0] if np.asarray(z).ndim == 1 else out


def f4_np(z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
    phi = stereo_np(z2)
    h = bump_field(z2)
    extra = (F4_BUMP_SCALE * np.atleast_1d(h))[:, None]
    y = np.concatenate([phi, extra], axis=1)
    y = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-15, None)
    out = rotate_ambient(pad_ambient(y, D_AMB), Q)
    return out[0] if np.asarray(z).ndim == 1 else out


GENERATORS_NP = {"F0": f0_np, "F1": f1_np, "F2": f2_np, "F4": f4_np}


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
        self.D = int(Q.shape[0])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = z.ndim == 1
        if squeeze:
            z = z.unsqueeze(0)
        y = pad_t(self._raw(z), self.D)
        out = y @ self.Q.T
        return out[0] if squeeze else out

    def _raw(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class F0Map(FixtureMap):
    def __init__(self, Q):
        super().__init__("F0", Q)

    def _raw(self, z):
        return stereo_t(z)


class F1Map(FixtureMap):
    def __init__(self, Q, c: float = F1_C):
        super().__init__("F1", Q)
        self.c = float(c)
        self.r = float(np.sqrt(max(1.0 - c * c, 0.0)))

    def _raw(self, z):
        phi = stereo_t(z)
        c = z.new_full((z.shape[0], 1), self.c)
        return torch.cat([self.r * phi, c], dim=-1)


class F2Map(FixtureMap):
    def __init__(self, Q):
        super().__init__("F2", Q)

    def _raw(self, z):
        r = s = 0.5 ** 0.5
        xu, yv = stereo_t(z[:, :CLIFFORD_P]), stereo_t(z[:, CLIFFORD_P:])
        return torch.cat([r * xu, s * yv], dim=-1)


class F4Map(FixtureMap):
    def __init__(self, Q):
        super().__init__("F4", Q)
        self.register_buffer("C", torch.as_tensor(np.asarray(F4_CENTERS), dtype=torch.float64))
        self.register_buffer("A", torch.as_tensor(np.asarray(F4_AMPLITUDES), dtype=torch.float64))
        self.register_buffer("W", torch.as_tensor(np.asarray(F4_WIDTHS), dtype=torch.float64))

    def _raw(self, z):
        phi = stereo_t(z)
        h = z.new_zeros(z.shape[0])
        for i in range(self.A.shape[0]):
            d2 = ((z - self.C[i]) ** 2).sum(dim=-1)
            h = h + self.A[i] * torch.exp(-d2 / (2.0 * self.W[i] ** 2))
        extra = (F4_BUMP_SCALE * h)[:, None]
        y = torch.cat([phi, extra], dim=-1)
        return y / torch.linalg.norm(y, dim=-1, keepdim=True)


def make_torch_map(name: str, Q: np.ndarray) -> FixtureMap:
    return {"F0": F0Map, "F1": F1Map, "F2": F2Map, "F4": F4Map}[name](Q)


def autodiff_jets(model: FixtureMap, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zt = torch.as_tensor(np.asarray(z, dtype=np.float64), dtype=torch.float64)
    if zt.ndim != 1:
        raise ValueError("autodiff_jets expects a single latent point")
    G = model.forward(zt)
    J = jacrev(model.forward)(zt)
    H = hessian(model.forward)(zt)
    return G.detach().numpy(), J.detach().numpy(), H.detach().numpy()


def energy_g(B: np.ndarray, ginv: np.ndarray) -> float:
    Bw = np.einsum("ac,bd,iab->icd", ginv, ginv, B)
    return float(np.tensordot(B, Bw, axes=([0, 1, 2], [0, 1, 2])))


def targets_from_jets(G: np.ndarray, J: np.ndarray, Qhess: np.ndarray) -> dict:
    """Full Euclidean II^E, residual B^S, unaveraged traces, Gauss scalar."""
    g, PT, PNS, Gh = sphere_normal_projectors(G, J)
    d = J.shape[1]
    eye = np.eye(G.shape[0])
    PN = eye - PT
    II_E = apply_P(PN, Qhess)
    B_S = apply_P(PNS, Qhess)
    II_R = -np.einsum("ab,i->iab", g, Gh)
    try:
        ginv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        ginv = np.linalg.pinv(g)
    H_E = np.einsum("ab,iab->i", ginv, II_E)
    H_S = np.einsum("ab,iab->i", ginv, B_S)
    eE, eS, eR = energy_g(II_E, ginv), energy_g(B_S, ginv), energy_g(II_R, ginv)
    scal = float(d * (d - 1) + np.dot(H_S, H_S) - eS)
    return {
        "G": G,
        "J": J,
        "Hess": Qhess,
        "g": g,
        "ginv": ginv,
        "P_T": PT,
        "P_NS": PNS,
        "G_hat": Gh,
        "II_E": II_E,
        "B_S": B_S,
        "II_R": II_R,
        "H_E": H_E,
        "H_S": H_S,
        "H_E_avg": H_E / d,
        "H_S_avg": H_S / d,
        "energy_II_E": eE,
        "energy_B_S": eS,
        "energy_II_R": eR,
        "f_res": float(eS / eE) if eE > 1e-30 else float("nan"),
        "Scal": scal,
        "reconstruct_rel": float(np.linalg.norm(II_E - (II_R + B_S)) / max(np.linalg.norm(II_E), 1e-12)),
        "H_E_norm": float(np.linalg.norm(H_E)),
        "H_S_norm": float(np.linalg.norm(H_S)),
        "B_S_fro": float(np.linalg.norm(B_S)),
        "II_E_fro": float(np.linalg.norm(II_E)),
    }


def truth_at(name: str, z: np.ndarray, Qrot: np.ndarray) -> dict:
    model = make_torch_map(name, Qrot)
    G, J, H = autodiff_jets(model, z)
    out = targets_from_jets(G, J, H)
    out["name"] = name
    return out


def jacobian_batch_fd(name: str, z: np.ndarray, Q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Vectorized central-difference Jacobian, shape (n, D, d)."""
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    fn = GENERATORS_NP[name]
    G0 = fn(z, Q)
    D = G0.shape[1]
    J = np.empty((n, D, d), dtype=np.float64)
    for a in range(d):
        zp, zm = z.copy(), z.copy()
        zp[:, a] += h
        zm[:, a] -= h
        J[:, :, a] = (fn(zp, Q) - fn(zm, Q)) / (2.0 * h)
    return G0, J


def haar_chart(n: int, dim_sphere: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    g = rng.standard_normal((n, dim_sphere + 1))
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-15, None)
    s = g[:, 0].copy()
    return stereo_inv_np(g), s


def sample_latent(name: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if name == "F2":
        u, su = haar_chart(n, CLIFFORD_P, rng)
        v, sv = haar_chart(n, CLIFFORD_P, rng)
        return np.concatenate([u, v], axis=1), su
    return haar_chart(n, D_LAT, rng)
