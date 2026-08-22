"""SAE tangent benchmark v2: code-manifold pushforward hypothesis.

Does NOT modify outputs/geometry/physics_tangent_estimator_benchmark/.
Estimator selection uses synthetic geometry only — never probe labels.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .curvature_probe_screen import partial_spearman, spearman_dict
from .gauss_map_curvature import (
    estimate_anchor_gauss_map,
    parallel_transport_basis,
    split_half_projectors,
)
from .multimodel_graph_prior_quadratic import load_model_X
from .paths import platonic_root, resolve_path
from .quadratic import quadratic_features
from .sphere_normal_quadratic import normalize_rows, sphere_project_basis
from .tangent_estimator_benchmark import (
    bootstrap_grassmann_pca,
    joint_quadratic_principal_manifold,
    same_patch_pca,
    train_synthetic_sae,
)
from .tangent_reliability import (
    grassmann_dist,
    kernel_weights,
    pca_tangent,
    principal_angles,
    projector,
)

EPS = 1e-12
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
PHYSICS_SAE = "outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0"
# Do not write here:
V1_OUT = "outputs/geometry/physics_tangent_estimator_benchmark"


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _case_seed(base: int, *parts: Any) -> int:
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(base + int(h[:8], 16) % 10_000)


def _ensure_sae_path() -> Path:
    sae_path = Path(__file__).resolve().parents[2] / "SAE-shared-basis"
    if str(sae_path) not in sys.path:
        sys.path.insert(0, str(sae_path))
    return sae_path


def projector_frobenius_sq_thin(U: np.ndarray, V: np.ndarray) -> float:
    """||P_U - P_V||_F^2 = 2d - 2||U^T V||_F^2 for orthonormal bases."""
    d = min(U.shape[1], V.shape[1])
    return float(2 * d - 2 * np.linalg.norm(U[:, :d].T @ V[:, :d], "fro") ** 2)


def E_T_thin(J_hat: np.ndarray, J_true: np.ndarray) -> float:
    d = min(J_hat.shape[1], J_true.shape[1])
    return float(np.sqrt(max(projector_frobenius_sq_thin(J_hat, J_true), 0.0) / max(2 * d, 1)))


@dataclass
class SAETangentConfig:
    output_dir: str = "outputs/geometry/physics_sae_tangent_benchmark_v2"
    multimodel_dir: str = SOURCE_MM
    physics_sae_dir: str = PHYSICS_SAE
    model: str = "vit_base"
    target: str = "mag_r_desi"
    ambient_D: int = 768
    dims: list[int] = field(default_factory=lambda: [8, 12, 16])
    primary_d: int = 12
    k_list: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    n_global: int = 16384
    n_anchors: int = 48
    sae_seeds: list[int] = field(default_factory=lambda: [0, 1])
    sae_feature_dim: int = 256
    sae_k: int = 32
    sae_steps: int = 400
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    smoke: bool = False
    max_seconds: float = 14400.0

    def resolved(self, root: Path) -> Path:
        out = resolve_path(root, self.output_dir)
        # safety: never write into v1 dir
        if out.resolve() == resolve_path(root, V1_OUT).resolve():
            raise RuntimeError("Refusing to write into v1 tangent benchmark directory")
        return out

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


# -------------------- latent sampling --------------------


def sample_latent(kind: str, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(size=(n, d))
    if kind == "laplace":
        return rng.laplace(size=(n, d))
    if kind == "mixture":
        # 3-mode mixture
        centers = rng.normal(size=(3, d)) * 2.0
        comp = rng.integers(0, 3, size=n)
        return centers[comp] + 0.4 * rng.normal(size=(n, d))
    if kind == "truncated":
        Z = rng.normal(size=(n, d))
        # nonuniform: keep only half-space + stretch
        Z = Z * (0.3 + 0.7 * (Z[:, 0:1] > 0))
        return Z
    raise ValueError(kind)


# -------------------- differentiable sphere decoders --------------------


@dataclass
class SynthDecoder:
    family: str
    d: int
    D: int
    seed: int
    # parameters filled at init
    x0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jlin: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    BS: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mlp: Any = None
    gate_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Jalt: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        J, _ = np.linalg.qr(rng.normal(size=(self.D, self.d)))
        x0 = rng.normal(size=self.D)
        x0 = x0 - J @ (J.T @ x0)
        x0 /= max(np.linalg.norm(x0), EPS)
        J = sphere_project_basis(x0, J)
        self.x0, self.Jlin = x0, J
        if self.family in ("quadratic", "sparse_gated"):
            q = self.d * (self.d + 1) // 2
            # normal directions for curvature
            Nspan, _ = np.linalg.qr(rng.normal(size=(self.D, min(8, self.D - self.d - 1))))
            Nspan = Nspan - x0[:, None] * (x0 @ Nspan)
            Nspan = Nspan - J @ (J.T @ Nspan)
            Qn, _ = np.linalg.qr(Nspan)
            BS = np.zeros((self.D, q))
            idx = 0
            for a in range(self.d):
                for b in range(a, self.d):
                    BS[:, idx] = 0.5 * Qn[:, idx % Qn.shape[1]]
                    idx += 1
            self.BS = BS
        if self.family == "mlp_silu":
            self.mlp = _make_mlp(self.d, self.D, self.seed)
        if self.family == "sparse_gated":
            self.gate_w = rng.normal(size=self.d)
            self.gate_w /= max(np.linalg.norm(self.gate_w), EPS)
        if self.family in ("piecewise", "stratified"):
            J2, _ = np.linalg.qr(rng.normal(size=(self.D, self.d)))
            self.Jalt = sphere_project_basis(x0, J2)

    def embed(self, Z: np.ndarray) -> np.ndarray:
        fam = self.family
        if fam == "linear":
            return normalize_rows(self.x0 + Z @ self.Jlin.T)
        if fam == "quadratic":
            Phi = quadratic_features(Z)
            return normalize_rows(self.x0 + Z @ self.Jlin.T + 0.5 * (Phi @ self.BS.T))
        if fam == "mlp_silu":
            with torch.no_grad():
                y = self.mlp(torch.tensor(Z, dtype=torch.float32)).numpy()
            return normalize_rows(y)
        if fam == "sparse_gated":
            g = 1.0 / (1.0 + np.exp(-3.0 * (Z @ self.gate_w)))
            Zg = Z * g[:, None]
            Phi = quadratic_features(Zg)
            return normalize_rows(self.x0 + Zg @ self.Jlin.T + 0.35 * (Phi @ self.BS.T))
        if fam == "piecewise":
            # support switch on first coord
            m = Z[:, 0] >= 0
            X = np.zeros((len(Z), self.D))
            X[m] = normalize_rows(self.x0 + Z[m] @ self.Jlin.T)
            X[~m] = normalize_rows(self.x0 + Z[~m] @ self.Jalt.T)
            return X
        if fam == "stratified":
            half = len(Z) // 2
            X1 = normalize_rows(self.x0 + Z[:half] @ self.Jlin.T)
            X2 = normalize_rows(self.x0 + Z[half:] @ self.Jalt.T)
            return np.vstack([X1, X2])
        raise ValueError(fam)

    def true_tangent(self, z: np.ndarray) -> np.ndarray:
        """Exact ambient tangent basis at f(z) via analytic / autograd Jf."""
        fam = self.family
        if fam == "linear":
            x = normalize_rows((self.x0 + z @ self.Jlin.T)[None, :])[0]
            return sphere_project_basis(x, self.Jlin)
        if fam in ("quadratic", "sparse_gated"):
            # finite-diff Jacobian of embed (cheap, d columns)
            return self._fd_tangent(z)
        if fam == "mlp_silu":
            return self._fd_tangent(z)
        if fam in ("piecewise", "stratified"):
            # local chart tangent (may be ill-defined on stratified)
            if fam == "piecewise" and z[0] < 0:
                x = normalize_rows((self.x0 + z @ self.Jalt.T)[None, :])[0]
                return sphere_project_basis(x, self.Jalt)
            x = normalize_rows((self.x0 + z @ self.Jlin.T)[None, :])[0]
            return sphere_project_basis(x, self.Jlin)
        raise ValueError(fam)

    def _fd_tangent(self, z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        x0 = self.embed(z[None, :])[0]
        cols = []
        for i in range(self.d):
            e = np.zeros(self.d)
            e[i] = eps
            xp = self.embed((z + e)[None, :])[0]
            xm = self.embed((z - e)[None, :])[0]
            cols.append((xp - xm) / (2 * eps))
        J = np.column_stack(cols)
        return sphere_project_basis(x0, J)


def _make_mlp(d: int, D: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    h = max(64, 4 * d)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, D),
            )

        def forward(self, z):
            return self.net(z)

    m = M()
    m.eval()
    return m


# -------------------- undercomplete AE control --------------------


class UndercompleteAE(nn.Module):
    def __init__(self, D: int, d: int):
        super().__init__()
        self.enc = nn.Linear(D, d)
        self.dec = nn.Linear(d, D)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_undercomplete_ae(
    X: np.ndarray, d: int, steps: int, seed: int, device: torch.device
) -> dict:
    torch.manual_seed(seed)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / scale, dtype=torch.float32, device=device)
    model = UndercompleteAE(X.shape[1], d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    bs = min(256, len(Xt))
    model.train()
    for _ in range(steps):
        idx = rng.integers(0, len(Xt), size=bs)
        xb = Xt[idx]
        xh, _ = model(xb)
        loss = F.mse_loss(xh, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return {"model": model, "mean": mean, "scale": scale, "device": device, "d": d}


# -------------------- SAE encode/decode helpers --------------------


def sae_encode_decode(bundle: dict, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    xs = (X - bundle["mean"]) / bundle["scale"]
    with torch.no_grad():
        xt = torch.tensor(xs, dtype=torch.float32, device=device)
        xh, z = model(xt)
    return z.cpu().numpy(), xh.cpu().numpy()


def decoder_weight(bundle: dict) -> np.ndarray:
    return bundle["model"].decoder.weight.detach().cpu().numpy().astype(np.float64)


def active_decoder_J(bundle: dict, s0: np.ndarray) -> np.ndarray:
    """JD for linear decoder at fixed support: columns of W for active features."""
    W = decoder_weight(bundle)
    active = s0 > 0
    if active.sum() == 0:
        return W[:, :1] * 0.0
    return W[:, active]


# -------------------- tangent estimators --------------------


def sae_reconstruction_jacobian_top_svd(
    Xn, x0, d, sae_bundle=None, device=None, **kw
) -> tuple[np.ndarray | None, dict]:
    """Negative-control: leading left SV of DR (active-set)."""
    if sae_bundle is None:
        return None, {"ok": False, "reason": "no_sae", "estimator": "sae_reconstruction_jacobian_top_svd"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    model = sae_bundle["model"]
    if model.input_dim != len(x0):
        return None, {"ok": False, "reason": "dim_mismatch", "estimator": "sae_reconstruction_jacobian_top_svd"}
    x = torch.tensor(((x0 - sae_bundle["mean"]) / sae_bundle["scale"])[None, :], device=device, dtype=torch.float32)
    with torch.no_grad():
        z_pre = F.relu(model.encoder(x))
        kk = min(model.k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, kk, dim=-1)
        mask = vals.squeeze(0) > 0
        if int(mask.sum()) < 1:
            return None, {"ok": False, "reason": "no_active", "estimator": "sae_reconstruction_jacobian_top_svd"}
        active = idx.squeeze(0)[mask]
        We = model.encoder.weight[active]
        Wd = model.decoder.weight[:, active]
        Jlin = (Wd @ We).detach().cpu().numpy()
        xh, _ = model(x)
        r = xh.squeeze(0).cpu().numpy()
        nr = max(np.linalg.norm(r), EPS)
        ru = r / nr
        Jnp = ((np.eye(len(r)) - np.outer(ru, ru)) / nr) @ Jlin
    U, S, _ = np.linalg.svd(Jnp, full_matrices=False)
    J = sphere_project_basis(x0, U[:, :d])
    return J, {
        "estimator": "sae_reconstruction_jacobian_top_svd",
        "ok": True,
        "singular_gap": float(S[d - 1] - S[d]) if len(S) > d else float("nan"),
    }


def sae_code_covariance_pushforward(
    Xn, x0, d, sae_bundle=None, device=None, **kw
) -> tuple[np.ndarray | None, dict]:
    """Leading d code-variation directions → push through linear decoder."""
    if sae_bundle is None:
        return None, {"ok": False, "reason": "no_sae", "estimator": "sae_code_covariance_pushforward"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    Z, _ = sae_encode_decode(sae_bundle, np.vstack([x0[None, :], Xn]), device)
    s0, Sn = Z[0], Z[1:]
    # allow different supports: use raw code differences
    dS = Sn - s0
    # PCA in code space
    if len(dS) < d + 2:
        return None, {"ok": False, "reason": "too_few", "estimator": "sae_code_covariance_pushforward"}
    _, _, Vt = np.linalg.svd(dS, full_matrices=False)
    Vs = Vt[:d].T  # (F, d)
    W = decoder_weight(sae_bundle)
    # scale space → ambient (decoder acts on scaled activations)
    Jamb = W @ Vs
    # map back: ambient of scaled space ≈ scale * J; then sphere-project at x0
    Jamb = Jamb * sae_bundle["scale"][:, None]
    J = sphere_project_basis(x0, Jamb)
    return J, {
        "estimator": "sae_code_covariance_pushforward",
        "ok": True,
        "code_rank": int(np.sum(np.linalg.svd(dS, compute_uv=False) > 1e-6)),
        "active0": int((s0 > 0).sum()),
    }


def sae_denoised_pca(
    Xn, x0, d, sae_bundle=None, device=None, weighted: bool = False, **kw
) -> tuple[np.ndarray | None, dict]:
    if sae_bundle is None:
        return None, {"ok": False, "reason": "no_sae", "estimator": "sae_denoised_pca"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    # reconstruct patch in scaled space then unscale
    Xall = np.vstack([x0[None, :], Xn])
    _, Xh_sc = sae_encode_decode(sae_bundle, Xall, device)
    Xh = Xh_sc * sae_bundle["scale"] + sae_bundle["mean"]
    Xh = normalize_rows(Xh)
    xh0, Xhn = Xh[0], Xh[1:]
    w = None
    if weighted:
        dists = np.linalg.norm(Xhn - xh0, axis=1)
        w = kernel_weights(dists, float(np.quantile(dists, 0.5)))
    J, _, diag = pca_tangent(Xhn, xh0, d, weights=w)
    # transport/project to true x0 sphere frame
    J = sphere_project_basis(x0, J)
    return J, {
        "estimator": "sae_denoised_pca_weighted" if weighted else "sae_denoised_pca",
        "ok": True,
        **{k: diag[k] for k in ("eigengap", "rel_eigengap", "d_eff")},
    }


def _local_isomap_coords(S: np.ndarray, n_comp: int, knn: int = 12) -> np.ndarray:
    """Classical MDS on truncated geodesic distances of a local code knn graph."""
    n = len(S)
    knn = min(knn, n - 1)
    # pairwise euclidean in code
    G = S @ S.T
    norms = np.diag(G)
    D2 = np.maximum(norms[:, None] + norms[None, :] - 2 * G, 0.0)
    D = np.sqrt(D2)
    # knn graph
    nn_idx = np.argsort(D, axis=1)[:, 1 : knn + 1]
    W = np.full((n, n), np.inf)
    np.fill_diagonal(W, 0.0)
    for i in range(n):
        for j in nn_idx[i]:
            W[i, j] = W[j, i] = D[i, j]
    # Floyd–Warshall (n small: ≤2048 but we use patch ≤512 typically)
    for k in range(n):
        W = np.minimum(W, W[:, k : k + 1] + W[k : k + 1, :])
    # unreachable → large
    finite = np.isfinite(W)
    if not finite.all():
        med = np.median(W[finite & (W > 0)])
        W = np.where(finite, W, 3.0 * med)
    # classical MDS
    D2 = W**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    pos = evals > 1e-10
    m = min(n_comp, int(pos.sum()))
    if m == 0:
        return np.zeros((n, n_comp))
    Z = evecs[:, :m] * np.sqrt(evals[:m])
    if m < n_comp:
        Z = np.pad(Z, ((0, 0), (0, n_comp - m)))
    return Z


def sae_graph_pushforward(
    Xn, x0, d, sae_bundle=None, device=None, **kw
) -> tuple[np.ndarray | None, dict]:
    """Primary hypothesis: local Isomap on codes → Vs → JD Vs; also direct lift."""
    if sae_bundle is None:
        return None, {"ok": False, "reason": "no_sae", "estimator": "sae_graph_pushforward"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    Xall = np.vstack([x0[None, :], Xn])
    # subsample patch if huge
    if len(Xall) > 400:
        rng = np.random.default_rng(0)
        keep = np.concatenate([[0], rng.choice(np.arange(1, len(Xall)), size=399, replace=False)])
        Xall = Xall[keep]
        Xn = Xall[1:]
    Z, Xh_sc = sae_encode_decode(sae_bundle, Xall, device)
    Xh = normalize_rows(Xh_sc * sae_bundle["scale"] + sae_bundle["mean"])
    s0, Sn = Z[0], Z[1:]
    # Isomap coords on codes including s0
    coords = _local_isomap_coords(Z, n_comp=d, knn=min(16, len(Z) - 1))
    z0, Zn = coords[0], coords[1:]
    dS = Sn - s0
    dZ = Zn - z0
    # Vs = argmin ||dS - Vs dZ|| ; Vs is (F, d)
    # solve least squares: dS ≈ dZ @ Vs.T → Vs.T = lstsq(dZ, dS)
    try:
        Vs_T, *_ = np.linalg.lstsq(dZ, dS, rcond=None)
        Vs = Vs_T.T
    except np.linalg.LinAlgError:
        return None, {"ok": False, "reason": "lstsq", "estimator": "sae_graph_pushforward"}
    W = decoder_weight(sae_bundle)
    J_push = (W @ Vs) * sae_bundle["scale"][:, None]
    J = sphere_project_basis(x0, J_push)
    # direct reconstructed lift
    dXh = Xh[1:] - Xh[0]
    try:
        Jdir_T, *_ = np.linalg.lstsq(dZ, dXh, rcond=None)
        J_direct = sphere_project_basis(x0, Jdir_T.T)
    except np.linalg.LinAlgError:
        J_direct = J
    agree = E_T_thin(J, J_direct)
    return J, {
        "estimator": "sae_graph_pushforward",
        "ok": True,
        "agree_direct_ET": agree,
        "active0": int((s0 > 0).sum()),
        "J_direct": J_direct,  # stripped before parquet
    }


def sae_fixed_point_secant(
    Xn, x0, d, sae_bundle=None, device=None, lam: float = 1.0, **kw
) -> tuple[np.ndarray | None, dict]:
    """Secants that are approximately fixed by JR = JD JE (active-set)."""
    if sae_bundle is None:
        return None, {"ok": False, "reason": "no_sae", "estimator": "sae_fixed_point_secant"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    model = sae_bundle["model"]
    Xall = np.vstack([x0[None, :], Xn])
    Z, Xh_sc = sae_encode_decode(sae_bundle, Xall, device)
    Xh = normalize_rows(Xh_sc * sae_bundle["scale"] + sae_bundle["mean"])
    dXh = Xh[1:] - Xh[0]
    # candidate basis Q from reconstructed secants, rank ≤ 3d
    rmax = min(3 * d, len(dXh), Xh.shape[1])
    if rmax < d:
        return None, {"ok": False, "reason": "rank", "estimator": "sae_fixed_point_secant"}
    _, _, Vt = np.linalg.svd(dXh, full_matrices=False)
    Q = Vt[:rmax].T  # (D, r)
    # build JR ≈ Wd We on active set at x0 (scaled space), map to ambient approx
    x = torch.tensor(((x0 - sae_bundle["mean"]) / sae_bundle["scale"])[None, :], device=device, dtype=torch.float32)
    with torch.no_grad():
        z_pre = F.relu(model.encoder(x))
        kk = min(model.k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, kk, dim=-1)
        mask = vals.squeeze(0) > 0
        if int(mask.sum()) < 1:
            return None, {"ok": False, "reason": "no_active", "estimator": "sae_fixed_point_secant"}
        active = idx.squeeze(0)[mask]
        We = model.encoder.weight[active].detach().cpu().numpy()
        Wd = model.decoder.weight[:, active].detach().cpu().numpy()
    Jlin = Wd @ We  # scaled→scaled
    # JR on ambient approx: scale * Jlin * inv_scale
    sc = sae_bundle["scale"]
    JR = (sc[:, None] * Jlin) / np.maximum(sc[None, :], EPS)
    # work in Q coords: minimize ||(JR-I) Q A|| + λ ||(I - Q A A^T Q^T) dXh||
    # with A (r,d) Stiefel — take top-d eigenspace of a surrogate
    M = Q.T @ (JR - np.eye(len(x0))).T @ (JR - np.eye(len(x0))) @ Q
    # encourage spanning secants: add -λ Q^T (dXh.T dXh) Q style energy
    Sec = dXh.T @ dXh
    M = M - lam * (Q.T @ Sec @ Q) / max(np.trace(Sec), EPS)
    evals, evecs = np.linalg.eigh(M)
    A = evecs[:, :d]  # smallest eigenvalues of M
    J = sphere_project_basis(x0, Q @ A)
    # fixedness diagnostic on J
    fix = float(np.linalg.norm(JR @ J - J, "fro") / np.sqrt(max(d, 1)))
    return J, {
        "estimator": "sae_fixed_point_secant",
        "ok": True,
        "fixness": fix,
        "r_candidate": rmax,
    }


def undercomplete_ae_tangent(
    Xn, x0, d, ae_bundle=None, **kw
) -> tuple[np.ndarray | None, dict]:
    """Positive control: Im JD for undercomplete AE with latent=d."""
    if ae_bundle is None:
        return None, {"ok": False, "reason": "no_ae", "estimator": "undercomplete_ae_tangent"}
    model = ae_bundle["model"]
    W = model.dec.weight.detach().cpu().numpy().astype(np.float64)
    J = sphere_project_basis(x0, W * ae_bundle["scale"][:, None])
    return J, {"estimator": "undercomplete_ae_tangent", "ok": True}


def inner_pca(Xn, x0, d, k_tan: int = 256, **kw):
    k_tan = min(k_tan, len(Xn))
    J, _, diag = pca_tangent(Xn[:k_tan], x0, d)
    return J, {**diag, "estimator": "inner_pca", "ok": True}


def _wrap_ok(fn, name):
    def _f(*a, **k):
        J, diag = fn(*a, **k)
        return J, {**diag, "estimator": name, "ok": True}

    return _f


ESTIMATOR_FNS = {
    "same_patch_pca": _wrap_ok(same_patch_pca, "same_patch_pca"),
    "inner_pca": inner_pca,
    "bootstrap_grassmann_pca": _wrap_ok(bootstrap_grassmann_pca, "bootstrap_grassmann_pca"),
    "joint_quadratic_principal_manifold": _wrap_ok(
        joint_quadratic_principal_manifold, "joint_quadratic_principal_manifold"
    ),
    "sae_reconstruction_jacobian_top_svd": sae_reconstruction_jacobian_top_svd,
    "sae_code_covariance_pushforward": sae_code_covariance_pushforward,
    "sae_denoised_pca": lambda *a, **k: sae_denoised_pca(*a, weighted=False, **k),
    "sae_denoised_pca_weighted": lambda *a, **k: sae_denoised_pca(*a, weighted=True, **k),
    "sae_graph_pushforward": sae_graph_pushforward,
    "sae_fixed_point_secant": sae_fixed_point_secant,
    "undercomplete_ae_tangent": undercomplete_ae_tangent,
}


# -------------------- jacobian diagnostics (no full D×D) --------------------


def jvp_JR(bundle, x0, V, device) -> np.ndarray:
    """Apply JR to columns of V via active-set factorization (D,d) out."""
    model = bundle["model"]
    x = torch.tensor(((x0 - bundle["mean"]) / bundle["scale"])[None, :], device=device, dtype=torch.float32)
    with torch.no_grad():
        z_pre = F.relu(model.encoder(x))
        kk = min(model.k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, kk, dim=-1)
        mask = vals.squeeze(0) > 0
        if int(mask.sum()) < 1:
            return np.zeros_like(V)
        active = idx.squeeze(0)[mask]
        We = model.encoder.weight[active].detach().cpu().numpy()
        Wd = model.decoder.weight[:, active].detach().cpu().numpy()
    sc = bundle["scale"]
    # V ambient → scaled: V / sc; apply Jlin; * sc
    Vs = V / np.maximum(sc[:, None], EPS)
    out = (sc[:, None] * (Wd @ (We @ Vs)))
    return out


def diag_fixedness(bundle, x0, Jstar, device) -> float:
    JV = jvp_JR(bundle, x0, Jstar, device)
    d = Jstar.shape[1]
    return float(np.linalg.norm(JV - Jstar, "fro") / np.sqrt(max(d, 1)))


def diag_containment(bundle, x0, Jstar, device) -> dict:
    W = decoder_weight(bundle)
    # full decoder span projector via thin QR of W
    Q, _ = np.linalg.qr(W, mode="reduced")
    resid = Jstar - Q @ (Q.T @ Jstar)
    d = Jstar.shape[1]
    e_full = float(np.linalg.norm(resid, "fro") / np.sqrt(max(d, 1)))
    Ja = active_decoder_J(bundle, sae_encode_decode(bundle, x0[None, :], device)[0][0])
    if Ja.shape[1] >= 1:
        Qa, _ = np.linalg.qr(Ja, mode="reduced")
        resid_a = Jstar - Qa @ (Qa.T @ Jstar)
        e_act = float(np.linalg.norm(resid_a, "fro") / np.sqrt(max(d, 1)))
    else:
        e_act = float("nan")
    return {"E_contain_full": e_full, "E_contain_active": e_act}


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: SAETangentConfig) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sae_checkpoints").mkdir(exist_ok=True)
    (out / "synthetic").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "protocol": "sae_tangent_benchmark_v2",
        "v1_dir_untouched": V1_OUT,
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[sae2] prepare hash={meta['config_hash']}", flush=True)
    return meta


def _dataset_specs(cfg: SAETangentConfig) -> list[dict]:
    if cfg.smoke:
        return [
            {"family": "mlp_silu", "latent": "gaussian", "d": 12, "noise": "zero"},
            {"family": "mlp_silu", "latent": "gaussian", "d": 12, "noise": "empirical"},
            {"family": "quadratic", "latent": "gaussian", "d": 12, "noise": "empirical"},
        ]
    specs = []
    # all dims on smooth MLP
    for d in cfg.dims:
        for lat in ("gaussian", "laplace", "mixture", "truncated"):
            for noise in ("zero", "empirical"):
                specs.append({"family": "mlp_silu", "latent": lat, "d": d, "noise": noise})
    # stress at d=12
    for fam in ("linear", "quadratic", "sparse_gated", "piecewise", "stratified"):
        for lat in ("gaussian", "mixture"):
            for noise in ("zero", "empirical"):
                specs.append({"family": fam, "latent": lat, "d": cfg.primary_d, "noise": noise})
    return specs


def stage_synthetic(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_index.parquet"
    if _done(path, cfg.force):
        return
    D = 64 if cfg.smoke else cfg.ambient_D
    N = 1024 if cfg.smoke else cfg.n_global
    n_anch = 16 if cfg.smoke else cfg.n_anchors
    noise_emp = 0.01
    # calibrate from Physics if available
    try:
        Xphys = load_model_X(cfg.mm(root), cfg.model)
        pack = np.load(cfg.mm(root) / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz")
        a = int(pack["anchors_local"][0])
        dx = Xphys[pack["neigh"][0, :256]] - Xphys[a]
        _, _, vt = np.linalg.svd(dx - dx.mean(0), full_matrices=False)
        resid = dx - dx @ vt[:16].T @ vt[:16]
        noise_emp = float(np.sqrt(np.mean(resid**2)))
    except Exception:  # noqa: BLE001
        pass
    rows = []
    device = torch.device("cpu")  # embed on CPU
    for spec in _dataset_specs(cfg):
        key = f"{spec['family']}_{spec['latent']}_d{spec['d']}_{spec['noise']}"
        seed = _case_seed(cfg.seed, key)
        dec = SynthDecoder(spec["family"], spec["d"], D, seed)
        rng = np.random.default_rng(seed)
        Z = sample_latent(spec["latent"], N, spec["d"], rng)
        X = dec.embed(Z).astype(np.float32)
        if spec["noise"] == "empirical":
            X = normalize_rows(X + noise_emp * rng.normal(size=X.shape)).astype(np.float32)
        # ambient knn via torch IP (sphere)
        Xt = torch.tensor(X, device=torch.device("cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"))
        # pick anchors
        anchors = rng.choice(N, size=min(n_anch, N), replace=False)
        # store dataset
        np.savez_compressed(
            out / "synthetic" / f"{key}.npz",
            X=X,
            Z=Z.astype(np.float32),
            anchors=anchors.astype(np.int32),
            x0_frame=dec.x0,
            Jlin=dec.Jlin,
            family=np.array([spec["family"]]),
            d=np.array([spec["d"]]),
            D=np.array([D]),
            seed=np.array([seed]),
            noise_sigma=np.array([noise_emp if spec["noise"] == "empirical" else 0.0]),
        )
        # save decoder mlp if needed
        if dec.mlp is not None:
            torch.save(dec.mlp.state_dict(), out / "synthetic" / f"{key}_mlp.pt")
        rows.append({**spec, "key": key, "N": N, "D": D, "n_anchors": len(anchors), "noise_sigma": noise_emp if spec["noise"] == "empirical" else 0.0})
        del Xt
        print(f"[sae2][syn] {key}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    (out / "calibration.json").write_text(json.dumps({"noise_empirical": noise_emp}, indent=2))
    print(f"[sae2] synthetic n_datasets={len(rows)}", flush=True)


def _load_decoder(out: Path, row, D: int) -> SynthDecoder:
    seed = int(np.load(out / "synthetic" / f"{row.key}.npz")["seed"][0])
    dec = SynthDecoder(row.family, int(row.d), D, seed)
    if row.family == "mlp_silu":
        mlp_path = out / "synthetic" / f"{row.key}_mlp.pt"
        if mlp_path.exists():
            dec.mlp.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
    return dec


def stage_train_sae(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "sae_train_metrics.parquet"
    if _done(path, cfg.force):
        return
    idx = pd.read_parquet(out / "synthetic_index.parquet")
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows = []
    seeds = [cfg.seed] if cfg.smoke else cfg.sae_seeds
    steps = 80 if cfg.smoke else cfg.sae_steps
    fdim = 128 if cfg.smoke else cfg.sae_feature_dim
    kk = 16 if cfg.smoke else cfg.sae_k
    t0 = time.time()
    for _, row in idx.iterrows():
        z = np.load(out / "synthetic" / f"{row.key}.npz")
        X = z["X"]
        # held-out split for checkpoint selection by recon only
        rng = np.random.default_rng(cfg.seed + 11)
        perm = rng.permutation(len(X))
        n_te = max(128, len(X) // 5)
        te, tr = perm[:n_te], perm[n_te:]
        for sd in seeds:
            if time.time() - t0 > cfg.max_seconds * 0.35:
                break
            bundle = train_synthetic_sae(
                X[tr],
                feature_dim=fdim,
                k=kk,
                steps=steps,
                seed=sd,
                device=device,
            )
            # held-out recon
            Zte, Xh = sae_encode_decode(bundle, X[te], device)
            Xh_amb = Xh * bundle["scale"] + bundle["mean"]
            mse = float(np.mean((Xh_amb - X[te]) ** 2))
            cos = float(np.mean(np.sum(normalize_rows(Xh_amb) * X[te], axis=1)))
            # knn radius proxy
            rho = float(np.median(np.linalg.norm(X[te] - X[te].mean(0), axis=1)))
            active = float(np.mean((Zte > 0).sum(axis=1)))
            # support turnover between halves of te
            half = len(te) // 2
            s1 = (Zte[:half] > 0).mean(0)
            s2 = (Zte[half:] > 0).mean(0)
            turnover = float(np.mean(np.abs(s1 - s2)))
            W = decoder_weight(bundle)
            cond = float(np.linalg.cond(W[:, : min(64, W.shape[1])]))
            ckpt = out / "sae_checkpoints" / f"{row.key}_seed{sd}.pt"
            torch.save(
                {
                    "state_dict": bundle["model"].state_dict(),
                    "mean": bundle["mean"],
                    "scale": bundle["scale"],
                    "feature_dim": fdim,
                    "k": kk,
                    "input_dim": X.shape[1],
                    "mse": mse,
                },
                ckpt,
            )
            rows.append(
                {
                    "key": row.key,
                    "family": row.family,
                    "latent": row.latent,
                    "d": int(row.d),
                    "noise": row.noise,
                    "seed": sd,
                    "mse": mse,
                    "cosine": cos,
                    "mse_over_rho2": mse / max(rho**2, EPS),
                    "active_mean": active,
                    "support_turnover": turnover,
                    "decoder_cond": cond,
                    "ckpt": str(ckpt.relative_to(out)),
                }
            )
            # undercomplete AE positive control (one seed)
            if sd == seeds[0]:
                ae = train_undercomplete_ae(X[tr], int(row.d), steps=max(60, steps // 2), seed=sd, device=device)
                torch.save(
                    {"state_dict": ae["model"].state_dict(), "mean": ae["mean"], "scale": ae["scale"], "d": int(row.d)},
                    out / "sae_checkpoints" / f"{row.key}_ae.pt",
                )
        print(f"[sae2][train] {row.key}", flush=True)
    # seed agreement
    df = pd.DataFrame(rows)
    if len(seeds) > 1 and len(df):
        ag = []
        for key, g in df.groupby("key"):
            if len(g) >= 2:
                ag.append({"key": key, "seed_mse_std": float(g.mse.std()), "seed_mse_mean": float(g.mse.mean())})
        pd.DataFrame(ag).to_parquet(out / "sae_seed_agreement.parquet", index=False)
    # select best seed per key by held-out mse
    if len(df):
        best = df.sort_values("mse").groupby("key", as_index=False).first()
        best.to_parquet(out / "sae_selected_checkpoints.parquet", index=False)
    df.to_parquet(path, index=False)
    print(f"[sae2] train_sae n={len(df)}", flush=True)


def _load_ckpt(out: Path, rel: str, device: torch.device) -> dict:
    _ensure_sae_path()
    from sae.sae_model import TopKSAE  # type: ignore

    ck = torch.load(out / rel, map_location=device, weights_only=False)
    model = TopKSAE(ck["input_dim"], ck["feature_dim"], ck["k"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(ck["mean"], dtype=np.float64),
        "scale": np.asarray(ck["scale"], dtype=np.float64),
        "device": device,
    }


def _knn_indices(X: np.ndarray, anchors: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    Xt = torch.tensor(X, device=device, dtype=torch.float32)
    At = Xt[torch.tensor(anchors, device=device)]
    # chunked IP knn
    sims = At @ Xt.T
    # exclude self
    for i, a in enumerate(anchors):
        sims[i, int(a)] = -1e9
    _, idx = torch.topk(sims, k=k, dim=1)
    return idx.cpu().numpy()


def stage_tangents(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "tangent_errors.parquet"
    if _done(path, cfg.force):
        return
    idx = pd.read_parquet(out / "synthetic_index.parquet")
    sel = pd.read_parquet(out / "sae_selected_checkpoints.parquet")
    sel_map = dict(zip(sel.key, sel.ckpt))
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    ks = [256, 512] if cfg.smoke else cfg.k_list
    est_names = [
        "same_patch_pca",
        "inner_pca",
        "bootstrap_grassmann_pca",
        "joint_quadratic_principal_manifold",
        "sae_reconstruction_jacobian_top_svd",
        "sae_code_covariance_pushforward",
        "sae_denoised_pca",
        "sae_denoised_pca_weighted",
        "sae_graph_pushforward",
        "sae_fixed_point_secant",
        "undercomplete_ae_tangent",
    ]
    rows = []
    t0 = time.time()
    for _, row in idx.iterrows():
        z = np.load(out / "synthetic" / f"{row.key}.npz")
        X, Zlat, anchors = z["X"], z["Z"], z["anchors"]
        D = int(z["D"][0])
        d = int(row.d)
        dec = _load_decoder(out, row, D)
        sae = _load_ckpt(out, sel_map[row.key], device) if row.key in sel_map else None
        ae = None
        ae_path = out / "sae_checkpoints" / f"{row.key}_ae.pt"
        if ae_path.exists():
            ck = torch.load(ae_path, map_location=device, weights_only=False)
            ae_m = UndercompleteAE(D, d).to(device)
            ae_m.load_state_dict(ck["state_dict"])
            ae_m.eval()
            ae = {"model": ae_m, "mean": ck["mean"], "scale": ck["scale"], "device": device}
        for k in ks:
            if k >= len(X):
                continue
            neigh = _knn_indices(X, anchors, k, device)
            for ai, a in enumerate(anchors):
                if time.time() - t0 > cfg.max_seconds * 0.55:
                    break
                x0 = X[int(a)]
                Xn = X[neigh[ai]]
                Jstar = dec.true_tangent(Zlat[int(a)])
                # density / boundary proxies
                dens = float(np.median(np.linalg.norm(Xn - x0, axis=1)))
                for name in est_names:
                    fn = ESTIMATOR_FNS[name]
                    t1 = time.time()
                    try:
                        J, diag = fn(
                            Xn,
                            x0,
                            d,
                            sae_bundle=sae,
                            ae_bundle=ae,
                            device=device,
                            n_boot=4,
                            seed=cfg.seed + int(a),
                            k_tan=min(256, k // 2),
                            n_iter=4 if cfg.smoke else 6,
                        )
                    except Exception as e:  # noqa: BLE001
                        rows.append(
                            {
                                "key": row.key,
                                "family": row.family,
                                "latent": row.latent,
                                "d": d,
                                "k": k,
                                "noise": row.noise,
                                "anchor": int(a),
                                "estimator": name,
                                "ok": False,
                                "error": type(e).__name__,
                            }
                        )
                        continue
                    if J is None:
                        rows.append(
                            {
                                "key": row.key,
                                "family": row.family,
                                "latent": row.latent,
                                "d": d,
                                "k": k,
                                "noise": row.noise,
                                "anchor": int(a),
                                "estimator": name,
                                "ok": False,
                                "reason": diag.get("reason"),
                            }
                        )
                        continue
                    et = E_T_thin(J, Jstar)
                    ang = principal_angles(J, Jstar)
                    rows.append(
                        {
                            "key": row.key,
                            "family": row.family,
                            "latent": row.latent,
                            "d": d,
                            "k": k,
                            "noise": row.noise,
                            "anchor": int(a),
                            "estimator": name,
                            "ok": True,
                            "E_T": et,
                            "ang_rms": float(np.sqrt(np.mean(ang**2))),
                            "ang_p90": float(np.quantile(ang, 0.9)),
                            "knn_radius": dens,
                            "runtime_s": time.time() - t1,
                            "agree_direct_ET": diag.get("agree_direct_ET", np.nan),
                            "stratified": row.family == "stratified",
                        }
                    )
            print(f"[sae2][tan] {row.key} k={k}", flush=True)
        if time.time() - t0 > cfg.max_seconds * 0.55:
            break
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[sae2] tangents n={len(rows)}", flush=True)


def stage_jacobian_diagnostics(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "jacobian_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    idx = pd.read_parquet(out / "synthetic_index.parquet")
    sel = pd.read_parquet(out / "sae_selected_checkpoints.parquet")
    sel_map = dict(zip(sel.key, sel.ckpt))
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows = []
    for _, row in idx.iterrows():
        if row.key not in sel_map:
            continue
        z = np.load(out / "synthetic" / f"{row.key}.npz")
        X, Zlat, anchors = z["X"], z["Z"], z["anchors"]
        D = int(z["D"][0])
        dec = _load_decoder(out, row, D)
        sae = _load_ckpt(out, sel_map[row.key], device)
        # few anchors
        for a in anchors[: min(8, len(anchors))]:
            x0 = X[int(a)]
            Jstar = dec.true_tangent(Zlat[int(a)])
            efix = diag_fixedness(sae, x0, Jstar, device)
            cont = diag_containment(sae, x0, Jstar, device)
            # idempotence on tangent / normal
            JRJ = jvp_JR(sae, x0, Jstar, device)
            idemp_t = float(np.linalg.norm(jvp_JR(sae, x0, JRJ, device) - JRJ, "fro") / np.sqrt(Jstar.shape[1]))
            # support sensitivity: perturb x0
            rng = np.random.default_rng(int(a))
            xp = normalize_rows((x0 + 1e-3 * rng.normal(size=x0.shape))[None, :])[0]
            # compare active sets
            s0, _ = sae_encode_decode(sae, x0[None, :], device)
            sp, _ = sae_encode_decode(sae, xp[None, :], device)
            jacc = float(np.sum((s0[0] > 0) & (sp[0] > 0)) / max(np.sum((s0[0] > 0) | (sp[0] > 0)), 1))
            rows.append(
                {
                    "key": row.key,
                    "family": row.family,
                    "latent": row.latent,
                    "d": int(row.d),
                    "noise": row.noise,
                    "anchor": int(a),
                    "E_fix": efix,
                    "idemp_tangent": idemp_t,
                    "support_jaccard": jacc,
                    **cont,
                }
            )
        print(f"[sae2][jac] {row.key}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[sae2] jacobian_diagnostics n={len(rows)}", flush=True)


def stage_gauss(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "gauss_map_recovery.parquet"
    if _done(path, cfg.force):
        return
    err = pd.read_parquet(out / "tangent_errors.parquet")
    ok = err[err.ok == True]  # noqa: E712
    if ok.empty:
        pd.DataFrame([]).to_parquet(path, index=False)
        return
    # best raw / best SAE / joint
    raw_names = ["same_patch_pca", "inner_pca", "bootstrap_grassmann_pca"]
    sae_names = [e for e in ok.estimator.unique() if e.startswith("sae_") and "top_svd" not in e and e != "sae_reconstruction_jacobian_top_svd"]
    best_raw = ok[ok.estimator.isin(raw_names)].groupby("estimator").E_T.median().sort_values().index[0]
    best_sae = (
        ok[ok.estimator.isin(sae_names)].groupby("estimator").E_T.median().sort_values().index[0]
        if sae_names and ok.estimator.isin(sae_names).any()
        else None
    )
    run_ests = [best_raw, "joint_quadratic_principal_manifold"]
    if best_sae:
        run_ests.append(best_sae)
    (out / "gauss_estimators.json").write_text(json.dumps({"estimators": run_ests}, indent=2))
    idx = pd.read_parquet(out / "synthetic_index.parquet")
    sel = pd.read_parquet(out / "sae_selected_checkpoints.parquet")
    sel_map = dict(zip(sel.key, sel.ckpt))
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    # only flat + curved + stratified at primary d, k=512
    rows = []
    for _, row in idx.iterrows():
        if row.family not in ("linear", "mlp_silu", "quadratic", "stratified"):
            continue
        if int(row.d) != (12 if cfg.smoke else cfg.primary_d) and not (cfg.smoke and int(row.d) == 12):
            continue
        z = np.load(out / "synthetic" / f"{row.key}.npz")
        X, anchors = z["X"], z["anchors"]
        d = int(row.d)
        sae = _load_ckpt(out, sel_map[row.key], device) if row.key in sel_map else None
        k = 256 if cfg.smoke else 512
        neigh = _knn_indices(X, anchors[: min(8, len(anchors))], k, device)
        for name in run_ests:
            fn = ESTIMATOR_FNS[name]
            for ai, a in enumerate(anchors[: min(8, len(anchors))]):
                x0 = X[int(a)]
                Xn = X[neigh[ai]]
                try:
                    J, diag = fn(Xn, x0, d, sae_bundle=sae, device=device, n_iter=3, seed=cfg.seed)
                except Exception:  # noqa: BLE001
                    continue
                if J is None:
                    continue
                Px = J @ J.T  # only for split helper; secondary uses thin
                _, _, split_x = split_half_projectors(Xn, x0, d, cfg.seed + int(a), pca_tangent)
                sites, splits = [], []
                dists = np.linalg.norm(X - x0, axis=1)
                order = np.argsort(dists)
                for rnk in (8, 16, 32, 64):
                    y = X[order[rnk]]
                    Yn = X[order[1:129]]
                    Jy, _, _ = pca_tangent(Yn, y, d)
                    # sphere-transport Jy → x0 for ΔP² via thin formula after PT
                    Jy_at = parallel_transport_basis(y, x0, Jy)
                    sites.append((y, Jy_at @ Jy_at.T))
                    _, _, sj = split_half_projectors(Yn, y, d, rnk, pca_tangent)
                    splits.append(sj)
                if len(sites) < 4:
                    continue
                g = estimate_anchor_gauss_map(x0, Px, sites, split_x, splits, d)
                rows.append(
                    {
                        "key": row.key,
                        "family": row.family,
                        "estimator": name,
                        "anchor": int(a),
                        "beta": g["beta"],
                        "energy": g["curvature_energy"],
                        "label": g["label"],
                        "false_curvature": bool(row.family == "linear" and np.isfinite(g["curvature_energy"]) and g["curvature_energy"] > 0.08),
                    }
                )
        print(f"[sae2][gauss] {row.key}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[sae2] gauss n={len(rows)} ests={run_ests}", flush=True)


def stage_select(root: Path, cfg: SAETangentConfig) -> dict:
    out = cfg.resolved(root)
    path = out / "SAE_TANGENT_FREEZE.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    err = pd.read_parquet(out / "tangent_errors.parquet")
    jac = (
        pd.read_parquet(out / "jacobian_diagnostics.parquet")
        if (out / "jacobian_diagnostics.parquet").exists()
        else pd.DataFrame()
    )
    gauss = (
        pd.read_parquet(out / "gauss_map_recovery.parquet")
        if (out / "gauss_map_recovery.parquet").exists()
        else pd.DataFrame()
    )
    ok = err[err.ok == True].copy()  # noqa: E712
    reject: set[str] = set()
    # recon/rank: SAE methods with high failure rate
    for est, g in err.groupby("estimator"):
        if str(est).startswith("sae_") and g.ok.mean() < 0.5:
            reject.add(est)
    # fixedness gate: if E_fix huge on average, mark sae_reconstructs_but_derivative_fails later
    mean_fix = float(jac.E_fix.median()) if len(jac) else float("nan")
    # false curvature
    if len(gauss):
        for est, g in gauss[gauss.family == "linear"].groupby("estimator"):
            if float(g.false_curvature.mean()) > 0.5:
                reject.add(est)
    # stratified: confidently unique tangent → reject
    strat = ok[ok.stratified == True]  # noqa: E712
    for est, g in strat.groupby("estimator"):
        if len(g) and float(g.E_T.median()) < 0.15:
            reject.add(est)
            print(f"[sae2] reject {est}: stratified confidence", flush=True)
    # selection conditions: nonlinear, non-Gaussian, empirical noise
    focus = ok[
        (ok.family.isin(["mlp_silu", "quadratic", "sparse_gated"]))
        & (ok.latent.isin(["laplace", "mixture", "truncated", "gaussian"]))
        & (ok.noise == "empirical")
        & (~ok.estimator.isin(reject))
        & (ok.estimator != "undercomplete_ae_tangent")
        & (ok.estimator != "sae_reconstruction_jacobian_top_svd")
    ]
    if focus.empty:
        focus = ok[(~ok.estimator.isin(reject)) & (ok.estimator != "undercomplete_ae_tangent")]
    # paired median E_T
    scores = focus.groupby("estimator").E_T.median().sort_values()
    sae_scores = scores[[e for e in scores.index if str(e).startswith("sae_")]]
    raw_scores = scores[[e for e in scores.index if not str(e).startswith("sae_")]]
    # choose
    if scores.empty:
        chosen, label = "same_patch_pca", "no_reliable_tangent_estimator"
    else:
        best = scores.index[0]
        se = {
            e: float(np.nanstd(g.E_T) / np.sqrt(max(len(g), 1)))
            for e, g in focus.groupby("estimator")
        }
        thresh = float(scores.iloc[0] + se.get(best, 0.0))
        near = [e for e in scores.index if scores.loc[e] <= thresh]
        # simplicity preference among near
        pref = [
            "same_patch_pca",
            "inner_pca",
            "bootstrap_grassmann_pca",
            "joint_quadratic_principal_manifold",
            "sae_denoised_pca",
            "sae_denoised_pca_weighted",
            "sae_graph_pushforward",
            "sae_code_covariance_pushforward",
            "sae_fixed_point_secant",
        ]
        chosen = next((p for p in pref if p in near), best)
        if chosen.startswith("sae_graph"):
            label = "sae_graph_tangent_validated"
        elif "denoised" in chosen:
            label = "sae_denoised_tangent_validated"
        elif "fixed_point" in chosen:
            label = "sae_fixed_point_tangent_validated"
        elif chosen.startswith("joint"):
            label = "joint_quadratic_tangent_preferred"
        elif chosen.startswith("sae_"):
            label = "sae_denoised_tangent_validated"
        else:
            label = "pca_tangent_preferred"
        if float(scores.iloc[0]) > 0.4:
            label = "no_reliable_tangent_estimator"
        if np.isfinite(mean_fix) and mean_fix > 0.7 and not chosen.startswith("sae_"):
            # SAE reconstructs but derivative fails (top-svd / fixedness)
            if len(sae_scores) and float(sae_scores.iloc[0]) > float(raw_scores.iloc[0]) + 0.05:
                label = "sae_reconstructs_but_derivative_fails"
    # top-svd vs best SAE to answer selection-rule question
    top_svd = ok[ok.estimator == "sae_reconstruction_jacobian_top_svd"].E_T.median() if (ok.estimator == "sae_reconstruction_jacobian_top_svd").any() else float("nan")
    best_sae_et = float(sae_scores.iloc[0]) if len(sae_scores) else float("nan")
    freeze = {
        "estimator": chosen,
        "synthetic_label": label,
        "median_E_T": float(scores.loc[chosen]) if chosen in scores.index else float("nan"),
        "scores": {k: float(v) for k, v in scores.items()},
        "rejected": sorted(reject),
        "mean_E_fix": mean_fix,
        "top_svd_median_ET": float(top_svd) if np.isfinite(top_svd) else None,
        "best_sae_median_ET": best_sae_et if np.isfinite(best_sae_et) else None,
        "top_svd_was_selection_failure": bool(
            np.isfinite(top_svd) and np.isfinite(best_sae_et) and best_sae_et + 0.05 < top_svd
        ),
        "config_hash": json.loads((out / "resolved_config.json").read_text())["config_hash"],
        "primary_d": cfg.primary_d,
        "sae_passes_gates": bool(chosen.startswith("sae_") and label not in ("no_reliable_tangent_estimator", "sae_reconstructs_but_derivative_fails")),
    }
    path.write_text(json.dumps(freeze, indent=2))
    (out / "config_hash").write_text(freeze["config_hash"])
    print(f"[sae2] FREEZE {chosen} label={label}", flush=True)
    return freeze


def stage_physics(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "physics_geometry.parquet"
    freeze = json.loads((out / "SAE_TANGENT_FREEZE.json").read_text())
    if not freeze.get("sae_passes_gates"):
        print("[sae2] SAE did not pass synthetic gates — skipping Physics SAE application", flush=True)
        pd.DataFrame([{"skipped": True, "reason": freeze["synthetic_label"]}]).to_parquet(path, index=False)
        (out / "geometry_tables_complete.json").write_text(
            json.dumps({"complete": True, "physics_sae_applied": False, "hash": freeze["config_hash"]}, indent=2)
        )
        return
    if _done(path, cfg.force):
        return
    # Physics SAE application
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    _ensure_sae_path()
    from sae_affine_basis_mknn_gpu import load_sae  # type: ignore

    sae_dir = resolve_path(root, cfg.physics_sae_dir)
    sae = load_sae(sae_dir, device)
    sae["device"] = device
    mm = cfg.mm(root)
    X = load_model_X(mm, cfg.model)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    all512 = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(all512.read_text())["sample_ids"] if all512.exists() else anchors_sid.tolist()
    if cfg.smoke:
        use_sids = use_sids[:8]
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    # verify recon quality
    sample = X[:: max(1, len(X) // 2048)]
    _, Xh = sae_encode_decode(sae, sample, device)
    Xh_amb = Xh * sae["scale"] + sae["mean"]
    mse = float(np.mean((Xh_amb - sample) ** 2))
    (out / "physics_sae_verify.json").write_text(
        json.dumps(
            {
                "sae_dir": str(sae_dir),
                "mse": mse,
                "n_verify": len(sample),
                "sample_id_alignment": "anchors from multimodel prepare",
            },
            indent=2,
        )
    )
    fn = ESTIMATOR_FNS[freeze["estimator"]]
    dims = [16, 12] if not cfg.smoke else [16]
    rows = []
    t0 = time.time()
    for d in dims:
        for si, sid in enumerate(use_sids):
            if si % 32 == 0:
                print(f"[sae2][phys] d={d} {si}/{len(use_sids)}", flush=True)
            ai = sid_to_ai[int(sid)]
            x0 = X[int(anchors_local[ai])]
            N = pack["neigh"][ai]
            Xn = X[N[:512]]
            Jpca, _ = same_patch_pca(Xn, x0, d)
            Jsae, diag = fn(Xn, x0, d, sae_bundle=sae, device=device)
            if Jsae is None:
                continue
            ang = principal_angles(Jsae, Jpca)
            rows.append(
                {
                    "sample_id": int(sid),
                    "d": d,
                    "estimator": freeze["estimator"],
                    "ET_vs_pca": E_T_thin(Jsae, Jpca),
                    "ang_rms_vs_pca": float(np.sqrt(np.mean(ang**2))),
                    "agree_direct_ET": diag.get("agree_direct_ET", np.nan),
                }
            )
            if time.time() - t0 > cfg.max_seconds * 0.9:
                break
        if time.time() - t0 > cfg.max_seconds * 0.9:
            break
    pd.DataFrame(rows).to_parquet(path, index=False)
    (out / "geometry_tables_complete.json").write_text(
        json.dumps({"complete": True, "physics_sae_applied": True, "hash": freeze["config_hash"]}, indent=2)
    )
    print(f"[sae2] physics n={len(rows)}", flush=True)


def stage_probe(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    path = out / "probe_joins.parquet"
    assert (out / "SAE_TANGENT_FREEZE.json").exists()
    assert (out / "geometry_tables_complete.json").exists(), "geometry must complete before probes"
    freeze = json.loads((out / "SAE_TANGENT_FREEZE.json").read_text())
    (out / "freeze_locked_before_probe.json").write_text(json.dumps(freeze, indent=2))
    if _done(path, cfg.force):
        return
    phys = pd.read_parquet(out / "physics_geometry.parquet")
    if "skipped" in phys.columns or len(phys) == 0 or "sample_id" not in phys.columns:
        pd.DataFrame([]).to_parquet(path, index=False)
        pd.DataFrame([]).to_parquet(out / "probe_control_path.parquet", index=False)
        return
    mm = cfg.mm(root)
    fields = pd.read_parquet(mm / "local_probe_fields.parquet")
    fields = fields[
        (fields.model == cfg.model)
        & (fields.target == cfg.target)
        & (fields.neighbourhood == "model")
        & (fields.scale_k == 2048)
    ]
    g = phys[phys.d == 16].merge(fields, on="sample_id", how="inner")
    g.to_parquet(path, index=False)
    rows = []
    y = g.local_r2.to_numpy(float)
    for col in ("ET_vs_pca", "ang_rms_vs_pca"):
        if col not in g.columns:
            continue
        x = g[col].to_numpy(float)
        raw = spearman_dict(x, y)
        C = g.log_knn_radius.to_numpy(float)[:, None]
        pathc = {"raw": raw["rho"], "radius_only": partial_spearman(x, y, C)["rho"]}
        Z = C
        for name, c in [
            ("label_var", g.local_label_variance.to_numpy(float)),
            ("eval_count", g.local_evaluation_count.to_numpy(float)),
        ]:
            Z = np.column_stack([Z, c])
            pathc[f"+{name}"] = partial_spearman(x, y, Z)["rho"]
        pathc["full_partial"] = list(pathc.values())[-1]
        rows.append({"stat": col, "n": int(np.isfinite(x).sum()), **pathc})
    pd.DataFrame(rows).to_parquet(out / "probe_control_path.parquet", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def stage_analyze(root: Path, cfg: SAETangentConfig) -> None:
    out = cfg.resolved(root)
    freeze = json.loads((out / "SAE_TANGENT_FREEZE.json").read_text())
    err = pd.read_parquet(out / "tangent_errors.parquet")
    jac = (
        pd.read_parquet(out / "jacobian_diagnostics.parquet")
        if (out / "jacobian_diagnostics.parquet").exists()
        else pd.DataFrame()
    )
    ok = err[err.ok == True]  # noqa: E712
    # plots
    if len(ok):
        fig, ax = plt.subplots(figsize=(8, 4))
        med = ok.groupby("estimator").E_T.median().sort_values()
        ax.barh(med.index.astype(str), med.values)
        ax.set_xlabel("median E_T")
        ax.set_title("SAE tangent v2 synthetic errors")
        fig.tight_layout()
        fig.savefig(out / "figures" / "median_ET_by_estimator.png", dpi=140)
        plt.close(fig)
    # answers
    top_svd = ok[ok.estimator == "sae_reconstruction_jacobian_top_svd"]
    graph = ok[ok.estimator == "sae_graph_pushforward"]
    pca = ok[ok.estimator == "same_patch_pca"]
    answers = {
        "good_recon_implies_fixedness": bool(len(jac) and float(jac.E_fix.median()) < 0.35),
        "median_E_fix": float(jac.E_fix.median()) if len(jac) else None,
        "top_svd_failure_is_selection_rule": freeze.get("top_svd_was_selection_failure"),
        "best_sae_estimator": min(
            ((e, float(g.E_T.median())) for e, g in ok[ok.estimator.str.startswith("sae_")].groupby("estimator")),
            key=lambda t: t[1],
            default=(None, None),
        )[0],
        "outperforms_pca_outside_gaussian": None,
        "frozen": freeze,
    }
    # nonlinear non-gaussian comparison
    sub = ok[(ok.family == "mlp_silu") & (ok.latent != "gaussian") & (ok.noise == "empirical")]
    if len(sub) and answers["best_sae_estimator"]:
        answers["outperforms_pca_outside_gaussian"] = bool(
            sub[sub.estimator == answers["best_sae_estimator"]].E_T.median()
            < sub[sub.estimator == "same_patch_pca"].E_T.median()
        )
    (out / "scientific_answers.json").write_text(json.dumps(answers, indent=2, default=str))
    report = f"""# SAE tangent benchmark v2

## Question

Given a well-reconstructing SAE, can the tangent of the encoded data manifold be estimated in code space and pushed through the decoder Jacobian to recover the ambient tangent?

## Freeze (synthetic only)

- Estimator: `{freeze['estimator']}`
- Label: `{freeze['synthetic_label']}`
- Median E_T: {freeze.get('median_E_T')}
- Config hash: `{freeze.get('config_hash')}`
- Top-SVD was selection-rule failure: {freeze.get('top_svd_was_selection_failure')}
- Mean E_fix: {freeze.get('mean_E_fix')}

## Scores (focus median E_T)

```json
{json.dumps(freeze.get('scores', {}), indent=2)}
```

## Answers

1. **Good recon ⇒ tangent fixedness?** {answers['good_recon_implies_fixedness']} (median E_fix={answers['median_E_fix']})
2. **Was v1 SAE failure the top-SVD rule?** {answers['top_svd_failure_is_selection_rule']}
3. **Best SAE construction:** `{answers['best_sae_estimator']}`
4. **Beats PCA outside Gaussian regimes?** {answers['outperforms_pca_outside_gaussian']}
5. **Justified for Physics?** {freeze.get('sae_passes_gates')}

## Notes

- `sae_reconstruction_jacobian_top_svd` is retained only as a negative/diagnostic control.
- v1 outputs under `{V1_OUT}` were not modified.
- Probe joins (if any) are post-freeze only.
"""
    (out / "REPORT.md").write_text(report)
    pd.DataFrame([{"stage": "synthetic", "label": freeze["synthetic_label"]}]).to_csv(
        out / "decision_labels.csv", index=False
    )
    print(f"[sae2] analyze label={freeze['synthetic_label']}", flush=True)


def run(cfg: SAETangentConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    profile: dict[str, Any] = {"stages": {}}
    stages = {
        "prepare": stage_prepare,
        "synthetic": stage_synthetic,
        "train_sae": stage_train_sae,
        "tangents": stage_tangents,
        "jacobian_diagnostics": stage_jacobian_diagnostics,
        "gauss": stage_gauss,
        "select": stage_select,
        "physics": stage_physics,
        "probe": stage_probe,
        "analyze": stage_analyze,
    }
    order = [
        "prepare",
        "synthetic",
        "train_sae",
        "tangents",
        "jacobian_diagnostics",
        "gauss",
        "select",
        "physics",
        "probe",
        "analyze",
    ]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    deps = {
        "synthetic": ["prepare"],
        "train_sae": ["synthetic"],
        "tangents": ["train_sae"],
        "jacobian_diagnostics": ["train_sae"],
        "gauss": ["tangents"],
        "select": ["tangents", "jacobian_diagnostics", "gauss"],
        "physics": ["select"],
        "probe": ["physics", "select"],
        "analyze": ["select", "probe"],
    }
    run_set = set(want)
    for s in want:
        for d in deps.get(s, []):
            run_set.add(d)
    for s in order:
        if s not in run_set:
            continue
        t1 = time.time()
        print(f"[sae2] stage={s}", flush=True)
        stages[s](root, cfg)
        profile["stages"][f"{s}_s"] = time.time() - t1
        if time.time() - t0 > cfg.max_seconds:
            print("[sae2] time budget", flush=True)
            break
    profile.update(
        {
            "total_seconds": time.time() - t0,
            "peak_rss_mb": _rss(),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[sae2] done in {profile['total_seconds']:.1f}s", flush=True)
    return profile

