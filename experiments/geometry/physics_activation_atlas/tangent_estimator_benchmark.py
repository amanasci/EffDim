"""Ground-truth tangent-estimator benchmark + frozen Gauss-map Physics analysis.

Estimator selection uses synthetic manifolds only — never probe performance.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .curvature_probe_screen import partial_spearman, spearman_dict
from .gauss_map_curvature import (
    estimate_anchor_gauss_map,
    split_half_projectors,
)
from .multimodel_graph_prior_quadratic import load_model_X
from .paths import platonic_root, resolve_path
from .quadratic import quadratic_features
from .sphere_normal_quadratic import normalize_rows, normal_projector_apply, sphere_project_basis
from .tangent_reliability import (
    bootstrap_grassmann_tangent,
    grassmann_dist,
    kernel_weights,
    pca_tangent,
    principal_angles,
    projector,
)

EPS = 1e-12
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SAE_DEFAULT = "outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0"


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


@dataclass
class BenchmarkConfig:
    output_dir: str = "outputs/geometry/physics_tangent_estimator_benchmark"
    multimodel_dir: str = SOURCE_MM
    sae_dir: str = SAE_DEFAULT
    model: str = "vit_base"
    target: str = "mag_r_desi"
    ambient_D: int = 768
    dims: list[int] = field(default_factory=lambda: [8, 12, 16])
    primary_d: int = 16
    k_tier1: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    k_tier2: list[int] = field(default_factory=lambda: [2048])
    n_synth_train: int = 4096
    n_synth_anchors: int = 48
    n_boot: int = 6
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    smoke: bool = False
    max_seconds: float = 14400.0
    physics_secondary_ranks: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024, 2048]
    )
    n_secondary_per_anchor: int = 24

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


# -------------------- synthetic truth --------------------


def _rand_sphere_frame(rng: np.random.Generator, D: int, d: int):
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= max(np.linalg.norm(x0), EPS)
    J = sphere_project_basis(x0, J)
    return x0.astype(np.float64), J.astype(np.float64)


def _pack_BS(d: int, H_amp: float, traceless_amp: float, rng: np.random.Generator, nvec: np.ndarray):
    D = len(nvec)
    q = d * (d + 1) // 2
    BS = np.zeros((D, q))
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                BS[:, idx] += H_amp * nvec
            else:
                g = rng.normal(size=D)
                g = g - np.dot(g, nvec) * nvec
                g /= max(np.linalg.norm(g), EPS)
                BS[:, idx] += traceless_amp * g
            idx += 1
    return BS


def sample_manifold(
    family: str,
    d: int,
    D: int,
    n: int,
    noise: float,
    seed: int,
    radius: float = 0.25,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x0, J = _rand_sphere_frame(rng, D, d)
    nvec = normal_projector_apply(rng.normal(size=D), x0, J)
    nvec /= max(np.linalg.norm(nvec), EPS)
    if family == "geodesic":
        BS = np.zeros((D, d * (d + 1) // 2))
        H_true = np.zeros(D)
        B0_true = 0.0
    elif family == "pure_mean":
        BS = _pack_BS(d, 0.8, 0.0, rng, nvec)
        H_true = 0.8 * nvec
        B0_true = 0.0
    elif family == "pure_traceless":
        BS = _pack_BS(d, 0.0, 0.7, rng, nvec)
        H_true = np.zeros(D)
        B0_true = 0.7
    elif family == "high_rank":
        BS = _pack_BS(d, 0.4, 0.5, rng, nvec)
        H_true = 0.4 * nvec
        B0_true = 0.5
    elif family in ("nonuniform", "boundary", "stratified"):
        BS = _pack_BS(d, 0.5, 0.3, rng, nvec)
        H_true = 0.5 * nvec
        B0_true = 0.3
    else:
        raise ValueError(family)

    if family == "nonuniform":
        U = rng.normal(size=(n, d))
        U = U * (0.15 + 0.35 * (U[:, 0:1] > 0))
        U *= radius
    elif family == "boundary":
        U = np.abs(rng.normal(size=(n, d))) * radius
    elif family == "stratified":
        J2, _ = np.linalg.qr(rng.normal(size=(D, d)))
        J2 = sphere_project_basis(x0, J2)
        half = n // 2
        U1 = rng.normal(size=(half, d)) * radius
        U2 = rng.normal(size=(n - half, d)) * radius
        X1 = normalize_rows(x0 + U1 @ J.T)
        X2 = normalize_rows(x0 + U2 @ J2.T)
        X = np.vstack([X1, X2])
        if noise > 0:
            X = normalize_rows(X + noise * rng.normal(size=X.shape))
        return {
            "family": family,
            "X": X.astype(np.float32),
            "x0": x0,
            "J_true": J,
            "J_alt": J2,
            "BS": BS,
            "H_true": H_true,
            "B0_amp": B0_true,
            "stratified": True,
            "d": d,
            "D": D,
        }
    else:
        U = rng.normal(size=(n, d)) * radius

    Phi = quadratic_features(U)
    BS_n = normal_projector_apply(BS, x0, J)
    X = normalize_rows(x0 + U @ J.T + 0.5 * (Phi @ BS_n.T))
    if noise > 0:
        X = normalize_rows(X + noise * rng.normal(size=X.shape))
    return {
        "family": family,
        "X": X.astype(np.float32),
        "x0": x0,
        "J_true": J,
        "BS": BS_n,
        "H_true": H_true,
        "B0_amp": B0_true,
        "stratified": False,
        "d": d,
        "D": D,
        "U": U,
    }


# -------------------- estimators --------------------


def same_patch_pca(Xn, x0, d, **kw):
    J, ev, diag = pca_tangent(Xn, x0, d)
    return J, {**diag, "estimator": "same_patch_pca"}


def inner_pca(Xn, x0, d, k_tan: int = 256, **kw):
    k_tan = min(k_tan, len(Xn))
    J, ev, diag = pca_tangent(Xn[:k_tan], x0, d)
    return J, {**diag, "estimator": "inner_pca", "k_tan": k_tan}


def kernel_weighted_pca(Xn, x0, d, dists=None, **kw):
    if dists is None:
        dists = np.linalg.norm(Xn - x0, axis=1)
    bw = float(np.quantile(dists, 0.5))
    w = kernel_weights(dists, bw)
    J, ev, diag = pca_tangent(Xn, x0, d, weights=w)
    return J, {**diag, "estimator": "kernel_weighted_pca", "bandwidth": bw}


def bootstrap_grassmann_pca(Xn, x0, d, n_boot=6, seed=0, **kw):
    J, T, ang, diag = bootstrap_grassmann_tangent(Xn, x0, d, n_boot, seed)
    return J, {**diag, "estimator": "bootstrap_grassmann_pca", "T_boot": T}


def multiscale_extrapolated_pca(Xn, x0, d, **kw):
    n = len(Xn)
    ks = sorted(set([max(d + 5, n // 8), max(d + 5, n // 4), max(d + 5, n // 2), n]))
    Ps, invk = [], []
    for k in ks:
        J, _, _ = pca_tangent(Xn[:k], x0, d)
        Ps.append(projector(J).reshape(-1))
        invk.append(1.0 / k)
    P = np.stack(Ps)
    A = np.column_stack([np.ones(len(invk)), invk])
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    P0 = coef[0].reshape(Xn.shape[1], Xn.shape[1])
    P0 = 0.5 * (P0 + P0.T)
    evals, evecs = np.linalg.eigh(P0)
    J = sphere_project_basis(x0, evecs[:, ::-1][:, :d])
    return J, {"estimator": "multiscale_extrapolated_pca", "d_eff": J.shape[1]}


def joint_quadratic_principal_manifold(
    Xn: np.ndarray,
    x0: np.ndarray,
    d: int,
    n_iter: int = 8,
    seed: int = 0,
    **kw,
) -> tuple[np.ndarray, dict]:
    """Alternate latent coords / B^S / Stiefel J under sphere-normal constraint."""
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J, _, diag0 = pca_tangent(Xn, x0, d)
    lam = 1e-2
    for _ in range(n_iter):
        U = (Xn - x0) @ J
        Phi = quadratic_features(U)
        L = x0 + U @ J.T
        resid = Xn - L
        resid_n = normal_projector_apply(resid.T, x0, J).T
        G = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        try:
            BS = np.linalg.solve(G, Phi.T @ resid_n).T
        except np.linalg.LinAlgError:
            BS = np.linalg.lstsq(G, Phi.T @ resid_n, rcond=None)[0].T
        BS = normal_projector_apply(BS, x0, J)
        corr = Xn - 0.5 * (Phi @ BS.T)
        dx = corr - x0
        dx = dx - np.outer(dx @ x0, x0)
        try:
            _, _, vt = np.linalg.svd(dx, full_matrices=False)
            J = sphere_project_basis(x0, vt[:d].T)
        except np.linalg.LinAlgError:
            break
    return J, {"estimator": "joint_quadratic_principal_manifold", "d_eff": J.shape[1], **diag0}


def train_synthetic_sae(
    X_train: np.ndarray,
    feature_dim: int = 256,
    k: int = 32,
    steps: int = 200,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Train TopKSAE on synthetic samples only (repository architecture)."""
    _ensure_sae_path()
    from sae.sae_model import TopKSAE  # type: ignore

    device = device or torch.device("cpu")
    mean = X_train.mean(0)
    scale = X_train.std(0) + 1e-6
    Xt = torch.tensor((X_train - mean) / scale, dtype=torch.float32, device=device)
    model = TopKSAE(Xt.shape[1], feature_dim, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    bs = min(256, len(Xt))
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(Xt), size=bs)
        xb = Xt[idx]
        xh, z = model(xb)
        loss = torch.mean((xh - xb) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return {
        "model": model,
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "device": device,
        "synthetic": True,
    }


def _active_set_jacobian(model, x: torch.Tensor) -> tuple[np.ndarray, int, float]:
    """Analytic DR for TopK at fixed support, then sphere-normalize Jacobian."""
    with torch.no_grad():
        z_pre = torch.relu(model.encoder(x))
        kk = min(model.k, z_pre.shape[-1])
        vals, idx = torch.topk(z_pre, kk, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, idx, vals)
        xh = model.decoder(z)
        recon = float(torch.mean((xh - x) ** 2).item())
        active = idx.squeeze(0)
        W_e = model.encoder.weight[active]  # (k, D)
        W_d = model.decoder.weight[:, active]  # (D, k)
        # pre-activation active where ReLU was positive (topk vals > 0)
        mask = vals.squeeze(0) > 0
        if int(mask.sum()) < 1:
            return np.zeros((x.shape[-1], x.shape[-1])), 0, recon
        W_e = W_e[mask]
        W_d = W_d[:, mask]
        J_lin = (W_d @ W_e).detach().cpu().numpy().astype(np.float64)
        r = xh.squeeze(0).detach().cpu().numpy().astype(np.float64)
        nr = max(np.linalg.norm(r), EPS)
        r_u = r / nr
        # d(r/||r||) = (I - rr^T)/||r|| * J_lin
        J_sphere = ((np.eye(len(r)) - np.outer(r_u, r_u)) / nr) @ J_lin
        return J_sphere, int(mask.sum()), recon


def sae_reconstruction_jacobian(
    Xn: np.ndarray,
    x0: np.ndarray,
    d: int,
    sae_bundle: dict | None = None,
    device: torch.device | None = None,
    **kw,
) -> tuple[np.ndarray | None, dict]:
    """Leading left singular vectors of DR(x0) for SAE reconstruction map."""
    if sae_bundle is None:
        return None, {"estimator": "sae_reconstruction_jacobian", "ok": False, "reason": "no_sae"}
    device = device or sae_bundle.get("device") or torch.device("cpu")
    model = sae_bundle["model"]
    mean = sae_bundle["mean"]
    scale = sae_bundle["scale"]
    # dimension mismatch (real SAE on synthetic smoke D)
    if model.input_dim != len(x0):
        return None, {
            "estimator": "sae_reconstruction_jacobian",
            "ok": False,
            "reason": "dim_mismatch",
            "input_dim": int(model.input_dim),
            "D": int(len(x0)),
        }
    model.eval()
    x = torch.tensor(((x0 - mean) / scale)[None, :], device=device, dtype=torch.float32)
    try:
        Jnp, active, recon = _active_set_jacobian(model, x)
    except Exception as e:  # noqa: BLE001
        return None, {
            "estimator": "sae_reconstruction_jacobian",
            "ok": False,
            "reason": type(e).__name__,
        }
    try:
        U, S, _ = np.linalg.svd(Jnp, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, {"ok": False, "reason": "svd_fail", "estimator": "sae_reconstruction_jacobian"}
    rank = int(np.sum(S > 1e-4 * max(S[0], EPS))) if len(S) else 0
    rho = float(np.median(np.linalg.norm(Xn - x0, axis=1)))
    # stability under perturbation
    rng = np.random.default_rng(0)
    eps = 1e-3 * rho
    x1 = x0 + eps * rng.normal(size=x0.shape)
    x1 = x1 / max(np.linalg.norm(x1), EPS)
    xt1 = torch.tensor(((x1 - mean) / scale)[None, :], device=device, dtype=torch.float32)
    J2, _, _ = _active_set_jacobian(model, xt1)
    try:
        U2, _, _ = np.linalg.svd(J2, full_matrices=False)
        stab = grassmann_dist(projector(U[:, :d]), projector(U2[:, :d]), d)
    except Exception:  # noqa: BLE001
        stab = float("inf")
    ok = bool(recon < 0.5 * max(rho**2, EPS) and active >= d and rank >= d and stab < 0.5)
    if not ok:
        return None, {
            "estimator": "sae_reconstruction_jacobian",
            "ok": False,
            "reason": "sae_gates",
            "recon": recon,
            "active": active,
            "rank": rank,
            "rho": rho,
            "stab": stab,
        }
    J = sphere_project_basis(x0, U[:, :d])
    # also record decoder-code path
    with torch.no_grad():
        z = model.encode(x)
        W_d = model.decoder.weight.detach().cpu().numpy()
        active_idx = (z.squeeze(0) > 0).cpu().numpy()
        if active_idx.sum() >= d:
            Dec_a = W_d[:, active_idx]
            Udec, _, _ = np.linalg.svd(Dec_a, full_matrices=False)
            J_code = sphere_project_basis(x0, Udec[:, :d])
            code_et = grassmann_dist(projector(J), projector(J_code), d)
        else:
            code_et = float("nan")
    return J, {
        "estimator": "sae_reconstruction_jacobian",
        "ok": True,
        "recon": recon,
        "active": active,
        "rank": rank,
        "stab": stab,
        "code_vs_jac_ET": code_et,
        "d_eff": J.shape[1],
    }


ESTIMATORS: dict[str, Callable] = {
    "same_patch_pca": same_patch_pca,
    "inner_pca": inner_pca,
    "kernel_weighted_pca": kernel_weighted_pca,
    "bootstrap_grassmann_pca": bootstrap_grassmann_pca,
    "multiscale_extrapolated_pca": multiscale_extrapolated_pca,
    "joint_quadratic_principal_manifold": joint_quadratic_principal_manifold,
    "sae_reconstruction_jacobian": sae_reconstruction_jacobian,
}


# -------------------- metrics --------------------


def tangent_error(J_hat: np.ndarray, J_true: np.ndarray) -> dict:
    d = min(J_hat.shape[1], J_true.shape[1])
    Ph, Pt = projector(J_hat[:, :d]), projector(J_true[:, :d])
    et = grassmann_dist(Ph, Pt, d)
    ang = principal_angles(J_hat[:, :d], J_true[:, :d])
    return {
        "E_T": et,
        "ang_median": float(np.median(ang)),
        "ang_p90": float(np.quantile(ang, 0.9)),
    }


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: BenchmarkConfig) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "protocol": "tangent_estimator_benchmark_v1",
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    cal = {"noise_empirical": 0.02, "radius_empirical": 0.25}
    try:
        mm = cfg.mm(root)
        X = load_model_X(mm, cfg.model)
        pack = np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz")
        a = int(pack["anchors_local"][0])
        N = pack["neigh"][0, :256]
        dx = X[N] - X[a]
        _, _, vt = np.linalg.svd(dx - dx.mean(0), full_matrices=False)
        proj = dx @ vt[:16].T @ vt[:16]
        resid = dx - proj
        cal["noise_empirical"] = float(np.sqrt(np.mean(resid**2)))
        cal["radius_empirical"] = float(np.median(np.linalg.norm(dx, axis=1)))
    except Exception as e:  # noqa: BLE001
        cal["cal_error"] = type(e).__name__
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"[teb] prepare cal={cal}", flush=True)
    return {**meta, **cal}


def _load_physics_sae(root: Path, cfg: BenchmarkConfig, device: torch.device) -> dict | None:
    sae_dir = resolve_path(root, cfg.sae_dir)
    if not (sae_dir / "model.pt").exists():
        return None
    _ensure_sae_path()
    from sae_affine_basis_mknn_gpu import load_sae  # type: ignore

    return load_sae(sae_dir, device)


def stage_synthetic(root: Path, cfg: BenchmarkConfig) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_truth_index.parquet"
    if _done(path, cfg.force):
        return
    cal = json.loads((out / "calibration.json").read_text())
    noise0 = float(cal.get("noise_empirical", 0.02))
    radius = float(cal.get("radius_empirical", 0.25))
    D = 64 if cfg.smoke else cfg.ambient_D
    n_anch = 8 if cfg.smoke else cfg.n_synth_anchors
    n_train = 512 if cfg.smoke else cfg.n_synth_train
    families_t1 = ["geodesic", "pure_mean", "pure_traceless", "high_rank"]
    families_t2 = ["nonuniform", "boundary", "stratified"]
    rows = []
    syn_dir = out / "synthetic"
    syn_dir.mkdir(exist_ok=True)
    dims = [cfg.primary_d] if cfg.smoke else cfg.dims
    ks = [256, 512] if cfg.smoke else cfg.k_tier1
    noise_levels = [("zero", 0.0), ("empirical", noise0)]
    if not cfg.smoke:
        noise_levels.append(("twice_empirical", 2.0 * noise0))
    for fam in families_t1 + (families_t2 if not cfg.smoke else []):
        for d in dims:
            for noise_name, noise in noise_levels:
                # twice_empirical only for core families (stress test)
                if noise_name == "twice_empirical" and fam not in families_t1:
                    continue
                for k in ks:
                    # tier1 only for k in k_tier1; stresses with large k later
                    if fam in families_t2 and k > 1024:
                        continue
                    seed = _case_seed(cfg.seed, fam, d, k, noise_name)
                    man = sample_manifold(
                        fam, d, D, n_train + n_anch, noise, seed, radius=radius
                    )
                    key = f"{fam}_d{d}_k{k}_{noise_name}"
                    np.savez_compressed(
                        syn_dir / f"{key}.npz",
                        X=man["X"],
                        x0=man["x0"],
                        J_true=man["J_true"],
                        H_true=man.get("H_true", np.zeros(D)),
                        BS=man.get("BS", np.zeros((D, 1))),
                        stratified=np.array([man["stratified"]]),
                    )
                    rows.append(
                        {
                            "key": key,
                            "family": fam,
                            "d": d,
                            "k": k,
                            "noise": noise_name,
                            "noise_sigma": noise,
                            "n": len(man["X"]),
                            "stratified": man["stratified"],
                            "tier": 1 if fam in families_t1 else 2,
                        }
                    )
    if not cfg.smoke:
        for fam in families_t2:
            for d in [cfg.primary_d]:
                for k in cfg.k_tier2:
                    seed = _case_seed(cfg.seed, "t2", fam, d, k)
                    man = sample_manifold(fam, d, D, n_train + n_anch, noise0, seed, radius=radius)
                    key = f"{fam}_d{d}_k{k}_empirical"
                    np.savez_compressed(
                        syn_dir / f"{key}.npz",
                        X=man["X"],
                        x0=man["x0"],
                        J_true=man["J_true"],
                        H_true=man.get("H_true", np.zeros(D)),
                        BS=man.get("BS", np.zeros((D, 1))),
                        stratified=np.array([man["stratified"]]),
                    )
                    rows.append(
                        {
                            "key": key,
                            "family": fam,
                            "d": d,
                            "k": k,
                            "noise": "empirical",
                            "noise_sigma": noise0,
                            "n": len(man["X"]),
                            "stratified": man["stratified"],
                            "tier": 2,
                        }
                    )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[teb] synthetic n_cases={len(rows)} D={D}", flush=True)


def _eval_case_estimators(
    case,
    X: np.ndarray,
    x0: np.ndarray,
    J_true: np.ndarray,
    est_names: list[str],
    cfg: BenchmarkConfig,
    sae_cache: dict,
    device: torch.device,
    t0: float,
) -> list[dict]:
    rows = []
    d = int(case.d)
    k = int(case.k)
    dists = np.linalg.norm(X - x0, axis=1)
    order = np.argsort(dists)
    neigh = order[1 : k + 1] if dists[order[0]] < 1e-6 else order[:k]
    Xn = X[neigh]
    dn = dists[neigh]
    # train synthetic SAE once per case if needed
    if "sae_reconstruction_jacobian" in est_names and case.key not in sae_cache:
        # train on points farther than nearest k (hold out neighbourhood)
        hold = order[k + 1 :]
        if len(hold) < 64:
            hold = order[max(k // 2, 1) :]
        Xtr = X[hold[: max(512, cfg.n_synth_train // 2 if not cfg.smoke else 400)]]
        steps = 60 if cfg.smoke else 200
        fdim = 128 if cfg.smoke else 256
        kk = 16 if cfg.smoke else 32
        sae_cache[case.key] = train_synthetic_sae(
            Xtr, feature_dim=fdim, k=kk, steps=steps, seed=cfg.seed + d, device=device
        )
    for name in est_names:
        if time.time() - t0 > cfg.max_seconds * 0.45:
            break
        fn = ESTIMATORS[name]
        kw = {
            "dists": dn,
            "n_boot": cfg.n_boot,
            "seed": cfg.seed + d * 17,
            "k_tan": min(256, max(d + 8, k // 2)),
            "sae_bundle": sae_cache.get(case.key),
            "device": device,
            "n_iter": 4 if cfg.smoke else 8,
        }
        t_est = time.time()
        try:
            J, diag = fn(Xn, x0, d, **kw)
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "key": case.key,
                    "family": case.family,
                    "d": d,
                    "k": k,
                    "noise": case.noise,
                    "estimator": name,
                    "ok": False,
                    "error": type(e).__name__,
                    "runtime_s": time.time() - t_est,
                }
            )
            continue
        if J is None:
            rows.append(
                {
                    "key": case.key,
                    "family": case.family,
                    "d": d,
                    "k": k,
                    "noise": case.noise,
                    "estimator": name,
                    "ok": False,
                    "runtime_s": time.time() - t_est,
                    **{
                        kk: diag.get(kk)
                        for kk in ("reason", "recon", "active", "rank", "stab")
                    },
                }
            )
            continue
        err = tangent_error(J, J_true)
        # bootstrap projector variance for PCA-like methods
        boot_var = float("nan")
        if name in ("same_patch_pca", "inner_pca", "kernel_weighted_pca"):
            _, T, _, _ = bootstrap_grassmann_tangent(Xn, x0, d, min(cfg.n_boot, 4), cfg.seed)
            boot_var = T
        rows.append(
            {
                "key": case.key,
                "family": case.family,
                "d": d,
                "k": k,
                "noise": case.noise,
                "estimator": name,
                "ok": True,
                "stratified": bool(case.stratified),
                "runtime_s": time.time() - t_est,
                "boot_var": boot_var,
                **err,
                "d_eff": diag.get("d_eff", d),
            }
        )
    return rows


def stage_estimators(root: Path, cfg: BenchmarkConfig) -> None:
    out = cfg.resolved(root)
    path = out / "tangent_error_tables.parquet"
    if _done(path, cfg.force):
        return
    idx = pd.read_parquet(out / "synthetic_truth_index.parquet")
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    rows: list[dict] = []
    sae_cache: dict = {}
    t0 = time.time()
    est_names = list(ESTIMATORS.keys())
    # Tier 1: all estimators on tier-1 / k<=1024
    for _, case in idx.iterrows():
        if int(case.k) > 1024:
            continue
        z = np.load(out / "synthetic" / f"{case.key}.npz")
        rows.extend(
            _eval_case_estimators(
                case, z["X"], z["x0"], z["J_true"], est_names, cfg, sae_cache, device, t0
            )
        )
        print(f"[teb][est] {case.key} done", flush=True)
        if time.time() - t0 > cfg.max_seconds * 0.45:
            break
    # provisional competitive shortlist for tier2
    df_tmp = pd.DataFrame(rows)
    if len(df_tmp) and (df_tmp.ok == True).any():  # noqa: E712
        med = df_tmp[df_tmp.ok == True].groupby("estimator").E_T.median().sort_values()  # noqa: E712
        competitive = list(med.head(4).index)
    else:
        competitive = ["same_patch_pca", "inner_pca", "kernel_weighted_pca", "joint_quadratic_principal_manifold"]
    # Tier 2: competitive only at k=2048 / stress families
    for _, case in idx.iterrows():
        if int(case.k) <= 1024 and case.tier != 2:
            continue
        if int(case.k) > 1024 or case.tier == 2:
            z = np.load(out / "synthetic" / f"{case.key}.npz")
            # avoid double-counting tier2 that already ran at k<=1024 for all ests
            if int(case.k) <= 1024:
                # already evaluated above for all; skip
                continue
            rows.extend(
                _eval_case_estimators(
                    case,
                    z["X"],
                    z["x0"],
                    z["J_true"],
                    competitive,
                    cfg,
                    sae_cache,
                    device,
                    t0,
                )
            )
            print(f"[teb][est-t2] {case.key} {competitive}", flush=True)
        if time.time() - t0 > cfg.max_seconds * 0.5:
            break
    pd.DataFrame(rows).to_parquet(path, index=False)
    (out / "competitive_estimators.json").write_text(json.dumps(competitive, indent=2))
    print(f"[teb] estimators n={len(rows)} competitive={competitive}", flush=True)


def stage_gauss(root: Path, cfg: BenchmarkConfig) -> None:
    """Gauss-map recovery + false-curvature by estimator on synthetic controls."""
    out = cfg.resolved(root)
    path = out / "gauss_map_recovery.parquet"
    if _done(path, cfg.force):
        return
    idx = pd.read_parquet(out / "synthetic_truth_index.parquet")
    # Compact estimator set for FP / recovery gates (skip SAE / bootstrap here).
    est_for_gauss = [
        "same_patch_pca",
        "inner_pca",
        "kernel_weighted_pca",
        "joint_quadratic_principal_manifold",
    ]
    rows = []
    # Compact design: primary_d, k∈{256,1024}, zero+empirical, key families
    for _, case in idx.iterrows():
        if case.family not in ("geodesic", "pure_mean", "high_rank", "stratified"):
            continue
        if case.noise not in ("zero", "empirical"):
            continue
        if int(case.k) not in ((256,) if cfg.smoke else (256, 1024)):
            continue
        if int(case.d) != (cfg.primary_d if not cfg.smoke else cfg.primary_d):
            continue
        z = np.load(out / "synthetic" / f"{case.key}.npz")
        X, J_true = z["X"], z["J_true"]
        d = int(case.d)
        rng = np.random.default_rng(cfg.seed + 3)
        n_anch = 4 if cfg.smoke else 4
        anchors = rng.choice(len(X), size=min(n_anch, len(X)), replace=False)
        t_case = time.time()
        for name in est_for_gauss:
            fn = ESTIMATORS[name]
            for ai in anchors:
                xa = X[ai]
                xa = xa / max(np.linalg.norm(xa), EPS)
                dists = np.linalg.norm(X - xa, axis=1)
                order = np.argsort(dists)
                k_tan = min(128, int(case.k))  # lighter secondary PCA
                Xn = X[order[1 : k_tan + 1]]
                try:
                    J, diag = fn(
                        Xn,
                        xa,
                        d,
                        dists=dists[order[1 : k_tan + 1]],
                        n_iter=3,
                        k_tan=min(64, k_tan),
                        seed=cfg.seed + int(ai),
                    )
                except Exception:  # noqa: BLE001
                    continue
                if J is None:
                    continue
                Px = J @ J.T
                _, _, split_x = split_half_projectors(Xn, xa, d, cfg.seed + ai, pca_tangent)
                sites, splits = [], []
                for rnk in (8, 16, 32, 48):
                    if rnk + 1 >= len(order):
                        continue
                    y = X[order[rnk]]
                    y = y / max(np.linalg.norm(y), EPS)
                    # reuse points near y from the same ambient pool (fast)
                    Yn = X[order[1 : k_tan + 1]]
                    Jy, _, _ = pca_tangent(Yn, y, d)
                    sites.append((y, Jy @ Jy.T))
                    # cheap split proxy: half the same patch
                    h = len(Yn) // 2
                    if h > d + 2:
                        J1, _, _ = pca_tangent(Yn[:h], y, d)
                        J2, _, _ = pca_tangent(Yn[h : 2 * h], y, d)
                        sj = float(np.linalg.norm(J1 @ J1.T - J2 @ J2.T, "fro") ** 2)
                    else:
                        sj = 0.0
                    splits.append(sj)
                if len(sites) < 4:
                    continue
                g = estimate_anchor_gauss_map(xa, Px, sites, split_x, splits, d)
                et = float("nan") if case.stratified else tangent_error(J, J_true)["E_T"]
                energy = g["curvature_energy"]
                rows.append(
                    {
                        "key": case.key,
                        "family": case.family,
                        "d": d,
                        "k": int(case.k),
                        "estimator": name,
                        "anchor": int(ai),
                        "beta": g["beta"],
                        "energy": energy,
                        "label": g["label"],
                        "E_T": et,
                        "false_curvature": bool(
                            case.family == "geodesic"
                            and np.isfinite(energy)
                            and energy > 0.08
                        ),
                        "confident_stratified": bool(
                            case.stratified and np.isfinite(et) and et < 0.15
                        ),
                    }
                )
        print(f"[teb][gauss] {case.key} ({time.time()-t_case:.1f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    if rows:
        gdf = pd.DataFrame(rows)
        fp = (
            gdf[gdf.family == "geodesic"]
            .groupby("estimator")
            .agg(
                fp_rate=("false_curvature", "mean"),
                median_energy=("energy", "median"),
                n=("energy", "count"),
            )
            .reset_index()
        )
        fp.to_parquet(out / "flat_false_positive_tables.parquet", index=False)
    print(f"[teb] gauss recovery n={len(rows)}", flush=True)


def stage_select(root: Path, cfg: BenchmarkConfig) -> dict:
    out = cfg.resolved(root)
    path = out / "estimator_freeze.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    err = pd.read_parquet(out / "tangent_error_tables.parquet")
    gauss = (
        pd.read_parquet(out / "gauss_map_recovery.parquet")
        if (out / "gauss_map_recovery.parquet").exists()
        else pd.DataFrame()
    )
    ok = err[err.ok == True].copy()  # noqa: E712
    reject: set[str] = set()
    # reject excessive false curvature on flat controls
    if len(gauss):
        geo = gauss[gauss.family == "geodesic"]
        for est, g in geo.groupby("estimator"):
            fp = float(g.false_curvature.mean()) if len(g) else 0.0
            med_e = float(np.nanmedian(g.energy)) if len(g) else 0.0
            if fp > 0.5 or med_e > 0.25:
                reject.add(est)
                print(f"[teb] reject {est}: false curvature fp={fp:.2f} medE={med_e:.3f}", flush=True)
        strat = gauss[gauss.family == "stratified"]
        for est, g in strat.groupby("estimator"):
            # Gauss stage marks confident_stratified only if E_T known; use err table
            pass
    strat_err = ok[ok.stratified == True]  # noqa: E712
    for est, g in strat_err.groupby("estimator"):
        if len(g) and float(np.nanmedian(g.E_T)) < 0.15:
            reject.add(est)
            print(f"[teb] reject {est}: fits stratified too confidently", flush=True)
    survivors = [e for e in ok.estimator.unique() if e not in reject]
    if not survivors:
        survivors = list(ok.estimator.unique()) if len(ok) else ["same_patch_pca"]
    core = ok[(ok.stratified == False) & (ok.estimator.isin(survivors))]  # noqa: E712
    # focus selection on empirical noise + k in {256,512,1024}
    core_sel = core[core.noise.isin(["empirical", "zero"])]
    if core_sel.empty:
        core_sel = core
    scores = core_sel.groupby("estimator").E_T.median().sort_values()
    if scores.empty:
        chosen = "same_patch_pca"
        se: dict[str, float] = {}
    else:
        best = scores.index[0]
        se = {
            est: float(np.nanstd(g.E_T) / np.sqrt(max(len(g), 1)))
            for est, g in core_sel.groupby("estimator")
        }
        thresh = float(scores.iloc[0] + se.get(best, 0.0))
        near = [e for e in scores.index if scores.loc[e] <= thresh]
        preference = [
            "same_patch_pca",
            "inner_pca",
            "kernel_weighted_pca",
            "bootstrap_grassmann_pca",
            "multiscale_extrapolated_pca",
            "joint_quadratic_principal_manifold",
            "sae_reconstruction_jacobian",
        ]
        chosen = next((p for p in preference if p in near), best)
    if chosen.startswith("sae"):
        syn_label = "sae_tangent_preferred"
    elif chosen.startswith("joint"):
        syn_label = "joint_quadratic_tangent_preferred"
    elif "kernel" in chosen or "weighted" in chosen:
        syn_label = "weighted_pca_preferred"
    elif chosen in (
        "same_patch_pca",
        "inner_pca",
        "bootstrap_grassmann_pca",
        "multiscale_extrapolated_pca",
    ):
        syn_label = "pca_tangent_validated"
    else:
        syn_label = "no_reliable_tangent_estimator"
    if scores.empty or float(scores.iloc[0]) > 0.45:
        syn_label = "no_reliable_tangent_estimator"
    freeze = {
        "estimator": chosen,
        "synthetic_label": syn_label,
        "median_E_T": float(scores.loc[chosen]) if chosen in scores.index else float("nan"),
        "survivors": survivors,
        "rejected": sorted(reject),
        "scores": {k: float(v) for k, v in scores.items()},
        "config_hash": json.loads((out / "resolved_config.json").read_text())["config_hash"],
        "primary_d": cfg.primary_d,
        "sensitivity_d": 12,
        "k_tan_default": 256,
        "selection_rule": "lexicographic_false_curvature_stratified_median_ET_simplicity",
    }
    path.write_text(json.dumps(freeze, indent=2))
    (out / "estimator_config_hash").write_text(freeze["config_hash"])
    print(f"[teb] FREEZE estimator={chosen} label={syn_label}", flush=True)
    return freeze


def stage_physics(root: Path, cfg: BenchmarkConfig) -> None:
    out = cfg.resolved(root)
    path = out / "physics_geometry.parquet"
    freeze_p = out / "estimator_freeze.json"
    if not freeze_p.exists():
        raise RuntimeError("estimator_freeze.json missing — run select first")
    freeze = json.loads(freeze_p.read_text())
    if freeze.get("synthetic_label") == "no_reliable_tangent_estimator":
        print("[teb] no reliable estimator — skipping physics", flush=True)
        pd.DataFrame([]).to_parquet(path, index=False)
        (out / "geometry_label.json").write_text(
            json.dumps({"label": "real_geometry_unresolved"}, indent=2)
        )
        return
    if _done(path, cfg.force):
        return
    mm = cfg.mm(root)
    X = load_model_X(mm, cfg.model)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    all512 = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(all512.read_text())["sample_ids"] if all512.exists() else anchors_sid.tolist()
    if cfg.smoke:
        use_sids = use_sids[:8]
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    knn_path = mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    pack = dict(np.load(knn_path))
    est_name = freeze["estimator"]
    fn = ESTIMATORS[est_name]
    dims = [int(freeze.get("primary_d", cfg.primary_d))]
    if not cfg.smoke:
        dims.append(int(freeze.get("sensitivity_d", 12)))
    rows = []
    t0 = time.time()
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    sae = _load_physics_sae(root, cfg, device) if "sae" in est_name else None
    # optional prior quadratic energies for agreement
    prior_q = None
    prior_path = mm / "quadratic_energies.parquet"
    if prior_path.exists():
        prior_q = pd.read_parquet(prior_path)

    for d in dims:
        for si, sid in enumerate(use_sids):
            if si % 32 == 0:
                print(f"[teb][phys] d={d} {si}/{len(use_sids)}", flush=True)
            ai = sid_to_ai[int(sid)]
            a_local = int(anchors_local[ai])
            x0 = X[a_local]
            x0 = x0 / max(np.linalg.norm(x0), EPS)
            N = pack["neigh"][ai]
            dists = pack["dists"][ai]
            k_tan = int(freeze.get("k_tan_default", 256))
            Xn = X[N[: max(k_tan, 512)]]
            J, diag = fn(
                Xn,
                x0,
                d,
                dists=dists[: len(Xn)],
                k_tan=k_tan,
                n_boot=cfg.n_boot,
                seed=cfg.seed + ai,
                sae_bundle=sae,
                device=device,
            )
            if J is None:
                continue
            Px = J @ J.T
            _, _, split_x = split_half_projectors(Xn[:k_tan], x0, d, cfg.seed + ai, pca_tangent)
            sites, splits = [], []
            ranks = [r for r in cfg.physics_secondary_ranks if r < len(N)]
            rng = np.random.default_rng(cfg.seed + ai)
            pick = []
            for r0, r1 in zip([0] + ranks[:-1], ranks):
                band = list(range(max(r0, 1), r1))
                if not band:
                    continue
                n_pick = max(1, cfg.n_secondary_per_anchor // max(len(ranks), 1))
                pick.extend(
                    rng.choice(band, size=min(n_pick, len(band)), replace=False).tolist()
                )
            for rnk in pick[: cfg.n_secondary_per_anchor]:
                y_idx = int(N[rnk])
                y = X[y_idx]
                y = y / max(np.linalg.norm(y), EPS)
                local = N[:2048]
                dy = np.linalg.norm(X[local] - y, axis=1)
                ord_y = np.argsort(dy)[:k_tan]
                Yn = X[local[ord_y]]
                Jy, _, _ = pca_tangent(Yn, y, d)
                sites.append((y, Jy @ Jy.T))
                _, _, sj = split_half_projectors(Yn, y, d, cfg.seed + 3 + rnk, pca_tangent)
                splits.append(sj)
            if len(sites) < 6:
                continue
            g = estimate_anchor_gauss_map(x0, Px, sites, split_x, splits, d)
            row = {
                "sample_id": int(sid),
                "estimator": est_name,
                "d": d,
                "T_uncertainty": float(diag.get("T_boot", diag.get("rel_eigengap", np.nan))),
                "beta": g["beta"],
                "curvature_energy": g["curvature_energy"],
                "gauss_label": g["label"],
                "plateau_score": g["score"],
                "median_delta_deb": g["median_delta_deb"],
                "n_pairs": g["n_pairs"],
                "eigengap": diag.get("eigengap", np.nan),
            }
            if prior_q is not None:
                m = prior_q.sample_id == int(sid)
                if "d" in prior_q.columns:
                    m = m & (prior_q.d == d)
                pq = prior_q[m]
                for col in ("K_mean", "K_traceless", "K_total", "d_H2_B0F2"):
                    if col in prior_q.columns:
                        row[f"prior_{col}"] = float(pq.iloc[0][col]) if len(pq) else float("nan")
            rows.append(row)
            if time.time() - t0 > cfg.max_seconds * 0.85:
                print("[teb][phys] time budget", flush=True)
                break
        if time.time() - t0 > cfg.max_seconds * 0.85:
            break
    pd.DataFrame(rows).to_parquet(path, index=False)
    if len(rows):
        labs = pd.DataFrame(rows)
        labs = labs[labs.d == dims[0]]
        if len(labs):
            top = labs.gauss_label.value_counts().index[0]
        else:
            top = "unresolved"
        geo_label = {
            "pointwise_gauss_regime": "gauss_curvature_validated",
            "finite_scale_tangent_heterogeneity": "finite_scale_tangent_heterogeneity",
            "noise_dominated": "real_geometry_unresolved",
            "stratified_or_boundary": "stratified_geometry",
            "unresolved": "real_geometry_unresolved",
        }.get(top, "real_geometry_unresolved")
    else:
        geo_label = "real_geometry_unresolved"
    (out / "geometry_label.json").write_text(json.dumps({"label": geo_label}, indent=2))
    print(f"[teb] physics n={len(rows)} geo_label={geo_label}", flush=True)


def stage_probe(root: Path, cfg: BenchmarkConfig) -> None:
    out = cfg.resolved(root)
    path = out / "probe_joins.parquet"
    freeze_p = out / "estimator_freeze.json"
    assert freeze_p.exists(), "Must freeze estimator before loading probe results"
    # never alter freeze after this point
    freeze = json.loads(freeze_p.read_text())
    (out / "freeze_locked_before_probe.json").write_text(json.dumps(freeze, indent=2))
    if _done(path, cfg.force):
        return
    geo_p = out / "physics_geometry.parquet"
    if not geo_p.exists() or len(pd.read_parquet(geo_p)) == 0:
        pd.DataFrame([]).to_parquet(path, index=False)
        pd.DataFrame([]).to_parquet(out / "probe_control_path.parquet", index=False)
        return
    phys = pd.read_parquet(geo_p)
    # primary d only for probe joins
    phys = phys[phys.d == int(freeze.get("primary_d", cfg.primary_d))]
    mm = cfg.mm(root)
    fields = pd.read_parquet(mm / "local_probe_fields.parquet")
    fields = fields[
        (fields.model == cfg.model)
        & (fields.target == cfg.target)
        & (fields.neighbourhood == "model")
        & (fields.scale_k == 2048)
    ]
    g = phys.merge(fields, on="sample_id", how="inner")
    gp = mm / "graph_dimension_prior.parquet"
    if gp.exists():
        gpp = pd.read_parquet(gp)
        gpp = gpp[(gpp.model == cfg.model) & (gpp.scale_k == 2048)][
            ["sample_id", "graph_support_turnover", "graph_boundary_imbalance"]
        ]
        g = g.merge(gpp, on="sample_id", how="left")
    g.to_parquet(path, index=False)

    rows = []
    y = g.local_r2.to_numpy(float)
    for curv_name in ["curvature_energy", "T_uncertainty", "median_delta_deb", "beta"]:
        if curv_name not in g.columns:
            continue
        x = g[curv_name].to_numpy(float)
        raw = spearman_dict(x, y)
        C_r = g.log_knn_radius.to_numpy(float)[:, None]
        pr = partial_spearman(x, y, C_r)
        path_coefs: dict[str, Any] = {
            "raw": raw["rho"],
            "raw_p": raw["pvalue"],
            "radius_only": pr["rho"],
            "radius_only_p": pr["pvalue"],
        }
        controls = [
            ("label_var", g.local_label_variance.to_numpy(float)),
            ("eval_count", g.local_evaluation_count.to_numpy(float)),
        ]
        if "graph_support_turnover" in g.columns:
            controls.append(("turnover", g.graph_support_turnover.fillna(0).to_numpy(float)))
            controls.append(("boundary", g.graph_boundary_imbalance.fillna(0).to_numpy(float)))
        Z = C_r
        for name, col in controls:
            Z = np.column_stack([Z, col])
            pc = partial_spearman(x, y, Z)
            path_coefs[f"+{name}"] = pc["rho"]
            path_coefs[f"+{name}_p"] = pc["pvalue"]
        path_coefs["full_partial"] = path_coefs[
            [k for k in path_coefs if k.startswith("+") and not k.endswith("_p")][-1]
            if any(k.startswith("+") and not k.endswith("_p") for k in path_coefs)
            else "radius_only"
        ]
        # fix full_partial to last control rho
        ctrl_keys = [k for k in path_coefs if k.startswith("+") and not k.endswith("_p")]
        if ctrl_keys:
            path_coefs["full_partial"] = path_coefs[ctrl_keys[-1]]
        else:
            path_coefs["full_partial"] = path_coefs["radius_only"]
        rows.append(
            {
                "curvature_stat": curv_name,
                "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                **path_coefs,
            }
        )
    pd.DataFrame(rows).to_parquet(out / "probe_control_path.parquet", index=False)
    print(f"[teb] probe joins n={len(g)} path_rows={len(rows)}", flush=True)
    if rows:
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


def stage_analyze(root: Path, cfg: BenchmarkConfig) -> None:
    out = cfg.resolved(root)
    freeze = json.loads((out / "estimator_freeze.json").read_text())
    err = pd.read_parquet(out / "tangent_error_tables.parquet")
    path_df = (
        pd.read_parquet(out / "probe_control_path.parquet")
        if (out / "probe_control_path.parquet").exists()
        else pd.DataFrame()
    )
    geo_lab = (
        json.loads((out / "geometry_label.json").read_text())["label"]
        if (out / "geometry_label.json").exists()
        else "real_geometry_unresolved"
    )
    probe_lab = "no_curvature_probe_association"
    if len(path_df):
        en = path_df[path_df.curvature_stat == "curvature_energy"]
        if len(en):
            raw = float(en.iloc[0].get("raw", 0) or 0)
            full = float(en.iloc[0].get("full_partial", 0) or 0)
            if abs(full) >= 0.08:
                probe_lab = "total_curvature_association"
            elif abs(raw) >= 0.08:
                probe_lab = "geographic_but_not_curvature_specific"

    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    ok = err[err.ok == True]  # noqa: E712
    if len(ok):
        fig1, ax = plt.subplots(figsize=(7, 4))
        for est, g in ok.groupby("estimator"):
            med = g.groupby("k").E_T.median()
            ax.plot(med.index, med.values, marker="o", label=est)
        ax.legend(fontsize=7)
        ax.set_xlabel("k")
        ax.set_ylabel("median E_T")
        ax.set_title("Synthetic tangent error")
        fig1.tight_layout()
        fig1.savefig(fig / "synthetic_ET_by_k.png", dpi=140)
        plt.close(fig1)

    if (out / "gauss_map_recovery.parquet").exists():
        gdf = pd.read_parquet(out / "gauss_map_recovery.parquet")
        if len(gdf):
            fig2, ax = plt.subplots(figsize=(6, 4))
            for fam, g in gdf.groupby("family"):
                ax.scatter(
                    np.clip(g.energy, 0, None),
                    g.E_T if "E_T" in g else np.zeros(len(g)),
                    s=12,
                    alpha=0.5,
                    label=fam,
                )
            ax.legend(fontsize=7)
            ax.set_xlabel("Gauss energy (clamped plot)")
            ax.set_ylabel("E_T")
            fig2.tight_layout()
            fig2.savefig(fig / "gauss_energy_vs_ET.png", dpi=140)
            plt.close(fig2)

    report = f"""# Tangent estimator benchmark + Gauss-map Physics

## Synthetic freeze (no probe labels used)

- Chosen estimator: `{freeze['estimator']}`
- Synthetic label: `{freeze['synthetic_label']}`
- Median E_T: {freeze.get('median_E_T')}
- Config hash: `{freeze.get('config_hash')}`
- Rejected: {freeze.get('rejected')}

## Scores (median E_T)

```json
{json.dumps(freeze.get('scores', {}), indent=2)}
```

## Real geometry

- Label: `{geo_lab}`

## Probe relationship (after freeze only)

- Label: `{probe_lab}`
- Sequential controls (every column printed; not omnibus C0):

```
{path_df.to_string(index=False) if len(path_df) else 'n/a — physics skipped or empty'}
```

## Strongest defensible interpretation

Estimator selected on synthetic ground truth only. Gauss-map curvature on Physics uses the frozen estimator.
Probe associations are post-hoc and must not be read as estimator selection evidence.
"""
    (out / "REPORT.md").write_text(report)
    pd.DataFrame(
        [
            {"stage": "synthetic", "label": freeze["synthetic_label"]},
            {"stage": "geometry", "label": geo_lab},
            {"stage": "probe", "label": probe_lab},
        ]
    ).to_csv(out / "decision_labels.csv", index=False)
    print(
        f"[teb] analyze labels synth={freeze['synthetic_label']} geo={geo_lab} probe={probe_lab}",
        flush=True,
    )


def run(cfg: BenchmarkConfig, root: Path | None = None) -> dict:
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
        "estimators": stage_estimators,
        "gauss": stage_gauss,
        "select": stage_select,
        "physics": stage_physics,
        "probe": stage_probe,
        "analyze": stage_analyze,
    }
    order = [
        "prepare",
        "synthetic",
        "estimators",
        "gauss",
        "select",
        "physics",
        "probe",
        "analyze",
    ]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    deps = {
        "synthetic": ["prepare"],
        "estimators": ["synthetic"],
        "gauss": ["synthetic"],
        "select": ["estimators", "gauss"],
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
        print(f"[teb] stage={s}", flush=True)
        stages[s](root, cfg)
        profile["stages"][f"{s}_s"] = time.time() - t1
        if time.time() - t0 > cfg.max_seconds:
            print("[teb] time budget", flush=True)
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
    print(f"[teb] done in {profile['total_seconds']:.1f}s", flush=True)
    return profile
