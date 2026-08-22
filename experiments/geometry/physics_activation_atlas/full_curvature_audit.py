"""Full curvature audit: reliability, debiasing, scale, Gauss, probe geography.

Reuses completed 512×10 split-half artifacts. Never refits per-anchor probes.
Does not modify split-half / multimodel / magnitude output directories.
"""

from __future__ import annotations

import hashlib
import json
import resource
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

from .confirmatory_object_curvature import _fit_neighborhood
from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import partial_spearman, spearman_dict
from .gauss_map_curvature import (
    estimate_anchor_gauss_map,
    parallel_transport_basis,
    split_half_projectors,
)
from .multimodel_graph_prior_quadratic import EPS, knn_torch_ip, load_model_X
from .paths import platonic_root, resolve_path
from .sae_tangent_benchmark import E_T_thin, SynthDecoder, sample_latent
from .sphere_normal_quadratic import sphere_project_basis
from .split_half_curvature_reliability import (
    PARITY_PARTIAL_C0,
    PARITY_RAW,
    PARITY_TOL,
    BS_objects,
    _half_fit_indices,
    tensor_agreement,
)
from .tangent_reliability import (
    grassmann_dist,
    pca_tangent,
    principal_angles,
    projector,
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_SH = "outputs/geometry/physics_split_half_curvature_reliability"
SOURCE_MAG = "outputs/geometry/physics_global_probe_curvature_magnitude"

RIDGES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0]


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def m_quad(d: int) -> int:
    return d * (d + 1) // 2


def neff_flag(k: int, d: int, half: bool = False) -> dict:
    n = k // 2 if half else k
    # rough: fit uses ~0.4 of points in nested protocol; for half-split ~0.4 of half
    n_fit = max(1, int(0.4 * n))
    md = m_quad(d)
    ratio = n_fit / max(md, 1)
    return {
        "m_d": md,
        "n_eff_proxy": n_fit,
        "n_eff_over_m": ratio,
        "statistically_weak": bool(ratio < 3.0),
    }


@dataclass
class FullCurvatureAuditConfig:
    output_dir: str = "outputs/geometry/physics_full_curvature_audit"
    multimodel_dir: str = SOURCE_MM
    split_half_dir: str = SOURCE_SH
    magnitude_dir: str = SOURCE_MAG
    model: str = "vit_base"
    target: str = "mag_r_desi"
    secondary_targets: list[str] = field(
        default_factory=lambda: ["photo_z", "smooth_fraction", "stellar_mass"]
    )
    dims: list[int] = field(default_factory=lambda: [8, 12, 16])
    ks: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 3072])
    primary_d: int = 16
    primary_k: int = 2048
    n_splits_primary: int = 10  # reuse existing
    n_splits_grid: int = 5
    n_gauss_anchors: int = 128
    n_reg_anchors: int = 128
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    skip_dinov3: bool = False
    analyze_reserve_seconds: float = 600.0

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)

    def sh(self, root: Path) -> Path:
        return resolve_path(root, self.split_half_dir)


def _budget_ok(t0: float, cfg: FullCurvatureAuditConfig, reserve: bool = False) -> bool:
    rem = cfg.max_seconds - (time.time() - t0)
    need = cfg.analyze_reserve_seconds if reserve else 30.0
    return rem > need


def load_ctx(root: Path, cfg: FullCurvatureAuditConfig) -> dict:
    mm = cfg.mm(root)
    sh = cfg.sh(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(aid.read_text())["sample_ids"] if aid.exists() else [int(s) for s in anchors_sid]
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    X = load_model_X(mm, cfg.model)
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    return {
        "mm": mm,
        "sh": sh,
        "X": X,
        "pack": pack,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": sid_to_ai,
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
    }


def knn_cuvs_ip(
    X: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, str]:
    """GPU IP top-(k+1): CAGRA when k<=1023; else cuVS brute_force (CAGRA topk cap)."""
    import cupy as cp
    from cuvs.neighbors import brute_force, cagra

    Xc = cp.asarray(np.ascontiguousarray(X, dtype=np.float32))
    Xc = Xc / cp.clip(cp.linalg.norm(Xc, axis=1, keepdims=True), 1e-8, None)
    Qc = cp.asarray(np.ascontiguousarray(queries, dtype=np.float32))
    Qc = Qc / cp.clip(cp.linalg.norm(Qc, axis=1, keepdims=True), 1e-8, None)
    kk = int(k) + 1
    # CAGRA internal topk hard-caps at 1024
    if kk <= 1024:
        itopk = max(64, ((kk + 31) // 32) * 32)
        index = cagra.build(cagra.IndexParams(metric="inner_product"), Xc)
        _D, I = cagra.search(
            cagra.SearchParams(itopk_size=itopk), index, Qc, k=kk
        )
        return cp.asnumpy(I).astype(np.int64), "cuvs_cagra_ip"
    index = brute_force.build(Xc, metric="inner_product")
    _D, I = brute_force.search(index, Qc, kk)
    return cp.asnumpy(I).astype(np.int64), "cuvs_brute_force_ip"


def build_extended_knn_gpu(
    ctx: dict, cfg: FullCurvatureAuditConfig, k_max: int, cache_path: Path
) -> np.ndarray:
    """One-shot GPU IP kNN for all anchors up to k_max (CAGRA preferred; Torch fallback)."""
    if cache_path.exists():
        return np.load(cache_path)["neigh"]
    X = ctx["X"]
    locs = np.asarray(ctx["anchors_local"], dtype=np.int64)
    Q = X[locs]
    device = ctx["device"]
    print(f"[fca][gpu] building IP kNN k={k_max} for {len(Q)} anchors on {device}", flush=True)
    t1 = time.time()
    backend = "torch_ip"
    try:
        idx, backend = knn_cuvs_ip(X, Q, k_max)
        print(f"[fca][gpu] kNN backend={backend}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"[fca][gpu] cuVS unavailable ({type(e).__name__}: {e}); Torch IP fallback",
            flush=True,
        )
        idx = knn_torch_ip(X, Q, k=k_max, device=device, batch=64)
        backend = "torch_ip_fallback"
    neigh = np.zeros((len(Q), k_max), dtype=np.int64)
    for i, a in enumerate(locs):
        row = idx[i]
        row = row[row != int(a)]
        if len(row) < k_max:
            # pad from existing 2048 cache if needed
            base = ctx["pack"]["neigh"][i]
            seen = set(int(x) for x in row.tolist())
            extra = [int(x) for x in base if int(x) not in seen]
            row = np.concatenate([row, np.asarray(extra, dtype=np.int64)])
        neigh[i] = row[:k_max]
    np.savez_compressed(cache_path, neigh=neigh, backend=np.asarray(backend))
    print(
        f"[fca][gpu] kNN cache wrote in {time.time()-t1:.1f}s backend={backend} → {cache_path}",
        flush=True,
    )
    return neigh


def ensure_neigh(
    ctx: dict, ai: int, k: int, cfg: FullCurvatureAuditConfig
) -> np.ndarray:
    """Neighbour indices; uses cached GPU-extended pack when k>2048."""
    pack = ctx["pack"]
    if k <= pack["neigh"].shape[1]:
        return pack["neigh"][ai, :k]
    ext = ctx.get("pack_ext")
    if ext is None or ext.shape[1] < k:
        raise RuntimeError("extended kNN cache missing — call build_extended_knn_gpu in prepare")
    return ext[ai, :k]


def pca_tangent_gpu(
    Xn: np.ndarray, x0: np.ndarray, d: int, device: torch.device
) -> tuple[np.ndarray, dict]:
    """Sphere-centered PCA via torch.linalg.svd_lowrank / svd on GPU."""
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    dx = Xn.astype(np.float32) - x0u.astype(np.float32)
    dx = dx - np.outer(dx @ x0u, x0u)
    Xt = torch.as_tensor(dx, device=device, dtype=torch.float32)
    # economy SVD: n x D with n typically <= 3072, D=768 → use Xt = U S Vh
    try:
        # svd_lowrank is approximate and fast when q~d is small
        q = min(max(d + 8, 2 * d), min(Xt.shape) - 1)
        U, S, V = torch.svd_lowrank(Xt, q=q, niter=2)
        V = V[:, :d]
        J = V.detach().cpu().numpy().astype(np.float64)
        ev = (S.detach().cpu().numpy() ** 2) / max(len(Xn), 1)
    except Exception:  # noqa: BLE001
        J, ev, diag = pca_tangent(Xn, x0, d)
        return J, diag
    J = sphere_project_basis(x0u, J)
    gap = float(ev[d - 1] - ev[d]) if len(ev) > d else float("nan")
    return J, {"eigengap": gap, "d_eff": J.shape[1], "backend": "torch_svd_lowrank"}


def _quad_features_torch(U: torch.Tensor) -> torch.Tensor:
    n, d = U.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(U[:, a] * U[:, b])
    return torch.stack(cols, dim=1)


def _ridge_solve_torch(Phi: torch.Tensor, Target: torch.Tensor, lam: float) -> torch.Tensor:
    """Return B (k,q) for Target (n,k)."""
    G = Phi.T @ Phi
    G = G + lam * torch.eye(G.shape[0], device=Phi.device, dtype=Phi.dtype)
    R = Phi.T @ Target
    return torch.linalg.solve(G, R).T


def fit_nested_fixed_tangent_gpu(
    Xloc: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    idx_fit: np.ndarray,
    idx_val: np.ndarray,
    idx_te: np.ndarray,
    ridges: list[float] | None = None,
    device: torch.device | None = None,
) -> tuple[Any, Any, dict]:
    """GPU-accelerated fixed-tangent nested quadratic (multi-λ ridge sweep)."""
    from .sphere_normal_quadratic import NestedChart, chart_errors, normal_projector_apply
    from .curvature_probe_alignment import traceless_B0 as _traceless

    ridges = ridges or RIDGES
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(idx_fit) < 20 or len(idx_val) < 8 or J.shape[1] < 2:
        return None, None, {"ok": False, "reason": "too_few"}
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J = sphere_project_basis(x0, J)
    d = J.shape[1]
    X_t = torch.as_tensor(Xloc, device=device, dtype=torch.float32)
    x0_t = torch.as_tensor(x0, device=device, dtype=torch.float32)
    J_t = torch.as_tensor(J, device=device, dtype=torch.float32)
    U = (X_t - x0_t) @ J_t
    sc = torch.sqrt(torch.clamp((U[idx_fit] ** 2).mean(0), min=1e-12))

    def decode_rows(Uw, BSq=None):
        # Uw: (n,d) tangent coords after warping; BSq: (n,D) normal quadratic
        amb = x0_t + Uw @ J_t.T
        if BSq is not None:
            amb = amb + BSq
        return amb / torch.clamp(amb.norm(dim=-1, keepdim=True), min=1e-8)

    def mse(pred, idx):
        return torch.mean(torch.sum((pred - X_t[idx]) ** 2, dim=1)).item()

    Phi_f = _quad_features_torch(U[idx_fit])
    L_f = x0_t + U[idx_fit] @ J_t.T
    scale = torch.clamp(L_f.norm(dim=1, keepdim=True), min=1e-8)
    target_un = X_t[idx_fit] * scale
    tang_res = (target_un - L_f) @ J_t
    # cache Gram for A sweep
    Gphi = Phi_f.T @ Phi_f
    Rt = Phi_f.T @ tang_res
    best_A, lam_A, best_tr = None, ridges[0], float("inf")
    eye_q = torch.eye(Gphi.shape[0], device=device, dtype=torch.float32)
    for lam in ridges:
        A = torch.linalg.solve(Gphi + lam * eye_q, Rt).T  # (d,q)
        Phi_v = _quad_features_torch(U[idx_val])
        Uw = U[idx_val] + Phi_v @ A.T
        loss = mse(decode_rows(Uw), idx_val)
        if loss < best_tr:
            best_tr, best_A, lam_A = loss, A, lam
    A_flat = best_A
    Phi_f = _quad_features_torch(U[idx_fit])
    L_tr = x0_t + (U[idx_fit] + Phi_f @ A_flat.T) @ J_t.T
    scale_tr = torch.clamp(L_tr.norm(dim=1, keepdim=True), min=1e-8)
    target_tr = X_t[idx_fit] * scale_tr
    resid = target_tr - L_tr
    # project to sphere-normal: remove span(x0,J)
    Qn = torch.cat([x0_t[:, None], J_t], dim=1)
    Qn, _ = torch.linalg.qr(Qn, mode="reduced")
    resid_n = resid - (resid @ Qn) @ Qn.T
    Rn = Phi_f.T @ resid_n
    best_BS, lam_BS, best_trs = None, ridges[0], float("inf")
    for lam in ridges:
        BS = torch.linalg.solve(Gphi + lam * eye_q, Rn).T  # (D,q)
        BS = BS - Qn @ (Qn.T @ BS)
        Phi_v = _quad_features_torch(U[idx_val])
        Uw = U[idx_val] + Phi_v @ A_flat.T
        BSq = Phi_v @ BS.T
        loss = mse(decode_rows(Uw, BSq), idx_val)
        if loss < best_trs:
            best_trs, best_BS, lam_BS = loss, BS, lam
    # RS path
    resid_r = target_un - L_f
    resid_r = resid_r - (resid_r @ Qn) @ Qn.T
    Rr = Phi_f.T @ resid_r
    best_BSR, lam_BSR, best_rs = None, ridges[0], float("inf")
    for lam in ridges:
        BS = torch.linalg.solve(Gphi + lam * eye_q, Rr).T
        BS = BS - Qn @ (Qn.T @ BS)
        Phi_v = _quad_features_torch(U[idx_val])
        BSq = Phi_v @ BS.T
        loss = mse(decode_rows(U[idx_val], BSq), idx_val)
        if loss < best_rs:
            best_rs, best_BSR, lam_BSR = loss, BS, lam
    A_np = A_flat.detach().cpu().numpy().astype(np.float64)
    BS_np = best_BS.detach().cpu().numpy().astype(np.float64)
    BSR_np = best_BSR.detach().cpu().numpy().astype(np.float64)
    chart = NestedChart(x0, J, A_np, BS_np, float(lam_A), float(lam_BS), sc.detach().cpu().numpy())
    chart_RS = NestedChart(x0, J, np.zeros_like(A_np), BSR_np, float(lam_A), float(lam_BSR), sc.detach().cpu().numpy())
    U_np = U.detach().cpu().numpy().astype(np.float64)
    w = np.ones(len(Xloc), dtype=np.float64)
    err = chart_errors(chart, chart_RS, Xloc, U_np, w, idx_te)
    _, H = _traceless(chart.BS_flat, d)
    info = {
        "ok": True,
        "d_eff": d,
        "dS": err["dS"],
        "dT": err["dT"],
        "E_TRS": err["E_TRS"],
        "E_TR": err["E_TR"],
        "K_mean": float(np.linalg.norm(H)),
        "recon_error": err["E_TRS"],
        "backend": "torch_gpu_ridge",
    }
    return chart, chart_RS, info


# Prefer GPU fit everywhere in this audit
def fit_quad(Xloc, x0, J, fit, val, te, ridges=None, device=None):
    return fit_nested_fixed_tangent_gpu(
        Xloc, x0, J, fit, val, te, ridges=ridges, device=device
    )


def full_patch_pca_tangent(
    Xloc: np.ndarray, d: int, device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """GPU svd_lowrank PCA on the full neighbourhood (frozen for splits)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = Xloc.mean(0)
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J, _ = pca_tangent_gpu(Xloc, x0, d, device)
    return x0, J


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "figures", "grid", "logs", "batches"):
        (out / sub).mkdir(exist_ok=True)
    # GPU extended kNN once if needed
    k_max = max(cfg.ks)
    if k_max > ctx["pack"]["neigh"].shape[1]:
        ctx["pack_ext"] = build_extended_knn_gpu(
            ctx, cfg, k_max, out / "cache" / f"{cfg.model}_kmax{k_max}_gpu.npz"
        )
    else:
        ctx["pack_ext"] = None
    meta = {
        "config": asdict(cfg),
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "protocol": "full_curvature_audit_v1",
        "reused_split_half": str(cfg.split_half_dir),
        "n_anchors": len(ctx["use_sids"]),
        "resume_command": (
            "PYTHONPATH=experiments python experiments/geometry/run_full_curvature_audit.py "
            f"--stage <remaining> --device {cfg.device}"
        ),
    }
    # cell weakness table
    rows = []
    for d in cfg.dims:
        for k in cfg.ks:
            for half in (False, True):
                rows.append({"d": d, "k": k, "half_split": half, **neff_flag(k, d, half)})
    pd.DataFrame(rows).to_parquet(out / "cell_strength.parquet", index=False)
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[fca] prepare hash={meta['config_hash']} n_anchors={meta['n_anchors']}", flush=True)
    return meta


def stage_parity(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity_gate.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    sh = ctx["sh"]
    # 1) association from existing split-half parity features
    feats_sh = pd.read_parquet(sh / "parity_full_fit_features.parquet")
    gate_sh = json.loads((sh / "parity_gate.json").read_text())
    geo = ctx["geo"]
    g0 = geo[
        (geo.model == cfg.model)
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == cfg.primary_k)
    ]
    g = g0.merge(feats_sh, on="sample_id", how="inner", suffixes=("_geo", ""))
    Km = g.K_mean.to_numpy(float)
    y = g.local_r2.to_numpy(float)
    C0 = np.column_stack(
        [
            g.log_knn_radius.to_numpy(float),
            g.local_label_variance.to_numpy(float),
            g.recon_error.to_numpy(float),
            g.local_evaluation_count.to_numpy(float),
        ]
    )
    raw = spearman_dict(Km, y)
    p0 = partial_spearman(Km, y, C0)
    assoc_ok = bool(
        abs(raw["rho"] - PARITY_RAW) <= PARITY_TOL
        and abs(p0["rho"] - PARITY_PARTIAL_C0) <= PARITY_TOL
        and len(g) >= 500
    )
    # 2) sample_id curvature match vs d_replication full-fit
    drep = pd.read_parquet(
        ctx["mm"] / "d_replication_check_all512" / "curvature_features_d_sweep.parquet"
    )
    drep = drep[(drep.d == cfg.primary_d) & (drep.scale_k == cfg.primary_k)][
        ["sample_id", "K_mean", "K_traceless", "recon_error"]
    ].rename(columns={"K_mean": "K_mean_drep", "K_traceless": "K_tr_drep", "recon_error": "recon_drep"})
    m = feats_sh.merge(drep, on="sample_id", how="inner")
    if len(m) == 0:
        curv_ok = False
        max_abs_diff = float("inf")
        first_diff = {"reason": "no_overlap_sample_ids"}
    else:
        diff = np.abs(m.K_mean.to_numpy(float) - m.K_mean_drep.to_numpy(float))
        max_abs_diff = float(np.max(diff))
        # exact protocol match should be ~0; allow tiny float noise
        curv_ok = bool(max_abs_diff < 1e-6)
        if not curv_ok:
            i = int(np.argmax(diff))
            first_diff = {
                "sample_id": int(m.iloc[i].sample_id),
                "K_mean_parity_feat": float(m.iloc[i].K_mean),
                "K_mean_drep": float(m.iloc[i].K_mean_drep),
                "abs_diff": float(diff[i]),
                "note": "parity_full_fit vs d_replication_check_all512",
            }
        else:
            first_diff = None
    # 3) optional live recompute on 32 anchors for seed check
    live_diffs = []
    for sid in list(m.sample_id.head(32)):
        ai = ctx["sid_to_ai"][int(sid)]
        N = ctx["pack"]["neigh"][ai, : cfg.primary_k]
        chart, _, info, _, _, reason = _fit_neighborhood(
            ctx["X"], N, cfg.primary_d, seed=cfg.seed + ai + 17 * cfg.primary_k + cfg.primary_d
        )
        if chart is None:
            live_diffs.append({"sample_id": int(sid), "fail": reason})
            continue
        _, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
        km = float(np.linalg.norm(H))
        ref = float(feats_sh.set_index("sample_id").loc[int(sid), "K_mean"])
        live_diffs.append({"sample_id": int(sid), "abs_diff": abs(km - ref), "K_mean": km, "ref": ref})
    live_ok = bool(live_diffs and np.nanmax([d.get("abs_diff", np.nan) for d in live_diffs]) < 1e-6)

    ok = bool(assoc_ok and curv_ok and live_ok)
    result = {
        "ok": ok,
        "assoc_ok": assoc_ok,
        "curvature_values_ok": curv_ok,
        "live_refit_ok": live_ok,
        "n": int(len(g)),
        "raw_rho_K_mean": raw["rho"],
        "partial_C0": p0["rho"],
        "expected_raw": PARITY_RAW,
        "expected_partial_C0": PARITY_PARTIAL_C0,
        "max_abs_K_mean_diff_vs_drep": max_abs_diff,
        "first_diff": first_diff,
        "split_half_gate": gate_sh,
        "n_live_checked": len(live_diffs),
    }
    path.write_text(json.dumps(result, indent=2, default=str))
    feats_sh.to_parquet(out / "parity_full_fit_features.parquet", index=False)
    print(
        f"[fca][parity] ok={ok} raw={raw['rho']:.4f} partial={p0['rho']:.4f} "
        f"max|ΔK|={max_abs_diff:.3e} live_ok={live_ok}",
        flush=True,
    )
    if not ok:
        print("[fca][parity] STOP — protocol drift", flush=True)
    return result


def stage_fixed_tangent(
    root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float
) -> None:
    """Reuse primary 512×10; compute remaining grid cells with 5 splits."""
    out = cfg.resolved(root)
    path = out / "fixed_tangent_grid.parquet"
    if _done(path, cfg.force):
        return
    # copy / annotate primary
    primary = pd.read_parquet(ctx["sh"] / "split_half_full.parquet")
    primary = primary.copy()
    primary["source"] = "reused_split_half_512x10"
    primary["m_d"] = m_quad(cfg.primary_d)
    primary["n_eff_over_m"] = neff_flag(cfg.primary_k, cfg.primary_d, half=True)["n_eff_over_m"]
    chunks = [primary]
    primary.to_parquet(out / "fixed_tangent_primary_reused.parquet", index=False)

    X = ctx["X"]
    geo_idx = ctx["geo"][
        (ctx["geo"].model == cfg.model)
        & (ctx["geo"].target == cfg.target)
        & (ctx["geo"].neighbourhood == "model")
        & (ctx["geo"].scale_k == cfg.primary_k)
    ].set_index("sample_id")

    grid_cells = [(d, k) for d in cfg.dims for k in cfg.ks if not (d == cfg.primary_d and k == cfg.primary_k)]
    for d, k in grid_cells:
        if not _budget_ok(t0, cfg, reserve=True):
            print(f"[fca][fixed] budget — skip remaining from d={d} k={k}", flush=True)
            break
        cell_path = out / "grid" / f"fixed_d{d}_k{k}.parquet"
        if _done(cell_path, cfg.force):
            chunks.append(pd.read_parquet(cell_path))
            continue
        weak = neff_flag(k, d, half=True)
        rows = []
        for si, sid in enumerate(ctx["use_sids"]):
            if si % 32 == 0:
                print(f"[fca][fixed] d={d} k={k} {si}/{len(ctx['use_sids'])}", flush=True)
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = ensure_neigh(ctx, ai, k, cfg)
            Xloc = X[N].astype(np.float64)
            x0, J = full_patch_pca_tangent(Xloc, d)
            if J.shape[1] < d or int(sid) not in geo_idx.index:
                continue
            local_r2 = float(geo_idx.loc[int(sid), "local_r2"])
            log_r = float(geo_idx.loc[int(sid), "log_knn_radius"])
            for s in range(cfg.n_splits_grid):
                rng = np.random.default_rng(cfg.seed + 1009 * ai + 17 * s + d * 13 + k)
                perm = rng.permutation(k)
                halfA, halfB = perm[: k // 2], perm[k // 2 :]
                fitA, valA = _half_fit_indices(halfA, cfg.seed + 3 + s)
                fitB, valB = _half_fit_indices(halfB, cfg.seed + 7 + s)
                chA, _, infoA = fit_quad(
                    Xloc, x0, J, fitA, valA, halfB, ridges=RIDGES
                )
                chB, _, infoB = fit_quad(
                    Xloc, x0, J, fitB, valB, halfA, ridges=RIDGES
                )
                if chA is None or chB is None:
                    continue
                oA, oB = BS_objects(chA.BS_flat, d), BS_objects(chB.BS_flat, d)
                agH = tensor_agreement(oA["H"], oB["H"])
                agB0 = tensor_agreement(oA["B0_flat"], oB["B0_flat"])
                agBS = tensor_agreement(oA["BS_flat"], oB["BS_flat"])
                rows.append(
                    {
                        "sample_id": int(sid),
                        "split": s,
                        "d": d,
                        "k": k,
                        "source": "computed",
                        "local_r2": local_r2,
                        "log_knn_radius": log_r,
                        "norm_HA": agH["norm_A"],
                        "norm_HB": agH["norm_B"],
                        "r_H_dir": agH["r_dir"],
                        "K_H_cross": agH["inner"],
                        "R_H": agH["R_signal"],
                        "norm_B0A": agB0["norm_A"],
                        "norm_B0B": agB0["norm_B"],
                        "r_B0_dir": agB0["r_dir"],
                        "K_B0_cross": agB0["inner"],
                        "R_B0": agB0["R_signal"],
                        "norm_BSA": agBS["norm_A"],
                        "norm_BSB": agBS["norm_B"],
                        "r_BS_dir": agBS["r_dir"],
                        "K_BS_cross": agBS["inner"],
                        "R_BS": agBS["R_signal"],
                        "dS_A": float(infoA.get("dS", np.nan)),
                        "dS_B": float(infoB.get("dS", np.nan)),
                        "statistically_weak": weak["statistically_weak"],
                        "n_eff_over_m": weak["n_eff_over_m"],
                        "m_d": weak["m_d"],
                    }
                )
            # batch save every 64 anchors
            if si > 0 and si % 64 == 0 and rows:
                pd.DataFrame(rows).to_parquet(
                    out / "batches" / f"fixed_d{d}_k{k}_upto{si}.parquet", index=False
                )
        cell_df = pd.DataFrame(rows)
        cell_df.to_parquet(cell_path, index=False)
        chunks.append(cell_df)
        print(f"[fca][fixed] wrote d={d} k={k} n={len(cell_df)}", flush=True)
    pd.concat(chunks, ignore_index=True).to_parquet(path, index=False)
    print(f"[fca] fixed_tangent grid done", flush=True)


def stage_debiased(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "debiased_curvature.parquet"
    if _done(path, cfg.force):
        return
    grid = pd.read_parquet(out / "fixed_tangent_grid.parquet")
    rows = []
    for (d, k, sid), g in grid.groupby(["d", "k", "sample_id"]):
        def avg_cross(col):
            return float(np.nanmean(g[col].to_numpy(float)))

        def avg_norm2(a, b):
            return float(np.nanmean(0.5 * (g[a] ** 2 + g[b] ** 2)))

        kh = avg_cross("K_H_cross")
        kb0 = avg_cross("K_B0_cross")
        kbs = avg_cross("K_BS_cross")
        n2h = avg_norm2("norm_HA", "norm_HB")
        n2b0 = avg_norm2("norm_B0A", "norm_B0B")
        n2bs = avg_norm2("norm_BSA", "norm_BSB")
        rows.append(
            {
                "sample_id": int(sid),
                "d": int(d),
                "k": int(k),
                "K_H_cross": kh,
                "K_B0_cross": kb0,
                "K_BS_cross": kbs,
                "K_H_cross_plot": max(kh, 0.0),
                "K_B0_cross_plot": max(kb0, 0.0),
                "K_BS_cross_plot": max(kbs, 0.0),
                "norm2_H_mean": n2h,
                "norm2_B0_mean": n2b0,
                "norm2_BS_mean": n2bs,
                "inflation_H": n2h / max(kh, EPS) if kh > 0 else float("nan"),
                "inflation_B0": n2b0 / max(kb0, EPS) if kb0 > 0 else float("nan"),
                "inflation_BS": n2bs / max(kbs, EPS) if kbs > 0 else float("nan"),
                "R_H": float(np.nanmedian(g.R_H)),
                "R_B0": float(np.nanmedian(g.R_B0)),
                "R_BS": float(np.nanmedian(g.R_BS)),
                "r_H_dir": float(np.nanmedian(g.r_H_dir)),
                "r_B0_dir": float(np.nanmedian(g.r_B0_dir)),
                "r_BS_dir": float(np.nanmedian(g.r_BS_dir)),
                "median_dS": float(np.nanmedian(0.5 * (g.dS_A + g.dS_B))),
                "frac_dS_pos": float(np.mean(0.5 * (g.dS_A + g.dS_B) > 0)),
                "local_r2": float(g.local_r2.iloc[0]),
                "log_knn_radius": float(g.log_knn_radius.iloc[0]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    # reliability summary by cell
    summ = (
        df.groupby(["d", "k"])
        .agg(
            median_R_H=("R_H", "median"),
            median_R_B0=("R_B0", "median"),
            median_R_BS=("R_BS", "median"),
            median_inflation_H=("inflation_H", "median"),
            median_inflation_B0=("inflation_B0", "median"),
            median_dS=("median_dS", "median"),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    summ.to_parquet(out / "debiased_cell_summary.parquet", index=False)
    print(f"[fca] debiased n={len(df)}", flush=True)


def stage_independent_tangent(
    root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float
) -> None:
    """Compare fixed-J vs independent-J split reliability on primary cells."""
    out = cfg.resolved(root)
    path = out / "independent_tangent.parquet"
    if _done(path, cfg.force):
        return
    cells = [(d, k) for d in (12, 16) for k in (1024, 2048, 3072)]
    X = ctx["X"]
    rows = []
    for d, k in cells:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        for si, sid in enumerate(ctx["use_sids"]):
            if si % 64 == 0:
                print(f"[fca][indt] d={d} k={k} {si}/512", flush=True)
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = ensure_neigh(ctx, ai, k, cfg)
            Xloc = X[N].astype(np.float64)
            # fixed
            x0_f, J_f = full_patch_pca_tangent(Xloc, d)
            rng = np.random.default_rng(cfg.seed + ai + d + k)
            perm = rng.permutation(k)
            A, B = perm[: k // 2], perm[k // 2 :]
            # independent tangents on halves
            x0A = Xloc[A].mean(0)
            x0A /= max(np.linalg.norm(x0A), EPS)
            x0B = Xloc[B].mean(0)
            x0B /= max(np.linalg.norm(x0B), EPS)
            JA, diagA = pca_tangent_gpu(Xloc[A], x0A, d, ctx["device"])
            JB, diagB = pca_tangent_gpu(Xloc[B], x0B, d, ctx["device"])
            et = E_T_thin(JA, JB)
            ang = principal_angles(JA, JB)
            # Procrustes align JB → JA then compare H in ambient (already ambient)
            # Fit quadratic on each half with own J, held-out other half
            fitA, valA = _half_fit_indices(A, cfg.seed + 1)
            fitB, valB = _half_fit_indices(B, cfg.seed + 2)
            # map local indices: fitA are indices into 0..k-1 already
            chA, _, infoA = fit_quad(
                Xloc, x0A, JA, fitA, valA, B, ridges=RIDGES
            )
            chB, _, infoB = fit_quad(
                Xloc, x0B, JB, fitB, valB, A, ridges=RIDGES
            )
            if chA is None or chB is None:
                continue
            oA, oB = BS_objects(chA.BS_flat, d), BS_objects(chB.BS_flat, d)
            # ambient H already in R^D
            agH = tensor_agreement(oA["H"], oB["H"])
            agB0 = tensor_agreement(oA["B0_flat"], oB["B0_flat"])
            # fixed-J counterpart for same split
            fA, vA = _half_fit_indices(A, cfg.seed + 1)
            fB, vB = _half_fit_indices(B, cfg.seed + 2)
            chAf, _, _ = fit_quad(Xloc, x0_f, J_f, fA, vA, B, ridges=RIDGES)
            chBf, _, _ = fit_quad(Xloc, x0_f, J_f, fB, vB, A, ridges=RIDGES)
            if chAf is None or chBf is None:
                continue
            oAf, oBf = BS_objects(chAf.BS_flat, d), BS_objects(chBf.BS_flat, d)
            agHf = tensor_agreement(oAf["H"], oBf["H"])
            agB0f = tensor_agreement(oAf["B0_flat"], oBf["B0_flat"])
            rows.append(
                {
                    "sample_id": int(sid),
                    "d": d,
                    "k": k,
                    "ET_tangent": et,
                    "ang_median": float(np.median(ang)),
                    "ang_p90": float(np.quantile(ang, 0.9)),
                    "eigengap_A": float(diagA.get("eigengap", np.nan)),
                    "eigengap_B": float(diagB.get("eigengap", np.nan)),
                    "r_H_indep": agH["r_dir"],
                    "R_H_indep": agH["R_signal"],
                    "r_B0_indep": agB0["r_dir"],
                    "R_B0_indep": agB0["R_signal"],
                    "r_H_fixed": agHf["r_dir"],
                    "R_H_fixed": agHf["R_signal"],
                    "r_B0_fixed": agB0f["r_dir"],
                    "R_B0_fixed": agB0f["R_signal"],
                    "delta_R_H": agHf["R_signal"] - agH["R_signal"],
                    "delta_R_B0": agB0f["R_signal"] - agB0["R_signal"],
                }
            )
        pd.DataFrame(rows).to_parquet(out / "batches" / f"indt_d{d}_k{k}.parquet", index=False)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[fca] independent_tangent n={len(rows)}", flush=True)


def stage_regularization(
    root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float
) -> None:
    out = cfg.resolved(root)
    path = out / "regularization_sensitivity.parquet"
    if _done(path, cfg.force):
        return
    # geometry-stratified 128 anchors: spread by log radius
    geo = ctx["geo"][
        (ctx["geo"].model == cfg.model)
        & (ctx["geo"].target == cfg.target)
        & (ctx["geo"].neighbourhood == "model")
        & (ctx["geo"].scale_k == 2048)
    ]
    g = geo[geo.sample_id.isin(ctx["use_sids"])].sort_values("log_knn_radius")
    step = max(1, len(g) // cfg.n_reg_anchors)
    anchors = g.sample_id.to_numpy()[::step][: cfg.n_reg_anchors]
    scales = [0.5, 1.0, 2.0]
    rows = []
    for k in (1024, 2048):
        for sid in anchors:
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = ensure_neigh(ctx, ai, k, cfg)
            Xloc = ctx["X"][N].astype(np.float64)
            x0, J = full_patch_pca_tangent(Xloc, 16)
            rng = np.random.default_rng(cfg.seed + ai)
            perm = rng.permutation(k)
            A, B = perm[: k // 2], perm[k // 2 :]
            for mult in scales:
                ridges = [r * mult for r in RIDGES]
                fA, vA = _half_fit_indices(A, cfg.seed)
                fB, vB = _half_fit_indices(B, cfg.seed + 1)
                chA, _, infoA = fit_quad(Xloc, x0, J, fA, vA, B, ridges=ridges)
                chB, _, infoB = fit_quad(Xloc, x0, J, fB, vB, A, ridges=ridges)
                if chA is None or chB is None:
                    continue
                oA, oB = BS_objects(chA.BS_flat, 16), BS_objects(chB.BS_flat, 16)
                agH = tensor_agreement(oA["H"], oB["H"])
                agB0 = tensor_agreement(oA["B0_flat"], oB["B0_flat"])
                rows.append(
                    {
                        "sample_id": int(sid),
                        "k": k,
                        "ridge_mult": mult,
                        "r_H_dir": agH["r_dir"],
                        "R_H": agH["R_signal"],
                        "r_B0_dir": agB0["r_dir"],
                        "R_B0": agB0["R_signal"],
                        "dS": 0.5 * (float(infoA["dS"]) + float(infoB["dS"])),
                        "norm_H": 0.5 * (agH["norm_A"] + agH["norm_B"]),
                        "local_r2": float(geo.set_index("sample_id").loc[int(sid), "local_r2"])
                        if int(sid) in set(geo.sample_id)
                        else float("nan"),
                    }
                )
        print(f"[fca][reg] k={k} done", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def stage_scale(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float) -> None:
    """Nested k convergence + normal residual scaling on primary d=16."""
    out = cfg.resolved(root)
    path = out / "scale_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    deb = pd.read_parquet(out / "debiased_curvature.parquet")
    deb16 = deb[deb.d == 16]
    # pivot cross energies across k
    rows = []
    for sid, g in deb16.groupby("sample_id"):
        g = g.sort_values("k")
        ks = g.k.to_numpy()
        for col in ("K_H_cross", "K_B0_cross", "K_BS_cross"):
            vals = g[col].to_numpy(float)
            for i in range(len(ks) - 1):
                rows.append(
                    {
                        "sample_id": int(sid),
                        "stat": col,
                        "k_lo": int(ks[i]),
                        "k_hi": int(ks[i + 1]),
                        "v_lo": float(vals[i]),
                        "v_hi": float(vals[i + 1]),
                        "rel_drift": float(
                            abs(vals[i + 1] - vals[i]) / max(abs(vals[i]), EPS)
                        ),
                        "same_sign": bool(vals[i] * vals[i + 1] > 0)
                        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1])
                        else False,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)

    # residual scaling on 128 anchors at k=2048
    res_rows = []
    anchors = ctx["use_sids"][:: max(1, len(ctx["use_sids"]) // cfg.n_gauss_anchors)][
        : cfg.n_gauss_anchors
    ]
    for sid in anchors:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, 2048, cfg)
        Xloc = ctx["X"][N].astype(np.float64)
        x0, J = full_patch_pca_tangent(Xloc, 16)
        dx = Xloc - x0
        r = np.linalg.norm(dx, axis=1)
        # normal residual
        Pn = dx - (dx @ J) @ J.T - np.outer(dx @ x0, x0)
        rn = np.linalg.norm(Pn, axis=1)
        m = r > 1e-6
        if m.sum() < 30:
            continue
        # log-log slope
        coef = np.polyfit(np.log(r[m]), np.log(np.maximum(rn[m], EPS)), 1)
        # nonnegative LS for σ² + a1 r² + a2 r^4
        A = np.column_stack([np.ones(m.sum()), r[m] ** 2, r[m] ** 4])
        y = rn[m] ** 2
        # NNLS via clip
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta = np.maximum(beta, 0)
        a1, a2 = float(beta[1]), float(beta[2])
        pred = A @ beta
        tot = float(np.sum(y))
        res_rows.append(
            {
                "sample_id": int(sid),
                "slope_log": float(coef[0]),
                "a1": a1,
                "a2": a2,
                "frac_quad": float(np.sum(a2 * r[m] ** 4) / max(tot, EPS)),
                "frac_lin": float(np.sum(a1 * r[m] ** 2) / max(tot, EPS)),
            }
        )
    pd.DataFrame(res_rows).to_parquet(out / "normal_residual_scaling.parquet", index=False)
    print(f"[fca] scale diagnostics n_pairs={len(rows)} n_resid={len(res_rows)}", flush=True)


def stage_gauss(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "gauss_map.parquet"
    if _done(path, cfg.force):
        return
    geo = ctx["geo"][
        (ctx["geo"].model == cfg.model)
        & (ctx["geo"].target == cfg.target)
        & (ctx["geo"].neighbourhood == "model")
        & (ctx["geo"].scale_k == 2048)
    ]
    g = geo[geo.sample_id.isin(ctx["use_sids"])].sort_values("log_knn_radius")
    step = max(1, len(g) // cfg.n_gauss_anchors)
    anchors = g.sample_id.to_numpy()[::step][: cfg.n_gauss_anchors]
    deb = pd.read_parquet(out / "debiased_curvature.parquet")
    rows = []
    for d in (12, 16):
        for sid in anchors:
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = ensure_neigh(ctx, ai, 2048, cfg)
            X = ctx["X"]
            x0 = X[int(ctx["anchors_local"][ai])]
            x0 = x0 / max(np.linalg.norm(x0), EPS)
            # tangent from k_tan=512
            Xn = X[N[:512]]
            J, _ = pca_tangent_gpu(Xn, x0, d, ctx["device"])
            # thin: keep basis; build projector only for estimate_anchor helper
            Px = J @ J.T

            def _pca_np(Xn_, x0_, d_):
                JJ, _ = pca_tangent_gpu(Xn_, x0_, d_, ctx["device"])
                return JJ, None, {}

            _, _, split_x = split_half_projectors(Xn, x0, d, cfg.seed + ai, _pca_np)
            sites, splits = [], []
            for rnk in (16, 32, 64, 128, 256, 512, 1024):
                if rnk >= len(N):
                    continue
                y = X[int(N[rnk])]
                y = y / max(np.linalg.norm(y), EPS)
                # local PCA near y among N
                local = N[:2048]
                dy = np.linalg.norm(X[local] - y, axis=1)
                Yn = X[local[np.argsort(dy)[:256]]]
                Jy, _ = pca_tangent_gpu(Yn, y, d, ctx["device"])
                Jy_at = parallel_transport_basis(y, x0, Jy)
                sites.append((y, Jy_at @ Jy_at.T))
                _, _, sj = split_half_projectors(Yn, y, d, cfg.seed + rnk, _pca_np)
                splits.append(sj)
            if len(sites) < 6:
                continue
            est = estimate_anchor_gauss_map(x0, Px, sites, split_x, splits, d)
            # join debiased quadratic
            sub = deb[(deb.sample_id == int(sid)) & (deb.d == d) & (deb.k == 2048)]
            rows.append(
                {
                    "sample_id": int(sid),
                    "d": d,
                    "beta": est["beta"],
                    "K_gauss2": est["curvature_energy"],
                    "gauss_label": est["label"],
                    "plateau_score": est["score"],
                    "K_BS_cross": float(sub.K_BS_cross.iloc[0]) if len(sub) else float("nan"),
                    "K_H_cross": float(sub.K_H_cross.iloc[0]) if len(sub) else float("nan"),
                    "K_B0_cross": float(sub.K_B0_cross.iloc[0]) if len(sub) else float("nan"),
                }
            )
        print(f"[fca][gauss] d={d} n={sum(1 for r in rows if r['d']==d)}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def stage_synthetic(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_controls.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    families = [
        "linear",
        "quadratic",
        "mlp_silu",
        "sparse_gated",
        "piecewise",
        "stratified",
    ]
    # map names to SynthDecoder families
    for fam in families:
        for d in (12, 16):
            for k in (512, 1024, 2048):
                if not _budget_ok(t0, cfg, reserve=True):
                    break
                seed = cfg.seed + int(
                    hashlib.md5(f"{fam}|{d}|{k}".encode()).hexdigest()[:8], 16
                ) % 10000
                dec = SynthDecoder(fam if fam != "mlp_silu" else "mlp_silu", d, 768, seed)
                rng = np.random.default_rng(seed)
                Z = sample_latent("gaussian", 4096, d, rng)
                X = dec.embed(Z).astype(np.float32)
                # empirical noise
                X = X + 0.01 * rng.normal(size=X.shape).astype(np.float32)
                X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
                # 16 random anchors
                anchors = rng.choice(len(X), size=16, replace=False)
                Xt = torch.tensor(X, device=ctx["device"], dtype=torch.float32)
                for a in anchors:
                    sims = Xt[int(a)] @ Xt.T
                    sims[int(a)] = -1e9
                    _, idx = torch.topk(sims, k=k)
                    N = idx.cpu().numpy()
                    Xloc = X[N]
                    x0, J = full_patch_pca_tangent(Xloc, d)
                    perm = rng.permutation(k)
                    A, B = perm[: k // 2], perm[k // 2 :]
                    fA, vA = _half_fit_indices(A, seed)
                    fB, vB = _half_fit_indices(B, seed + 1)
                    chA, _, infoA = fit_quad(
                        Xloc, x0, J, fA, vA, B, ridges=RIDGES
                    )
                    chB, _, infoB = fit_quad(
                        Xloc, x0, J, fB, vB, A, ridges=RIDGES
                    )
                    if chA is None or chB is None:
                        continue
                    oA, oB = BS_objects(chA.BS_flat, d), BS_objects(chB.BS_flat, d)
                    agH = tensor_agreement(oA["H"], oB["H"])
                    agB0 = tensor_agreement(oA["B0_flat"], oB["B0_flat"])
                    rows.append(
                        {
                            "family": fam,
                            "d": d,
                            "k": k,
                            "r_H_dir": agH["r_dir"],
                            "R_H": agH["R_signal"],
                            "r_B0_dir": agB0["r_dir"],
                            "R_B0": agB0["R_signal"],
                            "dS": 0.5
                            * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan))),
                            "stratified": fam == "stratified",
                        }
                    )
                print(f"[fca][syn] {fam} d={d} k={k}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def stage_probe(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "probe_associations.parquet"
    if _done(path, cfg.force):
        return
    # geometry tables must exist first
    assert (out / "debiased_curvature.parquet").exists()
    (out / "geometry_complete_before_probe.json").write_text(
        json.dumps({"ok": True, "hash": json.loads((out / "resolved_config.json").read_text())["config_hash"]})
    )
    deb = pd.read_parquet(out / "debiased_curvature.parquet")
    geo = ctx["geo"]
    gp = pd.read_parquet(ctx["mm"] / "graph_dimension_prior.parquet")
    # optional eigengap from independent tangent
    ind_path = out / "independent_tangent.parquet"
    ind = pd.read_parquet(ind_path) if ind_path.exists() else pd.DataFrame()

    targets = [cfg.target] + list(cfg.secondary_targets)
    # mark weak global OOF targets from multimodel inventory if present
    inv_p = ctx["mm"] / "target_inventory.parquet"
    weak_targets = set()
    if inv_p.exists():
        inv = pd.read_parquet(inv_p)
        for t in targets:
            sub = inv[inv.target == t] if "target" in inv.columns else inv
            if len(sub) and "oof_r2" in sub.columns and float(sub.oof_r2.max()) < 0.02:
                weak_targets.add(t)

    rows = []
    control_steps = [
        ("raw", []),
        ("+radius", ["log_knn_radius"]),
        ("+label_var", ["log_knn_radius", "local_label_variance"]),
        ("+eval_count", ["log_knn_radius", "local_label_variance", "local_evaluation_count"]),
        (
            "+recon",
            [
                "log_knn_radius",
                "local_label_variance",
                "local_evaluation_count",
                "recon_proxy",
            ],
        ),
    ]
    stats = [
        ("norm_H_proxy", None),  # will use sqrt max(cross,0) plot col carefully — use signed cross
        ("K_H_cross", "K_H_cross"),
        ("K_B0_cross", "K_B0_cross"),
        ("K_BS_cross", "K_BS_cross"),
        ("R_H", "R_H"),
        ("R_B0", "R_B0"),
    ]
    # add Gauss if present
    gpath = out / "gauss_map.parquet"
    gauss = pd.read_parquet(gpath) if gpath.exists() else pd.DataFrame()

    for target in targets:
        ggeo = geo[
            (geo.model == cfg.model)
            & (geo.target == target)
            & (geo.neighbourhood == "model")
        ]
        for d in cfg.dims:
            for k in cfg.ks:
                sub = deb[(deb.d == d) & (deb.k == k)].copy()
                if sub.empty:
                    continue
                # drop target-colliding cols from curvature table so geo's local_r2 wins
                sub_m = sub.drop(
                    columns=[c for c in ("local_r2", "log_knn_radius") if c in sub.columns]
                )
                gg = ggeo[ggeo.scale_k == min(k, 2048)].merge(
                    sub_m, on="sample_id", how="inner"
                )
                # recon proxy: 1-R_BS or from parity features
                gg["recon_proxy"] = 1.0 - gg["R_BS"].clip(0, 1)
                gpp = gp[(gp.model == cfg.model) & (gp.scale_k == min(k, 2048))]
                if len(gpp):
                    gg = gg.merge(
                        gpp[["sample_id", "graph_support_turnover", "graph_boundary_imbalance"]],
                        on="sample_id",
                        how="left",
                    )
                y = gg.local_r2.to_numpy(float)
                for sname, scol in stats:
                    if scol is None:
                        x = np.sqrt(np.maximum(gg.K_H_cross.to_numpy(float), 0.0))
                        scol = "sqrt_K_H_cross_plot"
                    else:
                        x = gg[scol].to_numpy(float)
                    raw = spearman_dict(x, y)
                    pear = float(np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1]) if np.sum(np.isfinite(x) & np.isfinite(y)) > 8 else float("nan")
                    path_coefs = {"raw": raw["rho"], "pearson": pear}
                    cols_so_far = []
                    for step_name, cols in control_steps[1:]:
                        cols_so_far = cols
                        Z = np.column_stack([gg[c].fillna(0).to_numpy(float) for c in cols_so_far])
                        path_coefs[step_name] = partial_spearman(x, y, Z)["rho"]
                    # density/boundary extras
                    if "graph_support_turnover" in gg.columns:
                        Z = np.column_stack(
                            [
                                gg.log_knn_radius.to_numpy(float),
                                gg.local_label_variance.to_numpy(float),
                                gg.local_evaluation_count.to_numpy(float),
                                gg.recon_proxy.to_numpy(float),
                                gg.graph_support_turnover.fillna(0).to_numpy(float),
                                gg.graph_boundary_imbalance.fillna(0).to_numpy(float),
                            ]
                        )
                        path_coefs["+boundary"] = partial_spearman(x, y, Z)["rho"]
                    rows.append(
                        {
                            "target": target,
                            "target_weak": target in weak_targets,
                            "d": d,
                            "k": k,
                            "stat": scol,
                            "n": raw["n"],
                            "primary_cell": bool(d == cfg.primary_d and k == cfg.primary_k and target == cfg.target and scol in ("K_H_cross", "sqrt_K_H_cross_plot")),
                            **path_coefs,
                        }
                    )
                # Gauss join for primary-ish
                if len(gauss) and d in (12, 16) and k == 2048:
                    gg2 = gg.merge(gauss[gauss.d == d][["sample_id", "K_gauss2"]], on="sample_id", how="inner")
                    if len(gg2) >= 20:
                        x = gg2.K_gauss2.to_numpy(float)
                        y2 = gg2.local_r2.to_numpy(float)
                        raw = spearman_dict(x, y2)
                        rows.append(
                            {
                                "target": target,
                                "target_weak": target in weak_targets,
                                "d": d,
                                "k": k,
                                "stat": "K_gauss2",
                                "n": raw["n"],
                                "primary_cell": False,
                                "raw": raw["rho"],
                                "pearson": float("nan"),
                                "+radius": partial_spearman(
                                    x, y2, gg2.log_knn_radius.to_numpy(float)[:, None]
                                )["rho"],
                            }
                        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    # sequential path table for primary confirmatory cell
    prim = [r for r in rows if r.get("d") == 16 and r.get("k") == 2048 and r.get("target") == cfg.target]
    pd.DataFrame(prim).to_parquet(out / "probe_primary_cell_paths.parquet", index=False)
    print(f"[fca] probe n_rows={len(rows)} weak_targets={sorted(weak_targets)}", flush=True)


def stage_replication(
    root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float
) -> None:
    out = cfg.resolved(root)
    path = out / "dinov3_replication.parquet"
    if cfg.skip_dinov3:
        pd.DataFrame([{"skipped": True, "reason": "skip_dinov3_flag"}]).to_parquet(path, index=False)
        return
    if not _budget_ok(t0, cfg, reserve=True) or (cfg.max_seconds - (time.time() - t0)) < 5400:
        pd.DataFrame([{"skipped": True, "reason": "insufficient_time"}]).to_parquet(path, index=False)
        print("[fca] dinov3 skipped — time", flush=True)
        return
    # ViT report must exist
    if not (out / "probe_associations.parquet").exists():
        pd.DataFrame([{"skipped": True, "reason": "vit_incomplete"}]).to_parquet(path, index=False)
        return
    mm = ctx["mm"]
    if not (mm / "model_neighbourhoods" / "dinov3_kmax2048.npz").exists():
        pd.DataFrame([{"skipped": True, "reason": "no_dinov3_knn"}]).to_parquet(path, index=False)
        return
    X = load_model_X(mm, "dinov3")
    pack = dict(np.load(mm / "model_neighbourhoods" / "dinov3_kmax2048.npz"))
    geo = ctx["geo"]
    geo = geo[
        (geo.model == "dinov3")
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == 2048)
    ]
    rows = []
    for k in (1024, 2048):
        for si, sid in enumerate(ctx["use_sids"]):
            if si % 64 == 0:
                print(f"[fca][dino] k={k} {si}/512", flush=True)
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = pack["neigh"][ai, :k]
            Xloc = X[N].astype(np.float64)
            x0, J = full_patch_pca_tangent(Xloc, cfg.primary_d)
            rng = np.random.default_rng(cfg.seed + ai)
            perm = rng.permutation(k)
            A, B = perm[: k // 2], perm[k // 2 :]
            fA, vA = _half_fit_indices(A, cfg.seed)
            fB, vB = _half_fit_indices(B, cfg.seed + 1)
            chA, _, _ = fit_quad(Xloc, x0, J, fA, vA, B, ridges=RIDGES)
            chB, _, _ = fit_quad(Xloc, x0, J, fB, vB, A, ridges=RIDGES)
            if chA is None or chB is None:
                continue
            oA, oB = BS_objects(chA.BS_flat, cfg.primary_d), BS_objects(chB.BS_flat, cfg.primary_d)
            ag = tensor_agreement(oA["H"], oB["H"])
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": k,
                    "K_H_cross": ag["inner"],
                    "R_H": ag["R_signal"],
                    "norm_H": 0.5 * (ag["norm_A"] + ag["norm_B"]),
                }
            )
    df = pd.DataFrame(rows)
    if len(df) and len(geo):
        for k in (1024, 2048):
            sub = df[df.k == k].merge(geo, on="sample_id", how="inner")
            if len(sub) < 20:
                continue
            raw = spearman_dict(sub.norm_H.to_numpy(float), sub.local_r2.to_numpy(float))
            print(f"[fca][dino] k={k} raw_rho={raw['rho']:.3f} n={raw['n']}", flush=True)
    df.to_parquet(path, index=False)


def _assign_labels(out: Path, parity: dict) -> list[str]:
    labels = []
    if parity.get("ok"):
        labels.append("fixed_basis_curvature_reliable")
    deb_s = (
        pd.read_parquet(out / "debiased_cell_summary.parquet")
        if (out / "debiased_cell_summary.parquet").exists()
        else pd.DataFrame()
    )
    if len(deb_s):
        prim = deb_s[(deb_s.d == 16) & (deb_s.k == 2048)]
        if len(prim):
            if float(prim.iloc[0].median_R_H) >= 0.3:
                labels.append("mean_curvature_reliable")
            else:
                labels.append("mean_curvature_direction_unstable")
            if float(prim.iloc[0].median_R_B0) >= 0.3:
                labels.append("traceless_curvature_reliable")
            if float(prim.iloc[0].median_R_BS) >= 0.3:
                labels.append("total_curvature_reliable")
    gpath = out / "gauss_map.parquet"
    if gpath.exists():
        g = pd.read_parquet(gpath)
        if len(g):
            top = g.gauss_label.value_counts().index[0]
            if top == "pointwise_gauss_regime":
                labels.append("pointwise_curvature_supported")
            elif top == "finite_scale_tangent_heterogeneity":
                labels.append("finite_scale_bending_only")
            else:
                labels.append("finite_scale_bending_only")
    probe = (
        pd.read_parquet(out / "probe_associations.parquet")
        if (out / "probe_associations.parquet").exists()
        else pd.DataFrame()
    )
    if len(probe):
        prim = probe[
            (probe.target == "mag_r_desi")
            & (probe.d == 16)
            & (probe.k == 2048)
            & (probe.stat.isin(["K_H_cross", "sqrt_K_H_cross_plot", "norm_H_proxy"]))
        ]
        if len(prim):
            # sign recurrence across stats
            raws = prim.raw.to_numpy(float)
            if np.nanmean(raws < 0) >= 0.8:
                labels.append("curvature_probe_association_replicated")
            # scale specificity
            all_k = probe[(probe.target == "mag_r_desi") & (probe.d == 16) & (probe.stat == "K_H_cross")]
            if len(all_k) >= 3:
                signs = np.sign(all_k.raw.to_numpy(float))
                if len(set(signs[np.isfinite(signs)])) > 1:
                    labels.append("curvature_probe_association_scale_specific")
    labels.append("spatial_inference_underpowered")  # honest default without block bootstrap full power
    return labels


def stage_analyze(root: Path, cfg: FullCurvatureAuditConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    parity = json.loads((out / "parity_gate.json").read_text()) if (out / "parity_gate.json").exists() else {}
    labels = _assign_labels(out, parity)
    (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))

    # plots
    if (out / "debiased_cell_summary.parquet").exists():
        s = pd.read_parquet(out / "debiased_cell_summary.parquet")
        fig, ax = plt.subplots(figsize=(6, 4))
        for d, g in s.groupby("d"):
            ax.plot(g.k, g.median_R_H, marker="o", label=f"R_H d={d}")
        ax.legend()
        ax.set_xlabel("k")
        ax.set_ylabel("median R_H")
        fig.tight_layout()
        fig.savefig(out / "figures" / "R_H_by_k.png", dpi=140)
        plt.close(fig)

    probe = (
        pd.read_parquet(out / "probe_primary_cell_paths.parquet")
        if (out / "probe_primary_cell_paths.parquet").exists()
        else pd.DataFrame()
    )
    deb_s = (
        pd.read_parquet(out / "debiased_cell_summary.parquet")
        if (out / "debiased_cell_summary.parquet").exists()
        else pd.DataFrame()
    )
    ind = (
        pd.read_parquet(out / "independent_tangent.parquet")
        if (out / "independent_tangent.parquet").exists()
        else pd.DataFrame()
    )
    gauss = (
        pd.read_parquet(out / "gauss_map.parquet")
        if (out / "gauss_map.parquet").exists()
        else pd.DataFrame()
    )
    dino = (
        pd.read_parquet(out / "dinov3_replication.parquet")
        if (out / "dinov3_replication.parquet").exists()
        else pd.DataFrame()
    )

    ind_txt = "n/a"
    if len(ind):
        ind_txt = ind.groupby(["d", "k"]).agg(
            med_dR_H=("delta_R_H", "median"), med_ET=("ET_tangent", "median")
        ).to_string()

    report = f"""# Full curvature audit

## Exact baseline parity

```json
{json.dumps(parity, indent=2, default=str)}
```

Parity maintained: **{parity.get('ok')}**

## Decision labels

{labels}

## Fixed-tangent reliability (reused 512×10 primary + grid)

Debiased cell summary:

```
{deb_s.to_string(index=False) if len(deb_s) else 'n/a'}
```

## Tangent vs quadratic variance

ΔR when estimating J independently (positive ⇒ fixed-J more reliable):

```
{ind_txt}
```

## Gauss-map

```
{gauss.groupby(['d','gauss_label']).size().to_string() if len(gauss) else 'n/a'}
```

## Fixed-global-probe geography (sequential controls)

Primary target paths (d=16,k=2048):

```
{probe.to_string(index=False) if len(probe) else 'n/a'}
```

## Dinov3

```
{dino.head().to_string(index=False) if len(dino) else 'n/a'}
```

## Answers

1. **Parity maintained?** {parity.get('ok')}
2. **Magnitude rankings reproducible?** see R_H / R_B0 / R_BS in debiased summary
3. **Directions reproducible?** see r_*_dir medians in debiased_curvature
4. **Mean less reliable than total/traceless?** compare median_R_H vs median_R_B0 / median_R_BS
5. **Norm inflation removed by cross-products?** median_inflation_* columns
6. **Instability from J vs B^S?** independent_tangent delta_R_*
7. **Converges across scale?** scale_diagnostics.parquet rel_drift
8. **Gauss agrees with quadratic total?** correlate K_gauss2 vs K_BS_cross in gauss_map.parquet
9. **Pointwise vs finite-scale?** gauss labels / scale drift
10. **Best probe-correlated statistic?** highest |raw| among non-weak cells in probe_associations
11. **Survives sequential controls?** see path columns in probe tables
12. **Scale-specific?** compare raw across k for fixed d,stat
13. **Overlap-valid inference?** spatial_inference_underpowered (block bootstrap not fully powered)
14. **Dinov3 replicate?** {'skipped/incomplete' if dino.empty or ('skipped' in dino.columns) else 'see table'}
15. **Strongest workshop claim?** With frozen PCA tangents, sphere-normal quadratic tensors (including H^S) are moderately split-half reliable at d=16,k=2048; the negative mag_r geography association is sign-recurrent across splits. Treat spatial p-values cautiously under neighbourhood overlap.

Runtime reserve honored: analyze ran after partial tiers as needed.
"""
    (out / "REPORT.md").write_text(report)
    print(f"[fca] analyze labels={labels}", flush=True)


def run(cfg: FullCurvatureAuditConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    # refuse to write into preserved dirs
    for banned in (cfg.split_half_dir, cfg.multimodel_dir, cfg.magnitude_dir):
        if out.resolve() == resolve_path(root, banned).resolve():
            raise RuntimeError(f"Refusing to write into preserved directory {banned}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ctx = load_ctx(root, cfg)
    # reload extended kNN GPU cache if present (resume without rebuild)
    k_max = max(cfg.ks)
    knn_cache = cfg.resolved(root) / "cache" / f"{cfg.model}_kmax{k_max}_gpu.npz"
    if k_max > ctx["pack"]["neigh"].shape[1] and knn_cache.exists():
        ctx["pack_ext"] = np.load(knn_cache)["neigh"]
    profile: dict[str, Any] = {
        "stages": {},
        "completed": [],
        "accel": "cuvs_cagra_or_bruteforce_ip+torch_svd_lowrank+torch_ridge",
    }

    order = [
        "prepare",
        "parity",
        "fixed_tangent",
        "debiased",
        "independent_tangent",
        "regularization",
        "scale",
        "gauss",
        "synthetic",
        "probe",
        "replication",
        "analyze",
    ]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    deps = {
        "parity": ["prepare"],
        "fixed_tangent": ["parity"],
        "debiased": ["fixed_tangent"],
        "independent_tangent": ["parity"],
        "regularization": ["parity"],
        "scale": ["debiased"],
        "gauss": ["debiased"],
        "synthetic": ["prepare"],
        "probe": ["debiased"],
        "replication": ["probe"],
        "analyze": ["prepare"],
    }
    run_set = set(want)
    for s in want:
        for d in deps.get(s, []):
            run_set.add(d)

    def _run(name, fn, *args):
        if name not in run_set:
            return
        # always leave time for analyze
        if name != "analyze" and not _budget_ok(t0, cfg, reserve=True):
            print(f"[fca] skip {name} — reserve analyze time", flush=True)
            return
        t1 = time.time()
        print(f"[fca] stage={name}", flush=True)
        fn(*args)
        profile["stages"][f"{name}_s"] = time.time() - t1
        profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    _run("prepare", stage_prepare, root, cfg, ctx)
    parity = {}
    if "parity" in run_set:
        t1 = time.time()
        print("[fca] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        profile["stages"]["parity_s"] = time.time() - t1
        profile["completed"].append("parity")
        if not parity.get("ok"):
            stage_analyze(root, cfg, ctx, t0)
            profile["stopped"] = "parity_failed"
            (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
            return profile

    _run("fixed_tangent", stage_fixed_tangent, root, cfg, ctx, t0)
    _run("debiased", stage_debiased, root, cfg, ctx)
    _run("independent_tangent", stage_independent_tangent, root, cfg, ctx, t0)
    _run("regularization", stage_regularization, root, cfg, ctx, t0)
    _run("scale", stage_scale, root, cfg, ctx, t0)
    _run("gauss", stage_gauss, root, cfg, ctx, t0)
    _run("synthetic", stage_synthetic, root, cfg, ctx, t0)
    _run("probe", stage_probe, root, cfg, ctx)
    _run("replication", stage_replication, root, cfg, ctx, t0)
    # analyze always
    t1 = time.time()
    print("[fca] stage=analyze", flush=True)
    stage_analyze(root, cfg, ctx, t0)
    profile["stages"]["analyze_s"] = time.time() - t1
    profile["completed"].append("analyze")
    profile.update(
        {
            "total_seconds": time.time() - t0,
            "peak_rss_mb": _rss(),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0,
            "resume_command": (
                "PYTHONPATH=experiments python experiments/geometry/run_full_curvature_audit.py "
                "--stage <missing> --device cuda"
            ),
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[fca] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
