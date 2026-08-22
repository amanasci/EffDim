"""Implicit normal-space inverse: config, freeze, and local constraint fitting.

Never writes into completed geometry output directories.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    angular_radii,
    rms_tangent_radius,
    row_l2_status,
    sphere_log_map,
)

from .algebra import (
    EPS,
    RIDGES,
    bottom_eigh,
    constraint_residuals,
    fit_h_for_a,
    implicit_shape_operators,
    loglog_exponent,
    mixed_var_nnls,
    n_quad_features,
    profiled_K,
    projector_overlap,
    qr_orthonormal,
    r2_cancel,
    sampson_batch,
    stiefel_qr,
    tangent_basis,
    unpack_h,
    weighted_phi,
)
from .classify import DEFAULT_THRESHOLDS, classify_one, consecutive_normal_count

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_STD = "outputs/geometry/physics_stable_tangent_dimension"
SOURCE_OSG = "outputs/geometry/physics_order_stratified_geometry"

PRESERVED = [SOURCE_MM, SOURCE_EDM, SOURCE_NDC, SOURCE_COV, SOURCE_STD, SOURCE_OSG]
PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"
K_CANDIDATES = [128, 256, 512, 768, 1024, 1536, 2048]


@dataclass
class ImplicitNormalConfig:
    output_dir: str = "outputs/geometry/physics_implicit_normal_inverse"
    multimodel_dir: str = SOURCE_MM
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    R: int = 20
    R_sens: list[int] = field(default_factory=lambda: [16, 24, 32])
    q_max: int = 10
    d_core: int = 12
    d_ref: int = 16
    n_null_draw: int = 8
    n_parity_anchors: int = 32
    n_synth_cal: int = 5
    n_synth_eval: int = 5
    n_refine_steps: int = 3
    n_gauss_anchors: int = 64
    n_gauss_neighbors: int = 8
    ks: list[int] = field(default_factory=list)
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 400.0
    smoke: bool = False
    n_anchors: int | None = None

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: ImplicitNormalConfig, reserve: bool = False) -> bool:
    return (cfg.max_seconds - (time.time() - t0)) > (cfg.analyze_reserve_seconds if reserve else 20.0)


def _sha16(payload: Any) -> str:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


def assert_not_preserved(out: Path, root: Path) -> None:
    resolved = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if resolved == pres or pres in resolved.parents:
            raise RuntimeError(f"refusing to write into preserved geometry dir {rel}")


def resolve_k_grid(k_max: int, *, smoke: bool, primary_k: int) -> list[int]:
    if smoke:
        grid = [k for k in [64, 96, 128, 192, 256] if k <= k_max]
        return grid[:5] if len(grid) >= 4 else grid
    grid = [k for k in K_CANDIDATES if k <= k_max]
    if primary_k <= k_max and primary_k not in grid:
        grid.append(int(primary_k))
    return sorted(set(grid))


def load_ctx(root: Path, cfg: ImplicitNormalConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(aid.read_text())["sample_ids"] if aid.exists() else [int(s) for s in anchors_sid]
    if cfg.n_anchors is not None:
        use_sids = use_sids[: int(cfg.n_anchors)]
    elif cfg.smoke:
        use_sids = use_sids[:8]
        cfg.R = min(int(cfg.R), 8)
        cfg.q_max = min(int(cfg.q_max), 6)
        cfg.n_synth_cal = min(cfg.n_synth_cal, 2)
        cfg.n_synth_eval = min(cfg.n_synth_eval, 2)
        cfg.n_parity_anchors = min(cfg.n_parity_anchors, 8)
        cfg.n_gauss_anchors = min(cfg.n_gauss_anchors, 8)
        cfg.R_sens = [r for r in cfg.R_sens if r <= 8] or [cfg.R]
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == cfg.model) & (geo.target == cfg.target) & (geo.neighbourhood == "model")]
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    freeze_p = resolve_path(root, SOURCE_EDM) / "dimension_freeze.json"
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else {}
    k_max = int(pack["neigh"].shape[1])
    ks = list(cfg.ks) if cfg.ks else resolve_k_grid(min(k_max, cfg.primary_k), smoke=cfg.smoke, primary_k=cfg.primary_k)
    X = load_model_X(mm, cfg.model)
    return {
        "mm": mm,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "pack": pack,
        "freeze": freeze,
        "X": X,
        "ks": ks,
        "k_max": k_max,
        "l2": row_l2_status(X),
        "std": resolve_path(root, SOURCE_STD),
        "ndc": resolve_path(root, SOURCE_NDC),
        "edm": resolve_path(root, SOURCE_EDM),
        "cov": resolve_path(root, SOURCE_COV),
        "osg": resolve_path(root, SOURCE_OSG),
    }


def ensure_neigh(ctx: dict, ai: int, k: int) -> np.ndarray:
    return ctx["pack"]["neigh"][ai, : min(k, ctx["pack"]["neigh"].shape[1])]


def _j_src(ctx: dict, cfg: ImplicitNormalConfig, sid: int, k: int) -> Path:
    return ctx["osg"] / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz"


def _j_ours(out: Path, cfg: ImplicitNormalConfig, sid: int, k: int) -> Path:
    return out / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz"


def cache_path(out: Path, cfg: ImplicitNormalConfig, sid: int, k: int, R: int) -> Path:
    return out / "cache" / f"{cfg.model}_{int(sid)}_k{int(k)}_R{int(R)}.npz"


def load_or_compute_J(
    out: Path, ctx: dict, cfg: ImplicitNormalConfig, sid: int, k: int, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_need = max([cfg.R, cfg.d_ref, *cfg.R_sens])
    theta = None
    rms = None
    for p in (_j_src(ctx, cfg, sid, k), _j_ours(out, cfg, sid, k)):
        if p.exists():
            zc = np.load(p)
            J = zc["J"]
            if J.shape[1] >= min(d_need, cfg.R):
                ev = zc["ev"] if "ev" in zc.files else np.zeros(J.shape[1])
                theta = zc["theta"] if "theta" in zc.files else None
                rms = zc["rms"] if "rms" in zc.files else None
                return J, ev, (theta, rms)
    J, ev = nested_uncentred_svd(Z, d_need, device=ctx["device"])
    dest = _j_ours(out, cfg, sid, k)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, J=J, ev=ev)
    return J, ev, (None, None)


def carrier_coords(Z: np.ndarray, J: np.ndarray, R: int) -> tuple[np.ndarray, float]:
    r = min(int(R), J.shape[1])
    UR = J[:, :r]
    Y = Z @ UR
    outside = float(np.mean(np.maximum(np.sum(Z * Z, axis=1) - np.sum(Y * Y, axis=1), 0.0)))
    return Y, outside


def pick_lambda(
    Ytr: np.ndarray, Phi_tr: np.ndarray, Yva: np.ndarray, Phi_va: np.ndarray, ridges: list[float]
) -> tuple[float, float]:
    losses, dfs = [], []
    for lam in ridges:
        info = profiled_K(Ytr, Phi_tr, lam)
        _, U = bottom_eigh(info["K"], min(4, Ytr.shape[1]))
        a = U[:, 0]
        h = fit_h_for_a(Ytr, Phi_tr, a, lam)
        r = Yva @ a + Phi_va @ h
        losses.append(float(np.mean(r * r)))
        dfs.append(float(info["df"]))
    best_idx = int(np.argmin(losses))
    se = float(np.std(losses) / max(np.sqrt(len(losses)), 1.0))
    thresh = losses[best_idx] + se
    chosen = best_idx
    for i, loss in enumerate(losses):  # increasing λ: keep last within 1-SE (more regularized)
        if loss <= thresh:
            chosen = i
    return float(ridges[chosen]), float(dfs[chosen])


def _refine(Y: np.ndarray, A: np.ndarray, Hs: np.ndarray, Phi: np.ndarray, lam: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    A = A.copy()
    Hs = Hs.copy()
    q, R = A.shape[1], A.shape[0]
    step = 0.05
    for _ in range(max(int(steps), 0)):
        samp = sampson_batch(Y, A, Hs)
        G = np.zeros_like(A)
        eps = 1e-3
        base = float(np.mean(samp))
        for i in range(R):
            for j in range(q):
                Ap = A.copy()
                Ap[i, j] += eps
                Ap = qr_orthonormal(Ap)
                G[i, j] = (float(np.mean(sampson_batch(Y, Ap, Hs))) - base) / eps
        A = stiefel_qr(A, G, step)
        for ℓ in range(q):
            h = fit_h_for_a(Y, Phi, A[:, ℓ], lam)
            Hs[ℓ] = unpack_h(h, R)
    return A, Hs


def fit_constraints(
    Y: np.ndarray,
    radii: np.ndarray,
    *,
    q_max: int,
    seed: int,
    n_null: int,
    refine_steps: int = 0,
) -> dict[str, Any]:
    n, R = Y.shape
    Aidx, Bidx = radial_stratified_halves(radii, seed)
    if min(len(Aidx), len(Bidx)) < R + 4:
        return {"ok": False, "reason": "split_too_small"}
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(Aidx)
    nva = max(8, len(perm) // 5)
    Aval, Afit = perm[:nva], perm[nva:]
    if len(Afit) < R + 2:
        Afit, Aval = Aidx, Bidx[: max(8, len(Bidx) // 5)]
    Phi = weighted_phi(Y)
    lam, df = pick_lambda(Y[Afit], Phi[Afit], Y[Aval], Phi[Aval], RIDGES)
    infoA = profiled_K(Y[Aidx], Phi[Aidx], lam)
    infoB = profiled_K(Y[Bidx], Phi[Bidx], lam)
    q_use = min(int(q_max), R)
    evA, UA = bottom_eigh(infoA["K"], q_use)
    evB, UB = bottom_eigh(infoB["K"], q_use)
    Cyy = (Y[Aidx].T @ Y[Aidx]) / max(len(Aidx), 1)
    ev_lin, Ulin = bottom_eigh(Cyy, q_use)
    ev_lin_all, Ulin_all = np.linalg.eigh(0.5 * (Cyy + Cyy.T))
    # tangent-matched null: random dirs in the top-variance half
    top = Ulin_all[:, max(R // 2, 1) :]
    null_corr = []
    for _ in range(max(int(n_null), 1)):
        g = rng.normal(size=top.shape[1])
        aN = top @ (g / max(np.linalg.norm(g), EPS))
        aN = aN / max(np.linalg.norm(aN), EPS)
        hN = fit_h_for_a(Y[Aidx], Phi[Aidx], aN, lam)
        rN = Y[Bidx] @ aN + Phi[Bidx] @ hN
        null_corr.append(float(np.mean(rN * rN)))
    # quadratic-structure null: permute columns within radial bins
    order = np.argsort(radii[Aidx])
    bins = np.array_split(Aidx[order], min(8, max(2, len(Aidx) // 16)))
    null_struct = []
    for _ in range(max(int(n_null) // 2, 1)):
        Yp = Y.copy()
        for b in bins:
            if len(b) < 2:
                continue
            for c in range(R):
                Yp[b, c] = Y[rng.permutation(b), c]
        infN = profiled_K(Yp[Aidx], weighted_phi(Yp)[Aidx], lam)
        evN, UN = bottom_eigh(infN["K"], 1)
        aN = UN[:, 0]
        hN = fit_h_for_a(Yp[Aidx], weighted_phi(Yp)[Aidx], aN, lam)
        rN = Yp[Bidx] @ aN + weighted_phi(Yp)[Bidx] @ hN
        null_struct.append(float(np.mean(rN * rN)))
    null_mse = float(np.quantile(null_corr, 0.20)) if null_corr else 0.0
    tot_var = float(np.mean(np.sum(Y[Bidx] ** 2, axis=1)))
    dir_rows = []
    h_pack = []
    Hs_dirs = []
    for j in range(UA.shape[1]):
        a = UA[:, j]
        h = fit_h_for_a(Y[Aidx], Phi[Aidx], a, lam)
        h_pack.append(h)
        Hs_dirs.append(unpack_h(h, R))
        rr = constraint_residuals(Y[Bidx], a, h, Phi[Bidx])
        ov1 = projector_overlap(UA[:, j : j + 1], UB[:, : min(j + 3, UB.shape[1])])
        # best 1D match in UB
        dots = np.abs(UB.T @ a)
        ov1 = float(dots.max() ** 2) if dots.size else float("nan")
        lin_mse = float(np.mean(rr["linear"] ** 2))
        corr_mse = float(np.mean(rr["corrected"] ** 2))
        pct = float(np.mean(np.asarray(null_corr) <= corr_mse)) if null_corr else float("nan")
        dir_rows.append(
            {
                "j": j,
                "eval_K": float(evA[j]) if j < len(evA) else float("nan"),
                "eval_lin": float(ev_lin[j]) if j < len(ev_lin) else float("nan"),
                "overlap": ov1,
                "lin_mse": lin_mse,
                "corr_mse": corr_mse,
                "cancel_r2": r2_cancel(rr["linear"], rr["corrected"]),
                "var_share": lin_mse / max(tot_var / max(R, 1), EPS),
                "null_mse": null_mse,
                "null_percentile": pct,
                "null_struct": float(np.median(null_struct)) if null_struct else float("nan"),
                "pred_cos": float(
                    np.abs(np.dot(rr["linear"], -rr["quadratic"]))
                    / max(np.linalg.norm(rr["linear"]) * np.linalg.norm(rr["quadratic"]), EPS)
                ),
            }
        )
    q_rows = []
    for q in range(0, UA.shape[1] + 1):
        if q == 0:
            q_rows.append(
                {
                    "q": 0,
                    "d1": R,
                    "lin_mse": float("nan"),
                    "corr_mse": float("nan"),
                    "cancel_r2": float("nan"),
                    "sampson": 0.0,
                    "sampson_refined": 0.0,
                    "overlap": float("nan"),
                    "eval_K": float("nan"),
                    "note": "q=0 is the full carrier and has trivial zero residual",
                }
            )
            continue
        A = qr_orthonormal(UA[:, :q])
        Hs = np.stack(Hs_dirs[:q], axis=0)
        samp = sampson_batch(Y[Bidx], A, Hs)
        samp_r = samp
        if refine_steps > 0:
            A_ref, Hs_ref = _refine(Y[Aidx], A, Hs, Phi[Aidx], lam, refine_steps)
            samp_r = sampson_batch(Y[Bidx], A_ref, Hs_ref)
        q_rows.append(
            {
                "q": q,
                "d1": R - q,
                "eval_K": float(evA[q - 1]),
                "eval_lin": float(ev_lin[q - 1]) if q - 1 < len(ev_lin) else float("nan"),
                "overlap": projector_overlap(UA[:, :q], UB[:, :q]),
                "lin_mse": float(np.mean([d["lin_mse"] for d in dir_rows[:q]])),
                "corr_mse": float(np.mean([d["corr_mse"] for d in dir_rows[:q]])),
                "cancel_r2": float(np.nanmean([d["cancel_r2"] for d in dir_rows[:q]])),
                "sampson": float(np.mean(samp)),
                "sampson_refined": float(np.mean(samp_r)),
                "var_share": float(np.mean([d["var_share"] for d in dir_rows[:q]])),
                "df": df,
                "lam": lam,
                "null_mse": null_mse,
            }
        )
    return {
        "ok": True,
        "lam": lam,
        "df": df,
        "ev_K": evA,
        "ev_lin": ev_lin,
        "UA": UA,
        "UB": UB,
        "Ulin": Ulin,
        "h_pack": np.stack(h_pack, axis=0) if h_pack else np.zeros((0, n_quad_features(R))),
        "null_mse": null_mse,
        "null_corr": np.asarray(null_corr, dtype=np.float64),
        "dir_rows": dir_rows,
        "q_rows": q_rows,
        "tot_var": tot_var,
        "n_tr": int(len(Aidx)),
        "n_te": int(len(Bidx)),
        "R": R,
        "Aidx": Aidx,
        "Bidx": Bidx,
    }


def scaling_for_directions(
    Y: np.ndarray, radii: np.ndarray, UA: np.ndarray, h_pack: np.ndarray, n_bins: int = 6
) -> list[dict[str, Any]]:
    Phi = weighted_phi(Y)
    order = np.argsort(radii)
    bins = np.array_split(order, n_bins)
    out = []
    for j in range(UA.shape[1]):
        a = UA[:, j]
        h = h_pack[j]
        rr = constraint_residuals(Y, a, h, Phi)
        r_cent, amp_raw, amp_corr, var_raw, var_corr = [], [], [], [], []
        for b in bins:
            if len(b) < 8:
                continue
            r_cent.append(float(np.median(radii[b])))
            amp_raw.append(float(np.sqrt(np.mean(rr["linear"][b] ** 2))))
            amp_corr.append(float(np.sqrt(np.mean(rr["corrected"][b] ** 2))))
            var_raw.append(float(np.mean(rr["linear"][b] ** 2)))
            var_corr.append(float(np.mean(rr["corrected"][b] ** 2)))
        raw_e = loglog_exponent(np.asarray(r_cent), np.asarray(amp_raw))
        corr_e = loglog_exponent(np.asarray(r_cent), np.asarray(amp_corr))
        raw_v = loglog_exponent(np.asarray(r_cent), np.asarray(var_raw))
        corr_v = loglog_exponent(np.asarray(r_cent), np.asarray(var_corr))
        mix_raw = mixed_var_nnls(np.asarray(r_cent), np.asarray(var_raw), "r2_r4_c")
        mix_corr = mixed_var_nnls(np.asarray(r_cent), np.asarray(var_corr), "r6_c")
        out.append(
            {
                "j": j,
                "amp_exp_raw": raw_e["alpha"],
                "amp_exp_corr": corr_e["alpha"],
                "var_exp_raw": raw_v["alpha"],
                "var_exp_corr": corr_v["alpha"],
                "amp_resolved": raw_e["resolved"],
                "mix_raw_r2": mix_raw.get("r2_fit", float("nan")),
                "mix_raw_alpha": mix_raw.get("alpha", float("nan")),
                "mix_raw_beta": mix_raw.get("beta", float("nan")),
                "mix_corr_gamma": mix_corr.get("gamma", float("nan")),
                "floor_raw": mix_raw.get("c", float("nan")),
                "floor_corr": mix_corr.get("c", float("nan")),
            }
        )
    return out


def classify_fit(fit: dict[str, Any], scaling: list[dict[str, Any]] | None, persist: list[float], thr: dict) -> list[str]:
    labels = []
    scaling = scaling or [{} for _ in fit["dir_rows"]]
    for j, d in enumerate(fit["dir_rows"]):
        sc = scaling[j] if j < len(scaling) else {}
        p = persist[j] if j < len(persist) else float("nan")
        labels.append(
            classify_one(
                cancel_r2=d["cancel_r2"],
                split_overlap=d["overlap"],
                persist=p,
                raw_mse=d["lin_mse"],
                corr_mse=d["corr_mse"],
                null_mse=d["null_mse"],
                amp_exp=sc.get("amp_exp_raw", float("nan")),
                corr_exp=sc.get("amp_exp_corr", float("nan")),
                var_share=d["var_share"],
                thr=thr,
            )
        )
    return labels


def dimension_from_labels(labels: list[str], R: int, ev_lin: np.ndarray, thr: dict) -> dict[str, Any]:
    cN_minus = consecutive_normal_count(labels)
    d1_plus = int(R - cN_minus)
    unscanned = max(int(R) - len(labels), 0)
    extra = 0
    if cN_minus > 0:
        for lab in labels[cN_minus:]:
            if lab == "first_order_tangent":
                extra += 1
            else:
                break
        d1_minus = int(unscanned + extra)
    else:
        d1_minus = 0
    n_flat = sum(1 for lab in labels[:cN_minus] if lab == "approximately_flat_normal")
    n_curv = sum(1 for lab in labels[:cN_minus] if lab == "curvature_active_normal")
    return {
        "cN_minus": int(cN_minus),
        "d1_plus": int(d1_plus),
        "d1_minus": int(d1_minus),
        "n_flat_prefix": int(n_flat),
        "n_curv_prefix": int(n_curv),
        "labels": labels,
    }


def implicit_q2_from_pack(UA: np.ndarray, h_pack: np.ndarray, q: int) -> dict[str, Any]:
    if q <= 0:
        return {"q": 0, "q2": 0, "s": []}
    R = UA.shape[0]
    A = qr_orthonormal(UA[:, :q])
    Hs = np.stack([unpack_h(h_pack[j], R) for j in range(q)], axis=0)
    Ss, Bflat = implicit_shape_operators(A, Hs)
    if Bflat.size == 0:
        return {"q": q, "q2": 0, "s": []}
    s = np.linalg.svd(Bflat, compute_uv=False)
    s = np.asarray(s, dtype=np.float64)
    energy = np.cumsum(s**2) / max(float(np.sum(s**2)), EPS)
    # consecutive prefix above 8% of leading singular value and 5% extra energy
    q2 = 0
    s0 = float(s[0]) if len(s) else 0.0
    for i, si in enumerate(s):
        if si >= 0.12 * max(s0, EPS) and (i == 0 or energy[i] - energy[i - 1] >= 0.04 or si >= 0.25 * s0):
            q2 = i + 1
        else:
            break
    return {"q": q, "q2": int(q2), "s": s.tolist(), "energy": energy.tolist(), "Tdim": int(tangent_basis(A, R).shape[1])}


def stage_prepare(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    assert_not_preserved(out, root)
    for sub in ("cache", "batches", "figures", "logs", "synth", "J"):
        (out / sub).mkdir(exist_ok=True)
    mm = ctx["mm"]
    x_path = mm / "prepare" / "models" / f"{cfg.model}.npz"
    pack_path = mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    probes = mm / "global_probes" / "oof_predictions"
    probe_hash = _sha16(sorted(p.name for p in probes.glob("*")) if probes.exists() else "missing")
    osg_q = ctx["osg"] / "quadratic_rank_summary.csv"
    meta = {
        "config": asdict(cfg),
        "protocol": "implicit_normal_inverse_v1",
        "preserved": PRESERVED,
        "ks": ctx["ks"],
        "n_anchors": len(ctx["use_sids"]),
        "l2_status": ctx["l2"],
        "primary_carrier_R": cfg.R,
        "R_sensitivity": cfg.R_sens,
        "q_scan": list(range(0, cfg.q_max + 1)),
        "no_assume_d12_or_d16": True,
        "sphere_log": True,
        "anchor_origin": True,
        "no_local_mean_centre": True,
        "software": {"numpy": np.__version__, "torch": torch.__version__, "pandas": pd.__version__},
        "hashes": {
            "activations": _file_sha(x_path) if x_path.exists() else None,
            "knn_pack": _file_sha(pack_path) if pack_path.exists() else None,
            "oof_probes": probe_hash,
            "freeze": ctx["freeze"].get("dimension_config_hash"),
            "osg_q2_summary": _file_sha(osg_q) if osg_q.exists() else None,
        },
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
        "config_hash": _sha16(asdict(cfg)),
        "seeds": {"analysis": cfg.seed, "null": cfg.seed + 17},
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "sample_ids": ctx["use_sids"],
                "model": cfg.model,
                "l2_normalized": ctx["l2"]["unit_normalized"],
                "neighbour_search_metric": "inner_product",
                "split_schedule": "radial_stratified_halves",
                "reference_k": cfg.primary_k,
                "R": cfg.R,
                "seed": cfg.seed,
                "ridges": RIDGES,
                "coord": "sphere_log",
                "thresholds_frozen_from": "synthetic_calibration",
                **meta["hashes"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"[ini] prepare ks={ctx['ks']} R={cfg.R} n={len(ctx['use_sids'])}", flush=True)
    return meta


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FREEZE_HASH_EXPECTED",
    "ImplicitNormalConfig",
    "K_CANDIDATES",
    "PARITY_D12_RHO",
    "PARITY_D16_RHO",
    "PARITY_TOL",
    "PRESERVED",
    "SOURCE_OSG",
    "SOURCE_STD",
    "_budget_ok",
    "_done",
    "_file_sha",
    "_j_ours",
    "_j_src",
    "_sha16",
    "assert_not_preserved",
    "cache_path",
    "carrier_coords",
    "classify_fit",
    "dimension_from_labels",
    "ensure_neigh",
    "fit_constraints",
    "implicit_q2_from_pack",
    "load_ctx",
    "load_or_compute_J",
    "platonic_root",
    "resolve_k_grid",
    "scaling_for_directions",
    "sphere_log_map",
    "stage_prepare",
    "angular_radii",
    "rms_tangent_radius",
]

from .stages_run import run  # noqa: E402
