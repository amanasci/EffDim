"""Confirmatory multi-seed Δ_S + frozen label-blind object-level curvature features."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .data import load_prepare
from .metrics import weighted_mse
from .paths import platonic_root, resolve_path
from .quadratic import n_quad_features, quadratic_features
from .quadratic_structure import StructureConfig, build_charts, matrix_rank_stats
from .sphere_normal_quadratic import (
    NestedChart,
    _ridge_multi,
    _ridge_solve,
    chart_errors,
    fit_nested_chart,
    flatten_BS_for_svd,
    mc_pvalue,
    normalize_rows,
    normal_projector_apply,
    predictive_rank_curve,
    SphereNormalConfig,
)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget(t0: float, max_seconds: float, where: str) -> None:
    if time.time() - t0 > max_seconds:
        raise RuntimeError(f"Hard stop at {where}: {time.time() - t0:.1f}s")


# -------------------- B^S mean / traceless --------------------


def unpack_BS_symmetric(BS_flat: np.ndarray, d: int) -> np.ndarray:
    """Unpack (D,q) flat a<=b coeffs into symmetric B[D,d,d]. Off-diag: flat/2."""
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


def decompose_BS(BS_flat: np.ndarray, d: int) -> dict[str, float]:
    """
    H^S = (1/d) sum_a B_aa;  B°_ab = B_ab - δ_ab H^S.
    Verify |B|_F^2 ≈ |B°|_F^2 + d |H|^2.
    """
    B = unpack_BS_symmetric(BS_flat, d)
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)  # (D,)
    B0 = B.copy()
    for a in range(d):
        B0[:, a, a] = B[:, a, a] - H
    B_fro = float(np.linalg.norm(B))
    B0_fro = float(np.linalg.norm(B0))
    H_norm = float(np.linalg.norm(H))
    lhs = B_fro**2
    rhs = B0_fro**2 + d * (H_norm**2)
    return {
        "B_fro": B_fro,
        "H_norm": H_norm,
        "B_traceless_fro": B0_fro,
        "identity_residual": float(abs(lhs - rhs) / max(lhs, 1e-12)),
        "mean_frac": float((d * H_norm**2) / max(lhs, 1e-12)),
        "traceless_frac": float((B0_fro**2) / max(lhs, 1e-12)),
    }


# -------------------- synthetics --------------------


def _synth_base(n: int, d: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d)) * 0.3
    return rng, x0, J, U


def synth_affine_sphere(n=800, d=4, D=32, seed=0):
    _, x0, J, U = _synth_base(n, d, D, seed)
    Y = normalize_rows(x0[None, :] + U @ J.T)
    return Y, U, np.zeros((D, n_quad_features(d)))


def synth_isotropic_normal(n=800, d=4, D=32, seed=1):
    _, x0, J, U = _synth_base(n, d, D, seed)
    nvec = normal_projector_apply(np.ones(D), x0, J)
    nvec /= max(np.linalg.norm(nvec), 1e-12)
    Phi = quadratic_features(U)
    BS = np.zeros((D, n_quad_features(d)))
    # isotropic: equal diagonal modes along same normal
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                BS[:, idx] = 0.8 * nvec
            idx += 1
    Y = normalize_rows(x0[None, :] + U @ J.T + Phi @ BS.T)
    return Y, U, BS


def synth_saddle_normal(n=800, d=4, D=32, seed=2):
    _, x0, J, U = _synth_base(n, d, D, seed)
    nvec = normal_projector_apply(np.arange(D, dtype=np.float64), x0, J)
    nvec /= max(np.linalg.norm(nvec), 1e-12)
    Phi = quadratic_features(U)
    BS = np.zeros((D, n_quad_features(d)))
    # traceless diagonal: +c, +c, -c, -c (for d=4)
    signs = np.array([1.0, 1.0, -1.0, -1.0][:d])
    signs = signs - signs.mean()
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                BS[:, idx] = 0.9 * signs[a] * nvec
            idx += 1
    Y = normalize_rows(x0[None, :] + U @ J.T + Phi @ BS.T)
    return Y, U, BS


def run_decomposition_synthetics(seed: int = 0) -> dict:
    rows = []
    for name, maker, expect in [
        ("normalized_affine", synth_affine_sphere, "near_zero"),
        ("isotropic_sphere_normal", synth_isotropic_normal, "mean"),
        ("saddle_sphere_normal", synth_saddle_normal, "traceless"),
    ]:
        X, Utrue, _ = maker(1000, 4, 32, seed)
        w = np.ones(len(X))
        idx = np.arange(len(X))
        rng = np.random.default_rng(seed + hash(name) % 1000)
        rng.shuffle(idx)
        g, f, v, te = np.split(idx, [250, 600, 800])
        chart, chart_RS, info, U = fit_nested_chart(X, Utrue, w, g, f, v)
        dec = decompose_BS(chart.BS_flat, chart.J.shape[1])
        err = chart_errors(chart, chart_RS, X, U, w, te)
        rows.append({"synth": name, "expect": expect, **dec, **err, **info})

    checks = {
        "affine_near_zero": rows[0]["B_fro"] < 0.15 and rows[0]["dS"] < 0.002,
        "isotropic_mean_dom": rows[1]["mean_frac"] > 0.55 and rows[1]["H_norm"] > 0.05,
        "saddle_traceless_dom": rows[2]["traceless_frac"] > 0.55 and rows[2]["B_traceless_fro"] > 0.05,
        "identity_ok": all(r["identity_residual"] < 1e-6 for r in rows),
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


# -------------------- confirmatory --------------------


@dataclass
class ConfirmatoryConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_quadratic_atlas_sphere_normal"
    structure_dir: str = "outputs/geometry/physics_quadratic_atlas_structure"
    ablation_prepare: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    n_charts: int = 6
    local_dim: int = 8
    atlas_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_bootstrap: int = 100
    n_null: int = 100
    n_anchors: int = 384
    knn_scales: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    n_feature_bootstrap: int = 4
    charts_per_sample: int = 3
    min_chart_samples: int = 40
    seed: int = 0
    force: bool = False
    max_seconds: float = 7200.0

    @property
    def primary(self) -> str:
        return f"n{self.n_charts}_d{self.local_dim}"

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def config_hash(self) -> str:
        payload = {
            "n_charts": self.n_charts,
            "local_dim": self.local_dim,
            "atlas_seeds": self.atlas_seeds,
            "n_bootstrap": self.n_bootstrap,
            "n_null": self.n_null,
            "n_anchors": self.n_anchors,
            "knn_scales": self.knn_scales,
            "n_feature_bootstrap": self.n_feature_bootstrap,
            "charts_per_sample": self.charts_per_sample,
            "min_chart_samples": self.min_chart_samples,
            "ablation_prepare": self.ablation_prepare,
            "structure_dir": self.structure_dir,
            "protocol": "sphere_normal_nested_v1",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _sn_cfg(cfg: ConfirmatoryConfig, seed: int) -> SphereNormalConfig:
    return SphereNormalConfig(
        output_dir=cfg.output_dir,
        structure_dir=cfg.structure_dir,
        ablation_prepare=cfg.ablation_prepare,
        primary=cfg.primary,
        configs=[cfg.primary],
        n_bootstrap=cfg.n_bootstrap,
        n_null=cfg.n_null,
        seed=seed,
        force=cfg.force,
        max_seconds=cfg.max_seconds,
    )


def ensure_seed_memberships(
    root: Path, cfg: ConfirmatoryConfig, atlas_seed: int
) -> Path:
    """Build or reuse chart memberships for (n_charts, seed)."""
    out = cfg.resolved_out(root) / "confirmatory" / "memberships" / f"seed{atlas_seed}"
    marker = out / "memberships_csr.npz"
    if _done(marker, cfg.force):
        return out
    out.mkdir(parents=True, exist_ok=True)
    # seed 0: prefer structure cache
    if atlas_seed == 0:
        src = resolve_path(root, cfg.structure_dir) / "grid" / cfg.primary / "memberships_csr.npz"
        if src.exists():
            W = sparse.load_npz(src)
            sparse.save_npz(marker, W)
            centres_src = resolve_path(root, cfg.structure_dir) / "grid" / cfg.primary / "centres.npz"
            if centres_src.exists():
                import shutil

                shutil.copy(centres_src, out / "centres.npz")
            (out / "provenance.json").write_text(
                json.dumps({"source": str(src), "atlas_seed": atlas_seed}, indent=2)
            )
            return out
    data = load_prepare(resolve_path(root, cfg.ablation_prepare))
    scfg = StructureConfig(
        seed=atlas_seed,
        charts_per_sample=cfg.charts_per_sample,
        min_chart_samples=cfg.min_chart_samples,
    )
    W, centres, meta = build_charts(data["X"], data["train_local"], cfg.n_charts, scfg)
    sparse.save_npz(marker, W)
    np.savez_compressed(out / "centres.npz", centres=centres)
    (out / "provenance.json").write_text(json.dumps({"meta": meta, "atlas_seed": atlas_seed}, indent=2))
    return out


def _split_train(tr_all: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    tr = np.asarray(tr_all, dtype=np.int64).copy()
    rng.shuffle(tr)
    n = len(tr)
    n_g = max(20, int(0.4 * n))
    n_f = max(20, int(0.4 * n))
    idx_geom, idx_fit, idx_val = tr[:n_g], tr[n_g : n_g + n_f], tr[n_g + n_f :]
    if len(idx_val) < 10:
        idx_val = idx_fit.copy()
    return idx_geom, idx_fit, idx_val


def _refit_quadratic_fixed_geom(
    X: np.ndarray,
    U: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    sc: np.ndarray,
    f_idx: np.ndarray,
    v_idx: np.ndarray,
    te_c: np.ndarray,
    lam_A: float,
    lam_BS: float,
    lam_BSR: float,
) -> dict:
    """Refit A/B^S with fixed (x0,J); evaluate Δ_S on te_c. Fast bootstrap path."""
    Phi_f = quadratic_features(U[f_idx])
    L_f = x0[None, :] + U[f_idx] @ J.T
    scale = np.linalg.norm(L_f, axis=1, keepdims=True)
    target_un = X[f_idx] * np.maximum(scale, 1e-8)
    tang_res = (target_un - L_f) @ J
    A = _ridge_solve(Phi_f, tang_res, w[f_idx], lam_A)
    L_tr = x0[None, :] + (U[f_idx] + Phi_f @ A.T) @ J.T
    scale_tr = np.linalg.norm(L_tr, axis=1, keepdims=True)
    target_tr = X[f_idx] * np.maximum(scale_tr, 1e-8)
    resid_n = normal_projector_apply((target_tr - L_tr).T, x0, J).T
    BS = normal_projector_apply(_ridge_solve(Phi_f, resid_n, w[f_idx], lam_BS), x0, J)
    resid_r = normal_projector_apply((target_un - L_f).T, x0, J).T
    BS_R = normal_projector_apply(_ridge_solve(Phi_f, resid_r, w[f_idx], lam_BSR), x0, J)
    chart = NestedChart(x0, J, A, BS, lam_A, lam_BS, sc)
    chart_RS = NestedChart(x0, J, np.zeros_like(A), BS_R, lam_A, lam_BSR, sc)
    return chart_errors(chart, chart_RS, X, U, w, te_c)


def prepare_fixed_geom_cache(
    X: np.ndarray,
    W: sparse.csr_matrix,
    data: dict,
    cfg: ConfirmatoryConfig,
    atlas_seed: int,
) -> tuple[dict, list[dict]]:
    """Primary fit + per-chart fixed geometry for quadratic bootstraps."""
    idx_geom, idx_fit, idx_val = _split_train(data["train_local"], atlas_seed)
    te = data["test_local"]
    d = cfg.local_dim
    chart_rows, rank_rows, BS_list, cache = [], [], [], []
    for c in range(W.shape[1]):
        w = np.asarray(W[:, c].todense()).ravel()

        def memb(idx, _w=w):
            return idx[_w[idx] > 1e-6]

        g, f, v = memb(idx_geom), memb(idx_fit), memb(idx_val)
        te_c = memb(te)
        if len(g) < d + 5 or len(f) < 20:
            continue
        chart, chart_RS, info, U = fit_nested_chart(
            X, np.zeros((len(X), d)), w, g, f, v
        )
        err = chart_errors(chart, chart_RS, X, U, w, te_c)
        sc = chart.coord_scale
        dd = chart.J.shape[1]
        feat_scale = np.array([sc[a] * sc[b] for a in range(dd) for b in range(a, dd)])
        BS_svd = flatten_BS_for_svd(chart.BS_flat * feat_scale[None, :], dd)
        stats = matrix_rank_stats(BS_svd)
        curve = predictive_rank_curve(chart, X, U, w, te_c)
        r90 = "full"
        for row in curve:
            if row["rank"] != "full" and row["frac_of_full_dS"] >= 0.9:
                r90 = row["rank"]
                break
        dec = decompose_BS(chart.BS_flat, dd)
        sJ = np.linalg.svd(chart.J, compute_uv=False)
        tcond = float(sJ[0] / max(sJ[-1], 1e-12)) if len(sJ) else float("nan")
        chart_rows.append(
            {
                "atlas_seed": atlas_seed,
                "chart": c,
                **err,
                **info,
                **dec,
                "BS_stable_rank": stats["stable_rank"],
                "BS_entropy_rank": stats["entropy_rank"],
                "BS_participation_ratio": stats["participation_ratio"],
                "BS_rank95": stats["rank95"],
                "pred_rank_90pct_dS": r90,
                "tangent_condition": tcond,
                "centre_norm": float(np.linalg.norm(chart.x0)),
            }
        )
        for row in curve:
            rank_rows.append({**row, "atlas_seed": atlas_seed, "chart": c})
        BS_list.append(chart.BS_flat)
        cache.append(
            {
                "c": c,
                "w": w,
                "f": f,
                "v": v,
                "te_c": te_c,
                "U": U,
                "x0": chart.x0,
                "J": chart.J,
                "sc": sc,
                "lam_A": float(info["ridge_A"]),
                "lam_BS": float(info["ridge_BS"]),
                "lam_BSR": float(info["ridge_BS_R"]),
            }
        )
    dS = np.array([r["dS"] for r in chart_rows], dtype=np.float64)
    res = {
        "atlas_seed": atlas_seed,
        "charts": chart_rows,
        "rank_curve": rank_rows,
        "BS_list": BS_list,
        "mean_dS": float(np.nanmean(dS)) if len(dS) else float("nan"),
        "median_dS": float(np.nanmedian(dS)) if len(dS) else float("nan"),
        "frac_charts_dS_pos": float(np.mean(dS > 0)) if len(dS) else float("nan"),
        "mean_dT": float(np.nanmean([r["dT"] for r in chart_rows])) if chart_rows else float("nan"),
        "mean_dSR": float(np.nanmean([r["dSR"] for r in chart_rows])) if chart_rows else float("nan"),
        "n_charts_fit": len(chart_rows),
    }
    return res, cache


def bootstrap_delta_S_fixed_geom(
    X: np.ndarray, cache: list[dict], n_bootstrap: int, atlas_seed: int
) -> list[dict]:
    """Bootstrap quadratic refits with fixed chart geometry (x0,J)."""
    rows = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(atlas_seed * 1000 + 1000 + b)
        dS_charts = []
        for cc in cache:
            f = cc["f"]
            if len(f) < 20:
                continue
            f_b = rng.choice(f, size=len(f), replace=True)
            err = _refit_quadratic_fixed_geom(
                X,
                cc["U"],
                cc["w"],
                cc["x0"],
                cc["J"],
                cc["sc"],
                f_b,
                cc["v"],
                cc["te_c"],
                cc["lam_A"],
                cc["lam_BS"],
                cc["lam_BSR"],
            )
            dS_charts.append(err["dS"])
        rows.append(
            {
                "atlas_seed": atlas_seed,
                "bootstrap": b,
                "mean_dS": float(np.nanmean(dS_charts)) if dS_charts else float("nan"),
                "frac_charts_dS_pos": float(np.mean(np.asarray(dS_charts) > 0)) if dS_charts else float("nan"),
            }
        )
    return rows


def fit_config_with_W(
    X: np.ndarray,
    W: sparse.csr_matrix,
    data: dict,
    cfg: ConfirmatoryConfig,
    atlas_seed: int,
) -> dict:
    res, _ = prepare_fixed_geom_cache(X, W, data, cfg, atlas_seed)
    return res


def _nulls_for_W(
    X: np.ndarray,
    W: sparse.csr_matrix,
    data: dict,
    cfg: ConfirmatoryConfig,
    atlas_seed: int,
    real_dS: float,
    t0: float,
) -> dict:
    tr = np.asarray(data["train_local"]).copy()
    te = data["test_local"]
    rng0 = np.random.default_rng(atlas_seed + 7)
    rng0.shuffle(tr)
    n = len(tr)
    idx_geom = tr[: int(0.4 * n)]
    idx_fit = tr[int(0.4 * n) : int(0.8 * n)]
    idx_val = tr[int(0.8 * n) :]
    d = cfg.local_dim
    chart_cache = []
    for c in range(W.shape[1]):
        w = np.asarray(W[:, c].todense()).ravel()

        def memb(idx, _w=w):
            return idx[_w[idx] > 1e-6]

        g, f, v = memb(idx_geom), memb(idx_fit), memb(idx_val)
        te_c = memb(te)
        if len(f) < 30 or len(te_c) < 10:
            continue
        chart0, _, _, U = fit_nested_chart(X, np.zeros((len(X), d)), w, g, f, v)
        Phi = quadratic_features(U[f])
        L_f = chart0.x0[None, :] + U[f] @ chart0.J.T
        scale = np.linalg.norm(L_f, axis=1, keepdims=True)
        target_un = X[f] * np.maximum(scale, 1e-8)
        tang_res = (target_un - L_f) @ chart0.J
        chart_cache.append(
            {
                "w": w,
                "f": f,
                "te_c": te_c,
                "chart0": chart0,
                "U": U,
                "Phi": Phi,
                "tang_res": tang_res,
                "target_un": target_un,
                "L_f": L_f,
            }
        )
    null_shuffle, null_random = [], []
    for nrep in range(cfg.n_null):
        _budget(t0, cfg.max_seconds, f"null_s{atlas_seed}_{nrep}")
        rng = np.random.default_rng(atlas_seed * 10007 + 9000 + nrep)
        dS_s, dS_r = [], []
        for cc in chart_cache:
            x0, J = cc["chart0"].x0, cc["chart0"].J
            f, te_c, w, U = cc["f"], cc["te_c"], cc["w"], cc["U"]
            Phi_s = cc["Phi"][rng.permutation(len(cc["Phi"]))]
            A_s, lamA, _ = _ridge_multi(Phi_s, cc["tang_res"], w[f], [1e-2, 1e-1, 1.0])
            L_tr = x0[None, :] + (U[f] + Phi_s @ A_s.T) @ J.T
            resid_n = normal_projector_apply(
                (X[f] * np.maximum(np.linalg.norm(L_tr, axis=1, keepdims=True), 1e-8) - L_tr).T,
                x0,
                J,
            ).T
            BS_s, lamB, _ = _ridge_multi(Phi_s, resid_n, w[f], [1e-2, 1e-1, 1.0])
            BS_s = normal_projector_apply(BS_s, x0, J)
            ch_s = NestedChart(x0, J, A_s, BS_s, lamA, lamB, cc["chart0"].coord_scale)
            dS_s.append(chart_errors(ch_s, ch_s, X, U, w, te_c)["dS"])
            R = rng.normal(size=(d, d))
            Phi_r = quadratic_features((U @ R)[f])
            BS_r, lamB, _ = _ridge_multi(
                Phi_r,
                normal_projector_apply((cc["target_un"] - cc["L_f"]).T, x0, J).T,
                w[f],
                [1e-2, 1e-1, 1.0],
            )
            BS_r = normal_projector_apply(BS_r, x0, J)
            Phi_te = quadratic_features((U @ R)[te_c])
            pred = normalize_rows(x0[None, :] + U[te_c] @ J.T + Phi_te @ BS_r.T)
            E_TRS = weighted_mse(pred.astype(np.float32), X[te_c].astype(np.float32), w[te_c])
            E_TR = weighted_mse(
                cc["chart0"].decode_TR(U[te_c]).astype(np.float32),
                X[te_c].astype(np.float32),
                w[te_c],
            )
            dS_r.append(float(E_TR - E_TRS))
        null_shuffle.append(float(np.nanmean(dS_s)) if dS_s else float("nan"))
        null_random.append(float(np.nanmean(dS_r)) if dS_r else float("nan"))
        if (nrep + 1) % 20 == 0:
            print(f"[confirm] seed={atlas_seed} null {nrep+1}/{cfg.n_null}", flush=True)
    p_s, B_s = mc_pvalue(real_dS, np.asarray(null_shuffle))
    p_r, B_r = mc_pvalue(real_dS, np.asarray(null_random))
    return {
        "p_shuffle": p_s,
        "p_random_features": p_r,
        "B_null": int(B_s),
        "null_shuffle": null_shuffle,
        "null_random": null_random,
    }


def stage_confirmatory(root: Path, cfg: ConfirmatoryConfig, t0: float) -> dict:
    out = cfg.resolved_out(root) / "confirmatory"
    marker = out / "confirmatory_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepare(resolve_path(root, cfg.ablation_prepare))
    X = data["X"]
    seed_rows = []
    all_charts = []
    all_rank = []
    boot_rows = []
    null_tables = []

    for atlas_seed in cfg.atlas_seeds:
        _budget(t0, cfg.max_seconds, f"confirm_seed{atlas_seed}")
        mdir = ensure_seed_memberships(root, cfg, atlas_seed)
        W = sparse.load_npz(mdir / "memberships_csr.npz")
        print(f"[confirm] atlas_seed={atlas_seed} fit", flush=True)
        res, cache = prepare_fixed_geom_cache(X, W, data, cfg, atlas_seed)
        # chart-centre sensitivity: alternate FPS centres (seed+100)
        scfg = StructureConfig(
            seed=atlas_seed + 100,
            charts_per_sample=cfg.charts_per_sample,
            min_chart_samples=cfg.min_chart_samples,
        )
        W_alt, _, _ = build_charts(X, data["train_local"], cfg.n_charts, scfg)
        res_alt = fit_config_with_W(X, W_alt, data, cfg, atlas_seed + 100)
        centre_sens = float(res["mean_dS"] - res_alt["mean_dS"])

        print(f"[confirm] seed={atlas_seed} quadratic bootstrap n={cfg.n_bootstrap}", flush=True)
        boot_part = bootstrap_delta_S_fixed_geom(X, cache, cfg.n_bootstrap, atlas_seed)
        boot_rows.extend(boot_part)
        boot_dS = np.asarray([r["mean_dS"] for r in boot_part], dtype=np.float64)
        nulls = _nulls_for_W(X, W, data, cfg, atlas_seed, res["mean_dS"], t0)
        seed_rows.append(
            {
                "atlas_seed": atlas_seed,
                "mean_dS": res["mean_dS"],
                "median_dS": res["median_dS"],
                "mean_dT": res["mean_dT"],
                "mean_dSR": res["mean_dSR"],
                "frac_charts_dS_pos": res["frac_charts_dS_pos"],
                "n_charts_fit": res["n_charts_fit"],
                "bootstrap_mean": float(np.nanmean(boot_dS)),
                "bootstrap_ci95": [
                    float(np.nanquantile(boot_dS, 0.025)),
                    float(np.nanquantile(boot_dS, 0.975)),
                ],
                "frac_boot_dS_pos": float(np.mean(boot_dS > 0)),
                "p_shuffle": nulls["p_shuffle"],
                "p_random_features": nulls["p_random_features"],
                "B_null": nulls["B_null"],
                "B_bootstrap": int(len(boot_dS)),
                "centre_sensitivity_dS": centre_sens,
                "alt_seed_mean_dS": res_alt["mean_dS"],
            }
        )
        all_charts.extend(res["charts"])
        all_rank.extend(res["rank_curve"])
        null_tables.append(
            {
                "atlas_seed": atlas_seed,
                "null_shuffle": nulls["null_shuffle"],
                "null_random": nulls["null_random"],
            }
        )
        print(
            f"[confirm] seed={atlas_seed} dS={res['mean_dS']:.5f} "
            f"p_s={nulls['p_shuffle']:.4f} boot+={np.mean(boot_dS>0):.2f}",
            flush=True,
        )

    seed_df = pd.DataFrame(seed_rows)
    pooled_dS = float(seed_df.mean_dS.mean())
    # pooled null: concatenate nulls and compare to pooled mean? Use mean of per-seed p / or pool
    # Corrected MC on pooled statistic: use average of seed null means vs pooled real
    pooled_null_s = np.concatenate([np.asarray(t["null_shuffle"]) for t in null_tables])
    pooled_null_r = np.concatenate([np.asarray(t["null_random"]) for t in null_tables])
    p_pool_s, B_ps = mc_pvalue(pooled_dS, pooled_null_s)
    p_pool_r, B_pr = mc_pvalue(pooled_dS, pooled_null_r)
    stable = bool(
        (seed_df.mean_dS > 0).all()
        and (seed_df.frac_charts_dS_pos >= 0.8).all()
        and (seed_df.frac_boot_dS_pos >= 0.8).all()
        and (seed_df.p_shuffle <= 0.05).all()
    )
    summary = {
        "primary": cfg.primary,
        "seeds": seed_rows,
        "pooled": {
            "mean_dS": pooled_dS,
            "std_dS_across_seeds": float(seed_df.mean_dS.std(ddof=0)),
            "min_dS": float(seed_df.mean_dS.min()),
            "max_dS": float(seed_df.mean_dS.max()),
            "p_shuffle": p_pool_s,
            "p_random_features": p_pool_r,
            "B_null_pooled": int(B_ps),
            "frac_seeds_dS_pos": float((seed_df.mean_dS > 0).mean()),
            "mean_centre_sensitivity_dS": float(seed_df.centre_sensitivity_dS.mean()),
        },
        "stable_across_seeds": stable,
        "config_hash": cfg.config_hash(),
    }
    out.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out / "delta_S_by_seed.csv", index=False)
    pd.DataFrame(all_charts).to_parquet(out / "charts_by_seed.parquet", index=False)
    pd.DataFrame(all_rank).to_parquet(out / "predictive_rank_by_seed.parquet", index=False)
    pd.DataFrame(boot_rows).to_parquet(out / "bootstrap_by_seed.parquet", index=False)
    (out / "confirmatory_summary.json").write_text(json.dumps(summary, indent=2))
    if not stable:
        (out / "STOP_UNSTABLE.txt").write_text(
            "Δ_S not stable across atlas seeds; object-feature freeze aborted.\n"
            + json.dumps(summary["pooled"], indent=2)
        )
    return summary


# -------------------- object-level features --------------------


def select_anchors(data: dict, n_anchors: int) -> np.ndarray:
    """Deterministic held-out anchors ordered by stable sample_id."""
    hold = np.asarray(data["holdout_local"], dtype=np.int64)
    sids = data["sample_ids"][hold]
    order = np.argsort(sids, kind="mergesort")
    hold = hold[order]
    n = min(n_anchors, len(hold))
    return hold[:n]


def _fit_neighborhood(
    X: np.ndarray,
    neigh_idx: np.ndarray,
    d: int,
    seed: int,
) -> tuple[NestedChart | None, NestedChart | None, dict, np.ndarray, np.ndarray, str]:
    """
    Cross-fit nested model on a neighborhood (subsetted X for speed).
    Returns (chart, chart_RS, info, U_local, local_global_idx, reason).
    """
    if len(neigh_idx) < max(40, 5 * d):
        return None, None, {}, np.zeros((0, d)), np.zeros(0, dtype=np.int64), "too_few_neighbors"
    rng = np.random.default_rng(seed)
    glob = np.asarray(neigh_idx, dtype=np.int64).copy()
    rng.shuffle(glob)
    n = len(glob)
    n_g = max(15, int(0.4 * n))
    n_f = max(15, int(0.4 * n))
    # work in local coordinates 0..n-1
    Xloc = X[glob].astype(np.float64)
    g = np.arange(0, n_g)
    f = np.arange(n_g, n_g + n_f)
    v = np.arange(n_g + n_f, n)
    if len(v) < 8:
        return None, None, {}, np.zeros((0, d)), glob, "too_few_val"
    w = np.ones(n, dtype=np.float64)
    try:
        chart, chart_RS, info, U = fit_nested_chart(
            Xloc, np.zeros((n, d)), w, g, f, v
        )
    except Exception as e:  # noqa: BLE001
        return None, None, {}, np.zeros((0, d)), glob, f"fit_error:{type(e).__name__}"
    if chart.J.shape[1] < d:
        return None, None, {}, U, glob, "rank_deficient_tangent"
    return chart, chart_RS, info, U, glob, ""


def features_from_chart(
    chart: NestedChart,
    chart_RS: NestedChart,
    X: np.ndarray,
    U: np.ndarray,
    eval_idx: np.ndarray,
    rho: float,
) -> dict:
    d = chart.J.shape[1]
    w = np.ones(len(eval_idx))
    err = chart_errors(chart, chart_RS, X, U, np.ones(len(X)), eval_idx)
    dec = decompose_BS(chart.BS_flat, d)
    sc = chart.coord_scale
    feat_scale = np.array([sc[a] * sc[b] for a in range(d) for b in range(a, d)])
    BS_svd = flatten_BS_for_svd(chart.BS_flat * feat_scale[None, :], d)
    stats = matrix_rank_stats(BS_svd)
    curve = predictive_rank_curve(chart, X, U, np.ones(len(X)), eval_idx)
    r90 = "full"
    for row in curve:
        if row["rank"] != "full" and row.get("frac_of_full_dS", 0) >= 0.9:
            r90 = row["rank"]
            break
    sJ = np.linalg.svd(chart.J, compute_uv=False)
    tcond = float(sJ[0] / max(sJ[-1], 1e-12)) if len(sJ) else float("nan")
    n_eff = float(len(eval_idx))  # uniform weights in neighborhood fit; report fit size separately
    return {
        "B_fro": dec["B_fro"],
        "H_norm": dec["H_norm"],
        "B_traceless_fro": dec["B_traceless_fro"],
        "rho_times_B_fro": rho * dec["B_fro"],
        "rho_times_H_norm": rho * dec["H_norm"],
        "rho_times_B_traceless_fro": rho * dec["B_traceless_fro"],
        "stable_rank": stats["stable_rank"],
        "participation_rank": stats["participation_ratio"],
        "entropy_rank": stats["entropy_rank"],
        "rank95": stats["rank95"],
        "predictive_rank_90": r90,
        "delta_s": err["dS"],
        "reconstruction_error": err["E_TRS"],
        "knn_radius": rho,
        "tangent_condition": tcond,
        "identity_residual": dec["identity_residual"],
        "mean_frac": dec["mean_frac"],
        "traceless_frac": dec["traceless_frac"],
        "E_TR": err["E_TR"],
        "E_R": err["E_R"],
    }


def stage_object_features(root: Path, cfg: ConfirmatoryConfig, t0: float, confirm: dict) -> dict:
    out = cfg.resolved_out(root)
    raw_path = out / "object_curvature_features.parquet"
    agg_path = out / "object_curvature_features_aggregated.parquet"
    meta_path = out / "object_curvature_features_meta.json"
    if _done(raw_path, cfg.force) and _done(agg_path, cfg.force):
        return json.loads(meta_path.read_text()) if meta_path.exists() else {"cached": True}
    if not confirm.get("stable_across_seeds", False):
        meta = {
            "frozen": False,
            "reason": "confirmatory_unstable",
            "confirm_pooled": confirm.get("pooled"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    data = load_prepare(resolve_path(root, cfg.ablation_prepare))
    X = data["X"].astype(np.float64)
    train = data["train_local"]
    anchors = select_anchors(data, cfg.n_anchors)
    k_max = max(cfg.knn_scales)
    print(f"[object] knn fit on {len(train)} train, {len(anchors)} anchors, k_max={k_max}", flush=True)
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric="euclidean")
    nn.fit(X[train])
    dists, inds = nn.kneighbors(X[anchors], return_distance=True)
    # inds are into train array
    train_arr = np.asarray(train)

    rows = []
    for ai, a_local in enumerate(anchors):
        _budget(t0, cfg.max_seconds, f"anchor{ai}")
        sample_id = int(data["sample_ids"][a_local])
        for k in cfg.knn_scales:
            rho = float(dists[ai, k - 1])
            neigh = train_arr[inds[ai, :k]]
            # exclude accidental self
            neigh = neigh[neigh != a_local]
            if len(neigh) < k // 2:
                rows.append(
                    _fail_row(sample_id, a_local, k, rho, cfg, "neighbors_collapsed")
                )
                continue
            # primary fit
            seed_fit = cfg.seed + 17 * ai + k
            chart, chart_RS, info, Uloc, glob, reason = _fit_neighborhood(
                X, neigh, cfg.local_dim, seed=seed_fit
            )
            if chart is None:
                rows.append(_fail_row(sample_id, a_local, k, rho, cfg, reason or "fit_failed"))
                continue
            n = len(glob)
            v_local = np.arange(int(0.8 * n), n)
            if len(v_local) < 8:
                v_local = np.arange(max(0, n - max(8, n // 10)), n)
            feat = features_from_chart(chart, chart_RS, X[glob], Uloc, v_local, rho)
            n_eff = float(len(neigh))
            boot_Bf = []
            for b in range(cfg.n_feature_bootstrap):
                rngb = np.random.default_rng(cfg.seed + 10007 * ai + 97 * k + b)
                nb = rngb.choice(neigh, size=len(neigh), replace=True)
                ch_b, _, _, _, _, _ = _fit_neighborhood(
                    X, nb, cfg.local_dim, seed=cfg.seed + 10007 * ai + 97 * k + b + 3
                )
                if ch_b is None:
                    continue
                boot_Bf.append(decompose_BS(ch_b.BS_flat, ch_b.J.shape[1])["B_fro"])
            boot_std = float(np.std(boot_Bf)) if boot_Bf else float("nan")
            valid = True
            failure = ""
            if feat["identity_residual"] > 1e-5:
                valid, failure = False, "identity_failed"
            if not np.isfinite(feat["delta_s"]):
                valid, failure = False, "delta_s_nan"
            if feat["tangent_condition"] > 1e6:
                valid, failure = False, "ill_conditioned_tangent"
            if n_eff < 40:
                valid, failure = False, "n_eff_low"
            rows.append(
                {
                    "sample_id": sample_id,
                    "local_index": int(a_local),
                    "scale_k": int(k),
                    "seed": int(cfg.seed),
                    "atlas_seed": int(cfg.atlas_seeds[0]),
                    "n_eff": n_eff,
                    "bootstrap_std": boot_std,
                    "valid": bool(valid),
                    "failure_reason": failure,
                    "config_hash": cfg.config_hash(),
                    "prepare_path": cfg.ablation_prepare,
                    "split": "holdout_local",
                    "n_charts": cfg.n_charts,
                    "local_dim": cfg.local_dim,
                    "ridge_A": info.get("ridge_A", float("nan")),
                    "ridge_BS": info.get("ridge_BS", float("nan")),
                    **feat,
                }
            )
        if (ai + 1) % 32 == 0:
            print(f"[object] anchors {ai+1}/{len(anchors)} rss={_rss():.0f}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_parquet(raw_path, index=False)

    # Aggregate: one row per sample_id × scale (mean over raw fits; here one seed)
    feat_cols = [
        "B_fro",
        "H_norm",
        "B_traceless_fro",
        "rho_times_B_fro",
        "rho_times_H_norm",
        "rho_times_B_traceless_fro",
        "stable_rank",
        "participation_rank",
        "entropy_rank",
        "rank95",
        "delta_s",
        "reconstruction_error",
        "knn_radius",
        "n_eff",
        "tangent_condition",
        "bootstrap_std",
        "mean_frac",
        "traceless_frac",
    ]
    agg_rows = []
    for (sid, k), g in raw.groupby(["sample_id", "scale_k"]):
        g_ok = g[g["valid"]]
        src = g_ok if len(g_ok) else g
        row = {
            "sample_id": int(sid),
            "scale_k": int(k),
            "n_estimates": int(len(g)),
            "n_valid": int(len(g_ok)),
            "valid": bool(len(g_ok) > 0),
            "config_hash": cfg.config_hash(),
            "predictive_rank_90_mode": (
                src["predictive_rank_90"].astype(str).mode().iloc[0]
                if len(src)
                else "nan"
            ),
        }
        for c in feat_cols:
            if c in src.columns:
                row[c] = float(src[c].mean()) if len(src) else float("nan")
                row[f"{c}_std"] = float(src[c].std(ddof=0)) if len(src) > 1 else 0.0
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_parquet(agg_path, index=False)

    # also copy aggregated to the canonical frozen name requested? User asked for
    # object_curvature_features.parquet as the frozen artifact — keep raw there,
    # and write aggregated alongside. Document both.
    # User: "Write: .../object_curvature_features.parquet" with sample_id, scale, seed, features
    # and "Also write a final aggregated Parquet with one row per sample_id × scale."
    # Raw matches the first; aggregated is the second.

    valid_frac = (
        raw.groupby("scale_k")["valid"].mean().to_dict() if len(raw) else {}
    )
    meta = {
        "frozen": True,
        "raw_path": str(raw_path),
        "aggregated_path": str(agg_path),
        "config_hash": cfg.config_hash(),
        "n_anchors": int(len(anchors)),
        "n_rows_raw": int(len(raw)),
        "n_rows_agg": int(len(agg)),
        "valid_frac_by_scale": {str(k): float(v) for k, v in valid_frac.items()},
        "knn_scales": cfg.knn_scales,
        "provenance": {
            "prepare": cfg.ablation_prepare,
            "split": "holdout_local",
            "selection": "first_n_by_sorted_sample_id",
            "labels_used": False,
        },
        "ready_for_retrieval": True,
        "note": "Frozen: do not alter curvature hyperparameters or validity gates for later retrieval.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    # freeze stamp
    (out / "OBJECT_FEATURES_FROZEN.txt").write_text(
        f"config_hash={cfg.config_hash()}\nraw={raw_path}\nagg={agg_path}\n"
    )
    return meta


def _fail_row(sample_id, local_index, k, rho, cfg: ConfirmatoryConfig, reason: str) -> dict:
    return {
        "sample_id": int(sample_id),
        "local_index": int(local_index),
        "scale_k": int(k),
        "seed": int(cfg.seed),
        "atlas_seed": int(cfg.atlas_seeds[0]),
        "B_fro": float("nan"),
        "H_norm": float("nan"),
        "B_traceless_fro": float("nan"),
        "rho_times_B_fro": float("nan"),
        "rho_times_H_norm": float("nan"),
        "rho_times_B_traceless_fro": float("nan"),
        "stable_rank": float("nan"),
        "participation_rank": float("nan"),
        "entropy_rank": float("nan"),
        "rank95": float("nan"),
        "predictive_rank_90": "nan",
        "delta_s": float("nan"),
        "reconstruction_error": float("nan"),
        "knn_radius": float(rho) if np.isfinite(rho) else float("nan"),
        "n_eff": float("nan"),
        "tangent_condition": float("nan"),
        "bootstrap_std": float("nan"),
        "valid": False,
        "failure_reason": reason,
        "config_hash": cfg.config_hash(),
        "prepare_path": cfg.ablation_prepare,
        "split": "holdout_local",
        "n_charts": cfg.n_charts,
        "local_dim": cfg.local_dim,
        "identity_residual": float("nan"),
        "mean_frac": float("nan"),
        "traceless_frac": float("nan"),
        "E_TR": float("nan"),
        "E_R": float("nan"),
        "ridge_A": float("nan"),
        "ridge_BS": float("nan"),
    }


def stage_report(root: Path, cfg: ConfirmatoryConfig, confirm: dict, obj_meta: dict, synth: dict) -> dict:
    out = cfg.resolved_out(root)
    seeds = pd.DataFrame(confirm["seeds"])
    pooled = confirm["pooled"]
    raw_path = out / "object_curvature_features.parquet"
    agg_path = out / "object_curvature_features_aggregated.parquet"

    # distributions
    dist_txt = "object features not frozen"
    scale_txt = ""
    pred_rank_txt = ""
    mean_vs_trace = ""
    valid_frac = {}
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
        ok = raw[raw.valid]
        valid_frac = raw.groupby("scale_k")["valid"].mean().to_dict()
        if len(ok):
            dist_txt = (
                f"B_fro median={ok.B_fro.median():.4g}  "
                f"H_norm median={ok.H_norm.median():.4g}  "
                f"B°_fro median={ok.B_traceless_fro.median():.4g}"
            )
            scale_txt = ok.groupby("scale_k")[["B_fro", "H_norm", "B_traceless_fro", "delta_s", "knn_radius"]].median().to_string()
            pred_rank_txt = ok.groupby(["scale_k", "predictive_rank_90"]).size().to_string()
            mean_vs_trace = (
                f"mean_frac median={ok.mean_frac.median():.3f}  "
                f"traceless_frac median={ok.traceless_frac.median():.3f}  "
                f"mostly={'mean' if ok.mean_frac.median() > ok.traceless_frac.median() else 'traceless'}"
            )

    ready = bool(obj_meta.get("ready_for_retrieval")) and bool(confirm.get("stable_across_seeds"))
    report = f"""# Sphere-normal confirmatory + object curvature features

## 1. Confirmatory Δ_S across atlas seeds

Primary `{cfg.primary}`, bootstraps={cfg.n_bootstrap}, nulls={cfg.n_null}.

{seeds.to_string(index=False)}

**Pooled:** mean Δ_S={pooled['mean_dS']:.6f} (seed std={pooled['std_dS_across_seeds']:.6f})
corrected MC p(shuffle)={pooled['p_shuffle']:.4f} (B_pooled={pooled['B_null_pooled']})
p(random)={pooled['p_random_features']:.4f}
stable_across_seeds={confirm['stable_across_seeds']}
mean |centre-sensitivity Δ_S|={abs(pooled['mean_centre_sensitivity_dS']):.6f}

## 2. Corrected null results

Per-seed p_shuffle / p_random in the table above. Never report p=0; formula (1+#{{T_null≥T_real}})/(B+1).

## 3. Valid anchors by scale

{json.dumps({str(k): float(v) for k, v in valid_frac.items()}, indent=2)}

## 4. Distributions (valid rows)

{dist_txt}

## 5. Scale dependence (median by k)

{scale_txt}

## 6. Bootstrap / seed stability

Feature bootstrap_std column in raw parquet; confirmatory seed table above.

## 7. Predictive-rank distribution

{pred_rank_txt}

## 8. Mean vs traceless

{mean_vs_trace}

Decomposition synthetics pass={synth.get('pass')} checks={json.dumps(synth.get('checks'))}

## 9. Frozen artifact paths

- raw (per estimate): `{raw_path}`
- aggregated (sample_id × scale): `{agg_path}`
- config_hash: `{cfg.config_hash()}`

## 10. Ready for separately specified retrieval?

**{ready}**

Labels / Fisher / JS / retrieval were not used. Do not alter curvature hyperparameters or validity gates after freeze.

## Exact next command (not run)

```bash
# retrieval must be specified separately; do not launch from this report
```
"""
    (out / "CONFIRMATORY_OBJECT_REPORT.md").write_text(report)
    analysis = {
        "stable_across_seeds": confirm.get("stable_across_seeds"),
        "pooled": pooled,
        "seeds": confirm.get("seeds"),
        "object_meta": obj_meta,
        "synthetic_decomp_pass": synth.get("pass"),
        "ready_for_retrieval": ready,
        "frozen_raw": str(raw_path),
        "frozen_aggregated": str(agg_path),
        "config_hash": cfg.config_hash(),
    }
    (out / "confirmatory_object_analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    return analysis


STAGES = ["synth_decomp", "confirmatory", "object_features", "report"]


def run_confirmatory_object(cfg: ConfirmatoryConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmatory_object_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    results: dict[str, Any] = {}
    order = STAGES if cfg.stage == "all" else [cfg.stage]
    confirm: dict = {}
    obj_meta: dict = {}
    synth: dict = {}
    for s in order:
        print(f"[confirm-object] stage={s} rss={_rss():.1f}", flush=True)
        _budget(t0, cfg.max_seconds, s)
        if s == "synth_decomp":
            sodir = out / "synthetic_decomposition"
            sodir.mkdir(parents=True, exist_ok=True)
            if _done(sodir / "synthetic_decomposition.json", cfg.force):
                synth = json.loads((sodir / "synthetic_decomposition.json").read_text())
            else:
                synth = run_decomposition_synthetics(cfg.seed)
                (sodir / "synthetic_decomposition.json").write_text(json.dumps(synth, indent=2))
                pd.DataFrame(synth["rows"]).to_csv(sodir / "synthetic_decomposition.csv", index=False)
            results[s] = synth
        elif s == "confirmatory":
            confirm = stage_confirmatory(root, cfg, t0)
            results[s] = {k: confirm[k] for k in ("pooled", "stable_across_seeds", "config_hash")}
        elif s == "object_features":
            if not confirm:
                cpath = out / "confirmatory" / "confirmatory_summary.json"
                confirm = json.loads(cpath.read_text()) if cpath.exists() else {"stable_across_seeds": False}
            obj_meta = stage_object_features(root, cfg, t0, confirm)
            results[s] = obj_meta
        elif s == "report":
            if not confirm:
                cpath = out / "confirmatory" / "confirmatory_summary.json"
                confirm = json.loads(cpath.read_text())
            if not obj_meta:
                mpath = out / "object_curvature_features_meta.json"
                obj_meta = json.loads(mpath.read_text()) if mpath.exists() else {}
            if not synth:
                sp = out / "synthetic_decomposition" / "synthetic_decomposition.json"
                synth = json.loads(sp.read_text()) if sp.exists() else {}
            results[s] = stage_report(root, cfg, confirm, obj_meta, synth)
        else:
            raise ValueError(s)
    results["total_seconds"] = time.time() - t0
    results["peak_rss_mb"] = _rss()
    (out / "confirmatory_object_run_summary.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    return results
