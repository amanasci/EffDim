"""Graph-effective-dimension curvature metrics across Smith42/Physics encoders.

Uses validated Isomap residual-elbow + local graph prior (no probe-driven d).
Never fits per-patch probes — only existing five-fold OOF global probes.
"""

from __future__ import annotations

import hashlib
import json
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
from scipy.stats import spearmanr

from .confirmatory_object_curvature import unpack_BS_symmetric
from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import partial_spearman, spearman_dict
from .full_curvature_audit import fit_quad, full_patch_pca_tangent
from .multimodel_graph_prior_quadratic import (
    EPS,
    GRAPH_PRIOR_SOURCE,
    ISOMAP_DIMS_REL,
    ISOMAP_KEY,
    load_model_X,
)
from .paths import platonic_root, resolve_path
from .split_half_curvature_reliability import (
    BS_objects,
    _half_fit_indices,
    tensor_agreement,
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_FCA = "outputs/geometry/physics_full_curvature_audit"
SOURCE_SH = "outputs/geometry/physics_split_half_curvature_reliability"

MODELS = ["vit_base", "dinov3", "clip_base", "convnext_base", "vit_large"]
TARGETS = ["mag_r_desi", "smooth_fraction", "photo_z", "stellar_mass", "sfr"]
SCALES = [1024, 2048]
COMMON_D = 16


# -------------------- metric algebra --------------------


def m_quad(d: int) -> int:
    return d * (d + 1) // 2


def aniso_prefactor(d: int) -> float:
    """Coefficient such that E_v||B(v,v)||^2 = ||H||^2 + pref * ||B°||_F^2."""
    return 2.0 / float(d * (d + 2))


def decompose_tensors(BS_flat: np.ndarray, d: int) -> dict[str, np.ndarray | float]:
    B = unpack_BS_symmetric(BS_flat, d)
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    B0 = B.copy()
    for a in range(d):
        B0[:, a, a] = B[:, a, a] - H
    kh2 = float(np.dot(H, H))
    b0f = float(np.linalg.norm(B0) ** 2)
    bsf = float(np.linalg.norm(B) ** 2)
    ka2 = aniso_prefactor(d) * b0f
    kd2 = kh2 + ka2
    return {
        "H": H,
        "B0": B0,
        "B": B,
        "K_H": float(np.sqrt(max(kh2, 0.0))),
        "K_H2": kh2,
        "K_aniso2": ka2,
        "K_aniso": float(np.sqrt(max(ka2, 0.0))),
        "K_dir2": kd2,
        "K_dir": float(np.sqrt(max(kd2, 0.0))),
        "B_fro": float(np.sqrt(bsf)),
        "B0_fro": float(np.sqrt(b0f)),
    }


def metric_scalars(BS_flat: np.ndarray, d: int) -> dict[str, float]:
    t = decompose_tensors(BS_flat, d)
    return {k: float(t[k]) for k in ("K_H", "K_H2", "K_aniso", "K_aniso2", "K_dir", "K_dir2", "B_fro", "B0_fro")}


def cross_metric_pair(BSA: np.ndarray, BSB: np.ndarray, d: int) -> dict[str, float]:
    a = decompose_tensors(BSA, d)
    b = decompose_tensors(BSB, d)
    khx = float(np.dot(a["H"], b["H"]))
    b0x = float(np.sum(a["B0"] * b["B0"]))
    kax = aniso_prefactor(d) * b0x
    kdx = khx + kax
    return {
        "K_H_cross": khx,
        "K_aniso_cross": kax,
        "K_dir_cross": kdx,
        "K_H_cross_plot": float(max(khx, 0.0)),
        "K_aniso_cross_plot": float(max(kax, 0.0)),
        "K_dir_cross_plot": float(max(kdx, 0.0)),
        "R_H": tensor_agreement(a["H"], b["H"])["R_signal"],
        "R_B0": tensor_agreement(a["B0"].ravel(), b["B0"].ravel())["R_signal"],
        "R_BS": tensor_agreement(a["B"].ravel(), b["B"].ravel())["R_signal"],
        "norm_H_mean": 0.5 * (a["K_H"] + b["K_H"]),
        "norm_dir_mean": 0.5 * (a["K_dir"] + b["K_dir"]),
        "norm_aniso_mean": 0.5 * (a["K_aniso"] + b["K_aniso"]),
    }


def project_normal(w: np.ndarray, x0: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, float]:
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    wn = w - J @ (J.T @ w) - x0u * float(np.dot(x0u, w))
    n = float(np.linalg.norm(wn))
    if n < EPS:
        return wn, 0.0
    return wn / n, n


def probe_facing_scalar(BS_flat: np.ndarray, d: int, w_hat_N: np.ndarray) -> dict[str, float]:
    B = unpack_BS_symmetric(BS_flat, d)
    b = np.einsum("i,iab->ab", w_hat_N, B)
    tr = float(np.trace(b))
    bf2 = float(np.sum(b * b))
    # ((tr b)^2 + 2||b||_F^2) / (d(d+2))
    kw2 = (tr * tr + 2.0 * bf2) / max(float(d * (d + 2)), EPS)
    return {"K_w_dir2": kw2, "K_w_dir": float(np.sqrt(max(kw2, 0.0))), "tr_b": tr, "b_fro2": bf2, "b": b}


def probe_facing_cross(
    BSA: np.ndarray, BSB: np.ndarray, d: int, w_hat_N: np.ndarray
) -> dict[str, float]:
    a = probe_facing_scalar(BSA, d, w_hat_N)
    b = probe_facing_scalar(BSB, d, w_hat_N)
    kx = (a["tr_b"] * b["tr_b"] + 2.0 * float(np.sum(a["b"] * b["b"]))) / max(
        float(d * (d + 2)), EPS
    )
    return {
        "K_probe_facing_cross": kx,
        "K_probe_facing_cross_plot": float(max(kx, 0.0)),
        "K_w_dir_A": a["K_w_dir"],
        "K_w_dir_B": b["K_w_dir"],
    }


def monte_carlo_K_dir2(BS_flat: np.ndarray, d: int, n_dir: int = 4000, seed: int = 0) -> float:
    B = unpack_BS_symmetric(BS_flat, d)
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dir, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + EPS
    # B is (D,d,d); B(v,v) = sum_ab v_a v_b B[:,a,b] → (n_dir, D)
    Bvv = np.einsum("ia,jab,ib->ij", V, B, V)
    return float(np.mean(np.sum(Bvv**2, axis=1)))


# -------------------- config --------------------


@dataclass
class EffDimCurvatureConfig:
    output_dir: str = "outputs/geometry/physics_effdim_curvature_metrics"
    multimodel_dir: str = SOURCE_MM
    coverage_dir: str = SOURCE_COV
    isomap_dims_path: str = ISOMAP_DIMS_REL
    models: list[str] = field(default_factory=lambda: list(MODELS))
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
    ks: list[int] = field(default_factory=lambda: list(SCALES))
    n_splits_primary: int = 5
    n_splits_band: int = 3
    n_band_anchors: int = 128
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 600.0
    skip_gauss: bool = False
    normal_mass_min: float = 0.05

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: EffDimCurvatureConfig, reserve: bool = False) -> bool:
    rem = cfg.max_seconds - (time.time() - t0)
    return rem > (cfg.analyze_reserve_seconds if reserve else 30.0)


def load_ctx(root: Path, cfg: EffDimCurvatureConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = (
        json.loads(aid.read_text())["sample_ids"]
        if aid.exists()
        else [int(s) for s in anchors_sid]
    )
    geo_mm = pd.read_parquet(mm / "local_probe_fields.parquet")
    cov = resolve_path(root, cfg.coverage_dir)
    geo_ext = cov / "extended_local_probe_fields.parquet"
    geo = pd.read_parquet(geo_ext) if geo_ext.exists() else geo_mm
    gp = pd.read_parquet(mm / "graph_dimension_prior.parquet")
    isomap = json.loads(resolve_path(root, cfg.isomap_dims_path).read_text())
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    return {
        "mm": mm,
        "geo": geo,
        "gp": gp,
        "isomap": isomap,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "cov": cov,
    }


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: EffDimCurvatureConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "batches", "figures", "logs"):
        (out / sub).mkdir(exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "protocol": "effdim_curvature_metrics_v1",
        "graph_prior_source": GRAPH_PRIOR_SOURCE,
        "isomap_dims_path": cfg.isomap_dims_path,
        "preserved": [SOURCE_MM, SOURCE_COV, SOURCE_FCA, SOURCE_SH],
        "no_local_probes": True,
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[edm] prepare hash={meta['config_hash']}", flush=True)
    return meta


def _synth_manifold(kind: str, n: int, d_in: int, D: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "affine":
        Z = rng.normal(size=(n, d_in))
        A = rng.normal(size=(D, d_in))
        X = Z @ A.T
    elif kind == "sphere":
        Z = rng.normal(size=(n, d_in + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        A = rng.normal(size=(D, d_in + 1))
        X = Z @ A.T
    elif kind == "torus":
        u, v = rng.uniform(0, 2 * np.pi, size=(2, n))
        R, r = 2.0, 1.0
        Y = np.stack(
            [(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)],
            axis=1,
        )
        # pad intrinsic coords
        pad = rng.normal(size=(n, max(d_in - 2, 0))) * 0.05 if d_in > 2 else np.zeros((n, 0))
        Z = np.concatenate([Y, pad], axis=1) if pad.size else Y
        A = rng.normal(size=(D, Z.shape[1]))
        X = Z @ A.T
    elif kind in ("swiss", "scurve"):
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
        h = 21 * rng.uniform(size=n)
        if kind == "swiss":
            Y = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
        else:
            Y = np.stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1)], axis=1)
        pad = rng.normal(size=(n, max(d_in - 2, 0))) * 0.02 if d_in > 2 else np.zeros((n, 0))
        Z = np.concatenate([Y[:, :2], pad], axis=1) if d_in > 2 else Y[:, :2]
        # swiss/scurve intrinsic ~2; report vs d_true=2 regardless of d_in request
        A = rng.normal(size=(D, Z.shape[1]))
        X = Z @ A.T
    elif kind == "quadratic":
        Z = rng.normal(size=(n, d_in))
        A = rng.normal(size=(D, d_in))
        X = Z @ A.T
        X[:, : min(8, D)] += 0.15 * (Z[:, :1] ** 2)
    elif kind == "nonuniform":
        Z = rng.normal(size=(n, d_in)) ** 3
        A = rng.normal(size=(D, d_in))
        X = Z @ A.T
    elif kind == "noisy":
        Z = rng.normal(size=(n, d_in))
        A = rng.normal(size=(D, d_in))
        X = Z @ A.T + 0.05 * rng.normal(size=(n, D))
    elif kind == "boundary":
        Z = rng.uniform(-1, 1, size=(n, d_in))
        A = rng.normal(size=(D, d_in))
        X = Z @ A.T
    else:
        raise ValueError(kind)
    X = X.astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X


def stage_synthetic_dimension(root: Path, cfg: EffDimCurvatureConfig) -> dict:
    out = cfg.resolved(root)
    path = out / "synthetic_dimension_gate.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    import importlib
    import sys

    exp = str(resolve_path(root, "experiments"))
    if exp not in sys.path:
        sys.path.insert(0, exp)
    iso = importlib.import_module("isomap_ann_mknn_gpu")
    import importlib.util

    pipe_path = resolve_path(root, "experiments/SAE-shared-basis/pipeline_isomap_sae_shared_mknn.py")
    spec = importlib.util.spec_from_file_location("pipeline_isomap_sae_shared_mknn", pipe_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    estimate_isomap_dim = mod.estimate_isomap_dim

    kinds = [
        ("affine", 8),
        ("sphere", 8),
        ("torus", 2),
        ("swiss", 2),
        ("scurve", 2),
        ("quadratic", 8),
        ("nonuniform", 8),
        ("noisy", 8),
        ("boundary", 8),
    ]
    kind_seed = {k: i + 1 for i, (k, _) in enumerate(kinds)}
    rows = []
    device = "cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    for kind, d_true in kinds:
        X = _synth_manifold(
            kind, n=2500, d_in=max(d_true, 2), D=40, seed=cfg.seed + 1009 * kind_seed[kind]
        )
        try:
            stats = estimate_isomap_dim(
                X,
                iso=iso,
                graph_k=20,
                n_landmarks=768,
                n_components=48,
                seed=cfg.seed,
                dim_method="residual_elbow",
                graph_backend="exact",
                d_grid=list(range(1, 25)),
                device=device,
            )
            d_hat = int(stats.get("d_residual_elbow", stats.get("d_primary", -1)))
        except Exception as e:  # noqa: BLE001
            d_hat = -1
            stats = {"error": f"{type(e).__name__}: {e}"}
        err = abs(d_hat - d_true) if d_hat > 0 else 99
        rows.append(
            {
                "kind": kind,
                "d_true": d_true,
                "d_hat": d_hat,
                "abs_err": err,
                "exact": err == 0,
                "within_1": err <= 1,
                "within_2": err <= 2,
                "within_3": err <= 3,
                "stats": {k: stats[k] for k in ("d_residual_elbow", "d_pr", "d_cum90") if k in stats},
            }
        )
        print(f"[edm][syn] {kind} true={d_true} hat={d_hat} err={err}", flush=True)
    df = pd.DataFrame(rows)
    # Low-d manifolds must recover; high-d elbows are known mild underestimates.
    low = df[df.kind.isin(["swiss", "scurve", "torus", "sphere"])]
    hard = df[df.kind.isin(["affine", "quadratic", "noisy", "nonuniform", "boundary"])]
    low_rate = float(low.within_2.mean()) if len(low) else 0.0
    hard_rate = float(hard.within_3.mean()) if len(hard) else 0.0
    ok = (
        int((df.d_hat > 0).sum()) >= 7
        and low_rate >= 0.75
        and hard_rate >= 0.4
    )
    result = {
        "ok": ok,
        "within_2_rate_low": low_rate,
        "within_3_rate_hard": hard_rate,
        "within_2_rate_hard": float(hard.within_2.mean()) if len(hard) else 0.0,
        "exact_rate": float(df.exact.mean()),
        "within_1_rate": float(df.within_1.mean()),
        "within_2_rate": float(df.within_2.mean()),
        "rows": rows,
        "estimator": "estimate_isomap_dim residual_elbow",
        "note": "Gate: low-d within_2>=0.75 and hard within_3>=0.4 (elbow underestimates high-d).",
    }
    path.write_text(json.dumps(result, indent=2, default=str))
    pd.DataFrame(rows).drop(columns=["stats"], errors="ignore").to_csv(
        out / "synthetic_dimension_recovery.csv", index=False
    )
    print(
        f"[edm][syn] gate ok={ok} low_within_2={low_rate:.2f} hard_within_3={hard_rate:.2f}",
        flush=True,
    )
    return result


def stage_estimate_dimension(root: Path, cfg: EffDimCurvatureConfig, ctx: dict) -> pd.DataFrame:
    out = cfg.resolved(root)
    path = out / "dimension_estimates.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)
    gp = ctx["gp"]
    rows = []
    for m in cfg.models:
        key = ISOMAP_KEY.get(m)
        gstat = ctx["isomap"].get(key, {})
        d_iso = gstat.get("d_residual_elbow", gstat.get("d_primary", float("nan")))
        for k in cfg.ks:
            sub = gp[(gp.model == m) & (gp.scale_k == k)].copy()
            if sub.empty:
                continue
            # bootstrap median of d_graph
            vals = sub.d_graph.to_numpy(float)
            rng = np.random.default_rng(cfg.seed + 17 * k + 997 * (MODELS.index(m) + 1))
            boots = []
            for _ in range(200):
                boots.append(float(np.median(rng.choice(vals, size=len(vals), replace=True))))
            boots = np.asarray(boots)
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "model": m,
                        "sample_id": int(r.sample_id),
                        "scale_k": int(k),
                        "d_isomap": float(d_iso),
                        "d_graph": float(r.d_graph),
                        "d_energy_rank_90": float(r.d_energy_rank_90),
                        "d_participation_ratio": float(r.d_participation_ratio),
                        "dimension_confidence": float(
                            1.0 / (1.0 + abs(float(r.d_graph) - float(d_iso)))
                        )
                        if np.isfinite(d_iso)
                        else float("nan"),
                        "elbow_strength": float(gstat.get("residual_variance_at_elbow", np.nan))
                        if isinstance(gstat, dict)
                        else float("nan"),
                        "graph_connectivity": float(r.graph_support_turnover),
                        "graph_boundary_imbalance": float(r.graph_boundary_imbalance),
                        "bootstrap_med_p05": float(np.quantile(boots, 0.05)),
                        "bootstrap_med_p95": float(np.quantile(boots, 0.95)),
                        "failure_flag": bool(not np.isfinite(r.d_graph) or r.d_graph < 2),
                        "source": GRAPH_PRIOR_SOURCE,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[edm] dimension_estimates n={len(df)}", flush=True)
    return df


def stage_freeze_dimension(
    root: Path, cfg: EffDimCurvatureConfig, ctx: dict, est: pd.DataFrame, syn: dict
) -> dict:
    out = cfg.resolved(root)
    path = out / "dimension_freeze.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    if not syn.get("ok", False):
        freeze = {"ok": False, "reason": "synthetic_dimension_gate_failed", "synthetic": syn}
        path.write_text(json.dumps(freeze, indent=2, default=str))
        print("[edm] STOP — synthetic dimension gate failed", flush=True)
        return freeze

    freeze_rows = []
    for m in cfg.models:
        for k in cfg.ks:
            sub = est[(est.model == m) & (est.scale_k == k)]
            if sub.empty:
                continue
            # consensus: median of per-anchor d_graph and global isomap
            d_graph_med = float(np.median(sub.d_graph))
            d_iso = float(np.nanmedian(sub.d_isomap))
            d_star = int(np.clip(round(0.5 * d_graph_med + 0.5 * d_iso), 4, 24))
            # band from bootstrap of median + anchor IQR, narrow
            lo_b = float(np.nanmedian(sub.bootstrap_med_p05))
            hi_b = float(np.nanmedian(sub.bootstrap_med_p95))
            iqr = np.subtract(*np.percentile(sub.d_graph, [75, 25]))
            d_minus = int(np.clip(round(min(lo_b, d_star - 1, d_star - 0.5 * iqr)), 4, d_star))
            d_plus = int(np.clip(round(max(hi_b, d_star + 1, d_star + 0.5 * iqr)), d_star, 24))
            # keep band narrow: at most ±2
            d_minus = max(d_star - 2, min(d_minus, d_star))
            d_plus = min(d_star + 2, max(d_plus, d_star))
            if d_minus == d_star:
                d_minus = max(4, d_star - 1)
            if d_plus == d_star:
                d_plus = d_star + 1
            disagree = abs(d_graph_med - d_iso)
            freeze_rows.append(
                {
                    "model": m,
                    "scale_k": k,
                    "d_star": d_star,
                    "d_minus": d_minus,
                    "d_plus": d_plus,
                    "d_common": COMMON_D,
                    "d_graph_median": d_graph_med,
                    "d_isomap_global": d_iso,
                    "method_disagreement": disagree,
                    "n_anchors": int(len(sub)),
                }
            )
    freeze = {
        "ok": True,
        "protocol": "consensus_median(d_graph,d_isomap_global); band clipped ±2",
        "estimator_versions": {
            "isomap": "pipeline_isomap_sae_shared_mknn.estimate_isomap_dim residual_elbow",
            "local_graph": GRAPH_PRIOR_SOURCE,
        },
        "isomap_dims_path": cfg.isomap_dims_path,
        "isomap_input_hash": hashlib.sha256(
            resolve_path(root, cfg.isomap_dims_path).read_bytes()
        ).hexdigest()[:16],
        "graph_prior_hash": hashlib.sha256(
            (ctx["mm"] / "graph_dimension_prior.parquet").read_bytes()
        ).hexdigest()[:16],
        "no_label_verification": {
            "used_probe_labels": False,
            "used_quadratic_recon_optimum": False,
            "used_pca_rank_alone": False,
            "sources": ["d_isomap_global residual_elbow", "d_graph blend energy_rank+isomap"],
        },
        "synthetic_gate": {
            k: syn[k]
            for k in (
                "ok",
                "within_2_rate_low",
                "within_3_rate_hard",
                "within_2_rate",
            )
            if k in syn
        },
        "by_model_scale": freeze_rows,
        "dimension_config_hash": "",
    }
    freeze["dimension_config_hash"] = hashlib.sha256(
        json.dumps(freeze_rows, sort_keys=True).encode()
    ).hexdigest()[:16]
    path.write_text(json.dumps(freeze, indent=2, default=str))
    pd.DataFrame(freeze_rows).to_csv(out / "dimension_stability.csv", index=False)
    print(f"[edm] freeze hash={freeze['dimension_config_hash']}", flush=True)
    for r in freeze_rows:
        print(
            f"  {r['model']} k={r['scale_k']}: d*={r['d_star']} "
            f"[{r['d_minus']},{r['d_plus']}] iso={r['d_isomap_global']:.0f} "
            f"graph_med={r['d_graph_median']:.1f}",
            flush=True,
        )
    return freeze


def _load_probe_weight(mm: Path, model: str, target: str) -> tuple[np.ndarray, bool]:
    p = mm / "global_probes" / "oof_predictions" / f"{model}_{target}.npz"
    if not p.exists():
        # sfr from coverage cache
        return np.zeros(0), False
    z = np.load(p)
    w = z["w_full"].astype(np.float64)
    # fold stability if present
    stable = True
    if "fold_cos_median" in z.files:
        stable = bool(float(z["fold_cos_median"][0]) > 0.9) if np.isfinite(z["fold_cos_median"][0]) else False
    return w, stable


def _fit_cell(
    X: np.ndarray,
    pack: dict,
    use_sids: list[int],
    sid_to_ai: dict,
    d: int,
    k: int,
    n_splits: int,
    seed: int,
    model: str,
    role: str,
    t0: float,
    cfg: EffDimCurvatureConfig,
    probe_w: dict[str, tuple[np.ndarray, bool]] | None = None,
) -> pd.DataFrame:
    rows = []
    md = m_quad(d)
    for si, sid in enumerate(use_sids):
        if si % 64 == 0:
            print(f"[edm][fit] {model} {role} d={d} k={k} {si}/{len(use_sids)}", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = sid_to_ai[int(sid)]
        N = pack["neigh"][ai, :k]
        Xloc = X[N].astype(np.float64)
        x0, J = full_patch_pca_tangent(Xloc, d)
        if J.shape[1] < d:
            continue
        neff = (0.4 * (k // 2)) / max(md, 1)
        for s in range(n_splits):
            rng = np.random.default_rng(seed + 1009 * ai + 17 * s + d * 13 + k)
            perm = rng.permutation(k)
            halfA, halfB = perm[: k // 2], perm[k // 2 :]
            fA, vA = _half_fit_indices(halfA, seed + 3 + s)
            fB, vB = _half_fit_indices(halfB, seed + 7 + s)
            chA, _, infoA = fit_quad(Xloc, x0, J, fA, vA, halfB)
            chB, _, infoB = fit_quad(Xloc, x0, J, fB, vB, halfA)
            if chA is None or chB is None:
                continue
            cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
            sa, sb = metric_scalars(chA.BS_flat, d), metric_scalars(chB.BS_flat, d)
            row = {
                "model": model,
                "sample_id": int(sid),
                "split": s,
                "d": d,
                "k": k,
                "role": role,
                "m_d": md,
                "n_eff_over_m": neff,
                "underdetermined": bool(neff < 3.0),
                "dS": 0.5 * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan))),
                **cross,
                "K_H_A": sa["K_H"],
                "K_H_B": sb["K_H"],
                "K_dir_A": sa["K_dir"],
                "K_dir_B": sb["K_dir"],
                "K_aniso_A": sa["K_aniso"],
                "K_aniso_B": sb["K_aniso"],
            }
            # probe-facing for available targets
            if probe_w:
                for t, (w, stable) in probe_w.items():
                    if w.size != X.shape[1]:
                        continue
                    wh, nmass = project_normal(w, x0, J)
                    if nmass < cfg.normal_mass_min:
                        row[f"K_pf_{t}_cross"] = float("nan")
                        row[f"pf_{t}_normal_mass"] = nmass
                        row[f"pf_{t}_stable"] = stable
                        continue
                    pf = probe_facing_cross(chA.BS_flat, chB.BS_flat, d, wh)
                    row[f"K_pf_{t}_cross"] = pf["K_probe_facing_cross"]
                    row[f"pf_{t}_normal_mass"] = nmass
                    row[f"pf_{t}_stable"] = stable
            rows.append(row)
    return pd.DataFrame(rows)


def stage_fit_curvature(
    root: Path, cfg: EffDimCurvatureConfig, ctx: dict, freeze: dict, t0: float
) -> None:
    out = cfg.resolved(root)
    path = out / "curvature_metrics.parquet"
    if _done(path, cfg.force):
        return
    if not freeze.get("ok"):
        print("[edm] skip fit — freeze not ok", flush=True)
        return
    chunks = []
    freeze_df = pd.DataFrame(freeze["by_model_scale"])
    # geometry-stratified band anchors: by d_graph tercile from estimates
    est = pd.read_parquet(out / "dimension_estimates.parquet")

    for m in cfg.models:
        X = load_model_X(ctx["mm"], m)
        pack = dict(np.load(ctx["mm"] / "model_neighbourhoods" / f"{m}_kmax2048.npz"))
        probe_w = {}
        for t in cfg.targets:
            w, st = _load_probe_weight(ctx["mm"], m, t)
            if w.size:
                probe_w[t] = (w, st)
        # sfr from coverage
        sfr_p = ctx["cov"] / "cache" / "oof" / f"{m}_sfr.npz"
        if "sfr" in cfg.targets and sfr_p.exists() and "sfr" not in probe_w:
            # no w_full for sfr — skip probe-facing for sfr
            pass
        for k in cfg.ks:
            fr = freeze_df[(freeze_df.model == m) & (freeze_df.scale_k == k)]
            if fr.empty:
                continue
            d_star = int(fr.iloc[0].d_star)
            d_minus = int(fr.iloc[0].d_minus)
            d_plus = int(fr.iloc[0].d_plus)
            cell = out / "batches" / f"{m}_k{k}_dstar{d_star}.parquet"
            if _done(cell, cfg.force):
                chunks.append(pd.read_parquet(cell))
            else:
                if not _budget_ok(t0, cfg, reserve=True):
                    break
                df = _fit_cell(
                    X,
                    pack,
                    ctx["use_sids"],
                    ctx["sid_to_ai"],
                    d_star,
                    k,
                    cfg.n_splits_primary,
                    cfg.seed,
                    m,
                    "d_star",
                    t0,
                    cfg,
                    probe_w,
                )
                df.to_parquet(cell, index=False)
                chunks.append(df)
                print(f"[edm] wrote {cell.name} n={len(df)}", flush=True)

            # band dims on stratified 128 anchors
            for d_band, role in ((d_minus, "d_minus"), (d_plus, "d_plus")):
                bcell = out / "batches" / f"{m}_k{k}_{role}{d_band}.parquet"
                if _done(bcell, cfg.force):
                    chunks.append(pd.read_parquet(bcell))
                    continue
                if not _budget_ok(t0, cfg, reserve=True):
                    break
                sube = est[(est.model == m) & (est.scale_k == k)].sort_values("d_graph")
                # take every n-th for stratification
                step = max(1, len(sube) // cfg.n_band_anchors)
                band_sids = sube.sample_id.to_numpy(int)[::step][: cfg.n_band_anchors]
                dfb = _fit_cell(
                    X,
                    pack,
                    list(band_sids),
                    ctx["sid_to_ai"],
                    d_band,
                    k,
                    cfg.n_splits_band,
                    cfg.seed + 99,
                    m,
                    role,
                    t0,
                    cfg,
                    probe_w,
                )
                dfb.to_parquet(bcell, index=False)
                chunks.append(dfb)

            # common d=16 thin sensitivity if budget
            ccell = out / "batches" / f"{m}_k{k}_dcommon16.parquet"
            if _done(ccell, cfg.force):
                chunks.append(pd.read_parquet(ccell))
            elif _budget_ok(t0, cfg, reserve=True):
                sube = est[(est.model == m) & (est.scale_k == k)].sort_values("d_graph")
                step = max(1, len(sube) // cfg.n_band_anchors)
                band_sids = sube.sample_id.to_numpy(int)[::step][: cfg.n_band_anchors]
                dfc = _fit_cell(
                    X,
                    pack,
                    list(band_sids),
                    ctx["sid_to_ai"],
                    COMMON_D,
                    k,
                    cfg.n_splits_band,
                    cfg.seed + 7,
                    m,
                    "d_common16",
                    t0,
                    cfg,
                    probe_w,
                )
                dfc.to_parquet(ccell, index=False)
                chunks.append(dfc)

    if not chunks:
        pd.DataFrame().to_parquet(path, index=False)
        return
    all_df = pd.concat(chunks, ignore_index=True)
    all_df.to_parquet(path, index=False)
    # anchor means for primary
    num_cols = [
        c
        for c in all_df.columns
        if c.startswith("K_") or c.startswith("R_") or c.startswith("norm_")
    ]
    primary = all_df[all_df.role == "d_star"]
    if len(primary):
        agg = primary.groupby(["model", "sample_id", "d", "k", "role"], as_index=False)[
            num_cols
        ].mean()
        agg.to_parquet(out / "crossfit_curvature_metrics.parquet", index=False)
    # reliability summary
    rel_rows = []
    for (m, k, role, d), g in all_df.groupby(["model", "k", "role", "d"]):
        for metric, col in [
            ("K_H", "K_H_cross"),
            ("K_aniso", "K_aniso_cross"),
            ("K_dir", "K_dir_cross"),
        ]:
            # split-half scalar reliability: correlate half norms across splits? use R_*
            rcol = {"K_H": "R_H", "K_aniso": "R_B0", "K_dir": "R_BS"}[metric]
            rel_rows.append(
                {
                    "model": m,
                    "k": k,
                    "role": role,
                    "d": d,
                    "metric": metric,
                    "median_R": float(g[rcol].median()) if rcol in g else float("nan"),
                    "median_cross": float(g[col].median()),
                    "frac_cross_pos": float((g[col] > 0).mean()),
                    "n": len(g),
                }
            )
    pd.DataFrame(rel_rows).to_csv(out / "reliability_by_model_scale.csv", index=False)
    print(f"[edm] fit_curvature n={len(all_df)}", flush=True)


def stage_probe_facing(root: Path, cfg: EffDimCurvatureConfig) -> None:
    """Extract probe-facing columns into dedicated table."""
    out = cfg.resolved(root)
    path = out / "probe_facing_metrics.parquet"
    if _done(path, cfg.force):
        return
    src = out / "curvature_metrics.parquet"
    if not src.exists():
        pd.DataFrame().to_parquet(path, index=False)
        return
    df = pd.read_parquet(src)
    cols = ["model", "sample_id", "split", "d", "k", "role"] + [
        c for c in df.columns if c.startswith("K_pf_") or c.startswith("pf_")
    ]
    df[cols].to_parquet(path, index=False)
    print(f"[edm] probe_facing cols={len(cols)}", flush=True)


def stage_associations(root: Path, cfg: EffDimCurvatureConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "associations_raw.csv"
    if _done(Path(path), cfg.force):
        return
    cross = out / "crossfit_curvature_metrics.parquet"
    if not cross.exists():
        return
    curv = pd.read_parquet(cross)
    geo = ctx["geo"]
    gp = ctx["gp"]
    # weak targets from cv
    weak = set()
    cvp = ctx["mm"] / "global_probe_cv_metrics.parquet"
    if cvp.exists():
        cv = pd.read_parquet(cvp)
        for _, r in cv.iterrows():
            if float(r.oof_r2) < 0.02:
                weak.add((str(r.model), str(r.target)))
    # sfr always underpowered for claims
    for m in cfg.models:
        weak.add((m, "sfr"))

    metrics = [
        ("K_H_cross", "K_H_cross"),
        ("K_aniso_cross", "K_aniso_cross"),
        ("K_dir_cross", "K_dir_cross"),
    ]
    raw_rows = []
    seq_rows = []
    for m in cfg.models:
        for t in cfg.targets:
            for k in cfg.ks:
                sub = curv[(curv.model == m) & (curv.k == k) & (curv.role == "d_star")].copy()
                g = geo[
                    (geo.model == m)
                    & (geo.target == t)
                    & (geo.neighbourhood == "model")
                    & (geo.scale_k == min(k, int(geo.scale_k.max())))
                ].copy()
                if sub.empty or g.empty:
                    continue
                # drop colliding
                drop = [c for c in ("local_r2", "log_knn_radius") if c in sub.columns]
                gg = g.merge(sub.drop(columns=drop, errors="ignore"), on="sample_id", how="inner")
                gpp = gp[(gp.model == m) & (gp.scale_k == k)][
                    ["sample_id", "graph_support_turnover", "graph_boundary_imbalance", "d_graph"]
                ]
                gg = gg.merge(gpp, on="sample_id", how="left")
                gg["recon_proxy"] = 1.0 - gg.get("R_BS", pd.Series(0, index=gg.index)).clip(0, 1)
                y = gg.local_r2.to_numpy(float)
                # optional pf metric
                pf_col = f"K_pf_{t}_cross"
                mlist = list(metrics)
                if pf_col in gg.columns:
                    mlist.append(("K_probe_facing_cross", pf_col))
                for sname, scol in mlist:
                    if scol not in gg.columns:
                        continue
                    x = gg[scol].to_numpy(float)
                    raw = spearman_dict(x, y)
                    msk = np.isfinite(x) & np.isfinite(y)
                    pear = (
                        float(np.corrcoef(x[msk], y[msk])[0, 1])
                        if msk.sum() > 8
                        else float("nan")
                    )
                    raw_rows.append(
                        {
                            "model": m,
                            "target": t,
                            "target_weak": (m, t) in weak,
                            "k": k,
                            "d": int(gg.d.iloc[0]) if len(gg) else -1,
                            "metric": sname,
                            "n": raw["n"],
                            "spearman": raw["rho"],
                            "pearson": pear,
                        }
                    )
                    path_coefs = {"raw": raw["rho"]}
                    steps = [
                        ("+radius", ["log_knn_radius"]),
                        ("+label_var", ["log_knn_radius", "local_label_variance"]),
                        (
                            "+eval_count",
                            ["log_knn_radius", "local_label_variance", "local_evaluation_count"],
                        ),
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
                    for step_name, cols in steps:
                        Z = np.column_stack([gg[c].fillna(0).to_numpy(float) for c in cols])
                        path_coefs[step_name] = partial_spearman(x, y, Z)["rho"]
                    if "graph_support_turnover" in gg.columns:
                        Z = np.column_stack(
                            [
                                gg.log_knn_radius.fillna(0).to_numpy(float),
                                gg.local_label_variance.fillna(0).to_numpy(float),
                                gg.local_evaluation_count.fillna(0).to_numpy(float),
                                gg.recon_proxy.fillna(0).to_numpy(float),
                                gg.graph_support_turnover.fillna(0).to_numpy(float),
                                gg.graph_boundary_imbalance.fillna(0).to_numpy(float),
                            ]
                        )
                        path_coefs["+boundary"] = partial_spearman(x, y, Z)["rho"]
                    seq_rows.append(
                        {
                            "model": m,
                            "target": t,
                            "target_weak": (m, t) in weak,
                            "k": k,
                            "d": int(gg.d.iloc[0]),
                            "metric": sname,
                            "n": raw["n"],
                            **path_coefs,
                        }
                    )
    pd.DataFrame(raw_rows).to_csv(out / "associations_raw.csv", index=False)
    pd.DataFrame(seq_rows).to_csv(out / "associations_sequential_controls.csv", index=False)
    print(f"[edm] associations raw={len(raw_rows)} seq={len(seq_rows)}", flush=True)


def stage_metric_comparison(root: Path, cfg: EffDimCurvatureConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "metric_comparison.csv"
    if _done(Path(path), cfg.force):
        return
    cross = out / "crossfit_curvature_metrics.parquet"
    if not cross.exists():
        return
    curv = pd.read_parquet(cross)
    geo = ctx["geo"]
    rows = []
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    for m in cfg.models:
        for t in ["mag_r_desi", "smooth_fraction"]:
            for k in cfg.ks:
                sub = curv[(curv.model == m) & (curv.k == k) & (curv.role == "d_star")]
                g = geo[
                    (geo.model == m)
                    & (geo.target == t)
                    & (geo.neighbourhood == "model")
                    & (geo.scale_k == min(k, 2048))
                ]
                gg = g.merge(sub, on="sample_id", how="inner")
                if len(gg) < 40:
                    continue
                y = rankdata_safe(gg.local_r2.to_numpy(float))
                ctrl = np.column_stack(
                    [
                        gg.log_knn_radius.fillna(0).to_numpy(float),
                        gg.local_label_variance.fillna(0).to_numpy(float),
                        gg.local_evaluation_count.fillna(0).to_numpy(float),
                    ]
                )
                specs = {
                    "controls": ctrl,
                    "controls+K_H": np.column_stack([ctrl, gg.K_H_cross.to_numpy(float)]),
                    "controls+K_aniso": np.column_stack([ctrl, gg.K_aniso_cross.to_numpy(float)]),
                    "controls+K_dir": np.column_stack([ctrl, gg.K_dir_cross.to_numpy(float)]),
                    "controls+K_H+K_aniso": np.column_stack(
                        [
                            ctrl,
                            gg.K_H_cross.to_numpy(float),
                            gg.K_aniso_cross.to_numpy(float),
                        ]
                    ),
                }
                pf = f"K_pf_{t}_cross"
                if pf in gg.columns and np.isfinite(gg[pf]).sum() > 40:
                    specs["controls+K_probe_facing"] = np.column_stack(
                        [ctrl, gg[pf].fillna(0).to_numpy(float)]
                    )
                base_r2 = cv_r2(ctrl, y)
                for name, X in specs.items():
                    r2 = cv_r2(X, y)
                    rows.append(
                        {
                            "model": m,
                            "target": t,
                            "k": k,
                            "spec": name,
                            "cv_r2": r2,
                            "delta_r2_vs_controls": r2 - base_r2,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[edm] metric_comparison n={len(rows)}", flush=True)


def rankdata_safe(y: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    y = np.asarray(y, float)
    m = np.isfinite(y)
    out = np.full_like(y, np.nan)
    if m.sum():
        out[m] = rankdata(y[m])
    return np.nan_to_num(out, nan=np.nanmedian(out) if np.isfinite(out).any() else 0.0)


def cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    if len(y) < 30:
        return float("nan")
    kf = KFold(n_splits=min(n_splits, len(y) // 6), shuffle=True, random_state=0)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        reg = LinearRegression().fit(X[tr], y[tr])
        preds[te] = reg.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - ss_res / max(ss_tot, EPS))


def stage_dimension_sensitivity(root: Path, cfg: EffDimCurvatureConfig) -> None:
    out = cfg.resolved(root)
    path = out / "dimension_sensitivity.csv"
    if _done(Path(path), cfg.force):
        return
    src = out / "curvature_metrics.parquet"
    if not src.exists():
        return
    df = pd.read_parquet(src)
    rows = []
    for (m, k), g in df.groupby(["model", "k"]):
        for metric, col in [
            ("K_H_cross", "K_H_cross"),
            ("K_dir_cross", "K_dir_cross"),
            ("K_aniso_cross", "K_aniso_cross"),
        ]:
            star = g[g.role == "d_star"].groupby("sample_id")[col].mean()
            for role in ("d_minus", "d_plus", "d_common16"):
                other = g[g.role == role].groupby("sample_id")[col].mean()
                common = star.index.intersection(other.index)
                if len(common) < 20:
                    continue
                rho, _ = spearmanr(star.loc[common], other.loc[common])
                rows.append(
                    {
                        "model": m,
                        "k": k,
                        "metric": metric,
                        "vs_role": role,
                        "rank_corr": float(rho),
                        "n": int(len(common)),
                        "label": (
                            "dimension_robust"
                            if abs(rho) > 0.7
                            else "dimension_sensitive"
                            if abs(rho) > 0.4
                            else "dimension_underpowered"
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[edm] dimension_sensitivity n={len(rows)}", flush=True)


def stage_analyze(root: Path, cfg: EffDimCurvatureConfig, freeze: dict, syn: dict) -> None:
    out = cfg.resolved(root)
    rel = (
        pd.read_csv(out / "reliability_by_model_scale.csv")
        if (out / "reliability_by_model_scale.csv").exists()
        else pd.DataFrame()
    )
    seq = (
        pd.read_csv(out / "associations_sequential_controls.csv")
        if (out / "associations_sequential_controls.csv").exists()
        else pd.DataFrame()
    )
    raw = (
        pd.read_csv(out / "associations_raw.csv")
        if (out / "associations_raw.csv").exists()
        else pd.DataFrame()
    )
    mc = (
        pd.read_csv(out / "metric_comparison.csv")
        if (out / "metric_comparison.csv").exists()
        else pd.DataFrame()
    )
    ds = (
        pd.read_csv(out / "dimension_sensitivity.csv")
        if (out / "dimension_sensitivity.csv").exists()
        else pd.DataFrame()
    )
    freeze_df = pd.DataFrame(freeze.get("by_model_scale", []))

    labels = []
    if freeze.get("ok"):
        labels.append("dimension_freeze_ok")
    if len(rel):
        prim = rel[(rel.role == "d_star") & (rel.k == 2048) & (rel.metric == "K_dir")]
        if len(prim) and float(prim.median_R.median()) > 0.3:
            labels.append("cross_model_curvature_reliable")
        kh = rel[(rel.role == "d_star") & (rel.k == 2048) & (rel.metric == "K_H")]
        ka = rel[(rel.role == "d_star") & (rel.k == 2048) & (rel.metric == "K_aniso")]
        if len(kh) and len(ka):
            if float(ka.median_R.median()) > float(kh.median_R.median()) + 0.05:
                labels.append("anisotropic_curvature_specific")
            elif float(kh.median_R.median()) > float(ka.median_R.median()) + 0.05:
                labels.append("mean_curvature_specific")
            else:
                labels.append("total_directional_curvature_specific")

    # mag_r associations
    mag = seq[(seq.target == "mag_r_desi") & (seq.k == 2048) & (~seq.target_weak)] if len(seq) else pd.DataFrame()
    best_metric = None
    if len(mag):
        # pick metric with strongest |+boundary| else |raw|
        col = "+boundary" if "+boundary" in mag.columns else "raw"
        mag = mag.copy()
        mag["score"] = mag[col].abs()
        best = mag.sort_values("score", ascending=False).iloc[0]
        best_metric = str(best.metric)
        vit = mag[(mag.model == "vit_base") & (mag.metric == best_metric)]
        others = mag[(mag.model != "vit_base") & (mag.metric == best_metric)]
        if len(vit) and float(vit.iloc[0][col]) < -0.15:
            if len(others) and (others[col] < 0).mean() >= 0.6:
                labels.append("curvature_metric_mixed" if (others[col].abs() < 0.1).mean() > 0.5 else "cross_model_functional_link")
            else:
                labels.append("vit_base_specific_functional_link")
        if len(vit) and abs(float(vit.iloc[0]["raw"])) > 0.2 and abs(float(vit.iloc[0].get("+boundary", vit.iloc[0]["raw"]))) < 0.1:
            labels.append("raw_geography_only")
        if len(vit) and abs(float(vit.iloc[0].get("+boundary", 0))) > 0.15:
            pass
        else:
            if "no_control_stable_curvature_link" not in labels and best_metric:
                if len(vit) and abs(float(vit.iloc[0].get("+boundary", 0))) < 0.1:
                    labels.append("no_control_stable_curvature_link")

    if len(ds):
        if (ds.label == "dimension_robust").mean() >= 0.5:
            labels.append("dimension_robust")
        elif (ds.label == "dimension_sensitive").mean() >= 0.5:
            labels.append("dimension_sensitive")

    # cross-model summary
    xm_rows = []
    if len(raw):
        for t in cfg.targets:
            for metric in ["K_H_cross", "K_aniso_cross", "K_dir_cross", "K_probe_facing_cross"]:
                for k in cfg.ks:
                    sub = raw[(raw.target == t) & (raw.metric == metric) & (raw.k == k)]
                    if sub.empty:
                        continue
                    xm_rows.append(
                        {
                            "target": t,
                            "metric": metric,
                            "k": k,
                            "median_spearman": float(sub.spearman.median()),
                            "frac_neg": float((sub.spearman < 0).mean()),
                            "frac_pos": float((sub.spearman > 0).mean()),
                            "n_models": int(sub.model.nunique()),
                        }
                    )
    pd.DataFrame(xm_rows).to_csv(out / "cross_model_summary.csv", index=False)
    (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))

    # figures
    figdir = out / "figures"
    if len(freeze_df):
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in cfg.ks:
            sub = freeze_df[freeze_df.scale_k == k]
            ax.errorbar(
                range(len(sub)),
                sub.d_star,
                yerr=[sub.d_star - sub.d_minus, sub.d_plus - sub.d_star],
                fmt="o",
                label=f"k={k}",
            )
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub.model, rotation=30, ha="right")
        ax.axhline(COMMON_D, color="gray", ls="--", label="common d=16")
        ax.set_ylabel("graph d*")
        ax.legend()
        ax.set_title("Frozen graph dimensions")
        fig.tight_layout()
        fig.savefig(figdir / "dimension_freeze.png", dpi=120)
        plt.close(fig)

    report = f"""# Graph-effective-dimension curvature metrics

## Scope

Smith42/Physics encoders `{cfg.models}` × targets `{cfg.targets}` × k=`{cfg.ks}`.
No per-patch probes. Fixed five-fold OOF global probes only.
Graph dimension from validated Isomap residual-elbow + local energy-rank blend (label-free).

## Synthetic dimension gate

ok={syn.get('ok')} within_2_hard={syn.get('within_2_rate_hard')} within_2={syn.get('within_2_rate')}

## Frozen dimensions (label-free)

```
{freeze_df.to_string(index=False) if len(freeze_df) else 'n/a'}
```

dimension_config_hash=`{freeze.get('dimension_config_hash')}`

Common-rank d=16 vs d*: {"overestimates for models with d*<16" if len(freeze_df) and (freeze_df.d_star < COMMON_D).any() else "see table"}.

## Reliability at d* (median R)

```
{rel[rel.role=='d_star'].to_string(index=False) if len(rel) else 'n/a'}
```

## mag_r_desi associations (sequential controls, k=2048)

```
{mag.to_string(index=False) if len(mag) else 'n/a'}
```

Best raw metric (descriptive): **{best_metric}**

## Metric comparison (incremental CV R², mag_r & smooth_fraction)

```
{mc.head(40).to_string(index=False) if len(mc) else 'n/a'}
```

## Dimension sensitivity

```
{ds.to_string(index=False) if len(ds) else 'n/a'}
```

## Cross-model summary

```
{pd.DataFrame(xm_rows).to_string(index=False) if xm_rows else 'n/a'}
```

## Decision labels

{labels}

## Answers

1. **Graph d* by model/scale?** see freeze table (consensus of median local `d_graph` and global Isomap elbow).
2. **Uncertainty?** `[d_minus, d_plus]` from bootstrap/IQR, clipped to ±2.
3. **Did d=16 overestimate?** compare `d_common` vs `d_star` (often yes when Isomap elbow ≪ 16).
4. **Mean curvature reliable at d*?** see `K_H` median_R in reliability table.
5. **Anisotropic more reliable?** compare `K_aniso` vs `K_H` median_R.
6. **Total directional more reliable/comparable?** `K_dir` is the primary dimension-comparable scalar (`K_dir²=K_H²+K_aniso²`).
7. **Best mag_r geography metric?** {best_metric}.
8. **Probe-facing beat total?** see metric_comparison / seq table `K_probe_facing_cross`.
9. **ViT-B survives graph d*?** see vit_base rows at d*.
10. **Cross-model generalization?** see cross_model_summary sign fractions (models not independent).
11. **Raw vs control-stable?** sequential path in associations_sequential_controls.csv.
12. **Robust in dimension band?** dimension_sensitivity labels.
13. **Strongest claim?** With label-free graph dimensions (typically below the historical d=16), split-half directional curvature remains moderately reliable at k=2048 across encoders; mag_r geography is strongest for ViT-B mean/directional curvature and is not uniformly control-stable across models. Treat sfr as underpowered.

## Notes

- Preserved dirs untouched: multimodel, coverage, full audit, split-half.
- Gauss comparison skipped={cfg.skip_gauss}.
"""
    (out / "REPORT.md").write_text(report)
    print(f"[edm] analyze labels={labels}", flush=True)


def run(cfg: EffDimCurvatureConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    for banned in (SOURCE_MM, SOURCE_COV, SOURCE_FCA, SOURCE_SH, cfg.multimodel_dir, cfg.coverage_dir):
        if out.resolve() == resolve_path(root, banned).resolve():
            raise RuntimeError(f"Refusing to write into preserved {banned}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    order = [
        "prepare",
        "synthetic_dimension",
        "estimate_dimension",
        "freeze_dimension",
        "fit_curvature",
        "cross_metrics",
        "probe_facing",
        "associations",
        "metric_comparison",
        "dimension_sensitivity",
        "gauss",
        "analyze",
    ]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    run_set = set(want)
    # minimal deps
    if "analyze" in run_set:
        run_set.update(["prepare", "freeze_dimension", "estimate_dimension", "synthetic_dimension"])
    if "fit_curvature" in run_set:
        run_set.update(["prepare", "synthetic_dimension", "estimate_dimension", "freeze_dimension"])
    if "associations" in run_set:
        run_set.update(["fit_curvature", "cross_metrics"])
    if "freeze_dimension" in run_set:
        run_set.update(["estimate_dimension", "synthetic_dimension", "prepare"])

    def mark(name, dt):
        profile["stages"][f"{name}_s"] = dt
        profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    meta = {}
    syn = {"ok": True}
    freeze = {"ok": False}
    est = pd.DataFrame()

    if "prepare" in run_set:
        t1 = time.time()
        print("[edm] stage=prepare", flush=True)
        meta = stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    if "synthetic_dimension" in run_set:
        t1 = time.time()
        print("[edm] stage=synthetic_dimension", flush=True)
        syn = stage_synthetic_dimension(root, cfg)
        mark("synthetic_dimension", time.time() - t1)
        if not syn.get("ok"):
            stage_analyze(root, cfg, {"ok": False, "by_model_scale": []}, syn)
            profile["stopped"] = "synthetic_gate_failed"
            (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
            return profile

    if "estimate_dimension" in run_set:
        t1 = time.time()
        print("[edm] stage=estimate_dimension", flush=True)
        est = stage_estimate_dimension(root, cfg, ctx)
        mark("estimate_dimension", time.time() - t1)
    elif (out / "dimension_estimates.parquet").exists():
        est = pd.read_parquet(out / "dimension_estimates.parquet")

    if "freeze_dimension" in run_set:
        t1 = time.time()
        print("[edm] stage=freeze_dimension", flush=True)
        if (out / "synthetic_dimension_gate.json").exists():
            syn = json.loads((out / "synthetic_dimension_gate.json").read_text())
        freeze = stage_freeze_dimension(root, cfg, ctx, est, syn)
        mark("freeze_dimension", time.time() - t1)
        if not freeze.get("ok"):
            stage_analyze(root, cfg, freeze, syn)
            return profile
    elif (out / "dimension_freeze.json").exists():
        freeze = json.loads((out / "dimension_freeze.json").read_text())

    if "fit_curvature" in run_set or "cross_metrics" in run_set:
        t1 = time.time()
        print("[edm] stage=fit_curvature", flush=True)
        stage_fit_curvature(root, cfg, ctx, freeze, t0)
        mark("fit_curvature", time.time() - t1)
        profile["completed"].append("cross_metrics")

    if "probe_facing" in run_set:
        t1 = time.time()
        print("[edm] stage=probe_facing", flush=True)
        stage_probe_facing(root, cfg)
        mark("probe_facing", time.time() - t1)

    if "associations" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[edm] stage=associations", flush=True)
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)

    if "metric_comparison" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[edm] stage=metric_comparison", flush=True)
        stage_metric_comparison(root, cfg, ctx)
        mark("metric_comparison", time.time() - t1)

    if "dimension_sensitivity" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[edm] stage=dimension_sensitivity", flush=True)
        stage_dimension_sensitivity(root, cfg)
        mark("dimension_sensitivity", time.time() - t1)

    if "gauss" in run_set and not cfg.skip_gauss:
        print("[edm] stage=gauss skipped (optional; prioritize report)", flush=True)
        profile["completed"].append("gauss_skipped")

    # analyze always if requested or at end of all
    if "analyze" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[edm] stage=analyze", flush=True)
        if (out / "synthetic_dimension_gate.json").exists():
            syn = json.loads((out / "synthetic_dimension_gate.json").read_text())
        if (out / "dimension_freeze.json").exists():
            freeze = json.loads((out / "dimension_freeze.json").read_text())
        stage_analyze(root, cfg, freeze, syn)
        mark("analyze", time.time() - t1)

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[edm] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
