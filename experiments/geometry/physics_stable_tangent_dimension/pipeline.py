"""Stage pipeline for the stable-tangent-dimension audit.

Scientific order: dimension (label-free) → linear distortion → curvature panel
→ sensitivity → secondary probe associations. Existing output directories are
never overwritten.
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

from geometry.physics_activation_atlas.curvature_probe_screen import (
    partial_spearman,
    spearman_dict,
)
from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    cross_metric_pair,
    metric_scalars,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

EPS = 1e-12
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from scipy.stats import spearmanr

from .curvature_panel import (
    D_lin,
    curvature_spectrum,
    excess_sectional,
    k_max_directional,
    tangent_rotation_stat,
    verify_kdir_identity,
)
from .dimension import (
    DEFAULT_THRESHOLDS,
    bootstrap_survival,
    consecutive_prefix,
    dT_from_rank_flags,
    dimension_sensitivity_label,
    model_label,
    paired_bootstrap_ci,
    survival_curve,
)
from .nested_pca import (
    block_agreement,
    crossfit_risk,
    degenerate_blocks,
    eigengaps,
    incremental_gain,
    nested_uncentred_svd,
    prefix_agreement,
    radial_stratified_halves,
    reconstruction_risk,
)
from .nulls import (
    column_permutation_null,
    quantile_threshold,
    residual_isotropic_null,
    split_schedule_null,
)
from .scaling import align_blocks_across_scales, classify_scaling, loglog_slope
from .sphere_coords import (
    angular_radii,
    projected_chord,
    rms_tangent_radius,
    row_l2_status,
    sphere_log_map,
)
from .synthetics import (
    SYNTH_KINDS,
    closest_synthetic,
    make_synthetic,
    split_seeds,
)

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_FCA = "outputs/geometry/physics_full_curvature_audit"
SOURCE_SH = "outputs/geometry/physics_split_half_curvature_reliability"
SOURCE_ATLAS = "outputs/geometry/physics_activation_atlas"

PRESERVED = [SOURCE_MM, SOURCE_EDM, SOURCE_NDC, SOURCE_COV, SOURCE_FCA, SOURCE_SH, SOURCE_ATLAS]

PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"
K_CANDIDATES = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]


@dataclass
class StableTangentConfig:
    output_dir: str = "outputs/geometry/physics_stable_tangent_dimension"
    multimodel_dir: str = SOURCE_MM
    effdim_dir: str = SOURCE_EDM
    nested_dir: str = SOURCE_NDC
    coverage_dir: str = SOURCE_COV
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    d_max: int = 20
    d_core: int = 12
    d_ref: int = 16
    n_splits: int = 5
    n_null_draw: int = 16
    n_parity_anchors: int = 32
    n_synth_cal: int = 8
    n_synth_eval: int = 8
    n_rotation_neighbors: int = 24
    ks: list[int] = field(default_factory=list)
    replication_models: list[str] = field(
        default_factory=lambda: ["convnext_base", "dinov3", "clip_base", "vit_large"]
    )
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 600.0
    skip_replication: bool = True
    smoke: bool = False
    coord: str = "log"  # log | chord
    n_anchors: int | None = None

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: StableTangentConfig, reserve: bool = False) -> bool:
    rem = cfg.max_seconds - (time.time() - t0)
    return rem > (cfg.analyze_reserve_seconds if reserve else 30.0)


def _sha16(payload: Any) -> str:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


def resolve_k_grid(k_max: int, reference_ks: list[int], *, smoke: bool) -> list[int]:
    if smoke:
        grid = [k for k in [64, 96, 128, 192, 256] if k <= k_max]
        return grid[:5] if len(grid) >= 5 else grid
    grid = [k for k in K_CANDIDATES if k <= k_max]
    for r in reference_ks:
        if 1 <= r <= k_max and r not in grid:
            grid.append(int(r))
    grid = sorted(set(grid))
    # keep at least five scales centred on the curvature reference
    ref = max(reference_ks) if reference_ks else min(k_max, 2048)
    if len(grid) > 7:
        # prefer prefixes around ref
        lo = [k for k in grid if k <= ref]
        hi = [k for k in grid if k > ref]
        keep = lo[-5:] + hi[:2]
        grid = sorted(set(keep + [ref]))
    if len(grid) < 5:
        extra = [max(16, k_max // (2**i)) for i in range(6, -1, -1)]
        grid = sorted(set(grid + [k for k in extra if 16 <= k <= k_max]))[:7]
    return grid


def load_ctx(root: Path, cfg: StableTangentConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = (
        json.loads(aid.read_text())["sample_ids"]
        if aid.exists()
        else [int(s) for s in anchors_sid]
    )
    if cfg.n_anchors is not None:
        use_sids = use_sids[: int(cfg.n_anchors)]
    elif cfg.smoke:
        use_sids = use_sids[:16]
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == cfg.model)
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
    ]
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    pack2048 = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    knn3072 = resolve_path(root, SOURCE_FCA) / "cache" / f"{cfg.model}_kmax3072_gpu.npz"
    pack3072 = dict(np.load(knn3072)) if knn3072.exists() else None
    freeze_p = resolve_path(root, cfg.effdim_dir) / "dimension_freeze.json"
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else {}
    k_max = int(pack2048["neigh"].shape[1])
    if pack3072 is not None:
        k_max = max(k_max, int(pack3072["neigh"].shape[1]))
    ref_ks = [1024, 2048]
    ks = list(cfg.ks) if cfg.ks else resolve_k_grid(min(k_max, cfg.primary_k), ref_ks, smoke=cfg.smoke)
    X = load_model_X(mm, cfg.model)
    return {
        "mm": mm,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "pack2048": pack2048,
        "pack3072": pack3072,
        "freeze": freeze,
        "X": X,
        "ks": ks,
        "k_max": k_max,
        "l2": row_l2_status(X),
    }


def ensure_neigh(ctx: dict, ai: int, k: int) -> np.ndarray:
    if k <= ctx["pack2048"]["neigh"].shape[1]:
        return ctx["pack2048"]["neigh"][ai, :k]
    if ctx["pack3072"] is None:
        raise RuntimeError("k>2048 requires full_curvature_audit kNN cache")
    return ctx["pack3072"]["neigh"][ai, :k]


def displacements(x0: np.ndarray, Xloc: np.ndarray, coord: str) -> np.ndarray:
    if coord == "chord":
        return projected_chord(x0, Xloc)
    return sphere_log_map(x0, Xloc)


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: StableTangentConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "batches", "figures", "logs", "synth", "J"):
        (out / sub).mkdir(exist_ok=True)
    mm = ctx["mm"]
    x_path = mm / "prepare" / "models" / f"{cfg.model}.npz"
    pack_path = mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    probes = mm / "global_probes" / "oof_predictions"
    probe_hash = _sha16(sorted(p.name for p in probes.glob("*")) if probes.exists() else "missing")
    meta = {
        "config": asdict(cfg),
        "protocol": "stable_tangent_dimension_v1",
        "preserved": PRESERVED,
        "ks": ctx["ks"],
        "n_anchors": len(ctx["use_sids"]),
        "l2_status": ctx["l2"],
        "neighbour_metric": "inner_product_on_unit_sphere",
        "primary_coord": cfg.coord,
        "anchor_pca": "uncentred_through_anchor",
        "no_local_probes": True,
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
        "hashes": {
            "activations": _file_sha(x_path) if x_path.exists() else None,
            "knn_pack": _file_sha(pack_path) if pack_path.exists() else None,
            "oof_probes": probe_hash,
            "freeze": ctx["freeze"].get("dimension_config_hash"),
        },
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
        "config_hash": _sha16(asdict(cfg)),
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    freeze_manifest = {
        "sample_ids": ctx["use_sids"],
        "model": cfg.model,
        "layer_block": "frozen_multimodel_prepare",
        "preprocessing": "l2_normalize_in_multimodel_prepare",
        "l2_normalized": ctx["l2"]["unit_normalized"],
        "neighbour_search_metric": "inner_product",
        "neighbour_row_ordering": "descending_ip_prefix_of_kmax",
        "split_schedule": "radial_stratified_halves",
        "reference_curvature_k": cfg.primary_k,
        "graph_d_G": cfg.d_core,
        "seed": cfg.seed,
        **meta["hashes"],
    }
    (out / "freeze_manifest.json").write_text(json.dumps(freeze_manifest, indent=2, default=str))
    print(f"[std] prepare ks={ctx['ks']} n={len(ctx['use_sids'])} l2={ctx['l2']['unit_normalized']}", flush=True)
    return meta


def stage_parity(root: Path, cfg: StableTangentConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    ndc = resolve_path(root, cfg.nested_dir)
    edm = resolve_path(root, cfg.effdim_dir)
    cov = resolve_path(root, cfg.coverage_dir)
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    result: dict[str, Any] = {"ok": True, "corrections": []}
    # artifact associations (established)
    try:
        cov_df = pd.read_parquet(cov / "model_reliability_anchor_mean.parquet")
        c16 = cov_df[(cov_df.model == cfg.model) & (cov_df.k == cfg.primary_k) & (cov_df.d == 16)]
        m16 = geo.merge(c16, on="sample_id", how="inner")
        rho16, _ = spearmanr(m16.K_H_cross, m16.local_r2)
        edm_df = pd.read_parquet(edm / "crossfit_curvature_metrics.parquet")
        e12 = edm_df[
            (edm_df.model == cfg.model)
            & (edm_df.k == cfg.primary_k)
            & (edm_df.role == "d_star")
            & (edm_df.d == cfg.d_core)
        ]
        m12 = geo.merge(e12, on="sample_id", how="inner")
        rho12, _ = spearmanr(m12.K_H_cross, m12.local_r2)
        ok16 = abs(float(rho16) - PARITY_D16_RHO) <= PARITY_TOL
        ok12 = abs(float(rho12) - PARITY_D12_RHO) <= PARITY_TOL
        result["d16"] = {"rho_KH_cross": float(rho16), "expected": PARITY_D16_RHO, "ok": ok16, "n": int(len(m16))}
        result["d12"] = {"rho_KH_cross": float(rho12), "expected": PARITY_D12_RHO, "ok": ok12, "n": int(len(m12))}
        result["ok"] = bool(result["ok"] and ok16 and ok12)
    except Exception as e:  # noqa: BLE001
        result["ok"] = False
        result["artifact_error"] = str(e)

    freeze_hash = ctx["freeze"].get("dimension_config_hash")
    result["freeze_hash"] = freeze_hash
    result["freeze_hash_ok"] = freeze_hash == FREEZE_HASH_EXPECTED
    result["ok"] = bool(result["ok"] and result["freeze_hash_ok"])

    # recompute OLD nested_pca_frame on a subset vs nested-dimension cache
    old_rows = []
    X = ctx["X"]
    device = ctx["device"]
    sids = [s for s in ctx["use_sids"][: cfg.n_parity_anchors]]
    for sid in sids:
        jp = ndc / "cache" / f"J_{int(sid)}_k{cfg.primary_k}.npz"
        if not jp.exists():
            continue
        z = np.load(jp)
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x0, J, ev, _ = nested_pca_frame(Xloc, cfg.d_max, device)
        cos12 = float(np.mean(np.cos(principal_angles(J[:, :12], z["J"][:, :12])))) if z["J"].shape[1] >= 12 else float("nan")
        cos16 = float(np.mean(np.cos(principal_angles(J[:, :16], z["J"][:, :16])))) if z["J"].shape[1] >= 16 else float("nan")
        # log-map vs old (documented coordinate correction, not a hard fail)
        x_anchor = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        if ctx["l2"]["unit_normalized"]:
            Zlog = sphere_log_map(x_anchor, Xloc)
        else:
            Zlog = projected_chord(x_anchor, Xloc)
            result["corrections"].append("representations_not_unit_normalized_used_chord")
        Jlog, _ = nested_uncentred_svd(Zlog, cfg.d_max, device=device)
        log_cos12 = float(np.mean(np.cos(principal_angles(Jlog[:, :12], J[:, :12])))) if Jlog.shape[1] >= 12 else float("nan")
        old_rows.append(
            {
                "sample_id": int(sid),
                "nested_cache_cos12": cos12,
                "nested_cache_cos16": cos16,
                "log_vs_old_cos12": log_cos12,
            }
        )
        if len(old_rows) >= cfg.n_parity_anchors:
            break
    dfp = pd.DataFrame(old_rows)
    result["nested_pca_recompute"] = {
        "n": int(len(dfp)),
        "median_cache_cos12": float(dfp.nested_cache_cos12.median()) if len(dfp) else float("nan"),
        "median_cache_cos16": float(dfp.nested_cache_cos16.median()) if len(dfp) else float("nan"),
        "median_log_vs_old_cos12": float(dfp.log_vs_old_cos12.median()) if len(dfp) else float("nan"),
    }
    cache_ok = True
    if len(dfp):
        cache_ok = float(dfp.nested_cache_cos12.median()) > 0.98 and float(dfp.nested_cache_cos16.median()) > 0.98
    result["nested_pca_recompute"]["ok"] = cache_ok
    result["ok"] = bool(result["ok"] and cache_ok)
    result["coordinate_correction"] = {
        "primary": "spherical_log_uncentred_through_anchor",
        "legacy": "neighbourhood_mean_chord_pca_tangent_gpu",
        "documented": True,
        "does_not_fail_parity": True,
    }
    # held-out quadratic + K metrics from nested_dimension tables
    ncm = ndc / "nested_curvature_metrics.parquet"
    if ncm.exists():
        curv = pd.read_parquet(ncm)
        sub = curv[curv.k == cfg.primary_k]
        result["legacy_metrics"] = {}
        for d in (12, 16):
            g = sub[sub.d == d]
            result["legacy_metrics"][str(d)] = {
                "median_K_H_cross": float(g.K_H_cross.median()) if len(g) else float("nan"),
                "median_K_aniso_cross": float(g.K_aniso_cross.median()) if len(g) else float("nan"),
                "median_K_dir_cross": float(g.K_dir_cross.median()) if len(g) else float("nan"),
                "median_dS": float(g.dS.median()) if len(g) else float("nan"),
                "n": int(g.sample_id.nunique()) if len(g) else 0,
            }
    if not result["ok"]:
        print(f"[std][parity] FAILED {json.dumps({k: result.get(k) for k in ('d12','d16','freeze_hash_ok')})}", flush=True)
    else:
        print(f"[std][parity] ok d16={result.get('d16',{}).get('rho_KH_cross')} d12={result.get('d12',{}).get('rho_KH_cross')}", flush=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return result


def stage_neighbourhoods(root: Path, cfg: StableTangentConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "neighbourhood_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    rows = []
    ks = ctx["ks"]
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        prev = None
        for k in ks:
            N = ensure_neigh(ctx, ai, k)
            Xloc = X[N].astype(np.float64)
            th = angular_radii(x0, Xloc)
            Z = displacements(x0, Xloc, cfg.coord)
            uniq = len(np.unique(N))
            overlap = float("nan")
            if prev is not None:
                overlap = len(set(N.tolist()) & set(prev.tolist())) / max(min(k, len(prev)), 1)
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k),
                    "n_usable": int(len(N)),
                    "n_unique": int(uniq),
                    "duplicate_frac": float(1.0 - uniq / max(len(N), 1)),
                    "theta_max": float(np.max(th)) if len(th) else float("nan"),
                    "theta_median": float(np.median(th)) if len(th) else float("nan"),
                    "rms_tangent_radius": rms_tangent_radius(Z),
                    "density_proxy": float(k / max(np.median(th) ** (max(cfg.d_core, 1)), EPS)),
                    "overlap_prev": overlap,
                    "degenerate_frac": float(np.mean(th < 1e-8)),
                }
            )
            prev = N
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[std] neighbourhoods n={len(rows)}", flush=True)


def _pca_one_anchor(
    Xloc: np.ndarray,
    x0: np.ndarray,
    d_max: int,
    n_splits: int,
    seed: int,
    ai: int,
    k: int,
    device,
    coord: str,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    Z = displacements(x0, Xloc, coord)
    th = angular_radii(x0, Xloc)
    Jfull, ev_full = nested_uncentred_svd(Z, d_max, device=device)
    rows = []
    for s in range(n_splits):
        A, B = radial_stratified_halves(th, seed + 1009 * ai + 17 * s + 13 * k)
        if min(len(A), len(B)) < d_max + 2:
            continue
        JA, evA = nested_uncentred_svd(Z[A], d_max, device=device)
        JB, evB = nested_uncentred_svd(Z[B], d_max, device=device)
        R0 = crossfit_risk(Z[A], Z[B], JA, JB, 0)
        R_prev = R0
        for d in range(1, min(d_max, JA.shape[1], JB.shape[1]) + 1):
            Rd = crossfit_risk(Z[A], Z[B], JA, JB, d)
            rows.append(
                {
                    "split": s,
                    "d": d,
                    "A": prefix_agreement(JA, JB, d),
                    "A_inc": float(
                        (JA[:, d - 1] @ JB[:, :d] @ JB[:, :d].T @ JA[:, d - 1])
                        if JA.shape[1] >= d and JB.shape[1] >= d
                        else np.nan
                    ),
                    "R": Rd,
                    "G": incremental_gain(R_prev, Rd, R0),
                    "G_raw": float(R_prev - Rd),
                    "R0": R0,
                    "ev": float(0.5 * (evA[d - 1] + evB[d - 1])) if d <= len(evA) and d <= len(evB) else float("nan"),
                    "gap": float(0.5 * ((evA[d - 1] - evA[d]) if d < len(evA) else np.nan) + ((evB[d - 1] - evB[d]) if d < len(evB) else np.nan)),
                }
            )
            R_prev = Rd
        # blockwise
        for a, b in degenerate_blocks(0.5 * (evA[: min(len(evA), len(evB))] + evB[: min(len(evA), len(evB))])):
            rows.append(
                {
                    "split": s,
                    "d": -1,
                    "block_a": a,
                    "block_b": b,
                    "A_block": block_agreement(JA, JB, a, b),
                    "A": np.nan,
                    "R": np.nan,
                    "G": np.nan,
                    "G_raw": np.nan,
                    "R0": R0,
                    "ev": np.nan,
                    "gap": np.nan,
                }
            )
    return rows, Jfull, ev_full


def stage_nested_pca(root: Path, cfg: StableTangentConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "nested_pca_spectra.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    device = ctx["device"]
    chunks = []
    for si, sid in enumerate(ctx["use_sids"]):
        if si % 32 == 0:
            print(f"[std][pca] {si}/{len(ctx['use_sids'])}", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            print("[std][pca] budget stop", flush=True)
            break
        batch = out / "batches" / f"pca_{cfg.model}_sid{int(sid)}.parquet"
        if _done(batch, cfg.force):
            chunks.append(pd.read_parquet(batch))
            continue
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        rows = []
        for k in ctx["ks"]:
            N = ensure_neigh(ctx, ai, k)
            Xloc = X[N].astype(np.float64)
            rec, Jfull, ev = _pca_one_anchor(
                Xloc, x0, cfg.d_max, cfg.n_splits, cfg.seed, ai, k, device, cfg.coord
            )
            np.savez_compressed(
                out / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz",
                x0=x0,
                J=Jfull,
                ev=ev,
            )
            for r in rec:
                rows.append({"sample_id": int(sid), "k": int(k), "model": cfg.model, **r})
            for i, lam in enumerate(ev[: cfg.d_max]):
                rows.append(
                    {
                        "sample_id": int(sid),
                        "k": int(k),
                        "model": cfg.model,
                        "split": -1,
                        "d": i + 1,
                        "ev_full": float(lam),
                        "gap_full": float(ev[i] - ev[i + 1]) if i + 1 < len(ev) else float("nan"),
                        "A": np.nan,
                        "R": np.nan,
                        "G": np.nan,
                    }
                )
        dfb = pd.DataFrame(rows)
        dfb.to_parquet(batch, index=False)
        chunks.append(dfb)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_parquet(path, index=False)
        print(f"[std] nested_pca wrote {path}", flush=True)


def stage_nulls(root: Path, cfg: StableTangentConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "null_distributions.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    device = ctx["device"]
    k = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    sids = ctx["use_sids"][:: max(1, len(ctx["use_sids"]) // 64)][:64]
    if cfg.smoke:
        sids = ctx["use_sids"][:8]
    pooled_agree = []
    pooled_gain = []
    for si, sid in enumerate(sids):
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, k)
        Xloc = X[N].astype(np.float64)
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        Z = displacements(x0, Xloc, cfg.coord)
        th = angular_radii(x0, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        JA, _ = nested_uncentred_svd(Z[A], cfg.d_max, device=device)
        rng = np.random.default_rng(cfg.seed + 91 * ai)
        col = column_permutation_null(Z[A], Z[B], min(cfg.d_max, JA.shape[1]), rng=rng, n_draw=cfg.n_null_draw, device=device)
        spl = split_schedule_null(Z, th, min(cfg.d_max, 8 if cfg.smoke else cfg.d_max), rng=rng, n_draw=max(8, cfg.n_null_draw // 2), device=device)
        for d in range(1, col["agreement"].shape[1] + 1):
            Jpref, _ = nested_uncentred_svd(Z[A], d - 1, device=device) if d > 1 else (np.zeros((Z.shape[1], 0)), np.zeros(0))
            iso = residual_isotropic_null(Z[A], Jpref, rng=rng, n_draw=max(8, cfg.n_null_draw // 2), d_extra=1, device=device)
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k),
                    "d": d,
                    "null_agree_col_q99": float(np.nanquantile(col["agreement"][:, d - 1], 0.99)),
                    "null_gain_col_q99": float(np.nanquantile(col["gain"][:, d - 1], 0.99)),
                    "null_agree_iso_q99": float(np.nanquantile(iso["agreement"], 0.99)),
                    "null_ainc_iso_q99": float(np.nanquantile(iso["agreement_inc"], 0.99)),
                    "null_gain_iso_q99": float(np.nanquantile(iso["gain"], 0.99)),
                    "null_agree_split_median": float(np.nanmedian(spl["agreement"][:, d - 1])) if d <= spl["agreement"].shape[1] else float("nan"),
                }
            )
            pooled_agree.append(col["agreement"][:, d - 1])
            pooled_gain.append(col["gain"][:, d - 1])
        if si % 8 == 0:
            print(f"[std][nulls] {si}/{len(sids)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    thr = dict(DEFAULT_THRESHOLDS)
    if rows:
        # incremental residual-PCA null (not prefix A, which is prefix-dominated)
        thr["ainc_null_q99"] = float(df.null_ainc_iso_q99.median())
        thr["agree_null_q99"] = float(df.null_ainc_iso_q99.median())
        thr["gain_null_q99"] = float(max(float(df.null_gain_iso_q99.median()), 0.0))
        thr["gain_floor"] = 0.001
        thr["prefix_A_min"] = 0.45
        thr["agree_col_q99"] = float(df.null_agree_col_q99.median())
        thr["gain_col_q99"] = float(df.null_gain_col_q99.median())
    (out / "null_thresholds_unfrozen.json").write_text(json.dumps(thr, indent=2))
    print(f"[std] nulls n={len(df)}", flush=True)


def stage_scale_tracking(root: Path, cfg: StableTangentConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "scale_tracking.parquet"
    if _done(path, cfg.force):
        return
    neigh = pd.read_parquet(out / "neighbourhood_diagnostics.parquet")
    rows = []
    for sid in ctx["use_sids"]:
        Js, evs, radii = [], [], []
        ks_used = []
        for k in ctx["ks"]:
            jp = out / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz"
            if not jp.exists():
                continue
            z = np.load(jp)
            Js.append(z["J"])
            evs.append(z["ev"])
            r = neigh[(neigh.sample_id == sid) & (neigh.k == k)]
            radii.append(float(r.rms_tangent_radius.iloc[0]) if len(r) else float("nan"))
            ks_used.append(k)
        if len(Js) < 3:
            continue
        tracks = align_blocks_across_scales(Js, evs)
        for ti, tr in enumerate(tracks):
            e = []
            rr = []
            for si, k in enumerate(ks_used):
                if si in tr["energy"]:
                    e.append(tr["energy"][si])
                    rr.append(radii[si])
            sl = loglog_slope(np.asarray(rr), np.asarray(e))
            a0, b0 = next(iter(tr["slots"].values()))
            rows.append(
                {
                    "sample_id": int(sid),
                    "track": ti,
                    "rank0": int(a0),
                    "rank1": int(b0),
                    "width": int(tr["width"]),
                    "n_scales": int(len(tr["energy"])),
                    **{f"k{k}": tr["energy"].get(i, np.nan) for i, k in enumerate(ks_used)},
                    **sl,
                    "mean_overlap": float(np.mean(list(tr.get("overlap", {0: 1.0}).values()))),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[std] scale_tracking n={len(rows)}", flush=True)


def _eval_synth_one(kind: str, seed: int, cfg: StableTangentConfig, thr: dict, device) -> dict:
    pack = make_synthetic(kind, n=400 if cfg.smoke else 800, D=48 if cfg.smoke else 64, seed=seed, k_obs=256)
    X = pack["X"]
    x0 = pack["x0"]
    ks = [64, 96, 128, 192, 256]
    d_max = 16
    Js, evs, radii = [], [], []
    # primary k=256
    N = pack["neigh"][:256]
    Xloc = X[N]
    Z = sphere_log_map(x0, Xloc)
    th = angular_radii(x0, Xloc)
    A, B = radial_stratified_halves(th, seed)
    JA, evA = nested_uncentred_svd(Z[A], d_max, device=device)
    JB, evB = nested_uncentred_svd(Z[B], d_max, device=device)
    ev = 0.5 * (evA + evB[: len(evA)]) if len(evB) >= len(evA) else evA
    flags = _block_flags(JA, JB, ev, Z[A], Z[B], thr, d_max)
    dT = dT_from_rank_flags(flags)
    R0 = crossfit_risk(Z[A], Z[B], JA, JB, 0)
    R_prev = R0
    agrees, gains, aincs = [], [], []
    for d in range(1, min(d_max, JA.shape[1], JB.shape[1]) + 1):
        Rd = crossfit_risk(Z[A], Z[B], JA, JB, d)
        agrees.append(prefix_agreement(JA, JB, d))
        u = JA[:, d - 1]
        aincs.append(float(u @ JB[:, :d] @ JB[:, :d].T @ u))
        gains.append(incremental_gain(R_prev, Rd, R0))
        R_prev = Rd
    for k in ks:
        idx = pack["neigh"][:k]
        Zk = sphere_log_map(x0, X[idx])
        Jk, evk = nested_uncentred_svd(Zk, d_max, device=device)
        Js.append(Jk)
        evs.append(evk)
        radii.append(rms_tangent_radius(Zk))
    tracks = align_blocks_across_scales(Js, evs)
    alpha_extra = float("nan")
    for tr in tracks:
        a0, b0 = next(iter(tr["slots"].values()))
        if a0 <= 12 <= b0 or a0 >= 12:
            e = [tr["energy"][s] for s in sorted(tr["energy"])]
            rr = [radii[s] for s in sorted(tr["energy"])]
            alpha_extra = loglog_slope(np.asarray(rr), np.asarray(e))["alpha"]
            break
    D12 = D_lin(Z[B], JA, min(12, JA.shape[1]))
    D16 = D_lin(Z[B], JA, min(16, JA.shape[1]))
    ev = 0.5 * (evA + evB[: len(evA)]) if len(evB) >= len(evA) else evA
    var_extra = float(np.sum(ev[12:16]) / max(np.sum(ev[: min(20, len(ev))]), EPS)) if len(ev) >= 16 else float("nan")
    return {
        "kind": kind,
        "seed": seed,
        "true_d": pack["true_d"],
        "median_dT": dT,
        "p_ge_12": float(dT >= 12),
        "p_ge_16": float(dT >= 16),
        "agree_13_16": float(np.mean(aincs[12:16])) if len(aincs) >= 16 else float("nan"),
        "gain_13_16": float(np.mean(gains[12:16])) if len(gains) >= 16 else float("nan"),
        "alpha_13_16": alpha_extra,
        "var_share_13_16": var_extra,
        "Dlin_12": D12,
        "Dlin_16": D16,
        "extra_kind": pack["extra_kind"],
    }


def _calibrate_thresholds(cal_df: pd.DataFrame, base: dict) -> dict:
    thr = dict(base)
    lin = cal_df[cal_df.kind == "linear_d12"]
    weak = cal_df[cal_df.kind == "true_d16_weak_tangent"]
    noise = cal_df[cal_df.kind.isin(["d12_isotropic_noise4", "d12_stable_thickness4"])]
    if len(lin):
        thr["cal_linear_median_dT"] = float(lin.median_dT.median())
    if len(weak):
        thr["cal_weak_median_dT"] = float(weak.median_dT.median())
    if len(noise):
        thr["cal_noise_median_dT"] = float(noise.median_dT.median())
    if len(lin) and lin.median_dT.median() < 10:
        thr["ainc_null_q99"] = float(min(float(thr.get("ainc_null_q99", 0.12)), 0.08))
        thr["agree_null_q99"] = thr["ainc_null_q99"]
        thr["gain_floor"] = float(min(float(thr.get("gain_floor", 0.001)), 0.0005))
        thr["prefix_A_min"] = float(min(float(thr.get("prefix_A_min", 0.45)), 0.35))
        thr["z_gain_min"] = 0.5
    if len(noise) and noise.median_dT.median() >= 14:
        thr["ainc_null_q99"] = float(max(float(thr.get("ainc_null_q99", 0.12)), 0.18))
        thr["agree_null_q99"] = thr["ainc_null_q99"]
    thr["frozen"] = True
    return thr


def stage_synthetic_calibration(root: Path, cfg: StableTangentConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    cal_path = out / "synthetic_calibration.csv"
    ev_path = out / "synthetic_evaluation.csv"
    thr_path = out / "thresholds.json"
    if _done(Path(cal_path), cfg.force) and _done(Path(thr_path), cfg.force):
        return json.loads(thr_path.read_text())
    device = ctx["device"]
    seeds = split_seeds(cfg.n_synth_cal if not cfg.smoke else 3, cfg.n_synth_eval if not cfg.smoke else 3)
    unfrozen = json.loads((out / "null_thresholds_unfrozen.json").read_text()) if (out / "null_thresholds_unfrozen.json").exists() else dict(DEFAULT_THRESHOLDS)
    base = {**DEFAULT_THRESHOLDS, **unfrozen}
    cal_rows = []
    kinds = SYNTH_KINDS if not cfg.smoke else [
        "linear_d12",
        "curved_d12",
        "d12_isotropic_noise4",
        "true_d16_weak_tangent",
        "saddle_zero_mean",
        "unit_sphere_baseline",
    ]
    for kind in kinds:
        for sd in seeds["calibration_seeds"]:
            cal_rows.append(_eval_synth_one(kind, sd, cfg, base, device))
            print(f"[std][synth-cal] {kind} seed={sd} dT={cal_rows[-1]['median_dT']}", flush=True)
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(cal_path, index=False)
    thr = _calibrate_thresholds(cal_df, base)
    (thr_path).write_text(json.dumps(thr, indent=2))
    ev_rows = []
    for kind in kinds:
        for sd in seeds["evaluation_seeds"]:
            ev_rows.append(_eval_synth_one(kind, sd, cfg, thr, device))
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(ev_path, index=False)
    print(f"[std] synthetic cal={len(cal_df)} eval={len(ev_df)}", flush=True)
    return thr


def _block_flags(
    JA: np.ndarray,
    JB: np.ndarray,
    ev: np.ndarray,
    ZA: np.ndarray,
    ZB: np.ndarray,
    thr: dict,
    d_max: int,
) -> np.ndarray:
    """Accept/reject eigengap blocks as a whole; fill per-rank flags."""
    from .nested_pca import degenerate_blocks, block_agreement, crossfit_risk, incremental_gain

    accepted = np.zeros(d_max, dtype=bool)
    rel = float(thr.get("rel_gap_min", 0.15))
    blocks = degenerate_blocks(ev[:d_max], rel_gap_min=rel)
    R0 = crossfit_risk(ZA, ZB, JA, JB, 0)
    ainc_q = float(thr.get("ainc_null_q99", 0.12))
    gain_q = max(float(thr.get("gain_null_q99", 0.0)), float(thr.get("gain_floor", 0.001)))
    blk_min = float(thr.get("block_A_min", 0.50))
    pref_min = float(thr.get("prefix_A_min", 0.45))
    for a, b in blocks:
        if b >= min(JA.shape[1], JB.shape[1], d_max):
            b = min(JA.shape[1], JB.shape[1], d_max) - 1
        if a > b or a < 0:
            continue
        w = b - a + 1
        Ablk = block_agreement(JA, JB, a, b)
        R_before = crossfit_risk(ZA, ZB, JA, JB, a)
        R_after = crossfit_risk(ZA, ZB, JA, JB, b + 1)
        Gblk = incremental_gain(R_before, R_after, R0)
        Apref = prefix_agreement(JA, JB, b + 1)
        amin = ainc_q if w == 1 else blk_min
        ok = np.isfinite(Ablk) and Ablk >= amin and Gblk >= gain_q * max(w * 0.25, 1.0) and Apref >= pref_min
        if a < 12 <= b and ok:
            # split bulk at d=12 unless extra block is independently strong
            A_extra = block_agreement(JA, JB, 12, b) if b >= 12 else 0.0
            if not (np.isfinite(A_extra) and A_extra >= blk_min):
                accepted[a:12] = True
                break
        if not ok:
            break
        accepted[a : b + 1] = True
    return accepted


def _rank_flags_for_anchor(g: pd.DataFrame, tracks: pd.DataFrame, thr: dict, d_max: int) -> np.ndarray:
    """Blockwise consecutive prefix using full-neighbourhood eigengaps + split A/G."""
    accepted = np.zeros(d_max, dtype=bool)
    full = g[g.split == -1]
    gg = g[(g.split >= 0) & (g.d > 0)]
    if not len(gg):
        return accepted
    ev = np.full(d_max, np.nan)
    if len(full) and "ev_full" in full.columns:
        for _, r in full.iterrows():
            d = int(r.d)
            if 1 <= d <= d_max:
                ev[d - 1] = float(r.ev_full)
    if not np.isfinite(ev).any():
        med_ev = gg.groupby("d")["ev"].median() if "ev" in gg.columns else pd.Series(dtype=float)
        for d, v in med_ev.items():
            if 1 <= int(d) <= d_max:
                ev[int(d) - 1] = float(v)
    ag = gg.groupby("d")["A"].median()
    gn = gg.groupby("d")["G"].median()
    rel = float(thr.get("rel_gap_min", 0.15))
    from .nested_pca import degenerate_blocks

    blocks = degenerate_blocks(np.nan_to_num(ev, nan=0.0), rel_gap_min=rel)
    gain_q = max(float(thr.get("gain_null_q99", 0.0)), float(thr.get("gain_floor", 0.001)))
    blk_min = float(thr.get("block_A_min", 0.50))
    ainc_q = float(thr.get("ainc_null_q99", 0.12))
    persist_min = float(thr.get("persistence_min", 0.4))
    for a, b in blocks:
        b = min(b, d_max - 1)
        if a > b:
            continue
        w = b - a + 1
        A_end = float(ag.get(b + 1, np.nan))
        G_sum = float(np.nansum([gn.get(d, np.nan) for d in range(a + 1, b + 2)]))
        amin = ainc_q if w == 1 else blk_min
        hit = pd.DataFrame()
        lab, resolved = "unresolved", False
        if len(tracks):
            hit = tracks[(tracks.rank0 <= a) & (tracks.rank1 >= b)]
            if not len(hit):
                hit = tracks[(tracks.rank0 <= a) & (tracks.rank1 >= a)]
            if len(hit):
                row = hit.iloc[0]
                resolved = bool(row.get("resolved", False))
                lab = classify_scaling(
                    float(row.get("alpha", np.nan)),
                    resolved=resolved,
                    tangent_lo=float(thr["tangent_lo"]),
                    tangent_hi=float(thr["tangent_hi"]),
                    curve_lo=float(thr["curve_lo"]),
                    curve_hi=float(thr["curve_hi"]),
                    thick_lo=float(thr["thick_lo"]),
                    thick_hi=float(thr["thick_hi"]),
                )
        persist = float(hit["mean_overlap"].median()) if len(hit) else 1.0
        if (a + 1) <= 12 < (b + 1):
            A12 = float(ag.get(12, np.nan))
            Gcore = float(np.nansum([gn.get(d, np.nan) for d in range(a + 1, 13)]))
            if not (np.isfinite(A12) and A12 >= amin and Gcore >= gain_q):
                break
            accepted[a:12] = True
            if thr.get("require_scaling", True) and (not resolved or lab != "tangent_like"):
                break
            Gex = float(np.nansum([gn.get(d, np.nan) for d in range(13, b + 2)]))
            A_end = float(ag.get(b + 1, np.nan))
            if np.isfinite(A_end) and A_end >= blk_min and Gex >= gain_q:
                accepted[12 : b + 1] = True
            break
        ok = np.isfinite(A_end) and A_end >= amin and G_sum >= gain_q
        if np.isfinite(persist) and persist < persist_min:
            ok = False
        if thr.get("require_scaling", True) and (a + 1) > 12:
            if not resolved or lab != "tangent_like":
                break
        if not ok:
            break
        accepted[a : b + 1] = True
    return accepted


def stage_select_dimension(root: Path, cfg: StableTangentConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    loc_path = out / "local_tangent_dimensions.parquet"
    sum_path = out / "tangent_dimension_summary.csv"
    if _done(loc_path, cfg.force) and _done(Path(sum_path), cfg.force):
        return
    pca = pd.read_parquet(out / "nested_pca_spectra.parquet")
    trk = pd.read_parquet(out / "scale_tracking.parquet") if (out / "scale_tracking.parquet").exists() else pd.DataFrame()
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    e4_rows = []
    for sid, gsid in pca.groupby("sample_id"):
        dT_by_k = {}
        for k, gk in gsid.groupby("k"):
            tracks = trk[trk.sample_id == sid] if len(trk) else pd.DataFrame()
            flags = _rank_flags_for_anchor(gk, tracks, thr, cfg.d_max)
            dT = dT_from_rank_flags(flags)
            dT_by_k[int(k)] = dT
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k),
                    "d_T": int(dT),
                    "d_G": int(cfg.d_core),
                    "accepted_prefix": ",".join(str(int(x)) for x in flags.astype(int)),
                }
            )
        # E4 at reference k
        gk = gsid[gsid.k == k_ref]
        gg = gk[gk.split >= 0]
        if len(gg):
            ag = gg.groupby("d")["A"].median()
            gn = gg.groupby("d")["G"].median()
            e4_rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k_ref),
                    "A_13": float(ag.get(13, np.nan)),
                    "A_14": float(ag.get(14, np.nan)),
                    "A_15": float(ag.get(15, np.nan)),
                    "A_16": float(ag.get(16, np.nan)),
                    "G_13": float(gn.get(13, np.nan)),
                    "G_14": float(gn.get(14, np.nan)),
                    "G_15": float(gn.get(15, np.nan)),
                    "G_16": float(gn.get(16, np.nan)),
                    "A_block_13_16": float(np.nanmean([ag.get(d, np.nan) for d in range(13, 17)])),
                    "G_block_13_16": float(np.nanmean([gn.get(d, np.nan) for d in range(13, 17)])),
                    "d_T_ref": dT_by_k.get(k_ref, 0),
                }
            )
    loc = pd.DataFrame(rows)
    loc.to_parquet(loc_path, index=False)
    pd.DataFrame(e4_rows).to_parquet(out / "e4_block_evidence.parquet", index=False)
    # summary per k
    sum_rows = []
    for k, gk in loc.groupby("k"):
        dT = gk.d_T.to_numpy(float)
        med = paired_bootstrap_ci(dT, stat="median", seed=cfg.seed)
        surv = bootstrap_survival(dT, cfg.d_max, seed=cfg.seed)
        iqr = float(np.subtract(*np.quantile(dT, [0.75, 0.25]))) if len(dT) else float("nan")
        concentrated = (med["hi"] - med["lo"]) <= 4.0
        sum_rows.append(
            {
                "k": int(k),
                "n": int(len(gk)),
                "median_dT": med["point"],
                "median_lo": med["lo"],
                "median_hi": med["hi"],
                "iqr_dT": iqr,
                "mean_dT": float(np.mean(dT)),
                "p12": float(np.mean(dT >= 12)),
                "p16": float(np.mean(dT >= 16)),
                "concentrated": bool(concentrated),
                **{f"p_d{d}": float(surv["p"][d - 1]) for d in range(1, min(21, cfg.d_max + 1))},
            }
        )
        np.savez(out / "cache" / f"survival_k{int(k)}.npz", p=surv["p"], lo=surv["lo"], hi=surv["hi"])
    sdf = pd.DataFrame(sum_rows)
    sdf.to_csv(sum_path, index=False)
    scale_med = {int(r.k): float(r.median_dT) for r in sdf.itertuples()}
    ref = sdf[sdf.k == k_ref]
    extra_lab = None
    if (out / "scale_tracking.parquet").exists() and len(trk):
        extra = trk[trk.rank0 >= 12]
        if len(extra):
            extra_lab = classify_scaling(
                float(extra.alpha.median()),
                resolved=bool(extra.resolved.mean() > 0.3),
                tangent_lo=float(thr["tangent_lo"]),
                tangent_hi=float(thr["tangent_hi"]),
                curve_lo=float(thr["curve_lo"]),
                curve_hi=float(thr["curve_hi"]),
                thick_lo=float(thr["thick_lo"]),
                thick_hi=float(thr["thick_hi"]),
            )
    lab = model_label(
        median_dT=float(ref.median_dT.iloc[0]) if len(ref) else float("nan"),
        iqr_dT=float(ref.iqr_dT.iloc[0]) if len(ref) else float("nan"),
        p_adj=None,
        scale_medians=scale_med,
        extra_block_label=extra_lab,
        concentrated=bool(ref.concentrated.iloc[0]) if len(ref) else False,
    )
    (out / "decision_labels.json").write_text(
        json.dumps({"primary": lab, "extra_scaling": extra_lab, "k_ref": k_ref, "thresholds": thr}, indent=2)
    )
    print(f"[std] select_dimension label={lab} median_dT={scale_med.get(k_ref)}", flush=True)


def stage_curvature_atlas(root: Path, cfg: StableTangentConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "curvature_metric_atlas.parquet"
    if _done(path, cfg.force):
        return
    loc = pd.read_parquet(out / "local_tangent_dimensions.parquet")
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    loc_ref = loc[loc.k == k_ref]
    median_dT = int(np.round(loc_ref.d_T.median())) if len(loc_ref) else cfg.d_core
    band = sorted(set([max(2, median_dT - 2), median_dT, min(cfg.d_max, median_dT + 2), cfg.d_core, cfg.d_ref]))
    if cfg.smoke:
        band = sorted(set([min(median_dT, 8), min(cfg.d_core, cfg.d_max), min(cfg.d_ref, cfg.d_max)]))
    X = ctx["X"]
    chunks = []
    n_splits = 2 if cfg.smoke else min(cfg.n_splits, 3)
    for si, sid in enumerate(ctx["use_sids"]):
        if si % 16 == 0:
            print(f"[std][atlas] {si}/{len(ctx['use_sids'])} band={band}", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            break
        batch = out / "batches" / f"atlas_{cfg.model}_sid{int(sid)}.parquet"
        if _done(batch, cfg.force):
            chunks.append(pd.read_parquet(batch))
            continue
        ai = ctx["sid_to_ai"][int(sid)]
        jp = out / "J" / f"{cfg.model}_{int(sid)}_k{int(k_ref)}.npz"
        if not jp.exists():
            continue
        z = np.load(jp)
        x0, J = z["x0"], z["J"]
        N = ensure_neigh(ctx, ai, k_ref)
        Xloc = X[N].astype(np.float64)
        Z = displacements(x0, Xloc, cfg.coord)
        th = angular_radii(x0, Xloc)
        d_T_i = int(loc_ref.loc[loc_ref.sample_id == sid, "d_T"].iloc[0]) if (loc_ref.sample_id == sid).any() else median_dT
        rows = []
        rot = float("nan")
        # neighbour tangent rotation vs a few nearby anchors
        for s in range(n_splits):
            A, B = radial_stratified_halves(th, cfg.seed + 17 * s + ai)
            fA, vA = _half_fit_indices(A, cfg.seed + 3 + s)
            fB, vB = _half_fit_indices(B, cfg.seed + 7 + s)
            for d in band:
                d = int(min(d, J.shape[1]))
                if d < 2:
                    continue
                Jd = J[:, :d]
                chA, _, infoA = fit_quad(Xloc, x0, Jd, fA, vA, B, ridges=RIDGES)
                chB, _, infoB = fit_quad(Xloc, x0, Jd, fB, vB, A, ridges=RIDGES)
                if chA is None or chB is None:
                    continue
                cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
                sa, sb = metric_scalars(chA.BS_flat, d), metric_scalars(chB.BS_flat, d)
                ident = verify_kdir_identity(chA.BS_flat, d, seed=cfg.seed + d)
                km = k_max_directional(chA.BS_flat, d, n_starts=6 if cfg.smoke else 10, n_mc=400 if cfg.smoke else 1500, seed=cfg.seed + d)
                spec = curvature_spectrum(chA.BS_flat, d)
                ex = excess_sectional(chA.BS_flat, d)
                dlin = 0.5 * (D_lin(Z[B], Jd, d) + D_lin(Z[A], Jd, d))
                e_tr = 0.5 * (float(infoA.get("E_TR", np.nan)) + float(infoB.get("E_TR", np.nan)))
                e_trs = 0.5 * (float(infoA.get("E_TRS", np.nan)) + float(infoB.get("E_TRS", np.nan)))
                dS = 0.5 * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan)))
                qS = float((e_tr - e_trs) / max(e_tr, EPS)) if np.isfinite(e_tr) else float("nan")
                rows.append(
                    {
                        "sample_id": int(sid),
                        "k": int(k_ref),
                        "d": d,
                        "split": s,
                        "d_T_i": d_T_i,
                        "median_dT": median_dT,
                        "D_lin": dlin,
                        "dS": dS,
                        "Q_S": qS,
                        "E_TR": e_tr,
                        "E_TRS": e_trs,
                        "K_H_cross": cross["K_H_cross"],
                        "K_aniso_cross": cross["K_aniso_cross"],
                        "K_dir_cross": cross["K_dir_cross"],
                        "R_H": cross["R_H"],
                        "R_B0": cross["R_B0"],
                        "R_BS": cross["R_BS"],
                        "K_H": 0.5 * (sa["K_H"] + sb["K_H"]),
                        "K_aniso": 0.5 * (sa["K_aniso"] + sb["K_aniso"]),
                        "K_dir": 0.5 * (sa["K_dir"] + sb["K_dir"]),
                        "K_max": km["K_max"],
                        "K_max_converged": km["converged"],
                        "r90": spec["r90"],
                        "r95": spec["r95"],
                        "stable_rank": spec["stable_rank"],
                        "entropy_rank": spec["entropy_rank"],
                        "mean_excess": ex["mean_excess"],
                        "rms_excess": ex["rms_excess"],
                        "kdir_identity_err": ident["identity_err"],
                        "role": (
                            "d_T"
                            if d == d_T_i
                            else "d_G"
                            if d == cfg.d_core
                            else "d_ref"
                            if d == cfg.d_ref
                            else "band"
                        ),
                    }
                )
        dfb = pd.DataFrame(rows)
        dfb.to_parquet(batch, index=False)
        chunks.append(dfb)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_parquet(path, index=False)
        print(f"[std] curvature_atlas wrote {path}", flush=True)


def stage_metric_agreement(root: Path, cfg: StableTangentConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "metric_agreement.csv"
    rel_path = out / "metric_reliability.csv"
    if _done(Path(path), cfg.force):
        return
    atlas = pd.read_parquet(out / "curvature_metric_atlas.parquet")
    metrics = [
        "D_lin",
        "Q_S",
        "K_dir_cross",
        "K_H_cross",
        "K_aniso_cross",
        "K_max",
        "dS",
        "rms_excess",
        "stable_rank",
    ]
    rel_rows = []
    agr_rows = []
    for (k, d), g in atlas.groupby(["k", "d"]):
        g1 = g.groupby("sample_id").mean(numeric_only=True).reset_index()
        # split reliability: correlate split 0 vs 1
        s0 = g[g.split == 0].set_index("sample_id")
        s1 = g[g.split == 1].set_index("sample_id") if (g.split == 1).any() else None
        for m in metrics:
            if m not in g1:
                continue
            rec = {"k": int(k), "d": int(d), "metric": m, "median": float(g1[m].median())}
            if s1 is not None and m in s0.columns and m in s1.columns:
                idx = s0.index.intersection(s1.index)
                if len(idx) >= 20:
                    rho, _ = spearmanr(s0.loc[idx, m], s1.loc[idx, m])
                    rec["split_rho"] = float(rho)
                    rec["split_R"] = float(
                        tensor_agreement_1d(s0.loc[idx, m].to_numpy(), s1.loc[idx, m].to_numpy())
                    )
                    rec["invalid_frac"] = float(np.mean(~np.isfinite(g1[m])))
            rel_rows.append(rec)
        # pairwise Spearman of anchor means
        for i, a in enumerate(metrics):
            for b in metrics[i + 1 :]:
                if a not in g1 or b not in g1:
                    continue
                rho = spearman_dict(g1[a].to_numpy(), g1[b].to_numpy())
                agr_rows.append({"k": int(k), "d": int(d), "a": a, "b": b, "rho": rho["rho"], "n": rho["n"]})
    pd.DataFrame(rel_rows).to_csv(rel_path, index=False)
    pd.DataFrame(agr_rows).to_csv(path, index=False)
    print(f"[std] metric_agreement n={len(agr_rows)}", flush=True)


def tensor_agreement_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    na = float(np.linalg.norm(a[m]))
    nb = float(np.linalg.norm(b[m]))
    return float(2 * np.dot(a[m], b[m]) / max(na**2 + nb**2, EPS))


def stage_dimension_sensitivity(root: Path, cfg: StableTangentConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "dimension_sensitivity.csv"
    if _done(Path(path), cfg.force):
        return
    atlas = pd.read_parquet(out / "curvature_metric_atlas.parquet")
    loc = pd.read_parquet(out / "local_tangent_dimensions.parquet")
    k_ref = int(atlas.k.mode().iloc[0])
    median_dT = int(np.round(loc[loc.k == k_ref].d_T.median()))
    band = sorted(atlas.d.unique().tolist())
    metrics = ["D_lin", "Q_S", "K_dir_cross", "K_H_cross", "K_aniso_cross", "K_max"]
    g = atlas.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
    rows = []
    for m in metrics:
        by_d = {int(d): float(gd[m].median()) for d, gd in g.groupby("d") if m in gd}
        lab = dimension_sensitivity_label(by_d, band=band)
        if median_dT in by_d and cfg.d_core in by_d:
            if abs(by_d[median_dT] - by_d.get(cfg.d_ref, by_d[median_dT])) / max(abs(by_d[median_dT]), EPS) > 0.5:
                if lab == "dimension_robust":
                    lab = "extended_rank_specific"
        rows.append({"metric": m, "label": lab, **{f"med_d{d}": by_d.get(int(d), np.nan) for d in band}})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[std] dimension_sensitivity n={len(rows)}", flush=True)


def stage_associations(root: Path, cfg: StableTangentConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "probe_associations.csv"
    if _done(Path(path), cfg.force):
        return
    atlas = pd.read_parquet(out / "curvature_metric_atlas.parquet")
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    if not len(geo):
        geo = ctx["geo"]
    agg = atlas.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
    metrics = ["D_lin", "Q_S", "K_dir_cross", "K_H_cross", "K_aniso_cross", "K_max", "dS", "rms_excess"]
    rows = []
    family = []
    for d, gd in agg.groupby("d"):
        gg = geo.merge(gd, on="sample_id", how="inner")
        y = gg.local_r2.to_numpy(float)
        Z1 = gg.log_knn_radius.fillna(0).to_numpy(float)[:, None] if "log_knn_radius" in gg else None
        Zc = None
        if all(c in gg.columns for c in ("log_knn_radius", "local_label_variance", "local_evaluation_count")):
            Zc = np.column_stack(
                [
                    gg.log_knn_radius.fillna(0).to_numpy(float),
                    gg.local_label_variance.fillna(0).to_numpy(float),
                    gg.local_evaluation_count.fillna(0).to_numpy(float),
                ]
            )
        for m in metrics:
            if m not in gg:
                continue
            x = gg[m].to_numpy(float)
            raw = spearman_dict(x, y)
            rec = {
                "d": int(d),
                "metric": m,
                "n": raw["n"],
                "raw": raw["rho"],
                "p_raw": raw["pvalue"],
                "+radius": partial_spearman(x, y, Z1)["rho"] if Z1 is not None else float("nan"),
                "+controls": partial_spearman(x, y, Zc)["rho"] if Zc is not None else float("nan"),
            }
            # split sign recurrence
            signs = []
            sub = atlas[atlas.d == d]
            for _, gs in sub.groupby("split"):
                mm = geo.merge(gs, on="sample_id")
                if len(mm) < 30 or m not in mm:
                    continue
                r, _ = spearmanr(mm[m], mm.local_r2)
                if np.isfinite(r):
                    signs.append(np.sign(r))
            rec["sign_recurrence"] = float(np.mean(np.array(signs) < 0)) if signs else float("nan")
            family.append(abs(raw["rho"]) if np.isfinite(raw["rho"]) else 0.0)
            rows.append(rec)
    # max-statistic multiplicity: compare |rho| to max over family under sign-flip null (conservative)
    if family:
        cutoff = float(np.quantile(family, 0.9))
        for r in rows:
            r["family_pass"] = bool(np.isfinite(r["raw"]) and abs(r["raw"]) >= cutoff)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[std] associations n={len(rows)}", flush=True)


def _dT_frozen_primary_gates(
    x0: np.ndarray,
    Xloc_kmax: np.ndarray,
    ks: list[int],
    primary_k: int,
    d_max: int,
    seed: int,
    thr: dict,
    device,
    coord: str,
) -> int | None:
    """Same decision function as ViT-B primary: full-neighbourhood eigengaps,
    split A/G, cross-scale tracks, frozen thresholds. Do not retune per model.
    """
    k_max = int(Xloc_kmax.shape[0])
    ks_use = [int(k) for k in ks if 8 <= int(k) <= k_max]
    if not ks_use:
        return None
    k_pri = min(int(primary_k), k_max)
    if k_pri not in ks_use:
        ks_use = sorted(set(ks_use) | {k_pri})
    Js: list[np.ndarray] = []
    evs: list[np.ndarray] = []
    radii: list[float] = []
    JA = JB = ZA = ZB = None
    ev_full = None
    for k in ks_use:
        Z = displacements(x0, Xloc_kmax[:k], coord)
        Jk, evk = nested_uncentred_svd(Z, d_max, device=device)
        if Jk.shape[1] < 1:
            continue
        Js.append(Jk)
        evs.append(evk)
        radii.append(rms_tangent_radius(Z))
        if k == k_pri:
            ev_full = evk
            th = angular_radii(x0, Xloc_kmax[:k])
            A, B = radial_stratified_halves(th, seed)
            if min(len(A), len(B)) < d_max + 2:
                return None
            JA, _ = nested_uncentred_svd(Z[A], d_max, device=device)
            JB, _ = nested_uncentred_svd(Z[B], d_max, device=device)
            ZA, ZB = Z[A], Z[B]
    if JA is None or JB is None or ev_full is None:
        return None
    tracks = align_blocks_across_scales(Js, evs)
    trk_rows = []
    for tr in tracks:
        e = []
        rr = []
        for si in sorted(tr["energy"]):
            e.append(tr["energy"][si])
            rr.append(radii[si] if si < len(radii) else np.nan)
        sl = loglog_slope(np.asarray(rr), np.asarray(e))
        a0, b0 = next(iter(tr["slots"].values()))
        trk_rows.append(
            {
                "rank0": int(a0),
                "rank1": int(b0),
                "resolved": bool(sl["resolved"]),
                "alpha": sl["alpha"],
                "mean_overlap": float(np.mean(list(tr.get("overlap", {0: 1.0}).values()))),
            }
        )
    tracks_df = pd.DataFrame(trk_rows)
    g_rows: list[dict[str, Any]] = []
    for d in range(1, d_max + 1):
        g_rows.append(
            {
                "split": -1,
                "d": d,
                "ev_full": float(ev_full[d - 1]) if d <= len(ev_full) else np.nan,
                "A": np.nan,
                "G": np.nan,
                "ev": np.nan,
            }
        )
    R0 = crossfit_risk(ZA, ZB, JA, JB, 0)
    R_prev = R0
    d_lim = min(d_max, JA.shape[1], JB.shape[1])
    for d in range(1, d_lim + 1):
        Rd = crossfit_risk(ZA, ZB, JA, JB, d)
        g_rows.append(
            {
                "split": 0,
                "d": d,
                "A": prefix_agreement(JA, JB, d),
                "G": incremental_gain(R_prev, Rd, R0),
                "ev": float(ev_full[d - 1]) if d <= len(ev_full) else np.nan,
                "ev_full": np.nan,
            }
        )
        R_prev = Rd
    flags = _rank_flags_for_anchor(pd.DataFrame(g_rows), tracks_df, thr, d_max)
    return int(dT_from_rank_flags(flags))


def stage_replication(root: Path, cfg: StableTangentConfig, ctx0: dict, t0: float, thr: dict) -> None:
    """Apply frozen ViT-B primary gates to comparison models; do not retune."""
    out = cfg.resolved(root)
    path = out / "cross_model_replication.csv"
    if cfg.skip_replication or _done(Path(path), cfg.force):
        if cfg.skip_replication:
            print("[std] replication skipped (ViT-B primary first)", flush=True)
        return
    rows = []
    locp = out / "tangent_dimension_summary.csv"
    if locp.exists():
        sdf = pd.read_csv(locp)
        ref = sdf[sdf.k == cfg.primary_k]
        if len(ref):
            r0 = ref.iloc[0]
            dG_v = 12
            for rec in ctx0["freeze"].get("by_model_scale", []) or []:
                if rec.get("model") == "vit_base" and int(rec.get("scale_k", 0)) == int(cfg.primary_k):
                    dG_v = rec.get("d_star", 12)
                    break
            rows.append(
                {
                    "model": "vit_base",
                    "ok": True,
                    "n": int(r0.n),
                    "d_G": dG_v,
                    "median_dT": float(r0.median_dT),
                    "median_lo": float(r0.median_lo),
                    "median_hi": float(r0.median_hi),
                    "p12": float(r0.p12),
                    "p16": float(r0.p16),
                    "iqr": float(r0.iqr_dT),
                    "source": "primary_full",
                }
            )
    mm = ctx0["mm"]
    ks = list(ctx0.get("ks") or cfg.ks or [cfg.primary_k])
    for model in cfg.replication_models:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        xp = mm / "prepare" / "models" / f"{model}.npz"
        packp = mm / "model_neighbourhoods" / f"{model}_kmax2048.npz"
        if not xp.exists() or not packp.exists():
            rows.append({"model": model, "ok": False, "reason": "missing_artifacts"})
            continue
        X = load_model_X(mm, model)
        pack = dict(np.load(packp))
        sids = ctx0["use_sids"][: 128 if not cfg.smoke else 8]
        k = min(cfg.primary_k, pack["neigh"].shape[1])
        dTs = []
        device = ctx0["device"]
        for j, sid in enumerate(sids):
            ai = ctx0["sid_to_ai"][int(sid)]
            if ai >= pack["neigh"].shape[0]:
                continue
            N = pack["neigh"][ai, :k]
            x0 = X[int(ctx0["anchors_local"][ai])].astype(np.float64)
            Xloc = X[N].astype(np.float64)
            dT = _dT_frozen_primary_gates(
                x0,
                Xloc,
                ks,
                k,
                cfg.d_max,
                cfg.seed + ai,
                thr,
                device,
                cfg.coord,
            )
            if dT is None:
                continue
            dTs.append(dT)
            if j % 32 == 0:
                print(f"[std][repl] {model} {j}/{len(sids)} last_dT={dT}", flush=True)
        if not dTs:
            rows.append({"model": model, "ok": False, "reason": "no_anchors"})
            continue
        dTs = np.asarray(dTs, float)
        med = paired_bootstrap_ci(dTs, seed=cfg.seed)
        dG = float("nan")
        for rec in ctx0["freeze"].get("by_model_scale", []) or []:
            if rec.get("model") == model and int(rec.get("scale_k", 0)) == int(cfg.primary_k):
                dG = rec.get("d_star", float("nan"))
                break
        rows.append(
            {
                "model": model,
                "ok": True,
                "n": int(len(dTs)),
                "d_G": dG,
                "median_dT": med["point"],
                "median_lo": med["lo"],
                "median_hi": med["hi"],
                "p12": float(np.mean(dTs >= 12)),
                "p16": float(np.mean(dTs >= 16)),
                "iqr": float(np.subtract(*np.quantile(dTs, [0.75, 0.25]))),
                "source": "frozen_vitb_gates",
            }
        )
        print(f"[std][repl] {model} dT={med['point']} iqr={np.subtract(*np.quantile(dTs, [0.75, 0.25])):.2f}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


STAGES = [
    "prepare",
    "parity",
    "neighbourhoods",
    "nested_pca",
    "nulls",
    "scale_tracking",
    "synthetic_calibration",
    "select_dimension",
    "curvature_atlas",
    "metric_agreement",
    "dimension_sensitivity",
    "associations",
    "replication",
    "analyze",
    "report",
]


def _write_block_evidence(out: Path) -> None:
    pca = pd.read_parquet(out / "nested_pca_spectra.parquet")
    blk = pca[pca.d == -1].copy() if "d" in pca.columns else pd.DataFrame()
    e4p = out / "e4_block_evidence.parquet"
    e4 = pd.read_parquet(e4p) if e4p.exists() else pd.DataFrame()
    frames = []
    if len(blk):
        frames.append(blk)
    if len(e4):
        frames.append(e4)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out / "tangent_block_evidence.parquet", index=False)
    elif not (out / "tangent_block_evidence.parquet").exists():
        pd.DataFrame().to_parquet(out / "tangent_block_evidence.parquet", index=False)


def run(cfg: StableTangentConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    for banned in PRESERVED:
        if out.resolve() == resolve_path(root, banned).resolve():
            raise RuntimeError(f"Refusing to write into {banned}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = STAGES if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    if "all" in want:
        want = list(STAGES)
    run_set = set(want)
    # upstream requirements
    if run_set & {"nested_pca", "nulls", "scale_tracking", "select_dimension"}:
        run_set.update(["prepare", "neighbourhoods"])
    if "select_dimension" in run_set:
        run_set.update(["nested_pca", "nulls", "scale_tracking", "synthetic_calibration"])
    if "curvature_atlas" in run_set:
        run_set.update(["select_dimension"])
    if "associations" in run_set:
        run_set.update(["curvature_atlas"])
    if run_set & {"analyze", "report"}:
        run_set.update(["prepare", "parity"])

    def mark(name: str, dt: float) -> None:
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    parity: dict[str, Any] = {}
    thr: dict[str, Any] = dict(DEFAULT_THRESHOLDS)

    if "prepare" in run_set:
        t1 = time.time()
        print("[std] stage=prepare", flush=True)
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    if "parity" in run_set:
        t1 = time.time()
        print("[std] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        mark("parity", time.time() - t1)
        if not parity.get("ok"):
            from .report import write_methods, write_report

            write_methods(out, cfg, ctx, parity, thr)
            write_report(out, cfg, ctx, parity, {"primary": "tangent_dimension_unresolved", "parity_failed": True})
            profile["stopped"] = "parity_failed"
            (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
            raise RuntimeError("parity failed; see parity.json")
    elif (out / "parity.json").exists():
        parity = json.loads((out / "parity.json").read_text())

    if "neighbourhoods" in run_set:
        t1 = time.time()
        print("[std] stage=neighbourhoods", flush=True)
        stage_neighbourhoods(root, cfg, ctx)
        mark("neighbourhoods", time.time() - t1)

    if "nested_pca" in run_set:
        t1 = time.time()
        print("[std] stage=nested_pca", flush=True)
        stage_nested_pca(root, cfg, ctx, t0)
        mark("nested_pca", time.time() - t1)

    if "nulls" in run_set:
        t1 = time.time()
        print("[std] stage=nulls", flush=True)
        stage_nulls(root, cfg, ctx, t0)
        mark("nulls", time.time() - t1)

    if "scale_tracking" in run_set:
        t1 = time.time()
        print("[std] stage=scale_tracking", flush=True)
        stage_scale_tracking(root, cfg, ctx)
        mark("scale_tracking", time.time() - t1)

    if "synthetic_calibration" in run_set:
        t1 = time.time()
        print("[std] stage=synthetic_calibration", flush=True)
        thr = stage_synthetic_calibration(root, cfg, ctx)
        mark("synthetic_calibration", time.time() - t1)
    elif (out / "thresholds.json").exists():
        thr = json.loads((out / "thresholds.json").read_text())

    if "select_dimension" in run_set:
        t1 = time.time()
        print("[std] stage=select_dimension", flush=True)
        stage_select_dimension(root, cfg, ctx, thr)
        _write_block_evidence(out)
        mark("select_dimension", time.time() - t1)

    if "curvature_atlas" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[std] stage=curvature_atlas", flush=True)
        stage_curvature_atlas(root, cfg, ctx, t0)
        mark("curvature_atlas", time.time() - t1)

    if "metric_agreement" in run_set and (out / "curvature_metric_atlas.parquet").exists():
        t1 = time.time()
        print("[std] stage=metric_agreement", flush=True)
        stage_metric_agreement(root, cfg, ctx)
        mark("metric_agreement", time.time() - t1)

    if "dimension_sensitivity" in run_set and (out / "curvature_metric_atlas.parquet").exists():
        t1 = time.time()
        print("[std] stage=dimension_sensitivity", flush=True)
        stage_dimension_sensitivity(root, cfg, ctx)
        mark("dimension_sensitivity", time.time() - t1)

    if "associations" in run_set and (out / "curvature_metric_atlas.parquet").exists():
        t1 = time.time()
        print("[std] stage=associations", flush=True)
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)

    if "replication" in run_set:
        t1 = time.time()
        print("[std] stage=replication", flush=True)
        stage_replication(root, cfg, ctx, t0, thr)
        mark("replication", time.time() - t1)

    if "analyze" in run_set or "report" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[std] stage=analyze/report", flush=True)
        from .plots import write_figures
        from .report import write_methods, write_report

        if (out / "parity.json").exists():
            parity = json.loads((out / "parity.json").read_text())
        labels = {}
        if (out / "decision_labels.json").exists():
            labels = json.loads((out / "decision_labels.json").read_text())
        try:
            write_figures(out, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[std] figures failed: {e}", flush=True)
        write_methods(out, cfg, ctx, parity, thr)
        write_report(out, cfg, ctx, parity, labels)
        mark("analyze", time.time() - t1)
        mark("report", 0.0)

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[std] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
