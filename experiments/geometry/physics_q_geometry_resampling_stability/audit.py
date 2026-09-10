"""Phase 0: frozen artifacts, sample-ID alignment, controlled-correlation parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_pointwise_residual_curvature_probe_relation.audit import aligned_outcomes, load_bundle
from geometry.physics_pointwise_residual_curvature_probe_relation.config import ExpConfig as ResidualCfg
from geometry.physics_pointwise_residual_curvature_probe_relation.io_util import file_meta, platonic_root, resolve_path

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    D,
    K,
    K_PRIME,
    MODEL,
    N_ANCHORS,
    PARITY_ATOL,
    PARITY_DMSE,
    PARITY_MSE_G,
    PARITY_R2,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_MM,
    SOURCE_NDC,
)
from .q_fit import production_mean_kh


def load_frozen() -> dict[str, Any]:
    cfg = ResidualCfg()
    bundle = load_bundle(cfg)
    df = aligned_outcomes(bundle)
    root = bundle["root"]
    mm = bundle["mm"]
    neigh_p = mm / "model_neighbourhoods" / f"{MODEL}_kmax{K}.npz"
    pack = dict(np.load(neigh_p))
    neigh = np.asarray(pack["neigh"], dtype=np.int64)
    sids = [int(s) for s in bundle["sids"][:N_ANCHORS]]
    # neighbour rows follow original 512-anchor order
    orig = [int(s) for s in bundle["orig_sids"]]
    sid_to_ai = {int(s): i for i, s in enumerate(orig)}
    order = [sid_to_ai[s] for s in sids]
    neigh = neigh[np.asarray(order, dtype=np.int64)]
    frames = []
    query_rows = []
    cmcla = resolve_path(root, SOURCE_CMCLA)
    ndc = resolve_path(root, SOURCE_NDC)
    for sid in sids:
        jp = ndc / "cache" / f"J_{int(sid)}_k{K}.npz"
        if not jp.exists():
            jp = cmcla / "geometry" / MODEL / f"J_{int(sid)}.npz"
        z = np.load(jp)
        frames.append((np.asarray(z["x0"], dtype=np.float64), np.asarray(z["J"], dtype=np.float64)))
        query_rows.append(int(bundle["sid_to_row"][int(sid)]))
    radii = []
    X = np.asarray(bundle["X"])
    for i, (x0, _) in enumerate(frames):
        N = neigh[i, :K]
        radii.append(float(np.median(np.linalg.norm(X[N] - x0[None, :], axis=1))))
    df = df[df.sample_id.astype(int).isin(sids)].drop_duplicates("sample_id")
    df = df.set_index("sample_id").loc[sids].reset_index()
    return {
        "bundle": bundle,
        "df": df,
        "X": X,
        "neigh": neigh,
        "frames": frames,
        "sids": sids,
        "query_rows": np.asarray(query_rows, dtype=np.int64),
        "orig_radius": np.asarray(radii, dtype=np.float64),
        "neigh_path": neigh_p,
        "root": root,
        "mm": mm,
        "cmcla": cmcla,
        "ndc": ndc,
    }


def reuse_manifest(ctx: dict) -> dict[str, Any]:
    root = ctx["root"]
    files = {
        "vit_b_embeddings": ctx["mm"] / "prepare" / "models" / f"{MODEL}.npz",
        "anchors": ctx["mm"] / "prepare" / "anchors.npz",
        "folds": ctx["mm"] / "sample_folds.parquet",
        "neighbourhoods": ctx["neigh_path"],
        "fcr": root / SOURCE_FCR / "tables" / f"{MODEL}_per_anchor_curvature.parquet",
        "cmcla_probes": ctx["cmcla"] / "probes" / f"{MODEL}_anchor_metrics.parquet",
        "q_impl": Path(__file__).resolve().parents[1] / "physics_activation_atlas" / "nested_dimension_curvature.py",
        "fit_quad": Path(__file__).resolve().parents[1] / "physics_activation_atlas" / "full_curvature_audit.py",
    }
    return {
        "model": MODEL,
        "target": CATALOG_FIELD,
        "d": D,
        "k": K,
        "k_prime": K_PRIME,
        "n_anchors": len(ctx["sids"]),
        "n_objects": int(ctx["X"].shape[0]),
        "align_by": "sample_id",
        "estimator": "nested_dimension_curvature._fit_rank → fit_quad → cross_metric_pair (unpacked)",
        "n_splits_production": 5,
        "production_seed": 0,
        "ridges": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0],
        "split_half": "disjoint 1024/1024; 80/20 ridge val inside half; no clamp",
        "controls": list(CONTROLS),
        "files": {k: file_meta(p) if Path(p).exists() else {"path": str(p), "exists": False} for k, p in files.items()},
        "no_decoder_training": True,
        "no_probe_refitting": True,
        "no_label_model_fitting": True,
    }


def association_parity(df: pd.DataFrame) -> dict[str, Any]:
    Z = control_matrix(df)
    specs = {
        "rho_ctl_KH_R2G": (df.K_H_cross, df.r2_G, PARITY_R2),
        "rho_ctl_KH_MSEG": (df.K_H_cross, df.mse_G, PARITY_MSE_G),
        "rho_ctl_KH_DeltaAdapt": (df.K_H_cross, df.delta_adapt, PARITY_DMSE),
    }
    out = {"n": int(len(df))}
    ok = True
    for name, (x, y, exp) in specs.items():
        rec = associate(np.asarray(x, float), np.asarray(y, float), Z)
        rec["expect"] = exp
        rec["match"] = bool(np.isfinite(rec["controlled"]) and abs(rec["controlled"] - exp) <= PARITY_ATOL)
        out[name] = rec
        ok = ok and rec["match"]
    if "r2_P" in df.columns:
        out["rho_ctl_KH_R2P"] = associate(df.K_H_cross.to_numpy(float), df.r2_P.to_numpy(float), Z)
    if "mse_P" in df.columns:
        out["rho_ctl_KH_MSEP"] = associate(df.K_H_cross.to_numpy(float), df.mse_P.to_numpy(float), Z)
    patch_col = next((c for c in ("local_r2", "local_oracle_R2", "r2_quad", "patch_r2") if c in df.columns), None)
    if patch_col:
        out["rho_ctl_KH_patchR2"] = associate(df.K_H_cross.to_numpy(float), df[patch_col].to_numpy(float), Z)
        out["patch_r2_column"] = patch_col
    else:
        out["patch_r2_column"] = None
        out["patch_r2_note"] = "no frozen patch-R^2 column on the aligned 512-anchor table"
    out["ok"] = bool(ok and len(df) == N_ANCHORS)
    out["sample_id_unique"] = bool(df.sample_id.nunique() == len(df))
    return out


def q_refit_parity(ctx: dict, n_check: int = 8) -> dict[str, Any]:
    X, neigh, frames, sids, df = ctx["X"], ctx["neigh"], ctx["frames"], ctx["sids"], ctx["df"]
    kh0 = df.set_index("sample_id")["K_H_cross"]
    diffs = []
    for i, sid in enumerate(sids[:n_check]):
        x0, J = frames[i]
        Xloc = np.asarray(X[neigh[i, :K]], dtype=np.float64)
        rec = production_mean_kh(Xloc, x0, J, ai=int(ctx["bundle"]["sid_to_ai"][int(sid)]), k=K)
        frozen = float(kh0.loc[int(sid)])
        diffs.append({"sample_id": int(sid), "refit": rec["K_H_cross"], "frozen": frozen, "abs_diff": abs(rec["K_H_cross"] - frozen)})
    med = float(np.median([d["abs_diff"] for d in diffs])) if diffs else float("nan")
    return {"n": len(diffs), "median_abs_diff": med, "ok": bool(med < 1e-6 if np.isfinite(med) else False), "rows": diffs}
