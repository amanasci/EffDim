"""Phase 0: locate artifacts, align by sample_id, frozen parity."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import (
    D_LAT,
    DECODER_SEEDS,
    FROZEN_CH_R2,
    FROZEN_Q_R2,
    HISTORICAL_AMEND01_RHO,
    HISTORICAL_RAW_RHO,
    K,
    MODEL,
    N_FOLDS,
    PARITY_ATOL,
    PROTOCOL,
    SOURCE_AE,
    SOURCE_ALIGN,
    SOURCE_AUSTIN,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_MM,
    SOURCE_MULTI,
    SOURCE_NDC,
    SOURCE_PRCR,
    TARGETS,
    ExpConfig,
)
from .io_util import file_meta, platonic_root, resolve_path, write_json


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sample_id_row = folds["sample_id"].to_numpy(int)
    sid_to_row = {int(s): int(i) for i, s in enumerate(sample_id_row)}
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = [int(s) for s in np.asarray(anchors["anchors_sample_id"])]
    sids = orig_sids[: cfg.n_anc()]
    X = np.asarray(load_model_X(mm, MODEL))
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{MODEL}_kmax{K}.npz"))
    neigh_all = np.asarray(pack["neigh"], dtype=np.int64)
    sid_to_ai = {int(s): i for i, s in enumerate(orig_sids)}
    neigh = neigh_all[np.asarray([sid_to_ai[s] for s in sids], dtype=np.int64)]
    Y = {}
    finite = {}
    for t in TARGETS:
        Y[t] = folds[f"y_{t}"].to_numpy(float)
        finite[t] = folds[f"finite_{t}"].to_numpy(bool)
    prcr = resolve_path(root, SOURCE_PRCR)
    fcr = pd.read_parquet(resolve_path(root, SOURCE_FCR) / "tables" / f"{MODEL}_per_anchor_curvature.parquet")
    cmcla = pd.read_parquet(resolve_path(root, SOURCE_CMCLA) / "probes" / f"{MODEL}_anchor_metrics.parquet")
    return {
        "root": root,
        "mm": mm,
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "sid_to_ai": sid_to_ai,
        "orig_sids": orig_sids,
        "sids": sids,
        "X": X,
        "neigh": neigh,
        "Y": Y,
        "finite": finite,
        "fold": folds["fold"].to_numpy(int),
        "n_obj": int(len(folds)),
        "prcr": prcr,
        "fcr": fcr,
        "cmcla": cmcla,
        "D": int(X.shape[1]),
    }


def load_q_frame(shared: dict, sid: int) -> tuple[np.ndarray, np.ndarray]:
    root = shared["root"]
    jp = resolve_path(root, SOURCE_NDC) / "cache" / f"J_{int(sid)}_k{K}.npz"
    if not jp.exists():
        jp = resolve_path(root, SOURCE_CMCLA) / "geometry" / MODEL / f"J_{int(sid)}.npz"
    z = np.load(jp)
    return np.asarray(z["x0"], dtype=np.float64), np.asarray(z["J"], dtype=np.float64)[:, :D_LAT]


def load_historical_weights(shared: dict) -> dict[str, dict[str, np.ndarray]]:
    p = resolve_path(shared["root"], SOURCE_ALIGN) / "global_probe_weights.npz"
    z = np.load(p)
    out = {}
    for t in TARGETS:
        key = f"w_{t}"
        if key not in z.files:
            raise RuntimeError(f"historical weight missing for {t}")
        out[t] = {"w": np.asarray(z[key], dtype=np.float64), "b": float(np.asarray(z[f"b_{t}"]).reshape(-1)[0])}
    return out


def historical_local_r2(shared: dict) -> pd.DataFrame:
    p = resolve_path(shared["root"], SOURCE_ALIGN) / "anchor_target_probe_results.parquet"
    df = pd.read_parquet(p)
    df["sample_id"] = df["sample_id"].astype(int)
    if "scale_k" in df.columns:
        df = df[df.scale_k.astype(int) == K]
    keep = df[df.target.isin(TARGETS) & df.sample_id.isin(shared["sids"])]
    return keep.reset_index(drop=True)


def decoder_ckpts(shared: dict) -> dict[str, Any]:
    prcr = shared["prcr"] / "checkpoints"
    found = {}
    for seed in DECODER_SEEDS:
        p = prcr / f"{MODEL}_plain_ae_seed{seed}.npz"
        meta_p = p.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        found[int(seed)] = {
            "path": str(p),
            "exists": p.exists(),
            "protocol": meta.get("protocol"),
            "anchors_excluded": meta.get("anchors_excluded"),
            "latent_dim": meta.get("latent_dim"),
            "compatible": bool(
                p.exists()
                and int(meta.get("latent_dim", -1)) == D_LAT
                and str(meta.get("protocol", "")) == PROTOCOL
                and bool(meta.get("anchors_excluded", False))
            ),
        }
    austin = resolve_path(shared["root"], SOURCE_AUSTIN)
    ae = resolve_path(shared["root"], SOURCE_AE)
    raw_ckpts = list(austin.rglob("*.npz")) + list(ae.rglob("*plain_ae*.npz")) + list(ae.rglob("*checkpoint*"))
    return {
        "three_seed": found,
        "three_seed_ok": all(v["compatible"] for v in found.values()),
        "austin_raw_decoder": {"searched": str(austin), "npz": [str(p) for p in raw_ckpts], "found": bool(raw_ckpts)},
        "amendment01_sphere_projected": {
            "summary": str(ae / "summary.json"),
            "exists": (ae / "summary.json").exists(),
            "checkpoint_found": any("plain_ae" in p.name or "ckpt" in p.name for p in ae.rglob("*")),
        },
    }


def reuse_manifest(shared: dict, ckpts: dict) -> dict[str, Any]:
    root = shared["root"]
    files = {
        "embeddings": resolve_path(root, SOURCE_MM) / "prepare" / "models" / f"{MODEL}.npz",
        "anchors": resolve_path(root, SOURCE_MM) / "prepare" / "anchors.npz",
        "folds": resolve_path(root, SOURCE_MM) / "sample_folds.parquet",
        "neighbours": resolve_path(root, SOURCE_MM) / "model_neighbourhoods" / f"{MODEL}_kmax{K}.npz",
        "historical_weights": resolve_path(root, SOURCE_ALIGN) / "global_probe_weights.npz",
        "prcr_decision": shared["prcr"] / "decision.json",
        "cae": root / "notebooks" / "pu_manifold" / "cae.py",
    }
    excluded = []
    per = {}
    for t in TARGETS:
        n_lab = int(shared["finite"][t].sum())
        per[t] = {
            "n_labelled": n_lab,
            "frac_labelled": float(shared["finite"][t].mean()),
            "joined_by": "sample_id",
        }
        if n_lab < 8:
            excluded.append(t)
    return {
        "align_by": "sample_id",
        "model": MODEL,
        "d": D_LAT,
        "k": K,
        "n_anchors": len(shared["sids"]),
        "n_objects": shared["n_obj"],
        "ambient_D": shared["D"],
        "targets": list(TARGETS),
        "targets_excluded": excluded,
        "four_target_claim_ok": len(excluded) == 0,
        "files": {k: file_meta(p) for k, p in files.items()},
        "checkpoints": ckpts,
        "target_details": per,
        "read_only": True,
        "no_new_autoencoders": True,
    }


def run_parity(shared: dict, cfg: ExpConfig, out) -> dict[str, Any]:
    fcr = shared["fcr"].copy()
    fcr["sample_id"] = fcr["sample_id"].astype(int)
    df = pd.DataFrame({"sample_id": shared["sids"]})
    df = df.merge(fcr, on="sample_id", how="inner")
    if len(df) != len(shared["sids"]):
        raise RuntimeError(f"FCR sample_id join dropped rows {len(shared['sids'])}->{len(df)}")
    # reuse mag_r controls from FCR (frozen OOF, not confirmatory eval)
    Z = np.column_stack(
        [
            df["log_knn_radius"].to_numpy(float),
            df["local_label_variance"].to_numpy(float),
            df["local_evaluation_count"].to_numpy(float),
        ]
    )
    a = associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)
    q_ok = abs(float(a["controlled"]) - FROZEN_Q_R2) <= PARITY_ATOL

    prcr_p = shared["prcr"] / "per_anchor_curvature.parquet"
    dres = {"available": prcr_p.exists()}
    dres_ok = False
    if prcr_p.exists():
        tab = pd.read_parquet(prcr_p)
        tab["sample_id"] = tab["sample_id"].astype(int)
        m = df.merge(tab[["sample_id", "C_H"]], on="sample_id", how="inner")
        ad = associate(m.C_H.to_numpy(float), m.r2_G.to_numpy(float), Z[: len(m)])
        dres.update({"n": int(len(m)), "rho_ctl_CH_R2G": ad, "expected": FROZEN_CH_R2})
        dres_ok = abs(float(ad["controlled"]) - FROZEN_CH_R2) <= PARITY_ATOL
        dres["ok"] = dres_ok

    hist = {
        "raw_decoder_mag_r": {
            "expected": HISTORICAL_RAW_RHO,
            "recoverable": False,
            "reason": "Austin physics raw-decoder checkpoint and serialized H^E field were not found",
        },
        "amendment01_sphere_projected": {
            "expected": HISTORICAL_AMEND01_RHO,
            "recoverable": False,
            "reason": "Amendment 01 sphere-projected checkpoint was not serialized; only H_tan scalars exist and do not reproduce +0.328 vs frozen R_G^2",
        },
    }
    ae_sum = resolve_path(shared["root"], SOURCE_AE) / "summary.json"
    if ae_sum.exists():
        sm = json.loads(ae_sum.read_text())
        hist["amendment01_sphere_projected"]["scale_match_rho_point_R2"] = sm.get("rho_point_R2")

    report = {
        "ok": bool(q_ok and (dres_ok or cfg.smoke)),
        "q_KH_R2G": a,
        "q_ok": q_ok,
        "vitb_dresidual": dres,
        "vitb_dresidual_ok": dres_ok,
        "historical": hist,
        "n": int(len(df)),
        "n_folds": N_FOLDS,
        "note": "Frozen files are authoritative. Historical +0.347/+0.328 are not recoverable without their checkpoints.",
    }
    write_json(out / "parity.json", report, force=True)
    return report
