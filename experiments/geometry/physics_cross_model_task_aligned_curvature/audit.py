"""Phase 0: per-model artifacts, sample_id alignment, frozen K_H parity."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_curvature_probe_rank_sweep.inference import associate
from geometry.physics_task_aligned_curvature.probes import confirmatory_for_target

from .config import (
    D_LAT,
    DECODER_SEEDS,
    FROZEN_Q,
    K,
    NATIVE_D,
    PARITY_ATOL,
    PROTOCOL,
    SOURCE_CMCLA,
    SOURCE_CMPR,
    SOURCE_FCR,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_PRCR,
    SOURCE_TAC,
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
    Y = {t: folds[f"y_{t}"].to_numpy(float) for t in TARGETS}
    finite = {t: folds[f"finite_{t}"].to_numpy(bool) for t in TARGETS}
    return {
        "root": root,
        "mm": mm,
        "folds": folds,
        "sample_id_row": sample_id_row,
        "sid_to_row": sid_to_row,
        "orig_sids": orig_sids,
        "sids": sids,
        "Y": Y,
        "finite": finite,
        "n_obj": int(len(folds)),
    }


def load_model_bundle(shared: dict, model: str) -> dict[str, Any]:
    mm = shared["mm"]
    root = shared["root"]
    X = np.asarray(load_model_X(mm, model))
    if int(X.shape[1]) != NATIVE_D[model]:
        raise RuntimeError(f"{model}: ambient D {X.shape[1]} != {NATIVE_D[model]}")
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{model}_kmax{K}.npz"))
    neigh_all = np.asarray(pack["neigh"], dtype=np.int64)
    sid_to_ai = {int(s): i for i, s in enumerate(shared["orig_sids"])}
    neigh = neigh_all[np.asarray([sid_to_ai[s] for s in shared["sids"]], dtype=np.int64)]
    fcr = pd.read_parquet(resolve_path(root, SOURCE_FCR) / "tables" / f"{model}_per_anchor_curvature.parquet")
    cmcla = pd.read_parquet(resolve_path(root, SOURCE_CMCLA) / "probes" / f"{model}_anchor_metrics.parquet")
    return {
        "model": model,
        "X": X,
        "neigh": neigh,
        "sid_to_ai": sid_to_ai,
        "fcr": fcr,
        "cmcla": cmcla,
        "D": int(X.shape[1]),
    }


def as_shared(shared: dict, bundle: dict) -> dict[str, Any]:
    """Shape expected by confirmatory_for_target / Q helpers."""
    return {
        **shared,
        "X": bundle["X"],
        "neigh": bundle["neigh"],
        "sid_to_ai": bundle["sid_to_ai"],
        "fcr": bundle["fcr"],
        "cmcla": bundle["cmcla"],
        "D": bundle["D"],
        "model": bundle["model"],
    }


def load_q_frame(shared: dict, model: str, sid: int) -> tuple[np.ndarray, np.ndarray]:
    root = shared["root"]
    cmcla = resolve_path(root, SOURCE_CMCLA) / "geometry" / model / f"J_{int(sid)}.npz"
    if cmcla.exists():
        z = np.load(cmcla)
        return np.asarray(z["x0"], dtype=np.float64), np.asarray(z["J"], dtype=np.float64)[:, :D_LAT]
    if model == "vit_base":
        jp = resolve_path(root, SOURCE_NDC) / "cache" / f"J_{int(sid)}_k{K}.npz"
        z = np.load(jp)
        return np.asarray(z["x0"], dtype=np.float64), np.asarray(z["J"], dtype=np.float64)[:, :D_LAT]
    raise FileNotFoundError(f"no Q frame for {model} sample_id={sid}")


def decoder_ckpts(shared: dict, model: str) -> dict[str, Any]:
    root = shared["root"]
    cmpr = resolve_path(root, SOURCE_CMPR) / "checkpoints"
    prcr = resolve_path(root, SOURCE_PRCR) / "checkpoints"
    found = {}
    for seed in DECODER_SEEDS:
        cands = [cmpr / f"{model}_plain_ae_seed{seed}.npz"]
        if model == "vit_base":
            cands.append(prcr / f"{model}_plain_ae_seed{seed}.npz")
        hit = next((p for p in cands if p.exists()), cands[0])
        meta_p = hit.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        found[int(seed)] = {
            "path": str(hit),
            "exists": hit.exists(),
            "protocol": meta.get("protocol"),
            "anchors_excluded": meta.get("anchors_excluded"),
            "latent_dim": meta.get("latent_dim"),
            "compatible": bool(
                hit.exists()
                and int(meta.get("latent_dim", -1)) == D_LAT
                and str(meta.get("protocol", "")) == PROTOCOL
                and bool(meta.get("anchors_excluded", False))
            ),
        }
    return {
        "three_seed": found,
        "three_seed_ok": all(v["compatible"] for v in found.values()),
    }


def log_radius(bundle: dict, sids: list[int]) -> np.ndarray:
    src = bundle["cmcla"].copy()
    src["sample_id"] = src.sample_id.astype(int)
    mp = {int(s): float(r) for s, r in zip(src.sample_id, src.log_knn_radius)}
    return np.array([mp.get(int(s), np.nan) for s in sids], dtype=float)


def reuse_vitb_D(shared: dict) -> dict[str, np.ndarray] | None:
    p = resolve_path(shared["root"], SOURCE_TAC) / "per_anchor_task_aligned.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["sample_id"] = df.sample_id.astype(int)
    out = {}
    sids = [int(s) for s in shared["sids"]]
    for t in TARGETS:
        sub = df[df.target == t].drop_duplicates("sample_id")
        mp = {int(s): float(v) for s, v in zip(sub.sample_id, sub.E_D_S)}
        arr = np.array([mp.get(s, np.nan) for s in sids], dtype=float)
        if not np.isfinite(arr).all():
            return None
        out[t] = arr
    return out


def run_parity(shared: dict, bundles: dict[str, dict], cfg: ExpConfig, out) -> dict[str, Any]:
    per = {}
    ok = True
    for model, bundle in bundles.items():
        fcr = bundle["fcr"].copy()
        fcr["sample_id"] = fcr.sample_id.astype(int)
        df = pd.DataFrame({"sample_id": shared["sids"]}).merge(fcr, on="sample_id", how="inner")
        if len(df) != len(shared["sids"]):
            per[model] = {"ok": False, "reason": "sample_id_join_dropped"}
            ok = False
            continue
        Z = np.column_stack(
            [
                df["log_knn_radius"].to_numpy(float),
                df["local_label_variance"].to_numpy(float),
                df["local_evaluation_count"].to_numpy(float),
            ]
        )
        a = associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)
        match = abs(float(a["controlled"]) - FROZEN_Q[model]) <= PARITY_ATOL
        per[model] = {"rho_ctl_KH_R2G": a, "expected": FROZEN_Q[model], "ok": match}
        ok = ok and match
    report = {"ok": ok or bool(cfg.smoke), "per_model": per, "align_by": "sample_id"}
    write_json(out / "parity.json", report, force=True)
    return report


def reuse_manifest(shared: dict, bundles: dict[str, dict], ckpts: dict[str, dict]) -> dict[str, Any]:
    root = shared["root"]
    files = {
        "folds": resolve_path(root, SOURCE_MM) / "sample_folds.parquet",
        "anchors": resolve_path(root, SOURCE_MM) / "prepare" / "anchors.npz",
        "vitb_task_aligned": resolve_path(root, SOURCE_TAC) / "decision.json",
    }
    excluded = []
    details = {}
    for t in TARGETS:
        n_lab = int(shared["finite"][t].sum())
        details[t] = {"n_labelled": n_lab, "frac_labelled": float(shared["finite"][t].mean()), "joined_by": "sample_id"}
        if n_lab < 8:
            excluded.append(t)
    models = {}
    for m, b in bundles.items():
        gdir = resolve_path(root, SOURCE_CMCLA) / "geometry" / m
        models[m] = {
            "ambient_D": b["D"],
            "n_q_frames": int(len(list(gdir.glob("J_*.npz")))) if gdir.exists() else 0,
            "neigh": file_meta(resolve_path(root, SOURCE_MM) / "model_neighbourhoods" / f"{m}_kmax{K}.npz"),
            "checkpoints": ckpts[m],
        }
    return {
        "align_by": "sample_id",
        "d": D_LAT,
        "k": K,
        "n_anchors": len(shared["sids"]),
        "targets": list(TARGETS),
        "targets_excluded": excluded,
        "four_target_claim_ok": len(excluded) == 0,
        "models": models,
        "files": {k: file_meta(p) for k, p in files.items()},
        "read_only_prior_trees": True,
        "no_new_autoencoders": True,
        "primary_instrument": "Q_residual_sphere_normal_cross",
        "split_salt": "task_aligned_curvature_v1",
    }


def coverage_for_model(shared: dict, bundle: dict) -> dict[str, Any]:
    sh = as_shared(shared, bundle)
    per = {}
    ok = True
    for t in TARGETS:
        rec = confirmatory_for_target(sh, t)
        cov = rec["coverage"]
        per[t] = cov
        ok = ok and bool(cov["ok"])
    return {"ok": ok, "per_target": per, "model": bundle["model"]}
