"""Eligible-model inventory. Must not inspect scientific curvature–probe outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X

from .config import (
    CATALOG_FIELD,
    MIN_COMMON_ANCHORS,
    MODEL_SPECS,
    N_ANCHORS,
    POSITIVE_CONTROL,
    PRIMARY_K,
    SOURCE_MM,
    ExpConfig,
)
from .io_util import file_sha256, platonic_root, resolve_path, sha256_json, write_json


def _row_l2(X: np.ndarray) -> dict[str, float]:
    n = np.linalg.norm(X[: min(256, len(X))], axis=1)
    return {
        "row_l2_mean": float(np.mean(n)),
        "row_l2_std": float(np.std(n)),
        "unit_sphere": bool(abs(float(np.mean(n)) - 1.0) < 1e-3 and float(np.std(n)) < 1e-3),
    }


def inventory_models(cfg: ExpConfig, out) -> dict[str, Any]:
    root = platonic_root()
    mm = resolve_path(root, SOURCE_MM)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    sid_row = folds["sample_id"].to_numpy(int)
    n_obj = int(len(folds))
    y = folds[f"y_{CATALOG_FIELD}"].to_numpy(float)
    n_y = int(np.isfinite(y).sum())
    anchors = np.load(mm / "prepare" / "anchors.npz")
    orig_sids = [int(s) for s in anchors["anchors_sample_id"]]
    wanted = cfg.models_override or [s["model_id"] for s in MODEL_SPECS]
    spec_by = {s["model_id"]: s for s in MODEL_SPECS}

    alias_hash: dict[str, str] = {}
    rows = []
    for mid in wanted:
        spec = spec_by[mid]
        rec: dict[str, Any] = {
            **spec,
            "checkpoint_or_artifact": str(mm / "prepare" / "models" / f"{mid}.npz"),
            "embedding_dimension": None,
            "n_aligned_objects": 0,
            "normalization": "unknown",
            "eligible": False,
            "reason": "",
        }
        xp = mm / "prepare" / "models" / f"{mid}.npz"
        npz_neigh = mm / "model_neighbourhoods" / f"{mid}_kmax{PRIMARY_K}.npz"
        oof_p = mm / "global_probes" / "oof_predictions" / f"{mid}_{CATALOG_FIELD}.npz"
        if not xp.exists():
            rec["reason"] = "missing_embedding_npz"
            rows.append(rec)
            continue
        X = load_model_X(mm, mid)
        rec["embedding_dimension"] = int(X.shape[1])
        rec["n_aligned_objects"] = int(X.shape[0])
        rec["source_artifact_sha16"] = file_sha256(xp)
        l2 = _row_l2(X)
        rec["normalization"] = "row_l2_unit_sphere" if l2["unit_sphere"] else "not_unit_sphere"
        rec["row_l2"] = l2
        alias_hash[mid] = rec["source_artifact_sha16"]
        if X.shape[0] != n_obj:
            rec["reason"] = f"object_count_mismatch {X.shape[0]}!={n_obj}"
            rows.append(rec)
            continue
        if not npz_neigh.exists():
            rec["reason"] = "missing_knn_pack"
            rows.append(rec)
            continue
        pack = dict(np.load(npz_neigh))
        neigh = np.asarray(pack["neigh"], dtype=np.int64)
        if neigh.shape[1] < PRIMARY_K:
            rec["reason"] = f"knn_k={neigh.shape[1]} < {PRIMARY_K}"
            rows.append(rec)
            continue
        if not oof_p.exists():
            rec["reason"] = "missing_global_oof"
            rows.append(rec)
            continue
        oof = np.asarray(np.load(oof_p)["oof"], dtype=float).reshape(-1)
        if len(oof) != n_obj:
            rec["reason"] = "oof_length_mismatch"
            rows.append(rec)
            continue
        if n_y < PRIMARY_K + 8:
            rec["reason"] = "insufficient_labelled_objects"
            rows.append(rec)
            continue
        rec["eligible"] = True
        rec["reason"] = "eligible"
        rec["n_original_anchors"] = int(len(orig_sids))
        rows.append(rec)

    # alias detection from embedding file hashes
    by_hash: dict[str, list[str]] = {}
    for r in rows:
        h = r.get("source_artifact_sha16")
        if h:
            by_hash.setdefault(h, []).append(r["model_id"])
    for r in rows:
        h = r.get("source_artifact_sha16")
        mates = [m for m in by_hash.get(h, []) if m != r["model_id"]]
        if mates and r["eligible"]:
            r["eligible"] = False
            r["reason"] = f"duplicate_alias_of {mates}"

    eligible = [r["model_id"] for r in rows if r["eligible"]]
    distinct_arch = {spec_by[m]["architecture"] for m in eligible}
    distinct_obj = {spec_by[m]["training_objective"] for m in eligible}
    payload = {
        "n_objects": n_obj,
        "n_finite_target": n_y,
        "n_original_anchors": len(orig_sids),
        "original_anchor_sample_ids_head": orig_sids[:8],
        "models": rows,
        "eligible_model_ids": eligible,
        "positive_control_eligible": POSITIVE_CONTROL in eligible,
        "n_eligible": len(eligible),
        "n_architectures": len(distinct_arch),
        "n_training_objectives": len(distinct_obj),
        "insufficient_model_diversity": bool(len(eligible) < 3),
        "min_common_anchors_required": MIN_COMMON_ANCHORS,
        "note": "Eligibility does not inspect curvature–probe associations.",
    }
    write_json(out / "model_inventory.json", payload, force=True)
    frozen = {
        "protocol": "physics_cross_model_curvature_local_adaptation_v1",
        "d": 16,
        "k": PRIMARY_K,
        "n_anchors_requested": N_ANCHORS,
        "target_field": CATALOG_FIELD,
        "probe_alpha": 100.0,
        "estimator": "fixed_alpha_ridge_not_nested",
        "controls": list(("log_knn_radius", "local_label_variance", "local_evaluation_count")),
        "eligible_model_ids": eligible,
        "models": [
            {
                k: r[k]
                for k in (
                    "model_id",
                    "source_parquet",
                    "embedding_column",
                    "architecture",
                    "training_objective",
                    "representation_layer",
                    "embedding_dimension",
                    "n_aligned_objects",
                    "normalization",
                    "checkpoint_or_artifact",
                    "source_artifact_sha16",
                    "eligible",
                    "reason",
                    "positive_control",
                )
                if k in r
            }
            for r in rows
        ],
    }
    frozen["manifest_sha256"] = sha256_json(frozen)
    write_json(out / "FROZEN_MODEL_MANIFEST.json", frozen, force=True)
    return payload
