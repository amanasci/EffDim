"""Phase 0: artifacts, sample_id alignment, checkpoint reuse."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_cross_model_task_aligned_curvature.audit import decoder_ckpts as _ckpts
from geometry.physics_cross_model_task_aligned_curvature.audit import load_model_bundle, load_shared as _cm_shared

from .config import (
    ALL_MODELS,
    D_LAT,
    K,
    PAPER_TABLE2,
    SOURCE_ALIGN,
    SOURCE_CMPR,
    SOURCE_TAC,
    TARGETS,
    ExpConfig,
)
from .io_util import file_meta, platonic_root, resolve_path, write_json


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    # Reuse the cross-model loader (same MM folds / anchors / four targets).
    from geometry.physics_cross_model_task_aligned_curvature.config import ExpConfig as Cfg2

    shared = _cm_shared(Cfg2(smoke=cfg.smoke, models_override=list(cfg.models())))
    shared["paper_pdf"] = _find_pdf(shared["root"])
    return shared


def _find_pdf(root) -> dict[str, Any]:
    cands = [
        root / "ml4ps_curvature.pdf",
        root / "papers" / "ml4ps_curvature.pdf",
        root / "submissions" / "ml4ps_curvature.pdf",
        root / "submissions" / "ml4ps_2026" / "ml4ps_curvature.pdf",
    ]
    hit = next((p for p in cands if p.exists()), None)
    return {"path": str(hit) if hit else None, "found": hit is not None, "searched": [str(p) for p in cands]}


def load_bundles(shared: dict, models: tuple[str, ...]) -> dict[str, dict]:
    return {m: load_model_bundle(shared, m) for m in models}


def checkpoints(shared: dict, models: tuple[str, ...]) -> dict[str, dict]:
    return {m: _ckpts(shared, m) for m in models}


def historical_weights(shared: dict) -> dict[str, dict] | None:
    p = resolve_path(shared["root"], SOURCE_ALIGN) / "global_probe_weights.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = {}
    for t in TARGETS:
        if f"w_{t}" not in z.files:
            return None
        out[t] = {"w": np.asarray(z[f"w_{t}"], dtype=np.float64), "b": float(np.asarray(z[f"b_{t}"]).reshape(-1)[0])}
    return out


def historical_local_r2(shared: dict) -> pd.DataFrame | None:
    p = resolve_path(shared["root"], SOURCE_ALIGN) / "anchor_target_probe_results.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["sample_id"] = df.sample_id.astype(int)
    if "scale_k" in df.columns:
        df = df[df.scale_k.astype(int) == K]
    return df[df.target.isin(TARGETS)].copy()


def embedding_unit_norm(X: np.ndarray) -> dict[str, float]:
    nrm = np.linalg.norm(X, axis=1)
    return {"median": float(np.median(nrm)), "min": float(np.min(nrm)), "max": float(np.max(nrm)), "already_unit": bool(np.allclose(nrm, 1.0, atol=1e-3))}


def multiscale_radii(X: np.ndarray, neigh: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    from .config import MULTISCALE_K

    out = {}
    Xa = X[rows]
    for k in MULTISCALE_K:
        nbr = neigh[:, k - 1]
        rad = np.linalg.norm(X[nbr] - Xa, axis=1)
        out[int(k)] = np.log(np.maximum(rad, 1e-15))
    return out


def reuse_manifest(shared: dict, bundles: dict, ckpts: dict) -> dict[str, Any]:
    root = shared["root"]
    n_trainable = sum(0 if c["three_seed_ok"] else 3 for c in ckpts.values())
    return {
        "align_by": "sample_id",
        "d": D_LAT,
        "k": K,
        "n_anchors": len(shared["sids"]),
        "targets": list(TARGETS),
        "models": {m: {"ambient_D": bundles[m]["D"], "unit_norm": embedding_unit_norm(bundles[m]["X"]), "ckpts": ckpts[m]} for m in bundles},
        "paper_pdf": shared["paper_pdf"],
        "paper_table2_approx": PAPER_TABLE2,
        "decoder_protocol_note": (
            "Reused existing PlainAutoEncoder checkpoints (250×3 SiLU, 400 epochs, anchors excluded). "
            "These train on all non-anchor objects, not a further 80/20 of the complement. "
            "No new autoencoders were trained. Paper PDF was not located in-repo."
        ),
        "split_salt": "task_aligned_curvature_v1",
        "n_new_decoders_needed": int(n_trainable),
        "max_new_decoders": 12,
        "files": {
            "folds": file_meta(resolve_path(root, "outputs/geometry/physics_multimodel_graph_prior_quadratic") / "sample_folds.parquet"),
            "tac_decision": file_meta(resolve_path(root, SOURCE_TAC) / "decision.json"),
            "cmpr": file_meta(resolve_path(root, SOURCE_CMPR) / "decision.json"),
        },
        "q_tensors_serialized": False,
        "q_status": "unavailable_no_serialized_BS",
        "read_only_prior_trees": True,
        "implementation": "fresh_from_spec_reusing_frozen_decoders_and_QLCA_features",
    }
