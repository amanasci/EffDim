"""Stage pipeline for order-stratified (d_1, q_2) geometry.

Scientific order: freeze/parity → carrier → quadratic-normal rank → tail
tests → mixed scaling → models → synthetics → probes last.
Never writes into preserved geometry output directories.
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
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    cross_metric_pair,
    metric_scalars,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_activation_atlas.sphere_normal_quadratic import NestedChart, chart_errors
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    angular_radii,
    projected_chord,
    rms_tangent_radius,
    row_l2_status,
    sphere_log_map,
)
from geometry.physics_stable_tangent_dimension.dimension import paired_bootstrap_ci

from .algebra import (
    EPS,
    ambient_mse,
    cross_frobenius,
    fit_quadratic_map,
    intersection_rank,
    mix_shares,
    mixed_scale_nnls,
    n_quad_features,
    odd_even_displacements,
    pair_antipodes,
    pca_subspace,
    per_col_r2,
    predict_quadratic_map,
    projector_overlap,
    r2_score,
    refine_chart_coords,
    svd_quadratic_image,
    truncate_bs_left,
    whiten_tangent,
)
from .rank import DEFAULT_Q_THRESHOLDS, classify_hypothesis, select_q2
from .synthetics import SYNTH_KINDS, closest_synthetic, make_order_synthetic, split_seeds

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_FCA = "outputs/geometry/physics_full_curvature_audit"
SOURCE_STD = "outputs/geometry/physics_stable_tangent_dimension"
SOURCE_ATLAS = "outputs/geometry/physics_activation_atlas"

PRESERVED = [SOURCE_MM, SOURCE_EDM, SOURCE_NDC, SOURCE_COV, SOURCE_FCA, SOURCE_STD, SOURCE_ATLAS]

PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"
K_CANDIDATES = [128, 256, 512, 768, 1024, 1536, 2048]


@dataclass
class OrderStratConfig:
    output_dir: str = "outputs/geometry/physics_order_stratified_geometry"
    multimodel_dir: str = SOURCE_MM
    effdim_dir: str = SOURCE_EDM
    nested_dir: str = SOURCE_NDC
    coverage_dir: str = SOURCE_COV
    std_dir: str = SOURCE_STD
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    d_core: int = 12
    d_ref: int = 16
    R: int = 20
    R_sens: list[int] = field(default_factory=lambda: [16, 24, 32])
    q_max: int = 8
    n_null_draw: int = 8
    n_parity_anchors: int = 32
    n_synth_cal: int = 6
    n_synth_eval: int = 6
    n_splits: int = 1
    batch_size: int = 32
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
    coord: str = "log"
    n_anchors: int | None = None

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: OrderStratConfig, reserve: bool = False) -> bool:
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


def resolve_k_grid(k_max: int, *, smoke: bool, primary_k: int) -> list[int]:
    if smoke:
        grid = [k for k in [64, 96, 128, 192, 256] if k <= k_max]
        return grid[:5] if len(grid) >= 4 else grid
    grid = [k for k in K_CANDIDATES if k <= k_max]
    if primary_k <= k_max and primary_k not in grid:
        grid.append(int(primary_k))
    return sorted(set(grid))


def quad_capable_ks(ks: list[int], d_core: int, *, smoke: bool) -> list[int]:
    need = n_quad_features(d_core) + (8 if smoke else 20)
    min_k = 64 if smoke else 768
    out = [k for k in ks if k >= min_k and 0.4 * k >= need]
    return out or [max(ks)]


def load_ctx(root: Path, cfg: OrderStratConfig) -> dict:
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
        use_sids = use_sids[:8]
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == cfg.model) & (geo.target == cfg.target) & (geo.neighbourhood == "model")]
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    pack2048 = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    freeze_p = resolve_path(root, cfg.effdim_dir) / "dimension_freeze.json"
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else {}
    k_max = int(pack2048["neigh"].shape[1])
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
        "pack2048": pack2048,
        "freeze": freeze,
        "X": X,
        "ks": ks,
        "ks_quad": quad_capable_ks(ks, cfg.d_core, smoke=cfg.smoke),
        "k_max": k_max,
        "l2": row_l2_status(X),
        "std": resolve_path(root, cfg.std_dir),
        "ndc": resolve_path(root, cfg.nested_dir),
        "edm": resolve_path(root, cfg.effdim_dir),
        "cov": resolve_path(root, cfg.coverage_dir),
    }


def ensure_neigh(ctx: dict, ai: int, k: int) -> np.ndarray:
    return ctx["pack2048"]["neigh"][ai, : min(k, ctx["pack2048"]["neigh"].shape[1])]


def displacements(x0: np.ndarray, Xloc: np.ndarray, coord: str) -> np.ndarray:
    if coord == "chord":
        return projected_chord(x0, Xloc)
    return sphere_log_map(x0, Xloc)


def _j_path(out: Path, model: str, sid: int, k: int) -> Path:
    return out / "J" / f"{model}_{int(sid)}_k{int(k)}.npz"


def _b_path(out: Path, model: str, sid: int, k: int) -> Path:
    return out / "B" / f"{model}_{int(sid)}_k{int(k)}.npz"


def stage_prepare(root: Path, cfg: OrderStratConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "batches", "figures", "logs", "synth", "J", "B"):
        (out / sub).mkdir(exist_ok=True)
    mm = ctx["mm"]
    x_path = mm / "prepare" / "models" / f"{cfg.model}.npz"
    pack_path = mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    probes = mm / "global_probes" / "oof_predictions"
    probe_hash = _sha16(sorted(p.name for p in probes.glob("*")) if probes.exists() else "missing")
    meta = {
        "config": asdict(cfg),
        "protocol": "order_stratified_geometry_v1",
        "preserved": PRESERVED,
        "ks": ctx["ks"],
        "ks_quad": ctx["ks_quad"],
        "n_anchors": len(ctx["use_sids"]),
        "l2_status": ctx["l2"],
        "primary_comparison": "(12, q2) vs (16, 0)",
        "d1_source": "frozen_stable_tangent_core",
        "no_local_probes": True,
        "ridges": RIDGES,
        "software": {"numpy": np.__version__, "torch": torch.__version__, "pandas": pd.__version__},
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
    (out / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "sample_ids": ctx["use_sids"],
                "model": cfg.model,
                "l2_normalized": ctx["l2"]["unit_normalized"],
                "neighbour_search_metric": "inner_product",
                "split_schedule": "radial_stratified_halves",
                "reference_k": cfg.primary_k,
                "d_core": cfg.d_core,
                "R": cfg.R,
                "seed": cfg.seed,
                "quadratic_regularization": RIDGES,
                **meta["hashes"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"[osg] prepare ks={ctx['ks']} ks_quad={ctx['ks_quad']} n={len(ctx['use_sids'])}", flush=True)
    return meta


from .stages import STAGES, run  # noqa: E402

