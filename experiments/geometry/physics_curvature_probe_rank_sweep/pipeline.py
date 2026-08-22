"""Config, reuse, parity. Never writes into completed geometry directories."""

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

from geometry.physics_activation_atlas.curvature_probe_screen import partial_spearman, spearman_dict
from geometry.physics_activation_atlas.effdim_curvature_metrics import decompose_tensors, metric_scalars
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.nested_dimension_curvature import (
    _fit_rank,
    ensure_neigh,
    nested_pca_frame,
)
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_stable_tangent_dimension.sphere_coords import row_l2_status

from .classify import DEFAULT_THRESHOLDS

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_STD = "outputs/geometry/physics_stable_tangent_dimension"
SOURCE_OSG = "outputs/geometry/physics_order_stratified_geometry"
SOURCE_INI = "outputs/geometry/physics_implicit_normal_inverse"
SOURCE_QPD = "outputs/geometry/physics_quadratic_predictive_dimension"

PRESERVED = [SOURCE_MM, SOURCE_EDM, SOURCE_NDC, SOURCE_COV, SOURCE_STD, SOURCE_OSG, SOURCE_INI, SOURCE_QPD]
PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_NDC_D16_RAW = -0.412430
PARITY_NDC_D12_RAW = -0.038426
PARITY_NDC_D16_CTL = -0.240484
PARITY_TOL = 0.03
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"


@dataclass
class RankSweepConfig:
    output_dir: str = "outputs/geometry/physics_curvature_probe_rank_sweep"
    multimodel_dir: str = SOURCE_MM
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    d_min: int = 8
    d_max: int = 20
    n_perm: int = 10000
    n_boot: int = 2000
    n_scale_anchors: int = 128
    n_scale_splits: int = 3
    n_parity_anchors: int = 32
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 300.0
    smoke: bool = False
    n_anchors: int | None = None
    skip_scale_fit: bool = False

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def ds(self) -> list[int]:
        return list(range(int(self.d_min), int(self.d_max) + 1))

    def primary_ds(self) -> list[int]:
        return [d for d in self.ds() if d >= 12]


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: RankSweepConfig, reserve: bool = False) -> bool:
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


def atomic_replace(tmp: Path, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        ts = time.strftime("%Y%m%dT%H%M%S")
        dest.rename(dest.with_name(f"{dest.stem}.superseded.{ts}{dest.suffix}"))
    tmp.replace(dest)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    atomic_replace(tmp, path, force=force)


def hash_select(sids: list[int], n: int, *, seed: int) -> list[int]:
    scored = [(hashlib.sha256(f"cprs:{seed}:{int(s)}".encode()).hexdigest(), int(s)) for s in sids]
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]


def load_ctx(root: Path, cfg: RankSweepConfig) -> dict:
    mm = resolve_path(root, cfg.multimodel_dir)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(aid.read_text())["sample_ids"] if aid.exists() else [int(s) for s in anchors_sid]
    if cfg.n_anchors is not None:
        use_sids = use_sids[: int(cfg.n_anchors)]
    elif cfg.smoke:
        use_sids = use_sids[:16]
        cfg.n_perm = min(cfg.n_perm, 200)
        cfg.n_boot = min(cfg.n_boot, 100)
        cfg.n_scale_anchors = min(cfg.n_scale_anchors, 8)
        cfg.n_scale_splits = 1
        cfg.n_parity_anchors = min(cfg.n_parity_anchors, 8)
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == cfg.model) & (geo.target == cfg.target) & (geo.neighbourhood == "model")]
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    freeze_p = resolve_path(root, SOURCE_EDM) / "dimension_freeze.json"
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else {}
    X = load_model_X(mm, cfg.model)
    scale_sids = hash_select(use_sids, cfg.n_scale_anchors, seed=cfg.seed)
    return {
        "mm": mm,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "scale_sids": scale_sids,
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "pack2048": pack,
        "pack3072": None,
        "freeze": freeze,
        "X": X,
        "l2": row_l2_status(X),
        "ndc": resolve_path(root, SOURCE_NDC),
        "edm": resolve_path(root, SOURCE_EDM),
        "cov": resolve_path(root, SOURCE_COV),
        "qpd": resolve_path(root, SOURCE_QPD),
        "std": resolve_path(root, SOURCE_STD),
    }


def kh_trace_identity(BS_flat: np.ndarray, d: int) -> float:
    """||H|| from diagonal mean; must match metric_scalars K_H."""
    B = decompose_tensors(BS_flat, d)["B"]
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    return float(np.sqrt(max(float(np.dot(H, H)), 0.0)))
