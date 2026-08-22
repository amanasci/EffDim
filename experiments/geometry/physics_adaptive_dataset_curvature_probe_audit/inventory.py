"""Phase 1: exact comparison manifest for frozen discovery vs adaptive run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DISCOVERY_LABEL, DISCOVERY_PROBE_COLUMN, PARITY_RANKS, SHARED_CORE_CONTROLS
from .pipeline import AuditConfig, file_sha, file_sha_full, hash_select, write_df, write_json


def _git(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "short": run(["rev-parse", "--short", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
        "status_short": run(["status", "-sb"]).splitlines()[0] if run(["status", "-sb"]) else "",
    }


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text()) if p.exists() else {}


def discovery_sids(mm: Path) -> list[int]:
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    if aid.exists():
        return [int(s) for s in json.loads(aid.read_text())["sample_ids"]]
    z = np.load(mm / "prepare" / "anchors.npz")
    return [int(s) for s in z["anchors_sample_id"]]


def adaptive_sids(adcp: Path) -> list[int]:
    p = adcp / "datasets" / "physics_vit_base" / "geometry_meta.json"
    if p.exists():
        return [int(s) for s in json.loads(p.read_text())["anchors"]]
    panel = pd.read_parquet(adcp / "datasets" / "physics_vit_base" / "per_anchor_curvature.parquet")
    return [int(s) for s in panel.sample_id.drop_duplicates().tolist()]


def load_frozen_probe(mm: Path, *, k: int = 2048) -> pd.DataFrame:
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == "vit_base") & (geo.target == DISCOVERY_LABEL) & (geo.neighbourhood == "model")]
    if "scale_k" in geo.columns:
        geo = geo[geo.scale_k == k]
    return geo.drop_duplicates("sample_id")


def build_inventory(root: Path, cfg: AuditConfig) -> dict[str, Any]:
    out = cfg.resolved(root)
    mm, cprs, adcp, ndc = cfg.mm(root), cfg.cprs(root), cfg.adcp(root), cfg.ndc(root)
    disc = discovery_sids(mm)
    adap = adaptive_sids(adcp)
    shared = sorted(set(disc) & set(adap))
    knn_p = mm / "model_neighbourhoods" / "vit_base_kmax2048.npz"
    knn = dict(np.load(knn_p)) if knn_p.exists() else {}
    sid_to_ai = {int(s): i for i, s in enumerate(knn["anchors_sample_id"])} if "anchors_sample_id" in knn else {}
    if "sample_ids" in knn and not sid_to_ai:
        # pack stores full-table sample_ids separately; neigh rows follow anchors.npz order
        az = np.load(mm / "prepare" / "anchors.npz")
        sid_to_ai = {int(s): i for i, s in enumerate(az["anchors_sample_id"])}

    rows = []
    for i, sid in enumerate(sorted(set(disc) | set(adap))):
        rows.append(
            {
                "sample_id": int(sid),
                "in_discovery": sid in set(disc),
                "in_adaptive": sid in set(adap),
                "discovery_order": disc.index(sid) if sid in set(disc) else -1,
                "adaptive_order": adap.index(sid) if sid in set(adap) else -1,
            }
        )
    anchor_df = pd.DataFrame(rows)
    write_df(out / "anchor_comparison.csv", anchor_df, force=cfg.force)

    neigh_rows = []
    if "neigh" in knn and sid_to_ai:
        for sid in shared:
            ai = sid_to_ai.get(int(sid))
            if ai is None:
                continue
            nids = [int(x) for x in knn["neigh"][ai, :2048]]
            neigh_rows.append(
                {
                    "sample_id": int(sid),
                    "k": 2048,
                    "n_neighbours": len(nids),
                    "first_neighbour": nids[0] if nids else -1,
                    "last_neighbour": nids[-1] if nids else -1,
                    "neigh_sha16": __import__("hashlib").sha256(np.asarray(nids, dtype=np.int64).tobytes()).hexdigest()[:16],
                    "source": "shared_vit_base_kmax2048",
                }
            )
    write_df(out / "neighbour_comparison.csv", pd.DataFrame(neigh_rows), force=cfg.force)

    labels_p = root / "data_hf/physics/vit_base_test_labels.npz"
    emb_p = mm / "prepare" / "models" / "vit_base.npz"
    ndc_p = ndc / "nested_curvature_metrics.parquet"
    cprs_panel = cprs / "per_anchor_rank_curve.parquet"
    ad_panel = adcp / "datasets" / "physics_vit_base" / "per_anchor_curvature.parquet"
    desi_pq = root / "data_hf/desi/desi_vit_base.parquet"
    desi_lab = adcp / "cache" / "desi_smith42_labels.npz"

    artifact = {
        "git": _git(root),
        "discovery": {
            "tree": str(cprs),
            "config": _read_json(cprs / "config.json"),
            "parity": _read_json(cprs / "parity.json"),
            "reuse": _read_json(cprs / "reuse_manifest.json"),
            "complete": _read_json(cprs / "COMPLETE.json"),
            "n_anchors": len(disc),
            "anchor_rule": "d_replication_check_all512/anchor_ids.json (no re-hash of the 512)",
            "hash_prefix_scale_only": "cprs",
            "probe": DISCOVERY_PROBE_COLUMN,
            "probe_target": DISCOVERY_LABEL,
            "panel": str(cprs_panel),
            "panel_sha": file_sha(cprs_panel),
        },
        "adaptive": {
            "tree": str(adcp),
            "config": _read_json(adcp / "config.json"),
            "reuse": _read_json(adcp / "reuse_manifest.json"),
            "complete": _read_json(adcp / "COMPLETE.json"),
            "n_anchors": len(adap),
            "anchor_rule": "sha256(adcp:{seed}:{sample_id}) prefix of the same 512",
            "hash_prefix": "adcp",
            "probe": "catalog mag_r_desi from vit_base_test_labels.npz",
            "panel": str(ad_panel),
            "panel_sha": file_sha(ad_panel),
        },
        "embeddings": {
            "physics_X": str(emb_p),
            "physics_X_sha": file_sha_full(emb_p) if emb_p.exists() else "missing",
            "desi_parquet": str(desi_pq),
            "desi_parquet_sha": file_sha(desi_pq),
        },
        "labels": {
            "physics_npz": str(labels_p),
            "physics_npz_sha": file_sha_full(labels_p) if labels_p.exists() else "missing",
            "desi_cache": str(desi_lab),
            "desi_cache_sha": file_sha(desi_lab),
        },
        "neighbours": {
            "physics_knn": str(knn_p),
            "physics_knn_sha": file_sha(knn_p),
            "desi_knn": str(adcp / "datasets/desi_vit_base_hsc/knn_k2048.npz"),
            "desi_knn_sha": file_sha(adcp / "datasets/desi_vit_base_hsc/knn_k2048.npz"),
        },
        "curvature": {
            "nested_metrics": str(ndc_p),
            "nested_sha": file_sha(ndc_p),
            "parity_ranks": list(PARITY_RANKS),
        },
        "controls": list(SHARED_CORE_CONTROLS),
        "permutation": {
            "discovery_n_perm": _read_json(cprs / "config.json").get("config", {}).get("n_perm"),
            "adaptive_n_perm": _read_json(adcp / "config.json").get("thresholds", {}).get("n_perm"),
        },
    }
    write_json(out / "artifact_comparison.json", artifact, force=cfg.force)

    ad_hash = hash_select(disc, len(disc), seed=cfg.seed, prefix="adcp")
    cfg_diff = {
        "anchor_set_equal": set(disc) == set(adap),
        "anchor_order_equal": disc == adap,
        "adaptive_is_adcp_hash_of_discovery": ad_hash == adap,
        "probe_column_discovery": DISCOVERY_PROBE_COLUMN,
        "probe_column_adaptive": "catalog_mag_r_desi",
        "probe_quantity_equal": False,
        "hash_prefix_discovery_primary": None,
        "hash_prefix_adaptive": "adcp",
        "controls_names_equal": True,
        "control_source_discovery": "local_probe_fields.parquet (probe-field neighbourhood summaries)",
        "control_source_adaptive": "recomputed local_confounders from catalog y and kNN",
        "curvature_source_d8_20": "reused nested_curvature_metrics.parquet",
        "normalization": "unit L2 via multimodel prepare/models/vit_base.npz",
        "spherical_log": "nested_pca_frame / sphere_log_map (reused for d=8..20)",
        "ridge": "unchanged nested _fit_rank grid for reused ranks",
        "metric_whitening": "per-dim RMS of fit-split tangent coords (reused)",
        "first_divergence": "probe / label quantity: local_r2 vs catalog mag_r_desi",
    }
    write_json(out / "configuration_diff.json", cfg_diff, force=cfg.force)
    write_json(
        out / "audit_manifest.json",
        {
            "protocol": "adaptive_dataset_curvature_probe_audit_v1",
            "read_only": [
                cfg.adaptive_dir,
                cfg.rank_sweep_dir,
                cfg.nested_dir,
                cfg.multimodel_dir,
                cfg.qpd_dir,
            ],
            "write_only": cfg.output_dir,
            "n_perm": cfg.perm_boot()[0],
            "n_boot": cfg.perm_boot()[1],
            "smoke": cfg.smoke,
            "git": artifact["git"],
        },
        force=cfg.force,
    )
    return {
        "discovery_sids": disc,
        "adaptive_sids": adap,
        "shared_sids": shared,
        "artifact": artifact,
        "configuration_diff": cfg_diff,
        "knn": knn,
        "sid_to_ai": sid_to_ai,
    }
