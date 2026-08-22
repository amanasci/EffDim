"""Enumerate Smith42 / registry datasets before any correlation is computed."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from geometry.physics_activation_atlas.paths import resolve_path

from .config import (
    DISCOVERY_DATASET,
    DISCOVERY_LABEL,
    K_FRAC_OF_N,
    K_PRESET,
    MIN_VALID_ANCHORS,
    PRIMARY_ENCODER_FAMILY,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QPD,
    TARGET_ANCHORS,
)
from .pipeline import AdaptiveProbeConfig, file_sha, primary_k, write_df, write_json

# Canonical semantic labels. Orientation is from documented photometry /
# redshift conventions, frozen before association. Never flipped from ρ.
CANONICAL = {
    "mag_r": {
        "meaning": "r-band apparent magnitude; larger = fainter",
        "orientation": "fainter_positive",
        "mag_like": True,
        "group": "photometry",
    },
    "spec_z": {
        "meaning": "spectroscopic redshift",
        "orientation": "redshift_positive",
        "mag_like": False,
        "group": "redshift",
    },
    "photo_z": {
        "meaning": "photometric redshift",
        "orientation": "redshift_positive",
        "mag_like": False,
        "group": "redshift",
    },
    "smooth_fraction": {
        "meaning": "Galaxy Zoo smooth-or-featured smooth vote fraction",
        "orientation": "smoother_positive",
        "mag_like": False,
        "group": "morphology",
    },
    "stellar_mass": {
        "meaning": "log stellar mass (photo-z / NSA-style)",
        "orientation": "more_massive_positive",
        "mag_like": False,
        "group": "physical",
    },
    "sfr": {
        "meaning": "star-formation rate (log)",
        "orientation": "higher_sfr_positive",
        "mag_like": False,
        "group": "star_formation",
    },
}


def _pq_n(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _pq_cols(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _valid_count(y: np.ndarray, *, sentinel: float | None = -99.0) -> int:
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if sentinel is not None:
        m = m & (y != sentinel)
    return int(m.sum())


def extract_desi_labels(root: Path, dest: Path) -> dict[str, Any] | None:
    """Load catalog columns from the cached Smith42/DESI HF dataset (no images)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        z = np.load(dest)
        return {"path": str(dest), "n": int(len(z["spec_z"])), "reused": True}
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("Smith42/desi_hsc_crossmatched", split="train")
        keep = [c for c in ("Z", "ZERR", "ZWARN", "r_cmodel_mag", "g_cmodel_mag", "EBV", "desi_object_id", "hsc_object_id") if c in ds.column_names]
        sub = ds.select_columns(keep)
        spec_z = np.asarray(sub["Z"], dtype=np.float64)
        mag_r = np.asarray(sub["r_cmodel_mag"], dtype=np.float64)
        zwarn = np.asarray(sub["ZWARN"]).astype(np.int8) if "ZWARN" in keep else np.zeros(len(spec_z), dtype=np.int8)
        zerr = np.asarray(sub["ZERR"], dtype=np.float64) if "ZERR" in keep else np.full(len(spec_z), np.nan)
        oids = np.asarray(sub["desi_object_id"]) if "desi_object_id" in keep else np.arange(len(spec_z)).astype(str)
        np.savez_compressed(
            dest,
            spec_z=spec_z,
            mag_r=mag_r,
            ZERR=zerr,
            ZWARN=zwarn,
            desi_object_id=oids.astype(str),
        )
        return {"path": str(dest), "n": int(len(spec_z)), "reused": False}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def build_inventory(root: Path, cfg: AdaptiveProbeConfig) -> dict[str, Any]:
    """Write dataset_inventory.csv, physics_label_manifest.csv, inclusion_manifest.json."""
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)

    mm = resolve_path(root, SOURCE_MM)
    ndc = resolve_path(root, SOURCE_NDC)
    qpd = resolve_path(root, SOURCE_QPD)
    inv_rows: list[dict[str, Any]] = []
    lab_rows: list[dict[str, Any]] = []

    # ---- Smith42/galaxies (physics) ----
    phys_pq = root / "data_hf/physics/vit_base_test.parquet"
    phys_lab = root / "data_hf/physics/vit_base_test_labels.npz"
    sel_p = root / "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    other_enc = {
        "dinov3_vitb16": "data_hf/physics/dinov3_vitb16_test.parquet",
        "clip_base": "data_hf/physics/clip_base_test.parquet",
        "convnext_base": "data_hf/physics/convnext_base_test.parquet",
        "vit_large": "data_hf/physics/vit_large_test.parquet",
    }
    phys_n = _pq_n(phys_pq) if phys_pq.exists() else 0
    phys_hash = file_sha(phys_pq) if phys_pq.exists() else "missing"
    y_phys = dict(np.load(phys_lab)) if phys_lab.exists() else {}
    sel = np.load(sel_p) if sel_p.exists() else None
    # selection.npz stores the 16384-row index into the 86471 test table
    if sel is not None:
        sel_idx = None
        for k in sel.files:
            a = sel[k]
            if getattr(a, "ndim", 0) == 1 and len(a) in (16384, phys_n) and np.issubdtype(a.dtype, np.integer):
                sel_idx = np.asarray(a, dtype=np.int64)
                break
        if sel_idx is None:
            sel_idx = np.arange(16384, dtype=np.int64)
    else:
        sel_idx = np.arange(min(16384, phys_n), dtype=np.int64)

    def _phys_valid(key: str, sentinel=None) -> int:
        if key not in y_phys:
            return 0
        y = np.asarray(y_phys[key], dtype=np.float64)
        if sel_idx is not None and y.size == phys_n:
            y = y[sel_idx]
        return _valid_count(y, sentinel=sentinel)

    phys_labels = [
        ("mag_r_desi", "mag_r", None, True),
        ("smooth_fraction", "smooth_fraction", None, False),
        ("photo_z", "photo_z", None, False),
        ("stellar_mass", "stellar_mass", -99.0, False),
        ("sfr", "sfr", None, False),
    ]
    pk = primary_k(len(sel_idx))
    for raw, canon, sent, is_disc in phys_labels:
        n_valid = _phys_valid(raw, sent)
        n_full = _valid_count(np.asarray(y_phys[raw], dtype=np.float64), sentinel=sent) if raw in y_phys else 0
        under = n_valid < MIN_VALID_ANCHORS
        lab_rows.append(
            {
                "dataset_id": "physics_vit_base",
                "raw_column": raw,
                "canonical_label": canon,
                "is_discovery": bool(is_disc and raw == DISCOVERY_LABEL),
                "semantic_group": CANONICAL[canon]["group"],
                "meaning": CANONICAL[canon]["meaning"],
                "orientation": CANONICAL[canon]["orientation"],
                "mag_like": CANONICAL[canon]["mag_like"],
                "valid_full": n_full,
                "valid_geometry_subset": n_valid,
                "sentinel_treated_missing": sent,
                "underpowered": under,
                "include_in_association": (not under) and raw in y_phys,
                "same_objects_as": "physics_vit_base",
            }
        )

    inv_rows.append(
        {
            "dataset_id": "physics_vit_base",
            "registry_alias": "galaxies",
            "hf_dataset": "Smith42/galaxies",
            "hf_revision": "v2.0",
            "split": "test",
            "embedding_artifact": str(phys_pq),
            "embedding_hash": phys_hash,
            "representation": "vit_base_galaxies",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": phys_n,
            "n_geometry": int(len(sel_idx)),
            "embedding_dim": 768,
            "row_id": "galaxies test row index; geometry subset via selection.npz",
            "labels_align_exactly": True,
            "alignment_note": "labels.npz is row-aligned to vit_base_test.parquet; selection indexes both",
            "available_labels": "mag_r_desi,smooth_fraction,photo_z,stellar_mass,sfr",
            "confounders": "log_knn_radius,local_label_variance,local_evaluation_count",
            "cached_neighbours": str(mm / "model_neighbourhoods/vit_base_kmax2048.npz") if (mm / "model_neighbourhoods/vit_base_kmax2048.npz").exists() else "",
            "cached_eigensystems": str(ndc / "cache") if ndc.exists() else "",
            "cached_curvature": str(ndc / "nested_curvature_metrics.parquet") if (ndc / "nested_curvature_metrics.parquet").exists() else "",
            "primary_k": pk,
            "inclusion_status": "included",
            "exclusion_reason": "",
            "role": "discovery_for_mag_r_desi; confirmatory_other_labels_same_objects",
            "other_encoders": ",".join(k for k, p in other_enc.items() if (root / p).exists()),
        }
    )
    for enc, rel in other_enc.items():
        p = root / rel
        if not p.exists():
            continue
        inv_rows.append(
            {
                "dataset_id": f"physics_{enc}",
                "registry_alias": "galaxies",
                "hf_dataset": "Smith42/galaxies",
                "hf_revision": "v2.0",
                "split": "test",
                "embedding_artifact": str(p),
                "embedding_hash": file_sha(p),
                "representation": enc,
                "encoder_family": enc,
                "primary_family": False,
                "n_full": _pq_n(p),
                "n_geometry": _pq_n(p),
                "embedding_dim": "",
                "row_id": "same galaxies test rows as physics_vit_base",
                "labels_align_exactly": True,
                "alignment_note": "listed only; not mixed into the ViT-B primary replication",
                "available_labels": "same as physics_vit_base",
                "confounders": "",
                "cached_neighbours": "",
                "cached_eigensystems": "",
                "cached_curvature": "",
                "primary_k": "",
                "inclusion_status": "inventory_only_other_encoder",
                "exclusion_reason": "additional encoder; primary replication is the frozen ViT-B family",
                "role": "inventory",
                "other_encoders": "",
            }
        )

    # ---- Smith42/DESI ----
    desi_pq = root / "data_hf/desi/desi_vit_base.parquet"
    desi_lab_p = out / "cache" / "desi_smith42_labels.npz"
    desi_info = extract_desi_labels(root, desi_lab_p) if desi_pq.exists() and not cfg.skip_desi else None
    desi_n_emb = _pq_n(desi_pq) if desi_pq.exists() else 0
    desi_cols = _pq_cols(desi_pq) if desi_pq.exists() else []
    desi_ok = False
    desi_reason = ""
    if not desi_pq.exists():
        desi_reason = "local ViT-B embeddings missing"
    elif desi_info is None or "error" in (desi_info or {}):
        desi_reason = f"Smith42/desi_hsc_crossmatched catalog not loaded: {(desi_info or {}).get('error', 'unknown')}"
    elif int(desi_info["n"]) != desi_n_emb:
        desi_reason = f"catalog n={desi_info['n']} != embedding n={desi_n_emb}; refuse positional join"
    else:
        desi_ok = True
    desi_y = dict(np.load(desi_lab_p, allow_pickle=True)) if desi_ok and desi_lab_p.exists() else {}
    pk_desi = primary_k(desi_n_emb) if desi_n_emb else None
    if desi_ok:
        for raw, canon in (("spec_z", "spec_z"), ("mag_r", "mag_r")):
            n_valid = _valid_count(np.asarray(desi_y[raw], dtype=np.float64))
            lab_rows.append(
                {
                    "dataset_id": "desi_vit_base_hsc",
                    "raw_column": raw if raw != "spec_z" else "Z",
                    "canonical_label": canon,
                    "is_discovery": False,
                    "semantic_group": CANONICAL[canon]["group"],
                    "meaning": CANONICAL[canon]["meaning"],
                    "orientation": CANONICAL[canon]["orientation"],
                    "mag_like": CANONICAL[canon]["mag_like"],
                    "valid_full": n_valid,
                    "valid_geometry_subset": n_valid,
                    "sentinel_treated_missing": None,
                    "underpowered": n_valid < MIN_VALID_ANCHORS,
                    "include_in_association": n_valid >= MIN_VALID_ANCHORS,
                    "same_objects_as": "desi_vit_base_hsc",
                }
            )
    inv_rows.append(
        {
            "dataset_id": "desi_vit_base_hsc",
            "registry_alias": "desi",
            "hf_dataset": "Smith42/desi_hsc_crossmatched",
            "hf_revision": "main",
            "split": "train",
            "embedding_artifact": str(desi_pq),
            "embedding_hash": file_sha(desi_pq) if desi_pq.exists() else "missing",
            "representation": "vit_base_hsc",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": desi_n_emb,
            "n_geometry": desi_n_emb,
            "embedding_dim": 768,
            "row_id": "positional row index; desi_object_id stored in label cache",
            "labels_align_exactly": bool(desi_ok),
            "alignment_note": "catalog n matches embedding n; PU convention is positional alignment of UniverseTBD embeddings to Smith42 catalogs",
            "available_labels": "Z (spec_z), r_cmodel_mag (mag_r); ZWARN stored as bool (not DESI bitmask)",
            "confounders": "log_knn_radius,local_label_variance,local_evaluation_count; ZERR/EBV secondary only",
            "cached_neighbours": "",
            "cached_eigensystems": "",
            "cached_curvature": "",
            "primary_k": pk_desi,
            "inclusion_status": "included" if desi_ok else "excluded",
            "exclusion_reason": desi_reason,
            "role": "independent_replication" if desi_ok else "excluded",
            "other_encoders": "vit_base_desi (paired DESI-side ViT-B; inventory only)," + ",".join(c for c in desi_cols if c != "vit_base_hsc"),
        }
    )

    # ---- JWST ----
    jwst_pq = root / "data_hf/jwst/jwst_vit_base.parquet"
    jwst_n = _pq_n(jwst_pq) if jwst_pq.exists() else 0
    inv_rows.append(
        {
            "dataset_id": "jwst_vit_base_hsc",
            "registry_alias": "jwst",
            "hf_dataset": "Smith42/jwst_hsc_crossmatched",
            "hf_revision": "main",
            "split": "train",
            "embedding_artifact": str(jwst_pq),
            "embedding_hash": file_sha(jwst_pq) if jwst_pq.exists() else "missing",
            "representation": "vit_base_hsc",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": jwst_n,
            "n_geometry": jwst_n,
            "embedding_dim": 768,
            "row_id": "none in parquet; catalog has jwst_object_id / hsc_object_id",
            "labels_align_exactly": False,
            "alignment_note": "Smith42 catalog n=1667 vs embedding n=1496; refuse positional join",
            "available_labels": "HSC cmodel mags, mag_auto; no spec-z. a_z is extinction, not redshift",
            "confounders": "",
            "cached_neighbours": "",
            "cached_eigensystems": "",
            "cached_curvature": "",
            "primary_k": primary_k(jwst_n) if jwst_n else None,
            "inclusion_status": "excluded",
            "exclusion_reason": "catalog n=1667 != embedding n=1496; also underpowered for k=2048 (n*0.125<256)",
            "role": "excluded",
            "other_encoders": "vit_base_jwst,dinov3,clip,convnext,astropt,ijepa,vit-mae",
        }
    )

    # ---- Legacy ----
    leg_pq = root / "data_hf/legacysurvey/legacysurvey_vit_base.parquet"
    leg_n = _pq_n(leg_pq) if leg_pq.exists() else 0
    inv_rows.append(
        {
            "dataset_id": "legacy_vit_base_hsc",
            "registry_alias": "legacysurvey",
            "hf_dataset": "Smith42/legacysurvey_hsc_crossmatched",
            "hf_revision": "main",
            "split": "train",
            "embedding_artifact": str(leg_pq),
            "embedding_hash": file_sha(leg_pq) if leg_pq.exists() else "missing",
            "representation": "vit_base_hsc",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": leg_n,
            "n_geometry": leg_n,
            "embedding_dim": 768,
            "row_id": "none in parquet; catalog has legacysurvey_object_id / hsc_object_id",
            "labels_align_exactly": False,
            "alignment_note": "hub cache present (~166G) but processed datasets cache missing; no local label table",
            "available_labels": "documented HSC/Legacy photometry and EBV; no spec-z or stellar mass in the adapter",
            "confounders": "",
            "cached_neighbours": "",
            "cached_eigensystems": "",
            "cached_curvature": "",
            "primary_k": primary_k(leg_n) if leg_n else None,
            "inclusion_status": "excluded",
            "exclusion_reason": "Smith42 catalog not loadable from the processed HF cache in this run; embeddings have no joined labels",
            "role": "excluded",
            "other_encoders": "vit_base_legacysurvey,dino,dinov3,convnext,vit_large",
        }
    )

    # ---- SDSS ----
    inv_rows.append(
        {
            "dataset_id": "sdss_hsc",
            "registry_alias": "sdss",
            "hf_dataset": "Smith42/sdss_hsc_crossmatched",
            "hf_revision": "main",
            "split": "train",
            "embedding_artifact": "",
            "embedding_hash": "missing",
            "representation": "",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": 2319,
            "n_geometry": 0,
            "embedding_dim": "",
            "row_id": "catalog only",
            "labels_align_exactly": False,
            "alignment_note": "HF catalog has Z / Z_ERR / r_cmodel_mag (n=2319); no local embedding parquet",
            "available_labels": "Z, Z_ERR, ZWARNING, VDISP, r_cmodel_mag",
            "confounders": "",
            "cached_neighbours": "",
            "cached_eigensystems": "",
            "cached_curvature": "",
            "primary_k": None,
            "inclusion_status": "excluded",
            "exclusion_reason": "no local ViT-B embeddings in data_hf",
            "role": "excluded",
            "other_encoders": "",
        }
    )

    # ---- CosmosWeb (not Smith42) ----
    cw_pq = root / "data_hf/cosmosweb/hsc_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_vit_base_45000.parquet"
    inv_rows.append(
        {
            "dataset_id": "cosmosweb_vit_base_hsc",
            "registry_alias": "cosmosweb",
            "hf_dataset": "Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2",
            "hf_revision": "",
            "split": "train",
            "embedding_artifact": str(cw_pq) if cw_pq.exists() else "",
            "embedding_hash": file_sha(cw_pq) if cw_pq.exists() else "missing",
            "representation": "vit_base",
            "encoder_family": PRIMARY_ENCODER_FAMILY,
            "primary_family": True,
            "n_full": _pq_n(cw_pq) if cw_pq.exists() else 0,
            "n_geometry": _pq_n(cw_pq) if cw_pq.exists() else 0,
            "embedding_dim": 768,
            "row_id": "none",
            "labels_align_exactly": False,
            "alignment_note": "adapter CATALOG_COLUMNS documents lephare_photozs/lp_mass/lp_ssfr/mags but prepare() drops them; HF not in local cache",
            "available_labels": "documented in cosmosweb.CATALOG_COLUMNS; not joined",
            "confounders": "",
            "cached_neighbours": "",
            "cached_eigensystems": "",
            "cached_curvature": "",
            "primary_k": primary_k(_pq_n(cw_pq)) if cw_pq.exists() else None,
            "inclusion_status": "excluded",
            "exclusion_reason": "not a Smith42 catalog; labels not locally aligned",
            "role": "excluded",
            "other_encoders": "dinov3; JWST-side ViT-B",
        }
    )

    inv = pd.DataFrame(inv_rows)
    labs = pd.DataFrame(lab_rows)
    included = inv[inv.inclusion_status == "included"]["dataset_id"].tolist()
    excluded = inv[inv.inclusion_status != "included"][["dataset_id", "inclusion_status", "exclusion_reason"]].to_dict("records")
    assoc_labels = labs[labs.include_in_association == True]  # noqa: E712
    discovery = [r for r in assoc_labels.to_dict("records") if r["is_discovery"]]
    confirmatory = [r for r in assoc_labels.to_dict("records") if not r["is_discovery"]]
    manifest = {
        "protocol": "adaptive_dataset_curvature_probe_v1",
        "unit_of_analysis": "dataset",
        "primary_encoder_family": PRIMARY_ENCODER_FAMILY,
        "discovery": {"dataset_id": DISCOVERY_DATASET, "label": DISCOVERY_LABEL},
        "k_rule": f"largest k in {list(K_PRESET)} with k <= {K_FRAC_OF_N} * n_obs",
        "min_valid_anchors": MIN_VALID_ANCHORS,
        "target_anchors": TARGET_ANCHORS,
        "included_datasets": included,
        "excluded": excluded,
        "confirmatory_family": [
            {"dataset_id": r["dataset_id"], "canonical_label": r["canonical_label"]}
            for r in confirmatory
        ],
        "same_object_groups": {
            "physics_vit_base": [r["canonical_label"] for r in assoc_labels.to_dict("records") if r["dataset_id"] == "physics_vit_base"],
            "desi_vit_base_hsc": [r["canonical_label"] for r in assoc_labels.to_dict("records") if r["dataset_id"] == "desi_vit_base_hsc"],
        },
        "do_not_reverse_orientation_from_results": True,
        "no_association_yet": True,
        "desi_label_cache": desi_info,
    }
    write_df(out / "dataset_inventory.csv", inv, force=cfg.force)
    write_df(out / "physics_label_manifest.csv", labs, force=cfg.force)
    write_json(out / "inclusion_manifest.json", manifest, force=cfg.force)
    return {"inventory": inv, "labels": labs, "manifest": manifest}
