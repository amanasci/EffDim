"""Phase 4: DESI alignment proof. Equal row count is not sufficient."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .pipeline import AuditConfig, file_sha, write_df, write_json


def run_desi_alignment(root: Path, cfg: AuditConfig) -> dict[str, Any]:
    out = cfg.resolved(root)
    adcp = cfg.adcp(root)
    emb_p = root / "data_hf/desi/desi_vit_base.parquet"
    lab_p = adcp / "cache" / "desi_smith42_labels.npz"
    checks = []

    n_emb = int(pq.ParquetFile(emb_p).metadata.num_rows) if emb_p.exists() else 0
    cols = list(pq.read_schema(emb_p).names) if emb_p.exists() else []
    id_like = [c for c in cols if any(s in c.lower() for s in ("id", "obj", "target", "ra", "dec", "index"))]
    checks.append({"check": "embedding_exists", "pass": emb_p.exists(), "detail": str(emb_p)})
    checks.append({"check": "embedding_n", "pass": n_emb == 20465, "detail": n_emb})
    checks.append({"check": "embedding_has_object_id", "pass": False, "detail": f"columns={cols}; id_like={id_like}"})
    checks.append(
        {
            "check": "equal_row_count_is_not_proof",
            "pass": False,
            "detail": "catalog n == embedding n is recorded but is not an alignment proof",
        }
    )

    n_lab = 0
    has_oid = False
    if lab_p.exists():
        z = np.load(lab_p, allow_pickle=True)
        n_lab = int(len(z["spec_z"]))
        has_oid = "desi_object_id" in z.files
        checks.append({"check": "label_cache_exists", "pass": True, "detail": str(lab_p)})
        checks.append({"check": "label_n", "pass": n_lab == n_emb, "detail": n_lab})
        checks.append({"check": "catalog_has_desi_object_id", "pass": has_oid, "detail": "present in label cache only"})
        checks.append(
            {
                "check": "shared_identifier_in_both_artifacts",
                "pass": False,
                "detail": "embedding parquet has only vit_base_hsc and vit_base_desi; no object id to join",
            }
        )
    else:
        checks.append({"check": "label_cache_exists", "pass": False, "detail": "missing"})

    # Look for a generation manifest that would prove source order.
    manifest_hits = []
    for rel in (
        "data_hf/desi/README.md",
        "data_hf/desi/manifest.json",
        "outputs/sae_shared_basis/paper_table2_official/hf_cache/UniverseTBD__desi_hsc_embeddings",
    ):
        p = root / rel
        manifest_hits.append({"path": rel, "exists": p.exists()})
    checks.append(
        {
            "check": "embedding_generation_manifest_proves_catalog_order",
            "pass": False,
            "detail": "UniverseTBD/desi_hsc_embeddings export is a different official pair table; no proven Smith42 catalog row order",
        }
    )
    checks.append(
        {
            "check": "reproducible_reconstruction_from_catalog",
            "pass": False,
            "detail": "no stable source-row identifier in the embedding parquet",
        }
    )
    checks.append(
        {
            "check": "did_not_search_for_correlation_maximizing_permutation",
            "pass": True,
            "detail": "forbidden by protocol",
        }
    )

    proved = False
    status = "desi_label_alignment_unresolved"
    proof = {
        "status": status,
        "proved": proved,
        "embedding_n": n_emb,
        "catalog_n": n_lab,
        "embedding_columns": cols,
        "shared_metadata": [],
        "manifest_hits": manifest_hits,
        "scientific_use": "retain DESI geometry; do not report DESI curvature–label associations as scientific results",
        "hashes": {"embedding": file_sha(emb_p), "labels": file_sha(lab_p)},
    }
    write_json(out / "desi_alignment_proof.json", proof, force=cfg.force)
    write_df(out / "desi_alignment_checks.csv", pd.DataFrame(checks), force=cfg.force)
    return proof
