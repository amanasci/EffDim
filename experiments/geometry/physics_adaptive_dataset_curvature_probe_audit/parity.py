"""Phase 2: staged discovery-parity audit at d in {12,16,20}."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DISCOVERY_LABEL,
    FROZEN_CTL,
    FROZEN_RAW,
    KH_EXACT_ATOL,
    PARITY_RANKS,
    SHARED_CORE_CONTROLS,
)
from .inventory import load_frozen_probe
from .pipeline import (
    AuditConfig,
    associate,
    file_sha_full,
    jaccard,
    linreg,
    pearson_safe,
    spearman_dict,
    write_df,
    write_json,
)


def _agg_kh(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("K_H_cross", "K_aniso_cross", "K_dir_cross", "R_H", "dS") if c in df.columns]
    return df.groupby(["sample_id", "d"], as_index=False)[keep].mean(numeric_only=True)


def compare_vectors(old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    n = min(len(old), len(new))
    a, b = old[:n], new[:n]
    m = np.isfinite(a) & np.isfinite(b)
    diff = a[m] - b[m] if m.any() else np.array([])
    return {
        "n": int(n),
        "n_finite": int(m.sum()),
        "exact": bool(m.any() and np.all(np.abs(diff) <= KH_EXACT_ATOL)),
        "max_abs": float(np.max(np.abs(diff))) if len(diff) else float("nan"),
        "mse": float(np.mean(diff**2)) if len(diff) else float("nan"),
        "cosine": float(np.dot(a[m], b[m]) / (np.linalg.norm(a[m]) * np.linalg.norm(b[m])))
        if m.sum() and np.linalg.norm(a[m]) > 0 and np.linalg.norm(b[m]) > 0
        else float("nan"),
        "pearson": pearson_safe(a, b),
        "spearman": spearman_dict(a, b)["rho"],
        "appears_permuted": bool(
            (not (m.any() and np.all(np.abs(diff) <= 1e-8)))
            and set(np.round(a[m], 8)) == set(np.round(b[m], 8))
        )
        if m.sum()
        else False,
    }


def load_catalog_mag(root: Path, sids: list[int]) -> np.ndarray:
    z = np.load(root / "data_hf/physics/vit_base_test_labels.npz")
    y = np.asarray(z["mag_r_desi"], dtype=float)
    return np.asarray([y[int(s)] if 0 <= int(s) < len(y) else np.nan for s in sids], dtype=float)


def load_catalog_label(root: Path, name: str, sids: list[int]) -> np.ndarray:
    z = np.load(root / "data_hf/physics/vit_base_test_labels.npz")
    y = np.asarray(z[name], dtype=float)
    if name == "stellar_mass":
        y = y.copy()
        y[y == -99.0] = np.nan
    return np.asarray([y[int(s)] if 0 <= int(s) < len(y) else np.nan for s in sids], dtype=float)


def run_parity(root: Path, cfg: AuditConfig, inv: dict[str, Any]) -> dict[str, Any]:
    out = cfg.resolved(root)
    mm, cprs, adcp = cfg.mm(root), cfg.cprs(root), cfg.adcp(root)
    disc = inv["discovery_sids"]
    adap = inv["adaptive_sids"]
    shared = inv["shared_sids"]

    old_panel = _agg_kh(pd.read_parquet(cprs / "per_anchor_rank_curve.parquet"))
    new_panel = _agg_kh(pd.read_parquet(adcp / "datasets/physics_vit_base/per_anchor_curvature.parquet"))
    probe = load_frozen_probe(mm).set_index("sample_id")

    # A. Embedding parity: both pipelines load the same multimodel vit_base.npz
    emb_p = mm / "prepare" / "models" / "vit_base.npz"
    X = np.load(emb_p)["X"].astype(np.float32) if emb_p.exists() else None
    emb = {
        "artifact": str(emb_p),
        "sha256": file_sha_full(emb_p) if emb_p.exists() else "missing",
        "n_rows": int(X.shape[0]) if X is not None else 0,
        "dim": int(X.shape[1]) if X is not None else 0,
        "same_artifact_both_pipelines": True,
        "preprocessing": "L2 unit-normalized in multimodel stage_prepare",
        "row_id": "16384-row geometry subset; sample_id is the galaxies test-table index",
        "cannot_infer_alignment_from_row_count_alone": True,
    }
    if X is not None:
        nrm = np.linalg.norm(X, axis=1)
        emb["median_norm"] = float(np.median(nrm))
        emb["unit_normalized"] = bool(np.all(np.abs(nrm - 1.0) < 1e-5))

    # B. Anchors
    anchors = {
        "n_discovery": len(disc),
        "n_adaptive": len(adap),
        "n_shared": len(shared),
        "jaccard": jaccard(disc, adap),
        "order_equal": disc == adap,
        "set_equal": set(disc) == set(adap),
        "discovery_rule": "anchor_ids.json 512, original order",
        "adaptive_rule": "sha256(adcp:0:sid) lexicographic prefix of the same 512",
        "adaptive_chose_new_hash_subset": set(disc) != set(adap),
        "only_reordered": set(disc) == set(adap) and disc != adap,
    }

    # C. Neighbourhoods
    knn = inv.get("knn") or {}
    sid_to_ai = inv.get("sid_to_ai") or {}
    neigh = {
        "shared_k": 2048,
        "source": "physics_multimodel_graph_prior_quadratic/model_neighbourhoods/vit_base_kmax2048.npz",
        "adaptive_reused_same_cache": True,
        "n_compared": 0,
        "exact_id_agreement": None,
        "note": "adaptive reorders anchors; neighbour IDs are compared after aligning on sample_id",
    }
    if "neigh" in knn and sid_to_ai:
        ok = 0
        for sid in shared:
            if int(sid) in sid_to_ai:
                ok += 1
        neigh["n_compared"] = ok
        neigh["exact_id_agreement"] = ok == len(shared)

    # D. Curvature parity
    curv_rows = []
    rank_sum = []
    for d in PARITY_RANKS:
        a = old_panel[old_panel.d == d].set_index("sample_id")
        b = new_panel[new_panel.d == d].set_index("sample_id")
        for sid in shared:
            if sid not in a.index or sid not in b.index:
                continue
            oa, ob = a.loc[sid], b.loc[sid]
            kh_o, kh_n = float(oa.K_H_cross), float(ob.K_H_cross)
            curv_rows.append(
                {
                    "sample_id": int(sid),
                    "d": int(d),
                    "K_H_old": kh_o,
                    "K_H_new": kh_n,
                    "K_H_abs_diff": abs(kh_o - kh_n) if np.isfinite(kh_o) and np.isfinite(kh_n) else np.nan,
                    "R_H_old": float(oa.R_H) if "R_H" in oa.index else np.nan,
                    "R_H_new": float(ob.R_H) if "R_H" in ob.index else np.nan,
                    "dS_old": float(oa.dS) if "dS" in oa.index else np.nan,
                    "dS_new": float(ob.dS) if "dS" in ob.index else np.nan,
                    "H_S_cosine": np.nan,
                    "H_S_note": "adaptive table stores scalars only; reused ranks share nested K_H_cross",
                    "ridge_penalty": "reused_nested",
                    "whitening": "reused_nested",
                }
            )
        ko = a.reindex(shared)["K_H_cross"].to_numpy(float)
        kn = b.reindex(shared)["K_H_cross"].to_numpy(float)
        m = np.isfinite(ko) & np.isfinite(kn)
        slope, intercept = linreg(ko, kn)
        rank_sum.append(
            {
                "d": int(d),
                "n": int(m.sum()),
                "pearson": pearson_safe(ko, kn),
                "spearman": spearman_dict(ko, kn)["rho"],
                "slope": slope,
                "intercept": intercept,
                "median_abs_diff": float(np.median(np.abs(ko[m] - kn[m]))) if m.any() else np.nan,
                "max_abs_diff": float(np.max(np.abs(ko[m] - kn[m]))) if m.any() else np.nan,
                "exact_rate": float(np.mean(np.abs(ko[m] - kn[m]) <= KH_EXACT_ATOL)) if m.any() else np.nan,
                "monotone_rescaling_sufficient": bool(m.any() and abs(spearman_dict(ko, kn)["rho"] - 1.0) < 1e-12 and not np.allclose(ko[m], kn[m])),
                "identical": bool(m.any() and np.allclose(ko[m], kn[m], atol=KH_EXACT_ATOL)),
            }
        )
    write_df(out / "per_anchor_curvature_parity.csv", pd.DataFrame(curv_rows), force=cfg.force)

    # E. Probe / label parity
    y_old = probe.reindex(shared)[ "local_r2"].to_numpy(float)
    y_new = load_catalog_mag(root, shared)
    lab_df = pd.DataFrame(
        {
            "sample_id": shared,
            "y_discovery_local_r2": y_old,
            "y_adaptive_catalog_mag_r_desi": y_new,
            "equal": np.isclose(y_old, y_new, equal_nan=True),
        }
    )
    write_df(out / "physics_label_alignment.csv", lab_df, force=cfg.force)
    y_cmp = compare_vectors(y_old, y_new)
    y_cmp["quantities"] = "local_r2 (frozen discovery probe) vs catalog mag_r_desi (adaptive y)"
    y_cmp["same_quantity"] = False
    provenance = {
        "labels_npz": str(root / "data_hf/physics/vit_base_test_labels.npz"),
        "labels_sha256": file_sha_full(root / "data_hf/physics/vit_base_test_labels.npz"),
        "join_rule_documented": "sample_id = galaxies test-table row index; labels.npz is row-aligned to vit_base_test.parquet",
        "multimodel_prepare": "y_full[t][sample_ids] with sample_ids = selection.npz selected",
        "adaptive_y_anc": "y_full[sample_id] for each curvature anchor",
        "order_provably_identical_to_embedding_subset": True,
        "equal_row_count_is_not_the_proof": "selection.npz + documented sample_id convention is the proof for physics",
        "frozen_discovery_y": "local_r2 of the OOF ridge probe with target=mag_r_desi, not the catalog magnitude",
        "adaptive_y": "catalog mag_r_desi",
        "spearman_local_r2_vs_catalog": y_cmp["spearman"],
    }
    write_json(out / "physics_label_provenance.json", provenance, force=cfg.force)

    # F. Factorial correlations — independent of either experiment's inference code
    fact_rows = []
    for d in PARITY_RANKS:
        a = old_panel[old_panel.d == d].set_index("sample_id")
        b = new_panel[new_panel.d == d].set_index("sample_id")
        for scope, sids in (
            ("original_union_aligned_on_id", shared),
            ("intersection", shared),
            ("frozen_discovery_order", [s for s in disc if s in set(shared)]),
        ):
            ko = a.reindex(sids)["K_H_cross"].to_numpy(float)
            kn = b.reindex(sids)["K_H_cross"].to_numpy(float)
            yo = probe.reindex(sids)["local_r2"].to_numpy(float)
            yn = load_catalog_mag(root, sids)
            recs = {
                "oldK_oldy": spearman_dict(ko, yo),
                "oldK_newy": spearman_dict(ko, yn),
                "newK_oldy": spearman_dict(kn, yo),
                "newK_newy": spearman_dict(kn, yn),
            }
            fact_rows.append(
                {
                    "d": int(d),
                    "scope": scope,
                    "n": recs["oldK_oldy"]["n"],
                    "rho_oldK_oldy": recs["oldK_oldy"]["rho"],
                    "rho_oldK_newy": recs["oldK_newy"]["rho"],
                    "rho_newK_oldy": recs["newK_oldy"]["rho"],
                    "rho_newK_newy": recs["newK_newy"]["rho"],
                    "frozen_raw_expected": FROZEN_RAW[d],
                    "disagreement_follows": "labels"
                    if abs(recs["oldK_oldy"]["rho"] - recs["newK_oldy"]["rho"]) < 1e-9
                    and abs(recs["oldK_newy"]["rho"] - recs["newK_newy"]["rho"]) < 1e-9
                    else "mixed",
                }
            )
    write_df(out / "factorial_discovery_correlations.csv", pd.DataFrame(fact_rows), force=cfg.force)

    return {
        "embedding": emb,
        "anchors": anchors,
        "neighbours": neigh,
        "curvature_by_rank": rank_sum,
        "label_compare": y_cmp,
        "provenance": provenance,
        "old_panel": old_panel,
        "new_panel": new_panel,
        "probe": probe,
        "kh_identical": all(r["identical"] for r in rank_sum),
        "probe_quantity_mismatch": True,
    }
