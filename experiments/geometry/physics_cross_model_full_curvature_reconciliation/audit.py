"""Phase 0: recover the historical full-curvature definition and stored result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import HISTORICAL_FULL_METRIC, PRIMARY_D, PRIMARY_K, SOURCE_CMCLA, SOURCE_EDM, SOURCE_NDC
from .io_util import resolve_path, sha256_file, sha256_file16, write_json, write_text


SOURCE_REL = {
    "effdim_metrics": "experiments/geometry/physics_activation_atlas/effdim_curvature_metrics.py",
    "nested_fit_rank": "experiments/geometry/physics_activation_atlas/nested_dimension_curvature.py",
    "fit_quad": "experiments/geometry/physics_activation_atlas/full_curvature_audit.py",
    "unpack_BS": "experiments/geometry/physics_activation_atlas/confirmatory_object_curvature.py",
    "sphere_project": "experiments/geometry/physics_activation_atlas/sphere_normal_quadratic.py",
}

FORMULA = (
    "K_dir_cross = <H_A, H_B> + [2/(d(d+2))] <B0_A, B0_B>_F "
    "with H = (1/d) tr B^S in ambient space and B0 = B^S - H ⊗ I. "
    "Algebraically identical to "
    "(2 <B_A^S, B_B^S>_F + <tr B_A^S, tr B_B^S>) / (d(d+2))."
)


def _hash_if(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "sha256_16": sha256_file16(path),
        "bytes": int(path.stat().st_size),
    }


def run_audit(shared: dict, out: Path) -> dict[str, Any]:
    root: Path = shared["root"]
    sources = {k: _hash_if(root / rel) for k, rel in SOURCE_REL.items()}
    artifacts = {
        "edm_cross_model_summary": _hash_if(resolve_path(root, SOURCE_EDM) / "cross_model_summary.csv"),
        "edm_associations": _hash_if(resolve_path(root, SOURCE_EDM) / "associations_sequential_controls.csv"),
        "edm_curvature_metrics": _hash_if(resolve_path(root, SOURCE_EDM) / "curvature_metrics.parquet"),
        "edm_report": _hash_if(resolve_path(root, SOURCE_EDM) / "REPORT.md"),
        "ndc_metrics": _hash_if(resolve_path(root, SOURCE_NDC) / "nested_curvature_metrics.parquet"),
        "cmcla_decision": _hash_if(resolve_path(root, SOURCE_CMCLA) / "decision.json"),
        "cmcla_vitb_probes": _hash_if(
            resolve_path(root, SOURCE_CMCLA) / "probes" / "vit_base_anchor_metrics.parquet"
        ),
    }

    remembered = {}
    edm_sum = resolve_path(root, SOURCE_EDM) / "cross_model_summary.csv"
    if edm_sum.exists():
        sm = pd.read_csv(edm_sum)
        hit = sm[
            (sm.target == "mag_r_desi")
            & (sm.metric == "K_dir_cross")
            & (sm.k == PRIMARY_K)
        ]
        if len(hit):
            remembered = {
                "source": str(edm_sum),
                "target": "mag_r_desi local_r2 (global five-fold OOF)",
                "k": PRIMARY_K,
                "dimension_convention": "per-model graph-effective d* (not frozen d=16)",
                "median_spearman": float(hit.iloc[0].median_spearman),
                "frac_neg": float(hit.iloc[0].frac_neg),
                "n_models": int(hit.iloc[0].n_models),
            }

    ndc_ok = False
    ndc_note = ""
    ndc_p = resolve_path(root, SOURCE_NDC) / "nested_curvature_metrics.parquet"
    if ndc_p.exists():
        ndc = pd.read_parquet(ndc_p)
        sub = ndc[(ndc.d == PRIMARY_D) & (ndc.k == PRIMARY_K)]
        ndc_ok = "K_dir_cross" in sub.columns and len(sub) >= 100
        ndc_note = f"n_rows_d16={len(sub)} n_anchors={sub.sample_id.nunique() if len(sub) else 0}"

    definition_ok = all(sources[k]["exists"] for k in ("effdim_metrics", "nested_fit_rank", "unpack_BS"))
    stored_full_ok = bool(ndc_ok and remembered.get("frac_neg") == 1.0)
    blocked = not definition_ok or not ndc_ok

    payload = {
        "ok": bool(definition_ok and ndc_ok),
        "blocked": blocked,
        "historical_primary_metric": HISTORICAL_FULL_METRIC,
        "formula": FORMULA,
        "used_full_fitted_hessian": True,
        "estimator": "split_half_cross via nested_dimension_curvature._fit_rank → fit_quad",
        "n_splits_historical_ndc": 5,
        "support_radius_normalization": "none on the reported scalar; neighbourhood radius is a control only",
        "tangent_metric": (
            "frozen orthonormal PCA frame after sphere_project_basis; "
            "U = (X-x0)J is not re-whitened before Phi(U); Euclidean Frobenius is correct"
        ),
        "packed_off_diagonal": (
            "BS_flat stores 2 B_ab for a<b; unpack_BS_symmetric divides by 2; "
            "contractions use the unpacked (D,d,d) tensor so off-diagonals are counted twice"
        ),
        "reported_transform": (
            "raw signed split-cross; K_*_cross_plot = max(cross, 0) is visualization only; "
            "rank correlations use the signed cross"
        ),
        "anchor_filter_historical": "all 512 frozen anchors; no probe-association filter",
        "probe_outcome_historical": "local_r2 of the frozen five-fold OOF global probe of mag_r_desi",
        "controls_historical_sequential": [
            "log_knn_radius",
            "local_label_variance",
            "local_evaluation_count",
            "recon",
            "boundary",
        ],
        "controls_this_reconciliation": [
            "log_knn_radius",
            "local_label_variance",
            "local_evaluation_count",
        ],
        "rank_correlation": "Spearman; this reconciliation uses the frozen Freedman–Lane / associate()",
        "remembered_cross_model_raw_K_dir": remembered,
        "stored_full_curvature_parity_target": {
            "artifact": str(ndc_p),
            "ok": ndc_ok,
            "note": ndc_note,
            "d": PRIMARY_D,
            "k": PRIMARY_K,
        },
        "not_the_primary": {
            "K_H_cross": "mean-curvature trace comparator used by the completed cross-model paper",
            "prior_decision_label": "representation_specific_effect (trace estimand only; not overwritten)",
        },
        "sources": sources,
        "artifacts": artifacts,
        "code_paths": {
            "definition": SOURCE_REL["effdim_metrics"] + ":cross_metric_pair",
            "fit": SOURCE_REL["nested_fit_rank"] + ":_fit_rank",
            "quadratic": SOURCE_REL["fit_quad"] + ":fit_quad",
        },
    }
    write_json(out / "curvature_definition_audit.json", payload, force=True)
    write_text(out / "CURVATURE_DEFINITION_AUDIT.md", _md(payload), force=True)
    if blocked:
        write_text(
            out / "BLOCKER.md",
            _blocker_md(payload),
            force=True,
        )
    return payload


def _md(p: dict[str, Any]) -> str:
    rem = p.get("remembered_cross_model_raw_K_dir") or {}
    src_lines = "\n".join(
        f"- `{k}`: `{v.get('path')}` sha256=`{v.get('sha256_16', 'missing')}` exists={v.get('exists')}"
        for k, v in p["sources"].items()
    )
    art_lines = "\n".join(
        f"- `{k}`: `{v.get('path')}` sha256=`{v.get('sha256_16', 'missing')}` exists={v.get('exists')}"
        for k, v in p["artifacts"].items()
    )
    return f"""# Curvature definition audit

This reconciliation does **not** replace the historical full-curvature scalar
with a newly convenient formula. The recovered primary estimand is
`{p['historical_primary_metric']}`.

## Formula

{p['formula']}

## Estimator

- Full fitted Hessian of the local quadratic chart (`fit_quad`), not a
  mean-curvature-only reduction at fit time.
- Split-half cross estimator (`_fit_rank`, 5 A/B splits).
- No support-radius normalization of the reported scalar.
- Tangent metric: {p['tangent_metric']}.
- Packed off-diagonals: {p['packed_off_diagonal']}.
- Reported transform: {p['reported_transform']}.

## Remembered cross-model association

The completed `physics_effdim_curvature_metrics` run stored a uniformly
negative raw Spearman between `K_dir_cross` and global `local_r2` for
`mag_r_desi` at k=2048 across all five encoders:

- median Spearman = `{rem.get('median_spearman')}`
- fraction negative = `{rem.get('frac_neg')}`
- n models = `{rem.get('n_models')}`
- dimension convention: `{rem.get('dimension_convention')}`

That historical table used per-model graph-effective d*, not the frozen
d=16 of the later trace-based cross-model experiment. This reconciliation
keeps the **same scalar definition** and applies it at the frozen d=16
charts. Parity is required against the stored ViT-B d=16 `K_dir_cross`
values in `physics_nested_dimension_curvature`.

## What this is not

`K_H^cross = <H_A, H_B>` is the **trace comparator**
used by the completed cross-model paper. Its decision label
`representation_specific_effect` remains correct for that estimand and is
not overwritten here.

## Source hashes

{src_lines}

## Artifact hashes

{art_lines}

## Gate

- definition recovered: `{p['ok']}`
- blocked: `{p['blocked']}`
"""


def _blocker_md(p: dict[str, Any]) -> str:
    return (
        "# BLOCKER\n\n"
        "The historical full-curvature definition or a stored full-curvature "
        "result could not be recovered unambiguously. Scientific inference "
        "is stopped.\n\n"
        f"```json\n{json.dumps({k: p[k] for k in ('ok', 'blocked', 'formula', 'stored_full_curvature_parity_target')}, indent=2)}\n```\n"
    )
