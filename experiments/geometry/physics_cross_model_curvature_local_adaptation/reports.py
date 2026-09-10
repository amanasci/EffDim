"""METHODS.md, REPORT.md, TARGET_DEFINITION.md, reuse_manifest."""

from __future__ import annotations

from pathlib import Path

from .config import (
    CATALOG_FIELD,
    CONTROLS,
    N_SHUFFLE,
    POSITIVE_CONTROL,
    PRIMARY_D,
    PRIMARY_K,
    PROBE_ALPHA,
    QLCA_A_B,
    QLCA_MEDIAN_DELTA_Q,
    QLCA_PARTIAL_ADAPT,
    QLCA_RHO_KH_DQ,
    R_H_FAIL,
    SOURCE_CPRS,
    SOURCE_LPA,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QLCA,
    SOURCE_QLCA_AUDIT,
    WEIGHT_COS_RELIABLE,
)
from .io_util import write_json


def write_target_definition(out: Path) -> None:
    (out / "TARGET_DEFINITION.md").write_text(
        f"""# Target definition

Primary photometric target: catalog field `{CATALOG_FIELD}` (apparent r-band magnitude).
This is an astronomical photometric observable, not a fundamental physical property
and not a spectroscopic DESI label.

Global-decoding outcomes (never substitute the catalog vector):

- `r2_G`: local OOF R^2 of the **global** five-fold ridge probe (alpha={PROBE_ALPHA})
  evaluated on the anchor neighbourhood.
- `mse_G`: local OOF MSE of the same global probe.

Local-adaptation outcomes:

- `mse_P`: local OOF MSE of the **patch** ambient ridge probe, trained only on
  neighbourhood objects from outer-train folds, evaluated on the same outer-test
  objects as G.
- `delta_adapt = mse_G - mse_P`. Positive means patch better on that anchor.

Alignment key: `sample_id` (object identity). Neighbourhood indices are local rows
into the embedding table; they are joined through `sample_folds.parquet`.

QLCA (ViT-B only, reused, not rerun): median Delta_Q~{QLCA_MEDIAN_DELTA_Q},
rho_ctl(K_H,Delta_Q)~{QLCA_RHO_KH_DQ}, A_B~{QLCA_A_B}. Conditioning the adaptation
association on Delta_Q raises it to ~{QLCA_PARTIAL_ADAPT}; quadratic structure does
not mediate local adaptation.
"""
    )


def write_reuse_manifest(out: Path, inventory: dict, parity: dict) -> None:
    write_json(
        out / "reuse_manifest.json",
        {
            "read_only_inputs": [
                SOURCE_MM,
                SOURCE_CPRS,
                SOURCE_LPA,
                SOURCE_NDC,
                SOURCE_QLCA,
                SOURCE_QLCA_AUDIT,
            ],
            "vit_base_KH": f"{SOURCE_CPRS}/per_anchor_rank_curve.parquet (d=16,k=2048)",
            "vit_base_G_P_parity": f"{SOURCE_LPA}/anchor_improvements.csv",
            "shared_folds": f"{SOURCE_MM}/sample_folds.parquet",
            "eligible_models": inventory.get("eligible_model_ids"),
            "parity_ok": parity.get("ok"),
            "wrote_into_preserved": False,
        },
        force=True,
    )


def write_methods(out: Path, cfg) -> None:
    (out / "METHODS.md").write_text(
        f"""# METHODS

Frozen confirmatory geometry: d={PRIMARY_D}, k={PRIMARY_K},
K_H^cross = <H^(A), H^(B)> (split-half inner product of sphere-normal mean curvature).
Estimator: nested_dimension_curvature._fit_rank (n_splits={cfg.n_splits_kh()}).
ViT-B curvature is reused from the rank-sweep panel, not refit.

Reliability: a model is `geometry_unreliable` if median R_H ≤ {R_H_FAIL}.

Probes: global G is the frozen five-fold OOF ridge (α={PROBE_ALPHA}, sum-of-squares).
Patch P uses the same outer-fold IDs, train on fold≠f inside the neighbourhood,
evaluate on fold=f. Confirmatory estimator is **fixed** α={PROBE_ALPHA} (LPA definition).
G-cal: intercept-only recalibration of G on patch train.
G-affine-cal (C): slope+intercept recalibration of G on patch train.
T: ridge in the frozen tangent chart J (same J as curvature), not a new transductive PCA.

Δ_adapt = MSE_G − MSE_P.
Controls: {", ".join(CONTROLS)}.
Inference: rank-space Freedman–Lane, B_perm={cfg.n_perm_eff()}, B_boot={cfg.n_boot_eff()}.
p=0 is never reported (plus-one Monte Carlo).
Holm correction is within-model over (C_G, C_A, A).

Cross-model aggregate: equal model weight, synchronized anchor resampling.
Label shuffles: {cfg.n_shuffle_eff()} end-to-end G+P refits per model.
Rotation stability gate: fold-cosine ≥ {WEIGHT_COS_RELIABLE} (frozen from LPA audit).
"""
    )


def write_report(out: Path, decision: dict, parity: dict, rel_rows: list, primaries: dict, agg: dict, t0: float, cfg) -> None:
    import time

    lines = [
        "# REPORT",
        "",
        f"## Decision label",
        "",
        f"`{decision['label']}`",
        "",
        f"reason: {decision.get('reason')}",
        "",
        "## Phase 0 ViT-B parity",
        "",
        f"ok={parity.get('ok')} n={parity.get('n_association')} "
        f"ρ(K_H,R²_G)={parity.get('rho_r2_G')} ρ(K_H,MSE_G)={parity.get('rho_mse_G')} "
        f"ρ(K_H,Δ_adapt)={parity.get('rho_dMSE_GP')} mean Δ_adapt={parity.get('mean_delta_adapt')}",
        "",
        f"reused primary CI={parity.get('primary_reused')}",
        "",
        "## Reliability",
        "",
    ]
    for r in rel_rows:
        lines.append(f"- {r['model']}: median R_H={r['R_H_median']:.3f} unreliable={r['geometry_unreliable']}")
    lines += ["", "## Per-model primaries", ""]
    for m, p in primaries.items():
        lines.append(
            f"- {m}: C_G={p['C_G']['observed']:+.3f} C_A={p['C_A']['observed']:+.3f} "
            f"A={p['A']['observed']:+.3f} C_P={p['C_P']['observed']:+.3f} "
            f"meanΔ={p['mean_delta_adapt']:+.3f} patch_worse={p['patch_worse_on_average']}"
        )
    lines += ["", "## Aggregate", "", str(agg.get("observed")), ""]
    lines += [
        "## QLCA (ViT-B reused, not mediating)",
        "",
        f"median Δ_Q≈{QLCA_MEDIAN_DELTA_Q}, ρ(K_H,Δ_Q)≈{QLCA_RHO_KH_DQ}, A_B≈{QLCA_A_B}.",
        f"Conditioning adaptation on Δ_Q raises the association to ≈{QLCA_PARTIAL_ADAPT}.",
        "",
        f"## Runtime",
        "",
        f"smoke={cfg.smoke} wall_s≈{time.time()-t0:.1f}",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
