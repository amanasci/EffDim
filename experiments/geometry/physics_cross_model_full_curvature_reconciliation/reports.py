"""METHODS, REPORT, manuscript recommendation. Does not edit any paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import HISTORICAL_FULL_METRIC, PRIMARY_D, PRIMARY_K, TRACE_DECISION_LABEL
from .io_util import write_json, write_text


def _rho(prim: dict, key: str) -> float:
    return float(prim.get(key, {}).get("observed", float("nan")))


def write_tables(
    out: Path,
    tables: dict[str, pd.DataFrame],
    primaries: dict[str, dict[str, dict]],
    aggs: dict[str, dict],
    reliability: pd.DataFrame,
    pairwise: dict[str, pd.DataFrame],
) -> None:
    rows = []
    for m, df in tables.items():
        full = primaries.get("K_dir_cross", {}).get(m, {})
        kh = primaries.get("K_H_cross", {}).get(m, {})
        rel = reliability[reliability.model == m]
        rows.append(
            {
                "model": m,
                "full_reliability_R_BS": float(rel.full_tensor_median_R.iloc[0]) if len(rel) and "full_tensor_median_R" in rel else float("nan"),
                "trace_reliability_R_H": float(rel.trace_median_R.iloc[0]) if len(rel) and "trace_median_R" in rel else float("nan"),
                "rho_Kdir_R2G": _rho(full, "C_R2"),
                "rho_Kdir_MSEG": _rho(full, "C_G"),
                "rho_Kdir_R2P": _rho(full, "C_R2P"),
                "rho_Kdir_MSEP": _rho(full, "C_P"),
                "rho_Kdir_Dadapt": _rho(full, "C_A"),
                "CG_minus_CP": _rho(full, "A"),
                "mean_Dadapt": float(df.delta_adapt.mean()),
                "rho_KH_R2G": _rho(kh, "C_R2"),
                "rho_KH_MSEG": _rho(kh, "C_G"),
                "rho_KH_R2P": _rho(kh, "C_R2P"),
                "rho_KH_MSEP": _rho(kh, "C_P"),
                "rho_KH_Dadapt": _rho(kh, "C_A"),
                "KH_CG_minus_CP": _rho(kh, "A"),
            }
        )
    pd.DataFrame(rows).to_csv(out / "central_comparison_table.csv", index=False)

    estim = []
    for m, df in tables.items():
        rec = {"model": m}
        for col in ("K_dir_cross", "K_B_cross", "K_H_cross", "K_aniso_cross"):
            if col in df.columns:
                rec[f"median_{col}"] = float(np.nanmedian(df[col]))
                rec[f"frac_pos_{col}"] = float((df[col] > 0).mean())
        if "traceless_fraction" in df.columns:
            rec["median_traceless_fraction"] = float(np.nanmedian(df.traceless_fraction))
        if "K_dir_cross" in df.columns and "K_aniso_cross" in df.columns:
            kd = df.K_dir_cross.to_numpy(float)
            ka = df.K_aniso_cross.to_numpy(float)
            rec["median_aniso_share_of_Kdir"] = float(np.nanmedian(np.where(np.abs(kd) > 1e-12, ka / kd, np.nan)))
        estim.append(rec)
    pd.DataFrame(estim).to_csv(out / "estimand_comparison_table.csv", index=False)

    for m, pw in pairwise.items():
        pw.to_csv(out / "tables" / f"{m}_curvature_estimand_correlations.csv", index=False)
    reliability.to_csv(out / "reliability_table.csv", index=False)


def write_methods(out: Path, audit: dict) -> None:
    write_text(
        out / "METHODS.md",
        f"""# Methods — full sphere-normal curvature reconciliation

## Scope

Audit of a previously observed cross-model curvature–probe association
under the **historical full sphere-normal curvature** scalar, not a new
estimator and not a manuscript rewrite.

## Frozen reuse

Same five encoders, 512 anchors, d={PRIMARY_D}, k={PRIMARY_K},
the same object identities, neighbours, local PCA frames, A/B splits,
five probe folds, targets, valid evaluation objects, and frozen global
and patch predictions. Artifacts are aligned by `sample_id`.
Global and patch probes are **not** refit.

Controls: log kNN radius, local target variance, evaluation count.
Inference: the frozen rank-space Freedman–Lane `associate()` used in
the completed cross-model run. 10,000 permutations and 2,000
joint-anchor bootstraps. Encoders share anchors, so resample IDs jointly.

## Primary curvature

Recovered historical definition (`{HISTORICAL_FULL_METRIC}`):

{audit.get("formula", "")}

This is the direction-averaged normal-curvature split-cross statistic.
`K_H_cross` is reported only as a comparator. Negative split-cross
values are not clamped before rank correlations.

## Other scalars

- `K_B_cross = <B_A^S, B_B^S>_F` (metric-correct tensor Frobenius)
- `K_aniso_cross` (traceless / saddle-like bending)
- energy fractions of the unpacked B^S when tensors are available

## What is not claimed

A positive rho(K, Delta_adapt) is a relative
local-adaptation gain, not absolute patch-probe performance.
The prior label `{TRACE_DECISION_LABEL}` remains the correct label for
the trace estimand K_H^cross.
""",
        force=True,
    )


def write_report(
    out: Path,
    *,
    decision: dict,
    audit: dict,
    parity: dict,
    primaries: dict,
    aggs: dict,
    tests: dict,
    runtime_s: float,
) -> None:
    full = primaries.get("K_dir_cross", {})
    kh = primaries.get("K_H_cross", {})
    fag = aggs.get("K_dir_cross", {})
    lines = [
        "# Full-curvature reconciliation report",
        "",
        f"Decision label: `{decision['label']}`",
        f"Reason: `{decision['reason']}`",
        f"Prior trace label (untouched): `{TRACE_DECISION_LABEL}`",
        f"Primary metric: `{HISTORICAL_FULL_METRIC}`",
        "",
        "## Recovered formula",
        "",
        audit.get("formula", ""),
        "",
        f"Runtime: {runtime_s:.1f} s. Unit tests: {tests.get('n_pass')}/{tests.get('n')} pass={tests.get('ok')}.",
        f"Parity ok: {parity.get('ok')}. Historical K_dir parity: {parity.get('historical_kdir_ok')}.",
        "",
        "## Per-encoder full curvature vs outcomes",
        "",
        "| model | ρ(K_dir, R²_G) | ρ(K_dir, MSE_G) | ρ(K_dir, R²_P) | ρ(K_dir, MSE_P) | ρ(K_dir, Δ_adapt) | C_G−C_P | mean Δ_adapt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, rec in full.items():
        lines.append(
            f"| {m} | {_rho(rec,'C_R2'):.3f} | {_rho(rec,'C_G'):.3f} | {_rho(rec,'C_R2P'):.3f} | "
            f"{_rho(rec,'C_P'):.3f} | {_rho(rec,'C_A'):.3f} | {_rho(rec,'A'):.3f} | "
            f"{float(rec.get('mean_delta_adapt', float('nan'))):.3f} |"
        )
    lines += [
        "",
        "## Cross-model aggregates (joint-anchor bootstrap)",
        "",
        f"- equal-weight mean ρ(K_dir, R²_G) = {fag.get('C_R2_bar', {}).get('observed')}",
        f"- CI95 = {fag.get('C_R2_bar', {}).get('ci95')}",
        f"- equal-weight mean C_G = {fag.get('C_G_bar', {}).get('observed')}",
        f"- equal-weight mean C_A = {fag.get('C_A_bar', {}).get('observed')}",
        f"- Fisher-z ρ(K_dir, R²_G) = {(fag.get('fisher') or {}).get('C_R2_bar')}",
        "",
        "## Trace comparator (should match the completed experiment)",
        "",
    ]
    for m, rec in kh.items():
        lines.append(
            f"- {m}: ρ(K_H, R²_G)={_rho(rec,'C_R2'):.3f}, C_G={_rho(rec,'C_G'):.3f}, "
            f"C_A={_rho(rec,'C_A'):.3f}"
        )
    lines += [
        "",
        "## Fixed questions",
        "",
        "1. Historical full curvature vs global performance: see `C_R2` / `C_G` under K_dir.",
        "2. Full-tensor K_B: see `primaries.K_B_cross` and `aggregates.K_B_cross`.",
        "3. Absolute patch reversal: `C_R2P` / `C_P`, not `C_A`.",
        "4. Relative adaptation: `C_A` and mean Δ_adapt (the latter is the average risk difference).",
        "5. The prior `representation_specific_effect` label is about K_H only.",
        "6. Traceless share: `estimand_comparison_table.csv`.",
        "",
    ]
    write_text(out / "REPORT.md", "\n".join(lines) + "\n", force=True)


def write_manuscript_recommendation(out: Path, decision: dict, aggs: dict) -> None:
    fag = aggs.get("K_dir_cross", {})
    kh = aggs.get("K_H_cross", {})
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        f"""# Manuscript recommendation (no edits applied)

This file is advisory only. It does **not** modify
`submissions/ml4ps_2026/`, `submissions/ml4ps_2026/cross_model/`,
`submissions/neurreps_2026/`, or `papers/curvature_photometric_decoding/`.

## Labels

- Keep `{TRACE_DECISION_LABEL}` as the label of the **completed
  trace-based** cross-model experiment (K_H^cross).
- New audit label for the **full** estimand: `{decision['label']}`.
- Reason: `{decision['reason']}`.

## If the living / ML4PS-cross-model paper is revised later

Do **not** silently retitle the paper as a full-curvature result.
The current title and abstract claim a ViT-B-specific association under
the **trace** statistic. Changes that would be required, if and only if
a later revision adopts this reconciliation:

1. **Title.** State which estimand is meant. A title that says
   "representation curvature" without "mean-curvature trace" or
   "full sphere-normal second fundamental form" is ambiguous after this
   audit.
2. **Abstract.** Any sentence that treats the five-encoder table as
   evidence about "curvature" in general must name K_H^cross
   or K_dir^cross. The two scalars are
   not interchangeable. Current aggregate under the trace:
   mean C_G ≈ {(kh.get('C_G_bar') or {}).get('observed')}.
   Full-curvature aggregate:
   mean rho(K_dir, R_G^2) ≈ {(fag.get('C_R2_bar') or {}).get('observed')}.
3. **Methods.** Replace or supplement the sentence that defines curvature
   by the trace H^S = d^{{-1}} tr B^S with the recovered
   historical full construction (remove Q^T and forced Q^R, then
   contract the entire remaining B^S).
4. **Table 1 / cross-model table.** Add columns for K_dir
   (and optionally K_B, traceless energy) beside the existing K_H
   columns. Do not relabel the existing K_H numbers as full curvature.
5. **Figures.** The current forest plot is a trace plot. A revision would
   need a paired full-vs-trace forest (this audit's fig. 1) and must keep
   absolute patch R_P^2 visually distinct from Delta_adapt.
6. **Discussion of `representation_specific_effect`.** That conclusion
   stays attached to K_H. If the new label is not the same claim,
   say so in one sentence rather than rewriting the old result.

Do not change the paper until this audit’s `COMPLETE.json` exists and
the new label is accepted.
""",
        force=True,
    )


def write_reuse_manifest(out: Path, audit: dict, parity: dict) -> None:
    write_json(
        out / "reuse_manifest.json",
        {
            "read_only_sources": [
                "outputs/geometry/physics_cross_model_curvature_local_adaptation",
                "outputs/geometry/physics_nested_dimension_curvature",
                "outputs/geometry/physics_effdim_curvature_metrics",
                "outputs/geometry/physics_multimodel_graph_prior_quadratic",
            ],
            "probes_refit": False,
            "alignment": "sample_id",
            "historical_metric": HISTORICAL_FULL_METRIC,
            "audit_ok": audit.get("ok"),
            "parity_ok": parity.get("ok"),
        },
        force=True,
    )
