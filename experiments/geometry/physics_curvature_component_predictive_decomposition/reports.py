"""Narrative outputs. Does not edit any manuscript."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PRIMARY_D, PRIMARY_K
from .io_util import write_text


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def write_methods(out: Path, cfg_d: dict, span: dict) -> None:
    write_text(
        out / "METHODS.md",
        f"""# Methods — curvature-component predictive decomposition

This analysis is an **exploratory decomposition** motivated by completed
full-curvature and QLCA results. It is **not** prospectively preregistered.

## Frozen reuse

Five encoders (ViT-B, DINOv3, CLIP, ConvNeXt-B, ViT-L), 512 hash-stable
anchors, $k={PRIMARY_K}$, $d={PRIMARY_D}$. Same object identities, neighbours,
A/B geometry splits, global/patch predictions, probe folds, evaluation
objects, physical target (`mag_r_desi`), and controls
(`log_knn_radius`, `local_label_variance`, `local_evaluation_count`).
ViT-B QLCA reuses the same charts, $Q$, $B^S$, folds, held-out L/UQ/BS,
and foldwise Hessians $\\Gamma$. Alignment is by `sample_id`.

Prior trees are read-only. New writes go only to
`experiments/geometry/physics_curvature_component_predictive_decomposition/`
and `outputs/geometry/physics_curvature_component_predictive_decomposition/`.

## Geometry

The sphere-normal residual of the fitted quadratic chart is $B^S=Q-Q_T-Q_R$.
In orthonormal PCA chart coordinates,

$$
B^S = B^H + \\mathring B,\\qquad B^H_{{ab}}=\\delta_{{ab}}H,\\qquad H=\\tfrac1d\\mathrm{{tr}}B^S.
$$

Split-cross statistics (signed, unclamped):

$$
K_H^{{\\mathrm{{cross}}}}=\\langle H_A,H_B\\rangle,\\qquad
K_{{\\mathrm{{TF}}}}^{{\\mathrm{{cross}}}}=\\frac{{2}}{{d(d+2)}}\\langle\\mathring B_A,\\mathring B_B\\rangle_F,
$$

and $K_{{\\mathrm{{dir}}}}^{{\\mathrm{{cross}}}}=K_H^{{\\mathrm{{cross}}}}+K_{{\\mathrm{{TF}}}}^{{\\mathrm{{cross}}}}$.
`K_aniso_cross` from the completed reconciliation **is** $K_{{\\mathrm{{TF}}}}^{{\\mathrm{{cross}}}}$.

Within-split descriptive energies $E_H=\\|H\\|^2$,
$E_{{\\mathrm{{TF}}}}=\\frac{{2}}{{d(d+2)}}\\|\\mathring B\\|_F^2$,
$F_H=E_H/E_{{\\mathrm{{dir}}}}$, and
$C_{{\\mathrm{{trace}}}}=\\|\\mathrm{{tr}}B^S\\|^2/(d\\|B^S\\|_F^2)$
are **not** substituted for split-cross association statistics.

## Inference

Controlled Spearman uses the frozen rank-space Freedman–Lane `associate()`
from `physics_curvature_probe_rank_sweep`. Component-conditional tests add
the other component to the control matrix. Do **not** condition on $K_{{\\mathrm{{dir}}}}$.

Primary cross-model outcome: global OOF MSE. $R_G^2$ is the sign-reversed
parity endpoint. Patch MSE and $\\Delta_{{\\mathrm{{adapt}}}}$ are secondary and
kept distinct from quadratic label gain $\\Delta_Q$.

Encoders share anchors: joint-anchor bootstrap (2,000) and Monte Carlo
permutations (10,000). Equal-weight and Fisher-$z$ means. Holm correction
within the two-component primary family (unique $K_H$ and unique $K_{{\\mathrm{{TF}}}}$
vs global MSE). This is a diagnostic family, not a confirmatory analysis.

## ViT-B Hessian and probes

$\\Gamma=\\Gamma_H+\\mathring\\Gamma$ in the same orthonormal chart.
A component is not interpreted if foldwise cosine $<0.5$.

Alignments $A_B$, $A_H$, $A_{{\\mathrm{{TF}}}}$ use the QLCA Frobenius-preserving
136-vector convention. Nulls: 2,000 Haar / random-$\\gamma$ draws per
anchor, A/B geometry splits, and the induced-energy cross term
$2\\gamma^\\top B_H^{{\\mathrm{{flat}}\\top}}B_{{\\mathrm{{TF}}}}^{{\\mathrm{{flat}}}}\\gamma$.

Held-out models on frozen coordinates and folds:

- L: tangent linear
- IQ: isotropic quadratic (1-D trace of $\\Gamma$)
- TQ: traceless quadratic (135-D)
- UQ-v2: full quadratic **with** $\\alpha_Q=\\infty$ omit-quadratic candidate
- BSH / BSTF / BS: chart-constrained, cap-48 SVD (90/95/99% energy ranks reported)

IQ+TQ span check: $n_{{\\mathrm{{TF}}}}={span.get("n_tf")}$, span error
${span.get("span_err")}$. Frozen QLCA UQ point estimates are retained for
parity and are not replaced by UQ-v2.

## Decision

Exactly one label is assigned by the frozen rule in `decision.py`
(`rules_version=1`). Labels are not retuned after seeing results.
""",
        force=True,
    )


def write_artifact_audit(out: Path, manifest: dict, parity: dict) -> None:
    lines = ["# Artifact audit", "", "Coordinate convention: orthonormal PCA chart; Euclidean Frobenius is metric-correct.", ""]
    for k, rec in (manifest.get("files") or {}).items():
        lines.append(
            f"- `{k}`: exists={rec.get('exists')} sha16={rec.get('sha16')} path=`{rec.get('path')}`"
        )
    lines += [
        "",
        f"Parity ok: {parity.get('ok')}",
        f"FCR aggregate: {parity.get('fcr', {}).get('aggregate')}",
        f"QLCA: { {k: parity.get('qlca', {}).get(k) for k in ('median_delta_Q','rho_KH_delta_Q','A_B_median','all_stable')} }",
        "",
        "All artifacts aligned by `sample_id`.",
    ]
    write_text(out / "ARTIFACT_AUDIT.md", "\n".join(lines) + "\n", force=True)


def write_report(
    out: Path,
    *,
    decision: dict,
    parity: dict,
    tests: dict,
    org_rows: list[dict],
    joint: dict,
    per_model: dict,
    vitb_sum: dict,
    runtime_s: float,
) -> None:
    jkh = joint.get("unique_KH_bar", {})
    jtf = joint.get("unique_KTF_bar", {})
    write_text(
        out / "REPORT.md",
        f"""# Report — curvature-component predictive decomposition

Exploratory (not prospectively preregistered). Decision **`{decision.get('label')}`**
({decision.get('reason')}). Runtime {runtime_s/60:.1f} min. Tests {tests.get('n_pass')}/{tests.get('n')}
parity={parity.get('ok')}.

## Which component carries decodability?

$B^S$ is the sphere-normal curvature tensor. $K_H$ and $K_{{\\mathrm{{TF}}}}$ quantify
distinct organizations of that tensor. $K_H$ is not the complete curvature;
$K_{{\\mathrm{{dir}}}}$ is not the unique mathematically correct scalar.

### Unique associations with global OOF error (primary)

Equal-weight mean unique $\\rho_{{\\mathrm{{ctl}}}}(K_H,\\mathrm{{MSE}}_G\\mid K_{{\\mathrm{{TF}}}})$
= {_f(jkh.get('observed'))} CI {jkh.get('ci95')} $p_{{\\mathrm{{Holm}}}}$={_f(jkh.get('p_holm'), 4)}.

Equal-weight mean unique $\\rho_{{\\mathrm{{ctl}}}}(K_{{\\mathrm{{TF}}}},\\mathrm{{MSE}}_G\\mid K_H)$
= {_f(jtf.get('observed'))} CI {jtf.get('ci95')} $p_{{\\mathrm{{Holm}}}}$={_f(jtf.get('p_holm'), 4)}.

Sign counts: $K_H$ positive in {joint.get('sign', {}).get('n_pos_KH')} / {len(per_model)} encoders;
$K_{{\\mathrm{{TF}}}}$ positive in {joint.get('sign', {}).get('n_pos_KTF')} / {len(per_model)}.

These are **absolute probe-error** associations, not local-adaptation gains
and not quadratic label gains.

### Organization

High total curvature is typically greater traceless bending: see
`organization_table.csv`. Cross-estimated $K_{{\\mathrm{{TF}}}}$ share of $K_{{\\mathrm{{dir}}}}$
remains ~90%+, consistent with the completed reconciliation.

### ViT-B quadratic gain and Hessian

{vitb_sum.get('narrative', '')}

Do not infer causality or mediation.

## Decision flags

```
{decision.get('flags')}
```
""",
        force=True,
    )


def write_manuscript_recommendation(out: Path, decision: dict, joint: dict, vitb_sum: dict) -> None:
    lab = decision.get("label")
    if lab == "mean_bending_specific_decodability_link":
        emp = "coherent mean bending"
    elif lab == "traceless_bending_specific_decodability_link":
        emp = "traceless saddle-like bending"
    elif lab == "distinct_mean_and_traceless_predictive_roles":
        emp = "distinct component roles"
    elif lab == "representation_specific_curvature_component_effects":
        emp = "representation-specific heterogeneity"
    elif lab == "total_bending_magnitude_link":
        emp = "total curvature magnitude (components not separable)"
    else:
        emp = "no stronger claim than the completed audits"
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        f"""# Manuscript recommendation

Do **not** edit the paper from this tree. Eventual text should emphasize:

**{emp}**

Mechanical label: `{lab}` ({decision.get('reason')}).

## What to say

- $B^S$ is the geometrically meaningful sphere-normal curvature tensor.
- $K_H$ and $K_{{\\mathrm{{TF}}}}$ are complementary organizations of $B^S$, not competing “true” scalars.
- Unique global-error associations (joint-anchor):
  $K_H\\mid K_{{\\mathrm{{TF}}}}$ = {joint.get('unique_KH_bar', {}).get('observed')};
  $K_{{\\mathrm{{TF}}}}\\mid K_H$ = {joint.get('unique_KTF_bar', {}).get('observed')}.
- Keep global error, relative $\\Delta_{{\\mathrm{{adapt}}}}$, and quadratic $\\Delta_Q$ as distinct claims.
- {vitb_sum.get('manuscript_line', '')}

## What not to say

- Do not call this prospectively preregistered.
- Do not overwrite the completed trace label `representation_specific_effect`
  or the full-curvature label `full_curvature_partial_cross_model_replication`.
- Do not describe $K_H$ as complete curvature or $K_{{\\mathrm{{dir}}}}$ as uniquely correct.
- Do not infer causality or mediation.
""",
        force=True,
    )


def summarize_vitb(vitb: pd.DataFrame | None, qlca: pd.DataFrame | None, hess: dict, align: dict, probes: dict) -> dict:
    if vitb is None or vitb.empty:
        return {"narrative": "ViT-B component pass unavailable.", "manuscript_line": ""}
    lines = []
    if "frac_Gamma_H" in vitb:
        lines.append(
            f"Median $\\|\\Gamma_H\\|_F^2/\\|\\Gamma\\|_F^2$={float(vitb.frac_Gamma_H.median()):.3f}; "
            f"traceless={float(vitb.frac_Gamma_TF.median()):.3f}."
        )
    if "delta_UQ2" in vitb:
        lines.append(
            f"Median $\\Delta_{{IQ}}$={float(vitb.delta_IQ.median()):.4f}, "
            f"$\\Delta_{{TQ}}$={float(vitb.delta_TQ.median()):.4f}, "
            f"$\\Delta_{{UQ2}}$={float(vitb.delta_UQ2.median()):.4f}."
        )
    lines.append(f"Alignment driver: {align.get('driver')}. BSTF explains BS: {probes.get('bstf_explains_bs')}.")
    return {
        "narrative": " ".join(lines),
        "manuscript_line": " ".join(lines),
        "median_frac_Gamma_H": float(vitb.frac_Gamma_H.median()) if "frac_Gamma_H" in vitb else None,
        "median_frac_Gamma_TF": float(vitb.frac_Gamma_TF.median()) if "frac_Gamma_TF" in vitb else None,
        "median_A_B": float(vitb.A_B.median()) if "A_B" in vitb else None,
        "median_A_H": float(vitb.A_H.median()) if "A_H" in vitb else None,
        "median_A_TF": float(vitb.A_TF.median()) if "A_TF" in vitb else None,
    }
