"""METHODS.md and REPORT.md for quadratic predictive dimension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read(out: Path, name: str, default=None):
    p = out / name
    if not p.exists():
        return default
    if name.endswith(".json"):
        return json.loads(p.read_text())
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _fmt(x, nd=3):
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def write_methods(out: Path, cfg: Any, ctx: dict, parity: dict, thr: dict) -> None:
    text = f"""# Methods: quadratic predictive dimension

## Estimand

This experiment asks for the smallest local coordinate dimension at which a
**regularized quadratic surface** reconstructs held-out ViT-B neighbours
adequately. Two questions are kept separate:

1. **Plateau dimension.** Where does adding coordinates stop improving held-out
   quadratic reconstruction by more than a frozen practical tolerance?
2. **Absolute adequacy.** At what candidate $d$, if any, does held-out
   reconstruction reach 90%, 95%, or 99% explained energy, using the *lower*
   bootstrap confidence bound? If a threshold is not reached by $d=20$, the
   answer is `not_reached`. No extrapolation.

A plateau does **not** imply adequacy. An error curve can flatten at $d=12$
while explaining only 80% of held-out energy; that would mean the quadratic
model class is inadequate, not that the cloud is 12-dimensional.

## Correction of a prior interpretation

The order-stratified experiment reported a regularized held-out quadratic
$R^2\\approx 0.15$ for predicting $E_4$ (ranks 13–16) from 12 tangent
coordinates. That is **approximately 15% of $E_4$ variance explained and
approximately 85% unexplained**. It did **not** explain 85%.

`mag_r_desi`, $K_H$, $d_G$, and other probes are never used to choose $d$,
ridge, thresholds, or labels.

## Frozen inputs (read-only)

Completed trees are not written:

- `physics_stable_tangent_dimension`
- `physics_nested_dimension_curvature`
- `physics_order_stratified_geometry`
- `physics_implicit_normal_inverse`

Parity records cache identity, freeze hash `{parity.get("freeze_hash")}`,
512 primary anchors, spherical-log neighbourhoods, frozen $S_{{20}}$,
certified $T_{{12}}^{{\\mathrm{{core}}}}$, and the decomposition
$T_{{12}}$ (ranks 1–12), $E_4$ (13–16), $U_4$ (17–20), $U_8=E_4\\oplus U_4$.
Scientific fitting aborts if cache identity, anchors, preprocessing, or freeze
hash fails.

## Model

Neighbour coordinates are $z=\\log_x(y)$ after removing a numerical radial
component. For each outer-training fold, uncentred SVD of training neighbours
only supplies nested prefixes $J_d$. Initial coordinates $u_0=J_d^\\top z$.

The symmetric degree-two map
$\\phi_2(u)=(u_a^2,\\sqrt{{2}}u_a u_b)$ has $d(d+1)/2$ features. The immersion
is $f_d(u)=J_d u + B_d\\phi_2(u)$ with **no intercept**, so $f_d(0)=0$ and
$Df_d(0)=J_d$. Quadratic columns are RMS-scaled on training data only (no
mean-centering).

Primary model: unrestricted $B_d$ predicting the full ambient residual.
Sensitivity: $B_d^N=(I-J_d J_d^\\top)B_d$. Linear baseline: $f_d^L(u)=J_d u$.
Training quadratic error is never compared to linear test error.

## Nested CV and closest-point evaluation

Outer radial-stratified halves estimate held-out risk. Inner splits of the
outer-training set select ridge independently for each $d$, variant, scale,
and outer fold. The ridge grid is log-spaced relative to the quadratic Gram
spectrum. The primary inner score is **closest-point** NMSE. A one-SE rule
prefers more regularization.

Closest-point solves $\\hat u=\\arg\\min_u |z-f_d(u)|^2$ from $u_0$, constrained
to the outer-training coordinate-norm quantile. Gauss–Newton is monotone:
it cannot worsen the fixed-coordinate objective beyond numerical tolerance.
Fixed-coordinate $u=u_0$ is retained as a diagnostic.

Primary NMSE is
$\\mathrm{{NMSE}}_d=\\sum_{{\\mathrm{{test}}}}|z-f(\\hat u)|^2/\\sum |z|^2$,
$R^2_d=1-\\mathrm{{NMSE}}_d$. Pooled energy-weighted values and
median/IQR across anchors are both reported. Bootstrap resamples **anchors**.

Tail $R_C^2$ uses frozen external frames on $T_{{12}}$, $E_4$, $U_4$, $U_8$,
and the complement of $S_{{20}}$. Negative $R^2$ is kept visible.

## Plateau, adequacy, synthetics

Plateau $d_L^{{\\mathrm{{plat}}}}$ and $d_Q^{{\\mathrm{{plat}}}}$ use paired
anchor bootstrap differences and a practical tolerance frozen on synthetic
**calibration** seeds. Adequacy ranks $d_{{90}},d_{{95}},d_{{99}}$ use the
lower bootstrap bound of total $R^2$. Labels are gate-derived from
`classify.primary_label`.

Synthetic families: flat $d\\in\\{{8,12,16,20\\}}$, quadratic $d\\in\\{{8,12,16\\}}$,
$d=12$ with $q\\in\\{{1,4,8\\}}$, cubic, isotropic, tangential/normal thickness,
and an anisotropic tail. Evaluation seeds are untouched.

Primary empirical object: ViT-B, 512 frozen anchors, $k=2048$, $d=4,\\ldots,20$.
Scale sensitivity uses the frozen $k$ grid; non-primary scales use a
hash-selected subset of at least 128 anchors disclosed in `config.json`
before fitting.

Thresholds: `{json.dumps(thr, default=str)}`

Parity excerpt: `{json.dumps(parity, default=str)[:2200]}`
"""
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict, parity: dict, labels: dict, thr: dict | None = None) -> None:
    curves = _read(out, "aggregate_risk_curves.csv", pd.DataFrame())
    plat = _read(out, "plateau_bootstrap.csv", pd.DataFrame())
    tail = _read(out, "tail_adequacy.csv", pd.DataFrame())
    seval = _read(out, "synthetic_evaluation.csv", pd.DataFrame())
    scale = _read(out, "scale_sensitivity.csv", pd.DataFrame())
    runtime = _read(out, "runtime_profile.json", {}) or {}
    lab = labels.get("primary", "quadratic_predictive_dimension_unresolved")
    dQ = labels.get("dQ_plat")
    dL = labels.get("dL_plat")
    band = labels.get("dQ_plateau_set") or labels.get("dQ_iqr")
    r2e4 = labels.get("r2_E4_d12")
    r2u8 = labels.get("r2_U8_d12")
    r2tot = labels.get("r2_total_best")
    r2_12 = labels.get("r2_total_d12")
    recovered = labels.get("E4_recovered_of_prior_unexplained")
    unexplained_now = labels.get("E4_unexplained_now")
    high_total_low_tail = (
        np.isfinite(float(r2tot)) if r2tot is not None and not isinstance(r2tot, str) else False
    ) and (
        (isinstance(r2e4, (int, float)) and r2e4 < 0.30)
        or (isinstance(r2u8, (int, float)) and r2u8 < 0.30)
    )
    warning = ""
    if high_total_low_tail and isinstance(r2tot, (int, float)) and r2tot >= 0.95:
        warning = (
            f"\n**High-total / low-tail warning.** Total $R^2={_fmt(r2tot)}$ "
            f"but $R^2_{{E_4}}={_fmt(r2e4)}$ and $R^2_{{U_8}}={_fmt(r2u8)}$ at $d=12$. "
            "Do not describe this model as 95% adequate.\n"
        )
    synth_note = "NA"
    if seval is not None and len(seval) and "kind" in seval.columns:
        if labels.get("closest_synthetic"):
            synth_note = str(labels.get("closest_synthetic"))
        else:
            synth_note = ", ".join(sorted(seval.kind.unique())[:8])
    scale_txt = "single scale only"
    if scale is not None and len(scale) and "k" in scale.columns:
        meds = scale.groupby("k").dQ.median()
        scale_txt = ", ".join(f"k={int(k)}: dQ={float(v):.1f}" for k, v in meds.items())
        if not labels.get("scale_stable", True):
            scale_txt += " (not stable across scale; a rank at one radius is not an identified predictive dimension)"

    text = f"""# Report: quadratic predictive dimension (ViT-B only)

Primary label: **`{lab}`**

This is a *predictive* quadratic reconstruction study. It does not, by itself,
identify exact tangent or intrinsic dimension.

## Correction

The earlier order-stratified experiment explained approximately **15%** of
$E_4$ variance ($R^2\\approx 0.15$) and left approximately **85% unexplained**.
It did not explain 85%.
{warning}
## Answers

### Where do linear and quadratic held-out errors plateau?

Quadratic plateau $d_Q^{{\\mathrm{{plat}}}}={_fmt(dQ, 1)}$ with interval/set
`{band}`. Linear plateau $d_L^{{\\mathrm{{plat}}}}={_fmt(dL, 1)}$.
Per-anchor IQR of quadratic plateaus is `{labels.get("dQ_iqr")}`.
A plateau is the smallest consecutive rank whose risk is within the frozen
tolerance of the best candidate and whose further gains stay below the frozen
practical threshold, provided ridge shrinkage / effective df do not collapse.

### Does quadratic fitting replace ranks 13–16 with curvature?

Held-out NMSE drop from $d=12$ to $d=16$: quadratic
$\\Delta={_fmt(labels.get("delta_Q_12_16"))}$, linear
$\\Delta={_fmt(labels.get("delta_L_12_16"))}$.
If those drops are similar, extra linear coordinates remain useful and
quadratic terms do not replace ranks 13–16. If the quadratic drop is larger
and reproducible, curvature is using those extra coordinates. If both flatten
near 12, ranks 13–16 are not predictively necessary *inside this model class*.

### What are $d_{{90}}$, $d_{{95}}$, $d_{{99}}$?

- $d_{{90}}$ = `{labels.get("d90")}`
- $d_{{95}}$ = `{labels.get("d95")}`
- $d_{{99}}$ = `{labels.get("d99")}`

These are empirical reconstruction thresholds from the lower bootstrap bound
of total held-out $R^2$. They are **not** noise-floor estimates. `not_reached`
means the bound never crossed the line by $d=20$.

### Is a high total $R^2$ hiding poor $E_4$/$U_8$ reconstruction?

Best total $R^2={_fmt(r2tot)}$ (at $d=12$: {_fmt(r2_12)}; at $d=16$: {_fmt(labels.get("r2_total_d16"))}).
At $d=12$: $R^2_{{T_{{12}}}}={_fmt(labels.get("r2_T12_d12"))}$,
$R^2_{{E_4}}={_fmt(r2e4)}$, $R^2_{{U_4}}={_fmt(labels.get("r2_U4_d12"))}$,
$R^2_{{U_8}}={_fmt(r2u8)}$.
At $d=16$: $R^2_{{E_4}}={_fmt(labels.get("r2_E4_d16"))}$,
$R^2_{{U_8}}={_fmt(labels.get("r2_U8_d16"))}$.
Negative tail $R^2$ is a real failure to beat a zero predictor on that
subspace, not a display artifact.

### How much of the previously unexplained 85% of $E_4$ variance is now recovered?

Prior explained fraction $\\approx 0.15$. Current $d=12$ closest-point
$R^2_{{E_4}}={_fmt(r2e4)}$ (fixed-coordinate {_fmt(labels.get("r2_E4_fixed_d12"))};
normal-only {_fmt(labels.get("r2_E4_normal_d12"))}).
Unexplained now: {_fmt(unexplained_now)}. Fraction of the *prior unexplained
85%* that is newly recovered: {_fmt(recovered)}.
A value near 0 means nested ridge / closest-point / unrestricted $B$ did not
materially change the earlier 15% result.

### Does the result recur across neighbourhood scales?

{scale_txt}

A rank that appears at only one radius is not an identified predictive
dimension.

### Which synthetic families does the empirical curve resemble?

Closest listed family: `{synth_note}`.
Synthetic evaluation `not_only12={labels.get("synth_not_only12")}`.
The procedure is not accepted if it mechanically returns 12 on every family.

### What can and cannot be concluded about exact tangent or intrinsic dimension?

**Can:** report a predictive plateau band, absolute reconstruction thresholds,
and whether the quadratic class is adequate on total energy and on the frozen
tail. Compare linear vs quadratic incremental gains on ranks 13–16.

**Cannot:** infer exact intrinsic or tangent dimension from a predictive
plateau alone. Inadequacy of $f_d$ can come from cubic curvature, thickness,
stratification, or branch mixing. Adequacy of a 12-coordinate quadratic model
would still be a statement about *this model class*, not a unique geometric
dimension.

## Gate label

`{lab}` is produced by `classify.primary_label` from plateau location, linear
vs quadratic gains, $d_{{95}}$, total $R^2$, $E_4$/$U_8$ tail $R^2$, synthetic
non-degeneracy, and scale stability. It is not hand-edited.

## Runtime and parity

Runtime seconds: `{runtime.get("total_seconds")}`. Stages: `{runtime.get("completed")}`.
Parity ok: `{parity.get("ok")}`. Freeze hash: `{parity.get("freeze_hash")}`.
Output directory: `{out}`.
"""
    (out / "REPORT.md").write_text(text)
