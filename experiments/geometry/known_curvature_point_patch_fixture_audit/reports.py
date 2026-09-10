"""METHODS, REPORT, MANUSCRIPT_RECOMMENDATION. Does not edit any manuscript."""

from __future__ import annotations

from pathlib import Path

from .config import (
    AE_HIDDEN,
    AMBIENT_ROTATION_HASH16,
    D_AMB,
    D_LAT,
    DECODER_SEEDS,
    F1_C,
    F3_R2,
    F3_S2,
    F4_WIDTHS,
    F5_WIDTHS,
    MAX_EPOCHS,
    PRIMARY_K,
    PRIMARY_N,
)


def write_reports(out: Path, *, summary: dict, decision: dict, fixtures: dict, tests: dict) -> None:
    suite_scope = (
        "Bounded fixture check: colleague-scale n=86471 and Suite F "
        "(fixed radius / adaptive k / inverse-density weights) were not run."
        if summary.get("bounded")
        else "Full plan also includes n=86471 and Suite F."
    )
    methods = rf"""# METHODS — known curvature point/patch fixture audit

## Estimands and scales

Two instruments are scored against *matched* geometric targets.

- **Estimator D** (learned decoder) estimates the *pointwise* sphere-normal second
  fundamental form of a globally fitted surface
  \(\widetilde F(z)=F(z)/\|F(z)\|\). Its proper target is **T1**.
- **Estimator Q** (frozen local quadratic) estimates a *finite-patch* quadratic
  representation of the observed cloud. Its proper target at neighbourhood
  radius \(r\) is **T2** (uniform volume) or **T3** (sampling-weighted). T1 is
  only the *asymptotic* target of Q as \(r\to 0\) with adequate density.

They are compared directly only after aligning contraction (mean vs full
\(B^S\)) *and* spatial scale.

## Common geometry

- Intrinsic dimension \(d={D_LAT}\), ambient \(D={D_AMB}\), unit-normalized.
- Frozen ambient rotation, seed 20260907, SHA256[:16] `{AMBIENT_ROTATION_HASH16}`.
- Float64 for all truth calculations and differential geometry.
- Hash-stable anchors by `sha256(seed:sample_id)`.
- Identical observed points and anchors for D and Q within every condition.

## Sphere-normal tensors

\[
J=\partial G,\quad g=J^\top J,\quad P_T=Jg^{{-1}}J^\top,\quad
P_{{N,S}}=I-GG^\top-P_T,\quad
B^S_{{ab}}=P_{{N,S}}\partial_{{ab}}G.
\]

\[
H^S=\frac1d g^{{ab}}B^S_{{ab}},\qquad
K_{{\mathrm{{dir}}}}=\frac{{2\|B^S\|_F^2+\|\mathrm{{tr}}B^S\|^2}}{{d(d+2)}}
\]

after metric whitening. Split \(B^S=gH^S+\mathring B^S\).

**Packing.** T1 \(B^S\) is the Hessian. Frozen Q stores \(\Phi=u_a u_b\)
coefficients \(S\); \(\mathrm{{Hess}}=2\,\mathrm{{unpack}}(S)\). Production
unpacked scalars equal one-quarter of the Hessian \(K_{{\mathrm{{dir}}}}\)
cross statistic. Tables labelled Hessian use the user formula; tables labelled
unpacked retain the production scalar for continuity with real-data papers.
Negative split-cross values are not clamped.

\(C_\rho=\mathrm{{sign}}(K_{{\mathrm{{dir}}}}^{{\mathrm{{cross}}}})\,
\rho\,\sqrt{{|K_{{\mathrm{{dir}}}}^{{\mathrm{{cross}}}}|}}\)
on Hessian tensors.

## Fixtures

- F0 great \(S^{{16}}\subset S^{{767}}\): \(B^S=0\).
- F1 latitude \(c={F1_C}\): pure mean curvature \(\kappa=c/r\).
- F2 minimal Clifford \(r^2=s^2=1/2\): \(H^S=0\), \(\mathring B^S\neq 0\).
- F3 nonminimal Clifford \(r^2={F3_R2}\), \(s^2={F3_S2}\).
- F4 low-frequency bumped sphere, widths {list(F4_WIDTHS)}.
- F5 high-frequency bumped sphere, widths {list(F5_WIDTHS)}.
  Bump parameters were frozen before estimator scoring.

Independent truth: analytic (F0–F3), torch autodiff, and central finite
differences, plus orthogonality, radial identity, latent- and ambient-rotation
invariance. A fixture is used only if these agree within frozen tolerances.

## Estimators

**D.** `cae.PlainAutoEncoder` \(768\to 16\to 768\), hidden {AE_HIDDEN}, SiLU,
{MAX_EPOCHS} epochs, `TRAIN_CFG` from the colleague protocol
(lr \(10^{{-3}}\), weight decay \(10^{{-4}}\), batch 128). Train/holdout split
seed 20260813, holdout fraction 0.2. Anchors are taken from the holdout only.
Seeds {DECODER_SEEDS} on primary conditions. Full \(B^S\) is retained (not
only the trace).

**Q.** Exact frozen path: `nested_pca_frame` + `fit_quad` (ridge grid
\([10^{{-4}},\ldots,3]\), A/B splits, sphere-radial removal, frozen packing).

## Finite-patch oracles

T2: population-optimal quadratic under uniform manifold-volume weights
\(\sqrt{{\det g}}\) over the ambient ball of radius \(r\), Sobol latent
candidates. T3: the same fit with the known fixture sampling density.
Inverse-density-weighted Q is a secondary path against T2.

## Suites

A clean geometry; B sampling; C noise; D density–noise; E scale sweep
on \(n={PRIMARY_N}\) only. {suite_scope}

Primary: \(n={PRIMARY_N}\), \(k={PRIMARY_K}\), 512 holdout anchors.
"""

    label = decision.get("label", "known_curvature_fixture_audit_unresolved")
    reason = decision.get("reason", "")
    report = rf"""# REPORT — known curvature point/patch fixture audit

## Decision

**{label}**

{reason}

Decoder pointwise mean-curvature accuracy: {summary.get("decoder_pointwise_H")}
Decoder pointwise full \(B^S\) accuracy: {summary.get("decoder_pointwise_Kdir")}
Decoder cross-seed stability: {summary.get("decoder_seed_stability")}
Quadratic pointwise convergence: {summary.get("quadratic_shrink")}
Quadratic matched finite-patch accuracy: {summary.get("quadratic_patch")}
False curvature on residual-flat F0: {summary.get("false_F0")}
Tests passed: {tests.get("n_passed")}/{tests.get("n_tests")}
Runtime s: {summary.get("runtime_s")}
Peak RSS MB: {summary.get("peak_rss_mb")}
Bounded: {summary.get("bounded")} skipped={summary.get("skipped")}

## What the two instruments may be called

See `MANUSCRIPT_RECOMMENDATION.md`. This audit does not pick a winner against
an unmatched target. Pointwise estimators were scored against pointwise truth;
finite-patch estimators against matched finite-patch truth.

## Fixture definitions

See `fixture_definitions.json`. Ambient rotation hash `{AMBIENT_ROTATION_HASH16}`.

## Outputs

All artifacts under this tree. No manuscript and no prior experiment tree was modified.
"""

    reco = rf"""# MANUSCRIPT RECOMMENDATION

Do not edit existing manuscripts from this audit. The following language is
what each estimator may *legitimately be called* in future drafts, and which
real-data claims require revision pending this known-answer result.

## Decision label

`{label}`

{reason}

## Estimator D (differentiated decoder)

May be called a **pointwise decoder curvature instrument** only to the extent
that it recovers T1 (analytic \(H^S\) and full \(B^S\)) with stable fields
across seeds. High reconstruction \(R^2\) alone is not a licence to treat it
as a curvature meter on real activations.

It must **not** be described as measuring the same finite-patch quadratic
functional as estimator Q, nor as refuting a local-chart association merely
because the two fields anticorrelate on real data.

## Estimator Q (local quadratic)

May be called a **finite-bandwidth residual curvature instrument**. Its
matched target is T2/T3 at the neighbourhood radius actually used (\(k=2048\)
or the corresponding physical radius), not infinitesimal mean curvature.

It may be described as converging to pointwise curvature only when the
radius-shrinkage diagnostic against T1 succeeds. Split-half reliability is
not evidence of truth by itself.

## Real-data claims that require revision if this audit so indicates

1. Treating colleague AE \(H_{{\mathrm{{tan}}}}\) (\(\rho\\approx +0.33\) on
   86,471 rows) and local-chart \(K_H^{{\mathrm{{cross}}}}\)
   (\(\rho_{{\mathrm{{ctl}}}}\\approx -0.240\) on ViT-B) as estimates of one
   shared curvature–decodability law.
2. Interpreting their anticorrelation as a contradiction rather than as a
   mismatch of estimand (mean vs full \(B^S\)) or scale (point vs patch),
   until those mismatches are ruled out by this fixture study.
3. Claiming either instrument is “the” residual curvature of activation space
   without naming T1 vs T2/T3.

## What may stand

- Local-chart \(K_H\) / \(K_{{\mathrm{{dir}}}}\) as finite-patch residual
  curvature under the frozen quadratic protocol, if Q recovers T2/T3 here.
- Decoder autodiff \(H^S\) as a pointwise property of a learned surface, if D
  recovers T1 here and is seed-stable.
- The statement that the two real-data signs can coexist because they are
  different functionals, if the distinct-scale label is assigned.

## What must not be claimed

A winner-takes-all comparison of D against Q on unmatched targets.
"""

    (out / "METHODS.md").write_text(methods)
    (out / "REPORT.md").write_text(report)
    (out / "MANUSCRIPT_RECOMMENDATION.md").write_text(reco)
