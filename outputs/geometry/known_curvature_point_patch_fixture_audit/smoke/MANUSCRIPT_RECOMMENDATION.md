# MANUSCRIPT RECOMMENDATION

Do not edit existing manuscripts from this audit. The following language is
what each estimator may *legitimately be called* in future drafts, and which
real-data claims require revision pending this known-answer result.

## Decision label

`mean_vs_full_curvature_divergence`

mean curvature discards traceless bending that full K_dir recovers

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

1. Treating colleague AE \(H_{\mathrm{tan}}\) (\(\rho\\approx +0.33\) on
   86,471 rows) and local-chart \(K_H^{\mathrm{cross}}\)
   (\(\rho_{\mathrm{ctl}}\\approx -0.240\) on ViT-B) as estimates of one
   shared curvature–decodability law.
2. Interpreting their anticorrelation as a contradiction rather than as a
   mismatch of estimand (mean vs full \(B^S\)) or scale (point vs patch),
   until those mismatches are ruled out by this fixture study.
3. Claiming either instrument is “the” residual curvature of activation space
   without naming T1 vs T2/T3.

## What may stand

- Local-chart \(K_H\) / \(K_{\mathrm{dir}}\) as finite-patch residual
  curvature under the frozen quadratic protocol, if Q recovers T2/T3 here.
- Decoder autodiff \(H^S\) as a pointwise property of a learned surface, if D
  recovers T1 here and is seed-stable.
- The statement that the two real-data signs can coexist because they are
  different functionals, if the distinct-scale label is assigned.

## What must not be claimed

A winner-takes-all comparison of D against Q on unmatched targets.
