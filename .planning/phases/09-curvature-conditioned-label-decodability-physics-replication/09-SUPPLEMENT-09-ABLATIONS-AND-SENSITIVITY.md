# 09-SUPPLEMENT-09 — decoder ablations, cross-fitted label Hessian, weak-ridge probe

**Status:** post-hoc. **Not pre-registered. Feeds no verdict.** **Written:** 2026-09-13 UTC.
Runner `notebooks/diagnostics/09_physics_probe_facing_split_run.py` (options `--fit-seed`, `--hidden`,
`--hessian-xfit`, `--alpha`; commits 79589d3, c54a2eb), pod `universetbd-0`, labels from the cached
parquet (sha256 60f2f82e…). Records in `notebooks/.cache/09_physics_probe_facing_split_{seed1,seed2,w400,xfit,alpha1}.jsonl`.
Manuscript Appendices A and B are generated from these by `docs/latex/ml4ps/appendix_gen.py`.

## A. Decoder ablations (d=16; seeds 1, 2 at 250³; seed 0 at 400³; var_explained 0.952 all)
Multi-scale partials vs local R². Sphere term −0.26…−0.35 on every label and variant; mismatch (emp)
−0.34…−0.45 on mag_r and photo_z, −0.05…−0.09 (n.s.) smooth, null stellar; alignment +0.31…+0.41 mag_r,
+0.24…+0.26 photo_z, +0.09…+0.19 smooth, null stellar; shape term mag_r +0.05*…+0.12, photo_z −0.06…−0.09,
smooth +0.07*…+0.20, stellar −0.06*…+0.09. Every significant sign of the main table is retained.

## B. Cross-fitted Hessian (seed-0 geometry; each 2,048-patch split into random halves)
Fit Hess_M y on half A, score local R² on half B, and vice versa; multi-scale controls from the full patch.
mag_r: mismatch −0.38/−0.34 (d=16), −0.43/−0.40 (d=20) vs same-half −0.38/−0.43; alignment +0.33/+0.31,
+0.39/+0.37. photo_z: mismatch −0.42/−0.37, −0.42/−0.42; alignment +0.23/+0.20, +0.22/+0.22. smooth:
mismatch −0.10/−0.17, −0.15/−0.21; alignment +0.17/+0.19 (d=16), n.s. (d=20). stellar: all n.s.
Split-half reliability of Hess_M y: norm rank ρ 0.91–0.94; full-tensor metric cosine median 0.31–0.35
(d=16), 0.19–0.24 (d=20); ρ(R²_A, R²_B) 0.93–0.96. Shared local label noise does not drive the mismatch
or alignment columns; the tensor direction is only moderately reliable, the norm highly.

## C. Weak-ridge probe (α = 1 instead of 100; global OOF R² 0.64–0.67 vs 0.48–0.53)
Sphere term: mag_r −0.21/−0.22, smooth −0.12/−0.13 (still negative, p ≤ 0.02); photo_z +0.05*/+0.03*,
stellar −0.02*/−0.02* (gone). Mismatch −0.49…−0.58 (mag_r, photo_z), −0.22/−0.27 smooth, −0.11/−0.13
stellar; alignment +0.25…+0.47 on all four labels; shape term mag_r +0.25/+0.11, photo_z +0.05/−0.09,
smooth +0.17/+0.20, stellar +0.13/+0.35.

**Reading.** The sealed-probe sphere term on photo_z and stellar_mass was ridge shrinkage of extreme
predictions; on mag_r and smooth_fraction it survives. Mismatch and alignment strengthen under the
weaker probe and are unchanged under cross-fitting: those are the robust readout-side results. The
manuscript states exactly this (abstract, Section 5, Discussion).

---
*Phase 09 follow-up. Not pre-registered, feeds no verdict.*
