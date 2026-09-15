# 09-SUPPLEMENT-10 — known-surface alignment demonstration (experiment K): partial negative

**Status:** post-hoc. **Not pre-registered. Feeds no verdict.** **Written:** 2026-09-15 UTC.
Runner `notebooks/diagnostics/09_fixture_probe_facing_split_run.py` (imports the fixture probe-facing
runner and the in-sphere generator unchanged). New this session: fixture overrides `--scale-mult`,
`--width-mult`, `--amp-mult` (multiply the sealed fixture's `scale_choices`, `bump_widths`, `bump_amps`;
recorded in the environment row) and the label family `bumpamb_<beta>`. Records
`notebooks/.cache/09_fixture_alignment_demo*.jsonl` (1,000 permutations, seed 20260905, gammas −1 and 0.6).

## Design
Surface: the in-sphere generator, normalize([stereo(z); 0.8 h(z); 0…]), h = four Gaussian bumps, so the
in-sphere bending is 0.8·Hess h. Labels:
- `bumpalt_β` = a·z + β h_alt(z), h_alt = the same bumps with two amplitude signs flipped (the label bends
  with the surface near two bumps and against it near the other two);
- `bumpamb_β` = (w·x)/σ + β h_alt(z)/σ' (ambient-linear part, whose covariant Hessian equals its own
  probe-facing curvature identically, plus the standardised bump-alt part; β is the bump-to-linear
  standard-deviation ratio).
Prediction for both: mismatch partial negative, alignment partial positive, at both γ.

## Why the sealed d=16 fixture cannot show it
Bump centres sit at radius ≈ 0.5·√16 = 2 in latent space and the latents at radius 1.6–3.6; with widths
0.7–1.0 the bumps are ≈ 0 everywhere (median pf_tan 0.04 vs label Hessian 25: the label's Hessian is the
Christoffel term of the chart-linear part a·z). First attempt (sealed fixture): alignment −0.03 / −0.08
(n.s.), mismatch −0.71 / −0.58, at γ = −1 / 0.6.

## Redesign grid (d = 16, n = 86,471, k = 2,048, 512 anchors)
Multi-scale partials vs local R² (γ = −1 / 0.6), median pf_tan : hess_label ratio in brackets.

| fixture | label | alignment | mismatch | shape | sphere |
|---|---|---|---|---|---|
| scale ×0.25, amp ×5 | bumpalt_0.3 | −0.33 / −0.28 | −0.59 / −0.60 | −0.11 / −0.14 | −0.52 / −0.52 [1:15] |
| scale ×0.25, amp ×5 | bumpalt_1.0 | **+0.39 / +0.35** | −0.74 / −0.75 | +0.33 / +0.30 | −0.50 / −0.47 [1:6] |
| scale ×0.25, amp ×10 | bumpalt_0.3 | −0.20 / −0.35 | −0.61 / −0.65 | +0.35 / +0.17 | −0.58 / −0.55 [1:9] |
| scale ×0.25, amp ×10 | bumpalt_1.0 | +0.07* / −0.05* | −0.66 / −0.73 | +0.21 / +0.30 | −0.38 / −0.41 [1:5] |
| scale ×1, width ×2.5, amp ×5 | bumpalt_0.3 | −0.53 / −0.64 | −0.60 / −0.34 | −0.43 / −0.45 | −0.66 / −0.61 [1:23] |
| scale ×1, width ×2.5, amp ×5 | bumpalt_1.0 | −0.10 / −0.26 | −0.69 / −0.39 | −0.30 / −0.18 | −0.36 / −0.26 [1:16] |
| scale ×1, width ×2.5, amp ×10 | bumpalt_0.3 | −0.41 / −0.40 | −0.48 / −0.43 | −0.39 / −0.32 | −0.57 / −0.45 [1:15] |
| scale ×1, width ×2.5, amp ×10 | bumpalt_1.0 | +0.06* / +0.09 | −0.54 / −0.50 | −0.29 / −0.25 | −0.40 / −0.49 [1:9] |
| scale ×0.5, width ×1.5, amp ×5 | bumpalt_0.3 | −0.43 / −0.40 | −0.56 / −0.57 | −0.47 / −0.50 | −0.46 / −0.38 [1:10] |
| scale ×0.5, width ×1.5, amp ×5 | bumpalt_1.0 | −0.48 / −0.43 | −0.69 / −0.65 | −0.65 / −0.61 | −0.38 / −0.19 [1:9] |
| scale ×0.25, amp ×5 | bumpamb_0.3 | −0.19 / −0.16 | −0.78 / −0.79 | +0.43 / +0.53 | −0.54 / −0.50 [1:7] |
| scale ×0.25, amp ×5 | bumpamb_1.0 | +0.02* / +0.17 | −0.64 / −0.68 | +0.23 / +0.39 | −0.40 / −0.45 [1:5] |
| scale ×0.5, width ×1.5, amp ×5 | bumpamb_0.3 | +0.17 / +0.06* | −0.86 / −0.88 | −0.24 / −0.29 | −0.18 / −0.29 [1:4] |
| scale ×0.5, width ×1.5, amp ×5 | bumpamb_1.0 | −0.08* / −0.26 | −0.82 / −0.80 | −0.59 / −0.57 | −0.11 / −0.17 [1:5] |
| scale ×1, width ×2.5, amp ×5 | bumpamb_0.3 | −0.09 / +0.11 | −0.40 / −0.44 | −0.26 / −0.02* | −0.32 / −0.12 [1:7] |
| scale ×1, width ×2.5, amp ×5 | bumpamb_1.0 | −0.04* / −0.14 | −0.14 / −0.32 | −0.03* / −0.22 | −0.24 / −0.19 [1:12] |

\* not significant at 0.05. Global OOF R² 0.70–0.96 (bumpalt), 0.93–0.99 (bumpamb).

d = 4 smoke fixture (n = 4,000, D = 64, k = 128, 64 anchors; sealed geometry): bumpalt alignment
−0.18* / +0.43, −0.02* / +0.45; bumpamb +0.36 / +0.04*, +0.57 / −0.21* (β = 0.3, 1.0; γ = −1 / 0.6);
mismatch −0.27 to −0.66 throughout.

## Reading
- The **mismatch** partial is negative in all 40 cells (−0.14 to −0.88), on every fixture geometry, both
  label families, both γ: the full second-order condition of Section 5 reproduces on a surface with known
  curvature, which is the one thing the fixture was able to say.
- The **alignment** partial has no stable sign: one cell pair meets the pass rule (scale ×0.25, amp ×5,
  bumpalt_1.0: +0.39 / +0.35), the neighbouring β and amplitude reverse or null it, and the wide-bump
  geometries make it negative. The alignment cosine is a direction statistic; on this fixture its variation
  across anchors is set by the shared part of the label and probe Hessians (the chart-linear or
  ambient-linear term), not by the bump term the demo was built around, and the local R² is driven by the
  first-order gradient residual that the mismatch column happens to track. The mechanism is not refuted,
  but it is not demonstrated either.
- Manuscript consequence: the alignment column stays an empirical regularity of the galaxy data; the
  Limitations paragraph now says so in one sentence. No Section 5 sentence was added. The pass rule of
  `09-NEXT-EXPERIMENTS-2026-09-14.md` (alignment positive and mismatch negative at both γ) is **not met**.

---
*Phase 09 follow-up. Not pre-registered, feeds no verdict.*
