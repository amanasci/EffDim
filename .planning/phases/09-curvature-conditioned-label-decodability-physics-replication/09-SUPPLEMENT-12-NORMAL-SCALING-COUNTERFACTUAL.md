# 09-SUPPLEMENT-12 — counterfactual local second-order surrogate of the readout: the probe-facing shape term helps, its sign reversal hurts

**Status:** post-hoc. **Not pre-registered. Feeds no verdict.** **Written:** 2026-09-15 UTC; **revised twice** after external review (v2: t=0 renamed shape-flat, random control matched on the contracted tensor norm, framing as a local surrogate; v3: random control matched on the centred quadratic amplitude ‖q_c‖₂, exact finite-sample criterion stated; v1/v2 records kept on the pod under `ns-out-v1/`, `ns-out-v2/`).
Runner `notebooks/diagnostics/09_physics_normal_scaling_run.py` (new; imports the split runner unchanged;
runner sha256 `6eb2d7bf61ff5fc6…`), pod `universetbd-0`, tmux `effdim-ns{A,B}`, 2–4 min per run. Geometry
from the stored anchor npz of each decoder (ViT-B d=16/20 from the Phase 9 run; the four X decoders,
seed 0). Records `notebooks/.cache/09_physics_normal_scaling_<enc>_d<d>.{jsonl,npz}`, logs
`09_ns_{A,B}.log`, sha256 verified both sides. Smoke mode on the in-sphere fixture ran locally first.

## Question and design
The partials are rank correlations across anchors; no anchor is flat, so they cannot say whether bending
*away* from the label's curvature hurts relative to no bending. This test scores a counterfactual local second-order surrogate of the
readout at a fixed manifold; it does not construct a new global weight vector. At each anchor the fitted
global ridge weight is split as w = w_T + w_rad + w_S (tangent, radial/sphere, in-sphere normal) and the
probe-facing Hessian as K = K_S − (w·x̂) g with K_S = ⟨w_S, II^S⟩. The surrogate
ŷ_t(u) = c + aᵀu + ½K_sph(u,u) + (t/2)K_S(u,u) (in practice: data-side tangent-plus-sphere readout
(w_T + w_rad)·x plus t·q, intercept c refit) has a local sum of squares that is an exact quadratic in t:
SS(t) = |r0 − t q_c|², r0 the centred residual of the **shape-flat** readout (t = 0 keeps the sphere term; it is
not K = 0), q_c = q − q̄·1 the **centred** quadratic on the anchor's actual neighbours (the code centres q before
forming ⟨q,q⟩; the denominator is ‖q_c‖², as the intercept refit requires). Finite-sample criterion:
2⟨r0, q_c⟩ > ‖q_c‖², t*_data = ⟨r0, q_c⟩/‖q_c‖². This is the finite-neighbourhood counterpart of the tensor-level
statement below, not the same quantity. With R = Hess_M y − K_sph the remaining label curvature, t = 1 beats
shape-flat iff 2⟨R, K_S⟩ > ‖K_S‖² and t* = ⟨R, K_S⟩/‖K_S‖²; "bending toward the label" means ⟨R, K_S⟩ > 0,
the manifold's bending *as seen in the readout's normal direction*. Hence t* = ⟨e0,q⟩/⟨q,q⟩ and R²(t) on the grid
t ∈ {−1, −0.5, 0, 0.5, 1, 1.5, 2} follow. Variants for q:
- **S_model** (primary): q = ½⟨w_S, II^S⟩(u,u) in the anchor's chart coordinates u = g⁻¹Jᵀ(x − x0) — the
  decoder's own second-order term, i.e. the quantity the residual expansion is about; t = +1 is the
  manifold bending as it does in the probe's normal direction, t = −1 the mirror-image bending, t = 0 flat.
- **S** (data): q = w_S·x on the data. Includes the off-manifold part of x (first order in the 5% reconstruction
  residual), so it measures how much the probe reads off the 16-d manifold, not the second-order term.
- **random_qmatched** (primary null, v3): the random quadratic rescaled so that its centred amplitude on the actual
  neighbours equals the fitted one, ‖q_v,c‖₂ = ‖q_c‖₂ — identical empirical quadratic strength, different orientation.
- **random_matched** (v2): v rescaled so that ‖⟨v, II^S⟩‖_g = ‖⟨w_S, II^S⟩‖_g (tensor-level match; on anisotropic
  k-NN patches equal tensor norm does not guarantee equal quadratic amplitude).
- **random_wnorm** (kept for comparison): q = r·x with ‖r‖ = ‖w_S‖. In codimension ≈ 750 a random normal
  direction is nearly orthogonal to the ≤ d(d+1)/2 = 136 directions II^S can span, so its contracted tensor is
  a fraction of the fitted one (ratio recorded per anchor); matching ‖v‖ alone is not a fair control.
Also `full` / `full_model` / `S_proj` (radial part included / model first-order base); same conclusions.

Key numbers per run and label (S_model; `random_qmatched` in brackets, v3): fraction of anchors where t = +1 beats
shape-flat ("help"), fraction where t = −1 is worse than shape-flat ("hurt"), median ΔR² at +1 and −1, median t*,
fraction t* > 1. (v2 records; the S_model columns are unchanged from v1 because the model term did not change.)

| run | label | help | hurt | ΔR²(+1) | ΔR²(−1) | t* p50 | t*>1 | [random help / hurt] |
|---|---|---|---|---|---|---|---|---|
| ViT-B d16 | mag_r | 0.97 | 1.00 | +0.030 | −0.051 | 2.3 | 0.91 | [0.26 / 0.80] |
| ViT-B d16 | photo_z | 0.99 | 0.99 | +0.028 | −0.038 | 3.0 | 0.98 | [0.33 / 0.68] |
| ViT-B d16 | smooth | 0.90 | 1.00 | +0.021 | −0.040 | 1.7 | 0.78 | [0.24 / 0.78] |
| ViT-B d16 | stellar | 0.96 | 0.99 | +0.026 | −0.040 | 2.6 | 0.93 | [0.29 / 0.70] |
| ViT-B d20 | mag_r | 0.99 | 1.00 | +0.031 | −0.049 | 2.6 | 0.94 | [0.24 / 0.78] |
| ViT-B d20 | photo_z | 0.99 | 1.00 | +0.029 | −0.038 | 3.6 | 0.99 | [0.36 / 0.71] |
| ViT-B d20 | smooth | 0.90 | 1.00 | +0.020 | −0.035 | 1.9 | 0.78 | [0.22 / 0.77] |
| ViT-B d20 | stellar | 0.97 | 1.00 | +0.027 | −0.038 | 3.1 | 0.95 | [0.32 / 0.72] |
| DINOv3 d16 | mag_r | 0.92 | 0.99 | +0.018 | −0.031 | 2.1 | 0.81 | [0.29 / 0.71] |
| DINOv3 d16 | photo_z | 0.96 | 0.99 | +0.020 | −0.031 | 2.6 | 0.91 | [0.29 / 0.71] |
| DINOv3 d16 | smooth | 0.93 | 1.00 | +0.024 | −0.055 | 1.5 | 0.73 | [0.22 / 0.78] |
| DINOv3 d16 | stellar | 0.95 | 1.00 | +0.016 | −0.028 | 2.1 | 0.87 | [0.26 / 0.74] |
| CLIP-B d16 | mag_r | 0.95 | 0.99 | +0.018 | −0.031 | 2.2 | 0.86 | [0.30 / 0.68] |
| CLIP-B d16 | photo_z | 0.99 | 1.00 | +0.019 | −0.026 | 3.4 | 0.97 | [0.36 / 0.64] |
| CLIP-B d16 | smooth | 0.92 | 0.99 | +0.014 | −0.027 | 1.7 | 0.78 | [0.25 / 0.69] |
| CLIP-B d16 | stellar | 0.97 | 1.00 | +0.020 | −0.028 | 2.9 | 0.93 | [0.35 / 0.68] |
| ConvNeXt-B d16 | mag_r | 0.92 | 1.00 | +0.018 | −0.031 | 2.0 | 0.81 | [0.26 / 0.73] |
| ConvNeXt-B d16 | photo_z | 0.99 | 1.00 | +0.018 | −0.026 | 2.9 | 0.98 | [0.28 / 0.64] |
| ConvNeXt-B d16 | smooth | 0.88 | 0.98 | +0.017 | −0.036 | 1.6 | 0.74 | [0.23 / 0.75] |
| ConvNeXt-B d16 | stellar | 0.99 | 1.00 | +0.022 | −0.032 | 3.1 | 0.98 | [0.29 / 0.65] |
| ViT-L d16 | mag_r | 0.96 | 0.99 | +0.022 | −0.034 | 2.6 | 0.90 | [0.32 / 0.74] |
| ViT-L d16 | photo_z | 0.97 | 0.99 | +0.018 | −0.025 | 3.2 | 0.93 | [0.34 / 0.67] |
| ViT-L d16 | smooth | 0.79 | 0.97 | +0.013 | −0.029 | 1.6 | 0.66 | [0.25 / 0.77] |
| ViT-L d16 | stellar | 0.98 | 1.00 | +0.022 | −0.032 | 2.9 | 0.94 | [0.30 / 0.67] |

q-matched random |ΔR²| medians are ≤ 0.014 in every cell and t* is −0.16 to +0.06 (tensor-matched v2: ≤ 0.013,
−0.16 to +0.04; same picture): an unaligned term of the same empirical quadratic amplitude is typically harmful under
either sign and shows none of the asymmetry, as the ‖q_c‖² penalty alone predicts — orientation, not magnitude, is what
the fitted term supplies. The ‖v‖-matched control
(`random_wnorm`, v1 numbers: help 24–40%, hurt 57–79%, |ΔR²| ≤ 0.009) is weaker only because its contracted
tensor is small: median ‖⟨r, II^S⟩‖_g / ‖⟨w_S, II^S⟩‖_g at equal norm is 0.26–0.38 across runs (p25–p75
0.21–0.42), the high-codimension effect the review anticipated. Data variant S: help 0.94–1.00, hurt 1.00, ΔR²(+1)
+0.05 to +0.13, ΔR²(−1) −0.10 to −0.24 — larger because it also carries the off-manifold label signal
(|w_S|/|w| p50 0.71–0.87: the ridge probe reads mostly off the 16-d manifold). cos(decoder image, data row)
p05 ≥ 0.95, p50 ≥ 0.98 on every run. The decoder's metric-weighted cross term ⟨Hess_M y, ⟨w_N, II^S⟩⟩_g is
positive at 371–509 of 512 anchors (73–99%); inside the negative stratum (3–141 anchors) the model term
still helps at 75–100% of anchors with positive median ΔR², and ρ(t*, cross/‖K‖²) is only +0.10 to +0.60.
The sign of the estimated alignment cosine therefore does not predict the anchor-level counterfactual; the
intervention uses the raw local residual and needs no label-Hessian estimate, whose direction is the
unreliable part (split-half tensor cosine 0.19–0.35, Supplement 09).

## Significance with dependent anchors (added 2026-09-16)
Over all 512 anchors the sign tests are absurdly small (largest one-sided binomial p across the 24 cells: 4×10⁻⁴² for
"t = +1 beats shape-flat", 2×10⁻¹²⁶ for "t = −1 worse", Wilcoxon on ΔR²(+1) ≤ 4×10⁻³³, paired fitted-vs-q-matched-random
≤ 4×10⁻⁶⁸) and not to be trusted as such: 512 neighbourhoods of 2,048 on 86,471 points put each point in ≈ 12
neighbourhoods (pairwise overlap median ≈ 0, p90 ≈ 0.12, max 0.90–0.94 of k). `09_physics_normal_scaling_thin_run.py`
recomputes each run's panel and saves the 512×512 overlap matrix (`*_thin.npz`, sha256 verified); exact disjointness from
the union of kept sets leaves only 6–7 anchors (a few kept sets already cover ~14% of the data), so the paper uses a
maximal independent set of the graph "pairwise overlap > 5% of k" (min-degree greedy, computed in `appendix_gen.py`):
19–21 anchors per run, t = +1 beats shape-flat at 81–100% (one-sided sign test p ≤ 10⁻³ in every cell), t = −1 worse at
95–100% (p ≤ 10⁻⁵). At 1% / 2% / 10% thresholds the sets have 12–14 / 13–16 / 30–34 members with the same picture.
These are the p-values to quote; the 512-anchor ones are not.

## Reading
- **The probe-facing shape term helps, its sign reversal hurts — at the anchor level, all five encoders, both d,
  all four labels.** Adding the decoder's in-sphere second-order term to the shape-flat surrogate raises local R²
  at 79–99% of anchors (median +0.013 to +0.031); reversing its sign lowers it at 97–100% (median −0.025 to
  −0.055); a random orientation with the same centred quadratic amplitude shows no such asymmetry and is typically
  harmful under either sign (help 22–36%, hurt 64–80%, |ΔR²| ≤ 0.014). The asymmetry between +1 and −1 is what a parabola with t* > 1 gives.
- **t* > 1 throughout** (median 1.5–3.6; t* > 1 at 66–99% of anchors): consistent with the globally
  ridge-regularized probe under-using a locally beneficial second-order term (the weak-ridge sensitivity points
  the same way), but not uniquely so — decoder scaling error, higher-order terms, or the gap between a global
  optimum and per-anchor optima could contribute. Stated as "consistent with", not as cause.
- **Cross-anchor vs within-anchor.** The partials ask whether variation in ‖Hess_M y − K‖ across anchors explains
  variation in local R²; the counterfactual asks whether the fitted probe-facing quadratic is useful *within* each
  anchor. The latter can be positive where the former has no cross-anchor signal — which is why stellar_mass shows
  t = 1 beating shape-flat at 96–99% of anchors while its mismatch partial is null.
- **What this is and is not.** It is a counterfactual local second-order surrogate scored at a fixed manifold,
  not a new global weight vector and not a manipulation of the manifold. At each anchor the manifold's bending
  *as seen in the fitted readout's normal direction* (K_S depends on II^S and on w_S) is shown to carry label
  curvature (t = +1 helps) and its reversal to remove it (t = −1 hurts). The fitted probe's alignment is itself a
  consequence of least squares choosing w_N to exploit the bending; the test shows that choice is right at nearly
  every anchor and under-sized.
- The alignment *partial* and this test measure different things. The partial asks whether the
  estimated cosine varies across anchors with local R²; the intervention asks, at each anchor, whether the
  decoder's bending term is useful at all. The second is nearly universal for a fitted probe (least squares
  chooses w_N to exploit the bending), which is why it can be tested only by intervention.
- stellar_mass shows the same pattern although its cross-anchor partials are null: the intervention asks
  whether K is useful at each anchor, the partial whether variation in K across anchors tracks variation in
  R²; the former can hold where the latter has no signal.
- Manuscript: Appendix E (generated table), one sentence in Section 5.

---
*Phase 09 follow-up. Not pre-registered, feeds no verdict.*
