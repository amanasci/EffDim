# Phase 3 Findings — the decoder curvature field

**Date:** 2026-08-17. **Milestone:** v1.1 PU Manifold Curvature. **Phase:** 03-decoder-curvature-field.

**One-line outcome.** The curvature instrument is validated and the PU curvature field is not: at
`d = 4` the pipeline recovers analytic mean curvature on a varying, mixed-sign field at
`rho = 0.989` and `R² = 0.980`, and at PU's own conditioning its false-positive floor is 3.87
against a measured median of 1359 — but no curved control ever reached PU's conditioning, so
nothing bounds the field's *accuracy*, and CURV-07 is answered **negatively**.

---

## 1. The override, in this phase's own words

ROADMAP Phase 3's `Depends on` line names a **PASS**. **No PASS exists anywhere in this
milestone.**

| Phase | Verdict | Scope |
|---|---|---|
| 02 | `GATE_VERDICT = FAIL` (`m = 0.412071`) | global |
| 02.2 | `CAE_VERDICT = FAIL` — T1 geodesic distortion `0.296981` (bar `<0.15`), T3 held-out reconstruction margin `3.586350` (bar `<0.90`) | global; T2 chart-transition cycle passed |
| 02.4 | `TOPOAE_VERDICT = FAIL` — T1, T2 | global; T3 passed |
| 02.5 stage 1 | `CURVATURE_VERDICT = FAIL`, marginal and seed-sensitive | local |

Phase 3 ran anyway, by **explicit developer decision on 2026-08-12**
(`02-NOTE-phase-2-stage-on-hold.md` §3). **This is an override, not a satisfied precondition**,
and this phase's record states so rather than inheriting a silent green light.

**The consequence the gate existed to prevent, carried rather than lost:** a curvature field
decoded from an unvalidated parameterization **conflates real curvature with parameterization
damage**, and the synthetic control provably cannot detect that — a synthetic manifold that
trains cleanly never reproduces the pathology (§6).

**The partial mitigation on record** (`02.4-FINDINGS.md`): every FAIL in this milestone is
**global-scoped**, no local-scoped gate has ever failed here (02.2 T2 and 02.4 T3 both passed),
and mean curvature is a **local invariant**. That supports "not globally coordinatizable" rather
than "no usable local structure." **This is an argument for proceeding. It is not a verdict, and
it does not substitute for the PASS that does not exist.**

## 2. The D-05 `n_charts` scope ruling, in this phase's own words

`n_charts` was opened as an in-scope Phase 3 hyperparameter by **explicit developer ruling**,
overriding Phase 02.3's on-hold status **for that knob only**.

Rationale: `02.5-09` measured the chart-decoder curvature read-out degrading monotonically in
the number of charts actually used, and identified the mechanism — atlas fragmentation, with
`max cond(g)` climbing `3.26 → 122.22` and second derivatives banding along artificial chart
seams. A phase asked to decode curvature through a chart decoder cannot hold fixed the one
parameter measured to drive that failure.

**Scope, stated exactly.** The ruling opened `n_charts`. It opened **nothing else**. Phases
02.3, 02.5, 02.6 and 02.7 remain **on hold** (`02-NOTE-phase-2-stage-on-hold.md`), and **no
sealed verdict from Phase 2, 02.1, 02.2, 02.4, 02.5, 02.6 or 02.7 is reopened, softened,
recomputed or reinterpreted anywhere in this phase's record.**

## 3. The working dimension

`chart_dim = 20`, justified against three independent estimates: **TwoNN 19.5**, **local-PCA
median 25.0** (std 2.0), and a **median of 18** across eight geometric estimators.

**`d_frozen = 5` is explicitly rejected.** Reason: at **41% negative eigenvalue mass**, the
residual curve saturates early, so the elbow that produced 5 measured the *flatness failure* of
the Isomap embedding rather than the geometry of the data. A module-level `ValueError` in
`curvature_field_pu_run.py` names D-11 so no edit can quietly reach `chart_dim = 5`.

## 4. Step 1 — the Swiss roll known-answer gate

The floor `ROLL_FLOOR = 0.65` on median `rho_chart` was declared in
`swiss_roll_curvature_sweep_run.py`'s source **before any Phase 3 number existed**. The sweep
ran across `n_charts` and seeds; the gate cleared, and Steps 2 through 4 proceeded.

**Printed with the outcome, not after it:**

- **The D-04 multiple-comparisons caveat.** The gate was read across a sweep rather than at one
  pre-committed point. Selecting the best cell of a sweep and then reporting its statistic
  inflates that statistic — the more cells searched, the better the winner looks by chance
  alone. The pre-declared floor and the multi-seed requirement limit this; they do not remove it.
- **`0.6712` is context.** The raw-point baseline gated nothing, and it **missed its own
  notebook's `>0.90` bar**. It must never be described as a validated benchmark that this phase
  met or missed.

## 5. Steps 2 and 3 — the PU field

**Selected `n_charts = 4`**, by the lexicographic rule declared in plan 03-07 before any PU
number existed and applied unchanged (`git diff 52cbb01..HEAD` shows no edit inside
`apply_selection_rule`). **Axis 1, metric conditioning, decided it outright; no axis was tied.**

What that answer rests on, stated rather than left to be derived: the selected config is the
**worst** of the three on reconstruction *and* on occupancy, clears the occupancy disqualifier
(`median < 2`) by **exactly zero margin** at `2.00`, and won on an axis whose across-seed spread
is **five orders of magnitude** — the seed pulling its median down having collapsed onto a
**single chart**, which is well-conditioned precisely because it is degenerate.

**The field** (converged fit, seed 20260813, full 10,000-row cloud):

| quantity | value |
|---|---|
| `‖H‖` (mean curvature **vector norm**, unflagged n=9,900) | min 681.3, p25 1157.9, **median 1359.0**, p75 1653.9, p95 2391.3, max 4283.9 |
| `cond(g)` | median 9.932e+06, p90 1.583e+07, p99 2.312e+07, max 3.821e+07 |
| flagged at the within-config 99th `cond(g)` percentile | 100 points (1.00%), `‖H‖` mean **2057.4** vs unflagged **1468.3** |
| chart assignment | independently recomputed, **MATCHES**; **0 constructed points** |
| second derivatives, 64 **held-out** rows | max\|Hessian\| 6.697e-01, strictly positive, all finite |
| full-cloud curvature wall clock at `d=20, D=768` | **3,129.5 s** — a measurement that did not exist in this milestone |

The flagging earned its place: the near-singular points carry a **40% higher** mean `‖H‖`, so
averaging them in would have inflated the field, while the median barely moves — which is why a
distribution was required rather than a statistic. The joint `‖H‖`-versus-`cond(g)` histogram
shows mass shifting toward higher conditioning as curvature rises, so the large-`‖H‖` tail is at
least partly conditioning-driven; that is invisible in either marginal alone.

**Step 3 — the finite-difference bridge** (96 held-out chart-assigned points, 3× `VMAP_CHUNK`,
so WR-03's chunking fix is exercised rather than assumed; `near_zero_reference_fraction = 0.0`
on every row, so every relative column is a genuine ratio):

| level | median relative | max relative |
|---|---|---|
| `full_hessian_agreement` | 4.68e-08 / 5.39e-08 | 1.87e-01 / 3.28e-01 |
| `reduced ... [H_vec]` | 4.69e-04 / 5.06e-04 | **6.21e+00 / 2.37e+01** |
| `reduced ... [H_norm]` | 3.52e-05 / 3.65e-05 | 1.48e-04 / 2.28e-04 |

Three readings. The **derivative computation is sound** — autodiff and an independent
non-`torch.func` stencil agree on the raw Hessian to ~5e-08 relative, which plan 03-05's
forward-versus-reverse test structurally could not establish since it compares two autodiff
paths that could share a bug. The **`g⁻¹` contraction amplifies that error ~750-fold**, the
`cond(g) ~ 10⁷` tax measured rather than argued. And the **norm is stable where the vector is
not**: `H_vec` relative disagreement reaches 6.21 and 23.7 while `H_norm` stays near 2e-04 at
the same points — so the quantity CURV-03 refuses to report is precisely the one shown to be
unstable.

**One seed.** The reported unit in this milestone is a three-seed spread. This is a **probe**;
no dispersion is claimed. Converging the other two seeds costs ~3.8 h and was not authorised.

## 6. Step 4 — the synthetic control, and CURV-07

**The parameterization-damage caveat, here beside the numbers rather than in a limitations
section:** these fixtures are sampled cleanly and fit to a clean, unfragmented atlas. They
**never reproduce the atlas fragmentation `02.5-09` measured on real data**. A control that
passes establishes only that the pipeline is correct on a manifold that is **easy to fit** —
necessary, **not sufficient** — and it can never be presented as evidence that the PU field is
free of parameterization damage, because **the one failure mode the override worries about is
precisely the one a cleanly-training synthetic manifold cannot exhibit.**

### The four axes, never collapsed

| fixture | `d` | recon | `cond(g)` med | cosine | mag ratio | `R²` | `rho` | charts |
|---|---|---|---|---|---|---|---|---|
| sphere | 20 | 2.13e-02 | 1.53e+08 | 0.008875 | 58.46 | *undef* | *undef* | 3/4 |
| saddle | 20 | 1.62e-02 | 1.88e+08 | −0.000478 | 9955 | **0.000002** | **−0.0151** | 4/4 |
| sphere | 4 | 1.25e-06 | 2.076 | **0.999893** | **0.999747** | *undef* | *undef* | 1/4 |
| saddle | 4 | 5.11e-07 | 4.098 | **0.992725** | **0.934713** | **0.979850** | **0.988709** | 1/4 |

*undef* is not a bad score: the sphere's analytic `‖H‖` is **constant**, so Spearman has no ranks
to correlate and the calibration regression has a zero-variance predictor. Flat gets no axes at
all — its truth is **exactly zero**, so cosine, ratio and slope are all undefined. **`d = 4` rows
are DIAGNOSTICS, not matched controls**, and must never be quoted as controls for the PU field.

### The `cond(g)` → artifact-curvature band table

From the flat fixture, whose analytic `‖H‖` is exactly zero, so every value is artifact:

| `cond(g)` | artifact `‖H‖` median | max |
|---|---|---|
| ≈ 1.4 (`d=4`) | **0.0073** | 0.0181 |
| ≤ 3.82e+07 (PU's full range) | **3.87** | 24.50 |
| 3.82e+07 – 1e+09 | 35.02 | 2,929 |
| > 1e+09 | 1,410 | 2.15e+06 |

**Monotone across four decades.** This is the first quantitative measurement in this milestone of
a mechanism previously only argued verbally, and it is the phase's most transferable result: it
says how much conditioning a curvature field can tolerate before its values stop meaning
anything. Caveat: each fixture rescales by its own `global_std`, so the rigorous part is the
trend within the three `d=20` bands.

At PU's own conditioning the false-positive floor is **3.87** against PU's median **1359** — a
**351×** margin, with PU's *minimum* (681) still 28× the artifact's *maximum* (24.50).

### CURV-07 — answered

**Is the measured PU curvature a property of the data manifold or an artifact of the fitted
decoder?**

**Neither has been established. The PU field is NOT validated.** Stated conditioned on the
override, as it must be: no PASS exists upstream, so this answer rests on §1's override and must
never be read as if the parameterization had been independently validated.

1. **It is not conditioning artifact** — 351× above the measured floor at PU's own `cond(g)`.
2. **The instrument is correct** — `d=4` recovers analytic curvature on a varying, mixed-sign
   field at `rho = 0.989`, `R² = 0.980`, and correctly detects trace cancellation (0.0200 against
   0.1235 overall). Any remaining failure is **upstream of the curvature computation**.
3. **PU's own accuracy is untested.** Every fixture with known curvature that reached PU's
   dimension failed to *train* to PU-comparable quality (`cond(g)` 15–19× worse, reconstruction
   345–453× worse). **No curved control reached PU's conditioning**, so nothing bounds `‖H‖ =
   1359` — neither its magnitude nor, after the saddle's `rho`, its ordering.

Point 3 is a **structural** gap, not an oversight: closing it needs a CAE that fits a *curved*
20-manifold as well as it fits PU, and no fit in this milestone has done that.

## 7. Library work delivered

- **D-08 / D-09 — the mode toggle.** `chart_curvature.CURVATURE_MODES = ("reverse", "forward")`,
  added alongside rather than replacing: `mode="reverse"` is the default and is **bit-identical**
  to the sealed pre-03-05 path. Measured forward-versus-reverse ratio **21.15–21.96×** (median
  ~21.7) at `d=20, D=768` — well inside D-08's ~38× operation-count ceiling, confirming that
  ceiling is an upper bound. Forward mode remains **opt-in**.
- **D-14 — the derivative bridge.** WR-01 (the float64 guard receiving a bound method instead of
  the model), WR-02 (relative columns exceeding 100% against near-zero references, now carrying
  `near_zero_reference_fraction`), and WR-03 (`calibrate_fd_step`'s unchunked autodiff Hessian)
  all closed by plan 03-03 and exercised at PU scale here at 96 points, 3× `VMAP_CHUNK`.
- **`synthetic_controls.py`** — flat, sphere and saddle with analytic `H` at PU's working scale,
  asserting at import time that its `CURVATURE_CONVENTION` agrees with both sealed modules, so a
  future drift breaks the import rather than silently propagating a factor-of-`d` error.
- **`decoder_priors.py` mode `"christoffel"`** (2026-08-17) — a second-order prior penalizing the
  **tangential** part of `D²F` only. Built, unit-proven not to bias the estimand, **never run**.

## 8. Carried limitations

1. **The gate override** (§1) and its parameterization-damage consequence.
2. **The PU field is unvalidated for accuracy** (§6). It is descriptive, not a verdict.
3. **One seed, not three**, for the field, the bridge and every `d=20` control.
4. **D-12's trigger fired on both legs** — the `d=20` CAE lost to a matched plain-AE control on
   held-out reconstruction (1.733e-04 vs 3.589e-05) and on PH agreement — and was then **retired**
   at plan 03-08's checkpoint as the wrong instrument for a C2 question
   (`03-NOTE-d12-retirement.md`). **No `d` sweep was escalated to.** The retirement rests on the
   C0/C2 argument, not on the result being unwelcome; the result is recorded above unhedged.
5. **Reconstruction and topology do not predict curvature quality.** A C0 quantity cannot bound a
   C2 one: a decoder learning `y = 0.7ax²` where truth is `y = ax²` attenuates curvature 30% with
   essentially no reconstruction signal. **Measured here:** removing total-loss early stopping cut
   held-out `mse_per_dim` **62.2%** and left `cond(g)` **unmoved** (9.758e+06 → 1.0033e+07).
6. **The training objective constrains no decoder derivative at any order.**
   `cae.lipschitz_penalty` regularizes `chart_encoders`; curvature is decoded through
   `chart_decoders` composed with `embedding_decoder`. The two sets **share no parameter**.
7. **The matched continuous protocol diverges.** A single continuous 300-epoch fit on the *flat*
   fixture reaches non-finite weights at ~epoch 220 (~27,500 steps); the same configuration in
   25-epoch blocks completes cleanly. Mechanism hypothesis — accumulated Adam state — is
   **consistent with the evidence, not confirmed.** **The converged PU fit ran deep into the same
   regime without crashing**, which is a weaker guarantee than was held when its field was computed.
8. **The converged PU fit never plateaued.** Best epoch was its **last**; the trailing 25-epoch
   window still improved 5.271e-02 against a 1.0e-03 tolerance. The budget ended training.
9. **Undeclared `ripser` / `persim` dependency**, inherited and unresolved because
   `pyproject.toml` is frozen this milestone.
10. **Split seed and torch seed are coupled** in the roll sweep, so a "seed spread" varies both
    together and cannot separate their contributions.
11. **Forward mode remains opt-in**, promotable only on a named condition it has not met.
12. **Flat at `d=20` is a 2,000-row diagnostic probe**, not a runner record — its production fit
    is what discovered limitation 7.

## 9. Handoff to Phase 4

Phase 4 receives a **per-point `‖H‖` field over the full 10,000-row PU cloud** at
`n_charts = 4`, `chart_dim = 20`, with `cond(g)` per point beside it and 100 near-singular points
flagged, in `notebooks/.cache/03_curvature_field_pu.jsonl` (`kind: "field_cell"`).

**Phase 4 consumes ORDERING**, not magnitude — it quantile-partitions the field into regions.
That is why Spearman gated Step 1 and magnitude did not, and it is the reason limitation 2 matters
so directly: **the saddle at `d=20` measured `rho = −0.0151`**, i.e. no ordering information at
all, on the only fixture at PU's dimension where ordering could be checked against truth.

**Conditions on that handoff, all of which Phase 4 must carry forward:**

- The field is **descriptive and unvalidated** (§6). It is not a verdict on PU's curvature.
- It is **one seed**. Any Phase 4 partition derived from it inherits that.
- Its **ordering is unvalidated at PU's dimension**, which is the specific property Phase 4 uses.
- The **override** (§1) travels with every number derived from it.

**The two levers if Phase 4 needs a better field**, both pointing at the parameterization rather
than the curvature code, and neither yet measured:

1. **The Christoffel prior** (`decoder_priors.py`, mode `"christoffel"`) — targets `cond(g)`
   directly, provably cannot bias `‖H‖`, built and unrun.
2. **Chart count.** All three `d=4` fits used **1 of 4 charts** and reached `cond(g)` 1.4–4.1; all
   three `d=20` fits fragmented and reached `~10⁸`. The pattern holds across zero, constant and
   varying curvature.

The `d=20` sphere or saddle is the right test bed for either, because both have known truth *and*
currently fail — so the four axes moving toward 1 as `cond(g)` falls is a directly readable
mechanism check rather than a guess.

---
*Phase: 03-decoder-curvature-field*
*Findings recorded: 2026-08-17*
