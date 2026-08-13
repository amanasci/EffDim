# Phase 3: Decoder & Curvature Field - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-13
**Phase:** 3-decoder-curvature-field
**Areas discussed:** Step-1 bar & stopping rule, Chart count as a knob, PU fit (which one, what d), Verification depth & carried debt

---

## Step-1 bar & stopping rule

### Q1 — What does the step-1 statistic get computed over?

| Option | Description | Selected |
|--------|-------------|----------|
| Median over ≥5 seeds | Fit at ≥5 torch seeds, compare the median to the bar, report spread. Matches `02.5-PREREGISTRATION-AMENDMENT-01`'s 5-seed rule | ✓ |
| Single seed, spread reported after | One fixed seed gates, as ROADMAP step 1 literally reads | |
| Seed-conditional: report the curve | Report ρ as a function of charts-actually-used; no single value to gate on | |

**User's choice:** Median over ≥5 seeds.

---

### Q2 — What bar must the median clear? *(asked twice; first attempt paused for clarification)*

| Option | Description | Selected |
|--------|-------------|----------|
| Beat the recomputed baseline | Recompute the raw-point baseline in-notebook, require median ρ > it; report 0.90 as context | |
| Absolute 0.90, baseline as context | The bar `02.5-05` set for itself | |
| Two-tier: continue vs claim | Baseline to continue, 0.90 to license calling the field trustworthy | |
| **Other (user-supplied)** | **Absolute floor 0.65. Baseline as context only. Swiss roll only. Exact mean curvature calculations only** | ✓ |

**User's choice:** free text — "Make the floor be 0.65. Baseline as context. Only for swiss roll. Only use exact mean curvature calculations for this."

**Notes:** Before answering, the user asked three clarifying questions, each of which changed what got asked next:

1. *"Is rho in this case the spearman correlation?"* — Confirmed: `curvature_probe.spearman_gate_statistic`, Spearman between per-point estimated `‖H‖` and analytic `‖H‖` over 3,000 fixture points. Locked as the gating statistic by 02.5's D-01 because Phase 4 partitions by `‖H‖` quantiles and so consumes ordering, not magnitude.
2. *"How are we estimating mean curvature?"* — Walked both arms: the chart-decoder arm (`J = DF`, `g = JᵀJ`, `P_N = I − Jg⁻¹Jᵀ`, `H = tr_g(P_N D²F)`, exact by autodiff) and the raw-point centroid arm (kNN, centroid displacement, local-PCA tangent basis, `H = gap_normal·(2d/r2)`).
3. *"A low rho would mean a failure of the way we estimate curvature compared against the analytical, not a failure of CAE, correct?"* — **Corrected: inverted for the chart arm, correct for the baseline arm.** The chart-arm estimator is exact to float64 given the decoder, so a low ρ indicts the *learned surface*, not the estimator. Two supporting facts: the estimator has no seed dependence yet ρ moved 0.93 across seeds; and cosine similarity `0.9706` shows chart routing, coordinate spaces, convention and scaling were all wired correctly. The baseline's `0.6712` *is* estimator error (`O(r²)` bias, density contamination, tangent-basis noise), no model involved.

**Consequence flagged at the time:** the 4-seed sweep `{−0.1444, −0.0604, 0.4250, 0.8665}` has median ≈ 0.18, so the 02.2 configuration fails a 0.65 floor decisively — which is what made chart count (area 2) the phase's only live lever rather than an optional topic.

---

### Q3 — Under an n_charts sweep, what does the floor apply to?

| Option | Description | Selected |
|--------|-------------|----------|
| Best config clears it | Passes if median ρ > 0.65 at any swept `n_charts`, full table printed, best-of-N named as a caveat | ✓ |
| Pre-declare one config, sweep is context | Name the gated config before running; no best-of inflation | |
| Majority of configs clear it | Require > half the swept values to clear; much stronger, likely to fail | |

**User's choice:** Best config clears it.

---

## Chart count as a knob

### Q1 — How does Phase 3 treat chart count?

| Option | Description | Selected |
|--------|-------------|----------|
| Model-side seed selection | Fit ≥5 seeds at `n_charts=8`, select by `cond(g)` and occupancy, never by ρ; no architecture change | |
| Lower n_charts is in scope | Treat `n_charts` as a Phase 3 hyperparameter and sweep it down; needs an explicit scope ruling since CAE iteration is on-hold 02.3 | ✓ |
| Freeze at 02.2 config, report | Change nothing; median lands near 0.18, phase stops and reports | |
| Selection now, n_charts if it fails | Ordered fallback decided in advance | |

**User's choice:** Lower `n_charts` is in scope — recorded as an explicit scope ruling over Phase 02.3's hold, for this knob only.

**Notes:** Presented with the distinction that `n_charts` is a hyperparameter while *charts actually used* is an outcome (all four seeds ran `n_charts=8` and occupied 3, 5, 8, 8), and that selecting on ρ itself would be tuning on the answer whereas `cond(g)` and occupancy are model-side and transfer to PU. Also noted, and not contested: as `n_charts` falls the CAE degenerates toward a plain autoencoder, so a win at 2–3 charts is itself a result about the model — and CLAUDE.md already mandates the matched `PlainAutoEncoder` baseline that would show it.

---

### Q2 — What transfers from the roll sweep to PU?

| Option | Description | Selected |
|--------|-------------|----------|
| Criterion transfers, value re-swept | Roll calibrates a model-side rule against known H; PU re-sweeps and applies it | |
| Carry the winning n_charts literally | Cheapest; imports a 2-D sheet's value into a `d≈20` manifold as a named assumption | |
| Sweep on PU, roll is validation only | Roll shows only that the pipeline recovers a known answer; PU sweeps independently, roll winner is context | ✓ |

**User's choice:** Sweep on PU, roll is validation only.

---

### Q3 — What does the PU-side selection rule key on? *(asked twice; first attempt paused for clarification)*

| Option | Description | Selected |
|--------|-------------|----------|
| cond(g) + occupancy + reconstruction | The three model-side diagnostics already emitted | |
| cond(g) alone | One number, no weighting question | |
| **Add persistent homology as a fourth** | PH agreement alongside the three; expensive at D=768, H2 power never measured, 02.6's PH axis carries two confounds | ✓ *(modified)* |
| Decide at plan time from the roll | Let the roll show which diagnostics predict ρ | |

**User's choice:** "add persistent homology as a fourth. only compute H0 and H1."

**Notes:** Two clarifying questions preceded this:

1. *"What is cond(g)?"* — Condition number of the pullback metric `g = JᵀJ`; ratio of largest to smallest eigenvalue; large values mean the chart is crushed nearly flat in some direction and at infinity the differential has dropped rank (non-immersion). Candidate for PU precisely because it needs no ground truth and tracked charts-used in lockstep on the roll (3.26, 7.64, 63.19, 122.22).
2. *"And for reconstruction do we use persistent homology?"* — No: reconstruction means `‖x − F(E(x))‖` via `cae.reconstruction_stats`, DEC-03/04's metric. PH was 02.6's separate ranking axis. Flagged that PH is where the CAE looks worst (`cae_retained = 0.183246`, ~5× worse H0 retention than a plain AE at ~8× the parameters), that 02.6's PH ranking carries two named confounds, and that H2 has no power analysis at all.

The H0/H1 restriction removes both of PH's known liabilities here: H2's unbounded nulls and its compute cost. H1's power *is* bounded at PU's own regime (cycle detected down to 3.3× transverse thickness at n=383, 2.0× at n=2000).

---

## Cross-cutting: autodiff cost at PU scale

*Raised by the user mid-discussion, not a pre-identified gray area.*

**User's opening:** *"But on PU, didn't we have a plan to use an estimate of curvature since the jacobian will be very expensive to calculate for high dimensions? I may want to change how we do that."*

**Response:** the remembered plan (Hutchinson randomized trace) was deliberately **demoted** by `02.5-NOTE-randomized-trace.md` — at `d=20` the exact trace is 20 HVPs against `K=8`, a 2.5× saving on something that was never the bottleneck, and the decoder arm's real advantage over raw points is statistical (it forms no neighbourhood, so `r/R` never enters its error) rather than computational. Ambient `D` was measured to cost ~9× wall-clock and *nothing* in accuracy.

**But the cost worry surfaced a real, uncovered inefficiency:** `chart_mean_curvature` uses `vmap(jacrev)` and `torch.func.hessian` (= `jacfwd∘jacrev`). On a `d ≪ D` map that costs ~`D` reverse passes where forward mode costs ~`d`. Invisible at the roll's `d=2, D=3`; ~768 vs ~20 at PU's `d=20, D=768`, and ~15,360 vs ~400 for the Hessian.

### Q — How to handle it?

| Option | Description | Selected |
|--------|-------------|----------|
| Measure first, then decide | Time the exact path at PU scale before choosing | |
| Switch to forward-mode, keep exact | Same mathematics, ~38× operation-count ceiling | |
| Promote the randomized estimator | Contradicts the note's demotion; ~2.5× and injects Monte-Carlo variance | |
| Subsample the points instead | Linear cost reduction; Phase 4 inherits a sparser field | |
| **Other (user-supplied)** | **Add forward mode as a toggle so reverse is easy to fall back to** | ✓ |

**User's choice:** free text — "Let's add the ability to do forward mode then. Just make it a toggle, so if it doesn't work we can go back to reverse easily."

**Notes:** The user asked *"Why reverse mode? What does that mean?"* first. Explained forward vs reverse as which side you multiply the Jacobian on — reverse computes `vᵀJ` (a row per pass, `~D` passes), forward computes `Jv` (a column per pass, `~d` passes); reverse is right for scalar-loss ML and is the reflex, the decoder is the opposite regime. Stated the honest caveat: those are operation counts, not wall-clock, PyTorch's forward path is less optimized, and forward-mode `vmap` may hit an unimplemented batching rule and not run at all — which is the argument for a toggle rather than a switch.

### Follow-ups

| Question | Options | Selected |
|---|---|---|
| Default mode | Reverse (opt in to forward) / Forward (opt out) / Forward with auto-fallback | **Reverse — opt in to forward** |
| Toggle scope | `chart_curvature.py` only / also `decoder_curvature.py` / also `derivative_bridge.py` | **`chart_curvature.py` only** |

---

## PU fit: which one, what d

### Q1 — Which fit does the field come from?

| Option | Description | Selected |
|--------|-------------|----------|
| Fresh sweep, sealed fit as one row | Sealed 02.2 fit read-only as a labelled `n_charts=16` row; keeps provenance continuity at no extra training | |
| Fresh sweep only, ignore sealed fit | One clean protocol, nothing to reconcile; discards the only pre-registered PU fit | ✓ |
| Sealed fit primary, sweep secondary | Maximum provenance; puts the field on the 16-chart config | |

**User's choice:** Fresh sweep only.

**Notes:** Grounded on the sealed 02.2 configuration read from `02.2-FINDINGS.md` §2 — `D_CHART=20`, `L_EMBED=40`, `N_CHARTS_INIT=16` with all 16 surviving pruning, SiLU, width 250, 8,000/2,000 split at seed 20260803. Flagged that 16 charts sits at the worst end of the only axis measured to move ρ.

---

### Q2 — What happens to chart_dim?

| Option | Description | Selected |
|--------|-------------|----------|
| Hold at 20, justified and restated | Keep 02.2's `D_CHART=20`, restate the justification, reject `d_frozen=5` explicitly | |
| Sweep d alongside n_charts | Two axes; multiplies fits at D=768 | |
| Hold at 20, one d-robustness check | Sweep `n_charts` at d=20, refit the winner at one alternative d | |
| **Other (user-supplied)** | **Try d = 20; if it's not good, then sweep** | ✓ |

**User's choice:** free text — "try d = 20, if its not good then sweep."

---

### Q3 — What triggers the escalation?

| Option | Description | Selected |
|--------|-------------|----------|
| Blocking checkpoint, you judge | Print diagnostics, stop for a human read, same shape as 02.5-07/09's GO/NO-GO; no invented threshold | |
| Lose to the plain-AE control | Escalate if the best d=20 config fails to beat a matched plain AE on held-out reconstruction and PH H0/H1 | ✓ |
| Widespread non-immersion | Escalate on a stated fraction of near-singular `cond(g)`; needs a fraction and cutoff with no precedent | |

**User's choice:** Lose to the plain-AE control.

**Notes:** The stated caveat on this option — reconstruction and topology do not predict curvature quality (02.5-09's D-09 finding; the `y = 0.7ax²` illustration where curvature attenuates 30% with no reconstruction signal) — was presented in the option text and the user selected it regardless. Recorded in CONTEXT.md as a known limitation rather than re-opened. Also noted after the fact: the CAE already loses to a plain AE on H0 retention, so the trigger should be expected to fire.

---

## Verification depth & carried debt

### Q1 — Does Phase 3 use derivative_bridge?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, and close WR-01/02/03 first | Forward-vs-reverse compares two autodiff paths and cannot catch a bug shared by both; finite differences can | ✓ |
| No — equivalence test is enough | Cheapest; no independent check on the autodiff | |
| Yes, but defer the WR fixes | Bridge as context only; WR-02's >100% relative errors make the table hard to read | |

**User's choice:** Yes, and close WR-01/02/03 first.

---

### Q2 — How much gate machinery?

| Option | Description | Selected |
|--------|-------------|----------|
| Roll floor only, no pre-registration | The 0.65 floor written into the plan before running; no PREREGISTRATION.md, no ratification, no ancestry proof, no verdict JSON | ✓ |
| Pre-register the PU selection rule too | Ratify the four-part rule and the escalation trigger with ancestry proof | |
| Full pre-registration, milestone-standard | Treat Phase 3 like every gated 02.x phase | |

**User's choice:** Roll floor only.

---

### Q3 — PU compute budget?

| Option | Description | Selected |
|--------|-------------|----------|
| Narrow sweep, 3 seeds | 3 × 3 = 9 fits, ~3–5 h; widen later if the trend is unclear | ✓ |
| Full sweep, 5 seeds | 4 × 5 = 20 fits, ~7–11 h, user-launched like 02.7's grid | |
| Timing probe first, then decide | Measure one fit plus one curvature field end to end, then size the sweep | |

**User's choice:** Narrow sweep, 3 seeds.

**Notes:** Anchored on 02.2's measured 1,941.2 s per PU fit — training only, predating the forward-mode toggle, and curvature at `d=20, D=768` has never been timed at all.

---

## Claude's Discretion

- Exact `n_charts` values in each sweep (roll spans the measured monotone range; PU picks 3 under the D-13 budget).
- Whether to run a timing probe before committing the PU sweep.
- Deliverable shape — runner script for the expensive PU grid plus a presentation notebook, per the milestone pattern; the new `notebooks/03_swiss_roll_*_check.ipynb` is mandated by CLAUDE.md and is additive.
- How the four D-07 diagnostics combine into one selection (weighted, lexicographic, or printed table plus a stated rule).

## Deferred Ideas

- PH **H2** agreement as a selection term — excluded until a power analysis exists.
- Resuming Phases 02.3 / 02.5 / 02.7 — still on hold; D-05 opens `n_charts` to Phase 3 and nothing else.
- The 5 open windows in `.planning/WINDOWS.md`; `VERIFICATION.md` missing on 02, 02.1, 02.6.
- 02.7's two Swiss roll defects, unresolved at the hold. Does not block Phase 3.
- Two gray areas offered at the close and declined: the exact PU deliverable shape, and how the step-4 synthetic control is built given it provably cannot detect parameterization damage. The second is the more consequential and lands on the planner.
