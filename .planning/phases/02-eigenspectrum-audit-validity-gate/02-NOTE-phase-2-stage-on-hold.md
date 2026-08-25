# Note — the phase-2 stage is ON HOLD; architecture selection tabled, CAE carried into Phase 3

**Status:** user decision, 2026-08-12. **Binding on routing, not on any sealed verdict.** No
verdict, finding, or measured number recorded anywhere in the 02.x family is reopened,
softened, recomputed, or reinterpreted by this note. What changes is only *what gets worked
on next*.

## 1. The decision

The milestone has spent phases 02.1 through 02.7 trying to select a representation/decoder
substrate and has not produced a PASS. The user's judgment is that further selection work has
stopped paying for itself. Therefore:

- **Architecture selection is tabled.** Phases 02.3, 02.5, 02.6 and 02.7 stop where they
  stand. No further substrate screening, re-gating, or front-end benchmarking is scheduled.
- **The Chart Auto-Encoder (CAE) is the substrate carried into Phase 3.**
- **Phase 3 proceeds** and is the milestone's active work.

## 2. What the CAE choice rests on — and what it does not

It rests on a **defect ledger and readiness** comparison, made under an explicit
no-more-selection-work constraint. It does **not** rest on measured superiority, and no
measurement in this milestone licenses that claim.

**For:**

- 02.2 delivered a CAE Swiss roll check (`notebooks/02.2_swiss_roll_cae_check.ipynb`) that
  passes, and the CAE implementation carries **no open correctness window**.
- TopoAE carries window **#3** — `train_topoae`'s reconstruction term sums over ambient
  features where the reference means over them, reparameterizing `LAMBDA_TOPO` by a factor of
  `D`. Closing it changes every sealed fit's training objective and needs a fresh
  pre-registration plus a full sixteen-fit re-run. Unfixed, and expensive to fix.
- Plain AE was never a candidate — it exists in this milestone as a matched baseline.

**Against — recorded here so the choice is not read as evidence-backed:**

- `CAE_VERDICT = FAIL` (02.2, sealed 2026-08-04): T1 geodesic distortion `0.296981`
  (threshold `<0.15`), T3 held-out reconstruction margin `3.586350` (threshold `<0.90`). T2
  chart-transition cycle residual passed (`1.089366 < 2.0`). Permanent history.
- **02.5-09 measured the CAE chart decoder failing a curvature Swiss roll check** — curvature
  Spearman `-0.0604` against the raw-point baseline's `0.6712`, at 8/8 charts. This is the
  single most directly adverse result for the specific use Phase 3 puts the CAE to.
- Quick task `20260809-topoae-vs-cae-persistence`: the CAE's 40-D embedding preserves H0
  merge structure ~5x worse than a plain AE at the same dimension (`cae_retained=0.183246`),
  using ~8x the parameters.
- Quick task `20260809-cae-undertraining-test`: across 24 Swiss roll fits, **no arm reaches
  the raw-point curvature baseline `0.6712`**; and `chart_survival` returns 8/8 in every fit
  while argmax occupancy falls to 6/8 — the paper's pruning criterion thresholds a *ratio*
  and provably cannot detect a dead chart.
- Phase 02.6 completed 15/15 replan plans and **promoted no substrate and eliminated none**;
  its ranking axis carries two named confounds (`02.6-FINDINGS-02.md`).

**Read this as:** the CAE is the substrate Phase 3 will use, chosen for readiness under a
tabling decision. If Phase 3's curvature field fails, 02.5-09's `-0.0604` is the first place
to look, and that failure would be an expected outcome rather than a surprise.

## 3. The hard gate is being overridden, deliberately

ROADMAP.md Phase 3 states: *"Depends on: Phase 02.4 **PASS** (the operative precondition —
Phase 3 must check its verdict artifact before running any expensive cell; until 02.4 exists
and passes, Phase 3 stays blocked)."* Phase 2 states the same shape of gate; so does 02.2.

Every sealed verdict in this milestone is FAIL:

| Phase | Verdict | Scope |
|---|---|---|
| 02 | `GATE_VERDICT = FAIL` (`m=0.412071`) | global |
| 02.2 | `CAE_VERDICT = FAIL` | global (T1, T3); local T2 passed |
| 02.4 | `TOPOAE_VERDICT = FAIL` | global (T1, T2); local T3 passed |
| 02.5 stage 1 | `CURVATURE_VERDICT = FAIL` | local, marginal + seed-sensitive |

**No PASS exists. Phase 3 therefore starts with its precondition unmet, by explicit user
decision.** This is an override, not a satisfied gate, and Phase 3's plan must say so in its
own record rather than inheriting a silent green light.

The consequence the gate existed to prevent, restated so it is carried rather than lost: a
curvature field decoded from an unvalidated parameterization **conflates real curvature with
parameterization damage**, and CURV-06/07's synthetic control provably cannot detect that — a
synthetic manifold that passes the Phase 2 gate never reproduces the pathology. Phase 3 will
produce numbers; this note is the reason those numbers cannot be read as validated curvature
without an additional argument Phase 3 must supply itself.

Partial mitigation already on record (`02.4-FINDINGS.md`): **every FAIL in this milestone is
global-scoped**, and no local-scoped gate has ever failed here (02.2 T2, 02.4 T3 both passed).
Mean curvature is a local invariant. That supports "not globally coordinatizable," not "no
usable structure" — it is the strongest available argument for proceeding, and it is an
argument, not a verdict.

## 4. What is on hold, and exactly where it stopped

| Phase | Stopped at | Open |
|---|---|---|
| 02 | sealed FAIL, 3/3 plans | `VERIFICATION.md` missing |
| 02.1 | sealed, 4/4 plans | `VERIFICATION.md` missing |
| 02.2 | sealed FAIL, 6/6 plans | — |
| 02.3 | never planned | proposed only; unretracted, available fallback |
| 02.4 | sealed FAIL, 8/8 plans | window #3 (λ×D fidelity gap) |
| 02.5 | **9/13 plans** | `02.5-09` Task 3 blocking checkpoint **OPEN** (human must judge the chart-decoder curvature read-out); plans `02.5-10`..`13` unstarted |
| 02.6 | 15/15 replan plans | `VERIFICATION.md` missing; no substrate promoted or eliminated |
| 02.7 | **10/12 plans** | `02.7-10` Tasks 2/3 unrun (the ~17h benchmark grid); `02.7-11`, `02.7-12` unstarted |

**Carried debt that does not disappear with the hold:**

- Three code-review warnings in `derivative_bridge.py` (`02.6-REVIEW.md`, commit `1d3f666`),
  **WR-01/02/03**. None affects any number Phase 02.6 recorded. They were routed to `02.5-10`,
  which is now on hold — so they land on **whoever next relies on the bridge for
  thresholding**, which may well be Phase 3.
  - WR-01 — `finite_difference_jacobian` / `finite_difference_hessian` / `calibrate_fd_step`
    pass a bound method to `chart_curvature._assert_float64`, which expects the model; the
    per-parameter float64 guard is silently skipped. Masked only because every current call
    site pre-casts with `.double()`.
  - WR-02 — relative-error columns exceed 100% against near-zero references
    (`full_hess_max_abs_rel = 1.1351e+00` already visible in the recorded PU table).
  - WR-03 — `calibrate_fd_step` computes its autodiff Hessian unchunked.
- **5 open windows** in `.planning/WINDOWS.md` (`/gsd-ship` blocks while any remain): #1
  (01, uneven `STAGE2_K` spacing), #3 (02.4, λ×D), #4 (02.6, batch-split exactness at real
  architecture scale), #5 (02.7, SC-5's abstain path (c) never exercised), #6 (02.7-10 Tasks
  2/3 unexecuted).
- **02.7's Swiss roll check does not pass as written.** `notebooks/02.7_swiss_roll_template_check.ipynb`
  prints 1 of 4 read-out lines true: the roll passes only by abstain, both in-library positive
  controls (`S1`, `T2`) fail to receive their labels, and abstain condition (c) never fires.
  Two independent defects: GMST local-dispersion instability (range `22.2` on the roll against
  the ratified bound `3.0`) fires condition (b) on everything first, and the banded β₀ is
  inflated (roll `(26,1,0)`/`(29,1,1)`, T2 `(22,2,0)`/`(53,3,1)` where truth needs β₀=1),
  which would have blocked correct Betti lookup independently. The geodesic k-sweep reports
  `n_components(k=15)=1` for all three clouds, contradicting the banded H0 count on the same
  clouds. **Unresolved at the time of the hold.**

## 5. What ends the hold

Nothing automatic. The hold ends only by an explicit user decision to resume a named phase.
The likeliest triggers:

1. **Phase 3's curvature field fails**, and 02.5-09's `-0.0604` is implicated — resume 02.5
   stage 2, or 02.3 (CAE iteration), against the failure's actual shape.
2. **Phase 3 needs the derivative bridge for thresholding** — close WR-01/02/03 first, inside
   Phase 3's own plan, since 02.5-10 is no longer coming.
3. **Shipping the milestone** — `/gsd-ship` blocks on 5 open windows; each needs closing or an
   explicit waiver with a recorded reason.
4. **The template front end is wanted for anything downstream** — 02.7's two Swiss roll defects
   must be resolved before the ~17h grid is worth running, not after.

## 6. Cross-references

- `02.6-FINDINGS-02.md` — the ranking, the 192-number matrix, the two named confounds, the
  D-15 separating-experiment PASS branch.
- `02.5-NOTE-substrate-selection.md` §4 — Swiss roll topology does not transfer to PU; nothing
  measured on the roll bounds PU's actual template, dimension, or curvature.
- `02.4-FINDINGS.md` — the every-FAIL-is-global-scoped finding this hold's Phase 3 argument
  leans on.
- `02.2-FINDINGS.md` — the sealed CAE FAIL the chosen substrate carries.
