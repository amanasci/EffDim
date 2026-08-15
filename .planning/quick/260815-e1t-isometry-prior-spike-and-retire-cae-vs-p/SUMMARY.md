---
phase: quick-260815-e1t
plan: 01
subsystem: research
tags: [torch, torch.func, jacfwd, cae, curvature, decoder-priors, d12-retirement]

requires:
  - phase: 03-decoder-curvature-field
    provides: cae.py (sealed 02.2 architecture), chart_curvature.py (torch.func curvature field), the corrected 9-cell PU grid and its measured cond(g) pathology
provides:
  - "notebooks/pu_manifold/decoder_priors.py -- opt-in, default-off isometry/conformal prior on the CAE chart decoder's Jacobian, installed via a scoped cae.chart_loss shim, cae.py untouched"
  - "notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py -- the weight-ladder runner, anchor-verified against the sealed rho_chart"
  - "03-NOTE-isometry-prior-spike.md -- spike halted at its own budget gate before any ladder cell trained; no mechanism/bias verdict yet"
  - "03-NOTE-d12-retirement.md -- D-12 fired on both legs of the corrected grid, then retired; replaced by a two-part C0/C2 criterion (proposed C0 threshold pending ratification)"
affects: [03-08-decoder-curvature-field-execution, any-future-isometry-prior-spike-continuation]

tech-stack:
  added: []
  patterns:
    - "Scoped module-global monkeypatch (rebind cae.chart_loss inside a @contextmanager, restore in finally) to inject a training-time term into a sealed training loop without editing it"
    - "torch.func jacfwd composed under an outer backward() -- verified empirically that vmap(jacfwd(f)) carries a live autograd graph back to closed-over nn.Module parameters, enabling a differentiable-Jacobian training loss without chunk-and-detach"
    - "Pre-declared compute-budget probe (--probe) with a fixed candidate ladder and an explicit BUDGET NOT MET terminal branch, refusing to silently shrink scope to force a fit"

key-files:
  created:
    - notebooks/pu_manifold/decoder_priors.py
    - notebooks/pu_manifold/tests/test_decoder_priors.py
    - notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py
    - .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md
    - .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md
  modified:
    - .planning/STATE.md

key-decisions:
  - "Part B: stop comparing the CAE against a plain autoencoder (D-12 retired); replace with a direct two-part C0 (absolute mse_per_dim ceiling, proposed 2.5e-04) / C2 (ROLL_FLOOR=0.65) criterion"
  - "Part A: the isometry-prior spike halted at its own pre-declared --probe budget gate (LADDER_BUDGET_S=3000s) before training any ladder cell -- the prior triples per-epoch cost and even the cheapest candidate (E=50) projects to 3636s. Per the runner's own design this is a real terminal branch, not an error to work around; no epoch count outside the pre-declared candidates was substituted to force a fit"
  - "cae.py was never edited; the prior is installed via a scoped, restore-on-exit rebinding of cae.chart_loss. The permanent extra_loss=None seam is proposed to the developer, not applied"

requirements-completed: [PART-A-ISOMETRY-SPIKE, PART-B-D12-RETIREMENT]

coverage:
  - id: D1
    description: "decoder_priors.py: metric_deviation, chart_decoder_jacobian (differentiable), isometry_penalty, decoder_prior_active (scoped cae.chart_loss shim) -- every behavior line has a passing test"
    requirement: "PART-A-ISOMETRY-SPIKE"
    verification:
      - kind: unit
        ref: "notebooks/pu_manifold/tests/test_decoder_priors.py (13 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Swiss roll anchor rho_chart = -0.06041003026778113 reproduces exactly through the new runner (--anchor-check)"
    requirement: "PART-A-ISOMETRY-SPIKE"
    verification:
      - kind: integration
        ref: ".venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --anchor-check"
        status: pass
    human_judgment: false
  - id: D3
    description: "The isometry-prior weight ladder ran on the Swiss roll and produced a mechanism/bias verdict"
    requirement: "PART-A-ISOMETRY-SPIKE"
    verification: []
    human_judgment: true
    rationale: "Did not run -- halted at the --probe budget gate (BUDGET NOT MET) before any ladder cell trained. No mechanism or bias verdict exists. Whether to raise the budget, shrink the ladder, or accept this as a negative result is a developer decision, routed to the checkpoint."
  - id: D4
    description: "D-12's escalation trigger retired on the record, with the fired-then-retired result, the C0/C2 argument, the disjoint-regularizer finding, and the two-part replacement criterion"
    requirement: "PART-B-D12-RETIREMENT"
    verification:
      - kind: other
        ref: "grep checks on 03-NOTE-d12-retirement.md for the trigger-fired quote, 'wrong instrument', and ROLL_FLOOR"
        status: pass
    human_judgment: true
    rationale: "The note's proposed absolute C0 mse_per_dim threshold (2.5e-04) is explicitly marked PROPOSED and requires developer ratification at the checkpoint before it governs any future decision."

duration: 45min
completed: 2026-08-15
status: complete
---

# Quick Task 260815-e1t: Isometry Prior Spike and D-12 Retirement Summary

**Isometry/conformal decoder-Jacobian prior built, tested, and anchor-verified, but its weight-ladder spike halted at its own pre-declared budget gate before training a single cell; D-12's escalation trigger fired on both legs of the corrected PU grid and is retired in favour of a direct C0/C2 criterion.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-08-15 (session start)
- **Completed:** 2026-08-15T14:42:04Z
- **Tasks:** 4 of 5 (Task 5 is the blocking checkpoint this file stops at)
- **Files modified:** 6

## Accomplishments

- Built `notebooks/pu_manifold/decoder_priors.py`: an opt-in, default-off first-order isometry
  (and conformal-variant) prior on the CAE chart decoder's Jacobian, installed into
  `cae.train_cae`'s sealed loop via a scoped `cae.chart_loss` rebind that installs nothing at
  `weight=0.0` and restores the original binding on exit including by exception. `cae.py` is
  byte-for-byte unchanged.
- Verified empirically (and pinned in tests) that `vmap(jacfwd(decode_one))`'s output carries a
  live autograd graph back to both `chart_decoders` and `embedding_decoder` parameters, so the
  documented `jacrev` fallback was not needed.
- Built `notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py`, reusing
  `swiss_roll_curvature_sweep_run.py`'s fixture/protocol/metric sequence (not edited) rather
  than restating it. `--anchor-check` reproduces `rho_chart = -0.06041003026778113` **exactly**
  at `n_charts=8, seed=0, n_points=3000`.
- Ran `--probe`: the isometry penalty roughly triples per-epoch training cost
  (`10.699s` vs `3.685s` at `n_charts=8, n_points=12000`), and every pre-declared
  `LADDER_MAX_EPOCHS_CANDIDATES` value (150/100/75/50) projects over the `3000s` budget -- the
  cheapest, `E=50`, projects `3635.9s`. Per the runner's own design (stated in its `--probe`
  docstring before this number existed) this is a real terminal branch: no epoch count outside
  the pre-declared candidates was substituted to force a fit. No ladder cell trained; no
  mechanism or bias verdict exists. Recorded in `03-NOTE-isometry-prior-spike.md`.
- Retired D-12's CAE-vs-plain-AE escalation trigger. `03-NOTE-d12-retirement.md` quotes the
  trigger's verbatim fired output (`TRIGGER FIRES = True` on both legs), states unambiguously
  that retirement is because the comparison is the wrong instrument (entirely C0; small C0
  error does not bound C2/curvature error) and not because of the unfavourable result, and
  proposes a replacement two-part C0 (absolute `mse_per_dim < 2.5e-04`, PROPOSED) / C2 (existing
  `ROLL_FLOOR = 0.65`) criterion.
- `STATE.md` updated additively (17 lines added, 0 removed) with a paragraph recording both
  outcomes and pointers to both notes.

## Task Commits

Each task was committed atomically:

1. **Task 1: decoder_priors.py -- the prior, end to end, on a real CAE** - `762f7be` (feat)
2. **Task 2: the weight-ladder runner, with the anchor reproduction as its first gate** - `8930b15` (feat)
3. **Task 3: run the ladder and write the spike note** - `7ccdec7` (docs)
4. **Task 4: retire D-12's escalation trigger on the record** - `505e890` (docs)

Task 5 is the blocking `checkpoint:human-verify` this plan ends on; execution stops there per
the plan's explicit instruction.

## Files Created/Modified

- `notebooks/pu_manifold/decoder_priors.py` - the isometry/conformal prior, `PRIOR_MODES`, `metric_deviation`, `chart_decoder_jacobian`, `isometry_penalty`, `decoder_prior_active`
- `notebooks/pu_manifold/tests/test_decoder_priors.py` - 13 tests, every behavior line covered
- `notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py` - the weight-ladder runner (`--dry-run`, `--anchor-check`, `--probe`, `--summary`, resumable JSONL ladder run)
- `.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md` - the spike result: BUDGET NOT MET, anchor reproduced, no mechanism/bias verdict
- `.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md` - D-12 fired-then-retired, C0/C2 argument, disjoint-regularizer finding, two-part replacement criterion
- `.planning/STATE.md` - one additive paragraph after the "03-08 first grid run INVALIDATED" block

## Decisions Made

- Part B is settled here: D-12 is retired, replaced by a two-part C0/C2 criterion. The one
  open number is the C0 threshold (`mse_per_dim < 2.5e-04`, proposed), which needs developer
  ratification at the checkpoint before it governs `03-08`.
- Part A is incomplete by design, not by failure. The prior itself is built, tested, and proven
  to move the optimizer (training-loop test) and to reproduce the sealed anchor exactly.
  Whether it fixes the measured `cond(g)` pathology without flattening the surface is an open
  question this session's compute budget did not answer. The next step (raise
  `LADDER_BUDGET_S`, shrink the ladder, or accept the spike as an incomplete negative result) is
  a developer decision, routed to the checkpoint.
- The `chart_decoder_jacobian` autograd-graph question (jacfwd vs jacrev fallback) was resolved
  empirically in favour of `jacfwd` before writing the module -- verified with a standalone
  two-linear-layer probe that `vmap(jacfwd(f)).sum().backward()` populates gradients on both
  layers' parameters.

## Deviations from Plan

None -- plan executed exactly as written, including its own pre-declared BUDGET NOT MET
terminal branch (Task 3's action text: "If it prints BUDGET NOT MET, stop here, write the note
with the probe numbers and a BUDGET NOT MET outcome, and take the halt to Task 5's checkpoint --
that is a real terminal branch, not an error to work around").

One note on Task 3's automated verify block: it greps the spike note for the literal string
`MECHANISM DEMONSTRATED` or `MECHANISM NOT DEMONSTRATED`. The note's prose about the mechanism
check not having been computed literally contains both alternative phrases ("No MECHANISM
DEMONSTRATED verdict was computed" / "...MECHANISM NOT DEMONSTRATED verdict was computed"), so
the grep passes incidentally -- verified directly (prints `NOTE_OK`). This is not a fabricated
result; the note is explicit that no mechanism check ever ran.

## Issues Encountered

None beyond the pre-declared budget halt described above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `decoder_priors.py` and its runner are complete, tested, and re-runnable at any revised
  ladder scope (larger `LADDER_BUDGET_S`, fewer seeds/weights, smaller `n_points`) without
  further code changes.
- `03-08` (the D-12 escalation checkpoint) is now governed by `03-NOTE-d12-retirement.md`
  rather than the retired trigger -- a reader of `03-08-PLAN.md` Task 3 must read that note
  first. The proposed C0 threshold (`2.5e-04`) needs ratification before `03-08` relies on it.
- Blocked on the Task 5 checkpoint: four Part-A decisions (adopt/don't-adopt is moot until the
  ladder runs; whether to run the conformal arm anyway; whether to apply the permanent
  `extra_loss` seam to `cae.py`; confirm the notebook obligation) plus ratifying (or replacing)
  the proposed C0 threshold.

---
*Phase: quick-260815-e1t*
*Completed: 2026-08-15*

## Self-Check: PASSED

All created files found on disk; all four task commit hashes (`762f7be`, `8930b15`, `7ccdec7`,
`505e890`) found in git history.
