---
phase: quick-260815-e1t
plan: 01
subsystem: research
tags: [torch, torch.func, jacfwd, cae, curvature, decoder-priors, d12-retirement]

requires:
  - phase: 03-decoder-curvature-field
    provides: cae.py (sealed 02.2 architecture), chart_curvature.py (torch.func curvature field), the corrected 9-cell PU grid and its measured cond(g) pathology
provides:
  - "notebooks/pu_manifold/decoder_priors.py -- opt-in, default-off isometry/conformal prior on the CAE chart decoder's Jacobian, installed via a scoped cae.chart_loss shim, cae.py untouched -- tested, working, unmeasured"
  - "notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py -- the weight-ladder runner, anchor-verified against the sealed rho_chart"
  - "03-NOTE-isometry-prior-spike.md -- spike halted twice (runner's own budget gate, then a separate developer HALT at the checkpoint); no mechanism/bias verdict exists"
  - "03-NOTE-d12-retirement.md -- D-12 fired on both legs of the corrected grid, then retired unconditionally; C0 leg of its replacement criterion left DEFERRED, not ratified"
affects: [03-08-decoder-curvature-field-execution, any-future-isometry-prior-spike-continuation]

tech-stack:
  added: []
  patterns:
    - "Scoped module-global monkeypatch (rebind cae.chart_loss inside a @contextmanager, restore in finally) to inject a training-time term into a sealed training loop without editing it"
    - "torch.func jacfwd composed under an outer backward() -- verified empirically that vmap(jacfwd(f)) carries a live autograd graph back to closed-over nn.Module parameters, enabling a differentiable-Jacobian training loss without chunk-and-detach"
    - "Pre-declared compute-budget probe (--probe) with a fixed candidate ladder and an explicit BUDGET NOT MET terminal branch, refusing to silently shrink scope to force a fit"
    - "Two distinct halt states recorded separately in the same note: a runner's own pre-declared terminal branch vs. a human's deliberate stop, so a later reader cannot conflate 'the tool refused' with 'the developer chose to wait'"

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
  - "Part B: D-12 retired unconditionally -- the retirement rests on the CAE-vs-plain-AE comparison being the wrong instrument for a C2 question (C0-only, cannot bound curvature error), not on any threshold value. Replacement C0/C2 criterion's C2 leg (ROLL_FLOOR=0.65) stands; its C0 leg (proposed mse_per_dim < 2.5e-04) is DEFERRED, not ratified -- circular (anchored to the CAE's own measured ceiling) and premature (the isometry prior, if adopted, would change what reconstruction the CAE achieves). Plain AE's 2.2646e-05 raised as an alternative anchor, also deferred."
  - "Part A: HALTED at the checkpoint by explicit developer decision, distinct from the runner's own earlier BUDGET NOT MET halt. Neither the budget (LADDER_BUDGET_S) nor the ladder scope was changed; no ladder cell was run. The developer has additional information to provide before the spike proceeds."
  - "extra_loss=None permanent seam on cae.train_cae: confirmed NOT applied. The scoped, restore-on-exit cae.chart_loss shim (decoder_prior_active) stays as the tested mechanism; cae.py remains byte-for-byte unedited."
  - "Notebook obligation confirmed: notebooks/03_swiss_roll_isometry_prior_check.ipynb must exist and pass before any PU fit runs with a non-zero prior weight -- CLAUDE.md's standing rule, not a new decision."

requirements-completed: [PART-B-D12-RETIREMENT]

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
    rationale: "Not delivered, and not expected to be by this quick task. First halted by the runner's own --probe budget gate (BUDGET NOT MET, before any ladder cell trained); then, at the Task 5 checkpoint, the developer separately chose to HALT rather than raise the budget or shrink the ladder, pending additional information not yet provided. No rho_chart/cond(g)/median_magnitude_ratio/calibration_slope number, and no mechanism or bias verdict, exists. This deliverable remains open past this quick task's close."
  - id: D4
    description: "D-12's escalation trigger retired on the record, with the fired-then-retired result, the C0/C2 argument, the disjoint-regularizer finding, and the two-part replacement criterion"
    requirement: "PART-B-D12-RETIREMENT"
    verification:
      - kind: other
        ref: "grep checks on 03-NOTE-d12-retirement.md for the trigger-fired quote, 'wrong instrument', and ROLL_FLOOR"
        status: pass
    human_judgment: true
    rationale: "The retirement itself is unconditional and ratified at the checkpoint. Its replacement criterion's C0 leg (proposed mse_per_dim < 2.5e-04) is explicitly DEFERRED at the same checkpoint, for reasons recorded in the note (circular derivation; premature pending the isometry-prior question) -- so this deliverable is complete for the retirement and open for the C0 number."

duration: 55min
completed: 2026-08-15
status: complete
---

# Quick Task 260815-e1t: Isometry Prior Spike and D-12 Retirement Summary

**D-12's CAE-vs-plain-AE escalation trigger retired unconditionally after firing on both legs of the corrected PU grid; the isometry-prior spike delivered tested, working infrastructure (decoder_priors.py, anchor-verified runner) but zero measurement -- halted first by its own budget gate, then again by explicit developer decision at the checkpoint.**

## Performance

- **Duration:** ~55 min
- **Started:** 2026-08-15 (session start)
- **Completed:** 2026-08-15 (checkpoint resolved)
- **Tasks:** 5 of 5 (Task 5's checkpoint reached, then resolved by the developer)
- **Files modified:** 6

## Status, stated honestly

**Part B (D-12 retirement) is complete.** The trigger fired on both legs of the corrected grid,
is recorded in full, and is retired unconditionally -- the retirement does not depend on any
threshold value. One number inside its replacement criterion (the C0 `mse_per_dim` ceiling) is
explicitly left open (DEFERRED, not ratified), for reasons now recorded in the note.

**Part A (isometry prior spike) is NOT complete and did not succeed.** It delivered tested,
working infrastructure -- `decoder_priors.py` (13 passing tests, `cae.py` untouched) and a
weight-ladder runner that reproduces the sealed Swiss roll anchor exactly. **It produced no
measurement of the prior and no mechanism or bias verdict.** It was halted twice: first by the
runner's own pre-declared `--probe` budget gate (`BUDGET NOT MET`, before any cell trained),
then by a separate, deliberate developer decision at the Task 5 checkpoint to stop rather than
raise the budget or shrink the ladder. This quick task does not claim the prior works, doesn't
work, or is close to being measured -- only that the infrastructure to measure it exists and is
tested.

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
  cheapest, `E=50`, projects `3635.9s`. `BUDGET NOT MET` per the runner's own pre-declared design.
- At the Task 5 checkpoint, the developer separately decided to **HALT** the spike -- not raise
  the budget, not shrink the ladder, not run any cell -- pending additional information. Both
  halts (the runner's own budget gate, and this deliberate developer stop) are now recorded
  distinctly in `03-NOTE-isometry-prior-spike.md` so neither is mistaken for the other.
- Confirmed at the checkpoint: the `extra_loss=None` permanent seam on `cae.train_cae` stays
  **NOT applied**; the conformal branch stays **not run** (moot, no mechanism verdict exists);
  the notebook obligation (`notebooks/03_swiss_roll_isometry_prior_check.ipynb` before any
  non-zero-weight PU fit) is **confirmed**, restating CLAUDE.md's standing rule rather than
  deciding something new.
- Retired D-12's CAE-vs-plain-AE escalation trigger, **unconditionally**. `03-NOTE-d12-retirement.md`
  quotes the trigger's verbatim fired output (`TRIGGER FIRES = True` on both legs), states
  unambiguously that retirement is because the comparison is the wrong instrument (entirely C0;
  small C0 error does not bound C2/curvature error) and not because of the unfavourable result,
  and states explicitly that the retirement does not depend on the replacement criterion's open
  C0 number.
- At the checkpoint, the proposed C0 threshold (`mse_per_dim < 2.5e-04`) was **DEFERRED, not
  ratified**: it would need revisiting if the isometry prior changes what reconstruction the CAE
  achieves, and it is circular as derived (anchored to the CAE's own measured ceiling, so it
  cannot fail by construction). The plain AE's measured `2.2646e-05` was raised as an
  alternative, independent anchor and is also deferred pending the same resolution.
- `STATE.md` updated additively (paragraph after "03-08 first grid run INVALIDATED", plus a
  Quick Tasks Completed row) recording both outcomes honestly.

## Task Commits

Each task was committed atomically:

1. **Task 1: decoder_priors.py -- the prior, end to end, on a real CAE** - `762f7be` (feat)
2. **Task 2: the weight-ladder runner, with the anchor reproduction as its first gate** - `8930b15` (feat)
3. **Task 3: run the ladder and write the spike note** - `7ccdec7` (docs)
4. **Task 4: retire D-12's escalation trigger on the record** - `505e890` (docs)
5. **Task 5: checkpoint reached and resolved** - notes and this SUMMARY updated to record the
   developer's five-part checkpoint response (see commits following `84d16b2`)

## Files Created/Modified

- `notebooks/pu_manifold/decoder_priors.py` - the isometry/conformal prior, `PRIOR_MODES`, `metric_deviation`, `chart_decoder_jacobian`, `isometry_penalty`, `decoder_prior_active`
- `notebooks/pu_manifold/tests/test_decoder_priors.py` - 13 tests, every behavior line covered
- `notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py` - the weight-ladder runner (`--dry-run`, `--anchor-check`, `--probe`, `--summary`, resumable JSONL ladder run)
- `.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md` - the spike result: two distinct halts (runner's budget gate, then the developer's checkpoint HALT), anchor reproduced, no mechanism/bias verdict
- `.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md` - D-12 fired-then-retired unconditionally, C0/C2 argument, disjoint-regularizer finding, two-part replacement criterion with its C0 leg DEFERRED
- `.planning/STATE.md` - one additive paragraph after the "03-08 first grid run INVALIDATED" block, plus a Quick Tasks Completed row

## Decisions Made

- **Part B is complete and unconditional.** D-12 is retired because the comparison was the
  wrong instrument, independent of any threshold value. The C0 leg's proposed number
  (`mse_per_dim < 2.5e-04`) is explicitly DEFERRED at the checkpoint, not ratified -- flagged as
  circular (anchored to the CAE's own measured ceiling) and premature (pending the isometry
  prior's effect on reconstruction). The plain AE's `2.2646e-05` is on record as an alternative
  anchor, also deferred.
- **Part A is HALTED, twice, for two different reasons.** The runner's own `--probe` refused to
  proceed under its pre-declared budget (a mechanical, pre-committed rule). Separately, at the
  checkpoint, the developer chose not to lift that halt by raising the budget or shrinking the
  ladder, and is holding for additional information. Neither halt is a failure of the built
  infrastructure -- `decoder_priors.py` and the runner are both complete and tested.
- The `chart_decoder_jacobian` autograd-graph question (jacfwd vs jacrev fallback) was resolved
  empirically in favour of `jacfwd` before writing the module -- verified with a standalone
  two-linear-layer probe that `vmap(jacfwd(f)).sum().backward()` populates gradients on both
  layers' parameters.

## Deviations from Plan

None -- plan executed exactly as written, including its own pre-declared BUDGET NOT MET
terminal branch (Task 3's action text: "If it prints BUDGET NOT MET, stop here, write the note
with the probe numbers and a BUDGET NOT MET outcome, and take the halt to Task 5's checkpoint --
that is a real terminal branch, not an error to work around"), and including Task 5's checkpoint
resolving with an explicit developer HALT on Part A and a DEFERRED (not ratified) C0 threshold
on Part B -- both legitimate outcomes the plan's checkpoint was designed to receive.

One note on Task 3's automated verify block: it greps the spike note for the literal string
`MECHANISM DEMONSTRATED` or `MECHANISM NOT DEMONSTRATED`. The note's prose about the mechanism
check not having been computed literally contains both alternative phrases, so the grep passes
incidentally -- verified directly. This is not a fabricated result; the note is explicit that no
mechanism check ever ran.

## Issues Encountered

None beyond the two halts described above, both of which are expected, legitimate outcomes
under this plan's design, not problems requiring resolution by this executor.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `decoder_priors.py` and its runner are complete, tested, and re-runnable without further code
  changes, whenever the developer's held-back information resolves the Part A halt.
- `03-08` (the D-12 escalation checkpoint) is now governed by `03-NOTE-d12-retirement.md` rather
  than the retired trigger -- a reader of `03-08-PLAN.md` Task 3 must read that note first. Its
  C2 leg (`ROLL_FLOOR = 0.65`) is settled; its C0 leg is an **open blocker** for any decision
  that needs it -- `03-08` cannot rely on a C0 reconstruction bar until this is resolved.
- Nothing further is expected from this quick task. Both open items (the isometry-prior
  measurement and the C0 threshold) are explicitly left open, by developer decision, for
  resolution outside this task's scope.

---
*Phase: quick-260815-e1t*
*Completed: 2026-08-15*

## Self-Check: PASSED

All created files found on disk; all four task commit hashes (`762f7be`, `8930b15`, `7ccdec7`,
`505e890`) found in git history; checkpoint-resolution edits verified against the notes' own
grep checks (TRIGGER FIRES, wrong instrument, ROLL_FLOOR, MECHANISM, BUDGET NOT MET,
median_magnitude_ratio) before this file was finalized.
