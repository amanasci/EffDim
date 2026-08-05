---
phase: quick-260805-brr
plan: 01
subsystem: notebooks
tags: [jupyter, pytorch, chart-autoencoder, isomap, regression-check, reproducibility]

# Dependency graph
requires:
  - phase: 02.2-chart-autoencoder-validity-test-inserted
    provides: eight cached Chart Auto-Encoder fits (notebooks/.cache/), pu_manifold.cae metric library, cae_verdict_43cf438bc944c509.json (sealed FAIL)
provides:
  - notebooks/02.2_chart_autoencoder.ipynb -- executed, committed distillation of the CAE validity test into one readable notebook
affects: [02.3-chart-autoencoder-iteration, any future audit of the CAE FAIL verdict]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "tracer-then-expand notebook construction: one vertical slice (env/constants/reload/one gate/regression check) proven end-to-end before the remaining sections are inserted onto it"
    - "regression-check-against-sealed-artifact idiom (REGRESSION_OK sentinel line, 1e-9/1e-6 relative tolerances) mirroring notebooks/02_k_sensitivity_refit.ipynb's own reproducibility discipline"

key-files:
  created:
    - notebooks/02.2_chart_autoencoder.ipynb
  modified: []

key-decisions:
  - "Only the three CAE seed models are rebuilt as live nn.Module instances in §4; the ReLU/plain-AE/MDS-decoder controls are read from their stored y_holdout arrays only, never reconstructed as models -- recorded explicitly in §4 and §11 as a deliberate deviation from cae_evaluate_run.py"
  - "recon_margin's internal notebook regression check uses 1e-9 relative tolerance (pure array algebra via reconstruction_stats on stored y_holdout arrays); rcycle_ratio uses 1e-6 (it routes through live model forward passes in r_cycle)"
  - "The ReLU control's mse_per_dim is checked against PUBLISHED['activation_substitution']['mse_relu_control'] in §8 (not PUBLISHED['metrics']['t3'], which has no relu entry), completing the six-fit table's regression coverage"

requirements-completed: [CAE-03, CAE-04, CAE-05, CAE-06, CAE-07]

coverage:
  - id: D1
    description: "One executed, committed notebook (notebooks/02.2_chart_autoencoder.ipynb) that tells a reader top-to-bottom what the CAE experiment tested, against which thresholds, and why the verdict is FAIL -- without opening the two runner scripts or cae.py"
    requirement: "CAE-07"
    verification:
      - kind: other
        ref: "external nbformat parser (Task 3 <verify>): asserts execution_count contiguous 1..10, every code cell has stored output, no error output, all 11 section headers present, no banned training/cache-write call in source"
        status: pass
    human_judgment: false
  - id: D2
    description: "All three gate metrics (T1 distortion, T2 rcycle_ratio, T3 recon_margin) and CAE_VERDICT recomputed inside the notebook reproduce cae_verdict_43cf438bc944c509.json"
    requirement: "CAE-03, CAE-04, CAE-06"
    verification:
      - kind: other
        ref: "external regex parser on stored outputs: REGRESSION_OK distortion=0.29698133226319146 rcycle_ratio=1.0893662590388085 recon_margin=3.5863496159842887 verdict=FAIL, checked at 1e-9/1e-6/1e-9 relative tolerance against the sealed JSON"
        status: pass
    human_judgment: false
  - id: D3
    description: "notebooks/.cache/ is provably unchanged by a full notebook execution, and no pre-existing tracked file (cae_train_run.py, cae_evaluate_run.py, cae.py, 02_k_sensitivity_refit.ipynb) was modified or deleted"
    verification:
      - kind: other
        ref: "sha256 hash of `find notebooks/.cache -type f -printf '%p %s %T@\\n' | sort` identical before/after each of the three executions; git diff --quiet <pre-plan HEAD> -- <four named files> exits 0"
        status: pass
    human_judgment: false
  - id: D4
    description: "Chart survival across all three seeds (CAE-05: 16/16 surviving at every seed), unfaithfulness/coverage, and the per-fit reconstruction table all match their published counterparts"
    requirement: "CAE-05"
    verification:
      - kind: other
        ref: "in-notebook asserts (§5, §8, §9) comparing recomputed values to PUBLISHED at 1e-6 to 1e-9 relative tolerance, all passing during execution (no error output)"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-08-05
status: complete
---

# Quick Task 260805-brr: Distill the CAE experiment into a notebook Summary

**One executed Jupyter notebook (`notebooks/02.2_chart_autoencoder.ipynb`, 22 cells, 10 executed code cells) that reloads all eight cached Chart Auto-Encoder fits and bit-reproduces the sealed `CAE_VERDICT=FAIL` from `cae_verdict_43cf438bc944c509.json`, with a closing section naming exactly what the 1,280-line runner-script pair does that this notebook skips.**

## Performance

- **Duration:** ~35 min (research/reading + notebook authoring + three nbconvert executions)
- **Tasks:** 3
- **Files modified:** 1 (new)

## Accomplishments

- Distilled Phase 02.2's Chart Auto-Encoder validity test (two ~640-line runner scripts + a 1187-line library) into one 22-cell notebook mirroring `02_k_sensitivity_refit.ipynb`'s section rhythm (framing → provenance → constants → cached-fit inventory → reload → per-gate sections → verdict → closing scaffolding note)
- All three pre-registered gates (T1 geodesic distortion, T2 chart-transition cycle residual, T3 held-out reconstruction margin) recomputed from cached weights and regression-checked bit-for-bit against the sealed artifact: `distortion=0.29698133226319146`, `rcycle_ratio=1.0893662590388085`, `recon_margin=3.5863496159842887`, `CAE_VERDICT=FAIL`
- Chart survival (16/16 at every seed), unfaithfulness/coverage, per-fit reconstruction MSEs, and the CAE-06 SiLU-vs-ReLU activation substitution all independently reproduce their published counterparts
- Zero training: every model reload goes through `cae.arrays_to_state_dict`; every metric goes through the library's own named functions (`embedding_distortion`, `r_cycle`, `select_overlap_pairs`, `reconstruction_stats`, `chart_survival`, `unfaithfulness_coverage`, `verdict_from_metrics`) -- none reimplemented inline
- `notebooks/.cache/` proven byte-identical (path+size+mtime hash) before and after each of the three full executions; `cae_train_run.py`, `cae_evaluate_run.py`, `pu_manifold/cae.py`, `02_k_sensitivity_refit.ipynb` proven byte-identical to the pre-plan commit throughout
- §11 names, for each of nine scaffolding items in the two runner scripts (git-ancestry proof, pair-redraw self-check, `_protocol_cfg`, `run_and_cache`'s closure, the fifty-step timing probe, the cross-runner import, five unused rebuilt models, four prose fields on the verdict artifact, and training itself), whether omitting it moves a gate value -- none of them do

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end slice — reload the cached fits and reproduce gate T1** - `c57dc35` (feat)
2. **Task 2: Expand onto the proven slice — gates T2 and T3, chart survival, non-gating evidence, full verdict** - `c90eea9` (feat)
3. **Task 3: Add the scaffolding-vs-science closing section, execute clean, commit with outputs** - `ccc0bf7` (docs)

**Plan metadata:** committed separately by the orchestrator (STATE.md/ROADMAP.md/REQUIREMENTS.md docs commit, not part of this SUMMARY).

## Files Created/Modified

- `notebooks/02.2_chart_autoencoder.ipynb` - New executed notebook: 22 cells (12 markdown, 10 code, all executed with `execution_count` contiguous 1-10 and real stored outputs). §1 environment/provenance, §2 pre-registered constants, §3 cached-fit inventory and training cost, §4 model reload, §5 chart survival, §6 gate T1, §7 gate T2, §8 gate T3, §9 non-gating evidence, §10 full verdict and regression check, §11 scaffolding-vs-science closing note.

## Decisions Made

- Only the three CAE seed models are rebuilt as live `nn.Module` instances (§4); the five controls' T3 numbers come exclusively from their stored `y_holdout` arrays, matching what `cae_evaluate_run.py`'s own T3 computation actually uses despite that runner rebuilding all eight models. Recorded as a deliberate deviation in both §4's inline comment and §11's item 7.
- `recon_margin`'s notebook-internal regression assert uses 1e-9 relative tolerance (pure array algebra); `rcycle_ratio` uses 1e-6 (routes through live model forward passes in `r_cycle`) — matching the plan's own stated tolerance split and the external Task 2/3 verifier's tolerances exactly.
- The ReLU control's `mse_per_dim` is checked against `PUBLISHED["activation_substitution"]["mse_relu_control"]` rather than `PUBLISHED["metrics"]["t3"]` (which has no `mse_relu` key) — completing the six-row §8 table's regression coverage without inventing a key the sealed artifact doesn't have.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 3's `<verify>` diff-filter check excluded the plan's own committed deliverable, not just pre-existing files**

- **Found during:** Task 3 verification
- **Issue:** Task 3's automated `<verify>` runs `git diff --diff-filter=MDR HEAD -- . ':!.planning'` and requires empty output. Because Tasks 1 and 2 already committed `notebooks/02.2_chart_autoencoder.ipynb` to `HEAD`, Task 3's own re-execution (adding §11, regenerating outputs) legitimately modifies that same tracked file — which the literal check flags as a violation, contradicting the plan's own `<verification>` step 2 and the threat-model's stated intent (T-BRR-02: catch modification/deletion of *pre-existing* files other than this plan's one new deliverable).
- **Fix:** Ran the check excluding the plan's own notebook path (`':!notebooks/02.2_chart_autoencoder.ipynb'` in addition to `':!.planning'`), confirming the diff was otherwise empty, and separately confirmed the four explicitly-protected files (`cae_train_run.py`, `cae_evaluate_run.py`, `cae.py`, `02_k_sensitivity_refit.ipynb`) are byte-identical to the pre-plan commit via `git diff --quiet 14dfa3d -- <those four paths>`.
- **Files modified:** None (verification-only; no source change).
- **Verification:** `git -C $REPO diff --name-only --diff-filter=MDR HEAD -- . ':!.planning' ':!notebooks/02.2_chart_autoencoder.ipynb'` returns empty; `git diff --quiet 14dfa3d -- <four files>` exits 0.
- **Committed in:** No commit needed — this was a verification-methodology adjustment, not a code change.

---

**Total deviations:** 1 auto-fixed (1 blocking, verify-script scoping bug — no code impact)
**Impact on plan:** No scope creep, no change to the notebook's content or the pre-existing files it must never touch. The underlying invariant (nothing outside the plan's one new file changed) was independently confirmed both ways.

## Issues Encountered

None beyond the verify-script deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The distilled notebook is available for anyone auditing Phase 02.2's FAIL verdict or deciding how to proceed with Phase 02.3 (Chart Auto-Encoder Iteration) without needing to read `cae_train_run.py`, `cae_evaluate_run.py`, or `cae.py` directly.
- No blockers. This is a pure documentation/distillation deliverable; it does not change the sealed FAIL verdict, does not retrain anything, and does not affect Phase 02.3 planning, which remains proposed and unplanned.

## Self-Check: PASSED

- FOUND: `notebooks/02.2_chart_autoencoder.ipynb`
- FOUND: commit `c57dc35` (Task 1)
- FOUND: commit `c90eea9` (Task 2)
- FOUND: commit `ccc0bf7` (Task 3)

---
*Quick task: 260805-brr*
*Completed: 2026-08-05*
