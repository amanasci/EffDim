---
status: complete
phase: 03-decoder-curvature-field
created: 2026-08-14
supplements: |
  notebooks/pu_manifold/persistence_probe.py, notebooks/pu_manifold/tests/test_persistence_probe.py,
  notebooks/diagnostics/curvature_field_pu_run.py,
  .planning/phases/02.6-decoder-substrate-screening/02.6-NOTE-ph-saturation-artifact.md
trigger: |
  03-08-DEFECTS-01.md: the nine-cell PU grid completed and applied its pre-declared
  selection rule, but all three read-out axes were corrupted by instrumentation defects
  unrelated to the chart auto-encoder itself -- an unmatched control (2x the CAE's actual
  bottleneck capacity), a training protocol that early-stopped 5 of 9 cells at 7 epochs, and
  a persistent-homology normalizer whose distance scale grows as sqrt(d), saturating every
  cross-dimension comparison by construction. Developer-directed fix pass, between plans
  03-07 and 03-08. Not a numbered plan.
commits:
  - 260a614 fix(persistence-probe): add dimension-invariant median_distance prescale (defect 3)
  - 34ccc54 fix(curvature-field-pu): apply defects 1-3 from 03-08-DEFECTS-01.md
  - 66ff107 docs(02.6): note PH prescale sqrt(d) artifact's exposure in this phase
---

# Phase 3 Plan 08 -- Supplement 2: fixing the three DEFECTS-01 instrumentation defects

**Developer-directed work, sits between plans 03-07 and 03-08. Not a numbered plan.** Same
artifact convention as `03-02-AMENDMENT-01.md` and `03-07-SUPPLEMENT-01.md`: this document is
committed alongside the code it describes, states exactly what was built and why, and every
claim below carries the actual verification evidence, not just an assertion that it was
checked. Read `03-08-DEFECTS-01.md` first -- it holds the full diagnosis and measured evidence
for each defect; this document records the fix, not a re-derivation of the diagnosis.

## 1. Why

The nine-cell PU grid ran to completion and its pre-declared selection rule selected
`n_charts=16` -- but every axis it ranked on was corrupted by one of three defects, none of
which concern chart auto-encoders (`03-08-DEFECTS-01.md` §Summary). The developer raised the
objection from theory (the CAE is proven in its source paper to beat a plain autoencoder on
reconstruction, so a 3.5x loss had to be an experimental fault) and traced it to three
independent defects. This document fixes all three, then audits prior phases for defect 3 (the
only one with a plausible cross-phase footprint), per `03-08-DEFECTS-01.md`'s own cross-phase
implication section. No `n_charts` selection is made here -- the orchestrator re-runs the grid
after this fix lands.

## 2. Fix 1 -- the D-12 control now matches the CAE's actual bottleneck

**The defect.** `_run_control_cell` built `cae.PlainAutoEncoder(768, PU_EMBED_DIM=40, ...)`.
`PlainAutoEncoder`'s second positional argument is `latent_dim` -- the bottleneck. The CAE's
own narrowest point is `chart_dim=20` (`encode -> z (embed_dim=40) -> chart_coords ->
chart_dim=20 -> chart_decoder -> embedding_decoder -> 768`), so the control solved a strictly
easier reconstruction problem at double the CAE's actual capacity -- exactly the asymmetry
`PlainAutoEncoder`'s own docstring exists to rule out ("whether the atlas bought anything over
one chart at matched capacity").

**The fix.** `build_control(latent_dim, device=...)` now takes `latent_dim` as an explicit,
required argument rather than hardcoding `PU_EMBED_DIM`. `run_grid` fits TWO labelled control
sets per seed: `control_seed{seed}` at `latent_dim=PU_CHART_DIM` (20, the matched D-12
comparison) and `control40_seed{seed}` at `latent_dim=PU_CONTROL_CAPACITY_DIM` (40, an
OPTIONAL, separately-labelled capacity reference -- genuinely informative on its own, per
`03-08-DEFECTS-01.md`'s "keep... if cheap" allowance, since `PlainAutoEncoder` fits in 11-32s
per 02.2's own measurement). `print_d12_trigger` self-filters `control_records` to
`latent_dim == PU_CHART_DIM` before computing anything, regardless of what its caller passes
in -- so the 40-dim reference can never silently enter the D-12 comparison even by a future
call-site mistake. `_control_records(completed, latent_dim=...)` exposes the same filter for
`run_select_only`, which now reports both control sets' presence counts separately and only
ever feeds the matched set to the trigger.

**Verification.** `build_control(20)` constructs a `PlainAutoEncoder` whose encoder's final
layer has 20 output features (checked directly by import); `--dry-run` prints both control
counts and their `latent_dim`s; `--smoke` still exits 0 (the smoke path does not touch
`_run_control_cell`, unaffected by this fix).

## 3. Fix 2 -- the training protocol no longer truncates on a confounded plateau

**The defect.** `MAX_EPOCHS=40`, `EARLY_STOP_PATIENCE=5`, `LIP_WEIGHT=1e-2` against the Swiss
roll's `300/25/1e-3`. `train_cae` early-stops on TOTAL loss -- reconstruction plus
cross-entropy plus the Lipschitz penalty -- and with the penalty 10x heavier than the roll's
own protocol, a plateau in the *penalty* term (not reconstruction) could and did end training
early: `epochs_run` across the nine grid cells was `[7, 7, 7, 7, 7, 34, 34, 40, 40]`. All three
40-dim controls ran the full 40 epochs without early stopping, i.e. were still improving at
the cap under the identical `max_epochs`/`patience`/`lip_weight` values -- direct evidence the
early-stopping criterion, not genuine convergence, ended the five 7-epoch cells.

**Investigation -- was `40/5/1e-2` justified anywhere?** Read `03-07-PLAN.md`,
`03-CONTEXT.md` (D-13), `03-RESEARCH.md`, and 02.2's own training runner
(`notebooks/diagnostics/cae_train_run.py`). Finding: `40/5/1e-2` is not an independently
chosen PU-scale value -- it is 02.2's own sealed-fit protocol, carried across "for
comparability" (`03-07-PLAN.md` line 126-129) with no PU-scale justification of its own. More
directly relevant: 02.2's own findings document (`02.2-06-SUMMARY.md`) named *"training
budget/epochs, chart latent dimension, loss weighting, the Lipschitz penalty schedule"* as
candidate investigation axes for a proposed Phase 02.3 iteration that never ran -- i.e. 02.2
itself flagged these exact three values as unresolved, not as validated-sufficient. No
document in this milestone independently justifies `EARLY_STOP_PATIENCE=5` or
`LIP_WEIGHT=1e-2` as adequate at PU scale. D-13's `3-5h` budget for the nine-fit grid is
anchored on 02.2's own `1941.2s`-per-fit timing-probe projection at `MAX_EPOCHS=40` -- and
`cae.timing_probe` (checked directly) projects `cfg["max_epochs"]`'s wall-clock from a
per-step measurement, structurally IGNORING early stopping. The already-declared D-13 budget
therefore already assumed the worst case (every fit running the full 40-epoch cap, no early
stopping), which the five 7-epoch cells fell dramatically short of -- not because the budget
was generous, but because the fits were truncated before that budget was used.

**The fix, and the reasoning for what changed and what did not.** The binding requirement
(`03-08-DEFECTS-01.md`) is that training must not be truncated by early stopping before the
model has converged, not that PU must adopt the roll's protocol verbatim -- the roll is a
2-D toy manifold trained on ~3,000 points, PU trains at `chart_dim=20`, `D=768` on 8,000 rows,
a materially different scale. Realigned exactly the two parameters the diagnosed mechanism
implicates:

- `EARLY_STOP_PATIENCE`: `5 -> 25` (matches the roll). A 5-epoch patience on a noisy
  multi-term total loss is exactly what let a Lipschitz-penalty plateau end training early;
  25 gives reconstruction loss room to keep driving total loss down through a temporary
  penalty plateau.
- `LIP_WEIGHT`: `1e-2 -> 1e-3` (matches the roll). Removes the 10x-heavier-than-the-roll
  confound directly -- the penalty term no longer dominates total loss, so a plateau in it
  no longer masks continued reconstruction improvement.
- `MAX_EPOCHS`: **left at 40, deliberately NOT raised to the roll's 300.** Two independent
  reasons. First, the wall-clock argument above: `cae.timing_probe`'s projection (and
  therefore D-13's declared 3-5h budget, and 03-07-SUPPLEMENT-01's measured ~5.6-5.7h ceiling
  that motivated the GPU supplement) already assumes the full 40-epoch cap runs on every cell
  with no early stopping credited -- raising `MAX_EPOCHS` to 300 would raise that already-
  measured ceiling by ~7.5x (to roughly 30-40h projected), a new budget commitment this fix
  is not authorized to make unilaterally (Rule 4 territory, not Rules 1-3). Second, the
  evidence in hand does not show `MAX_EPOCHS=40` itself was the truncating limit: even under
  the OLD `patience=5`, four of nine grid cells (plus all three controls) already reached
  34-40 epochs without hitting a hard ceiling artifact -- the cap was rarely the actual
  binding constraint; the plateau-detection rule was. `FPS_PRETRAIN_EPOCHS` (5 vs the roll's
  20) is left unchanged for the same reason: the diagnosed mechanism (total-loss early
  stopping under a tight patience and a heavy Lipschitz term) does not implicate the
  pre-training stage, and CLAUDE.md's keep-it-simple rule argues against widening the fix
  beyond the mechanism actually diagnosed.

**Expected wall-clock consequence.** Because `MAX_EPOCHS` is unchanged and `cae.timing_probe`
already projects the full-cap wall-clock regardless of early stopping, this fix should not
raise the grid's *projected* wall-clock ceiling beyond what `03-07-SUPPLEMENT-01` already
measured (~5.6-5.7h on this CPU machine, already over the nominal 3-5h envelope and the
reason GPU support was added). What changes is the *actual* wall-clock: cells that previously
finished in ~110s (7 epochs, `n_charts=4`) are now expected to run substantially longer,
likely trending toward the ~1941.2s-per-fit anchor most cells will now approach or reach,
since a real fit at PU scale is no longer artificially short-circuited. The nine-cell grid
should now cost closer to its already-budgeted ceiling than it previously did, not more than
that ceiling.

**Verification.** `--dry-run` and `--smoke` still exit 0 after the change (smoke uses its own
`SMOKE_MAX_EPOCHS=2`, unaffected by the protocol constants). The orchestrator re-runs the real
grid; this fix does not re-run it here per the objective's explicit instruction.

## 4. Fix 3 -- a dimension-invariant PH normalizer, opt-in

**The defect.** `cloud_distance_matrix(points, prescale=True)` scales each cloud by
`topoae.latent_unit_scale(cloud)` -- mean per-dimension variance fixed to 1, isotropic and
correct for what it was designed for, but leaving the *distance* scale growing as `sqrt(d)`.
Comparing the PU runner's 40-dim latent against its 768-dim ambient reference
(`sqrt(768/40) ~ 4.4x`) therefore saturates by construction: measured directly, identical
intrinsic structure isometrically embedded at d=10/40/128/768 gives `applied_scale` ratios
matching `sqrt(d/10)` exactly (`1, 2.00, 3.578, 8.763`), and the bottleneck between the 40-dim
and 768-dim diagrams of the *same* structure equals the saturation ceiling exactly
(`15.2054 == 15.2054`). Internally in the nine-cell grid, every `latent|*` PH cell read
`0.5, saturated=True` in all 12 records including controls -- the PH axis measured ambient
dimension, not topology.

**The fix.** `persistence_probe.cloud_distance_matrix` and `.readout_matrix`'s `prescale`
argument is widened from `bool` to `Union[bool, str]`. `True`/`False` behaviour is
byte-identical to before this fix -- verified by the full existing test suite passing
unmodified (§6) and by every existing call site (`decoder_substrate_ph_screen_run.py`,
`template_benchmark_run.py`, `ph_budget_calibration_run.py`, `template_tracer_run.py`) being
left untouched. A new value, `prescale="median_distance"`, applies
`median_pairwise_scale(points)` instead -- a NEW function that scales the cloud so its own
median pairwise Euclidean distance is 1, a dimension-invariant statistic by construction
(returns `1.0`, never raises or divides by zero, for fewer than 2 points or a degenerate
all-zero-distance cloud). Any other `prescale` value raises `ValueError` naming the three
accepted values.

**Verification against the defect's own measured evidence.** Re-ran the identical-structure
test with the new mode:

```
current (variance, prescale=True):        bottleneck=2.4144  ceiling=2.4144  saturated=True
new (median_distance, prescale="median_distance"): bottleneck=0.0000  ceiling=0.0637  saturated=False
```

matching `03-08-DEFECTS-01.md`'s own verified-fix table exactly.

**Regression test.** `test_median_distance_prescale_is_dimension_invariant_defect3` in
`notebooks/pu_manifold/tests/test_persistence_probe.py`: a fixed `(300, 10)` cloud is
zero-padded (an exactly distance-preserving isometric embedding, asserted directly) out to
d=40 and d=768; `prescale=True` gives `saturated=True` on the H0 bottleneck between the two
embeddings of the identical structure (characterizing the OLD default, not fixing it);
`prescale="median_distance"` gives `saturated=False` and `bottleneck < 0.05` (the fix).
Two supporting tests: `test_median_pairwise_scale_normalizes_median_distance_to_one`
(the new normalizer's own defining property) and
`test_cloud_distance_matrix_rejects_unknown_prescale_value` (the three-value contract).

**Wiring.** `curvature_field_pu_run.py`'s `PRESCALE_PU` is switched from `True` to
`"median_distance"` -- the PU runner's own PH diagnostic (`_ph_diagnostic`, called once per
grid/control cell) now compares its 40-dim latent against the `AMBIENT_DIM`-dim (768)
reference under the dimension-invariant normalizer, closing exactly the gap the defect
measured. This is the one intentional behaviour change on a currently-live call site,
explicitly authorized by the fix task (`03-08-DEFECTS-01.md`: "Point
`curvature_field_pu_run.py` at the dimension-invariant mode").

## 5. Prior-phase audit for defect 3

Per `03-08-DEFECTS-01.md`'s cross-phase implication section: `02.6-SCREENING-RULE-02.md`
already carries a "bottleneck-saturation travelling caveat" that flagged the symptom without
diagnosing the cause. Grepped the repo for every user of `persistence_probe.cloud_distance_matrix`
/ `.readout_matrix` (the only two functions that can apply `prescale`).

**Callers found, beyond the two named in `03-08-DEFECTS-01.md`:**
`notebooks/diagnostics/template_tracer_run.py` and
`notebooks/diagnostics/ph_budget_calibration_run.py` (both Phase 02.7). No other caller
exists in the repository (`notebooks/pu_manifold/confidence_band.py` and
`notebooks/pu_manifold/geodesic_graph.py` reference `persistence_probe` in prose/docstring
only, calling `persistence_diagram` directly on a distance matrix supplied by their own
callers, never `cloud_distance_matrix` or `readout_matrix` themselves).

**Which pass `prescale=True` (the trigger condition) at all:** only two, matching
`03-08-DEFECTS-01.md`'s own naming --
`notebooks/diagnostics/decoder_substrate_ph_screen_run.py` (Phase 02.6, `PRESCALE_CLOUDS =
True`, every seed/candidate) and `notebooks/diagnostics/curvature_field_pu_run.py` (Phase 3,
the primary defect, fixed above). `template_tracer_run.py`, `ph_budget_calibration_run.py`,
and `template_benchmark_run.py` (Phase 02.7) all call `cloud_distance_matrix(points,
prescale=False)` exclusively, and none of them calls `readout_matrix` at all -- confirmed by
reading each call site directly. `prescale=False` never applies any scalar (`cloud_distance_matrix`
returns `scale=1.0` unconditionally on that branch), so defect 3's mechanism cannot manifest
in these three files regardless of whether they ever compare clouds of differing ambient
dimension. **Phase 02.7 is unaffected by defect 3** and needs no note.

**Phase 02.6 is affected** and its own predecessor plans compare clouds of differing ambient
dimension under `prescale=True` at several of the sixteen read-out cells (8-12 of 16 cells per
candidate, depending on the candidate's own latent dimension). A note is written --
`.planning/phases/02.6-decoder-substrate-screening/02.6-NOTE-ph-saturation-artifact.md`,
committed at `66ff107`. Summary of that note's own careful scoping (full detail there, not
repeated here): the mismatch magnitudes in 02.6 (`1.22x`-`2.0x`) are far smaller than the PU
case's `4.4x`; the recorded data itself carries evidence AGAINST defect 3's `sqrt(d)`
mechanism being the dominant cause of this phase's one observed saturated cell
(`latent|ambient|H1|bottleneck` saturates for `plainae`/`topoae` but never for `cae`, despite
`cae` carrying the LARGER dimension mismatch on that exact cell -- the opposite of what the
`sqrt(d)` mechanism alone would predict); `02.6-FINDINGS-02.md` §7b had already independently
discovered and named a closely related dimensionality confound before this diagnosis existed.
The note states precisely what is and is not settled, and what re-running the read-out matrix
under `prescale="median_distance"` (now available) would show if someone wants to check
further. No sealed 02.6 artifact is edited; every recorded number is unchanged on disk.

## 6. Verification

```
# hard constraint 1 -- the roll anchor, verified by direct import and call, not from cache
.venv/bin/python -c "
import sys; sys.path.insert(0, 'notebooks'); sys.path.insert(0, 'notebooks/diagnostics')
import swiss_roll_curvature_sweep_run as m; import torch
rec = m.run_cell(n_charts=8, seed=0, device=torch.device('cpu'))
assert rec['rho_chart'] == -0.06041003026778113, rec['rho_chart']
print('MATCH')
"
-> -0.06041003026778113
-> MATCH

# hard constraint 4 -- full suite, before any change in this supplement (baseline)
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
-> 286 passed, 1 skipped, 19 warnings in 124.93s

# after persistence_probe.py's defect-3 fix + regression tests (commit 260a614)
.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_persistence_probe.py -q
-> 16 passed in 35.30s
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
-> 289 passed, 1 skipped, 19 warnings in 112.42s

# after curvature_field_pu_run.py's defects 1-3 (commit 34ccc54)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --dry-run
-> exits 0, prints both control sets (matched latent_dim=20, capacity-reference latent_dim=40)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --smoke --record-path <scratch>
-> exits 0, SMOKE TALLY printed, ph cells computed under the new median_distance normalizer
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
-> 289 passed, 1 skipped, 19 warnings in 113.02s (unchanged, final confirmation)
```

`build_control(20)` verified directly to construct a `PlainAutoEncoder` whose encoder's final
layer has 20 output features (not 40). No sealed cache artifact
(`notebooks/.cache/03_curvature_field_pu.jsonl` from the invalidated grid, or any prior
phase's gitignored cache) was read, written, or deleted by this work -- every check above used
a scratch `--record-path` or ran no cache-touching command at all.

## 7. What this supplement does not do

Per the objective's explicit instruction, this supplement does NOT re-run the nine-cell PU
grid -- the orchestrator does that after reviewing these fixes. No `n_charts` selection exists
anywhere in this repository as a result of this work. `notebooks/pu_manifold/cae.py` was not
modified (hard constraint 2). No prior phase's sealed findings, summaries, or verdicts were
edited (02.6's note is additive-only, confirmed above). `MAX_EPOCHS` was deliberately left
unchanged rather than raised to the roll's `300` -- see §3's full reasoning; if the re-run
grid still shows truncation-shaped `epochs_run` patterns even under the widened patience, that
would be new evidence for a future, explicitly-authorized `MAX_EPOCHS` increase, not something
this fix pre-empts.

## 8. State

Phase 3's PU grid is unblocked to re-run under the three fixes above. `03-08-DEFECTS-01.md`'s
consequence section ("No `n_charts` may be selected from this grid... The grid must be
re-run after defects 1 and 2 are fixed and defect 3 is either fixed or its axis explicitly
withdrawn") is discharged: defects 1 and 2 are fixed, and defect 3 is fixed (not withdrawn) via
the opt-in `median_distance` normalizer now wired into `curvature_field_pu_run.py`.
