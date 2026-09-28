# Phase 9: Curvature-Conditioned Label Decodability (Physics Replication) - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Recreate the colleague's curvature–decodability experiment from `origin/curvature-experiments` with
this milestone's instrument, on the same data and the same outcome. His frozen result: controlled
Spearman `rho(K_H, local OOF R^2 of a global ridge probe for r-band magnitude) = -0.240` (raw
`-0.412`) at chart rank `d=16`, `k=2048` neighbours, `n=512` anchors, ViT-B Physics embeddings; `+0.143`
at `d=12`, `-0.233` at `d=20`. His association exists only at his largest `k` (k=1024 controlled
`-0.027`), so neighbourhood scale is load-bearing.

**One thing changes:** his `k=2048` nested-PCA quadratic-chart curvature estimator is replaced by the
plain-autoencoder decoder curvature (`cae.PlainAutoEncoder` + `decoder_curvature.plain_decoder_curvature`,
trace convention, Phase 7's frozen fit protocol). Everything else — outcome, controls, permutation
scheme, gates — matches his `METHODS_FOR_PAPER.md` §9–§11 or this milestone's stricter equivalent.

**In scope:** data download and row-alignment proof; the OOF ridge probe and per-anchor local `R^2`;
AE fits at `D_SWEEP = (16, 20, 25, 32)`; `||H||` and `||H_tan||` at anchors; the 3-control partial with
Freedman–Lane/FWER null and 07.1's stratified null beside it; positive control, shuffled-label
calibration, fit-quality read-out per `d`; a two-wave seed design; frozen pre-registration; the
reporting notebook and `09-FINDINGS.md`.

**Out of scope:** modifying any sealed module; touching `src/effdim/`; reinterpreting any sealed
verdict from Phases 2–8; a per-anchor comparison against his `K_H` values (his `selection.npz` is
not on his branch — see Deferred); any label he excluded (`sfr`, DESI fields).

**The decisions below (D9-01..D9-18) are this phase's requirement set**, the arrangement Phases 7
and 8 used.

</domain>

<decisions>
## Implementation Decisions

### Sample, neighbourhood and anchor scale

- **D9-01 — Full 86,471 Physics rows.** `UniverseTBD/pu-embeddings` config `physics_vit_base_test`
  (768-D, single column `vit_base_galaxies`), the whole test split. No subsample. His host used a
  16,384-row hash-selected subset; we do not reproduce that subset.
- **D9-02 — Local `R^2` neighbourhood `k = 2048` only.** His absolute `k`, no `k` grid. With
  `n = 86,471` this is 1/42 of the data where his was 1/8; the record must state that ratio.
- **D9-03 — 512 anchors, seeded uniform draw**, matching his `n = 512` so the final Spearman sits on
  the same sample-size footing and his paired-anchor bootstrap bands are comparable.
- **D9-04 — Anchors are drawn from the AE holdout rows only** (~17k at Phase 7's
  `HOLDOUT_FRACTION = 0.2`). **A deliberate departure from Phase 7's
  `FIELD_EVALUATED_ON = all_10000_rows_including_the_8000_training_rows`**: curvature is measured only
  where the decoder never trained. Neighbourhoods (`k`-NN over all 86,471 rows) and the OOF probe
  folds are independent of the AE split. — **Reversibility:** costly — flipping to all-rows anchors
  after any number exists changes the anchor set and voids the freeze.

### Row-alignment proof (serves the roadmap's "no proof, no Physics number")

- **D9-05 — The colleague's standard is a principle, not a method.** His branch records "equal row
  count is **not** the proof" and struck DESI associations as `desi_label_alignment_unresolved`
  (`Proved=False`). For Physics his join is a documented convention — `sample_id` = galaxies
  test-table row index; `vit_base_test_labels.npz` "row-aligned to `vit_base_test.parquet`" — with no
  test anywhere on the branch and the labels-build script absent. Phase 9 supplies the method.
- **D9-06 — Method: statistical shifted-row check.** Fit the 5-fold OOF ridge probe embedding →
  `mag_r` at the assumed alignment (shift 0), then at each alignment in a frozen shift set. Aligned
  data gives `R^2(shift 0)` far above every shifted `R^2` (≈0). Re-embedding galaxies with the ViT-B
  checkpoint was considered and not chosen (needs the authors' exact weights and preprocessing).
- **D9-07 — Shift set, frozen before download:** `mag_r` only; row shifts `±1..±10`, `±100`, `±1000`,
  plus 20 seeded random permutations. Pass rule: `R^2(shift 0)` exceeds the maximum over every
  shifted/permuted alignment by a pre-registered margin (margin value: planner).
- **D9-08 — On failure, SEARCH for the true offset.** If shift 0 fails but some other shift passes,
  adopt that offset and proceed. **Recorded as a post-hoc, data-chosen step**: the developer chose
  this over "halt with no Physics number". The plan must make the adoption explicit (its own
  amendment document and freeze commit), and `09-FINDINGS.md` must state which offset was used and
  that it was found rather than assumed. — **Reversibility:** one-way — an adopted offset is baked
  into every downstream number; changing it means a fresh freeze and full re-run.

### Replication verdict rule

- **D9-09 — The verdict statistic is his exact 3-control rank-partial Spearman:**
  `rho(curvature, local OOF R^2 | log_knn_radius, local_label_variance, local_evaluation_count)`,
  ranks residualized on ranks, as `inference.py`'s `associate`/`control_matrix`. Raw `rho` and
  07.1's within-density-stratum permutation partial are reported beside it, non-gating. Both nulls
  are reported unconditionally (roadmap).
- **D9-10 — "Replicates" = controlled partial is NEGATIVE and clears its own Freedman–Lane rank
  permutation null with FWER (max-|rho| envelope) across `d`, at one or more `d`.** No magnitude
  threshold. Magnitude is printed beside his `-0.240` with both bootstrap bands (his B=2000 paired
  anchor resamples; ours the same). Every-`d` and magnitude-band rules were considered and not
  chosen. Vocabulary follows 07.1's `SURVIVES AT SUBSET OF d` / per-`d` independent reporting
  (D8-13 pattern). — **Reversibility:** one-way — frozen verdict rule under D7-06.
- **D9-11 — `||H_tan||` (sphere-tangential mean curvature) carries the verdict; `||H||` is in the
  same table, non-gating.** His estimator removes the sphere-radial component before the quadratic
  fit, so `||H_tan||` is the like-for-like quantity. `08-DIAGNOSTICS.md` §2 measured the two fields
  disagreeing on a partial by 2.8x at `d=25`, so this choice is made before any number and is not
  revisited after. Decomposition machinery: `08_radial_curvature_decomposition_run.py`.
  — **Reversibility:** one-way — same freeze discipline.
- **D9-12 — `D_SWEEP = (16, 20, 25, 32)`.** `d=16` added so one cell matches his chart rank
  directly. Phase 9 declares its own sweep constant in its own module; Phase 7's frozen `D_SWEEP`
  is not edited. Fit-quality read-out (`var_explained`, `cond(g)`) is required at every `d`
  including 16. Fixture fidelity at `d=16` is currently unmeasured (the small-ambient fixture arm
  supports `d ≤ 27`); whether to measure it is Claude's discretion. His `d=12` `+0.143` lies
  outside our sweep and is reported as non-comparable.

### Probe and control construction

- **D9-13 — Ridge `alpha = 100`, fixed**, his `METHODS_FOR_PAPER.md` §9 value. No grid, no selection.
  Five-fold OOF: fold `f` predicted from weights fit on the other four. Local `R^2`, MSE and SST
  computed per anchor over its 2048 neighbours with finite `y`, `ŷ`, uniform weights, exactly §10.
  `linear_probe.fit_probe` / `predict_probe` are the existing implementations to reuse (sealed;
  import only).
- **D9-14 — Positive control: curvature-side rank-plant** on the pattern of
  `crossmodal_curvature.plant_positive_control`. Real local `R^2` kept; a synthetic curvature array
  spread-matched to the realized `||H_tan||` range is planted at a grid of target `rho` by
  bisection, then pushed through the identical 3-control partial and null. Reports the smallest
  cleared target (detection floor). An `R^2`-side perturbation (degrading predictions in
  high-curvature neighbourhoods) was considered and not chosen.
- **D9-15 — Shuffled-label calibration** (roadmap): shuffle the label vector across rows, run the
  entire pipeline, read the false-positive rate. Shape is Claude's discretion (see below).
- **D9-16 — Secondary labels `photo_z`, `smooth_fraction`, `stellar_mass` are reported,
  non-gating.** Same pipeline per label, same table, own nulls; `mag_r` alone decides.
  `stellar_mass` has ~7k unlabeled rows and needs a missing-value mask (his record: 79,490 of
  86,471 labelled). `sfr` excluded as underpowered, per his record.
- **D9-17 — Seeds, two waves.** Wave A: single `TORCH_INIT_SEED` across all four `d` (Phase 7's
  `SEED_HANDLING_RULE`). Wave B, conditional: three seeds at every `d` where the wave-A verdict
  fired; unanimity 3-of-3 or the cell is `SPLIT ACROSS SEEDS`; **seeds are never pooled**
  (`05-03-DECISION.md`, one-way). — **Reversibility:** one-way — inherits the ratified do-not-pool
  constraint.

### Inherited, non-negotiable

- **D9-18 — Freeze before any number (D7-06 / D8-22).** Every constant above — sample, `k`, anchor
  seed and pool, shift set and margin, verdict statistic and rule, `D_SWEEP`, `alpha`, positive-control
  grid, seed rule, and the unconditional reporting block — committed in one freeze commit,
  git-ancestry-proved to precede every measured value (`merge-base --is-ancestor` AND
  `rev-list --count ≥ 1`). Additive only; no sealed module mutated on import (D8-23); `src/effdim/`
  untouched (D8-24). The verdict sentence must name the instrument and `d` beside his and print
  the per-`d` table, D8-21's caveat-bearing-verdict pattern. Report `p < 1/(B+1)`, never `p = 0`.

### Claude's Discretion

- The alignment margin's numeric value (D9-07) and how a found offset is ratified (D9-08).
- Whether to measure fixture fidelity at `d=16` before the Physics run (D9-12).
- Shuffled-label calibration shape: number of repeats, whether labels are shuffled globally or
  local `R^2` shuffled across anchors (D9-15).
- Permutation count (his `B = 10^4`) and bootstrap count (his `B = 2000`) — inherit or reduce with a
  measured cost table, the 08-PREREGISTRATION-AMENDMENT-01 pattern.
- The OOF fold seed, the anchor-draw seed, the density/radius `k` for controls (his
  `log_knn_radius` is the radius of the same 2048-neighbourhood).
- Positive-control target-`rho` grid values.
- Module naming, runner layout, wave decomposition, runtime budget. Curvature at 512 anchors is
  seconds per `d`; four AE fits at 86k rows are the main cost (~50 min each, serial,
  `OMP_NUM_THREADS` capped — `07-CONTEXT.md` §7).
- How 07.1's stratified null attaches: strata on `log_knn_radius` rank (his radius) versus
  `curvature_probe.local_density_weights` — either is defensible; state which.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The experiment being replicated (colleague's branch `origin/curvature-experiments`, read via `git show`)
- `origin/curvature-experiments:paper/curvature_neurreps/audit_outputs/submission_validation/METHODS_FOR_PAPER.md`
  — §9 OOF probe (`alpha = 100`, 5-fold), §10 exact local `R^2`, §11 controls / Freedman–Lane /
  FWER / paired bootstrap / `p < 1/(B+1)`. §1 confirms his embeddings are row-ℓ₂ normalised.
- `origin/curvature-experiments:experiments/geometry/physics_curvature_probe_rank_sweep/inference.py`
  — `CONTROLS`, `associate`, `control_matrix`, `freedman_lane_y`, `permutation_curves`,
  `paired_bootstrap_curves`. The reference implementation of D9-09/D9-10.
- `origin/curvature-experiments:CONTEXT.md` lines 60–75 — target identity table; "Proven join" wording;
  `desi_label_alignment_unresolved`; the `probe_label_alignment_failure` historical bug (typed
  targets: never substitute catalog `mag_r` for local OOF `R^2`).
- `origin/curvature-experiments:paper/curvature_neurreps/audit_outputs/multilabel_chart_screen/mag_r_desi/global_anchor_metrics.csv`
  — his 512-anchor table (`K_H_cross`, `log_knn_radius`, `r2_G`, `mse_G`, `local_label_variance`,
  `local_evaluation_count`); the same table exists for `photo_z`, `smooth_fraction`, `stellar_mass`.
- `origin/curvature-experiments:paper/curvature_neurreps/audit_outputs/submission_validation/scale_sensitivity.csv`
  — the `k` sensitivity (association only at k=2048).
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-COLLEAGUE-REANALYSIS.md`
  — his numbers reproduced from those tables on 2026-09-02, the stratified-null re-analysis, and
  the reproduction script.

### This milestone's instrument and protocol
- `.planning/ROADMAP.md` — Phase 9 entry (goal, data, outcome, controls, radial term, gates).
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-CONTEXT.md` — D7-01 (instrument),
  D7-06 (freeze), §4 (fixture fidelity), §5 (PU fit protocol), §7 (cost model).
- `notebooks/pu_manifold/crossmodal_curvature.py` — the frozen fit protocol constants (`AE_HIDDEN`,
  `AE_ACTIVATION`, `MAX_EPOCHS`, `TRAIN_CFG`, `HOLDOUT_FRACTION`, `SPLIT_SEED`), `plant_positive_control`,
  `assert_preregistered`. Sealed; import, never edit.
- `notebooks/diagnostics/08_radial_curvature_decomposition_run.py` and
  `.planning/phases/08-curvature-conditioned-cka-alignment/08-DIAGNOSTICS.md` §2 — the `||H_tan||`
  decomposition D9-11 uses and the measured 2.8x collapse that motivates it.
- `.planning/phases/07.1-density-stratified-null-and-seed-stability/07.1-CONTEXT.md` — D-01..D-08
  (stratified null construction), D-11 (unanimity), D-14/D-15 (per-`d` independent verdicts,
  `SURVIVES AT SUBSET OF d`).
- `.planning/phases/08-curvature-conditioned-cka-alignment/08-CONTEXT.md` — D8-21 (caveat-bearing
  verdict), D8-22/23/24 (freeze, additive-only, `src/effdim/` untouched).
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-03-DECISION.md` — the one-way
  do-not-pool-seeds ratification D9-17 carries.
- `notebooks/pu_manifold/linear_probe.py` — `fit_probe`, `predict_probe`, `per_point_residuals`,
  `aggregate_r2`, `combine_seed_verdicts`, freeze machinery. Sealed.
- `notebooks/pu_manifold/density_stratified_null.py` — `density_strata`, stratified permutation.
- `notebooks/pu_manifold/subsample.py` — `l2_normalize`; the `physics_vit_base_test` loader is new
  (single-column config, no pairing, no `object_id`).
- `HANDOFF-v1.1.md` §5–§7 — post-freeze diagnostics and practices worth keeping.

### Project rules
- `CLAUDE.md` — additive only, `src/effdim/` frozen for v1.1, KEEP THINGS SIMPLE FIRST. The Swiss
  roll rule introduces no new model here (plain AE already has
  `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb`), the Phase 7 declaration.
- `.claude/skills/spike-findings-effdim/SKILL.md` — anchor at low `d`, write the decision rule
  before running, compare fixtures spread-for-spread.

### Data
- `UniverseTBD/pu-embeddings`, config `physics_vit_base_test` (86,471 rows, `physics/vit_base_test.parquet`,
  not cached locally as of 2026-09-02; only `desi` and `legacysurvey` parquets are).
- `Smith42/galaxies` test split (86,471 rows; `mag_r`, `photo_z`, `smooth_fraction`, `stellar_mass`),
  not cached locally.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cae.PlainAutoEncoder`, `cae.train_plain_ae`, `decoder_curvature.plain_decoder_curvature` — the
  instrument, unchanged.
- `crossmodal_curvature.py` fit constants and `plant_positive_control` — the protocol and the
  positive-control pattern D9-14 follows.
- `08_radial_curvature_decomposition_run.py` — `||H_tan||` from `||H||` and the decoder image norm.
- `linear_probe.fit_probe` / `predict_probe` — ridge with intercept at a given `alpha`; wrap for
  5-fold OOF.
- `density_stratified_null.density_strata` + stratified permutation loop — 07.1's null.
- `07_crossmodal_curvature_run.py` / `08_cka_alignment_run.py` — runner shape: `assert_preregistered`,
  freeze-SHA strict-ancestor gate, `--mode smoke|...`, jsonl records in `notebooks/.cache/`.

### Established Patterns
- Freeze before any number with git-ancestry proof (D7-06, D8-22).
- Non-gating diagnostics reported beside the verdict (D7-03).
- Per-`d` verdicts independent, no pooled headline (07.1 D-14, D8-13).
- Caveat-bearing verdict sentence and a frozen unconditional reporting block (D8-21).
- Never pool seeds; unanimity or `SPLIT ACROSS SEEDS` (05-03).
- Report to full precision; `p < 1/(B+1)` never `p = 0`.

### Integration Points
- New module(s) under `notebooks/pu_manifold/` and runner(s) under `notebooks/diagnostics/`, importing
  sealed modules read-only. New Physics loader alongside `subsample.py`, not inside it.
- Records to `notebooks/.cache/09_*.jsonl`; the frozen anchor table (512 rows × curvature, `R^2`,
  MSE, SST, controls, per `d` and label) is the phase's primary artifact.
- Reporting notebook committed with outputs; `09-FINDINGS.md`.

</code_context>

<specifics>
## Specific Ideas

- **His scale table is the warning:** k=1024 controlled `-0.027`, k=1536 `-0.080` (p 0.37), k=2048
  `-0.240`. If our `-0.24`-class number appears only at his `k`, that is a finding about scale, not
  about curvature, and the record must say so. D9-02 fixes `k` anyway; the `n`-ratio difference
  (1/42 vs 1/8) must be stated next to it.
- **`R_H` cannot see shared bias.** His reliability gate is split-half `R_H` only;
  `06-FINDINGS.md` measured `R_H = 0.990` beside `rho = 0.469` against truth on the Swiss roll. Our
  fixture-fidelity range is the analogue and must be quoted (`(0.53, 0.99)` at `d=20`, `(0.17, 0.97)`
  at `d=25`, none at `d=16` or `d=32`).
- **Typed targets.** The outcome is local OOF `R^2`, never catalog `mag_r`; his historical
  `probe_label_alignment_failure` bug was exactly that substitution (Spearman ≈ −0.215 between the
  two `y` vectors looked like a sign flip).
- **`rho(K_H, log_knn_radius) = +0.765`** on his table; radius control is doing most of the work
  (`-0.412 → -0.246` with radius alone). Expect the same on ours.
- **Execution host (added 2026-09-02, plan-phase):** Phase 9 will NOT execute on the developer's
  local machine. Compute is either an SSH remote server or the colleague's box (his call which);
  undecided at planning time. Runners, data download, cache paths, the freeze-SHA ancestry gate,
  and the returned artifacts must therefore work from a fresh clone on a machine we do not
  control: no hard-coded local paths, HF cache and `notebooks/.cache/` locations configurable,
  outputs bundled for transfer back, cost stated per-thread not "on this machine". Smoke mode
  may still run locally; no real number does.

</specifics>

<deferred>
## Deferred Ideas

- **Per-anchor instrument comparison against his `K_H`.** Needs his `selection.npz` (indexes his
  16,384-row subset); not on the branch. Optional per the roadmap; requesting it from the colleague
  is the developer's call, not a planning task.
- **`k` sensitivity grid on our data** (e.g. 512/1024/2048/4096) — presented and declined for this
  phase; a natural follow-up if D9-10 fires.
- **`R^2`-side positive control** (degrading predictions in high-curvature neighbourhoods) —
  considered, not chosen.
- **Re-embedding galaxies with the ViT-B checkpoint** as a direct alignment proof — considered, not
  chosen.
- **Fixture fidelity at `d=32`** remains unmeasured (08-07 halt on the `D=28` literal); Phase 9 does
  not fix the fixture.

</deferred>

---

*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Context gathered: 2026-09-02*
