# Phase 9 Pre-Registration — every gating constant frozen before a Physics number exists

**Date:** 2026-09-03
**Freeze commit SHA:** `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`
**Filled by:** plan `09-05`, Task 1 (this document), Task 2 wires the SHA above and into
`notebooks/diagnostics/09_physics_curvature_run.py`, `notebooks/diagnostics/09_row_alignment_proof_run.py`,
`notebooks/pu_manifold/tests/test_physics_curvature_probe.py` and
`notebooks/pu_manifold/tests/test_physics_labels.py`.
**Committed by:** plan `09-05`, Task 1's single freeze commit, which touches exactly three files —
`notebooks/pu_manifold/physics_labels.py`, `notebooks/pu_manifold/physics_curvature_probe.py`, and
this document. No file that produces a number is touched by the freeze commit (D9-18 precision).
No `notebooks/.cache/09_row_alignment.jsonl` or `notebooks/.cache/09_physics_curvature.jsonl`
exists anywhere in this repository as of this document's commit.

## What this document forecloses

Once the freeze commit named above lands, none of the values in the table below may change for
the remainder of Phase 9 without a numbered amendment (the `08-PREREGISTRATION-AMENDMENT-01.md`
pattern) superseding this document, and a full re-run of anything already measured under the
superseded value. This applies in full even though no Physics number exists yet — the discipline
begins at the freeze, not at the first number (D9-18).

## Checkpoint ratification this document rests on

**`09-04` Task 2 — the one-way column-mapping / sentinel / margin ratification.** Put to the
developer at a blocking `checkpoint:decision` in `09-DATA-MANIFEST.md` Section 5 (two readings of
`09-CONTEXT.md`'s "any label he excluded (`sfr`, DESI fields)" out-of-scope line). **Developer's
reply, verbatim, received 2026-09-03 UTC (`09-DATA-MANIFEST.md` Section 7):**

> ratify-as-proposed

This resolved the reconciliation under the **narrow reading**: the out-of-scope line strikes only
the colleague's unresolved DESI cross-match associations he marked `desi_label_alignment_unresolved`
and struck as `Proved=False` — it does not strike `mag_r_desi`, a photometry column that is not one
of those associations. The `LABEL_COLUMN_MAP`, `SENTINEL_VALUES` and `ALIGNMENT_MARGIN_R2` values
in the table below are transcribed exactly from `09-DATA-MANIFEST.md` Section 7, entry for entry,
with no addition, omission or substitution — see the entry-for-entry comparison in this plan's
`09-05-SUMMARY.md`.

Every other value in the table below is not a checkpoint ratification: it comes directly from
`09-CONTEXT.md`'s locked decisions D9-01 through D9-18 (the developer's own `/gsd-discuss-phase`
choices), from a measurement already on record (`09-DATA-MANIFEST.md`, `09-FIXTURE-FIDELITY-D16.md`),
from a sealed module's own frozen value re-declared fresh across the freeze boundary
(`crossmodal_curvature.py`), or from a planner discretion decision recorded in `09-05-PLAN.md`'s
`<discretion_decisions>` block with its reasoning stated there in full.

## Pre-registered constants

Every value below is the verbatim value of the identically-named constant in
`notebooks/pu_manifold/physics_labels.py` or `notebooks/pu_manifold/physics_curvature_probe.py`,
as committed at this freeze commit. Long string/tuple/dict values are truncated with `...` in this
table for readability; the committed source is authoritative in every case — this table is a
transcription of it, never a parallel specification. One shared literal seed, `20260902`, is used
across every Phase 9 statistical component (anchor draw, OOF fold split, alignment permutations,
Freedman-Lane null, bootstrap, positive control, shuffled-label calibration) — noted once here
rather than repeated at each row; the autoencoder fit protocol keeps Phase 7's own
`SPLIT_SEED = 20260813` and `TORCH_INIT_SEED = 0` so the fit matches Phase 7/8 exactly.

| Module | Constant | Value | Source / rationale |
|---|---|---|---|
| `physics_labels.py` | `PHYSICS_REPO` | `'UniverseTBD/pu-embeddings'` | 09-DATA-MANIFEST.md §1 (embeddings source) |
| `physics_labels.py` | `PHYSICS_CONFIG` | `'physics_vit_base_test'` | 09-DATA-MANIFEST.md §1 |
| `physics_labels.py` | `PHYSICS_PARQUET_PATH` | `'hf://datasets/UniverseTBD/pu-embeddings/physics/vit_base_test.parq...` | 09-DATA-MANIFEST.md §1, single fixed parquet path |
| `physics_labels.py` | `PHYSICS_COLUMN` | `'vit_base_galaxies'` | 09-DATA-MANIFEST.md §1 |
| `physics_labels.py` | `EXPECTED_N_PHYSICS_ROWS` | `86471` | Measured full-scale, 09-DATA-MANIFEST.md §2 (Assumption A2 resolved) |
| `physics_labels.py` | `EMBEDDING_NORMALIZATION` | `"row_l2_via_subsample.l2_normalize applied to every embedding row b...` | Matches colleague's METHODS_FOR_PAPER.md §1 (row-L2-normalised) |
| `physics_labels.py` | `LABEL_REPO` | `'Smith42/galaxies'` | 09-DATA-MANIFEST.md §1 (label source) |
| `physics_labels.py` | `LABEL_REVISION` | `'v2.0'` | Default revision lacks label columns (09-RESEARCH.md Pitfall 1) |
| `physics_labels.py` | `LABEL_SPLIT` | `'test'` | 09-DATA-MANIFEST.md §1 |
| `physics_labels.py` | `LABEL_N_SHARDS` | `16` | 09-DATA-MANIFEST.md §1, all 16 shards measured |
| `physics_labels.py` | `LABEL_SHARD_ORDER_RULE` | `'Shards are concatenated in ascending index order, 0..LABEL_N_SHARD...` | D9-05 join convention; see value (self-describing rule string) |
| `physics_labels.py` | `LABEL_COLUMN_MAP` | `{'mag_r': 'mag_r_desi', 'photo_z': 'photo_z', 'smooth_fraction': 's...` | 09-DATA-MANIFEST.md §7 Ruling, transcribed verbatim (ratify-as-proposed, 2026-09-03) |
| `physics_labels.py` | `LABEL_COLUMN_MAP_PROVENANCE` | `{'mag_r': "Phase 9's own documented convention: mag_r_desi is 100.0...` | 09-DATA-MANIFEST.md §4/§6 |
| `physics_labels.py` | `PRIMARY_LABEL` | `'mag_r'` | D9-06/D9-09, the gating label |
| `physics_labels.py` | `SECONDARY_LABELS` | `('photo_z', 'smooth_fraction', 'stellar_mass')` | D9-16, non-gating |
| `physics_labels.py` | `SECONDARY_LABELS_ARE_NON_GATING` | `True` | D9-16 |
| `physics_labels.py` | `EXCLUDED_LABELS` | `('sfr',)` | D9-16, ~8.45% coverage measured |
| `physics_labels.py` | `EXCLUDED_LABELS_RULE` | `"sfr (raw column total_sfr_median) excluded as underpowered: measur...` | 09-DATA-MANIFEST.md §3; see value |
| `physics_labels.py` | `SENTINEL_VALUES` | `(-99.0,)` | 09-DATA-MANIFEST.md §7 Ruling, transcribed verbatim |
| `physics_labels.py` | `ALIGNMENT_LABEL` | `'mag_r'` | D9-06 |
| `physics_labels.py` | `ALIGNMENT_SHIFT_SET` | `(-1000, -100, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, ...` | D9-07, frozen before download, 24 non-zero shifts |
| `physics_labels.py` | `ALIGNMENT_N_PERMUTATIONS` | `20` | D9-07, "20 seeded random permutations", inherited verbatim |
| `physics_labels.py` | `ALIGNMENT_PERMUTATION_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_labels.py` | `ALIGNMENT_MARGIN_R2` | `0.1` | 09-DATA-MANIFEST.md §7 Ruling, transcribed verbatim |
| `physics_labels.py` | `ALIGNMENT_PASS_RULE` | `'passed is True iff gap = r2_shift0 - best_other_r2 is STRICTLY gre...` | Exact-equality guarded; see value (self-describing rule string) |
| `physics_labels.py` | `ALIGNMENT_SEARCH_RULE` | `"the D9-08 SEARCH branch adopts a non-zero alignment only when exac...` | Exact-equality guarded; see value (self-describing rule string) |
| `physics_labels.py` | `ALIGNMENT_ASSUMED_OFFSET` | `0` | D9-08 |
| `physics_labels.py` | `HF_CACHE_ENV_VARS` | `('HF_HOME', 'HF_DATASETS_CACHE')` | resolve_hf_cache_dir's execution-host knob |
| `physics_labels.py` | `MANIFEST_RECORD_STEM` | `'09_data_manifest'` | D9-05 record stem |
| `physics_labels.py` | `ALIGNMENT_RECORD_STEM` | `'09_row_alignment'` | D9-06/D9-07 record stem |
| `physics_curvature_probe.py` | `K_NEIGHBOURS` | `2048` | D9-02, his absolute k, no grid |
| `physics_curvature_probe.py` | `NEIGHBOURHOOD_RATIO_RULE` | `'K_NEIGHBOURS=2048 of n=86,471 is 1/42 of the Physics sample (his 2...` | D9-02; see value (self-describing rule string) |
| `physics_curvature_probe.py` | `N_ANCHORS` | `512` | D9-03, matches colleague's n=512 |
| `physics_curvature_probe.py` | `ANCHOR_DRAW_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `ANCHOR_POOL` | `'ae_holdout_rows_only'` | D9-04 |
| `physics_curvature_probe.py` | `ANCHOR_POOL_RULE` | `"Anchors are drawn only from the AE holdout rows (~17k at HOLDOUT_F...` | D9-04; see value (self-describing rule string) |
| `physics_curvature_probe.py` | `SPLIT_SEED` | `20260813` | crossmodal_curvature.SPLIT_SEED, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `HOLDOUT_FRACTION` | `0.2` | crossmodal_curvature.HOLDOUT_FRACTION, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `AE_IN_DIM` | `768` | crossmodal_curvature.AE_IN_DIM, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `AE_HIDDEN` | `(250, 250, 250)` | crossmodal_curvature.AE_HIDDEN, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `AE_ACTIVATION` | `'silu'` | crossmodal_curvature.AE_ACTIVATION, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `MAX_EPOCHS` | `600` | crossmodal_curvature.MAX_EPOCHS, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `TORCH_INIT_SEED` | `0` | crossmodal_curvature.TORCH_INIT_SEED, re-declared fresh (byte-identical) |
| `physics_curvature_probe.py` | `TRAIN_CFG` | `{'lr': 0.001, 'weight_decay': 0.0001, 'batch': 128, 'lip_weight': 0...` | crossmodal_curvature.TRAIN_CFG, re-declared fresh (byte-identical, early-stop disabled) |
| `physics_curvature_probe.py` | `CURVATURE_SOURCE_FUNCTION` | `'decoder_curvature.plain_decoder_curvature'` | crossmodal_curvature.CURVATURE_SOURCE_FUNCTION, re-declared fresh |
| `physics_curvature_probe.py` | `CURVATURE_CONVENTION` | `'trace'` | Must equal "trace" (guarded); matches Phase 7/8 |
| `physics_curvature_probe.py` | `D_SWEEP` | `(16, 20, 25, 32)` | D9-12, Phase 9's own fresh literal; crossmodal_curvature.D_SWEEP unedited |
| `physics_curvature_probe.py` | `FIT_QUALITY_KEYS` | `('var_explained', 'cond_g_median')` | D9-12, required at every d |
| `physics_curvature_probe.py` | `INSTRUMENT_FIDELITY_RANGE_D16` | `(0.8376, 0.9882)` | 09-FIXTURE-FIDELITY-D16.md §3, measured 2026-09-02 |
| `physics_curvature_probe.py` | `INSTRUMENT_FIDELITY_RANGE_D20` | `(0.53, 0.99)` | 07-CONTEXT.md §4 / HANDOFF-v1.1.md §5.3 |
| `physics_curvature_probe.py` | `INSTRUMENT_FIDELITY_RANGE_D25` | `(0.17, 0.97)` | HANDOFF-v1.1.md §5.3, plan 08-07 |
| `physics_curvature_probe.py` | `INSTRUMENT_FIDELITY_D32_RULE` | `"d=32 fixture fidelity is NOT measured and cannot be measured with ...` | 09-FIXTURE-FIDELITY-D16.md §5; see value |
| `physics_curvature_probe.py` | `CURVATURE_FIELD_FOR_VERDICT` | `'H_tan_norm'` | D9-11; exact-equality guarded, module's own required text is "H_tan_norm" (corrected from plan prose "H_tan" — see Deviations in `09-05-SUMMARY.md`) |
| `physics_curvature_probe.py` | `H_NORM_IS_NON_GATING` | `True` | D9-11 |
| `physics_curvature_probe.py` | `RADIAL_DECOMPOSITION_RULE` | `"decompose_radial_tangential, copying 08_radial_curvature_decomposi...` | D9-11, 08-DIAGNOSTICS.md §2 2.8x collapse; see value |
| `physics_curvature_probe.py` | `MIN_IMAGE_NORM` | `1e-12` | Excluded rather than divided by |
| `physics_curvature_probe.py` | `ALPHA_RIDGE` | `100.0` | D9-13, his METHODS_FOR_PAPER.md §9 value |
| `physics_curvature_probe.py` | `ALPHA_GRID` | `(100.0,)` | D9-13, diagnostic only, never used to select ALPHA_RIDGE |
| `physics_curvature_probe.py` | `ALPHA_SELECTION_RULE` | `"alpha_grid passed to oof_ridge_predictions holds exactly one DISTI...` | Exact-equality guarded; see value |
| `physics_curvature_probe.py` | `N_OOF_FOLDS` | `5` | D9-13, five-fold OOF |
| `physics_curvature_probe.py` | `OOF_FOLD_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `OOF_IMPLEMENTATION_RULE` | `'oof_ridge_predictions wraps linear_probe.fit_probe/predict_probe i...` | Exact-equality guarded; see value |
| `physics_curvature_probe.py` | `LOCAL_R2_RULE` | `"local_r2_panel computes, per anchor over its K_NEIGHBOURS neighbou...` | D9-13, METHODS_FOR_PAPER.md §10; see value |
| `physics_curvature_probe.py` | `MIN_FINITE_NEIGHBOURS` | `32` | 1.6% of K_NEIGHBOURS, discretion decision |
| `physics_curvature_probe.py` | `CONTROLS` | `('log_knn_radius', 'local_label_variance', 'local_evaluation_count')` | D9-09, ordered 3-control tuple |
| `physics_curvature_probe.py` | `VERDICT_STATISTIC` | `"controlled_partial(H_tan_norm, local_oof_r2, Z=[log_knn_radius, lo...` | D9-09; see value |
| `physics_curvature_probe.py` | `RAW_RHO_IS_NON_GATING` | `True` | D9-09 |
| `physics_curvature_probe.py` | `STRATIFIED_NULL_IS_NON_GATING` | `True` | D9-09/07.1 |
| `physics_curvature_probe.py` | `STRATIFICATION_FIELD` | `'log_knn_radius'` | Discretion decision, matches 09-COLLEAGUE-REANALYSIS.md construction |
| `physics_curvature_probe.py` | `STRATA_GRID` | `(10, 20)` | Matches 09-COLLEAGUE-REANALYSIS.md S=10/S=20 |
| `physics_curvature_probe.py` | `STRATIFIED_NULL_DRAWS` | `5000` | Matches 09-COLLEAGUE-REANALYSIS.md draw count |
| `physics_curvature_probe.py` | `STRATIFIED_NULL_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `N_PERMUTATIONS` | `10000` | D9-18 discretion, inherited from colleague's B=10^4, not reduced |
| `physics_curvature_probe.py` | `PERMUTATION_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `NULL_CONSTRUCTION_RULE` | `"The null for the 3-control partial is Freedman-Lane: freedman_lane...` | Exact-equality guarded; see value |
| `physics_curvature_probe.py` | `FWER_ALPHA` | `0.05` | D9-10, standard 0.05 level |
| `physics_curvature_probe.py` | `P_VALUE_FLOOR_RULE` | `"p_value_from_null never reports a zero p; when the observed statis...` | D9-18, "p < 1/(B+1)", never p=0 |
| `physics_curvature_probe.py` | `N_BOOTSTRAP` | `2000` | Matches colleague's B=2000 paired anchor resamples |
| `physics_curvature_probe.py` | `BOOTSTRAP_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `BOOTSTRAP_RULE` | `"paired_anchor_bootstrap resamples anchor ROWS with replacement, ca...` | See value |
| `physics_curvature_probe.py` | `REPORT_BOTH_NULLS_UNCONDITIONALLY` | `True` | D9-09/roadmap |
| `physics_curvature_probe.py` | `VERDICT_RULE` | `'D9-10 VERDICT_RULE -- frozen in committed source before any Physic...` | D9-10, transcribed in full below |
| `physics_curvature_probe.py` | `VERDICT_VALUES` | `('REPLICATES AT EVERY d', 'REPLICATES AT SUBSET OF d', 'DOES NOT RE...` | D9-10, four phase-verdict strings (see Deviations in `09-05-SUMMARY.md` re: fourth entry) |
| `physics_curvature_probe.py` | `PER_D_VERDICT_VALUES` | `('NEGATIVE AND CLEARS FWER NULL', 'DOES NOT CLEAR')` | D9-10, two per-d verdict strings (see Deviations in `09-05-SUMMARY.md` re: dropped third entry) |
| `physics_curvature_probe.py` | `VERDICT_SENTENCE_RULE` | `"verdict_sentence must name the instrument (cae.PlainAutoEncoder + ...` | D9-10/D8-21 pattern; transcribed in full below |
| `physics_curvature_probe.py` | `REPORTING_BLOCK_ROWS` | `('raw_rho', 'controlled_partial', 'fwer_p_display', 'stratified_nul...` | D9-18 unconditional reporting block |
| `physics_curvature_probe.py` | `REPORTING_BLOCK_RULE` | `'Every row named in REPORTING_BLOCK_ROWS is printed and written to ...` | D9-18 |
| `physics_curvature_probe.py` | `POSITIVE_CONTROL_TARGET_RHOS` | `(0.05, 0.1, 0.15, 0.2, 0.25)` | D9-14, straddles colleague's -0.240 |
| `physics_curvature_probe.py` | `POSITIVE_CONTROL_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `POSITIVE_CONTROL_RULE` | `"The plant is on the curvature side, spread-matched to the realized...` | D9-14; see value |
| `physics_curvature_probe.py` | `SHUFFLED_LABEL_REPEATS` | `20` | D9-15 discretion |
| `physics_curvature_probe.py` | `SHUFFLED_LABEL_SEED` | `20260902` | Phase 9 shared seed (see above) |
| `physics_curvature_probe.py` | `SHUFFLED_LABEL_RULE` | `'shuffled_label_repeat performs a global row shuffle of the label v...` | D9-15; see value |
| `physics_curvature_probe.py` | `SEED_HANDLING_RULE` | `'no_pooling_per_seed_verdicts'` | D9-17, exact-equality guarded, 05-03-DECISION.md do-not-pool |
| `physics_curvature_probe.py` | `TORCH_INIT_SEEDS_WAVE_B` | `(0, 1, 2)` | D9-17, matches Phase 8's own seed axis |
| `physics_curvature_probe.py` | `SEED_VERDICT_COMBINATION_RULE` | `"Wave B runs three torch init seeds (TORCH_INIT_SEEDS_WAVE_B) at ev...` | D9-17; see value |
| `physics_curvature_probe.py` | `WAVE_B_TRIGGER_RULE` | `'Wave B (the three-seed sweep, TORCH_INIT_SEEDS_WAVE_B) runs only a...` | D9-17; see value |
| `physics_curvature_probe.py` | `PREREGISTRATION_FREEZE_RULE` | `'The freeze commit -- the commit that fills every constant in this ...` | D9-18, adapted from crossmodal_curvature.py wording |
| `physics_curvature_probe.py` | `RECORD_STEM` | `'09_physics_curvature'` | D9-18 record stem |
| `physics_curvature_probe.py` | `RECORD_LOCATION_RULE` | `"The frozen record is written under resolve_output_root() via recor...` | D9-18; see value |
| `physics_curvature_probe.py` | `OUTPUT_ROOT_ENV_VAR` | `'EFFDIM_09_OUTPUT_ROOT'` | Execution-host override knob (09-06) |
| `physics_curvature_probe.py` | `EXECUTION_HOST_RULE` | `"No real number is produced on the developer's machine (09-CONTEXT....` | 09-06 hand-off; see value |
| `physics_curvature_probe.py` | `SWISS_ROLL_APPLICABILITY_RULE` | `"CLAUDE.md's Swiss roll standing rule introduces no new notebook in...` | CLAUDE.md standing rule; filled at 09-01 (non-gating declarative fact) |

103 constants total: 30 in `physics_labels.py`, 73 in `physics_curvature_probe.py` (including the
one, `SWISS_ROLL_APPLICABILITY_RULE`, that was already filled at 09-01 as a non-gating declarative
fact rather than adjudicated by this freeze).

## `VERDICT_RULE` (verbatim, from `physics_curvature_probe.py` at the freeze commit)

```
D9-10 VERDICT_RULE -- frozen in committed source before any Physics probe
number existed (D9-18).

"Replicates" at a given d in D_SWEEP means the controlled 3-control partial
(VERDICT_STATISTIC, on CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm") is STRICTLY NEGATIVE
(rho < 0.0) AND clears its own Freedman-Lane rank-permutation null under the family-wise
envelope (the per-draw maximum absolute controlled partial across D_SWEEP) at FWER_ALPHA = 0.05,
using a strict < on p_fwer. No magnitude threshold. Magnitude is printed beside the colleague's
-0.240 with both bootstrap bands (his B=2000 paired anchor resamples; ours the same, N_BOOTSTRAP
= 2000).

Per-d cells are reported independently -- PER_D_VERDICT_VALUES[0] ("NEGATIVE AND CLEARS FWER
NULL") or PER_D_VERDICT_VALUES[1] ("DOES NOT CLEAR") -- with NO pooled headline number across d.
The phase verdict then aggregates the per-d cells: every d fired gives VERDICT_VALUES[0]
("REPLICATES AT EVERY d"), at least one but not all gives VERDICT_VALUES[1] ("REPLICATES AT
SUBSET OF d"), none gives VERDICT_VALUES[2] ("DOES NOT REPLICATE"). VERDICT_VALUES[3]
("HALTED - ALIGNMENT NOT PROVED") is reserved for the case where the D9-06/D9-07 row-alignment
proof itself never clears ALIGNMENT_MARGIN_R2 at any candidate offset (D9-08's SEARCH branch
finding zero or more-than-one clearing shift) -- in that case no Physics number is ever computed
and phase_verdict() is never called.

The raw (uncontrolled) rho and the density-stratified null (STRATIFICATION_FIELD =
"log_knn_radius") are both reported unconditionally beside the headline (REPORT_BOTH_NULLS_
UNCONDITIONALLY = True) but neither gates the verdict alone (RAW_RHO_IS_NON_GATING,
STRATIFIED_NULL_IS_NON_GATING). ||H_norm|| is reported beside ||H_tan_norm|| and never promoted
to the headline (H_NORM_IS_NON_GATING).

Every-d and magnitude-band alternative rules were considered and not chosen. Vocabulary follows
07.1's SURVIVES AT SUBSET OF d / per-d independent reporting (D8-13 pattern).
```

## `VERDICT_SENTENCE_RULE` (verbatim, from `physics_curvature_probe.py` at the freeze commit)

```
verdict_sentence must name the instrument (cae.PlainAutoEncoder + decoder_curvature.plain_decoder_curvature), the d values in D_SWEEP, the colleague's own -0.240 at his d=16 beside this phase's measured value at each d, both nulls (Freedman-Lane FWER p and density-stratified p, both as p_display strings per P_VALUE_FLOOR_RULE), the instrument-fidelity ranges INSTRUMENT_FIDELITY_RANGE_D16/D20/D25 including the unmeasured d=32 (INSTRUMENT_FIDELITY_D32_RULE), and the neighbourhood ratio (NEIGHBOURHOOD_RATIO_RULE) -- and must say 'reproduces the same sign under a different, differently-validated instrument', never 'confirms' (D9-10, D8-21's caveat-bearing pattern).
```

## What is explicitly NOT frozen, and why

- **Test tolerances** (e.g. `RTOL_DECOMPOSITION`, `ATOL_PARITY`, `ATOL_TARGET_RHO`,
  `FRAC_SPHERE_TOLERANCE` in `test_physics_curvature_probe.py`). These are test-local literals
  governing how tightly a unit test compares a computed value to a known answer — D9-18 enumerates
  what the freeze covers, and a numerical comparison tolerance is in none of those categories.
- **Print formatting** (e.g. `p_display`'s `.3e`/`.6g` format specifiers, JSONL key ordering). No
  Phase 9 verdict depends on how a number is rendered to a terminal or a file.
- **Runner mode names** (`--mode smoke|manifest|proof|search|dsweep|...`). These are CLI ergonomics
  added incrementally by each plan (09-01 through 09-09); none of them is a value the freeze
  adjudicates, and none is read by `assert_preregistered()`.

## Sources cited by date

- `09-DATA-MANIFEST.md` § Ruling (Section 7): developer ratification `ratify-as-proposed`,
  received 2026-09-03 UTC, applied verbatim as `LABEL_COLUMN_MAP`, `SENTINEL_VALUES` and
  `ALIGNMENT_MARGIN_R2` above.
- `09-DATA-MANIFEST.md` Sections 1-3 (measurement): full-scale manifest run, measured 2026-09-02,
  recorded `notebooks/.cache/09_data_manifest.jsonl` — `EXPECTED_N_PHYSICS_ROWS = 86471`.
- `09-FIXTURE-FIDELITY-D16.md` § 3 (measurement): `d=16` fixture sweep, measured 2026-09-02 —
  `INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)`.

## Flagged assumptions this phase carries forward

From `09-01-PLAN.md`'s `<flagged_assumptions>`:

- **`stellar_mass` -> `mass_med_photoz`** (09-RESEARCH.md Assumption A1) — was `[ASSUMED]`, now
  **`[RATIFIED 2026-09-03]`** per the `09-04` Task 2 checkpoint (`09-DATA-MANIFEST.md` Section 7).
  Non-gating either way (D9-16).
- **`mag_r` -> `mag_r_desi`** (09-RESEARCH.md Data Section) — ratified alongside the above at the
  same checkpoint. **Gating** — this is the primary label.
- **`Smith42/galaxies@v2.0` test split has exactly 86,471 rows** (Assumption A2) — measured for
  real by `09-03`/`09-04`'s `--mode manifest` run; confirmed exactly (`09-DATA-MANIFEST.md`
  Section 2), no mismatch, no halt.
- **`ALIGNMENT_MARGIN_R2 = 0.10`** — the planner's value for D9-07's "pre-registered margin", with
  no measured precedent; reasoning recorded in `09-05-PLAN.md`'s `<discretion_decisions>` and
  ratified unchanged at the `09-04` Task 2 checkpoint.

## Closing rule

A later edit to any constant listed above, after a Physics number exists anywhere in the tree
(`notebooks/.cache/09_row_alignment.jsonl`, `notebooks/.cache/09_physics_curvature.jsonl`, or any
downstream artifact derived from either), is a pre-registration BREACH. The only remedy is a fresh
freeze commit, a numbered amendment document superseding this one in full, and a complete re-run
of the execution-host workload affected by the changed value — never a silent fix, and never a
partial re-run that leaves a pre-change number standing beside a post-change one.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Plan: 09-05, Task 1*
