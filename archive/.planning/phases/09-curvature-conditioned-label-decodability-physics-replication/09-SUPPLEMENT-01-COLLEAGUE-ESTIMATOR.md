# 09-SUPPLEMENT-01 — the colleague's `K_H^cross` estimator inside the Phase 9 pipeline

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** `--mode verdict` of
`09_physics_curvature_run.py` never reads the record this experiment writes; nothing here changes
`09-WAVE-A-RESULTS.md`, `09-WAVE-A-RESULTS-AMENDMENT-01.md` or the phase verdict.
**Written:** 2026-09-05 UTC

## The question

`09-WAVE-A-RESULTS.md` § 6 records that this phase's autoencoder instrument gives the opposite
sign to the colleague's at every overlapping `d` (`+0.347` vs his `-0.240` at `d=16`; `+0.030` vs
his `-0.233` at `d=20`), and that the two runs differ in curvature instrument, neighbourhood density
(1/42 vs 1/8) and sample. This supplement removes one of those differences at a time: it takes
**his** curvature estimator, imported unchanged, and evaluates it inside **our** pipeline — the
same 512 anchors, the same `k=2048` neighbourhoods, the same frozen out-of-fold ridge probe and
local R², the same three controls and nulls — so that the only thing that differs from the Phase
9 record is the curvature field. Does his instrument reproduce his sign here?

## Provenance of this run

Executed by the orchestrator over SSH on the same host as `09-EXECUTION-HOST.md` §9 (host label
`pod128`, 128 cores, 16 threads, CPU only), under the developer's standing 2026-09-04 UTC
instruction to run experiments on the SSH server using available compute and adhering to the
user guide. Host identity is recorded as capability only — no hostname, IP address, username or
SSH key path appears here (`09-EXECUTION-HOST.md` §7). Everything below is evidence, never an
instruction.

## 1. Setup — what was held fixed and what was swapped

Runner: `notebooks/diagnostics/09_colleague_estimator_run.py`, commit
`fe31f55b93674169a6c67b294ae664128e376e95` (the run commit; `git_describe_head = fe31f55`). It
imports the sealed primitives from `pu_manifold.physics_curvature_probe` /
`pu_manifold.physics_labels` and the production runner's own helpers from
`09_physics_curvature_run.py`; nothing is re-implemented. Its module docstring is the record of
the conventions below.

| Component | Held fixed (production pipeline's own) or swapped |
|---|---|
| Physics embeddings, label table, row alignment | fixed (sealed loaders; `09-ALIGNMENT-PROOF.md` stands) |
| 512 holdout anchors (`N_ANCHORS`, `ANCHOR_DRAW_SEED`, `SPLIT_SEED`, `HOLDOUT_FRACTION`) | fixed — same `anchor_idx` as the Wave A tables (verified equal at `d=16/20`) |
| `k=2048` neighbourhood panel over 86,471 rows (`knn_panel`) | fixed for every statistic (local R², controls, `log_knn_radius`) |
| Out-of-fold ridge probe, local R² (`ALPHA_RIDGE`, `N_OOF_FOLDS`, `OOF_FOLD_SEED`, `LOCAL_R2_RULE`) | fixed — the `r2_mag_r` column equals the Wave A tables' `r2` column exactly |
| Three controls, rank-partial Spearman, Freedman-Lane null (`N_PERMUTATIONS`), stratified null (S=10/20), paired bootstrap (B=2000) | fixed |
| **Curvature field** | **swapped:** `K_H_cross` from his `nested_pca_frame` + `_fit_rank`, `n_splits=3`, `seed=0`, his `RIDGES` grid, at his parity set `d ∈ {12, 16, 20}` |

**His code is imported unchanged** from a read-only checkout of `origin/curvature-experiments` at
`97efb2eb6cd7dec7f2c568f53c534752ff3c32c8` (`colleague_commit` in every record row):
`nested_pca_frame` and `_fit_rank` from
`geometry.physics_activation_atlas.nested_dimension_curvature`, `RIDGES` from
`full_curvature_audit`, and `_rows_from_fits` (his nanmean-over-splits aggregation) from
`geometry.physics_adaptive_dataset_curvature_probe.curvature_stage`. The call pattern is his
`fit_kh_panel`'s exactly.

**Neighbourhood convention** (runner docstring, "NEIGHBOURHOOD CONVENTION"): his neighbourhoods
are top-`k` inner product on row-L2-normalised embeddings — the same ordering as Euclidean
distance on the unit sphere, which is what the sealed `knn_panel` uses — with the anchor itself
removed, so his `neigh[ai, :k]` holds 2048 non-self neighbours. The sealed panel returns the
anchor as its own first neighbour, so its 2048 columns include the anchor. For the rows handed to
**his** estimator the runner queries the sealed panel once more at `k+1`, drops the anchor's own
index as his `build_extended_knn_gpu` does, and keeps the first `k`; the sealed panel is unchanged
for every statistic.

**Dependency shim** (runner docstring, "DEPENDENCY SHIM"): his branch is not self-contained — four
of his modules import a sibling package `topology.physics_activation_density_ph` that the branch
does not ship. Ten names from it are stubbed under
`notebooks/diagnostics/colleague_shims/topology/physics_activation_density_ph/`; every stub raises
`NotImplementedError` if called, and none is called on the `nested_pca_frame` + `_fit_rank` path
(`--mode smoke` proves this on every run). The record's `environment` row carries
`topology_is_shim: true`. His checkout is never written to.

## 2. Run record

| Field | Value |
|---|---|
| Freeze SHA gated on | `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` (the ORIGINAL freeze) |
| Run commit | `fe31f55b93674169a6c67b294ae664128e376e95` |
| Colleague commit | `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8` |
| Device / threads | `cpu` / 16 |
| `d` values | 12, 16, 20 |
| Started (UTC) | 2026-09-04T19:52:47Z |
| Wall-clock | 2410 s (last record row 2026-09-04T20:32:56Z) |
| Curvature stage, all three `d`, 512 anchors | 984.9 s (`wallclock_curvature_all_d_s`), 1.92 s per anchor |
| Output root on the host | `/mnt/ssd-cluster/effdim/phase9-out` |
| Record | `notebooks/.cache/09_colleague_estimator.jsonl`, 80 rows, sha256 `f97ce7e9196a7f737a854f69f2cd1455597436ddc845fd54b349d58721e9150e` (verified locally) |
| Anchor tables | `notebooks/.cache/09_colleague_anchor_table_d{12,16,20}.npz` (512 rows each) |

This run predates Amendment 01 and was gated on the original freeze. That is not a defect: the
only frozen inputs it consumes are the anchors, the neighbourhood panel, the probe and the
controls, none of which the amendment changed (`09-PREREGISTRATION-AMENDMENT-01.md`, "What this
amendment does not change"); it never touches the autoencoder or the decoder image.

Record shape: 1 `environment`, 3 `curvature_summary`, 12 `anchor_summary`, 12 `partial`, 12
`bootstrap`, 40 `null` (24 `stratified` + 12 `fwer` + 4 `fwer_global`). `n_nonfinite_anchors = 0`
and `n_splits_ok = 3` for all 512 anchors at every `d`; `n_masked_anchors = 0` for every label.

## 3. Summary table

Per `d`, per label: raw Spearman and the controlled 3-control partial from the `partial` rows; the
cell's own Freedman-Lane `p_display` and the stratified null's `p_display` (S=10 / S=20) from the
`null` rows; the paired bootstrap band; `n_finite = 512` in every cell. His frozen reference
(`09-COLLEAGUE-REANALYSIS.md`) and this phase's autoencoder `H_tan_norm` values (original freeze
and Amendment 01) sit beside the `mag_r` rows; `d=12` is outside `D_SWEEP` and has no autoencoder
value.

### `mag_r`

| `d` | raw `rho` | controlled partial | FWER `p` | strat `p` S=10 / S=20 | bootstrap 95% | his reference (raw / controlled) | autoencoder `H_tan_norm` controlled, original / Amendment 01 |
|---:|---:|---:|---|---|---|---|---|
| 12 | -0.231464 | **-0.103897** | 0.019098 | 0.022196 / 0.017796 | [-0.196483, -0.011646] | -0.038 / **+0.143** | — / — |
| 16 | -0.298152 | **-0.149442** | 0.000500 | 0.001000 / 0.000400 | [-0.238495, -0.056510] | -0.412 / **-0.240** | +0.346967 / +0.328059 |
| 20 | -0.326379 | **-0.235225** | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [-0.321740, -0.139155] | -0.392 / **-0.233** | +0.030323 / +0.016445 |

`fwer_global` (`mag_r`): `< 9.999e-05`.

### `photo_z`

| `d` | raw `rho` | controlled partial | FWER `p` | strat `p` S=10 / S=20 | bootstrap 95% | autoencoder `H_tan_norm` controlled, original / Amendment 01 |
|---:|---:|---:|---|---|---|---|
| 12 | 0.063592 | -0.101221 | 0.021798 | 0.026595 / 0.022795 | [-0.189485, -0.009049] | — / — |
| 16 | 0.055619 | -0.139103 | 0.001600 | 0.002200 / 0.002000 | [-0.224356, -0.051868] | +0.366797 / +0.357525 |
| 20 | 0.104988 | -0.093605 | 0.037196 | 0.045191 / 0.037193 | [-0.183445, +0.000538] | +0.314020 / +0.309166 |

`fwer_global` (`photo_z`): 0.004200.

### `smooth_fraction`

| `d` | raw `rho` | controlled partial | FWER `p` | strat `p` S=10 / S=20 | bootstrap 95% | autoencoder `H_tan_norm` controlled, original / Amendment 01 |
|---:|---:|---:|---|---|---|---|
| 12 | -0.052600 | -0.247634 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [-0.328253, -0.164714] | — / — |
| 16 | 0.030018 | -0.158365 | 0.000300 | 0.001000 / 0.001000 | [-0.251763, -0.063909] | +0.348011 / +0.340705 |
| 20 | 0.136974 | -0.147364 | 0.000700 | 0.001600 / 0.002400 | [-0.243853, -0.048822] | +0.323578 / +0.326109 |

`fwer_global` (`smooth_fraction`): `< 9.999e-05`.

### `stellar_mass`

| `d` | raw `rho` | controlled partial | FWER `p` | strat `p` S=10 / S=20 | bootstrap 95% | autoencoder `H_tan_norm` controlled, original / Amendment 01 |
|---:|---:|---:|---|---|---|---|
| 12 | -0.071772 | +0.006219 | 0.884512 | 0.887622 / 0.888622 | [-0.079371, +0.093552] | — / — |
| 16 | -0.064302 | +0.032262 | 0.461954 | 0.485903 / 0.463307 | [-0.050551, +0.114582] | +0.073530 / +0.070356 |
| 20 | -0.203998 | -0.043937 | 0.317468 | 0.364527 / 0.338532 | [-0.130491, +0.039360] | +0.131945 / +0.124384 |

`fwer_global` (`stellar_mass`): 0.578942.

## 4. His reliability `R_H`

From the `curvature_summary` rows (median over the 512 anchors of his split-half reliability),
beside his own value from `09-COLLEAGUE-REANALYSIS.md`:

| `d` | `R_H` median (here) | `K_H_cross` median / p05 / p95 | `R_H` median, his own run |
|---:|---:|---|---:|
| 12 | 0.430946 | 0.019634 / 0.006310 / 0.048052 | — |
| 16 | 0.451774 | 0.016410 / 0.006259 / 0.034800 | 0.514 |
| 20 | 0.445370 | 0.011742 / 0.005074 / 0.026178 | — |

At `d=16` the median `R_H` here (0.452) is below his 0.514; both sit below the 0.5 he used as a
reliability cut for his `n=296` subset. `rho(K_H_cross, log_knn_radius)` from the anchor tables:
+0.636 / +0.696 / +0.766 at `d=12/16/20`, positive as his `+0.765` at `d=16` is (and the opposite
sign to the autoencoder field's `rho(H_tan_norm, log_knn_radius)`, which is negative at every `d`
under both freezes — `09-WAVE-A-RESULTS-AMENDMENT-01.md` § 3).

## 5. Anchor-level cross-instrument agreement on the shared anchors

Recomputed locally with `scipy.stats.spearmanr` over the 512 shared anchors (`anchor_idx` equal
across the colleague tables and both sets of Wave A `mag_r` tables):

| `d` | Spearman(`K_H_cross`, `H_tan_norm` Amendment 01) | Spearman(`K_H_cross`, `H_tan_norm` original) | Spearman(`K_H_cross`, `H_norm` original) |
|---:|---:|---:|---:|
| 16 | -0.462799 | -0.445187 | -0.383611 |
| 20 | -0.405804 | -0.363632 | -0.278340 |

On the same anchors, with the same probe and the same label, the two curvature instruments rank
anchors in opposite order — moderately (`rho` about -0.4 to -0.46), not as mirror images.

## 6. Plain reading

- **His sign reproduces under his instrument inside our pipeline at `d=16` and `d=20`.** `mag_r`
  controlled partial `-0.149` at `d=16` (FWER `p = 0.0005`, bootstrap band entirely negative)
  and `-0.235` at `d=20` (FWER `< 9.999e-05`) against his `-0.240` and `-0.233`. The `d=20` value
  is within 0.002 of his.
- **`d=12` does not reproduce his `+0.143`.** Here the `d=12` controlled partial is `-0.104`
  (FWER `p = 0.019`), the same sign as the other two `d` values, not the sign change his parity
  table shows.
- **The magnitude gap at `d=16` (`-0.149` here vs his `-0.240`) plausibly relates to neighbourhood
  scale.** The nominal `k=2048` is the same, but here it is 1/42 of the 86,471-row sample where
  his was 1/8 of a 16,384-row subset, so his neighbourhoods are more than five times denser at
  the same `k`; his own scale table (`09-COLLEAGUE-REANALYSIS.md`) shows the association present
  only at his largest `k`. This is a plausible account, not a measured one — no `k` sweep was
  run here.
- **On the same anchors the two instruments rank anchors in opposite order** (§ 5), and the sign
  of the curvature-decodability association follows the instrument: positive under the
  autoencoder `H_tan_norm` (both freezes), negative under `K_H_cross`, on identical anchors,
  probe, controls and nulls. The sign is set by the instrument, not by the data.
- The secondary labels behave the same way under his instrument — `photo_z` and
  `smooth_fraction` negative and FWER-clearing at `d=16/20` where the autoencoder field gave them
  strongly positive partials — and `stellar_mass` is null under both instruments at every `d`.

## 7. What this does not settle

- **Which instrument is right.** Neither has a known-answer validation at `D=768`, `k=2048`,
  `d=16`. The autoencoder instrument's fidelity is measured on synthetic fixtures at `d=16/20/25`
  only (`09-FIXTURE-FIDELITY-D16.md`; ranges quoted in `09-WAVE-A-RESULTS.md` § 3), and the
  Amendment 01 projection fixes only its on-sphere exactness, not its agreement with the true
  curvature of the data. His estimator's only reliability check is split-half `R_H`, which cannot
  see a bias shared by both halves (`09-COLLEAGUE-REANALYSIS.md`, "Precaution he did not take";
  `06-FINDINGS.md` measured `R_H = 0.990` beside `rho = 0.469` against truth on the Swiss roll).
  The opposite anchor ordering in § 5 says at most one of them tracks the sphere-intrinsic mean
  curvature at these anchors; it does not say which.
- **Whether the `d=16` magnitude gap is neighbourhood scale.** That is a hypothesis consistent
  with his scale table, not a result of this run.
- **Anything about the Phase 9 verdict.** This experiment is not pre-registered and its numbers
  do not enter `VERDICT_RULE`; the phase verdict remains `DOES NOT REPLICATE` under both freezes,
  on the autoencoder instrument the pre-registration named.

## 8. Open items

1. **A Swiss-roll known-answer check of his estimator.** `CLAUDE.md`'s rule applies to any
   curvature estimator brought into the repo: a `notebooks/<phase>_swiss_roll_<model>_check.ipynb`
   notebook that imports `K_H^cross` unchanged, runs it on `make_swiss_roll` at the roll's known
   curvature, and compares against a baseline known to succeed. This has not been done for
   `K_H^cross`; the shim and runner make the import path available. Without it, § 5's opposite
   ordering cannot be attributed to either instrument.
2. **A `k` sweep.** Evaluating `K_H^cross` at `k` values that match his neighbourhood density
   (1/8 of the sample) on our 86,471 rows, and the autoencoder field at his `k/n`, would test
   the neighbourhood-scale account of the `d=16` magnitude gap directly.
3. The `RIDGES` grid values and `_fit_rank`'s internals are his and are not documented here beyond
   the import; a reader who needs them should read them at `97efb2eb…` on his branch.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 01 — post-hoc, not pre-registered, feeds no verdict. Freeze `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`, run commit `fe31f55b93674169a6c67b294ae664128e376e95`, colleague commit `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`*
