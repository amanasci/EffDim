# C1: the paper's pipeline on the pod GPUs for the published five and the DINOv3 size ladder

Date: 2026-09-30
Branch: `c1-ladder` (from `encoder-scaling@0a4cb72`)

## Goal

For the ML4PS rebuttal (and the extended version), run the paper's battery on the pod GPUs for ten galaxy
encoders: the published five and the DINOv3 size ladder. It shows whether the paper's claims hold across
model size within one family, and it reproduces the published five on a second platform.

This is sub-project C1. It replaces the paused 31-encoder plan (`docs/superpowers/plans/2026-09-28-encoder-scaling.md`,
Task 7) and reuses its sweep machinery. C2 (QM9 molecules) follows with its own spec.

Two Opus design reviews shaped this spec. Their findings are folded in below.

Success criteria:

1. All 60 jobs (6 per encoder x 10 encoders) complete with validated records and done-markers.
2. Every `robust` job's reproduction guard passes in exact mode against its own encoder's `main_xfit`
   and `cf` outputs.
3. For the five published encoders, the sweep's global out-of-fold R^2 equals the published value.
   This is a data-identity check (embeddings, label table, folds, probe), not a GPU check: the value is
   computed before any geometry step. A difference below 1e-12 (BLAS thread-count rounding; the published
   runs used 16 threads) is reported and the gate passes; a difference of 1e-12 or more stops the run.
4. The report regenerates from the records: `.tex` and `.md` byte-for-byte (pinned by a test); figures
   identical between two runs in the same environment.
5. The CPU equivalence gate passes; `paper/latex/main.tex` is unchanged.

## Encoders

| Encoder | D | Params | Set |
|---|---|---|---|
| vit_base | 768 | 86,389,248 | published |
| clip_base | 512 | 86,192,837 | published |
| convnext_base | 1024 | 88,717,800 | published |
| vit_large | 1024 | 304,351,232 | published |
| dinov3_vits16 | 384 | 21,596,544 | ladder |
| dinov3_vits16plus | 384 | 28,692,864 | ladder |
| dinov3_vitb16 | 768 | 85,660,416 | published and ladder |
| dinov3_vitl16 | 1024 | 303,129,600 | ladder |
| dinov3_vith16plus | 1280 | 840,592,640 | ladder |
| dinov3_vit7b16 | 4096 | 6,716,035,072 | ladder |

Selected with a `--encoders` filter over the existing `curvature-experiment/encoders.yaml`; no new
manifest field.

## Battery (6 jobs per encoder)

| # | Job | What | Depends on |
|---|---|---|---|
| 1 | `main_xfit` | split runner `--fit-seed 0 --hessian-xfit --geometry-out` | — |
| 2 | `seed1` | split runner `--fit-seed 1` | — |
| 3 | `seed2` | split runner `--fit-seed 2` | — |
| 4 | `cf` | counterfactual runner on job 1's geometry | 1 |
| 5 | `thin` | thin runner on job 4's arrays | 4 |
| 6 | `robust` | `11_review_robustness_run.py` on job 1's geometry, alpha = 100 and alpha*, guard `exact` against job 1's record and job 4's arrays | 1, 4 |

Dropped from the original battery: `main` (duplicates `main_xfit`; `main_xfit` replaces it everywhere
aggregation read `main`), `w400`, `alpha1` (covered by `robust`'s tuned alpha), `d20`. Published CPU
d20, w400 and alpha1 results exist for ViT-B only; for every other encoder robustness is seed-only
(three decoder fits), and the report says so.

All GPU jobs pass `--device cuda --deterministic`. `cf`, `thin` and `robust` are CPU work that holds a GPU
slot while it runs; this is accepted (a separate CPU lane would add at most two jobs within the 30-core
budget).

Record names follow the sweep's layout: `records/scaling__<enc>__<suffix>.jsonl` for every job,
`arrays/scaling__<enc>__cf.npz` and `arrays/scaling__<enc>__thin.npz`, geometry under `geometry/<enc>/`.

## Code changes (additive to the CPU path)

1. **GPU geometry in forward mode.** `ppf.decoder_geometry` gains `forward_mode: bool = False`. True uses
   `vmap(jacfwd(f))` and `vmap(jacfwd(jacfwd(f)))` with the sealed `VMAP_CHUNK = 32`; False is the existing
   reverse-mode path, unchanged. The split runner passes `forward_mode=True` on CUDA only. This replaces
   commit ac13891's `out_chunk` (which does not bound memory; measured) and its tests and docstring.
2. **Exact guard on the GPU path.** On CUDA only, the split runner always casts the geometry to float32
   and rebuilds it with `geometry_from_arrays` before scoring (right after `decoder_geometry`, before the
   save), whether or not `--geometry-out` is given. All three seed jobs are therefore scored the same way,
   and the saved npz and the `main_xfit` record agree exactly. `cond_g`, dropped by the rebuild, is used
   nowhere downstream. When it saves geometry, the split runner records the npz's sha256 in its `fit` row.
   The CPU path is unchanged.
3. **Runner 11 in sweep use.** `--geometry-sha256` becomes optional (the hash is computed and recorded
   either way). A new `--published-cf-record` (the `cf` jsonl) gives the guard the cf run's `threads`;
   the guard fails if they differ from runner 11's, since the counterfactual is compared at 1e-12.
4. **Manifest.** A top-level `label_table_sha256` next to the existing top-level `label_table` in
   `encoders.yaml` (value `60f2f82e64e4036eb4eff9a448b15365dd1e974e52779fe7c1323db0d9dabfd9`, the table every
   published record names). No per-encoder field changes.
5. **Sweep.** `jobs.py`: `build_jobs(..., encoders=None)` filters by name; the 6-job battery above; the
   `robust` argv is `--encoder <enc> --geometry-npz <main_xfit npz> --parquet-path <enc parquet>
   --embedding-column <enc column> --label-table <m.label_table> --label-table-sha256 <m.label_table_sha256>
   --published-split records/scaling__<enc>__main_xfit.jsonl --published-cf arrays/scaling__<enc>__cf.npz
   --published-cf-record records/scaling__<enc>__cf.jsonl --guard exact --threads <n>
   --record-path records/scaling__<enc>__robust.jsonl`, with outputs `[records/scaling__<enc>__robust.jsonl]`.
   `run_queue.py`: a `--encoders a,b,c` flag passed to `build_jobs`; encoders ordered largest D first; the
   two thin-triggered prune sites also require the encoder's `robust` done (C1 runs with `--keep-geometry`
   regardless). Step 2's settings are `--encoders dinov3_vit7b16 --gpus <two free GPUs> --threads 12`
   (the three split jobs share two GPUs; the CPU jobs then run at 12 threads each).
6. **Aggregation.** `JOB_SUFFIXES` becomes the 6-job list; `SPLIT_JOBS = (main_xfit, seed1, seed2)`
   (`robust` is not a split job: its rows have `partials`, not `columns`); every read of `rows["main"]`
   reads `rows["main_xfit"]`; `ROBUST_VARIANTS = (main_xfit, seed1, seed2)` with the robust-table caption
   rewritten as a seed spread. New: the DINOv3 size table; the published-versus-GPU table (below); a
   freshness check (the geometry sha256 in `main_xfit`'s `fit` row must equal the one in `cf`'s and
   `robust`'s environment rows, else the encoder is flagged stale in the report). Aggregation reads the
   published CPU records only for the comparison table, from `--published-dir` (default `notebooks/.cache`,
   read-only); every other number comes from the sweep records.

## Published-versus-GPU comparison (fixed before any result is seen)

Published records (read-only, `notebooks/.cache/`): split `09_physics_probe_facing_split_<enc>.jsonl` for
dinov3_vitb16, clip_base, convnext_base and vit_large (fit seed 0); for vit_base the unsuffixed
`09_physics_probe_facing_split.jsonl`, which was scored on the stored Supplement-06 geometry (fit seed
recorded as null), so it counts as ViT-B's seed-0 value although it is not a seed-0 refit; ViT-B's seeds 1
and 2 are `09_physics_probe_facing_split_seed1.jsonl` and `_seed2.jsonl`. Counterfactual:
`09_physics_normal_scaling_<enc>_d16.npz` and `_d16_thin.npz`. All at d = 16 rows only.

Split cells: for each label and each of the two columns (the multiscale partial of `hess_mismatch_emp` and
of `align_cos_tan`), compare the published value with the sweep's `main_xfit`. A cell agrees when the sign
matches, the significance at 0.05 matches, and |GPU - CPU| is at most the largest absolute difference among
ViT-B's published seeds 0, 1, 2 for that label and column (the only CPU seed spread that exists; the report
says it is used for all five encoders). A cell with 0.01 < p < 0.1 on either side is marked borderline and
counted separately, not as a disagreement. A missing published value prints `--` and is left out of the
counts.

Counterfactual cells: boolean agreement, per label, on S_model help > 0.5, S_model hurt > 0.5 and the
thinned sign test p_help < 0.05 (there is no CPU seed spread for the counterfactual).

This table is reported, not gating.

## What the ladder can claim

Six sizes of one family and one training recipe (21M to 6.7B parameters). It supports: the sign and
significance of the mismatch and alignment partials, and the counterfactual help/hurt pattern, hold across
sizes. It does not support "the effect scales with size" (D changes with size at fixed d = 16, and n = 6).
The size table reports, per label and column, the `main_xfit` multiscale partial of mismatch and alignment,
the seed range (min-max over the three fits), the counterfactual help and hurt, and the tuned-alpha
partials from `robust`, for the six DINOv3 sizes; the Spearman correlation with log parameters is computed
per (label, column) over those six points and labelled descriptive.

## Run order on the pod (each gate must pass before the next)

1. `setup_pod.sh` on `encoder-scaling`/`c1-ladder`; cgroup memory limit checked;
   `TMPDIR=/mnt/ssd-cluster/EffDim/tmp`. GPU checks: memory diagnostic with the forward path at D = 4096;
   GPU forward versus CPU reverse on a real decoder at D = 768 (max absolute difference of `Hess` and `J`
   <= 1e-10); GPU determinism smoke twice,
   `IDENTICAL`, confirmed to take the forward branch.
2. `dinov3_vit7b16` alone (settings in Code change 5): `main_xfit`, `seed1`, `seed2`, then `cf`, `thin`,
   `robust`. Record wall time and peak RSS of every job. Stop and report if `robust` exceeds 6 h or 60 GB
   RAM (one float64 II at D = 4096 is about 4.3 GB; `split_columns` holds II, II_tan and Hess together).
3. `dinov3_vitb16` end to end: the exact guard passes and global OOF R^2 matches the published value to
   1e-10 (gating); first published-versus-GPU cells.
4. The remaining 8 encoders: 8 GPUs, `--keep-geometry`, 3 threads per job, `--min-free-gb 10`, largest D
   first.
5. Fetch the records to `curvature-experiment/.cache/scaling/`, aggregate to `curvature-experiment/results/scaling/`,
   commit.

Pod rules (CLAUDE.md and the pod guide) bind every remote step.

## Estimates

D <= 1280: about 45-60 min of jobs per encoder; with 8 in parallel, steps 3-4 take about 2-3 h wall.
vit7b16: 3-6 h (about 44 large SVDs in `robust`). About a day in total. Kept geometry about 7 GB of the
25 GB free.

## Tests (TDD, local CPU)

1. Forward-mode geometry equals reverse mode within 1e-10 on the smoke decoder; the CPU default path is
   unchanged (gate).
2. The split runner selects forward mode on CUDA and reverse on CPU, and on CUDA scores from the
   float32-cast geometry (tested through a device-agnostic helper).
3. `jobs.py`: `--encoders` filter; the 6-job battery; `robust` depends on `main_xfit` and `cf`; `robust`
   argv as specified.
4. Runner 11 runs without `--geometry-sha256` and records the hash; its guard reads sweep-produced records
   (xfit rows ignored by `extract.split_cells`); a thread mismatch with the `cf` record fails the guard.
5. `run_queue.py`: pruning waits for `cf`, `thin`, `robust`; `--keep-geometry` never prunes; largest D first.
6. `aggregate.py`: the job list, the size table, the published-versus-GPU table and its agreement rule
   (including borderline and missing cells), the freshness check, and `.tex`/`.md` byte-for-byte
   regeneration.
7. Existing tests rewritten for the new battery, not deleted: `test_sweep_jobs.py` (counts `31*9`,
   `JOB_SUFFIXES[:7]`), `test_sweep_aggregate.py` (job lists and `main` reads), `test_device_flags.py`
   (the ac13891 `out_chunk` tests become forward-mode tests).

## Out of scope

QM9 (C2); encoders outside the ten; d = 20 and width 400 for the ladder; editing `main.tex`.
