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
3. For the five published encoders, the sweep's global out-of-fold R^2 equals the published value to 1e-10.
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

Dropped from the original battery: `main` (duplicates `main_xfit`), `w400`, `alpha1` (covered by
`robust`'s tuned alpha), `d20`. The published five keep their published CPU d20 and w400 results;
ladder robustness is seed-only, and the report says so.

All GPU jobs pass `--device cuda --deterministic`.

## Code changes (additive to the CPU path)

1. **GPU geometry in forward mode.** `ppf.decoder_geometry` gains `forward_mode: bool = False`. True uses
   `vmap(jacfwd(f))` and `vmap(jacfwd(jacfwd(f)))` with the sealed `VMAP_CHUNK = 32`; False is the existing
   reverse-mode path, unchanged. The split runner passes `forward_mode=True` on CUDA only. This replaces
   commit ac13891's `out_chunk` (which does not bound memory; measured) and its tests and docstring.
2. **Exact guard on the GPU path.** On CUDA only, the split runner scores its partials from the geometry
   rebuilt from the float32 arrays it saves (`geometry_from_arrays` of the cast arrays), so the saved npz
   and the record agree exactly. The CPU path is unchanged.
3. **Runner 11 in sweep use.** `--geometry-sha256` becomes optional (the hash is computed and recorded
   either way); the recorded `threads` must equal the `cf` record's threads, since the counterfactual is
   compared at 1e-12 (asserted by the guard).
4. **Sweep.** `jobs.py`: `--encoders` filter; the 6-job battery above; `robust` argv with
   `--published-split <main_xfit record> --published-cf <cf arrays> --guard exact --label-table-sha256 <pinned>`.
   `run_queue.py`: when pruning is on, prune only after `cf`, `thin` and `robust` are all done (C1 runs
   with `--keep-geometry`); encoders ordered largest D first.
5. **Aggregation.** The 6-job list; a DINOv3 size table (each quantity against log params, Spearman with
   size reported as descriptive); the published-versus-GPU table (below); freshness checks: `robust` and
   `cf` geometry sha256 must equal `main_xfit`'s done-marker's, else the encoder is flagged stale.

## Published-versus-GPU comparison (fixed before any result is seen)

For the five published encoders, per label and column (mismatch `hess_mismatch_emp`, alignment
`align_cos_tan`, multi-scale controls), a cell agrees when all three hold: same sign; same significance at
0.05; and |GPU - CPU| <= the largest absolute difference among the published CPU seeds 0, 1, 2 for that
label and column. Published CPU seed records exist for ViT-B only, so ViT-B's CPU seed spread is the
tolerance for all five, and the report says so. Counterfactual help/hurt signs (S_model help > 0.5, hurt
> 0.5) are compared the same way. This table is reported, not gating.

## What the ladder can claim

Six sizes of one family and one training recipe (21M to 6.7B parameters). It supports: the sign and
significance of the mismatch and alignment partials, and the counterfactual help/hurt pattern, hold across
sizes. It does not support "the effect scales with size" (D changes with size at fixed d = 16, and n = 6).

## Run order on the pod (each gate must pass before the next)

1. `setup_pod.sh` on `encoder-scaling`/`c1-ladder`; cgroup memory limit checked;
   `TMPDIR=/mnt/ssd-cluster/EffDim/tmp`. GPU checks: memory diagnostic with the forward path at D = 4096;
   GPU forward versus CPU reverse on a real decoder at D = 768 (<= 1e-10); GPU determinism smoke twice,
   `IDENTICAL`, confirmed to take the forward branch.
2. `dinov3_vit7b16` alone at 24 threads: `main_xfit`, `seed1`, `seed2`, then `cf`, `thin`, `robust`. Stop and
   report if `robust` exceeds 6 h or 60 GB RAM.
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
6. `aggregate.py`: the job list, the size table, the published-versus-GPU table and its agreement rule,
   the freshness check, and `.tex`/`.md` byte-for-byte regeneration.

## Out of scope

QM9 (C2); encoders outside the ten; d = 20 and width 400 for the ladder; editing `main.tex`.
