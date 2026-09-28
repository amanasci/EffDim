# Reproducing the records

This documents how the records `paper/generate/appendix_gen.py`, `paper/generate/table_main_gen.py`
and `paper/latex/figures/*.py` read were produced. Smoke mode (`--mode smoke`) runs the same code
paths at small `n` and is what `docs/superpowers/harness/gate.sh` checks; it does not reproduce
any paper number. Full physics mode needs the pod-only inputs below and hours of GPU/CPU time and
is not covered by any automated check in this repository.

## 1. Environment

```bash
pip install -r curvature-experiment/requirements.txt
```

Records are read from `EFFDIM_CACHE_DIR` if set, otherwise `curvature-experiment/.cache/`
(gitignored). The existing record store referenced throughout this document is about 8.7 GB; a
reader who has it should either set `EFFDIM_CACHE_DIR` to its path or symlink it into place:

```bash
ln -s /path/to/your/record/store curvature-experiment/.cache
# or
export EFFDIM_CACHE_DIR=/path/to/your/record/store
```

## 2. Inputs

- **Embeddings.** `UniverseTBD/pu-embeddings`, physics test split, files
  `physics/{vit_base,dinov3_vitb16,clip_base,convnext_base,vit_large}_test.parquet`, read via
  `hf://` or fetched locally with `huggingface_hub.hf_hub_download`. The snapshot used throughout
  begins `bc081f8a` (locally: `bc081f8a5db4767edcd958653d96efde9137de0b`, from
  `ls ~/.cache/huggingface/hub/datasets--UniverseTBD--pu-embeddings/snapshots/`).
  `pu_manifold/physics_labels.py`'s `PHYSICS_PARQUET_PATH` does not pin this revision.
- **Labels.** `Smith42/galaxies` at the pinned revision `v2.0`, split `test`, 16 shards
  (`pu_manifold/physics_labels.py`'s `LABEL_REPO`/`LABEL_REVISION`/`LABEL_SPLIT`/`LABEL_N_SHARDS`).
  The `hf://` read through `HfFileSystem` hung for some runs (Supplement 07); the 16 shards were
  instead fetched with `hf_hub_download`, column-projected and concatenated in shard order into a
  local parquet, `labels_Smith42_galaxies_v2.0_test.parquet` (86,471 rows, sha256
  `60f2f82e64e4036e…`, pod path `/mnt/ssd-cluster/effdim/labels_Smith42_galaxies_v2.0_test.parquet`),
  substituted via the runner's `--label-table` option. No script in this repository builds that
  parquet; it is a one-off `hf_hub_download` + concatenate, documented but not committed as code.
- **ViT-B per-anchor geometry.** `09_probe_facing_geometry_d{16,20}.npz` (Jacobian, Hessian, image
  and latent codes at the 512 sealed anchors), written by `09_physics_probe_facing_run.py` and
  read by `09_physics_probe_facing_split_run.py` via `--geometry-root` instead of refitting the
  decoder. Kept on the pod under `probe-facing-out/probe-facing/` (about 400 MB each); not
  transferred, and no sha256 for it is recorded in `paper/latex/COMPLIANCE.md` or any supplement.
- Analogous per-encoder geometry npz files exist for the four cross-encoder decoders
  (DINOv3 ViT-B/16, CLIP ViT-B, ConvNeXt-B, ViT-L) at `d=16`, used by
  `09_physics_normal_scaling_run.py` for those runs; same pod-only, no-sha256 status.

## 3. Invocations

Command lines below are copied from the archived phase supplements
(`archive/.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/`)
where a supplement gives one. Flags and structure are copied verbatim; paths are rewritten from
the supplement's original `notebooks/diagnostics/` to this repository's
`curvature-experiment/runners/`, and from `notebooks/.cache/` to `curvature-experiment/.cache/`.
Where the supplement itself left a placeholder inside the command (`09-SUPPLEMENT-11`'s
`<local parquet>`, `<enc>`, `<cached labels>`), §3.8 below expands it with the concrete value the
same supplement gives elsewhere in prose (the snapshot hash, the encoder name, the label
parquet's sha256), rather than reproducing the bare placeholder. No invocation below used
`--colleague-root` or `--skip-colleague`, so nothing was dropped on that account. Where a
supplement names a runner and settings in prose but does not give a full command line, that is
stated below instead of an invented one.

### 3.1 `09_instrument_adjudication.jsonl`

Source: `09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md`, §4 "Run record".

Invocation not recorded in the supplements as a literal command line. The supplement's own
description: runner `09_instrument_adjudication_run.py`, mode `sphere-fixture`, run twice — once
at `--noise 0` and once at `--noise patch` — at the fixture's frozen defaults (`seed 20260905`,
`D=768`, `d=16`, `k=2048`, `n=86,471`, 512 anchors), 16 threads, run commit `068490d34b1f200bf0ffc9ec69c55e0c6f6ebaeb`.
Backs the Section 2 "Validation" paragraph numbers (cosine 0.999, ratio 1.00, rank 0.94 — the
noiseless row) and the equivalent numbers quoted in Appendix E.

### 3.2 `09_fixture_probe_decodability.jsonl`, `09_fixture_probe_decodability_gamma2.jsonl`

Source: `09-SUPPLEMENT-03-FIXTURE-PROBE-DECODABILITY.md` (Provenance) for the first record;
`09-SUPPLEMENT-04-DENSITY-ISOLATION.md`'s Provenance list (~line 31, "Experiment 7") for the
second record's invocation, and its §8 for that record's results and sha256.

`09_fixture_probe_decodability.jsonl`: invocation not recorded in the supplements as a literal
command line. Runner `09_fixture_probe_decodability_run.py`, run commit `3c1260d640706e3c0dde714c020b83b07ff05b4e`,
gammas `-1,0,1` (the runner's own default), 16 threads.

`09_fixture_probe_decodability_gamma2.jsonl`: partial invocation recorded in the Provenance
list — `09_fixture_probe_decodability_run.py --gammas 0.4,0.6,0.8` — same generator, seed, anchors
and pipeline as the record above; full command line (mode, thread count, record path) not given
in the supplement. Together the two records cover `gamma in {-1, 0, 0.4, 0.6, 0.8, 1}`.

### 3.3 `09_fixture_probe_facing.jsonl`

Source: `09-SUPPLEMENT-05-PROBE-FACING-CURVATURE.md`, §2 "What was computed".

Invocation not recorded in the supplements as a literal command line. Runner
`09_fixture_probe_facing_run.py` (imports `09_fixture_probe_decodability_run.py` unchanged),
executed locally at 16 threads, exact
geometry only (no autoencoder, no colleague estimator), `gammas -1,0,0.4,0.6,0.8,1` (the union of
Supplements 03 and 04 §8, and the runner's own default). Backs `figF` panel (a).

### 3.4 `09_fixture_probe_facing_split.jsonl`

Source: `09-SUPPLEMENT-07-PROBE-FACING-SPLIT.md`, §1 "Fixture (exact geometry)".

Invocation not recorded in the supplements as a literal command line. Runner
`09_fixture_probe_facing_split_run.py` (imports `09_fixture_probe_facing_run.py` unchanged; same
pool, seeds, samples, labels, probe, controls), executed locally at 16 threads,
`gammas -1,0.6` (the steep and zero-coupling samplings of Supplement 05), 1,000 Freedman-Lane
draws. Backs the Appendix E known-surface numbers.

### 3.5 `09_physics_probe_facing_split.jsonl` (seed 0, `d=16` and `d=20`)

Source: `09-SUPPLEMENT-07-PROBE-FACING-SPLIT.md`, §2 "Physics anchors (decoder geometry, label
Hessian from data)".

Invocation not recorded in the supplements as a literal command line. Runner
`09_physics_probe_facing_split_run.py`, run on the pod from the stored per-anchor geometry of
Supplement 06 (`09_probe_facing_geometry_d{16,20}.npz`; no decoder refit), 16 threads. The
supplement names one option explicitly: `--label-table` pointed at the local
`labels_Smith42_galaxies_v2.0_test.parquet` (§2 above), because the `hf://` label read hung.
Runner untracked at commit `71914dd` on both hosts. Record sha256
`09e869ff61a1626a24773437834aa66dd3de02c4fdae80ec4498ece243e7be65` (matches the sha256 quoted for
this file in `paper/latex/COMPLIANCE.md`). This is the record behind `tab:real`,
`tab:meancurv`, the main-text checks, the baseline column of `tab:ablate`, the
`alpha=100` parenthetical column of `tab:sens` (`appendix_gen.py`'s `main` lookup, ~line 65),
`figF` panel (b) and the generated-but-not-included `fig1_probe_facing.pdf`, and the ViT-B row of
`tab:xenc`/`tab:xencx`. It does **not** back the compiled Figure 1 (`fig1_intervention.pdf`) — see
§3.9 for that.

### 3.6 `09_physics_probe_facing_split_{seed1,seed2,w400}.jsonl` (Appendix A)

Source: `09-SUPPLEMENT-09-ABLATIONS-AND-SENSITIVITY.md`, header and §A.

Invocation not recorded in the supplements as a literal command line. Runner
`09_physics_probe_facing_split_run.py` at `d=16`, commits `79589d3`/`c54a2eb`, pod
`universetbd-0`, labels from the cached parquet (§2 above). The supplement names the options used
without giving full command lines: `_seed1` and `_seed2` are `--fit-seed 1` and `--fit-seed 2`
(decoder width 250^3, matching the main record); `_w400` is a decoder of width 400^3 at seed 0
(`--hidden 400,400,400`, inferred from the runner's own `--hidden` help text and the supplement's
"seed 0 at 400^3"; not stated as a literal flag value in the supplement). All four variants (main
plus these three) reach variance explained 0.952.

### 3.7 `09_physics_probe_facing_split_{xfit,alpha1}.jsonl` (Appendix B)

Source: `09-SUPPLEMENT-09-ABLATIONS-AND-SENSITIVITY.md`, header, §B and §C.

Invocation not recorded in the supplements as a literal command line. Runner
`09_physics_probe_facing_split_run.py`, commits `79589d3`/`c54a2eb`, pod `universetbd-0`. `_xfit`
is the `--hessian-xfit` flag (cross-fitted Hessian: fit on one half of each 2,048-patch, score on
the other); `_alpha1` is `--alpha 1` (weak-ridge probe, global out-of-sample R^2 rises to
0.64-0.67 from 0.48-0.53 under the sealed `alpha=100`).

### 3.8 `09_physics_probe_facing_split_{dinov3_vitb16,clip_base,convnext_base,vit_large}.jsonl` (Appendix C, cross-encoder)

Source: `09-SUPPLEMENT-11-CROSS-ENCODER-PROBE-FACING.md`.

Literal invocation given (one run per encoder, `<enc>` in
`{dinov3_vitb16, clip_base, convnext_base, vit_large}`):

```bash
python curvature-experiment/runners/09_physics_probe_facing_split_run.py \
  --mode physics --d-values 16 --fit-seed 0 --hessian-xfit \
  --parquet-path <local copy of physics/<enc>_test.parquet, snapshot bc081f8a…> \
  --embedding-column <enc>_galaxies \
  --label-table <cached labels_Smith42_galaxies_v2.0_test.parquet, sha256 60f2f82e…>
```

Run on the pod (`universetbd-0`), one tmux session per encoder, 16 threads each, 48-52 minutes
wall clock each; runner sha256 `ce63a83ce4b0d2f6…`, commit `139950f`. Embeddings parquet files
were fetched with `hf_hub_download` from the `bc081f8a…` snapshot. Records
`09_physics_probe_facing_split_<enc>.jsonl`, logs `09_xenc_<enc>.log`. Row-order correctness was
checked via the label probes' global out-of-sample R^2 per encoder rather than a join key (no id
column exists in these parquet files); a misaligned row order would have collapsed those R^2
values to approximately zero.

The per-encoder geometry npz that §3.9 reads is written only when `--geometry-out <dir>` is
passed: the runner treats the value as a directory and saves
`<dir>/09_probe_facing_geometry_d16_seed0.npz` (`anchor_idx`, `J`, `Hess`, `image`) alongside
the refit. The command recorded in Supplement 11 above omits `--geometry-out`, even though the
§3.9 cross-encoder runs read geometry saved by these runs.

### 3.9 `09_physics_normal_scaling_{vit_base_d16,vit_base_d20,dinov3_vitb16_d16,clip_base_d16,convnext_base_d16,vit_large_d16}.{jsonl,npz}` (Appendix D, Figure 1)

Source: `09-SUPPLEMENT-12-NORMAL-SCALING-COUNTERFACTUAL.md`, header and "Question and design".

Invocation not recorded in the supplements as a literal command line. Runner
`09_physics_normal_scaling_run.py` (imports `09_physics_probe_facing_split_run.py` unchanged),
run on the pod (`universetbd-0`, tmux sessions `effdim-nsA`/`effdim-nsB`), 2-4 minutes per run,
runner sha256 `6eb2d7bf61ff5fc6…`. Geometry taken from each decoder's stored per-anchor npz (the
ViT-B `d=16`/`d=20` geometry from §3.5 above; the four cross-encoder decoders at seed 0, `d=16`,
from the geometry saved by the §3.8 runs). Smoke mode on the in-sphere fixture was run locally
first as a check, not to produce a paper record. Records at three revisions exist on the pod
(`ns-out-v1/`, `ns-out-v2/`, current); only the current (v3) records back the manuscript.

The runner's own documented command form (its module docstring; not a recorded invocation):

```bash
python curvature-experiment/runners/09_physics_normal_scaling_run.py --mode physics --d 16 --threads 16 \
  --geometry-npz <geometry npz> --parquet-path <embeddings parquet> --embedding-column <enc>_galaxies \
  --label-table <cached labels parquet> --record-path curvature-experiment/.cache/09_physics_normal_scaling_<run>.jsonl
```

`--geometry-npz` is `probe-facing/09_probe_facing_geometry_d{16,20}.npz` (§2) for the two ViT-B
runs (with `--d 20` for the `d=20` run) and the §3.8 `--geometry-out` file
`09_probe_facing_geometry_d16_seed0.npz` for each cross-encoder run. The `.npz` half of each record is written only when `--arrays-out <file>` is
also passed.

### 3.10 `09_physics_normal_scaling_*_thin.npz` (Appendix D sign tests)

Source: `09-SUPPLEMENT-12-NORMAL-SCALING-COUNTERFACTUAL.md`, "Significance with dependent
anchors".

Invocation not recorded in the supplements as a literal command line. Runner
`09_physics_normal_scaling_thin_run.py`, recomputes each run's anchor panel deterministically
(same seeds, same embeddings) and saves the 512x512 pairwise neighbourhood-overlap matrix.
`paper/generate/appendix_gen.py` computes the maximal independent set (pairwise overlap at most
5% of `k`) from this matrix at generation time; it is not precomputed into the record.

## 4. Record hashes

Hashes below are as quoted in the archived supplements (§3) and, where noted, cross-checked
against `paper/latex/COMPLIANCE.md`. A blank cell means no hash for that record is recorded
anywhere in the supplements or `COMPLIANCE.md` — most records were "verified both sides" of a
pod-to-local transfer without the hash itself being written down in prose.

| File | sha256 | Source |
|---|---|---|
| `09_instrument_adjudication.jsonl` | `2779170f0359c8d813f671f607976341dd6518d41e9b48a7e0a06e422961da32` | Supplement 02 §4 |
| `09_fixture_probe_decodability.jsonl` | `bf455f03f979e7f151b4cfbaaa444fd80359eaa65855ce24a69d4687ce750a6b` | Supplement 03 |
| `09_fixture_probe_decodability_gamma2.jsonl` | `ce3863f9…2adbb7dd` | Supplement 04 §8 |
| `09_fixture_probe_facing.jsonl` | `d6396ca012a09006…` | Supplement 05 §2 |
| `09_fixture_probe_facing_split.jsonl` | (not recorded) | Supplement 07 §1 |
| `09_physics_probe_facing_split.jsonl` (seed 0) | `09e869ff61a1626a24773437834aa66dd3de02c4fdae80ec4498ece243e7be65` | Supplement 07 §2; matches `paper/latex/COMPLIANCE.md` |
| `09_physics_probe_facing_split_{seed1,seed2,w400,xfit,alpha1}.jsonl` | (not recorded) | Supplement 09 |
| `09_physics_probe_facing_split_{dinov3_vitb16,clip_base,convnext_base,vit_large}.jsonl` | (not recorded; "verified on both sides") | Supplement 11 |
| `09_physics_normal_scaling_*.{jsonl,npz}` | (not recorded; "sha256 verified both sides") | Supplement 12 |
| `09_physics_normal_scaling_*_thin.npz` | (not recorded; "sha256 verified") | Supplement 12 |
| `labels_Smith42_galaxies_v2.0_test.parquet` (pod-only input) | `60f2f82e64e4036e…` | Supplement 07 §2 |

## 5. Caveats

- **`repo_head` does not pin the code version.** Records from 2026-09-12 through 2026-09-15
  carry `repo_head 71914dd`, although several of the CLI flags they were run with
  (`--fit-seed`, `--hidden`, `--hessian-xfit`, `--alpha`, `--parquet-path`, `--embedding-column`,
  `--label-table`) were added in later commits (`79589d3`, `c54a2eb`, `139950f`, `9ec69d7`
  through `28d3ee6`). The field records the tree the run started from, not a claim that every
  flag it accepted existed in that tree.
- **Appendix E's "mean-bending trace −0.24 to +0.27 across five encoders" is not produced by this
  repository.** Source: `origin/curvature-experiments@dabe5e2:outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md`
  §6, mirrored in that branch's `experiments/curvature_program/EXPERIMENT_REGISTRY.md`. No
  runner here reproduces it.
- **Two mismatch columns coexist, differing on the probe side, not the label side.** Both
  `hess_mismatch_emp` (used in the main tables) and `hess_mismatch_dec` (used in the cross-fitted
  columns; see `paper/generate/appendix_gen.py`) subtract the same estimated label Hessian
  `hess_y` (the local-quadratic fit to the label). They differ in what they subtract it from:
  `_dec` is `||hess_y - pf_full||`, against the decoder's own analytic `<w_N, II>` (`pf_full`,
  from autodiff); `_emp` is `||hess_y - probe_emp||`, against `probe_emp`, a second local-quadratic
  fit — this time to the probe's own prediction `w.x` on the data — as a data-side check of the
  decoder's curvature estimate (`09_physics_probe_facing_split_run.py`, ~lines 148-149). Both are
  read from the records as-is; this documentation does not reconcile them.
- **The colleague estimator comparison was removed from this code.** Earlier phases of this work
  compared the decoder instrument against a colleague's split-half nested-chart estimator
  (`K_H^cross`); several supplements above discuss that comparison for context (it is how the
  decoder instrument came to be validated against a known answer, Supplement 02). That
  estimator's code, CLI flags (`--colleague-root`, `--skip-colleague`) and record columns are not
  part of this repository; see `archive/README.md`.
- **Smoke mode is not a reproduction.** `--mode smoke` (or the in-sphere fixture at small `n`)
  exercises the same code paths at a scale the equivalence gate
  (`docs/superpowers/harness/gate.sh`) can check quickly; none of its output is a paper number.
  Every invocation in §3 above needs the pod-only inputs of §2 and ran at production scale
  (`n=86,471`, `k=2,048`, 512 anchors), which this repository's automated checks do not attempt.
