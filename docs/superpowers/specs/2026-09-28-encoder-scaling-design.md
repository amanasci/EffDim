# Encoder scaling: the paper's battery on all 31 Platonic Universe galaxy encoders

Date: 2026-09-28
Branch: `encoder-scaling` (from `linear-probes-curvature@bbaced5`)

## Goal

Test whether the paper's results ("Linear Probes on Curved Latent Spaces": the mismatch and
alignment partials, and the counterfactual help/hurt asymmetry) hold across every encoder in the
`physics` set of `UniverseTBD/pu-embeddings`, not just the five the paper used, including
model-size families. This is sub-project 1 of "scale to the whole PU release"; other datasets and
modalities (legacysurvey, cosmosweb, desi, jwst) need their own label source and are sub-project 2,
out of scope here.

Success criteria:

1. All 279 jobs (9 per encoder × 31 encoders) complete on the pod GPU, each with a validated
   record and a done-marker.
2. `aggregate.py` regenerates every committed table, figure and the report byte-for-byte from the
   records (pinned by a test).
3. The existing CPU equivalence gate (`docs/superpowers/harness/gate.sh`) still passes: the new
   `--device` / `--deterministic` options leave the default CPU path unchanged.
4. `paper/latex/main.tex` is unchanged (the submitted manuscript is not edited).

## Decisions (agreed in brainstorming)

| Topic | Decision |
|---|---|
| Scope | Sub-project 1 = all 31 encoders of the `physics` galaxy set. Other datasets = sub-project 2. |
| Per-encoder analyses | The paper's full battery plus robustness (option B): see Jobs. |
| Platform | Pod GPUs (8× A100 80GB), all 31 encoders including the paper's five. |
| CPU vs GPU comparison | None. Results stand on the GPU platform alone; the decoder-seed spread (jobs 3-4) is the noise floor. Every record states its device. |
| Determinism | `--deterministic` on: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |
| Architecture | Encoder manifest + resumable job queue around the existing runners (no new pipeline). |
| Manuscript | Not edited. New results live in `curvature-experiment/results/scaling/`. |

## Encoders

31 encoders, all 86,471 rows, parquet `physics/<name>_test.parquet`, column `<name>_galaxies`,
snapshot `bc081f8a5db4767edcd958653d96efde9137de0b`:

| Family | Encoders (embedding dim) |
|---|---|
| AstroPT | astropt_015M (384), astropt_095M (768), astropt_850M (2048) |
| CLIP | clip_base (512)*, clip_large (768) |
| ConvNeXt | convnext_nano (640), convnext_tiny (768), convnext_base (1024)*, convnext_large (1536) |
| DINOv3 | dinov3_vits16 (384), dinov3_vits16plus (384), dinov3_vitb16 (768)*, dinov3_vitl16 (1024), dinov3_vith16plus (1280), dinov3_vit7b16 (4096) |
| ViT | vit_base (768)*, vit_large (1024)*, vit_huge (1280) |
| ViT-MAE | vit-mae_base (768), vit-mae_large (1024), vit-mae_huge (1280) |
| I-JEPA | ijepa_huge (1280), ijepa_giant (1408) |
| V-JEPA | vjepa_large (1024), vjepa_huge (1280), vjepa_giant (1408) |
| LLaVA-1.5 | llava_15_7b (4096), llava_15_13b (5120) |
| PaliGemma | paligemma_3b (2304), paligemma_10b (3584), paligemma_28b (4608) |

`*` = in the paper. Parameter counts are entered per model in the manifest with a cited source
(model card).

## Jobs (per encoder)

All at d=16 unless stated; labels mag_r, photo_z, smooth_fraction, stellar_mass; 512 anchors,
k=2048, α=100, multi-scale radius control — the paper's protocol with only the embedding source
changed.

| # | Job id suffix | Runner and flags | Depends on |
|---|---|---|---|
| 1 | `main_xfit` | split runner `--fit-seed 0 --hessian-xfit --geometry-out <dir>` | — |
| 2 | `main` | split runner `--fit-seed 0` | — |
| 3 | `seed1` | split runner `--fit-seed 1` | — |
| 4 | `seed2` | split runner `--fit-seed 2` | — |
| 5 | `w400` | split runner `--fit-seed 0 --hidden 400,400,400` | — |
| 6 | `alpha1` | split runner `--fit-seed 0 --alpha 1` | — |
| 7 | `d20` | split runner `--fit-seed 0 --d-values 20` | — |
| 8 | `cf` | normal-scaling runner `--geometry-npz <job 1 geometry> --arrays-out <npz>` | 1 |
| 9 | `thin` | thin runner `--arrays-npz <job 8 arrays> --out <npz>` | 8 |

Every split/normal-scaling/thin invocation also passes `--parquet-path` (the encoder's parquet,
downloaded once to the pod), `--embedding-column`, `--label-table` (the pod-resident label
parquet), `--device cuda --deterministic`, and an explicit `--record-path`.

## Code changes

Existing runners (in `curvature-experiment/runners/`), minimal and default-preserving:

- `09_physics_probe_facing_split_run.py`, `09_physics_normal_scaling_run.py` (and whatever they
  call that builds the decoder or geometry): add `--device {cpu,cuda}` (default `cpu`) and
  `--deterministic`. On `cuda`, the decoder, its training tensors and the Jacobian/Hessian `vmap`
  run on the device; results return to CPU as float64 before any statistic. Ridge probes, partial
  Spearman, permutations and bootstrap stay on CPU numpy, unchanged.
- The thin runner is CPU-only (it recomputes neighbourhoods); it is unchanged.
- Environment rows gain `device`, `gpu_name`, `cuda_version`, `torch`, `deterministic`.
- The default path (`--device cpu`, no `--deterministic`) must stay byte-identical: the existing
  gate proves it.

New, in `curvature-experiment/`:

- `encoders.yaml` — the manifest: name, family, parquet path, column, dim, params, params source,
  `in_paper` flag; snapshot sha pinned once at the top.
- `sweep/jobs.py` — expands manifest × battery into job specs: id `<encoder>__<suffix>`, argv,
  record path(s), dependencies, expected record checks. `python -m sweep.jobs --list` is a dry run.
- `sweep/run_queue.py` — scheduler: runs ready jobs (dependencies done), one per GPU via
  `CUDA_VISIBLE_DEVICES`, up to `--gpus` concurrently; writes `<id>.done` (with record sha256)
  only after exit 0 and record validation; skips jobs with a valid `.done`; moves a partial record
  aside and re-runs a job without one; logs to `<id>.log`; `--only`, `--dry-run`.
- `sweep/aggregate.py` — reads records, writes tables, figures and the report (see Outputs).
- `sweep/setup_pod.sh` — idempotent environment setup on `/mnt/ssd-cluster/EffDim` (venv on
  `/mnt`, pinned requirements, HF cache on `/mnt`), re-runnable after a pod restart.
- Tests (`curvature-experiment/tests/`): manifest schema and 31 entries; job expansion and
  dependency graph; queue done-marker/resume/partial-record handling with a fake runner;
  aggregate on synthetic records; aggregate regeneration of the committed outputs byte-for-byte
  (skips without the scaling records).

## Pod operation

Binding rules from `docs/remote-compute/eleutherai-pod-user-guide.md` and `CLAUDE.md`:

- Read the local guide before any SSH command; each session, check the remote
  `/root/user-guide.md` sha256 against the local copy's and re-read on change.
- Everything persistent under `/mnt/ssd-cluster/EffDim` (repo clone at this branch, venv, HF
  cache, `sweep-out/`). Nothing we need lives in `/root`.
- Long jobs run inside `tmux`. After a pod restart: re-run `setup_pod.sh`, restart the queue; done
  jobs are skipped.
- No concurrent or unbounded `find`/`ls -R` on `/mnt`; direct paths, `-maxdepth`, `timeout`.
- Never write to `/mnt/datasets` or shared `/mnt/ssd-1..4`; touch only `/mnt/ssd-cluster/EffDim`.
- The pod is shared: check `nvidia-smi` for free GPUs before launching and use only free ones;
  verify the effective cgroup CPU limit before setting `--threads`.
- Embedding parquets (~16 GB for the physics set) download once into the HF cache on `/mnt`.
- Records return to this machine with `rsync` into `curvature-experiment/.cache/scaling/`.

Run order:

1. GPU smoke of each changed runner, twice; records must be identical (determinism check).
2. One encoder end to end (vit_base), all 9 jobs; sanity read of the outputs.
3. All 31 encoders.

## Outputs

Records: `curvature-experiment/.cache/scaling/` (gitignored, like all records), with a
`SHA256SUMS` file committed to `curvature-experiment/results/scaling/`.

Committed to `curvature-experiment/results/scaling/`, all written by `aggregate.py`:

1. `tab_scaling_main.tex` — 31 rows × 4 labels: mismatch and alignment partials (multi-scale
   control) and variance explained; grouped by family, sorted by parameters; `tab:xenc` cell format.
2. `tab_scaling_xfit.tex` — cross-fitted mismatch (`tab:xencx` format).
3. `tab_scaling_cf.tex` — counterfactual help/hurt fractions, t*, sign-test bounds (`tab:cf` format).
4. `tab_scaling_robust.tex` — per encoder, min-max of each headline partial across seeds 0/1/2,
   width 400, α=1, d=20, and the number of sign changes.
5. Figures (PDF + PNG), paper styling: partials vs log parameters (one panel per label, colour by
   family); counterfactual help fraction vs log parameters; robustness spread per encoder.
6. `SCALING_REPORT.md` — computed summary: per paper claim, how many encoders reproduce it (sign and
   significance of mismatch and alignment per label; counterfactual help > hurt), and the list of
   exceptions. No hand-typed numbers.

## Risks

- **Pod restarts mid-run.** Mitigated by done-markers and idempotent jobs; at worst the in-flight
  jobs (≤ 8) rerun.
- **GPU memory for the largest encoders** (D up to 5120: decoder output layer and D×d×d Hessians
  at 512 anchors). Chunked `vmap` exists (`VMAP_CHUNK`); the ViT-B end-to-end run plus a one-off
  memory probe on llava_15_13b before the full sweep.
- **Determinism is not guaranteed by every CUDA kernel.** If `use_deterministic_algorithms` raises
  on an op, the run stops; the fix is a per-op workaround recorded in the environment row, never
  silently dropping the flag.
- **Row alignment across encoders.** All 31 parquets have 86,471 rows. The paper's row-alignment proof
  (`09_row_alignment_proof_run.py`) established the embedding-to-label pairing for ViT-B, and the
  cross-encoder runs assumed the same row order. The manifest loader asserts row count per encoder; a cross-encoder alignment check
  is out of scope unless a mismatch in results suggests it.
- **Shared GPUs.** Other users may hold GPUs; the queue takes `--gpus` explicitly from the free set.
