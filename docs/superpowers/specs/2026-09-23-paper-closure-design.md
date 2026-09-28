# Paper closure: reduce the repo to code supporting "Linear Probes on Curved Latent Spaces"

Date: 2026-09-23
Branch: `simplify/paper-closure` (worktree `.claude/worktrees/paper-closure`), based on
`fixture-validity-audit@78dced8` plus the two commits of `simplify/previous-phases`
(`build/` removal, behaviour-preserving `pu_manifold` cleanup), cherry-picked cleanly.

## Goal

A human reviewer can start at the paper's LaTeX, follow every table, figure and number to
exactly one generator, one runner and a small set of modules, and meet no unrelated code on
the way. Code that does not support the paper moves to `archive/`; nothing is deleted.

Success criteria:

1. The paper and its supporting code are decoupled. `paper/` holds only the manuscript and
   the scripts that turn records into LaTeX and figures. `curvature-experiment/` holds only
   code reachable from the runner invocations in `curvature-experiment/REPRODUCE.md`, the
   paper's generators, and the kept Swiss roll notebooks.
2. Every kept runner's smoke output is identical to the pre-change baseline (see
   Verification), except for the colleague fields removed on purpose.
3. `paper/generate/appendix_gen.py` run against the record cache leaves `main.tex`
   byte-identical.
4. `paper/latex/check.sh` builds the PDF.
5. The `effdim` package, its tests, CI and docs site are unchanged and still pass.

## Decisions (agreed in brainstorming)

| Topic | Decision |
|---|---|
| Paper source | OpenReview PDF blocked by bot check; the local LaTeX is authoritative. Its final state is the **uncommitted working tree** of the main checkout (see Stage 0). |
| Depth of cut | Archive whole files, cut import chains, **and** trim unreachable functions inside kept modules. |
| `effdim` package | Stays at the repo root untouched (`src/effdim`, `tests/`, `benchmarks/`, `pyproject.toml`, CI, `mkdocs.yml`, `docs/`, `README.md`, `MANIFEST.in`). The manuscript moves to `paper/`; its supporting code moves to `curvature-experiment/`. |
| `.planning/` | Archived whole. Provenance is rewritten into `curvature-experiment/REPRODUCE.md`, which cites the archived source path for every invocation. |
| Untracked figures | Committed. |
| `appendix_gen.py` clobbering Table 1 | Fixed: generator emits `tab:real` and the hand-edited Appendix C prose. |
| Colleague estimator | Archived. Its CLI flags and record columns are removed. |
| Appendix E mean-bending trace | Not ported. `REPRODUCE.md` documents its source (`origin/curvature-experiments@dabe5e2:outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md` §6). |
| Records, pod-only inputs | Not committed (8.7G). `REPRODUCE.md` lists paths, sha256s and origins. |
| Wrong `repo_head` in records | Documented in `REPRODUCE.md`, not fixed. |
| Missing Swiss roll check for the sphere-projected decoder | Out of scope; separate task. |
| Paper / code split (added 2026-09-27) | `paper/` = manuscript, figures, figure scripts, table/appendix generators. `curvature-experiment/` = runners, `pu_manifold`, tests, Swiss roll notebooks, requirements, provenance. Generators read records from `curvature-experiment/.cache/` (or `EFFDIM_CACHE_DIR`). |
| CLAUDE.md "additive only" rule | The user's request is explicit sign-off for this task. |

## Target layout

```
paper/                       the manuscript and record -> LaTeX/figure scripts
  README.md                  paper element -> generator -> record -> runner map
  latex/                     was docs/latex/ml4ps: main.tex, references.bib, neurips_2026.sty,
                             checklist.tex, neurips_2026.tex, check.sh, COMPLIANCE.md
    figures/                 fig1_intervention, figF_mean_curvature, fig1_probe_facing (.pdf/.png),
                             make_fig1.py, make_fig_intervention.py
  generate/                  table_main_gen.py, appendix_gen.py
  tests/                     test_generators.py (appendix_gen reproduces main.tex)
curvature-experiment/        the supporting code that produces the records
  README.md                  what each runner computes, how to run tests and smoke checks
  REPRODUCE.md               invocations, record sha256s, pod-only inputs, repo_head caveat,
                             Appendix E mean-bending source, what smoke mode does and does not cover
  requirements.txt           trimmed from notebooks/requirements-notebooks.txt
  runners/                   kept 09_* runners, original file names
  pu_manifold/               trimmed modules
  tests/                     tests for kept modules and runners
  notebooks/                 02.6_swiss_roll_plainae_curvature_check (new CLAUDE.md reference),
                             09.1_swiss_roll_probe_decodability_check,
                             09.2_swiss_roll_density_decoupling_check
                             (02.2_swiss_roll_cae_check is archived: it tests ChartAutoEncoder,
                             which the paper does not use)
  .cache/                    gitignored; records. CACHE_DIR resolves here unless EFFDIM_CACHE_DIR is set
archive/
  README.md                  what is here, why, original path of every item
  notebooks/...              all other notebooks, runners, modules, tests (original relative paths)
  pu_manifold_trimmed/       functions removed from kept modules, one file per source module
  .planning/  sweep/  docs/latex/ (old long-form main.tex, main.pdf, references.bib)
  HANDOFF-v1.1.md  TODO.md  PYPI_SETUP.md
```

Dependency direction is one-way: `paper/` reads records and imports nothing from
`curvature-experiment/` except the record location; `curvature-experiment/` never references
`paper/`. Root `README.md` gains two lines pointing to `paper/` and `curvature-experiment/`.
`CLAUDE.md` paths are updated (`notebooks/pu_manifold` -> `curvature-experiment/pu_manifold`,
reference notebook becomes
`curvature-experiment/notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb`, remote-compute
doc path unchanged). `.gitignore` gains `curvature-experiment/.cache/`. Existing records stay
in `notebooks/.cache`; `REPRODUCE.md` tells the reader to symlink
`curvature-experiment/.cache` to it or set `EFFDIM_CACHE_DIR`. Generators take the cache path from the same resolution instead of the
hard-coded `C = "notebooks/.cache/"`, and no longer depend on the working directory.

## The closure

Starting set, from the paper-to-code trace:

- Runners: `09_physics_probe_facing_split_run`, `09_physics_normal_scaling_run`,
  `09_physics_normal_scaling_thin_run`, `09_physics_probe_facing_run`,
  `09_fixture_probe_facing_run`, `09_fixture_probe_facing_split_run`,
  `09_fixture_probe_decodability_run`, `09_instrument_adjudication_run`.
- Modules: `__init__`, `cache`, `subsample`, `physics_curvature_probe`, `physics_labels`,
  `cae`, `geometry_probes`, `chart_curvature`, `decoder_curvature`, `curvature_probe`,
  `crossmodal_curvature`, `cross_split_curvature`, `density_stratified_null`, `linear_probe`.
- Borderline: `09_row_alignment_proof_run.py` justifies `ALIGNMENT_ASSUMED_OFFSET` and is
  loaded by `test_physics_labels`. Kept in `curvature-experiment/runners/` as provenance for that constant.

Leaving the closure through this work:

- Colleague path: `09_colleague_estimator_run`, `colleague_shims/`, and
  `test_colleague_estimator_run`. (Correction found while planning: `09_physics_curvature_run`
  is **not** colleague-only. Kept runners use its `SphereProjectedDecoder`,
  `fit_and_field_at_anchors`, `_oof_predictions_for_label` and `_THREADS` cap, so it stays in
  the closure, is loaded directly instead of through the colleague runner, and is trimmed in
  stage 5 like any kept module.)
- `_fidelity_axes` chain: `synthetic_control_run`, `curvature_field_pu_run`,
  `derivative_bridge_run`, `derivative_bridge`, `topoae`, `persistence_probe`,
  `synthetic_controls`, and `mknn` if it proves import-only after trimming.
- Everything listed as outside the closure in the trace (about 50 runners, 13 modules and
  their tests, all other notebooks).

## Trimming rules

A symbol survives only if it is reachable from a kept runner's entry point under the
`REPRODUCE.md` invocations, a generator, a kept Swiss roll notebook, or a kept test of a
surviving symbol. Reachability is computed by static call graph and grep, then confirmed by
the harness.

Stages, one commit each, harness gate after each:

0. **Snapshot the submitted paper.** Copy the main checkout's working-tree versions of
   `docs/latex/ml4ps/{main.tex, references.bib, COMPLIANCE.md, appendix_gen.py,
   figures/make_fig1.py, figures/fig1_probe_facing.{pdf,png}, preview_times_metric.pdf}` and
   the untracked `figures/{make_fig_intervention.py, fig1_intervention.{pdf,png},
   figF_mean_curvature.{pdf,png}}`, `preview_cm_fallback.pdf`, plus the uncommitted
   `HANDOFF-v1.1.md` and novelty-review `README.md` edits. The `fixture-validity-audit`
   checkout is not modified. Commit message names the source and its HEAD.
1. **Capture baseline** (not committed; see Verification).
2. **Archive whole files** outside the closure with `git mv`, preserving relative paths
   under `archive/`.
3. **Cut the colleague path.** Archive its items; point the adjudication runner's loader at
   `09_physics_curvature_run.py` directly. Remove `--colleague-root`,
   `--skip-colleague`, colleague columns and their plumbing from the adjudication, physics
   and fixture runners.
4. **Cut the `_fidelity_axes` chain.** Move `_fidelity_axes` and only the private helpers it
   needs into `09_instrument_adjudication_run.py`; archive the chain.
5. **Trim inside kept modules.** Unreachable functions and classes (for example
   `cae.ChartAutoEncoder`, most of `curvature_probe`, the verdict, bootstrap and stratified
   parts of `physics_curvature_probe`) move verbatim to
   `archive/pu_manifold_trimmed/<module>.py`, headed with the source commit. Their tests
   move with them.
6. **Remove dead parameters and branches** only where every caller passes the same value
   (for example `INCLUDE_RELATIVE_II=False` and its block in `appendix_gen.py`).
7. **Move to the `paper/` + `curvature-experiment/` layout.** `git mv` into the target layout; rewrite `parents[1]`,
   `DIAGNOSTICS_ROOT`, `CACHE_DIR` and notebook import paths; add `EFFDIM_CACHE_DIR`.
8. **Fix `appendix_gen.py`** to emit `tab:real` and the hand-edited Appendix C prose.
9. **Docs.** `paper/README.md`, `curvature-experiment/README.md`,
   `curvature-experiment/REPRODUCE.md`, `archive/README.md`, root README pointers, `CLAUDE.md`
   paths, `curvature-experiment/requirements.txt`.

Invariants for every stage:

- No renames of kept public symbols.
- No change to numerics, defaults, dtypes, RNG call order or seeds.
- No reformatting of untouched lines, so `git diff -M` shows moves as moves.
- The nine-hop `importlib` runner chain is kept as a chain (flattening deferred); after
  stages 3-4 it shrinks to about five hops, all inside `curvature-experiment/runners/`.

## Verification

Baseline, captured once before stage 2 into the job's tmp directory (not committed):

- `pytest` on the kept `pu_manifold` tests.
- Every kept runner in `--mode smoke` (the thin runner on the smoke `.npz`), writing to a
  temp output root via `EFFDIM_09_OUTPUT_ROOT`.
- `table_main_gen.py` stdout.
- `appendix_gen.py` rendered to a temp file instead of `main.tex`.

Gate after each stage:

- Tests pass.
- Smoke records equal the baseline field by field. Ignored: timestamps, `repo_head`,
  absolute paths, and after stage 3 the removed colleague fields. Floats must match
  exactly; seeds are fixed. Any drift stops the work and is reported.
- Generator output byte-identical. Exception: stage 8, whose acceptance is that
  `appendix_gen.py` against the real cache leaves `main.tex` unchanged.

Final checks:

- Swiss roll notebooks re-executed with `nbconvert --execute`; their pass/fail lines match
  the committed outputs.
- `appendix_gen.py` against the real cache leaves `main.tex` unchanged
  (`git diff --exit-code`).
- `check.sh` builds the PDF.
- Root `pytest` (the `effdim` suite) still passes.

Not covered: full physics mode, which needs the pod and hours of GPU time. Smoke mode runs
the same code paths at small n. `REPRODUCE.md` states this.

## Risks

- **Smoke mode may not reach every code path used in physics mode** (for example
  `--hessian-xfit`, `--alpha`, `--hidden`, per-encoder parquet paths). Mitigation: run smoke
  once per distinct flag combination in `REPRODUCE.md` where smoke accepts it; any symbol
  reached only by physics mode is kept, not trimmed, unless static analysis proves it dead.
- **Runtime monkeypatching** in the split and normal-scaling runners
  (`pl.load_physics_embeddings`, `pcp.ALPHA_RIDGE`, `pcp.AE_HIDDEN`, and others) makes some
  symbols look unused to static analysis. Mitigation: every patched attribute is kept.
- **Two mismatch definitions** (`hess_mismatch_emp` in the main tables,
  `hess_mismatch_dec` in the cross-fitted columns) are preserved as they are; this work
  does not change what the paper reports. Flagged in `REPRODUCE.md` for the reviewer.
