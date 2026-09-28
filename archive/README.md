# archive/

This holds everything that was in the repository before the paper-closure restructure but is not
reachable from the ML4PS 2026 manuscript's tables, figures and generators. Nothing was deleted:
whole files that fell outside the closure were moved here with `git mv`, and code cut out of a
kept file (a function, a block of dead test cases, an unreachable CLI mode) was copied verbatim
into a new file here in the same commit that removed it from the kept file. The pre-closure tree is
`fixture-validity-audit@78dced8` plus the Stage 0 snapshot commit (which copied in the submitted
paper's uncommitted working-tree state); either of those, or any commit before this restructure,
has every file below at its original path and in its original, untrimmed form.

Code under this directory is **not importable as-is**: paths changed (for example
`notebooks/pu_manifold` no longer exists; the package now lives at
`curvature-experiment/pu_manifold`), and archived files still import from the old paths. Treat
everything here as a historical record, not as a package to run.

| Archived path | Original path | Reason |
|---|---|---|
| `archive/notebooks/` (runners, notebooks, and `pu_manifold/`'s non-kept modules and tests) | `notebooks/` | Outside the paper-to-code trace: about 50 runners, 17 `pu_manifold` modules and their tests, and every notebook except the three kept Swiss roll checks. Covers earlier milestones (chart auto-encoders, topological auto-encoders, region partitioning, persistence, the crossmodal/Phase 7-8 alignment work) that the manuscript does not use. |
| `archive/notebooks/diagnostics/colleague_removed/` | (new; created in this restructure, Stage 3, commit `f235f49`) | Copies of three runners (`09_fixture_probe_decodability_run.py`, `09_instrument_adjudication_run.py`, `09_physics_probe_facing_run.py`) as they stood immediately before that commit cut the colleague estimator's `--colleague-root`/`--skip-colleague` flags, columns and plumbing from the kept versions. Reference for what changed, not a second copy to run. |
| `archive/notebooks/diagnostics/trimmed/` | (new; created in this restructure, Stage 5a, commit `cb9e502`) | `09_physics_curvature_run.py`'s functions that were not reachable from any kept runner (CLI modes, verdict and pre-registration machinery, `__main__`), cut out and archived verbatim in the same commit; headed with the source commit. |
| `archive/notebooks/diagnostics/colleague_shims/` | `notebooks/diagnostics/colleague_shims/` | The `topology` shim the colleague's estimator code needed when imported from a read-only checkout of his branch. Only meaningful together with `09_colleague_estimator_run.py`, also archived. |
| `archive/notebooks/pu_manifold/tests/*_removed.py` | (new; created in this restructure, Stage 2, commit `7e4eb93`) | Dead test cases for already-archived modules (curvature stubs, the 07.1 runner), cut out of the surviving test files and archived verbatim in the same commit that archived their target module. |
| `archive/notebooks/pu_manifold/tests/trimmed/` | (new; created in this restructure, Stage 5a, commit `cb9e502`) | `test_physics_curvature_probe.py`'s tests of the functions removed into `archive/notebooks/diagnostics/trimmed/`, moved with them in the same commit. |
| `archive/pu_manifold_trimmed/` | (new; extracted from `notebooks/pu_manifold/*.py`, one file per source module) | Top-level functions and classes cut from modules that were otherwise kept (for example `cae.ChartAutoEncoder`, most of `curvature_probe.py`, the verdict/bootstrap/stratified parts of `physics_curvature_probe.py`). Each file is headed with the source commit; order and content are verbatim. Their tests moved with them into `pu_manifold_trimmed/tests/`. |
| `archive/closure_dead_branches.md` | (new) | Dead parameters and branches removed from kept files during this restructure (Stage 6) — for example `appendix_gen.py`'s `INCLUDE_RELATIVE_II = False` guard and the code it always skipped. One heading per removed block, with the original file and line range, verbatim. |
| `archive/.planning/` | `.planning/` | The full planning history for every phase of this project (pre-registrations, plans, summaries, wave results, the Phase 9 supplements, handoffs). `curvature-experiment/REPRODUCE.md` cites the relevant Phase 9 supplement and section for every record it documents. |
| `archive/sweep/` | `sweep/` | A standalone parameter sweep (its own `requirements.txt`, `run_sweep.py`) outside the paper's closure. |
| `archive/docs/latex/` | `docs/latex/` (the pre-restructure long-form manuscript, kept alongside the ML4PS version before this project settled on the workshop paper as authoritative) | Superseded by `paper/latex/`; kept for history (`main.tex`, `main.pdf`, `references.bib`). |
| `archive/HANDOFF-v1.1.md` | `HANDOFF-v1.1.md` | Milestone handoff document for a reader starting fresh on v1.1; describes the pre-restructure layout and is now superseded by this README and the `paper/`/`curvature-experiment/` READMEs. |
| `archive/TODO.md` | `TODO.md` | Future-work task list predating this restructure. |
| `archive/PYPI_SETUP.md` | `PYPI_SETUP.md` | PyPI publishing setup notes for the `effdim` package, unrelated to the paper closure. |

## Two things worth knowing if you go looking in here

- **A frozen skill snapshot still names the old module paths.** `.claude/skills/spike-findings-effdim/sources/*`
  references `notebooks/pu_manifold/...` module names from before this restructure; it was not
  updated, since it is a frozen snapshot of prior findings rather than live documentation.
- **Archived test files that end in `_removed.py`, or that live under a `trimmed/` directory,
  carry no working imports.** Unlike most of this directory, these files did not exist before this
  restructure: they were created by it, as verbatim copies of test cases cut out of a surviving
  test file when their target module or function was archived (see the table above for which
  commit created each one). They were detached from a runnable module tree the moment they were
  written, and are historical record only, same as everything else in this directory.
