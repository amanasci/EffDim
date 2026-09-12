# Repository audit (curvature handoff)

Audit time: 2026-09-12 (UTC+10 session). Worktree:
`/home/angus/Documents/Code/PlatonicUniverse/EffDim-worktrees/SAE-shared-basis`.

Branch: `curvature-experiments`. HEAD: `dabe5e2`.
`git status --short` was dirty before any handoff edit. **No branch switch.**
Unrelated and user-authored changes were left in place.

## Existing agent-context files

| Path | Status | Action |
|---|---|---|
| `AGENTS.md` (root) | absent | Created a short pointer only |
| `CLAUDE.md` | present; Swiss-roll / `src/effdim/` rules | Not modified |
| `CONTEXT.md` | present; dirty; manuscript-oriented onboarding | **Not overwritten** |
| `experiments/geometry/AGENTS.md` | absent | Created scoped instructions |
| `experiments/curvature/README.md` | present (older paper-working index) | Not overwritten |
| `experiments/geometry/*/CONTEXT.md` | five experiment-local files | Not overwritten |
| `.cursor/rules/` | absent | Created scoped `curvature-program.mdc` |

## Package roots and conventions

- Shipped library: `src/effdim/` (`pyproject.toml` name `effdim`). **Do not modify during v1.1.**
- Geometry experiments import as `geometry.*` with `PYTHONPATH=experiments`.
- Tests: `tests/test_*.py` via pytest (`tool.pytest.ini_options`).
- No second package manager. Optional extra groups `curvature` and `curvature-torch` added to `pyproject.toml`.
- Reusable handoff code: `experiments/geometry/curvature/` (not `src/effdim/`, not a forced `src/platonic_universe/` path).

## Experiment registries already present

- Frozen synthesis (read-only): `outputs/geometry/curvature_program_synthesis/`
- Older paper-working notes: `experiments/curvature/paper_working/`
- Per-experiment `CONTEXT.md` files under later geometry packages
- This handoff adds `experiments/curvature_program/EXPERIMENT_REGISTRY.{md,json}`

## Curvature-related modules (historical, not rewritten)

Under `experiments/geometry/`:

- `physics_curvature_probe_submission_validation`
- `physics_local_probe_adaptation` (+ `_audit`)
- `physics_quadratic_label_chart_alignment` (+ `_audit`)
- `physics_cross_model_curvature_local_adaptation`
- `physics_cross_model_full_curvature_reconciliation`
- `pointwise_decoder_curvature_reproduction`
- `known_curvature_*`
- `physics_pointwise_residual_curvature_probe_relation`
- `physics_q_geometry_resampling_stability`
- `physics_task_aligned_curvature`
- `physics_cross_model_task_aligned_curvature`
- `physics_cross_model_pointwise_residual_curvature`
- `physics_cross_model_hessian_mismatch`
- `physics_curvature_component_predictive_decomposition`
- `curvature_program_synthesis`
- plus earlier adaptive-dataset / AE scale-match trees

Historical numerical code remains in those packages (`algebra.py`, `features.py`, runners).
The new package **copies** reusable algebra; it does not silently rewire runners.

## Safety constraints observed

1. Dirty worktree: no checkout/switch.
2. Existing `outputs/` trees treated as immutable.
3. Manuscripts (`paper/`, `papers/`, `submissions/`) not edited.
4. `CONTEXT.md` and `CLAUDE.md` not overwritten.
5. No training, Q refits, permutations, or bootstraps launched.
6. Source-of-truth hierarchy: `COMPLETE.json` > `decision.json` > `summary.json` > tables > reports > manuscript > notes.

## Local vs host output trees

This worktree is a **partial mirror**. Canonical large artifacts live on the science host
under `$PLATONIC_ROOT/outputs/geometry` (historically `/home/angus/platonic-universe/outputs/geometry`).
Host-only trees required by this programme are inventoried from the frozen synthesis
`SOURCE_MANIFEST.json` plus later local `decision.json` files. They are marked
`present_locally=false` and must not be reconstructed from conversation.
