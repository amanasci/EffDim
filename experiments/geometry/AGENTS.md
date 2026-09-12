# Geometry experiments — agent instructions

Read [`experiments/curvature_program/AGENT_CONTEXT.md`](../curvature_program/AGENT_CONTEXT.md) first.
Then consult [`experiments/curvature_program/EXPERIMENT_REGISTRY.md`](../curvature_program/EXPERIMENT_REGISTRY.md) before proposing a new run.

Do not reconstruct months of context from chat. Inspect frozen artifacts.

## Safety

- Never modify a completed output tree under `outputs/`.
- Create a **new** experiment package and a **new** output path for new work.
- Preserve strict `sample_id` alignment. Never join by row position.
- Distinguish raw, normalized-full, sphere-radial, sphere-normal residual, trace, and full-form curvature.
- Distinguish pointwise decoder curvature from finite-patch quadratic curvature.
- Distinguish global \(R^2\), global MSE, patch performance, and adaptation gain.
- When a statistic contains \(w\), use leakage-safe train-only weights.
- Use train-only labels for \(H_y\).
- Retain signed split-half cross estimates. Do not clamp negatives. Do not average-then-square.
- Avoid causal or universal wording.
- Avoid new broad sweeps without explicit authorization.
- Stop when a reliability or identity gate fails. Do not reinterpret a failed gate as a positive mechanism.
- Write `COMPLETE.json` only after required stages finish.
- Never edit a manuscript automatically.
- Do not modify `src/effdim/` during the v1.1 milestone.
- Do not rerun Q fits, decoder training, permutations, or bootstraps unless the user explicitly authorizes that expensive job.

## Source of truth

`COMPLETE.json` > `decision.json` > `summary.json` > tables > reports > manuscript > notes.

## Package

Reusable algebra: `experiments/geometry/curvature/` (import as `geometry.curvature`).
Historical runners are frozen; do not silently rewire them to the new package.
