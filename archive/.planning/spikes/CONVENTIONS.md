# Spike Conventions

Patterns established across spike sessions. New spikes follow these unless the question requires
otherwise.

## Stack

Plain Python — numpy, scipy, scikit-learn, matplotlib if a plot is needed. No new dependencies.
**Use the repo `.venv`**: `.venv/bin/python`, not the system python, which has no torch. Several
sealed modules import torch at module scope even when the spike path is pure numpy.

## Structure

```
.planning/spikes/NNN-name/
  run_*.py          the spike's main measurement
  probe_*.py        follow-up probes, one per hypothesis, named for what they isolate
  *.out             recorded stdout, committed beside the script that produced it
  README.md         frontmatter, research, investigation trail, results
```

Scripts are standalone and take `REPO = Path(__file__).resolve().parents[3]`. Import the notebook
package via `sys.path.insert(0, REPO / "notebooks")`, or via `REPO / "notebooks" / "diagnostics"`
when a diagnostics runner is needed (those self-bootstrap `pu_manifold` onto the path).

## Patterns

- **Import sealed code unchanged; never edit `notebooks/pu_manifold/` or `notebooks/diagnostics/`
  from a spike.** A model that only works after the sealed module is rewritten is itself the
  finding. Anything new lives in spike-local code and says so in its docstring.
- **Score with the sealed scorer.** Reuse `synthetic_control_run._fidelity_axes` rather than
  recomputing axes, so spike numbers are comparable to sealed rows by construction. Where a
  spike-local statistic is genuinely needed (the floored CV), it is one function, documented as
  spike-local, and reported *alongside* the sealed number rather than replacing it.
- **Anchor at low `d` before interpreting a failure at high `d`.** A FAIL with no anchor cannot
  distinguish the phenomenon from broken wiring. Spike 001 is the template.
- **Write the decision rule into the script before running it.** Every probe here prints its own
  verdict from a threshold fixed in the source. Two of them then refuted the hypothesis that
  motivated writing them, which is only credible because the rule predated the data.
- **Record revised criteria, do not silently apply them.** When spike 001's thresholds turned out
  to be wrong, the revision note went into the docstring of the file that passes, so the original
  failure is visible from the passing artifact.
- **One probe, one hypothesis, named for what it isolates** — `probe_confound.py`,
  `probe_scale_confound.py`, `probe_dynamic_range.py`. Each is cheap enough to re-run on a whim.
- **Measure the constant rather than quoting it** where it is cheap. `r/R`, `D`-invariance and
  §1's `d=20` row were all recomputed from scratch; two of the three independently reproduced the
  recorded values, which is what makes the third believable.
- **Time before gridding.** A single timing probe turned a ~20-hour grid into a 30-minute one.

## Tools & Libraries

- `curvature_probe.quadric_mean_curvature` — the local-polynomial geometry teacher `(P̂, ÎI)`.
  Already exists, sealed. `O(D²)` memory per point from `svd(..., full_matrices=True)`.
- `synthetic_control_run._fidelity_axes` — the four axes. Private but stable; the comparability
  is worth the private import.
- `synthetic_controls.make_saddle_control` — the sealed `d=20` control. **Constant analytic
  Hessian by construction**; its `||H||` varies only through the pullback metric. See spike 002
  finding 3 before using it for an ordering question.
- `curvature_probe.make_graph_of_function_fixture` — Gaussian bumps, genuinely varying second
  derivatives, `||H_true||` spread ~1095×.
- `curvature_probe.make_swiss_roll_fixture` — the CLAUDE.md-mandated anchor. Returns `H_norm`
  only; the analytic `H` *vector* needed for the direction axis is derived in
  `001-teacher-low-d-anchor/run_anchor.py`, pinned to the fixture's `H_norm` at `1e-12`.

## Conventions that are not negotiable

`CURVATURE_CONVENTION = "trace"`, `H = tr_g(II)` unnormalized — a unit `d`-sphere gives
`||H|| = d`. The averaged convention differs by a factor of `d`, and this codebase has already
shipped and fixed one factor-of-`d` bug.

## The confound trio

A positive result at high `d` is not believed until three probes clear, each one file, each named
for what it isolates, each printing a decision rule fixed in its source before the run:

1. **Fixture structure** — does the result depend on the fixture's own functional form? Rerun on a
   fixture from a different family, everything else held.
2. **Local scale** — is the ordering curvature, or local sampling density? Spearman partial
   correlation against the local kNN radius.
3. **Dynamic range** — compare fixtures spread-for-spread by restricting to quantile windows of
   `||H_true||` and reporting each window's realized `p95/p05` beside its `rho`.

In spike 002 all three were necessary and each changed the conclusion; two refuted the hypothesis
that motivated writing them.

## Wrapped findings

Packaged as `Skill("spike-findings-effdim")` — requirements, the validation protocol, measured dead
ends, and the open saddle-fixture question. Load it before proposing any curvature estimator,
prior, or control fixture.
