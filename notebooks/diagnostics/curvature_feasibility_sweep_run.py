"""Phase 02.5's stage-1 feasibility sweep runner -- the phase's structural gate.

Computes exactly and only the gating metrics named in
``curvature_probe.STAGE1_GATING_METRICS`` (``spearman_rho`` AND ``quantile_bin_concordance``,
Section 3h's ratified option-scale-C -- a cell PASSES stage 1 only if BOTH clear their own
null-calibrated threshold and their own absolute floor), plus every non-gating context metric
``curvature_probe.measure_cell`` already returns. Applies the tested strict-comparison rule
``curvature_probe.verdict_from_stage1_metrics``, never re-derived inline. Writes
``curvature_verdict_stage1_{stage1_key}.json`` on PASS and on FAIL alike.

**This script writes NO Phase 3 handoff under any circumstance** -- the handoff belongs to
stage 2's substrate verdict (D-15), and stage 1 answers a different question (is local
curvature estimable at all, not which substrate is best). The handoff-writing sibling
function this module deliberately never calls or names is confirmed absent from this whole
file by Task 1's own negative-grep acceptance criterion.

``notebooks/pu_manifold/cae.py``, ``topoae.py``, and ``curvature.py`` are never edited by this
script -- every estimator, gate statistic, and artifact writer is imported and called
unmodified. ``curvature_probe.py`` itself was extended by this SAME plan (02.5-07) to add
``quantile_bin_concordance`` as a tested module function -- that extension is a separate,
already-committed change this script only consumes.

**Precondition, enforced in STEP 0 before any computation runs:** ``02.5-PREREGISTRATION.md``
must exist, its adding commit (``PREREG_SHA``) must be a proven ``git`` ancestor of ``HEAD``,
and every pre-registered constant copied into this file below must be found, verbatim, inside
that document. A runner whose constants have drifted from its own pre-registration is
measuring a different rule than the one that was ratified -- this precondition proves the
ordering (pre-register, then ratify, then measure) rather than merely asserting it, following
``topoae_evaluate_run.py``'s own STEP 0b/0c pattern.

**A note on wall-clock reality, disclosed here rather than only in the FINDINGS document:**
the non-gating local-quadric cross-check (``curvature_probe.quadric_mean_curvature``,
required at every cell by Section 9: "every existing non-gating quantity measure_cell already
returns") is, at the PU regime's own ambient dimension ``D=768`` and working dimension
``d=20``, measured (independently, before this file was written) at roughly 6-8 minutes of
CPU time for a single ``(n=10_000, d=20, D=768, k=30)`` call -- ``quadric_fit_curvature``'s
least-squares design matrix has ``d*(d+1)/2 = 210`` columns at ``d=20``, and the per-point
Python-level loop pays that cost ``n`` times. Running this cross-check to completion at every
cell in the full pre-registered grid would take multiple HOURS, far beyond
``WALL_CLOCK_BUDGET_S = 1800`` (30 minutes total, Section 9's own ratified figure). Per this
document's own escape valve ("A cell exceeding it is recorded as timed_out with its partial
state ... never silently dropped"), this runner therefore times out the EXPENSIVE, NON-GATING
quadric cross-check separately and more tightly than the two GATING statistics (which are
themselves comparatively cheap -- centroid estimation plus both permutation-null calibrations
completes in well under a minute per cell even at the PU regime's own scale) -- see STEP 2's
own comments for the exact budgets and the reasoning for treating the base cell specially. No
RATIFIED constant is touched by this choice: ``WALL_CLOCK_BUDGET_S`` itself, and every gating
threshold, are exactly as ratified. Only the internal ALLOCATION of that budget across a
cell's gating-vs-non-gating computation is this runner's own orchestration choice, disclosed
here and in the SUMMARY/FINDINGS documents rather than hidden.

Invoke: ``PYTHONPATH=notebooks python notebooks/diagnostics/curvature_feasibility_sweep_run.py``
"""

import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from pu_manifold import cache
from pu_manifold import curvature_probe as cp

# =============================================================================================
# Pre-registered constants, copied VERBATIM from 02.5-PREREGISTRATION.md Sections 3, 4, 6, 7,
# 8, 9 -- duplicated here rather than imported so the STEP 0 precondition assertions below run
# BEFORE anything else, mirroring topoae_evaluate_run.py:45-49's own explanatory comment. If a
# constant here ever drifts from the ratified document, STEP 0 halts naming exactly which one.
# =============================================================================================

PREREG_PATH = (
    ".planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-PREREGISTRATION.md"
)
PREREG_SHA = "eecdb15c70dc3f833e31c54ec7864543579dddb7"  # Section 15's recorded adding commit

NULL_QUANTILE = 0.99
N_PERMUTATIONS = 999
SPEARMAN_ABSOLUTE_FLOOR = 0.50
REGION_N_BINS = 4
REGION_PAIR_COUNT = 200_000
REGION_PAIR_SEED = 20260731
REGION_ABSOLUTE_FLOOR = 0.4750
K_GATING = 30
K_DENSITY = 30
D_PRIMARY = 20
LADDER = (8, 16, 20, 24, 32, 40)
WALL_CLOCK_BUDGET_S = 1800.0  # "30 minutes total on CPU" (Section 9)

# Fixture parameters (Section 4)
SIGMA = 0.6
AMPLITUDE = 1.0
DOMAIN_RADIUS = 2.0
DENSITY_SKEW_VALUES = (0.0, 3.0)
N_BUMPS_VALUES = (1, 8)

# "3 seeds at the base cell" (Section 9) -- the base cell's own seed (20260807, Section 9's
# BASE_CELL dict) plus two more, sequentially chosen (the pre-registration fixes the COUNT of
# seeds, not their numeric values, which are this runner's own orchestration choice).
SEEDS = (20260807, 20260808, 20260809)

BASE_CELL = {
    "d": 20,
    "D": 768,
    "n": 10_000,
    "k": 30,
    "n_bumps": 8,
    "density_correct": True,
    "k_density": 30,
    "seed": 20260807,
}

# One axis at a time from BASE_CELL (Section 9). "ladder_d" is D-07's own D_PRIMARY ladder
# {8,16,20,24,32,40} -- a DIFFERENT value set from the "d" axis's {2,5,10,20,25} and reported
# as its own axis in STEP 3's per-axis table, per Section 9's "Plus the full D_PRIMARY ladder
# ... at the base cell's other settings" (a distinct sweep, not a restatement of the "d" axis).
SWEEP_AXES = {
    "d": [2, 5, 10, 20, 25],
    "D": [BASE_CELL["d"] + BASE_CELL["n_bumps"], 50, 200, 768],
    "n": [1000, 3000, 10000],
    "k": [10, 15, 20, 30, 50, 100],
    "n_bumps": list(N_BUMPS_VALUES),
    "density_correct": [False, True],
    "density_skew": list(DENSITY_SKEW_VALUES),
    "ladder_d": list(LADDER),
}

# --- this runner's own orchestration choices -- NOT pre-registered constants, disclosed here
# and in 02.5-07-SUMMARY.md / 02.5-FINDINGS.md rather than silently baked in. See the module
# docstring's "wall-clock reality" note for the measured cost this responds to. Only the
# non-gating quadric cross-check is time-bounded (via curvature_probe.measure_cell's own
# quadric_timeout_s argument, added by this same plan) -- the two GATING statistics and both
# their permutation-null calibrations carry no timeout of their own because they are measured
# (module docstring) to reliably complete in well under a minute even at the PU regime's own
# n=10,000/D=768 scale.
# Re-tuned after the coordinator-directed shared-pass optimization
# (centroid_mean_curvature_both_densities) roughly halved the GATING computation's own
# cost (measured: two separate centroid_mean_curvature calls ~51s at PU-regime scale vs
# one shared pass ~27s, bit-identical -- see that function's own docstring). Even so,
# GATING ALONE across the full 25-cell grid (measured: ~14.7s per permutation-null
# calibration PER VARIANT, roughly independent of n/D/d since it is dominated by
# N_PERMUTATIONS=999 x REGION_PAIR_COUNT=200_000, plus the shared centroid pass) leaves
# only a modest remainder of WALL_CLOCK_BUDGET_S for the non-gating quadric cross-check --
# these two budgets were re-tuned from the pre-optimization values to fit that remainder,
# giving the base cell (needed for STEP 4's verdict) the larger share.
BASE_CELL_QUADRIC_TIMEOUT_S = 120.0  # the phase's single most important cell gets a real chance
OTHER_CELL_QUADRIC_TIMEOUT_S = 2.0  # a genuine, but tightly bounded, attempt for every other cell


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


# =============================================================================================
_banner("Phase 02.5 stage-1 feasibility sweep runner")

# =============================================================================================
_banner("STEP 0a -- precondition: 02.5-PREREGISTRATION.md exists at PREREG_PATH")

_prereg_file = Path(PREREG_PATH)
if not _prereg_file.exists():
    raise FileNotFoundError(
        f"STEP 0a failed: {PREREG_PATH} does not exist. The pre-registration must precede "
        "the measurement; this script refuses to guess at a rule nobody ratified."
    )
_prereg_text = _prereg_file.read_text()
print(f"  found: {PREREG_PATH} ({len(_prereg_text)} chars)")

# =============================================================================================
_banner("STEP 0b -- precondition: PREREG_SHA is a proven git ancestor of HEAD")

_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, "HEAD"])
if _ancestor_check.returncode != 0:
    raise RuntimeError(
        f"STEP 0b failed: pre-registration commit {PREREG_SHA} is NOT an ancestor of HEAD. "
        "The pre-registration must precede the measurement, and this ordering is PROVED "
        "(via git ancestry), never merely asserted."
    )
print(f"  pre-registration commit {PREREG_SHA} confirmed an ancestor of HEAD -- proved, not asserted")

# =============================================================================================
_banner("STEP 0c -- precondition: every constant in this file matches 02.5-PREREGISTRATION.md")


def _assert_constant_in_doc(name: str, value) -> None:
    """Halts unless `name` and THIS RUNNER'S OWN LIVE VALUE for it (never a hardcoded
    string independent of the variable actually defined above) co-occur on some line of
    the ratified document -- proving the constant as it stands in this file RIGHT NOW was
    not silently edited relative to what was ratified. Checks both the plain string form
    of `value` and, for integers of five digits or more, Python's underscore-grouped
    format (``f"{value:_}"``) too, since the document itself uses underscore grouping
    (e.g. ``200_000``) that ``str(int)`` alone never produces."""
    candidates = {str(value)}
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10_000:
        candidates.add(f"{value:_}")
    for line in _prereg_text.splitlines():
        if name in line and any(c in line for c in candidates):
            return
    raise RuntimeError(
        f"STEP 0c failed: constant {name!r}={value!r} (this runner's own live value; "
        f"checked forms {sorted(candidates)!r}) not found co-occurring on any line of "
        f"{PREREG_PATH} -- this runner's copy may have drifted from the ratified "
        "document. Halting before computing anything."
    )


_assert_constant_in_doc("NULL_QUANTILE", NULL_QUANTILE)
_assert_constant_in_doc("N_PERMUTATIONS", N_PERMUTATIONS)
_assert_constant_in_doc("SPEARMAN_ABSOLUTE_FLOOR", SPEARMAN_ABSOLUTE_FLOOR)
_assert_constant_in_doc("REGION_N_BINS", REGION_N_BINS)
_assert_constant_in_doc("REGION_PAIR_COUNT", REGION_PAIR_COUNT)
_assert_constant_in_doc("REGION_PAIR_SEED", REGION_PAIR_SEED)
_assert_constant_in_doc("REGION_ABSOLUTE_FLOOR", REGION_ABSOLUTE_FLOOR)
_assert_constant_in_doc("K_GATING", K_GATING)
_assert_constant_in_doc("K_DENSITY", K_DENSITY)
_assert_constant_in_doc("D_PRIMARY", D_PRIMARY)

_ladder_spaced = "{" + ", ".join(str(v) for v in LADDER) + "}"
_ladder_tight = "{" + ",".join(str(v) for v in LADDER) + "}"
if _ladder_spaced not in _prereg_text and _ladder_tight not in _prereg_text:
    raise RuntimeError(
        f"STEP 0c failed: this runner's own live LADDER={LADDER} -- neither "
        f"{_ladder_spaced!r} nor {_ladder_tight!r} found in the document."
    )

_bumps_spaced = "{" + ", ".join(str(v) for v in N_BUMPS_VALUES) + "}"
_bumps_tight = "{" + ",".join(str(v) for v in N_BUMPS_VALUES) + "}"
if _bumps_spaced not in _prereg_text and _bumps_tight not in _prereg_text:
    raise RuntimeError(
        f"STEP 0c failed: this runner's own live N_BUMPS_VALUES={N_BUMPS_VALUES} -- neither "
        f"{_bumps_spaced!r} nor {_bumps_tight!r} found in the document."
    )

assert WALL_CLOCK_BUDGET_S == 1800.0, (
    f"STEP 0c failed: this runner's own live WALL_CLOCK_BUDGET_S={WALL_CLOCK_BUDGET_S} does "
    "not equal 1800.0 (the ratified 30-minute-total figure)."
)
if "30 minutes" not in _prereg_text and "30-minute" not in _prereg_text:
    raise RuntimeError("STEP 0c failed: the 30-minute wall-clock budget prose not found in the document.")
print("  every pre-registered constant in this file confirmed present, verbatim (and tied to "
      "this runner's own live variable values), in the ratified document")

# =============================================================================================
_banner("STEP 0d -- precondition: this module's imports succeed and CURVATURE_CONVENTION is 'trace'")

assert cp.CURVATURE_CONVENTION == "trace", (
    f"STEP 0d failed: curvature_probe.CURVATURE_CONVENTION={cp.CURVATURE_CONVENTION!r}, "
    "expected 'trace'."
)
print(f"  curvature_probe imported OK, CURVATURE_CONVENTION={cp.CURVATURE_CONVENTION!r}")

# =============================================================================================
_banner("STEP 1 -- build the cell list: one axis at a time from BASE_CELL, deduplicated")


def _cell_from_base(**overrides) -> dict:
    cell = dict(BASE_CELL)
    cell["density_skew"] = 0.0  # BASE_CELL is implicitly uniform (Section 9's own BASE_CELL dict has none)
    cell.update(overrides)
    return cell


def _cell_key(cell: dict) -> tuple:
    return tuple(sorted(cell.items()))


CELL_REGISTRY: dict = {}  # canonical key -> cell dict (each UNIQUE cell computed exactly once)
AXIS_MEMBERSHIP: dict = {}  # axis name -> list of (axis_value, canonical key)


def _register(axis_name, axis_value, cell: dict) -> None:
    key = _cell_key(cell)
    CELL_REGISTRY.setdefault(key, cell)
    AXIS_MEMBERSHIP.setdefault(axis_name, []).append((axis_value, key))


_base_key = _cell_key(_cell_from_base())
CELL_REGISTRY[_base_key] = _cell_from_base()
AXIS_MEMBERSHIP.setdefault("base_cell", []).append(("BASE_CELL", _base_key))

for _value in SWEEP_AXES["d"]:
    _register("d", _value, _cell_from_base(d=_value))
for _value in SWEEP_AXES["D"]:
    _register("D", _value, _cell_from_base(D=_value))
for _value in SWEEP_AXES["n"]:
    _register("n", _value, _cell_from_base(n=_value))
for _value in SWEEP_AXES["k"]:
    _register("k", _value, _cell_from_base(k=_value))
for _value in SWEEP_AXES["n_bumps"]:
    _register("n_bumps", _value, _cell_from_base(n_bumps=_value))
for _value in SWEEP_AXES["density_correct"]:
    _register("density_correct", _value, _cell_from_base(density_correct=_value))
for _value in SWEEP_AXES["density_skew"]:
    _register("density_skew", _value, _cell_from_base(density_skew=_value))
for _value in SWEEP_AXES["ladder_d"]:
    _register("ladder_d", _value, _cell_from_base(d=_value))
for _value in SEEDS[1:]:
    _register("seed", _value, _cell_from_base(seed=_value))

_n_distinct_cells = len(CELL_REGISTRY)
_n_axis_entries = sum(len(v) for v in AXIS_MEMBERSHIP.values())
print(f"  {_n_axis_entries} axis-value entries across {len(AXIS_MEMBERSHIP)} axes "
      f"(including the base-cell anchor) map onto {_n_distinct_cells} DISTINCT cells after dedup")
print("  D-06: BOTH density_correct=False and True run at EVERY distinct cell (2 measure_cell "
      "calls per cell) regardless of whether density_correct is itself the swept axis")
print(f"  total measure_cell invocations this sweep will attempt: {_n_distinct_cells * 2}")

_infeasible_cells = sum(
    1 for c in CELL_REGISTRY.values() if not cp.centroid_tangent_basis_feasible(k=c["k"], d=c["d"])
)
_expensive_cells = sum(
    1
    for c in CELL_REGISTRY.values()
    if c["n"] >= 10_000 and c["d"] > 2 and cp.centroid_tangent_basis_feasible(k=c["k"], d=c["d"])
)
print(f"  {_infeasible_cells} of {_n_distinct_cells} cells are INFEASIBLE (k <= d -- cannot "
      "extract a d-dimensional tangent basis from k neighbours) and will be recorded without "
      "any estimator call, per the precondition check in STEP 2")
print(f"  {_expensive_cells} of {_n_distinct_cells} FEASIBLE cells are at the PU-regime's own "
      "n=10,000, D=768-ish scale, where the non-gating quadric cross-check alone costs several "
      "minutes (module docstring's 'wall-clock reality' note) -- most of these will exercise "
      f"the {OTHER_CELL_QUADRIC_TIMEOUT_S:.0f}s quadric timeout below rather than complete it")

# GATING-per-variant estimate below the shared-pass optimization (measured, PU-regime scale:
# permutation_null costs ~4.78s (spearman, N_PERMUTATIONS=999) + ~9.93s (region,
# REGION_PAIR_COUNT=200_000) per variant -- both roughly independent of n/D/d since they are
# dominated by the pre-registered N_PERMUTATIONS/REGION_PAIR_COUNT themselves, not by cell
# scale -- so this ~15s/variant floor applies to every FEASIBLE cell, cheap or expensive). The
# shared centroid_mean_curvature_both_densities pass itself is ~27s at D=768 (both variants
# combined, ~48% less than two separate calls) and considerably cheaper at the D-axis's
# smaller-D rows and the d=2 Swiss-roll anchor. Infeasible cells cost ~0s (a precondition
# check only, no fixture, no estimator call).
_est_null_calib_s_per_variant = 15.0
_est_shared_centroid_s_expensive = 27.0  # D=768-ish, both variants combined
_est_shared_centroid_s_cheap = 8.0  # smaller D / n / Swiss-roll anchor, both variants combined
_n_feasible_cheap = _n_distinct_cells - _expensive_cells - _infeasible_cells
_rough_gating_only_s = (
    _expensive_cells * (_est_shared_centroid_s_expensive + 2 * _est_null_calib_s_per_variant)
    + _n_feasible_cheap * (_est_shared_centroid_s_cheap + 2 * _est_null_calib_s_per_variant)
)
_rough_quadric_s = (
    (_n_distinct_cells - 1 - _infeasible_cells) * 2 * OTHER_CELL_QUADRIC_TIMEOUT_S
    + 2 * BASE_CELL_QUADRIC_TIMEOUT_S
)
_rough_total_estimate_s = _rough_gating_only_s + _rough_quadric_s
print(f"  rough GATING-ONLY estimate (shared-pass optimization applied): "
      f"~{_rough_gating_only_s / 60:.1f} minutes")
print(f"  rough quadric-timeout-budget estimate: ~{_rough_quadric_s / 60:.1f} minutes")
print(f"  rough total wall-clock ESTIMATE: ~{_rough_total_estimate_s / 60:.1f} minutes "
      f"(WALL_CLOCK_BUDGET_S nominal total: {WALL_CLOCK_BUDGET_S / 60:.0f} minutes) -- "
      "printed here so an over-budget sweep is visible BEFORE it starts, per Section 9")
if _rough_total_estimate_s > WALL_CLOCK_BUDGET_S:
    print(
        "  *** WARNING: this estimate EXCEEDS WALL_CLOCK_BUDGET_S. Per the sealed "
        "document's own rule, this runner does NOT silently overrun -- if the real run "
        "confirms this, it must halt and be reported as a checkpoint, not absorbed "
        "silently. ***"
    )

SWEEP_CFG = {
    "phase": "02.5",
    "stage": 1,
    "prereg_sha": PREREG_SHA,
    "null_quantile": NULL_QUANTILE,
    "n_permutations": N_PERMUTATIONS,
    "spearman_absolute_floor": SPEARMAN_ABSOLUTE_FLOOR,
    "region_n_bins": REGION_N_BINS,
    "region_pair_count": REGION_PAIR_COUNT,
    "region_pair_seed": REGION_PAIR_SEED,
    "region_absolute_floor": REGION_ABSOLUTE_FLOOR,
    "k_gating": K_GATING,
    "k_density": K_DENSITY,
    "d_primary": D_PRIMARY,
    "ladder": list(LADDER),
    "wall_clock_budget_s": WALL_CLOCK_BUDGET_S,
    "sigma": SIGMA,
    "amplitude": AMPLITUDE,
    "domain_radius": DOMAIN_RADIUS,
    "density_skew_values": list(DENSITY_SKEW_VALUES),
    "n_bumps_values": list(N_BUMPS_VALUES),
    "seeds": list(SEEDS),
    "base_cell": BASE_CELL,
    "sweep_axes": SWEEP_AXES,
}
stage1_key = cache.config_key(SWEEP_CFG)
print(f"  stage1_key = {stage1_key}")

# STEP 2/3's banners print inside _compute_sweep below, at the point they actually run --
# cache.json_cache only calls _compute_sweep on a cache MISS, so printing here (at module
# scope) would misleadingly claim STEP 2/3 ran even on a pure cache-hit re-invocation.


def _make_fixture(cell: dict) -> dict:
    if cell["d"] == 2:
        # The mandatory Swiss roll d=2 anchor (D-03) -- never substituted, per CLAUDE.md.
        return cp.make_swiss_roll_fixture(n=cell["n"], seed=cell["seed"])
    return cp.make_graph_of_function_fixture(
        n=cell["n"],
        d=cell["d"],
        D=cell["D"],
        n_bumps=cell["n_bumps"],
        seed=cell["seed"],
        sigma=SIGMA,
        amplitude=AMPLITUDE,
        density_skew=cell["density_skew"],
        domain_radius=DOMAIN_RADIUS,
    )


def _measure_one(
    fixture: dict,
    cell: dict,
    density_correct: bool,
    quadric_budget_s: float,
    precomputed_H_vec: np.ndarray,
) -> dict:
    """One (cell, density_correct) measurement -- a single call to
    ``curvature_probe.measure_cell``, exactly as Task 1's own action text specifies, so this
    runner contains orchestration only and no statistics of its own. The two GATING
    statistics and both their permutation-null calibrations are always computed and
    returned (measured, in every case tried while writing this runner, to complete in well
    under a minute even at the PU regime's own n=10,000/D=768 scale). The non-gating quadric
    cross-check is bounded by `measure_cell`'s own `quadric_timeout_s` argument (added by
    this same plan) -- if it does not finish in time, `measure_cell` itself records
    `quadric_timed_out: True` with the quadric-derived fields set to `None`, while still
    returning the cell's GATING result (the actual boundary-relevant data) rather than
    discarding it. See `measure_cell`'s own docstring and this module's 'wall-clock
    reality' note above for the measured cost this responds to -- not a change to any
    ratified statistic or threshold, only to how this runner's own time budget is spent.

    ``precomputed_H_vec`` (coordinator-directed shared-pass optimization, plan 02.5-07):
    the centroid estimate for THIS density_correct variant, already computed once by
    `curvature_probe.centroid_mean_curvature_both_densities` for BOTH variants in a single
    pass over the cell's fixture -- shared here via `measure_cell`'s own
    `precomputed_H_vec` argument, so this call never re-runs the k-NN query or any
    per-point tangent-basis SVD a second time. Bit-identical to letting `measure_cell`
    compute it internally (pinned by
    `test_measure_cell_precomputed_h_vec_is_bit_identical_and_shares_the_pass` and
    `test_centroid_mean_curvature_both_densities_is_bit_identical` in
    `test_curvature_probe.py`)."""
    t0 = time.perf_counter()
    result = cp.measure_cell(
        fixture=fixture,
        k=cell["k"],
        d=cell["d"],
        k_density=K_DENSITY,
        density_correct=density_correct,
        n_resamples=N_PERMUTATIONS,
        seed=cell["seed"],
        quantile=NULL_QUANTILE,
        region_n_bins=REGION_N_BINS,
        region_pair_seed=REGION_PAIR_SEED,
        region_n_pairs=REGION_PAIR_COUNT,
        quadric_timeout_s=quadric_budget_s,
        precomputed_H_vec=precomputed_H_vec,
    )
    result["status"] = "partial_quadric_timeout" if result["quadric_timed_out"] else "complete"
    result["wrapper_elapsed_s"] = float(time.perf_counter() - t0)
    return result


def _compute_sweep() -> dict:
    _banner("STEP 2 -- run the cells (per-cell time budgets; timed-out cells recorded, never dropped)")

    per_cell_results: dict = {}
    sweep_t0 = time.perf_counter()

    for _i, (key, cell) in enumerate(CELL_REGISTRY.items(), start=1):
        is_base = key == _base_key
        quadric_budget = BASE_CELL_QUADRIC_TIMEOUT_S if is_base else OTHER_CELL_QUADRIC_TIMEOUT_S

        # Coordinator's explicit requirement: a cell that is TIMED OUT mid-computation
        # (recorded status="timed_out"/"partial_quadric_timeout") is compliant with the
        # sealed text; a cell NEVER ATTEMPTED because the cumulative budget ran out in
        # iteration order is NOT -- that would be an order-dependent, silently incomplete
        # boundary. If the running total already exceeds WALL_CLOCK_BUDGET_S before this
        # cell (the base cell is always first -- see STEP 1 -- so it is never starved),
        # HALT LOUDLY here rather than silently skip the remainder of the grid.
        _elapsed_before_this_cell = time.perf_counter() - sweep_t0
        if _elapsed_before_this_cell > WALL_CLOCK_BUDGET_S and _i > 1:
            raise RuntimeError(
                f"STEP 2 halted: cumulative wall-clock ({_elapsed_before_this_cell / 60:.1f} "
                f"min) already exceeds WALL_CLOCK_BUDGET_S ({WALL_CLOCK_BUDGET_S / 60:.0f} "
                f"min) before cell [{_i}/{_n_distinct_cells}] was ever attempted. Per the "
                "sealed document's own rule, this is NOT a timed-out cell (which would be "
                "compliant, per Section 9's own tolerance) -- it is a cell this runner "
                "would otherwise silently never measure at all, purely because of "
                "iteration order. Halting rather than producing an order-dependent, "
                "silently incomplete boundary. This must be reported as a checkpoint, not "
                "resolved by shrinking the grid or the timeouts further without review."
            )

        cell_label = ", ".join(f"{k}={v}" for k, v in sorted(cell.items()))
        print(f"  [{_i}/{_n_distinct_cells}] {'BASE ' if is_base else ''}cell: {cell_label}")

        # Precondition check (coordinator-directed, added after this runner crashed
        # mid-sweep on k=10, d=20): centroid_mean_curvature_both_densities's shared
        # local_tangent_basis SVD requires k > d strictly (curvature_probe's own
        # centroid_tangent_basis_feasible predicate, k == d also treated as infeasible --
        # see that function's docstring). Checked BEFORE building the fixture or running
        # any k-NN query, not by catching the ValueError after paying for that work. An
        # infeasible cell is neither a PASS nor a FAIL -- it contributes no gating verdict
        # and is recorded, never silently dropped, per Section 9's own tolerance for a
        # cell that cannot produce a number.
        if not cp.centroid_tangent_basis_feasible(k=cell["k"], d=cell["d"]):
            reason = (
                f"k={cell['k']} <= d={cell['d']}: cannot extract a {cell['d']}-dimensional "
                f"tangent basis from {cell['k']} neighbours -- k must strictly exceed d "
                "for centroid_mean_curvature's local_tangent_basis SVD to have any "
                "redundancy left to average the tangent direction over. This is a hard "
                "structural floor of the point-cloud estimator, not a bug and not a gate "
                "outcome (neither PASS nor FAIL)."
            )
            per_cell_results[key] = {
                "cell_params": cell,
                "status": "infeasible_k_le_d",
                "reason": reason,
                "variants": {},
            }
            print(f"      INFEASIBLE (k<=d), not attempted: {reason}")
            continue

        try:
            fixture = _make_fixture(cell)
        except Exception as exc:  # a fixture that cannot be built at all is a full timeout/failure
            per_cell_results[key] = {
                "cell_params": cell,
                "status": "timed_out",
                "elapsed_s": 0.0,
                "error": f"fixture construction failed: {exc}",
                "variants": {},
            }
            print(f"      fixture construction FAILED: {exc}")
            continue

        # Coordinator-directed shared-pass optimization: the k-NN query and every
        # per-point tangent-basis SVD are IDENTICAL between the two density_correct
        # variants (only the cheap gap/r2 sums differ) -- compute both variants' centroid
        # estimate in ONE pass, bit-identical to two separate centroid_mean_curvature
        # calls (test_centroid_mean_curvature_both_densities_is_bit_identical), instead of
        # paying for that expensive shared part twice.
        H_uncorrected, H_corrected = cp.centroid_mean_curvature_both_densities(
            fixture["X"], k=cell["k"], d=cell["d"], k_density=K_DENSITY
        )
        precomputed_by_variant = {False: H_uncorrected, True: H_corrected}

        variants = {}
        for density_correct in (False, True):
            # No outer signal-based timeout wraps this call: measure_cell's OWN
            # quadric_timeout_s already guards the one expensive (non-gating) part
            # internally (cp.QuadricTimeout, caught inside measure_cell itself), and
            # signal.setitimer is a single, non-nestable, per-process resource -- an
            # outer alarm here would silently stomp on measure_cell's own inner one.
            # The gating computation ahead of the quadric guard is measured (module
            # docstring) to reliably complete in well under a minute even at the PU
            # regime's own scale, so no separate outer guard is needed around it.
            variant_result = _measure_one(
                fixture, cell, density_correct, quadric_budget,
                precomputed_by_variant[density_correct],
            )
            variants[str(density_correct)] = variant_result
            elapsed = variant_result["wrapper_elapsed_s"]
            status = variant_result["status"]
            print(f"      density_correct={density_correct}: {status} in {elapsed:.1f}s "
                  f"(spearman_rho={variant_result['spearman_rho']:.4f}, "
                  f"quantile_bin_concordance={variant_result['quantile_bin_concordance']:.4f})")

        per_cell_results[key] = {
            "cell_params": cell,
            "status": "complete",
            "variants": variants,
        }

        elapsed_total = time.perf_counter() - sweep_t0
        print(f"      cumulative sweep wall-clock so far: {elapsed_total / 60:.1f} min")

    total_elapsed_s = time.perf_counter() - sweep_t0

    _banner("STEP 3 -- locate the feasibility boundary, per axis")

    def _cell_clears(variant_result: dict) -> bool:
        if variant_result.get("status") == "timed_out":
            return None
        spearman_threshold = max(variant_result["null_threshold"], SPEARMAN_ABSOLUTE_FLOOR)
        region_threshold = max(variant_result["region_null_threshold"], REGION_ABSOLUTE_FLOOR)
        metrics = {
            "spearman_rho": variant_result["spearman_rho"],
            "quantile_bin_concordance": variant_result["quantile_bin_concordance"],
        }
        thresholds = {
            "spearman_rho": spearman_threshold,
            "quantile_bin_concordance": region_threshold,
        }
        verdict, gate_detail = cp.verdict_from_stage1_metrics(metrics, thresholds)
        return {
            "verdict": verdict,
            "clears": verdict == "PASS",
            "gate_detail": gate_detail,
            "spearman_threshold": spearman_threshold,
            "region_threshold": region_threshold,
        }

    per_axis_boundary: dict = {}
    for axis_name, entries in AXIS_MEMBERSHIP.items():
        rows = []
        for axis_value, cell_key in sorted(entries, key=lambda e: (str(type(e[0])), e[0])):
            cell_record = per_cell_results[cell_key]
            row = {"axis_value": axis_value, "cell_status": cell_record["status"]}
            for density_correct_str in ("False", "True"):
                variant = cell_record.get("variants", {}).get(density_correct_str)
                if variant is None:
                    # A cell-level status ("infeasible_k_le_d" or "timed_out" from a
                    # failed fixture build) means BOTH variants share that same reason --
                    # label it explicitly rather than the uninformative generic "missing"
                    # (which would read as "we ran out of time to try this", conflating a
                    # structural impossibility with a budget outcome).
                    row[f"density_correct_{density_correct_str}"] = {
                        "status": cell_record["status"],
                        "reason": cell_record.get("reason") or cell_record.get("error"),
                    }
                    continue
                verdict_info = _cell_clears(variant)
                row[f"density_correct_{density_correct_str}"] = {
                    "status": variant.get("status"),
                    "spearman_rho": variant.get("spearman_rho"),
                    "quantile_bin_concordance": variant.get("quantile_bin_concordance"),
                    "verdict": verdict_info["verdict"] if verdict_info else None,
                    "clears": verdict_info["clears"] if verdict_info else None,
                }
            rows.append(row)

        # crossing point: at density_correct=True (BASE_CELL's own regime), the last value
        # that clears and the first that does not, walking the axis in value order.
        clears_sequence = [
            r["density_correct_True"].get("clears") for r in rows
        ]
        crossing = None
        if all(c is True for c in clears_sequence if c is not None) and any(
            c is not None for c in clears_sequence
        ):
            crossing = "clears throughout (density_correct=True)"
        elif all(c is False for c in clears_sequence if c is not None) and any(
            c is not None for c in clears_sequence
        ):
            crossing = "fails throughout (density_correct=True)"
        else:
            last_clear, first_fail = None, None
            for r, c in zip(rows, clears_sequence):
                if c is True:
                    last_clear = r["axis_value"]
                if c is False and first_fail is None:
                    first_fail = r["axis_value"]
            crossing = {"last_clearing_value": last_clear, "first_failing_value": first_fail}

        per_axis_boundary[axis_name] = {"rows": rows, "crossing": crossing}

    base_variants = per_cell_results[_base_key]["variants"]
    base_true = base_variants.get("True")
    base_verdict_info = _cell_clears(base_true) if base_true else None

    env = {
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "stage1_key": stage1_key,
        "sweep_cfg": SWEEP_CFG,
        "n_distinct_cells": _n_distinct_cells,
        "per_cell_results": {str(k): v for k, v in per_cell_results.items()},
        "per_axis_boundary": per_axis_boundary,
        "base_cell": BASE_CELL,
        "base_cell_verdict": base_verdict_info,
        "total_elapsed_s": float(total_elapsed_s),
        "environment": env,
    }


# =============================================================================================
_banner("STEP 4 -- persist through cache.json_cache, then write the stage-1 verdict")

sweep_result = cache.json_cache(f"curvature_feasibility_{stage1_key}", SWEEP_CFG, _compute_sweep)

_base_verdict_info = sweep_result["base_cell_verdict"]
if _base_verdict_info is None:
    raise RuntimeError(
        "STEP 4 failed: the base cell's own density_correct=True measurement did not "
        "complete (its gating computation is measured to reliably finish in well under a "
        "minute even at this scale, and its non-gating quadric cross-check has its own "
        f"extended {BASE_CELL_QUADRIC_TIMEOUT_S:.0f}s budget) -- cannot write a stage-1 "
        "verdict without the base cell's own measured rho. This is itself a reportable "
        "finding if it occurs; it must not be silently worked around."
    )

_base_metrics = {
    "spearman_rho": sweep_result["per_cell_results"][str(_base_key)]["variants"]["True"]["spearman_rho"],
    "quantile_bin_concordance": sweep_result["per_cell_results"][str(_base_key)]["variants"]["True"]["quantile_bin_concordance"],
}
_base_thresholds = {
    "spearman_rho": _base_verdict_info["spearman_threshold"],
    "quantile_bin_concordance": _base_verdict_info["region_threshold"],
}
_base_verdict_str = _base_verdict_info["verdict"]

_context_extra = {
    "per_axis_boundary": sweep_result["per_axis_boundary"],
    "n_distinct_cells": sweep_result["n_distinct_cells"],
    "total_elapsed_s": sweep_result["total_elapsed_s"],
    "environment": sweep_result["environment"],
    "quadric_context": {
        "base_variant": sweep_result["per_cell_results"][str(_base_key)]["variants"]["True"],
    },
}

written_verdict = cp.write_curvature_verdict(
    key=stage1_key,
    stage=1,
    metrics=_base_metrics,
    thresholds=_base_thresholds,
    verdict=_base_verdict_str,
    extra=_context_extra,
)

# Stage 1 writes NO Phase 3 handoff under any circumstance -- see module docstring. The
# handoff-writing sibling function is never called or named anywhere in this file.

_verdict_path = cache.cache_path(f"curvature_verdict_stage1_{stage1_key}", "json")
print(f"  written to: {_verdict_path}")
print(f"  CURVATURE_VERDICT = {_base_verdict_str}")

# =============================================================================================
_banner("STEP 5 -- read-out")

print(f"  base cell: rho(spearman)={_base_metrics['spearman_rho']:.4f} "
      f"(threshold {_base_thresholds['spearman_rho']:.4f}), "
      f"quantile_bin_concordance={_base_metrics['quantile_bin_concordance']:.4f} "
      f"(threshold {_base_thresholds['quantile_bin_concordance']:.4f})")
print(f"  CURVATURE_VERDICT (stage 1) = {_base_verdict_str}")
print()
print("  per-axis crossing table:")
for axis_name, info in sweep_result["per_axis_boundary"].items():
    print(f"    {axis_name}: {info['crossing']}")
print()
print(f"  total sweep wall-clock: {sweep_result['total_elapsed_s'] / 60:.1f} minutes "
      f"(nominal budget: {WALL_CLOCK_BUDGET_S / 60:.0f} minutes)")
print()
if _base_verdict_str != "PASS":
    print(
        "  D-04: the base cell does not clear -- per 02.5-PREREGISTRATION.md Section 10, the "
        "phase HALTS for a user decision with NO auto-fallback. Ollivier-Ricci / intrinsic "
        "curvature is the live remaining option: it is defined on the k-NN graph via optimal "
        "transport, never touches the ambient space, and so is not subject to the sampling "
        "constraint that just failed."
    )
else:
    print("  D-04: the base cell clears both gates -- stage 2 may proceed to plan 02.5-08, "
          "pending the Task 3 human checkpoint's own review of the full boundary report.")
print()
print("  No 'recommended k' is emitted by this script, per D-04/Section 8's own prohibition.")
print()
print(f"Done. stage1_key = {stage1_key}")
