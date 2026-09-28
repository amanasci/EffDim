"""
notebooks/diagnostics/template_benchmark_run.py -- Phase 02.7 Plan 10: D-15's benchmark
runner. Resumable, fair-by-assertion, per-cell timed.

**This runner MEASURES and ASSEMBLES; it does not choose or re-tune any constant, and it
DECIDES nothing on its own.** Unlike the 02.6 axis (no ratified decision rule existed
there at all), this phase HAS a ratified decision rule -- the separation this runner keeps
is therefore between itself (which composes measurements and calls the ratified rule
module once per cloud) and `template_decision.decide`/`.tally` (which apply the actual
Betti-vector-lookup criterion, thresholds and abstain conditions). Every ratified constant
below is READ, verbatim, from `02.7-SCREENING-RULE.md` (committed alone at
`7aa699de94206efe141288b2480eb1d42c045029`) and its `AMENDMENT-01` (committed alone at
`948d13fe4a16e68a3a5c13da504ce6ba0ad769f6`) -- this module introduces no new default for
any of them, and chooses no grid level, threshold or abstain rule of its own.

**D-12's amendment is honored via a general structural predicate, not a name list.** This
runner assembles `dimension_readout["local_dispersion"]` for `template_decision.decide`
via `local_dimension.gating_dispersion` (added in `235916c`), never via
`local_dimension.dispersion` directly -- `gating_dispersion` excludes any estimator whose
local call is not provenance-identical to its global call (currently `gmst`, as a
CONSEQUENCE of that estimator's own call-path asymmetry, never hardcoded by name here).
`local_dimension.dispersion`'s own full, unrestricted 8-estimator report is ALSO recorded
on every draw (`local_dispersion_full`) purely for completeness -- `decide` itself is
never handed that unrestricted dict.

**The `ball` timeout fallback (ratified, `02.7-SCREENING-RULE.md` Sec.9) is implemented,
not aspirational.** If a `ball` combination's wall clock would exceed roughly
`2 x` the same grid configuration's own measured `S2` combination cost, this runner
enforces that ceiling via a hard, forked-subprocess timeout (:func:`run_with_timeout`,
the same isolation shape `ph_budget_calibration_run.py`'s own `run_timed` uses) and
records the `ball` cell as an incomplete `"timeout"` finding -- it never blocks or crashes
the rest of the grid. When no same-configuration `S2` cost has been measured yet (the
combination order below runs `S2` before `ball` within every configuration specifically so
this reference is usually available, but a `--resume`'d partial run or a `--templates
ball`-only invocation may not have one), a stated, documented fallback reference cost is
used instead (`FALLBACK_S2_PER_COMBO_S`, sourced directly from
`02.7-BUDGET-CALIBRATION.md` Sec.4's own measured `S2` whole-cell total, scaled by
`DRAWS`) -- never silently assumed precise.

**The `n_ph` fairness assertion (D-08) is asserted in code, not stated in a comment.**
:func:`_assert_budget_parity` raises :class:`BudgetParityError` if the two metric arms are
ever handed different point counts, or if either differs from the cell's own ratified
`N_PH` -- called on every draw, before either arm's distance matrix is built. The ONE
place the two arms legitimately differ afterward is D-07's own no-bridging rule: the
geodesic arm computes its persistence diagram on the largest connected component only, so
its EFFECTIVE point count can end up lower than what it was OFFERED when the cloud
disconnects. That is tracked separately (`geo_dropped_fraction`), never folded into the
fairness assertion, and a disconnected cell already routes to abstain condition (b)
through `connectivity_readout["n_components"] > 1` -- D-08's own instruction is honored
structurally, not by a second, redundant check here.

**Immersion validity (D-14) is checked on every generated cloud, not assumed.**
`template_immersion.jacobian_rank` is called, at the ratified `n_check`/`rank_tol`, on
every draw before either metric arm is measured. A cloud whose minimum rank does not equal
its own template's true intrinsic dimension is EXCLUDED from the tally (its ground-truth
label is not trustworthy) and the exclusion is counted and reported, never silently
dropped.

**Resumability is the point.** Every (grid configuration, template) combination -- the
"cell" `02.7-BUDGET-CALIBRATION.md`'s own arithmetic counts (60 of them at the full ratified
grid) -- is written to an append-only JSON-lines record as it completes
(:func:`append_record`), and `--resume` skips any combination already present in that
record (:func:`load_completed`) rather than recomputing it. Per-combination wall clock is
printed as each one finishes, together with a running mean-based projection of the time
remaining -- so a multi-hour unattended run can be interrupted and resumed in segments
without losing completed work or hiding its own cost.

**No composite, weighted or averaged figure is ever computed across the two metric arms,
across templates, or across abstain conditions.** Every read-out this runner emits is
either scalar-per-metric-label (mirroring `template_decision.assert_labelled`'s own
invariant) or reported per template/per configuration separately. `template_decision.tally`
is called exactly once, at the very end, over every valid (non-excluded) draw across the
whole run -- the single D-13 three-way tally (`correct`/`wrong`/`abstained`/`total`, the
three separate rates, and `by_condition`), never an accuracy computed over non-abstained
cells only.

**Two implementation decisions this runner makes that the ratified rule does NOT pin
down** (stated here explicitly, exactly as `02.7-SCREENING-RULE.md`'s own convention
requires for anything that is a judgment call rather than a measured or ratified number):

1. **Which single `k` actually builds the geodesic distance matrix and the local-dispersion
   neighbourhood width.** D-06/D-10 ratify the SWEEP (`K_VALUES`, `MIN_RUN`) and the
   stability rule's discipline, but `ripser` needs one concrete `D_geo` and
   `local_estimates` needs one concrete `k` -- neither module picks a winning `k` for its
   caller (`geodesic_graph.contiguous_stable_range`'s own docstring: "deciding whether a
   curve counts as stable" is the caller's job). :func:`_select_geodesic_k` uses the FIRST
   `k` of `n_components`'s longest stable run (cheapest connected `k`) when that run reaches
   `MIN_RUN`, and falls back to the largest swept `k` (most likely to be connected) when it
   does not -- an instability that itself does not silently vanish: `k_stable_enough=False`
   is recorded on every draw. :func:`_select_dimension_k` uses the LATEST `start_index`
   among the winning `d_hat`'s own supporting estimators' runs (guaranteeing the chosen `k`
   sits inside every one of those runs), falling back to the largest swept `k` when there is
   no consensus at all (in which case `decide` abstains via (d) before this choice matters).
2. **Ground-truth correctness for the `ball` template.** D-04's joint key means `lookup`
   returns `"ball_d{d_hat}"` for WHATEVER `d_hat` the measurement itself produced, not the
   cloud's own true `BALL_D` -- the benchmark's job is to test whether the FAMILY is
   recovered, not whether the measured `d_hat` exactly equals the ground-truth dimension (a
   separate question the benchmark's own `d_hat`-versus-`d_true` read-out already answers
   without conflating it with the three-way tally). :func:`_classify_outcome` therefore
   scores a `ball` draw `"correct"` iff `decide`'s returned template starts with `"ball_d"`,
   regardless of which `d_hat` it names.

Invoke:
    .venv/bin/python notebooks/diagnostics/template_benchmark_run.py --smoke
    .venv/bin/python notebooks/diagnostics/template_benchmark_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/template_benchmark_run.py
    .venv/bin/python notebooks/diagnostics/template_benchmark_run.py --resume
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effdim import geometry  # noqa: E402
from pu_manifold import (  # noqa: E402
    confidence_band,
    geodesic_graph,
    local_dimension,
    persistence_probe,
    template_decision,
    template_immersion,
)
from pu_manifold.cache import CACHE_DIR  # noqa: E402

# =============================================================================================
# Ratified constants -- every one of these is READ verbatim from 02.7-SCREENING-RULE.md
# (7aa699de94206efe141288b2480eb1d42c045029) and its AMENDMENT-01
# (948d13fe4a16e68a3a5c13da504ce6ba0ad769f6). This module chooses none of them and
# introduces no default for any of them; see each constant's own source citation below.
# =============================================================================================

TEMPLATES: Tuple[str, ...] = ("S1", "S2", "T2", "ball")
"""Fixed order deliberately: within any one grid configuration, S1/S2/T2 all run before
`ball`, so a same-configuration S2 cost is usually already on record by the time `ball`'s
combination is reached (feeding the ratified timeout fallback's reference cost, Sec.9)."""

N_PH = 300
"""SCREENING-RULE.md Sec.7, D-08: pinned identically across the Euclidean arm, the
geodesic arm, and every benchmark grid cell -- except the n-OFAT arm's own three levels
(N_OFAT_LEVELS below), which vary n_ph BY DESIGN as the ratified grid's own n-arm."""

B = 10
ALPHA = 0.05
"""SCREENING-RULE.md Sec.2, D-02."""

DRAWS = 3
"""SCREENING-RULE.md Sec.7/Sec.9, D-08: independent immersion draws per grid cell --
fresh clouds with new sampling/noise realizations, feeding assumption A-04 (PH draw
variance). A DIFFERENT loop from D-02's within-cloud bootstrap (B above); costs multiply."""

K_VALUES: Tuple[int, ...] = (5, 10, 15, 20, 30)
MIN_RUN = 3
TOLERANCE = 0.5
COUNT_DISTINCT = True
MAJORITY = 5
"""SCREENING-RULE.md Sec.4, D-09/D-10/D-12; shared verbatim by D-06's geodesic k-sweep."""

DISPERSION_BOUND = 3.0
SPREAD_BOUND = 3.0
"""SCREENING-RULE.md Sec.5, D-03(b)/D-03(d) -- judgment calls, not calibration-grounded,
but ratified exactly as written; this module supplies neither a default nor a substitute."""

NOISE_LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.1)
D_LEVELS: Tuple[int, ...] = (18, 100, 768)
"""SCREENING-RULE.md Sec.8, D-13/D-15. D=768 is a REQUIRED level -- it MUST actually
execute in the real grid, never be skipped for cost (critical execution note #7)."""

BASELINE_D = 768
BASELINE_NOISE = 0.05
BASELINE_N_PH = N_PH
BASELINE_DENSITY = 1.0
"""SCREENING-RULE.md Sec.8: the ratified baseline cell the two OFAT arms sweep off of."""

N_OFAT_LEVELS: Tuple[int, ...] = (200, 300, 400)
DENSITY_OFAT_LEVELS: Tuple[float, ...] = (0.5, 1.0, 2.0)
"""SCREENING-RULE.md Sec.8: the two one-factor-at-a-time arms off the baseline cell."""

WARP_PARAMS: Dict[str, float] = {"strength": 0.05, "freq": 1.0}
"""SCREENING-RULE.md Sec.7, D-14 -- the exact pair 02.7-05 measured all four templates
PASS the immersion Jacobian rank check at, at D=768."""

BALL_D = 18
"""SCREENING-RULE.md Sec.7, D-14 -- carried forward from 02.7-BUDGET-CALIBRATION.md's own
proxy value for cost-transfer reasons, stated there as not independently
calibration-grounded for correctness."""

JACOBIAN_N_CHECK = 30
JACOBIAN_RANK_TOL = 1e-6
"""SCREENING-RULE.md Sec.7, D-14: `jacobian_rank(n_check=30, rank_tol=1e-6)` -- required
arguments with no module default; this runner is the one place they are supplied."""

NEIGHBOURHOOD_SIZE = 30
N_ANCHORS = 40
"""SCREENING-RULE.md Sec.4, D-12."""

BALL_TIMEOUT_MULTIPLIER = 2.0
"""SCREENING-RULE.md Sec.9: "if ball's wall clock ... exceeds ~2x S2's per-cell cost, the
benchmark runner reports ball's cells as an incomplete/timeout finding rather than
blocking the whole grid" -- ratified blind, before this runner or any real cell existed."""

FALLBACK_S2_PER_COMBO_S = 341.03 * DRAWS
"""Stated fallback reference cost for the ball-timeout ceiling when no same-configuration
S2 combination has completed yet in this run's own record (e.g. a --resume that has not
reached S2 for that configuration, or a --templates ball-only invocation).
341.03s is 02.7-BUDGET-CALIBRATION.md Sec.4's own measured ONE-DRAW S2 whole-cell total
(both metrics x 3 degrees x (1 + B=10) calls) at the ratified baseline configuration
(n_ph=300, B=10, D=768) -- scaled by DRAWS to approximate a whole COMBINATION (not a
single draw). This is coarse and configuration-invariant by construction (it does not
account for a different configuration's own noise/D/n cost); it is used ONLY as a
fallback when a real, same-configuration measurement is not yet available, and a real
measurement always takes precedence once one exists (see _s2_reference_cost)."""

# =============================================================================================
# --smoke: a tiny, wall-clock-tractable subset (one template, the baseline cell's own
# D/noise/density, minimal draws and B) -- n_ph is a STATED reduction from the ratified 300
# purely for wall-clock tractability, the same precedent 02.7-09's own Swiss-roll notebook
# used (its n_ph=150 reduction, verified there not to change the wiring's own correctness).
# =============================================================================================

SMOKE_N_PH = 80
SMOKE_B = 2
SMOKE_DRAWS = 1
SMOKE_TEMPLATES: Tuple[str, ...] = ("S1",)


class BudgetParityError(ValueError):
    """D-08's fairness assertion, raised in code -- see _assert_budget_parity."""


# =============================================================================================
# Grid construction -- the ratified full factorial on (D, noise) plus the two OFAT arms off
# the baseline cell. 3x3 (9) + 3 (n-OFAT) + 3 (density-OFAT) = 15 configurations x 4
# templates = 60 (configuration, template) combinations (SCREENING-RULE.md Sec.8).
# =============================================================================================


def build_grid_configs() -> List[Dict[str, Any]]:
    """The 15 ratified grid configurations, each carrying its own D/noise/n_ph/density and
    a stable `config_id` (used both for the resumability key and the ball-timeout
    same-configuration S2 lookup)."""
    configs: List[Dict[str, Any]] = []

    for D in D_LEVELS:
        for noise in NOISE_LEVELS:
            configs.append(
                {
                    "config_id": f"factorial_D{D}_noise{noise}",
                    "arm": "factorial",
                    "D": D,
                    "noise": noise,
                    "n_ph": BASELINE_N_PH,
                    "density": BASELINE_DENSITY,
                }
            )

    for n in N_OFAT_LEVELS:
        configs.append(
            {
                "config_id": f"n_ofat_n{n}",
                "arm": "n_ofat",
                "D": BASELINE_D,
                "noise": BASELINE_NOISE,
                "n_ph": n,
                "density": BASELINE_DENSITY,
            }
        )

    for density in DENSITY_OFAT_LEVELS:
        configs.append(
            {
                "config_id": f"density_ofat_density{density}",
                "arm": "density_ofat",
                "D": BASELINE_D,
                "noise": BASELINE_NOISE,
                "n_ph": BASELINE_N_PH,
                "density": density,
            }
        )

    return configs


def build_combinations(
    configs: Sequence[Dict[str, Any]], templates: Sequence[str]
) -> List[Tuple[Dict[str, Any], str]]:
    """(configuration, template) pairs, templates in :data:`TEMPLATES`' own fixed order
    within each configuration (S2 before ball) -- never re-ordered by a caller-supplied
    template subset."""
    ordered_templates = [t for t in TEMPLATES if t in templates]
    combos: List[Tuple[Dict[str, Any], str]] = []
    for config in configs:
        for template in ordered_templates:
            combos.append((config, template))
    return combos


# =============================================================================================
# Deterministic seed derivation -- independent of run order, so a --resume'd run reproduces
# bit-identical draws for any combination it re-executes.
# =============================================================================================


def _combo_seed(config_id: str, template: str) -> int:
    digest = hashlib.sha256(f"template_benchmark|{config_id}|{template}".encode()).hexdigest()
    return int(digest[:15], 16) % (2**31 - 1)


def _draw_seed(combo_seed: int, draw_index: int) -> int:
    digest = hashlib.sha256(f"{combo_seed}|draw{draw_index}".encode()).hexdigest()
    return int(digest[:15], 16) % (2**31 - 1)


# =============================================================================================
# D-08's fairness assertion, in code.
# =============================================================================================


def _assert_budget_parity(n_offered: Dict[str, int], expected_n_ph: int) -> None:
    """Raises :class:`BudgetParityError` if the two metric arms were offered different
    point counts, or if either differs from the cell's own ratified `n_ph` -- D-08's "same
    budget ... asserted in code" discipline, carried forward from 02.6-FINDINGS-02.md's own
    training-budget-asymmetry finding. Called on every draw, before either arm's distance
    matrix is built."""
    distinct = set(n_offered.values())
    if len(distinct) != 1:
        raise BudgetParityError(
            f"budget parity violated (D-08): metric arms were offered different point "
            f"counts {n_offered!r} -- n_ph must be pinned identically across the "
            f"euclidean arm, the geodesic arm and every benchmark cell"
        )
    (actual,) = distinct
    if actual != expected_n_ph:
        raise BudgetParityError(
            f"budget parity violated (D-08): arms were offered n={actual} points but this "
            f"cell's ratified n_ph is {expected_n_ph} -- every arm must be offered exactly "
            f"the cell's own ratified n_ph, asserted against the constant, not merely "
            f"against each other"
        )


# =============================================================================================
# The two implementation decisions the ratified rule does not pin down -- see module
# docstring items 1/2.
# =============================================================================================


def _connectivity_summary(readout: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-serializable subset of a geodesic_graph read-out (drops the numpy
    label/mask arrays neither `decide` nor the record needs)."""
    return {
        "n_components": int(readout["n_components"]),
        "largest_size": int(readout["largest_size"]),
        "dropped_fraction": float(readout["dropped_fraction"]),
    }


def _select_geodesic_k(
    points: np.ndarray, k_values: Sequence[int], min_run: int
) -> Tuple[int, Dict[str, Any], Dict[str, Any], bool]:
    """D-06's sweep, reduced to one winning k for the actual D_geo build -- see module
    docstring item 1. Returns (winning_k, connectivity_summary_at_winning_k,
    stable_range_over_n_components, stable_enough)."""
    sweep = geodesic_graph.k_sweep_components(points, k_values)
    n_components_seq = [entry["n_components"] for entry in sweep]
    stable = geodesic_graph.contiguous_stable_range(n_components_seq)
    stable_enough = stable["length"] >= min_run
    idx = stable["start_index"] if stable_enough else len(k_values) - 1
    winning_k = int(k_values[idx])
    return winning_k, _connectivity_summary(sweep[idx]), stable, stable_enough


def _select_dimension_k(consensus: Dict[str, Any], k_values: Sequence[int]) -> int:
    """D-10's plateau consensus, reduced to one representative k for `spread`/local
    dispersion -- see module docstring item 1."""
    if consensus["d_hat"] is None or not consensus["run"]:
        return int(k_values[-1])
    idx = max(run["start_index"] for run in consensus["run"].values())
    idx = min(idx, len(k_values) - 1)
    return int(k_values[idx])


def _classify_outcome(template: str, decision: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """D-13's per-draw outcome against ground truth -- see module docstring item 2 for the
    `ball` family-vs-exact-d_hat rule. Returns (outcome, detail) where `detail` is the
    abstain condition letter (abstained) or the decided template name (correct/wrong)."""
    if decision["abstained"]:
        return "abstained", decision["abstain_condition"]
    actual = decision["template"]
    if template == "ball":
        correct = actual is not None and actual.startswith("ball_d")
    else:
        correct = actual == template
    return ("correct" if correct else "wrong"), actual


# =============================================================================================
# One metric arm: persistence diagram at maxdim=2 (D-11's H_2 ceiling), one bootstrap band
# per degree (D-02), the significant Betti vector.
# =============================================================================================


def _run_metric_arm(name: str, D_mat: np.ndarray, seed: int, b_boot: int) -> Dict[str, Any]:
    t0 = time.monotonic()
    dgms = persistence_probe.persistence_diagram(D_mat, maxdim=2)
    dgm_wallclock_s = time.monotonic() - t0

    bands: List[float] = []
    band_details: List[Dict[str, Any]] = []
    for degree in range(3):
        t1 = time.monotonic()
        band_info = confidence_band.bootstrap_band(D_mat, degree, b_boot, ALPHA, seed)
        band_wallclock_s = time.monotonic() - t1
        bands.append(band_info["band"])
        band_details.append(
            {
                "degree": degree,
                "c_alpha": band_info["c_alpha"],
                "band": band_info["band"],
                "wallclock_s": band_wallclock_s,
            }
        )

    betti = confidence_band.betti_vector(dgms, bands)
    n_bars = [int(dgms[d].shape[0]) for d in range(3)]

    return {
        "metric": name,
        "betti": list(betti),
        "bands": band_details,
        "n_bars_h0_h1_h2": n_bars,
        "persistence_wallclock_s": dgm_wallclock_s,
    }


# =============================================================================================
# One draw: immersion -> Jacobian validity -> budget parity -> both metric arms -> dimension
# consensus + local dispersion (amended gate) -> connectivity -> decide().
# =============================================================================================


def _run_one_draw(config: Dict[str, Any], template: str, seed: int, b_boot: int) -> Dict[str, Any]:
    d_kwarg = BALL_D if template == "ball" else None
    cloud = template_immersion.immerse(
        template,
        config["n_ph"],
        config["D"],
        config["noise"],
        config["density"],
        seed,
        WARP_PARAMS,
        d=d_kwarg,
        check_immersion=False,
    )
    points = cloud["points"]

    # --- D-14: immersion validity, checked on every generated cloud, never assumed. -------
    jac = template_immersion.jacobian_rank(cloud, n_check=JACOBIAN_N_CHECK, rank_tol=JACOBIAN_RANK_TOL)
    if not bool(jac["is_immersion"]):
        return {
            "seed": seed,
            "excluded_invalid_immersion": True,
            "jacobian_rank": {
                "min_rank": int(jac["min_rank"]),
                "expected_rank": int(jac["expected_rank"]),
                "worst_point_index": int(jac["worst_point_index"]),
            },
        }

    # --- D-08: budget parity, asserted in code before either arm's matrix is built. -------
    n_offered = {"euclidean": int(points.shape[0]), "geodesic": int(points.shape[0])}
    _assert_budget_parity(n_offered, config["n_ph"])

    D_euclidean, _ = persistence_probe.cloud_distance_matrix(points, prescale=False)

    winning_k, connectivity_readout, _geodesic_k_stable_range, k_stable_enough = _select_geodesic_k(
        points, K_VALUES, MIN_RUN
    )
    D_geodesic, geo_readout = geodesic_graph.geodesic_distance_matrix(points, winning_k)

    arms: Dict[str, Any] = {}
    for name, D_mat in (("euclidean", D_euclidean), ("geodesic", D_geodesic)):
        arms[name] = _run_metric_arm(name, D_mat, seed, b_boot)

    # --- D-09/D-10: all eight estimators, every swept k, no aggregation; plateau consensus. -
    k_max = max(K_VALUES)
    dist_sq_global = geometry.compute_knn_distances(points, k_max)
    estimates_by_k, _k_applicable = local_dimension.global_estimates(points, K_VALUES)
    consensus = local_dimension.plateau_consensus(estimates_by_k, MAJORITY, MIN_RUN, TOLERANCE, COUNT_DISTINCT)
    dim_k = _select_dimension_k(consensus, K_VALUES)
    spread_at_k = local_dimension.spread(estimates_by_k[dim_k])

    # --- D-12 + AMENDMENT-01: local dispersion, gated through the provenance-matched subset.
    n_anchors = min(N_ANCHORS, points.shape[0])
    rng_anchor = np.random.default_rng(seed)
    anchor_indices = rng_anchor.choice(points.shape[0], size=n_anchors, replace=False)
    local_by_anchor = local_dimension.local_estimates(
        points, dim_k, anchor_indices, NEIGHBOURHOOD_SIZE, dist_sq_global[:, :dim_k]
    )
    local_dispersion_full = local_dimension.dispersion(local_by_anchor)
    local_dispersion_gating = local_dimension.gating_dispersion(local_by_anchor)

    dimension_readout = {
        "d_hat": consensus["d_hat"],
        "reason": consensus["reason"],
        "supporting_estimators": consensus["supporting_estimators"],
        "excluded_estimators": consensus["excluded_estimators"],
        "spread": spread_at_k,
        "representative_k": dim_k,
        "local_dispersion": local_dispersion_gating,
    }

    bounds = {"dispersion_bound": DISPERSION_BOUND, "spread_bound": SPREAD_BOUND}
    arms_for_decide = {
        "euclidean": {"betti": arms["euclidean"]["betti"]},
        "geodesic": {"betti": arms["geodesic"]["betti"]},
    }
    decision = template_decision.decide(arms_for_decide, dimension_readout, connectivity_readout, bounds)

    outcome, outcome_detail = _classify_outcome(template, decision)

    return {
        "seed": seed,
        "excluded_invalid_immersion": False,
        "d_true": cloud["d_true"],
        "estimates_by_k": estimates_by_k,
        "dimension_readout": {
            "d_hat": dimension_readout["d_hat"],
            "reason": dimension_readout["reason"],
            "supporting_estimators": dimension_readout["supporting_estimators"],
            "excluded_estimators": dimension_readout["excluded_estimators"],
            "spread": dimension_readout["spread"],
            "representative_k": dim_k,
            "local_dispersion_gating": local_dispersion_gating,
            "local_dispersion_full": local_dispersion_full,
        },
        "geodesic_winning_k": winning_k,
        "geodesic_k_stable_enough": k_stable_enough,
        "connectivity": connectivity_readout,
        "geo_dropped_fraction": float(geo_readout["dropped_fraction"]),
        "arms": arms,
        "decision": decision,
        "outcome": outcome,
        "outcome_detail": outcome_detail,
    }


# =============================================================================================
# One combination (config, template): DRAWS independent draws, per-combination timing.
# =============================================================================================


def _run_combo(config: Dict[str, Any], template: str, draws: int, b_boot: int) -> Dict[str, Any]:
    combo_seed = _combo_seed(config["config_id"], template)
    draw_results: List[Dict[str, Any]] = []
    excluded_count = 0

    t0 = time.monotonic()
    for draw_index in range(draws):
        draw_seed = _draw_seed(combo_seed, draw_index)
        result = _run_one_draw(config, template, draw_seed, b_boot)
        if result.get("excluded_invalid_immersion"):
            excluded_count += 1
        draw_results.append(result)
    combo_wallclock_s = time.monotonic() - t0

    valid = [r for r in draw_results if not r.get("excluded_invalid_immersion")]
    outcomes = [r["outcome"] for r in valid]

    return {
        "config_id": config["config_id"],
        "template": template,
        "arm": config["arm"],
        "D": config["D"],
        "noise": config["noise"],
        "n_ph": config["n_ph"],
        "density": config["density"],
        "draws_requested": draws,
        "excluded_count": excluded_count,
        "outcomes": outcomes,
        "draw_results": draw_results,
        "combo_wallclock_s": combo_wallclock_s,
        "status": "ok",
    }


# =============================================================================================
# Timeout isolation -- the same forked-subprocess ceiling shape as
# ph_budget_calibration_run.py's own run_timed, generalised to run one full combination.
# Used ONLY for `ball` combinations (SCREENING-RULE.md Sec.9's own named, ratified fallback).
# =============================================================================================


def _timeout_worker(fn, args, conn) -> None:
    try:
        result = fn(*args)
        conn.send({"status": "ok", "result": result})
    except Exception as exc:  # pragma: no cover -- defensive; reported, never swallowed
        conn.send({"status": "error", "error": repr(exc)})
    finally:
        conn.close()


def run_with_timeout(fn, args: tuple, timeout_s: float) -> Dict[str, Any]:
    """Run `fn(*args)` in a forked child process with a hard `timeout_s` ceiling. Returns
    `{"status": "ok"|"timeout"|"error"|"crashed", "wallclock_s", "result"?}` -- a call that
    does not finish inside `timeout_s` is recorded as `"timeout"` at that known bound,
    never silently retried and never left to hang."""
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_timeout_worker, args=(fn, args, child_conn))
    proc.start()
    child_conn.close()

    t0 = time.monotonic()
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = {"status": "crashed"}
        proc.join(timeout=5)
    else:
        result = {"status": "timeout"}
        proc.terminate()
        proc.join(timeout=5)

    if proc.is_alive():
        proc.kill()
        proc.join()

    result.setdefault("wallclock_s", time.monotonic() - t0)
    return result


def _s2_reference_cost(config_id: str, completed_by_key: Dict[Tuple[str, str], Dict[str, Any]]) -> Optional[float]:
    """The same configuration's own already-completed S2 combination cost, if on record --
    None if not yet measured (the caller falls back to FALLBACK_S2_PER_COMBO_S)."""
    entry = completed_by_key.get((config_id, "S2"))
    if entry is None or entry.get("status") != "ok":
        return None
    return entry.get("combo_wallclock_s")


# =============================================================================================
# JSON-lines append-only record: resumability.
# =============================================================================================


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_completed(record_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Reads every JSON-lines record already on disk, keyed by (config_id, template) --
    the resumability index. Missing file -> empty dict (a fresh run, not an error)."""
    completed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed[(rec["config_id"], rec["template"])] = rec
    return completed


def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(_to_jsonable(record)) + "\n")


# =============================================================================================
# main
# =============================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: one template (S1), the baseline cell's own D/noise/density at a "
            "stated-reduction n_ph, minimal draws and B -- exercises immersion, the "
            "Jacobian check, budget parity, both metric arms, the dimension/dispersion "
            "layer (amended gate), decide(), and the D-13 tally in well under a minute. "
            "This is the runner's automated <verify>."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid (configuration count, combination count, D levels "
        "present) and exit without running or writing anything.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any (configuration, template) combination already present in the "
        "record file rather than recomputing it.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Override the append-only record path (default: "
        "notebooks/.cache/template_benchmark_run_record.jsonl).",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        choices=TEMPLATES,
        default=None,
        help="Restrict to a subset of templates -- for segmented or targeted runs.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Run at most this many combinations this invocation, then stop -- for "
        "deliberately segmented manual runs (the user drives a multi-hour grid in "
        "pieces).",
    )
    parser.add_argument(
        "--force-ball-timeout-s",
        type=float,
        default=None,
        help="TEST-ONLY debug hook: override the computed ball-timeout ceiling directly "
        "instead of the ratified 2x-S2 rule, so the fallback path can be exercised "
        "without a real S2 measurement. Never used by the real grid run.",
    )
    args = parser.parse_args()

    record_path = Path(args.record_path) if args.record_path else (CACHE_DIR / "template_benchmark_run_record.jsonl")

    print("=" * 92)
    print("Phase 02.7 Plan 10: template-inference benchmark runner (D-15)")
    print(
        "Assembles measurements and calls the ratified decision rule once per cloud -- "
        "chooses no threshold, grid level or default of its own."
    )

    if args.dry_run:
        configs = build_grid_configs()
        templates = args.templates or list(TEMPLATES)
        combos = build_combinations(configs, templates)
        print(f"configurations={len(configs)}  templates={templates}  combinations={len(combos)}")
        for config in configs:
            print(f"  [{config['arm']}] {config['config_id']}: D={config['D']} noise={config['noise']} n_ph={config['n_ph']} density={config['density']}")
        print(f"D_LEVELS present: {D_LEVELS}  (768 required, and present)")
        print("--dry-run: nothing executed, nothing written.")
        return

    if args.smoke:
        smoke_config = {
            "config_id": "smoke_baseline_shaped",
            "arm": "smoke",
            "D": BASELINE_D,
            "noise": BASELINE_NOISE,
            "n_ph": SMOKE_N_PH,
            "density": BASELINE_DENSITY,
        }
        configs = [smoke_config]
        templates = args.templates or list(SMOKE_TEMPLATES)
        draws = SMOKE_DRAWS
        b_boot = SMOKE_B
        print(
            f"--smoke: n_ph={SMOKE_N_PH} (stated reduction from ratified {N_PH}, wall-clock "
            f"tractability only), B={SMOKE_B}, draws={SMOKE_DRAWS}, templates={templates}, "
            f"D={BASELINE_D} noise={BASELINE_NOISE} density={BASELINE_DENSITY}"
        )
    else:
        configs = build_grid_configs()
        templates = args.templates or list(TEMPLATES)
        draws = DRAWS
        b_boot = B
        print(
            f"full grid: n_ph={N_PH} B={B} draws={DRAWS} k_values={K_VALUES} "
            f"D_levels={D_LEVELS} noise_levels={NOISE_LEVELS} templates={templates}"
        )

    combos = build_combinations(configs, templates)
    print(f"configurations={len(configs)}  combinations_this_invocation={len(combos)}")
    print(f"record_path={record_path}")
    print("=" * 92)

    completed_by_key = load_completed(record_path) if args.resume else {}
    if args.resume:
        print(f"--resume: {len(completed_by_key)} combination(s) already on record, will be skipped.")

    total_excluded = 0
    combo_wallclocks: List[float] = []
    n_run_this_invocation = 0

    for combo_index, (config, template) in enumerate(combos):
        key = (config["config_id"], template)
        if args.resume and key in completed_by_key:
            print(f"  [skip, resumed] {key[0]} / {key[1]}")
            continue

        if args.max_combos is not None and n_run_this_invocation >= args.max_combos:
            print(f"--max-combos={args.max_combos} reached; stopping this invocation ({len(combos) - combo_index} combination(s) remain).")
            break

        t_combo_start = time.monotonic()

        if template == "ball":
            s2_cost = _s2_reference_cost(config["config_id"], completed_by_key)
            reference_s = s2_cost if s2_cost is not None else FALLBACK_S2_PER_COMBO_S
            ceiling = (
                args.force_ball_timeout_s
                if args.force_ball_timeout_s is not None
                else BALL_TIMEOUT_MULTIPLIER * reference_s
            )
            timed = run_with_timeout(_run_combo, (config, template, draws, b_boot), ceiling)
            if timed["status"] == "timeout":
                record = {
                    "config_id": config["config_id"],
                    "template": template,
                    "arm": config["arm"],
                    "D": config["D"],
                    "noise": config["noise"],
                    "n_ph": config["n_ph"],
                    "density": config["density"],
                    "status": "timeout",
                    "timeout_ceiling_s": ceiling,
                    "s2_reference_s": reference_s,
                    "s2_reference_measured": s2_cost is not None,
                    "combo_wallclock_s": timed["wallclock_s"],
                    "outcomes": [],
                    "excluded_count": 0,
                }
                print(
                    f"  [TIMEOUT] {key[0]} / ball: exceeded ceiling={ceiling:.1f}s "
                    f"(2x reference={reference_s:.1f}s, measured={s2_cost is not None}) -- "
                    f"reported as incomplete, grid continues"
                )
            elif timed["status"] == "ok":
                record = timed["result"]
            else:
                record = {
                    "config_id": config["config_id"],
                    "template": template,
                    "arm": config["arm"],
                    "D": config["D"],
                    "noise": config["noise"],
                    "n_ph": config["n_ph"],
                    "density": config["density"],
                    "status": timed["status"],
                    "combo_wallclock_s": timed["wallclock_s"],
                    "error": timed.get("error"),
                    "outcomes": [],
                    "excluded_count": 0,
                }
                print(f"  [{timed['status'].upper()}] {key[0]} / ball: {timed.get('error')}")
        else:
            record = _run_combo(config, template, draws, b_boot)

        append_record(record_path, record)
        combo_wallclock_s = time.monotonic() - t_combo_start
        combo_wallclocks.append(combo_wallclock_s)
        n_run_this_invocation += 1
        completed_by_key[key] = record

        excluded_count = record.get("excluded_count", 0)
        total_excluded += excluded_count
        outcomes = record.get("outcomes", [])

        remaining = len(combos) - combo_index - 1
        eta_s = (sum(combo_wallclocks) / len(combo_wallclocks)) * remaining if combo_wallclocks else 0.0
        print(
            f"  [{record.get('status', 'ok')}] {key[0]} / {key[1]}: wallclock_s={combo_wallclock_s:.2f} "
            f"excluded={excluded_count} outcomes={outcomes}  "
            f"(remaining={remaining}, projected_remaining={eta_s / 3600.0:.2f}h)"
        )

    print("=" * 92)
    print(f"This invocation: {n_run_this_invocation} combination(s) executed, {total_excluded} draw(s) excluded (failed Jacobian rank check).")

    # --- D-13's three-way tally, over every valid draw on record so far ----------------------
    # tally() requires an explicit abstain_condition on every "abstained" entry, so this is
    # recovered per draw from draw_results (outcomes alone is insufficient for that).
    tally_entries: List[Dict[str, Any]] = []
    for record in completed_by_key.values():
        for draw in record.get("draw_results", []):
            if draw.get("excluded_invalid_immersion"):
                continue
            entry = {"outcome": draw["outcome"]}
            if draw["outcome"] == "abstained":
                entry["abstain_condition"] = draw["outcome_detail"]
            tally_entries.append(entry)

    if tally_entries:
        tally = template_decision.tally(tally_entries)
        print(
            f"D-13 tally (this run's cumulative record): correct={tally['correct']} "
            f"wrong={tally['wrong']} abstained={tally['abstained']} total={tally['total']}"
        )
        print(
            f"  accuracy={tally['accuracy']:.4f}  abstention_rate={tally['abstention_rate']:.4f}  "
            f"error_rate={tally['error_rate']:.4f}"
        )
        print(f"  by_condition={tally['by_condition']}")
        assert tally["correct"] + tally["wrong"] + tally["abstained"] == tally["total"]
    else:
        print("D-13 tally: no valid (non-excluded) draws on record yet -- nothing to tally.")

    print("=" * 92)
    print("Done. No composite, weighted or averaged figure was computed across metric arms, ")
    print("templates or abstain conditions anywhere above.")


if __name__ == "__main__":
    main()
