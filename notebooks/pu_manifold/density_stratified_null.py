"""Phase 07.1 pre-registration: 07.1's own gating constants block, its guard, and the two
independent verdict rules for D7.1-01 (a density-stratified null for
``partial_rho_density_controlled``) and D7.1-02 (seed stability at ``d=25``).

**This module adds; it does not edit.** ``notebooks/pu_manifold/crossmodal_curvature.py``
(Phase 7, sealed by D7-05) is never imported for a gating VALUE here -- every constant this
module needs is re-declared as a fresh top-level literal, even where the value is identical to
Phase 7's own (``N_PERMUTATIONS``, ``PERMUTATION_SEED``, ``NULL_QUANTILE_PER_TAIL``,
``SPLIT_SEED``, ``HOLDOUT_FRACTION``). D7-05 sealed ``crossmodal_curvature.py`` as
import-never-edit, and a gating constant imported ACROSS that freeze boundary would not be
covered by this module's own ``assert_preregistered()`` or its own git-ancestry proof -- exactly
the gap `.planning/phases/07.1-density-stratified-null-and-seed-stability/07.1-CONTEXT.md`'s
D-08 exists to close. Compute functions that reuse Phase 7's pure utilities
(``_relative_precision_distinct_count``, ``split_indices``) are fine and expected in later
07.1 plans -- it is only the pre-registered VALUES that must never cross the boundary.

**The constants below are FROZEN.** They are committed in this file, in this commit, before any
07.1 number exists anywhere in the tree. A later edit to any of them after a 07.1 number exists
is a pre-registration BREACH: the only remedy is a fresh freeze and a fresh run, never a silent
fix (mirrors D7-06's discipline, applied here to 07.1's own constants).

**This plan produces NO numbers.** Only the frozen constants block, ``assert_preregistered()``,
``verdict_is_terminal()`` and the two verdict functions live here. The stratified-permutation
routine (D7.1-01), the three ``d=25`` fits (D7.1-02), and the positive-control extension all land
in later 07.1 plans, strictly below this freeze line.

**What each decision below governs, by ID** (full text:
``.planning/phases/07.1-density-stratified-null-and-seed-stability/07.1-CONTEXT.md``):

- **D-01/D-02/D-03** -- stratum design: equal-count quantile bins on density RANK
  (``STRATIFICATION_RULE``), ``N_STRATA_HEADLINE = 20`` as the gating count, a non-gating
  ``STRATA_GRID`` of thresholds (``SENSITIVITY_GRID_RULE``).
- **D-04** -- the stratified null's own positive control, ratified at this plan's Task 1
  checkpoint (``ratify-as-proposed``): a finer target grid bracketing both the ``d=20`` and
  ``d=32`` residuals, run in BOTH directions because the residuals this phase adjudicates are
  negative and coarse strata bias the null MEAN positive (liberal on the negative tail).
- **D-05** -- ``h`` and ``m`` permuted independently within strata (``PERMUTATION_SCHEME_RULE``),
  with the mixed-residualization trade-off recorded verbatim in the rule text itself.
- **D-07** -- ``FROZEN_PARTIAL_REFERENCE``, the frozen Phase 7 record values a later plan
  recomputes and asserts against at tight tolerance before trusting any new number.
- **D-08** -- this module's own freeze-before-any-number discipline, applied to its own
  constants rather than inherited from Phase 7's.
- **D-09/D-13** -- the seed axis at ``d=25``: ``TORCH_INIT_SEEDS`` ratified at Task 1
  (``0, 1, 2`` -- seed ``0`` doubles as a reproduction check against Phase 7's frozen field),
  ``SPLIT_SEED``/``PERMUTATION_SEED`` held fixed so any inter-seed difference is attributable to
  the curvature field alone.
- **D-11/D-14/D-15/D-16** -- verdict rules, both ratified at this plan's Task 2 checkpoint
  (``ratify``): unanimity across the three seeds (2-of-3 is the terminal, non-supportive
  ``SPLIT ACROSS SEEDS``), a verdict value distinct from Phase 7's ``SPLIT ACROSS d`` vocabulary
  for subset survival across ``d``, two independent verdict functions each structurally unable to
  accept a quantity they may not gate on.
- **D-12** -- ``SEED_GATING_STATISTIC`` names the load-bearing quantity for the seed verdict; the
  raw statistic is reported alongside and gates nothing (``RAW_STATISTIC_IS_NON_GATING``).

No file I/O happens in this module, following ``crossmodal_curvature.py``'s and
``linear_probe.py``'s stated convention: a default is how a pre-registered value gets inherited
by accident instead of by an explicit call-site choice. This file defines no computable defaults
either -- only flat literals and prose-rule strings.
"""

from typing import Any, Dict, Tuple

# =============================================================================================
# Field and d-sweep (D-08, re-declared fresh -- identical to Phase 7's own D_SWEEP, never
# imported: the per-d clearance mapping apply_partial_verdict consumes is keyed on this tuple).
# =============================================================================================

D_SWEEP = (20, 25, 32)
"""The three d values this phase's stratified null adjudicates -- identical to Phase 7's own
D_SWEEP by construction (same three decoder fits, same frozen field), re-declared fresh rather
than imported per D-08."""

# =============================================================================================
# Significance (D-08, re-declared fresh -- values happen to equal Phase 7's own, never imported).
# =============================================================================================

N_PERMUTATIONS = 1000
PERMUTATION_SEED = 20260825
NULL_QUANTILE_PER_TAIL = 0.975

# =============================================================================================
# Density stratum design (D-01, D-02, D-03).
# =============================================================================================

N_STRATA_HEADLINE = 20
"""The gating stratum count -- 500 points per stratum at n = 10,000. Deliberately the middle of
the defensible range: finer strata make the test MORE sensitive (the permuted partial varies
less, the band narrows), not more conservative."""

STRATA_GRID = (10, 20, 50)
"""Non-gating threshold grid reported alongside the headline S=20 result (D-03)."""

STRATIFICATION_RULE = (
    "Strata are equal-count quantile bins on density RANK, not equal-width bins in log-density "
    "(density spans a 7.8-order range, p05=6.07e4 / p50=2.29e9 / p95=3.63e12, so rank bins give "
    "every stratum identical permutation entropy). Stratum assignment is by np.argsort(density) "
    "POSITION: exactly-tied densities are separated by index order and are never merged into "
    "the same rank; any remainder rows after equal division go to the LAST stratum."
)

SENSITIVITY_GRID_RULE = (
    "STRATA_GRID = (10, 20, 50) is a grid of THRESHOLDS, not of point estimates -- unlike "
    "MKNN_K_GRID, the observed partial_rho_density_controlled is computed with controls=density "
    "and never touches strata, so S moves only the null, never the observed statistic. Its "
    "purpose is to expose the N_STRATA_HEADLINE artifact directly: a residual that clears at "
    "S=50 but not at S=10 cleared on stratification tightness, not on curvature. It may qualify "
    "a reading but never overturn or escalate it -- mirrors crossmodal_curvature.py's own "
    "SENSITIVITY_GRID_RULE precedent for MKNN_K_GRID. null_vals.mean() is reported at every S "
    "alongside the band, because 07.1-RESEARCH.md Pitfall 1 measured the stratum-count artifact "
    "as a null-MEAN shift, not a band-width change."
)

# =============================================================================================
# The null construction (D-05, D-06).
# =============================================================================================

PERMUTATION_SCHEME_RULE = (
    "h and m are each permuted INDEPENDENTLY within strata, mirroring "
    "scipy.stats.permutation_test(permutation_type='pairings')'s global re-pairing but confined "
    "to one stratum at a time. Recorded trade-off (D-05, stated here and not only in prose "
    "surrounding it): because h's residualization also varies per permutation, the resulting "
    "band mixes both residualization sides and is NOT attributable to the MKNN link alone. The "
    "considered alternative -- shuffling m only, holding h's joint distribution with density "
    "exactly fixed -- was evaluated and not chosen."
)

# =============================================================================================
# Frozen Phase 7 reference values this phase recomputes against (D-07).
# =============================================================================================

FROZEN_PARTIAL_REFERENCE = {
    20: -0.024188908104711526,
    25: -0.06583482510693942,
    32: -0.02171865371304575,
}
"""Read directly from notebooks/.cache/07_crossmodal_curvature.jsonl's three row_kind: "sweep"
rows. A later plan recomputes partial_rho_density_controlled from the frozen npz and asserts
np.isclose against these at PARTIAL_REFERENCE_RTOL / PARTIAL_REFERENCE_ATOL before trusting any
new number -- proving the frozen field and the reload path are intact, not a nuisance check."""
PARTIAL_REFERENCE_RTOL = 1e-9
PARTIAL_REFERENCE_ATOL = 1e-12

# =============================================================================================
# Positive control for the stratified null (D-04). Ratified at this plan's Task 1 checkpoint,
# option ratify-as-proposed.
# =============================================================================================

POSITIVE_CONTROL_TARGET_RHOS = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.10, 0.20)
"""Strictly increasing, finer than Phase 7's (0.02, 0.05, 0.10, 0.20) and bracketing both the
d=20 (-0.02419) and d=32 (-0.02172) residuals with grid points on either side -- Phase 7's own
grid could not resolve the detection floor between 0.021 and 0.05, exactly where those two
residuals sit."""
POSITIVE_CONTROL_DIRECTIONS = ("positive", "negative")
"""The stratified control is run in BOTH directions, not positive-only. 07.1-RESEARCH.md
Pitfall 1 measured that coarse strata bias the null MEAN positive, which is liberal on the
NEGATIVE tail -- and every residual this phase adjudicates (d=20, d=25, d=32, all in
FROZEN_PARTIAL_REFERENCE) is negative. The power argument that licenses an "underpowered, no
claim" override (see PARTIAL_VERDICT_RULE) is gated on the NEGATIVE-direction floor, measured in
the same direction as the residuals it is used to interpret."""
POSITIVE_CONTROL_SEED = 20260825
POSITIVE_CONTROL_RULE = (
    "Extends crossmodal_curvature.plant_positive_control's bisection mechanism to target "
    "partial_rho_density_controlled instead of the raw Spearman rho: the planted array is built "
    "on PU's own realized density-controlled residual field, bisected against "
    "cross_split_curvature.partial_spearman(h, planted, controls=density) rather than "
    "scipy.stats.spearmanr(h, planted), because the partial residualizes both arrays against "
    "density first and Phase 7's D7-02 validated the RAW statistic's detection floor only -- an "
    "inherited floor for the partial would be an assumption, not a measurement. For each entry "
    "of POSITIVE_CONTROL_TARGET_RHOS, the search is run once per entry of "
    "POSITIVE_CONTROL_DIRECTIONS ('positive' bisects toward +target_rho, 'negative' toward "
    "-target_rho), each against the SAME stratified null machinery (N_STRATA_HEADLINE, "
    "N_PERMUTATIONS, PERMUTATION_SEED, NULL_QUANTILE_PER_TAIL) the headline stratified test "
    "uses. The reported power quantity -- the smallest target_rho at which either tail clears --"
    " is read off the NEGATIVE-direction search only, because every residual this phase "
    "adjudicates is negative (see POSITIVE_CONTROL_DIRECTIONS). The achieved rho is recorded "
    "beside every target, never silently substituted for it."
)

# =============================================================================================
# Seed axis at d=25 (D-09, D-13). SPLIT_SEED and HOLDOUT_FRACTION are held fixed across seeds --
# only TORCH_INIT_SEEDS varies -- so the split and the training data are identical across the
# three fits and any inter-seed difference is attributable to the curvature field alone.
# =============================================================================================

SPLIT_SEED = 20260813
HOLDOUT_FRACTION = 0.2
TORCH_INIT_SEEDS = (0, 1, 2)
"""Ratified at this plan's Task 1 checkpoint. Seed 0 is Phase 7's own realized TORCH_INIT_SEED,
so one of the three d=25 fits doubles as a reproduction check against the frozen h_norm_25
field. PERMUTATION_SEED is fixed at 20260825 across all three runs (D-13): all three nulls draw
identical permutation index sets, so any difference between the three thresholds comes from the
curvature field alone -- the one thing D7.1-02 measures."""

SEED_COMBINATION_RULE = (
    "Unanimity: all three TORCH_INIT_SEEDS must clear the stratified partial null for "
    "d=25 for the seed verdict to report support. Any 2-of-3 (or 1-of-3, or 0-of-3-with-a-"
    "cleared-positive-control) pattern is SPLIT ACROSS SEEDS -- a COMPLETE, TERMINAL, "
    "non-supportive outcome, never a stall and never resolved by majority vote, escalation, or a "
    "different bucketing. Directly inherits Phase 5's ratified don't-pool decision "
    "(05-03-DECISION.md) and its SPLIT ACROSS SEEDS framing verbatim. Ratified at this plan's "
    "Task 2 checkpoint, option ratify: on Phase 5's own measured evidence a split is a live "
    "possibility, so this is the consequential call of the phase and it is made in the "
    "direction that refuses to overclaim."
)

SEED_GATING_STATISTIC = "partial_rho_density_controlled"
RAW_STATISTIC_IS_NON_GATING = True
"""D-12: each of the three d=25 seeds is gated on D7.1-01's stratified partial -- the raw
statistic is reported alongside every seed's result and gates none of the seed verdict."""

# =============================================================================================
# Diagnostics (structural, not promissory -- D-16).
# =============================================================================================

DIAGNOSTICS_ARE_NON_GATING = True
"""Whatever diagnostic quantities a later 07.1 plan reports alongside either verdict, neither
apply_partial_verdict nor apply_seed_verdict's signature can accept them -- both take exactly a
clearance mapping and a positive-control result, mechanically enforcing this rather than only
documenting it (mirrors crossmodal_curvature.py's own DIAGNOSTICS_ARE_NON_GATING precedent)."""

# =============================================================================================
# Provenance and record (D-08's freeze discipline, applied to this module).
# =============================================================================================

RECORD_STEM = "07.1_density_stratified_null"
RECORD_LOCATION_RULE = (
    "The frozen record is written under cache.CACHE_DIR via cache.cache_path, which routes "
    "every write through cache._assert_inside_cache's containment guard, the same real security "
    "mitigation Phase 7 relied on (T-07-01). It is written to "
    "notebooks/.cache/07.1_density_stratified_null.jsonl, gitignored per CLAUDE.md, and is "
    "distinct from notebooks/.cache/07_crossmodal_curvature.jsonl and "
    "notebooks/.cache/07_crossmodal_curvature_fields.npz, both of which this phase reads but "
    "never writes."
)
PREREGISTRATION_FREEZE_RULE = (
    "No 07.1 number of any kind may be produced at or before the commit that adds this file -- "
    "that commit is 07.1's own FREEZE_COMMIT_SHA and it is a STRICT git ancestor of every later "
    "07.1 number-producing commit. The proof, mirroring D7-06's discipline exactly: equality "
    "against a hardcoded FREEZE_COMMIT_SHA (not merely ancestry -- a wrong-but-genuine ancestor "
    "must be rejected, per crossmodal_curvature_run.py's own CR-01 fix), THEN "
    "`git merge-base --is-ancestor <freeze> HEAD` (exits 0), AND "
    "`git rev-list --count <freeze>..HEAD` (>= 1, because a commit is its own ancestor and "
    "`--is-ancestor` alone would pass even if a number were produced in the freeze commit "
    "itself). Phase 7's own gating constants (N_PERMUTATIONS, PERMUTATION_SEED, "
    "NULL_QUANTILE_PER_TAIL, SPLIT_SEED, HOLDOUT_FRACTION) are RE-DECLARED above rather than "
    "imported from crossmodal_curvature, because a gating constant crossing the D7-05 freeze "
    "boundary is not covered by this module's own guard."
)

# =============================================================================================
# Verdict rules (D-11, D-14, D-15, D-16). Ratified at this plan's Task 2 checkpoint, option
# ratify. Two INDEPENDENT verdicts -- D7.1-01's per-d stratified-null verdict and D7.1-02's
# per-seed verdict at d=25 -- because D7.1-02 is gated on D7.1-01 (D-12) and a null result there
# must not silently void the seed work.
# =============================================================================================

PARTIAL_VERDICT_RULE = """D-14/D-15 PARTIAL_VERDICT_RULE -- frozen in committed source before any
07.1 stratified-null number existed. Ratified at this plan's Task 2 blocking checkpoint, ratify.

apply_partial_verdict consumes a per-d clearance mapping (one boolean per d in D_SWEEP -- did
the stratified null, at N_STRATA_HEADLINE, clear for partial_rho_density_controlled at that d)
and the stratified positive control's smallest cleared target (or None if it recovered nothing).
The per-d clearance table is reported UNCONDITIONALLY regardless of which verdict fires, because
the phase goal names the individual d=20/d=25/d=32 residuals to adjudicate, not only the
aggregate verdict.

The three-outcome mapping onto PARTIAL_VERDICT_VALUES:
  (a) every d clears -> RESIDUAL SURVIVES AT ALL d;
  (b) no d clears, AND the positive control cleared at some target
      -> NO SURVIVING RESIDUAL;
  (c) no d clears, AND the positive control cleared NOTHING
      -> UNDERPOWERED -- NO CLAIM (D-04's power requirement made mechanical -- without this
      override, a null could not be distinguished from an underpowered test);
  (d) some d clear and some do not -> SURVIVES AT SUBSET OF d.

SURVIVES AT SUBSET OF d is a value DISTINCT from Phase 7's SPLIT ACROSS d vocabulary (D-15):
the phase's most likely outcome -- d=25 clears, d=20 and d=32 do not -- is a genuine finding
(most of Phase 7's association was density everywhere except d=25), and reusing Phase 7's
SPLIT ACROSS d label would describe it identically to a failure to detect anything.

apply_partial_verdict raises ValueError unless its mapping's keys are exactly set(D_SWEEP) --
a verdict from a partial sweep is not a verdict."""

PARTIAL_VERDICT_VALUES = (
    "RESIDUAL SURVIVES AT ALL d",
    "SURVIVES AT SUBSET OF d",
    "NO SURVIVING RESIDUAL",
    "UNDERPOWERED -- NO CLAIM",
)
"""The four terminal outcomes for D7.1-01. SURVIVES AT SUBSET OF d is a value Phase 7's
VERDICT_VALUES does not have (D-15) -- it names subset survival honestly instead of collapsing
it into Phase 7's SPLIT ACROSS d."""

SEED_VERDICT_RULE = """D-11/D-14 SEED_VERDICT_RULE -- frozen in committed source before any
07.1 seed-stability number existed. Ratified at this plan's Task 2 blocking checkpoint, ratify.

apply_seed_verdict consumes a per-seed clearance mapping (one boolean per seed in
TORCH_INIT_SEEDS -- did that seed's d=25 fit clear the stratified null for
partial_rho_density_controlled) and the same stratified positive control's smallest cleared
target used by the partial verdict (or None).

Combination rule is UNANIMITY (SEED_COMBINATION_RULE, D-11): all three seeds must clear for the
seed verdict to report support. The three-outcome mapping onto SEED_VERDICT_VALUES:
  (a) all three seeds clear -> SEED STABLE AT d=25;
  (b) no seed clears, AND the positive control cleared at some target
      -> NO SURVIVING RESIDUAL AT ANY SEED;
  (c) no seed clears, AND the positive control cleared NOTHING
      -> UNDERPOWERED -- NO CLAIM;
  (d) any mixed pattern (2-of-3, 1-of-3) -> SPLIT ACROSS SEEDS, a COMPLETE, TERMINAL,
      non-supportive outcome, mirroring Phase 5's SPLIT ACROSS SEEDS precedent
      (05-03-DECISION.md) -- never upgradable by majority vote, and never re-litigated after
      seeing the three numbers.

apply_seed_verdict raises ValueError unless its mapping's keys are exactly set(TORCH_INIT_SEEDS)
-- a verdict from a partial seed set is not a verdict. It is order-invariant: the same mapping
built in any key order produces the same verdict string, because the check and the branch logic
both operate on the mapping's values, never its iteration order."""

SEED_VERDICT_VALUES = (
    "SEED STABLE AT d=25",
    "SPLIT ACROSS SEEDS",
    "NO SURVIVING RESIDUAL AT ANY SEED",
    "UNDERPOWERED -- NO CLAIM",
)
"""The four terminal outcomes for D7.1-02. SPLIT ACROSS SEEDS is a complete, non-supportive
terminal outcome (D-11), never a stall."""


_REQUIRED_CONSTANTS = (
    "D_SWEEP",
    "N_PERMUTATIONS", "PERMUTATION_SEED", "NULL_QUANTILE_PER_TAIL",
    "N_STRATA_HEADLINE", "STRATA_GRID", "STRATIFICATION_RULE", "SENSITIVITY_GRID_RULE",
    "PERMUTATION_SCHEME_RULE",
    "FROZEN_PARTIAL_REFERENCE", "PARTIAL_REFERENCE_RTOL", "PARTIAL_REFERENCE_ATOL",
    "POSITIVE_CONTROL_TARGET_RHOS", "POSITIVE_CONTROL_DIRECTIONS", "POSITIVE_CONTROL_SEED",
    "POSITIVE_CONTROL_RULE",
    "SPLIT_SEED", "HOLDOUT_FRACTION", "TORCH_INIT_SEEDS", "SEED_COMBINATION_RULE",
    "SEED_GATING_STATISTIC", "RAW_STATISTIC_IS_NON_GATING",
    "DIAGNOSTICS_ARE_NON_GATING",
    "RECORD_STEM", "RECORD_LOCATION_RULE", "PREREGISTRATION_FREEZE_RULE",
    "PARTIAL_VERDICT_RULE", "PARTIAL_VERDICT_VALUES",
    "SEED_VERDICT_RULE", "SEED_VERDICT_VALUES",
)


def assert_preregistered() -> None:
    """Refuse to proceed while any pre-registered constant is unset, malformed, or absent.

    Mirrors ``crossmodal_curvature.assert_preregistered``'s contract shape exactly: loop
    ``_REQUIRED_CONSTANTS`` against ``globals()``, collecting ``"{name} (absent)"`` /
    ``"{name} (None)"`` / ``"{name} (empty string)"`` / ``"{name} (empty sequence)"`` for every
    offender, and raise one ``RuntimeError`` naming all of them. The number-producing path in
    every later 07.1 plan calls this first, so a 07.1 number cannot be computed by a build of
    this module that predates the freeze.
    """
    g = globals()
    missing = []
    for name in _REQUIRED_CONSTANTS:
        if name not in g:
            missing.append(f"{name} (absent)")
            continue
        value = g[name]
        if value is None:
            missing.append(f"{name} (None)")
        elif isinstance(value, str) and not value.strip():
            missing.append(f"{name} (empty string)")
        elif isinstance(value, (tuple, list)) and len(value) == 0:
            missing.append(f"{name} (empty sequence)")

    if missing:
        raise RuntimeError(
            "density_stratified_null.assert_preregistered: 07.1 is not frozen -- the "
            "following pre-registered constants are unset: " + ", ".join(missing) + ". No "
            "07.1 number may be computed before the freeze."
        )


def verdict_is_terminal(verdict: str, values: Tuple[str, ...]) -> bool:
    """``verdict`` is one of ``values`` -- the vocabulary tuple is taken explicitly (D-14) so
    this single function serves both PARTIAL_VERDICT_VALUES and SEED_VERDICT_VALUES rather than
    being duplicated per vocabulary."""
    return verdict in values


def apply_partial_verdict(per_d_results: Dict[int, bool], positive_control_cleared_at: Any) -> str:
    """Mechanically applies PARTIAL_VERDICT_RULE (D7.1-01's verdict).

    ``per_d_results`` maps each ``d`` this phase's D_SWEEP covers to that ``d``'s stratified-null
    clearance boolean; ``positive_control_cleared_at`` is the smallest target rho the stratified
    positive control cleared, or ``None`` if it recovered nothing.

    Raises ``ValueError`` if ``per_d_results``' keys are not exactly ``set(D_SWEEP)`` -- a
    verdict computed from a partial sweep is not a verdict.
    """
    if set(per_d_results.keys()) != set(D_SWEEP):
        raise ValueError(
            f"apply_partial_verdict: per_d_results keys {sorted(per_d_results.keys())} do not "
            f"exactly match D_SWEEP {D_SWEEP}."
        )

    clears = list(per_d_results.values())
    if all(clears):
        verdict = "RESIDUAL SURVIVES AT ALL d"
    elif not any(clears):
        verdict = (
            "UNDERPOWERED -- NO CLAIM"
            if positive_control_cleared_at is None
            else "NO SURVIVING RESIDUAL"
        )
    else:
        verdict = "SURVIVES AT SUBSET OF d"

    assert verdict_is_terminal(verdict, PARTIAL_VERDICT_VALUES)
    return verdict


def apply_seed_verdict(per_seed_results: Dict[int, bool], positive_control_cleared_at: Any) -> str:
    """Mechanically applies SEED_VERDICT_RULE (D7.1-02's verdict).

    ``per_seed_results`` maps each seed in ``TORCH_INIT_SEEDS`` to that seed's d=25 stratified-
    null clearance boolean; ``positive_control_cleared_at`` is the same quantity
    ``apply_partial_verdict`` consumes.

    Combination rule is UNANIMITY (D-11): any pattern short of all-three-clear that is not
    all-three-fail is ``SPLIT ACROSS SEEDS``, a complete terminal non-supportive outcome.

    Raises ``ValueError`` if ``per_seed_results``' keys are not exactly
    ``set(TORCH_INIT_SEEDS)`` -- a verdict computed from a partial seed set is not a verdict.
    """
    seed_set = frozenset(TORCH_INIT_SEEDS)
    if set(per_seed_results.keys()) != seed_set:
        raise ValueError(
            f"apply_seed_verdict: per_seed_results keys {sorted(per_seed_results.keys())} do "
            f"not exactly match TORCH_INIT_SEEDS {TORCH_INIT_SEEDS}."
        )

    clears = list(per_seed_results.values())
    if all(clears):
        verdict = "SEED STABLE AT d=25"
    elif not any(clears):
        verdict = (
            "UNDERPOWERED -- NO CLAIM"
            if positive_control_cleared_at is None
            else "NO SURVIVING RESIDUAL AT ANY SEED"
        )
    else:
        verdict = "SPLIT ACROSS SEEDS"

    assert verdict_is_terminal(verdict, SEED_VERDICT_VALUES)
    return verdict


# =============================================================================================
# Compute functions (plan 07.1-03, Task 1). Everything above this line -- the frozen
# pre-registration from 07.1-01 -- is untouched by this addition. Pure functions only: no file
# I/O, and no defaults on any pre-registered parameter, matching crossmodal_curvature.py's own
# convention at its own line 512.
# =============================================================================================

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from . import cross_split_curvature  # noqa: E402


def density_strata(density: Any, n_strata: int) -> np.ndarray:
    """D-01/D-02: equal-count quantile bins on density RANK, per ``STRATIFICATION_RULE``.

    Assignment is by ``np.argsort(density, kind="stable")`` POSITION -- a stable sort so
    exactly-tied density values are separated by their original index order into adjacent
    strata rather than merged into one (``STRATIFICATION_RULE``'s explicit requirement; a
    non-stable sort would make tied points' stratum assignment depend on the sort algorithm's
    internal tie-breaking, which is an implementation detail, not a specification). Bin size is
    ``n // n_strata``; the ``n % n_strata`` remainder rows go to the LAST stratum, so every
    stratum holds exactly ``n // n_strata`` points except the last, which holds
    ``n // n_strata + n % n_strata``.

    Raises ``ValueError`` if ``n_strata < 1``, or if ``n // n_strata`` is below 3 -- the same
    floor ``cross_split_curvature.partial_spearman`` enforces on any single rank correlation,
    named explicitly in the message since a stratum below it could not support the
    within-stratum permutation this stratification exists to feed.
    """
    density_arr = np.asarray(density, dtype=np.float64).ravel()
    n = density_arr.shape[0]

    if n_strata < 1:
        raise ValueError(f"density_strata: n_strata={n_strata} must be at least 1.")

    bin_size = n // n_strata
    if bin_size < 3:
        raise ValueError(
            f"density_strata: n // n_strata = {n} // {n_strata} = {bin_size} is below the "
            "3-point floor cross_split_curvature.partial_spearman enforces -- a stratum this "
            "small cannot support a rank correlation, let alone a permutation over one."
        )

    order = np.argsort(density_arr, kind="stable")
    strata = np.empty(n, dtype=int)
    for i in range(n_strata):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < n_strata - 1 else n
        strata[order[lo:hi]] = i
    return strata


def stratified_partial_null(
    h: Any,
    m: Any,
    density: Any,
    n_strata: int,
    n_resamples: int,
    seed: int,
    quantile_per_tail: float,
) -> Dict[str, Any]:
    """D-06's restricted (within-stratum) permutation null for
    ``partial_rho_density_controlled``, calibrated against the exact statistic
    :func:`cross_split_curvature.partial_spearman` computes.

    ``h`` and ``m`` are each permuted INDEPENDENTLY within the strata
    :func:`density_strata` assigns from ``density``, mirroring
    ``scipy.stats.permutation_test(permutation_type="pairings")``'s global re-pairing but
    confined to one stratum at a time (``PERMUTATION_SCHEME_RULE``). Recorded trade-off,
    verbatim from that frozen rule: because ``h``'s residualization also varies per
    permutation, the resulting band mixes both residualization sides and is NOT attributable
    to the MKNN link alone. The considered alternative -- shuffling ``m`` only, holding ``h``'s
    joint distribution with density exactly fixed -- was evaluated and not chosen.

    Guard clauses run first, mirroring ``curvature_probe.permutation_null``'s clause order and
    message style, every message beginning ``"stratified_partial_null: "``: non-finite on
    ``h``, non-finite on ``m``, non-finite on ``density``, a length mismatch across the three,
    then a zero-peak-to-peak constant check on ``h`` and on ``m``.

    Then, ONCE outside the resample loop: ``rankdata(h)``, ``rankdata(m)``,
    ``rankdata(density)``, the design matrix ``column_stack([ones(n), rank(density)])`` (the
    control, FIXED across every resample -- only ``h`` and ``m`` move), and the list of
    per-stratum index arrays from :func:`density_strata`. Each resample permutes the
    PRECOMPUTED RANK values within each stratum independently for ``h`` and for ``m``, then
    computes the residual-Pearson statistic directly from the permuted rank vectors --
    ``rankdata(x)[perm] == rankdata(x[perm])`` is what licenses skipping the rank transform
    inside the loop (pinned by ``test_rank_permutation_equivariance``). All permutations are
    drawn from a single ``np.random.default_rng(seed)``, so the same ``seed`` reproduces the
    same null exactly (D-13).

    The observed statistic is computed directly via
    ``cross_split_curvature.partial_spearman(h, m, controls=density)`` -- reused, not
    reimplemented -- rather than re-derived from the precomputed ranks, so this function's own
    ``observed`` key can never silently diverge from the statistic every other 07.1 call site
    reports.

    Reading both tails off ONE null distribution is equivalent to Phase 7's mirror-call
    two-tailed construction (``crossmodal_curvature.two_tailed_permutation_null``) because
    ``partial_spearman(-h, m, controls=c) == -partial_spearman(h, m, controls=c)`` exactly
    (pinned by ``test_partial_spearman_is_exactly_odd_under_negation``): a rank reversal is
    affine and the design carries an intercept, so negating ``h`` before residualizing is the
    same operation as residualizing then negating, and the same holds for every permuted
    resample of ``h`` inside this null. One restricted-permutation null of the UNNEGATED pair
    therefore already carries both tails of the mirror-call construction, and reading
    ``null_low``/``null_high`` off it is the single-null equivalent of running the mirror call
    twice.

    Clearance is STRICT on both tails, mirroring ``curvature_probe.permutation_null``'s own
    ``observed_rho > null_threshold``: ``clears_positive`` is ``observed > null_high``,
    ``clears_negative`` is ``observed < null_low``. An observed value exactly equal to either
    edge does NOT clear. ``null_high`` is the ``quantile_per_tail`` quantile of the null values
    and ``null_low`` is the ``1 - quantile_per_tail`` quantile.

    Returns a flat dict of plain Python scalars: ``observed``, ``null_mean``, ``null_std``,
    ``null_low``, ``null_high``, ``n_strata``, ``stratum_size_min``, ``stratum_size_max``,
    ``n_resamples``, ``seed``, ``quantile_per_tail``, ``clears_positive``, ``clears_negative``,
    ``clears_either``, ``direction`` (``"positive"``, ``"negative"`` or ``"neither"``, with
    ``"positive"`` winning a simultaneous tie deterministically, exactly as
    ``two_tailed_permutation_null`` resolves it).
    """
    h_arr = np.asarray(h, dtype=np.float64).ravel()
    m_arr = np.asarray(m, dtype=np.float64).ravel()
    density_arr = np.asarray(density, dtype=np.float64).ravel()

    if not np.all(np.isfinite(h_arr)):
        raise ValueError("stratified_partial_null: h contains a non-finite value.")
    if not np.all(np.isfinite(m_arr)):
        raise ValueError("stratified_partial_null: m contains a non-finite value.")
    if not np.all(np.isfinite(density_arr)):
        raise ValueError("stratified_partial_null: density contains a non-finite value.")
    if not (h_arr.shape[0] == m_arr.shape[0] == density_arr.shape[0]):
        raise ValueError(
            f"stratified_partial_null: h (len={h_arr.shape[0]}), m (len={m_arr.shape[0]}) and "
            f"density (len={density_arr.shape[0]}) must all have the same length."
        )
    if np.ptp(h_arr) == 0:
        raise ValueError("stratified_partial_null: h is constant (np.ptp(h) == 0).")
    if np.ptp(m_arr) == 0:
        raise ValueError("stratified_partial_null: m is constant (np.ptp(m) == 0).")

    n = h_arr.shape[0]
    strata = density_strata(density_arr, n_strata)
    strat_indices = [np.where(strata == s)[0] for s in range(n_strata)]
    stratum_sizes = [int(idx.shape[0]) for idx in strat_indices]

    rank_h = rankdata(h_arr)
    rank_m = rankdata(m_arr)
    rank_density = rankdata(density_arr)
    design = np.column_stack([np.ones(n), rank_density])  # controls FIXED across all resamples

    def _partial_from_ranks(rh_p: np.ndarray, rm_p: np.ndarray) -> float:
        def _resid(v: np.ndarray) -> np.ndarray:
            coef, *_ = np.linalg.lstsq(design, v, rcond=None)
            return v - design @ coef

        ex, ey = _resid(rh_p), _resid(rm_p)
        return float(np.corrcoef(ex, ey)[0, 1])

    rng = np.random.default_rng(seed)
    null_vals = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        rh_p, rm_p = rank_h.copy(), rank_m.copy()
        for idx in strat_indices:
            rh_p[idx] = rank_h[rng.permutation(idx)]
            rm_p[idx] = rank_m[rng.permutation(idx)]
        null_vals[b] = _partial_from_ranks(rh_p, rm_p)

    observed = float(cross_split_curvature.partial_spearman(h_arr, m_arr, controls=density_arr))
    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    null_high = float(np.quantile(null_vals, quantile_per_tail))
    null_low = float(np.quantile(null_vals, 1.0 - quantile_per_tail))

    clears_positive = bool(observed > null_high)
    clears_negative = bool(observed < null_low)
    clears_either = clears_positive or clears_negative
    if clears_positive:
        direction = "positive"
    elif clears_negative:
        direction = "negative"
    else:
        direction = "neither"

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_low": null_low,
        "null_high": null_high,
        "n_strata": int(n_strata),
        "stratum_size_min": int(min(stratum_sizes)),
        "stratum_size_max": int(max(stratum_sizes)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "quantile_per_tail": float(quantile_per_tail),
        "clears_positive": clears_positive,
        "clears_negative": clears_negative,
        "clears_either": clears_either,
        "direction": direction,
    }
