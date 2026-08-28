"""Phase 8 CKA estimator: the Song et al. (2012) unbiased-HSIC estimator, linear and RBF Gram
builders, and the pre-registration freeze machinery for Phase 8's curvature-conditioned CKA
alignment work.

**This module adds; it does not edit.** ``notebooks/pu_manifold/density_stratified_null.py``
(Phase 07.1) and ``notebooks/pu_manifold/crossmodal_curvature.py`` (Phase 7, sealed by D7-05)
are never imported for a gating VALUE here -- every constant this module needs is re-declared as
a fresh top-level literal, even where a later value might coincide with either sealed module's
own. D7-05/D8-23 sealed those modules as import-never-edit, and a gating constant imported
ACROSS that freeze boundary would not be covered by this module's own
``assert_preregistered()`` or by this phase's own git-ancestry proof.

**THIS COMMIT IS THE FREEZE COMMIT (D8-22).** Every one of the 45 constants named in
``_REQUIRED_CONSTANTS`` is filled below, together, in this single commit, ratified by the
developer directly in
``.planning/phases/08-curvature-conditioned-cka-alignment/08-04-DECISION.md`` (not by a standing
authorization). This commit must be a strict git ancestor of every commit that computes a Phase 8
number -- ``08_cka_alignment_run.py``'s ``_strict_ancestor_or_exit`` and
``tests/test_cka.py``'s ancestry test both pin this commit's SHA once it exists. **A later edit
to any of these 45 constants after a Phase 8 number exists anywhere in the tree is a
pre-registration BREACH: the only remedy is a fresh freeze and a fresh run, never a silent fix**
(mirrors D7-06's discipline, carried into this phase's own constants exactly as
``density_stratified_null.py`` carried it into 07.1's, and the discipline `02.2`'s sealed FAIL
and `06-PREREGISTRATION-AMENDMENT-01` both establish the cost of skipping).

**This plan (08-04) produces NO Phase 8 number itself.** ``--mode selfcheck`` and ``--mode
sigma`` in the accompanying runner never call :func:`assert_preregistered`, for the same reasons
recorded at 08-01/08-03: the former is a pure in-memory known-answer check, the latter measures
pre-registration INPUTS, not Phase 8 results. This commit only fills the constants that make
every later production run (08-05 onward) computable and provable.

**Supersession, not an edit.** ``crossmodal_curvature.py`` line 109 freezes
``ALIGNMENT_METRIC = "mknn"`` under D7-07 ("CKA is out of scope and not implemented anywhere in
this codebase"). Phase 8 supersedes that scope decision BY PHASE DECISION, taken by the
developer on 2026-08-27 and recorded in ``08-CONTEXT.md`` -- never by patching the sealed
module. ``SUPERSEDES`` (filled at the freeze) names ``crossmodal_curvature.ALIGNMENT_METRIC``
as a positive, checkable fact; ``SUPERSESSION_RULE`` (also filled at the freeze) states the
supersession in prose. Phase 7's own ``ALIGNMENT_METRIC = "mknn"`` remains true of Phase 7's own
record rows and is not reinterpreted.

**The Swiss roll standing rule (CLAUDE.md) does not apply here, by decision (D8-17).** CKA has
no decoder and no representation map -- it is a statistic computed over two representations that
already exist. The rule's purpose (telling a broken implementation apart from a real FAIL on
data with no known answer) is served instead by D8-16's invariance ladder, whose answers are
known in closed form. ``SWISS_ROLL_APPLICABILITY_RULE`` (filled at the freeze) carries this
declaration as a checkable fact, not only as this docstring's prose.

No file I/O happens anywhere in this module, following ``crossmodal_curvature.py``'s and
``density_stratified_null.py``'s stated convention: a default is how a pre-registered value gets
inherited by accident instead of by an explicit call-site choice. Every pure function below
takes its parameters as explicit arguments -- ``sigma`` in particular has no default anywhere in
this module, so no call site that only sees a subset of the full point cloud can silently supply
a per-subset bandwidth (D8-03's named confound).
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

# =============================================================================================
# Frozen constants block -- ALL 45 FILLED IN THIS COMMIT (D8-22, the freeze commit). Ratified by
# the developer directly in 08-04-DECISION.md. Never filled piecemeal, never edited after this
# commit once a Phase 8 number exists -- see module docstring.
# =============================================================================================

KERNELS = ("linear", "rbf")
"""D8-01: linear CKA carries the headline verdict; RBF CKA is reported as robustness and gates
nothing (see ``RBF_IS_NON_GATING``)."""

SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)
"""D8-04: the RBF bandwidth sensitivity ladder. ``sigma`` (the ``1.0`` rung) carries the
headline; ``0.5x`` and ``2x`` are reported beside it as diagnostics that gate nothing (see
``SIGMA_LADDER_IS_NON_GATING``)."""

SIGMA_HSC = 0.6420152563705613
"""D8-03: the frozen RBF bandwidth for the HSC modality -- the median pairwise Euclidean
distance over all 10,000 HSC points, measured by ``--mode sigma`` (08-03) before any subset
existed, quoted at full precision from ``08-03-SUMMARY.md``."""

SIGMA_LEGACYSURVEY = 0.5696337821442163
"""D8-03: the frozen RBF bandwidth for the Legacy Survey modality, computed the same way,
independently, over all 10,000 Legacy Survey points, quoted at full precision from
``08-03-SUMMARY.md``."""

GRAM_DTYPE = "float32"
"""The storage dtype for the precomputed Gram matrices (discretion decision, RESEARCH.md A3's
memory argument; ``08-01``'s ``test_gram_dtype_agreement`` measured float32/float64 CKA agreement
at 1.71e-11 absolute at n=3000, well inside the 1e-5 acceptance bound)."""

HSIC_ESTIMATOR_RULE = (
    "The Song et al. (2012) unbiased HSIC form is computed on raw, zero-diagonal Gram matrices "
    "and must never be double-centered: the 1/(n(n-3)) correction terms already perform the "
    "debiasing, and applying them to a pre-centered matrix silently reproduces the biased "
    "estimator under the unbiased one's name (D8-02)."
)
"""D8-02: names the Song et al. (2012) unbiased-HSIC form and the double-centering trap it must
never fall into. Behaviorally pinned by
``tests/test_cka.py::test_double_centering_changes_the_answer``."""

SIGMA_FREEZE_RULE = (
    "sigma is the median pairwise Euclidean distance over all 10,000 points, computed once per "
    "modality, before any subset exists, and reused unchanged for every subset, every d, every "
    "seed and every S (D8-03). A per-subset median is rejected: the high-||H|| subset is "
    "measurably denser at d=20/25 (spearman(density, ||H||) = +0.4281 and +0.3150), so a "
    "per-subset bandwidth would shrink for density reasons and make the RBF gap a density "
    "artifact by construction -- the exact confound this phase is built to exclude. sigma is a "
    "pre-registration constant under D7-06's freeze-before-any-number discipline: changing it "
    "after any Phase 8 number exists requires a new pre-registration and a full re-run, as "
    "02.2's sealed FAIL and 06-PREREGISTRATION-AMENDMENT-01 both establish."
)
"""D8-03: sigma is computed once, globally, per modality, before any subset exists, and reused
unchanged for every subset/d/seed/S, including the measured density-curvature correlations that
make a per-subset median a confound by construction."""

ALIGNMENT_METRIC = "cka"
"""Phase 8's own alignment-metric name -- a positive, checkable fact distinct from and not
overwriting ``crossmodal_curvature.ALIGNMENT_METRIC`` (see ``SUPERSEDES``/``SUPERSESSION_RULE``
below)."""

SUPERSEDES = ("crossmodal_curvature.ALIGNMENT_METRIC",)
"""Names the sealed constant this phase supersedes by decision -- as a positive, checkable fact
(see module docstring's "Supersession, not an edit" section). ``crossmodal_curvature.py`` is
never edited."""

SUPERSESSION_RULE = (
    "Phase 8 supersedes D7-07's CKA-out-of-scope scope decision by PHASE DECISION, taken by the "
    "developer on 2026-08-27 and recorded in 08-CONTEXT.md -- never by editing "
    "crossmodal_curvature.py. crossmodal_curvature.ALIGNMENT_METRIC = 'mknn' remains true of "
    "Phase 7's own record rows and is not reinterpreted; SUPERSEDES names it as a positive, "
    "checkable fact of this supersession, not an edit to the sealed module."
)
"""States that Phase 8 supersedes D7-07's CKA-out-of-scope decision by phase decision, never by
editing the sealed module."""

SWISS_ROLL_APPLICABILITY_RULE = (
    "The CLAUDE.md Swiss roll standing rule is declared NOT APPLICABLE to Phase 8, on purpose "
    "(D8-17). CKA is not a manifold or representation-learning model -- it has no decoder and no "
    "representation map; it is a statistic computed over two representations that already exist. "
    "The rule's purpose (telling a broken implementation apart from a real FAIL on data with no "
    "known answer) is served instead by D8-16's invariance ladder, whose answers are known in "
    "closed form. A Swiss roll option was presented and not chosen -- this is a deliberate "
    "declaration recorded so the gate is satisfied by decision rather than by omission."
)
"""Records D8-17's declaration that the CLAUDE.md Swiss roll standing rule is NOT APPLICABLE to
Phase 8, on purpose (see module docstring)."""

RBF_IS_NON_GATING = True
"""D8-01: RBF CKA is reported as robustness and gates nothing; linear CKA alone carries the
headline verdict."""

SIGMA_LADDER_IS_NON_GATING = True
"""D8-04: the 0.5x/2x sigma sensitivity rungs are diagnostics only and gate nothing; only the
``sigma`` rung itself feeds the headline."""

DIAGNOSTICS_ARE_NON_GATING = True
"""The D7-03 non-gating-diagnostic pattern, carried into this phase for every diagnostic
quantity it reports beside a verdict."""

# --- 08-02 additions: the within-density-stratum tertile split (D8-05/06/07/08) --------------

S_GRID = (10, 20, 50)
"""D8-08: the threshold grid of stratum counts ``S`` this phase's tertile split and null are
computed at -- a grid of THRESHOLDS, not a headline value. See ``SENSITIVITY_GRID_RULE`` below
for what a reader may and may not do with it."""

N_TERTILES = 3
"""D8-05: the number of ``||H||``-magnitude buckets the within-stratum split produces. Not a
discretion value: Phase 8's whole design is built on three tertiles."""

DENSITY_K = 30
"""D8-07: the ``k`` used by ``curvature_probe.local_density_weights`` to build the per-point
density field this phase stratifies on -- re-declared fresh, inherited unchanged from
``crossmodal_curvature.py``'s own ``DENSITY_K = 30``, never imported across the freeze
boundary."""

DENSITY_FIELD_D = 20
"""D8-07: the ambient dimension the density field is computed at -- re-declared fresh, inherited
unchanged from ``crossmodal_curvature.py``'s own ``DENSITY_FIELD_D = 20``."""

DENSITY_INPUT = "legacysurvey_ambient_768"
"""D8-07: which modality's embedding the density field is computed over -- re-declared fresh
from ``crossmodal_curvature.py``'s own ``DENSITY_INPUT``."""

DENSITY_SIGN_CONVENTION = (
    "curvature_probe.local_density_weights returns the per-point INVERSE density w, "
    "mean-normalized to 1. The density used throughout this phase is the RELATIVE density "
    "1.0 / w, matching Phase 4's printed convention (region_partition_mknn_run.py REGN-01) so "
    "Phase 4 / 7 / 07.1 / 8 density numbers stay comparable rather than needing translation "
    "(D8-07)."
)
"""D8-07's sign convention: the density used throughout this phase is the RELATIVE density
``1.0 / w``, matching Phase 4's printed convention."""

STRATIFICATION_RULE = (
    "Strata are density_stratified_null.density_strata's equal-count quantile bins on density "
    "RANK, reused unchanged (D8-06); the within-stratum tertile split "
    "(tertile_split_within_strata) is built on top of that stratification, never a "
    "reimplementation of it. Because the tertile split is computed WITHIN each stratum, the "
    "three tertiles' density marginals are identical by construction (up to each stratum's own "
    "remainder), and equal-n falls out for free. This means the tertiles rank "
    "DENSITY-RESIDUALIZED CURVATURE, not raw ||H|| -- a semantic consequence that must be stated "
    "explicitly in 08-FINDINGS.md, not buried."
)
"""Names ``density_stratified_null.density_strata``'s exact binning convention this phase
reuses, PLUS D8-06's semantic consequence: the tertiles this phase computes rank
DENSITY-RESIDUALIZED CURVATURE, not raw ``||H||``."""

SENSITIVITY_GRID_RULE = (
    "S_GRID is a grid of THRESHOLDS, not point estimates (07.1's SENSITIVITY_GRID_RULE "
    "pattern). S does not buy sample size -- pooled subset size is ~3,333 per tertile at any S; "
    "S instead trades density-match tightness against realized ||H|| contrast, because "
    "within-stratum tertiles are computed on n/S points. A reading at one S may qualify a "
    "reading at another but never overturns or escalates it; the only verdict-bearing use of "
    "this grid is D8-09's clearance-at-every-S requirement."
)
"""States D8-08/D8-09's grid semantics -- ``S_GRID`` is a grid of THRESHOLDS, not point
estimates; there is NO headline ``S``; clearance is required at EVERY grid point; an
``S``-dependent gap is self-reporting as an artifact rather than something a reader has to
notice."""

# --- 08-02 additions: the tertile-difference panel and its within-stratum label-permutation
# null (D8-10/D8-11) --------------------------------------------------------------------------

N_PERMUTATIONS = 1000
"""D8-11: the number of resamples the stratified label-permutation null draws (Phase 7/07.1's
own convention)."""

PERMUTATION_SEED = 20260827
"""D8-11: the RNG seed for the null -- a fresh date-stamped literal, re-declared across the
freeze boundary rather than imported (07.1's ``PERMUTATION_SEED`` idiom)."""

NULL_QUANTILE_PER_TAIL = 0.975
"""D8-11: the two-tailed empirical quantile the null threshold is read at (Phase 7's own
two-tailed permutation-wrapper convention)."""

NULL_KERNELS = ("linear", "rbf_sigma")
"""D8-11: which kernel(s) the permutation null is computed for -- not necessarily every entry of
``KERNELS``, since a null computed on a non-gating diagnostic kernel would gate nothing."""

TERTILE_STATISTIC_RULE = (
    "The statistic is CKA(tertile 3) minus CKA(tertile 1); the middle tertile is printed beside "
    "it as a shape diagnostic and gates nothing (MIDDLE_TERTILE_IS_NON_GATING = True). "
    "Monotone-trend and Phase-6-style compound criteria were both considered and rejected: "
    "Phase 6 died on a monotonicity criterion while its other two criteria held, and at three "
    "buckets a trend statistic is near-powerless (D8-10)."
)
"""States D8-10's statistic is ``CKA(tertile 3) - CKA(tertile 1)``; the middle tertile is
printed beside it as a shape diagnostic and gates nothing; monotone-trend and Phase-6-style
compound criteria were considered and rejected -- Phase 6 died on a monotonicity criterion while
its other two criteria held, and at three buckets a trend statistic is near-powerless."""

NULL_CONSTRUCTION_RULE = (
    "The null permutes ||H|| tertile LABELS within density strata, then recomputes the entire "
    "three-subset CKA panel -- preserving density structure and subset sizes exactly, breaking "
    "only the curvature link (D8-11). This is NOT mknn.permutation_null's "
    "permutation_type='pairings' row-pairing shuffle, which nulls alignment itself (a question "
    "Phase 7 already settled), not whether alignment differs by curvature. It is also NOT a "
    "bootstrap CI on the difference: CKA is a nonlinear function of the whole subset, its "
    "bootstrap bias is uncharacterized, and this record has no precedent for it."
)
"""States D8-11's null permutes ``||H||`` tertile LABELS within density strata and recomputes
the entire three-subset panel, preserving density structure and subset sizes exactly while
breaking only the curvature link; it is explicitly NOT ``mknn.permutation_null``'s row-pairing
shuffle (which nulls alignment itself, a question Phase 7 already settled) and NOT a bootstrap CI
(whose bias on a nonlinear whole-subset statistic is uncharacterized and has no precedent in this
record)."""

MIDDLE_TERTILE_IS_NON_GATING = True
"""D8-10: the middle tertile's CKA is a shape diagnostic printed beside the verdict and is never
read by any verdict function."""

# --- 08-02 additions: verdict rules -- clearance at every S, per-d independence, seed
# unanimity, and the four non-gating declarations (D8-09/12/13/15, D8-20) --------------------

D_SWEEP = (20, 25, 32)
"""D8-13/D8-14: the ``d`` values this phase's verdict sweeps -- re-declared fresh from
``crossmodal_curvature.py``'s own ``D_SWEEP``, never imported across the freeze boundary."""

SEED_FIELD_D = 25
"""D8-14: the single ``d`` value the ``TORCH_INIT_SEEDS`` axis is measured at -- the only ``d``
with three existing seed fields to reuse; no decoder is ever retrained."""

TORCH_INIT_SEEDS = (0, 1, 2)
"""D8-14: the three decoder-initialization seeds the seed-stability verdict is measured across --
07.1's own three existing ``d=25`` seed fields, never retrained. Reported at all three ``S``
values (18 cells total: 3 fields x 6 seed/d cells at all three S -- the resolution of
``08-RESEARCH.md`` Open Question 1, departing from its single-headline-``S`` recommendation
because D8-09 leaves no headline ``S`` to restrict to)."""

VERDICT_RULE = (
    "There is NO headline S in S_GRID; the verdict fires only if the curvature-CKA gap clears "
    "its two-tailed null at EVERY grid point in S_GRID. Relaxing this after seeing an "
    "S-dependent result is exactly the post-hoc retuning the k*=15 and 02.2 pre-registrations "
    "exist to prevent; an S-dependent gap is self-reporting as an artifact rather than something "
    "a reader has to notice."
)
"""States D8-09 in full -- there is NO headline ``S``; the verdict fires only if the gap clears
its two-tailed null at EVERY point in ``S_GRID``; relaxing this after seeing an ``S``-dependent
result is exactly the post-hoc retuning the ``k*=15`` and ``02.2`` pre-registrations exist to
prevent. Names ``S_GRID`` (checked below by ``assert_preregistered``, mirroring
``linear_probe.assert_preregistered``'s own ``VERDICT_RULE`` / ``N_BUCKETS`` naming check)."""

SEED_HANDLING_RULE = "no_pooling_per_seed_verdicts"
"""D8-15: the exact ratified string, carrying ``05-03-DECISION.md``'s one-way no-pooling
constraint verbatim. Checked by EXACT STRING EQUALITY below, not truthiness, mirroring
``linear_probe.py``'s own guard -- a future edit that assigns any other non-empty string
(re-entering the pooled design ``05-03-DECISION.md`` rejected) must fail this guard rather than
pass it."""

SEED_VERDICT_COMBINATION_RULE = (
    "Unanimous 3-of-3 clearance -> 'CLEARS IN ALL THREE SEEDS'; zero clearances -> 'NO "
    "CLEARANCE IN ANY SEED'; one or two clearances -> the terminal, non-supportive 'SPLIT ACROSS "
    "SEEDS', never upgraded by majority vote (D8-15, carrying 05-03-DECISION.md's ratified "
    "never-pool constraint)."
)
"""States :func:`combine_seed_verdicts`' unanimous-3-of-3-or-nothing combination -- three
clearances ``"CLEARS IN ALL THREE SEEDS"``, zero ``"NO CLEARANCE IN ANY SEED"``, one or two the
terminal ``"SPLIT ACROSS SEEDS"`` -- never upgraded by majority vote."""

D32_IS_NON_GATING = True
"""D8-12: ``d=32`` is a REPORTED DIAGNOSTIC that gates nothing and is NOT a hard invalidator; a
hard-invalidator reading was offered and explicitly declined by the developer on 2026-08-27."""

VALIDATION_LADDER_IS_NON_GATING = True
"""D8-20: all three validation-ladder rungs (D8-16/18/19) run and are reported beside the
verdict, and none of them gates it; a hard-gate ordering was offered and explicitly declined by
the developer on 2026-08-27."""

# --- 08-04 additions: control and reporting constants (D8-18/19/21), born already-frozen -- ---
# nothing before this freeze commit reads them; D8-22 only requires that they be committed in
# the single freeze commit that precedes every Phase 8 number (08-04-PLAN.md
# <artifacts_this_phase_produces>).

N_REPEATS = 30
"""D8-19: the number of shuffled-``||H||`` end-to-end calibration repeats defining the
false-positive rate. Frozen rather than left as a ``--n-repeats`` flag, because it determines the
precision of a number D8-21 makes mandatory -- a post-hoc ``--n-repeats 3`` could shrink an
inconvenient rate. Run at all three ``S`` (90 full null computations total)."""

NEGATIVE_CONTROL_FIELD = "h_norm_25"
"""D8-19: the frozen curvature field whose ``||H||`` values are shuffled across points (marginal
preserved, point correspondence destroyed) to measure the negative-control false-positive
rate."""

PLANTED_EFFECT_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
"""D8-18: the fraction of the high-``||H||`` tertile's rows in one modality whose crossmodal
pairing is destroyed, swept to read a detection floor at PU's realized ~1.5x dynamic range rather
than at Phase 6's ~20x."""

PLANTED_EFFECT_SEED = 20260827
"""D8-18: the RNG seed for the planted-effect ladder's row-destruction draws."""

RECORD_STEM = "08_cka_alignment"
"""D8-22: the frozen record stem 08-05 onward's production modes append their JSONL rows to,
via ``cache.cache_path(RECORD_STEM, "jsonl")``. No file at this stem exists before this freeze
commit -- ``notebooks/.cache/08_cka_alignment.jsonl`` is created only when a production mode
first runs, never during this plan."""

REPORTING_BLOCK_ROWS = (
    "d32_gap",
    "shuffled_h_false_positive_rate",
    "planted_effect_detection_floor",
    "realized_h_contrast_per_s",
    "sigma_rungs",
)
"""D8-21: the exact set of five rows ``08-FINDINGS.md`` must print regardless of outcome, each
beside the headline and not in an appendix."""

REPORTING_BLOCK_RULE = (
    "08-FINDINGS.md prints all five REPORTING_BLOCK_ROWS regardless of outcome, each beside the "
    "headline and not in an appendix -- 07.1's D-15 (per-d table reported unconditionally) is "
    "the precedent (D8-21)."
)
"""States that ``08-FINDINGS.md`` prints all five ``REPORTING_BLOCK_ROWS`` unconditionally,
beside the headline, never in an appendix."""

VERDICT_SENTENCE_RULE = (
    "The verdict sentence in 08-FINDINGS.md cannot be written without stating d=32's gap and "
    "the shuffled-||H|| false-positive rate in the same sentence -- this makes it structurally "
    "impossible to quote a headline without its caveat, the failure mode by which Phase 4's "
    "number escaped its confound (D8-21)."
)
"""States that the verdict sentence cannot be written without the ``d=32`` gap and the
shuffled-``||H||`` false-positive rate in the same sentence."""


_REQUIRED_CONSTANTS = (
    "KERNELS",
    "SIGMA_MULTIPLIERS",
    "SIGMA_HSC",
    "SIGMA_LEGACYSURVEY",
    "GRAM_DTYPE",
    "HSIC_ESTIMATOR_RULE",
    "SIGMA_FREEZE_RULE",
    "ALIGNMENT_METRIC",
    "SUPERSEDES",
    "SUPERSESSION_RULE",
    "SWISS_ROLL_APPLICABILITY_RULE",
    "RBF_IS_NON_GATING",
    "SIGMA_LADDER_IS_NON_GATING",
    "DIAGNOSTICS_ARE_NON_GATING",
    "S_GRID",
    "N_TERTILES",
    "DENSITY_K",
    "DENSITY_FIELD_D",
    "DENSITY_INPUT",
    "DENSITY_SIGN_CONVENTION",
    "STRATIFICATION_RULE",
    "SENSITIVITY_GRID_RULE",
    "N_PERMUTATIONS",
    "PERMUTATION_SEED",
    "NULL_QUANTILE_PER_TAIL",
    "NULL_KERNELS",
    "TERTILE_STATISTIC_RULE",
    "NULL_CONSTRUCTION_RULE",
    "MIDDLE_TERTILE_IS_NON_GATING",
    "D_SWEEP",
    "SEED_FIELD_D",
    "TORCH_INIT_SEEDS",
    "VERDICT_RULE",
    "SEED_HANDLING_RULE",
    "SEED_VERDICT_COMBINATION_RULE",
    "D32_IS_NON_GATING",
    "VALIDATION_LADDER_IS_NON_GATING",
    "N_REPEATS",
    "NEGATIVE_CONTROL_FIELD",
    "PLANTED_EFFECT_GRID",
    "PLANTED_EFFECT_SEED",
    "RECORD_STEM",
    "REPORTING_BLOCK_ROWS",
    "REPORTING_BLOCK_RULE",
    "VERDICT_SENTENCE_RULE",
)
"""Every gating constant this module declares, in declaration order -- 45 total after the 08-04
freeze commit (37 declared through 08-03 plus the eight control/reporting constants born
already-frozen here per 08-04-DECISION.md's guard-coverage fix). A constant added later without
a guard entry here fails the parametrized rejection sweep in
``tests/test_cka.py::test_assert_preregistered_rejects_unset_constant`` -- that is the mechanism
this tuple exists to serve."""


def assert_preregistered() -> None:
    """Refuse to proceed while any pre-registered Phase 8 constant is UNSET.

    One check per name in :data:`_REQUIRED_CONSTANTS`, in declaration order, raising
    ``RuntimeError`` on the FIRST failure. A value is UNSET if it is ``None``, an empty tuple,
    or an empty-or-whitespace-only string -- the three UNSET sentinels this module's own
    constants block used before the 08-04 freeze. As of the 08-04 freeze commit, all 45
    constants are filled and this function returns without raising; a later edit that reverts
    any one of them to an UNSET sentinel is caught by this same generic sweep.
    """
    g = globals()
    for name in _REQUIRED_CONSTANTS:
        value = g.get(name, None)
        is_unset = (
            value is None
            or (isinstance(value, tuple) and len(value) == 0)
            or (isinstance(value, str) and not value.strip())
        )
        if is_unset:
            raise RuntimeError(
                f"assert_preregistered: {name}={value!r} is UNSET. Every Phase 8 gating "
                "constant must be filled by the single 08-04 freeze commit (D8-22) before any "
                "Phase 8 number may be computed. A later edit to a filled constant after any "
                "Phase 8 number exists is a pre-registration breach -- the only remedy is a "
                "fresh freeze and a fresh run."
            )

    # T-08-09 (D8-15): SEED_HANDLING_RULE must equal the exact ratified value, not merely be a
    # non-empty string, so a future edit reintroducing seed pooling under a differently worded
    # rule string still fails loudly -- linear_probe.py's own precedent (SEED_HANDLING_RULE !=
    # "no_pooling_per_seed_verdicts"). Reached only once the generic loop above has already
    # confirmed SEED_HANDLING_RULE is non-empty.
    if SEED_HANDLING_RULE != "no_pooling_per_seed_verdicts":
        raise RuntimeError(
            f"assert_preregistered: SEED_HANDLING_RULE={SEED_HANDLING_RULE!r} does not equal "
            '"no_pooling_per_seed_verdicts" -- the ratified no-pooling decision '
            "(05-03-DECISION.md, carried by D8-15)."
        )

    # T-08-10 (D8-09): VERDICT_RULE must NAME S_GRID, so a future edit relaxing the
    # clearance-at-every-S requirement to a subset check is caught even though the string
    # remains non-empty -- mirrors linear_probe.assert_preregistered's own VERDICT_RULE /
    # N_BUCKETS naming check.
    if "S_GRID" not in VERDICT_RULE:
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} does not name S_GRID."
        )


# =============================================================================================
# Estimator functions -- pure numpy, no file I/O, no module-level default parameters that could
# be inherited silently across a call site.
# =============================================================================================


def _zero_diag(K: np.ndarray) -> np.ndarray:
    """Copy of `K` with the diagonal set to 0.0. Never mutates the caller's array."""
    K = np.asarray(K).copy()
    np.fill_diagonal(K, 0.0)
    return K


def unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """The Song et al. (2012) unbiased HSIC estimator, computed on RAW Gram matrices with only
    the diagonal zeroed.

    ``HSIC_1(K, L) = 1/(n(n-3)) * [ tr(K~L~) + (1'K~1)(1'L~1)/((n-1)(n-2)) - (2/(n-2))*1'K~L~1 ]``
    where ``K~``/``L~`` are `K`/`L` with the diagonal zeroed.

    **CRITICAL: `K` and `L` must be the RAW Gram matrices, only zero-diagonalized -- never
    double-centered (`H K H`) first.** The `1/(n(n-3))` correction terms above already perform
    the debiasing; applying them to a pre-centered matrix silently reproduces (a scaled variant
    of) the *biased* estimator under this unbiased formula's name. This is the exact trap D8-02
    exists to avoid; ``tests/test_cka.py::test_double_centering_changes_the_answer`` pins it
    behaviorally.

    Raises ``ValueError`` on non-square, shape-mismatched, or non-finite input before doing any
    arithmetic, and on ``n <= 3`` (the estimator's own floor -- `(n-1)(n-2)` and `(n-2)` in the
    denominators must be non-zero and positive).
    """
    K = np.asarray(K)
    L = np.asarray(L)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"unbiased_hsic: K has shape {K.shape}; must be a square 2D array.")
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"unbiased_hsic: L has shape {L.shape}; must be a square 2D array.")
    if K.shape != L.shape:
        raise ValueError(
            f"unbiased_hsic: K has shape {K.shape} but L has shape {L.shape}; they must match."
        )
    if not np.all(np.isfinite(K)):
        raise ValueError("unbiased_hsic: K contains non-finite values.")
    if not np.all(np.isfinite(L)):
        raise ValueError("unbiased_hsic: L contains non-finite values.")
    n = K.shape[0]
    if n <= 3:
        raise ValueError(f"unbiased_hsic: n={n} must exceed 3 (Song et al. 2012 floor).")
    Kt, Lt = _zero_diag(K), _zero_diag(L)
    ones = np.ones(n)
    term1 = np.trace(Kt @ Lt)
    term2 = (ones @ Kt @ ones) * (ones @ Lt @ ones) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ Kt @ Lt @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """Centered Kernel Alignment, composed from the unbiased HSIC estimator above:
    ``CKA(K, L) = HSIC_1(K, L) / sqrt(HSIC_1(K, K) * HSIC_1(L, L))``.

    `K` and `L` are RAW Gram matrices (see :func:`unbiased_hsic`'s critical note); this function
    never centers them itself.
    """
    hsic_kl = unbiased_hsic(K, L)
    hsic_kk = unbiased_hsic(K, K)
    hsic_ll = unbiased_hsic(L, L)
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


def linear_gram(X: np.ndarray, dtype: Any) -> np.ndarray:
    """Linear kernel Gram matrix, ``X @ X.T``, cast to `dtype`. `dtype` is a required, explicit
    call-site argument -- never a module-level default -- so a caller can never silently inherit
    a stale precision choice."""
    X = np.asarray(X)
    return (X @ X.T).astype(dtype)


def median_pairwise_distance(X: np.ndarray) -> float:
    """D8-03's sigma: the median Euclidean pairwise distance over ALL rows of `X`. In
    production this means all 10,000 points of one modality, computed once, before any subset
    (tertile, stratum, permutation) ever exists. Never call this on a subset -- :func:`rbf_gram`
    requires `sigma` explicitly and has no default specifically so this mistake cannot happen
    silently at a call site that only has access to a subset (D8-03's named confound)."""
    X = np.asarray(X)
    return float(np.median(pdist(X, metric="euclidean")))


def rbf_gram(X: np.ndarray, sigma: float, dtype: Any) -> np.ndarray:
    """RBF/Gaussian kernel Gram matrix at a REQUIRED, explicit bandwidth `sigma` -- no default
    value, by design. A call site that sees only a subset of the full point cloud can never
    silently compute and use a per-subset bandwidth, because there is nothing to fall back on if
    `sigma` is omitted (D8-03's named confound, restated as an interface property).

    Raises ``ValueError`` when `sigma` is non-finite or `sigma <= 0`.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"rbf_gram: sigma={sigma!r} must be finite and > 0.")
    X = np.asarray(X)
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))
    K = np.exp(-sq_dists / (2.0 * sigma ** 2))
    return K.astype(dtype)


def cka_on_subset(K_full: np.ndarray, L_full: np.ndarray, idx: np.ndarray) -> float:
    """CKA on the subset named by `idx`, computed via submatrix indexing into already-built
    full Gram matrices: ``K_full[np.ix_(idx, idx)]`` / ``L_full[np.ix_(idx, idx)]`` then
    :func:`cka`. This is EXACT, not an approximation -- a kernel value `K(x_i, x_j)` depends
    only on the pair `(x_i, x_j)`, never on which other points are present in the batch. This is
    the Gram-matrix-once/index-many architecture this phase's entire runtime budget depends on
    (08-RESEARCH.md's Runtime/Cost Model)."""
    idx = np.asarray(idx)
    K_sub = K_full[np.ix_(idx, idx)]
    L_sub = L_full[np.ix_(idx, idx)]
    return cka(K_sub, L_sub)


# =============================================================================================
# 08-02 additions: the within-density-stratum tertile split and the realized-contrast
# diagnostic (D8-05/06/07/08). ``strata`` is always an array already produced by
# ``density_stratified_null.density_strata(density, S)`` at some call site upstream of these
# functions -- imported there as a pure function only; no gating value ever crosses the freeze
# boundary. These functions never call ``density_strata`` themselves; they only ever consume the
# stratum-id array it produces, exactly as D8-06's split is specified to be built ON TOP of it,
# never a reimplementation of it.
# =============================================================================================


def tertile_split_within_strata(
    h: np.ndarray, strata: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D8-06's within-density-stratum ``||H||`` tertile split.

    For each unique stratum id in `strata`, rank that stratum's points by `h` using a stable
    argsort (ascending), then cut into three contiguous rank blocks of size ``n_s // 3`` with the
    ``n_s % 3`` remainder going to the LAST (highest-``h``) block -- the same remainder-to-last
    convention ``density_stratified_null.density_strata`` itself uses when dividing `n` points
    into `S` strata, so the two binning rules agree rather than each inventing their own. The
    per-stratum blocks are pooled across strata into three global index arrays, each returned
    sorted ascending.

    Because the split is computed WITHIN each stratum independently, tertile 3 holds the
    highest-``h`` third within every stratum, never the globally highest third -- this is what
    makes the three returned subsets' density-stratum marginals identical by construction
    (D8-06), up to each stratum's own ``n_s % 3`` remainder.

    Raises ``ValueError`` when `h` and `strata` have different lengths, when `h` contains a
    non-finite value, or when any stratum holds fewer than 3 points (naming the offending
    stratum and its size) -- a stratum that small cannot support a three-way split at all.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    strata = np.asarray(strata).ravel()
    if h.shape[0] != strata.shape[0]:
        raise ValueError(
            f"tertile_split_within_strata: h has {h.shape[0]} entries but strata has "
            f"{strata.shape[0]}; they must be row-aligned."
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("tertile_split_within_strata: h contains non-finite values.")

    tertile_blocks: Tuple[list, list, list] = ([], [], [])
    for stratum_id in np.unique(strata):
        idx = np.where(strata == stratum_id)[0]
        n_s = idx.shape[0]
        if n_s < 3:
            raise ValueError(
                f"tertile_split_within_strata: stratum {stratum_id!r} holds {n_s} point(s), "
                "below the 3-point floor a within-stratum tertile split requires."
            )
        order = idx[np.argsort(h[idx], kind="stable")]
        bin_size = n_s // 3
        tertile_blocks[0].append(order[:bin_size])
        tertile_blocks[1].append(order[bin_size:2 * bin_size])
        tertile_blocks[2].append(order[2 * bin_size:])  # remainder -> last (highest-h) block

    return tuple(np.sort(np.concatenate(blocks)) for blocks in tertile_blocks)


def realized_h_contrast(h: np.ndarray, tertiles: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """D8-21's mandatory "realized ``||H||`` contrast per `S`" row: the tertile-3 median of `h`
    over the tertile-1 median, strictly greater than 1.0 whenever `h` is non-constant. This is
    the number that makes D8-18's planted effect calibratable against PU's measured ~1.5x
    spread. Reported, never gated on."""
    h = np.asarray(h, dtype=np.float64).ravel()
    t1, _t2, t3 = tertiles
    return float(np.median(h[t3]) / np.median(h[t1]))


# =============================================================================================
# 08-02 additions: the tertile-difference panel and its within-stratum label-permutation null
# (D8-10/D8-11). Every kernel-recomputation call site below reads ONLY already-built Gram
# matrices via cka_on_subset's np.ix_ submatrix path -- no linear_gram/rbf_gram call happens
# inside stratified_tertile_label_null's resample loop (RESEARCH.md Pitfall 3).
# =============================================================================================


def tertile_gap_panel(
    K_full: Dict[str, np.ndarray], L_full: Dict[str, np.ndarray],
    tertiles: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """D8-10's tertile-difference panel: for every kernel present in both `K_full` and `L_full`
    (already-built full ``(n, n)`` Gram matrices), computes CKA on all three tertiles plus the
    tertile-3-minus-tertile-1 gap. The middle tertile's CKA is present in the returned panel
    (a shape diagnostic) but is not consumed by any verdict function below.

    Raises ``ValueError`` when `K_full` and `L_full` do not share exactly the same kernel-name
    key set.
    """
    if set(K_full.keys()) != set(L_full.keys()):
        raise ValueError(
            f"tertile_gap_panel: K_full keys {sorted(K_full)} and L_full keys {sorted(L_full)} "
            "differ; both dicts must be keyed by the same kernel names."
        )
    t1, t2, t3 = tertiles
    panel: Dict[str, Dict[str, float]] = {}
    for name in K_full:
        K, L = K_full[name], L_full[name]
        cka_t1 = cka_on_subset(K, L, t1)
        cka_t2 = cka_on_subset(K, L, t2)
        cka_t3 = cka_on_subset(K, L, t3)
        panel[name] = {
            "cka_t1": cka_t1,
            "cka_t2": cka_t2,
            "cka_t3": cka_t3,
            "gap": cka_t3 - cka_t1,
        }
    return panel


def stratified_tertile_label_null(
    h: np.ndarray, strata: np.ndarray, K_full: Dict[str, np.ndarray], L_full: Dict[str, np.ndarray],
    n_resamples: int, seed: int,
) -> Dict[str, np.ndarray]:
    """D8-11's within-stratum ``||H||`` tertile-label permutation null.

    Permutes ``||H||`` LABELS within each density stratum -- preserving density structure and
    every tertile's size exactly, breaking only the curvature link -- then recomputes the entire
    three-subset panel and records ``CKA(tertile 3) - CKA(tertile 1)`` per kernel, for
    `n_resamples` resamples. This is NOT ``mknn.permutation_null``'s row-pairing shuffle (that
    nulls global alignment, a question Phase 7 already settled) and never rebuilds a Gram matrix
    -- every kernel value below comes from :func:`cka_on_subset` indexing into the caller's
    already-built `K_full`/`L_full`.

    All `n_resamples` within-stratum permutation index sets are precomputed from a single
    ``np.random.default_rng(seed)`` BEFORE entering the recomputation loop (mirroring
    ``density_stratified_null.stratified_partial_null``'s own precompute-outside-the-loop
    optimization), so the loop body performs no further RNG calls -- pure array indexing plus
    HSIC arithmetic. Two calls at the same `seed` reproduce the same null arrays exactly; a
    different `seed` gives different arrays.

    Returns one ``float64`` array of length `n_resamples` per kernel name present in `K_full`.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    strata = np.asarray(strata).ravel()
    if h.shape[0] != strata.shape[0]:
        raise ValueError(
            f"stratified_tertile_label_null: h has {h.shape[0]} entries but strata has "
            f"{strata.shape[0]}; they must be row-aligned."
        )

    rng = np.random.default_rng(seed)
    strat_indices = [np.where(strata == s)[0] for s in np.unique(strata)]

    # Precompute every resample's within-stratum permutation BEFORE the recomputation loop, so
    # the loop below draws no further randomness (RESEARCH.md's precompute-outside-the-loop
    # optimization, applied at the label-permutation level).
    precomputed_perms = [
        [rng.permutation(idx) for idx in strat_indices] for _ in range(n_resamples)
    ]

    null_by_kernel: Dict[str, np.ndarray] = {
        name: np.empty(n_resamples, dtype=np.float64) for name in K_full
    }
    for b in range(n_resamples):
        h_perm = h.copy()
        for idx, perm in zip(strat_indices, precomputed_perms[b]):
            h_perm[idx] = h[perm]
        tertiles = tertile_split_within_strata(h_perm, strata)
        for name in K_full:
            c3 = cka_on_subset(K_full[name], L_full[name], tertiles[2])
            c1 = cka_on_subset(K_full[name], L_full[name], tertiles[0])
            null_by_kernel[name][b] = c3 - c1
    return null_by_kernel


def null_threshold(null_array: np.ndarray, quantile_per_tail: float) -> Tuple[float, float]:
    """The two-tailed empirical thresholds off one null array: ``(1 - quantile_per_tail)`` and
    ``quantile_per_tail``. Two-tailed is inherited from Phase 7's own two-tailed permutation
    wrapper -- a negative gap (CKA higher in the LOW-``||H||`` tertile) is a real possible
    finding here, not a nuisance to discard by a one-tailed test."""
    null_array = np.asarray(null_array, dtype=np.float64)
    low = float(np.quantile(null_array, 1.0 - quantile_per_tail))
    high = float(np.quantile(null_array, quantile_per_tail))
    return low, high


# =============================================================================================
# 08-02 additions: verdict rules -- clearance at every S (D8-09), independent per-d reporting
# (D8-13), unanimous-or-nothing seed combination (D8-15), and the pooled-field guard the
# never-pool-seeds ratification requires (05-03-DECISION.md).
# =============================================================================================

_PER_D_VERDICT_VALUES = ("CLEARS AT EVERY S", "DOES NOT CLEAR")
"""The two terminal outcome strings :func:`per_d_verdict` can produce -- the only values
:func:`combine_seed_verdicts` accepts as a per-seed input."""


def per_d_verdict(gaps_by_s: Dict[Any, float], thresholds_by_s: Dict[Any, Tuple[float, float]], rule: str) -> Dict[str, Any]:
    """D8-09's clearance-at-every-``S`` verdict, reported independently per ``d`` (D8-13) --
    this function reads only the gap/threshold values it is handed for ONE ``d``/seed cell and
    never touches another cell's inputs.

    `gaps_by_s` maps each ``S`` in the grid to its observed ``CKA(tertile 3) - CKA(tertile 1)``
    gap; `thresholds_by_s` maps the same ``S`` values to the ``(null_low, null_high)`` pair
    :func:`null_threshold` returns. The middle tertile's CKA is never read here -- it is not a
    parameter of this function at all.

    Clearance at one ``S`` is two-tailed: ``gap > null_high`` OR ``gap < null_low``. The verdict
    fires (``"CLEARS AT EVERY S"``) only when EVERY ``S`` clears; a single non-clearing ``S``
    yields ``"DOES NOT CLEAR"`` -- there is no headline ``S`` to average over or defer to.

    Raises ``RuntimeError`` when `rule` is empty -- this cannot run before the pre-registration
    freeze, mirroring :func:`combine_seed_verdicts`'s own guard. Raises ``ValueError`` when
    `gaps_by_s` and `thresholds_by_s` do not share exactly the same ``S`` key set, or when
    `S_GRID` is frozen (non-empty) and either mapping is missing a value in it.

    Returns a dict with ``verdict`` (the terminal outcome string), ``per_s`` (a mapping from
    each ``S`` to its ``gap``, ``null_low``, ``null_high`` and boolean ``clears``), and
    ``n_s_cleared`` -- so a reader can see exactly which ``S`` failed, not only the terminal
    string.
    """
    if not isinstance(rule, str) or not rule.strip():
        raise RuntimeError(
            "per_d_verdict: rule is empty; cannot run before the pre-registration freeze."
        )
    if set(gaps_by_s.keys()) != set(thresholds_by_s.keys()):
        raise ValueError(
            f"per_d_verdict: gaps_by_s keys {sorted(gaps_by_s)} and thresholds_by_s keys "
            f"{sorted(thresholds_by_s)} differ; both must cover exactly the same S grid."
        )
    if S_GRID and not set(S_GRID).issubset(gaps_by_s.keys()):
        raise ValueError(
            f"per_d_verdict: gaps_by_s keys {sorted(gaps_by_s)} do not cover every S in the "
            f"frozen S_GRID={S_GRID!r}."
        )

    per_s: Dict[Any, Dict[str, Any]] = {}
    for s_value, gap in gaps_by_s.items():
        null_low, null_high = thresholds_by_s[s_value]
        clears = bool(gap > null_high or gap < null_low)
        per_s[s_value] = {
            "gap": gap,
            "null_low": null_low,
            "null_high": null_high,
            "clears": clears,
        }
    n_s_cleared = sum(1 for entry in per_s.values() if entry["clears"])
    verdict = _PER_D_VERDICT_VALUES[0] if n_s_cleared == len(per_s) else _PER_D_VERDICT_VALUES[1]
    return {"verdict": verdict, "per_s": per_s, "n_s_cleared": n_s_cleared}


def combine_seed_verdicts(per_seed_verdicts: Dict[int, str], rule: str) -> Dict[str, Any]:
    """D8-15's unanimous-3-of-3-or-nothing seed combination, copying ``linear_probe.py`` lines
    831-887's shape exactly: three clearances -> ``"CLEARS IN ALL THREE SEEDS"``, zero ->
    ``"NO CLEARANCE IN ANY SEED"``, one or two -> the terminal, non-supportive ``"SPLIT ACROSS
    SEEDS"`` -- never upgraded by majority vote.

    `per_seed_verdicts` maps seed int to that seed's :func:`per_d_verdict` terminal string, each
    one of :data:`_PER_D_VERDICT_VALUES`. Raises ``RuntimeError`` when `rule` is empty (cannot
    run before the freeze). Raises ``ValueError`` unless `per_seed_verdicts` holds exactly three
    entries (naming the count actually supplied), and ``ValueError`` when any value is not a
    member of :data:`_PER_D_VERDICT_VALUES`.
    """
    if not isinstance(rule, str) or not rule.strip():
        raise RuntimeError(
            "combine_seed_verdicts: rule is empty; cannot run before the pre-registration "
            "freeze."
        )
    if not isinstance(per_seed_verdicts, dict) or len(per_seed_verdicts) != 3:
        n_seeds = len(per_seed_verdicts) if isinstance(per_seed_verdicts, dict) else None
        raise ValueError(
            "combine_seed_verdicts: per_seed_verdicts must hold exactly three seeds, got "
            f"{n_seeds if n_seeds is not None else per_seed_verdicts!r}."
        )
    for seed, verdict in per_seed_verdicts.items():
        if verdict not in _PER_D_VERDICT_VALUES:
            raise ValueError(
                f"combine_seed_verdicts: per-seed verdict for seed {seed} is {verdict!r}, not "
                f"one of {_PER_D_VERDICT_VALUES}."
            )

    sorted_seeds = sorted(per_seed_verdicts.keys())
    n_cleared = sum(1 for s in sorted_seeds if per_seed_verdicts[s] == _PER_D_VERDICT_VALUES[0])
    if n_cleared == 3:
        phase_verdict = "CLEARS IN ALL THREE SEEDS"
    elif n_cleared == 0:
        phase_verdict = "NO CLEARANCE IN ANY SEED"
    else:
        phase_verdict = "SPLIT ACROSS SEEDS"

    return {
        "phase_verdict": phase_verdict,
        "n_cleared": n_cleared,
        "n_seeds": len(sorted_seeds),
        "per_seed_verdicts": {s: per_seed_verdicts[s] for s in sorted_seeds},
        "rule": rule,
    }


def pooled_field_guard(fields: Any) -> None:
    """Raises ``RuntimeError`` naming ``05-03-DECISION.md`` and D8-15 whenever `fields` names
    more than one seed field to be combined into a single pooled field. Seeds are NEVER pooled:
    each seed gets its own within-stratum tertile split and its own verdict. Exists so any
    future pooled-mode equivalent fails at the FIRST call rather than silently averaging."""
    if len(fields) > 1:
        raise RuntimeError(
            "pooled_field_guard: received more than one seed field to combine into a single "
            "pooled field. Seeds are NEVER pooled (05-03-DECISION.md, carried by D8-15) -- each "
            "seed gets its own within-stratum tertile split and its own verdict."
        )
