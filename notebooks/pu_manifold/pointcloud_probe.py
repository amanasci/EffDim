"""Phase 6 point-cloud curvature-conditioned linear decodability: the pre-registration
constants block, its guard, and the single-verdict rule.

**This module adds; it does not edit.** ``linear_probe.py`` carries Phase 5's 31 frozen
constants and its git history is the ordering proof for Phase 5's sealed result, so it is
imported here and never modified (D6-04). Every computational function this phase needs --
the split, the ridge fit, the per-point residuals, the tertile edges, the bootstrap CI, the
size-matched re-check, and ``apply_verdict_rule`` itself -- already exists there, is pure,
and is reused unchanged. What this module contributes is a constants block for the ONE thing
Phase 6 changes and a verdict rule restated for one field instead of three.

**What Phase 6 changes, and the exhaustive list of what it does not.** The field the held-out
residuals are bucketed by, and nothing else. The ridge map (``hsc -> legacysurvey``), the one
70/30 split, the alpha grid and its selection rule, the intercept and per-target settings, the
preprocessing, the residual metric, the bucket count and the tertile rule are all Phase 5's own
frozen values, inherited by explicit re-declaration below so that a reader can diff the two
blocks rather than take this sentence on trust.

**Why the swap (D6-01).** Three facts, each from a sealed artifact:

(a) Phase 4 already partitions PU on ``centroid_mean_curvature``, density-corrected, at
    ``K_FROZEN = 500`` (``04-FINDINGS.md``). Phase 5 reverted to the CAE decoder route. The two
    consecutive phases use different instruments on the same manifold.

(b) The decoder route does not reproduce across seeds. ``05-03-DECISION.md`` measured the three
    seeds' fields mutually anti-correlated on rank (pairwise Spearman ``-0.1402``, ``+0.2019``,
    ``-0.2725`` -- sign-inconsistent, two of three negative) and directionally orthogonal
    (median cosine ``0.0007``-``0.0039``, 46-48 percent of points anti-aligned). Two of the
    three fields are near-degenerate as bucketing variables: seeds 20260814 and 20260815 take
    **4 and 3** effective distinct levels across all 10,000 points. The Phase 4 field takes
    9,750 distinct values of 10,000.

(c) The point-cloud estimator does no training and carries no model seed. It is deterministic
    given ``k``. Phase 5's terminal outcome ``SPLIT ACROSS SEEDS`` is therefore not reachable
    here, which is why ``SEED_VERDICT_COMBINATION_RULE`` and ``PHASE_VERDICT_VALUES`` are NOT
    carried over (D6-03) rather than carried over and left unused.

**The free cross-estimator reading (D6-06).** Phase 6 buckets the SAME 3,000 held-out per-point
residuals Phase 5 scored -- same split, same map, same residual metric -- so the rank agreement
between the Phase 4 field and each Phase 5 decoder field costs nothing extra and closes D4-08,
recommended at the Phase 3 close and declined twice (D4-03, D4-08). It is a DISCLOSURE. No
verdict here may be upgraded or downgraded by it.

**G6-01, carried at full strength and not buried in a limitations section.** The field this
module buckets on is validated on PU by split-half reliability alone, and split-half reliability
cannot detect a bias both halves share. That is not hypothetical: measured on the Swiss roll,
where the answer is known, ``R_H = 0.990`` (near-perfect two-half agreement) sat alongside
``rho = 0.469`` against true curvature (``04-FINDINGS.md`` Gap 1). There is no ground truth for
PU, so no amount of further split-half measurement can close it. **Phase 6 inherits this gap
verbatim and closes nothing.**

**G6-03.** ``K_FROZEN = 500`` is the largest ``k`` actually run, not a ``k`` the freeze rule
selected: ``04_k_freeze.json`` records ``rule_fired: false``. At ``k = 500`` on 10,000 rows a
neighbourhood is 5 percent of the cloud, and whether that is still local on the PU manifold is
unmeasured.

**G6-04.** A disagreement between Phase 5's verdict and Phase 6's localizes to the instrument. It
does NOT establish which instrument is correct, does not validate either field, does not show
that either measures true curvature, and reopens no sealed verdict -- not ``CAE_VERDICT = FAIL``
(Phase 02.2), not ``GATE_VERDICT = FAIL`` (Phase 2), and not Phase 5's ``SPLIT ACROSS SEEDS``.

No file I/O happens in this module. Every value is passed in by the caller with no default,
following ``linear_probe.py``'s and ``region_partition.py``'s stated convention: a default is how
a pre-registered value gets inherited by accident instead of by an explicit call-site choice.
"""

from typing import Any, Dict

# --- inherited from Phase 5, re-declared so the two blocks can be diffed -----------------
# Every value below is the verbatim value of the identically-named constant in
# linear_probe.py as committed at Phase 5's D5-09 freeze commit 32dabe3. Phase 6 changes
# none of them; re-declaring rather than importing makes the inheritance auditable.

TRAIN_FRACTION = 0.7
SPLIT_SEED = 20260824
SPLIT_RULE = (
    "One permutation of np.arange(10000) under SPLIT_SEED; first 7,000 of the permutation "
    "train, last 3,000 test; both sorted ascending; NOT stratified by bucket. Inherited from "
    "Phase 5 unchanged (D6-02) so the two phases score the identical held-out residuals."
)
RIDGE_ALPHA_GRID = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
RIDGE_SELECTION_RULE = (
    "scikit-learn RidgeCV's generalized leave-one-out CV on the training split alone, "
    "selecting one alpha from the grid"
)
ALPHA_PER_TARGET = False
FIT_INTERCEPT = True
EMBEDDING_PREPROCESSING = (
    "raw_as_cached -- both modalities are already L2-normalized upstream (every row norm "
    "equals 1.0 to float64 rounding in the resolved npz), so re-normalizing would be a "
    "no-op dressed as a decision."
)
RESIDUAL_METRIC = "squared_l2_per_point"
N_BUCKETS = 3
BUCKET_RULE = (
    "Tertiles of the curvature field over ALL 10,000 rows -- edges at the 33.333rd and "
    "66.667th percentiles of the full field, computed once and then applied to the test-split "
    "rows. Edges are NOT recomputed on the test split alone."
)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260824
CONFIDENCE_LEVEL = 0.95
SIZE_MATCH_RULE = (
    "Re-check the headline sign after subsampling the two headline buckets to their realized "
    "test-split counts, SIZE_MATCH_N_REPEATS times under SIZE_MATCH_SEED; the sign is stable "
    "if the CIs are disjoint in at least half the repeats."
)
SIZE_MATCH_N_REPEATS = 200
SIZE_MATCH_SEED = 20260824

# --- Phase 6's own constants: the one thing that changes ----------------------------------

CURVATURE_SOURCE = "phase_4_sealed_point_cloud_field"
"""D6-01. The field is READ, never recomputed. Recomputing would silently re-tune k and make
Phase 4's freeze meaningless."""

CURVATURE_SOURCE_ARTIFACT = "notebooks/.cache/04_region_partition.npz"
CURVATURE_SOURCE_KEY = "h_norm"
CURVATURE_SOURCE_FUNCTION = "curvature_probe.centroid_mean_curvature"
CURVATURE_DENSITY_CORRECTED = True
K_FROZEN = 500
K_DENSITY = 30
FIELD_D = 20
CURVATURE_CONVENTION = "trace"
SEED_HANDLING_RULE = "single_field_no_seeds"
"""D6-03. The point-cloud estimator is deterministic given k and carries no model seed, so
there is one field, one bucketing and one verdict. SPLIT ACROSS SEEDS is not reachable and
must not appear in any Phase 6 artifact."""

CROSS_ESTIMATOR_DISCLOSURE_SEEDS = (20260813, 20260814, 20260815)
"""D6-06. The three Phase 5 decoder fields, read only to report rank agreement against this
phase's field. A disclosure closing D4-08 -- never a gate."""

PREREGISTRATION_PATH = (
    ".planning/phases/06-point-cloud-curvature-conditioned-linear-decodability/"
    "06-PREREGISTRATION.md"
)

VERDICT_RULE = """D6-05 VERDICT_RULE -- frozen in committed source before any Phase 6 probe
number existed. ONE field, ONE verdict (D6-03).

The headline comparison is the highest-||H|| bucket (of N_BUCKETS = 3 tertiles of the Phase 4
sealed point-cloud field) against the lowest, on mean per-point squared L2 residual over the
ONE 70/30 test split inherited from Phase 5 (TRAIN_FRACTION, SPLIT_SEED), under BUCKET_RULE.

The verdict is HOLDS if and only if ALL three of:
  (a) the highest and lowest bucket's CONFIDENCE_LEVEL (0.95) percentile bootstrap CIs on mean
      per-point squared L2 residual are disjoint;
  (b) the highest bucket's mean residual strictly exceeds the lowest bucket's; AND
  (c) the sign survives SIZE_MATCH_RULE (subsampled to the realized test-split bucket counts)
      with CIs disjoint in at least half of SIZE_MATCH_N_REPEATS = 200 repeats.

NO DETECTABLE RELATIONSHIP is the verdict whenever any one of (a)/(b)/(c) fails. It is a
complete, valid, TERMINAL outcome -- never a phase failure, never escalated by the continuous
Spearman statistic, and never re-decided by trying a different N_BUCKETS or a different k.

These are the same three criteria Phase 5's per-seed VERDICT_RULE applied, applied once. That
is deliberate: holding the decision rule fixed while changing only the field is what makes the
two phases comparable at all.

The continuous Spearman between curvature magnitude and per-point residual on the test split is
reported as SENSITIVITY ONLY; it can neither establish nor overturn the verdict.

G6-01 CAVEAT, carried in this rule's own text rather than only alongside it: the field this rule
buckets on is validated on PU by split-half reliability alone, which cannot detect a bias both
halves share -- measured directly on the Swiss roll at R_H = 0.990 alongside rho = 0.469 against
true curvature. No verdict produced under this rule may be read as evidence that the field
measures true curvature.

G6-03 CAVEAT: K_FROZEN = 500 is the largest k actually run, not a k the Phase 4 freeze rule
selected (04_k_freeze.json records rule_fired: false). At k = 500 on 10,000 rows a neighbourhood
is 5 percent of the cloud; whether that is still local on PU is unmeasured.

G6-04 CAVEAT: a disagreement with Phase 5's SPLIT ACROSS SEEDS localizes to the instrument and
does NOT establish which instrument is correct. No sealed verdict is reopened by this rule.

D6-06 NOTE: cross-estimator rank agreement against the three Phase 5 decoder fields is reported
alongside the verdict as a disclosure closing D4-08. It is not a gate under this rule.

D6-08 NOTE: spearman(density, ||H||) is reported alongside the verdict as a disclosure only,
following Phase 5's D5-13 precedent. It is not a gate under this rule.
"""

VERDICT_VALUES = ("HOLDS", "NO DETECTABLE RELATIONSHIP")
"""The two terminal outcomes. There is no third, and no phase-level combination step (D6-03)."""


_REQUIRED_CONSTANTS = (
    "TRAIN_FRACTION", "SPLIT_SEED", "SPLIT_RULE", "RIDGE_ALPHA_GRID", "RIDGE_SELECTION_RULE",
    "ALPHA_PER_TARGET", "FIT_INTERCEPT", "EMBEDDING_PREPROCESSING", "RESIDUAL_METRIC",
    "N_BUCKETS", "BUCKET_RULE", "N_BOOTSTRAP", "BOOTSTRAP_SEED", "CONFIDENCE_LEVEL",
    "SIZE_MATCH_RULE", "SIZE_MATCH_N_REPEATS", "SIZE_MATCH_SEED", "CURVATURE_SOURCE",
    "CURVATURE_SOURCE_ARTIFACT", "CURVATURE_SOURCE_KEY", "CURVATURE_SOURCE_FUNCTION",
    "CURVATURE_DENSITY_CORRECTED", "K_FROZEN", "K_DENSITY", "FIELD_D",
    "CURVATURE_CONVENTION", "SEED_HANDLING_RULE", "CROSS_ESTIMATOR_DISCLOSURE_SEEDS",
    "PREREGISTRATION_PATH", "VERDICT_RULE", "VERDICT_VALUES",
)


def assert_preregistered() -> None:
    """Refuse to proceed while any pre-registered constant is unset (D6-05).

    Mirrors ``linear_probe.assert_preregistered``'s contract: the bucketed, number-producing
    path calls this first, so a Phase 6 probe number cannot be computed by a build of this
    module that predates the freeze. Raises ``RuntimeError`` naming every offending constant.
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
            "pointcloud_probe.assert_preregistered: Phase 6 is not frozen -- the following "
            "pre-registered constants are unset: " + ", ".join(missing) + ". No probe number "
            "may be computed before the freeze (D6-05)."
        )


def verdict_is_terminal(verdict: str) -> bool:
    """``verdict`` is one of :data:`VERDICT_VALUES`. Used by the runner to refuse to write a
    record carrying anything else -- in particular anything mentioning a seed split, which is
    unreachable for a single deterministic field (D6-03)."""
    return verdict in VERDICT_VALUES


def describe_inheritance() -> Dict[str, Any]:
    """The audit surface: every Phase 5 constant this phase inherits unchanged, so a reader can
    confirm mechanically that only the field differs. Pure; no I/O."""
    return {
        "inherited_from_phase_5": {
            "TRAIN_FRACTION": TRAIN_FRACTION,
            "SPLIT_SEED": SPLIT_SEED,
            "RIDGE_ALPHA_GRID": RIDGE_ALPHA_GRID,
            "ALPHA_PER_TARGET": ALPHA_PER_TARGET,
            "FIT_INTERCEPT": FIT_INTERCEPT,
            "EMBEDDING_PREPROCESSING": EMBEDDING_PREPROCESSING,
            "RESIDUAL_METRIC": RESIDUAL_METRIC,
            "N_BUCKETS": N_BUCKETS,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
            "CONFIDENCE_LEVEL": CONFIDENCE_LEVEL,
            "SIZE_MATCH_N_REPEATS": SIZE_MATCH_N_REPEATS,
            "SIZE_MATCH_SEED": SIZE_MATCH_SEED,
        },
        "changed_by_phase_6": {
            "CURVATURE_SOURCE": CURVATURE_SOURCE,
            "CURVATURE_SOURCE_FUNCTION": CURVATURE_SOURCE_FUNCTION,
            "K_FROZEN": K_FROZEN,
            "SEED_HANDLING_RULE": SEED_HANDLING_RULE,
        },
    }
