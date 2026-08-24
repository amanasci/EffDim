"""Diametrical sign-split partition (D4-09) on a curvature-field-like vector array.

``canonical_eigvec_sign(v)`` fixes an eigenvector's arbitrary sign (``numpy.linalg.eigh``
returns eigenvectors up to sign, and that sign is not guaranteed stable across
otherwise-identical runs). ``region_partition(H, min_norm_percentile)`` implements D4-09:
exclude points whose ``||H||`` falls below a within-config percentile of the field's OWN
``||H||`` distribution (never a fixed absolute magnitude -- Pitfall 2), then split the
survivors into two regions by the sign of their projection onto the leading eigenvector of
``Cov(H_i / ||H_i||)`` over the surviving points. This is precisely **diametrical
clustering** for k=2 clusters (Dhillon, Marcotte & Roshan, *Diametrical clustering for
identifying anti-correlated gene clusters*, Bioinformatics 19(13), 2003) -- the citation was
read only in secondary summary, not the primary source, so it is named here with that
caveat rather than asserted as directly verified.

``region_counts(labels, n_excluded, n_zero_projection)`` reports the region/exclusion counts
a caller already computed, in one place, with fractions.

The covariance form. D4-09's own text says "the unit-``H`` covariance." The
diametrical-clustering literature conventionally uses the *uncentered* second-moment
matrix (``mean(u_i u_i^T)``), while ``numpy.cov`` is mean-centered by construction. The two
coincide exactly when the unit-vector mean is near zero. This module implements the
mean-centered form -- matching 04-PATTERNS.md's composed snippet and D4-09's own word --
and ``region_partition`` reports ``mean_unit_norm`` (``||mean(unit)||``) alongside every
result so a reader can see for themselves whether the distinction bites on a given input.
It is not resolved silently.

The codimension gap this helper does not close. Every fixture the direction-partition
decision (D4-01) rests on is a codimension-1 graph, where ``H = H_scalar * n_hat`` -- so a
cosine near 1.000 on those fixtures demonstrates recovery of the surface's NORMAL
ORIENTATION, a tangent-space problem known to converge well, not resolution of ``H``'s
direction within a high-dimensional normal space. PU's codimension is roughly 748 (``d ~
20`` inside ``D = 768``), and nothing in this milestone measures or closes that gap
(03-NOTE-phase-4-decisions.md Amendment 01). This sign split partitions whatever direction
structure the field happens to carry; it does not itself validate that structure.

D4-10 overrides D4-01's body text naming the ridge-fixture check a Phase 4 precondition:
neither ``make_ridge_graph_control`` nor ``make_multinormal_ridge_control`` is run before
the PU split is frozen. This module's own known-answer coverage is the two-antipodal-cone
fixture in ``tests/test_region_partition.py``, which is a stricter test of the split logic
itself than either control fixture would be.
"""

from typing import Any, Dict

import numpy as np


# --- Pre-registration (D4-11, ratified at this plan's blocking checkpoint) -----------------
#
# PRE-REGISTERED under the ROADMAP's Ordering constraint: every constant below, and
# VERDICT_RULE's full text, were ratified at this plan's Task 2 blocking decision checkpoint
# BEFORE any regional MKNN number existed. Amending any of MIN_NORM_PERCENTILE, MIN_REGION_N,
# HEADLINE_K, MKNN_K_GRID, NULL_QUANTILE, CONFIDENCE_LEVEL, or VERDICT_RULE after a regional
# MKNN number has been computed invalidates the phase -- a rule chosen after seeing the
# numbers is a rationalization, not a pre-registration. See
# `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-PREREGISTRATION.md`
# for the full committed record, including the checkpoint's ratification note.

MIN_NORM_PERCENTILE = 5.0  # within-config percentile of the field's own ||H||, never absolute
MIN_REGION_N = 500  # = 10 * k_max at k_max = 50 (RESEARCH A4's number; no literature precedent)
MKNN_K_GRID = (5, 10, 20, 50)
HEADLINE_K = 20
NULL_QUANTILE = 0.99
CONFIDENCE_LEVEL = 0.95
N_PERMUTATIONS = 1000  # D4-17
N_BOOTSTRAP = 1000  # D4-17
FIELD_D = 20  # D-07: explicit call-site value, never re-derived
K_DENSITY = 30  # D4-15
SEED = 20260822  # existing runner's seed, kept for continuity
COVARIANCE_FORM = "mean_centered"  # np.cov's own form; see module docstring's caveat

# Copied verbatim from notebooks/.cache/04_k_freeze.json (plan 04-02's output). rule_fired is
# False: D4-07's freeze rule never fired anywhere across k in {30, 60, 120, 231, 350, 500) --
# median_R_H reached only 0.3436 against its 0.5 floor, and the per-step increment never
# collapsed toward the 0.03 ceiling (it ROSE at the last step, 0.0516 -> 0.0583). K_FROZEN=500
# is therefore the pre-registered fallback -- "the largest k actually run", a compute-budget
# ceiling -- NOT a detected reliability plateau. Never described as converged or settled.
K_FROZEN = 500
K_FREEZE_RULE = (
    "D4-07: freeze the curvature-field k at the smallest k in the ordered sweep grid whose "
    "median_R_H gain over the immediately preceding sweep point is strictly less than 0.03 "
    "AND whose median_R_H is greater than or equal to 0.5. The rule is evaluated from the "
    "SECOND sweep point onward, because the gain at the first point is undefined. If no k in "
    "the grid satisfies both conditions, the frozen k is the largest k actually run and the "
    "outcome is recorded as not-fired -- never adjusted post hoc."
)

VERDICT_RULE = """MKNN-07 verdict rule -- ratified at this plan's Task 2 blocking checkpoint,
before any regional MKNN number existed.

The high-vs-low regional MKNN result HOLDS at a given k if and only if BOTH:
  (a) the two regions' CONFIDENCE_LEVEL (0.95) percentile bootstrap CIs at that k are
      disjoint, AND
  (b) the higher-scoring region's observed MKNN strictly exceeds the NULL_QUANTILE (0.99)
      percentile of its OWN region-scoped permutation null.

The headline call is made at HEADLINE_K = 20 alone. The remaining grid values, k in
MKNN_K_GRID = (5, 10, 50), are reported as sensitivity only: they cannot overturn or escalate
the headline verdict, and take no separate multiplicity correction. No multiplicity correction
is applied across the 2x4 grid, because the four k values are a nested sensitivity sweep on
the same two regions and the same embeddings, not independent trials.

"NO DETECTABLE DIFFERENCE" at the headline k is a complete, valid outcome. It is never treated
as a phase failure and it is never escalated by a majority vote across the sensitivity k --
that alternative verdict shape was considered and rejected at the Task 2 checkpoint.

D4-14 CAVEAT, carried in this rule's own text rather than only alongside it: the
density-confound battery run in this phase is the REGN-02 correlation only -- no
density-matched null. MKNN is itself a k-NN statistic and therefore directly
density-sensitive. A detected regional MKNN difference under this rule CANNOT be attributed
to curvature rather than to regional density by anything in this phase.
"""

MIN_REGION_N_UNDEFINED_REASON = "n_region < MIN_REGION_N"


def assert_preregistered() -> None:
    """Raise ``RuntimeError`` unless the pre-registration is intact: ``VERDICT_RULE`` is a
    non-empty string naming ``HEADLINE_K``, ``K_FROZEN`` is a positive int, and
    ``MIN_REGION_N`` is a positive int. Called at the top of the runner's ``--mode regional``
    branch so the regional path fails loudly rather than computing anything when the
    pre-registration is absent or malformed."""
    if not isinstance(VERDICT_RULE, str) or not VERDICT_RULE.strip():
        raise RuntimeError("assert_preregistered: VERDICT_RULE is empty or not a string.")
    if "HEADLINE_K" not in VERDICT_RULE:
        raise RuntimeError("assert_preregistered: VERDICT_RULE does not name HEADLINE_K.")
    if not isinstance(K_FROZEN, int) or isinstance(K_FROZEN, bool) or K_FROZEN <= 0:
        raise RuntimeError(f"assert_preregistered: K_FROZEN={K_FROZEN!r} is not a positive int.")
    if (
        not isinstance(MIN_REGION_N, int)
        or isinstance(MIN_REGION_N, bool)
        or MIN_REGION_N <= 0
    ):
        raise RuntimeError(
            f"assert_preregistered: MIN_REGION_N={MIN_REGION_N!r} is not a positive int."
        )


def canonical_eigvec_sign(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled so its largest-magnitude component is positive. Without this,
    ``numpy.linalg.eigh``'s arbitrary eigenvector sign would flip region labels between
    otherwise identical runs on the same input."""
    v = np.asarray(v, dtype=np.float64)
    idx = int(np.argmax(np.abs(v)))
    sign = 1.0 if v[idx] >= 0 else -1.0
    return v * sign


def region_partition(H: np.ndarray, min_norm_percentile: float) -> Dict[str, Any]:
    """D4-09's diametrical sign split. ``min_norm_percentile`` is a required argument with
    no default, following this module's convention for a pre-registered constant: a
    default value is exactly how such a value gets inherited by accident rather than by an
    explicit call-site choice.

    Steps, in order: compute ``norm = ||H_i||``; take ``floor`` as the
    ``min_norm_percentile``-th percentile of ``norm`` (a within-config statistic of this
    field's own distribution, never a fixed absolute magnitude); keep points with
    ``norm >= floor`` (greater-than-or-equal -- a point exactly at the percentile is kept,
    a stated deterministic tie rule); normalize the survivors; form the mean-centered
    covariance of the unit vectors; take its leading eigenvector, sign-canonicalized;
    project every survivor onto it; label by the sign of that projection, with an exact
    zero projection going to region 0.

    Raises ``ValueError`` naming the offending argument: ``H`` non-finite; ``H`` not
    two-dimensional; fewer than 2 points survive the exclusion; ``min_norm_percentile``
    outside ``[0, 100)``.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(
            f"region_partition: H must be two-dimensional, got shape {H.shape}."
        )
    if not np.all(np.isfinite(H)):
        raise ValueError("region_partition: H contains a non-finite value.")
    if not (0.0 <= min_norm_percentile < 100.0):
        raise ValueError(
            f"region_partition: min_norm_percentile={min_norm_percentile} must be in "
            "[0, 100)."
        )

    norm = np.linalg.norm(H, axis=1)
    floor = float(np.percentile(norm, min_norm_percentile))
    keep = norm >= floor
    keep_idx = np.flatnonzero(keep)
    excluded_idx = np.flatnonzero(~keep)

    if keep_idx.shape[0] < 2:
        raise ValueError(
            f"region_partition: only {keep_idx.shape[0]} point(s) survive the "
            f"min_norm_percentile={min_norm_percentile} exclusion; need at least 2."
        )

    unit = H[keep_idx] / np.maximum(norm[keep_idx, None], 1e-12)
    cov = np.cov(unit, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top = int(np.argmax(eigvals))
    v = canonical_eigvec_sign(eigvecs[:, top])
    proj = unit @ v
    labels = np.where(proj >= 0, 0, 1)
    n_zero_projection = int(np.sum(proj == 0.0))
    mean_unit_norm = float(np.linalg.norm(unit.mean(axis=0)))

    return {
        "v": v,
        "labels": labels,
        "keep_idx": keep_idx,
        "excluded_idx": excluded_idx,
        "proj": proj,
        "h_norm": norm,
        "floor": floor,
        "min_norm_percentile": float(min_norm_percentile),
        "n_zero_projection": n_zero_projection,
        "eigval_top": float(eigvals[top]),
        "eigval_spectrum": eigvals,
        "mean_unit_norm": mean_unit_norm,
    }


def region_counts(
    labels: np.ndarray, n_excluded: int, n_zero_projection: int = 0
) -> Dict[str, Any]:
    """Region/exclusion counts and fractions from an already-computed ``labels`` array
    (region_partition's ``labels`` field) plus the exclusion count. ``n_zero_projection``
    is accepted as an optional pass-through (default 0) so the field can be reported
    alongside the counts without recomputing it from ``proj``, which this function does not
    receive; callers pass ``region_partition``'s own ``n_zero_projection`` value here.

    Counts come from ``np.bincount(labels, minlength=2)`` as plain ints; fractions are
    plain floats. The three counts (``n_region_0``, ``n_region_1``, ``n_excluded``) sum
    exactly to ``n_total``, the original point count.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    n_region_0 = int(counts[0])
    n_region_1 = int(counts[1])
    n_excluded = int(n_excluded)
    n_total = n_region_0 + n_region_1 + n_excluded

    def _frac(x: int) -> float:
        return float(x) / n_total if n_total > 0 else 0.0

    return {
        "n_region_0": n_region_0,
        "n_region_1": n_region_1,
        "n_excluded": n_excluded,
        "n_total": n_total,
        "fraction_region_0": _frac(n_region_0),
        "fraction_region_1": _frac(n_region_1),
        "fraction_excluded": _frac(n_excluded),
        "n_zero_projection": int(n_zero_projection),
    }
