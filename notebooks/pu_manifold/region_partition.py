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
