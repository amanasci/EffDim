"""
Pure numpy/scipy probe functions for Phase 02.1's geometry-representation evidence.

Every function takes arrays (or a memmap) and returns arrays or plain dicts -- no file
I/O, no cache handling, no torch. ``notebooks/diagnostics/geometry_probes_run.py`` owns
all cache/path handling, which keeps every function testable against synthetic fixtures
with a known answer (the Wave 0 gap ``02.1-VALIDATION.md`` records).

Constants (``P_LADDER``, ``Q_LADDER``, ``DELTA_QUADRUPLES``, ``DELTA_SEED``, the
distortion statistic, the working-dimension rule, the delta reading rule) live in
``02.1-PREREGISTRATION.md``, not here.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def draw_geo_pairs(rng: np.random.Generator, n: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``count`` off-diagonal (row, col) index pairs into ``[0, n)``, self-pairs
    rejected and redrawn. Reproduces Phase 2's pair-drawing idiom exactly, recorded in
    ``02-PATTERNS.md``'s ``_draw_geo_pairs``: draw both index arrays with
    ``rng.integers(0, n, size=count)``, then while any ``rows == cols``, redraw only the
    offending entries of ``cols``. Bit-compatibility with this idiom is not cosmetic --
    the runner asserts the resulting pairs' geodesic distances are bit-identical to
    Phase 2's cached ``geo_pairs_r2``. Returns ``(rows, cols)``, each shape ``(count,)``."""
    rows = rng.integers(0, n, size=count)
    cols = rng.integers(0, n, size=count)
    self_pairs = rows == cols
    while np.any(self_pairs):
        n_bad = int(self_pairs.sum())
        cols[self_pairs] = rng.integers(0, n, size=n_bad)
        self_pairs = rows == cols
    return rows, cols


def sampled_delta_hyperbolicity(D, n_quadruples: int, seed: int) -> Dict[str, float]:
    """Sampled Gromov four-point relative delta-hyperbolicity.

    RESEARCH.md's ``Don't Hand-Roll`` table names this the one item where hand-rolling is
    the right call: the numerically delicate primitives it warns against (optimal
    transport, Riemannian exponential/logarithmic maps, kernel bandwidth selection) are
    absent here -- this is a plain distance-matrix reduction. Exhaustive enumeration of
    all quadruples is ``C(n, 4)``, intractable at ``n = 10,000``, so quadruples are drawn
    at random and rejected on any repeated index; this project's pre-registered count
    (``DELTA_QUADRUPLES = 200_000`` in ``02.1-PREREGISTRATION.md``) is well above the few
    hundred to few thousand the literature typically samples, since the compute budget
    allows it.

    For each sampled quadruple ``(a, b, c, d)``, the three pair sums
    ``d(a,b)+d(c,d)``, ``d(a,c)+d(b,d)``, ``d(a,d)+d(b,c)`` are sorted and
    ``delta = (largest - second_largest) / 2``, the standard four-point definition. A
    tree metric gives ``delta`` identically zero for every quadruple, which is the Wave 0
    fixture this function is proved against.

    ``D`` may be a memmap (only fancy-indexed reads are performed, never a full-array
    copy). Returns ``delta_max``, ``delta_p95``, ``delta_mean``, ``diam_sample`` (max
    pairwise geodesic distance within the sampled index set), ``delta_rel_max``/
    ``delta_rel_p95`` (``2 * delta / diam_sample`` at max and 95th percentile), and
    ``n_quadruples`` (the realised count).
    """
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    quads = np.empty((n_quadruples, 4), dtype=np.int64)
    filled = 0
    while filled < n_quadruples:
        need = n_quadruples - filled
        cand = rng.integers(0, n, size=(need, 4))
        # vectorised "all four indices distinct" check
        sorted_cand = np.sort(cand, axis=1)
        distinct = np.all(np.diff(sorted_cand, axis=1) != 0, axis=1)
        good = cand[distinct]
        take = min(len(good), need)
        quads[filled:filled + take] = good[:take]
        filled += take
    a, b, c, d = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    delta = (sums[:, 2] - sums[:, 1]) / 2.0

    sample_idx = np.unique(quads)
    diam_sample = float(np.max(D[np.ix_(sample_idx, sample_idx)]))

    delta_max = float(delta.max())
    delta_p95 = float(np.percentile(delta, 95))
    delta_mean = float(delta.mean())

    return {
        "delta_max": delta_max,
        "delta_p95": delta_p95,
        "delta_mean": delta_mean,
        "diam_sample": diam_sample,
        "delta_rel_max": 2.0 * delta_max / diam_sample if diam_sample > 0 else 0.0,
        "delta_rel_p95": 2.0 * delta_p95 / diam_sample if diam_sample > 0 else 0.0,
        "n_quadruples": int(n_quadruples),
    }


def krein_mass_capture(
    eigvals: np.ndarray, p_ladder: Sequence[int], q_ladder: Sequence[int]
) -> List[Dict[str, float]]:
    """Fraction of total absolute eigenvalue mass captured by the top-``p`` positive and
    top-``q`` most-negative eigenvalues, for every ``(p, q)`` rung of the pre-registered
    ladder. Needs only the eigenvalues, not the eigenvectors -- this is the spectral half
    of the dimension question, independently checkable against
    ``pseudo_euclidean_sq_distances``'s distortion numbers (which do need the
    eigenvectors). On an all-non-negative spectrum there is nothing for the negative block
    to capture, so capture at fixed ``p`` is identical across every ``q`` -- the Wave 0
    fixture this is proved against. Returns one dict per ``(p, q)`` rung with keys ``p``,
    ``q``, ``mass_capture``."""
    pos_sorted = np.sort(eigvals[eigvals > 0])[::-1]  # largest positive first
    neg_sorted = np.sort(eigvals[eigvals < 0])         # most negative first
    total_mass = np.abs(eigvals).sum()

    rows = []
    for p in p_ladder:
        pos_take = pos_sorted[: min(p, len(pos_sorted))]
        for q in q_ladder:
            neg_take = neg_sorted[: min(q, len(neg_sorted))]
            captured = np.abs(pos_take).sum() + np.abs(neg_take).sum()
            mass_capture = float(captured / total_mass) if total_mass > 0 else 0.0
            rows.append({"p": int(p), "q": int(q), "mass_capture": mass_capture})
    return rows


def pseudo_euclidean_sq_distances(
    eigvals_sel: np.ndarray, eigvecs_sel: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Reconstruct signed squared distances from a selected set of signed eigenpairs
    (``eigvals_sel`` shape (k,), ``eigvecs_sel`` shape (n, k) column-aligned to it, over
    pairs ``rows``/``cols``) of the double-centred geodesic matrix -- the pseudo-Euclidean
    /Krein generalisation of the classical-MDS reconstruction identity.

    With coordinates ``X = eigvecs_sel * sqrt(abs(eigvals_sel))`` and signs
    ``s = sign(eigvals_sel)``, the reconstructed squared distance for pair ``(i, j)`` is
    ``sum_a s_a * (X[i, a] - X[j, a]) ** 2``. This does **not** take an absolute value or
    a square root: in a pseudo-Euclidean space a squared distance is genuinely allowed to
    be negative, and hiding that behind ``abs`` or ``sqrt`` would erase the property this
    representation exists to express. When ``eigvals_sel``/``eigvecs_sel`` is the *full*
    signed eigendecomposition of a double-centred squared-distance matrix, this identity
    reconstructs the original squared distances exactly -- restricting to the positive
    block only (``q = 0``) drops a subtracted term and biases the reconstruction upward,
    which is exactly what ``q = 0``'s classical-MDS baseline is expected to do on
    indefinite input. Returns shape (m,) signed squared distances; may be negative.
    """
    signs = np.sign(eigvals_sel)
    X = eigvecs_sel * np.sqrt(np.abs(eigvals_sel))[None, :]
    diff = X[rows] - X[cols]
    return (signs[None, :] * diff ** 2).sum(axis=1)


def distortion_stats(d2_rep: np.ndarray, d2_geo: np.ndarray) -> Dict[str, float]:
    """The pre-registered distortion statistic (``02.1-PREREGISTRATION.md``'s
    ``## Distortion Statistic``), computed over squared distances ``d2_rep``
    (reconstructed) vs. ``d2_geo`` (true geodesic, same pair sample). Returns
    ``median_abs_rel`` = ``median(|d2_rep - d2_geo| / d2_geo)``, ``median_signed_rel`` =
    ``median((d2_rep - d2_geo) / d2_geo)`` (reporting only the absolute form would hide
    systematic over/under-estimation inside the median), and ``p95_abs_rel`` = the 95th
    percentile of the absolute relative deviation."""
    rel = (d2_rep - d2_geo) / d2_geo
    return {
        "median_abs_rel": float(np.median(np.abs(rel))),
        "median_signed_rel": float(np.median(rel)),
        "p95_abs_rel": float(np.percentile(np.abs(rel), 95)),
    }


def correction_blindness(eigvals: np.ndarray) -> Dict[str, float]:
    """Demonstrate, rather than cite, that eigenvalue clipping and the Lingoes-type
    additive shift are diagnostic-blind: both force zero negative mass on *any* input,
    curved or not, by construction.

    Clipping sets every negative eigenvalue to zero. The Lingoes-type additive shift adds
    ``abs(min(eigvals))`` to every eigenvalue -- the spectral form of adding a constant to
    squared off-diagonal distances -- which is guaranteed to leave nothing negative.
    Because both corrections succeed on any input by construction, neither can serve as
    evidence the underlying geometry is "basically flat" (RESEARCH.md Pattern 1, Pitfall
    2) -- ``geometry_probes_run.py`` demonstrates this by running this function on the
    real spectrum *and* on synthetic controls and showing the corrected columns are
    identical across all of them. Cailliez's constant-shift correction belongs to the same
    diagnostic-blind family but is not implemented here: its exact constant requires an
    eigenproblem on a ``2n x 2n`` matrix, a 20,000x20,000 solve at ``n = 10,000``, and it
    would demonstrate nothing the two cheaper corrections here do not already demonstrate.

    Returns ``m_before`` (the pre-correction ``m`` statistic:
    ``sum(abs(negative)) / sum(abs(all))``), ``m_after_clip``, ``m_after_shift`` (both
    exactly 0.0 by construction on any input), and ``shift_constant`` (the Lingoes-type
    additive constant used).
    """
    total_mass = np.abs(eigvals).sum()
    neg_mass = np.abs(eigvals[eigvals < 0]).sum()
    m_before = float(neg_mass / total_mass) if total_mass > 0 else 0.0

    clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    clipped_total = np.abs(clipped).sum()
    clipped_neg = np.abs(clipped[clipped < 0]).sum()
    m_after_clip = float(clipped_neg / clipped_total) if clipped_total > 0 else 0.0

    shift_constant = float(abs(eigvals.min())) if eigvals.min() < 0 else 0.0
    shifted = eigvals + shift_constant
    shifted_total = np.abs(shifted).sum()
    shifted_neg = np.abs(shifted[shifted < 0]).sum()
    m_after_shift = float(shifted_neg / shifted_total) if shifted_total > 0 else 0.0

    return {
        "m_before": m_before,
        "m_after_clip": m_after_clip,
        "m_after_shift": m_after_shift,
        "shift_constant": shift_constant,
    }


def kneedle_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Maximum-curvature (kneedle) elbow index over curve ``(x, y)``, implementing
    ``gate_verdict_43cf438bc944c509.json``'s ``elbow_criterion`` verbatim:

    "Maximum-curvature (kneedle) elbow on the Tenenbaum residual-variance curve (1 - R^2
    between geodesic and embedded pairwise distances): both axes are normalized to [0, 1]
    by their own range, and the elbow is the point of greatest perpendicular distance from
    the chord connecting the curve's first and last normalized point. ... ties broken to
    the lower d (ELBOW_TIE_BREAK='lower')."

    Range-normalising both axes to ``[0, 1]`` first makes the result invariant to any
    affine rescaling of either axis -- the same normalisation is why a perfectly straight
    line makes every point equidistant from the chord (distance identically zero), at
    which point the lower-index tie-break fires and the first point is returned. Returns
    the index into ``x``/``y`` of the elbow point.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_norm = (x - x.min()) / x_range if x_range > 0 else np.zeros_like(x)
    y_norm = (y - y.min()) / y_range if y_range > 0 else np.zeros_like(y)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    chord = p2 - p1
    chord_len = np.linalg.norm(chord)

    if chord_len == 0:
        return 0

    points = np.stack([x_norm, y_norm], axis=1)
    diffs = points - p1
    # perpendicular distance from each point to the chord line
    cross = diffs[:, 0] * chord[1] - diffs[:, 1] * chord[0]
    distances = np.abs(cross) / chord_len

    max_dist = distances.max()
    candidates = np.flatnonzero(np.isclose(distances, max_dist, atol=1e-12))
    return int(candidates.min())
