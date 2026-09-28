"""Pure numpy/scipy probe functions for Phase 02.1's geometry evidence.

Arrays in, arrays/dicts out — no file I/O, no cache handling, no torch; the runner
(diagnostics/geometry_probes_run.py) owns paths. Constants live in
02.1-PREREGISTRATION.md.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def draw_geo_pairs(rng: np.random.Generator, n: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``count`` off-diagonal (row, col) pairs into [0, n), self-pairs redrawn.

    Reproduces Phase 2's draw idiom exactly — the runner asserts bit-identity against
    the cached ``geo_pairs_r2``, so do not "improve" the redraw scheme."""
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

    Sampling is the method, not a shortcut — exhaustive C(n, 4) is intractable at
    n=10,000. Per quadruple: sort the three pair sums, delta = (largest - second)/2;
    a tree metric gives delta identically zero (the Wave 0 fixture). ``D`` may be a
    memmap (fancy-indexed reads only). Returns delta_max/p95/mean, diam_sample, and
    delta_rel_* = 2*delta/diam_sample."""
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
    """Fraction of total |eigenvalue| mass captured by the top-p positive and top-q
    most-negative eigenvalues, per (p, q) rung. Values only, no eigenvectors. Returns
    one dict per rung: p, q, mass_capture."""
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
    """Signed squared distances from selected signed eigenpairs — the Krein
    generalisation of the classical-MDS reconstruction identity.

    X = eigvecs * sqrt(|eigvals|), s = sign(eigvals); d2(i,j) = sum_a s_a (X_ia - X_ja)^2.
    Deliberately no abs, no sqrt: a pseudo-Euclidean squared distance may be negative,
    and hiding that erases the property the representation exists to express. Exact on a
    full signed eigendecomposition; positive-only (q=0) is biased upward."""
    signs = np.sign(eigvals_sel)
    X = eigvecs_sel * np.sqrt(np.abs(eigvals_sel))[None, :]
    diff = X[rows] - X[cols]
    return (signs[None, :] * diff ** 2).sum(axis=1)


def distortion_stats(d2_rep: np.ndarray, d2_geo: np.ndarray) -> Dict[str, float]:
    """Pre-registered distortion statistic: median_abs_rel, median_signed_rel (the
    signed form exposes systematic bias the absolute median hides), p95_abs_rel."""
    rel = (d2_rep - d2_geo) / d2_geo
    return {
        "median_abs_rel": float(np.median(np.abs(rel))),
        "median_signed_rel": float(np.median(rel)),
        "p95_abs_rel": float(np.percentile(np.abs(rel), 95)),
    }


def correction_blindness(eigvals: np.ndarray) -> Dict[str, float]:
    """m before and after eigenvalue clipping and the Lingoes-type additive shift
    (add |min|). Both force zero negative mass on ANY input by construction — that
    blindness is the point being demonstrated. Cailliez omitted: needs a 2n x 2n solve
    and demonstrates nothing more. Returns m_before, m_after_clip, m_after_shift,
    shift_constant."""
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
    """Kneedle elbow index, implementing the gate verdict's elbow_criterion verbatim:
    both axes range-normalised to [0, 1], elbow = greatest perpendicular distance from
    the first-to-last chord, ties to the lower index. Affine-rescaling invariant; a
    straight line ties everywhere and returns index 0."""
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
