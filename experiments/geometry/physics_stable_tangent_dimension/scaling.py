"""Cross-scale subspace tracking and eigenvalue scaling exponents.

Do not regress the j-th sorted eigenvalue independently when branches cross.
Align eigengap-defined blocks by projector overlap, then fit log λ vs log r.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .nested_pca import block_agreement, degenerate_blocks, prefix_agreement
from .sphere_coords import EPS

SCALE_LABELS = (
    "tangent_like",
    "curvature_normal_like",
    "scale_independent_thickness",
    "mixed_or_crossing",
    "unresolved",
)


def align_blocks_across_scales(
    Js: list[np.ndarray],
    evs: list[np.ndarray],
    *,
    overlap_min: float = 0.35,
) -> list[dict[str, Any]]:
    """Track blocks from the smallest scale outward.

    Returns a list of tracks, each with `slots` (scale index → (a,b)) and
    `energy` (sum of eigenvalues in the aligned block).
    """
    if not Js:
        return []
    blocks0 = degenerate_blocks(evs[0])
    tracks = []
    for a, b in blocks0:
        tracks.append(
            {
                "slots": {0: (a, b)},
                "energy": {0: float(np.sum(evs[0][a : b + 1]))},
                "width": b - a + 1,
            }
        )
    for s in range(1, len(Js)):
        used = set()
        blocks = degenerate_blocks(evs[s])
        for tr in tracks:
            prev_s = max(tr["slots"])
            a0, b0 = tr["slots"][prev_s]
            Jprev = Js[prev_s]
            best, best_ov = None, -1.0
            for a, b in blocks:
                if (a, b) in used:
                    continue
                if (b - a + 1) != tr["width"]:
                    # still allow if overlap is high and widths differ by 1
                    if abs((b - a + 1) - tr["width"]) > 1:
                        continue
                ov = block_agreement(Jprev, Js[s], a0, min(b0, Jprev.shape[1] - 1))
                # compare previous block to this block
                w = min(b0 - a0 + 1, b - a + 1, Jprev.shape[1] - a0, Js[s].shape[1] - a)
                if w <= 0:
                    continue
                M = Jprev[:, a0 : a0 + w].T @ Js[s][:, a : a + w]
                ov = float(np.sum(M * M) / w)
                if ov > best_ov:
                    best_ov, best = ov, (a, b)
            if best is None or best_ov < overlap_min:
                continue
            used.add(best)
            tr["slots"][s] = best
            a, b = best
            tr["energy"][s] = float(np.sum(evs[s][a : b + 1]))
            tr.setdefault("overlap", {})[s] = best_ov
    return tracks


def loglog_slope(
    radii: np.ndarray,
    energy: np.ndarray,
    *,
    min_points: int = 4,
) -> dict[str, float]:
    """Robust log-log slope α = d log E / d log r. Theil–Sen if possible."""
    r = np.asarray(radii, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    m = np.isfinite(r) & np.isfinite(e) & (r > EPS) & (e > EPS)
    if int(m.sum()) < min_points:
        return {
            "alpha": float("nan"),
            "alpha_lo": float("nan"),
            "alpha_hi": float("nan"),
            "n": int(m.sum()),
            "resolved": False,
            "leverage_max": float("nan"),
            "r_span_log": float("nan"),
        }
    x = np.log(r[m])
    y = np.log(e[m])
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) < 1e-9:
                continue
            slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return {
            "alpha": float("nan"),
            "alpha_lo": float("nan"),
            "alpha_hi": float("nan"),
            "n": n,
            "resolved": False,
            "leverage_max": float("nan"),
            "r_span_log": float(x.max() - x.min()),
        }
    slopes = np.asarray(slopes)
    alpha = float(np.median(slopes))
    lo, hi = float(np.quantile(slopes, 0.16)), float(np.quantile(slopes, 0.84))
    # hat-matrix leverage of OLS for radius dominance
    X = np.column_stack([np.ones(n), x])
    try:
        xtx = X.T @ X
        H = X @ np.linalg.solve(xtx, X.T)
        lev = float(np.max(np.diag(H)))
    except np.linalg.LinAlgError:
        lev = float("nan")
    span = float(x.max() - x.min())
    # one scale dominates if max leverage >> 2/n
    dominated = np.isfinite(lev) and lev > max(0.7, 4.0 / n)
    resolved = (n >= min_points) and (span >= np.log(1.8)) and (not dominated)
    return {
        "alpha": alpha,
        "alpha_lo": lo,
        "alpha_hi": hi,
        "n": n,
        "resolved": bool(resolved),
        "leverage_max": lev,
        "r_span_log": span,
    }


def classify_scaling(
    alpha: float,
    *,
    resolved: bool,
    tangent_lo: float,
    tangent_hi: float,
    curve_lo: float,
    curve_hi: float,
    thick_lo: float,
    thick_hi: float,
) -> str:
    if (not resolved) or (not np.isfinite(alpha)):
        return "unresolved"
    hits = []
    if tangent_lo <= alpha <= tangent_hi:
        hits.append("tangent_like")
    if curve_lo <= alpha <= curve_hi:
        hits.append("curvature_normal_like")
    if thick_lo <= alpha <= thick_hi:
        hits.append("scale_independent_thickness")
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "mixed_or_crossing"
    return "mixed_or_crossing"


def persistence_score(overlaps: list[float]) -> float:
    x = np.asarray(overlaps, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")
