# Top-level definitions removed from notebooks/pu_manifold/cross_split_curvature.py in the paper-closure
# Stage 5: not reachable from any kept runner, generator or notebook. Verbatim, original order.


# --- removed from cross_split_curvature.py:74-84 ---
INDEPENDENCE_MODES = ("disjoint_data", "seed_only")
"""What made the two arms independent.

``"disjoint_data"``  -- arms trained on disjoint sample halves. Cancels sampling AND
                        optimisation noise. The faithful analogue of the source branch's
                        split-half neighbourhood.
``"seed_only"``      -- arms share a training sample, differing only in initialisation /
                        optimisation seed. Cancels optimisation noise only. ``R_H`` from
                        such a pair is an UPPER BOUND on reliability and must never be
                        reported as if it were a split-half number.
"""


# --- removed from cross_split_curvature.py:87-91 ---
def _as_2d_float64(H: Any, name: str) -> np.ndarray:
    arr = np.asarray(H, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (n, D) mean curvature vectors; got shape {arr.shape}.")
    return arr


# --- removed from cross_split_curvature.py:94-137 ---
def tensor_agreement(H_A: Any, H_B: Any) -> Dict[str, np.ndarray]:
    """Row-wise port of the source branch's ``tensor_agreement``.

    ``H_A``, ``H_B``: ``(n, D)`` mean curvature vectors for the SAME ``n`` points, in the
    same ambient frame. Returns per-row arrays:

      ``norm_A``, ``norm_B``  ``||H^(A)||``, ``||H^(B)||``
      ``inner``               ``<H^(A), H^(B)>`` -- this IS ``K_H_cross``
      ``r_dir``               cosine similarity, ``inner / (norm_A * norm_B)``
      ``R_signal``            ``2 * inner / (norm_A^2 + norm_B^2)`` -- the reliability ratio

    **Reading ``R_signal``.** It is a per-point signal-to-total ratio, not a correlation: it
    is bounded above by 1 (attained iff ``H^(A) == H^(B)``), reaches 0 when the two arms are
    orthogonal, and goes negative when they point in opposing directions -- which is the
    diagnostic case, meaning the two fits disagree on the SIGN of the curvature and no
    amount of averaging will rescue the field. It differs from ``r_dir`` by penalising
    magnitude disagreement as well as direction; two arms that agree perfectly in direction
    but differ 10x in scale score ``r_dir = 1`` and ``R_signal = 0.198``.

    The ambient frame requirement is not a formality. Two independently trained chart
    auto-encoders assign different charts to the same point and use different chart
    coordinates, so their ``H`` agree only because ``chart_curvature.chart_mean_curvature``
    returns ``tr_g(II)`` projected into AMBIENT coordinates. Comparing chart-space
    quantities across two such models would be meaningless.
    """
    a = _as_2d_float64(H_A, "H_A")
    b = _as_2d_float64(H_B, "H_B")
    if a.shape != b.shape:
        raise ValueError(
            f"H_A and H_B must have identical shape -- the two arms must be evaluated at the "
            f"same points, in the same order. Got {a.shape} and {b.shape}."
        )
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    inner = np.einsum("ij,ij->i", a, b)
    r_dir = inner / np.maximum(na * nb, EPS)
    R = (2.0 * inner) / np.maximum(na**2 + nb**2, EPS)
    return {
        "norm_A": na,
        "norm_B": nb,
        "inner": inner,
        "r_dir": r_dir,
        "R_signal": R,
    }


# --- removed from cross_split_curvature.py:140-187 ---
def cross_curvature_field(
    H_A: Any,
    H_B: Any,
    independence: str,
    single_split_arm: str = "A",
) -> Dict[str, Any]:
    """The per-point cross statistics, plus the single-split quantity they replace, so a
    caller can score BOTH on one field and report the difference rather than swapping one
    number for another and asserting an improvement.

    ``independence``: one of :data:`INDEPENDENCE_MODES`. Required, not defaulted -- see the
    module docstring; a ``"seed_only"`` pair produces a real ``R_H`` that means something
    weaker than a ``"disjoint_data"`` one, and the distinction is invisible in the numbers.

    ``single_split_arm``: which arm supplies the legacy ``||H_est||`` baseline, ``"A"`` or
    ``"B"``. Both are recorded; this only picks which one the convenience key points at.

    Returns per-point arrays ``K_H_cross``, ``R_H``, ``r_dir``, ``norm_H_A``, ``norm_H_B``,
    ``norm_H_mean``, and ``h_norm_single`` (the legacy statistic), plus the ``independence``
    tag and ``curvature_convention`` for provenance.

    ``K_H_cross`` is SIGNED and is meant to stay signed. The source branch carries a
    separate ``K_H_cross_plot = max(K_H_cross, 0)`` for figures only and ranks on the signed
    value; clipping before a rank statistic would collapse every disagreeing point into a
    tie and manufacture rank agreement out of estimator failure.
    """
    if independence not in INDEPENDENCE_MODES:
        raise ValueError(
            f"independence must be one of {INDEPENDENCE_MODES}; got {independence!r}. "
            f"This is not a defaultable argument -- see the module docstring."
        )
    if single_split_arm not in ("A", "B"):
        raise ValueError(f"single_split_arm must be 'A' or 'B'; got {single_split_arm!r}.")
    agree = tensor_agreement(H_A, H_B)
    h_single = agree["norm_A"] if single_split_arm == "A" else agree["norm_B"]
    return {
        "K_H_cross": agree["inner"],
        "R_H": agree["R_signal"],
        "r_dir": agree["r_dir"],
        "norm_H_A": agree["norm_A"],
        "norm_H_B": agree["norm_B"],
        "norm_H_mean": 0.5 * (agree["norm_A"] + agree["norm_B"]),
        "h_norm_single": h_single,
        "single_split_arm": single_split_arm,
        "independence": independence,
        "curvature_convention": CURVATURE_CONVENTION,
        "n_points": int(agree["inner"].shape[0]),
    }


# --- removed from cross_split_curvature.py:190-229 ---
def reliability_summary(
    R_H: Any, threshold: float, min_fraction: float = 0.5
) -> Dict[str, Any]:
    """Aggregate a per-point ``R_H`` field into the admissibility verdict the source branch
    applies per neighbourhood scale ("``k=512`` fails ``R_H`` reliability and is not
    confirmatory").

    ``threshold``: the ``R_H`` a point must exceed to count as reliably estimated.
    ``min_fraction``: the fraction of points that must clear ``threshold`` for the FIELD to
    be admissible.

    **Both bounds are the caller's to declare and neither has a defensible default**, which
    is why ``threshold`` is required. The source branch's own cutoff is not transferable: it
    was set against per-anchor split-half local quadratic fits on unit-normalised ViT
    embeddings, and nothing establishes that the same number separates signal from noise for
    a global chart auto-encoder on a different manifold. Declare it before looking at the
    field, or the gate is a post-hoc rationalisation of whatever was measured.

    Returns ``median_R_H``, ``mean_R_H``, ``fraction_above``, ``n_above``, ``n_points``,
    ``fraction_negative`` (points where the two arms disagree on sign -- the pure-failure
    diagnostic), the echoed bounds, and ``admissible``.
    """
    r = np.asarray(R_H, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("reliability_summary over an empty field is not a measurement.")
    above = r > threshold
    frac_above = float(above.mean())
    return {
        "median_R_H": float(np.median(r)),
        "mean_R_H": float(r.mean()),
        "p05_R_H": float(np.percentile(r, 5)),
        "p95_R_H": float(np.percentile(r, 95)),
        "fraction_above": frac_above,
        "n_above": int(above.sum()),
        "n_points": int(r.size),
        "fraction_negative": float((r < 0.0).mean()),
        "threshold": float(threshold),
        "min_fraction": float(min_fraction),
        "admissible": bool(frac_above >= min_fraction),
    }
