"""Split-half cross statistics for a mean-curvature field, ported from the
``curvature-experiments`` branch so the two research lines share one estimator.

**Where this comes from.** ``curvature-experiments`` (fork point ``7b2401e``) scores its
local-quadratic charts with a CROSS statistic rather than a single-split point estimate:
two independent fits ``A`` and ``B`` of the same neighbourhood produce ``H^(A)``, ``H^(B)``,
and the reported quantity is their inner product

    ``K_H_cross = <H^(A), H^(B)>``

alongside a reliability ratio

    ``R_H = 2 <H^(A), H^(B)> / (||H^(A)||^2 + ||H^(B)||^2)``

which that branch uses as an ADMISSIBILITY GATE -- its ``k=512`` neighbourhood scale is
excluded from confirmatory analysis because it "fails ``R_H`` reliability". Sources:
``experiments/geometry/physics_activation_atlas/effdim_curvature_metrics.py``
(``cross_metric_pair``) and ``.../split_half_curvature_reliability.py``
(``tensor_agreement``).

**Why we want it here.** Everything this milestone has scored curvature with is a
single-split ``||H_est||`` compared against a truth by Spearman rank. The norm of a noisy
vector is POSITIVELY BIASED: ``E||H_hat|| >= ||E H_hat||``, with the gap growing as the
estimator degrades. At ``d = 20`` the local-polynomial teacher's median magnitude ratio was
measured at 224 (spike 002, ``k=231``), so ``||H_est||`` there is dominated by noise
magnitude, and a rank correlation against truth is being computed on a quantity that is
mostly not curvature. The cross statistic removes exactly that term in expectation: for
independent, zero-mean estimation errors ``e_A``, ``e_B``,

    ``E<H + e_A, H + e_B> = ||H||^2``

so the noise contributes no positive offset, only variance. This is the single most
transferable idea on that branch and it costs one extra fit.

**Scope, deliberately narrow (CLAUDE.md: keep things simple first).** The source module also
computes ``K_aniso`` and ``K_dir`` from the traceless part ``B0`` of the full second
fundamental form. ``chart_curvature.chart_mean_curvature`` returns only ``H_vec``, never
``II`` itself, so those two statistics are NOT ported -- adding them means changing what the
sealed curvature path returns, which is a larger change than this module is entitled to
make. Only the ``H``-based statistics are here. If ``II`` is ever exposed, the missing
algebra is ``aniso_prefactor(d) = 2 / (d * (d + 2))`` applied to ``sum(B0_A * B0_B)``.

**A difference from the source that matters when reading results.** On that branch, ``A``
and ``B`` are two halves of ONE anchor's neighbourhood, so the cross statistic is a scalar
per anchor. Here the two arms are two independently fitted GLOBAL models evaluated at the
same ambient points, so the statistic is a field with one value per point. The estimator
algebra is identical; what counts as "independent" is not. Two models differing only in
initialisation seed share their training sample and therefore share sampling noise -- such a
pair cancels optimisation noise ONLY, and ``R_H`` from it is an optimistic bound on
reliability. A pair trained on disjoint data halves cancels both. Callers must record which
they used; :func:`cross_curvature_field` takes ``independence`` for exactly that reason and
refuses to guess.

Pure numpy, no module-level ``torch`` -- same posture as ``curvature_probe.py``, so this is
importable and testable without a torch install. Callers holding
``chart_curvature.chart_curvature_field`` output pass ``field["H_vec"].detach().cpu().numpy()``.
"""

from typing import Any, Dict, Optional

import numpy as np

EPS = 1e-12

CURVATURE_CONVENTION = "trace"
"""Drift guard. ``H`` arriving here must be in the same convention the rest of this package
uses (``chart_curvature.CURVATURE_CONVENTION``). Nothing in this module rescales ``H``, and
every statistic below is either a bilinear form in ``H`` or a ratio of such forms, so a
convention mismatch between the two arms would corrupt ``K_H_cross`` silently. The source
branch normalises its ``H`` by ``1/d`` where this package does not; that constant cancels in
``R_H`` and rescales ``K_H_cross`` by ``1/d^2``, which is invisible to any rank statistic at
fixed ``d`` and NOT invisible when comparing across ``d``."""

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


def _as_2d_float64(H: Any, name: str) -> np.ndarray:
    arr = np.asarray(H, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (n, D) mean curvature vectors; got shape {arr.shape}.")
    return arr


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


def partial_spearman(
    x: Any, y: Any, controls: Optional[Any] = None
) -> float:
    """Spearman rank correlation between ``x`` and ``y``, optionally controlling for
    ``controls`` -- the source branch's "controlled ρ", which is a partial correlation taken
    in RANK SPACE (its ``report.py``: "Controls: ``log_knn_radius``,
    ``local_label_variance``, ``local_evaluation_count``. Partial Spearman on ranks.").

    ``controls``: ``(n, c)`` array of covariates, or ``None`` for the raw statistic. Each of
    ``x``, ``y`` and every control column is rank-transformed, then ``x`` and ``y`` are each
    residualised against the rank-transformed controls (with intercept) by least squares,
    and the Pearson correlation of the residuals is returned.

    That procedure is what makes the source branch's ``-0.240`` at ``d=16`` differ from its
    raw ``-0.412``: over half the raw association there is carried by the covariates. Any
    comparison against their numbers has to use the same transform or it is comparing
    different statistics.

    ``scipy`` is imported inside the function so the rest of this module stays importable in
    environments where only numpy is present.
    """
    from scipy.stats import rankdata

    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.shape != yv.shape:
        raise ValueError(f"x and y must have the same length; got {xv.shape} and {yv.shape}.")
    if xv.size < 3:
        raise ValueError("A rank correlation over fewer than three points is not a measurement.")

    rx = rankdata(xv)
    ry = rankdata(yv)
    if controls is None:
        return float(np.corrcoef(rx, ry)[0, 1])

    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if C.shape[0] != xv.size:
        raise ValueError(
            f"controls must have one row per point; got {C.shape[0]} rows for {xv.size} points."
        )
    RC = np.column_stack([rankdata(C[:, j]) for j in range(C.shape[1])])
    design = np.column_stack([np.ones(xv.size), RC])

    def residual(v: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ coef

    ex = residual(rx)
    ey = residual(ry)
    if np.std(ex) < EPS or np.std(ey) < EPS:
        raise ValueError(
            "A control set that explains all variance in x or y leaves no residual to "
            "correlate; the partial statistic is undefined here."
        )
    return float(np.corrcoef(ex, ey)[0, 1])
