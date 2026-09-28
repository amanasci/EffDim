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

from typing import Any, Optional

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
