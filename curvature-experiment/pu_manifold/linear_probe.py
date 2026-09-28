"""Phase 5 curvature-conditioned linear decodability: probe fit/score, seed pooling, bucketing
and verdict functions, plus the pre-registration constants block and its guard.

**Closure note.** Only ``fit_probe`` and ``predict_probe`` (wrapped by
``physics_curvature_probe.oof_ridge_predictions``) remain in this file; the Phase 5 constants
block, verdict rules and the other functions described below are archived verbatim in
``archive/pu_manifold_trimmed/linear_probe.py``.

(a) **D5-03's own citation is corrected here, not silently followed.** D5-03 names
``pu_manifold/decoder_curvature.py`` as the decoder-side curvature source. That
module's own docstring states it is ``chart_curvature.py`` with the
``chart_decoders[chart_idx]`` two-hop composition removed, built for Phase 02.6's
no-chart-index substrates (a plain autoencoder and a ``PlainAutoEncoder`` trained under
``topoae.train_topoae``, both decoding through one smooth MLP with no chart index at all).
``ChartAutoEncoder`` is chart-routed and has no bare single-hop decode entry point matching
that module's signature. The function this phase actually uses, matching the CAE Phase 3
built, is ``chart_curvature.chart_curvature_field(model, x, mode="reverse")`` -- frozen as
the named constant ``CURVATURE_SOURCE_FUNCTION`` (unset here, filled at the 05-04 freeze) so
the correction is auditable rather than buried in a comment. This module imports nothing
from ``decoder_curvature``, and the runner does not either.

(b) **The pooled design is REMOVED from this module, not merely left unused.**
``05-CONTEXT.md`` D5-04 said to pool the three cached CAE seeds into one averaged ``||H||``
field. That was put to the developer at the ``05-03`` Task 1 blocking checkpoint and REJECTED
-- one-way, per ``05-03-DECISION.md``, which SUPERSEDES D5-04. The evidence: measured at
``05-02`` over all 10,000 PU points, the three seeds' fields are mutually anti-correlated on
rank (pairwise Spearman on ``H_norm`` -0.1402, +0.2019, -0.2725 -- sign-inconsistent, two of
three negative) and directionally orthogonal (median cosine of unit ``H_vec`` 0.0007 to 0.0039,
with 46 to 48 percent of points anti-aligned between any pair), with seeds 20260814 and
20260815 taking 4 and 3 effective distinct levels (see the correction below) at a metric
determinant around ``1e-166``, roughly 100 orders of magnitude from seed 20260813's continuous
field. Any pooled field would not be a consensus: it would be seed 20260813's structure plus
two step-like functions that disagree with it and with each other. Rejected alongside the raw
mean: per-seed median-divide then average (``05-RESEARCH.md``'s own recommendation), per-seed
percentile-rank then average, and halting the phase. :func:`pool_seed_fields` is RETAINED as
tested but unused code, per CLAUDE.md's additive-only rule -- Phase 5 calls it nowhere.

**Correction to 05-02-SUMMARY.md.** That summary reported seeds 20260814/15 as "not literally
piecewise-constant -- 5,301 / 9,852 exact distinct ``H_norm`` values (not 3-4)". That claim is
WRONG: those counts are float noise in the last ULPs. Measured directly from the cached fields
at RELATIVE precision, seed 20260814 has 4 distinct levels and seed 20260815 has 3, stable from
rel 1e-9 through rel 1e-3. ``05-RESEARCH.md`` Pitfall 2 and ``03-09-SUMMARY.md``'s original
measurement were both correct.

(c) **D5-11's accepted gap.** The field this phase splits on has no demonstrated relationship
to true curvature: the sealed ``d=20`` decoder row is
``rank_spearman_rho = -0.015106571347065712`` against the only analytic-curvature control
that tests it (spike-findings-effdim, ``high-d-curvature-feasibility.md``), and direction is
near a coin flip (52-75 percent of points anti-aligned). A Swiss roll / low-``d`` anchor was
offered and declined for this phase. Any relationship Phase 5 measures between ``||H||`` and
probe residual therefore cannot be attributed to curvature by anything in this phase --
stated here, in this module's own words, and not only by cross-reference.

(d) **D5-12's inherited chain.** The CAE underlying every field this module consumes failed
its own validity gate (``CAE_VERDICT = FAIL``, Phase 02.2); Phase 3 ran on a deliberate
override of that gate; Phase 03.1 found the pullback metric fully repaired by a ``scale``
prior while the curvature ordering only partially and non-seed-consistently moved. Every
number this module's functions eventually help produce inherits that chain.

No file I/O happens in this module -- every function is pure, operates on arrays the caller
already has in memory, and every pre-registered value is passed in by the caller with no
default (mirroring ``region_partition.py``'s own stated convention: a default is how a
pre-registered value gets inherited by accident instead of by an explicit call-site choice).
"""

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import RidgeCV


# --- D5-01/D5-02: the probe itself -----------------------------------------------------------


def fit_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha_grid: Any,
    alpha_per_target: bool,
    fit_intercept: bool,
) -> Dict[str, Any]:
    """Wraps ``sklearn.linear_model.RidgeCV(alphas=alpha_grid,
    alpha_per_target=alpha_per_target, fit_intercept=fit_intercept)``. Never hand-rolls a CV
    loop or a least-squares solver. Returns a flat dict carrying the fitted estimator and
    everything a caller or test might want to inspect without refitting.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError(f"fit_probe: X_train must be two-dimensional, got shape {X_train.shape}.")
    if Y_train.ndim != 2:
        raise ValueError(f"fit_probe: Y_train must be two-dimensional, got shape {Y_train.shape}.")
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"fit_probe: X_train has {X_train.shape[0]} rows but Y_train has "
            f"{Y_train.shape[0]} rows."
        )
    if not np.all(np.isfinite(X_train)):
        raise ValueError("fit_probe: X_train contains a non-finite value.")
    if not np.all(np.isfinite(Y_train)):
        raise ValueError("fit_probe: Y_train contains a non-finite value.")
    alpha_grid = tuple(float(a) for a in alpha_grid)
    if len(alpha_grid) == 0:
        raise ValueError("fit_probe: alpha_grid must be non-empty.")

    estimator = RidgeCV(
        alphas=alpha_grid, alpha_per_target=alpha_per_target, fit_intercept=fit_intercept
    )
    estimator.fit(X_train, Y_train)

    return {
        "estimator": estimator,
        "coef_shape": tuple(np.asarray(estimator.coef_).shape),
        "intercept_shape": tuple(np.asarray(estimator.intercept_).shape),
        "alpha_": estimator.alpha_,
        "alpha_grid": alpha_grid,
        "alpha_per_target": bool(alpha_per_target),
        "fit_intercept": bool(fit_intercept),
        "n_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_targets": int(Y_train.shape[1]),
    }


def predict_probe(fit: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """``fit["estimator"].predict(X)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"predict_probe: X must be two-dimensional, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("predict_probe: X contains a non-finite value.")
    return fit["estimator"].predict(X)
