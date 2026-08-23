"""Evaluate a fixture's analytic curvature AT THE POINT A DECODER ACTUALLY PLACED, rather
than at the input point it was handed.

**The defect this addresses.** Every rank correlation this milestone has computed for a
decoder arm pairs

    ``H_est(i)``  = curvature of the LEARNED manifold at ``F_a(z_chart(x_i))``
    ``H_true(i)`` = curvature of the TRUE manifold at ``x_i``

and those are the same location only when reconstruction is exact. ``F_a(z_chart(x_i))`` is
the reconstruction, so the two differ by the reconstruction error. On a fixture whose
``||H||`` varies over the domain, a displaced evaluation point has a genuinely different
true curvature, so the score charges the estimator for drift that is not a curvature error
at all. ``chart_curvature.curvature_fidelity_report``'s docstring already warns that
reconstruction quality cannot VALIDATE a curvature estimate; this module is about the
converse -- reconstruction error silently CONTAMINATING one.

Scale of the concern, from ``notebooks/.cache/03_synthetic_controls.jsonl``: the sealed
``d=4`` saddle fit reconstructs at ``mse_per_dim = 5.1e-07`` and scores ``rho = +0.989``,
while the ``d=20`` saddle fit reconstructs at ``1.6e-02`` -- four orders of magnitude worse
-- and scores ``rho = -0.015``. Whether any of that collapse is drift rather than curvature
failure has never been measured. This module makes it measurable.

**What it does not do.** It does not find the nearest point on the true manifold in the
Euclidean sense. For every fixture here the true surface is a graph or an explicit
parameterisation, so a decoded ambient point is mapped back to the PARAMETER that generates
it and the analytic curvature is evaluated there. That is exact when the reconstruction lies
on the true surface and is the natural projection when it does not. A Euclidean nearest-point
projection would be a different (and for these fixtures, unnecessary) choice.

**Both scores are meant to be reported, never one instead of the other.** They answer
different questions -- "did the decoder get curvature right at the point you handed it"
versus "did it get curvature right where it actually put the surface" -- and the gap between
them is itself the diagnostic.

Pure numpy; no module-level ``torch``, same posture as ``curvature_probe.py``. Nothing in
this module edits or re-derives any sealed curvature mathematics: the Swiss roll magnitude
comes from ``curvature_probe.swiss_roll_analytic_H_scaled`` and the graph-control curvature
from ``curvature_probe.graph_mean_curvature``, both called unmodified.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.datasets import make_swiss_roll

from . import curvature_probe

CURVATURE_CONVENTION = "trace"
"""Drift guard, matching ``chart_curvature.CURVATURE_CONVENTION``. Every value this module
returns comes from a sealed helper in that convention and is never rescaled here."""


# --- Swiss roll --------------------------------------------------------------------------


def swiss_roll_preprocessing(n: int, seed: int) -> Dict[str, Any]:
    """The centring offset and scale ``curvature_probe.make_swiss_roll_fixture`` applied,
    recovered by regenerating its raw cloud.

    The fixture returns ``global_std`` but not the mean, and inverting the preprocessing
    needs both. ``make_swiss_roll`` is deterministic in ``(n_samples, random_state)`` with
    ``noise=0.0``, so regenerating costs nothing and cannot drift from the fixture -- both
    call the same generator with the same arguments.
    """
    X_raw, t = make_swiss_roll(n_samples=n, noise=0.0, random_state=seed)
    return {
        "mean": X_raw.mean(axis=0),
        "global_std": float(X_raw.std()),
        "t": t,
        "X_raw": X_raw,
    }


def swiss_roll_t_at(P_scaled: Any, n: int, seed: int) -> np.ndarray:
    """The roll's arc-length parameter ``t`` at arbitrary points ``P_scaled`` ``(m, 3)`` given
    in the fixture's preprocessed coordinates.

    ``sklearn.datasets.make_swiss_roll`` parametrizes the surface as
    ``X(t, y) = (t*cos(t), y, t*sin(t))`` with ``t`` drawn from ``[1.5*pi, 4.5*pi]``, so ``t``
    is strictly positive and

        ``x^2 + z^2 = t^2 (cos^2 t + sin^2 t) = t^2``   =>   ``t = sqrt(x^2 + z^2)``.

    The inversion is therefore exact and closed-form -- no optimisation, no nearest-neighbour
    search, no branch ambiguity from the spiral wrapping, because the radius alone determines
    ``t`` on this parameter range. The ``y`` coordinate is the ruling direction and carries no
    curvature, so it is not needed.

    Points are un-preprocessed first (``P_raw = P_scaled * global_std + mean``) because the
    identity above holds in the generator's raw coordinates.
    """
    P = np.asarray(P_scaled, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P_scaled must be (m, 3) Swiss roll points; got {P.shape}.")
    pre = swiss_roll_preprocessing(n, seed)
    P_raw = P * pre["global_std"] + pre["mean"]
    return np.sqrt(P_raw[:, 0] ** 2 + P_raw[:, 2] ** 2)


def swiss_roll_truth_at(P_scaled: Any, n: int, seed: int) -> np.ndarray:
    """``||H||`` of the TRUE Swiss roll at the points ``P_scaled``, in the fixture's scaled
    units -- directly comparable to what ``chart_curvature.chart_curvature_field`` returns.

    Delegates the curvature itself to the sealed
    ``curvature_probe.swiss_roll_analytic_H_scaled``; this function only supplies the ``t`` to
    evaluate it at.
    """
    pre = swiss_roll_preprocessing(n, seed)
    t_hat = swiss_roll_t_at(P_scaled, n, seed)
    return curvature_probe.swiss_roll_analytic_H_scaled(t_hat, pre["global_std"])


# --- graph-of-function controls (flat / saddle) ------------------------------------------


def _rotation(D: int, seed: int) -> np.ndarray:
    """``synthetic_controls.rotate_and_pad``'s fixed orthogonal ``Q``, rebuilt from the same
    ``(D, seed)``. Kept byte-identical to that function's own construction -- a QR of one
    ``default_rng(seed).standard_normal((D, D))`` draw -- because any divergence would rotate
    the inverse map into the wrong frame and produce plausible, wrong numbers."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return Q


def saddle_truth_at(
    P_scaled: Any,
    fixture: Dict[str, Any],
    d: int,
    D: int,
    seed: int,
) -> np.ndarray:
    """``||H||`` of the TRUE saddle at arbitrary ambient points ``P_scaled`` ``(k, D)``, in the
    fixture's scaled units.

    ``fixture`` is the dict ``synthetic_controls.make_saddle_control`` returned, which carries
    ``X``, ``x_param``, ``signs`` and ``global_std``. The saddle is the graph
    ``f(x) = 0.5 x^T diag(signs) x`` over ``R^d``, so a decoded ambient point is mapped back to
    its parameter ``x`` and the analytic curvature is evaluated there:

        ``grad = x * signs``,  ``hess = diag(signs)``  (constant),

    then handed to the sealed ``curvature_probe.graph_mean_curvature`` unmodified -- this
    module writes no curvature mathematics of its own, exactly as
    ``synthetic_controls.make_saddle_control`` does not.

    The local frame is recovered by inverting ``rotate_and_pad``. The offset is fixed by
    requiring that feeding the fixture's own ``X`` back through this function reproduces the
    fixture's own ``H_norm``; that identity is the module's decisive test
    (``test_saddle_truth_at_reproduces_fixture_on_its_own_points``).
    """
    P = np.asarray(P_scaled, dtype=np.float64)
    signs = np.asarray(fixture["signs"], dtype=np.float64)
    global_std = float(fixture["global_std"])
    x_param = np.asarray(fixture["x_param"], dtype=np.float64)
    Xref = np.asarray(fixture["X"], dtype=np.float64)
    m = d + 1

    Q = _rotation(D, seed)
    # Undo scaling and rotation. Xref is centred, so both P and Xref land in a frame that is
    # the true local cloud minus the same constant offset.
    local_centred = (P * global_std @ Q)[:, :m]
    ref_local_centred = (Xref * global_std @ Q)[:, :m]

    # The offset: the reference's own local coordinates are known exactly from the generator
    # (x_param, and f(x) from it), so the constant is their mean minus the centred version's.
    f_ref = 0.5 * np.einsum("ij,j,ij->i", x_param, signs, x_param)
    ref_local_true = np.concatenate([x_param, f_ref[:, None]], axis=1)
    offset = ref_local_true.mean(axis=0) - ref_local_centred.mean(axis=0)

    local = local_centred + offset
    x_hat = local[:, :d]

    grad = (x_hat * signs)[:, None, :]
    hess = np.repeat(np.diag(signs)[None, None, :, :], x_hat.shape[0], axis=0)
    H_local = curvature_probe.graph_mean_curvature(grad, hess)  # (k, d + 1)

    H_padded = np.zeros((x_hat.shape[0], D), dtype=np.float64)
    H_padded[:, :m] = H_local
    H_vec = (H_padded @ Q.T) * global_std
    return np.linalg.norm(H_vec, axis=-1)


# --- drift -------------------------------------------------------------------------------


def reconstruction_drift(X_input: Any, X_recon: Any) -> Dict[str, Any]:
    """Per-point ``||F(z(x_i)) - x_i||`` plus a summary, so drift can be carried as a CONTROL
    in a partial rank correlation rather than only inspected.

    Returned under ``"drift"``; ``"drift_relative"`` divides by each input point's own norm.
    """
    a = np.asarray(X_input, dtype=np.float64)
    b = np.asarray(X_recon, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"X_input and X_recon must have the same shape; got {a.shape}, {b.shape}.")
    drift = np.linalg.norm(a - b, axis=-1)
    denom = np.maximum(np.linalg.norm(a, axis=-1), 1e-12)
    return {
        "drift": drift,
        "drift_relative": drift / denom,
        "median_drift": float(np.median(drift)),
        "median_drift_relative": float(np.median(drift / denom)),
        "p95_drift_relative": float(np.percentile(drift / denom, 95)),
    }


def rescore(
    h_est_norm: Any,
    truth_at_input: Any,
    truth_at_recon: Any,
    drift: Optional[Any] = None,
) -> Dict[str, Any]:
    """Both rank correlations side by side, plus the drift-controlled variant when ``drift``
    is supplied.

    ``rho_at_input``  -- the legacy score: estimate vs truth at the point handed in.
    ``rho_at_recon``  -- the corrected score: estimate vs truth where the decoder put it.
    ``delta``         -- ``rho_at_recon - rho_at_input``; a large magnitude means the legacy
                         number was carrying reconstruction error.
    ``rho_input_given_drift`` -- legacy score with drift partialled out, present only when
                         ``drift`` is given. A third, independent read on the same question.
    """
    from scipy.stats import spearmanr

    from . import cross_split_curvature as csc

    h = np.asarray(h_est_norm, dtype=np.float64).ravel()
    ti = np.asarray(truth_at_input, dtype=np.float64).ravel()
    tr = np.asarray(truth_at_recon, dtype=np.float64).ravel()
    rho_i = float(spearmanr(h, ti).statistic)
    rho_r = float(spearmanr(h, tr).statistic)
    out = {
        "rho_at_input": rho_i,
        "rho_at_recon": rho_r,
        "delta": rho_r - rho_i,
        "truth_rank_agreement": float(spearmanr(ti, tr).statistic),
        "curvature_convention": CURVATURE_CONVENTION,
    }
    if drift is not None:
        out["rho_input_given_drift"] = csc.partial_spearman(h, ti, controls=np.asarray(drift))
    return out
