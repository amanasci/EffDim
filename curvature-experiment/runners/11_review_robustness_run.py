"""Review robustness: the reviewer's concerns 2-5 on the published five encoders, from their stored decoder geometry.

2 validation-tuned probe (nested RidgeCV for OOF predictions, RidgeCV for the global w), every published quantity
  at alpha = 100 and alpha*; 3 target-difficulty controls (hess_label, local label roughness) and a held-out Delta R^2
  on cluster-split anchors; 4 cluster-bootstrap intervals and thinned-anchor partials; 5 surrogate (S_model) vs
  exact data-side (S) change in local R^2 at t = 1.

REPRODUCTION GUARD. At alpha = 100 the runner recomputes the published partials and counterfactual summaries from the
same stored geometry and compares them with the published records (read through sweep/extract.py) before writing any
new number; any difference stops the run.

NOT PRE-REGISTERED, GATES NOTHING. Reads only; writes only its own record.
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent


def _runner(name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, DIAGNOSTICS_ROOT / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pfs = _runner("09_physics_probe_facing_split_run.py", "physics_probe_facing_split_run")
ns = _runner("09_physics_normal_scaling_run.py", "physics_normal_scaling_run")
th = _runner("09_physics_normal_scaling_thin_run.py", "physics_normal_scaling_thin_run")
ppf, adj, runner = pfs.ppf, pfs.adj, pfs.runner

from typing import Any, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from pu_manifold import linear_probe  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "review-robustness"
ALPHA_GRID = tuple(float(a) for a in np.logspace(-3, 4, 15))
PUBLISHED_ALPHA = 100.0
N_BOOT = 2000
BLOCKS_MAIN = 32
BLOCKS_SENS = (16, 64)
N_SPLITS = 20
THIN_THR = 0.10
SIGN_THR = 0.05
BOOT_SEED = 20260930
CF_SEED = 20260915
MISMATCH, ALIGN = "hess_mismatch_emp", "align_cos_tan"


def oof_predictions(X: np.ndarray, y: np.ndarray, alpha_grid, n_folds: int = pcp.N_OOF_FOLDS,
                    fold_seed: int = pcp.OOF_FOLD_SEED) -> Tuple[np.ndarray, List[float]]:
    """pcp.oof_ridge_predictions on the finite rows, with an alpha grid: a one-value grid (a, a) is the published
    fixed-alpha path; a longer grid selects alpha by RidgeCV inside each outer fold's training rows only."""
    y = np.asarray(y, dtype=np.float64).ravel()
    fin = np.isfinite(y)
    y_hat = np.full(y.shape[0], np.nan)
    Xf, yf = np.asarray(X, dtype=np.float64)[fin], y[fin]
    out = np.full(yf.shape[0], np.nan); alphas: List[float] = []
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=fold_seed).split(Xf):
        fit = linear_probe.fit_probe(Xf[tr], yf[tr].reshape(-1, 1), alpha_grid=tuple(float(a) for a in alpha_grid),
                                     alpha_per_target=False, fit_intercept=True)
        out[te] = np.asarray(linear_probe.predict_probe(fit, Xf[te]), dtype=np.float64).ravel()
        alphas.append(float(fit["estimator"].alpha_))
    y_hat[fin] = out
    return y_hat, alphas


def select_alpha(X: np.ndarray, y: np.ndarray) -> float:
    fin = np.isfinite(y)
    fit = linear_probe.fit_probe(np.asarray(X, float)[fin], np.asarray(y, float)[fin].reshape(-1, 1),
                                 alpha_grid=ALPHA_GRID, alpha_per_target=False, fit_intercept=True)
    return float(fit["estimator"].alpha_)


def global_probe(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    fin = np.isfinite(y)
    ridge = Ridge(alpha=float(alpha)).fit(X[fin], y[fin])
    return ridge.coef_.astype(np.float64), float(ridge.intercept_)
