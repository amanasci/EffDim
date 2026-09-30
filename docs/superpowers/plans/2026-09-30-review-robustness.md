# Review Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer ML4PS reviewer concerns 2–5 on the paper's five galaxy encoders, from the paper's own stored decoder geometry: validation-tuned probe, target-difficulty controls and a held-out test, dependence-aware uncertainty, and surrogate fidelity.

**Architecture:** One additive runner, `runners/11_review_robustness_run.py`, imports the split, counterfactual and thin runners unchanged and composes their functions; the published readers in `sweep/extract.py` define every reference quantity. A reproduction guard at alpha = 100 must match the published numbers before any new number is written. A report script turns the records into a rebuttal-ready `REPORT.md`.

**Tech Stack:** Python 3.14 (`.venv`), numpy, scipy (hierarchy clustering, stats), scikit-learn (RidgeCV via `pu_manifold.linear_probe`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-review-robustness-design.md`

## Global Constraints

- Repo `R=/home/akagi/Documents/Projects/EffDim`, branch `review-robustness`. `PY=$R/.venv/bin/python`. Tests: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/<file>`.
- Additive only: no existing runner, `pu_manifold` module or `sweep` module changes.
- CPU gate at the end: `CLOSURE_WORK=$HOME/.cache/effdim-closure $R/docs/superpowers/harness/gate.sh $R review-robustness` → `GATE PASS`.
- `paper/latex/main.tex` unchanged. No experiment code may contain the strings `paper/`, `"paper"` or `'paper'` (guard test `test_no_paper_dependency.py`) — use the word `published` in names instead.
- Read-only inputs: `/mnt/ssd-cluster/effdim/**` on the pod and `notebooks/.cache/**` locally are never written.
- Protocol unchanged: labels mag_r, photo_z, smooth_fraction, stellar_mass; d = 16; 512 anchors; k = 2048; `pcp` seeds; OOF folds `pcp.N_OOF_FOLDS = 5`, `pcp.OOF_FOLD_SEED`; permutations 2000; counterfactual seed 20260915 (the counterfactual runner's default, recorded in the published records).
- Constants (fixed in source before any run): `ALPHA_GRID = tuple(np.logspace(-3, 4, 15))`, `PUBLISHED_ALPHA = 100.0`, `N_BOOT = 2000`, `BLOCKS_MAIN = 32`, `BLOCKS_SENS = (16, 64)`, `N_SPLITS = 20`, `THIN_THR = 0.10`, `SIGN_THR = 0.05` (the published sign-test threshold), `BOOT_SEED = 20260930`.
- Pod rules (CLAUDE.md, pod guide) bind Task 8: read the guide and check the remote sha256 before the first SSH; everything we write under `/mnt/ssd-cluster/EffDim`; long jobs in tmux; no `du`/`find`/`ls -R` on `/mnt`; `timeout 30 ssh` for checks; `df -h` before writing; CPU budget 30 cores.
- Keep it simple first (CLAUDE.md).
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Ruling carried from planning (deviation from the spec's guard tolerance)

The spec asks every partial to reproduce within 1e-6. Provenance (environment rows): the ViT-B published split record read its geometry from the stored npz (`fit_seed` null), so it can reproduce exactly; the other four were computed in the same process that fitted the decoder, from float64 geometry in memory, while the npz stores float32 (`J`, `Hess`, `image` cast in the split runner). Their partials can only reproduce approximately. The guard therefore uses `exact` (partial and p within 1e-6) for vit_base and `refit` (partial within 0.02, p not compared) for the other four; the counterfactual read the npz for all five and is compared exactly (help, hurt, d_r2_plus, d_r2_minus, t_star within 1e-12). Cost if wrong: an unfaithful recomputation that drifts by < 0.02 on a partial goes unnoticed for four encoders.

## Review Focus

1. **Library drift on the pod** (numpy/scikit-learn differ from the published run's numpy 2.5.1) — must stop at the guard with the differing cells, not write numbers. Pinned by `test_guard_stops_on_perturbed_reference` (Task 5); versions recorded in the environment row (Task 6).
2. **A label with sentinel NaN rows** (photo_z, stellar_mass) — tuned OOF must predict only finite rows, as the published helper does. Pinned by `test_oof_fixed_alpha_matches_published_helper` using a label with NaNs (Task 1).
3. **Bootstrap replicate with too few finite rows or a degenerate control matrix** — skip that replicate and count it, never crash. Pinned by `test_bootstrap_skips_degenerate_replicates` (Task 3).
4. **Wrong geometry file** (sha256 mismatch) — refuse before compute. Pinned by `test_refuses_geometry_sha_mismatch` (Task 6).
5. **d = 20 rows in the ViT-B published record** — the reference must use d = 16 rows only. Pinned by `test_reference_uses_d16_rows_only` (Task 5).

---

## File structure

```
curvature-experiment/
  runners/11_review_robustness_run.py      CREATE  probe/alpha, split quantities, dependence, held-out, counterfactual, guard, CLI
  runners/11_review_robustness_report.py   CREATE  records -> results/review-robustness/REPORT.md
  tests/test_review_robustness.py          CREATE  all tests
  results/review-robustness/               CREATE  (Task 8) REPORT.md, records/*.jsonl
```

Facts the implementer needs (verified in the repo):

- Runner files start with digits; load with `importlib` (pattern in every 09 runner and in `tests/test_tensor_fidelity.py::_load`). Importing `09_physics_probe_facing_split_run.py` gives `pfs` with `pfs.ppf`, `pfs.adj`, `pfs.runner`, `pfs.local_quadratics(X, x0_idx, neigh, geo, targets, min_finite) -> {"hess": {name: (b,d,d)}, "r2_lin": {name: (b,)}, "r2_quad", "u_scale"}`, `pfs.split_columns(geo, w, b0, hess_y, probe_emp, d) -> {"cols": {...}, "checks": {...}}`, `pfs.geometry_from_arrays(J, Hess, image) -> geo`.
- `09_physics_normal_scaling_run.py` (`ns`): `ns.scaling_at_anchor(Xn, yn, x0, w, J, g, ginv, II, xhat, rng)`, `ns.VARIANTS`, `ns.T_GRID` (index 2 is t = 0, index 4 is t = 1, index 0 is t = -1). Its main loop (lines 226–246) is the reference for `counterfactual` below.
- `09_physics_normal_scaling_thin_run.py` (`th`): `th.overlap_matrix(neigh) -> (b,b)` fraction overlap.
- `sweep/extract.py`: `read_rows(path)`, `split_cells(rows) -> {(label, col): multiscale_dict}` (`multiscale_dict` has `partial`, `p`), `cf_summary(npz_path) -> {label: {"S_model": {help, hurt, d_r2_plus, d_r2_minus, t_star}, "random_qmatched": {...}}}`, `sign_test(cf_npz, thin_npz, thr=0.05)`, `_indep(ov, thr) -> bool mask`. Import as `from sweep import extract` (curvature-experiment/ is on `sys.path` once a runner module is loaded).
- `pcp`: `anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]`, `knn_panel(X, a, k) -> {"indices", "distances", "log_knn_radius"}`, `local_r2_panel(y, y_hat, neigh, min_finite) -> {"r2", "local_label_variance", "local_evaluation_count", "n_masked_anchors"}`, `controlled_partial(x, y, Z) -> float`, `MIN_FINITE_NEIGHBOURS`, `K_NEIGHBOURS`, `N_ANCHORS`, `N_OOF_FOLDS`, `OOF_FOLD_SEED`.
- `ppf`: `partial_row(x, r2, Z, n_perm) -> {"n_finite", "partial", "p", "undefined"}`, `MULTISCALE_KS = (16, 64, 256, 1024, 2048)`, `load_physics(args)`, `load_smoke(args)`, `fit_decoder`, `decoder_geometry`, `_append`, `_spearman`, `PRODUCTION_STEMS`, `pl` (the labels module whose loaders the shims replace).
- `pu_manifold.linear_probe`: `fit_probe(X, Y2d, alpha_grid, alpha_per_target, fit_intercept) -> {"estimator": RidgeCV, ...}`, `predict_probe(fit, X)`. `pcp.oof_ridge_predictions` calls it with `alpha_grid=(alpha, alpha)` (a two-entry tuple; the sklearn single-candidate workaround).
- The published split runner's local R^2 and controls (lines 283–292 of the split runner): `y_hat = runner._oof_predictions_for_label(X, y, alpha, 5, OOF_FOLD_SEED)`; `loc = pcp.local_r2_panel(y, y_hat, panel["indices"], MIN_FINITE)`; `Z_multi = column_stack([log r at each k in MULTISCALE_KS <= k, loc["local_label_variance"], loc["local_evaluation_count"]])`; probe `Ridge(alpha).fit(X[fin], y[fin])`; targets `{"y": y, "p": X @ w}` into `local_quadratics`.
- The Section-4 figure (`make_fig_intervention.py`, read only) plots variant `S_model`: base = data readout `X (w_T + w_rad)`, scaled term = decoder's `(1/2) <w_S, II^S>(u, u)`. Variant `S` scales the data-side `X w_S` instead (the exact ambient term). Surrogate fidelity compares `S_model` with `S` at t = 1.

---

### Task 1: Probe with fixed and tuned alpha

**Files:** Create `curvature-experiment/runners/11_review_robustness_run.py`, `curvature-experiment/tests/test_review_robustness.py`.

**Interfaces — Produces:**
- `oof_predictions(X, y, alpha_grid, n_folds=pcp.N_OOF_FOLDS, fold_seed=pcp.OOF_FOLD_SEED) -> (y_hat: (n,), fold_alphas: list[float])` — finite rows only, NaN elsewhere.
- `select_alpha(X, y) -> float` — RidgeCV over `ALPHA_GRID` on all finite rows.
- `global_probe(X, y, alpha) -> (w: (D,), b0: float)` — `Ridge(alpha).fit` on finite rows (as both published runners).
- Constants listed in Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
"""Review robustness: concerns 2-5 on stored geometry."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


rr = _load("11_review_robustness_run.py")


def _ridge_data(n=600, D=12, noise=0.5, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D)); beta = rng.standard_normal(D)
    y = X @ beta + noise * rng.standard_normal(n)
    if nan_frac:
        y[rng.random(n) < nan_frac] = np.nan
    return X, y


def test_oof_fixed_alpha_matches_published_helper():
    X, y = _ridge_data(nan_frac=0.1)
    got, alphas = rr.oof_predictions(X, y, (100.0, 100.0))
    want = rr.runner._oof_predictions_for_label(X, y, 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[np.isfinite(got)], want[np.isfinite(want)])
    assert alphas == [100.0] * rr.pcp.N_OOF_FOLDS


def test_tuned_oof_never_uses_held_out_rows():
    """Scrambling the labels of fold 0's held-out rows must not change the alpha chosen for fold 0."""
    from sklearn.model_selection import KFold
    X, y = _ridge_data()
    _, a_ref = rr.oof_predictions(X, y, rr.ALPHA_GRID)
    test0 = next(iter(KFold(rr.pcp.N_OOF_FOLDS, shuffle=True, random_state=rr.pcp.OOF_FOLD_SEED).split(X)))[1]
    y2 = y.copy(); y2[test0] = np.random.default_rng(9).standard_normal(len(test0)) * 100
    _, a_new = rr.oof_predictions(X, y2, rr.ALPHA_GRID)
    assert a_new[0] == a_ref[0]


def test_select_alpha_small_for_strong_signal():
    X, y = _ridge_data(noise=0.01)
    assert rr.select_alpha(X, y) <= 1.0
    assert rr.PUBLISHED_ALPHA in rr.ALPHA_GRID and 1.0 in [round(a, 12) for a in rr.ALPHA_GRID]


def test_global_probe_matches_ridge():
    from sklearn.linear_model import Ridge
    X, y = _ridge_data(nan_frac=0.1)
    w, b0 = rr.global_probe(X, y, 100.0)
    fin = np.isfinite(y)
    ref = Ridge(alpha=100.0).fit(X[fin], y[fin])
    np.testing.assert_array_equal(w, ref.coef_); assert b0 == float(ref.intercept_)
```

- [ ] **Step 2: Run — expect FAIL** (`FileNotFoundError` for the runner).

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_review_robustness.py`

- [ ] **Step 3: Implement** — create the runner:

```python
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
```

- [ ] **Step 4: Run — expect 4 passed.** If `test_oof_fixed_alpha_matches_published_helper` fails on exact equality, that is a finding (the published helper path is not what we replicate): stop and use systematic-debugging; do not loosen to allclose.
- [ ] **Step 5: Commit** — `feat(review-robustness): probe with published and validation-tuned alpha` + trailer.

---

### Task 2: Split quantities, extended controls, partials

**Interfaces — Consumes:** Task 1. **Produces:**
- `probe_panel(X, y, a, panel, alpha_grid) -> {"y_hat", "fold_alphas", "r2", "Z_multi", "global_oof_r2"}`
- `split_quantities(X, y, a, panel, geo, w, b0, d) -> {"cols": dict, "roughness": (b,)}` — roughness = `1 - r2_lin["y"]`.
- `extended_controls(Z_multi, cols, roughness) -> (b, m+2)` — `column_stack([Z_multi, cols["hess_label"], roughness])`.
- `partials(cols, r2, Z, n_perm) -> {MISMATCH: partial_row, ALIGN: partial_row}`

- [ ] **Step 1: Failing tests** (append)

```python
@pytest.fixture(scope="module")
def smoke():
    """The split runner's smoke fixture: generator data, a 3-epoch decoder, its geometry at the anchors."""
    data = rr.ppf.load_smoke(argparse.Namespace(seed=20260905))
    X = data["X"]; n = X.shape[0]
    k, n_anchors, d = rr.adj.SMOKE["k"], rr.adj.SMOKE["n_anchors"], rr.adj.SMOKE["d"]
    a = rr.pcp.anchor_indices(n, rr.pcp.SPLIT_SEED, rr.pcp.HOLDOUT_FRACTION, n_anchors, rr.pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
    panel = rr.pcp.knn_panel(X, a, k)
    fit = rr.ppf.fit_decoder(X, d, X.shape[1], 3)
    with torch.no_grad():
        z = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
    geo = rr.ppf.decoder_geometry(fit["curvature_model"], z)
    return {"X": X, "y": np.asarray(data["labels"]["lin"], float), "a": a, "panel": panel, "geo": geo, "d": d}


def test_probe_panel_matches_published_construction(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    y_hat = rr.runner._oof_predictions_for_label(s["X"], s["y"], 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    loc = rr.pcp.local_r2_panel(s["y"], y_hat, s["panel"]["indices"], rr.pcp.MIN_FINITE_NEIGHBOURS)
    np.testing.assert_array_equal(pp["r2"], loc["r2"])
    ks = [kk for kk in rr.ppf.MULTISCALE_KS if kk <= s["panel"]["indices"].shape[1]]
    assert pp["Z_multi"].shape == (len(s["a"]), len(ks) + 2)


def test_extended_controls_and_partials(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    w, b0 = rr.global_probe(s["X"], s["y"], 100.0)
    sq = rr.split_quantities(s["X"], s["y"], s["a"], s["panel"], s["geo"], w, b0, s["d"])
    Z_ext = rr.extended_controls(pp["Z_multi"], sq["cols"], sq["roughness"])
    assert Z_ext.shape[1] == pp["Z_multi"].shape[1] + 2
    np.testing.assert_array_equal(Z_ext[:, -2], sq["cols"]["hess_label"])
    parts = rr.partials(sq["cols"], pp["r2"], Z_ext, 50)
    ref = rr.ppf.partial_row(sq["cols"][rr.MISMATCH], pp["r2"], Z_ext, 50)
    assert parts[rr.MISMATCH]["partial"] == ref["partial"] and set(parts) == {rr.MISMATCH, rr.ALIGN}
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: ... 'probe_panel'`).
- [ ] **Step 3: Implement** (append)

```python
def probe_panel(X, y, a, panel, alpha_grid) -> Dict[str, Any]:
    """The split runner's per-label probe block (OOF predictions, local R^2, multi-scale controls), at an alpha grid."""
    y = np.asarray(y, dtype=np.float64)
    y_hat, fold_alphas = oof_predictions(X, y, alpha_grid)
    loc = pcp.local_r2_panel(y, y_hat, panel["indices"], pcp.MIN_FINITE_NEIGHBOURS)
    k = panel["indices"].shape[1]
    log_r_multi = np.column_stack([np.log(panel["distances"][:, kk - 1]) for kk in ppf.MULTISCALE_KS if kk <= k])
    Z_multi = np.column_stack([log_r_multi, loc["local_label_variance"], loc["local_evaluation_count"]])
    gr2 = 1.0 - float(np.nansum((y - y_hat) ** 2) / np.nansum((y - np.nanmean(y)) ** 2))
    return {"y_hat": y_hat, "fold_alphas": fold_alphas, "r2": loc["r2"], "Z_multi": Z_multi, "global_oof_r2": gr2}


def split_quantities(X, y, a, panel, geo, w, b0, d) -> Dict[str, Any]:
    lq = pfs.local_quadratics(X, a, panel["indices"], geo, {"y": y, "p": X @ w}, pcp.MIN_FINITE_NEIGHBOURS)
    cols = pfs.split_columns(geo, w, b0, lq["hess"]["y"], lq["hess"]["p"], d)["cols"]
    return {"cols": cols, "roughness": 1.0 - lq["r2_lin"]["y"]}


def extended_controls(Z_multi: np.ndarray, cols: Dict[str, np.ndarray], roughness: np.ndarray) -> np.ndarray:
    return np.column_stack([Z_multi, cols["hess_label"], roughness])


def partials(cols, r2, Z, n_perm: int) -> Dict[str, Dict[str, Any]]:
    return {c: ppf.partial_row(cols[c], r2, Z, n_perm) for c in (MISMATCH, ALIGN)}
```

- [ ] **Step 4: Run — expect 6 passed.**
- [ ] **Step 5: Commit** — `feat(review-robustness): split quantities with target-difficulty controls` + trailer.

---

### Task 3: Dependence — blocks, cluster bootstrap, thinned partials

**Interfaces — Produces:**
- `overlap_blocks(ov: (b,b), n_blocks: int) -> (b,) int labels 0..n_blocks-1`
- `cluster_bootstrap(x, r2, Z, blocks, n_boot, seed) -> {"lo", "hi", "excludes_zero", "n_ok", "n_skipped"}` (95% percentile interval of `pcp.controlled_partial`)
- `thinned_partial(x, r2, Z, ov, thr, n_perm) -> partial_row dict + "n_kept"`

- [ ] **Step 1: Failing tests** (append)

```python
def test_overlap_blocks_count():
    rng = np.random.default_rng(0)
    neigh = np.array([rng.choice(500, 40, replace=False) for _ in range(120)])
    ov = rr.th.overlap_matrix(neigh)
    b = rr.overlap_blocks(ov, 16)
    assert b.shape == (120,) and len(np.unique(b)) == 16


def _synthetic_partial(n, rng):
    Z = rng.standard_normal((n, 2)); x = rng.standard_normal(n)
    r2 = 0.4 * x + Z @ np.array([0.5, -0.3]) + rng.standard_normal(n)
    return x, r2, Z


def test_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(1)
    truth = rr.pcp.controlled_partial(*_synthetic_partial(200000, rng))
    cover = 0
    for rep in range(200):
        x, r2, Z = _synthetic_partial(256, rng)
        blocks = rng.integers(0, 32, 256)
        ci = rr.cluster_bootstrap(x, r2, Z, blocks, 200, rep)
        cover += ci["lo"] <= truth <= ci["hi"]
    assert 0.90 <= cover / 200 <= 0.99


def test_bootstrap_skips_degenerate_replicates():
    x = np.full(40, np.nan); x[:3] = [1.0, 2.0, 3.0]
    r2 = np.arange(40.0); Z = np.ones((40, 1)); blocks = np.arange(40) % 8
    ci = rr.cluster_bootstrap(x, r2, Z, blocks, 50, 0)
    assert ci["n_ok"] + ci["n_skipped"] == 50 and ci["n_skipped"] > 0


def test_thinned_partial_uses_independent_anchors():
    rng = np.random.default_rng(2)
    neigh = np.array([rng.choice(2000, 30, replace=False) for _ in range(100)])
    ov = rr.th.overlap_matrix(neigh)
    x, r2, Z = _synthetic_partial(100, rng)
    t = rr.thinned_partial(x, r2, Z, ov, 0.10, 50)
    assert t["n_kept"] == int(rr.extract._indep(ov, 0.10).sum()) and "partial" in t
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** (append; add `from sweep import extract  # noqa: E402` after the `pcp` import)

```python
def overlap_blocks(ov: np.ndarray, n_blocks: int) -> np.ndarray:
    """Average-linkage clustering of the anchors on 1 - neighbourhood overlap, cut into n_blocks clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - np.asarray(ov, float); D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), "average"), n_blocks, criterion="maxclust") - 1


def cluster_bootstrap(x, r2, Z, blocks, n_boot: int, seed: int) -> Dict[str, Any]:
    """Resample whole blocks with replacement; 95% percentile interval of the controlled partial Spearman."""
    x, r2, Z = np.asarray(x, float), np.asarray(r2, float), np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(x) & np.isfinite(r2) & np.all(np.isfinite(Z), axis=1)
    ub = np.unique(blocks); members = [np.where(blocks == b)[0] for b in ub]
    rng = np.random.default_rng(seed)
    vals: List[float] = []; skipped = 0
    for _ in range(n_boot):
        ii = np.concatenate([members[j] for j in rng.integers(0, len(ub), len(ub))])
        ii = ii[ok[ii]]
        if ii.size < Z.shape[1] + 4:
            skipped += 1; continue
        try:
            v = float(pcp.controlled_partial(x[ii], r2[ii], Z[ii]))
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1; continue
        if np.isfinite(v):
            vals.append(v)
        else:
            skipped += 1
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "excludes_zero": False, "n_ok": 0, "n_skipped": skipped}
    lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return {"lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0), "n_ok": len(vals), "n_skipped": skipped}


def thinned_partial(x, r2, Z, ov, thr: float, n_perm: int) -> Dict[str, Any]:
    keep = extract._indep(np.asarray(ov, float), thr)
    out = ppf.partial_row(np.asarray(x, float)[keep], np.asarray(r2, float)[keep], np.asarray(Z, float)[keep], n_perm)
    out["n_kept"] = int(keep.sum())
    return out
```

- [ ] **Step 4: Run — expect 10 passed** (the coverage test takes ~30 s).
- [ ] **Step 5: Commit** — `feat(review-robustness): cluster bootstrap and thinned-anchor partials` + trailer.

---

### Task 4: Held-out Delta R^2

**Interfaces — Produces:** `heldout_delta_r2(r2, Z_base, Z_geo, blocks, n_splits, seed) -> {"median", "p05", "p95", "frac_pos", "n_splits"}`. Columns and r2 are rank-transformed (on the finite rows) before the OLS fits, matching the rank-based partials.

- [ ] **Step 1: Failing tests** (append)

```python
def test_heldout_positive_with_signal_zero_with_noise():
    rng = np.random.default_rng(3)
    n = 512
    Zb = rng.standard_normal((n, 3)); g = rng.standard_normal((n, 2)); blocks = np.arange(n) % 32
    r2_sig = Zb @ np.array([0.3, 0.2, 0.1]) + g @ np.array([0.8, -0.5]) + 0.5 * rng.standard_normal(n)
    r2_noise = Zb @ np.array([0.3, 0.2, 0.1]) + 0.5 * rng.standard_normal(n)
    sig = rr.heldout_delta_r2(r2_sig, Zb, g, blocks, 20, 0)
    noi = rr.heldout_delta_r2(r2_noise, Zb, g, blocks, 20, 0)
    assert sig["median"] > 0.05 and sig["frac_pos"] == 1.0
    assert abs(noi["median"]) < 0.02 and sig["n_splits"] == 20
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** (append)

```python
def _ols_r2_heldout(yA, XA, yB, XB) -> float:
    A = np.column_stack([np.ones(len(yA)), XA]); B = np.column_stack([np.ones(len(yB)), XB])
    beta, *_ = np.linalg.lstsq(A, yA, rcond=None)
    resid = yB - B @ beta
    return 1.0 - float(resid @ resid) / max(float(((yB - yB.mean()) ** 2).sum()), 1e-300)


def heldout_delta_r2(r2, Z_base, Z_geo, blocks, n_splits: int, seed: int) -> Dict[str, Any]:
    """Out-of-sample R^2 of (base + geometry) minus (base), fitting on half the blocks and scoring on the rest."""
    from scipy.stats import rankdata
    Zb = np.asarray(Z_base, float); Zg = np.asarray(Z_geo, float)
    Zb = Zb[:, None] if Zb.ndim == 1 else Zb; Zg = Zg[:, None] if Zg.ndim == 1 else Zg
    m = np.isfinite(r2) & np.all(np.isfinite(Zb), axis=1) & np.all(np.isfinite(Zg), axis=1)
    y = rankdata(np.asarray(r2, float)[m])
    Zb = np.column_stack([rankdata(c) for c in Zb[m].T]); Zg = np.column_stack([rankdata(c) for c in Zg[m].T])
    bl = np.asarray(blocks)[m]; ub = np.unique(bl)
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(n_splits):
        inA = np.isin(bl, rng.permutation(ub)[: len(ub) // 2]); inB = ~inA
        base = _ols_r2_heldout(y[inA], Zb[inA], y[inB], Zb[inB])
        full = _ols_r2_heldout(y[inA], np.column_stack([Zb, Zg])[inA], y[inB], np.column_stack([Zb, Zg])[inB])
        deltas.append(full - base)
    d = np.asarray(deltas)
    return {"median": float(np.median(d)), "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
            "frac_pos": float(np.mean(d > 0)), "n_splits": int(n_splits)}
```

- [ ] **Step 4: Run — expect 11 passed.**
- [ ] **Step 5: Commit** — `feat(review-robustness): held-out Delta R^2 on cluster-split anchors` + trailer.

---

### Task 5: Counterfactual, surrogate fidelity, references and the guard

**Interfaces — Produces:**
- `counterfactual(X, y, a, neigh, geo, w, b0, d, seed=CF_SEED) -> arrays: dict "{var}:{key}" -> np.ndarray` (keys `t_star, dR2, eq, qq, r2_curve` for every `ns.VARIANTS`), same loop and rng order as the published counterfactual runner.
- `cf_tables(arrays_by_label: {label: arrays}, ov, tmpdir) -> {"summary": extract.cf_summary(...), "sign": extract.sign_test(..., thr=SIGN_THR)}` — writes the arrays as `{label}:{var}:{key}` into a temporary npz so the published readers compute every number.
- `surrogate_fidelity(arrays) -> {"spearman", "median_abs_diff", "n"}` — t = 1 change in local R^2, `S_model` vs `S`.
- `published_reference(split_record, cf_npz) -> {"split": {(label, col): {"partial", "p"}}, "cf": extract.cf_summary(cf_npz)}` — d = 16 result rows only.
- `reproduction_diffs(ours, ref, mode) -> list[str]` (`mode` in `{"exact", "refit"}`); `enforce_reproduction(ours, ref, mode)` raises `SystemExit` listing the diffs.

- [ ] **Step 1: Failing tests** (append)

```python
def _arrays(n=64, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for v in rr.ns.VARIANTS:
        cv = rng.standard_normal((n, len(rr.ns.T_GRID)))
        out.update({f"{v}:r2_curve": cv, f"{v}:eq": rng.standard_normal(n), f"{v}:qq": rng.random(n) + 0.1,
                    f"{v}:t_star": rng.standard_normal(n), f"{v}:dR2": rng.standard_normal(n)})
    return out


def test_surrogate_fidelity_identical_curves():
    arr = _arrays(); arr["S:r2_curve"] = arr["S_model:r2_curve"].copy()
    s = rr.surrogate_fidelity(arr)
    assert s["spearman"] == pytest.approx(1.0) and s["median_abs_diff"] == 0.0 and s["n"] == 64


def test_cf_tables_use_published_readers(tmp_path):
    rng = np.random.default_rng(4)
    neigh = np.array([rng.choice(3000, 40, replace=False) for _ in range(64)])
    ov = rr.th.overlap_matrix(neigh)
    by_label = {"mag_r": _arrays(seed=1)}
    t = rr.cf_tables(by_label, ov, tmp_path)
    cv = by_label["mag_r"]["S_model:r2_curve"]; m = np.isfinite(by_label["mag_r"]["S_model:eq"])
    assert t["summary"]["mag_r"]["S_model"]["help"] == pytest.approx(float(np.mean(cv[m, 4] - cv[m, 2] > 0)))
    assert t["sign"]["mag_r"]["n"] == int((rr.extract._indep(ov, rr.SIGN_THR) & np.isfinite(cv[:, 0])).sum())


def _write_published(tmp_path, partial=0.3):
    rows = [{"row": "environment"}]
    for d in (16, 20):
        rows.append({"row": "result", "d": d, "label": "mag_r",
                     "columns": {rr.MISMATCH: {"multiscale": {"partial": partial if d == 16 else 0.9, "p": 0.01}},
                                 rr.ALIGN: {"multiscale": {"partial": 0.1, "p": 0.2}}}})
    p = tmp_path / "split.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    arr = {f"mag_r:{k}": v for k, v in _arrays(seed=5).items()}
    c = tmp_path / "cf.npz"; np.savez(c, **arr)
    return p, c


def test_reference_uses_d16_rows_only(tmp_path):
    p, c = _write_published(tmp_path)
    ref = rr.published_reference(p, c)
    assert ref["split"][("mag_r", rr.MISMATCH)]["partial"] == 0.3


def test_guard_stops_on_perturbed_reference(tmp_path):
    p, c = _write_published(tmp_path)
    ref = rr.published_reference(p, c)
    ours = {"split": {k: dict(v) for k, v in ref["split"].items()}, "cf": json.loads(json.dumps(ref["cf"]))}
    assert rr.reproduction_diffs(ours, ref, "exact") == []
    ours["split"][("mag_r", rr.MISMATCH)]["partial"] += 0.01
    assert rr.reproduction_diffs(ours, ref, "refit") == []                      # within 0.02
    with pytest.raises(SystemExit, match=f"mag_r.*{rr.MISMATCH}"):
        rr.enforce_reproduction(ours, ref, "exact")
    ours["cf"]["mag_r"]["S_model"]["help"] += 1e-3
    assert any("help" in d for d in rr.reproduction_diffs(ours, ref, "refit"))
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** (append; `import tempfile` is not needed — callers pass a directory)

```python
TOLERANCE = {"exact": 1e-6, "refit": 0.02}
CF_KEYS = ("help", "hurt", "d_r2_plus", "d_r2_minus", "t_star")


def counterfactual(X, y, a, neigh, geo, w, b0, d, seed: int = CF_SEED) -> Dict[str, np.ndarray]:
    """The published counterfactual runner's per-anchor loop (its main, lines 226-246) for one label and readout w."""
    y = np.asarray(y, dtype=np.float64)
    n_anchors = len(a)
    image = geo["image"]; xhat = image / np.linalg.norm(image, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    out = {f"{v}:{key}": np.full(n_anchors, np.nan) for v in ns.VARIANTS for key in ("t_star", "dR2", "eq", "qq")}
    out.update({f"{v}:r2_curve": np.full((n_anchors, len(ns.T_GRID)), np.nan) for v in ns.VARIANTS})
    for i in range(n_anchors):
        idx = neigh[i]; yn = y[idx]; m = np.isfinite(yn)
        if m.sum() < pcp.MIN_FINITE_NEIGHBOURS:
            continue
        sc = ns.scaling_at_anchor(X[idx][m], yn[m], X[a[i]], w, geo["J"][i], geo["g"][i], geo["ginv"][i], geo["II"][i], xhat[i], rng)
        for v in ns.VARIANTS:
            for key in ("t_star", "dR2", "eq", "qq"):
                out[f"{v}:{key}"][i] = sc[v][key]
            out[f"{v}:r2_curve"][i] = sc[v]["r2_curve"]
    return out


def cf_tables(arrays_by_label: Dict[str, Dict[str, np.ndarray]], ov: np.ndarray, tmpdir) -> Dict[str, Any]:
    tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    cf_npz, thin_npz = tmpdir / "cf.npz", tmpdir / "thin.npz"
    np.savez(cf_npz, **{f"{lab}:{k}": v for lab, arr in arrays_by_label.items() for k, v in arr.items()})
    np.savez(thin_npz, overlap=np.asarray(ov, np.float32))
    return {"summary": extract.cf_summary(cf_npz), "sign": extract.sign_test(cf_npz, thin_npz, thr=SIGN_THR)}


def surrogate_fidelity(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    ds = arrays["S:r2_curve"][:, 4] - arrays["S:r2_curve"][:, 2]
    dm = arrays["S_model:r2_curve"][:, 4] - arrays["S_model:r2_curve"][:, 2]
    m = np.isfinite(ds) & np.isfinite(dm)
    return {"spearman": ppf._spearman(dm[m], ds[m]), "median_abs_diff": float(np.median(np.abs(dm[m] - ds[m]))), "n": int(m.sum())}


def published_reference(split_record, cf_npz) -> Dict[str, Any]:
    rows = [r for r in extract.read_rows(split_record) if r.get("row") != "result" or r.get("d") == 16]
    return {"split": {k: {"partial": v["partial"], "p": v["p"]} for k, v in extract.split_cells(rows).items()
                      if k[1] in (MISMATCH, ALIGN)},
            "cf": extract.cf_summary(cf_npz)}


def reproduction_diffs(ours: Dict[str, Any], ref: Dict[str, Any], mode: str) -> List[str]:
    tol = TOLERANCE[mode]; out: List[str] = []
    for key, r in ref["split"].items():
        o = ours["split"].get(key)
        if o is None:
            out.append(f"{key[0]} {key[1]}: missing"); continue
        if not abs(o["partial"] - r["partial"]) <= tol:
            out.append(f"{key[0]} {key[1]}: partial {o['partial']:+.6f} vs published {r['partial']:+.6f}")
        if mode == "exact" and not abs(o["p"] - r["p"]) <= tol:
            out.append(f"{key[0]} {key[1]}: p {o['p']:.6f} vs published {r['p']:.6f}")
    for lab, by_var in ref["cf"].items():
        for var, r in by_var.items():
            for k in CF_KEYS:
                if k in r and not abs(ours["cf"][lab][var][k] - r[k]) <= 1e-12:
                    out.append(f"{lab} {var}: {k} {ours['cf'][lab][var][k]!r} vs published {r[k]!r}")
    return out


def enforce_reproduction(ours, ref, mode: str) -> None:
    diffs = reproduction_diffs(ours, ref, mode)
    if diffs:
        raise SystemExit("reproduction guard FAILED at alpha = 100 (no new numbers written):\n  " + "\n  ".join(diffs))
```

- [ ] **Step 4: Run — expect 15 passed.**
- [ ] **Step 5: Commit** — `feat(review-robustness): counterfactual, surrogate fidelity and the reproduction guard` + trailer.

---

### Task 6: One encoder end to end — CLI, records, smoke

**Interfaces — Consumes:** Tasks 1–5. **Produces:**
- `install_shims(parquet_path, column, label_table) -> label_table_sha256 | None` — the published runners' loader shims (copied from the counterfactual runner's `main`, lines 158–186), so `ppf.load_physics` reads the given parquet and label table.
- `analyse_label(X, y, a, panel, geo, d, ov, blocks: {G: labels}, alpha_mode, n_perm, n_boot, tmpdir) -> (row: dict, cf_arrays: dict)` — `alpha_mode` in `{"published", "tuned"}`.
- `main()`; flags `--encoder`, `--geometry-npz`, `--geometry-sha256`, `--parquet-path`, `--embedding-column`, `--label-table`, `--label-table-sha256`, `--published-split`, `--published-cf`, `--guard {exact,refit}`, `--threads`, `--record-path`, `--n-perm` (default 2000), `--n-boot` (default `N_BOOT`), `--smoke`.
- Records: `environment` row (versions, shas, flags), then after the guard a `guard` row `{"mode", "passed": true}`, then one `result` row per (label, alpha_mode).

- [ ] **Step 1: Failing tests** (append)

```python
def test_refuses_geometry_sha_mismatch(tmp_path, monkeypatch):
    g = tmp_path / "g.npz"; np.savez(g, anchor_idx=np.arange(3))
    monkeypatch.setattr(sys, "argv", ["x", "--encoder", "vit_base", "--geometry-npz", str(g), "--geometry-sha256", "0" * 64,
                                      "--threads", "8", "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="sha256"):
        rr.main()
    assert not (tmp_path / "r.jsonl").exists()


def test_smoke_end_to_end(tmp_path, monkeypatch):
    rec = tmp_path / "11_review_robustness_smoke.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--smoke", "--threads", "8", "--record-path", str(rec), "--n-perm", "20", "--n-boot", "30"])
    rr.main()
    rows = [json.loads(l) for l in rec.read_text().splitlines()]
    assert rows[0]["row"] == "environment"
    res = [r for r in rows if r["row"] == "result"]
    assert {(r["label"], r["alpha_mode"]) for r in res} == {(l, m) for l in ("lin", "nonlin_with_nan") for m in ("published", "tuned")}
    for r in res:
        assert np.isfinite(r["partials"]["published_controls"][rr.MISMATCH]["partial"])
        assert set(r["bootstrap"]) == {"16", "32", "64"} and "heldout" in r and "surrogate" in r and "cf" in r
    assert all(r["alpha"] == 100.0 for r in res if r["alpha_mode"] == "published")
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** (append; add `import argparse`, `import hashlib`, `import time`, `from datetime import datetime, timezone` with `# noqa: E402`)

```python
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_shims(parquet_path, column, label_table):
    """The published runners' loader shims (counterfactual runner main): read this parquet and this label table."""
    ppf.pl.PHYSICS_PARQUET_PATH = parquet_path
    ppf.pl.PHYSICS_COLUMN = column

    def _load_embeddings(parquet_path=None, column=None, expected_rows=None, normalize=True):
        import pyarrow.parquet as pq
        path = parquet_path or ppf.pl.PHYSICS_PARQUET_PATH; col = column or ppf.pl.PHYSICS_COLUMN
        tbl = pq.read_table(path, columns=[col])
        raw = np.stack([np.asarray(v, dtype=np.float64) for v in tbl.column(col).to_pylist()])
        want = expected_rows or ppf.pl.EXPECTED_N_PHYSICS_ROWS
        if raw.shape[0] != want:
            raise RuntimeError(f"{path}: {raw.shape[0]} rows, expected {want}")
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        X = raw / np.maximum(norms, 1e-12) if normalize else raw
        return {"X": X, "n_rows": int(X.shape[0]), "n_features": int(X.shape[1])}
    ppf.pl.load_physics_embeddings = _load_embeddings
    import pandas as pd
    frame = pd.read_parquet(label_table)

    def _load_label_table(columns, expected_rows=None):
        out = frame[list(columns)].reset_index(drop=True)
        want = expected_rows if expected_rows is not None else ppf.pl.EXPECTED_N_PHYSICS_ROWS
        if len(out) != want:
            raise RuntimeError(f"label table has {len(out)} rows, expected {want}")
        return out
    ppf.pl.load_label_table = _load_label_table


def analyse_label(X, y, a, panel, geo, d, ov, blocks, alpha_mode: str, n_perm: int, n_boot: int, tmpdir):
    grid = (PUBLISHED_ALPHA, PUBLISHED_ALPHA) if alpha_mode == "published" else ALPHA_GRID
    pp = probe_panel(X, y, a, panel, grid)
    alpha = PUBLISHED_ALPHA if alpha_mode == "published" else select_alpha(X, y)
    w, b0 = global_probe(X, y, alpha)
    sq = split_quantities(X, y, a, panel, geo, w, b0, d)
    cols, r2 = sq["cols"], pp["r2"]
    Z_ext = extended_controls(pp["Z_multi"], cols, sq["roughness"])
    arrays = counterfactual(X, y, a, panel["indices"], geo, w, b0, d)
    row = {"alpha_mode": alpha_mode, "alpha": float(alpha), "fold_alphas": pp["fold_alphas"], "global_oof_r2": pp["global_oof_r2"],
           "partials": {"published_controls": partials(cols, r2, pp["Z_multi"], n_perm),
                        "extended_controls": partials(cols, r2, Z_ext, n_perm)},
           "bootstrap": {str(G): {c: cluster_bootstrap(cols[c], r2, pp["Z_multi"], blocks[G], n_boot, BOOT_SEED) for c in (MISMATCH, ALIGN)}
                         for G in (BLOCKS_SENS[0], BLOCKS_MAIN, BLOCKS_SENS[1])},
           "thinned": {c: thinned_partial(cols[c], r2, pp["Z_multi"], ov, THIN_THR, n_perm) for c in (MISMATCH, ALIGN)},
           "heldout": heldout_delta_r2(r2, Z_ext, np.column_stack([cols[MISMATCH], cols[ALIGN]]), blocks[BLOCKS_MAIN], N_SPLITS, BOOT_SEED),
           "surrogate": surrogate_fidelity(arrays)}
    return row, arrays


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder", type=str, default="smoke")
    p.add_argument("--geometry-npz"); p.add_argument("--geometry-sha256")
    p.add_argument("--parquet-path"); p.add_argument("--embedding-column")
    p.add_argument("--label-table"); p.add_argument("--label-table-sha256")
    p.add_argument("--published-split"); p.add_argument("--published-cf")
    p.add_argument("--guard", choices=["exact", "refit"], default="exact")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--record-path", type=str, required=True)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    import tempfile
    args = build_parser().parse_args()
    record_path = Path(args.record_path).resolve()
    for stem in ppf.PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    t0 = time.monotonic()
    if args.smoke:
        import torch
        data = ppf.load_smoke(argparse.Namespace(seed=20260905))
        X, labels = data["X"], data["labels"]
        k, n_anchors, d = adj.SMOKE["k"], adj.SMOKE["n_anchors"], adj.SMOKE["d"]
        a = pcp.anchor_indices(X.shape[0], pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
        fit = ppf.fit_decoder(X, d, X.shape[1], adj.SMOKE_EPOCHS)
        with torch.no_grad():
            z = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
        geo = ppf.decoder_geometry(fit["curvature_model"], z)
        geo_sha = lab_sha = None
    else:
        geo_sha = _sha256(args.geometry_npz)
        if geo_sha != args.geometry_sha256:
            raise SystemExit(f"geometry sha256 {geo_sha} != expected {args.geometry_sha256}: {args.geometry_npz}")
        lab_sha = _sha256(args.label_table)
        if lab_sha != args.label_table_sha256:
            raise SystemExit(f"label table sha256 {lab_sha} != expected {args.label_table_sha256}")
        install_shims(args.parquet_path, args.embedding_column, args.label_table)
        data = ppf.load_physics(argparse.Namespace(labels=",".join((ppf.pl.PRIMARY_LABEL,) + ppf.pl.SECONDARY_LABELS)))
        X, labels = data["X"], data["labels"]
        k, n_anchors, d = pcp.K_NEIGHBOURS, pcp.N_ANCHORS, 16
        a = pcp.anchor_indices(X.shape[0], pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
        z = np.load(args.geometry_npz)
        assert np.array_equal(z["anchor_idx"], a), "anchor draw differs from the stored geometry"
        geo = pfs.geometry_from_arrays(z["J"], z["Hess"], z["image"])
    panel = pcp.knn_panel(X, a, k)
    ov = th.overlap_matrix(panel["indices"])
    blocks = {G: overlap_blocks(ov, G) for G in (BLOCKS_SENS[0], BLOCKS_MAIN, BLOCKS_SENS[1])}
    import sklearn, scipy
    env = {"experiment": EXPERIMENT, "row": "environment", "encoder": args.encoder, "timestamp": _utc_now(),
           "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "guard": args.guard,
           "geometry_npz": args.geometry_npz, "geometry_sha256": geo_sha, "parquet_path": args.parquet_path,
           "embedding_column": args.embedding_column, "label_table": args.label_table, "label_table_sha256": lab_sha,
           "published_split": args.published_split, "published_cf": args.published_cf, "n_perm": args.n_perm, "n_boot": args.n_boot,
           "alpha_grid": list(ALPHA_GRID), "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
           "python": sys.version.split()[0], "pre_registered": False, "gates": "nothing"}
    rows_by_mode: Dict[str, List[dict]] = {"published": [], "tuned": []}
    cf_by_mode: Dict[str, Dict[str, Any]] = {"published": {}, "tuned": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("published", "tuned"):
            for name, y in labels.items():
                row, arrays = analyse_label(X, np.asarray(y, float), a, panel, geo, d, ov, blocks, mode, args.n_perm, args.n_boot, tmp)
                row.update({"experiment": EXPERIMENT, "row": "result", "encoder": args.encoder, "label": name})
                rows_by_mode[mode].append(row); cf_by_mode[mode][name] = arrays
                print(f"[{mode}] {name}: alpha {row['alpha']:g} OOF R2 {row['global_oof_r2']:.3f} "
                      f"mismatch {row['partials']['published_controls'][MISMATCH]['partial']:+.3f} "
                      f"align {row['partials']['published_controls'][ALIGN]['partial']:+.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
            tables = cf_tables(cf_by_mode[mode], ov, Path(tmp) / mode)
            for row in rows_by_mode[mode]:
                row["cf"] = tables["summary"].get(row["label"]); row["sign_test"] = tables["sign"].get(row["label"])
            if mode == "published" and not args.smoke:
                ours = {"split": {(r["label"], c): r["partials"]["published_controls"][c] for r in rows_by_mode[mode] for c in (MISMATCH, ALIGN)},
                        "cf": tables["summary"]}
                enforce_reproduction(ours, published_reference(args.published_split, args.published_cf), args.guard)
    ppf._append(env, record_path)
    if not args.smoke:
        ppf._append({"experiment": EXPERIMENT, "row": "guard", "encoder": args.encoder, "mode": args.guard, "passed": True,
                     "timestamp": _utc_now()}, record_path)
    for mode in ("published", "tuned"):
        for row in rows_by_mode[mode]:
            row["timestamp"] = _utc_now(); ppf._append(row, record_path)
    print(f"DONE {args.encoder} in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

Note: `json` cannot serialise numpy bool/float scalars inside nested dicts only if `ppf._append` does not handle them; if the smoke test fails with a `TypeError: Object of type bool_ ...`, convert at the source (the `excludes_zero` and partial_row values) with `bool()`/`float()` — do not add a custom encoder.

- [ ] **Step 4: Run — expect 17 passed.**
- [ ] **Step 5: Gate and full suite** — `EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests` (all pass) and `gate.sh $R review-robustness` → `GATE PASS`.
- [ ] **Step 6: Commit** — `feat(review-robustness): one encoder end to end with the guard and records` + trailer.

---

### Task 7: Report

**Files:** Create `curvature-experiment/runners/11_review_robustness_report.py`; append tests.

**Interfaces — Produces:** `load(paths) -> {"env": [..], "guard": [..], "rows": [..]}` (last row wins per (encoder, label, alpha_mode)); `write_report(data, out_dir)` → `REPORT.md` with sections `## Concern 2`, `## Concern 3`, `## Concern 4`, `## Concern 5`, `## Limitation`, and a guard line per encoder. CLI `--record-path` (1+), `--out-dir` (default `curvature-experiment/results/review-robustness`).

- [ ] **Step 1: Failing test** (append)

```python
rep = _load("11_review_robustness_report.py")


def _res(enc, lab, mode, mis):
    part = {rr.MISMATCH: {"partial": mis, "p": 0.001}, rr.ALIGN: {"partial": 0.2, "p": 0.01}}
    boot = {c: {"lo": mis - 0.1, "hi": mis + 0.1, "excludes_zero": True, "n_ok": 2000, "n_skipped": 0} for c in (rr.MISMATCH, rr.ALIGN)}
    return {"row": "result", "encoder": enc, "label": lab, "alpha_mode": mode, "alpha": 100.0 if mode == "published" else 3.7,
            "global_oof_r2": 0.6, "partials": {"published_controls": part, "extended_controls": part},
            "bootstrap": {"16": boot, "32": boot, "64": boot}, "thinned": {c: {"partial": 0.1, "p": 0.3, "n_kept": 22} for c in part},
            "heldout": {"median": 0.03, "p05": 0.01, "p95": 0.05, "frac_pos": 1.0, "n_splits": 20},
            "cf": {"S_model": {"help": 0.8, "hurt": 0.9, "t_star": 0.7}, "random_qmatched": {"help": 0.4, "hurt": 0.5}},
            "sign_test": {"n": 15, "help": 0.8, "p_help": 0.01}, "surrogate": {"spearman": 0.95, "median_abs_diff": 0.001, "n": 512}}


def test_report_sections(tmp_path):
    rows = [{"row": "environment", "encoder": "vit_base", "numpy": "2.5.1"}, {"row": "guard", "encoder": "vit_base", "mode": "exact", "passed": True}]
    rows += [_res("vit_base", "mag_r", m, -0.3) for m in ("published", "tuned")] + [_res("vit_base", "mag_r", "tuned", -0.4)]
    p = tmp_path / "r.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = rep.load([p])
    assert len(data["rows"]) == 2
    rep.write_report(data, tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    for h in ("## Concern 2", "## Concern 3", "## Concern 4", "## Concern 5", "## Limitation", "guard: exact PASS"):
        assert h in text, h
    assert "-0.400" in text      # the re-run tuned row won
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**

```python
"""Review robustness report: records -> REPORT.md, one rebuttal-ready table per concern."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "review-robustness"
MISMATCH, ALIGN = "hess_mismatch_emp", "align_cos_tan"
ENC_ORDER = ["vit_base", "dinov3_vitb16", "clip_base", "convnext_base", "vit_large"]
LABELS = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]


def load(paths: List[Path]) -> Dict[str, Any]:
    env, guard, rows = [], [], {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["row"] == "environment":
                env.append(r)
            elif r["row"] == "guard":
                guard.append(r)
            elif r["row"] == "result":
                rows[(r["encoder"], r["label"], r["alpha_mode"])] = r
    return {"env": env, "guard": guard, "rows": list(rows.values())}


def _f(v, fmt="+.3f"):
    return "--" if v is None else format(v, fmt)


def _p(v):
    return "--" if v is None else (f"{v:.3f}" if v >= 0.001 else "<0.001")


def _get(rows, enc, lab, mode):
    return next((r for r in rows if r["encoder"] == enc and r["label"] == lab and r["alpha_mode"] == mode), None)


def _encs(rows):
    have = {r["encoder"] for r in rows}
    return [e for e in ENC_ORDER if e in have] + sorted(have - set(ENC_ORDER))


def write_report(data: Dict[str, Any], out_dir: Path) -> None:
    rows = data["rows"]; out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Review robustness (concerns 2-5)", "",
         "All numbers recomputed from the published decoder geometry (sha256-verified). Mismatch = `hess_mismatch_emp`, "
         "alignment = `align_cos_tan`, partial Spearman with the published multi-scale controls unless stated.", ""]
    for g in data["guard"]:
        L.append(f"- {g['encoder']}: guard: {g['mode']} {'PASS' if g['passed'] else 'FAIL'}")
    L += ["", "## Concern 2: validation-tuned probe", "",
          "| encoder | label | alpha* | OOF R2 @100 | OOF R2 @alpha* | mismatch @100 (p) | mismatch @alpha* (p) | align @100 (p) | align @alpha* (p) | help/hurt @100 | help/hurt @alpha* |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            a, b = _get(rows, e, lab, "published"), _get(rows, e, lab, "tuned")
            if not (a and b):
                continue
            pa, pb = a["partials"]["published_controls"], b["partials"]["published_controls"]
            L.append(f"| {e} | {lab} | {b['alpha']:.3g} | {a['global_oof_r2']:.3f} | {b['global_oof_r2']:.3f} | "
                     f"{_f(pa[MISMATCH]['partial'])} ({_p(pa[MISMATCH]['p'])}) | {_f(pb[MISMATCH]['partial'])} ({_p(pb[MISMATCH]['p'])}) | "
                     f"{_f(pa[ALIGN]['partial'])} ({_p(pa[ALIGN]['p'])}) | {_f(pb[ALIGN]['partial'])} ({_p(pb[ALIGN]['p'])}) | "
                     f"{a['cf']['S_model']['help']:.2f}/{a['cf']['S_model']['hurt']:.2f} | {b['cf']['S_model']['help']:.2f}/{b['cf']['S_model']['hurt']:.2f} |")
    for mode, title in (("published", "alpha = 100"), ("tuned", "alpha*")):
        L += ["", f"## Concern 3: added value beyond target difficulty ({title})", "",
              "Extended controls = published controls + label-Hessian norm + local label roughness (1 - local linear R2). "
              "Held-out: out-of-sample Delta R2 of local R2 from adding mismatch and alignment to the extended controls, "
              "fit on half the overlap blocks, scored on the rest (20 splits; ranks).", "",
              "| encoder | label | mismatch published ctl | mismatch extended ctl (p) | align extended ctl (p) | held-out Delta R2 median [p05, p95] | splits > 0 |",
              "|---|---|---|---|---|---|---|"]
        for e in _encs(rows):
            for lab in LABELS:
                r = _get(rows, e, lab, mode)
                if not r:
                    continue
                pe = r["partials"]["extended_controls"]; h = r["heldout"]
                L.append(f"| {e} | {lab} | {_f(r['partials']['published_controls'][MISMATCH]['partial'])} | "
                         f"{_f(pe[MISMATCH]['partial'])} ({_p(pe[MISMATCH]['p'])}) | {_f(pe[ALIGN]['partial'])} ({_p(pe[ALIGN]['p'])}) | "
                         f"{_f(h['median'])} [{_f(h['p05'])}, {_f(h['p95'])}] | {h['frac_pos']:.2f} |")
    L += ["", "## Concern 4: dependence across anchors (alpha = 100)", "",
          "Cluster bootstrap: 32 overlap blocks (average linkage on 1 - neighbourhood overlap), 2000 resamples of whole blocks, "
          "95% percentile interval; excludes-0 at 16 and 64 blocks as sensitivity. Adjacent blocks still share boundary points, "
          "so these intervals are more honest than anchor-level permutation, not exact. Thinned: anchors with pairwise overlap "
          "<= 0.10 (low power).", "",
          "| encoder | label | column | partial | 95% CI (32 blocks) | excl. 0 at 16/32/64 | thinned partial (p, n) |",
          "|---|---|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            r = _get(rows, e, lab, "published")
            if not r:
                continue
            for c in (MISMATCH, ALIGN):
                b = {G: r["bootstrap"][G][c] for G in ("16", "32", "64")}; t = r["thinned"][c]
                L.append(f"| {e} | {lab} | {c} | {_f(r['partials']['published_controls'][c]['partial'])} | "
                         f"[{_f(b['32']['lo'])}, {_f(b['32']['hi'])}] | {'/'.join('y' if b[G]['excludes_zero'] else 'n' for G in ('16', '32', '64'))} | "
                         f"{_f(t['partial'])} ({_p(t['p'])}, {t['n_kept']}) |")
    L += ["", "## Concern 5: surrogate fidelity (alpha = 100)", "",
          "The Section 4 figure plots the counterfactual variant `S_model`: the data readout's tangent-plus-radial part is "
          "kept exactly on the neighbours, and the scaled term is the decoder's second-order in-sphere term "
          "(1/2)<w_S, II^S>(u,u). Variant `S` scales the data-side in-sphere readout w_S.x instead (exact, ambient). "
          "Below: agreement of their t = 1 change in local R2 across anchors.", "",
          "| encoder | label | Spearman(S_model, S) | median abs diff | n |", "|---|---|---|---|---|"]
    for e in _encs(rows):
        for lab in LABELS:
            r = _get(rows, e, lab, "published")
            if r:
                s = r["surrogate"]
                L.append(f"| {e} | {lab} | {_f(s['spearman'])} | {s['median_abs_diff']:.4f} | {s['n']} |")
    L += ["", "## Limitation", "",
          "On a known manifold (results/tensor-fidelity) the label-Hessian estimate has the right direction but its magnitude "
          "is about 2x too large at paper scale (d=16, n=86,471). The partials above are rank-based and unaffected; any claim "
          "about mismatch magnitudes would be.", ""]
    (out_dir / "REPORT.md").write_text("\n".join(L))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-path", nargs="+", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = p.parse_args()
    data = load([Path(x) for x in a.record_path])
    write_report(data, Path(a.out_dir))
    print(f"{len(data['rows'])} result rows -> {a.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect 18 passed.**
- [ ] **Step 5: Commit** — `feat(review-robustness): rebuttal report` + trailer.

---

### Task 8: Run the five encoders on the pod, report, commit

- [ ] **Step 1: Push the branch** — `git -C $R push -u origin review-robustness`.
- [ ] **Step 2: Pod setup** — guide sha256 check; then in tmux window `rr`: `cd /mnt/ssd-cluster/EffDim/repo && git fetch -q origin review-robustness && git checkout -q review-robustness && git merge -q --ff-only origin/review-robustness`, then `BRANCH=review-robustness bash curvature-experiment/sweep/setup_pod.sh` → exits 0. `df -h /mnt/ssd-cluster`.
- [ ] **Step 3: Copy the published references to the pod** (from this machine; read-only sources):

```bash
ssh root@216.153.49.26 'mkdir -p /mnt/ssd-cluster/EffDim/review-robustness/published'
cd $R/notebooks/.cache && rsync -a 09_physics_probe_facing_split.jsonl 09_physics_probe_facing_split_{dinov3_vitb16,clip_base,convnext_base,vit_large}.jsonl \
  09_physics_normal_scaling_{vit_base,dinov3_vitb16,clip_base,convnext_base,vit_large}_d16.npz root@216.153.49.26:/mnt/ssd-cluster/EffDim/review-robustness/published/
```

- [ ] **Step 4: Run, in tmux, five encoders in parallel at 6 threads each (30-core budget)** — a script `/mnt/ssd-cluster/EffDim/review-robustness/run.sh`:

```bash
B=/mnt/ssd-cluster/EffDim; E=/mnt/ssd-cluster/effdim; RR=$B/review-robustness; P=$RR/published
SNAP=$E/hf-cache/hub/datasets--UniverseTBD--pu-embeddings/snapshots/bc081f8a5db4767edcd958653d96efde9137de0b/physics
LAB=$E/labels_Smith42_galaxies_v2.0_test.parquet; LABSHA=60f2f82e64e4036eb4eff9a448b15365dd1e974e52779fe7c1323db0d9dabfd9
cd $B/repo/curvature-experiment
run() { enc=$1 geo=$2 sha=$3 split=$4 guard=$5
  $B/venv/bin/python runners/11_review_robustness_run.py --encoder $enc --geometry-npz $geo --geometry-sha256 $sha \
    --parquet-path $SNAP/${enc}_test.parquet --embedding-column ${enc}_galaxies --label-table $LAB --label-table-sha256 $LABSHA \
    --published-split $P/$split --published-cf $P/09_physics_normal_scaling_${enc}_d16.npz --guard $guard --threads 6 \
    --record-path $RR/11_review_robustness_${enc}.jsonl > $RR/${enc}.log 2>&1 && echo "DONE $enc" || echo "FAILED $enc"; }
run vit_base $E/probe-facing-out/probe-facing/09_probe_facing_geometry_d16.npz 477886ad7036ff6a18897409bdbb1db15262457bb618da64fadaaba138b82ff9 09_physics_probe_facing_split.jsonl exact &
run clip_base $E/xenc-out/geometry_clip_base/09_probe_facing_geometry_d16_seed0.npz 602cf931f1dd5ff42a40beece81f2a724c6eece5238e15c40fe202dd4daead80 09_physics_probe_facing_split_clip_base.jsonl refit &
run convnext_base $E/xenc-out/geometry_convnext_base/09_probe_facing_geometry_d16_seed0.npz 14db5894cf8d6eecb7e3e50146032e95b1bea05ddda6e0c5df99a34449fb4904 09_physics_probe_facing_split_convnext_base.jsonl refit &
run dinov3_vitb16 $E/xenc-out/geometry_dinov3_vitb16/09_probe_facing_geometry_d16_seed0.npz b4176c43b0171869f16e4bb4a93290e18c4b298f25fc6a392219a1def9f299de 09_physics_probe_facing_split_dinov3_vitb16.jsonl refit &
run vit_large $E/xenc-out/geometry_vit_large/09_probe_facing_geometry_d16_seed0.npz f8c1015ea589e378e33d36529ca3ea0499246408f432cd289e1dcf53db181718 09_physics_probe_facing_split_vit_large.jsonl refit &
wait; echo RR_DONE
```
Expected: five `DONE` lines. A `FAILED` whose log ends in `reproduction guard FAILED` is a finding, not a bug to paper over: read the listed cells; if every diff is a tiny floating-point difference traceable to library versions (compare the environment row's numpy/sklearn with the published numpy 2.5.1), report it and stop — do not change the tolerance without the user.

- [ ] **Step 5: Fetch and report** — `rsync -a root@216.153.49.26:/mnt/ssd-cluster/EffDim/review-robustness/11_review_robustness_\*.jsonl $R/curvature-experiment/results/review-robustness/records/`; then `$PY curvature-experiment/runners/11_review_robustness_report.py --record-path curvature-experiment/results/review-robustness/records/*.jsonl`.
- [ ] **Step 6: Read-out** — per concern, in plain words: does any mismatch/alignment sign or significance change at alpha*; do the partials survive the extended controls; is held-out Delta R^2 above 0; do the 32-block intervals exclude 0; how close is S_model to S. Report what the numbers say; a weakened result is a result.
- [ ] **Step 7: Final checks and commit** — full suite, `gate.sh $R review-robustness` → `GATE PASS`, `git diff --exit-code encoder-scaling -- paper/latex/main.tex`; commit `results/review-robustness/` as `feat(results): review robustness for concerns 2-5 on the five published encoders` + trailer; push.
