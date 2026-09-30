# Tensor Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how well the split runner's full probe-facing tensors (`pf_full`, `pf_tan`, `hess_y`, mismatch) recover exact truth on the known in-sphere manifold, at small scale (CPU grid) and paper scale (pod GPU).

**Architecture:** One new runner, `runners/10_tensor_fidelity_run.py`, imports the split runner unchanged (and through it the probe-facing and adjudication runners) and adds only truth, labels, a chart-invariant ambient comparison and scoring. One report script turns its JSONL records into `REPORT.md` and a figure. No existing file changes.

**Tech Stack:** Python 3.14 (`.venv`), numpy, torch (`torch.func` vmap/jacrev/hessian), scikit-learn Ridge, scipy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-tensor-fidelity-design.md`

## Global Constraints

- Repo `R=/home/akagi/Documents/Projects/EffDim`, branch `tensor-fidelity`. Python `PY=$R/.venv/bin/python`. Tests: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/<file>`.
- Additive only: no existing runner or `pu_manifold` module changes (CLAUDE.md: never rewrite existing runners).
- The CPU gate must pass at the end: `CLOSURE_WORK=$HOME/.cache/effdim-closure $R/docs/superpowers/harness/gate.sh $R tensor-fidelity` → `GATE PASS`.
- `paper/latex/main.tex` must not change. Never import or run `paper/generate/appendix_gen.py`.
- Never write a record whose filename starts with a Phase 9 production stem (`ppf.PRODUCTION_STEMS`); new records go to `10_tensor_fidelity.jsonl`.
- Pass lines, fixed in source before any run: `TENSOR_COS_PASS = 0.8` (median `pf_full` cosine), `MISMATCH_RHO_PASS = 0.7` (mismatch-norm Spearman), on the noise-free small fixture.
- Paper protocol unchanged: `ppf.fit_decoder` with `pcp.MAX_EPOCHS`, `pcp.AE_HIDDEN`; probe `Ridge(alpha=pcp.ALPHA_RIDGE)`; sphere-projected decoder.
- GPU runs pass `--device cuda --deterministic`. Truth is always float64 on CPU.
- Pod rules (CLAUDE.md, `docs/remote-compute/eleutherai-pod-user-guide.md`), binding for Task 5: read the guide and check the remote sha256 before the first SSH; everything under `/mnt/ssd-cluster/EffDim`; long jobs in `tmux`; no `du`/`find`/`ls -R` on `/mnt`; wrap status checks in `timeout 30 ssh ...`; use only GPUs with no processes in `nvidia-smi`; `df -h /mnt/ssd-cluster` before writing.
- Keep it simple first (CLAUDE.md): no extra options, caching or abstractions beyond this plan.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Near-zero true mismatch (label `lam0`)** — a relative error divided by `|mismatch_true| ≈ 0` explodes and a cosine of two near-zero tensors is noise. Expected: mismatch relative error is normalised by `|hess_y_true|`, and mismatch cosine is NaN (excluded) where `|mismatch_true| < 1e-6 |hess_y_true|`. Pinned by `test_mismatch_scoring_near_zero_truth` (Task 1).
2. **Anchors the local quadratic fit skips** (`local_quadratics` returns NaN rows) — must be excluded from medians and counted, not crash or poison the median. Pinned by `test_score_skips_nan_anchors` (Task 1).
3. **Re-running a configuration** appends duplicate rows — the report must keep the last row per (mode, n, noise, seed, label), not double-count. Pinned by `test_report_dedupes_reruns` (Task 4).
4. **Record path with a production stem** — must refuse before any compute. Pinned by `test_refuses_production_record` (Task 3).
5. **`--device cuda` without CUDA** — must stop with a clear message, not fall back to CPU. Pinned by `test_cuda_without_gpu_refuses` (Task 3).

---

## File structure

```
curvature-experiment/
  runners/10_tensor_fidelity_run.py      CREATE  truth, labels, ambient comparison, scoring, CLI
  runners/10_tensor_fidelity_report.py   CREATE  records -> REPORT.md + fig_tensor_fidelity.png
  tests/test_tensor_fidelity.py          CREATE  all tests for both files
  results/tensor-fidelity/               CREATE  (Task 6) REPORT.md, fig, records/*.jsonl
```

Module loading pattern (runner files start with digits, so they are loaded with `importlib`), used by the tests:

```python
import importlib.util, sys
from pathlib import Path
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
```

Facts the implementer needs (all verified in the repo):

- `09_physics_probe_facing_split_run.py` (module-level names): `ppf` (the probe-facing runner), `adj` (the adjudication runner), `runner`, `local_quadratics(X, x0_idx, neigh, geo, targets, min_finite) -> {"hess": {name: (b,d,d)}, "r2_lin", "r2_quad", "u_scale"}`, `split_columns(geo, w, b0, hess_y, probe_emp, d) -> {"cols": {...}, "checks": {...}}`. `cols` includes `hess_mismatch_dec` and `align_cos_full`.
- `ppf.decoder_geometry(model, z: torch.Tensor, out_chunk=None) -> {"J": (b,D,d), "Hess": (b,D,d,d), "image": (b,D), "g", "ginv", "II": (b,D,d,d), "H": (b,D), "cond_g"}` (numpy). Works on any module with `.decode(z)`.
- `ppf.fit_decoder(X, d, in_dim, max_epochs, device="cpu") -> {"model", "curvature_model", "x64", "var_explained", "wallclock_fit_s"}`; init seed is `pcp.TORCH_INIT_SEED` (module global).
- `ppf._spearman(a, b) -> float`; `ppf._append(row, path)`; `ppf.PRODUCTION_STEMS`.
- `adj.InSphereGenerator(d, D, a, widths, amps, seed)` (float64, CPU, `.decode`, `.d`, `.D`), `adj.draw_latents(n, d, scale_choices, scale_probs, seed)`, `adj.generate_points(G, z)`, `adj.FIXTURE` (d=16, D=768, n=86471, k=2048, n_anchors=512, a, bump_widths, bump_amps, scale_choices, scale_probs), `adj.SMOKE` (d=4, D=64, n=4000, k=128, n_anchors=64), `adj._git_head(path)`.
- `pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]`, `pcp.knn_panel(X, anchor_idx, k) -> {"indices", "distances", "log_knn_radius"}`, `pcp.MIN_FINITE_NEIGHBOURS = 32`.
- Threads: `runner._THREADS` is read from `sys.argv --threads` at import; `main()` asserts it equals `args.threads` (as every 09 runner does).

---

### Task 1: Geometry comparison and scoring

**Files:**
- Create: `curvature-experiment/runners/10_tensor_fidelity_run.py`
- Create: `curvature-experiment/tests/test_tensor_fidelity.py`

**Interfaces:**
- Consumes: split runner module (`ppf`, `adj`, `split_columns`), `ppf.decoder_geometry`, `ppf.metric_norms`.
- Produces:
  - `ambient_inner(Ta, geo_a, Tb, geo_b) -> np.ndarray (b,)` — Frobenius inner product of the ambient lifts `J ginv T ginv J^T`.
  - `tensor_cosine(Ta, geo_a, Tb, geo_b) -> (b,)`
  - `relative_error(T_est, geo_est, T_true, geo_true, ref_norm=None) -> (b,)` — `|lift(T_est)-lift(T_true)|_F / ref_norm` (default `|lift(T_true)|_F`).
  - `probe_facing_tensors(geo, w) -> {"w_N": (b,D), "pf_full": (b,d,d), "pf_tan": (b,d,d)}`
  - `truth_geometry(G, z_anchor: np.ndarray) -> geo dict` plus key `"max_abs_H_rad_plus_d"`; raises `AssertionError` if > 1e-8.
  - `score_tensors(est: dict, truth: dict, geo_est, geo_true) -> dict` — `est`/`truth` map `"pf_full","pf_tan","hess_y","mismatch"` to (b,d,d); returns for each tensor `t`: `cos_{t}_p25`, `cos_{t}_p50`, `relerr_{t}_p50`, `n_valid_{t}`.
  - Constants `TENSOR_COS_PASS = 0.8`, `MISMATCH_RHO_PASS = 0.7`, `MISMATCH_FLOOR = 1e-6`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tensor fidelity: full probe-facing tensors vs exact truth on the in-sphere fixture."""
import importlib.util
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


tf = _load("10_tensor_fidelity_run.py")


def _small_generator(seed=20260905):
    c = tf.adj.SMOKE
    return tf.adj.InSphereGenerator(c["d"], c["D"], c["a"], c["bump_widths"], c["bump_amps"], seed=seed)


class _Reparam(torch.nn.Module):
    """G'(z') = G(A z'): the same manifold in another chart."""
    def __init__(self, G, A):
        super().__init__(); self.G = G; self.A = torch.as_tensor(A, dtype=torch.float64)
        self.d, self.D = G.d, G.D
    def decode(self, z):
        return self.G.decode(z @ self.A.T)


def test_truth_exactness():
    G = _small_generator()
    z = tf.adj.draw_latents(8, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=1)
    geo = tf.truth_geometry(G, z)
    assert geo["max_abs_H_rad_plus_d"] < 1e-8


def test_ambient_lift_is_chart_invariant():
    G = _small_generator()
    rng = np.random.default_rng(3)
    A = rng.standard_normal((G.d, G.d)) + 2 * np.eye(G.d)
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=2)
    zp = np.linalg.solve(A, z.T).T                       # A z' = z
    geo = tf.truth_geometry(G, z)
    geo_p = tf.truth_geometry(_Reparam(G, A), zp)
    w = rng.standard_normal(G.D)
    T = tf.probe_facing_tensors(geo, w)["pf_full"]
    Tp = tf.probe_facing_tensors(geo_p, w)["pf_full"]
    np.testing.assert_allclose(tf.tensor_cosine(T, geo, Tp, geo_p), 1.0, atol=1e-10)
    np.testing.assert_allclose(tf.relative_error(Tp, geo_p, T, geo), 0.0, atol=1e-10)


def test_probe_facing_tensors_match_split_columns_norms():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=4)
    geo = tf.truth_geometry(G, z)
    w = np.random.default_rng(5).standard_normal(G.D)
    pt = tf.probe_facing_tensors(geo, w)
    hess = np.zeros_like(pt["pf_full"])
    cols = tf.split_columns(geo, w, 0.0, hess, pt["pf_full"], G.d)["cols"]
    np.testing.assert_allclose(tf.ppf.metric_norms(pt["pf_full"], geo["ginv"])["fro"], cols["pf_full"], rtol=1e-12)
    np.testing.assert_allclose(tf.ppf.metric_norms(pt["pf_tan"], geo["ginv"])["fro"], cols["pf_tan"], rtol=1e-12)


def test_scoring_self_and_negative():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=6)
    geo = tf.truth_geometry(G, z)
    T = tf.probe_facing_tensors(geo, np.random.default_rng(7).standard_normal(G.D))["pf_full"]
    np.testing.assert_allclose(tf.tensor_cosine(T, geo, T, geo), 1.0, atol=1e-12)
    np.testing.assert_allclose(tf.tensor_cosine(-T, geo, T, geo), -1.0, atol=1e-12)
    np.testing.assert_allclose(tf.relative_error(T, geo, T, geo), 0.0, atol=1e-12)


def _four(T):
    return {"pf_full": T, "pf_tan": T, "hess_y": T, "mismatch": T}


def test_score_skips_nan_anchors():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=8)
    geo = tf.truth_geometry(G, z)
    T = tf.probe_facing_tensors(geo, np.random.default_rng(9).standard_normal(G.D))["pf_full"]
    T_est = T.copy(); T_est[2] = np.nan
    s = tf.score_tensors(_four(T_est), _four(T), geo, geo)
    assert s["n_valid_pf_full"] == 5
    assert s["cos_pf_full_p50"] == pytest.approx(1.0, abs=1e-12)


def test_mismatch_scoring_near_zero_truth():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=10)
    geo = tf.truth_geometry(G, z)
    H = tf.probe_facing_tensors(geo, np.random.default_rng(11).standard_normal(G.D))["pf_full"]
    est = _four(H); truth = _four(H)
    truth["mismatch"] = np.zeros_like(H)                     # true mismatch exactly zero
    est["mismatch"] = 1e-3 * H                               # small estimated mismatch
    s = tf.score_tensors(est, truth, geo, geo)
    assert s["n_valid_mismatch"] == 0 and np.isnan(s["cos_mismatch_p50"])
    assert s["relerr_mismatch_p50"] == pytest.approx(1e-3, rel=1e-9)   # normalised by |hess_y_true|
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py`
Expected: collection error / FAIL — `10_tensor_fidelity_run.py` does not exist.

- [ ] **Step 3: Implement**

Create `curvature-experiment/runners/10_tensor_fidelity_run.py`:

```python
"""Tensor fidelity: the split runner's full probe-facing tensors against exact truth on the in-sphere fixture.

PURPOSE. ``09_instrument_adjudication_run.py`` validated only the trace of the second fundamental
form (``H_tan``). The paper's diagnostic uses the full contraction of II with the probe. This runner
estimates, exactly as the split runner does, ``pf_full = <w_N, II>``, its in-sphere part ``pf_tan``,
the label Hessian ``hess_y`` (local quadratic in tangent-projected coordinates) and the mismatch
``hess_y - pf_full``, and compares each with exact truth from the generator.

CHART INVARIANCE. The decoder's chart is not the generator's. Every (0,2) tensor T with chart
Jacobian J is compared through its ambient lift ``J ginv T ginv J^T`` (the tangent-plane tensor,
chart-free); inner products are formed as ``tr(M_a C M_b C^T)`` with ``M = ginv T ginv`` and
``C = J_a^T J_b``, never materialising D x D matrices.

TRUTH. float64 autodiff of the generator on CPU at each anchor's own latent: J, Hess, II by
``ppf.decoder_geometry(G, z)``; covariant label Hessian ``d^2 f - Gamma^k d_k f``.

DECISION RULES (fixed here, before any number is printed): on the noise-free small fixture,
median pf_full cosine >= TENSOR_COS_PASS and mismatch-norm Spearman >= MISMATCH_RHO_PASS.

NOT PRE-REGISTERED FOR THE PAPER, GATES NOTHING.

Usage:
    python curvature-experiment/runners/10_tensor_fidelity_run.py --mode small --n 4000 --noise 0 --seed 0 --threads 8
    python curvature-experiment/runners/10_tensor_fidelity_run.py --mode full --noise 0.25 --seed 0 --threads 8 \\
        --device cuda --deterministic
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_SPLIT_PATH = DIAGNOSTICS_ROOT / "09_physics_probe_facing_split_run.py"
_spec = importlib.util.spec_from_file_location("physics_probe_facing_split_run", _SPLIT_PATH)
_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split)
ppf, adj, runner = _split.ppf, _split.adj, _split.runner
local_quadratics, split_columns = _split.local_quadratics, _split.split_columns

from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "tensor-fidelity"
DEFAULT_RECORD_PATH = Path(os.environ.get("EFFDIM_CACHE_DIR") or NOTEBOOK_ROOT / ".cache") / "10_tensor_fidelity.jsonl"
TENSORS = ("pf_full", "pf_tan", "hess_y", "mismatch")

# Decision rules -- fixed in source before any run.
TENSOR_COS_PASS = 0.8
MISMATCH_RHO_PASS = 0.7
MISMATCH_FLOOR = 1e-6
"""Mismatch cosine is undefined where |mismatch_true| < MISMATCH_FLOOR * |hess_y_true| (label lam0)."""


def _M(T: np.ndarray, geo: Dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("bij,bjk,bkl->bil", geo["ginv"], T, geo["ginv"])


def ambient_inner(Ta: np.ndarray, geo_a: Dict[str, np.ndarray], Tb: np.ndarray, geo_b: Dict[str, np.ndarray]) -> np.ndarray:
    """<J_a M_a J_a^T, J_b M_b J_b^T>_F per anchor, M = ginv T ginv, via C = J_a^T J_b (d x d)."""
    C = np.einsum("bai,baj->bij", geo_a["J"], geo_b["J"])
    return np.einsum("bij,bjk,bkl,bil->b", _M(Ta, geo_a), C, _M(Tb, geo_b), C)


def tensor_cosine(Ta, geo_a, Tb, geo_b) -> np.ndarray:
    ab = ambient_inner(Ta, geo_a, Tb, geo_b)
    aa = ambient_inner(Ta, geo_a, Ta, geo_a); bb = ambient_inner(Tb, geo_b, Tb, geo_b)
    return ab / np.sqrt(np.maximum(aa * bb, 1e-300))


def relative_error(T_est, geo_est, T_true, geo_true, ref_norm=None) -> np.ndarray:
    ee = ambient_inner(T_est, geo_est, T_est, geo_est)
    tt = ambient_inner(T_true, geo_true, T_true, geo_true)
    et = ambient_inner(T_est, geo_est, T_true, geo_true)
    diff = np.sqrt(np.maximum(ee + tt - 2 * et, 0.0))
    ref = np.sqrt(np.maximum(tt, 0.0)) if ref_norm is None else ref_norm
    return diff / np.maximum(ref, 1e-300)


def probe_facing_tensors(geo: Dict[str, np.ndarray], w: np.ndarray) -> Dict[str, np.ndarray]:
    """The split runner's pf_full / pf_tan formulas (split_columns), returning the tensors, not norms."""
    J, ginv, II, x = geo["J"], geo["ginv"], geo["II"], geo["image"]
    xhat = x / np.linalg.norm(x, axis=1, keepdims=True)
    wT = np.einsum("bai,a->bi", J, w)
    w_N = w[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, wT)
    II_rad = np.einsum("baij,ba->bij", II, xhat)
    II_tan = II - np.einsum("ba,bij->baij", xhat, II_rad)
    return {"w_N": w_N, "pf_full": np.einsum("baij,ba->bij", II, w_N), "pf_tan": np.einsum("baij,ba->bij", II_tan, w_N)}


def truth_geometry(G: torch.nn.Module, z_anchor: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact geometry of the generator at the anchors' own latents (float64, CPU)."""
    geo = ppf.decoder_geometry(G, torch.as_tensor(z_anchor, dtype=torch.float64))
    xhat = geo["image"] / np.linalg.norm(geo["image"], axis=1, keepdims=True)
    dev = float(np.max(np.abs(np.einsum("ba,ba->b", geo["H"], xhat) + G.d)))
    if not dev < 1e-8:
        raise AssertionError(f"exactness check failed: max|H_rad + d| = {dev:.3e} (expected < 1e-8)")
    geo["max_abs_H_rad_plus_d"] = dev
    return geo


def _norm(T, geo) -> np.ndarray:
    return np.sqrt(np.maximum(ambient_inner(T, geo, T, geo), 0.0))


def score_tensors(est: Dict[str, np.ndarray], truth: Dict[str, np.ndarray],
                  geo_est: Dict[str, np.ndarray], geo_true: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per tensor: cosine p25/p50, relative-error p50 and valid-anchor count; NaN anchors excluded.
    The mismatch is normalised by |hess_y_true| and its cosine dropped where the true mismatch is ~0."""
    out: Dict[str, Any] = {}
    hess_ref = _norm(truth["hess_y"], geo_true)
    for t in TENSORS:
        cos = tensor_cosine(est[t], geo_est, truth[t], geo_true)
        rel = relative_error(est[t], geo_est, truth[t], geo_true, ref_norm=hess_ref if t == "mismatch" else None)
        if t == "mismatch":
            cos = np.where(_norm(truth[t], geo_true) < MISMATCH_FLOOR * hess_ref, np.nan, cos)
        ok = np.isfinite(cos)
        out[f"n_valid_{t}"] = int(ok.sum())
        out[f"cos_{t}_p25"] = float(np.percentile(cos[ok], 25)) if ok.any() else float("nan")
        out[f"cos_{t}_p50"] = float(np.median(cos[ok])) if ok.any() else float("nan")
        okr = np.isfinite(rel)
        out[f"relerr_{t}_p50"] = float(np.median(rel[okr])) if okr.any() else float("nan")
    return out
```

Note: if `ppf.decoder_geometry` fails on the generator under `vmap` (it should not: `G.decode` is batch-generic), stop and report BLOCKED with the traceback — do not reimplement the geometry.

- [ ] **Step 4: Run to verify they pass**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd $R && git add curvature-experiment/runners/10_tensor_fidelity_run.py curvature-experiment/tests/test_tensor_fidelity.py
git commit -m "feat(tensor-fidelity): chart-invariant tensor comparison and exact truth geometry

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Labels and the covariant label Hessian

**Files:**
- Modify: `curvature-experiment/runners/10_tensor_fidelity_run.py` (append after `score_tensors`)
- Modify: `curvature-experiment/tests/test_tensor_fidelity.py` (append)

**Interfaces:**
- Consumes: `truth_geometry`, `probe_facing_tensors` (Task 1).
- Produces:
  - `LAMBDAS = (0.0, 0.5, 1.0, 2.0)`, `BUMP_WIDTH = 0.8`, `LABEL_SEED = 20260930`.
  - `label_name(lam: float) -> str` → `"lam0"`, `"lam0.5"`, `"lam1"`, `"lam2"`.
  - `make_labels(G, z_all: np.ndarray) -> {"f": {name: callable(torch (b,d)) -> (b,)}, "y": {name: np.ndarray (n,)}, "w0": np.ndarray (D,), "scales": {"ambient": float, "bump": float}}` with names `lin, nonlin, lam0, lam0.5, lam1, lam2`.
  - `covariant_hessian(f, geo, z_anchor: np.ndarray) -> np.ndarray (b,d,d)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_labels_names_and_scaling():
    G = _small_generator()
    z = tf.adj.draw_latents(2000, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=12)
    lab = tf.make_labels(G, z)
    assert list(lab["y"]) == ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]
    assert np.std(lab["y"]["lam0"]) == pytest.approx(1.0, rel=1e-9)     # ambient term at unit std
    for name, y in lab["y"].items():
        assert y.shape == (2000,) and np.all(np.isfinite(y)), name


def test_covariant_hessian_identity_at_lambda_zero():
    """At lambda = 0 the label is <w0, G(z)>/s: its covariant Hessian is <w0, II>/s = <w0_N, II>/s."""
    G = _small_generator()
    z = tf.adj.draw_latents(500, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=13)
    lab = tf.make_labels(G, z)
    za = z[:8]
    geo = tf.truth_geometry(G, za)
    got = tf.covariant_hessian(lab["f"]["lam0"], geo, za)
    want = tf.probe_facing_tensors(geo, lab["w0"])["pf_full"] / lab["scales"]["ambient"]
    np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-8)


def test_label_functions_match_values():
    G = _small_generator()
    z = tf.adj.draw_latents(300, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=14)
    lab = tf.make_labels(G, z)
    zt = torch.as_tensor(z[:5], dtype=torch.float64)
    for name, f in lab["f"].items():
        with torch.no_grad():
            np.testing.assert_allclose(f(zt).numpy(), lab["y"][name][:5], rtol=1e-12, err_msg=name)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py -k "labels or covariant or label_functions"`
Expected: FAIL — `AttributeError: module ... has no attribute 'make_labels'`.

- [ ] **Step 3: Implement** (append to the runner, and add `from torch.func import hessian, jacrev, vmap  # noqa: E402` next to the other imports)

```python
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
BUMP_WIDTH = 0.8
LABEL_SEED = 20260930


def label_name(lam: float) -> str:
    return f"lam{lam:g}"


def _batched(f, zt: torch.Tensor, batch: int = 8192) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([f(zt[s:s + batch]).numpy() for s in range(0, zt.shape[0], batch)])


def make_labels(G: torch.nn.Module, z_all: np.ndarray) -> Dict[str, Any]:
    """lin, nonlin (the smoke fixture's shapes) and y_lambda = <w0, G(z)>/s_a + lambda * h(z)/s_h."""
    rng = np.random.default_rng(LABEL_SEED)
    d, D = G.d, G.D
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)  # noqa: E731
    a1 = rng.standard_normal(d); a1 /= np.linalg.norm(a1)
    a2 = rng.standard_normal(d)
    w0 = rng.standard_normal(D); w0 /= np.linalg.norm(w0)
    centres = rng.standard_normal((2, d)) * 0.5
    a1t, a2t, w0t, ct = t(a1), t(a2), t(w0), t(centres)

    def lin(z): return z @ a1t
    def nonlin(z): return torch.sin(2 * (z @ a1t)) + (z @ a2t) ** 2
    def ambient(z): return G.decode(z) @ w0t
    def bump(z): return torch.exp(-((z[:, None, :] - ct[None, :, :]) ** 2).sum(-1) / (2 * BUMP_WIDTH ** 2)).sum(-1)

    zt = t(z_all)
    s_a = float(np.std(_batched(ambient, zt))); s_h = float(np.std(_batched(bump, zt)))
    f: Dict[str, Any] = {"lin": lin, "nonlin": nonlin}
    for lam in LAMBDAS:
        f[label_name(lam)] = (lambda lam_: (lambda z: ambient(z) / s_a + lam_ * bump(z) / s_h))(lam)
    y = {name: _batched(fn, zt) for name, fn in f.items()}
    return {"f": f, "y": y, "w0": w0, "scales": {"ambient": s_a, "bump": s_h}}


def covariant_hessian(f, geo: Dict[str, np.ndarray], z_anchor: np.ndarray) -> np.ndarray:
    """nabla^2 f = d^2 f - Gamma^k d_k f, Gamma^k_ij = ginv^{kl} J_l . d_i d_j G (float64 autodiff)."""
    zt = torch.as_tensor(z_anchor, dtype=torch.float64)
    f_one = lambda z1: f(z1.unsqueeze(0)).squeeze(0)  # noqa: E731
    grad = vmap(jacrev(f_one))(zt).detach().numpy()
    hess = vmap(hessian(f_one))(zt).detach().numpy()
    Gamma = np.einsum("bkl,bal,baij->bkij", geo["ginv"], geo["J"], geo["Hess"])
    return hess - np.einsum("bkij,bk->bij", Gamma, grad)
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd $R && git add curvature-experiment/runners/10_tensor_fidelity_run.py curvature-experiment/tests/test_tensor_fidelity.py
git commit -m "feat(tensor-fidelity): labels with a controlled mismatch and the covariant label Hessian

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: One configuration end to end, CLI and records

**Files:**
- Modify: `curvature-experiment/runners/10_tensor_fidelity_run.py` (append)
- Modify: `curvature-experiment/tests/test_tensor_fidelity.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2; `local_quadratics`, `split_columns`, `ppf.fit_decoder`, `ppf.decoder_geometry`, `ppf._spearman`, `ppf._append`, `adj.*`, `pcp.*`.
- Produces:
  - `FIXTURE_SEED = 20260905`
  - `config_for(mode: str, n: int | None) -> dict` with keys `d, D, n, k, n_anchors, a, bump_widths, bump_amps, scale_choices, scale_probs`. `small`: `adj.SMOKE` with `n` (default 4000), `k = 128 if n <= 4000 else n // 32`, `n_anchors = 64`. `paper`: `adj.FIXTURE` unchanged.
  - `run_config(cfg, noise_frac, seed, max_epochs, device, mode, record_path) -> list[dict]` (the result rows it appended).
  - `build_parser()`, `main()`.
  - Record rows (JSONL): an `environment` row, then one `result` row per label with keys `experiment, row="result", mode, n, k, n_anchors, noise_frac, seed, label, lam (float or None), device, var_explained, max_abs_H_rad_plus_d, rho_mismatch, rho_align, timestamp` plus every key from `score_tensors`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import json


def test_config_for():
    c = tf.config_for("small", 16000)
    assert (c["d"], c["D"], c["n"], c["k"], c["n_anchors"]) == (4, 64, 16000, 500, 64)
    assert tf.config_for("small", None)["k"] == 128
    p = tf.config_for("paper", None)
    assert (p["d"], p["D"], p["n"], p["k"], p["n_anchors"]) == (16, 768, 86471, 2048, 512)


def test_end_to_end_smoke(tmp_path):
    rec = tmp_path / "10_tensor_fidelity.jsonl"
    rows = tf.run_config(tf.config_for("small", 4000), 0.0, 0, 3, "cpu", "small", rec)
    assert [r["label"] for r in rows] == ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]
    on_disk = [json.loads(l) for l in rec.read_text().splitlines()]
    assert [r["label"] for r in on_disk] == [r["label"] for r in rows]
    for r in rows:
        for t in ("pf_full", "pf_tan", "hess_y"):
            assert np.isfinite(r[f"cos_{t}_p50"]) and -1.0 <= r[f"cos_{t}_p50"] <= 1.0, (r["label"], t)
        assert np.isfinite(r["rho_mismatch"]) and np.isfinite(r["rho_align"])
        assert r["max_abs_H_rad_plus_d"] < 1e-8


def test_refuses_production_record(tmp_path, monkeypatch):
    bad = tmp_path / f"{tf.ppf.PRODUCTION_STEMS[0]}_x.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "small", "--threads", "8", "--record-path", str(bad)])
    with pytest.raises(SystemExit, match="refusing"):
        tf.main()
    assert not bad.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU refusal")
def test_cuda_without_gpu_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "small", "--threads", "8", "--device", "cuda",
                                      "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="CUDA is not available"):
        tf.main()
```

Note on `--threads 8` in these tests: `runner._THREADS` was fixed at import from the test's `sys.argv` (default 8 when `--threads` is absent — verify with `tf.runner._THREADS`; if the default differs, use that value in both tests).

- [ ] **Step 2: Run to verify they fail**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py -k "config_for or end_to_end or refuses"`
Expected: FAIL — `AttributeError: ... 'config_for'`.

- [ ] **Step 3: Implement** (append; add `import argparse`, `import time`, `from datetime import datetime, timezone` and `from sklearn.linear_model import Ridge` to the imports with `# noqa: E402`)

```python
FIXTURE_SEED = 20260905


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_for(mode: str, n) -> Dict[str, Any]:
    if mode == "paper":
        return dict(adj.FIXTURE)
    n = int(n) if n is not None else int(adj.SMOKE["n"])
    return {**adj.SMOKE, "n": n, "k": 128 if n <= 4000 else n // 32, "n_anchors": 64}


def run_config(cfg: Dict[str, Any], noise_frac: float, seed: int, max_epochs: int, device: str, mode: str,
               record_path: Path) -> list:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print(f"\n[config] mode={mode} d={d} D={D} n={n} k={k} anchors={n_anchors} noise={noise_frac} seed={seed}", flush=True)
    G = adj.InSphereGenerator(d, D, cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=FIXTURE_SEED)
    z = adj.draw_latents(n, d, cfg["scale_choices"], cfg["scale_probs"], seed=FIXTURE_SEED + 1 + 1000 * seed)
    X0 = adj.generate_points(G, z)
    a = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
    panel = pcp.knn_panel(X0, a, k)
    if noise_frac > 0:
        # the adjudication runner's patch-noise protocol, at a chosen fraction of the median patch radius
        sigma = noise_frac * float(np.median(panel["distances"][:, -1])) / np.sqrt(D)
        X = X0 + np.random.default_rng(FIXTURE_SEED + 2 + 1000 * seed).standard_normal((n, D)) * sigma
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        panel = pcp.knn_panel(X, a, k)
    else:
        X = X0
    labels = make_labels(G, z)
    truth = truth_geometry(G, z[a])

    default_init = pcp.TORCH_INIT_SEED
    pcp.TORCH_INIT_SEED = int(default_init) + int(seed)
    try:
        fit = ppf.fit_decoder(X, d, D, max_epochs, device=device)
    finally:
        pcp.TORCH_INIT_SEED = default_init
    with torch.no_grad():
        z_hat = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long, device=fit["x64"].device)])
    est = ppf.decoder_geometry(fit["curvature_model"], z_hat)
    print(f"[fit] var_explained={fit['var_explained']:.4f} in {fit['wallclock_fit_s']:.0f}s", flush=True)

    rows = []
    for name, y in labels["y"].items():
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X, y)
        w, b0 = ridge.coef_.astype(np.float64), float(ridge.intercept_)
        lq = local_quadratics(X, a, panel["indices"], est, {"y": y, "p": X @ w}, pcp.MIN_FINITE_NEIGHBOURS)
        pf_e, pf_t = probe_facing_tensors(est, w), probe_facing_tensors(truth, w)
        hess_e = lq["hess"]["y"]
        hess_t = covariant_hessian(labels["f"][name], truth, z[a])
        T_e = {"pf_full": pf_e["pf_full"], "pf_tan": pf_e["pf_tan"], "hess_y": hess_e, "mismatch": hess_e - pf_e["pf_full"]}
        T_t = {"pf_full": pf_t["pf_full"], "pf_tan": pf_t["pf_tan"], "hess_y": hess_t, "mismatch": hess_t - pf_t["pf_full"]}
        c_e = split_columns(est, w, b0, hess_e, lq["hess"]["p"], d)["cols"]
        c_t = split_columns(truth, w, b0, hess_t, pf_t["pf_full"], d)["cols"]
        row = {"experiment": EXPERIMENT, "row": "result", "mode": mode, "n": n, "k": k, "n_anchors": n_anchors,
               "noise_frac": float(noise_frac), "seed": int(seed), "label": name,
               "lam": next((lam for lam in LAMBDAS if label_name(lam) == name), None), "device": device,
               "var_explained": float(fit["var_explained"]), "max_abs_H_rad_plus_d": truth["max_abs_H_rad_plus_d"],
               "rho_mismatch": ppf._spearman(c_e["hess_mismatch_dec"], c_t["hess_mismatch_dec"]),
               "rho_align": ppf._spearman(c_e["align_cos_full"], c_t["align_cos_full"]),
               **score_tensors(T_e, T_t, est, truth), "timestamp": _utc_now()}
        ppf._append(row, record_path)
        rows.append(row)
        print(f"  {name:8s} cos pf_full {row['cos_pf_full_p50']:+.3f} pf_tan {row['cos_pf_tan_p50']:+.3f} "
              f"hess_y {row['cos_hess_y_p50']:+.3f} mismatch {row['cos_mismatch_p50']:+.3f} | "
              f"rho mismatch {row['rho_mismatch']:+.3f} align {row['rho_align']:+.3f}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["small", "paper"], required=True)
    p.add_argument("--n", type=int, default=None, help="small mode only; default 4000")
    p.add_argument("--noise", type=float, default=0.0, help="fraction of the median k-NN patch radius")
    p.add_argument("--seed", type=int, default=0, help="latent draw, noise and decoder init")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8")
    p.add_argument("--max-epochs", type=int, default=None, help="test override only; default pcp.MAX_EPOCHS")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    return p


def main() -> None:
    args = build_parser().parse_args()
    record_path = Path(args.record_path).resolve()
    for stem in ppf.PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available on this machine")
    gpu_name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None
    cfg = config_for(args.mode, args.n)
    max_epochs = args.max_epochs if args.max_epochs is not None else pcp.MAX_EPOCHS
    print(f"record -> {record_path}\nNOT PRE-REGISTERED FOR THE PAPER; GATES NOTHING.\n"
          f"DECISION RULE: noise-free small fixture, median pf_full cosine >= {TENSOR_COS_PASS} and "
          f"mismatch Spearman >= {MISMATCH_RHO_PASS}.")
    ppf._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
                 "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "cfg": cfg,
                 "noise_frac": args.noise, "seed": args.seed, "max_epochs": max_epochs,
                 "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0],
                 "device": args.device, "deterministic": args.deterministic, "gpu_name": gpu_name,
                 "cuda_version": torch.version.cuda}, record_path)
    t0 = time.monotonic()
    run_config(cfg, args.noise, args.seed, max_epochs, args.device, args.mode, record_path)
    print(f"DONE in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py`
Expected: 13 passed (the CUDA refusal test is skipped on a GPU machine).

- [ ] **Step 5: Run the CLI once by hand**

Run: `cd $R && $PY curvature-experiment/runners/10_tensor_fidelity_run.py --mode small --n 4000 --noise 0 --seed 0 --threads 8 --max-epochs 3 --record-path $CLAUDE_JOB_DIR/tmp/tf-smoke.jsonl` (any scratch path outside the repo)
Expected: six label lines with finite cosines, then `DONE in <n>s`.

- [ ] **Step 6: Commit**

```bash
cd $R && git add curvature-experiment/runners/10_tensor_fidelity_run.py curvature-experiment/tests/test_tensor_fidelity.py
git commit -m "feat(tensor-fidelity): one configuration end to end with records and CLI

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Report

**Files:**
- Create: `curvature-experiment/runners/10_tensor_fidelity_report.py`
- Modify: `curvature-experiment/tests/test_tensor_fidelity.py` (append)

**Interfaces:**
- Consumes: record rows from Task 3; `TENSOR_COS_PASS`, `MISMATCH_RHO_PASS`, `TENSORS`, `LAMBDAS`, `label_name` from the runner (imported through `_load`-style importlib).
- Produces:
  - `load_rows(paths: list[Path]) -> list[dict]` — result rows only, last row wins per key `(mode, n, noise_frac, seed, label)`, returned sorted by that key.
  - `pass_lines(rows) -> {"n": int, "per_label": {label: {"cos": float, "rho": float | None, "pass": bool}}, "pass": bool}` — small mode, `noise_frac == 0`, the largest `n` present; per label the median over seeds; `rho` is `None` and not tested for `lam0` (true mismatch ~ 0 by construction); overall pass iff every label passes.
  - `write_report(rows, out_dir: Path) -> None` — writes `REPORT.md` and `fig_tensor_fidelity.png`.
  - CLI: `--record-path` (one or more), `--out-dir` (default `curvature-experiment/results/tensor-fidelity`).

- [ ] **Step 1: Write the failing tests** (append)

```python
rep = _load("10_tensor_fidelity_report.py")


def _row(n, noise, seed, label, cos, rho, mode="small"):
    r = {"experiment": "tensor-fidelity", "row": "result", "mode": mode, "n": n, "noise_frac": noise, "seed": seed,
         "label": label, "rho_mismatch": rho, "rho_align": rho, "var_explained": 0.99}
    for t in tf.TENSORS:
        r.update({f"cos_{t}_p50": cos, f"cos_{t}_p25": cos, f"relerr_{t}_p50": 0.1, f"n_valid_{t}": 64})
    return r


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


LABELS = ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]


def test_report_dedupes_reruns(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, [{"row": "environment"}, _row(4000, 0.0, 0, "lin", 0.1, 0.1), _row(4000, 0.0, 0, "lin", 0.9, 0.9)])
    rows = rep.load_rows([p])
    assert len(rows) == 1 and rows[0]["cos_pf_full_p50"] == 0.9


def test_pass_lines_use_largest_n_and_skip_lam0_rho(tmp_path):
    rows = [_row(4000, 0.0, s, lab, 0.1, 0.1) for s in range(3) for lab in LABELS]            # small n fails
    rows += [_row(64000, 0.0, s, lab, 0.9, (0.0 if lab == "lam0" else 0.8)) for s in range(3) for lab in LABELS]
    rows += [_row(64000, 0.5, s, lab, 0.1, 0.1) for s in range(3) for lab in LABELS]           # noisy ignored
    pl = rep.pass_lines(rows)
    assert pl["n"] == 64000 and pl["pass"] is True
    assert pl["per_label"]["lam0"]["rho"] is None


def test_pass_lines_fail(tmp_path):
    rows = [_row(64000, 0.0, s, lab, 0.9, 0.5) for s in range(3) for lab in LABELS]
    assert rep.pass_lines(rows)["pass"] is False


def test_write_report(tmp_path):
    rows = [_row(n, nz, s, lab, 0.9, 0.8) for n in (4000, 16000) for nz in (0.0, 0.25) for s in range(2) for lab in LABELS]
    rows += [_row(86471, nz, 0, lab, 0.85, 0.75, mode="paper") for nz in (0.0, 0.25) for lab in LABELS]
    rep.write_report(rows, tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    assert text.startswith("# Tensor fidelity") and "PASS" in text and "## Paper scale" in text
    assert (tmp_path / "fig_tensor_fidelity.png").stat().st_size > 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py -k "report or pass_lines"`
Expected: collection error — `10_tensor_fidelity_report.py` does not exist.

- [ ] **Step 3: Implement** `curvature-experiment/runners/10_tensor_fidelity_report.py`

```python
"""Tensor fidelity report: records -> REPORT.md + fig_tensor_fidelity.png.

Usage:
    python curvature-experiment/runners/10_tensor_fidelity_report.py \\
        --record-path curvature-experiment/results/tensor-fidelity/records/*.jsonl
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = DIAGNOSTICS_ROOT.parent / "results" / "tensor-fidelity"
_spec = importlib.util.spec_from_file_location("tensor_fidelity_run", DIAGNOSTICS_ROOT / "10_tensor_fidelity_run.py")
tf = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
try:
    _spec.loader.exec_module(tf)
finally:
    sys.argv = _argv

KEY = ("mode", "n", "noise_frac", "seed", "label")
LABELS = ["lin", "nonlin"] + [tf.label_name(lam) for lam in tf.LAMBDAS]


def load_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("row") == "result":
                by_key[tuple(r[k] for k in KEY)] = r
    return [by_key[k] for k in sorted(by_key, key=lambda k: tuple(str(v) for v in k))]


def _med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.median(v)) if v else float("nan")


def pass_lines(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean = [r for r in rows if r["mode"] == "small" and r["noise_frac"] == 0.0]
    n = max(r["n"] for r in clean)
    per: Dict[str, Any] = {}
    for lab in LABELS:
        rs = [r for r in clean if r["n"] == n and r["label"] == lab]
        if not rs:
            continue
        cos = _med(rs, "cos_pf_full_p50")
        rho = None if lab == tf.label_name(0.0) else _med(rs, "rho_mismatch")
        ok = cos >= tf.TENSOR_COS_PASS and (rho is None or rho >= tf.MISMATCH_RHO_PASS)
        per[lab] = {"cos": cos, "rho": rho, "pass": bool(ok)}
    return {"n": n, "per_label": per, "pass": bool(per) and all(v["pass"] for v in per.values())}


def _f(v) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:+.3f}"


def _table(rows: List[Dict[str, Any]], mode: str) -> List[str]:
    L = ["| n | noise | label | cos pf_full | cos pf_tan | cos hess_y | cos mismatch | rho mismatch | rho align | var. expl. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rs = [r for r in rows if r["mode"] == mode]
    for n in sorted({r["n"] for r in rs}):
        for nz in sorted({r["noise_frac"] for r in rs}):
            for lab in LABELS:
                g = [r for r in rs if r["n"] == n and r["noise_frac"] == nz and r["label"] == lab]
                if g:
                    L.append(f"| {n} | {nz:g} | {lab} | " + " | ".join(_f(_med(g, f"cos_{t}_p50")) for t in tf.TENSORS)
                             + f" | {_f(_med(g, 'rho_mismatch'))} | {_f(_med(g, 'rho_align'))} | {_med(g, 'var_explained'):.3f} |")
    return L


def write_report(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = pass_lines(rows)
    L = ["# Tensor fidelity", "",
         f"Pass lines (fixed in source before any run): noise-free small fixture at n={pl['n']}, per label the median over "
         f"seeds of the median pf_full cosine >= {tf.TENSOR_COS_PASS} and mismatch-norm Spearman >= {tf.MISMATCH_RHO_PASS} "
         f"(Spearman not tested for {tf.label_name(0.0)}, whose true mismatch is ~0 by construction).", "",
         f"**Overall: {'PASS' if pl['pass'] else 'FAIL'}**", "", "| label | cos pf_full | rho mismatch | result |", "|---|---|---|---|"]
    for lab, v in pl["per_label"].items():
        L.append(f"| {lab} | {_f(v['cos'])} | {_f(v['rho'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += ["", "## Small fixture (d=4, D=64), medians over seeds", ""] + _table(rows, "small")
    L += ["", "## Paper scale (d=16, D=768, n=86,471)", ""] + _table(rows, "paper") + [""]
    (out_dir / "REPORT.md").write_text("\n".join(L))

    small = [r for r in rows if r["mode"] == "small"]
    ns = sorted({r["n"] for r in small})
    fig, axes = plt.subplots(1, max(len(ns), 1), figsize=(3.2 * max(len(ns), 1), 2.8), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        nzs = sorted({r["noise_frac"] for r in small if r["n"] == n})
        for t in tf.TENSORS:
            ax.plot(nzs, [_med([r for r in small if r["n"] == n and r["noise_frac"] == nz], f"cos_{t}_p50") for nz in nzs],
                    marker="o", label=t)
        ax.axhline(tf.TENSOR_COS_PASS, color="0.6", lw=0.8, ls="--")
        ax.set_title(f"n={n}", fontsize=9); ax.set_xlabel("noise / patch radius")
    axes[0][0].set_ylabel("median tensor cosine"); axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tensor_fidelity.png", dpi=150, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-path", nargs="+", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    rows = load_rows([Path(x) for x in args.record_path])
    write_report(rows, Path(args.out_dir))
    print(f"{len(rows)} result rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd $R && $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests/test_tensor_fidelity.py`
Expected: 17 passed.

- [ ] **Step 5: CPU gate and full suite**

Run: `cd $R && EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests` → all pass; then `CLOSURE_WORK=$HOME/.cache/effdim-closure docs/superpowers/harness/gate.sh $R tensor-fidelity` → `GATE PASS`.

- [ ] **Step 6: Commit**

```bash
cd $R && git add curvature-experiment/runners/10_tensor_fidelity_report.py curvature-experiment/tests/test_tensor_fidelity.py
git commit -m "feat(tensor-fidelity): report with pre-registered pass lines and figure

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run the grids

No code changes. Records go to `curvature-experiment/results/tensor-fidelity/records/` (new files, committed in Task 6).

- [ ] **Step 1: Small grid, local CPU, in the background** (36 runs; the paper's 600 epochs, so expect 1–2 h)

```bash
cd $R && OUT=curvature-experiment/results/tensor-fidelity/records && mkdir -p $OUT
for n in 4000 16000 64000; do for nz in 0 0.125 0.25 0.5; do for s in 0 1 2; do
  $PY curvature-experiment/runners/10_tensor_fidelity_run.py --mode small --n $n --noise $nz --seed $s --threads 8 \
    --record-path $OUT/small.jsonl || echo "FAILED n=$n noise=$nz seed=$s"
done; done; done 2>&1 | tee $CLAUDE_JOB_DIR/tmp/tf-small.log
```
Expected: 36 `DONE` lines, no `FAILED`. `wc -l $OUT/small.jsonl` = 36 × 7 = 252.

- [ ] **Step 2: Paper scale on the pod GPU** (follow `curvature-experiment/sweep/POD_RUNBOOK.md` §1–§3 and the Global Constraints pod rules)
  1. Read `docs/remote-compute/eleutherai-pod-user-guide.md`; `timeout 30 ssh root@216.153.49.26 'sha256sum /root/user-guide.md'` equals the local sha256.
  2. Push the branch: `git -C $R push -u origin tensor-fidelity`.
  3. On the pod, in tmux: `BRANCH=tensor-fidelity bash /mnt/ssd-cluster/EffDim/repo/curvature-experiment/sweep/setup_pod.sh` → `SETUP_EXIT=0` semantics (script exits 0) and `torch ... available True`.
  4. `nvidia-smi` → pick two GPUs with no processes; `df -h /mnt/ssd-cluster` (≥ 5 GB free).
  5. In tmux window `tf-paper`, one command per GPU:

```bash
cd /mnt/ssd-cluster/EffDim/repo/curvature-experiment && mkdir -p /mnt/ssd-cluster/EffDim/tensor-fidelity
CUDA_VISIBLE_DEVICES=<gpu_a> /mnt/ssd-cluster/EffDim/venv/bin/python runners/10_tensor_fidelity_run.py --mode full --noise 0 --seed 0 \
  --threads 8 --device cuda --deterministic --record-path /mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise0.jsonl 2>&1 | tee /mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise0.log
CUDA_VISIBLE_DEVICES=<gpu_b> /mnt/ssd-cluster/EffDim/venv/bin/python runners/10_tensor_fidelity_run.py --mode full --noise 0.25 --seed 0 \
  --threads 8 --device cuda --deterministic --record-path /mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise025.jsonl 2>&1 | tee /mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise025.log
```
  Expected: each log ends `DONE in <n>s`. If `use_deterministic_algorithms` raises on an op or CUDA runs out of memory, stop and report BLOCKED with the log tail.
  6. Fetch: `rsync -a root@216.153.49.26:/mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise0.jsonl root@216.153.49.26:/mnt/ssd-cluster/EffDim/tensor-fidelity/paper_noise025.jsonl $R/curvature-experiment/results/tensor-fidelity/records/`

---

### Task 6: Report, record the result, commit

- [ ] **Step 1: Generate** — `cd $R && $PY curvature-experiment/runners/10_tensor_fidelity_report.py --record-path curvature-experiment/results/tensor-fidelity/records/*.jsonl`
  Expected: `REPORT.md` with `**Overall: PASS**` or `**Overall: FAIL**`, both tables filled, and the figure.
- [ ] **Step 2: Read-out** — in the task report state: overall result; per label the cosine and Spearman at n=64000, noise 0; how each tensor's cosine falls with noise and rises with n; paper-scale numbers next to the small-fixture ones; any label or tensor below its line. Do not tune thresholds, labels or the grid after seeing the numbers — a FAIL is a result.
- [ ] **Step 3: Final checks** — full suite, `gate.sh $R tensor-fidelity` → `GATE PASS`, `git diff --exit-code encoder-scaling -- paper/latex/main.tex`.
- [ ] **Step 4: Commit and push**

```bash
cd $R && git add curvature-experiment/results/tensor-fidelity
git commit -m "feat(results): tensor fidelity of the probe-facing tensors on the in-sphere fixture

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push
```
