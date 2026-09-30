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
from sweep import extract  # noqa: E402

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
