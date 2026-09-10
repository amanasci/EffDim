"""Rank, discrimination, calibration, reliability, and dynamic-range metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .config import (
    BAND_CEIL_MODERATE,
    BAND_CEIL_STRONG,
    BAND_PAIR_MODERATE,
    BAND_PAIR_STRONG,
    BAND_RANK_MODERATE,
    BAND_RANK_STRONG,
    BAND_REL_MODERATE,
    BAND_REL_STRONG,
    EPS,
    MIN_CLEAN_RHO_FOR_RETAINED,
    N_BOOT,
    RANK_DEGENERATE_IQR_MULT,
    TOL_TRUTH_ABS,
    TOL_TRUTH_REL,
)


def _finite_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def spearman_safe(a, b) -> float:
    a, b = _finite_pair(a, b)
    if a.size < 8 or float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def kendall_safe(a, b) -> float:
    a, b = _finite_pair(a, b)
    if a.size < 8 or float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def pearson_safe(a, b) -> float:
    a, b = _finite_pair(a, b)
    if a.size < 8 or float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rank_rmse(est, truth) -> float:
    est, truth = _finite_pair(est, truth)
    if est.size < 4:
        return float("nan")
    re = _ranks(est)
    rt = _ranks(truth)
    return float(np.sqrt(np.mean((re - rt) ** 2)))


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(counts.size, dtype=np.float64)
        np.add.at(sums, inv, ranks)
        ranks = sums[inv] / counts[inv]
    return ranks


def pairwise_ordering_accuracy(est, truth) -> float:
    """P(sign(est_i-est_j) matches sign(truth_i-truth_j) | truth_i != truth_j)."""
    est, truth = _finite_pair(est, truth)
    n = est.size
    if n < 3:
        return float("nan")
    dt = truth[:, None] - truth[None, :]
    de = est[:, None] - est[None, :]
    iu = np.triu_indices(n, k=1)
    dt, de = dt[iu], de[iu]
    m = np.abs(dt) > 1e-15
    if int(m.sum()) < 1:
        return float("nan")
    return float(np.mean(np.sign(de[m]) == np.sign(dt[m])))


def chance_pairwise() -> float:
    return 0.5


def dynamic_range(truth, *, analytic_constant: bool = False) -> dict:
    x = np.asarray(truth, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    med = float(np.median(np.abs(x))) if x.size else float("nan")
    mean = float(np.mean(x)) if x.size else float("nan")
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    q25, q75 = (np.percentile(x, [25, 75]) if x.size else (np.nan, np.nan))
    iqr = float(q75 - q25) if x.size else float("nan")
    cv = sd / (abs(mean) + EPS)
    r_iqr = iqr / (med + EPS)
    tol = max(TOL_TRUTH_ABS, TOL_TRUTH_REL * max(med, abs(mean), 1e-12))
    degenerate = bool(analytic_constant) or (np.isfinite(iqr) and iqr < RANK_DEGENERATE_IQR_MULT * tol)
    return {
        "n": int(x.size),
        "mean": mean,
        "sd": sd,
        "median_abs": med,
        "iqr": iqr,
        "cv": float(cv),
        "r_iqr": float(r_iqr),
        "tol": float(tol),
        "analytic_constant": bool(analytic_constant),
        "rank_target_degenerate": bool(degenerate),
    }


def retained_ratio(rho_stress: float, rho_clean: float) -> float:
    if not np.isfinite(rho_stress) or not np.isfinite(rho_clean):
        return float("nan")
    if abs(rho_clean) < MIN_CLEAN_RHO_FOR_RETAINED:
        return float("nan")
    return float(rho_stress / rho_clean)


def delta_rho(rho_stress: float, rho_clean: float) -> float:
    if not np.isfinite(rho_stress) or not np.isfinite(rho_clean):
        return float("nan")
    return float(rho_stress - rho_clean)


def quartile_discrimination(est, truth) -> dict:
    est, truth = _finite_pair(est, truth)
    n = est.size
    out = {
        "n": int(n),
        "top_precision": float("nan"),
        "top_recall": float("nan"),
        "top_jaccard": float("nan"),
        "p_high_above_low": float("nan"),
        "roc_auc_top_bottom": float("nan"),
        "chance_precision": 0.25,
        "chance_recall": 0.25,
        "chance_jaccard": 0.25 / (0.25 + 0.75),  # 0.25 if independent 25% sets
        "chance_p_high_above_low": 0.5,
        "chance_roc_auc": 0.5,
        "n_high": 0,
        "n_low": 0,
    }
    if n < 16:
        return out
    t75, t25 = np.percentile(truth, [75, 25])
    e75 = np.percentile(est, 75)
    high_t = truth >= t75
    low_t = truth <= t25
    high_e = est >= e75
    nh = int(high_t.sum())
    nl = int(low_t.sum())
    ne = int(high_e.sum())
    inter = int((high_e & high_t).sum())
    union = int((high_e | high_t).sum())
    out["n_high"] = nh
    out["n_low"] = nl
    out["top_precision"] = float(inter / max(ne, 1))
    out["top_recall"] = float(inter / max(nh, 1))
    out["top_jaccard"] = float(inter / max(union, 1))
    # P(est_high > est_low) over truth-defined groups
    eh, el = est[high_t], est[low_t]
    if eh.size and el.size:
        # Mann–Whitney U / (n_h n_l)
        wins = float(np.mean(eh[:, None] > el[None, :]))
        ties = float(np.mean(eh[:, None] == el[None, :]))
        out["p_high_above_low"] = wins + 0.5 * ties
        scores = np.concatenate([eh, el])
        labels = np.concatenate([np.ones(eh.size), np.zeros(el.size)])
        out["roc_auc_top_bottom"] = _roc_auc(scores, labels)
    # chance Jaccard for two independent 25% subsets of n is 1/7
    out["chance_jaccard"] = 1.0 / 7.0
    return out


def _roc_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = scores[labels > 0.5]
    neg = scores[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = float(np.mean(pos[:, None] > neg[None, :]))
    ties = float(np.mean(pos[:, None] == neg[None, :]))
    return wins + 0.5 * ties


def calibration_split(n: int, seed: int) -> np.ndarray:
    """Frozen boolean mask: True = calibration-fit half of anchors."""
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    mask = np.zeros(n, dtype=bool)
    mask[idx[: n // 2]] = True
    return mask


def fit_global_scale(est, truth, train_mask) -> float:
    e = np.asarray(est, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    m = train_mask & np.isfinite(e) & np.isfinite(t) & (np.abs(t) > 1e-12)
    if int(m.sum()) < 4:
        return float("nan")
    return float(np.median(e[m] / t[m]))


def apply_scale(est, factor) -> np.ndarray:
    e = np.asarray(est, dtype=np.float64)
    if not np.isfinite(factor):
        return e.copy()
    return e / factor if abs(factor) > 1e-18 else e.copy()


def calibration_report(est, truth, train_mask) -> dict:
    e = np.asarray(est, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=np.float64).ravel()
    factor = fit_global_scale(e, t, train_mask)
    test = (~train_mask) & np.isfinite(e) & np.isfinite(t)
    raw_ratio = float(np.median(e[test] / np.clip(t[test], 1e-18, None))) if test.sum() else float("nan")
    e_cal = apply_scale(e, factor)
    rel_abs = float(np.median(np.abs(e_cal[test] - t[test]) / np.maximum(np.abs(t[test]), 1e-12))) if test.sum() else float("nan")
    rel_sq = float(np.median(((e_cal[test] - t[test]) / np.maximum(np.abs(t[test]), 1e-12)) ** 2)) if test.sum() else float("nan")
    # log-scale slope on test half
    slope = float("nan")
    if int(test.sum()) >= 8:
        lt = np.log(np.maximum(np.abs(t[test]), 1e-12))
        le = np.log(np.maximum(np.abs(e[test]), 1e-12))
        if float(np.std(lt)) > 1e-12 and float(np.std(le)) > 1e-12:
            slope = float(np.polyfit(lt, le, 1)[0])
    rho_raw = spearman_safe(e[test], t[test]) if test.sum() else float("nan")
    rho_cal = spearman_safe(e_cal[test], t[test]) if test.sum() else float("nan")
    return {
        "global_scale_factor": factor,
        "median_est_over_truth": raw_ratio,
        "log_slope": slope,
        "rel_abs_err_calibrated": rel_abs,
        "rel_sq_err_calibrated": rel_sq,
        "spearman_test_raw": rho_raw,
        "spearman_test_calibrated": rho_cal,
        "n_train": int((train_mask & np.isfinite(e) & np.isfinite(t)).sum()),
        "n_test": int(test.sum()),
    }


def residualize_linear(y, *controls) -> np.ndarray:
    """OLS residuals of y on 1 + controls. Ranks should be passed in if rank-residualizing."""
    y = np.asarray(y, dtype=np.float64).ravel()
    cols = [np.asarray(c, dtype=np.float64).ravel() for c in controls]
    n = y.size
    X = np.column_stack([np.ones(n)] + cols)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full(n, np.nan)
    if int(m.sum()) < X.shape[1] + 2:
        return out
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    out[m] = y[m] - X[m] @ beta
    return out


def attenuation_ceiling(r_rel: float, rho_truth: float) -> dict:
    ok = bool(np.isfinite(r_rel) and r_rel > 0.0 and np.isfinite(rho_truth))
    r_max = float(np.sqrt(r_rel)) if ok else float("nan")
    f = float(abs(rho_truth) / r_max) if ok and r_max > 1e-15 else float("nan")
    return {
        "r_rel": float(r_rel) if np.isfinite(r_rel) else float("nan"),
        "r_max": r_max,
        "f_ceiling": f,
        "f_ceiling_display": float(min(f, 1.0)) if np.isfinite(f) else float("nan"),
        "attenuation_applicable": ok,
    }


def bootstrap_stat(fn, arrays: list[np.ndarray], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    vals = []
    n = arrays[0].size
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[np.asarray(a)[idx] for a in arrays])
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 8:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(vals)}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "lo": float(lo), "hi": float(hi), "n": len(vals)}


def band_rank(rho: float, *, degenerate: bool = False) -> str:
    if degenerate:
        return "degenerate_target"
    if not np.isfinite(rho):
        return "not_estimable"
    a = abs(float(rho))
    if a >= BAND_RANK_STRONG:
        return "strong"
    if a >= BAND_RANK_MODERATE:
        return "moderate"
    return "weak"


def band_pair(acc: float) -> str:
    if not np.isfinite(acc):
        return "not_estimable"
    if acc >= BAND_PAIR_STRONG:
        return "strong"
    if acc >= BAND_PAIR_MODERATE:
        return "moderate"
    return "weak"


def band_rel(r: float) -> str:
    if not np.isfinite(r):
        return "not_estimable"
    if r >= BAND_REL_STRONG:
        return "strong"
    if r >= BAND_REL_MODERATE:
        return "moderate"
    return "weak"


def band_ceil(f: float) -> str:
    if not np.isfinite(f):
        return "not_estimable"
    if f >= BAND_CEIL_STRONG:
        return "strong"
    if f >= BAND_CEIL_MODERATE:
        return "moderate"
    return "weak"


def rank_bundle(est, truth, *, analytic_constant: bool = False) -> dict:
    dr = dynamic_range(truth, analytic_constant=analytic_constant)
    rho = spearman_safe(est, truth)
    tau = kendall_safe(est, truth)
    acc = pairwise_ordering_accuracy(est, truth)
    return {
        **dr,
        "spearman": rho,
        "kendall_tau": tau,
        "pairwise_acc": acc,
        "rank_rmse": rank_rmse(est, truth),
        "pearson": pearson_safe(est, truth),
        "rank_band": band_rank(rho, degenerate=dr["rank_target_degenerate"]),
        "pair_band": band_pair(acc),
    }
