#!/usr/bin/env python3
"""Stage B: does local curvature explain probe error and SAE reconstruction error?

Two corrections to the earlier pipeline are applied here.

1. The probe-error target is cleaned. Only 8 of the 38 `independent` probes reach
   r2_cv > 0, and those 8 are exactly the ones with ~93-100% label coverage; the
   other 30 have n_valid ~ 725-1187 against 768 features, so the fit interpolates
   (elpetro_theta reaches r2_cv = -11117). A nanmean over all 38 therefore partly
   encodes *whether a galaxy has an NSA/MaNGA cross-match* rather than how hard it
   is to predict. `n_valid_probes` is exported so that confound can be partialled
   out explicitly.

2. Two bugs in _common.compute_probe_residuals are fixed:
   - the intercept was dropped (`y_hat = Z_test @ w_m`), which is only valid when
     Z_train is mean-centred; L2-normalised ViT embeddings have a large mean
     vector, so every prediction carried a constant offset;
   - test targets were re-standardised with test-split statistics while W was
     fitted against train-standardised ones.

_common is left untouched so the earlier run stays reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from _common import ALL_PROBES, DEFAULT_11_PROBES, INDEPENDENT_PROBES, load_physics_labels
from density_stats import partial_spearman

CURV_METRICS = ["kappa_ratio", "kappa_jet", "rf_k", "kappa_naive_ratio"]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def fit_probes(Z, y_dict, probe_keys):
    """Mirror of _common.train_probes that additionally retains the intercept and
    the train-split standardisation constants, both of which the residual
    computation needs and the original discards. Fit settings (LinearRegression,
    5-fold KFold seed 42) are identical, so r2_cv matches the earlier run.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    D, M = Z.shape[1], len(probe_keys)
    W = np.zeros((D, M), dtype=np.float64)
    b = np.zeros(M, dtype=np.float64)
    y_mu = np.full(M, np.nan)
    y_sd = np.full(M, np.nan)
    stats = {}

    for m, key in enumerate(probe_keys):
        y = y_dict[key]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            stats[key] = {"r2_train": float("nan"), "r2_cv": float("nan"),
                          "n_valid": int(valid.sum())}
            continue

        Zv, yv = Z[valid], y[valid]
        mu, sd = float(yv.mean()), float(yv.std() + 1e-12)
        ys = (yv - mu) / sd
        y_mu[m], y_sd[m] = mu, sd

        model = LinearRegression(fit_intercept=True).fit(Zv, ys)
        W[:, m], b[m] = model.coef_, float(model.intercept_)

        cv = []
        kf = KFold(n_splits=min(5, len(ys)), shuffle=True, random_state=42)
        for tr, te in kf.split(Zv):
            mcv = LinearRegression(fit_intercept=True).fit(Zv[tr], ys[tr])
            cv.append(r2_score(ys[te], mcv.predict(Zv[te])))
        stats[key] = {
            "r2_train": float(r2_score(ys, model.predict(Zv))),
            "r2_cv": float(np.mean(cv)),
            "n_valid": int(valid.sum()),
        }
    return W, b, y_mu, y_sd, stats


def probe_residuals(Z_test, y_test, W, b, y_mu, y_sd, probe_keys):
    """Per-point, per-probe squared residuals, standardised with TRAIN statistics
    and including the fitted intercept."""
    n, M = Z_test.shape[0], len(probe_keys)
    res = np.full((n, M), np.nan)
    for m, key in enumerate(probe_keys):
        if key not in y_test or not np.isfinite(y_sd[m]):
            continue
        y = y_test[key]
        ys = (y - y_mu[m]) / y_sd[m]
        err = (ys - (Z_test @ W[:, m] + b[m])) ** 2
        err[np.isnan(y)] = np.nan
        res[:, m] = err
    return res


def probe_targets(Z, idx_train, idx_test, labels, probe_keys, r2_min):
    """Return the cleaned target, the legacy target, and the confound array."""
    y_train = {k: v[idx_train] for k, v in labels.items()}
    y_test = {k: v[idx_test] for k, v in labels.items()}

    W, b, y_mu, y_sd, stats = fit_probes(Z[idx_train], y_train, probe_keys)
    res = probe_residuals(Z[idx_test], y_test, W, b, y_mu, y_sd, probe_keys)

    good = np.array([stats.get(k, {}).get("r2_cv", np.nan) > r2_min for k in probe_keys])
    with np.errstate(invalid="ignore"):
        targets = {
            "mean_residual_all": np.nanmean(res, axis=1),
            "mean_residual_good": (np.nanmean(res[:, good], axis=1) if good.any()
                                   else np.full(len(res), np.nan)),
            "n_valid_probes": np.isfinite(res).sum(axis=1).astype(float),
        }
    if "redshift" in probe_keys:
        targets["redshift_residual"] = res[:, probe_keys.index("redshift")]
    return targets, stats, [k for k, g in zip(probe_keys, good) if g]


def load_labels_cached(root: Path, n: int) -> dict[str, np.ndarray]:
    """Stream Smith42/galaxies once and cache it; the original re-streams every run."""
    cache = root / "data_hf" / "physics" / f"labels_test_n{n}.npz"
    if cache.is_file():
        print(f"  Loading cached labels from {cache}", flush=True)
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    print(f"  Streaming {n} label rows from HF (will cache to {cache})...", flush=True)
    labels = load_physics_labels(n, split="test")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **labels)
    return labels


# ---------------------------------------------------------------------------
# SAE join
# ---------------------------------------------------------------------------

def load_sae_arrays(sae_npz_dir: Path, tag: str, idx_test: np.ndarray) -> dict:
    """Reuse the already-computed per-point SAE outputs instead of retraining.

    Valid because train_test_split(np.arange(N), test_size, random_state) is
    deterministic; the assertion below fails loudly if max_n / seed / test_size
    have drifted from the run that produced the npz.
    """
    path = Path(sae_npz_dir) / f"{tag}_curvature.npz"
    if not path.is_file():
        print(f"  WARNING: no SAE npz at {path}; skipping the SAE leg.", flush=True)
        return {}
    z = np.load(path)
    if not np.array_equal(z["idx_test"], idx_test):
        raise SystemExit(
            f"idx_test mismatch against {path}. The SAE outputs were produced with a "
            f"different --max-n/--seed/--test-size, so the per-point join would be wrong."
        )
    return {
        "sae_reconstruction_error": z["reconstruction_error"].astype(float),
        "sae_atom_turnover_rate": z["atom_turnover_rate"].astype(float),
        "legacy_mean_residual": z["mean_residual"].astype(float),
    }


# ---------------------------------------------------------------------------
# Entry point called by density_curvature_probe.main
# ---------------------------------------------------------------------------

def attach(payload: dict, args, root: Path, out_dir: Path) -> None:
    probe_keys = {
        "independent": INDEPENDENT_PROBES,
        "all": list(ALL_PROBES),
        "default11": DEFAULT_11_PROBES,
    }.get(args.probes, [p.strip() for p in args.probes.split(",")])

    labels_full = load_labels_cached(root, args.max_n)
    payload["connectback"] = []

    for entry in payload["per_model"]:
        data, real = entry["_data"], entry["_real"]
        n_total = data["n_total"]
        labels = {k: v[:n_total] for k, v in labels_full.items()}

        print(f"\n  Stage B — {entry['model']}: fitting {len(probe_keys)} probes...", flush=True)
        targets, stats, good_probes = probe_targets(
            data["Z"], data["idx_train"], data["idx_test"], labels,
            probe_keys, args.probe_r2_min)
        entry["probe_stats"] = stats
        entry["good_probes"] = good_probes
        print(f"    {len(good_probes)}/{len(probe_keys)} probes pass "
              f"r2_cv > {args.probe_r2_min}: {good_probes}", flush=True)

        if args.sae_npz_dir:
            targets.update(load_sae_arrays(
                Path(args.sae_npz_dir), entry["model"].replace("/", "_"),
                data["idx_test"]))

        d_k, n_valid = real["d_k"], targets["n_valid_probes"]
        target_keys = [k for k in ("mean_residual_good", "redshift_residual",
                                   "mean_residual_all", "sae_reconstruction_error",
                                   "sae_atom_turnover_rate") if k in targets]

        for K, suite in real["scales"].items():
            rows = []
            for curv_name in CURV_METRICS:
                curv = suite[curv_name].astype(float)
                for tk in target_keys:
                    t = targets[tk]
                    ok = np.isfinite(curv) & np.isfinite(t)
                    rows.append({
                        "curv": curv_name,
                        "target": tk,
                        "rho": float(spearmanr(curv[ok], t[ok]).statistic),
                        "partial_given_dk": partial_spearman(curv, t, d_k)["rho"],
                        "partial_given_nvalid": partial_spearman(curv, t, n_valid)["rho"],
                        "rho_vs_nvalid": float(
                            spearmanr(curv[np.isfinite(curv)], n_valid[np.isfinite(curv)]).statistic),
                    })
            payload["connectback"].append(
                {"model": entry["model"], "K": int(K), "rows": rows})

        np.savez_compressed(
            out_dir / f"{entry['model'].replace('/', '_')}_targets.npz", **targets)

        if "probe_health" not in payload:
            a, g = targets["mean_residual_all"], targets["mean_residual_good"]
            ok = np.isfinite(a) & np.isfinite(n_valid)
            payload["probe_health"] = {
                "model": entry["model"],
                "r2_min": args.probe_r2_min,
                "n_good": len(good_probes),
                "n_total": len(probe_keys),
                "all_median": float(np.nanmedian(a)), "all_mean": float(np.nanmean(a)),
                "good_median": float(np.nanmedian(g)), "good_mean": float(np.nanmean(g)),
                "all_vs_nvalid": float(spearmanr(a[ok], n_valid[ok]).statistic),
            }
