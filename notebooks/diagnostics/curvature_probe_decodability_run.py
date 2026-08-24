"""Phase 5 curvature-conditioned linear decodability runner.

`--mode field` extracts the decoder-side `||H||` field for each seed in `--seeds`, via
`chart_curvature.chart_curvature_field` against a genuine sealed CAE checkpoint, caching each
seed's field through `cache.npz_cache`. `--mode pool` is pre-registered but not implemented
until plan 05-03. `--mode bucketed` requires both `linear_probe.assert_preregistered()` and the
frozen pooled curvature field artifact to exist before it will even attempt anything -- the
D5-10 guard -- and its body is not implemented until plan 05-05. `--selfcheck` is this plan's
own automated implementation check: it runs the complete probe-to-verdict path on a synthetic,
dimensionally PU-shaped fixture with a planted linear map and a planted curvature-to-residual
ordering, and writes exactly one JSONL row tagged `data_source = "synthetic_planted"`. No PU
probe number is computed by any command below.

    python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field --smoke
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode pool
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode bucketed
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from scipy.stats import spearmanr

from pu_manifold import cache, linear_probe

DEFAULT_RECORD = cache.cache_path("05_curvature_probe_decodability", "jsonl")
SELFCHECK_RECORD = cache.cache_path("05_probe_selfcheck", "jsonl")


def load_pu_pair(
    column_a: str = "hsc", column_b: str = "legacysurvey"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path. Keeps
    only files carrying both columns, selects the one with the most rows; on a tie keeps the
    lexicographically first path. Copied unchanged from
    `region_partition_mknn_run.load_pu_pair` -- its `"hsc"` / `"legacysurvey"` defaults
    already match D5-01's probe target.
    """
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column_a in z.files and column_b in z.files and z[column_a].shape[0] > best_n:
                best, best_n = c, z[column_a].shape[0]
    if best is None:
        raise KeyError(
            f"no cached subsample carries both {column_a!r} and {column_b!r} columns"
        )
    with np.load(best) as z:
        Xa = np.asarray(z[column_a], dtype=np.float64)
        Xb = np.asarray(z[column_b], dtype=np.float64)
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError(
            f"load_pu_pair: {column_a!r} has {Xa.shape[0]} rows but {column_b!r} has "
            f"{Xb.shape[0]} rows in {best!r}."
        )
    print(f"loaded {column_a} {Xa.shape} and {column_b} {Xb.shape} from {Path(best).name}")
    return Xa, Xb, best


def _spearman_report(a: np.ndarray, b: np.ndarray, name: str) -> Dict[str, Any]:
    """One plain Spearman correlation, printed with its p-value and point count. When either
    input is constant, `scipy.stats.spearmanr` returns NaN rather than raising; that case is
    reported with an explicit undefined marker rather than a number.
    """
    n_pts = int(a.shape[0])
    rho, p = spearmanr(a, b)
    if np.isnan(rho):
        print(f"{name}: UNDEFINED (constant input, spearmanr returned NaN) -- n={n_pts}")
        return {"rho": None, "p_value": None, "n": n_pts, "undefined": True}
    print(f"{name}: rho={rho:+.4f}  p={p:.4g}  n={n_pts}")
    return {"rho": float(rho), "p_value": float(p), "n": n_pts, "undefined": False}


def _piecewise_constant_field(values: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantize `values` into `n_levels` equal-count levels `1..n_levels`, via the same
    rank-based `np.array_split` idiom `bucket_edges_from_field` uses. Reproduces this
    milestone's measured seed-field shape (RESEARCH Pitfall 2): two of the three cached CAE
    seeds' `||H||` fields are piecewise-constant on collapsed metrics, not continuous.
    """
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, n_levels)
    quantized = np.empty_like(values, dtype=np.float64)
    for level, g in enumerate(groups, start=1):
        quantized[g] = float(level)
    return quantized


def _make_rank_subspace_X(n: int, d_ambient: int, d_rank: int, rng: np.random.Generator) -> np.ndarray:
    """`n` rows drawn from a `d_rank`-dimensional random subspace of `R^d_ambient`, so the
    design matrix has the same ill-conditioning shape (rank << ambient dimension) as the real
    768-d probe design matrix at the manifold's established ~18-25 intrinsic dimension.
    """
    basis = rng.normal(size=(d_rank, d_ambient))
    coeffs = rng.normal(size=(n, d_rank))
    return coeffs @ basis


def selfcheck() -> bool:
    """Known-answer self-check on synthetic, dimensionally PU-shaped data (n=900, d=768).
    This runner flag is the phase's automated implementation check, run before any PU number
    is ever computed. Touches no PU embedding and no CAE checkpoint -- `data_source` in the
    written JSONL row is always `"synthetic_planted"`.
    """
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            ok = False

    rng = np.random.default_rng(20260824)
    n, d_ambient, d_rank = 900, 768, 25

    X = _make_rank_subspace_X(n, d_ambient, d_rank, rng)
    A = rng.normal(size=(d_ambient, d_ambient)) / np.sqrt(d_ambient)
    b = rng.normal(size=d_ambient)
    base_noise = rng.normal(size=(n, d_ambient))
    # `h`: fabricated per-point curvature-analogue field. Constructed so per-point residual
    # rises monotonically with `h` BY CONSTRUCTION -- extra noise added to a point in
    # proportion to its own `h`, unrelated to X, so it inflates the unexplained residual and
    # nothing else.
    h = rng.uniform(0.0, 1.0, size=n)
    extra_noise = h[:, None] * rng.normal(size=(n, d_ambient)) * 1.5e-2

    Y = X @ A.T + b + 1e-6 * base_noise + extra_noise

    train_idx, test_idx = linear_probe.train_test_split_indices(
        n, train_fraction=0.8, split_seed=20260824
    )

    alpha_grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    fit = linear_probe.fit_probe(
        X[train_idx], Y[train_idx], alpha_grid=alpha_grid, alpha_per_target=False,
        fit_intercept=True,
    )
    Y_pred_test = linear_probe.predict_probe(fit, X[test_idx])
    residuals_test = linear_probe.per_point_residuals(Y[test_idx], Y_pred_test)
    r2 = linear_probe.aggregate_r2(Y[test_idx], Y_pred_test, multioutput="variance_weighted")
    check("recovered aggregate R-squared exceeds 0.99", r2 > 0.99)

    ybar = Y[test_idx].mean(axis=0)
    frob_denominator = float(np.sum((Y[test_idx] - ybar) ** 2))
    frob_r2 = 1.0 - float(residuals_test.sum()) / frob_denominator
    check(
        "per_point_residuals/aggregate_r2 agree on the Frobenius identity to 1e-9",
        abs(frob_r2 - r2) < 1e-9,
    )

    # Three synthetic seed fields at deliberately mismatched scales (1x, 40x, 55x), built from
    # the SAME base `h` field over all n rows -- pooling operates on the full field, not only
    # the test split. Reproducing this milestone's measured shape (RESEARCH Pitfall 2): one
    # continuous field, two piecewise-constant fields with only four distinct levels each.
    seed_fields = {
        20260813: h * 1.0,
        20260814: _piecewise_constant_field(h, 4) * 40.0,
        20260815: _piecewise_constant_field(h, 4) * 55.0,
    }
    pool_result = linear_probe.pool_seed_fields(seed_fields, method="per_seed_median_divide")
    pooled = pool_result["pooled"]
    largest_seed = max(seed_fields, key=lambda s: float(np.median(seed_fields[s])))
    rho_report = _spearman_report(
        pooled, seed_fields[largest_seed], "selfcheck pooled vs. largest-magnitude seed"
    )
    check(
        "pooled field Spearman against the largest-magnitude seed is strictly below 1.0",
        rho_report["rho"] is not None and rho_report["rho"] < 1.0,
    )

    n_buckets_sc = 3
    h_test = pooled[test_idx]
    labels_test, edges_test = linear_probe.bucket_by_field(h_test, n_buckets_sc)
    counts_info = linear_probe.bucket_counts(labels_test, n_buckets_sc)
    check(
        "bucket counts partition the test split exactly",
        int(counts_info["counts"].sum()) == test_idx.shape[0],
    )

    bucket_stats = []
    for bucket_idx in range(n_buckets_sc):
        mask = labels_test == bucket_idx
        r_bucket = residuals_test[mask]
        ci = linear_probe.bucket_residual_ci(
            r_bucket, n_resamples=500, seed=20260824, confidence_level=0.95
        )
        bucket_stats.append(ci)
    check(
        "highest bucket's mean residual strictly exceeds the lowest bucket's",
        bucket_stats[-1]["score"] > bucket_stats[0]["score"],
    )

    size_match = linear_probe.size_matched_check(
        residuals_test, labels_test, n_repeats=50, seed=20260824, confidence_level=0.95
    )

    # Self-check-scoped VERDICT_RULE literal -- supplied locally, clearly commented, and never
    # read back into the module. The sealed VERDICT_RULE constant stays "" until the 05-04
    # freeze; this string exists only so apply_verdict_rule (which requires a non-empty
    # verdict_rule) can be exercised here.
    local_verdict_rule = (
        "selfcheck-scoped VERDICT_RULE naming N_BUCKETS -- local literal, never read into "
        "the linear_probe module."
    )
    verdict_result = linear_probe.apply_verdict_rule(
        bucket_stats, size_match, local_verdict_rule
    )
    check("apply_verdict_rule returns HOLDS on the planted ordering", verdict_result["verdict"] == "HOLDS")

    try:
        linear_probe.assert_preregistered()
        check("assert_preregistered raises RuntimeError while constants are unset", False)
    except RuntimeError:
        check("assert_preregistered raises RuntimeError while constants are unset", True)

    row = {
        "kind": "selfcheck",
        "data_source": "synthetic_planted",
        "n": int(n),
        "n_train": int(train_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "r2_overall": float(r2),
        "frob_r2": float(frob_r2),
        "spearman_pooled_vs_largest_seed": rho_report["rho"],
        "bucket_counts": [int(c) for c in counts_info["counts"]],
        "bucket_mean_residuals": [float(s["score"]) for s in bucket_stats],
        "size_match_n": size_match["n_match"],
        "size_match_sign_stable": size_match["sign_stable"],
        "verdict": verdict_result["verdict"],
        "ok": ok,
    }
    SELFCHECK_RECORD.parent.mkdir(parents=True, exist_ok=True)
    with SELFCHECK_RECORD.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()

    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["field", "pool", "bucketed"], default="field")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260813, 20260814, 20260815])
    p.add_argument("--n-charts", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--field-stem", type=str, default="05_curvature_field")
    p.add_argument("--pooling-method", type=str, default=None)
    return p


def main() -> None:
    a = build_arg_parser().parse_args()

    if a.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    raise NotImplementedError(
        f"--mode {a.mode!r} is pre-registered but not implemented until a later task in this "
        "plan (Task 3 completes --mode field and the --mode bucketed guard; --mode pool is "
        "implemented at plan 05-03; --mode bucketed's body is implemented at plan 05-05)."
    )


if __name__ == "__main__":
    main()
