"""Phase 4 region-partitioning MKNN runner. `--mode global` (plan 04-01) computes the
region-blind, partition-blind crossmodal HSC-vs-Legacy-Survey MKNN across the frozen
`--mknn-k` grid. `--mode partition` (plan 04-04) computes the density-corrected PU field
at the frozen k, REGN-01/REGN-02's density diagnostics, the D4-09 sign split, and REGN-06's
frozen partition artifact -- all before any regional MKNN number exists. `--mode regional`
(plan 04-03 added the guard; plan 04-05 implements the cell computation) requires both
`region_partition.assert_preregistered()` and the frozen partition artifact to exist before
it will even attempt a regional MKNN cell -- the eight-cell grid, region-scoped nulls,
bootstrap CIs, hubness, and the pre-registered verdict applied mechanically.

    python notebooks/diagnostics/region_partition_mknn_run.py --selfcheck
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global
    python notebooks/diagnostics/region_partition_mknn_run.py --mode partition --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode partition
    python notebooks/diagnostics/region_partition_mknn_run.py --mode regional
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pu_manifold import cache
from pu_manifold import curvature_probe
from pu_manifold import mknn
from pu_manifold import region_partition

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "04_region_partition_mknn.jsonl"


def load_pu_pair(
    column_a: str = "hsc", column_b: str = "legacysurvey"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path.
    Keeps only files carrying both columns, selects the one with the most rows; on a
    tie keeps the lexicographically first path (mirrors `pu_curvature_rankability_run
    .load_pu`'s existing strictly-greater-than comparison over a `sorted(glob)`)."""
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


def run_global_cell(
    X_hsc: np.ndarray,
    X_ls: np.ndarray,
    k_mknn: int,
    n_permutations: int,
    n_resamples: int,
    seed: int,
    null_quantile: float,
    confidence_level: float,
    subsample_file: str,
) -> Dict[str, Any]:
    """One flat, JSONL-serializable row: mknn_score, permutation_null, bootstrap_ci,
    hubness_skewness for both sides, chance_floor and the ratio over it."""
    t0 = time.monotonic()
    n = X_hsc.shape[0]

    score = mknn.mknn_score(X_hsc, X_ls, k_mknn)
    perm = mknn.permutation_null(X_hsc, X_ls, k_mknn, n_permutations, seed, null_quantile)
    boot = mknn.bootstrap_ci(X_hsc, X_ls, k_mknn, n_resamples, seed, confidence_level)
    hub_hsc = mknn.hubness_skewness(X_hsc, k_mknn)
    hub_ls = mknn.hubness_skewness(X_ls, k_mknn)
    floor = mknn.chance_floor(n, k_mknn)

    return {
        "kind": "mknn_global",
        "region": "global",
        "null_scope": "global",
        "n": int(n),
        "k_mknn": int(k_mknn),
        "score": score,
        "chance_floor": floor,
        "ratio_over_chance": score / floor,
        "p_value": perm["p_value"],
        "null_mean": perm["null_mean"],
        "null_std": perm["null_std"],
        "null_threshold": perm["null_threshold"],
        "null_quantile": perm["null_quantile"],
        "clears_null": perm["clears_null"],
        "n_permutations": perm["n_permutations"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "degenerate": boot["degenerate"],
        "confidence_level": boot["confidence_level"],
        "n_resamples": boot["n_resamples"],
        "seed": int(seed),
        "hubness_skewness_hsc": hub_hsc,
        "hubness_skewness_legacysurvey": hub_ls,
        "subsample_file": subsample_file,
        "wallclock_s": time.monotonic() - t0,
    }


def _spearman_report(a: np.ndarray, b: np.ndarray, name: str) -> Dict[str, Any]:
    """One plain Spearman correlation, printed with its p-value and point count. When
    either input is constant, ``scipy.stats.spearmanr`` returns NaN rather than raising;
    that case is reported with an explicit undefined marker rather than a number, per
    REGN-02's own read-out requirement."""
    n_pts = int(a.shape[0])
    rho, p = spearmanr(a, b)
    if np.isnan(rho):
        print(f"REGN-02 {name}: UNDEFINED (constant input, spearmanr returned NaN) -- n={n_pts}")
        return {"rho": None, "p_value": None, "n": n_pts, "undefined": True}
    print(f"REGN-02 {name}: rho={rho:+.4f}  p={p:.4g}  n={n_pts}")
    return {"rho": float(rho), "p_value": float(p), "n": n_pts, "undefined": False}


def run_partition(
    X_ls: np.ndarray,
    k: int,
    d: int,
    k_density: int,
    min_norm_percentile: float,
    seed: int,
    subsample_file: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    """``--mode partition``'s full compute, in the Ordering constraint's own order: the
    density-corrected PU field at ``k`` (D4-13's estimator, computed on the ``legacysurvey``
    column, matching ``pu_curvature_rankability_run.py``'s protocol), REGN-01's ambient
    768-d local density (printed BEFORE the split is trusted), the D4-09 diametrical sign
    split, REGN-05's region counts, REGN-06's frozen artifact (written before any regional
    MKNN number is computed), and REGN-02's density correlations plus the region-level
    Mann-Whitney comparison.

    Every parameter is a required argument with no default, so the caller always names the
    frozen constants explicitly at the call site (D-07) -- this function chooses nothing.
    ``smoke=True`` runs the identical path at whatever (reduced) size/``k``/``d`` the caller
    passes and skips both cache writes, matching ``--mode global --smoke``'s "writes
    nothing" convention -- a smoke pass must never collide with the real frozen artifact's
    manifest.
    """
    n = X_ls.shape[0]
    print(
        f"computing density-corrected PU field: n={n}, k={k}, d={d}, k_density={k_density} "
        "(density_correct=True) ..."
    )
    t0 = time.monotonic()
    H = curvature_probe.centroid_mean_curvature(
        X_ls, k=k, d=d, density_correct=True, k_density=k_density
    )
    field_wallclock = time.monotonic() - t0
    print(f"field wallclock: {field_wallclock:.1f}s")

    # REGN-01: ambient 768-d local density, printed before the split is trusted.
    w = curvature_probe.local_density_weights(X_ls, k_density=k_density, d=d)
    rho = 1.0 / w
    rho_p05 = float(np.percentile(rho, 5))
    rho_p50 = float(np.percentile(rho, 50))
    rho_p95 = float(np.percentile(rho, 95))
    rho_ratio = rho_p95 / rho_p05 if rho_p05 > 0 else float("inf")
    print(
        "REGN-01: local_density_weights' w is mean-normalized to 1, so rho = 1/w is a "
        "RELATIVE density, not an absolute one."
    )
    print(
        f"  rho p05={rho_p05:.6g}  p50={rho_p50:.6g}  p95={rho_p95:.6g}  "
        f"p95/p05={rho_ratio:.3f}"
    )

    # The D4-09 diametrical sign-split partition -- every parameter is the frozen constant
    # the caller passed in, chosen nowhere in this function.
    result = region_partition.region_partition(H, min_norm_percentile)
    v = result["v"]
    labels = result["labels"]
    keep_idx = result["keep_idx"]
    excluded_idx = result["excluded_idx"]
    h_norm_full = result["h_norm"]
    proj = result["proj"]
    floor = result["floor"]
    eigval_spectrum = result["eigval_spectrum"]
    eigval_top5 = sorted(eigval_spectrum.tolist(), reverse=True)[:5]
    print(
        f"partition: floor={floor:.6g} (min_norm_percentile={min_norm_percentile}), "
        f"excluded={int(excluded_idx.shape[0])}, mean_unit_norm={result['mean_unit_norm']:.6g}"
    )
    print(f"  eigval_top={result['eigval_top']:.6g}")
    print(f"  eigval_spectrum top-5: {[f'{e:.6g}' for e in eigval_top5]}")

    # REGN-05: region/exclusion counts, closure asserted against n.
    counts = region_partition.region_counts(labels, int(excluded_idx.shape[0]), result["n_zero_projection"])
    n_region_0 = counts["n_region_0"]
    n_region_1 = counts["n_region_1"]
    n_excluded = counts["n_excluded"]
    n_zero_projection = counts["n_zero_projection"]
    if n_region_0 + n_region_1 + n_excluded != n:
        raise AssertionError(
            f"region membership counts ({n_region_0} + {n_region_1} + {n_excluded}) do not "
            f"sum to n={n}."
        )
    print(
        f"REGN-05: region_0={n_region_0} ({counts['fraction_region_0'] * 100:.1f}%)  "
        f"region_1={n_region_1} ({counts['fraction_region_1'] * 100:.1f}%)  "
        f"excluded={n_excluded} ({counts['fraction_excluded'] * 100:.1f}%)  "
        f"n_zero_projection={n_zero_projection}"
    )
    undersized_region: List[int] = []
    for label_val, count in ((0, n_region_0), (1, n_region_1)):
        if count < region_partition.MIN_REGION_N:
            undersized_region.append(label_val)
            print(
                f"  PRE-REGISTERED CONSEQUENCE: region {label_val} has n={count} < "
                f"MIN_REGION_N={region_partition.MIN_REGION_N} -- recorded as undefined "
                f"('{region_partition.MIN_REGION_N_UNDEFINED_REASON}'), nothing computed "
                "for it, no adjustment made."
            )

    # REGN-06: freeze v, labels, keep_idx, excluded_idx, h_norm, signed_projection -- BEFORE
    # anything downstream reads a label. Skipped in smoke mode so a reduced-size smoke pass
    # never collides with the real frozen artifact's config manifest.
    partition_cfg = {
        "K_FROZEN": int(k),
        "FIELD_D": int(d),
        "K_DENSITY": int(k_density),
        "MIN_NORM_PERCENTILE": float(min_norm_percentile),
        "SEED": int(seed),
        "COVARIANCE_FORM": region_partition.COVARIANCE_FORM,
        "subsample_file": subsample_file,
    }
    if not smoke:
        cache.npz_cache(
            "04_region_partition",
            partition_cfg,
            lambda: {
                "v": v,
                "labels": labels,
                "keep_idx": keep_idx,
                "excluded_idx": excluded_idx,
                "h_norm": h_norm_full,
                "signed_projection": proj,
                # Extra key beyond REGN-06's required six: the full unit-H covariance
                # spectrum, so the notebook's Section 5 can render the top-five eigenvalues
                # (04-RESEARCH.md's codimension caveat) without recomputing the heavy field.
                "eigval_spectrum": eigval_spectrum,
            },
        )
        print(f"REGN-06: frozen partition artifact written to {cache.cache_path('04_region_partition', 'npz')}")
    else:
        print("SMOKE: REGN-06 artifact write skipped -- smoke mode writes nothing.")

    # REGN-02: both Spearman correlations, plain, on surviving non-excluded points only.
    rho_survivors = rho[keep_idx]
    h_norm_survivors = h_norm_full[keep_idx]
    spearman_density_vs_hnorm = _spearman_report(rho_survivors, h_norm_survivors, "density vs ||H||")
    spearman_density_vs_projection = _spearman_report(rho_survivors, proj, "density vs signed projection <H/||H||, v>")

    # Region-level density comparison (D4-14's other half of the evidence): median/IQR per
    # region plus a two-sided Mann-Whitney U test.
    rho_region_0 = rho_survivors[labels == 0]
    rho_region_1 = rho_survivors[labels == 1]
    median_density_region_0 = float(np.median(rho_region_0)) if rho_region_0.size else None
    median_density_region_1 = float(np.median(rho_region_1)) if rho_region_1.size else None
    iqr_density_region_0 = (
        float(np.percentile(rho_region_0, 75) - np.percentile(rho_region_0, 25)) if rho_region_0.size else None
    )
    iqr_density_region_1 = (
        float(np.percentile(rho_region_1, 75) - np.percentile(rho_region_1, 25)) if rho_region_1.size else None
    )
    if rho_region_0.size and rho_region_1.size:
        mw_stat, mw_p = mannwhitneyu(rho_region_0, rho_region_1, alternative="two-sided")
        mw_stat, mw_p = float(mw_stat), float(mw_p)
    else:
        mw_stat, mw_p = None, None
    print(
        "region-level density comparison (the single most decision-relevant density number "
        "this phase can report, given D4-14's declined controls):"
    )
    print(
        f"  region_0: median={median_density_region_0}  IQR={iqr_density_region_0}  "
        f"n={int(rho_region_0.size)}"
    )
    print(
        f"  region_1: median={median_density_region_1}  IQR={iqr_density_region_1}  "
        f"n={int(rho_region_1.size)}"
    )
    print(f"  Mann-Whitney U (two-sided): statistic={mw_stat}  p={mw_p}")

    h_p05 = float(np.percentile(h_norm_full, 5))
    h_p95 = float(np.percentile(h_norm_full, 95))
    h_spread = float(h_p95 / h_p05) if h_p05 > 0 else float("inf")

    diagnostics = {
        "spearman_density_vs_hnorm": spearman_density_vs_hnorm,
        "spearman_density_vs_projection": spearman_density_vs_projection,
        "mannwhitneyu_statistic": mw_stat,
        "mannwhitneyu_pvalue": mw_p,
        "median_density_region_0": median_density_region_0,
        "median_density_region_1": median_density_region_1,
        "iqr_density_region_0": iqr_density_region_0,
        "iqr_density_region_1": iqr_density_region_1,
        "n_region_0": int(n_region_0),
        "n_region_1": int(n_region_1),
        "n_excluded": int(n_excluded),
        "n_zero_projection": int(n_zero_projection),
        "undersized_region": undersized_region,
        "rho_p05": rho_p05,
        "rho_p50": rho_p50,
        "rho_p95": rho_p95,
        "rho_p95_over_p05": rho_ratio,
        "mean_unit_norm": float(result["mean_unit_norm"]),
        "eigval_top": float(result["eigval_top"]),
        "eigval_spectrum_top5": [float(e) for e in eigval_top5],
        "h_spread": h_spread,
        "floor": float(floor),
        "min_norm_percentile": float(min_norm_percentile),
        "k_frozen": int(k),
        "d": int(d),
        "k_density": int(k_density),
        "seed": int(seed),
        "subsample_file": subsample_file,
        "field_wallclock_s": float(field_wallclock),
        "smoke": bool(smoke),
    }
    if not smoke:
        cache.json_cache("04_density_diagnostics", partition_cfg, lambda: diagnostics)
        print(f"density diagnostics written to {cache.cache_path('04_density_diagnostics', 'json')}")
    else:
        print("SMOKE: density diagnostics write skipped -- smoke mode writes nothing.")

    print(
        "\nD4-14: the density confound is REPORTED, not CONTROLLED, in this phase. The only "
        "density-confound check run is the REGN-02 Spearman correlation (before and after "
        "the split) plus the region-level Mann-Whitney comparison above -- no partial "
        "regression, no density-matched null, no centroid-distance control, no "
        "density-matched stratification. MKNN is itself a k-NN statistic and therefore "
        "directly density-sensitive by construction, so a later regional MKNN difference "
        "CANNOT be attributed to curvature rather than to regional density by anything in "
        "this phase."
    )
    print(
        f"D4-05: PU's measured ||H|| spread is {h_spread:.2f}x (p95/p05 over the full field) "
        "against the runner's own calibration -- unrankable quadratic_bowl at 1.4x, "
        "rankable cubic/ridge at 28.2x/34.3x. PU sits far nearer the unrankable end. This "
        "gates nothing: direction is a unit vector and does not consume the magnitude "
        "spread."
    )
    print(
        "CODIMENSION CAVEAT: every fixture the direction-partition decision (D4-01) rests "
        "on is codimension 1, where H = H_scalar * n_hat; PU's codimension is roughly 748 "
        "(d ~ 20 inside D = 768). A cosine near 1.000 on those fixtures demonstrates "
        "recovery of a surface's normal orientation, a tangent-space problem known to "
        "converge well -- not resolution of H's direction within a 748-wide normal space. "
        "That gap is unmeasured on PU and unclosed by anything in this milestone."
    )
    return diagnostics


def run_regional_cell(
    region: int,
    k_mknn: int,
    X_hsc_region: np.ndarray,
    X_ls_region: np.ndarray,
    n_permutations: int,
    n_resamples: int,
    seed: int,
    null_quantile: float,
    confidence_level: float,
    k_frozen: int,
    subsample_file: str,
) -> Dict[str, Any]:
    """One flat, JSONL-serializable regional MKNN row (MKNN-03..MKNN-06, MKNN-08). Every
    parameter is a required argument sourced from `region_partition`'s frozen constants at
    the call site -- this function chooses nothing.

    Per D4-16, `X_hsc_region`/`X_ls_region` are ALREADY subset to the region's own rows by
    the caller; every neighbour set, score, null and CI computed here therefore lives
    entirely inside that region's own index set -- no global null is computed anywhere in
    this function.

    Skip conditions are checked BEFORE anything is computed: `n_region < MIN_REGION_N`
    (RESEARCH Pitfall 3 -- an undersized region's own null and CI become unstable) and
    `k_mknn + 1 > n_region` (a k-NN query cannot ask for more neighbours than the region
    has points). Both are pre-registered; neither is a judgement made here. A skipped cell
    is recorded with `status: "undefined"`, `score: None` and an explicit `reason`, never
    silently dropped.
    """
    t0 = time.monotonic()
    n_region = int(X_hsc_region.shape[0])
    k_over_n_region = float(k_mknn) / n_region if n_region > 0 else None

    row: Dict[str, Any] = {
        "kind": "mknn_regional",
        "region": int(region),
        "k_mknn": int(k_mknn),
        "n_region": n_region,
        "k_over_n_region": k_over_n_region,
        "null_scope": "region",
        "null_n": n_region,
        "n_permutations": int(n_permutations),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "null_quantile": float(null_quantile),
        "confidence_level": float(confidence_level),
        "k_frozen": int(k_frozen),
        "subsample_file": subsample_file,
    }
    undefined_fields: Dict[str, Any] = {
        "score": None,
        "chance_floor": None,
        "ratio_over_chance": None,
        "p_value": None,
        "null_mean": None,
        "null_std": None,
        "null_threshold": None,
        "clears_null": None,
        "ci_low": None,
        "ci_high": None,
        "degenerate": None,
        "hubness_skewness_hsc": None,
        "hubness_skewness_legacysurvey": None,
    }

    if n_region < region_partition.MIN_REGION_N:
        row.update(undefined_fields)
        row["status"] = "undefined"
        row["reason"] = region_partition.MIN_REGION_N_UNDEFINED_REASON
        row["wallclock_s"] = time.monotonic() - t0
        return row
    if k_mknn + 1 > n_region:
        row.update(undefined_fields)
        row["status"] = "undefined"
        row["reason"] = "k+1 > n_region"
        row["wallclock_s"] = time.monotonic() - t0
        return row

    score = mknn.mknn_score(X_hsc_region, X_ls_region, k_mknn)
    perm = mknn.permutation_null(
        X_hsc_region, X_ls_region, k_mknn, n_permutations, seed, null_quantile
    )
    boot = mknn.bootstrap_ci(
        X_hsc_region, X_ls_region, k_mknn, n_resamples, seed, confidence_level
    )
    hub_hsc = mknn.hubness_skewness(X_hsc_region, k_mknn)
    hub_ls = mknn.hubness_skewness(X_ls_region, k_mknn)
    floor = mknn.chance_floor(n_region, k_mknn)

    row["status"] = "ok"
    row["reason"] = None
    row["score"] = score
    row["chance_floor"] = floor
    row["ratio_over_chance"] = score / floor
    row["p_value"] = perm["p_value"]
    row["null_mean"] = perm["null_mean"]
    row["null_std"] = perm["null_std"]
    row["null_threshold"] = perm["null_threshold"]
    row["clears_null"] = perm["clears_null"]
    row["ci_low"] = boot["ci_low"]
    row["ci_high"] = boot["ci_high"]
    row["degenerate"] = boot["degenerate"]
    row["hubness_skewness_hsc"] = hub_hsc
    row["hubness_skewness_legacysurvey"] = hub_ls
    row["wallclock_s"] = time.monotonic() - t0
    return row


def _regional_row_print(r: Dict[str, Any]) -> None:
    if r["status"] != "ok":
        print(
            f"  region={r['region']} k={r['k_mknn']:>3} n_region={r['n_region']:>5}: "
            f"UNDEFINED ({r['reason']})"
        )
        return
    print(
        f"  region={r['region']} k={r['k_mknn']:>3} n_region={r['n_region']:>5} "
        f"k/n={r['k_over_n_region']:.4g} score%={r['score'] * 100:.4g} "
        f"floor%={r['chance_floor'] * 100:.4g} ratio={r['ratio_over_chance']:.3f} "
        f"p={r['p_value']:.4f} CI=[{r['ci_low'] * 100:.4g}%, {r['ci_high'] * 100:.4g}%] "
        f"clears_null={r['clears_null']}"
    )


def apply_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply `region_partition.VERDICT_RULE` mechanically to the eight regional rows
    (MKNN-07). Reads `HEADLINE_K`, `NULL_QUANTILE` and `CONFIDENCE_LEVEL` from
    `region_partition` -- chosen nowhere in this function, and never amended now that a
    regional number exists.

    A `k` HOLDS iff BOTH: (a) the two regions' `CONFIDENCE_LEVEL` bootstrap CIs at that k
    are disjoint, AND (b) the higher-scoring region's observed score strictly exceeds its
    OWN region-scoped null threshold (already the `NULL_QUANTILE` percentile of that
    region's own permutation null, per `mknn.permutation_null`). The headline call is made
    at `HEADLINE_K` alone; the remaining three k are reported as sensitivity only and
    cannot overturn or escalate it. Any cell recorded `status: "undefined"` makes that k's
    outcome `UNDEFINED`, and if that k is the headline k, the headline verdict becomes
    "NO DETECTABLE DIFFERENCE (comparison undefined at this split balance)".

    Returns a dict with `headline_k`, `headline_verdict`, `per_k`, `rule_text` (the
    verbatim `VERDICT_RULE`) and `density_caveat`. Prints the whole thing, including the
    verbatim rule text, so the applied rule and the rule as frozen are visible side by
    side in one output.
    """
    by_region_k: Dict[Tuple[int, int], Dict[str, Any]] = {
        (r["region"], r["k_mknn"]): r for r in rows
    }
    per_k: Dict[int, Dict[str, Any]] = {}
    for k in region_partition.MKNN_K_GRID:
        r0 = by_region_k.get((0, k))
        r1 = by_region_k.get((1, k))
        if r0 is None or r1 is None or r0["status"] != "ok" or r1["status"] != "ok":
            per_k[k] = {
                "outcome": "UNDEFINED",
                "reason": "a required regional cell is undefined at this k",
            }
            continue
        ci_disjoint = bool((r0["ci_high"] < r1["ci_low"]) or (r1["ci_high"] < r0["ci_low"]))
        higher, lower = (r0, r1) if r0["score"] >= r1["score"] else (r1, r0)
        exceeds_own_null = bool(higher["score"] > higher["null_threshold"])
        holds = bool(ci_disjoint and exceeds_own_null)
        per_k[k] = {
            "outcome": "HOLDS" if holds else "NO DETECTABLE DIFFERENCE",
            "ci_disjoint": ci_disjoint,
            "higher_region": higher["region"],
            "higher_score": higher["score"],
            "higher_null_threshold": higher["null_threshold"],
            "lower_region": lower["region"],
            "exceeds_own_null": exceeds_own_null,
        }

    headline = per_k.get(region_partition.HEADLINE_K, {"outcome": "UNDEFINED"})
    if headline["outcome"] == "UNDEFINED":
        headline_verdict = "NO DETECTABLE DIFFERENCE (comparison undefined at this split balance)"
    elif headline["outcome"] == "HOLDS":
        headline_verdict = (
            f"HOLDS at HEADLINE_K={region_partition.HEADLINE_K}: region "
            f"{headline['higher_region']} scores higher, CIs disjoint, and its observed "
            f"score strictly exceeds its own {region_partition.NULL_QUANTILE:.0%} null "
            "threshold."
        )
    else:
        headline_verdict = (
            f"NO DETECTABLE DIFFERENCE at HEADLINE_K={region_partition.HEADLINE_K} -- a "
            "complete, valid outcome per VERDICT_RULE; never a phase failure and never "
            "escalated by the sensitivity k."
        )

    density_caveat = (
        "D4-14 CAVEAT: a detected regional MKNN difference under this rule CANNOT be "
        "attributed to curvature rather than to regional density by anything in this "
        "phase -- the density-confound battery run in this phase is the REGN-02 "
        "correlation only, no density-matched null, no partial regression, no "
        "centroid-distance control, no density-matched stratification. MKNN is itself a "
        "k-NN statistic and therefore directly density-sensitive by construction."
    )

    result = {
        "headline_k": region_partition.HEADLINE_K,
        "headline_verdict": headline_verdict,
        "per_k": per_k,
        "rule_text": region_partition.VERDICT_RULE,
        "density_caveat": density_caveat,
    }

    print("\n" + "=" * 78)
    print("VERDICT_RULE (verbatim, from region_partition.py):")
    print("=" * 78)
    print(region_partition.VERDICT_RULE)
    print("=" * 78)
    print(
        f"Per-k outcomes (headline k = {region_partition.HEADLINE_K}, the remaining three "
        "are sensitivity only and cannot overturn or escalate the headline):"
    )
    for k in region_partition.MKNN_K_GRID:
        label = "HEADLINE" if k == region_partition.HEADLINE_K else "sensitivity"
        print(f"  k={k:>3} [{label:>11}]: {per_k[k]['outcome']}")
    print(f"\nHEADLINE VERDICT (k={region_partition.HEADLINE_K}): {headline_verdict}")
    print(f"\n{density_caveat}")

    return result


def _print_regional_closing_statements(rows: List[Dict[str, Any]]) -> None:
    """The three fixed closing statements this task's action requires: D4-14 (density
    confound, in the caveat's own words rather than a weaker restatement), D4-16 (regional
    numbers are not directly comparable to the global MKNN-02 figure), and MKNN-08 (the
    hubness caveat, substantiated by the printed k-occurrence skewness immediately beside
    it rather than asserted as prose alone)."""
    print("\n" + "=" * 78)
    print("CLOSING STATEMENTS")
    print("=" * 78)
    print(
        "D4-14: a regional MKNN difference cannot be separated from a regional density "
        "difference by anything in this phase. MKNN is itself a k-NN statistic and "
        "therefore directly density-sensitive by construction, and no density-matched "
        "null, partial regression, centroid-distance control or density-matched "
        "stratification was run -- see 04-04-SUMMARY.md's measured confound "
        "(spearman(density, signed_projection) = +0.8208, n=9500)."
    )
    print(
        "D4-16: the regional numbers above are NOT directly comparable to the global "
        "MKNN-02 figure, because k/n differs between a region and the whole cloud. A "
        "smaller region raises chance alignment, which is exactly what that region's own "
        "null absorbs -- the comparison that carries meaning is each region against its "
        "own null and its own chance floor, never region against global."
    )
    print("MKNN-08 hubness caveat, substantiated by the printed k-occurrence skewness:")
    for r in rows:
        if r["status"] != "ok":
            print(f"  region={r['region']} k={r['k_mknn']:>3}: n/a (cell undefined)")
            continue
        print(
            f"  region={r['region']} k={r['k_mknn']:>3}: "
            f"hub_hsc={r['hubness_skewness_hsc']:.3f}  "
            f"hub_legacysurvey={r['hubness_skewness_legacysurvey']:.3f}"
        )
    print(
        "\nEvery regional result above carries the hubness caveat: k-NN alignment metrics "
        "are hubness-sensitive in high dimensions (Radovanovic, Nanopoulos and Ivanovic, "
        "JMLR 2010)."
    )


def selfcheck() -> bool:
    """MKNN-01's known-answer assertions on synthetic data. This runner flag is the
    phase's automated implementation check (D4-18 declines `tests/test_mknn.py`)."""
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            ok = False

    rng = np.random.default_rng(20260822)

    X = rng.normal(size=(400, 16))
    Y = X.copy()
    check(
        "identical row-aligned pair scores exactly 1.0 at k=10",
        mknn.mknn_score(X, Y, 10) == 1.0,
    )

    Z = rng.normal(size=(20, 8))
    check("k = n - 1 scores exactly 1.0", mknn.mknn_score(Z, Z.copy(), 19) == 1.0)

    A = rng.normal(size=(400, 16))
    B = rng.normal(size=(400, 16))
    ind_score = mknn.mknn_score(A, B, 10)
    floor = mknn.chance_floor(400, 10)
    check(
        "independent Gaussian clouds (400, 16) land within a factor of 3 of "
        "chance_floor(400, 10)",
        floor / 3.0 <= ind_score <= floor * 3.0,
    )

    perm = rng.permutation(400)
    base = mknn.mknn_score(X, Y, 10)
    check(
        "simultaneous row permutation of both sides leaves the score unchanged",
        mknn.mknn_score(X[perm], Y[perm], 10) == base,
    )
    check(
        "permuting one side only changes the score",
        mknn.mknn_score(X, Y[perm], 10) != base,
    )

    for name, fn in [
        ("k + 1 > n raises ValueError", lambda: mknn.mknn_score(X[:5], Y[:5], 10)),
        ("n < 2 raises ValueError", lambda: mknn.mknn_score(X[:1], Y[:1], 1)),
        ("k < 1 raises ValueError", lambda: mknn.mknn_score(X, Y, 0)),
    ]:
        try:
            fn()
            check(name, False)
        except ValueError:
            check(name, True)

    return ok


PAPER_RANGE_LOW = 0.34   # percent, arXiv:2509.19453 Table 2, Legacy-vs-HSC column
PAPER_RANGE_HIGH = 2.25  # percent, same source
PAPER_N = 101725


def _header() -> None:
    print(
        f"{'k_mknn':>7} {'n':>7} {'score%':>9} {'floor%':>9} {'ratio':>8} "
        f"{'p':>8} {'ci_low%':>9} {'ci_high%':>9}"
    )


def _row(r: Dict[str, Any]) -> None:
    print(
        f"{r['k_mknn']:>7} {r['n']:>7} {r['score'] * 100:>9.4g} "
        f"{r['chance_floor'] * 100:>9.4g} {r['ratio_over_chance']:>8.3f} "
        f"{r['p_value']:>8.4f} {r['ci_low'] * 100:>9.4g} {r['ci_high'] * 100:>9.4g}"
    )


def summarize(records: List[Dict[str, Any]]) -> None:
    """Full read-out: one line per k with the raw score, the chance floor, the ratio
    over chance, the permutation p-value, the 95% CI and BOTH sides' k-occurrence
    hubness skewness (MKNN-08) -- printed beside the numbers, not asserted in prose
    alone. Followed by a fixed comparison block naming this run's n, the origin
    paper's raw range and n (D4-19), and the k caveat (MKNN-06 sets the grid, not
    the paper)."""
    print("\nGlobal crossmodal MKNN read-out, full grid:")
    print(
        f"{'k_mknn':>7} {'n':>7} {'score%':>9} {'floor%':>9} {'ratio':>8} "
        f"{'p':>8} {'ci_low%':>9} {'ci_high%':>9} {'hub_hsc':>9} {'hub_ls':>9}"
    )
    for r in records:
        print(
            f"{r['k_mknn']:>7} {r['n']:>7} {r['score'] * 100:>9.4g} "
            f"{r['chance_floor'] * 100:>9.4g} {r['ratio_over_chance']:>8.3f} "
            f"{r['p_value']:>8.4f} {r['ci_low'] * 100:>9.4g} {r['ci_high'] * 100:>9.4g} "
            f"{r['hubness_skewness_hsc']:>9.3f} {r['hubness_skewness_legacysurvey']:>9.3f}"
        )

    n_this = records[0]["n"] if records else None
    print("\n--- comparison against the origin paper (D4-19) ---")
    print(f"this run:  n = {n_this}, raw MKNN 4sf shown above per k, ratio-over-chance carries")
    print("           the comparison across the n mismatch, NOT the raw number")
    print(
        f"paper:     Legacy-vs-HSC published range {PAPER_RANGE_LOW}% - {PAPER_RANGE_HIGH}% "
        f"at n = {PAPER_N} (arXiv:2509.19453 Table 2)"
    )
    print(
        f"NOTE: n = {n_this} here vs n = {PAPER_N} in the paper -- raw numbers are NOT "
        "directly comparable across this n mismatch; the ratio-over-chance framing is what"
    )
    print("carries the comparison, per D4-19.")
    print(
        "NOTE: the paper does not state the k behind Table 2. This phase's grid "
        "{5, 10, 20, 50} is set by MKNN-06, never by the paper."
    )
    if records:
        hub_all = [r["hubness_skewness_hsc"] for r in records] + [
            r["hubness_skewness_legacysurvey"] for r in records
        ]
        hub_range = f"{min(hub_all):.3f} to {max(hub_all):.3f}"
    else:
        hub_range = "n/a"
    print(
        f"CAVEAT (MKNN-08): k-occurrence hubness skewness ranges {hub_range} across both "
        "embedding sides and every k above (see hub_hsc / hub_ls columns) -- MKNN alignment "
        "metrics are hubness-sensitive in high dimensions; this caveat applies to every "
        "result above."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["global", "partition", "regional"], default="global")
    p.add_argument("--mknn-k", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--null-quantile", type=float, default=0.99)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    a = build_arg_parser().parse_args()

    if a.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    if a.mode == "regional":
        # D4-11/REGN-04/T-04-07: fail loudly rather than compute anything when the
        # pre-registration or the frozen partition artifact is absent. This guard must run
        # BEFORE any regional cell is computed -- it is what makes the pre-registration
        # commit's ordering enforceable, not merely documented.
        region_partition.assert_preregistered()
        partition_artifact = cache.cache_path("04_region_partition", "npz")
        if not partition_artifact.exists():
            raise FileNotFoundError(
                f"--mode regional requires the frozen partition artifact at "
                f"{partition_artifact}, which does not exist. Run --mode partition first "
                "to produce it (a later plan in this phase implements --mode partition)."
            )

        X_hsc, X_ls, subsample_file = load_pu_pair()
        with np.load(partition_artifact) as part:
            labels = part["labels"]
            keep_idx = part["keep_idx"]

        record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
        record_path.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 78)
        print(
            f"Regional crossmodal MKNN -- MKNN_K_GRID={region_partition.MKNN_K_GRID}, "
            f"HEADLINE_K={region_partition.HEADLINE_K}, "
            f"N_PERMUTATIONS={region_partition.N_PERMUTATIONS}, "
            f"N_BOOTSTRAP={region_partition.N_BOOTSTRAP}"
        )
        print("=" * 78)
        print(f"record_path = {record_path}\n")

        rows: List[Dict[str, Any]] = []
        with record_path.open("a") as fh:
            for region in (0, 1):
                region_rows_idx = keep_idx[labels == region]
                n_region = int(region_rows_idx.shape[0])
                # Per D4-16: subset BOTH embeddings to the region's own rows first, then
                # every N_k, score, null and CI computed below lives inside that same
                # index set. No global null is computed anywhere in this branch.
                X_hsc_region = X_hsc[region_rows_idx]
                X_ls_region = X_ls[region_rows_idx]
                print(f"region {region}: n_region={n_region}")
                for k in region_partition.MKNN_K_GRID:
                    r = run_regional_cell(
                        region,
                        k,
                        X_hsc_region,
                        X_ls_region,
                        n_permutations=region_partition.N_PERMUTATIONS,
                        n_resamples=region_partition.N_BOOTSTRAP,
                        seed=region_partition.SEED,
                        null_quantile=region_partition.NULL_QUANTILE,
                        confidence_level=region_partition.CONFIDENCE_LEVEL,
                        k_frozen=region_partition.K_FROZEN,
                        subsample_file=subsample_file,
                    )
                    fh.write(json.dumps(r, default=float) + "\n")
                    fh.flush()
                    rows.append(r)
                    _regional_row_print(r)

        apply_verdict(rows)
        _print_regional_closing_statements(rows)
        return

    if a.mode == "partition":
        _, X_ls, subsample_file = load_pu_pair()
        if a.smoke:
            print(
                "SMOKE: n=800, k=25, d=3, k_density=10 -- proves the partition path runs "
                "end to end, writes nothing.\n"
            )
            run_partition(
                X_ls[:800],
                k=25,
                d=3,
                k_density=10,
                min_norm_percentile=region_partition.MIN_NORM_PERCENTILE,
                seed=a.seed,
                subsample_file=subsample_file,
                smoke=True,
            )
            return
        print("=" * 78)
        print(
            f"Region partition -- n={X_ls.shape[0]}, k={region_partition.K_FROZEN}, "
            f"d={region_partition.FIELD_D}, k_density={region_partition.K_DENSITY}"
        )
        print("=" * 78)
        run_partition(
            X_ls,
            k=region_partition.K_FROZEN,
            d=region_partition.FIELD_D,
            k_density=region_partition.K_DENSITY,
            min_norm_percentile=region_partition.MIN_NORM_PERCENTILE,
            seed=region_partition.SEED,
            subsample_file=subsample_file,
        )
        return

    if a.mode != "global":
        raise NotImplementedError(
            f"--mode {a.mode!r} is pre-registered but not implemented until a later "
            "plan in this phase (04-03 onward)."
        )

    X_hsc, X_ls, subsample_file = load_pu_pair()

    if a.smoke:
        print(
            "SMOKE: 800 rows, k_mknn=10, 200 permutations, 200 resamples -- proves the "
            "path runs, writes nothing.\n"
        )
        _header()
        r = run_global_cell(
            X_hsc[:800], X_ls[:800], 10, 200, 200, a.seed, a.null_quantile,
            a.confidence_level, subsample_file,
        )
        _row(r)
        return

    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Global crossmodal MKNN -- n={X_hsc.shape[0]}, mknn_k={a.mknn_k}")
    print("=" * 78)
    print(f"record_path = {record_path}\n")

    _header()
    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for k in a.mknn_k:
            r = run_global_cell(
                X_hsc, X_ls, k, a.n_permutations, a.n_resamples, a.seed,
                a.null_quantile, a.confidence_level, subsample_file,
            )
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _row(r)

    summarize(records)


if __name__ == "__main__":
    main()
