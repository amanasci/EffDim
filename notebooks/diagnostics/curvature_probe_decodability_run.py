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
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

from pu_manifold import cache, cae, chart_curvature, linear_probe

DEFAULT_RECORD = cache.cache_path("05_curvature_probe_decodability", "jsonl")
SELFCHECK_RECORD = cache.cache_path("05_probe_selfcheck", "jsonl")

CANONICAL_SEED_STEMS = (20260813, 20260814, 20260815)
"""The three sealed CAE seeds D5-05's inter-seed diagnostics run over -- fixed regardless of
what `--seeds` a given `--mode field` invocation requested, since Task 1's own action text
runs the three seeds one at a time across separate invocations. The diagnostics step only
runs once all three of these seeds' cached field artifacts exist on disk."""

# --- Sealed CAE architecture constants -- must match curvature_field_pu_run.py's build_cae /
# load_converged_model exactly, since these are the same three sealed checkpoints. ------------
PU_IN_DIM = 768
PU_EMBED_DIM = 40
PU_CHART_DIM = 20
PU_HIDDEN_WIDTH = 250
PU_ACTIVATION = "silu"
CONVERGED_CKPT_STEM = "03_converged_cae_pu"

CURVATURE_SOURCE_FUNCTION_NAME = "chart_curvature.chart_curvature_field"
"""The load-bearing correction to D5-03: the string this phase's CURVATURE_SOURCE_FUNCTION
constant will carry at the 05-04 freeze, naming the function actually called below."""


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


def build_cae(n_charts: int, device: torch.device = torch.device("cpu")) -> "cae.ChartAutoEncoder":
    """Constructed exactly as `curvature_field_pu_run.build_cae`, then moved to `device`
    AFTER construction (`model.to(device)`) -- never by passing `device=` into the
    constructor -- so `torch.manual_seed(seed)`'s RNG consumption order is unaffected. Default
    `cpu`: zero behaviour change. All three sealed checkpoints are `n_charts=4`.
    """
    model = cae.ChartAutoEncoder(
        in_dim=PU_IN_DIM,
        embed_dim=PU_EMBED_DIM,
        chart_dim=PU_CHART_DIM,
        n_charts=n_charts,
        hidden=[PU_HIDDEN_WIDTH, PU_HIDDEN_WIDTH, PU_HIDDEN_WIDTH],
        activation=PU_ACTIVATION,
    )
    return model.to(device)


def load_converged_model(n_charts: int, seed: int, device: torch.device) -> Any:
    """Rebuild the converged CAE from its sealed checkpoint. Never retrains: a missing
    checkpoint is a named `FileNotFoundError`, never a silent fallback to training -- silently
    training a replacement would put a DIFFERENT model behind the same reported number.
    Validates the checkpoint's own recorded `n_charts` and `seed` match the requested ones,
    raising `ValueError` on mismatch.
    """
    ckpt_path = cache.cache_path(f"{CONVERGED_CKPT_STEM}_nc{n_charts}_seed{seed}", "pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No converged checkpoint at {ckpt_path}. This runner never trains a replacement."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if int(ckpt["n_charts"]) != n_charts or int(ckpt["seed"]) != seed:
        raise ValueError(
            f"{ckpt_path} carries n_charts={ckpt['n_charts']} seed={ckpt['seed']}, but "
            f"n_charts={n_charts} seed={seed} was requested."
        )
    model = build_cae(n_charts, device=device).double()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def extract_seed_field(
    seed: int,
    n_charts: int,
    batch_size: int,
    x64: np.ndarray,
    field_stem_prefix: str,
    subsample_file: str,
) -> Dict[str, np.ndarray]:
    """The decoder-side curvature field for ONE seed, cached through `cache.npz_cache` at
    stem `f"{field_stem_prefix}_seed{seed}"`. The `cfg` dict keyed into the manifest carries
    `seed`, `n_charts`, `mode`, `batch_size`, `n_rows`, `subsample_file`,
    `curvature_convention` and `source_function`, so a re-run with a different configuration
    raises the manifest mismatch rather than silently reusing a stale field.

    This is the load-bearing correction to D5-03: the call is routed through
    `chart_curvature.chart_curvature_field`, which takes AMBIENT 768-d rows (not a latent),
    internally encodes, takes `chart_probs(z).argmax(dim=1)`, computes per-chart curvature and
    reassembles in the original row order. `notebooks/pu_manifold/decoder_curvature.py` --
    the module D5-03 itself names -- is `chart_curvature.py` with the
    `chart_decoders[chart_idx]` two-hop composition removed, built for Phase 02.6's
    no-chart-index substrates; `ChartAutoEncoder` has no bare single-hop decode entry point
    matching that module's signature, so nothing is imported from it here.
    """
    stem = f"{field_stem_prefix}_seed{seed}"
    cfg: Dict[str, Any] = {
        "seed": int(seed),
        "n_charts": int(n_charts),
        "mode": "reverse",
        "batch_size": int(batch_size),
        "n_rows": int(x64.shape[0]),
        "subsample_file": str(subsample_file),
        "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
        "source_function": CURVATURE_SOURCE_FUNCTION_NAME,
    }

    def _compute() -> Dict[str, np.ndarray]:
        model, _ = load_converged_model(n_charts, seed, torch.device("cpu"))
        x_tensor = torch.as_tensor(x64, dtype=torch.float64)
        field = chart_curvature.chart_curvature_field(
            model, x_tensor, batch_size=batch_size, mode="reverse"
        )
        return {
            "H_norm": field["H_norm"].detach().cpu().numpy().astype(np.float64),
            "H_vec": field["H_vec"].detach().cpu().numpy().astype(np.float64),
            "chart_assignment": field["chart_assignment"].detach().cpu().numpy().astype(np.int64),
            "metric_condition_number": field["metric_condition_number"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
            "lambda_min": field["lambda_min"].detach().cpu().numpy().astype(np.float64),
            "lambda_max": field["lambda_max"].detach().cpu().numpy().astype(np.float64),
            "det_g": field["det_g"].detach().cpu().numpy().astype(np.float64),
            "log10_det_g": field["log10_det_g"].detach().cpu().numpy().astype(np.float64),
            "n_charts_used": np.asarray(field["n_charts_used"]),
        }

    return cache.npz_cache(stem, cfg, _compute)


def run_field_mode(a: argparse.Namespace) -> None:
    """`--mode field`: extract the decoder-side `||H||` field for each seed in `a.seeds`,
    over the `legacysurvey` embedding (the CAE's own fitted substrate -- see
    `curvature_field_pu_run._load_subsample`, which reads the same column). `--smoke` uses
    only the first `a.smoke_n` rows and only the first seed, bypasses the cache entirely so a
    reduced field can never be mistaken for the real one, and writes nothing.
    """
    _, X_ls, subsample_file = load_pu_pair()

    if a.smoke:
        seed = a.seeds[0]
        print(
            f"SMOKE: n={a.smoke_n}, seed={seed} only -- proves the field extraction path "
            "runs end to end against a genuine sealed checkpoint, bypasses the cache, writes "
            "nothing.\n"
        )
        model, _ = load_converged_model(a.n_charts, seed, torch.device("cpu"))
        x64 = X_ls[: a.smoke_n].astype(np.float64)
        x_tensor = torch.as_tensor(x64, dtype=torch.float64)
        field = chart_curvature.chart_curvature_field(
            model, x_tensor, batch_size=a.batch_size, mode="reverse"
        )
        H_norm = field["H_norm"].detach().cpu().numpy().astype(np.float64)
        print(
            f"seed {seed}: n={x64.shape[0]}  median H_norm={float(np.median(H_norm)):.6g}  "
            f"n_distinct_H_norm={len(np.unique(H_norm))}  n_charts_used={field['n_charts_used']}"
        )
        return

    print("=" * 78)
    print(f"Curvature field extraction -- seeds={a.seeds}, n_charts={a.n_charts}")
    print("=" * 78)
    for seed in a.seeds:
        t0 = time.monotonic()
        x64 = X_ls.astype(np.float64)
        result = extract_seed_field(seed, a.n_charts, a.batch_size, x64, a.field_stem, subsample_file)
        H_norm = result["H_norm"]
        print(
            f"seed {seed}: wallclock={time.monotonic() - t0:.1f}s  "
            f"median H_norm={float(np.median(H_norm)):.6g}  "
            f"n_distinct_H_norm={len(np.unique(H_norm))}  "
            f"n_charts_used={int(np.asarray(result['n_charts_used']))}"
        )

    run_inter_seed_diagnostics(a.field_stem)


def run_pool_mode(a: argparse.Namespace) -> None:
    """`--mode pool` is pre-registered but not implemented until plan 05-03."""
    raise NotImplementedError(
        "--mode pool is pre-registered but not implemented until plan 05-03."
    )


def run_bucketed_mode(a: argparse.Namespace) -> None:
    """`--mode bucketed` -- the D5-10 guard, complete in this task even though the branch it
    guards is not. Before touching any data: calls `linear_probe.assert_preregistered()`,
    then checks the pooled field artifact exists, raising `FileNotFoundError` naming the
    missing path. Only after both pass does it reach the body, which raises
    `NotImplementedError` until plan 05-05. With the pre-registration constants unset today,
    the first guard fires and this mode is dead.
    """
    linear_probe.assert_preregistered()
    field_path = cache.cache_path(a.field_stem, "npz")
    if not field_path.exists():
        raise FileNotFoundError(
            f"--mode bucketed requires the frozen pooled curvature field artifact at "
            f"{field_path}, which does not exist. Run --mode field then --mode pool first "
            "to produce it."
        )
    raise NotImplementedError(
        "--mode bucketed's body is pre-registered but not implemented until plan 05-05."
    )


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


def _direction_axis_report(H_vec_a: np.ndarray, H_vec_b: np.ndarray, name: str) -> Dict[str, Any]:
    """The direction axis, reported beside every rank statistic and never separately, per the
    spike-findings-effdim rule that a rank statistic from this teacher is never reported
    without it. This is INTER-SEED agreement, NOT agreement with true curvature: no analytic
    truth exists for the PU field, so this is a weaker check than the spike-findings rule
    intends -- it is used because it is the strongest direction axis this phase can construct,
    not because it is equivalent.

    Row-normalizes both ``H_vec`` fields with the codebase's own divide-by-zero floor
    (``np.maximum(norm, 1e-12)``), takes the per-row dot product of the resulting unit
    vectors, and reports the median cosine, the 25th/75th percentile cosine, and the fraction
    of rows whose cosine is negative (the anti-alignment fraction).
    """
    a = np.asarray(H_vec_a, dtype=np.float64)
    b = np.asarray(H_vec_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"_direction_axis_report: shape mismatch {a.shape} vs {b.shape}.")
    norm_a = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    norm_b = np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    unit_a = a / norm_a[:, None]
    unit_b = b / norm_b[:, None]
    cosine = np.sum(unit_a * unit_b, axis=1)
    median_cosine = float(np.median(cosine))
    q25_cosine = float(np.percentile(cosine, 25))
    q75_cosine = float(np.percentile(cosine, 75))
    fraction_negative_cosine = float(np.mean(cosine < 0.0))
    print(
        f"{name} (direction axis, inter-seed only -- not agreement with true curvature): "
        f"median_cosine={median_cosine:+.4f}  q25={q25_cosine:+.4f}  q75={q75_cosine:+.4f}  "
        f"fraction_negative_cosine={fraction_negative_cosine:.4f}"
    )
    return {
        "median_cosine": median_cosine,
        "q25_cosine": q25_cosine,
        "q75_cosine": q75_cosine,
        "fraction_negative_cosine": fraction_negative_cosine,
        "n": int(a.shape[0]),
    }


def _load_cached_seed_field(seed: int, field_stem_prefix: str) -> Optional[Dict[str, np.ndarray]]:
    """Load one seed's cached field npz if it exists, else `None`. Never computes -- diagnostics
    only read artifacts Task 1 has already produced."""
    stem = f"{field_stem_prefix}_seed{seed}"
    path = cache.cache_path(stem, "npz")
    if not path.exists():
        return None
    return dict(np.load(path))


def run_inter_seed_diagnostics(field_stem_prefix: str) -> None:
    """D5-05: pairwise inter-seed Spearman (the rank axis) with the direction axis reported
    beside every entry, per-seed summary statistics (RESEARCH Pitfall 2's piecewise-constant
    symptom measurements), and the r/R non-application disclosure -- written through
    `cache.json_cache` to `notebooks/.cache/05_inter_seed_diagnostics.json`. Runs only once all
    three of `CANONICAL_SEED_STEMS`' field artifacts already exist on disk; otherwise prints a
    one-line notice and returns without writing anything, since a partial invocation (this
    plan's Task 1 runs one seed per command) leaves diagnostics with fewer than three seeds to
    compare. Computes no pooled field and chooses no pooling method -- that is 05-03's blocking
    checkpoint.
    """
    fields: Dict[int, Dict[str, np.ndarray]] = {}
    for seed in CANONICAL_SEED_STEMS:
        loaded = _load_cached_seed_field(seed, field_stem_prefix)
        if loaded is None:
            missing_path = cache.cache_path(f"{field_stem_prefix}_seed{seed}", "npz")
            print(
                f"Inter-seed diagnostics skipped: seed {seed}'s field artifact does not yet "
                f"exist at {missing_path}. Diagnostics run once all three canonical seeds "
                f"({CANONICAL_SEED_STEMS}) are cached."
            )
            return
        fields[seed] = loaded

    print("=" * 78)
    print("Inter-seed diagnostics (D5-05): rank axis + direction axis beside it")
    print("=" * 78)

    def _compute() -> Dict[str, Any]:
        pairwise_spearman: Dict[str, Any] = {}
        pairwise_direction: Dict[str, Any] = {}
        for seed_a, seed_b in itertools.combinations(CANONICAL_SEED_STEMS, 2):
            pair_name = f"{seed_a}_vs_{seed_b}"
            spearman_report = _spearman_report(
                fields[seed_a]["H_norm"],
                fields[seed_b]["H_norm"],
                f"inter-seed spearman({seed_a}, {seed_b})",
            )
            direction_report = _direction_axis_report(
                fields[seed_a]["H_vec"],
                fields[seed_b]["H_vec"],
                f"inter-seed direction({seed_a}, {seed_b})",
            )
            pairwise_spearman[pair_name] = spearman_report
            pairwise_direction[pair_name] = direction_report

        per_seed: Dict[str, Any] = {}
        for seed in CANONICAL_SEED_STEMS:
            H_norm = fields[seed]["H_norm"]
            chart_assignment = fields[seed]["chart_assignment"]
            chart_fractions = {
                str(int(c)): float(np.mean(chart_assignment == c))
                for c in sorted(np.unique(chart_assignment).tolist())
            }
            per_seed[str(seed)] = {
                "median_h_norm": float(np.median(H_norm)),
                "min_h_norm": float(np.min(H_norm)),
                "max_h_norm": float(np.max(H_norm)),
                "n_distinct_h_norm": int(len(np.unique(np.round(H_norm, 6)))),
                "chart_fractions": chart_fractions,
                "median_log10_det_g": float(np.median(fields[seed]["log10_det_g"])),
                "n_charts_used": int(np.asarray(fields[seed]["n_charts_used"])),
                "n": int(H_norm.shape[0]),
            }

        return {
            "pairwise_spearman": pairwise_spearman,
            "pairwise_direction": pairwise_direction,
            "per_seed": per_seed,
            "r_over_R": None,
            "r_over_R_reason": (
                "not defined for an autodiff decoder-side estimator: chart_curvature_field "
                "has no neighbourhood and no k"
            ),
            "seed_stems": list(CANONICAL_SEED_STEMS),
        }

    cfg = {
        "kind": "05_inter_seed_diagnostics",
        "field_stem_prefix": field_stem_prefix,
        "seed_stems": list(CANONICAL_SEED_STEMS),
    }
    cache.json_cache("05_inter_seed_diagnostics", cfg, _compute)
    print(f"wrote {cache.cache_path('05_inter_seed_diagnostics', 'json')}")


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

    if a.mode == "field":
        run_field_mode(a)
        return

    if a.mode == "pool":
        run_pool_mode(a)
        return

    if a.mode == "bucketed":
        run_bucketed_mode(a)
        return

    raise NotImplementedError(f"--mode {a.mode!r} is not a recognized mode.")


if __name__ == "__main__":
    main()
