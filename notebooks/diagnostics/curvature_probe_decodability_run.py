"""Phase 5 curvature-conditioned linear decodability runner.

`--mode field` extracts the decoder-side `||H||` field for each seed in `--seeds`, via
`chart_curvature.chart_curvature_field` against a genuine sealed CAE checkpoint, caching each
seed's field through `cache.npz_cache`. `--mode pool` REFUSES BY NAME (05-03-DECISION.md
ratified, one-way, NOT to pool the three seeds -- superseding 05-CONTEXT.md D5-04); use
`--mode perseed` instead, which bucketizes each seed's own field independently and writes three
per-seed bucket artifacts, never one pooled artifact. `--mode bucketed` requires both
`linear_probe.assert_preregistered()` and all three per-seed bucket artifacts to exist before it
will even attempt anything -- the D5-10 guard -- and, as of plan 05-05, its body fits ONE global
ridge map (D5-02) and buckets the held-out residuals three times, once per seed's frozen
`BUCKET_EDGES_PER_SEED` entry, producing three per-seed verdicts and one phase verdict, both
applied only through the frozen `linear_probe` rule functions. `--selfcheck` is this plan's own automated implementation check: it runs the complete
probe-to-verdict path on a synthetic, dimensionally PU-shaped fixture with a planted linear map
and a planted curvature-to-residual ordering, and writes exactly one JSONL row tagged
`data_source = "synthetic_planted"`. No PU probe number is computed by any command below.

    python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field --smoke
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode pool     # refuses
    python notebooks/diagnostics/curvature_probe_decodability_run.py --mode perseed
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

from pu_manifold import cache, cae, chart_curvature, curvature_probe, linear_probe

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


POOLING_REFUSAL_MESSAGE = (
    "Seed pooling was put to the developer at the 05-03 Task 1 blocking checkpoint and "
    "ratified as NOT DONE. See "
    ".planning/phases/05-curvature-conditioned-linear-decodability/05-03-DECISION.md -- "
    "05-CONTEXT.md D5-04 (pool the three cached CAE seeds into one averaged ||H|| field) is "
    "superseded by that ratified, one-way decision. Use --mode perseed instead."
)
"""The refusal text `run_pool_mode` raises and `--pooling-method`'s tripwire raises. A named,
durable pointer to the record rather than a bare NotImplementedError, so a later reader lands
on the decision, not on a stub."""


def run_pool_mode(a: argparse.Namespace) -> None:
    """`--mode pool` no longer stubs with `NotImplementedError` -- it refuses BY NAME. Seed
    pooling was ratified NOT DONE at the `05-03` Task 1 blocking checkpoint
    (`05-03-DECISION.md`), which supersedes `05-CONTEXT.md` D5-04. Raises `RuntimeError`
    naming the decision record and directing the caller to `--mode perseed`.
    """
    raise RuntimeError(POOLING_REFUSAL_MESSAGE)


def _fit_and_evaluate(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alpha_grid: Any,
    alpha_per_target: bool,
    fit_intercept: bool,
    r2_multioutput: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, float]:
    """The ONE call site of the ``linear_probe`` module's ridge-fit entry point in this file.
    Fits on ``(X_train, Y_train)``, then predicts and scores on ``(X_test, Y_test)`` via
    ``linear_probe.predict_probe`` / ``per_point_residuals`` / ``aggregate_r2``. Shared by BOTH
    ``selfcheck()`` (the synthetic fixture) and ``run_bucketed_mode()`` (the real PU probe) so
    there is structurally one fit path in this file, not one path per caller.
    Returns ``(fit, Y_pred_test, residuals_test, r2)``.
    """
    fit = linear_probe.fit_probe(X_train, Y_train, alpha_grid, alpha_per_target, fit_intercept)
    Y_pred_test = linear_probe.predict_probe(fit, X_test)
    residuals_test = linear_probe.per_point_residuals(Y_test, Y_pred_test)
    r2 = linear_probe.aggregate_r2(Y_test, Y_pred_test, multioutput=r2_multioutput)
    return fit, Y_pred_test, residuals_test, r2


def _score_one_seed(
    seed: int,
    H_norm: np.ndarray,
    bucket_labels: np.ndarray,
    test_idx: np.ndarray,
    residuals_test: np.ndarray,
    Y_true_test: np.ndarray,
    Y_pred_test: np.ndarray,
    n_buckets: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    confidence_level: float,
    size_match_n_repeats: int,
    size_match_seed: int,
    verdict_rule: str,
    r2_multioutput: str,
) -> Dict[str, Any]:
    """One seed's D5-07/D5-08/D5-09 scoring over the SAME shared ``residuals_test`` /
    ``Y_pred_test`` every seed reads (one global fit, D5-02) -- only the bucketing differs per
    seed. ``H_norm``/``bucket_labels`` are that seed's full-field arrays (the ones the frozen
    ``BUCKET_EDGES_PER_SEED[i]`` entry was cut over); ``bucket_labels[test_idx]`` is a lookup
    against those already-assigned labels, never a re-cut on the test split.

    Returns a dict carrying: ``bucket_stats`` (per-bucket n / mean residual / R-squared / CI, in
    ascending bucket-index order so index 0 is lowest-curvature and -1 is highest),
    ``full_field_bucket_counts``, ``realized_bucket_counts`` (the TEST-split counts D5-08
    requires -- RESEARCH Pitfall 4), ``size_match`` (``linear_probe.size_matched_check``'s
    return, called with the REALIZED test-split arrays), ``spearman`` (the D5-07 continuous
    statistic), ``verdict`` and ``criteria`` (``linear_probe.apply_verdict_rule``'s own return,
    the ONLY source of this seed's verdict string).
    """
    labels_test = bucket_labels[test_idx]
    full_field_counts = linear_probe.bucket_counts(bucket_labels, n_buckets)["counts"]

    bucket_stats = []
    for b in range(n_buckets):
        mask = labels_test == b
        r_bucket = residuals_test[mask]
        ci = linear_probe.bucket_residual_ci(r_bucket, n_bootstrap, bootstrap_seed, confidence_level)
        r2_bucket = linear_probe.aggregate_r2(
            Y_true_test[mask], Y_pred_test[mask], multioutput=r2_multioutput
        )
        bucket_stats.append(
            {
                "bucket_index": b,
                "n": int(mask.sum()),
                "score": ci["score"],
                "r2": r2_bucket,
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "degenerate": ci["degenerate"],
                "confidence_level": ci["confidence_level"],
                "n_resamples": ci["n_resamples"],
            }
        )

    size_match = linear_probe.size_matched_check(
        residuals_test, labels_test, size_match_n_repeats, size_match_seed, confidence_level
    )

    spearman_result = _spearman_report(
        H_norm[test_idx], residuals_test, f"D5-07 continuous seed {seed}"
    )

    verdict_result = linear_probe.apply_verdict_rule(bucket_stats, size_match, verdict_rule)

    return {
        "bucket_stats": bucket_stats,
        "full_field_bucket_counts": tuple(int(c) for c in full_field_counts),
        "realized_bucket_counts": tuple(b["n"] for b in bucket_stats),
        "size_match": size_match,
        "spearman": spearman_result,
        "verdict": verdict_result["verdict"],
        "criteria": verdict_result["criteria"],
    }


def _load_bucket_artifact(seed: int, bucket_stem_prefix: str) -> Dict[str, Any]:
    """Load ONE seed's frozen per-seed bucket artifact -- never recomputes, never falls back to
    ``--mode perseed``. Returns the raw arrays plus the sidecar manifest dict (used for the
    subsample-path provenance check)."""
    stem = f"{bucket_stem_prefix}_seed{seed}"
    npz_path = cache.cache_path(stem, "npz")
    with np.load(npz_path) as z:
        H_norm = np.asarray(z["H_norm"], dtype=np.float64)
        bucket_labels = np.asarray(z["bucket_labels"], dtype=np.int64)
        bucket_edges = tuple(float(v) for v in np.asarray(z["bucket_edges"], dtype=np.float64))
    manifest = json.loads(cache.cache_path(stem, "meta.json").read_text())
    return {
        "H_norm": H_norm,
        "bucket_labels": bucket_labels,
        "bucket_edges": bucket_edges,
        "manifest": manifest,
    }


def _conditioning_diagnostics(
    X_train: np.ndarray, alpha_grid: Any, selected_alpha: float
) -> Dict[str, Any]:
    """RESEARCH A2's stated reason for ridge -- that the 768-d design matrix is effectively
    rank-deficient at the manifold's established 18-to-25 intrinsic dimension -- CHECKED here
    against the training split's own measured singular spectrum, per A2's own instruction that
    the reason should be verified before being asserted as fact. Computes no statistic that
    feeds any verdict and calls no fit: it reads `selected_alpha` (an OUTPUT of the frozen
    RidgeCV selection rule, computed once at the one `_fit_and_evaluate` call site) rather than
    refitting. There is one training split and one fit, so this is called exactly once, never
    per seed.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    centered = X_train - X_train.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1])
    largest = float(singular_values[0])

    def _effective_rank(threshold_fraction: float) -> int:
        return int(np.sum(singular_values > threshold_fraction * largest))

    variance = singular_values**2
    total_variance = float(variance.sum())
    cumvar = np.cumsum(variance) / total_variance

    alpha_grid_floats = tuple(float(v) for v in alpha_grid)
    alpha_at_grid_boundary = bool(
        selected_alpha == min(alpha_grid_floats) or selected_alpha == max(alpha_grid_floats)
    )

    return {
        "kind": "probe_conditioning",
        "singular_values_head": [float(v) for v in singular_values[:40]],
        "condition_number": condition_number,
        "effective_rank_1pct": _effective_rank(0.01),
        "effective_rank_0p1pct": _effective_rank(0.001),
        "effective_rank_0p01pct": _effective_rank(0.0001),
        "cumvar_first_20": float(cumvar[19]),
        "cumvar_first_25": float(cumvar[24]),
        "selected_alpha": float(selected_alpha),
        "alpha_grid": list(alpha_grid_floats),
        "alpha_at_grid_boundary": alpha_at_grid_boundary,
    }


def run_bucketed_mode(a: argparse.Namespace) -> None:
    """`--mode bucketed` -- the phase's headline computation. The D5-10 guard --
    `assert_preregistered()` then the three-artifact existence check -- runs first and is
    unchanged from the 05-03 stub. Every constant this function reads comes off `linear_probe`,
    echoed verbatim into every emitted row; no CLI flag can override any of them. ONE global fit
    (`_fit_and_evaluate`, this file's sole `fit_probe` call site) on the training split; the
    held-out test-split residuals are bucketed THREE times, once per seed's frozen
    `BUCKET_EDGES_PER_SEED` entry (`_score_one_seed`), never refit per bucket and never refit
    per seed (D5-02). Per-seed verdicts come only from `linear_probe.apply_verdict_rule`; the
    phase verdict comes only from `linear_probe.combine_seed_verdicts`.

    Under `--smoke`, every provenance assertion below runs against the full 10,000-row
    artifacts first; only the split/fit/bucket/verdict path itself then runs on the first
    `a.smoke_n` rows (for wall-clock speed), and nothing is written to the JSONL record.
    """
    linear_probe.assert_preregistered()
    seed_stems = linear_probe.SEED_STEMS
    bucket_paths = [cache.cache_path(f"{a.bucket_stem}_seed{seed}", "npz") for seed in seed_stems]
    missing = [str(p) for p in bucket_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "--mode bucketed requires all three per-seed bucket artifacts to exist. Missing: "
            f"{missing}. Run --mode perseed first to produce them."
        )

    t0 = time.monotonic()

    TRAIN_FRACTION = linear_probe.TRAIN_FRACTION
    SPLIT_SEED = linear_probe.SPLIT_SEED
    RIDGE_ALPHA_GRID = linear_probe.RIDGE_ALPHA_GRID
    ALPHA_PER_TARGET = linear_probe.ALPHA_PER_TARGET
    FIT_INTERCEPT = linear_probe.FIT_INTERCEPT
    R2_MULTIOUTPUT = linear_probe.R2_MULTIOUTPUT
    N_BUCKETS = linear_probe.N_BUCKETS
    BUCKET_EDGES_PER_SEED = linear_probe.BUCKET_EDGES_PER_SEED
    SEED_HANDLING_RULE = linear_probe.SEED_HANDLING_RULE
    N_BOOTSTRAP = linear_probe.N_BOOTSTRAP
    BOOTSTRAP_SEED = linear_probe.BOOTSTRAP_SEED
    CONFIDENCE_LEVEL = linear_probe.CONFIDENCE_LEVEL
    SIZE_MATCH_N_REPEATS = linear_probe.SIZE_MATCH_N_REPEATS
    SIZE_MATCH_SEED = linear_probe.SIZE_MATCH_SEED
    VERDICT_RULE = linear_probe.VERDICT_RULE
    SEED_VERDICT_COMBINATION_RULE = linear_probe.SEED_VERDICT_COMBINATION_RULE
    PHASE_VERDICT_VALUES = linear_probe.PHASE_VERDICT_VALUES

    X_hsc, X_ls, subsample_file = load_pu_pair()
    resolved_subsample_file = str(Path(subsample_file).resolve())
    n_full = int(X_hsc.shape[0])
    if X_ls.shape[0] != n_full:
        raise ValueError(
            f"run_bucketed_mode: hsc has {n_full} rows but legacysurvey has {X_ls.shape[0]} rows."
        )

    seed_data: Dict[int, Dict[str, Any]] = {}
    for i, seed in enumerate(seed_stems):
        artifact = _load_bucket_artifact(seed, a.bucket_stem)
        manifest_subsample = str(Path(artifact["manifest"]["subsample_file"]).resolve())
        if manifest_subsample != resolved_subsample_file:
            raise ValueError(
                f"run_bucketed_mode: seed {seed}'s bucket artifact carries "
                f"subsample_file={manifest_subsample!r}, which does not match the resolved "
                f"subsample {resolved_subsample_file!r}."
            )
        frozen_edges = tuple(float(v) for v in BUCKET_EDGES_PER_SEED[i])
        if artifact["bucket_edges"] != frozen_edges:
            raise ValueError(
                f"run_bucketed_mode: seed {seed}'s artifact bucket_edges="
                f"{artifact['bucket_edges']!r} does not equal the frozen "
                f"BUCKET_EDGES_PER_SEED[{i}]={frozen_edges!r}."
            )
        if artifact["H_norm"].shape[0] != n_full or artifact["bucket_labels"].shape[0] != n_full:
            raise ValueError(
                f"run_bucketed_mode: seed {seed}'s artifact carries {artifact['H_norm'].shape[0]} "
                f"rows, expected {n_full} (matching hsc/legacysurvey)."
            )
        seed_data[seed] = artifact

    if n_full != 10000:
        raise ValueError(f"run_bucketed_mode: expected 10,000 rows everywhere, got {n_full}.")

    if a.smoke:
        n_use = min(a.smoke_n, n_full)
        print(
            f"SMOKE: reduced to n={n_use} of {n_full} rows for the split/fit/bucket/verdict "
            "path only -- every provenance assertion above already ran against the full "
            "10,000-row artifacts. Writes nothing.\n"
        )
        X_hsc_use = X_hsc[:n_use]
        X_ls_use = X_ls[:n_use]
        seed_data_use = {
            seed: {
                "H_norm": seed_data[seed]["H_norm"][:n_use],
                "bucket_labels": seed_data[seed]["bucket_labels"][:n_use],
            }
            for seed in seed_stems
        }
    else:
        n_use = n_full
        X_hsc_use = X_hsc
        X_ls_use = X_ls
        seed_data_use = {
            seed: {
                "H_norm": seed_data[seed]["H_norm"],
                "bucket_labels": seed_data[seed]["bucket_labels"],
            }
            for seed in seed_stems
        }

    train_idx, test_idx = linear_probe.train_test_split_indices(n_use, TRAIN_FRACTION, SPLIT_SEED)
    n_train, n_test = int(train_idx.shape[0]), int(test_idx.shape[0])

    fit, Y_pred_test, residuals_test, r2_overall = _fit_and_evaluate(
        X_hsc_use[train_idx],
        X_ls_use[train_idx],
        X_hsc_use[test_idx],
        X_ls_use[test_idx],
        RIDGE_ALPHA_GRID,
        ALPHA_PER_TARGET,
        FIT_INTERCEPT,
        R2_MULTIOUTPUT,
    )
    Y_true_test = X_ls_use[test_idx]
    selected_alpha = float(fit["alpha_"])
    mean_residual_overall = float(residuals_test.mean())

    print("=" * 78)
    print(
        f"Bucketed probe (D5-01/D5-02) -- n_train={n_train} n_test={n_test} "
        f"selected_alpha={selected_alpha:g} r2_overall={r2_overall:.6g}"
    )
    print("=" * 78)

    per_seed_results: Dict[int, Dict[str, Any]] = {}
    verdicts: Dict[int, str] = {}
    for seed in seed_stems:
        result = _score_one_seed(
            seed,
            seed_data_use[seed]["H_norm"],
            seed_data_use[seed]["bucket_labels"],
            test_idx,
            residuals_test,
            Y_true_test,
            Y_pred_test,
            N_BUCKETS,
            N_BOOTSTRAP,
            BOOTSTRAP_SEED,
            CONFIDENCE_LEVEL,
            SIZE_MATCH_N_REPEATS,
            SIZE_MATCH_SEED,
            VERDICT_RULE,
            R2_MULTIOUTPUT,
        )
        per_seed_results[seed] = result
        verdicts[seed] = result["verdict"]
        print(
            f"seed {seed}: realized test-split bucket counts={result['realized_bucket_counts']}  "
            f"size_match_n={result['size_match']['n_match']}  "
            f"full_field_bucket_counts={result['full_field_bucket_counts']}"
        )

    for seed in seed_stems:
        print(f"seed {seed} verdict: {verdicts[seed]}")

    combo = linear_probe.combine_seed_verdicts(verdicts, SEED_VERDICT_COMBINATION_RULE)
    phase_verdict = combo["phase_verdict"]
    if phase_verdict not in PHASE_VERDICT_VALUES:
        raise RuntimeError(
            f"run_bucketed_mode: combine_seed_verdicts produced {phase_verdict!r}, not a member "
            f"of PHASE_VERDICT_VALUES={PHASE_VERDICT_VALUES!r}."
        )
    print(f"phase verdict: {phase_verdict}  (n_holds={combo['n_holds']} of 3)")
    print("=" * 78)

    if a.smoke:
        return

    wallclock_s = time.monotonic() - t0
    record_path = cache.cache_path("05_curvature_probe_decodability", "jsonl")
    record_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "data_source": "pu",
        "subsample_file": resolved_subsample_file,
        "curvature_convention": linear_probe.CURVATURE_CONVENTION,
        "curvature_source_function": linear_probe.CURVATURE_SOURCE_FUNCTION,
        "field_stem": a.field_stem,
        "bucket_stem": a.bucket_stem,
    }

    with record_path.open("a") as fh:
        for seed in seed_stems:
            r = per_seed_results[seed]
            for stat in r["bucket_stats"]:
                row = {
                    "kind": "probe_bucket",
                    "seed_stem": int(seed),
                    "bucket_index": int(stat["bucket_index"]),
                    "bucket_n": int(stat["n"]),
                    "full_field_bucket_n": int(r["full_field_bucket_counts"][stat["bucket_index"]]),
                    "bucket_mean_residual": float(stat["score"]),
                    "bucket_r2": float(stat["r2"]),
                    "ci_low": float(stat["ci_low"]),
                    "ci_high": float(stat["ci_high"]),
                    "degenerate": bool(stat["degenerate"]),
                    "confidence_level": float(stat["confidence_level"]),
                    "n_resamples": int(stat["n_resamples"]),
                    **provenance,
                }
                fh.write(json.dumps(row, default=float) + "\n")
                fh.flush()

        for seed in seed_stems:
            r = per_seed_results[seed]
            row = {
                "kind": "probe_seed",
                "seed_stem": int(seed),
                "bucket_edges": list(seed_data[seed]["bucket_edges"]),
                "realized_bucket_counts": list(r["realized_bucket_counts"]),
                "full_field_bucket_counts": list(r["full_field_bucket_counts"]),
                "verdict": r["verdict"],
                "verdict_criteria": r["criteria"],
                "spearman_h_residual_rho": r["spearman"]["rho"],
                "spearman_h_residual_p": r["spearman"]["p_value"],
                "spearman_h_residual_n": r["spearman"]["n"],
                "spearman_h_residual_undefined": r["spearman"]["undefined"],
                "spearman_direction_axis": None,
                "spearman_direction_axis_reason": (
                    "both operands are scalar fields -- curvature magnitude and per-point "
                    "residual -- so there is no pair of vectors to take a cosine between and no "
                    "vector direction axis exists; the sign of rho is the direction."
                ),
                "size_match_n": int(r["size_match"]["n_match"]),
                "size_match_median_diff": float(r["size_match"]["median_diff"]),
                "size_match_sign_stable": bool(r["size_match"]["sign_stable"]),
                "size_match_ci_disjoint_fraction": float(r["size_match"]["ci_disjoint_fraction"]),
                "size_match_n_repeats": int(r["size_match"]["n_repeats"]),
                "selected_alpha": selected_alpha,
                "r2_overall": r2_overall,
                **provenance,
            }
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()

        overall_row = {
            "kind": "probe_overall",
            "n": n_use,
            "n_train": n_train,
            "n_test": n_test,
            "train_fraction": TRAIN_FRACTION,
            "split_seed": SPLIT_SEED,
            "selected_alpha": selected_alpha,
            "alpha_grid": list(RIDGE_ALPHA_GRID),
            "r2_overall": r2_overall,
            "mean_residual_overall": mean_residual_overall,
            "n_buckets": N_BUCKETS,
            "seed_stems": list(seed_stems),
            "seed_handling_rule": SEED_HANDLING_RULE,
            "bucket_edges_per_seed": [list(e) for e in BUCKET_EDGES_PER_SEED],
            "per_seed_verdicts": {str(s): verdicts[s] for s in seed_stems},
            "n_holds": combo["n_holds"],
            "phase_verdict": phase_verdict,
            "phase_verdict_values": list(PHASE_VERDICT_VALUES),
            "seed_verdict_combination_rule": SEED_VERDICT_COMBINATION_RULE,
            "wallclock_s": wallclock_s,
            **provenance,
        }
        fh.write(json.dumps(overall_row, default=float) + "\n")
        fh.flush()

        conditioning_row = _conditioning_diagnostics(
            X_hsc_use[train_idx], RIDGE_ALPHA_GRID, selected_alpha
        )
        fh.write(json.dumps(conditioning_row, default=float) + "\n")
        fh.flush()

    print(f"wrote {record_path}")


def _effective_distinct_levels(values: np.ndarray, tolerances: Tuple[float, ...]) -> Tuple[int, ...]:
    """Count distinct levels in `values` at each RELATIVE tolerance in `tolerances`. Sorts
    `values` once, then for each tolerance walks the sorted array once, opening a new level
    whenever the gap to the current level's representative value exceeds
    `tolerance * abs(representative)`. This is the measurement that corrects
    `05-02-SUMMARY.md`: that summary's exact-float64 counts (5,301 / 9,852) and its
    `np.round(H_norm, 6)` counts are both ABSOLUTE, and at magnitude ~5e4 an absolute
    six-decimal rounding is a relative precision of ~2e-11 -- fine enough to count last-ULP
    float noise as structure. Returns one count per tolerance, in the same order as
    `tolerances`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"_effective_distinct_levels: values must be one-dimensional, got shape {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError("_effective_distinct_levels: values must be non-empty.")
    sorted_values = np.sort(values)
    counts = []
    for tol in tolerances:
        n_levels = 1
        representative = sorted_values[0]
        for v in sorted_values[1:]:
            if abs(v - representative) > tol * abs(representative):
                n_levels += 1
                representative = v
        counts.append(n_levels)
    return tuple(counts)


def run_perseed_mode(a: argparse.Namespace) -> None:
    """`--mode perseed`: three INDEPENDENT bucketings, one per seed in `CANONICAL_SEED_STEMS`
    order, each cut over THAT seed's own 10,000-point field -- never over a test split, never
    mixed across seeds. This is the ratified replacement for `--mode pool`
    (`05-03-DECISION.md`): `linear_probe.pool_seed_fields` is never called and no pooled
    artifact is ever written by this mode.
    """
    _, _, subsample_file = load_pu_pair()

    print("=" * 78)
    print(f"Per-seed bucketing (D5-07/D5-09) -- seeds={CANONICAL_SEED_STEMS}, n_buckets=3")
    print("no pooled field is built here; three independent per-seed bucketings follow")
    print("=" * 78)

    tolerances: Tuple[float, ...] = (1e-9, 1e-6, 1e-3)
    n_buckets = 3
    levels_by_seed: Dict[int, Tuple[int, ...]] = {}

    for seed in CANONICAL_SEED_STEMS:
        loaded = _load_cached_seed_field(seed, a.field_stem)
        if loaded is None:
            missing_stem = f"{a.field_stem}_seed{seed}"
            raise FileNotFoundError(
                f"--mode perseed requires seed {seed}'s cached field artifact at stem "
                f"{missing_stem!r}, which does not exist. This mode never recomputes a "
                "field -- run --mode field first."
            )
        H_norm = loaded["H_norm"]
        labels, edges = linear_probe.bucket_by_field(H_norm, n_buckets)
        levels = _effective_distinct_levels(H_norm, tolerances)
        levels_by_seed[seed] = levels
        counts_info = linear_probe.bucket_counts(labels, n_buckets)

        print(f"seed {seed}: effective distinct levels at rel {tolerances} = {levels}")
        print(f"seed {seed}: bucket edges (full float64 repr) = ({edges[0]!r}, {edges[1]!r})")
        print(
            f"seed {seed}: full-field bucket counts = {list(counts_info['counts'])}  "
            f"median={float(np.median(H_norm)):.6g}  min={float(np.min(H_norm)):.6g}  "
            f"max={float(np.max(H_norm)):.6g}"
        )

        cfg: Dict[str, Any] = {
            "kind": "05_curvature_buckets_perseed",
            "seed": int(seed),
            "source_field_stem": f"{a.field_stem}_seed{seed}",
            "n_buckets": int(n_buckets),
            "bucket_rule": "equal_frequency_rank_partition_stable_argsort",
            "subsample_file": str(subsample_file),
            "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
            "seed_handling_rule": "no_pooling_per_seed_verdicts",
        }

        def _compute() -> Dict[str, np.ndarray]:
            return {
                "H_norm": H_norm,
                "bucket_labels": labels.astype(np.int64),
                "bucket_edges": np.asarray(edges, dtype=np.float64),
                "seed_stem": np.asarray(seed, dtype=np.int64),
                "n_buckets": np.asarray(n_buckets, dtype=np.int64),
                "effective_distinct_levels": np.asarray(levels, dtype=np.int64),
                "effective_level_tolerances": np.asarray(tolerances, dtype=np.float64),
                "n_charts_used": np.asarray(loaded["n_charts_used"]),
            }

        stem = f"{a.bucket_stem}_seed{seed}"
        cache.npz_cache(stem, cfg, _compute)
        print(f"wrote {cache.cache_path(stem, 'npz')}")

    print("=" * 78)
    print(
        "no pooled field was built; the three seeds' bucket labels are independent; the "
        "phase read-out is three per-seed verdicts and their spread (05-03-DECISION.md)"
    )
    expected_levels = {20260814: (4, 4, 4), 20260815: (3, 3, 3)}
    for seed, expected in expected_levels.items():
        measured = levels_by_seed.get(seed)
        if measured == expected:
            print(f"seed {seed}: measured effective levels {measured} match the expected {expected}")
        else:
            print(
                f"seed {seed}: measured effective levels {measured} DO NOT match the expected "
                f"{expected} -- this changes what 05-04 freezes"
            )
    print("=" * 78)

    run_density_diagnostics(a)


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


DENSITY_K_DENSITY = 30
"""Phase 4's pre-registered `region_partition.K_DENSITY` (D4-15), reused unchanged so Phase 4
and Phase 5's density numbers sit on the same footing."""

DENSITY_FIELD_D = 20
"""Phase 4's pre-registered `region_partition.FIELD_D` (D-07), reused unchanged for the same
reason -- the intrinsic-dimension parameter `local_density_weights`' k-NN density formula
takes, not the ambient ``X``'s own column count."""


def run_density_diagnostics(a: argparse.Namespace) -> None:
    """D5-13: re-measure ``spearman(density, ||H||)`` PER SEED on the decoder-side fields,
    with Phase 4's own point-cloud (-0.0273) and direction (+0.8208) reference numbers quoted
    beside them, and dispose of D5-05's pooled-versus-seed half (which has no referent under
    ``05-03-DECISION.md``'s ratified refusal to pool) rather than silently dropping it. Writes
    ``notebooks/.cache/05_density_diagnostics.json`` through ``cache.json_cache``. Mirrors
    :func:`run_inter_seed_diagnostics`'s own partial-invocation guard: runs only once all
    three canonical seeds' field artifacts already exist, otherwise prints a one-line notice
    and returns without writing anything.
    """
    fields: Dict[int, Dict[str, np.ndarray]] = {}
    for seed in CANONICAL_SEED_STEMS:
        loaded = _load_cached_seed_field(seed, a.field_stem)
        if loaded is None:
            missing_path = cache.cache_path(f"{a.field_stem}_seed{seed}", "npz")
            print(
                f"Density diagnostics skipped: seed {seed}'s field artifact does not yet "
                f"exist at {missing_path}."
            )
            return
        fields[seed] = loaded

    print("=" * 78)
    print("Density confound diagnostics (D5-13): per seed, Phase 4's own estimator/constants")
    print("=" * 78)

    _, X_ls, _ = load_pu_pair()
    X64 = X_ls.astype(np.float64)

    def _compute() -> Dict[str, Any]:
        # REGN-01 convention, unchanged: `weight` is the INVERSE local density, mean-normalized
        # to 1; `reciprocal` (= 1 / weight) is the RELATIVE density itself.
        weight = curvature_probe.local_density_weights(
            X64, k_density=DENSITY_K_DENSITY, d=DENSITY_FIELD_D
        )
        reciprocal = 1.0 / weight

        spearman_density_per_seed_h: Dict[str, Any] = {}
        spearman_inverse_of_weight_per_seed_h: Dict[str, Any] = {}
        for seed in CANONICAL_SEED_STEMS:
            H_norm = fields[seed]["H_norm"]
            spearman_density_per_seed_h[str(seed)] = _spearman_report(
                weight, H_norm, f"spearman(inverse_density_weight, ||H||) seed={seed}"
            )
            spearman_inverse_of_weight_per_seed_h[str(seed)] = _spearman_report(
                reciprocal, H_norm, f"spearman(relative_density, ||H||) seed={seed}"
            )

        return {
            "spearman_density_per_seed_h": spearman_density_per_seed_h,
            "spearman_inverse_of_weight_per_seed_h": spearman_inverse_of_weight_per_seed_h,
            "k_density": DENSITY_K_DENSITY,
            "field_d": DENSITY_FIELD_D,
            "density_definition": (
                "curvature_probe.local_density_weights returns the per-point INVERSE local "
                "density, normalized to mean 1 (REGN-01's own convention). "
                "spearman_density_per_seed_h correlates THAT quantity against each seed's "
                "||H||; spearman_inverse_of_weight_per_seed_h correlates its reciprocal "
                "(the RELATIVE density itself) against the same field -- the two keys have "
                "opposite sign by construction, and both are recorded so the sign of a "
                "density correlation is never ambiguous."
            ),
            "phase4_pointcloud_reference": (
                "-0.0273 (n=9500, p=0.0078) -- 04-FINDINGS.md's "
                "spearman(density, centroid_mean_curvature), measured on the POINT-CLOUD "
                "field. A different curvature estimator (centroid_mean_curvature, "
                "point-cloud-side) than these decoder-side chart_curvature_field values; "
                "does not transfer."
            ),
            "phase4_direction_reference": (
                "+0.8208 (n=9500, p≈0) -- 04-FINDINGS.md's "
                "spearman(density, signed_projection onto v). This attached to curvature "
                "DIRECTION (the sign of the projection onto the frozen split axis v), which "
                "is the axis Phase 5 is not splitting on -- Phase 5 buckets by ||H|| "
                "magnitude only."
            ),
            "direction_axis": None,
            "direction_axis_reason": (
                "both operands are scalar fields -- a density weight and a curvature "
                "magnitude -- so there is no pair of vectors to take a cosine between and no "
                "vector direction axis exists; the sign of rho is the direction."
            ),
            "seed_stems": list(CANONICAL_SEED_STEMS),
            "pooled_field_disposition": (
                "05-CONTEXT.md D5-05 asks for the Spearman between each seed and the pooled "
                "field. No pooled field exists: seed pooling was put to the developer at the "
                "05-03 Task 1 blocking checkpoint and ratified as NOT DONE in "
                "05-03-DECISION.md, superseding D5-04. The statistic therefore has no "
                "referent and was NOT computed against a substitute. D5-05's first half -- "
                "the pairwise inter-seed Spearman with its direction axis -- was measured at "
                "05-02 and is recorded in notebooks/.cache/05_inter_seed_diagnostics.json."
            ),
        }

    cfg = {
        "kind": "05_density_diagnostics",
        "field_stem_prefix": a.field_stem,
        "seed_stems": list(CANONICAL_SEED_STEMS),
        "k_density": DENSITY_K_DENSITY,
        "field_d": DENSITY_FIELD_D,
    }
    result = cache.json_cache("05_density_diagnostics", cfg, _compute)
    measured = {s: v["rho"] for s, v in result["spearman_density_per_seed_h"].items()}
    print(f"measured spearman(inverse_density_weight, ||H||) per seed: {measured}")
    print(f"wrote {cache.cache_path('05_density_diagnostics', 'json')}")


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
    fit, Y_pred_test, residuals_test, r2 = _fit_and_evaluate(
        X[train_idx], Y[train_idx], X[test_idx], Y[test_idx],
        alpha_grid, False, True, "variance_weighted",
    )
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
    p.add_argument("--mode", choices=["field", "pool", "perseed", "bucketed"], default="field")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260813, 20260814, 20260815])
    p.add_argument("--n-charts", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--field-stem", type=str, default="05_curvature_field")
    p.add_argument("--bucket-stem", type=str, default="05_curvature_buckets")
    p.add_argument(
        "--pooling-method",
        type=str,
        default=None,
        help=(
            "Retained only as a tripwire: supplying this flag under ANY mode raises the same "
            "RuntimeError --mode pool raises, naming 05-03-DECISION.md. It exists so passing "
            "it fails loudly instead of being silently ignored."
        ),
    )
    return p


def main() -> None:
    a = build_arg_parser().parse_args()

    if a.pooling_method is not None:
        raise RuntimeError(POOLING_REFUSAL_MESSAGE)

    if a.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    if a.mode == "field":
        run_field_mode(a)
        return

    if a.mode == "pool":
        run_pool_mode(a)
        return

    if a.mode == "perseed":
        run_perseed_mode(a)
        return

    if a.mode == "bucketed":
        run_bucketed_mode(a)
        return

    raise NotImplementedError(f"--mode {a.mode!r} is not a recognized mode.")


if __name__ == "__main__":
    main()
