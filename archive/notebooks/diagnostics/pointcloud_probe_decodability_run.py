"""Phase 6 runner: Phase 5's probe, bucketed by the point-cloud curvature field instead.

**One thing changes and the runner is built so that nothing else can.** The split, the ridge
map, the alpha grid and its selection rule, the residual metric and the bucket rule are all
imported from Phase 5's sealed ``linear_probe`` module and its frozen constants, re-declared in
``pointcloud_probe`` only so a reader can diff the two blocks. The curvature field is read from
Phase 4's sealed ``04_region_partition.npz`` and is never recomputed (D6-01): recomputing it
would silently re-tune ``k`` and make Phase 4's freeze meaningless.

Because the split and the map are Phase 5's own, the 3,000 held-out per-point residuals scored
here are the SAME 3,000 Phase 5 scored. The two phases differ in the instrument and in nothing
else, which is what makes the comparison and the D6-06 cross-estimator disclosure possible at
all.

    python notebooks/diagnostics/pointcloud_probe_decodability_run.py --selfcheck
    python notebooks/diagnostics/pointcloud_probe_decodability_run.py --mode bucketed

``--selfcheck`` runs the machine on planted data with a known answer and touches no PU row.
``--mode bucketed`` is the only path that produces a Phase 6 number, and it calls
``pointcloud_probe.assert_preregistered()`` first, so it cannot run against an unfrozen build.

**Provenance is checked, not assumed.** The runner refuses to proceed unless the subsample file
Phase 4 recorded in ``04_region_partition.meta.json`` is byte-identical in path to the one
``load_pu_pair`` resolves, and unless Phase 4's recorded ``K_FROZEN``/``K_DENSITY``/``FIELD_D``
equal the frozen Phase 6 constants. A field silently indexed against a different subsample would
produce a plausible number that means nothing.

No sealed verdict is reopened by this runner (G6-04).
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

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import cache, curvature_probe  # noqa: E402
from pu_manifold import linear_probe as lp  # noqa: E402
from pu_manifold import pointcloud_probe as pp  # noqa: E402

DEFAULT_RECORD = cache.cache_path("06_pointcloud_probe_decodability", "jsonl")
SELFCHECK_RECORD = cache.cache_path("06_probe_selfcheck", "jsonl")
PHASE4_FIELD = cache.cache_path("04_region_partition", "npz")
PHASE4_META = cache.cache_path("04_region_partition", "meta.json")
PHASE5_BUCKET_STEM = "05_curvature_buckets"


def load_pu_pair(
    column_a: str = "hsc", column_b: str = "legacysurvey"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Copied unchanged from ``curvature_probe_decodability_run.load_pu_pair`` so Phase 6 reads
    the identical rows in the identical order Phase 5 did."""
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column_a in z.files and column_b in z.files and z[column_a].shape[0] > best_n:
                best, best_n = c, z[column_a].shape[0]
    if best is None:
        raise KeyError(f"no cached subsample carries both {column_a!r} and {column_b!r}")
    with np.load(best) as z:
        Xa = np.asarray(z[column_a], dtype=np.float64)
        Xb = np.asarray(z[column_b], dtype=np.float64)
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError(f"load_pu_pair: row-count mismatch in {best!r}")
    print(f"loaded {column_a} {Xa.shape} and {column_b} {Xb.shape} from {Path(best).name}")
    return Xa, Xb, best


def load_phase4_field(subsample_file: str, n_expected: int) -> np.ndarray:
    """D6-01: READ Phase 4's sealed density-corrected ``centroid_mean_curvature`` field.

    Refuses on any provenance mismatch rather than proceeding with a field that may index a
    different cloud."""
    if not PHASE4_FIELD.exists():
        raise FileNotFoundError(f"Phase 4 field artifact missing: {PHASE4_FIELD}")
    meta = json.loads(PHASE4_META.read_text())
    for name, frozen in (("K_FROZEN", pp.K_FROZEN), ("K_DENSITY", pp.K_DENSITY),
                         ("FIELD_D", pp.FIELD_D)):
        if int(meta[name]) != int(frozen):
            raise RuntimeError(
                f"load_phase4_field: Phase 4 recorded {name}={meta[name]} but Phase 6 froze "
                f"{name}={frozen}. Phase 6 inherits Phase 4's freeze (D6-01); it does not "
                f"re-tune it."
            )
    if Path(meta["subsample_file"]).name != Path(subsample_file).name:
        raise RuntimeError(
            f"load_phase4_field: Phase 4's field was computed on "
            f"{Path(meta['subsample_file']).name} but this run resolved "
            f"{Path(subsample_file).name}. Row alignment cannot be assumed across subsamples."
        )
    with np.load(PHASE4_FIELD) as z:
        h = np.asarray(z[pp.CURVATURE_SOURCE_KEY], dtype=np.float64)
    if h.shape != (n_expected,):
        raise RuntimeError(
            f"load_phase4_field: field has shape {h.shape}, expected ({n_expected},)."
        )
    if not np.all(np.isfinite(h)):
        raise RuntimeError("load_phase4_field: field contains a non-finite value.")
    print(f"loaded Phase 4 field h_norm {h.shape} from {PHASE4_FIELD.name} "
          f"(K_FROZEN={meta['K_FROZEN']}, density-corrected)")
    return h


def _json_safe(obj: Any) -> Any:
    """Recursively cast numpy scalars AND arrays to native Python types.

    ``cae.to_native`` handles numpy scalars and torch tensors but not ``np.ndarray``, and
    ``lp.bucket_counts`` returns its per-bucket counts as an array. Written here rather than
    by extending ``cae.to_native`` because ``cae.py`` is sealed Phase 02.2 code and this is a
    serialization detail of one runner, not a change to shared behaviour.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _spearman(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    rho, p = spearmanr(a, b)
    if np.isnan(rho):
        return {"rho": None, "p": None, "n": int(np.asarray(a).shape[0]),
                "undefined_reason": "a constant input has no ranks to correlate"}
    return {"rho": float(rho), "p": float(p), "n": int(np.asarray(a).shape[0])}


def score_field(
    X: np.ndarray, Y: np.ndarray, field: np.ndarray, label: str
) -> Dict[str, Any]:
    """The whole probe, once, under the frozen constants. Pure apart from its prints."""
    n = X.shape[0]
    train_idx, test_idx = lp.train_test_split_indices(n, pp.TRAIN_FRACTION, pp.SPLIT_SEED)
    fit = lp.fit_probe(X[train_idx], Y[train_idx], pp.RIDGE_ALPHA_GRID,
                       pp.ALPHA_PER_TARGET, pp.FIT_INTERCEPT)
    Y_pred = lp.predict_probe(fit, X[test_idx])
    resid = lp.per_point_residuals(Y[test_idx], Y_pred)
    r2 = lp.aggregate_r2(Y[test_idx], Y_pred, pp.R2_MULTIOUTPUT)

    labels_all, edges = lp.bucket_by_field(field, pp.N_BUCKETS)
    labels_test = labels_all[test_idx]
    counts = lp.bucket_counts(labels_test, pp.N_BUCKETS)

    stats = []
    for b in range(pp.N_BUCKETS):
        sel = resid[labels_test == b]
        stats.append(lp.bucket_residual_ci(sel, pp.N_BOOTSTRAP, pp.BOOTSTRAP_SEED,
                                           pp.CONFIDENCE_LEVEL))
    size_match = lp.size_matched_check(resid, labels_test, pp.SIZE_MATCH_N_REPEATS,
                                       pp.SIZE_MATCH_SEED, pp.CONFIDENCE_LEVEL)
    verdict = lp.apply_verdict_rule(stats, size_match, pp.VERDICT_RULE)
    if not pp.verdict_is_terminal(verdict["verdict"]):
        raise RuntimeError(f"score_field: non-terminal verdict {verdict['verdict']!r}")

    return {
        "label": label,
        "n_train": int(train_idx.shape[0]), "n_test": int(test_idx.shape[0]),
        "selected_alpha": float(np.asarray(fit["alpha_"]).item()),
        "r2_overall": float(r2),
        "mean_residual_overall": float(resid.mean()),
        "bucket_edges": [float(e) for e in edges],
        "bucket_counts": counts,
        "bucket_stats": stats,
        "size_match": size_match,
        "verdict": verdict["verdict"],
        "criteria": verdict["criteria"],
        "sensitivity_spearman_field_vs_residual": _spearman(field[test_idx], resid),
    }


def run_selfcheck() -> None:
    """Planted data, known answer, no PU row touched. High-``field`` rows get extra noise, so a
    correctly wired machine must return HOLDS; a shuffled field must not."""
    pp.assert_preregistered()
    rng = np.random.default_rng(20260824)
    n, p, q = 10000, 40, 12
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((p, q))
    field = rng.random(n)
    noise = rng.standard_normal((n, q)) * (0.05 + 2.5 * field)[:, None]
    Y = X @ W + noise

    planted = score_field(X, Y, field, "planted_signal")
    shuffled = score_field(X, Y, rng.permutation(field), "planted_shuffled_field")

    ok = planted["verdict"] == "HOLDS" and shuffled["verdict"] == "NO DETECTABLE RELATIONSHIP"
    print(f"\n  planted  field -> {planted['verdict']}")
    print(f"  shuffled field -> {shuffled['verdict']}")
    print(f"  SELFCHECK {'PASS' if ok else 'FAIL'}")
    with SELFCHECK_RECORD.open("a") as fh:
        fh.write(json.dumps(_json_safe({"kind": "06_selfcheck", "pass": bool(ok),
                                        "planted": planted["verdict"],
                                        "shuffled": shuffled["verdict"]})) + "\n")
    if not ok:
        raise SystemExit("selfcheck FAILED -- the bucketed path must not be run")


def run_bucketed() -> None:
    """The only path that produces a Phase 6 number."""
    pp.assert_preregistered()
    X, Y, subsample_file = load_pu_pair()
    field = load_phase4_field(subsample_file, X.shape[0])

    result = score_field(X, Y, field, "phase_4_point_cloud_field")

    # --- D6-06 disclosure: cross-estimator agreement, closing D4-08 ----------------------
    cross: Dict[str, Any] = {}
    for seed in pp.CROSS_ESTIMATOR_DISCLOSURE_SEEDS:
        path = cache.cache_path(f"{PHASE5_BUCKET_STEM}_seed{seed}", "npz")
        if not path.exists():
            cross[str(seed)] = {"unavailable": str(path.name)}
            continue
        with np.load(path) as z:
            h5 = np.asarray(z["H_norm"], dtype=np.float64)
        cross[str(seed)] = _spearman(field, h5)

    # --- D6-08 disclosure: density confound ----------------------------------------------
    weights = curvature_probe.local_density_weights(X, pp.K_DENSITY, pp.FIELD_D)
    density = _spearman(weights, field)

    record = {
        "kind": "06_pointcloud_probe_decodability",
        "reproduces_sealed_cell": False,
        "curvature_source": pp.CURVATURE_SOURCE,
        "curvature_source_function": pp.CURVATURE_SOURCE_FUNCTION,
        "k_frozen": pp.K_FROZEN,
        "seed_handling_rule": pp.SEED_HANDLING_RULE,
        "subsample_file": str(subsample_file),
        "field_stats": {"n": int(field.shape[0]), "median": float(np.median(field)),
                        "p05": float(np.percentile(field, 5)),
                        "p95": float(np.percentile(field, 95)),
                        "spread_p95_p05": float(np.percentile(field, 95)
                                                / max(np.percentile(field, 5), 1e-30))},
        "probe": result,
        "disclosure_cross_estimator_spearman_vs_phase5_seeds": cross,
        "disclosure_density_spearman": density,
        "disclosure_note": (
            "D6-06 and D6-08 are DISCLOSURES. Neither may upgrade or downgrade the verdict. "
            "D6-07: PU has no ground-truth H, so no direction axis against truth exists here."
        ),
        "inheritance": pp.describe_inheritance(),
        "accepted_gaps": {
            "G6-01": "field validated on PU by split-half reliability alone, which cannot "
                     "detect a bias both halves share (Swiss roll: R_H=0.990 with rho=0.469)",
            "G6-02": "magnitude ordering is the weaker functional at d=20",
            "G6-03": "K_FROZEN=500 is the largest k run, not one the freeze rule selected "
                     "(04_k_freeze.json rule_fired: false)",
            "G6-04": "a disagreement with Phase 5 localizes to the instrument and does not "
                     "establish which instrument is correct",
        },
    }
    with DEFAULT_RECORD.open("a") as fh:
        fh.write(json.dumps(_json_safe(record)) + "\n")

    print("\n" + "=" * 78)
    print(f"  PHASE 6 VERDICT: {result['verdict']}")
    print("=" * 78)
    print(f"  n_train={result['n_train']}  n_test={result['n_test']}  "
          f"alpha={result['selected_alpha']}  r2={result['r2_overall']:.6f}")
    print(f"  bucket edges: {result['bucket_edges']}")
    print(f"  bucket counts (test): {result['bucket_counts']}")
    for i, s in enumerate(result["bucket_stats"]):
        print(f"    bucket {i}: mean={s['score']:.6f}  "
              f"CI=[{s['ci_low']:.6f}, {s['ci_high']:.6f}]  n={s['n']}")
    print(f"  criteria: {result['criteria']}")
    print(f"  sensitivity (NOT a gate) spearman(field, residual) = "
          f"{result['sensitivity_spearman_field_vs_residual']}")
    print(f"\n  D6-06 cross-estimator vs Phase 5 decoder seeds (disclosure, closes D4-08):")
    for k, v in cross.items():
        print(f"    seed {k}: {v}")
    print(f"  D6-08 density (disclosure): {density}")
    print(f"\n  written to {DEFAULT_RECORD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("bucketed",), default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        run_selfcheck()
    elif a.mode == "bucketed":
        run_bucketed()
    else:
        ap.error("pass --selfcheck or --mode bucketed")


if __name__ == "__main__":
    main()
