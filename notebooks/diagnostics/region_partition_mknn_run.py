"""Phase 4 region-partitioning MKNN runner. `--mode global` (this plan) computes the
region-blind, partition-blind crossmodal HSC-vs-Legacy-Survey MKNN across the frozen
`--mknn-k` grid. `--mode partition`/`--mode regional` are pre-registered by later plans
in this phase (04-03 onward) and are not implemented here.

    python notebooks/diagnostics/region_partition_mknn_run.py --selfcheck
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global --smoke
    python notebooks/diagnostics/region_partition_mknn_run.py --mode global
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

from pu_manifold import mknn

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
    print("\nGlobal crossmodal MKNN read-out (raw numbers, not yet framed against the")
    print("origin paper -- see notebooks/04_region_partition_mknn.ipynb for that read-out).")


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
