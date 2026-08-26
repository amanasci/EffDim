"""Phase 7 crossmodal curvature runner. `--mode smoke` (07-02) proves the pipeline end to end;
`--mode dsweep` (07-04) and `--mode positive-control` (07-03) are declared CLI choices only,
raising NotImplementedError until those plans land. Distinct from the nine pre-existing
`07_*_run.py` spike scripts, which stay untouched and satisfy no pre-registration. Usage:
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --selfcheck
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode smoke
"""

import os
import sys

# Thread cap MUST be set before any import pulling in torch/numpy -- NEW engineering here (no
# prior notebooks/ runner does this): 3 concurrent torch jobs measured load 44, ~10x slowdown.
_THREADS = 8
if "--threads" in sys.argv and sys.argv.index("--threads") + 1 < len(sys.argv):
    try:
        _THREADS = int(sys.argv[sys.argv.index("--threads") + 1])
    except ValueError:
        pass
os.environ["OMP_NUM_THREADS"] = str(_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_THREADS)

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import glob  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_THREADS)

from pu_manifold import cache  # noqa: E402
from pu_manifold import cae  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import mknn  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402

# The freeze commit (the commit that added crossmodal_curvature.py, plan 07-01, Task 2) --
# every PU number this runner produces must be a strict git descendant of it
# (PREREGISTRATION_FREEZE_RULE). Recorded on every non-smoke record row as
# ``preregistration_commit``.
FREEZE_COMMIT_SHA = "f032745f6450068c63763993d39fa112fd36bb8c"


def _git_rev_parse(rev: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def load_pu_pair(
    column_a: str = "hsc", column_b: str = "legacysurvey"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path. Copied
    verbatim from `region_partition_mknn_run.py` lines 42-70 (Task 1's `<read_first>`):
    keeps only files carrying both columns, selects the one with the most rows; on a tie
    keeps the lexicographically first path."""
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


def fit_and_field(
    X: np.ndarray, d: int, max_epochs: int, n_rows: int, smoke: bool = False
) -> Dict[str, Any]:
    """The fit-to-field path: build `cae.PlainAutoEncoder(latent_dim=d)`, split with
    `crossmodal_curvature.split_indices` at the frozen `SPLIT_SEED`/`HOLDOUT_FRACTION`,
    train with `cae.train_plain_ae` on the float32 train rows using `TRAIN_CFG` plus
    `max_epochs`, then `model.eval().double()`, encode the FULL float64 array, and call
    `decoder_curvature.plain_decoder_curvature(model, z)` passing the MODEL -- not a bound
    method -- so its `_assert_float64` guard actually runs its per-parameter check (WR-01's
    defect shape is exactly a bound method reaching that guard and silently skipping it).

    Refuses a `d` outside `D_SWEEP` unless `smoke=True`. Returns the per-point
    `np.linalg.norm(H_vec, axis=1)` array, the `metric_condition_number` array, and the
    holdout reconstruction variance-explained (via `cae.reconstruction_stats`, matching
    `07_pu_plain_ae_fit_run.py`'s measured `var_explained = 1 - mse_total / E||x_holdout||^2`
    convention).
    """
    if not smoke and d not in cc.D_SWEEP:
        raise ValueError(f"fit_and_field: d={d} is not in D_SWEEP={cc.D_SWEEP}.")
    if X.shape[0] != n_rows:
        raise ValueError(
            f"fit_and_field: X has {X.shape[0]} rows but n_rows={n_rows} was declared."
        )

    in_dim = X.shape[1]
    torch.manual_seed(cc.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(
        in_dim=in_dim, latent_dim=d, hidden=cc.AE_HIDDEN, activation=cc.AE_ACTIVATION
    )

    train_idx, holdout_idx = cc.split_indices(X.shape[0], cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]

    train_cfg = dict(cc.TRAIN_CFG)
    train_cfg["max_epochs"] = max_epochs
    cae.train_plain_ae(model, x_train32, train_cfg)

    model.eval().double()
    with torch.no_grad():
        z_full = model.encode(x64)
        y_holdout = model(x_holdout64)["y"]
    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    var_explained = 1.0 - recon["mse_total"] / sig

    field = decoder_curvature.plain_decoder_curvature(model, z_full)
    h_norm = np.linalg.norm(field["H_vec"].detach().cpu().numpy(), axis=1)
    cond = field["metric_condition_number"].detach().cpu().numpy()

    return {
        "h_norm": h_norm,
        "metric_condition_number": cond,
        "var_explained": float(var_explained),
        "reconstruction_stats": recon,
    }


def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    """Default resolves through `cache.cache_path(RECORD_STEM, "jsonl")`; a supplied value is
    passed through `cache._assert_inside_cache` before it is ever opened, so a traversal path
    raises rather than writes (T-07-01)."""
    if record_path_arg is None:
        return cache.cache_path(cc.RECORD_STEM, "jsonl")
    candidate = Path(record_path_arg)
    cache._assert_inside_cache(candidate)
    return candidate


def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar. Phase 6's
    `fix(06): serialize numpy arrays in the Phase 6 record` amendment is the cautionary
    precedent for what happens when this is not enforced."""
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                f"append_record_row: row[{key!r}] is a raw numpy value ({type(value)!r}); "
                "serialize it to a plain Python scalar/list before appending."
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_smoke(args: argparse.Namespace) -> str:
    """The tracer path: load the pair, take the first `--smoke-rows` rows of both arrays, run
    `fit_and_field` at d=20 with max_epochs=2, compute `per_point_mknn` at HEADLINE_K, run
    `two_tailed_permutation_null` with `--smoke-permutations`, apply `apply_verdict` with all
    three `d` keys mapped to the one measured boolean and a stubbed
    `positive_control_cleared_at`, and print the resulting verdict plus diagnostics. Writes
    NOTHING -- no record, no cache file, matching `region_partition_mknn_run.py`'s own smoke
    convention. Returns the printed verdict string."""
    cc.assert_preregistered()
    t_start = time.monotonic()

    X_hsc_full, X_ls_full, subsample_file = load_pu_pair(cc.PU_COLUMN_A, cc.PU_COLUMN_B)
    n_rows = args.smoke_rows
    X_hsc = X_hsc_full[:n_rows]
    X_ls = X_ls_full[:n_rows]
    print(
        f"\nSMOKE: {n_rows} rows, d=20, max_epochs=2, {args.smoke_permutations} permutations "
        "-- proves the whole path runs end to end on real PU rows. Writes NOTHING.\n"
    )

    t0 = time.monotonic()
    fit = fit_and_field(X_ls, d=20, max_epochs=2, n_rows=n_rows, smoke=True)
    t_fit = time.monotonic() - t0
    print(
        f"[fit+field]   wallclock={t_fit:.1f}s  var_explained={fit['var_explained']:.4f}  "
        f"cond(g) median={float(np.median(fit['metric_condition_number'])):.4e}"
    )

    t0 = time.monotonic()
    mknn_arr = cc.per_point_mknn(X_hsc, X_ls, cc.HEADLINE_K)
    t_mknn = time.monotonic() - t0
    scale = max(float(np.max(np.abs(mknn_arr))), 1e-300)
    n_distinct = int(np.unique(np.round(mknn_arr / scale, 12)).shape[0])
    print(
        f"[per_point_mknn] wallclock={t_mknn:.1f}s  n_distinct={n_distinct} "
        f"(<= HEADLINE_K + 1 = {cc.HEADLINE_K + 1})"
    )

    t0 = time.monotonic()
    two_tail = cc.two_tailed_permutation_null(
        fit["h_norm"], mknn_arr, args.smoke_permutations, cc.PERMUTATION_SEED,
        cc.NULL_QUANTILE_PER_TAIL,
    )
    t_perm = time.monotonic() - t0
    print(
        f"[permutation]  wallclock={t_perm:.1f}s  observed_rho={two_tail['observed_rho']:.4f}  "
        f"direction={two_tail['direction']}  clears_either={two_tail['clears_either']}"
    )
    print(
        f"  positive tail: threshold={two_tail['positive_tail']['null_threshold']:.4f}  "
        f"clears={two_tail['positive_tail']['clears_null']}"
    )
    print(
        f"  negative tail: threshold={two_tail['negative_tail']['null_threshold']:.4f}  "
        f"clears={two_tail['negative_tail']['clears_null']}"
    )

    stub_positive_control_cleared_at = cc.POSITIVE_CONTROL_TARGET_RHOS[0]
    per_d_results = {d: two_tail["clears_either"] for d in cc.D_SWEEP}
    verdict = cc.apply_verdict(per_d_results, stub_positive_control_cleared_at)
    print(
        f"\nVERDICT (smoke -- all three d stubbed to the one measured d=20 boolean, "
        f"positive_control_cleared_at stubbed to {stub_positive_control_cleared_at}): {verdict}"
    )
    assert verdict in cc.VERDICT_VALUES

    t_total = time.monotonic() - t_start
    print(f"\ntotal wallclock: {t_total:.1f}s")
    print("SMOKE MODE: writes nothing -- no record row, no cache file.")
    return verdict


def selfcheck() -> bool:
    """No PU data, no torch training. Known-answer assertions on synthetic arrays, mirroring
    `region_partition_mknn_run.selfcheck`'s own tally convention."""
    counts = {"pass": 0, "fail": 0}

    def check(name: str, cond: bool) -> None:
        if cond:
            counts["pass"] += 1
        else:
            counts["fail"] += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(20260826)

    z = rng.normal(size=(300, 8))
    check(
        "per_point_mknn(z, z, k) is all-ones for a cloud compared against itself",
        bool(np.allclose(cc.per_point_mknn(z, z, 10), 1.0)),
    )

    z1 = rng.normal(size=(400, 16))
    z2 = rng.normal(size=(400, 16))
    per_point = cc.per_point_mknn(z1, z2, 10)
    mean_score = mknn.mknn_score(z1, z2, 10)
    check(
        "per_point_mknn(z1, z2, k).mean() equals mknn.mknn_score(z1, z2, k)",
        bool(np.isclose(per_point.mean(), mean_score)),
    )
    floor = mknn.chance_floor(400, 10)
    check(
        "independent-cloud per_point_mknn mean lands near chance_floor(n, k) (factor of 3)",
        floor / 3.0 <= per_point.mean() <= floor * 3.0,
    )

    import curvature_field_pu_run as pu_field_run  # local diagnostics module; selfcheck only

    train_idx, holdout_idx = cc.split_indices(10000, cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    train_ref, holdout_ref = pu_field_run._split(
        10000, pu_field_run.PU_SPLIT_SEED, pu_field_run.PU_HOLDOUT_FRACTION
    )
    check(
        "split_indices reproduces curvature_field_pu_run._split element for element",
        bool(np.array_equal(train_idx, train_ref)) and bool(np.array_equal(holdout_idx, holdout_ref)),
    )

    h = rng.normal(size=500)
    m_anti = -h + rng.normal(scale=0.01, size=500)
    two_tail_anti = cc.two_tailed_permutation_null(h, m_anti, 200, 20260826, 0.975)
    check(
        "two_tailed_permutation_null recovers direction=='negative' on an anti-correlated pair",
        two_tail_anti["direction"] == "negative" and two_tail_anti["clears_either"],
    )

    m_pos = h + rng.normal(scale=0.01, size=500)
    two_tail_pos = cc.two_tailed_permutation_null(h, m_pos, 200, 20260826, 0.975)
    check(
        "two_tailed_permutation_null recovers direction=='positive' on a correlated pair",
        two_tail_pos["direction"] == "positive" and two_tail_pos["clears_either"],
    )

    total = counts["pass"] + counts["fail"]
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {total} total")
    return counts["fail"] == 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "dsweep", "positive-control"], default="smoke")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke-rows", type=int, default=800)
    p.add_argument("--smoke-permutations", type=int, default=50)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    if args.mode == "dsweep":
        raise NotImplementedError(
            "--mode dsweep is a pre-registered CLI surface (crossmodal_curvature.D_SWEEP) "
            "but its compute is implemented by plan 07-04, not this plan (07-02)."
        )
    if args.mode == "positive-control":
        raise NotImplementedError(
            "--mode positive-control is a pre-registered CLI surface "
            "(crossmodal_curvature.POSITIVE_CONTROL_RULE) but its compute is implemented by "
            "plan 07-03, not this plan (07-02)."
        )

    run_smoke(args)


if __name__ == "__main__":
    main()
