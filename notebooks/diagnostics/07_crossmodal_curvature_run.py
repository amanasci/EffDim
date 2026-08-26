"""Phase 7 crossmodal curvature runner. `--mode smoke` (07-02) proves the pipeline end to end;
`--mode positive-control` (07-03) plants D7-02's positive control against a real d=20 field
supplied via `--field-npz`, refusing to regenerate one if none is available; `--mode dsweep`
(07-04) is the deliverable: one serial in-process loop over `crossmodal_curvature.D_SWEEP`,
refusing to run without a `--freeze-commit` strict-ancestor proof (D7-06). Distinct from the
nine pre-existing `07_*_run.py` spike scripts, which stay untouched and satisfy no
pre-registration. Usage:
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --selfcheck
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode smoke
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode positive-control --field-npz <path>
    python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode dsweep --freeze-commit <sha>
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` (a token equal to `flag` followed by another token) and
    `--flag=value` (a single token starting with `flag=`) -- or `None` if `flag` was not
    passed in either form. CR-03: a raw `flag in argv` token-equality scan silently misses the
    `=` form entirely (the token is `"--flag=value"`, never `"--flag"`), even though argparse
    itself parses both identically. Kept dependency-free (only `sys`/plain strings) so it can
    run here, above the torch import, and be reused below for `--smoke-rows`/`--max-epochs`."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in torch/numpy -- NEW engineering here (no
# prior notebooks/ runner does this): 3 concurrent torch jobs measured load 44, ~10x slowdown.
_THREADS = 8
_threads_arg = _flag_value_from_argv("--threads", sys.argv)
if _threads_arg is not None:
    try:
        _THREADS = int(_threads_arg)
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
from scipy.stats import spearmanr  # noqa: E402

torch.set_num_threads(_THREADS)

from pu_manifold import cache  # noqa: E402
from pu_manifold import cae  # noqa: E402
from pu_manifold import cross_split_curvature  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import mknn  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402

# The freeze commit (the commit that added crossmodal_curvature.py, plan 07-01, Task 2) --
# every PU number this runner produces must be a strict git descendant of it
# (PREREGISTRATION_FREEZE_RULE). Recorded on every non-smoke record row as
# ``preregistration_commit``.
FREEZE_COMMIT_SHA = "f032745f6450068c63763993d39fa112fd36bb8c"

# Cost model (07-CONTEXT.md Section 7), measured on the real d=20 PU field: curvature
# computation dominates over training and scales as D * d**2. Printed as a per-d banner in
# --mode dsweep so a human watching the ~2h real run can see whether a given d is on pace.
DSWEEP_COST_MODEL_MINUTES = {20: 24, 25: 38, 32: 62}


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
    t0 = time.monotonic()
    cae.train_plain_ae(model, x_train32, train_cfg)
    wallclock_fit_s = time.monotonic() - t0

    model.eval().double()
    with torch.no_grad():
        z_full = model.encode(x64)
        y_holdout = model(x_holdout64)["y"]
    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    var_explained = 1.0 - recon["mse_total"] / sig

    t0 = time.monotonic()
    field = decoder_curvature.plain_decoder_curvature(model, z_full)
    wallclock_field_s = time.monotonic() - t0
    h_norm = np.linalg.norm(field["H_vec"].detach().cpu().numpy(), axis=1)
    cond = field["metric_condition_number"].detach().cpu().numpy()

    return {
        "h_norm": h_norm,
        "metric_condition_number": cond,
        "var_explained": float(var_explained),
        "reconstruction_stats": recon,
        "wallclock_fit_s": wallclock_fit_s,
        "wallclock_field_s": wallclock_field_s,
    }


def _distinct_value_count(arr: np.ndarray) -> int:
    """Distinct-value count at RELATIVE precision. Thin wrapper around
    `crossmodal_curvature._relative_precision_distinct_count` (plan 07-03) -- reused, not
    reimplemented; divide by the array's own maximum absolute value, round to 12 decimals,
    then `np.unique`, never raw float equality. `05-02-SUMMARY.md`'s retracted 5,301/9,852-
    vs-4/3 distinct-value miscount is the cautionary precedent."""
    return cc._relative_precision_distinct_count(arr)


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


def run_positive_control(args: argparse.Namespace) -> str:
    """D7-02's positive control, run against a real PU ``d=20`` ``||H||`` field. Refuses to
    invent a field it does not have: the field must be supplied via ``--field-npz`` and must
    carry an ``"h_norm"`` array (the same key ``run_smoke``'s ``fit_and_field`` returns), read
    from a path resolved through ``cache.cache_path`` / ``cache._assert_inside_cache`` so a
    traversal path raises rather than reads (T-07-01). If no field is available -- either
    ``--field-npz`` was not passed, or the path it names does not exist -- this raises naming
    plan 07-04 rather than regenerating one, matching plan 07-04's own d=20 sweep as the sole
    intended producer of that field.

    Calls ``crossmodal_curvature.assert_preregistered()`` first, then the same strict-ancestor
    freeze gate ``run_dsweep`` uses (CR-02: this path writes to the same sealed record
    ``run_dsweep`` protects and must be gated identically -- ``assert_preregistered()`` alone
    only checks that the constants themselves are well-formed, it says nothing about which git
    commit is checked out). Runs ``crossmodal_curvature.plant_positive_control`` at the frozen
    ``HEADLINE_K``, ``POSITIVE_CONTROL_TARGET_RHOS`` and ``POSITIVE_CONTROL_SEED``, appends one
    flat record row per target to the frozen record (``preregistration_commit`` / ``run_commit``
    per T-07-03), and prints ``smallest_cleared_target``'s value -- or, if nothing cleared, the
    string naming that the verdict is therefore forced to ``UNDERPOWERED -- NO CLAIM``
    (D7-02's override). Returns the printed ``smallest_cleared_target`` string.
    """
    cc.assert_preregistered()
    # CR-02: gate identically to run_dsweep before any row can be written. Unlike run_dsweep,
    # this path never took a caller-supplied --freeze-commit -- it already hardcodes
    # FREEZE_COMMIT_SHA (preserved below) -- so only the strict-ancestor-of-HEAD half of the
    # gate is new here; the equality-with-FREEZE_COMMIT_SHA half is trivially satisfied by
    # construction.
    _strict_ancestor_or_exit(FREEZE_COMMIT_SHA)

    if not args.field_npz:
        raise FileNotFoundError(
            "--mode positive-control requires --field-npz pointing at a real d=20 ||H|| field "
            "written by plan 07-04's sweep; none was provided, and this mode refuses to "
            "regenerate one. Run plan 07-04's d=20 sweep first, or pass an existing field's "
            "path via --field-npz."
        )

    candidate = Path(args.field_npz)
    cache._assert_inside_cache(candidate)
    if not candidate.exists():
        raise FileNotFoundError(
            f"--mode positive-control: {candidate} does not exist -- plan 07-04's d=20 sweep "
            "has not written a field there yet, and this mode refuses to regenerate one."
        )

    with np.load(candidate) as z:
        if "h_norm" not in z.files:
            raise KeyError(
                f"--mode positive-control: {candidate} does not carry an 'h_norm' array "
                f"(found: {sorted(z.files)}); this mode expects the same key run_smoke's "
                "fit_and_field returns."
            )
        h_real = np.asarray(z["h_norm"], dtype=np.float64)

    print(
        f"\nPOSITIVE CONTROL: {h_real.shape[0]} points loaded from {candidate.name}, planting "
        f"at HEADLINE_K={cc.HEADLINE_K}, targets={cc.POSITIVE_CONTROL_TARGET_RHOS}, "
        f"seed={cc.POSITIVE_CONTROL_SEED}.\n"
    )

    results = cc.plant_positive_control(
        h_real, cc.HEADLINE_K, cc.POSITIVE_CONTROL_TARGET_RHOS, cc.POSITIVE_CONTROL_SEED
    )
    cleared_at = cc.smallest_cleared_target(results)

    preregistration_commit = _git_rev_parse(FREEZE_COMMIT_SHA)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path)

    for result in results:
        row = {
            "row_kind": "positive_control",
            "target_rho": result["target_rho"],
            "achieved_rho": result["achieved_rho"],
            "slope": result["slope"],
            "n_distinct": result["n_distinct"],
            "clears_either": bool(result["clears_either"]),
            "direction": result["direction"],
            "positive_tail_threshold": float(result["positive_tail"]["null_threshold"]),
            "negative_tail_threshold": float(result["negative_tail"]["null_threshold"]),
            "field_npz": str(candidate),
            "preregistration_commit": preregistration_commit,
            "run_commit": run_commit,
        }
        append_record_row(row, record_path)
        print(
            f"  target_rho={result['target_rho']:.3f}  achieved_rho={result['achieved_rho']:.4f}  "
            f"clears_either={result['clears_either']}  direction={result['direction']}"
        )

    if cleared_at is None:
        outcome = (
            "positive control recovered NOTHING at the pre-registered effect-size grid "
            f"{cc.POSITIVE_CONTROL_TARGET_RHOS} -- verdict is forced to UNDERPOWERED -- NO CLAIM "
            "(D7-02 override)."
        )
    else:
        outcome = f"smallest_cleared_target: {cleared_at}"
    print(f"\n{outcome}")
    return outcome


def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """D7-06's freeze-ancestry gate. Exits 1 naming D7-06 unless `freeze_commit` resolves to
    EXACTLY the module's hardcoded `FREEZE_COMMIT_SHA` (CR-01: a wrong-but-plausible SHA that
    merely happens to precede HEAD in history -- a typo, an unrelated earlier commit -- must
    not silently pass and get stamped as `preregistration_commit`) AND is BOTH an ancestor of
    HEAD (`git merge-base --is-ancestor`) AND a STRICT one
    (`git rev-list --count <freeze>..HEAD >= 1`) -- `--is-ancestor` alone is insufficient
    because a commit is its own ancestor, so it would pass even for a number produced in the
    freeze commit itself (PREREGISTRATION_FREEZE_RULE). Both checks must hold."""
    if not freeze_commit:
        print(
            "ERROR (D7-06): --mode dsweep requires --freeze-commit naming the frozen "
            "commit's SHA. Refusing to compute a PU number without a strict-ancestor proof.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        resolved_commit = _git_rev_parse(freeze_commit)
    except subprocess.CalledProcessError:
        resolved_commit = None

    if resolved_commit != FREEZE_COMMIT_SHA:
        print(
            f"ERROR (D7-06): --freeze-commit {freeze_commit} (resolves to {resolved_commit}) "
            f"does not equal the known freeze commit FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA}. "
            "Refusing to stamp a PU number with the wrong preregistration_commit -- "
            "--freeze-commit must name THE freeze commit, not merely some earlier ancestor.",
            file=sys.stderr,
        )
        sys.exit(1)

    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"],
        cwd=str(NOTEBOOK_ROOT.parent),
    )
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{freeze_commit}..HEAD"],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
    )
    count = -1
    if count_result.returncode == 0 and count_result.stdout.strip().isdigit():
        count = int(count_result.stdout.strip())

    if is_ancestor.returncode != 0 or count < 1:
        print(
            f"ERROR (D7-06): --freeze-commit {freeze_commit} is not a STRICT git ancestor of "
            f"HEAD. is_ancestor_exit={is_ancestor.returncode} "
            f"rev_list_count({freeze_commit}..HEAD)={count}. A commit is its own ancestor, so "
            "`git merge-base --is-ancestor` alone is insufficient -- "
            "`git rev-list --count <freeze>..HEAD` must also be >= 1. "
            "PREREGISTRATION_FREEZE_RULE: no PU number may be produced at or before the "
            "freeze commit itself.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_dsweep(args: argparse.Namespace) -> str:
    """`--mode dsweep`: ONE sequential in-process loop over `crossmodal_curvature.D_SWEEP`,
    never concurrent tasks or processes (three concurrent torch jobs on this machine were
    measured driving load to 44 for roughly a 10x slowdown -- 07-CONTEXT.md Section 7).

    Order, and the order matters: `assert_preregistered()` first; the strict-ancestor freeze
    proof before any compute; load the pair once; compute every `MKNN_K_GRID` per-point array
    once, before the `d` loop (MKNN depends only on the two frozen embeddings and `k`, never
    on `d`); compute the density array once, before the `d` loop, and reuse it inside the loop
    for each `d`'s density statistics (`local_density_weights` is itself a k-NN computation on
    the full ambient cloud and does not depend on `d`, so recomputing it inside the loop would
    triple its cost for an identical value); then, for each `d` in `D_SWEEP` in tuple order,
    fit, field, both permutation tails, the non-gating sensitivity grid, that `d`'s density
    statistics, and one appended record row.

    `--resume`: a `d` already present in the record under a matching `preregistration_commit`
    is skipped. `--max-epochs` / `--smoke-rows`: reduced-scale exercise only -- when either is
    passed, this prints a prominent NOT-THE-DELIVERABLE banner and requires `--record-path`,
    refusing to let a reduced-scale run silently land in the frozen record or the frozen npz.
    """
    cc.assert_preregistered()
    _strict_ancestor_or_exit(args.freeze_commit)

    # CR-03: accept both `--flag value` and `--flag=value` -- a raw `"--flag" in sys.argv`
    # token scan silently misses the `=` form, which would let a reduced-scale request fall
    # through to the full-scale/production-record path below with no error.
    is_scratch = (
        _flag_value_from_argv("--smoke-rows", sys.argv) is not None
        or _flag_value_from_argv("--max-epochs", sys.argv) is not None
    )
    if is_scratch and not args.record_path:
        print(
            "ERROR: --smoke-rows / --max-epochs is a reduced-scale exercise, not the "
            "deliverable, and MUST be paired with --record-path pointing at a scratch path -- "
            "refusing to let a reduced-scale run default onto the frozen record.",
            file=sys.stderr,
        )
        sys.exit(1)

    preregistration_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path)

    if is_scratch:
        fields_path = record_path.with_name(record_path.stem + "_fields.npz")
        cache._assert_inside_cache(fields_path)
        print(
            f"\n{'=' * 78}\n"
            "THIS IS A REDUCED-SCALE EXERCISE RUN, NOT THE DELIVERABLE (--smoke-rows and/or "
            "--max-epochs were passed). Writing to the scratch paths "
            f"{record_path} / {fields_path}, never the frozen record.\n"
            f"{'=' * 78}\n"
        )
    else:
        fields_path = cache.cache_path("07_crossmodal_curvature_fields", "npz")

    max_epochs = (
        args.max_epochs if _flag_value_from_argv("--max-epochs", sys.argv) is not None
        else cc.MAX_EPOCHS
    )
    n_rows_override = (
        args.smoke_rows if _flag_value_from_argv("--smoke-rows", sys.argv) is not None
        else None
    )

    already_done_ds = set()
    if args.resume and record_path.exists():
        with record_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    existing.get("row_kind") == "sweep"
                    and existing.get("preregistration_commit") == preregistration_commit
                ):
                    already_done_ds.add(existing.get("d"))
        if already_done_ds:
            print(f"[resume] d values already recorded under a matching preregistration_commit: "
                  f"{sorted(already_done_ds)} -- these will be skipped.")

    X_hsc_full, X_ls_full, subsample_file = load_pu_pair(cc.PU_COLUMN_A, cc.PU_COLUMN_B)
    n_rows = n_rows_override if n_rows_override is not None else X_hsc_full.shape[0]
    X_hsc = X_hsc_full[:n_rows]
    X_ls = X_ls_full[:n_rows]
    print(
        f"\nDSWEEP: {n_rows} rows from {Path(subsample_file).name}, D_SWEEP={cc.D_SWEEP}, "
        f"max_epochs={max_epochs}, HEADLINE_K={cc.HEADLINE_K}.\n"
    )

    print(f"[per_point_mknn] computing once for every k in MKNN_K_GRID={cc.MKNN_K_GRID} "
          "-- MKNN depends only on the frozen embeddings and k, never on d.")
    mknn_by_k: Dict[int, np.ndarray] = {}
    mknn_distinct_by_k: Dict[str, int] = {}
    for k in cc.MKNN_K_GRID:
        t0 = time.monotonic()
        mknn_by_k[k] = cc.per_point_mknn(X_hsc, X_ls, k)
        mknn_distinct_by_k[str(k)] = _distinct_value_count(mknn_by_k[k])
        print(
            f"  k={k}: wallclock={time.monotonic() - t0:.1f}s  "
            f"n_distinct={mknn_distinct_by_k[str(k)]} (<= k + 1 = {k + 1})"
        )
    m_headline = mknn_by_k[cc.HEADLINE_K]

    print(
        f"\n[density] computing local_density_weights once on the {cc.DENSITY_INPUT} array at "
        f"DENSITY_K={cc.DENSITY_K}, DENSITY_FIELD_D={cc.DENSITY_FIELD_D} -- this does not "
        "depend on d, so it is computed once here and reused for each d's density statistics."
    )
    t0 = time.monotonic()
    w = curvature_probe.local_density_weights(X_ls, cc.DENSITY_K, cc.DENSITY_FIELD_D)
    density = 1.0 / w  # DENSITY_SIGN_CONVENTION: report on 1/w, matching Phase 4's REGN-01
    print(f"  wallclock={time.monotonic() - t0:.1f}s")

    spearman_density_vs_mknn = float(spearmanr(density, m_headline).statistic)
    density_p05 = float(np.percentile(density, 5))
    density_p50 = float(np.percentile(density, 50))
    density_p95 = float(np.percentile(density, 95))
    density_ratio_p95_p05 = density_p95 / density_p05 if density_p05 > 0 else float("inf")
    hubness_skewness_a = float(mknn.hubness_skewness(X_hsc, cc.HEADLINE_K))
    hubness_skewness_b = float(mknn.hubness_skewness(X_ls, cc.HEADLINE_K))
    chance_floor_val = float(mknn.chance_floor(X_hsc.shape[0], cc.HEADLINE_K))

    for d in cc.D_SWEEP:
        if d in already_done_ds:
            print(f"\n[resume] skipping d={d} -- already recorded.")
            continue

        projected_min = DSWEEP_COST_MODEL_MINUTES.get(d)
        print(
            f"\n{'-' * 78}\n"
            f"[d={d}] starting fit + field. Projected field time from the cost model: "
            f"~{projected_min} min (07-CONTEXT.md Section 7).\n"
            f"{'-' * 78}"
        )

        fit = fit_and_field(X_ls, d=d, max_epochs=max_epochs, n_rows=n_rows)
        print(
            f"[d={d}] fit+field done. wallclock_fit={fit['wallclock_fit_s']:.1f}s  "
            f"wallclock_field={fit['wallclock_field_s']:.1f}s  "
            f"var_explained={fit['var_explained']:.4f}  "
            f"cond(g) median={float(np.median(fit['metric_condition_number'])):.4e}"
        )

        two_tail = cc.two_tailed_permutation_null(
            fit["h_norm"], m_headline, cc.N_PERMUTATIONS, cc.PERMUTATION_SEED,
            cc.NULL_QUANTILE_PER_TAIL,
        )
        print(
            f"[d={d}] observed_rho={two_tail['observed_rho']:.4f}  "
            f"direction={two_tail['direction']}  clears_either={two_tail['clears_either']}"
        )

        sensitivity_grid = {}
        for k_other in cc.MKNN_K_GRID:
            if k_other == cc.HEADLINE_K:
                continue
            sensitivity_grid[str(k_other)] = float(
                spearmanr(fit["h_norm"], mknn_by_k[k_other]).statistic
            )

        spearman_density_vs_h = float(spearmanr(density, fit["h_norm"]).statistic)
        partial_rho_raw = float(
            cross_split_curvature.partial_spearman(fit["h_norm"], m_headline, controls=None)
        )
        partial_rho_density_controlled = float(
            cross_split_curvature.partial_spearman(fit["h_norm"], m_headline, controls=density)
        )

        row = {
            "row_kind": "sweep",
            "d": d,
            "alignment_metric": cc.ALIGNMENT_METRIC,
            "n": n_rows,
            "k": cc.HEADLINE_K,
            "var_explained": fit["var_explained"],
            "cond_g_median": float(np.median(fit["metric_condition_number"])),
            "cond_g_p95": float(np.percentile(fit["metric_condition_number"], 95)),
            "cond_g_max": float(np.max(fit["metric_condition_number"])),
            "h_norm_median": float(np.median(fit["h_norm"])),
            "h_norm_p05": float(np.percentile(fit["h_norm"], 5)),
            "h_norm_p95": float(np.percentile(fit["h_norm"], 95)),
            "observed_rho": two_tail["observed_rho"],
            "positive_tail_threshold": float(two_tail["positive_tail"]["null_threshold"]),
            "positive_tail_clears_null": bool(two_tail["positive_tail"]["clears_null"]),
            "negative_tail_threshold": float(two_tail["negative_tail"]["null_threshold"]),
            "negative_tail_clears_null": bool(two_tail["negative_tail"]["clears_null"]),
            "clears_either": bool(two_tail["clears_either"]),
            "direction": two_tail["direction"],
            "sensitivity_grid": sensitivity_grid,
            "spearman_density_vs_h": spearman_density_vs_h,
            "spearman_density_vs_mknn": spearman_density_vs_mknn,
            "partial_rho_raw": partial_rho_raw,
            "partial_rho_density_controlled": partial_rho_density_controlled,
            "density_p05": density_p05,
            "density_p50": density_p50,
            "density_p95": density_p95,
            "density_ratio_p95_p05": density_ratio_p95_p05,
            "hubness_skewness_a": hubness_skewness_a,
            "hubness_skewness_b": hubness_skewness_b,
            "chance_floor": chance_floor_val,
            "mknn_n_distinct_by_k": mknn_distinct_by_k,
            "preregistration_commit": preregistration_commit,
            "run_commit": run_commit,
            "wallclock_s": {"fit": fit["wallclock_fit_s"], "field": fit["wallclock_field_s"]},
        }
        append_record_row(row, record_path)

        savez_kwargs = {f"h_norm_{d}": fit["h_norm"], f"cond_g_{d}": fit["metric_condition_number"]}
        if fields_path.exists():
            with np.load(fields_path) as existing_z:
                for key in existing_z.files:
                    savez_kwargs.setdefault(key, existing_z[key])
        if d == 20:
            # --mode positive-control (run_positive_control) reads the bare "h_norm" key --
            # plant at PU's own realized d=20 dynamic range (D7-02's key link).
            savez_kwargs["h_norm"] = fit["h_norm"]
            savez_kwargs["cond_g"] = fit["metric_condition_number"]
        np.savez(fields_path, **savez_kwargs)
        print(f"[d={d}] wrote fields to {fields_path}")

    print(f"\nDSWEEP done. Record: {record_path}. Fields: {fields_path}.")
    return str(record_path)


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

    # D7-02: plant_positive_control's j/k discretization, determinism, and (via
    # partial_spearman below) the D7-03 partial-correlation route -- a single small target on a
    # small array keeps this within selfcheck's own fast-path budget (still pays for
    # N_PERMUTATIONS x 2 tails once per call, unlike every other check here).
    h_positive_control = rng.lognormal(mean=0.0, sigma=0.12, size=300)
    pc_results_a = cc.plant_positive_control(h_positive_control, cc.HEADLINE_K, (0.10,), 20260826)
    pc_results_b = cc.plant_positive_control(h_positive_control, cc.HEADLINE_K, (0.10,), 20260826)
    planted = pc_results_a[0]["planted"]
    check(
        "plant_positive_control's planted array is exactly j/HEADLINE_K discretized",
        bool(np.allclose(planted * cc.HEADLINE_K, np.round(planted * cc.HEADLINE_K))),
    )
    check(
        "plant_positive_control is deterministic across two identical calls",
        bool(np.array_equal(planted, pc_results_b[0]["planted"]))
        and pc_results_a[0]["achieved_rho"] == pc_results_b[0]["achieved_rho"],
    )

    from pu_manifold import cross_split_curvature

    h_tie_free = rng.normal(size=300)
    m_tie_free = rng.normal(size=300)
    raw_spearman = float(spearmanr(h_tie_free, m_tie_free).statistic)
    partial_no_controls = cross_split_curvature.partial_spearman(h_tie_free, m_tie_free, controls=None)
    check(
        "partial_spearman(h, m, controls=None) agrees with raw Spearman on a tie-free fixture",
        bool(np.isclose(partial_no_controls, raw_spearman, atol=1e-6)),
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
    p.add_argument(
        "--freeze-commit",
        type=str,
        default=None,
        help=(
            "--mode dsweep only, REQUIRED: the frozen commit's SHA (read from "
            "07-01-SUMMARY.md, not re-derived from git log). Must be a STRICT git ancestor "
            "of HEAD (D7-06) -- a commit is its own ancestor, so passing the current HEAD's "
            "own SHA here is rejected."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="--mode dsweep only: skip any d already recorded under a matching preregistration_commit.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "--mode dsweep only: override MAX_EPOCHS for a reduced-scale exercise run. When "
            "passed (together with --record-path), this is NOT the deliverable."
        ),
    )
    p.add_argument(
        "--field-npz",
        type=str,
        default=None,
        help=(
            "--mode positive-control only: path to a .npz carrying an 'h_norm' array -- a "
            "real d=20 ||H|| field written by plan 07-04's sweep. Resolved through "
            "cache._assert_inside_cache before it is ever opened (T-07-01); a traversal path "
            "raises rather than reads."
        ),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    if args.mode == "dsweep":
        run_dsweep(args)
        return
    if args.mode == "positive-control":
        run_positive_control(args)
        return

    run_smoke(args)


if __name__ == "__main__":
    main()
