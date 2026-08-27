"""Phase 07.1 density-stratified null runner. `--mode smoke` (07.1-03) is the tracer: reload the
frozen d=20 field, recompute MKNN and density from the frozen ambient embeddings, recompute the
observed `partial_rho_density_controlled` and assert it against Phase 7's frozen record (D-07),
build a within-stratum permutation null at `N_STRATA_HEADLINE` and read a two-tailed band off it
(D-06), and append exactly one scratch record row. `--selfcheck` (or `--mode selfcheck`) runs pure
in-memory known-answer checks plus a frozen-artifact existence check -- no PU data is loaded, no
model is trained. Distinct from `notebooks/diagnostics/07_crossmodal_curvature_run.py`, Phase 7's
own sealed runner, which is never imported here (its own thread-env-var side effects at import
time would collide with this runner's own `_THREADS` setup) -- its shape is mirrored instead.
Usage:
    python notebooks/diagnostics/07.1_density_stratified_null_run.py --selfcheck
    python notebooks/diagnostics/07.1_density_stratified_null_run.py --mode smoke --record-path notebooks/.cache/07.1_scratch_tracer.jsonl
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` and `--flag=value` -- or `None` if `flag` was not passed in either
    form. Copied verbatim from `07_crossmodal_curvature_run.py` (CR-03): a raw `flag in argv`
    token-equality scan silently misses the `=` form. Kept dependency-free (only `sys`/plain
    strings) so it can run here, above the numpy import."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in numpy -- mirrors 07_crossmodal_curvature_
# run.py's own discipline (07-CONTEXT.md Section 7: concurrent jobs measured driving load up
# ~10x). This runner never trains a model, so the cap matters far less here, but the discipline
# is cheap to keep and avoids a silent behavioural difference between the two 07.1/07 runners.
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

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional, Tuple  # noqa: E402

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import numpy as np  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cross_split_curvature  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402

# 07.1's own freeze commit (07.1-01-SUMMARY.md) -- the commit that added
# density_stratified_null.py alone. Every 07.1 number this runner produces must be a strict git
# descendant of it (D-08 / PREREGISTRATION_FREEZE_RULE). Only the production modes 07.1-04 and
# 07.1-05 add call _strict_ancestor_or_exit; --mode smoke is a reduced-resample exercise and
# never writes to the frozen record, so it does not gate on this.
FREEZE_COMMIT_SHA = "676866657676a36abb639782fa10ecb3061fd688"


def _git_rev_parse(rev: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """07.1's own freeze-ancestry gate (D-08), copying `07_crossmodal_curvature_run.py`'s
    CR-01-hardened `_strict_ancestor_or_exit` shape exactly: exits 1 naming D-08 unless
    `freeze_commit` resolves to EXACTLY this module's hardcoded `FREEZE_COMMIT_SHA` (a
    wrong-but-genuine ancestor must not silently pass and get stamped as
    `preregistration_commit`) AND is BOTH an ancestor of HEAD (`git merge-base --is-ancestor`)
    AND a STRICT one (`git rev-list --count <freeze>..HEAD >= 1`) -- a commit is its own
    ancestor, so `--is-ancestor` alone would pass even for a number produced in the freeze
    commit itself. Not called by `--mode smoke` (a scratch-only exercise); the production modes
    07.1-04 and 07.1-05 add call this before writing anything to the frozen record.
    """
    if not freeze_commit:
        print(
            "ERROR (D-08): this mode requires --freeze-commit naming the frozen commit's SHA. "
            "Refusing to compute a 07.1 number without a strict-ancestor proof.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        resolved_commit = _git_rev_parse(freeze_commit)
    except subprocess.CalledProcessError:
        resolved_commit = None

    if resolved_commit != FREEZE_COMMIT_SHA:
        print(
            f"ERROR (D-08): --freeze-commit {freeze_commit} (resolves to {resolved_commit}) "
            f"does not equal the known freeze commit FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA}. "
            "Refusing to stamp a 07.1 number with the wrong preregistration_commit -- "
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
            f"ERROR (D-08): --freeze-commit {freeze_commit} is not a STRICT git ancestor of "
            f"HEAD. is_ancestor_exit={is_ancestor.returncode} "
            f"rev_list_count({freeze_commit}..HEAD)={count}. A commit is its own ancestor, so "
            "`git merge-base --is-ancestor` alone is insufficient -- "
            "`git rev-list --count <freeze>..HEAD` must also be >= 1. "
            "PREREGISTRATION_FREEZE_RULE: no 07.1 number may be produced at or before the "
            "freeze commit itself.",
            file=sys.stderr,
        )
        sys.exit(1)


def _distinct_value_count(arr: np.ndarray) -> int:
    """Distinct-value count at RELATIVE precision. Thin delegation to
    `crossmodal_curvature._relative_precision_distinct_count` -- never a second
    implementation, since duplicating it is precisely the defect 07.1-02 (WR-02) removed
    elsewhere in this tree. `05-02-SUMMARY.md`'s retracted 5,301/9,852-vs-4/3 distinct-value
    miscount is the standing cautionary precedent."""
    return cc._relative_precision_distinct_count(arr)


def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    """Default resolves through `cache.cache_path(dsn.RECORD_STEM, "jsonl")`; a supplied value
    is passed through `cache._assert_inside_cache` before it is ever opened, so a traversal path
    raises rather than writes (T-07.1-09)."""
    if record_path_arg is None:
        return cache.cache_path(dsn.RECORD_STEM, "jsonl")
    candidate = Path(record_path_arg)
    cache._assert_inside_cache(candidate)
    return candidate


def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar. Copied from
    `07_crossmodal_curvature_run.py`, including its raw-numpy-value `TypeError` guard --
    Phase 6's `fix(06): serialize numpy arrays in the Phase 6 record` amendment is the
    cautionary precedent for what happens when this is not enforced (T-07.1-12)."""
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                f"append_record_row: row[{key!r}] is a raw numpy value ({type(value)!r}); "
                "serialize it to a plain Python scalar/list before appending."
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def load_pu_pair(column_a: str, column_b: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path. Mirrors
    `07_crossmodal_curvature_run.py`'s `load_pu_pair` verbatim (Phase 7's runner is never
    imported here -- its module-level thread-env-var writes would collide with this runner's
    own `_THREADS` setup above): keeps only files carrying both columns, selects the one with
    the most rows; on a tie keeps the lexicographically first path."""
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


def load_frozen_field(d: int) -> np.ndarray:
    """Loads `h_norm_{d}` from `notebooks/.cache/07_crossmodal_curvature_fields.npz` through
    `cache.cache_path`, so the path is containment-checked the same way every other 07.1 cache
    access is. Raises naming the missing key or file if absent -- this field is never
    regenerated here, only Phase 7's own dsweep produces it."""
    path = cache.cache_path("07_crossmodal_curvature_fields", "npz")
    if not path.exists():
        raise FileNotFoundError(
            f"load_frozen_field: {path} does not exist -- Phase 7's --mode dsweep has not been "
            "run in this checkout."
        )
    key = f"h_norm_{d}"
    with np.load(path) as z:
        if key not in z.files:
            raise KeyError(
                f"load_frozen_field: {path} does not carry key {key!r} (found: "
                f"{sorted(z.files)})."
            )
        return np.asarray(z[key], dtype=np.float64)


def recompute_mknn_and_density() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Recomputes MKNN and density from the frozen ambient embeddings -- NEITHER is stored in
    `07_crossmodal_curvature_fields.npz` (07.1-CONTEXT.md D-10: MKNN depends only on the two
    frozen embeddings and `HEADLINE_K`, never on `d`; density is a property of the ambient
    `legacysurvey` cloud and is also `d`-independent). Computed ONCE per run and reused across
    every `d` and every `S` a later 07.1 plan sweeps -- recomputing either inside a `d`/`S` loop
    would pay for an identical value repeatedly. Prints the wallclock of each step and the
    density p05/p50/p95 percentiles (`DENSITY_SIGN_CONVENTION`: reported on `1.0 / w`, matching
    Phase 4's REGN-01 convention).

    Returns `(mknn_arr, density, X_hsc, X_ls, subsample_file)`.
    """
    X_hsc, X_ls, subsample_file = load_pu_pair(cc.PU_COLUMN_A, cc.PU_COLUMN_B)

    t0 = time.monotonic()
    mknn_arr = cc.per_point_mknn(X_hsc, X_ls, cc.HEADLINE_K)
    print(f"[per_point_mknn]      wallclock={time.monotonic() - t0:.2f}s  k={cc.HEADLINE_K}")

    t0 = time.monotonic()
    w = curvature_probe.local_density_weights(X_ls, cc.DENSITY_K, cc.DENSITY_FIELD_D)
    density = 1.0 / w  # DENSITY_SIGN_CONVENTION: report on 1/w, matching Phase 4's REGN-01
    print(
        f"[local_density_weights] wallclock={time.monotonic() - t0:.2f}s  "
        f"density_k={cc.DENSITY_K}  density_field_d={cc.DENSITY_FIELD_D}"
    )
    print(
        f"  density p05={np.percentile(density, 5):.4e}  p50={np.percentile(density, 50):.4e}  "
        f"p95={np.percentile(density, 95):.4e}"
    )

    return mknn_arr, density, X_hsc, X_ls, subsample_file


def run_smoke(args: argparse.Namespace) -> str:
    """The tracer path (07.1-03): calls `dsn.assert_preregistered()` first, then reloads the
    frozen d=20 field, recomputes MKNN and density, recomputes
    `partial_spearman(h, m, controls=density)` and compares it against
    `dsn.FROZEN_PARTIAL_REFERENCE[20]` at the frozen rtol/atol -- HALTS with a non-zero exit
    naming D-07 if it does not match -- builds a `stratified_partial_null` at
    `N_STRATA_HEADLINE` with `--smoke-permutations` resamples and the frozen `PERMUTATION_SEED`
    / `NULL_QUANTILE_PER_TAIL`, prints every intermediate number, and appends exactly one row of
    plain Python scalars to the scratch record path. `--mode smoke` REQUIRES `--record-path`
    and refuses to default onto the frozen record -- it is a reduced-resample exercise, not the
    deliverable (T-07.1-10) -- and prints a banner saying so, mirroring `run_dsweep`'s own
    scratch banner. Does NOT call `_strict_ancestor_or_exit`; the production modes 07.1-04 and
    07.1-05 add do.
    """
    dsn.assert_preregistered()

    if not args.record_path:
        print(
            "ERROR: --mode smoke requires --record-path -- this is a reduced-resample "
            "exercise, not the deliverable, and refuses to default onto the frozen record.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\n{'=' * 78}\n"
        "THIS IS A REDUCED-RESAMPLE SCRATCH EXERCISE, NOT THE DELIVERABLE "
        f"({args.smoke_permutations} permutations vs the frozen N_PERMUTATIONS="
        f"{dsn.N_PERMUTATIONS}). Writing to {args.record_path}, never the frozen record.\n"
        f"{'=' * 78}\n"
    )

    t_start = time.monotonic()

    mknn_arr, density, X_hsc, X_ls, subsample_file = recompute_mknn_and_density()

    h = load_frozen_field(20)
    print(f"\n[load_frozen_field] d=20, {h.shape[0]} points loaded.")
    n_distinct_mknn = _distinct_value_count(mknn_arr)
    print(f"  per_point_mknn n_distinct={n_distinct_mknn} (<= HEADLINE_K + 1 = {cc.HEADLINE_K + 1})")

    t0 = time.monotonic()
    observed_partial = float(
        cross_split_curvature.partial_spearman(h, mknn_arr, controls=density)
    )
    print(
        f"\n[D-07] recomputed partial_rho_density_controlled (d=20): {observed_partial!r} "
        f"(wallclock={time.monotonic() - t0:.3f}s)"
    )
    frozen_reference = dsn.FROZEN_PARTIAL_REFERENCE[20]
    matches = bool(
        np.isclose(
            observed_partial,
            frozen_reference,
            rtol=dsn.PARTIAL_REFERENCE_RTOL,
            atol=dsn.PARTIAL_REFERENCE_ATOL,
        )
    )
    if not matches:
        print(
            f"ERROR (D-07): recomputed partial {observed_partial!r} does NOT match Phase 7's "
            f"frozen record {frozen_reference!r} at rtol={dsn.PARTIAL_REFERENCE_RTOL}, "
            f"atol={dsn.PARTIAL_REFERENCE_ATOL}. Halting -- this is a real finding about the "
            "frozen field or the reload path, never a nuisance to round past.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[D-07] matches Phase 7's frozen reference {frozen_reference!r} within tolerance.")

    t0 = time.monotonic()
    null_result = dsn.stratified_partial_null(
        h,
        mknn_arr,
        density,
        n_strata=dsn.N_STRATA_HEADLINE,
        n_resamples=args.smoke_permutations,
        seed=dsn.PERMUTATION_SEED,
        quantile_per_tail=dsn.NULL_QUANTILE_PER_TAIL,
    )
    t_null = time.monotonic() - t0
    print(
        f"\n[stratified_partial_null] wallclock={t_null:.3f}s  "
        f"N_STRATA_HEADLINE={dsn.N_STRATA_HEADLINE}  n_resamples={args.smoke_permutations}"
    )
    print(f"  observed          = {null_result['observed']!r}")
    print(f"  null_mean          = {null_result['null_mean']!r}")
    print(f"  null_std           = {null_result['null_std']!r}")
    print(f"  null_low           = {null_result['null_low']!r}")
    print(f"  null_high          = {null_result['null_high']!r}")
    print(f"  stratum_size_min   = {null_result['stratum_size_min']}")
    print(f"  stratum_size_max   = {null_result['stratum_size_max']}")
    print(f"  clears_positive    = {null_result['clears_positive']}")
    print(f"  clears_negative    = {null_result['clears_negative']}")
    print(f"  clears_either      = {null_result['clears_either']}")
    print(f"  direction          = {null_result['direction']}")

    record_path = resolve_record_path(args.record_path)
    row: Dict[str, Any] = dict(null_result)
    row["row_kind"] = "smoke_tracer"
    row["d"] = 20
    row["mknn_n_distinct"] = n_distinct_mknn
    row["subsample_file"] = str(subsample_file)
    append_record_row(row, record_path)
    print(f"\nappended one scratch row to {record_path}")

    t_total = time.monotonic() - t_start
    print(f"\ntotal wallclock: {t_total:.2f}s")
    print("SMOKE MODE: writes only to the scratch record path passed via --record-path.")
    return "smoke tracer complete"


def run_positive_control(args: argparse.Namespace) -> str:
    """D-04's positive control for the stratified null (07.1-04, Task 1): plants a controlled
    relationship on PU's own realized ``d=20`` ``||H||`` field and measures whether the
    stratified null recovers it, in both directions, across the frozen target grid. Calls
    ``dsn.assert_preregistered()`` first, then ``_strict_ancestor_or_exit(args.freeze_commit)``
    BEFORE touching any field or writing any row -- the CR-02 ordering ``c92260f`` hardened into
    Phase 7's runner for exactly this reason. Reloads ``h_norm_20`` and the recomputed MKNN and
    density, runs ``dsn.plant_positive_control_partial`` across
    ``POSITIVE_CONTROL_TARGET_RHOS`` x ``POSITIVE_CONTROL_DIRECTIONS``, prints one line per
    cell, and appends one ``row_kind: "positive_control"`` row per cell plus a
    ``row_kind: "positive_control_summary"`` row carrying the smallest cleared target per
    direction (``None`` when nothing cleared in that direction).
    """
    dsn.assert_preregistered()
    _strict_ancestor_or_exit(args.freeze_commit)

    mknn_arr, density, X_hsc, X_ls, subsample_file = recompute_mknn_and_density()
    h = load_frozen_field(20)
    print(f"\n[load_frozen_field] d=20, {h.shape[0]} points loaded.")

    print(
        f"\nPOSITIVE CONTROL (partial statistic): {h.shape[0]} points, "
        f"HEADLINE_K={cc.HEADLINE_K}, targets={dsn.POSITIVE_CONTROL_TARGET_RHOS}, "
        f"directions={dsn.POSITIVE_CONTROL_DIRECTIONS}, seed={dsn.POSITIVE_CONTROL_SEED}, "
        f"N_STRATA_HEADLINE={dsn.N_STRATA_HEADLINE}, N_PERMUTATIONS={dsn.N_PERMUTATIONS}.\n"
    )

    t0 = time.monotonic()
    results = dsn.plant_positive_control_partial(
        h, mknn_arr, density, cc.HEADLINE_K,
        dsn.POSITIVE_CONTROL_TARGET_RHOS, dsn.POSITIVE_CONTROL_DIRECTIONS,
        dsn.POSITIVE_CONTROL_SEED,
        n_strata=dsn.N_STRATA_HEADLINE, n_resamples=dsn.N_PERMUTATIONS,
        quantile_per_tail=dsn.NULL_QUANTILE_PER_TAIL,
    )
    print(f"[plant_positive_control_partial] wallclock={time.monotonic() - t0:.2f}s "
          f"for {len(results)} cells")

    preregistration_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path)

    for result in results:
        row: Dict[str, Any] = dict(result)
        row["row_kind"] = "positive_control"
        row["preregistration_commit"] = preregistration_commit
        row["run_commit"] = run_commit
        append_record_row(row, record_path)
        print(
            f"  target_rho={result['target_rho']:.3f}  direction={result['direction']:>8}  "
            f"achieved_rho={result['achieved_rho']:.4f}  slope={result['slope']:.4f}  "
            f"bracket_exhausted={result['bracket_exhausted']}  "
            f"clears_either={result['clears_either']}"
        )

    smallest_positive = dsn.smallest_cleared_target(results, "positive")
    smallest_negative = dsn.smallest_cleared_target(results, "negative")

    summary_row = {
        "row_kind": "positive_control_summary",
        "smallest_cleared_target_positive": smallest_positive,
        "smallest_cleared_target_negative": smallest_negative,
        "preregistration_commit": preregistration_commit,
        "run_commit": run_commit,
    }
    append_record_row(summary_row, record_path)

    print(f"\nsmallest_cleared_target (positive direction): {smallest_positive}")
    print(f"smallest_cleared_target (negative direction):  {smallest_negative}")
    if smallest_negative is None:
        print(
            "\nEvery residual this phase adjudicates is negative -- the positive control "
            "recovered NOTHING in the NEGATIVE direction across the pre-registered grid. D7.1-01's "
            "verdict is forced to UNDERPOWERED -- NO CLAIM if no d clears at N_STRATA_HEADLINE "
            "(PARTIAL_VERDICT_RULE); no non-detection reading may be reported without this power "
            "argument."
        )

    print(f"\nrecord: {record_path}")
    return f"positive-control complete: negative={smallest_negative} positive={smallest_positive}"


def _read_positive_control_summary(record_path: Path, preregistration_commit: str) -> Dict[str, Any]:
    """Reads the LAST ``row_kind: "positive_control_summary"`` row in ``record_path`` whose
    ``preregistration_commit`` matches. ``run_null`` (D7.1-01's verdict) depends on Task 1's
    positive control having already run under the SAME freeze commit -- 07.1-04's own
    Task-ordering precondition. Raises ``RuntimeError`` naming ``--mode positive-control`` if
    none is found, rather than silently proceeding with an unlicensed verdict."""
    if not record_path.exists():
        raise RuntimeError(
            f"run_null: {record_path} does not exist -- run `--mode positive-control` first "
            "(07.1-04 Task 1's power requirement must be satisfied before any verdict)."
        )
    summary = None
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("row_kind") == "positive_control_summary"
                and row.get("preregistration_commit") == preregistration_commit
            ):
                summary = row
    if summary is None:
        raise RuntimeError(
            f"run_null: no row_kind='positive_control_summary' row found in {record_path} "
            f"under preregistration_commit={preregistration_commit!r} -- run "
            "`--mode positive-control` first (07.1-04 Task 1)."
        )
    return summary


def run_null(args: argparse.Namespace) -> str:
    """D7.1-01's headline stratified null across the full ``d`` x ``S`` grid (D-02, D-03), the
    null-mean-vs-``S`` bias diagnostic (07.1-RESEARCH.md Pitfall 1 / Open Question 1), and the
    D7.1-01 verdict (07.1-04, Task 2).

    Order, and the order matters: ``assert_preregistered()``; the strict-ancestor freeze proof
    BEFORE any 07.1 number is produced; resolve the record path and (with ``--resume``) the set
    of ``(d, S)`` cells already recorded under a matching ``preregistration_commit``;
    ``recompute_mknn_and_density()`` ONCE (density is ``d``-independent, D-10); then for each
    ``d`` in ``D_SWEEP`` order, the observed partial recomputed once and D-07-asserted against
    ``FROZEN_PARTIAL_REFERENCE``, then for each ``S`` in ``STRATA_GRID`` order a stratified null
    at ``N_PERMUTATIONS``/``PERMUTATION_SEED``, appended as one ``row_kind: "null_grid"`` row.
    After the full grid, prints the null-mean-vs-``S`` diagnostic table BEFORE building any
    clearance mapping, reads Task 1's ``positive_control_summary`` row for the
    negative-direction ``positive_control_cleared_at``, builds the per-``d`` clearance mapping
    from ``N_STRATA_HEADLINE`` only, and calls the frozen ``dsn.apply_partial_verdict``,
    printing the per-``d`` clearance table unconditionally (D-15).
    """
    dsn.assert_preregistered()
    _strict_ancestor_or_exit(args.freeze_commit)

    preregistration_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path)

    already_done_cells = set()
    existing_null_grid_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    has_null_mean_vs_strata = False
    has_verdict = False
    if record_path.exists():
        with record_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if existing.get("preregistration_commit") != preregistration_commit:
                    continue
                if existing.get("row_kind") == "null_grid":
                    key = (existing.get("d"), existing.get("n_strata"))
                    existing_null_grid_rows[key] = existing
                    if args.resume:
                        already_done_cells.add(key)
                elif existing.get("row_kind") == "null_mean_vs_strata":
                    has_null_mean_vs_strata = True
                elif existing.get("row_kind") == "verdict":
                    has_verdict = True
        if already_done_cells:
            print(
                "[resume] (d, S) cells already recorded under a matching preregistration_commit: "
                f"{sorted(already_done_cells)} -- these will be skipped."
            )

    mknn_arr, density, X_hsc, X_ls, subsample_file = recompute_mknn_and_density()
    density_p05 = float(np.percentile(density, 5))
    density_p50 = float(np.percentile(density, 50))
    density_p95 = float(np.percentile(density, 95))
    density_ratio = density_p95 / density_p05 if density_p05 > 0 else float("inf")

    null_results: Dict[int, Dict[int, Dict[str, Any]]] = {d: {} for d in dsn.D_SWEEP}
    observed_by_d: Dict[int, float] = {}

    for d in dsn.D_SWEEP:
        h = load_frozen_field(d)
        t0 = time.monotonic()
        observed = float(cross_split_curvature.partial_spearman(h, mknn_arr, controls=density))
        frozen_reference = dsn.FROZEN_PARTIAL_REFERENCE[d]
        matches = bool(
            np.isclose(
                observed, frozen_reference,
                rtol=dsn.PARTIAL_REFERENCE_RTOL, atol=dsn.PARTIAL_REFERENCE_ATOL,
            )
        )
        print(
            f"\n[d={d}] recomputed partial_rho_density_controlled: {observed!r} "
            f"(wallclock={time.monotonic() - t0:.3f}s)"
        )
        if not matches:
            print(
                f"ERROR (D-07): d={d} recomputed partial {observed!r} does NOT match Phase 7's "
                f"frozen record {frozen_reference!r} at rtol={dsn.PARTIAL_REFERENCE_RTOL}, "
                f"atol={dsn.PARTIAL_REFERENCE_ATOL}. Halting -- this is a real finding about "
                "the frozen field or the reload path, never a nuisance to round past.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[d={d}] matches Phase 7's frozen reference {frozen_reference!r} within tolerance.")
        observed_by_d[d] = observed

        for n_strata in dsn.STRATA_GRID:
            key = (d, n_strata)
            if key in already_done_cells:
                print(f"[resume] skipping d={d}, S={n_strata} -- already recorded.")
                existing_row = existing_null_grid_rows[key]
                null_results[d][n_strata] = {
                    "null_mean": existing_row["null_mean"],
                    "null_std": existing_row["null_std"],
                    "null_low": existing_row["null_low"],
                    "null_high": existing_row["null_high"],
                    "clears_positive": existing_row["clears_positive"],
                    "clears_negative": existing_row["clears_negative"],
                    "clears_either": existing_row["clears_either"],
                    "direction": existing_row["direction"],
                    "own_edge": existing_row["own_edge"],
                    "signed_margin": existing_row["signed_margin"],
                    "margin_fraction": existing_row["margin_fraction"],
                }
                continue

            t0 = time.monotonic()
            null_result = dsn.stratified_partial_null(
                h, mknn_arr, density,
                n_strata=n_strata, n_resamples=dsn.N_PERMUTATIONS, seed=dsn.PERMUTATION_SEED,
                quantile_per_tail=dsn.NULL_QUANTILE_PER_TAIL,
            )
            t_null = time.monotonic() - t0

            if null_result["observed"] != observed:
                print(
                    f"ERROR: d={d} S={n_strata} observed {null_result['observed']!r} differs "
                    f"from the once-computed observed {observed!r} -- the grid must move only "
                    "the null (D-03); the observed statistic touched the strata, which it must "
                    "not. Halting.",
                    file=sys.stderr,
                )
                sys.exit(1)

            own_edge = null_result["null_low"] if observed < 0 else null_result["null_high"]
            signed_margin = observed - own_edge
            band_half_width = (null_result["null_high"] - null_result["null_low"]) / 2.0
            margin_fraction = (
                signed_margin / band_half_width if band_half_width != 0 else None
            )

            null_results[d][n_strata] = dict(null_result)
            null_results[d][n_strata]["own_edge"] = own_edge
            null_results[d][n_strata]["signed_margin"] = signed_margin
            null_results[d][n_strata]["margin_fraction"] = margin_fraction

            print(
                f"  [d={d} S={n_strata}] wallclock={t_null:.3f}s  "
                f"null_mean={null_result['null_mean']!r}  null_low={null_result['null_low']!r}  "
                f"null_high={null_result['null_high']!r}  clears_either={null_result['clears_either']}  "
                f"own_edge={own_edge!r}  signed_margin={signed_margin!r}  "
                f"margin_fraction={margin_fraction!r}"
            )

            row = {
                "row_kind": "null_grid",
                "d": d,
                "n_strata": n_strata,
                "observed": observed,
                "null_mean": null_result["null_mean"],
                "null_std": null_result["null_std"],
                "null_low": null_result["null_low"],
                "null_high": null_result["null_high"],
                "stratum_size_min": null_result["stratum_size_min"],
                "stratum_size_max": null_result["stratum_size_max"],
                "clears_positive": null_result["clears_positive"],
                "clears_negative": null_result["clears_negative"],
                "clears_either": null_result["clears_either"],
                "direction": null_result["direction"],
                "own_edge": own_edge,
                "signed_margin": signed_margin,
                "margin_fraction": margin_fraction,
                "n_permutations": dsn.N_PERMUTATIONS,
                "permutation_seed": dsn.PERMUTATION_SEED,
                "preregistration_commit": preregistration_commit,
                "run_commit": run_commit,
            }
            append_record_row(row, record_path)

    # --- Bias diagnostic: null_mean vs S, printed BEFORE any clearance verdict (RESEARCH Open
    # Question 1) --------------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("NULL-MEAN-vs-S DIAGNOSTIC (07.1-RESEARCH.md Pitfall 1 / Open Question 1) -- printed")
    print("BEFORE any clearance verdict.")
    print(f"{'=' * 78}")
    print(
        f"  recomputed density p05={density_p05:.4e}  p50={density_p50:.4e}  "
        f"p95={density_p95:.4e}  ratio(p95/p05)={density_ratio:.4e}"
    )

    warning_by_d: Dict[str, str] = {}
    null_mean_table: Dict[str, Dict[str, Any]] = {}
    for d in dsn.D_SWEEP:
        print(f"\n  d={d}  (observed={observed_by_d[d]!r})")
        null_mean_table[str(d)] = {}
        clears_by_s = {}
        for n_strata in dsn.STRATA_GRID:
            r = null_results[d].get(n_strata)
            if r is None:
                print(f"    S={n_strata}  (not available this run)")
                continue
            print(
                f"    S={n_strata:>3}  null_mean={r['null_mean']:.6f}  null_std={r['null_std']:.6f}  "
                f"clears_either={r['clears_either']}"
            )
            null_mean_table[str(d)][str(n_strata)] = {
                "null_mean": r["null_mean"], "null_std": r["null_std"],
                "clears_either": r["clears_either"],
            }
            clears_by_s[n_strata] = r["clears_either"]

        warning = "none"
        if 10 in clears_by_s and 50 in clears_by_s:
            if clears_by_s[10] and not clears_by_s[50]:
                warning = (
                    "BIAS SIGNATURE (clears at S=10, not S=50) -- 07.1-RESEARCH.md Pitfall 1's "
                    "positive null-mean bias, liberal on the negative tail"
                )
            elif clears_by_s[50] and not clears_by_s[10]:
                warning = (
                    "TIGHTNESS MECHANISM (clears at S=50, not S=10) -- D-02's stated "
                    "finer-strata-narrows-the-band mechanism"
                )
        warning_by_d[str(d)] = warning
        print(f"    warning sign: {warning}")

    if not has_null_mean_vs_strata:
        diag_row = {
            "row_kind": "null_mean_vs_strata",
            "density_p05": density_p05,
            "density_p50": density_p50,
            "density_p95": density_p95,
            "density_ratio_p95_p05": density_ratio,
            "null_mean_by_d_s": null_mean_table,
            "warning_by_d": warning_by_d,
            "preregistration_commit": preregistration_commit,
            "run_commit": run_commit,
        }
        append_record_row(diag_row, record_path)

    # --- D7.1-01 verdict: headline S only (D-02) --------------------------------------------
    positive_control_summary = _read_positive_control_summary(record_path, preregistration_commit)
    positive_control_cleared_at = positive_control_summary.get("smallest_cleared_target_negative")

    per_d_results = {}
    print(
        f"\n{'=' * 78}\nPER-d CLEARANCE TABLE (N_STRATA_HEADLINE={dsn.N_STRATA_HEADLINE}, "
        f"D-15: printed unconditionally, whichever verdict fires)\n{'=' * 78}"
    )
    for d in dsn.D_SWEEP:
        r = null_results[d].get(dsn.N_STRATA_HEADLINE)
        if r is None:
            raise RuntimeError(
                f"run_null: no headline S={dsn.N_STRATA_HEADLINE} result available for d={d} -- "
                "cannot build the verdict's clearance mapping."
            )
        per_d_results[d] = bool(r["clears_either"])
        print(
            f"  d={d:>2}  observed={observed_by_d[d]!r}  own_edge={r['own_edge']!r}  "
            f"signed_margin={r['signed_margin']!r}  margin_fraction={r['margin_fraction']!r}  "
            f"clears_either={r['clears_either']}"
        )

    verdict = dsn.apply_partial_verdict(per_d_results, positive_control_cleared_at)
    print(f"\nD7.1-01 VERDICT: {verdict}")
    print(f"positive_control_cleared_at (negative direction): {positive_control_cleared_at}")

    if not has_verdict:
        verdict_row = {
            "row_kind": "verdict",
            "verdict": verdict,
            "per_d_results": {str(d): v for d, v in per_d_results.items()},
            "positive_control_cleared_at": positive_control_cleared_at,
            "n_strata_headline": dsn.N_STRATA_HEADLINE,
            "partial_verdict_rule_first_line": dsn.PARTIAL_VERDICT_RULE.splitlines()[0],
            "preregistration_commit": preregistration_commit,
            "run_commit": run_commit,
        }
        append_record_row(verdict_row, record_path)

    print(f"\nrecord: {record_path}")
    return verdict


def selfcheck() -> bool:
    """No PU training, no permutation loop over full-scale data. Unlike Phase 7's selfcheck
    (pure in-memory only), this ALSO checks that the frozen artifacts this runner depends on
    exist in the current checkout -- imports clean, `assert_preregistered()` passes, both
    verdict functions raise on a partial key set, and the frozen npz/jsonl/subsample files are
    present. Prints one line per check and returns a bool, mirroring
    `07_crossmodal_curvature_run.selfcheck`'s own tally convention."""
    counts = {"pass": 0, "fail": 0}

    def check(name: str, cond: bool) -> None:
        if cond:
            counts["pass"] += 1
        else:
            counts["fail"] += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    try:
        dsn.assert_preregistered()
        check("dsn.assert_preregistered() passes", True)
    except Exception as exc:  # noqa: BLE001 -- selfcheck reports every failure, never raises
        check(f"dsn.assert_preregistered() passes ({exc})", False)

    try:
        dsn.apply_partial_verdict({20: True, 25: True}, 0.02)
        check("apply_partial_verdict raises ValueError on a partial d-key set", False)
    except ValueError:
        check("apply_partial_verdict raises ValueError on a partial d-key set", True)

    try:
        dsn.apply_seed_verdict({0: True, 1: True}, 0.02)
        check("apply_seed_verdict raises ValueError on a partial seed-key set", False)
    except ValueError:
        check("apply_seed_verdict raises ValueError on a partial seed-key set", True)

    fields_path = cache.cache_path("07_crossmodal_curvature_fields", "npz")
    check(f"frozen field artifact exists: {fields_path.name}", fields_path.exists())
    if fields_path.exists():
        with np.load(fields_path) as z:
            check("h_norm_20 key present in frozen field artifact", "h_norm_20" in z.files)

    record_path = cache.cache_path("07_crossmodal_curvature", "jsonl")
    check(f"frozen record artifact exists: {record_path.name}", record_path.exists())

    subsample_cands = glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz"))
    check("at least one subsample_*.npz carrying both PU columns exists", len(subsample_cands) > 0)

    total = counts["pass"] + counts["fail"]
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {total} total")
    return counts["fail"] == 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode", choices=["smoke", "selfcheck", "positive-control", "null"], default="smoke"
    )
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke-permutations", type=int, default=100)
    p.add_argument(
        "--freeze-commit",
        type=str,
        default=None,
        help=(
            "Production modes only (added by plans 07.1-04 / 07.1-05): the frozen commit's "
            "SHA (read from 07.1-01-SUMMARY.md, not re-derived from git log). Must be a "
            "STRICT git ancestor of HEAD (D-08). --mode smoke does not use this flag."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Production modes only (added by plans 07.1-04 / 07.1-05): skip work already "
            "recorded under a matching preregistration_commit. --mode smoke does not use this "
            "flag."
        ),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.selfcheck or args.mode == "selfcheck":
        ok = selfcheck()
        sys.exit(0 if ok else 1)

    if args.mode == "positive-control":
        run_positive_control(args)
        return

    if args.mode == "null":
        run_null(args)
        return

    run_smoke(args)


if __name__ == "__main__":
    main()
