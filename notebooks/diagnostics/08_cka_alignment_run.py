"""Phase 8 CKA alignment runner. `--mode selfcheck` (08-01) is the tracer: drives the Song et
al. (2012) unbiased-HSIC / CKA estimator through D8-16's invariance ladder for BOTH kernels
(D8-01) on synthetic pairs generated in-process from a fixed RNG seed -- no PU data is opened,
no subset exists, no null is built. It does NOT call `cka.assert_preregistered()`: it is a pure
in-memory known-answer check, following 07.1's own `--mode smoke` convention. It prints one line
per invariance-ladder rung and appends exactly one JSONL row per rung to a scratch record.

`--mode sigma` (08-03) is a PRE-FREEZE MEASUREMENT: it loads the real PU pair, computes the two
D8-03 global RBF bandwidths (the median pairwise Euclidean distance per modality, over all
10,000 points, before any subset exists), builds all eight Gram matrices once at the
0.5x/1x/2x sigma ladder (D8-04), and reports each build's wallclock and the process peak RSS. It
computes NO CKA value, constructs NO subset and constructs NO tertile -- its two sigma values
are pre-registration INPUTS under D8-03, frozen as literals by 08-04, not Phase 8 results. It
does NOT call `cka.assert_preregistered()` for the same reason `--mode selfcheck` does not.

`--mode sweep`, `--mode positive-control` and `--mode negative-control` (all 08-05) are
production modes: each first calls `_strict_ancestor_or_exit`, which requires `--freeze-commit`
to resolve to EXACTLY this module's own `FREEZE_COMMIT_SHA` and be a STRICT git ancestor of HEAD
(D8-22). `FREEZE_COMMIT_SHA` is still `""` in this commit -- 08-04 is the single commit that
wires the real SHA -- so every call to any of these three modes exits 1 today, regardless of
what `--freeze-commit` names; their actual sweep/control logic is not yet implemented and lands
in 08-05.

Usage:
    python notebooks/diagnostics/08_cka_alignment_run.py --mode selfcheck --record-path notebooks/.cache/08_scratch_tracer.jsonl
    python notebooks/diagnostics/08_cka_alignment_run.py --mode sigma --record-path notebooks/.cache/08_scratch_sigma.jsonl
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` and `--flag=value` -- or `None` if `flag` was not passed in either
    form. Copied verbatim from `07.1_density_stratified_null_run.py` (itself copied from
    `07_crossmodal_curvature_run.py`, CR-03): a raw `flag in argv` token-equality scan silently
    misses the `=` form. Kept dependency-free (only `sys`/plain strings) so it can run here,
    above the numpy import."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in numpy -- mirrors 07/07.1's own discipline
# (07-CONTEXT.md Section 7: concurrent jobs measured driving load up ~10x). This runner never
# trains a model and this plan's Gram matrices are all small synthetic arrays, so the cap
# matters far less here, but the discipline is cheap to keep for the runner's later modes
# (08-03/08-05), which DO build the full 10,000-point Gram matrices.
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
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Optional, Tuple  # noqa: E402

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cka  # noqa: E402


# Modes not yet implemented by any Phase 8 plan through 08-03: the plan that will implement each.
# "sweep"/"positive-control"/"negative-control" ARE dispatched by this plan (they call
# `_strict_ancestor_or_exit` first, per D8-22), but their actual sweep/control logic still lands
# in 08-05 -- while `FREEZE_COMMIT_SHA` is `""`, every call to one of them exits 1 at the
# ancestor gate before reaching any "not implemented" branch.
NOT_YET_IMPLEMENTED_MODES: Dict[str, str] = {
    "sweep": "08-05",
    "positive-control": "08-05",
    "negative-control": "08-05",
}

PRODUCTION_MODES_REQUIRING_FREEZE = ("sweep", "positive-control", "negative-control")

# 08-04 wires the real freeze commit SHA (D8-22). Empty in this commit: `_strict_ancestor_or_exit`
# below refuses EVERY `--freeze-commit` value, including a correct-looking SHA, until then.
FREEZE_COMMIT_SHA = ""

# 08-04 wires the real record stem (D8-22). Empty in this commit -- `resolve_record_path`'s
# default branch composes `cache.cache_path(RECORD_STEM, "jsonl")` from it, but every mode this
# plan implements (`selfcheck`, `sigma`) REQUIRES `--record-path` explicitly and never reaches
# this default branch; it exists only so 08-05's production modes have somewhere to land.
RECORD_STEM = ""

# D8-16's invariance ladder is run at these fixed literals -- declared at the call site, per the
# plan's own instruction, NOT as pre-registered constants in `cka.py` (tolerances gate nothing;
# see `08-01-PLAN.md`'s `<discretion_decisions>`).
SELFCHECK_N_POINTS = 2000
SELFCHECK_P_DIM = 64
SELFCHECK_SEED = 20260827
SELFCHECK_GRAM_DTYPE = np.float64
SELFCHECK_ATOL_CLOSED_FORM = 1e-6
SELFCHECK_ATOL_INDEPENDENCE = 0.05
SELFCHECK_NOISE_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _repo_root() -> Path:
    """The git repository root -- `NOTEBOOK_ROOT`'s parent -- mirroring
    `07.1_density_stratified_null_run.py`'s own `cwd=str(NOTEBOOK_ROOT.parent)` idiom, named as
    its own function here per this plan's action item."""
    return NOTEBOOK_ROOT.parent


def _git_rev_parse(rev: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """D8-22's freeze-ancestry gate, copying `07.1_density_stratified_null_run.py`'s
    CR-01-hardened `_strict_ancestor_or_exit` shape exactly (D-08 renamed to D8-22 throughout):
    exits 1 unless `freeze_commit` resolves to EXACTLY this module's own `FREEZE_COMMIT_SHA` (a
    wrong-but-genuine ancestor must not silently pass and get stamped as
    `preregistration_commit`) AND is BOTH an ancestor of HEAD (`git merge-base --is-ancestor`)
    AND a STRICT one (`git rev-list --count <freeze>..HEAD >= 1` -- a commit is its own ancestor,
    so `--is-ancestor` alone would pass even for a number produced in the freeze commit itself).

    `FREEZE_COMMIT_SHA` is `""` in this commit, so `resolved_commit != FREEZE_COMMIT_SHA` is
    true for every possible `--freeze-commit` value -- this function exits 1 unconditionally
    until 08-04 wires the real SHA. Called by `sweep`/`positive-control`/`negative-control`
    before any of their logic runs; `selfcheck` and `sigma` never call this -- both print a
    banner stating they are pre-freeze exercises producing no verdict number instead.
    """
    if not freeze_commit:
        print(
            "ERROR (D8-22): this mode requires --freeze-commit naming the frozen commit's SHA. "
            "Refusing to compute a Phase 8 number without a strict-ancestor proof.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        resolved_commit = _git_rev_parse(freeze_commit)
    except subprocess.CalledProcessError:
        resolved_commit = None

    if resolved_commit != FREEZE_COMMIT_SHA:
        print(
            f"ERROR (D8-22): --freeze-commit {freeze_commit} (resolves to {resolved_commit}) "
            f"does not equal the known freeze commit FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA!r}. "
            "FREEZE_COMMIT_SHA is still empty in this commit -- 08-04 is the single commit that "
            "wires the real SHA (D8-22) -- so no --freeze-commit value can pass yet. Refusing "
            "to stamp a Phase 8 number with the wrong preregistration_commit.",
            file=sys.stderr,
        )
        sys.exit(1)

    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"],
        cwd=str(_repo_root()),
    )
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{freeze_commit}..HEAD"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
    )
    count = -1
    if count_result.returncode == 0 and count_result.stdout.strip().isdigit():
        count = int(count_result.stdout.strip())

    if is_ancestor.returncode != 0 or count < 1:
        print(
            f"ERROR (D8-22): --freeze-commit {freeze_commit} is not a STRICT git ancestor of "
            f"HEAD. is_ancestor_exit={is_ancestor.returncode} "
            f"rev_list_count({freeze_commit}..HEAD)={count}. A commit is its own ancestor, so "
            "`git merge-base --is-ancestor` alone is insufficient -- "
            "`git rev-list --count <freeze>..HEAD` must also be >= 1. "
            "PREREGISTRATION_FREEZE_RULE: no Phase 8 number may be produced at or before the "
            "freeze commit itself.",
            file=sys.stderr,
        )
        sys.exit(1)


def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    """Defaults to `cache.cache_path(RECORD_STEM, "jsonl")` once `RECORD_STEM` is frozen (08-04);
    a supplied value is passed through `cache._assert_inside_cache` before it is ever opened, so
    a traversal path raises rather than writes -- copying 07.1's `resolve_record_path` shape.
    Every mode this plan implements (`selfcheck`, `sigma`) requires `--record-path` explicitly at
    its own call site and never reaches this default branch."""
    if record_path_arg is None:
        return cache.cache_path(RECORD_STEM, "jsonl")
    candidate = Path(record_path_arg)
    cache._assert_inside_cache(candidate)
    return candidate


def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar. Copied verbatim
    (in behavior) from `07.1_density_stratified_null_run.py` / `07_crossmodal_curvature_run.py`,
    including the raw-numpy-value `TypeError` guard -- Phase 6's
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


# 07.1's `d=25` seed field keys, plus Phase 7's own three `d` values -- the six frozen curvature
# fields D8-14 names verbatim. No decoder is ever retrained to produce any of these; they are
# read-only inputs already sitting in the gitignored cache from Phase 7 / 07.1.
FROZEN_FIELD_KEYS: Tuple[str, ...] = (
    "h_norm_20",
    "h_norm_25",
    "h_norm_32",
    "h_norm_25_seed0",
    "h_norm_25_seed1",
    "h_norm_25_seed2",
)


def load_pu_pair(column_a: str, column_b: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path. Mirrors
    `07.1_density_stratified_null_run.py`'s own `load_pu_pair` verbatim in behavior -- copied,
    not imported, because `notebooks/diagnostics/` is a plain directory (not a `pu_manifold`
    package member) and 07.1's runner performs its own module-level thread-environment writes
    that would collide with this runner's `_THREADS` setup above if imported at module scope
    (the same reason 07.1 itself copied this from Phase 7's runner rather than importing it).
    Keeps only files carrying both columns, selects the one with the most rows; on a tie keeps
    the lexicographically first path -- so Phase 8 resolves the exact same `subsample_*.npz`
    Phase 7 and 07.1 resolved."""
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz found in the notebook cache directory")
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


def load_frozen_fields() -> Dict[str, np.ndarray]:
    """D8-14: loads all six frozen curvature fields read-only from
    `07_crossmodal_curvature_fields.npz` (`h_norm_20`/`h_norm_25`/`h_norm_32`) and
    `07.1_seed_fields_d25.npz` (`h_norm_25_seed0`/`h_norm_25_seed1`/`h_norm_25_seed2`), each
    validated as shape `(10000,)` float64. There is NO compute-if-missing branch -- on a missing
    file or a missing key this prints a message naming the exact absent path or key and the fact
    that these are gitignored Phase 7 / 07.1 artifacts this phase does not regenerate, then exits
    1 (D8-14: no decoder retraining, halt-not-regenerate, the convention every prior frozen-cache
    phase in this milestone uses)."""
    cc_path = cache.cache_path("07_crossmodal_curvature_fields", "npz")
    seed_path = cache.cache_path("07.1_seed_fields_d25", "npz")

    for path in (cc_path, seed_path):
        if not path.exists():
            print(
                f"ERROR (D8-14): {path} does not exist. This is a gitignored Phase 7 / 07.1 "
                "artifact that Phase 8 does not regenerate -- run Phase 7's --mode dsweep (for "
                "07_crossmodal_curvature_fields.npz) or 07.1's seed runner (for "
                "07.1_seed_fields_d25.npz) in this checkout first. No decoder is retrained here.",
                file=sys.stderr,
            )
            sys.exit(1)

    fields: Dict[str, np.ndarray] = {}
    key_to_path = {
        "h_norm_20": cc_path,
        "h_norm_25": cc_path,
        "h_norm_32": cc_path,
        "h_norm_25_seed0": seed_path,
        "h_norm_25_seed1": seed_path,
        "h_norm_25_seed2": seed_path,
    }
    opened: Dict[Path, Any] = {}
    try:
        for path in (cc_path, seed_path):
            opened[path] = np.load(path)
        for key in FROZEN_FIELD_KEYS:
            path = key_to_path[key]
            z = opened[path]
            if key not in z.files:
                print(
                    f"ERROR (D8-14): {path} does not carry key {key!r} (found: "
                    f"{sorted(z.files)}). This field is not regenerated here.",
                    file=sys.stderr,
                )
                sys.exit(1)
            fields[key] = np.asarray(z[key], dtype=np.float64)
    finally:
        for z in opened.values():
            z.close()

    for key, arr in fields.items():
        if arr.shape != (10000,):
            print(
                f"ERROR (D8-14): field {key!r} has shape {arr.shape}, expected (10000,).",
                file=sys.stderr,
            )
            sys.exit(1)

    return fields


def _random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    """A random p x p orthogonal matrix via QR decomposition of a random Gaussian matrix."""
    a = rng.standard_normal((p, p))
    q, _ = np.linalg.qr(a)
    return q


def run_selfcheck(args: argparse.Namespace) -> bool:
    """D8-16's invariance ladder, for BOTH kernels (D8-01), on synthetic pairs generated
    in-process from `SELFCHECK_SEED`. Does NOT call `cka.assert_preregistered()` -- this is a
    pure in-memory known-answer check that opens no PU file and computes no Phase 8 number.

    Rungs, in order: orthogonal rotation of Z1 (linear and RBF CKA must both read 1.0);
    isotropic scaling of Z1 by 3.0 (linear CKA must read 1.0; RBF at the SAME fixed sigma must
    NOT read 1.0 -- RBF is not scale-invariant at fixed bandwidth); independent columns (both
    kernels below `SELFCHECK_ATOL_INDEPENDENCE` in absolute value); and an additive-noise ladder
    over `SELFCHECK_NOISE_SCALES`, which must be strictly decreasing for both kernels. For every
    RBF rung, the bandwidth is `cka.median_pairwise_distance(Z1)`, computed ONCE and reused for
    every `rbf_gram` call in the rung -- including the transformed side -- never recomputed per
    side (Pitfall 4).

    Prints one line per rung: `check=<name> kernel=<linear|rbf> measured=<value>
    expected=<value> PASS|FAIL`, then a final `SELFCHECK PASS`/`SELFCHECK FAIL` line. Appends
    exactly one JSONL row per rung to `--record-path`, which is REQUIRED (this mode refuses to
    default onto any frozen record path -- there is no frozen Phase 8 record yet, and this is a
    pre-freeze exercise, not a deliverable).
    """
    if not args.record_path:
        print(
            "ERROR: --mode selfcheck requires --record-path -- this is a PRE-FREEZE EXERCISE, "
            "not a deliverable, and refuses to default onto any frozen record path.",
            file=sys.stderr,
        )
        sys.exit(1)
    record_path = resolve_record_path(args.record_path)

    print(
        f"\n{'=' * 78}\n"
        "THIS IS A PRE-FREEZE EXERCISE, NOT A DELIVERABLE. Every Phase 8 gating constant in "
        "cka.py is UNSET (see cka.assert_preregistered) -- this selfcheck computes no Phase 8 "
        f"number and opens no PU file. Writing to {record_path}.\n"
        f"{'=' * 78}\n"
    )

    t_start = time.monotonic()
    rows = []
    all_passed = True

    def emit(check: str, kernel: str, measured, expected, passed: bool) -> None:
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        measured_str = "nan" if measured is None else f"{float(measured):.6f}"
        expected_str = "n/a" if expected is None else f"{float(expected):.6f}"
        print(f"check={check} kernel={kernel} measured={measured_str} expected={expected_str} {status}")
        all_passed = all_passed and passed
        rows.append({
            "mode": "selfcheck",
            "kernel": kernel,
            "check": check,
            "measured": None if measured is None else float(measured),
            "expected": None if expected is None else float(expected),
            "passed": bool(passed),
            "n_points": SELFCHECK_N_POINTS,
            "gram_dtype": np.dtype(SELFCHECK_GRAM_DTYPE).name,
        })

    rng = np.random.default_rng(SELFCHECK_SEED)
    Z1 = rng.standard_normal((SELFCHECK_N_POINTS, SELFCHECK_P_DIM))
    sigma = cka.median_pairwise_distance(Z1)  # frozen for this rung's entire RBF ladder

    K1_lin = cka.linear_gram(Z1, SELFCHECK_GRAM_DTYPE)
    K1_rbf = cka.rbf_gram(Z1, sigma, SELFCHECK_GRAM_DTYPE)

    # --- orthogonal rotation: linear and RBF CKA must both read 1.0 ------------------------
    Q = _random_orthogonal(SELFCHECK_P_DIM, rng)
    Z2_rot = Z1 @ Q
    v = cka.cka(K1_lin, cka.linear_gram(Z2_rot, SELFCHECK_GRAM_DTYPE))
    emit("orthogonal_rotation", "linear", v, 1.0, abs(v - 1.0) < SELFCHECK_ATOL_CLOSED_FORM)
    v = cka.cka(K1_rbf, cka.rbf_gram(Z2_rot, sigma, SELFCHECK_GRAM_DTYPE))
    emit("orthogonal_rotation", "rbf", v, 1.0, abs(v - 1.0) < SELFCHECK_ATOL_CLOSED_FORM)

    # --- isotropic scaling by 3.0: linear CKA = 1.0; RBF at the SAME sigma must NOT be 1.0 --
    Z2_scale = 3.0 * Z1
    v = cka.cka(K1_lin, cka.linear_gram(Z2_scale, SELFCHECK_GRAM_DTYPE))
    emit("isotropic_scaling", "linear", v, 1.0, abs(v - 1.0) < SELFCHECK_ATOL_CLOSED_FORM)
    v = cka.cka(K1_rbf, cka.rbf_gram(Z2_scale, sigma, SELFCHECK_GRAM_DTYPE))
    emit("isotropic_scaling", "rbf", v, 1.0, abs(v - 1.0) > SELFCHECK_ATOL_CLOSED_FORM)

    # --- independent columns: both kernels below ATOL_INDEPENDENCE in absolute value -------
    Z2_indep = rng.standard_normal((SELFCHECK_N_POINTS, SELFCHECK_P_DIM))
    v = cka.cka(K1_lin, cka.linear_gram(Z2_indep, SELFCHECK_GRAM_DTYPE))
    emit("independent_columns", "linear", v, 0.0, abs(v) < SELFCHECK_ATOL_INDEPENDENCE)
    v = cka.cka(K1_rbf, cka.rbf_gram(Z2_indep, sigma, SELFCHECK_GRAM_DTYPE))
    emit("independent_columns", "rbf", v, 0.0, abs(v) < SELFCHECK_ATOL_INDEPENDENCE)

    # --- additive-noise ladder: strictly decreasing for both kernels ------------------------
    lin_values, rbf_values = [], []
    for scale in SELFCHECK_NOISE_SCALES:
        Z2_noise = Z1 + scale * rng.standard_normal((SELFCHECK_N_POINTS, SELFCHECK_P_DIM))
        lin_values.append(cka.cka(K1_lin, cka.linear_gram(Z2_noise, SELFCHECK_GRAM_DTYPE)))
        rbf_values.append(cka.cka(K1_rbf, cka.rbf_gram(Z2_noise, sigma, SELFCHECK_GRAM_DTYPE)))
    lin_decreasing = all(lin_values[i] > lin_values[i + 1] for i in range(len(lin_values) - 1))
    rbf_decreasing = all(rbf_values[i] > rbf_values[i + 1] for i in range(len(rbf_values) - 1))
    for value in lin_values:
        emit("noise_ladder", "linear", value, None, lin_decreasing)
    for value in rbf_values:
        emit("noise_ladder", "rbf", value, None, rbf_decreasing)

    wallclock_s = time.monotonic() - t_start
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row["wallclock_s"] = wallclock_s
        row["timestamp"] = timestamp
        append_record_row(row, record_path)

    print("SELFCHECK PASS" if all_passed else "SELFCHECK FAIL")
    return all_passed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=["selfcheck", "sigma", "sweep", "positive-control", "negative-control"],
        default="selfcheck",
    )
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--freeze-commit",
        type=str,
        default=None,
        help=(
            "Production modes only (added by later plans, 08-05 onward): the frozen commit's "
            "SHA, a STRICT git ancestor of HEAD (D8-22). --mode selfcheck does not use this "
            "flag -- it computes no Phase 8 number."
        ),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode in PRODUCTION_MODES_REQUIRING_FREEZE:
        # D8-22: every production mode calls the strict-ancestor gate FIRST. While
        # FREEZE_COMMIT_SHA is "" (08-04 has not landed), this always exits 1 -- so the
        # NOT_YET_IMPLEMENTED_MODES branch below is presently unreachable for these three modes,
        # kept only for 08-05 to remove once it wires the real sweep/control logic.
        _strict_ancestor_or_exit(args.freeze_commit)
        print(
            f"ERROR: --mode {args.mode} is not implemented in this plan (08-03); it lands in "
            f"plan {NOT_YET_IMPLEMENTED_MODES[args.mode]}.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.mode == "selfcheck":
        ok = run_selfcheck(args)
        sys.exit(0 if ok else 1)

    if args.mode == "sigma":
        ok = run_sigma(args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
