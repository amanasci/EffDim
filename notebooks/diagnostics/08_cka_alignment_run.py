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
production modes: each first calls `cka.assert_preregistered()` (refusing to run against a
drifted or incomplete freeze) and then `_strict_ancestor_or_exit`, which requires
`--freeze-commit` to resolve to EXACTLY this module's own `FREEZE_COMMIT_SHA` and be a STRICT
git ancestor of HEAD (D8-22). `FREEZE_COMMIT_SHA` was wired to the real freeze commit by 08-04;
their actual sweep/control logic is not yet implemented and lands in 08-05, so today these three
modes pass both pre-flight checks and then exit 2 naming the plan that implements them.

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
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cka  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402


# All three production modes are implemented by this plan (08-05): `run_positive_control`
# (Task 1, D8-18), `run_negative_control` (Task 2, D8-19), `run_sweep` (Task 3, D8-09/13/15).
# Kept as an empty dict (rather than deleted) so `main()`'s dispatch shape below does not need to
# change if a future plan ever adds a new not-yet-implemented mode.
NOT_YET_IMPLEMENTED_MODES: Dict[str, str] = {}

PRODUCTION_MODES_REQUIRING_FREEZE = ("sweep", "positive-control", "negative-control")

# D8-22's freeze commit: the commit that filled every constant in cka.py's frozen block (37
# constants through 08-03, plus the eight control/reporting constants born already-frozen at
# 08-04, 45 total). Every Phase 8 number must be a STRICT git descendant of this commit --
# `_strict_ancestor_or_exit` below enforces this exactly.
FREEZE_COMMIT_SHA = "816863cae2209261470d1d041dcc4484a3056947"

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

    `FREEZE_COMMIT_SHA` was wired to the real 08-04 freeze commit's SHA, so a correct
    `--freeze-commit` value now passes. Called by `sweep`/`positive-control`/`negative-control`
    SECOND, after `cka.assert_preregistered()` has already confirmed no constant is UNSET or
    drifted; `selfcheck` and `sigma` never call either -- both print a banner stating they are
    pre-freeze exercises producing no verdict number instead.
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
            "Refusing to stamp a Phase 8 number with the wrong preregistration_commit.",
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
    """Defaults to `cache.cache_path(cka.RECORD_STEM, "jsonl")` -- `cka.RECORD_STEM` was frozen
    to `"08_cka_alignment"` by 08-04; a supplied value is passed through
    `cache._assert_inside_cache` before it is ever opened, so a traversal path raises rather than
    writes -- copying 07.1's `resolve_record_path` shape. `selfcheck` and `sigma` both require
    `--record-path` explicitly at their own call site and never reach this default branch; it is
    08-05's production modes that land on it."""
    if record_path_arg is None:
        return cache.cache_path(cka.RECORD_STEM, "jsonl")
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


# --- Task 3: --mode sigma -- the two D8-03 frozen global RBF bandwidths and the ------------------
# Gram-matrix-once proof (D8-04's 0.5x/1x/2x sensitivity ladder). ------------------------------

# Discretion decision (08-03-PLAN.md's <discretion_decisions>, ratified at the freeze as
# cka.GRAM_DTYPE): float32 halves eight (10000, 10000) float64 matrices' ~6.4 GB to ~3.2 GB;
# 08-01's test_gram_dtype_agreement is the evidence the choice does not move CKA beyond 1e-5.
# `cka.GRAM_DTYPE` itself is still UNSET (""), so this is a runner-local literal, not a read
# across the freeze boundary.
SIGMA_GRAM_DTYPE = np.float32
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0)

# T-08-07 mitigation: if currently AVAILABLE physical memory is below this many MB, run_sigma
# takes the one-at-a-time disk-cache fallback instead of holding all eight Gram matrices in
# memory simultaneously (RESEARCH.md's ~3.2 GB float32 / ~6.4 GB float64 estimate for all eight;
# 6 GB leaves comfortable headroom either way). Never silently changes SIGMA_GRAM_DTYPE.
FALLBACK_THRESHOLD_MB = 6144.0


def _available_memory_mb() -> float:
    """Best-effort estimate of currently available physical memory, in MB, via POSIX
    `os.sysconf` -- no extra dependency (`psutil` is not part of this project's notebook
    requirements). Returns `-1.0` (treated as "unknown, assume scarce") if the platform does not
    expose these sysconf names."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024.0 * 1024.0)
    except (ValueError, OSError, AttributeError):
        return -1.0


def _peak_rss_mb() -> float:
    """The process's peak resident set size, in MB, read from `resource.getrusage` -- the same
    idiom `geometry_probes_run.py` / `ph_budget_calibration_run.py` already use elsewhere in this
    tree (`ru_maxrss` is reported in KB on Linux)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _rbf_kernel_name(multiplier: float) -> str:
    """`"rbf_sigma"` at multiplier 1.0 (the headline rung); `"rbf_{multiplier}sigma"` otherwise
    (e.g. `"rbf_0.5sigma"`, `"rbf_2sigma"` at the diagnostic rungs, D8-04)."""
    if multiplier == 1.0:
        return "rbf_sigma"
    return f"rbf_{multiplier:g}sigma"


def build_gram_matrices(
    X_hsc: np.ndarray,
    X_ls: np.ndarray,
    sigma_hsc: float,
    sigma_ls: float,
    multipliers: Tuple[float, ...],
    dtype: Any,
    on_build: Optional[Callable[[str, str, float], None]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """D8-03/D8-04's eight Gram matrices, each built EXACTLY ONCE over all 10,000 points of its
    own modality and stored at `dtype` -- the Gram-matrix-once architecture 08-05's sweep, null
    and controls all sub-index (`cka.cka_on_subset`), never rebuild.

    Returns `(grams_hsc, grams_ls)`, each a dict keyed by kernel name --
    `"linear"`, `"rbf_0.5sigma"`, `"rbf_sigma"`, `"rbf_2sigma"` at the default `multipliers`.
    The RBF bandwidth for a modality is that modality's OWN frozen scalar (`sigma_hsc` /
    `sigma_ls`) times the multiplier -- HSC and Legacy Survey never share a scale, because they
    are different spaces with no reason to (D8-03). Accepts no subset index at all, so no call
    site can pass one and manufacture a per-subset bandwidth.

    `on_build`, if given, is called as `on_build(modality_name, kernel_name, wallclock_s)`
    immediately after each individual matrix finishes building -- optional instrumentation hook
    for a caller that wants per-matrix timing (`run_sigma` uses this to print and record each
    build); never required for correctness, and this function computes nothing from its return
    value.
    """

    def _one_modality(modality_name: str, X: np.ndarray, sigma: float) -> Dict[str, np.ndarray]:
        grams: Dict[str, np.ndarray] = {}
        t0 = time.monotonic()
        grams["linear"] = cka.linear_gram(X, dtype)
        if on_build is not None:
            on_build(modality_name, "linear", time.monotonic() - t0)
        for multiplier in multipliers:
            name = _rbf_kernel_name(multiplier)
            t0 = time.monotonic()
            grams[name] = cka.rbf_gram(X, sigma * multiplier, dtype)
            if on_build is not None:
                on_build(modality_name, name, time.monotonic() - t0)
        return grams

    grams_hsc = _one_modality("hsc", X_hsc, sigma_hsc)
    grams_ls = _one_modality("legacysurvey", X_ls, sigma_ls)
    return grams_hsc, grams_ls


def run_sigma(args: argparse.Namespace) -> bool:
    """D8-03's pre-freeze measurement mode: measures the two global RBF bandwidths (the median
    pairwise Euclidean distance per modality, over ALL 10,000 points, before any subset exists)
    and proves the Gram-matrix-once architecture by building all eight Gram matrices once at the
    0.5x/1x/2x sigma ladder (D8-04), reporting each build's wallclock, dtype and the process peak
    RSS. Computes NO CKA value, constructs NO density stratum and constructs NO tertile -- those
    all depend on constants that are still UNSET (D8-22); this mode's two sigma outputs are
    pre-registration INPUTS, not Phase 8 results, so it does NOT call
    `cka.assert_preregistered()` -- the same discipline `--mode selfcheck` follows.

    REQUIRES `--record-path` and refuses to default onto any frozen record path (there is no
    frozen Phase 8 record yet). Appends one JSONL row per (modality, kernel) -- eight rows total
    -- to the scratch record, every value cast to a plain Python scalar before
    `append_record_row` ever sees it.

    If currently available physical memory is below `FALLBACK_THRESHOLD_MB`, falls back to
    building and releasing one Gram matrix at a time, caching each to `notebooks/.cache/` as a
    `.npy` through `cache.cache_path` rather than holding all eight simultaneously -- and records
    in both the JSONL rows and the printed summary that the fallback was taken. Never silently
    changes `SIGMA_GRAM_DTYPE`.
    """
    if not args.record_path:
        print(
            "ERROR: --mode sigma requires --record-path -- this is a PRE-FREEZE MEASUREMENT, "
            "not a deliverable, and refuses to default onto any frozen record path.",
            file=sys.stderr,
        )
        sys.exit(1)
    record_path = resolve_record_path(args.record_path)

    available_mb = _available_memory_mb()
    disk_fallback_taken = available_mb >= 0.0 and available_mb < FALLBACK_THRESHOLD_MB

    print(
        f"\n{'=' * 78}\n"
        "THIS IS A PRE-FREEZE MEASUREMENT, NOT A DELIVERABLE. Every Phase 8 gating constant in "
        "cka.py is UNSET (see cka.assert_preregistered). The two sigma values below are "
        "PRE-REGISTRATION INPUTS under D8-03 -- 08-04 freezes them as SIGMA_HSC / "
        f"SIGMA_LEGACYSURVEY, computed once over all 10,000 points, before any subset exists. "
        f"This mode computes no CKA value and constructs no subset. Writing to {record_path}.\n"
        f"available_memory_mb={available_mb:.1f} fallback_threshold_mb={FALLBACK_THRESHOLD_MB:.1f} "
        f"disk_fallback_taken={disk_fallback_taken}\n"
        f"{'=' * 78}\n"
    )

    X_hsc, X_ls, subsample_path = load_pu_pair("hsc", "legacysurvey")
    n_points, n_features = X_hsc.shape

    sigma_hsc = cka.median_pairwise_distance(X_hsc)
    sigma_ls = cka.median_pairwise_distance(X_ls)
    print(f"sigma_median_pairwise modality=hsc value={sigma_hsc!r}")
    print(f"sigma_median_pairwise modality=legacysurvey value={sigma_ls!r}")

    sigma_by_modality = {"hsc": sigma_hsc, "legacysurvey": sigma_ls}
    build_records = []

    def _on_build(modality: str, kernel: str, wallclock_s: float) -> None:
        print(
            f"gram_build modality={modality} kernel={kernel} wallclock_s={wallclock_s:.6f} "
            f"dtype={np.dtype(SIGMA_GRAM_DTYPE).name}"
        )
        build_records.append((modality, kernel, wallclock_s))

    if disk_fallback_taken:
        # One rung at a time: build, save to notebooks/.cache/ as .npy, then release the array
        # before building the next one, so peak RSS never holds more than one (10000, 10000)
        # matrix at once. SIGMA_GRAM_DTYPE is unchanged either way.
        def _build_and_release(modality_name: str, X: np.ndarray, sigma: float) -> None:
            kernel_and_builder = [("linear", lambda: cka.linear_gram(X, SIGMA_GRAM_DTYPE))]
            for multiplier in SIGMA_MULTIPLIERS:
                name = _rbf_kernel_name(multiplier)
                kernel_and_builder.append(
                    (name, lambda m=multiplier: cka.rbf_gram(X, sigma * m, SIGMA_GRAM_DTYPE))
                )
            for kernel_name, builder in kernel_and_builder:
                t0 = time.monotonic()
                matrix = builder()
                wallclock_s = time.monotonic() - t0
                _on_build(modality_name, kernel_name, wallclock_s)
                npy_path = cache.cache_path(
                    f"08_scratch_gram_{modality_name}_{kernel_name}", "npy"
                )
                np.save(npy_path, matrix)
                del matrix

        _build_and_release("hsc", X_hsc, sigma_hsc)
        _build_and_release("legacysurvey", X_ls, sigma_ls)
    else:
        build_gram_matrices(
            X_hsc, X_ls, sigma_hsc, sigma_ls, SIGMA_MULTIPLIERS, SIGMA_GRAM_DTYPE,
            on_build=_on_build,
        )

    peak_rss_mb = _peak_rss_mb()
    print(f"peak_rss_mb={peak_rss_mb:.2f}")

    timestamp = datetime.now(timezone.utc).isoformat()
    for modality, kernel, wallclock_s in build_records:
        row = {
            "mode": "sigma",
            "modality": modality,
            "sigma_median_pairwise": float(sigma_by_modality[modality]),
            "n_points": int(n_points),
            "n_features": int(n_features),
            "subsample_file": str(subsample_path),
            "gram_kernel": kernel,
            "gram_build_s": float(wallclock_s),
            "gram_dtype": np.dtype(SIGMA_GRAM_DTYPE).name,
            "peak_rss_mb": float(peak_rss_mb),
            "disk_fallback_taken": bool(disk_fallback_taken),
            "timestamp": timestamp,
        }
        append_record_row(row, record_path)

    print("SIGMA MEASUREMENT COMPLETE (pre-freeze; no verdict, no CKA value, no subset)")
    return True


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


# =============================================================================================
# 08-05: the three production modes (D8-09/13/15/18/19). Every one of these calls
# cka.assert_preregistered() then _strict_ancestor_or_exit BEFORE this point (main()'s dispatch,
# below) -- no function in this section re-checks the freeze, matching 07.1's own runner
# discipline (the gate is checked once, at the CLI dispatch boundary).
#
# MODALITY_A ("hsc") is held fixed throughout -- its Gram matrices are built once and never
# rebuilt. MODALITY_B ("legacysurvey") is the modality D8-18's planted-effect ladder degrades;
# it is also the modality D8-06/D8-07's density field and every D8-14 curvature field are
# computed over (DENSITY_INPUT = "legacysurvey_ambient_768"), so a single load_pu_pair() call
# and a single compute_density() call serve every mode below.
# =============================================================================================

MODALITY_A = "hsc"
MODALITY_B = "legacysurvey"

# D8-14: which `d` (and, for the seed axis, which TORCH_INIT_SEED) each frozen field name
# belongs to -- re-declared here as a runner-local literal (never crossing the freeze boundary
# into cka.py, which declares no per-field-name mapping of its own).
FIELD_D_AND_SEED: Dict[str, Tuple[int, Optional[int]]] = {
    "h_norm_20": (20, None),
    "h_norm_25": (25, None),
    "h_norm_32": (32, None),
    "h_norm_25_seed0": (25, 0),
    "h_norm_25_seed1": (25, 1),
    "h_norm_25_seed2": (25, 2),
}


def compute_density(X_ls: np.ndarray) -> np.ndarray:
    """D8-07: the relative density `1.0 / w` (`DENSITY_SIGN_CONVENTION`), computed ONCE per
    process on the `legacysurvey` ambient cloud via `curvature_probe.local_density_weights` at
    `cka.DENSITY_K` / `cka.DENSITY_FIELD_D`, and reused across every `d`, seed and `S` -- density
    is a property of the ambient cloud and is `d`-independent (mirrors
    `07.1_density_stratified_null_run.py`'s own `recompute_mknn_and_density`). Prints p05/p50/p95,
    matching 07.1's own printed diagnostic."""
    w = curvature_probe.local_density_weights(X_ls, cka.DENSITY_K, cka.DENSITY_FIELD_D)
    density = 1.0 / w
    print(
        f"[compute_density] DENSITY_K={cka.DENSITY_K}  DENSITY_FIELD_D={cka.DENSITY_FIELD_D}  "
        f"p05={np.percentile(density, 5):.4e}  p50={np.percentile(density, 50):.4e}  "
        f"p95={np.percentile(density, 95):.4e}"
    )
    return density


def _sigma_multiplier_for_kernel_name(kernel_name: str) -> Optional[float]:
    """Inverse of `_rbf_kernel_name`: `"rbf_sigma"` -> `1.0`, `"rbf_0.5sigma"` -> `0.5`,
    `"rbf_2sigma"` -> `2.0`; `"linear"` (or any name not produced by `_rbf_kernel_name`) -> `None`
    -- D8-04's sigma ladder has no multiplier for the linear kernel."""
    for multiplier in cka.SIGMA_MULTIPLIERS:
        if _rbf_kernel_name(multiplier) == kernel_name:
            return float(multiplier)
    return None


def run_cell(
    h: np.ndarray,
    density: np.ndarray,
    K_full: Dict[str, np.ndarray],
    L_full: Dict[str, np.ndarray],
    s_strata: int,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    """The unit every production mode reuses (D8-06/07/08/09/10/11): builds `density_strata(
    density, s_strata)`, splits with `cka.tertile_split_within_strata`, computes
    `cka.tertile_gap_panel` for every kernel present in `K_full`/`L_full` (all four kernel
    variants in production), computes `cka.realized_h_contrast`, runs
    `cka.stratified_tertile_label_null` over `cka.NULL_KERNELS` ONLY, and derives the two-tailed
    thresholds via `cka.null_threshold(..., cka.NULL_QUANTILE_PER_TAIL)`.

    Never rebuilds a Gram matrix and never recomputes a bandwidth -- every kernel value comes
    from indexing into the caller's already-built `K_full`/`L_full` (via `cka.tertile_gap_panel`
    / `cka.stratified_tertile_label_null`'s own `cka_on_subset` calls).

    Returns a dict: `per_kernel` (one flat dict of plain Python scalars per kernel name in
    `K_full`, carrying `cka_t1`/`cka_t2`/`cka_t3`/`gap`, plus `null_lo`/`null_hi`/`cleared` for
    kernels in `cka.NULL_KERNELS` and `None` for the two non-null diagnostic rungs),
    `realized_h_contrast`, and the three subset sizes `n_t1`/`n_t2`/`n_t3`.
    """
    strata = dsn.density_strata(density, s_strata)
    tertiles = cka.tertile_split_within_strata(h, strata)
    panel = cka.tertile_gap_panel(K_full, L_full, tertiles)
    contrast = cka.realized_h_contrast(h, tertiles)
    n_t1, n_t2, n_t3 = (int(t.shape[0]) for t in tertiles)

    K_null = {name: K_full[name] for name in cka.NULL_KERNELS}
    L_null = {name: L_full[name] for name in cka.NULL_KERNELS}
    null_by_kernel = cka.stratified_tertile_label_null(
        h, strata, K_null, L_null, n_permutations, seed
    )

    per_kernel: Dict[str, Dict[str, Any]] = {}
    for name, values in panel.items():
        row: Dict[str, Any] = {
            "cka_t1": float(values["cka_t1"]),
            "cka_t2": float(values["cka_t2"]),
            "cka_t3": float(values["cka_t3"]),
            "gap": float(values["gap"]),
            "null_lo": None,
            "null_hi": None,
            "cleared": None,
        }
        if name in cka.NULL_KERNELS:
            null_lo, null_hi = cka.null_threshold(
                null_by_kernel[name], cka.NULL_QUANTILE_PER_TAIL
            )
            gap = values["gap"]
            row["null_lo"] = float(null_lo)
            row["null_hi"] = float(null_hi)
            row["cleared"] = bool(gap > null_hi or gap < null_lo)
        per_kernel[name] = row

    return {
        "per_kernel": per_kernel,
        "realized_h_contrast": float(contrast),
        "n_t1": n_t1,
        "n_t2": n_t2,
        "n_t3": n_t3,
    }


def plant_alignment_degradation(
    X_b: np.ndarray, tertile_hi_idx: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """D8-18's graded alignment-degradation injection: returns a COPY of modality-B's matrix in
    which a `magnitude` fraction of the rows indexed by `tertile_hi_idx` (the high-``||H||``
    tertile) have had their crossmodal pairing destroyed by permuting those rows AMONG
    THEMSELVES. Preserves modality-B's marginal distribution exactly (every value present before
    is present after, just reassigned among the chosen rows) and preserves every subset size, so
    the injected effect is an alignment degradation, never a distributional one.

    `magnitude=0.0` selects zero rows and returns a byte-identical copy -- the no-injection
    anchor (`PLANTED_EFFECT_GRID`'s first rung) that must NOT clear the null.
    """
    X_degraded = np.array(X_b, copy=True)
    tertile_hi_idx = np.asarray(tertile_hi_idx)
    n_hi = tertile_hi_idx.shape[0]
    n_destroy = int(round(magnitude * n_hi))
    if n_destroy >= 2:
        destroy_idx = rng.choice(tertile_hi_idx, size=n_destroy, replace=False)
        permuted = rng.permutation(destroy_idx)
        X_degraded[destroy_idx] = X_b[permuted]
    return X_degraded


def run_positive_control(args: argparse.Namespace) -> bool:
    """D8-18's planted-effect ladder on real PU geometry (Task 1): keeps PU's actual ``||H||``
    field, actual density strata and actual subset sizes throughout; only the crossmodal pairing
    of a `PLANTED_EFFECT_GRID` fraction of the high-``||H||`` tertile's rows in modality B
    (`legacysurvey`) is destroyed. Modality A's (`hsc`) Gram matrices are built once and never
    rebuilt; modality B's Gram matrices are rebuilt from the degraded embeddings at every
    (`S`, magnitude) cell -- the density/`||H||`-based tertile split itself never changes, since
    it depends only on `h` and `density`, both computed once, before any injection.

    Field: `cka.NEGATIVE_CONTROL_FIELD` (`"h_norm_25"`) -- the same field the negative control
    uses, so the two controls are read against one another without a field confound.

    Reports, per `S`, the smallest magnitude (of the kernels in `cka.NULL_KERNELS`) whose gap
    clears the null -- `detection_floor`, a power curve, not a single pass/fail. Prints the
    decision rule BEFORE the first measured number: the `magnitude=0.0` rung is the no-injection
    anchor and must NOT clear; if it does, this prints `POSITIVE CONTROL INVALID` (still exits 0)
    rather than silently working around it.
    """
    record_path = resolve_record_path(args.record_path)
    preregistration_commit = _git_rev_parse(args.freeze_commit)

    print(
        f"\n{'=' * 78}\n"
        "D8-18 POSITIVE CONTROL -- DECISION RULE (stated before any measured number, per the "
        "estimator-validation protocol):\n"
        "  1. The magnitude=0.0 rung is the NO-INJECTION ANCHOR and must NOT clear the null at "
        "any S. A 0.0 rung that clears invalidates this control -- it will be reported as "
        "POSITIVE CONTROL INVALID, not worked around.\n"
        "  2. For every other magnitude, the smallest magnitude whose gap clears its null is "
        "reported as detection_floor, per S, per kernel in NULL_KERNELS -- a power curve, not a "
        "single pass/fail.\n"
        f"  PLANTED_EFFECT_GRID={cka.PLANTED_EFFECT_GRID}  S_GRID={cka.S_GRID}  "
        f"NULL_KERNELS={cka.NULL_KERNELS}  field={cka.NEGATIVE_CONTROL_FIELD!r}\n"
        f"{'=' * 78}\n"
    )

    X_hsc, X_ls, subsample_path = load_pu_pair(MODALITY_A, MODALITY_B)
    fields = load_frozen_fields()
    h = fields[cka.NEGATIVE_CONTROL_FIELD]
    density = compute_density(X_ls)

    # Modality A's own four kernel variants only -- built directly (not via build_gram_matrices,
    # which always builds BOTH modalities together) so this mode never pays for a modality-B
    # Gram build it would immediately discard. Modality A's Gram matrices never change for the
    # rest of this function; only modality B's are rebuilt, once per (S, magnitude) cell, from
    # the degraded embeddings.
    grams_a: Dict[str, np.ndarray] = {"linear": cka.linear_gram(X_hsc, np.dtype(cka.GRAM_DTYPE))}
    for multiplier in cka.SIGMA_MULTIPLIERS:
        grams_a[_rbf_kernel_name(multiplier)] = cka.rbf_gram(
            X_hsc, cka.SIGMA_HSC * multiplier, np.dtype(cka.GRAM_DTYPE)
        )
    sigma_b = cka.SIGMA_LEGACYSURVEY

    rng = np.random.default_rng(cka.PLANTED_EFFECT_SEED)

    all_valid = True
    detection_floor_by_s: Dict[int, Dict[str, Optional[float]]] = {}
    t_start_all = time.monotonic()

    for s_strata in cka.S_GRID:
        strata = dsn.density_strata(density, s_strata)
        tertiles = cka.tertile_split_within_strata(h, strata)
        tertile_hi_idx = tertiles[2]

        cleared_at: Dict[str, Optional[float]] = {name: None for name in cka.NULL_KERNELS}
        for magnitude in cka.PLANTED_EFFECT_GRID:
            t_start = time.monotonic()
            X_b_degraded = plant_alignment_degradation(X_ls, tertile_hi_idx, magnitude, rng)
            grams_b_degraded = {
                "linear": cka.linear_gram(X_b_degraded, np.dtype(cka.GRAM_DTYPE)),
            }
            for multiplier in cka.SIGMA_MULTIPLIERS:
                grams_b_degraded[_rbf_kernel_name(multiplier)] = cka.rbf_gram(
                    X_b_degraded, sigma_b * multiplier, np.dtype(cka.GRAM_DTYPE)
                )

            cell = run_cell(
                h, density, grams_a, grams_b_degraded, s_strata, cka.N_PERMUTATIONS,
                cka.PERMUTATION_SEED,
            )
            wallclock_s = time.monotonic() - t_start
            timestamp = datetime.now(timezone.utc).isoformat()

            for kernel_name, kernel_row in cell["per_kernel"].items():
                cleared = kernel_row["cleared"]
                if magnitude == 0.0 and cleared:
                    all_valid = False
                if (
                    kernel_name in cka.NULL_KERNELS
                    and cleared
                    and cleared_at[kernel_name] is None
                ):
                    cleared_at[kernel_name] = float(magnitude)
                row = {
                    "mode": "positive-control",
                    "field": cka.NEGATIVE_CONTROL_FIELD,
                    "s_strata": int(s_strata),
                    "kernel": kernel_name,
                    "sigma_multiplier": _sigma_multiplier_for_kernel_name(kernel_name),
                    "planted_magnitude": float(magnitude),
                    "cka_t1": kernel_row["cka_t1"],
                    "cka_t2": kernel_row["cka_t2"],
                    "cka_t3": kernel_row["cka_t3"],
                    "gap": kernel_row["gap"],
                    "null_lo": kernel_row["null_lo"],
                    "null_hi": kernel_row["null_hi"],
                    "cleared": kernel_row["cleared"],
                    "n_t1": cell["n_t1"],
                    "n_t2": cell["n_t2"],
                    "n_t3": cell["n_t3"],
                    "realized_h_contrast": cell["realized_h_contrast"],
                    "n_permutations": int(cka.N_PERMUTATIONS),
                    "permutation_seed": int(cka.PERMUTATION_SEED),
                    "preregistration_commit": preregistration_commit,
                    "wallclock_s": float(wallclock_s),
                    "timestamp": timestamp,
                }
                append_record_row(row, record_path)

            print(
                f"[positive-control] S={s_strata} magnitude={magnitude} wallclock_s="
                f"{wallclock_s:.2f} elapsed_total_min={(time.monotonic() - t_start_all) / 60.0:.1f}"
            )

        detection_floor_by_s[s_strata] = cleared_at
        for kernel_name, floor in cleared_at.items():
            print(
                f"detection_floor S={s_strata} kernel={kernel_name} value={floor} "
                f"({'DETECTED within grid' if floor is not None else 'NOT DETECTED within PLANTED_EFFECT_GRID'})"
            )
            append_record_row(
                {
                    "mode": "positive-control",
                    "row_kind": "detection_floor_summary",
                    "field": cka.NEGATIVE_CONTROL_FIELD,
                    "s_strata": int(s_strata),
                    "kernel": kernel_name,
                    "detection_floor": floor,
                    "preregistration_commit": preregistration_commit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                record_path,
            )

    if not all_valid:
        print(
            "\nPOSITIVE CONTROL INVALID: the magnitude=0.0 no-injection anchor cleared its null "
            "at at least one (S, kernel) cell. This invalidates the detection-floor reading above "
            "-- reported as such, not worked around.\n"
        )
    else:
        print("\nPOSITIVE CONTROL: the magnitude=0.0 anchor did not clear at any (S, kernel) cell.\n")

    print("POSITIVE CONTROL COMPLETE")
    return True


def shuffle_h_field(h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """D8-19's negative control: a GLOBAL permutation of the ``||H||`` field `h` -- the marginal
    preserved exactly, point correspondence destroyed entirely. Deliberately NOT a within-stratum
    shuffle (that would leave the density-conditioning link partially intact): the point is to
    break the curvature-to-point link completely and see whether the split-plus-null machinery
    still manufactures a gap."""
    return rng.permutation(np.asarray(h, dtype=np.float64))


def run_negative_control(args: argparse.Namespace) -> bool:
    """D8-19's shuffled-``||H||`` end-to-end calibration run (Task 2): for each `S` in
    `cka.S_GRID`, runs `cka.N_REPEATS` independent repeats. Each repeat draws a global shuffle of
    `cka.NEGATIVE_CONTROL_FIELD` from a single RNG seeded at `cka.PERMUTATION_SEED`, then runs the
    ENTIRE pipeline (`run_cell` -- within-stratum splitting AND the full `cka.N_PERMUTATIONS`-draw
    permutation null) exactly as the sweep will. Reports, per `S` per kernel in
    `cka.NULL_KERNELS`, the fraction of repeats whose gap cleared its null (`false_positive_rate`)
    beside the nominal rate implied by `cka.NULL_QUANTILE_PER_TAIL`.

    `cka.N_REPEATS`, `cka.N_PERMUTATIONS` and `cka.S_GRID` are all frozen constants read directly
    off `cka` -- there is no CLI-flag override of N_REPEATS anywhere in this function,
    because reducing any of them after seeing an inconvenient rate is exactly the post-hoc move
    D8-22 exists to prevent.
    """
    record_path = resolve_record_path(args.record_path)
    preregistration_commit = _git_rev_parse(args.freeze_commit)

    X_hsc, X_ls, subsample_path = load_pu_pair(MODALITY_A, MODALITY_B)
    fields = load_frozen_fields()
    h = fields[cka.NEGATIVE_CONTROL_FIELD]
    density = compute_density(X_ls)

    grams_a, grams_b = build_gram_matrices(
        X_hsc, X_ls, cka.SIGMA_HSC, cka.SIGMA_LEGACYSURVEY, cka.SIGMA_MULTIPLIERS,
        np.dtype(cka.GRAM_DTYPE),
    )

    rng = np.random.default_rng(cka.PERMUTATION_SEED)
    nominal_rate = 2.0 * (1.0 - cka.NULL_QUANTILE_PER_TAIL)

    print(
        f"\n{'=' * 78}\n"
        "D8-19 NEGATIVE CONTROL: cka.N_REPEATS repeats of the ENTIRE pipeline (within-stratum "
        "split + full cka.N_PERMUTATIONS-draw null) per S in cka.S_GRID -- "
        f"{cka.N_REPEATS} x {len(cka.S_GRID)} = {cka.N_REPEATS * len(cka.S_GRID)} full null "
        "computations total. Per RESEARCH.md's cost model this was estimated at 1-2 hours; a "
        "measured empirical timing during this plan's own execution (see 08-05-SUMMARY.md) may "
        "differ substantially. N_REPEATS/N_PERMUTATIONS/S_GRID are frozen constants and are never "
        "reduced here to finish faster -- if this run must be interrupted, the completed S values "
        "are recorded and the remainder is reported as not-run, never extrapolated.\n"
        f"nominal_two_tailed_false_positive_rate={nominal_rate}\n"
        f"{'=' * 78}\n"
    )

    for s_strata in cka.S_GRID:
        cleared_counts: Dict[str, int] = {name: 0 for name in cka.NULL_KERNELS}
        t_start_s = time.monotonic()
        for repeat_index in range(cka.N_REPEATS):
            h_shuffled = shuffle_h_field(h, rng)
            cell = run_cell(
                h_shuffled, density, grams_a, grams_b, s_strata, cka.N_PERMUTATIONS,
                cka.PERMUTATION_SEED,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            for kernel_name in cka.NULL_KERNELS:
                kernel_row = cell["per_kernel"][kernel_name]
                if kernel_row["cleared"]:
                    cleared_counts[kernel_name] += 1
                append_record_row(
                    {
                        "mode": "negative-control",
                        "field": cka.NEGATIVE_CONTROL_FIELD,
                        "s_strata": int(s_strata),
                        "kernel": kernel_name,
                        "repeat_index": int(repeat_index),
                        "cka_t1": kernel_row["cka_t1"],
                        "cka_t2": kernel_row["cka_t2"],
                        "cka_t3": kernel_row["cka_t3"],
                        "gap": kernel_row["gap"],
                        "null_lo": kernel_row["null_lo"],
                        "null_hi": kernel_row["null_hi"],
                        "cleared": kernel_row["cleared"],
                        "n_t1": cell["n_t1"],
                        "n_t2": cell["n_t2"],
                        "n_t3": cell["n_t3"],
                        "realized_h_contrast": cell["realized_h_contrast"],
                        "n_permutations": int(cka.N_PERMUTATIONS),
                        "permutation_seed": int(cka.PERMUTATION_SEED),
                        "preregistration_commit": preregistration_commit,
                        "timestamp": timestamp,
                    },
                    record_path,
                )
            elapsed_s_min = (time.monotonic() - t_start_s) / 60.0
            print(
                f"[negative-control] S={s_strata} repeat={repeat_index + 1}/{cka.N_REPEATS} "
                f"elapsed_this_S_min={elapsed_s_min:.1f}"
            )

        wallclock_s_min = (time.monotonic() - t_start_s) / 60.0
        for kernel_name in cka.NULL_KERNELS:
            rate = cleared_counts[kernel_name] / float(cka.N_REPEATS)
            print(
                f"false_positive_rate S={s_strata} kernel={kernel_name} rate={rate} "
                f"nominal={nominal_rate} n_repeats={cka.N_REPEATS} wallclock_min={wallclock_s_min:.1f}"
            )
            append_record_row(
                {
                    "mode": "negative-control",
                    "row_kind": "summary",
                    "field": cka.NEGATIVE_CONTROL_FIELD,
                    "s_strata": int(s_strata),
                    "kernel": kernel_name,
                    "false_positive_rate": float(rate),
                    "nominal_rate": float(nominal_rate),
                    "n_repeats": int(cka.N_REPEATS),
                    "preregistration_commit": preregistration_commit,
                    "wallclock_s": float(wallclock_s_min * 60.0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                record_path,
            )

    print("NEGATIVE CONTROL COMPLETE")
    return True


def run_sweep(args: argparse.Namespace) -> bool:
    """D8-09/13/15's 18-cell sweep (Task 3): six fields (`h_norm_20`, `h_norm_25`, `h_norm_32`,
    `h_norm_25_seed0/1/2`) crossed with `cka.S_GRID`. Reports the full four-kernel panel per cell;
    verdicts are read off the LINEAR kernel (D8-01's headline). `per_d_verdict` is called with
    ONLY each `d`'s own per-`S` gaps/thresholds -- a null at one `d` never voids another and no
    pooled headline is invented (D8-13). The `d=25` seed axis calls `per_d_verdict` once per seed
    on that seed's own three-`S` results, then combines the three with `cka.combine_seed_verdicts`
    -- the three seed fields are never pooled into one (D8-15), proven at the top of this function
    by a self-check that `cka.pooled_field_guard` refuses a 3-field pooling attempt.
    """
    record_path = resolve_record_path(args.record_path)
    preregistration_commit = _git_rev_parse(args.freeze_commit)

    seed_field_names = tuple(
        name for name, (d, seed) in FIELD_D_AND_SEED.items() if seed is not None
    )
    try:
        cka.pooled_field_guard(seed_field_names)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "run_sweep: cka.pooled_field_guard did not raise for a 3-field pooling attempt -- "
            "the never-pool-seeds guard (05-03-DECISION.md, D8-15) is not doing its job."
        )
    print(
        "[pooled_field_guard] confirmed: pooling the three d=25 seed fields into one raises "
        "RuntimeError -- proceeding with each seed's OWN within-stratum split and OWN verdict, "
        "never pooled into one (D8-15).\n"
    )

    X_hsc, X_ls, subsample_path = load_pu_pair(MODALITY_A, MODALITY_B)
    fields = load_frozen_fields()
    density = compute_density(X_ls)

    grams_a, grams_b = build_gram_matrices(
        X_hsc, X_ls, cka.SIGMA_HSC, cka.SIGMA_LEGACYSURVEY, cka.SIGMA_MULTIPLIERS,
        np.dtype(cka.GRAM_DTYPE),
    )

    gaps_by_field_s: Dict[str, Dict[int, float]] = {}
    thresholds_by_field_s: Dict[str, Dict[int, Tuple[float, float]]] = {}

    t_start_all = time.monotonic()
    for field_name, (d_value, seed_value) in FIELD_D_AND_SEED.items():
        h = fields[field_name]
        gaps_by_s: Dict[int, float] = {}
        thresholds_by_s: Dict[int, Tuple[float, float]] = {}
        for s_strata in cka.S_GRID:
            t_start = time.monotonic()
            cell = run_cell(
                h, density, grams_a, grams_b, s_strata, cka.N_PERMUTATIONS, cka.PERMUTATION_SEED
            )
            wallclock_s = time.monotonic() - t_start
            timestamp = datetime.now(timezone.utc).isoformat()

            for kernel_name, kernel_row in cell["per_kernel"].items():
                append_record_row(
                    {
                        "mode": "sweep",
                        "field": field_name,
                        "d": int(d_value),
                        "seed": None if seed_value is None else int(seed_value),
                        "s_strata": int(s_strata),
                        "kernel": kernel_name,
                        "sigma_multiplier": _sigma_multiplier_for_kernel_name(kernel_name),
                        "cka_t1": kernel_row["cka_t1"],
                        "cka_t2": kernel_row["cka_t2"],
                        "cka_t3": kernel_row["cka_t3"],
                        "gap": kernel_row["gap"],
                        "null_lo": kernel_row["null_lo"],
                        "null_hi": kernel_row["null_hi"],
                        "cleared": kernel_row["cleared"],
                        "n_t1": cell["n_t1"],
                        "n_t2": cell["n_t2"],
                        "n_t3": cell["n_t3"],
                        "realized_h_contrast": cell["realized_h_contrast"],
                        "n_permutations": int(cka.N_PERMUTATIONS),
                        "permutation_seed": int(cka.PERMUTATION_SEED),
                        "preregistration_commit": preregistration_commit,
                        "wallclock_s": float(wallclock_s),
                        "timestamp": timestamp,
                    },
                    record_path,
                )

            gaps_by_s[s_strata] = cell["per_kernel"]["linear"]["gap"]
            thresholds_by_s[s_strata] = (
                cell["per_kernel"]["linear"]["null_lo"], cell["per_kernel"]["linear"]["null_hi"]
            )
            print(
                f"[sweep] field={field_name} d={d_value} seed={seed_value} S={s_strata} "
                f"gap={gaps_by_s[s_strata]:.6f} realized_h_contrast="
                f"{cell['realized_h_contrast']:.6f} wallclock_s={wallclock_s:.2f} "
                f"elapsed_total_min={(time.monotonic() - t_start_all) / 60.0:.1f}"
            )

        gaps_by_field_s[field_name] = gaps_by_s
        thresholds_by_field_s[field_name] = thresholds_by_s

    # --- per-d verdicts, independent (D8-13) -------------------------------------------------
    per_d_verdicts: Dict[int, Dict[str, Any]] = {}
    print("\n" + "=" * 78 + "\nPER-D TABLE (07.1's own per-d clearance table shape)\n" + "=" * 78)
    for field_name, d_value in (("h_norm_20", 20), ("h_norm_25", 25), ("h_norm_32", 32)):
        verdict = cka.per_d_verdict(
            gaps_by_field_s[field_name], thresholds_by_field_s[field_name], cka.VERDICT_RULE
        )
        per_d_verdicts[d_value] = verdict
        s_clearances = {s: verdict["per_s"][s]["clears"] for s in cka.S_GRID}
        print(f"d={d_value:<3} verdict={verdict['verdict']:<20} per_S_clears={s_clearances}")
        append_record_row(
            {
                "mode": "sweep",
                "row_kind": "per_d_verdict",
                "field": field_name,
                "d": int(d_value),
                "per_d_verdict": verdict["verdict"],
                "n_s_cleared": verdict["n_s_cleared"],
                "per_s": {
                    str(s): {
                        "gap": entry["gap"], "null_low": entry["null_low"],
                        "null_high": entry["null_high"], "clears": entry["clears"],
                    }
                    for s, entry in verdict["per_s"].items()
                },
                "preregistration_commit": preregistration_commit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            record_path,
        )

    d32_gaps = ", ".join(f"S={s}:{gaps_by_field_s['h_norm_32'][s]:.6f}" for s in cka.S_GRID)
    print(f"\nd=32 GAP (D8-12 + D8-21, printed prominently, not only as a table row): {d32_gaps}\n")

    # --- d=25 seed axis: per-seed verdicts, unanimous-or-nothing combination (D8-15) ---------
    per_seed_verdicts: Dict[int, str] = {}
    for field_name, seed_value in (
        ("h_norm_25_seed0", 0), ("h_norm_25_seed1", 1), ("h_norm_25_seed2", 2)
    ):
        verdict = cka.per_d_verdict(
            gaps_by_field_s[field_name], thresholds_by_field_s[field_name], cka.VERDICT_RULE
        )
        per_seed_verdicts[seed_value] = verdict["verdict"]
        print(f"seed={seed_value} verdict={verdict['verdict']}")
        append_record_row(
            {
                "mode": "sweep",
                "row_kind": "per_d_verdict",
                "field": field_name,
                "d": 25,
                "seed": int(seed_value),
                "per_d_verdict": verdict["verdict"],
                "n_s_cleared": verdict["n_s_cleared"],
                "preregistration_commit": preregistration_commit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            record_path,
        )

    seed_combined = cka.combine_seed_verdicts(per_seed_verdicts, cka.SEED_VERDICT_COMBINATION_RULE)
    print(f"seed_combined_verdict={seed_combined['phase_verdict']}  (n_cleared={seed_combined['n_cleared']}/3)")
    append_record_row(
        {
            "mode": "sweep",
            "row_kind": "seed_combined_verdict",
            "seed_combined_verdict": seed_combined["phase_verdict"],
            "n_cleared": seed_combined["n_cleared"],
            "n_seeds": seed_combined["n_seeds"],
            "per_seed_verdicts": {str(k): v for k, v in seed_combined["per_seed_verdicts"].items()},
            "preregistration_commit": preregistration_commit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        record_path,
    )

    print("SWEEP COMPLETE")
    return True


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
        # D8-22: every production mode calls cka.assert_preregistered() FIRST (refusing to run
        # against an UNSET or drifted constant) and the strict-ancestor gate SECOND -- so no
        # number can be produced by a tree whose constants drifted or whose freeze proof is
        # missing. Both pre-flight checks now pass against the real 08-04 freeze commit.
        cka.assert_preregistered()
        _strict_ancestor_or_exit(args.freeze_commit)

        if args.mode in NOT_YET_IMPLEMENTED_MODES:
            print(
                f"ERROR: --mode {args.mode} is not implemented yet; it lands in "
                f"plan {NOT_YET_IMPLEMENTED_MODES[args.mode]}.",
                file=sys.stderr,
            )
            sys.exit(2)

        if args.mode == "positive-control":
            ok = run_positive_control(args)
            sys.exit(0 if ok else 1)

        if args.mode == "negative-control":
            ok = run_negative_control(args)
            sys.exit(0 if ok else 1)

        if args.mode == "sweep":
            ok = run_sweep(args)
            sys.exit(0 if ok else 1)

    if args.mode == "selfcheck":
        ok = run_selfcheck(args)
        sys.exit(0 if ok else 1)

    if args.mode == "sigma":
        ok = run_sigma(args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
