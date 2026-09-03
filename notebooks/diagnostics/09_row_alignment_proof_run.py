"""Phase 9 row-alignment proof runner: the D9-06/07 alignment machinery
(`physics_labels.alignment_r2_curve`/`alignment_verdict`) as a CLI, built in 09-03 and RUN for
real on the execution host in 09-07.

`--mode smoke` -- the pre-freeze exercise: the whole proof on synthetic arrays with a known
injected row offset, no HuggingFace read, no `--freeze-commit` required. Exercises both the PASS
branch (aligned case, argmax at shift 0) and the branch D9-08's SEARCH mode reads (offset case,
single clearing alignment at the injected offset).

`--mode manifest` -- reads both real datasets at full scale and writes dataset metadata ONLY:
row counts, widths, dtypes, per-column finite counts and sentinel counts. Computes no
correlation, no R2, no curvature. Takes every value from the command line, never from the
frozen constants in `physics_labels.py`/`physics_curvature_probe.py`, because it runs BEFORE the
09-05 freeze and produces the evidence 09-04's blocking checkpoint reads to decide them.

`--mode proof` -- D9-06/D9-07 for real: the out-of-fold R2 curve at shift 0 and at every entry
of `physics_labels.ALIGNMENT_SHIFT_SET`, plus `ALIGNMENT_N_PERMUTATIONS` seeded permutations,
then `alignment_verdict` at `ALIGNMENT_MARGIN_R2`. Requires `--freeze-commit` and both frozen
modules' `assert_preregistered()` to pass.

`--mode search` -- D9-08's branch: reads the proof record's verdict, and when shift 0 FAILED
reports every alignment that clears the margin, classifying exactly one as a CANDIDATE OFFSET,
two or more as AMBIGUOUS (halt), zero as NO ALIGNMENT FOUND (halt). Never adopts an offset by
itself -- adoption is 09-07's blocking developer decision plus a fresh freeze.

Usage:
    python notebooks/diagnostics/09_row_alignment_proof_run.py --mode smoke --record-path notebooks/.cache/09_scratch_alignment.jsonl
    python notebooks/diagnostics/09_row_alignment_proof_run.py --mode manifest --candidate-columns mag_r_desi mag_r photo_z
    python notebooks/diagnostics/09_row_alignment_proof_run.py --mode proof --freeze-commit <sha>
    python notebooks/diagnostics/09_row_alignment_proof_run.py --mode search --freeze-commit <sha>
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` and `--flag=value` -- or `None` if `flag` was not passed in either
    form. Copied verbatim from `09_physics_curvature_run.py` (itself copied from
    `07_crossmodal_curvature_run.py`, CR-03). Kept dependency-free so it can run here, above the
    numpy import."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in numpy (07_crossmodal_curvature_run.py
# precedent: concurrent jobs measured driving load up ~10x). This runner never trains a model,
# but the discipline is cheap and keeps every Phase 9 runner behaviourally identical.
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
import hashlib
import io
import json
import platform
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402  (only for _describe_environment's version string -- this runner
                             # never trains or evaluates a model)

# Imported only so _describe_environment() can report their installed versions (T-09-41) --
# every one of these is already a project dependency, imported elsewhere transitively.
import datasets  # noqa: E402
import pandas  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from pu_manifold import cache  # noqa: E402  (imported for parity with sibling runners; not
                                              # called directly -- pcp.resolve_output_root owns
                                              # the fallback to cache.CACHE_DIR)
from pu_manifold import linear_probe  # noqa: E402  (imported so an import-purity check can see
                                                     # this runner's full dependency surface;
                                                     # not called directly -- pcp.oof_ridge_
                                                     # predictions wraps it)
from pu_manifold import subsample  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

# FREEZE_COMMIT_SHA wired to plan 09-05 Task 1's freeze commit (D9-18): the commit that filled
# every gating constant in physics_labels.py and physics_curvature_probe.py. _strict_ancestor_or_
# exit now also enforces the exact-equality check below -- a --freeze-commit resolving to any
# other genuine ancestor of HEAD is rejected (CR-01).
FREEZE_COMMIT_SHA = "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5"

# JSONL record stems this runner writes. --mode manifest and --mode proof/search each own one;
# --mode smoke must never default onto either (it always requires an explicit --record-path,
# see resolve_record_path).
RECORD_STEM = {
    "manifest": "09_data_manifest",
    "proof": "09_row_alignment",
    "search": "09_row_alignment",
}


def _git_rev_parse(rev: str) -> Optional[str]:
    """`git rev-parse rev`, returning `None` (rather than raising) on any failure -- callers
    decide what a failed resolution means."""
    result = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """D9-18's freeze-ancestry gate, `07_crossmodal_curvature_run.py`'s `_strict_ancestor_or_exit`
    shape. Exits 1 naming D9-18 when `--freeze-commit` is absent or empty. Resolves the supplied
    value through `git rev-parse` to a full 40-hex SHA -- never trusts an abbreviation. When
    `FREEZE_COMMIT_SHA` is set (post-09-05), also requires the resolved value to equal it
    exactly (CR-01: a wrong-but-genuine ancestor must not silently pass). Requires BOTH
    `git merge-base --is-ancestor <freeze> HEAD` to exit 0 AND `git rev-list --count
    <freeze>..HEAD` to be at least 1, so a freeze commit equal to HEAD (a commit is its own
    ancestor) is rejected. Writes no record row before this gate passes."""
    if not freeze_commit:
        print(
            "ERROR (D9-18): this mode requires --freeze-commit naming the frozen commit's SHA. "
            "Refusing to compute a Physics number without a strict-ancestor proof.",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_commit = _git_rev_parse(freeze_commit)
    if resolved_commit is None or len(resolved_commit) != 40:
        print(
            f"ERROR (D9-18): --freeze-commit {freeze_commit!r} did not resolve to a full "
            f"40-hex git SHA (resolved={resolved_commit!r}). Refusing to proceed on an "
            "abbreviation or an unresolved ref.",
            file=sys.stderr,
        )
        sys.exit(1)

    if FREEZE_COMMIT_SHA is not None and resolved_commit != FREEZE_COMMIT_SHA:
        print(
            f"ERROR (D9-18): --freeze-commit {freeze_commit!r} (resolves to {resolved_commit}) "
            f"does not equal the known freeze commit FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA!r}. "
            "--freeze-commit must name THE freeze commit, not merely some earlier ancestor.",
            file=sys.stderr,
        )
        sys.exit(1)

    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_commit, "HEAD"],
        cwd=str(NOTEBOOK_ROOT.parent),
    )
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{resolved_commit}..HEAD"],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
    )
    count = -1
    if count_result.returncode == 0 and count_result.stdout.strip().isdigit():
        count = int(count_result.stdout.strip())

    if is_ancestor.returncode != 0 or count < 1:
        print(
            f"ERROR (D9-18): --freeze-commit {freeze_commit!r} is not a STRICT git ancestor of "
            f"HEAD. is_ancestor_exit={is_ancestor.returncode} "
            f"rev_list_count({resolved_commit}..HEAD)={count}. A commit is its own ancestor, so "
            "`git merge-base --is-ancestor` alone is insufficient -- `git rev-list --count "
            "<freeze>..HEAD` must also be >= 1. No Physics number may be produced at or before "
            "the freeze commit itself.",
            file=sys.stderr,
        )
        sys.exit(1)


def _describe_environment() -> Dict[str, Any]:
    """The host's capability, reported before any read or write (T-09-41): core count, thread
    cap, Python and library versions, the resolved HuggingFace cache directory and output root,
    HEAD's `git describe` and the frozen `FREEZE_COMMIT_SHA`. Duplicated verbatim from
    `09_physics_curvature_run.py` -- neither runner imports the other, per this codebase's
    existing per-runner-duplication convention (`_flag_value_from_argv`, `_git_rev_parse`,
    `_strict_ancestor_or_exit` are all duplicated the same way). Calls neither
    `assert_preregistered()` -- every value this reads is either environment-only or a constant
    already frozen by 09-05, never gated by this call itself."""
    git_describe = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=str(NOTEBOOK_ROOT.parent),
        capture_output=True,
        text=True,
    )
    env: Dict[str, Any] = {
        "row_kind": "environment",
        "core_count": os.cpu_count(),
        "thread_cap": _THREADS,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pyarrow_version": pyarrow.__version__,
        "pandas_version": pandas.__version__,
        "datasets_version": datasets.__version__,
        "hf_cache_dir": pl.resolve_hf_cache_dir(),
        "output_root": str(pcp.resolve_output_root()),
        "git_describe_head": git_describe.stdout.strip() if git_describe.returncode == 0 else None,
        "freeze_commit_sha": FREEZE_COMMIT_SHA,
    }
    print(
        f"environment: core_count={env['core_count']} thread_cap={env['thread_cap']} "
        f"python={env['python_version']} torch={env['torch_version']} numpy={env['numpy_version']} "
        f"scipy={env['scipy_version']} scikit-learn={env['scikit_learn_version']} "
        f"pyarrow={env['pyarrow_version']} pandas={env['pandas_version']} "
        f"datasets={env['datasets_version']}"
    )
    print(f"resolved HF cache dir: {env['hf_cache_dir']}")
    print(f"resolved output root: {env['output_root']}")
    return env


def _sanitized_host_label(explicit: Optional[str]) -> str:
    """`--host-label` verbatim-sanitized if supplied, else the machine's own hostname with every
    non-alphanumeric character replaced by `-` -- a safe, portable archive filename component.
    Never committed to any file this plan writes (T-09-40); the archive stays a local, hand-
    transferred artifact on the execution host's own disk. Duplicated verbatim from
    `09_physics_curvature_run.py`."""
    raw = explicit if explicit else (platform.node() or "host")
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")
    return sanitized or "host"


def run_bundle(args: argparse.Namespace) -> bool:
    """Collects every `09_`-prefixed file directly under `resolve_output_root()` into one
    gzipped, checksummed tar -- the same naming `09_physics_curvature_run.py`'s own `run_bundle`
    uses, so either runner produces an archive distinguishable from the other's only by its UTC
    stamp. Embeds the environment description as an `environment.json` archive member. Exits 0
    even on a partial set (T-09-42)."""
    env = _describe_environment()
    output_root = pcp.resolve_output_root()

    host_label = _sanitized_host_label(args.host_label)
    utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_name = f"09-artifacts-{host_label}-{utc_stamp}.tar.gz"
    archive_path = output_root / archive_name
    pcp._assert_inside_output_root(archive_path)

    members = sorted(p for p in output_root.glob("09_*") if p.is_file())

    with tarfile.open(archive_path, "w:gz") as tar:
        for member in members:
            tar.add(member, arcname=member.name)
        env_bytes = json.dumps(env, indent=2).encode("utf-8")
        env_info = tarfile.TarInfo(name="environment.json")
        env_info.size = len(env_bytes)
        tar.addfile(env_info, io.BytesIO(env_bytes))

    archive_bytes = archive_path.read_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    size = len(archive_bytes)

    print(f"\nbundled {len(members)} artifact file(s) plus environment.json:")
    for member in members:
        print(f"  {member.name}")
    print(f"\narchive: {archive_path}")
    print(f"size: {size} bytes")
    print(f"sha256: {digest}")
    return True


_FROZEN_RECORD_STEMS = tuple(sorted(set(RECORD_STEM.values())))


def resolve_record_path(record_path_arg: Optional[str], default_stem: Optional[str]) -> Path:
    """Caller-supplied paths are routed through `pcp._assert_inside_output_root` (T-09-17); a
    traversal path raises rather than writes. `default_stem=None` (used only by `--mode smoke`)
    means: no default is offered, `--record-path` is REQUIRED, and a supplied path matching one
    of `_FROZEN_RECORD_STEMS` is refused -- a smoke row must never land in a frozen record
    (T-09-18). `default_stem` set (manifest/proof/search) means a bare invocation still writes
    to that mode's own record stem under the resolved output root."""
    if record_path_arg is None:
        if default_stem is None:
            raise ValueError(
                "resolve_record_path: no --record-path was supplied and this mode refuses to "
                "default onto any frozen record stem."
            )
        candidate = pcp.record_path(default_stem, "jsonl")
    else:
        candidate = Path(record_path_arg)
        if default_stem is None and candidate.stem in _FROZEN_RECORD_STEMS:
            raise ValueError(
                f"resolve_record_path: --record-path {candidate} matches a frozen record stem "
                f"{_FROZEN_RECORD_STEMS!r}; --mode smoke must write to a scratch path outside "
                "the frozen stems."
            )
    pcp._assert_inside_output_root(candidate)
    return candidate


def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar (07.1's own defect
    precedent, copied verbatim in behaviour)."""
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                f"append_record_row: row[{key!r}] is a raw numpy value ({type(value)!r}); "
                "serialize it to a plain Python scalar/list before appending."
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _read_last_verdict_row(record_path: Path) -> Dict[str, Any]:
    """Scans `record_path` for the LAST row carrying `row_kind == "verdict"` -- the one
    `--mode proof` appended. Raises when the file is absent or no verdict row is found."""
    if not record_path.exists():
        raise FileNotFoundError(f"_read_last_verdict_row: no record found at {record_path}.")
    last_verdict: Optional[Dict[str, Any]] = None
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "verdict":
                last_verdict = row
    if last_verdict is None:
        raise ValueError(f"_read_last_verdict_row: no verdict row found in {record_path}.")
    return last_verdict


def _load_label_table_with_overrides(
    columns: List[str], expected_rows: int, repo: str, revision: str, split: str, n_shards: int
) -> Any:
    """`--mode manifest`-only helper: `physics_labels.load_label_table` only accepts a
    `(columns, expected_rows)` override (Task 1's sealed-for-this-plan signature) -- everything
    else it reads (`LABEL_REPO`/`LABEL_REVISION`/`LABEL_SPLIT`/`LABEL_N_SHARDS`) comes from that
    module's own UNSET constants. This helper temporarily overrides those four module globals
    for the duration of ONE `load_label_table` call, then restores them exactly -- so `--mode
    manifest` can source the label catalog entirely from the command line while every gating
    constant in `physics_labels.py` is UNSET on entry and UNSET again on exit, unchanged by this
    plan (D9-18's "no gating constant is filled by this plan" prohibition)."""
    names = ("LABEL_REPO", "LABEL_REVISION", "LABEL_SPLIT", "LABEL_N_SHARDS")
    saved = {name: getattr(pl, name) for name in names}
    try:
        pl.LABEL_REPO = repo
        pl.LABEL_REVISION = revision
        pl.LABEL_SPLIT = split
        pl.LABEL_N_SHARDS = n_shards
        return pl.load_label_table(columns, expected_rows=expected_rows)
    finally:
        for name, value in saved.items():
            setattr(pl, name, value)


def run_smoke(args: argparse.Namespace) -> bool:
    """The pre-freeze exercise: the whole alignment proof on synthetic arrays with a known
    injected offset. Opens no HuggingFace dataset, requires no `--freeze-commit`, requires an
    explicit `--record-path` outside the frozen stems. Runs the full curve twice -- once with
    the label aligned at offset 0 (the PASS branch) and once rolled to a known non-zero offset
    (the branch D9-08's SEARCH mode reads) -- and asserts the PASS branch reports shift 0 as the
    argmax and the offset branch reports the injected offset as the single clearing alignment.
    Returns True iff both assertions hold."""
    print(
        "\n" + "=" * 78 +
        "\nPRE-FREEZE EXERCISE ON SYNTHETIC ARRAYS -- NOT A DELIVERABLE, PRODUCES NO PHYSICS "
        "NUMBER.\nEvery gating constant in physics_labels/physics_curvature_probe is still "
        "UNSET.\n" + "=" * 78 + "\n"
    )

    _describe_environment()  # prints only; smoke never gets a JSONL environment row (T-09-39)

    record_path = resolve_record_path(args.record_path, default_stem=None)

    n = args.smoke_rows
    ambient = 64
    smoke_seed = 20260902
    # alpha=1.0, not the ALPHA_RIDGE=100.0 the real proof mode uses -- measured: on this
    # smoke fixture's feature scale (unit-norm rows), alpha=100.0 over-shrinks the OOF fit to
    # R2~0.29, well under the 0.30 smoke margin below, even for the perfectly-aligned case; the
    # real proof's D=768 embeddings and ALPHA_RIDGE are calibrated to each other separately and
    # are not this fixture's concern.
    alpha = 1.0
    n_folds = 5

    rng = np.random.default_rng(smoke_seed)
    X_raw = rng.normal(size=(n, ambient))
    X, _ = subsample.l2_normalize(X_raw)
    w = rng.normal(size=ambient)
    y_true = X @ w + 0.05 * rng.normal(size=n)

    def oof_fn(X_f: np.ndarray, y_f: np.ndarray) -> np.ndarray:
        return pcp.oof_ridge_predictions(X_f, y_f, alpha=alpha, n_folds=n_folds, fold_seed=smoke_seed)

    shifts = (-3, -2, -1, 0, 1, 2, 3)

    # --- PASS branch: aligned case ------------------------------------------------------------
    curve_aligned = pl.alignment_r2_curve(
        X, y_true, shifts, n_permutations=5, permutation_seed=smoke_seed, oof_fn=oof_fn
    )
    verdict_aligned = pl.alignment_verdict(curve_aligned, margin=0.30)
    shift_rows_aligned = {row["shift"]: row["r2"] for row in curve_aligned if row["alignment"] == "shift"}
    argmax_shift = max(shift_rows_aligned, key=lambda s: shift_rows_aligned[s])
    aligned_ok = bool(verdict_aligned["passed"] and argmax_shift == 0)
    print(
        f"aligned case: argmax_shift={argmax_shift} r2_shift0={verdict_aligned['r2_shift0']:.4f} "
        f"passed={verdict_aligned['passed']} {'PASS' if aligned_ok else 'FAIL'}"
    )
    append_record_row(
        {
            "mode": "smoke", "case": "aligned", "argmax_shift": int(argmax_shift),
            "r2_shift0": verdict_aligned["r2_shift0"], "passed": aligned_ok,
        },
        record_path,
    )

    # --- SEARCH branch: offset case ------------------------------------------------------------
    injected_offset = 5
    y_offset = np.roll(y_true, injected_offset)
    curve_offset = pl.alignment_r2_curve(
        X, y_offset, shifts + (injected_offset,), n_permutations=5, permutation_seed=smoke_seed,
        oof_fn=oof_fn,
    )
    verdict_offset = pl.alignment_verdict(curve_offset, margin=0.30)
    offset_ok = bool(
        (not verdict_offset["passed"]) and verdict_offset["clearing_alignments"] == [injected_offset]
    )
    print(
        f"offset case: injected_offset={injected_offset} "
        f"clearing_alignments={verdict_offset['clearing_alignments']} "
        f"passed={verdict_offset['passed']} {'PASS' if offset_ok else 'FAIL'}"
    )
    append_record_row(
        {
            "mode": "smoke", "case": "offset", "injected_offset": injected_offset,
            "clearing_alignments": verdict_offset["clearing_alignments"], "passed": offset_ok,
        },
        record_path,
    )

    all_ok = aligned_ok and offset_ok
    print(f"\nrecord written to: {record_path}")
    print("\nALIGNMENT SMOKE PASS" if all_ok else "\nALIGNMENT SMOKE FAIL")
    return all_ok


def run_manifest(args: argparse.Namespace) -> bool:
    """The phase's one pre-freeze evidence mode: reads both real datasets at full scale and
    writes dataset metadata ONLY -- row counts, widths, dtypes, per-column finite and sentinel
    counts. Does NOT call `assert_preregistered()` -- it must run while every gating constant is
    still UNSET, and every value it needs is taken from the command line instead. Exits 2 when
    `--candidate-columns` is absent. No row this mode writes carries a key named `r2`, `rho`,
    `p` or `passed`, and no correlation/regression/curvature function is called anywhere in this
    function."""
    if not args.candidate_columns:
        print(
            "ERROR: --mode manifest requires --candidate-columns naming at least one raw "
            "catalog column to measure.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --candidate-columns is nargs="+", but 09-04's own documented invocation passes it as ONE
    # comma-separated token (`--candidate-columns a,b,c`) rather than space-separated
    # (`--candidate-columns a b c`) -- accept both by splitting every token on "," and dropping
    # empties, so a single comma-joined argv token explodes into the same list a space-separated
    # invocation would produce.
    candidate_columns: List[str] = []
    for token in args.candidate_columns:
        candidate_columns.extend(part for part in token.split(",") if part)
    args.candidate_columns = candidate_columns

    print(
        "\n" + "=" * 78 +
        "\nPRE-FREEZE EVIDENCE RUN -- dataset metadata only. Computes NO Physics number (no "
        "correlation, R2 or curvature). Every value below comes from the command line, never "
        "from the frozen constants (those stay UNSET until 09-05).\n" + "=" * 78 + "\n"
    )

    env = _describe_environment()
    hf_cache_dir = env["hf_cache_dir"]

    record_path = resolve_record_path(args.record_path, default_stem=RECORD_STEM["manifest"])
    append_record_row(env, record_path)

    run_commit = _git_rev_parse("HEAD") or "UNKNOWN"
    timestamp = _utc_timestamp()

    embeddings = pl.load_physics_embeddings(
        parquet_path=args.physics_parquet_path,
        column=args.physics_column,
        expected_rows=args.expected_rows,
        normalize=True,
    )
    labels = _load_label_table_with_overrides(
        list(args.candidate_columns),
        expected_rows=args.expected_rows,
        repo=args.label_repo,
        revision=args.label_revision,
        split=args.label_split,
        n_shards=args.shards,
    )

    column_map = {name: name for name in args.candidate_columns}
    missingness = pl.label_missingness_report(labels, column_map, args.sentinels)

    print(f"{'column':<45}{'n_total':>10}{'n_finite_raw':>15}{'n_sentinel':>12}{'n_finite_masked':>18}")
    for name, stats in missingness.items():
        print(
            f"{name:<45}{stats['n_total']:>10}{stats['n_finite_raw']:>15}"
            f"{stats['n_sentinel']:>12}{stats['n_finite_masked']:>18}"
        )
        row = dict(stats)
        row.update(
            {
                "mode": "manifest", "row_kind": "column", "canonical_name": name,
                "run_commit": run_commit, "timestamp_utc": timestamp,
            }
        )
        append_record_row(row, record_path)

    summary_row = {
        "mode": "manifest",
        "row_kind": "summary",
        "n_rows_embeddings": embeddings["n_rows"],
        "n_rows_labels": int(len(labels)),
        "n_features": embeddings["n_features"],
        "label_repo": args.label_repo,
        "label_revision": args.label_revision,
        "shard_order_rule": "ascending shard index, 0..LABEL_N_SHARDS-1, concatenated in that order",
        "hf_cache_dir": hf_cache_dir,
        "run_commit": run_commit,
        "timestamp_utc": timestamp,
    }
    append_record_row(summary_row, record_path)

    print(f"\nrecord written to: {record_path}")
    print("\nMANIFEST COMPLETE (pre-freeze evidence run, no Physics number produced)")
    return True


def run_proof(args: argparse.Namespace) -> int:
    """D9-06/D9-07 for real. `_strict_ancestor_or_exit` first, then both frozen modules'
    `assert_preregistered()`. Loads the embeddings and `ALIGNMENT_LABEL`, runs
    `alignment_r2_curve` at shift 0 plus every entry of `ALIGNMENT_SHIFT_SET` plus
    `ALIGNMENT_N_PERMUTATIONS` seeded permutations, appends one JSONL row per curve entry as it
    iterates the (already-computed, sealed-function-returned) curve -- never building one
    combined blob and writing it in a single call -- then appends one verdict row. Prints the
    full curve in frozen order, then the verdict line. Returns 0 on PASS, 1 on FAIL (a FAIL is a
    real terminal state, not an error)."""
    _strict_ancestor_or_exit(args.freeze_commit)
    pl.assert_preregistered()
    pcp.assert_preregistered()

    env = _describe_environment()

    record_path = resolve_record_path(args.record_path, default_stem=RECORD_STEM["proof"])
    append_record_row(env, record_path)

    embeddings = pl.load_physics_embeddings()
    raw_column = pl.LABEL_COLUMN_MAP[pl.ALIGNMENT_LABEL]
    label_table = pl.load_label_table([raw_column])
    y = pl.canonical_label(label_table, pl.ALIGNMENT_LABEL, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)

    def oof_fn(X_f: np.ndarray, y_f: np.ndarray) -> np.ndarray:
        return pcp.oof_ridge_predictions(
            X_f, y_f, alpha=pcp.ALPHA_RIDGE, n_folds=pcp.N_OOF_FOLDS, fold_seed=pcp.OOF_FOLD_SEED
        )

    shifts = (0,) + tuple(pl.ALIGNMENT_SHIFT_SET)
    curve = pl.alignment_r2_curve(
        embeddings["X"], y, shifts, pl.ALIGNMENT_N_PERMUTATIONS, pl.ALIGNMENT_PERMUTATION_SEED,
        oof_fn,
    )

    run_commit = _git_rev_parse("HEAD") or "UNKNOWN"
    freeze_commit = _git_rev_parse(args.freeze_commit) or args.freeze_commit
    timestamp = _utc_timestamp()

    print(f"{'alignment':<12}{'label':>8}{'r2':>14}{'n_finite':>12}")
    for row in curve:
        label = row.get("shift", row.get("draw"))
        print(f"{row['alignment']:<12}{label!s:>8}{row['r2']:>14.6f}{row['n_finite']:>12}")
        row_full = dict(row)
        row_full.update(
            {
                "mode": "proof", "row_kind": "curve", "run_commit": run_commit,
                "freeze_commit": freeze_commit, "timestamp_utc": timestamp,
            }
        )
        append_record_row(row_full, record_path)

    verdict = pl.alignment_verdict(curve, pl.ALIGNMENT_MARGIN_R2)
    verdict_row = dict(verdict)
    verdict_row.update(
        {
            "mode": "proof", "row_kind": "verdict", "margin": pl.ALIGNMENT_MARGIN_R2,
            "run_commit": run_commit, "freeze_commit": freeze_commit, "timestamp_utc": timestamp,
        }
    )
    append_record_row(verdict_row, record_path)

    print(
        f"\nverdict: r2_shift0={verdict['r2_shift0']:.6f} best_other_r2={verdict['best_other_r2']:.6f} "
        f"gap={verdict['gap']:.6f} margin={pl.ALIGNMENT_MARGIN_R2} passed={verdict['passed']}"
    )
    print(f"record written to: {record_path}")
    return 0 if verdict["passed"] else 1


def run_search(args: argparse.Namespace) -> int:
    """D9-08's branch: only runs on the failure branch. `_strict_ancestor_or_exit` first. Reads
    the proof record's last verdict row and exits 2 naming the reason when `passed` is already
    true. When `passed` is false, classifies `clearing_alignments`: exactly one is a CANDIDATE
    OFFSET (returns 0), two or more is AMBIGUOUS (halts, returns 1), zero is NO ALIGNMENT FOUND
    (halts, returns 1). Never adopts an offset itself -- adoption is 09-07's blocking developer
    decision plus a fresh freeze."""
    _strict_ancestor_or_exit(args.freeze_commit)
    env = _describe_environment()
    record_path = resolve_record_path(args.record_path, default_stem=RECORD_STEM["search"])
    append_record_row(env, record_path)
    verdict_row = _read_last_verdict_row(record_path)

    if verdict_row.get("passed"):
        print(
            "ERROR: --mode search only runs on the failure branch; the proof record's verdict "
            "already PASSED (shift 0 cleared the margin). Nothing to search for.",
            file=sys.stderr,
        )
        sys.exit(2)

    clearing = list(verdict_row.get("clearing_alignments", []))
    print(f"clearing_alignments: {clearing}")

    if len(clearing) == 1:
        classification = "CANDIDATE_OFFSET"
        print(f"CANDIDATE OFFSET: {clearing[0]}")
    elif len(clearing) >= 2:
        classification = "AMBIGUOUS"
        print(
            "AMBIGUOUS: more than one alignment clears the margin -- halting rather than "
            "picking one."
        )
    else:
        classification = "NO_ALIGNMENT_FOUND"
        print("NO ALIGNMENT FOUND -- halting.")

    row = {
        "mode": "search", "row_kind": "search_classification",
        "clearing_alignments": clearing, "classification": classification,
        "run_commit": _git_rev_parse("HEAD") or "UNKNOWN",
        "freeze_commit": _git_rev_parse(args.freeze_commit) or args.freeze_commit,
        "timestamp_utc": _utc_timestamp(),
    }
    append_record_row(row, record_path)

    print(f"record written to: {record_path}")
    return 0 if classification == "CANDIDATE_OFFSET" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "manifest", "proof", "search", "bundle"], default="smoke")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--freeze-commit", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--smoke-rows", type=int, default=1500)
    p.add_argument("--shards", type=int, default=16)
    # --mode manifest only -- every value is a command-line-supplied literal, never a frozen
    # constant, because this mode runs before the freeze.
    p.add_argument("--candidate-columns", type=str, nargs="+", default=None)
    p.add_argument("--expected-rows", type=int, default=86_471)
    p.add_argument("--sentinels", type=float, nargs="+", default=[-99.0])
    p.add_argument("--label-repo", type=str, default="Smith42/galaxies")
    p.add_argument("--label-revision", type=str, default="v2.0")
    p.add_argument("--label-split", type=str, default="test")
    p.add_argument(
        "--physics-parquet-path", type=str,
        default="hf://datasets/UniverseTBD/pu-embeddings/physics/vit_base_test.parquet",
    )
    p.add_argument("--physics-column", type=str, default="vit_base_galaxies")
    p.add_argument("--host-label", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.output_root and pcp.OUTPUT_ROOT_ENV_VAR:
        os.environ[pcp.OUTPUT_ROOT_ENV_VAR] = args.output_root

    if args.mode == "smoke":
        ok = run_smoke(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "manifest":
        ok = run_manifest(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "proof":
        sys.exit(run_proof(args))
    elif args.mode == "search":
        sys.exit(run_search(args))
    elif args.mode == "bundle":
        ok = run_bundle(args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
