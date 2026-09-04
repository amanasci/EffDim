"""Phase 9 curvature-conditioned label-decodability runner. `--mode smoke` (09-01) is the
tracer: it wires the WHOLE Phase 9 statistical path -- synthetic aligned (X, y) pair,
shifted-alignment R2 curve, 5-fold OOF ridge, one autoencoder fit at one small `d`, curvature at
anchors, radial/tangential decomposition, 3-control partial, Freedman-Lane null, JSONL record --
through one command, entirely on synthetic arrays generated in-process. It opens no HuggingFace
dataset and computes no Physics number. Every other `--mode` value is not yet implemented by
this plan and exits 2 naming the plan that adds it.

Usage:
    python notebooks/diagnostics/09_physics_curvature_run.py --mode smoke --record-path notebooks/.cache/09_scratch_tracer.jsonl
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` and `--flag=value` -- or `None` if `flag` was not passed in either
    form. Kept dependency-free so it can run here, above the torch import."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in torch/numpy (07_crossmodal_curvature_run.py
# precedent: concurrent torch jobs measured driving load up ~10x).
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_THREADS)

# Imported only so _describe_environment() can report their installed versions (T-09-41) --
# every one of these is already a project dependency, imported elsewhere transitively.
import datasets  # noqa: E402
import pandas  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cae  # noqa: E402
from pu_manifold import cross_split_curvature  # noqa: E402
from pu_manifold import crossmodal_curvature  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import linear_probe  # noqa: E402
from pu_manifold import subsample  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

# Modes not yet implemented by this plan, and which plan implements each. "bundle" and
# "print-cost-model" were removed from this dict by 09-06; "dsweep", "positive-control",
# "shuffled-label" and "verdict" were removed by 09-08 -- all eight are implemented below and
# dispatched directly in main(), never falling through to this "not implemented" table.
_MODE_IMPLEMENTING_PLAN = {
    "seeds": "09-09",
    "selfcheck": "a later plan (not yet scheduled)",
}

# FREEZE_COMMIT_SHA wired to plan 09-05 Task 1's freeze commit (D9-18): the commit that filled
# every gating constant in physics_labels.py and physics_curvature_probe.py -- mirrors
# 09_row_alignment_proof_run.py's own FREEZE_COMMIT_SHA/_strict_ancestor_or_exit pair exactly.
# No mode in THIS runner produces a Physics number yet (every non-smoke mode above exits 2,
# implemented by a later plan); this gate is wired now so 09-06/09-08/09-09 call it, rather than
# each re-deriving the freeze-ancestry check independently.
FREEZE_COMMIT_SHA = "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5"


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


def _strict_ancestor_or_exit(freeze_commit: Optional[str]) -> None:
    """D9-18's freeze-ancestry gate, `09_row_alignment_proof_run.py`'s `_strict_ancestor_or_exit`
    shape exactly. Exits 1 naming D9-18 when `--freeze-commit` is absent or empty. Resolves the
    supplied value through `git rev-parse` to a full 40-hex SHA -- never trusts an abbreviation.
    Requires the resolved value to equal `FREEZE_COMMIT_SHA` exactly (CR-01: a wrong-but-genuine
    ancestor must not silently pass). Requires BOTH `git merge-base --is-ancestor <freeze> HEAD`
    to exit 0 AND `git rev-list --count <freeze>..HEAD` to be at least 1, so a freeze commit equal
    to HEAD (a commit is its own ancestor) is rejected. Writes no record row before this gate
    passes."""
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

    if resolved_commit != FREEZE_COMMIT_SHA:
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


# --- Cost model (D9-06/09-06): CORE-HOURS, portable across an unknown execution host ------------
# Derived from Phase 7's measured DSWEEP_COST_MODEL_MINUTES (07-CONTEXT.md Section 7, an 8-thread
# cap, 10,000 rows, the curvature field evaluated at every one of those rows). Two independent
# scalings, kept separate below so a reader can see which dominates:
#   - training scales LINEARLY in rows: Phase 9 trains on EXPECTED_N_PHYSICS_ROWS=86,471 rows
#     against Phase 7's 10,000, an 8.65x multiplier. Training cost is d-INDEPENDENT to first
#     order -- AE_HIDDEN=(250, 250, 250) is a fixed encoder/decoder width, unchanged across
#     D_SWEEP; only the bottleneck layer's width changes with d -- so ONE training figure applies
#     to every entry of D_SWEEP, unlike curvature below.
#   - curvature drops by the ANCHOR-evaluation ratio: D9-04 evaluates the field at N_ANCHORS=512
#     points only, never at every row -- Phase 7 evaluated at all 10,000 rows. This single
#     departure removes most of Phase 7's dominant cost term (a 512/10,000 = 0.0512x multiplier,
#     applied BEFORE the d-scaling below), which is why training -- not curvature -- dominates
#     Phase 9's cost, the reverse of Phase 7's own shape.
# d=16 has no entry in Phase 7's own table (Phase 7's D_SWEEP was (20, 25, 32)); its relative-cost
# multiplier is derived from 07-CONTEXT.md Section 7's own stated scaling law ("scales as D*d^2",
# D=768 fixed across both phases): (16/20)**2 = 0.64, consistent with the measured ratios Phase 7
# recorded for d=25 ((25/20)**2 = 1.5625 against its measured ~1.6x) and d=32 ((32/20)**2 = 2.56
# against its measured ~2.6x) -- both within Phase 7's own rounding.
_D20_CURVATURE_CORE_HOURS_10K_ROWS_ALL_EVAL = 1457.0 / 3600.0 * 8  # 07-CONTEXT.md Sec 7, d=20
_D20_TRAINING_CORE_HOURS_10K_ROWS = 374.0 / 3600.0 * 8  # 07-CONTEXT.md Sec 7, 600 epochs, d=20
_ROW_SCALING_RATIO = 86_471 / 10_000  # physics_labels.EXPECTED_N_PHYSICS_ROWS over Phase 7's 10,000
_ANCHOR_SCALING_RATIO = 512 / 10_000  # pcp.N_ANCHORS over Phase 7's every-row evaluation

_TRAINING_CORE_HOURS = _D20_TRAINING_CORE_HOURS_10K_ROWS * _ROW_SCALING_RATIO  # d-independent

DSWEEP_COST_MODEL_CORE_HOURS: Dict[int, Dict[str, float]] = {
    d: {
        "training_core_hours": _TRAINING_CORE_HOURS,
        "curvature_core_hours": (
            _D20_CURVATURE_CORE_HOURS_10K_ROWS_ALL_EVAL * ((d / 20.0) ** 2) * _ANCHOR_SCALING_RATIO
        ),
    }
    for d in pcp.D_SWEEP
}
"""Per-`d` mapping of {training_core_hours, curvature_core_hours}, stated in CORE-HOURS -- a
portable form, unlike Phase 7's own DSWEEP_COST_MODEL_MINUTES wall-clock-minute figures, because
the execution host's core count is unknown at planning time. Printed by `print_cost_model`. An
ESTIMATE scaled from Phase 7's measurements; 09-08 records the measured figure."""


def _describe_environment() -> Dict[str, Any]:
    """The host's capability, reported before any read or write (T-09-41): core count, thread
    cap, Python and library versions, the resolved HuggingFace cache directory and output root,
    HEAD's `git describe` and the frozen `FREEZE_COMMIT_SHA`. Printed here; the caller decides
    whether to also append it as a `row_kind="environment"` JSONL row or embed it in an archive.
    Calls neither `assert_preregistered()` -- every value this reads is either environment-only
    or a constant already frozen by 09-05, never gated by this call itself."""
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
    transferred artifact on the execution host's own disk."""
    raw = explicit if explicit else (platform.node() or "host")
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")
    return sanitized or "host"


def run_bundle(args: argparse.Namespace) -> bool:
    """Collects every `09_`-prefixed file directly under `resolve_output_root()` into one
    gzipped, checksummed tar -- the artifact 09-EXECUTION-HOST.md's Task 3 transfers back.
    Embeds the environment description as a `environment.json` archive member. Exits 0 even on a
    partial set (T-09-42): an interrupted multi-hour run is exactly the case whose evidence must
    not be thrown away for being incomplete. The archive's own name (`09-artifacts-...`, a hyphen
    after `09`) never matches the `09_`-prefix (underscore) glob, so re-running `--mode bundle`
    never bundles a prior archive into a new one."""
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


def print_cost_model(threads: int) -> None:
    """Prints one row per `d` in `D_SWEEP`: training core-hours, curvature core-hours, their
    total, and the implied wall-clock at `threads`. Header line names `threads` and the host's
    own `os.cpu_count()` so a reader on an unfamiliar host sees both numbers at once. Touches no
    data and calls neither `assert_preregistered()`."""
    host_cores = os.cpu_count() or 0
    print(
        f"\n{'-' * 78}\n"
        f"Phase 9 cost model -- CORE-HOURS, portable across hosts. threads={threads} "
        f"host_core_count={host_cores}\n"
        f"{'-' * 78}"
    )
    header = (
        f"{'d':>4}{'training core-hr':>20}{'curvature core-hr':>20}{'total core-hr':>16}"
        f"{'wallclock@' + str(threads) + 't (hr)':>22}"
    )
    print(header)
    for d in pcp.D_SWEEP:
        entry = DSWEEP_COST_MODEL_CORE_HOURS[d]
        total = entry["training_core_hours"] + entry["curvature_core_hours"]
        wallclock_hours = total / threads if threads else float("inf")
        print(
            f"{d:>4}{entry['training_core_hours']:>20.3f}{entry['curvature_core_hours']:>20.3f}"
            f"{total:>16.3f}{wallclock_hours:>22.3f}"
        )
    print(
        "\nModel scaled from Phase 7's measured DSWEEP_COST_MODEL_MINUTES (07-CONTEXT.md Section "
        "7, 8-thread cap, 10,000 rows, curvature evaluated at every row): training scaled "
        "linearly by rows (86,471/10,000 ~= 8.65x), curvature scaled by the anchor-evaluation "
        "ratio (512/10,000 = 0.0512x, D9-04's single biggest cost difference from Phase 7). This "
        "is an ESTIMATE; 09-08 replaces it with the measured figure."
    )


def resolve_record_path(record_path_arg: Optional[str], default_stem: Optional[str] = None) -> Path:
    """Caller-supplied paths are routed through `pcp._assert_inside_output_root` (T-09-03); a
    traversal path raises rather than writes. `--mode smoke` calls this with `default_stem=None`
    and refuses to default onto any frozen record stem -- an explicit `--record-path` is
    required. Every production mode (09-08) instead passes `default_stem=pcp.RECORD_STEM`, so an
    omitted `--record-path` falls onto the frozen `09_physics_curvature.jsonl` record via
    `pcp.record_path`, itself containment-checked."""
    if record_path_arg is None:
        if default_stem is not None:
            return pcp.record_path(default_stem, "jsonl")
        raise ValueError(
            "resolve_record_path: no --record-path was supplied and this mode refuses to "
            "default onto any frozen record stem."
        )
    candidate = Path(record_path_arg)
    pcp._assert_inside_output_root(candidate)
    return candidate


def append_record_row(row: Dict[str, Any], record_path: Path) -> None:
    """Write one flat JSON-serializable dict per line. Every value must already be a plain
    Python scalar, list or string -- never a raw numpy array or numpy scalar (07.1's own
    defect precedent, copied verbatim in behaviour)."""
    for key, value in row.items():
        if isinstance(value, (np.ndarray, np.generic)):
            raise TypeError(
                f"append_record_row: row[{key!r}] is a raw numpy value ({type(value)!r}); "
                "serialize it to a plain Python scalar/list before appending."
            )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _gate_and_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Common preamble every non-smoke production mode (09-08) runs, in this exact order, before
    any read or write: describe the environment, the strict-ancestor freeze proof, then BOTH
    `assert_preregistered()` calls (`pl` then `pcp`). Returns the environment dict; the caller
    resolves its own `--record-path` (each mode may need mode-specific validation, e.g.
    `--field-npz`, before it is safe to write anything) and appends this dict as the first row
    of its own successful run -- never on an error path, so a validation failure never leaves a
    stray row in the record (T-09-58)."""
    env = _describe_environment()
    _strict_ancestor_or_exit(args.freeze_commit)
    pl.assert_preregistered()
    pcp.assert_preregistered()
    return env


_ANCHOR_TABLE_NAME_RE = re.compile(r"^09_anchor_table_d(\d+)_(.+)\.npz$")


def _anchor_table_path(output_root: Path, d: int, label: str) -> Path:
    """`{output_root}/09_anchor_table_d{d}_{label}.npz`, containment-checked -- the filename
    pattern Task 2's host instructions name literally (D9-12 ordering: `d` is part of the
    filename)."""
    path = output_root / f"09_anchor_table_d{d}_{label}.npz"
    pcp._assert_inside_output_root(path)
    return path


def _parse_anchor_table_filename(path: Path) -> Dict[str, Any]:
    """Recovers `{"d": int, "label": str}` from an anchor table's own filename (the two gates
    receive only a `--field-npz` path, not a `d`/label pair passed separately) -- `{"d": None,
    "label": None}` when the name does not match `_anchor_table_path`'s own pattern."""
    match = _ANCHOR_TABLE_NAME_RE.match(Path(path).name)
    if not match:
        return {"d": None, "label": None}
    return {"d": int(match.group(1)), "label": match.group(2)}


def _oof_predictions_for_label(
    X: np.ndarray, y_full: np.ndarray, alpha: float, n_folds: int, fold_seed: int
) -> np.ndarray:
    """`pcp.oof_ridge_predictions` requires every row of `y` to be finite (its own structural
    out-of-fold proof guard); a Physics label may carry sentinel-masked `NaN` rows (`photo_z`
    and `stellar_mass` are not 100% populated -- 09-DATA-MANIFEST.md). Fits and predicts OOF
    only on the finite subset of rows, scattering the result back into a full-length array with
    `NaN` at every non-finite row -- never widening the fold structure to hold out a row with no
    real label value. Returns an all-`NaN` array, rather than raising, when no row is finite."""
    y_full = np.asarray(y_full, dtype=np.float64).ravel()
    finite = np.isfinite(y_full)
    y_hat_full = np.full(y_full.shape[0], np.nan, dtype=np.float64)
    if not np.any(finite):
        return y_hat_full
    y_hat_full[finite] = pcp.oof_ridge_predictions(
        X[finite], y_full[finite], alpha=alpha, n_folds=n_folds, fold_seed=fold_seed
    )
    return y_hat_full


def local_mse_sst_panel(
    y: np.ndarray, y_hat: np.ndarray, neighbour_idx: np.ndarray, min_finite: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirrors `pcp.LOCAL_R2_RULE`'s construction EXACTLY -- same finite mask, same `min_finite`
    floor, same zero-SST exclusion -- to expose `mse`/`sst` per anchor. `pcp.local_r2_panel`
    computes both internally but returns only their ratio (`r2`); this is the same loop, never a
    reimplementation of the R2 formula itself, just retaining the two intermediate values
    `local_r2_panel` discards -- needed because this plan's anchor table (`must_haves`) carries
    `mse` and `sst` as their own columns, mirroring the colleague's own
    `global_anchor_metrics.csv`. Returns `(mse, sst)`, each `NaN` at exactly the anchors where
    `pcp.local_r2_panel`'s own `r2` would be `NaN`."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
    neighbour_idx = np.asarray(neighbour_idx)
    n_anchors = neighbour_idx.shape[0]
    mse = np.full(n_anchors, np.nan, dtype=np.float64)
    sst = np.full(n_anchors, np.nan, dtype=np.float64)
    for i in range(n_anchors):
        nbrs = neighbour_idx[i]
        y_n = y[nbrs]
        yhat_n = y_hat[nbrs]
        finite = np.isfinite(y_n) & np.isfinite(yhat_n)
        if int(finite.sum()) < min_finite:
            continue
        y_f = y_n[finite]
        yhat_f = yhat_n[finite]
        mean_y = float(np.mean(y_f))
        sst_v = float(np.sum((y_f - mean_y) ** 2))
        if sst_v == 0.0:
            continue
        sst[i] = sst_v
        mse[i] = float(np.sum((y_f - yhat_f) ** 2))
    return mse, sst


def build_anchor_table(
    anchor_idx: np.ndarray,
    decomp: Dict[str, np.ndarray],
    cond_g: np.ndarray,
    panel: Dict[str, np.ndarray],
    mse: np.ndarray,
    sst: np.ndarray,
    log_knn_radius: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assembles the per-`(d, label)` anchor table this plan's `must_haves` names: the anchor
    row index, `H_norm`, `H_tan_norm`, `H_rad`, `image_norm`, `cond_g`, the local `r2`, `mse`,
    `sst`, `local_label_variance`, `local_evaluation_count` and `log_knn_radius` -- the
    colleague's own `global_anchor_metrics.csv` column set plus the curvature/decomposition
    columns his table cannot have."""
    return {
        "anchor_idx": np.asarray(anchor_idx, dtype=np.int64),
        "H_norm": np.asarray(decomp["H_norm"], dtype=np.float64),
        "H_tan_norm": np.asarray(decomp["H_tan_norm"], dtype=np.float64),
        "H_rad": np.asarray(decomp["H_rad"], dtype=np.float64),
        "image_norm": np.asarray(decomp["image_norm"], dtype=np.float64),
        "cond_g": np.asarray(cond_g, dtype=np.float64),
        "r2": np.asarray(panel["r2"], dtype=np.float64),
        "mse": np.asarray(mse, dtype=np.float64),
        "sst": np.asarray(sst, dtype=np.float64),
        "local_label_variance": np.asarray(panel["local_label_variance"], dtype=np.float64),
        "local_evaluation_count": np.asarray(panel["local_evaluation_count"], dtype=np.int64),
        "log_knn_radius": np.asarray(log_knn_radius, dtype=np.float64),
    }


def write_anchor_table(table: Dict[str, np.ndarray], path: Path) -> Path:
    """Containment-checked `np.savez` of `table` to `path`. Returns `path` for chaining."""
    pcp._assert_inside_output_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)
    return path


def load_anchor_table(path: Any) -> Dict[str, np.ndarray]:
    """Containment-checked `np.load` of an anchor table written by `write_anchor_table`. Raises
    `FileNotFoundError` naming `path` when it does not exist -- the two gates refuse to
    regenerate a field and must see this raised, never a silent empty read (T-09-51)."""
    candidate = Path(path)
    pcp._assert_inside_output_root(candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"load_anchor_table: {candidate} does not exist.")
    with np.load(candidate) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def fit_and_field_at_anchors(
    X: np.ndarray,
    d: int,
    anchor_idx: np.ndarray,
    in_dim: int,
    hidden: Any,
    activation: str,
    train_cfg: Dict[str, Any],
    max_epochs: int,
    torch_init_seed: int,
    split_seed: int,
    holdout_fraction: float,
) -> Dict[str, Any]:
    """`08_radial_curvature_decomposition_run.py`'s `fit_and_decompose` path, except
    `plain_decoder_curvature` and `model.decode` are evaluated at the ANCHOR latent codes only,
    never at all rows -- D9-04's deliberate departure from Phase 7's `FIELD_EVALUATED_ON`
    convention. Passes the MODEL to `plain_decoder_curvature`, never a bound method, so its
    float64 guard actually runs. Returns `H_vec`, `image`, `metric_condition_number`,
    `var_explained` and the two wallclocks."""
    torch.manual_seed(torch_init_seed)
    model = cae.PlainAutoEncoder(in_dim=in_dim, latent_dim=d, hidden=hidden, activation=activation)

    train_idx, holdout_idx = crossmodal_curvature.split_indices(X.shape[0], split_seed, holdout_fraction)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]

    cfg = dict(train_cfg)
    cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x_train32, cfg)
    wallclock_fit_s = time.monotonic() - t0

    model.eval().double()
    anchor_idx_t = torch.as_tensor(np.asarray(anchor_idx), dtype=torch.long)
    x_anchor64 = x64[anchor_idx_t]
    with torch.no_grad():
        z_anchor = model.encode(x_anchor64)
        y_holdout = model(x_holdout64)["y"]
        image = model.decode(z_anchor).detach().cpu().numpy()

    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    var_explained = 1.0 - recon["mse_total"] / sig

    t0 = time.monotonic()
    field = decoder_curvature.plain_decoder_curvature(model, z_anchor)
    wallclock_field_s = time.monotonic() - t0

    H_vec = field["H_vec"].detach().cpu().numpy()
    metric_condition_number = field["metric_condition_number"].detach().cpu().numpy()

    return {
        "H_vec": H_vec,
        "image": image,
        "metric_condition_number": metric_condition_number,
        "var_explained": float(var_explained),
        "wallclock_fit_s": wallclock_fit_s,
        "wallclock_field_s": wallclock_field_s,
    }


def run_smoke(args: argparse.Namespace) -> bool:
    """The tracer: proves the WHOLE Phase 9 statistical path wires together, entirely on
    synthetic arrays. Does NOT call either `assert_preregistered()` -- a pure in-memory
    known-answer exercise, following 07.1's and 08's own selfcheck convention. Prints a
    pre-freeze banner, requires `--record-path`, and writes no `verdict`/`phase_verdict` key.
    Returns True iff every stage PASSED."""
    print(
        "\n" + "=" * 78 +
        "\nPRE-FREEZE EXERCISE ON SYNTHETIC ARRAYS -- NOT A DELIVERABLE, PRODUCES NO PHYSICS "
        "NUMBER.\nEvery gating constant in physics_labels/physics_curvature_probe is still "
        "UNSET.\n" + "=" * 78 + "\n"
    )

    _describe_environment()  # prints only; smoke never gets a JSONL environment row (T-09-39)

    record_path = resolve_record_path(args.record_path)

    n = args.smoke_rows
    ambient = 64
    d = 4
    n_anchors = 128
    k = 64
    alpha = 100.0
    n_folds = 5
    smoke_seed = 20260902

    stages = []

    def _stage(name: str, measured: Any, expected: Any, passed: bool) -> None:
        status = "PASS" if passed else "FAIL"
        print(f"stage={name} measured={measured} expected={expected} {status}")
        row = {
            "stage": name,
            "measured": measured,
            "expected": expected,
            "passed": bool(passed),
        }
        append_record_row(row, record_path)
        stages.append((name, passed))

    t_start = time.monotonic()

    # --- 1. synthetic aligned (X, y) pair, built so the true offset is 0 by construction -----
    # X sits near a genuine d=4 submanifold of R^ambient (a smooth nonlinear image of a d=4
    # Gaussian latent), not raw ambient noise -- an autoencoder has something learnable to
    # reconstruct, and the anchor curvature check below has a meaningful (non-degenerate) field
    # to measure. y is a fixed linear functional of the ambient embedding plus small noise, so
    # the true row alignment is offset 0 by construction.
    rng = np.random.default_rng(smoke_seed)
    latent_dim_true = d
    Z_true = rng.normal(size=(n, latent_dim_true))
    A1 = rng.normal(size=(latent_dim_true, 32)) / np.sqrt(latent_dim_true)
    A2 = rng.normal(size=(32, ambient)) / np.sqrt(32)
    X_raw = np.tanh(Z_true @ A1) @ A2
    X, _ = subsample.l2_normalize(X_raw)
    w = rng.normal(size=ambient)
    y = X @ w + 0.01 * rng.normal(size=n)

    def oof_fn(X_f: np.ndarray, y_f: np.ndarray) -> np.ndarray:
        return pcp.oof_ridge_predictions(X_f, y_f, alpha=alpha, n_folds=n_folds, fold_seed=smoke_seed)

    # --- 2. shifted-alignment R2 curve --------------------------------------------------------
    shifts = (-3, -2, -1, 0, 1, 2, 3)
    curve = pl.alignment_r2_curve(
        X, y, shifts, n_permutations=5, permutation_seed=smoke_seed, oof_fn=oof_fn
    )
    verdict = pl.alignment_verdict(curve, margin=0.30)
    _stage(
        "alignment",
        0 if (verdict["passed"] and verdict["r2_shift0"] >= verdict["best_other_r2"]) else -1,
        0,
        bool(verdict["passed"] and verdict["r2_shift0"] >= verdict["best_other_r2"]),
    )

    # --- 3. anchor draw, disjoint from train rows (D9-04) -------------------------------------
    idx = pcp.anchor_indices(
        n_rows=n, split_seed=smoke_seed, holdout_fraction=0.2, n_anchors=n_anchors, anchor_seed=smoke_seed
    )
    n_intersect = int(np.intersect1d(idx["anchor_idx"], idx["train_idx"]).size)
    _stage("anchor_disjoint", n_intersect, 0, n_intersect == 0)

    # --- 4. one autoencoder fit at one small d, curvature at anchors --------------------------
    max_epochs = 150
    train_cfg = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch": 128,
        "early_stop_min_delta": 1e-9,
        "early_stop_patience": max_epochs + 1,
        "wallclock_ceiling_s": float("inf"),
    }
    fit = fit_and_field_at_anchors(
        X,
        d=d,
        anchor_idx=idx["anchor_idx"],
        in_dim=ambient,
        hidden=(64, 64),
        activation="silu",
        train_cfg=train_cfg,
        max_epochs=max_epochs,
        torch_init_seed=0,
        split_seed=smoke_seed,
        holdout_fraction=0.2,
    )
    _stage("ae_fit", round(fit["var_explained"], 4), ">0.7", fit["var_explained"] > 0.7)

    # --- 5. radial/tangential decomposition ---------------------------------------------------
    decomp = pcp.decompose_radial_tangential(fit["H_vec"], fit["image"], min_image_norm=1e-9)
    h_rad_median = float(np.nanmedian(decomp["H_rad"]))
    # Sign-and-order-of-magnitude check: H_rad should be negative (mean curvature points at the
    # origin for points near the unit sphere) and within the same order of magnitude as -d.
    radial_ok = (h_rad_median < 0.0) and (0.2 * d <= abs(h_rad_median) <= 5.0 * d)
    _stage("radial_decomposition", round(h_rad_median, 4), -d, radial_ok)

    # --- 6. k-NN panel, 5-fold OOF ridge on the anchors' neighbourhoods, local R2 -------------
    knn = pcp.knn_panel(X, idx["anchor_idx"], k=k)
    _stage("knn", int(knn["indices"].shape[1]), k, int(knn["indices"].shape[1]) == k)

    y_hat = pcp.oof_ridge_predictions(X, y, alpha=alpha, n_folds=n_folds, fold_seed=smoke_seed)
    n_finite_oof = int(np.sum(np.isfinite(y_hat)))
    _stage("oof_ridge", n_finite_oof, n, n_finite_oof == n)

    panel = pcp.local_r2_panel(y, y_hat, knn["indices"], min_finite=10)
    n_valid_r2 = int(np.sum(np.isfinite(panel["r2"])))
    _stage("local_r2", n_valid_r2, ">0", n_valid_r2 > 0)

    # --- 7. 3-control partial and Freedman-Lane null ------------------------------------------
    controls = np.column_stack(
        [knn["log_knn_radius"], panel["local_label_variance"], panel["local_evaluation_count"]]
    )
    finite = np.isfinite(panel["r2"])
    h_anchor = decomp["H_tan_norm"]
    cp_val = pcp.controlled_partial(h_anchor[finite], panel["r2"][finite], controls[finite])
    _stage("controlled_partial", round(float(cp_val), 4), "finite", bool(np.isfinite(cp_val)))

    fwer = pcp.permutation_fwer(
        {d: h_anchor[finite]},
        panel["r2"][finite],
        controls[finite],
        n_permutations=args.smoke_permutations,
        seed=smoke_seed,
    )
    p_display = fwer["global"]["p_display"]
    _stage("permutation_fwer", p_display, "p or '< ...' string", bool(fwer["global"]["p"] > 0.0))

    t_total = time.monotonic() - t_start
    all_passed = all(passed for _, passed in stages)
    print(f"\ntotal wallclock: {t_total:.1f}s")
    print(f"record written to: {record_path}")
    print("\nSMOKE PASS" if all_passed else "\nSMOKE FAIL")
    return all_passed


def run_dsweep(args: argparse.Namespace) -> bool:
    """`--mode dsweep` (D9-12): the phase's deliverable. ONE sequential in-process loop over
    `pcp.D_SWEEP`, never concurrent (09-EXECUTION-HOST.md Section 4's own precedent). Loads the
    embeddings and every label once; computes the anchor draw, the k-NN neighbourhood panel and
    every label's out-of-fold ridge probe once, before the `d` loop, all three independent of
    `d` by construction (this plan's own `<discretion_decisions>`). For each `d`: fits the
    autoencoder and evaluates curvature at the anchors only (D9-04), decomposes radial/tangential
    (D9-11), writes one anchor table per label, and computes the raw/controlled partial and the
    density-stratified null for both `H_tan_norm` and `H_norm` against every label. After the
    loop, computes ONE Freedman-Lane family-wise envelope per label/field across ALL `d` at once
    (the null construction the family-wise `p_fwer` needs a common surrogate for)."""
    env = _gate_and_environment(args)
    freeze_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path, default_stem=pcp.RECORD_STEM)
    append_record_row(env, record_path)

    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    output_root = pcp.resolve_output_root()
    all_labels = (pl.PRIMARY_LABEL,) + pl.SECONDARY_LABELS

    print(
        f"\nDSWEEP: D_SWEEP={pcp.D_SWEEP} N_ANCHORS={pcp.N_ANCHORS} K_NEIGHBOURS={pcp.K_NEIGHBOURS} "
        f"labels={all_labels}\n{pcp.NEIGHBOURHOOD_RATIO_RULE}\n"
    )

    t0 = time.monotonic()
    emb = pl.load_physics_embeddings()
    X = emb["X"]
    n_rows = emb["n_rows"]
    print(f"[load] physics embeddings: n_rows={n_rows} wallclock={time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    table = pl.load_label_table(columns=list(pl.LABEL_COLUMN_MAP.values()))
    print(f"[load] label table: wallclock={time.monotonic() - t0:.1f}s")

    offset_perm = pl.shifted_pairing(n_rows, pl.ALIGNMENT_ASSUMED_OFFSET)
    y_by_label: Dict[str, np.ndarray] = {}
    for name in all_labels:
        y_raw = pl.canonical_label(table, name, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)
        y_by_label[name] = y_raw[offset_perm]

    idx = pcp.anchor_indices(
        n_rows=n_rows, split_seed=pcp.SPLIT_SEED, holdout_fraction=pcp.HOLDOUT_FRACTION,
        n_anchors=pcp.N_ANCHORS, anchor_seed=pcp.ANCHOR_DRAW_SEED,
    )
    anchor_idx = idx["anchor_idx"]

    t0 = time.monotonic()
    knn = pcp.knn_panel(X, anchor_idx, pcp.K_NEIGHBOURS)
    print(f"[knn] k={pcp.K_NEIGHBOURS} n_anchors={anchor_idx.shape[0]} wallclock={time.monotonic() - t0:.1f}s")

    panel_by_label: Dict[str, Dict[str, Any]] = {}
    for name in all_labels:
        t0 = time.monotonic()
        y_hat = _oof_predictions_for_label(X, y_by_label[name], pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
        panel = pcp.local_r2_panel(y_by_label[name], y_hat, knn["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        mse, sst = local_mse_sst_panel(y_by_label[name], y_hat, knn["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        const_eval = bool(np.all(panel["local_evaluation_count"] == panel["local_evaluation_count"][0]))
        panel_by_label[name] = {"panel": panel, "mse": mse, "sst": sst, "const_eval": const_eval}
        print(
            f"[oof/local_r2] label={name} n_masked_anchors={panel['n_masked_anchors']} "
            f"local_evaluation_count_constant={const_eval} wallclock={time.monotonic() - t0:.1f}s"
        )

    controls_by_label = {
        name: np.column_stack([
            knn["log_knn_radius"],
            panel_by_label[name]["panel"]["local_label_variance"],
            panel_by_label[name]["panel"]["local_evaluation_count"],
        ])
        for name in all_labels
    }

    H_tan_full_by_d: Dict[int, np.ndarray] = {}
    H_norm_full_by_d: Dict[int, np.ndarray] = {}

    for d in pcp.D_SWEEP:
        cost = DSWEEP_COST_MODEL_CORE_HOURS[d]
        projected_total = cost["training_core_hours"] + cost["curvature_core_hours"]
        print(
            f"\n{'-' * 78}\n[d={d}] starting fit + field. cost-model estimate: "
            f"{projected_total:.3f} core-hr (~{projected_total / max(args.threads, 1):.3f}h @ "
            f"{args.threads}t).\n{'-' * 78}"
        )
        t_d0 = time.monotonic()

        fit = fit_and_field_at_anchors(
            X, d=d, anchor_idx=anchor_idx, in_dim=pcp.AE_IN_DIM, hidden=pcp.AE_HIDDEN,
            activation=pcp.AE_ACTIVATION, train_cfg=pcp.TRAIN_CFG, max_epochs=pcp.MAX_EPOCHS,
            torch_init_seed=pcp.TORCH_INIT_SEED, split_seed=pcp.SPLIT_SEED,
            holdout_fraction=pcp.HOLDOUT_FRACTION,
        )
        decomp = pcp.decompose_radial_tangential(fit["H_vec"], fit["image"], pcp.MIN_IMAGE_NORM)
        cond_g = fit["metric_condition_number"]
        cond_g_median = float(np.median(cond_g))
        h_rad_median = float(np.nanmedian(decomp["H_rad"]))

        print(
            f"[d={d}] fit done (elapsed so far {time.monotonic() - t_d0:.1f}s). "
            f"wallclock_fit={fit['wallclock_fit_s']:.1f}s wallclock_field={fit['wallclock_field_s']:.1f}s "
            f"var_explained={fit['var_explained']:.4f} cond(g) median={cond_g_median:.4e} "
            f"H_rad median={h_rad_median:.4f} (expected ~{-d})"
        )

        append_record_row(
            {
                "row_kind": "fit",
                "d": d,
                "var_explained": fit["var_explained"],
                "cond_g_median": cond_g_median,
                "cond_g_p95": float(np.percentile(cond_g, 95)),
                "H_rad_median": h_rad_median,
                "H_rad_expected": float(-d),
                "n_excluded_low_image_norm": decomp["n_excluded_low_norm"],
                "wallclock_fit_s": fit["wallclock_fit_s"],
                "wallclock_field_s": fit["wallclock_field_s"],
                "freeze_commit": freeze_commit,
                "run_commit": run_commit,
                "timestamp_utc": _utc_now(),
            },
            record_path,
        )

        H_tan_full_by_d[d] = decomp["H_tan_norm"]
        H_norm_full_by_d[d] = decomp["H_norm"]

        for name in all_labels:
            panel = panel_by_label[name]["panel"]
            anchor_table = build_anchor_table(
                anchor_idx=anchor_idx, decomp=decomp, cond_g=cond_g, panel=panel,
                mse=panel_by_label[name]["mse"], sst=panel_by_label[name]["sst"],
                log_knn_radius=knn["log_knn_radius"],
            )
            table_path = write_anchor_table(anchor_table, _anchor_table_path(output_root, d, name))

            append_record_row(
                {
                    "row_kind": "anchor_summary",
                    "d": d,
                    "label": name,
                    "gating": bool(name == pl.PRIMARY_LABEL),
                    "n_anchors": int(anchor_idx.shape[0]),
                    "n_masked_anchors": panel["n_masked_anchors"],
                    "local_evaluation_count_constant": panel_by_label[name]["const_eval"],
                    "anchor_table_path": str(table_path),
                    "freeze_commit": freeze_commit,
                    "run_commit": run_commit,
                    "timestamp_utc": _utc_now(),
                },
                record_path,
            )

            finite = np.isfinite(panel["r2"]) & np.isfinite(decomp["H_tan_norm"]) & np.isfinite(decomp["H_norm"])
            controls = controls_by_label[name]

            for field_name, field_full in (("H_tan_norm", decomp["H_tan_norm"]), ("H_norm", decomp["H_norm"])):
                x_f = field_full[finite]
                y_f = panel["r2"][finite]
                z_f = controls[finite]
                raw_rho = float(spearmanr(x_f, y_f).statistic) if x_f.size > 1 else float("nan")
                controlled = float(pcp.controlled_partial(x_f, y_f, z_f))
                gating = bool(name == pl.PRIMARY_LABEL and field_name == pcp.CURVATURE_FIELD_FOR_VERDICT)

                append_record_row(
                    {
                        "row_kind": "partial",
                        "d": d,
                        "label": name,
                        "field": field_name,
                        "gating": gating,
                        "n_finite_anchors": int(finite.sum()),
                        "raw_rho": raw_rho,
                        "controlled_partial": controlled,
                        "freeze_commit": freeze_commit,
                        "run_commit": run_commit,
                        "timestamp_utc": _utc_now(),
                    },
                    record_path,
                )

                for n_strata in pcp.STRATA_GRID:
                    strat = pcp.stratified_partial_null_3control(
                        x_f, y_f, z_f, knn["log_knn_radius"][finite], n_strata,
                        pcp.STRATIFIED_NULL_DRAWS, pcp.STRATIFIED_NULL_SEED,
                    )
                    append_record_row(
                        {
                            "row_kind": "null",
                            "null_type": "stratified",
                            "d": d,
                            "label": name,
                            "field": field_name,
                            "n_strata": n_strata,
                            "observed": strat["observed"],
                            "p": strat["p"],
                            "p_display": strat["p_display"],
                            "floor_reached": strat["floor_reached"],
                            "freeze_commit": freeze_commit,
                            "run_commit": run_commit,
                            "timestamp_utc": _utc_now(),
                        },
                        record_path,
                    )

                boot = pcp.paired_anchor_bootstrap(x_f, y_f, z_f, pcp.N_BOOTSTRAP, pcp.BOOTSTRAP_SEED)
                append_record_row(
                    {
                        "row_kind": "bootstrap",
                        "d": d,
                        "label": name,
                        "field": field_name,
                        "ci_low": boot["ci_low"],
                        "ci_high": boot["ci_high"],
                        "n_boot": boot["n_boot"],
                        "freeze_commit": freeze_commit,
                        "run_commit": run_commit,
                        "timestamp_utc": _utc_now(),
                    },
                    record_path,
                )

        print(f"[d={d}] total wallclock: {time.monotonic() - t_d0:.1f}s")

    # --- Family-wise Freedman-Lane envelope, ALL d at once, per label/field (D9-10) -------------
    # The envelope needs a common Freedman-Lane surrogate drawn once per permutation and applied
    # to every d's curvature array -- this can only happen after every d's field is known, hence
    # after the loop above, never inside it. Anchors excluded at ANY d (a low-image-norm row at
    # that d) are excluded from every d's array here, so the same physical anchor set backs the
    # whole envelope (never a shifting row set from one d to the next).
    for name in all_labels:
        panel = panel_by_label[name]["panel"]
        controls = controls_by_label[name]
        for field_name, field_by_d in (("H_tan_norm", H_tan_full_by_d), ("H_norm", H_norm_full_by_d)):
            finite = np.isfinite(panel["r2"])
            for d in pcp.D_SWEEP:
                finite = finite & np.isfinite(field_by_d[d])
            curvature_by_d = {d: field_by_d[d][finite] for d in pcp.D_SWEEP}
            y_f = panel["r2"][finite]
            z_f = controls[finite]
            fwer = pcp.permutation_fwer(curvature_by_d, y_f, z_f, pcp.N_PERMUTATIONS, pcp.PERMUTATION_SEED)

            for d in pcp.D_SWEEP:
                per_d = fwer["per_d"][d]
                append_record_row(
                    {
                        "row_kind": "null",
                        "null_type": "fwer",
                        "d": d,
                        "label": name,
                        "field": field_name,
                        "gating": bool(name == pl.PRIMARY_LABEL and field_name == pcp.CURVATURE_FIELD_FOR_VERDICT),
                        "n_finite_anchors": int(finite.sum()),
                        "observed_rho": per_d["observed_rho"],
                        "p": per_d["p"],
                        "p_display": per_d["p_display"],
                        "floor_reached": per_d["floor_reached"],
                        "freeze_commit": freeze_commit,
                        "run_commit": run_commit,
                        "timestamp_utc": _utc_now(),
                    },
                    record_path,
                )
            append_record_row(
                {
                    "row_kind": "null",
                    "null_type": "fwer_global",
                    "label": name,
                    "field": field_name,
                    "d_values": list(pcp.D_SWEEP),
                    "n_finite_anchors": int(finite.sum()),
                    "p": fwer["global"]["p"],
                    "p_display": fwer["global"]["p_display"],
                    "floor_reached": fwer["global"]["floor_reached"],
                    "freeze_commit": freeze_commit,
                    "run_commit": run_commit,
                    "timestamp_utc": _utc_now(),
                },
                record_path,
            )
            print(f"[fwer] label={name} field={field_name} global p_display={fwer['global']['p_display']}")

    print(f"\nDSWEEP done. Record: {record_path}.")
    return True


def run_positive_control(args: argparse.Namespace) -> bool:
    """`--mode positive-control` (D9-14): requires `--field-npz` naming a Wave A anchor table and
    refuses to regenerate one, following `07_crossmodal_curvature_run.py`'s own discipline
    (T-09-51). Plants on the curvature side at every entry of `POSITIVE_CONTROL_TARGET_RHOS`,
    read as MAGNITUDES straddling the colleague's observed `-0.240` -- the plant targets the
    NEGATIVE of each magnitude, matching this phase's own negative-association hypothesis
    (D9-09/D9-10) and `plant_curvature_positive_control`'s own direction note. Validates each
    planted array through the identical three-control partial and `permutation_fwer`'s
    Freedman-Lane construction (never `two_tailed_permutation_null`, Phase 7's own null), and
    reports the smallest cleared MAGNITUDE as the detection floor."""
    env = _gate_and_environment(args)

    if not args.field_npz:
        print(
            "ERROR: --mode positive-control requires --field-npz naming a Wave A anchor table "
            "(e.g. 09_anchor_table_d16_mag_r.npz, written by --mode dsweep); this mode refuses "
            "to regenerate a curvature field.",
            file=sys.stderr,
        )
        sys.exit(2)

    field_path = Path(args.field_npz)
    pcp._assert_inside_output_root(field_path)
    if not field_path.exists():
        print(
            f"ERROR: --mode positive-control: {field_path} does not exist -- --mode dsweep has "
            "not written a field there yet, and this mode refuses to regenerate one.",
            file=sys.stderr,
        )
        sys.exit(2)

    table = load_anchor_table(field_path)
    required_keys = {"H_tan_norm", "r2", "log_knn_radius", "local_label_variance", "local_evaluation_count"}
    missing_keys = required_keys - set(table.keys())
    if missing_keys:
        print(
            f"ERROR: --mode positive-control: {field_path} is missing key(s) {sorted(missing_keys)} "
            f"(found: {sorted(table.keys())}).",
            file=sys.stderr,
        )
        sys.exit(2)

    parsed = _parse_anchor_table_filename(field_path)
    freeze_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path, default_stem=pcp.RECORD_STEM)
    append_record_row(env, record_path)

    h_real = np.asarray(table["H_tan_norm"], dtype=np.float64)
    r2 = np.asarray(table["r2"], dtype=np.float64)
    controls = np.column_stack(
        [table["log_knn_radius"], table["local_label_variance"], table["local_evaluation_count"]]
    )
    finite = np.isfinite(h_real) & np.isfinite(r2)
    h_real_f, r2_f, controls_f = h_real[finite], r2[finite], controls[finite]

    print(
        f"\nPOSITIVE CONTROL: {h_real_f.shape[0]} finite anchors loaded from {field_path.name} "
        f"(d={parsed['d']} label={parsed['label']}), magnitudes={pcp.POSITIVE_CONTROL_TARGET_RHOS}, "
        f"seed={pcp.POSITIVE_CONTROL_SEED}.\n"
    )

    cleared_magnitudes = []
    for magnitude in pcp.POSITIVE_CONTROL_TARGET_RHOS:
        target_rho = -float(magnitude)  # plant toward the NEGATIVE (D9-09's own hypothesis)
        plant = pcp.plant_curvature_positive_control(
            h_real_f, r2_f, controls_f, target_rho=target_rho, seed=pcp.POSITIVE_CONTROL_SEED, n_bisect=40,
        )
        fwer = pcp.permutation_fwer(
            {0: plant["planted"]}, r2_f, controls_f, pcp.N_PERMUTATIONS, pcp.PERMUTATION_SEED,
        )
        per = fwer["per_d"][0]
        verdict = pcp.per_d_verdict(rho=per["observed_rho"], p_fwer=per["p"], fwer_alpha=pcp.FWER_ALPHA)
        cleared = bool(verdict == pcp.PER_D_VERDICT_VALUES[0])
        if cleared:
            cleared_magnitudes.append(float(magnitude))

        append_record_row(
            {
                "row_kind": "positive_control",
                "d": parsed["d"],
                "label": parsed["label"],
                "target_magnitude": float(magnitude),
                "target_rho": target_rho,
                "achieved_controlled_partial": plant["achieved_controlled_partial"],
                "slope": plant["slope"],
                "cleared": cleared,
                "p": per["p"],
                "p_display": per["p_display"],
                "floor_reached": per["floor_reached"],
                "field_npz": str(field_path),
                "freeze_commit": freeze_commit,
                "run_commit": run_commit,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            record_path,
        )
        print(
            f"  target_magnitude={magnitude:.3f} achieved={plant['achieved_controlled_partial']:.4f} "
            f"p_display={per['p_display']} cleared={cleared}"
        )

    if cleared_magnitudes:
        floor = min(cleared_magnitudes)
        print(f"\ndetection floor (smallest cleared magnitude): {floor}")
    else:
        print(
            "\ndetection floor: NONE CLEARED -- the instrument did not recover any planted "
            f"magnitude at this d/label."
        )
    return True


def run_shuffled_label(args: argparse.Namespace) -> bool:
    """`--mode shuffled-label` (D9-15): requires `--field-npz` likewise (refuses to regenerate a
    field). For `SHUFFLED_LABEL_REPEATS` repeats from `SHUFFLED_LABEL_SEED`, shuffles the label
    vector globally and recomputes ONLY the out-of-fold predictions and both label-derived
    controls -- the embedding matrix, curvature field and anchor index array stay
    byte-identical. `pcp.shuffled_label_repeat`'s own return signature does not expose the
    shuffled local-R2 panel (needed to build THIS repeat's own Freedman-Lane null for the
    false-positive count), so this mode composes the same sealed primitives
    (`oof_ridge_predictions`, `local_r2_panel`, `controlled_partial`) in the SAME order that
    function uses, consuming the SAME `rng` -- never a reimplementation of any formula, purely
    retaining the intermediate array the sealed wrapper discards (see the module docstring's own
    `plant_curvature_positive_control` precedent for composing sealed primitives in the runner)."""
    env = _gate_and_environment(args)

    if not args.field_npz:
        print(
            "ERROR: --mode shuffled-label requires --field-npz naming a Wave A anchor table "
            "(e.g. 09_anchor_table_d16_mag_r.npz, written by --mode dsweep); this mode refuses "
            "to regenerate a curvature field.",
            file=sys.stderr,
        )
        sys.exit(2)

    field_path = Path(args.field_npz)
    pcp._assert_inside_output_root(field_path)
    if not field_path.exists():
        print(
            f"ERROR: --mode shuffled-label: {field_path} does not exist -- --mode dsweep has "
            "not written a field there yet, and this mode refuses to regenerate one.",
            file=sys.stderr,
        )
        sys.exit(2)

    table = load_anchor_table(field_path)
    required_keys = {"H_tan_norm", "anchor_idx", "log_knn_radius"}
    missing_keys = required_keys - set(table.keys())
    if missing_keys:
        print(
            f"ERROR: --mode shuffled-label: {field_path} is missing key(s) {sorted(missing_keys)} "
            f"(found: {sorted(table.keys())}).",
            file=sys.stderr,
        )
        sys.exit(2)

    parsed = _parse_anchor_table_filename(field_path)
    freeze_commit = _git_rev_parse(args.freeze_commit)
    run_commit = _git_rev_parse("HEAD")
    record_path = resolve_record_path(args.record_path, default_stem=pcp.RECORD_STEM)
    append_record_row(env, record_path)

    emb = pl.load_physics_embeddings()
    X = emb["X"]
    n_rows = emb["n_rows"]
    label_table = pl.load_label_table(columns=list(pl.LABEL_COLUMN_MAP.values()))
    offset_perm = pl.shifted_pairing(n_rows, pl.ALIGNMENT_ASSUMED_OFFSET)
    y_raw = pl.canonical_label(label_table, pl.PRIMARY_LABEL, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)
    y_full = y_raw[offset_perm]

    anchor_idx = np.asarray(table["anchor_idx"], dtype=np.int64)
    knn = pcp.knn_panel(X, anchor_idx, pcp.K_NEIGHBOURS)
    h_field = np.asarray(table["H_tan_norm"], dtype=np.float64)
    log_knn_radius = np.asarray(table["log_knn_radius"], dtype=np.float64)

    print(
        f"\nSHUFFLED-LABEL: {pcp.SHUFFLED_LABEL_REPEATS} repeats from seed={pcp.SHUFFLED_LABEL_SEED}, "
        f"field={field_path.name} (d={parsed['d']} label={parsed['label']}).\n"
    )

    rng = np.random.default_rng(pcp.SHUFFLED_LABEL_SEED)
    n_cleared = 0
    for repeat in range(pcp.SHUFFLED_LABEL_REPEATS):
        perm = rng.permutation(y_full.shape[0])
        y_shuffled = y_full[perm]
        y_hat = _oof_predictions_for_label(X, y_shuffled, pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
        panel = pcp.local_r2_panel(y_shuffled, y_hat, knn["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        controls = np.column_stack(
            [log_knn_radius, panel["local_label_variance"], panel["local_evaluation_count"]]
        )
        finite = np.isfinite(panel["r2"]) & np.isfinite(h_field)
        controlled = float(pcp.controlled_partial(h_field[finite], panel["r2"][finite], controls[finite]))

        fwer = pcp.permutation_fwer(
            {0: h_field[finite]}, panel["r2"][finite], controls[finite], pcp.N_PERMUTATIONS, pcp.PERMUTATION_SEED,
        )
        per = fwer["per_d"][0]
        verdict = pcp.per_d_verdict(rho=per["observed_rho"], p_fwer=per["p"], fwer_alpha=pcp.FWER_ALPHA)
        cleared = bool(verdict == pcp.PER_D_VERDICT_VALUES[0])
        if cleared:
            n_cleared += 1

        append_record_row(
            {
                "row_kind": "shuffled_label",
                "d": parsed["d"],
                "label": parsed["label"],
                "repeat": repeat,
                "controlled_partial": controlled,
                "n_masked_anchors": panel["n_masked_anchors"],
                "cleared": cleared,
                "p": per["p"],
                "p_display": per["p_display"],
                "floor_reached": per["floor_reached"],
                "field_npz": str(field_path),
                "freeze_commit": freeze_commit,
                "run_commit": run_commit,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            record_path,
        )
        print(f"  repeat={repeat} controlled_partial={controlled:.4f} p_display={per['p_display']} cleared={cleared}")

    fp_rate = n_cleared / pcp.SHUFFLED_LABEL_REPEATS
    print(
        f"\nfalse-positive count: {n_cleared}/{pcp.SHUFFLED_LABEL_REPEATS} ({fp_rate:.3f}) vs "
        f"nominal FWER_ALPHA={pcp.FWER_ALPHA}"
    )
    return True


def run_verdict(args: argparse.Namespace) -> bool:
    """`--mode verdict` (D9-10/D9-18): reads the record and the anchor tables only and
    recomputes nothing, so the printed verdict cannot differ from the recorded numbers. Exits 2
    naming the missing gate row kind(s) when the record carries no `positive_control` row or no
    `shuffled_label` row -- the verdict has no scale before both gates have run (T-09-53). Prints
    the per-`d` table for `H_tan_norm` (gating) beside `H_norm` (non-gating), both nulls, the
    detection floor, the false-positive rate, the fit-quality read-out, the fidelity ranges, the
    neighbourhood ratio and the caveat-bearing verdict sentence -- REPORTING_BLOCK_ROWS order,
    unconditionally. Appends exactly one `verdict` record row; never overwrites a prior one."""
    env = _gate_and_environment(args)
    record_path = resolve_record_path(args.record_path, default_stem=pcp.RECORD_STEM)

    rows = []
    if record_path.exists():
        with record_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    has_positive_control = any(r.get("row_kind") == "positive_control" for r in rows)
    has_shuffled_label = any(r.get("row_kind") == "shuffled_label" for r in rows)
    if not has_positive_control or not has_shuffled_label:
        missing = [
            kind for kind, present in (
                ("positive_control", has_positive_control), ("shuffled_label", has_shuffled_label),
            ) if not present
        ]
        print(
            f"ERROR: --mode verdict refuses to print before both gates have run -- the record at "
            f"{record_path} carries no row(s) of kind {missing}. The positive control establishes "
            "the detection floor and the shuffled-label calibration establishes the "
            "false-positive rate; a verdict read before them has no scale.",
            file=sys.stderr,
        )
        sys.exit(2)

    append_record_row(env, record_path)

    gating_field = pcp.CURVATURE_FIELD_FOR_VERDICT
    print(f"\n{'=' * 78}\nPHASE 9 WAVE A VERDICT (reads the record only; recomputes nothing)\n{'=' * 78}\n")

    per_d_verdicts: Dict[int, str] = {}
    for d in pcp.D_SWEEP:
        partial_row = next(
            (r for r in rows if r.get("row_kind") == "partial" and r.get("d") == d
             and r.get("label") == pl.PRIMARY_LABEL and r.get("field") == gating_field), None,
        )
        fwer_row = next(
            (r for r in rows if r.get("row_kind") == "null" and r.get("null_type") == "fwer" and r.get("d") == d
             and r.get("label") == pl.PRIMARY_LABEL and r.get("field") == gating_field), None,
        )
        h_norm_partial_row = next(
            (r for r in rows if r.get("row_kind") == "partial" and r.get("d") == d
             and r.get("label") == pl.PRIMARY_LABEL and r.get("field") == "H_norm"), None,
        )
        if partial_row is None or fwer_row is None:
            print(f"[d={d}] no recorded {gating_field}/{pl.PRIMARY_LABEL} partial+fwer row -- skipping.")
            continue

        rho = partial_row["controlled_partial"]
        verdict = pcp.per_d_verdict(rho=rho, p_fwer=fwer_row["p"], fwer_alpha=pcp.FWER_ALPHA)
        per_d_verdicts[d] = verdict
        print(
            f"[d={d}] raw_rho={partial_row['raw_rho']:.6f} controlled_partial={rho:.6f} "
            f"fwer_p_display={fwer_row['p_display']} verdict={verdict}"
        )
        if h_norm_partial_row is not None:
            print(f"        [non-gating H_norm] controlled_partial={h_norm_partial_row['controlled_partial']:.6f}")

    phase = pcp.phase_verdict(per_d_verdicts)
    print(f"\nPER-D VERDICTS: {per_d_verdicts}")
    print(f"PHASE VERDICT: {phase}")

    pc_rows = [r for r in rows if r.get("row_kind") == "positive_control"]
    cleared_magnitudes = sorted({r["target_magnitude"] for r in pc_rows if r.get("cleared")})
    detection_floor = cleared_magnitudes[0] if cleared_magnitudes else None
    print(f"\nPOSITIVE CONTROL detection floor: {detection_floor}")

    sl_rows = [r for r in rows if r.get("row_kind") == "shuffled_label"]
    n_fp = sum(1 for r in sl_rows if r.get("cleared"))
    n_total = len(sl_rows)
    fp_rate = (n_fp / n_total) if n_total else float("nan")
    print(f"SHUFFLED-LABEL false-positive rate: {n_fp}/{n_total} ({fp_rate:.3f}) vs nominal FWER_ALPHA={pcp.FWER_ALPHA}")

    fidelity = {
        16: pcp.INSTRUMENT_FIDELITY_RANGE_D16,
        20: pcp.INSTRUMENT_FIDELITY_RANGE_D20,
        25: pcp.INSTRUMENT_FIDELITY_RANGE_D25,
        32: f"UNMEASURED -- {pcp.INSTRUMENT_FIDELITY_D32_RULE}",
    }
    print(f"\nInstrument fidelity ranges: {fidelity}")
    print(f"Neighbourhood ratio: {pcp.NEIGHBOURHOOD_RATIO_RULE}")

    last_fwer_global = next(
        (r for r in reversed(rows) if r.get("row_kind") == "null" and r.get("null_type") == "fwer_global"
         and r.get("label") == pl.PRIMARY_LABEL and r.get("field") == gating_field), None,
    )
    sentence = pcp.verdict_sentence(
        instrument="cae.PlainAutoEncoder + decoder_curvature.plain_decoder_curvature",
        d_values=pcp.D_SWEEP,
        colleague_rho=-0.2405,
        colleague_d=16,
        fwer_p_display=(last_fwer_global["p_display"] if last_fwer_global else "n/a"),
        stratified_p_display="see per-d 'null' rows (null_type='stratified')",
        instrument_fidelity_ranges=fidelity,
        neighbourhood_ratio=pcp.NEIGHBOURHOOD_RATIO_RULE,
    )
    print(f"\n{sentence}")

    append_record_row(
        {
            "row_kind": "verdict",
            "phase_verdict": phase,
            "per_d_verdicts": {str(k): v for k, v in per_d_verdicts.items()},
            "positive_control_detection_floor": detection_floor,
            "shuffled_label_false_positive_count": n_fp,
            "shuffled_label_repeats": n_total,
            "verdict_sentence": sentence,
            "freeze_commit": _git_rev_parse(args.freeze_commit),
            "run_commit": _git_rev_parse("HEAD"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        record_path,
    )

    print(f"\nVERDICT recorded to {record_path}.")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=[
            "smoke", "dsweep", "positive-control", "shuffled-label", "seeds", "verdict",
            "bundle", "selfcheck",
        ],
        default="smoke",
    )
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--freeze-commit", type=str, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--smoke-rows", type=int, default=2000)
    p.add_argument("--smoke-permutations", type=int, default=200)
    p.add_argument("--field-npz", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--print-cost-model", action="store_true")
    p.add_argument("--host-label", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.output_root and pcp.OUTPUT_ROOT_ENV_VAR:
        os.environ[pcp.OUTPUT_ROOT_ENV_VAR] = args.output_root

    if args.print_cost_model:
        print_cost_model(args.threads)
        sys.exit(0)

    if args.mode == "bundle":
        ok = run_bundle(args)
        sys.exit(0 if ok else 1)

    if args.mode == "smoke":
        ok = run_smoke(args)
        sys.exit(0 if ok else 1)

    if args.mode == "dsweep":
        ok = run_dsweep(args)
        sys.exit(0 if ok else 1)

    if args.mode == "positive-control":
        ok = run_positive_control(args)
        sys.exit(0 if ok else 1)

    if args.mode == "shuffled-label":
        ok = run_shuffled_label(args)
        sys.exit(0 if ok else 1)

    if args.mode == "verdict":
        ok = run_verdict(args)
        sys.exit(0 if ok else 1)

    plan = _MODE_IMPLEMENTING_PLAN.get(args.mode, "a later plan")
    print(
        f"ERROR: --mode {args.mode} is not implemented by this plan (09-08) -- it is added "
        f"by plan {plan}.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
