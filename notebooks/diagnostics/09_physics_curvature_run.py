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
from typing import Any, Dict, Optional

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
# "print-cost-model" were removed from this dict by 09-06 -- both are implemented below and
# dispatched directly in main(), never falling through to this "not implemented" table.
_MODE_IMPLEMENTING_PLAN = {
    "dsweep": "09-08",
    "positive-control": "09-08",
    "shuffled-label": "09-08",
    "verdict": "09-08",
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


def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    """Caller-supplied paths are routed through `pcp._assert_inside_output_root` (T-09-03); a
    traversal path raises rather than writes. No default is offered here -- `--mode smoke`
    requires an explicit `--record-path` and refuses to default onto any frozen record stem."""
    if record_path_arg is None:
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

    if args.mode != "smoke":
        plan = _MODE_IMPLEMENTING_PLAN.get(args.mode, "a later plan")
        print(
            f"ERROR: --mode {args.mode} is not implemented by this plan (09-01) -- it is added "
            f"by plan {plan}.",
            file=sys.stderr,
        )
        sys.exit(2)

    ok = run_smoke(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
