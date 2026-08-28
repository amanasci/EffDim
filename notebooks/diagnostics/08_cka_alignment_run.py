"""Phase 8 CKA alignment runner. `--mode selfcheck` (08-01) is the tracer: drives the Song et
al. (2012) unbiased-HSIC / CKA estimator through D8-16's invariance ladder for BOTH kernels
(D8-01) on synthetic pairs generated in-process from a fixed RNG seed -- no PU data is opened,
no subset exists, no null is built. It does NOT call `cka.assert_preregistered()`: it is a pure
in-memory known-answer check, following 07.1's own `--mode smoke` convention. It prints one line
per invariance-ladder rung and appends exactly one JSONL row per rung to a scratch record.
`--mode sigma` (08-03), `--mode sweep` (08-05), `--mode positive-control` (08-05) and
`--mode negative-control` (08-05) are not implemented in this plan and exit 2, naming the plan
that implements each.

Usage:
    python notebooks/diagnostics/08_cka_alignment_run.py --mode selfcheck --record-path notebooks/.cache/08_scratch_tracer.jsonl
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
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np  # noqa: E402

from pu_manifold import cache  # noqa: E402
from pu_manifold import cka  # noqa: E402


# Modes not implemented by this plan (08-01): the plan that will implement each.
NOT_YET_IMPLEMENTED_MODES: Dict[str, str] = {
    "sigma": "08-03",
    "sweep": "08-05",
    "positive-control": "08-05",
    "negative-control": "08-05",
}

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


def resolve_record_path(record_path_arg: Optional[str]) -> Path:
    """A supplied value is passed through `cache._assert_inside_cache` before it is ever
    opened, so a traversal path raises rather than writes -- copying 07.1's
    `resolve_record_path` shape. Unlike 07.1, this runner has no frozen record to default onto
    (Phase 8 has no freeze yet), so a missing value is the caller's error, checked at each mode's
    own call site."""
    if record_path_arg is None:
        raise ValueError("resolve_record_path: no --record-path was supplied.")
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

    if args.mode in NOT_YET_IMPLEMENTED_MODES:
        print(
            f"ERROR: --mode {args.mode} is not implemented in this plan (08-01); it lands in "
            f"plan {NOT_YET_IMPLEMENTED_MODES[args.mode]}.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.mode == "selfcheck":
        ok = run_selfcheck(args)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
