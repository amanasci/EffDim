"""Phase 9 data-loading and row-alignment module: the `UniverseTBD/pu-embeddings`
``physics_vit_base_test`` embeddings and the `Smith42/galaxies@v2.0` catalog labels share no
``object_id`` and no explicit join key -- exactly the situation ``subsample.py`` documents for
the paired HSC/Legacy-Survey columns, except this time the two sides come from two SEPARATE
HuggingFace datasets rather than two columns of one dataset. Row order is therefore a
CONVENTION, never an assumption: :func:`alignment_r2_curve` and :func:`alignment_verdict` are
D9-06's runtime statistical proof that the convention holds (an out-of-fold R2 curve over a
shift set, with shift 0 required to win by a pre-registered margin), not a structural check
alone -- equal row counts (:func:`assert_expected_rows`) catch a silently changed upstream file,
but they are explicitly NOT the alignment proof (D9-05).

**This module adds; it does not edit.** No sealed ``pu_manifold`` module is imported for a
gating VALUE here -- every constant this module needs is a fresh top-level literal, declared
even where a value happens to coincide with a sealed module's own (the fresh-redeclaration
discipline ``density_stratified_null.py`` documents at its own lines 30-39).

**The constants below are UNSET in this commit.** Every one is ``None`` (scalar), ``()``
(tuple), ``""`` (rule string) or ``{}`` (mapping) until Phase 9's single freeze commit (09-05,
D9-18) fills them. :func:`assert_preregistered` raises ``RuntimeError`` on the first UNSET
constant it finds, in declaration order, so no Physics number can be computed against a build of
this module that predates the freeze. A later edit to any of these constants after a Physics
number exists anywhere in the tree is a pre-registration BREACH -- the only remedy is a fresh
freeze and a full re-run, never a silent fix.

This module never imports the sibling Phase 9 statistics module (the two modules stay acyclic
by dependency injection: :func:`alignment_r2_curve` takes the out-of-fold estimator as a
callable parameter rather than importing one), never imports ``torch``, and performs no network
access or parquet read of its own -- callers wire this module's pure functions to whatever
loader 09-03 adds.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

# =============================================================================================
# Constants -- ALL UNSET in this commit. Filled only by Phase 9's single freeze commit (09-05).
# =============================================================================================

PHYSICS_REPO = None
"""HuggingFace repo id for the Physics ViT-B embeddings, e.g. "UniverseTBD/pu-embeddings"."""

PHYSICS_CONFIG = None
"""The `physics_*_test` config name within PHYSICS_REPO."""

PHYSICS_PARQUET_PATH = None
"""The `hf://datasets/...` column-projected parquet path pattern for the embeddings side."""

PHYSICS_COLUMN = None
"""The embedding column name within the Physics config (e.g. a `<model>_galaxies` column)."""

EXPECTED_N_PHYSICS_ROWS = None
"""Row count the Physics test split must report exactly -- catches a silently changed upstream
file (D9-05). NOT the alignment proof; see assert_expected_rows's own docstring."""

EMBEDDING_NORMALIZATION = ""
"""Rule string stating whether/how the embedding matrix is normalized before any statistic."""

LABEL_REPO = None
"""HuggingFace repo id for the label catalog, e.g. "Smith42/galaxies"."""

LABEL_REVISION = None
"""The pinned revision string (e.g. "v2.0") -- the default branch/revision silently lacks every
label column (09-RESEARCH.md Pitfall 1), so this must never be left as the library default."""

LABEL_SPLIT = None
"""The split name within LABEL_REPO@LABEL_REVISION to read."""

LABEL_N_SHARDS = None
"""Number of parquet shard files the label split is stored across."""

LABEL_SHARD_ORDER_RULE = ""
"""Rule string stating shards are concatenated in ascending index order and that this order is
the entire basis of the positional row-index join with the embeddings side."""

LABEL_COLUMN_MAP = {}
"""Canonical label name -> raw catalog column name, e.g. {"mag_r": "mag_r_desi"}."""

LABEL_COLUMN_MAP_PROVENANCE = {}
"""Canonical label name -> a short string recording why that raw column was chosen (this
phase's own documented convention; the colleague's labels-build script is absent from his
branch, so this cannot be confirmed byte-for-byte against it)."""

PRIMARY_LABEL = None
"""The gating label -- the one whose local out-of-fold R2 the headline curvature statistic is
computed against."""

SECONDARY_LABELS = ()
"""Non-gating labels reported alongside the primary label."""

SECONDARY_LABELS_ARE_NON_GATING = None
"""True: SECONDARY_LABELS never gate the phase verdict."""

EXCLUDED_LABELS = ()
"""Catalog labels considered and explicitly excluded from this phase (e.g. for coverage)."""

EXCLUDED_LABELS_RULE = ""
"""Rule string recording why each EXCLUDED_LABELS entry was excluded."""

SENTINEL_VALUES = ()
"""Sentinel values (e.g. -99.0) that mean "missing" in the raw catalog and must be masked to
NaN before any statistic -- applied by mask_sentinels before any mean/variance/ridge fit."""

ALIGNMENT_LABEL = None
"""The single label alignment_r2_curve is run against for D9-06/07's row-alignment proof."""

ALIGNMENT_SHIFT_SET = ()
"""The frozen non-zero shift set D9-07's alignment curve is evaluated over, in addition to
shift 0 (supplied separately by the runner so the assumed alignment is never just one entry in
a list)."""

ALIGNMENT_N_PERMUTATIONS = None
"""Permutation draws alignment_r2_curve adds beyond the shift set, for a null comparison."""

ALIGNMENT_PERMUTATION_SEED = None
"""Seed for the permutation draws above."""

ALIGNMENT_MARGIN_R2 = None
"""D9-07's pre-registered margin: alignment_verdict's `passed` requires
`gap = r2_shift0 - best_other_r2` to STRICTLY exceed this margin."""

ALIGNMENT_PASS_RULE = ""
"""Exact-equality-guarded rule string (see _REQUIRED_ALIGNMENT_PASS_RULE below): states the
strict `>` gap comparison -- a gap exactly equal to the margin FAILS."""

ALIGNMENT_SEARCH_RULE = ""
"""Exact-equality-guarded rule string (see _REQUIRED_ALIGNMENT_SEARCH_RULE below): states the
D9-08 SEARCH branch adopts a non-zero alignment only when EXACTLY ONE shift clears the margin;
two or more clearing shifts is AMBIGUOUS and halts rather than picking one."""

ALIGNMENT_ASSUMED_OFFSET = None
"""The assumed row offset (0) between the two sources. Any other value may only arrive through
a numbered amendment and a fresh freeze (D9-08)."""

HF_CACHE_ENV_VARS = ()
"""Environment variable names checked, in order, for a HuggingFace cache directory override --
the execution-host knob resolve_hf_cache_dir reads."""

MANIFEST_RECORD_STEM = None
"""Record stem for the D9-05 data-manifest JSONL."""

ALIGNMENT_RECORD_STEM = None
"""Record stem for the D9-06/07 row-alignment JSONL."""


# --- Canonical values for the two exact-equality-guarded rule strings above -------------------
# This module owns the meaning of ALIGNMENT_PASS_RULE and ALIGNMENT_SEARCH_RULE (it implements
# alignment_verdict, the function whose behaviour they describe), so it also owns the required
# canonical text 09-05's freeze must copy verbatim -- mirroring linear_probe.py's
# SEED_HANDLING_RULE idiom (an exact-equality guard, never a truthiness check) in spirit.

_REQUIRED_ALIGNMENT_PASS_RULE = (
    "passed is True iff gap = r2_shift0 - best_other_r2 is STRICTLY greater than "
    "ALIGNMENT_MARGIN_R2; a gap exactly equal to the margin FAILS."
)

_REQUIRED_ALIGNMENT_SEARCH_RULE = (
    "the D9-08 SEARCH branch adopts a non-zero alignment only when exactly one shift's own gap "
    "over the remaining maximum clears ALIGNMENT_MARGIN_R2; two or more clearing shifts is "
    "AMBIGUOUS and halts rather than picking one."
)


_REQUIRED_CONSTANTS = (
    "PHYSICS_REPO", "PHYSICS_CONFIG", "PHYSICS_PARQUET_PATH", "PHYSICS_COLUMN",
    "EXPECTED_N_PHYSICS_ROWS", "EMBEDDING_NORMALIZATION",
    "LABEL_REPO", "LABEL_REVISION", "LABEL_SPLIT", "LABEL_N_SHARDS", "LABEL_SHARD_ORDER_RULE",
    "LABEL_COLUMN_MAP", "LABEL_COLUMN_MAP_PROVENANCE", "PRIMARY_LABEL", "SECONDARY_LABELS",
    "SECONDARY_LABELS_ARE_NON_GATING", "EXCLUDED_LABELS", "EXCLUDED_LABELS_RULE",
    "SENTINEL_VALUES",
    "ALIGNMENT_LABEL", "ALIGNMENT_SHIFT_SET", "ALIGNMENT_N_PERMUTATIONS",
    "ALIGNMENT_PERMUTATION_SEED", "ALIGNMENT_MARGIN_R2", "ALIGNMENT_PASS_RULE",
    "ALIGNMENT_SEARCH_RULE", "ALIGNMENT_ASSUMED_OFFSET",
    "HF_CACHE_ENV_VARS", "MANIFEST_RECORD_STEM", "ALIGNMENT_RECORD_STEM",
)


def _is_unset(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (tuple, list)) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


def assert_preregistered() -> None:
    """Raise ``RuntimeError`` on the FIRST unset entry of :data:`_REQUIRED_CONSTANTS`, checked
    in declaration order -- one check per constant, mirroring ``linear_probe.py``'s
    per-constant-message idiom rather than ``crossmodal_curvature.py``'s collect-then-raise
    shape. ``None``, an empty tuple/list, an empty dict and an empty-or-whitespace-only string
    are all treated as UNSET.

    Once every constant above is non-empty, two additional checks fire by EXACT STRING
    EQUALITY, never by truthiness (copying ``linear_probe.py``'s ``SEED_HANDLING_RULE`` idiom in
    spirit): ``ALIGNMENT_PASS_RULE`` must equal :data:`_REQUIRED_ALIGNMENT_PASS_RULE` and
    ``ALIGNMENT_SEARCH_RULE`` must equal :data:`_REQUIRED_ALIGNMENT_SEARCH_RULE` -- a reworded
    rule string fails this guard even though it is non-empty.
    """
    g = globals()
    for name in _REQUIRED_CONSTANTS:
        if name not in g:
            raise RuntimeError(f"assert_preregistered: {name} is absent from physics_labels.")
        value = g[name]
        if _is_unset(value):
            raise RuntimeError(
                f"assert_preregistered: {name}={value!r} is UNSET. No Physics number may be "
                "computed before the freeze (D9-18)."
            )

    if g["ALIGNMENT_PASS_RULE"] != _REQUIRED_ALIGNMENT_PASS_RULE:
        raise RuntimeError(
            f"assert_preregistered: ALIGNMENT_PASS_RULE={g['ALIGNMENT_PASS_RULE']!r} does not "
            f"equal the required text {_REQUIRED_ALIGNMENT_PASS_RULE!r}."
        )
    if g["ALIGNMENT_SEARCH_RULE"] != _REQUIRED_ALIGNMENT_SEARCH_RULE:
        raise RuntimeError(
            f"assert_preregistered: ALIGNMENT_SEARCH_RULE={g['ALIGNMENT_SEARCH_RULE']!r} does "
            f"not equal the required text {_REQUIRED_ALIGNMENT_SEARCH_RULE!r}."
        )


# =============================================================================================
# Compute functions. Pure, no file I/O of their own (callers do the parquet/HF reads), and no
# defaults on any pre-registered parameter -- a default is how a pre-registered value gets
# inherited by accident instead of chosen explicitly at every call site.
# =============================================================================================

# Assumed OOF fold count used only to size the "not a vacuous alignment" finite-row floor below
# -- a structural sanity floor, not a pre-registered scientific parameter.
_ALIGNMENT_ASSUMED_FOLD_COUNT = 5


def resolve_hf_cache_dir() -> Optional[str]:
    """The first of :data:`HF_CACHE_ENV_VARS` set (non-empty) in the environment, else ``None``
    meaning the library default. Reads the environment only; sets nothing. The execution host
    may point this at a different disk (09-06); the library default is preserved when unset."""
    for var in HF_CACHE_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def mask_sentinels(y: Any, sentinels: Any) -> np.ndarray:
    """Float64 copy of ``y`` with every value equal to any entry of ``sentinels``, and every
    non-finite value, set to ``np.nan``. Guard first: raise ``ValueError`` on a non-1-D input."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"mask_sentinels: y must be one-dimensional; got shape {arr.shape}.")
    out = arr.copy()
    sentinel_list = list(sentinels)
    if sentinel_list:
        sentinel_arr = np.asarray(sentinel_list, dtype=np.float64)
        is_sentinel = np.isin(out, sentinel_arr)
        out[is_sentinel] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def assert_expected_rows(n_seen: int, n_expected: int, what: str) -> None:
    """Exact integer comparison, raising ``ValueError`` naming both counts and ``what``. Equal
    row count is NOT the alignment proof (D9-05) -- this check exists only to catch a silently
    changed upstream file."""
    if int(n_seen) != int(n_expected):
        raise ValueError(
            f"assert_expected_rows: {what} has n_seen={int(n_seen)} rows; expected "
            f"n_expected={int(n_expected)}. This is NOT the alignment proof (D9-05) -- it only "
            "catches a silently changed upstream file."
        )


def canonical_label(table: Any, name: str, column_map: Dict[str, str], sentinels: Any) -> np.ndarray:
    """Resolve ``name`` through ``column_map``, raise ``KeyError`` naming both the canonical
    name, the resolved raw column and the revision when the column is absent, then return
    :func:`mask_sentinels` of the column as float64."""
    if name not in column_map:
        raise KeyError(
            f"canonical_label: canonical name={name!r} is not a key of column_map "
            f"(known: {sorted(column_map)!r})."
        )
    raw_column = column_map[name]
    columns = getattr(table, "columns", None)
    has_column = (raw_column in columns) if columns is not None else (raw_column in table)
    if not has_column:
        raise KeyError(
            f"canonical_label: canonical name={name!r} resolves to raw column={raw_column!r} "
            f"at revision={LABEL_REVISION!r}, which is absent from the table."
        )
    values = np.asarray(table[raw_column], dtype=np.float64)
    return mask_sentinels(values, sentinels)


def shifted_pairing(n: int, shift: int) -> np.ndarray:
    """``(np.arange(n) + shift) % n`` -- every alignment uses all ``n`` pairs and no row is
    dropped at any shift magnitude. Raise ``ValueError`` when ``n < 1``."""
    if n < 1:
        raise ValueError(f"shifted_pairing: n={n} must be at least 1.")
    return (np.arange(n) + shift) % n


def alignment_r2_curve(
    X: np.ndarray,
    y: np.ndarray,
    shifts: Any,
    n_permutations: int,
    permutation_seed: int,
    oof_fn: Any,
) -> List[Dict[str, Any]]:
    """The D9-06/07 row-alignment R2 curve. ``oof_fn`` is a REQUIRED callable parameter taking
    ``(X, y)`` and returning the out-of-fold prediction array, so this module never imports the
    sibling Phase 9 statistics module and the two modules stay acyclic.

    For each shift in ``shifts`` (evaluated and recorded in the given order, shift 0 first)
    build ``y_shifted = y[shifted_pairing(n, shift)]``, drop rows where ``y_shifted`` is NaN,
    call ``oof_fn``, and record ``{"alignment": "shift", "shift": int, "r2": float,
    "n_finite": int}``. Then for ``n_permutations`` draws from
    ``np.random.default_rng(permutation_seed)`` record the same 4-field shape with
    ``"alignment": "permutation"`` and the draw index under ``"draw"`` in place of ``"shift"``.

    Raises ``ValueError`` when ``shifts`` is empty -- an empty shift set must never yield a
    vacuous PASS. Raises ``ValueError`` when any alignment leaves fewer finite rows than five
    times the assumed OOF fold count (a structural sanity floor, not a pre-registered value)."""
    shifts = list(shifts)
    if len(shifts) == 0:
        raise ValueError(
            "alignment_r2_curve: shifts must be non-empty -- an empty shift set must never "
            "yield a vacuous PASS."
        )
    X = np.asarray(X, dtype=np.float64)
    y_full = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    if y_full.shape[0] != n:
        raise ValueError(
            f"alignment_r2_curve: X has {n} rows but y has {y_full.shape[0]} rows."
        )

    min_finite = 5 * _ALIGNMENT_ASSUMED_FOLD_COUNT

    def _one_row(kind: str, label: int, y_variant: np.ndarray) -> Dict[str, Any]:
        finite = np.isfinite(y_variant)
        n_finite = int(finite.sum())
        if n_finite < min_finite:
            raise ValueError(
                f"alignment_r2_curve: alignment {kind}={label} leaves {n_finite} finite rows, "
                f"fewer than the {min_finite}-row floor (5x the assumed OOF fold count)."
            )
        X_f = X[finite]
        y_f = y_variant[finite]
        y_hat = np.asarray(oof_fn(X_f, y_f), dtype=np.float64).ravel()
        ss_res = float(np.sum((y_f - y_hat) ** 2))
        ss_tot = float(np.sum((y_f - y_f.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        return {"r2": r2, "n_finite": n_finite}

    curve: List[Dict[str, Any]] = []
    for shift in shifts:
        y_shifted = y_full[shifted_pairing(n, shift)]
        result = _one_row("shift", int(shift), y_shifted)
        curve.append(
            {"alignment": "shift", "shift": int(shift), "r2": result["r2"], "n_finite": result["n_finite"]}
        )

    rng = np.random.default_rng(permutation_seed)
    for draw_idx in range(n_permutations):
        perm = rng.permutation(n)
        y_perm = y_full[perm]
        result = _one_row("permutation", draw_idx, y_perm)
        curve.append(
            {"alignment": "permutation", "draw": int(draw_idx), "r2": result["r2"], "n_finite": result["n_finite"]}
        )

    return curve


def alignment_verdict(curve: List[Dict[str, Any]], margin: float) -> Dict[str, Any]:
    """``passed`` is ``gap > margin`` using a STRICT ``>`` (a gap exactly equal to the margin
    FAILS, mirroring ``subsample.ALIGNMENT_MARGIN_Z``'s own documented strict comparison).
    ``clearing_alignments`` lists every non-zero shift whose OWN gap over the remaining maximum
    (every other shift row's r2) clears ``margin`` -- the input D9-08's SEARCH branch reads.
    Raise ``ValueError`` when ``curve`` contains no shift-0 row."""
    shift_rows = [row for row in curve if row.get("alignment") == "shift"]
    zero_rows = [row for row in shift_rows if row["shift"] == 0]
    if not zero_rows:
        raise ValueError("alignment_verdict: curve contains no shift-0 row.")
    r2_shift0 = zero_rows[0]["r2"]

    other_rows = [row for row in shift_rows if row["shift"] != 0]
    if other_rows:
        best_other_row = max(other_rows, key=lambda row: row["r2"])
        best_other_alignment = best_other_row["shift"]
        best_other_r2 = best_other_row["r2"]
    else:
        best_other_alignment = None
        best_other_r2 = float("-inf")

    gap = r2_shift0 - best_other_r2
    passed = gap > margin

    clearing_alignments = []
    for row in other_rows:
        remaining = [r["r2"] for r in shift_rows if r is not row]
        remaining_max = max(remaining) if remaining else float("-inf")
        if row["r2"] - remaining_max > margin:
            clearing_alignments.append(row["shift"])
    clearing_alignments = sorted(clearing_alignments)

    return {
        "r2_shift0": r2_shift0,
        "best_other_alignment": best_other_alignment,
        "best_other_r2": best_other_r2,
        "gap": gap,
        "passed": bool(passed),
        "clearing_alignments": clearing_alignments,
    }
