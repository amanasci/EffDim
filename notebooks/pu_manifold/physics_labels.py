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
callable parameter rather than importing one) and never imports ``torch``. This commit (09-03)
adds the module's own parquet-reading loaders (:func:`load_physics_embeddings`,
:func:`load_label_table`, :func:`label_missingness_report`): every value they read is either
overridden explicitly by the caller (the pre-freeze ``--mode manifest`` path) or resolved from
the still-UNSET constants above, so none of the three can run to completion until the freeze
fills the constants they depend on and no override is supplied.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import cache
from . import subsample

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
"""HuggingFace repo id for the label catalog, e.g. "Smith42/galaxies" -- always read at
LABEL_REVISION (e.g. "v2.0"), never at this repo's default revision."""

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


# =============================================================================================
# Loaders -- real HuggingFace reads. Every pre-registered value below (the parquet path/column
# for the embeddings side; the repo/revision/split/shard-count for the label side) defaults to
# reading the module-level UNSET constant, and every one of these three functions therefore
# cannot run to completion before the 09-05 freeze fills them, UNLESS the caller supplies an
# explicit override -- the pre-freeze `--mode manifest` path 09_row_alignment_proof_run.py adds.
# =============================================================================================

# Environment variable this module temporarily exports resolve_hf_cache_dir()'s resolved value
# into, for the duration of a read only, and only when not already set -- huggingface_hub's
# `hf://` filesystem handler consults HF_HOME directly; HF_CACHE_ENV_VARS may name a different
# variable (e.g. HF_DATASETS_CACHE) that resolve_hf_cache_dir checks first.
_HF_HOME_ENV_VAR = "HF_HOME"


def _require_label_source_constants() -> None:
    """Raise ``RuntimeError`` naming the first UNSET label-source constant among
    ``LABEL_REPO``, ``LABEL_REVISION``, ``LABEL_SPLIT`` and ``LABEL_N_SHARDS`` --
    :func:`load_label_table` and :func:`_shard_url` cannot resolve a shard URL before the
    freeze fills these (D9-18)."""
    for name in ("LABEL_REPO", "LABEL_REVISION", "LABEL_SPLIT", "LABEL_N_SHARDS"):
        value = globals()[name]
        if _is_unset(value):
            raise RuntimeError(
                f"_require_label_source_constants: {name}={value!r} is UNSET. No Physics "
                "number may be computed before the freeze (D9-18); see assert_preregistered."
            )


def _shard_url(index: int) -> str:
    """``hf://datasets/{LABEL_REPO}@{LABEL_REVISION}/data/{LABEL_SPLIT}-{index:05d}-of-
    {LABEL_N_SHARDS:05d}.parquet`` -- the revision lives in the URL fragment, not a keyword
    that a future caller could omit (RESEARCH.md Pitfall 1). Raises ``ValueError`` naming the
    index and the (possibly UNSET) shard count when ``index`` is outside
    ``range(LABEL_N_SHARDS)``, including when ``LABEL_N_SHARDS`` itself is UNSET."""
    n_shards = LABEL_N_SHARDS
    if n_shards is None or index not in range(n_shards):
        raise ValueError(
            f"_shard_url: index={index} is outside range(LABEL_N_SHARDS={n_shards!r})."
        )
    return (
        f"hf://datasets/{LABEL_REPO}@{LABEL_REVISION}/data/"
        f"{LABEL_SPLIT}-{index:05d}-of-{n_shards:05d}.parquet"
    )


class _hf_cache_env_override:
    """Context manager: exports :func:`resolve_hf_cache_dir`'s resolved value into
    :data:`_HF_HOME_ENV_VAR` for the duration of a read, ONLY when that variable is not
    already set in the environment -- an execution host that has already set ``HF_HOME`` (or
    any other :data:`HF_CACHE_ENV_VARS` entry the library itself consults) is never
    overridden. Restores the prior (absent) state on exit, never leaving a stray env var set
    across calls."""

    def __enter__(self) -> None:
        self._did_set = False
        resolved = resolve_hf_cache_dir()
        if resolved is not None and _HF_HOME_ENV_VAR not in os.environ:
            os.environ[_HF_HOME_ENV_VAR] = resolved
            self._did_set = True
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._did_set:
            os.environ.pop(_HF_HOME_ENV_VAR, None)


def load_physics_embeddings(
    parquet_path: Optional[str] = None,
    column: Optional[str] = None,
    expected_rows: Optional[int] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Column-projected read of the Physics ViT-B embeddings parquet. ``parquet_path``,
    ``column`` and ``expected_rows`` each default to ``None`` meaning "read the frozen
    constant" (``PHYSICS_PARQUET_PATH``, ``PHYSICS_COLUMN``, ``EXPECTED_N_PHYSICS_ROWS``); an
    explicit value overrides it -- the override path is what lets ``--mode manifest`` run
    before the freeze without weakening any post-freeze call site, which passes nothing and
    therefore gets the frozen values. Raises ``RuntimeError`` naming the UNSET constant(s) when
    no override is supplied and the frozen value is still UNSET.

    Reads with ``pyarrow.parquet.read_table(path, columns=[column])``, converts to a float64
    ``(n, 768)`` array, raises ``ValueError`` when the table has zero rows or the width is not
    768, then calls :func:`assert_expected_rows` against the resolved expected row count. When
    ``normalize`` is true the array is passed through ``subsample.l2_normalize`` and the raw
    (pre-normalization) row norms are returned alongside it, so the sphere premise the radial
    curvature decomposition rests on can be checked numerically later; ``EMBEDDING_NORMALIZATION``
    is recorded in the returned dict as provenance regardless.

    Caches the result under ``cache.npz_cache`` keyed on a cfg dict carrying the resolved URL,
    column name, expected row count and normalisation, so a re-run on the execution host does
    not re-download and a changed upstream file (a different resolved cfg) misses the cache
    rather than silently reusing a stale array.

    Returns ``{"X", "row_norm", "n_rows", "n_features", "source_url", "normalization"}``.
    """
    resolved_path = parquet_path if parquet_path is not None else PHYSICS_PARQUET_PATH
    resolved_column = column if column is not None else PHYSICS_COLUMN
    resolved_expected_rows = expected_rows if expected_rows is not None else EXPECTED_N_PHYSICS_ROWS
    if resolved_path is None or resolved_column is None or resolved_expected_rows is None:
        raise RuntimeError(
            "load_physics_embeddings: PHYSICS_PARQUET_PATH/PHYSICS_COLUMN/"
            "EXPECTED_N_PHYSICS_ROWS is UNSET and no override was supplied for it. No Physics "
            "number may be computed before the freeze (D9-18); see assert_preregistered."
        )

    normalization = EMBEDDING_NORMALIZATION if normalize else "none (normalize=False)"
    cfg = {
        "source_url": resolved_path,
        "column": resolved_column,
        "expected_rows": int(resolved_expected_rows),
        "normalize": bool(normalize),
        "normalization": normalization,
    }

    def _compute() -> Dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        with _hf_cache_env_override():
            table = pq.read_table(resolved_path, columns=[resolved_column])
        n_rows = table.num_rows
        if n_rows == 0:
            raise ValueError(
                f"load_physics_embeddings: read of {resolved_path!r} returned zero rows."
            )
        assert_expected_rows(n_rows, resolved_expected_rows, "physics embeddings")

        raw = np.asarray(table.column(resolved_column).to_pylist(), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 768:
            raise ValueError(
                f"load_physics_embeddings: expected width 768, got shape {raw.shape} from "
                f"column {resolved_column!r} of {resolved_path!r}."
            )

        if normalize:
            X, row_norm = subsample.l2_normalize(raw)
        else:
            X = raw
            row_norm = np.linalg.norm(raw, axis=1)
        return {"X": X, "row_norm": row_norm}

    stem = f"physics_embeddings_{cache.config_key(cfg)}"
    arrays = cache.npz_cache(stem, cfg, _compute)

    return {
        "X": arrays["X"],
        "row_norm": arrays["row_norm"],
        "n_rows": int(arrays["X"].shape[0]),
        "n_features": int(arrays["X"].shape[1]),
        "source_url": resolved_path,
        "normalization": normalization,
    }


def load_label_table(columns: Any, expected_rows: Optional[int] = None) -> Any:
    """Column-projected, revision-pinned read of ``LABEL_REPO@LABEL_REVISION``'s
    ``LABEL_SPLIT`` shards, concatenated in ascending shard-index order (the entire basis of
    the positional row-index join with the embeddings side --
    :data:`LABEL_SHARD_ORDER_RULE`). ``expected_rows`` defaults to ``None`` meaning
    ``EXPECTED_N_PHYSICS_ROWS``, overridable for the same pre-freeze reason
    :func:`load_physics_embeddings` documents. Raises ``RuntimeError`` naming the first UNSET
    label-source constant (:func:`_require_label_source_constants`) when the repo/revision/
    split/shard-count are not yet frozen.

    Before the first shard read, verifies shard 0's schema contains every requested column and
    raises ``KeyError`` naming the missing column(s), the repository and the revision when it
    does not -- so a future ``v3.0`` fails loudly instead of silently returning an empty label
    set. After concatenation, raises ``ValueError`` on a zero-row result and calls
    :func:`assert_expected_rows` against the resolved expected row count.

    Honours :func:`resolve_hf_cache_dir` by exporting its resolved value into the environment
    for the duration of the read only if not already set (:class:`_hf_cache_env_override`),
    never overriding a value the execution host chose.

    Returns a ``pandas.DataFrame``.
    """
    _require_label_source_constants()
    resolved_expected_rows = expected_rows if expected_rows is not None else EXPECTED_N_PHYSICS_ROWS
    if resolved_expected_rows is None:
        raise RuntimeError(
            "load_label_table: expected_rows was not supplied and EXPECTED_N_PHYSICS_ROWS is "
            "UNSET. No Physics number may be computed before the freeze (D9-18); see "
            "assert_preregistered."
        )

    column_list = list(columns)
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    with _hf_cache_env_override():
        for index in range(LABEL_N_SHARDS):
            url = _shard_url(index)
            if index == 0:
                schema_names = set(pq.read_schema(url).names)
                missing = [c for c in column_list if c not in schema_names]
                if missing:
                    raise KeyError(
                        f"load_label_table: column(s) {missing!r} are absent from "
                        f"{LABEL_REPO!r} at revision={LABEL_REVISION!r} (checked shard 0's "
                        f"schema: {url!r})."
                    )
            tables.append(pq.read_table(url, columns=column_list))

    full_table = pa.concat_tables(tables)
    frame = full_table.to_pandas()

    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError(
            f"load_label_table: concatenated read of {LABEL_N_SHARDS} shards from "
            f"{LABEL_REPO!r}@{LABEL_REVISION!r} returned zero rows."
        )
    assert_expected_rows(n_rows, resolved_expected_rows, "label table")
    return frame


def label_missingness_report(
    table: Any, column_map: Dict[str, str], sentinels: Any
) -> Dict[str, Dict[str, Any]]:
    """Counts only: no mean, no variance, no correlation, no regression (D9-18's manifest-mode
    prohibition -- this is the evidence 09-04's blocking checkpoint reads, and dataset metadata
    is not a Physics number).

    For every canonical name in ``column_map``, resolves the raw column (raising ``KeyError``
    naming the canonical name and the raw column when it is absent from ``table``), and reports
    ``{"raw_column", "n_total", "n_finite_raw", "n_sentinel", "n_finite_masked",
    "fraction_finite"}``. ``n_finite_raw`` counts finite entries BEFORE sentinel masking;
    ``n_sentinel`` counts entries equal to any of ``sentinels``; ``n_finite_masked`` counts
    finite entries after :func:`mask_sentinels`. A column that is entirely sentinel reports
    ``n_finite_masked == 0`` rather than raising, so the manifest can record the fact."""
    sentinel_list = list(sentinels)
    sentinel_arr = np.asarray(sentinel_list, dtype=np.float64) if sentinel_list else None

    report: Dict[str, Dict[str, Any]] = {}
    for canonical_name, raw_column in column_map.items():
        columns = getattr(table, "columns", None)
        has_column = (raw_column in columns) if columns is not None else (raw_column in table)
        if not has_column:
            raise KeyError(
                f"label_missingness_report: canonical name={canonical_name!r} resolves to raw "
                f"column={raw_column!r}, which is absent from the table."
            )
        values = np.asarray(table[raw_column], dtype=np.float64)
        n_total = int(values.shape[0])
        n_finite_raw = int(np.sum(np.isfinite(values)))
        n_sentinel = int(np.sum(np.isin(values, sentinel_arr))) if sentinel_arr is not None else 0
        n_finite_masked = int(np.sum(np.isfinite(mask_sentinels(values, sentinels))))
        report[canonical_name] = {
            "raw_column": raw_column,
            "n_total": n_total,
            "n_finite_raw": n_finite_raw,
            "n_sentinel": n_sentinel,
            "n_finite_masked": n_finite_masked,
            "fraction_finite": (n_finite_masked / n_total) if n_total > 0 else 0.0,
        }
    return report
