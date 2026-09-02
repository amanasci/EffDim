"""Phase 9 curvature-conditioned label-decodability statistics module: the pre-registration
constants block, its guard, the OOF ridge wrapper, the anchor draw, the radial/tangential
decomposition, the 3-control partial Spearman, the Freedman-Lane null, and the verdict rules.

**This module adds; it does not edit.** ``notebooks/pu_manifold/crossmodal_curvature.py``
(Phase 7, sealed) and ``notebooks/pu_manifold/density_stratified_null.py`` (Phase 07.1, sealed)
are never imported for a gating VALUE here -- every constant this module needs is a fresh
top-level literal, declared even where the value is identical to Phase 7's own (D_SWEEP,
AE_IN_DIM, AE_HIDDEN, AE_ACTIVATION, TRAIN_CFG, SPLIT_SEED, HOLDOUT_FRACTION,
CURVATURE_SOURCE_FUNCTION, CURVATURE_CONVENTION all happen to coincide with Phase 7's, and are
still re-declared fresh here). This module MAY import pure functions from those and other
sealed modules -- ``crossmodal_curvature.split_indices``, ``subsample.draw_row_indices``,
``subsample.l2_normalize``, ``cross_split_curvature.partial_spearman``,
``density_stratified_null.density_strata``, ``linear_probe.fit_probe``/``predict_probe``,
``decoder_curvature.plain_decoder_curvature`` -- it is only the pre-registered VALUES that must
never cross a freeze boundary.

**The constants below are UNSET in this commit.** Every one is ``None`` (scalar), ``()``
(tuple), ``""`` (rule string) or ``{}`` (mapping) until Phase 9's single freeze commit (09-05,
D9-18) fills them, with one deliberate exception: ``SWISS_ROLL_APPLICABILITY_RULE`` is filled
NOW, because it is a non-gating declarative fact about this phase's own methodology (it produces
no Physics number and gates nothing), not a value the freeze adjudicates -- the same status
``DIAGNOSTICS_ARE_NON_GATING = True`` has in ``crossmodal_curvature.py`` from that module's own
first commit. :func:`assert_preregistered` still validates it is well-formed as part of the
same sweep, so a future accidental un-set would still be caught.

Any later edit to a gating constant after a Physics number exists anywhere in the tree is a
pre-registration BREACH -- the only remedy is a fresh freeze and a full re-run, never a silent
fix. This module imports **no torch anywhere**, so the freeze guard stays importable without the
training stack and the test suite stays fast; the training stack lives in the runner
(``09_physics_curvature_run.py``), exactly as ``07_crossmodal_curvature_run.py`` already does
with its own ``fit_and_field``.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import cache
from . import cross_split_curvature
from . import crossmodal_curvature
from . import density_stratified_null
from . import linear_probe
from . import subsample

# =============================================================================================
# Constants -- UNSET in this commit except SWISS_ROLL_APPLICABILITY_RULE (see module docstring).
# Filled by Phase 9's single freeze commit (09-05, D9-18).
# =============================================================================================

K_NEIGHBOURS = None
"""Neighbourhood size for knn_panel's per-anchor query (D9-02)."""

NEIGHBOURHOOD_RATIO_RULE = ""
"""Rule string stating K_NEIGHBOURS as a fraction of the Physics sample and requiring that
ratio be printed beside every number."""

N_ANCHORS = None
"""Anchor count drawn from the AE holdout pool (D9-03)."""

ANCHOR_DRAW_SEED = None
"""Seed passed to subsample.draw_row_indices for the anchor draw."""

ANCHOR_POOL = None
"""Which row pool anchors are drawn from -- e.g. "ae_holdout_rows_only" (D9-04)."""

ANCHOR_POOL_RULE = ""
"""Rule string stating the deliberate departure from Phase 7's FIELD_EVALUATED_ON convention:
curvature is measured only where the decoder never trained."""

SPLIT_SEED = None
"""Seed for the AE train/holdout split (crossmodal_curvature.split_indices)."""

HOLDOUT_FRACTION = None
"""Holdout fraction for the AE train/holdout split."""

AE_IN_DIM = None
"""Autoencoder input width."""

AE_HIDDEN = ()
"""Autoencoder hidden-layer widths."""

AE_ACTIVATION = None
"""Autoencoder activation name."""

MAX_EPOCHS = None
"""Autoencoder training epoch cap."""

TORCH_INIT_SEED = None
"""torch.manual_seed value at model construction."""

TRAIN_CFG = {}
"""Autoencoder training protocol dict (lr, weight_decay, batch, ...)."""

CURVATURE_SOURCE_FUNCTION = None
"""Dotted name of the curvature function this phase uses."""

CURVATURE_CONVENTION = None
"""Must equal "trace" once filled -- H = tr_g(II), never the averaged convention."""

D_SWEEP = ()
"""The d values this phase fits and reports (D9-12)."""

FIT_QUALITY_KEYS = ()
"""Which fit-quality diagnostics are recorded per d."""

INSTRUMENT_FIDELITY_RANGE_D16 = ()
INSTRUMENT_FIDELITY_RANGE_D20 = ()
INSTRUMENT_FIDELITY_RANGE_D25 = ()
INSTRUMENT_FIDELITY_D32_RULE = ""
"""Analytic-fixture instrument-fidelity ranges per d, and (for d=32) a rule string stating why
it is unmeasurable at this milestone's fixture width."""

CURVATURE_FIELD_FOR_VERDICT = None
"""Exact-equality-guarded (see _REQUIRED_CURVATURE_FIELD_FOR_VERDICT below): the single field
name the verdict functions read -- "H_tan_norm", never "H_norm"."""

H_NORM_IS_NON_GATING = None
"""True: ||H|| is reported beside ||H_tan|| and never promoted to the headline (D9-11)."""

RADIAL_DECOMPOSITION_RULE = ""
"""Rule string naming the decompose_radial_tangential formula this phase uses."""

MIN_IMAGE_NORM = None
"""Rows whose decoder-image norm falls below this are excluded from the decomposition rather
than divided by."""

ALPHA_RIDGE = None
"""The single pinned ridge alpha oof_ridge_predictions fits at."""

ALPHA_GRID = ()
"""Non-gating diagnostic alpha grid -- never used to select ALPHA_RIDGE post-hoc."""

ALPHA_SELECTION_RULE = ""
"""Exact-equality-guarded (see _REQUIRED_ALPHA_SELECTION_RULE below)."""

N_OOF_FOLDS = None
"""Fold count for the explicit KFold OOF wrapper."""

OOF_FOLD_SEED = None
"""random_state passed to KFold."""

OOF_IMPLEMENTATION_RULE = ""
"""Exact-equality-guarded (see _REQUIRED_OOF_IMPLEMENTATION_RULE below)."""

LOCAL_R2_RULE = ""
"""Rule string naming the local out-of-fold R2 as the outcome -- never the catalog label
value itself (the substitution behind the colleague's own probe_label_alignment_failure)."""

MIN_FINITE_NEIGHBOURS = None
"""Floor on finite (y, y_hat) pairs in a neighbourhood before local_r2_panel masks the anchor."""

CONTROLS = ()
"""The ordered tuple of control names composing the 3-control partial's Z matrix."""

VERDICT_STATISTIC = None
"""Name of the statistic the phase verdict is computed on -- the controlled 3-control partial,
never the raw Spearman."""

RAW_RHO_IS_NON_GATING = None
"""True: the raw (uncontrolled) rho is reported beside the controlled partial and never gates."""

STRATIFIED_NULL_IS_NON_GATING = None
"""True: the density-stratified null is a secondary check and never gates the verdict alone."""

STRATIFICATION_FIELD = None
"""Field the density-stratified null bins on -- e.g. "log_knn_radius"."""

STRATA_GRID = ()
"""Non-gating stratum-count grid reported alongside the headline stratified null."""

STRATIFIED_NULL_DRAWS = None
"""Draw count for stratified_partial_null_3control."""

STRATIFIED_NULL_SEED = None
"""Seed for stratified_partial_null_3control."""

N_PERMUTATIONS = None
"""Draw count for permutation_fwer's Freedman-Lane null."""

PERMUTATION_SEED = None
"""Seed for permutation_fwer's Freedman-Lane null."""

NULL_CONSTRUCTION_RULE = ""
"""Exact-equality-guarded (see _REQUIRED_NULL_CONSTRUCTION_RULE below)."""

FWER_ALPHA = None
"""Significance level per_d_verdict compares p_fwer against, using a strict <."""

P_VALUE_FLOOR_RULE = ""
"""Rule string stating p_value_from_null never reports a zero p; it reports the
"< 1/(B+1)" string form instead."""

N_BOOTSTRAP = None
"""Draw count for paired_anchor_bootstrap."""

BOOTSTRAP_SEED = None
"""Seed for paired_anchor_bootstrap."""

BOOTSTRAP_RULE = ""
"""Rule string naming the paired-anchor-row resampling scheme."""

REPORT_BOTH_NULLS_UNCONDITIONALLY = None
"""True: the Freedman-Lane FWER null and the density-stratified null are both reported on
every run, never conditionally."""

VERDICT_RULE = ""
"""Rule string naming the phase-verdict decision procedure over VERDICT_VALUES."""

VERDICT_VALUES = ()
"""Exactly three phase-verdict strings: every-d, subset-of-d, does-not-replicate."""

PER_D_VERDICT_VALUES = ()
"""Exactly two per-d verdict strings: fired, not-fired."""

VERDICT_SENTENCE_RULE = ""
"""Rule string naming what verdict_sentence must state."""

REPORTING_BLOCK_ROWS = ()
"""Ordered tuple of row labels the final reporting block must include."""

REPORTING_BLOCK_RULE = ""
"""Rule string naming the reporting-block assembly procedure."""

POSITIVE_CONTROL_TARGET_RHOS = ()
"""Target controlled-partial grid plant_curvature_positive_control bisects against."""

POSITIVE_CONTROL_SEED = None
"""Seed for plant_curvature_positive_control's deterministic re-creation per bisection trial."""

POSITIVE_CONTROL_RULE = ""
"""Rule string naming the positive-control mechanism and its Freedman-Lane validation route."""

SHUFFLED_LABEL_REPEATS = None
"""Repeat count for shuffled_label_repeat."""

SHUFFLED_LABEL_SEED = None
"""Seed feeding the rng shuffled_label_repeat's caller re-creates per repeat."""

SHUFFLED_LABEL_RULE = ""
"""Rule string naming what is held fixed and what varies across shuffled_label_repeat calls."""

SEED_HANDLING_RULE = ""
"""Exact-equality-guarded (see _REQUIRED_SEED_HANDLING_RULE below): D9-17's never-pool rule."""

TORCH_INIT_SEEDS_WAVE_B = ()
"""The three torch init seeds Wave B fits at, feeding combine_seed_verdicts."""

SEED_VERDICT_COMBINATION_RULE = ""
"""Rule string naming the unanimous-3-of-3 combination rule and the SPLIT ACROSS SEEDS outcome."""

WAVE_B_TRIGGER_RULE = ""
"""Rule string naming when Wave B (the 3-seed sweep) is triggered."""

PREREGISTRATION_FREEZE_RULE = ""
"""Rule string naming the freeze-commit strict-ancestry requirement (D9-18)."""

RECORD_STEM = None
"""Base filename stem for the frozen Phase 9 JSONL record."""

RECORD_LOCATION_RULE = ""
"""Rule string naming where the frozen record and anchor-table npz files live."""

OUTPUT_ROOT_ENV_VAR = None
"""Environment variable name resolve_output_root checks for an execution-host override."""

EXECUTION_HOST_RULE = ""
"""Rule string naming the execution-host hand-off (09-06)."""

SWISS_ROLL_APPLICABILITY_RULE = (
    "CLAUDE.md's Swiss roll standing rule introduces no new notebook in Phase 9: this phase's "
    "instrument is cae.PlainAutoEncoder + decoder_curvature.plain_decoder_curvature, unchanged "
    "from Phase 7/8, and that model already has its Swiss roll notebook at "
    "notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb -- the same declaration Phase 7 "
    "made for the same instrument. No Swiss roll notebook is planned or written for Phase 9."
)
"""Filled now, deliberately -- a non-gating declarative fact, not a value the freeze
adjudicates. See module docstring."""


# --- Canonical values for the five exact-equality-guarded rule strings above -------------------
# This module owns the meaning of each of these five rule strings (it implements the behaviour
# each one describes), so it also owns the required canonical text 09-05's freeze must copy
# verbatim -- linear_probe.py's SEED_HANDLING_RULE idiom, applied here to five constants rather
# than one.

_REQUIRED_SEED_HANDLING_RULE = "no_pooling_per_seed_verdicts"

_REQUIRED_CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"

_REQUIRED_ALPHA_SELECTION_RULE = (
    "alpha_grid passed to oof_ridge_predictions holds exactly one DISTINCT value (ALPHA_RIDGE, "
    "duplicated as a two-entry tuple only to route around a sklearn==1.9.0 RidgeCV in-place-"
    "mutation defect on a one-element tuple -- see oof_ridge_predictions's own docstring); no "
    "alpha selection occurs at fit time. ALPHA_GRID is a diagnostic-only grid, never used to "
    "choose ALPHA_RIDGE post-hoc."
)

_REQUIRED_OOF_IMPLEMENTATION_RULE = (
    "oof_ridge_predictions wraps linear_probe.fit_probe/predict_probe inside an explicit KFold "
    "loop supplied by this module; sklearn.model_selection.cross_val_predict is never used, and "
    "a single whole-dataset fit is never presented as an out-of-fold prediction."
)

_REQUIRED_NULL_CONSTRUCTION_RULE = (
    "The null for the 3-control partial is Freedman-Lane: freedman_lane_y permutes the residual "
    "of the outcome's rank on the controls' ranks and adds the fit back. "
    "crossmodal_curvature.two_tailed_permutation_null is the wrong null for this phase and is "
    "never used here."
)


_REQUIRED_CONSTANTS = (
    "K_NEIGHBOURS", "NEIGHBOURHOOD_RATIO_RULE", "N_ANCHORS", "ANCHOR_DRAW_SEED", "ANCHOR_POOL",
    "ANCHOR_POOL_RULE", "SPLIT_SEED", "HOLDOUT_FRACTION", "AE_IN_DIM", "AE_HIDDEN",
    "AE_ACTIVATION", "MAX_EPOCHS", "TORCH_INIT_SEED", "TRAIN_CFG", "CURVATURE_SOURCE_FUNCTION",
    "CURVATURE_CONVENTION", "D_SWEEP", "FIT_QUALITY_KEYS", "INSTRUMENT_FIDELITY_RANGE_D16",
    "INSTRUMENT_FIDELITY_RANGE_D20", "INSTRUMENT_FIDELITY_RANGE_D25",
    "INSTRUMENT_FIDELITY_D32_RULE", "CURVATURE_FIELD_FOR_VERDICT", "H_NORM_IS_NON_GATING",
    "RADIAL_DECOMPOSITION_RULE", "MIN_IMAGE_NORM", "ALPHA_RIDGE", "ALPHA_GRID",
    "ALPHA_SELECTION_RULE", "N_OOF_FOLDS", "OOF_FOLD_SEED", "OOF_IMPLEMENTATION_RULE",
    "LOCAL_R2_RULE", "MIN_FINITE_NEIGHBOURS", "CONTROLS", "VERDICT_STATISTIC",
    "RAW_RHO_IS_NON_GATING", "STRATIFIED_NULL_IS_NON_GATING", "STRATIFICATION_FIELD",
    "STRATA_GRID", "STRATIFIED_NULL_DRAWS", "STRATIFIED_NULL_SEED", "N_PERMUTATIONS",
    "PERMUTATION_SEED", "NULL_CONSTRUCTION_RULE", "FWER_ALPHA", "P_VALUE_FLOOR_RULE",
    "N_BOOTSTRAP", "BOOTSTRAP_SEED", "BOOTSTRAP_RULE", "REPORT_BOTH_NULLS_UNCONDITIONALLY",
    "VERDICT_RULE", "VERDICT_VALUES", "PER_D_VERDICT_VALUES", "VERDICT_SENTENCE_RULE",
    "REPORTING_BLOCK_ROWS", "REPORTING_BLOCK_RULE", "POSITIVE_CONTROL_TARGET_RHOS",
    "POSITIVE_CONTROL_SEED", "POSITIVE_CONTROL_RULE", "SHUFFLED_LABEL_REPEATS",
    "SHUFFLED_LABEL_SEED", "SHUFFLED_LABEL_RULE", "SEED_HANDLING_RULE",
    "TORCH_INIT_SEEDS_WAVE_B", "SEED_VERDICT_COMBINATION_RULE", "WAVE_B_TRIGGER_RULE",
    "PREREGISTRATION_FREEZE_RULE", "RECORD_STEM", "RECORD_LOCATION_RULE",
    "OUTPUT_ROOT_ENV_VAR", "EXECUTION_HOST_RULE", "SWISS_ROLL_APPLICABILITY_RULE",
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
    in declaration order -- one check per constant. ``None``, an empty tuple/list, an empty
    dict and an empty-or-whitespace-only string are all treated as UNSET.

    Once every constant is non-empty, five additional checks fire by EXACT STRING EQUALITY,
    never by truthiness: ``SEED_HANDLING_RULE``, ``CURVATURE_FIELD_FOR_VERDICT``,
    ``ALPHA_SELECTION_RULE``, ``OOF_IMPLEMENTATION_RULE`` and ``NULL_CONSTRUCTION_RULE`` must
    each equal this module's own required canonical text -- a reworded rule string fails the
    guard even though it is non-empty."""
    g = globals()
    for name in _REQUIRED_CONSTANTS:
        if name not in g:
            raise RuntimeError(
                f"assert_preregistered: {name} is absent from physics_curvature_probe."
            )
        value = g[name]
        if _is_unset(value):
            raise RuntimeError(
                f"assert_preregistered: {name}={value!r} is UNSET. No Physics number may be "
                "computed before the freeze (D9-18)."
            )

    _equality_checks = (
        ("SEED_HANDLING_RULE", _REQUIRED_SEED_HANDLING_RULE),
        ("CURVATURE_FIELD_FOR_VERDICT", _REQUIRED_CURVATURE_FIELD_FOR_VERDICT),
        ("ALPHA_SELECTION_RULE", _REQUIRED_ALPHA_SELECTION_RULE),
        ("OOF_IMPLEMENTATION_RULE", _REQUIRED_OOF_IMPLEMENTATION_RULE),
        ("NULL_CONSTRUCTION_RULE", _REQUIRED_NULL_CONSTRUCTION_RULE),
    )
    for name, required in _equality_checks:
        if g[name] != required:
            raise RuntimeError(
                f"assert_preregistered: {name}={g[name]!r} does not equal the required text "
                f"{required!r}."
            )
    if g["CURVATURE_CONVENTION"] != "trace":
        raise RuntimeError(
            f"assert_preregistered: CURVATURE_CONVENTION={g['CURVATURE_CONVENTION']!r} does "
            'not equal "trace".'
        )


# =============================================================================================
# Compute functions. Pure, no torch import anywhere in this module. Every parameter this phase
# pre-registers is an explicit argument -- no defaults on any pre-registered parameter.
# =============================================================================================


def resolve_output_root() -> Path:
    """``Path(os.environ[OUTPUT_ROOT_ENV_VAR]).resolve()`` when that variable is set and
    non-empty, created with ``mkdir(parents=True, exist_ok=True)``; otherwise
    ``cache.CACHE_DIR``. The execution-host knob: the default is byte-identical to today's
    behaviour and ``cache.py`` is not edited."""
    env_var = OUTPUT_ROOT_ENV_VAR
    if env_var:
        value = os.environ.get(env_var)
        if value:
            root = Path(value).resolve()
            root.mkdir(parents=True, exist_ok=True)
            return root
    return cache.CACHE_DIR


def _assert_inside_output_root(path: Any) -> None:
    """Phase-9-owned containment guard mirroring ``cache._assert_inside_cache``'s logic against
    :func:`resolve_output_root`; raises ``ValueError`` naming both resolved paths. Written here
    rather than reused so no sealed module is touched."""
    root = resolve_output_root().resolve()
    resolved = Path(path).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(
            f"Refusing to use path outside the output root: {resolved} is not inside {root}."
        )


def record_path(stem: str, ext: str) -> Path:
    """``resolve_output_root() / f"{stem}.{ext}"``, containment-checked."""
    path = resolve_output_root() / f"{stem}.{ext}"
    _assert_inside_output_root(path)
    return path


def oof_ridge_predictions(X: np.ndarray, y: np.ndarray, alpha: float, n_folds: int, fold_seed: int) -> np.ndarray:
    """Explicit ``sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True,
    random_state=fold_seed)``; for each fold calls ``linear_probe.fit_probe`` on the train rows
    with an ``alpha_grid`` holding exactly ONE DISTINCT value (pinning alpha with no selection
    possible) and ``linear_probe.predict_probe`` on the held-out rows, writing into a
    full-length output array pre-filled with NaN. The fold structure is supplied here and never
    by the estimator's own internal machinery; a single whole-dataset fit is not an out-of-fold
    prediction. Raises ``ValueError`` when any output entry is still NaN after the loop -- the
    structural proof every row got exactly one held-out prediction.

    ``alpha_grid`` is passed as ``(float(alpha), float(alpha))`` -- a two-entry tuple whose
    entries are bit-identical -- rather than the more obvious one-element ``(float(alpha),)``.
    This is a measured, sealed-module-adjacent workaround (Rule 1/3, documented in
    ``09-01-SUMMARY.md``): ``sklearn==1.9.0``'s ``RidgeCV.fit`` mutates ``self.alphas[0]``
    in-place on the single-candidate fast path, which raises ``TypeError`` when ``alphas`` is an
    immutable one-element tuple (as ``linear_probe.fit_probe`` always constructs, regardless of
    what the caller passes in) but not when it holds two or more entries. Measured bit-identical
    to a genuine single-alpha ``sklearn.linear_model.Ridge`` fit (same ``coef_``, same
    predictions) because a candidate grid with only one DISTINCT value cannot select anything
    but that value -- "no selection possible" is preserved exactly; only the sklearn-internal
    fast-path branch changes. ``linear_probe.py`` is sealed and is not edited to work around
    this."""
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    y_hat = np.full(n, np.nan, dtype=np.float64)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    for train_idx, test_idx in kfold.split(X):
        fit = linear_probe.fit_probe(
            X[train_idx],
            y[train_idx].reshape(-1, 1),
            alpha_grid=(float(alpha), float(alpha)),
            alpha_per_target=False,
            fit_intercept=True,
        )
        preds = linear_probe.predict_probe(fit, X[test_idx])
        y_hat[test_idx] = np.asarray(preds, dtype=np.float64).ravel()

    if not np.all(np.isfinite(y_hat)):
        n_missing = int(np.sum(~np.isfinite(y_hat)))
        raise ValueError(
            f"oof_ridge_predictions: {n_missing} rows never received a held-out prediction -- "
            "every row must be assigned to exactly one test fold."
        )
    return y_hat


def anchor_indices(
    n_rows: int, split_seed: int, holdout_fraction: float, n_anchors: int, anchor_seed: int
) -> Dict[str, np.ndarray]:
    """``crossmodal_curvature.split_indices(n_rows, split_seed, holdout_fraction)``, then
    ``subsample.draw_row_indices(len(holdout_idx), n_anchors, anchor_seed)`` indexed into
    ``holdout_idx``. Returns ``{"train_idx", "holdout_idx", "anchor_idx"}``. Raises
    ``ValueError`` naming both counts when ``len(holdout_idx) < n_anchors``; returns the whole
    holdout when they are exactly equal. Curvature is measured only where the decoder never
    trained (D9-04) -- the deliberate departure from Phase 7's ``FIELD_EVALUATED_ON``
    convention. The returned anchor array is sorted and duplicate-free, and depends on no fitted
    model, so it is bit-for-bit identical across every ``d`` and every seed that shares
    ``split_seed``/``holdout_fraction``/``anchor_seed``."""
    train_idx, holdout_idx = crossmodal_curvature.split_indices(n_rows, split_seed, holdout_fraction)
    holdout_idx = np.asarray(holdout_idx)
    if holdout_idx.shape[0] < n_anchors:
        raise ValueError(
            f"anchor_indices: holdout pool has {holdout_idx.shape[0]} rows, fewer than "
            f"n_anchors={n_anchors}."
        )
    if holdout_idx.shape[0] == n_anchors:
        anchor_idx = np.sort(holdout_idx)
    else:
        anchor_pos = subsample.draw_row_indices(holdout_idx.shape[0], n_anchors, anchor_seed)
        anchor_idx = np.sort(holdout_idx[anchor_pos])
    return {
        "train_idx": np.sort(np.asarray(train_idx)),
        "holdout_idx": np.sort(holdout_idx),
        "anchor_idx": anchor_idx,
    }


def knn_panel(X_all: np.ndarray, anchor_idx: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """``sklearn.neighbors.NearestNeighbors(n_neighbors=k)`` fit on ``X_all``, queried at
    ``X_all[anchor_idx]`` only -- a query from ``n_anchors`` points, never an all-pairs graph.
    Returns ``{"distances", "indices", "log_knn_radius"}`` with
    ``log_knn_radius = np.log(distances[:, -1])``."""
    from sklearn.neighbors import NearestNeighbors

    X_all = np.asarray(X_all, dtype=np.float64)
    anchor_idx = np.asarray(anchor_idx)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_all)
    distances, indices = nn.kneighbors(X_all[anchor_idx])
    log_knn_radius = np.log(distances[:, -1])
    return {"distances": distances, "indices": indices, "log_knn_radius": log_knn_radius}


def local_r2_panel(
    y: np.ndarray, y_hat: np.ndarray, neighbour_idx: np.ndarray, min_finite: int
) -> Dict[str, Any]:
    """For each anchor row of ``neighbour_idx``, takes its neighbours with BOTH ``y`` and
    ``y_hat`` finite, uniform weights, and computes ``mse``, ``sst`` (about the neighbourhood
    mean of ``y``), ``r2 = 1 - mse_sum / sst_sum``, ``local_label_variance`` (variance of the
    finite ``y`` in the neighbourhood) and ``local_evaluation_count`` (that finite count). An
    anchor with fewer than ``min_finite`` finite pairs, or with ``sst_sum == 0.0``, gets NaN for
    ``r2`` and is counted in the returned ``n_masked_anchors``. ``y`` and ``y_hat`` are separate
    REQUIRED positional arguments -- the outcome is the local out-of-fold R2, never the catalog
    label value, the exact substitution behind the colleague's own
    ``probe_label_alignment_failure``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y_hat = np.asarray(y_hat, dtype=np.float64).ravel()
    neighbour_idx = np.asarray(neighbour_idx)
    n_anchors = neighbour_idx.shape[0]

    r2 = np.full(n_anchors, np.nan, dtype=np.float64)
    local_label_variance = np.full(n_anchors, np.nan, dtype=np.float64)
    local_evaluation_count = np.zeros(n_anchors, dtype=np.int64)
    n_masked = 0

    for i in range(n_anchors):
        nbrs = neighbour_idx[i]
        y_n = y[nbrs]
        yhat_n = y_hat[nbrs]
        finite = np.isfinite(y_n) & np.isfinite(yhat_n)
        n_finite = int(finite.sum())
        local_evaluation_count[i] = n_finite
        if n_finite < min_finite:
            n_masked += 1
            continue
        y_f = y_n[finite]
        yhat_f = yhat_n[finite]
        local_label_variance[i] = float(np.var(y_f))
        mean_y = float(np.mean(y_f))
        sst = float(np.sum((y_f - mean_y) ** 2))
        if sst == 0.0:
            n_masked += 1
            continue
        mse = float(np.sum((y_f - yhat_f) ** 2))
        r2[i] = 1.0 - mse / sst

    return {
        "r2": r2,
        "local_label_variance": local_label_variance,
        "local_evaluation_count": local_evaluation_count,
        "n_masked_anchors": n_masked,
    }


def decompose_radial_tangential(H_vec: np.ndarray, image: np.ndarray, min_image_norm: float) -> Dict[str, Any]:
    """Copies ``08_radial_curvature_decomposition_run.py``'s ``decompose()`` formula exactly:
    ``img_norm``, ``u = image / img_norm[:, None]``, ``H_rad = einsum("ij,ij->i", H_vec, u)``,
    ``H_tan = H_vec - H_rad[:, None] * u``. Rows whose ``img_norm`` is below ``min_image_norm``
    are excluded with their count returned as ``n_excluded_low_norm``, never divided by. Returns
    ``{"H_rad", "H_tan_norm", "H_norm", "image_norm", "n_excluded_low_norm"}``."""
    H_vec = np.asarray(H_vec, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    img_norm = np.linalg.norm(image, axis=1)
    keep = img_norm >= min_image_norm
    n_excluded = int((~keep).sum())

    n = H_vec.shape[0]
    H_rad = np.full(n, np.nan, dtype=np.float64)
    H_tan_norm = np.full(n, np.nan, dtype=np.float64)
    H_norm = np.full(n, np.nan, dtype=np.float64)

    if np.any(keep):
        u = image[keep] / img_norm[keep, None]
        H_rad_kept = np.einsum("ij,ij->i", H_vec[keep], u)
        H_tan_kept = H_vec[keep] - H_rad_kept[:, None] * u
        H_rad[keep] = H_rad_kept
        H_tan_norm[keep] = np.linalg.norm(H_tan_kept, axis=1)
        H_norm[keep] = np.linalg.norm(H_vec[keep], axis=1)

    return {
        "H_rad": H_rad,
        "H_tan_norm": H_tan_norm,
        "H_norm": H_norm,
        "image_norm": img_norm,
        "n_excluded_low_norm": n_excluded,
    }


def controlled_partial(x: Any, y: Any, controls: Any) -> float:
    """Thin delegation to ``cross_split_curvature.partial_spearman(x, y, controls=controls)``.
    Never reimplemented. A control matrix in which one column is constant is collinear with the
    intercept: least squares returns the minimum-norm solution, so the residuals and therefore
    the returned value are unchanged -- callers should record the constancy as a flag rather
    than dropping the column."""
    return cross_split_curvature.partial_spearman(x, y, controls=controls)


def freedman_lane_y(y: np.ndarray, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ports the colleague's ``inference.py`` lines 58-73 construction exactly: mask to rows
    where ``y`` and every column of ``Z`` are finite; rank-transform ``y`` and each control
    column on that mask; least-squares fit with an intercept; permute the RESIDUAL and add the
    fit back. Permuting the raw outcome instead is the documented bug class this construction
    exists to avoid."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    yr = rankdata(y[m])
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    fit = A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    resid = yr - fit
    y2 = y.copy()
    y2[m] = fit + rng.permutation(resid)
    return y2


def p_value_from_null(observed: float, null_draws: np.ndarray) -> Dict[str, Any]:
    """Returns ``{"p", "p_display", "n_draws", "floor_reached"}`` where
    ``p = (1 + count(|null| >= |observed|)) / (n_draws + 1)`` and ``p_display`` is the string
    ``f"< {1.0 / (n_draws + 1):.3e}"`` when the count is zero, otherwise the numeric value
    formatted to full precision. Never returns or formats a zero p-value."""
    null_draws = np.asarray(null_draws, dtype=np.float64)
    n_draws = int(null_draws.shape[0])
    count = int(np.sum(np.abs(null_draws) >= abs(float(observed))))
    p = (1 + count) / (n_draws + 1)
    floor_reached = count == 0
    p_display = f"< {1.0 / (n_draws + 1):.3e}" if floor_reached else f"{p:.6g}"
    return {"p": p, "p_display": p_display, "n_draws": n_draws, "floor_reached": floor_reached}


def permutation_fwer(
    curvature_by_d: Dict[int, np.ndarray], y: np.ndarray, Z: np.ndarray, n_permutations: int, seed: int
) -> Dict[str, Any]:
    """``curvature_by_d`` is an ordered mapping from ``d`` to the curvature array. For each draw
    builds one :func:`freedman_lane_y` surrogate and computes :func:`controlled_partial` at
    every ``d`` against it, keeping both the per-``d`` null and the per-draw ``max_d |rho|``
    envelope. Returns per-``d`` observed rho, per-``d`` ``p``, the global FWER ``p`` from the
    envelope, and the ``p_display`` strings."""
    rng = np.random.default_rng(seed)
    d_values = list(curvature_by_d.keys())
    observed = {d: controlled_partial(curvature_by_d[d], y, Z) for d in d_values}
    null_by_d: Dict[int, List[float]] = {d: [] for d in d_values}
    envelope: List[float] = []

    for _ in range(n_permutations):
        y_surrogate = freedman_lane_y(y, Z, rng)
        draw_vals = {}
        for d in d_values:
            val = controlled_partial(curvature_by_d[d], y_surrogate, Z)
            null_by_d[d].append(val)
            draw_vals[d] = val
        envelope.append(max(abs(v) for v in draw_vals.values()))

    per_d = {}
    for d in d_values:
        pv = p_value_from_null(observed[d], np.asarray(null_by_d[d], dtype=np.float64))
        per_d[d] = {"observed_rho": observed[d], **pv}

    max_observed = max(abs(v) for v in observed.values())
    global_pv = p_value_from_null(max_observed, np.asarray(envelope, dtype=np.float64))

    return {"per_d": per_d, "global": global_pv}


def stratified_partial_null_3control(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray, strata_field: np.ndarray, n_strata: int, n_draws: int, seed: int
) -> Dict[str, Any]:
    """Bins with ``density_stratified_null.density_strata(strata_field, n_strata)``, then
    permutes ``x`` and ``y`` INDEPENDENTLY within each stratum per draw and calls
    :func:`controlled_partial` with the full 3-column ``Z`` inside the loop. Does not edit
    ``density_stratified_null.py`` to generalise its single-control ``stratified_partial_null``
    -- additive only."""
    strata = density_stratified_null.density_strata(strata_field, n_strata)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    observed = controlled_partial(x, y, Z)

    rng = np.random.default_rng(seed)
    null_draws = np.empty(n_draws, dtype=np.float64)
    for b in range(n_draws):
        xp = x.copy()
        yp = y.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            xp[idx] = x[rng.permutation(idx)]
            yp[idx] = y[rng.permutation(idx)]
        null_draws[b] = controlled_partial(xp, yp, Z)

    pv = p_value_from_null(observed, null_draws)
    return {"observed": observed, "null_draws": null_draws, **pv}


def paired_anchor_bootstrap(x: np.ndarray, y: np.ndarray, Z: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """Resamples anchor ROWS with replacement, carrying ``x``, ``y`` and every control column
    together so the pairing is preserved, recomputes :func:`controlled_partial` per draw, and
    returns the 2.5/97.5 percentile band and the draw count."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = controlled_partial(x[idx], y[idx], Z[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": int(n_boot), "draws": draws}


def plant_curvature_positive_control(
    h_real: np.ndarray, y: np.ndarray, Z: np.ndarray, target_rho: float, seed: int, n_bisect: int
) -> Dict[str, Any]:
    """Reuses ``crossmodal_curvature.plant_positive_control``'s MECHANISM (guard first: raise
    ``ValueError`` naming ``h_real`` before any search when it is constant or non-finite;
    rank-transform; bisect a slope over ``n_bisect`` iterations on the bracket ``[0.0, 2.0]``)
    but retargets the achieved statistic at
    ``controlled_partial(planted, y, Z)``, spread-matched to the realized range of ``h_real``.
    Returns the planted array, the achieved controlled partial, and the slope. The null
    validation is the caller's job and must be :func:`permutation_fwer`'s Freedman-Lane
    construction -- ``crossmodal_curvature.two_tailed_permutation_null`` is the wrong null for
    this phase.

    Direction note (a real adaptation, not present in the sealed mechanism): the sealed
    ``plant_positive_control`` always bisects assuming ``spearmanr(h_real, planted)`` INCREASES
    with slope, which holds unconditionally there because the achieved statistic is measured
    against ``h_real`` itself. Here the achieved statistic is measured against ``y`` (e.g. the
    local out-of-fold R2), so whether ``controlled_partial(planted, y, Z)`` increases or
    decreases with slope depends on the empirical sign of the ``h_real``-``y`` relationship --
    for this phase's own negative-association hypothesis (D9-09), it decreases. The direction is
    therefore measured once (achieved at slope 0.0 vs slope 2.0) before bisecting, rather than
    assumed fixed."""
    from scipy.stats import rankdata

    h = np.asarray(h_real, dtype=np.float64).ravel()
    if not np.all(np.isfinite(h)):
        raise ValueError("plant_curvature_positive_control: h_real contains a non-finite value.")
    if np.ptp(h) == 0:
        raise ValueError("plant_curvature_positive_control: h_real is constant (np.ptp(h_real) == 0).")

    n = h.shape[0]
    u = (rankdata(h) - 0.5) / n
    lo_val, hi_val = float(np.min(h)), float(np.max(h))
    spread = hi_val - lo_val
    # A small discretization (mirroring the sealed mechanism's own k-sized binomial trial count,
    # rather than an arbitrary fine-grained one) keeps controlled_partial(planted, y, Z) a
    # smooth, near-monotonic function of slope across the whole [0.0, 2.0] bracket; a much finer
    # discretization saturates the achieved statistic within the first few percent of the
    # bracket, making bisection unable to resolve intermediate targets.
    _discretization = 10

    def _planted(slope: float) -> np.ndarray:
        p = np.clip(0.5 + slope * (u - 0.5), 0.0, 1.0)
        rng_ = np.random.default_rng(seed)
        j = rng_.binomial(_discretization, p)
        return lo_val + spread * (j / float(_discretization))

    achieved_at_low = controlled_partial(_planted(0.0), y, Z)
    achieved_at_high = controlled_partial(_planted(2.0), y, Z)
    increasing = achieved_at_high >= achieved_at_low

    low, high = 0.0, 2.0
    for _ in range(n_bisect):
        mid = (low + high) / 2.0
        mid_planted = _planted(mid)
        mid_achieved = controlled_partial(mid_planted, y, Z)
        if increasing:
            if mid_achieved < target_rho:
                low = mid
            else:
                high = mid
        else:
            if mid_achieved > target_rho:
                low = mid
            else:
                high = mid

    slope = high
    planted = _planted(slope)
    achieved = controlled_partial(planted, y, Z)
    return {
        "planted": planted,
        "achieved_controlled_partial": float(achieved),
        "slope": float(slope),
        "target_rho": float(target_rho),
    }


def shuffled_label_repeat(
    X: np.ndarray,
    y: np.ndarray,
    neighbour_idx: np.ndarray,
    log_knn_radius: np.ndarray,
    h_field: np.ndarray,
    alpha: float,
    n_folds: int,
    fold_seed: int,
    min_finite: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Permutes ``y`` across rows with ``rng``, recomputes the OOF predictions, recomputes
    :func:`local_r2_panel` and therefore BOTH label-derived controls, reuses the caller's
    ``log_knn_radius`` and ``h_field`` unchanged, and returns the controlled partial plus the
    masked count. The embedding matrix, the curvature field and the anchor index array are held
    byte-identical across repeats -- only the label vector moves."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    perm = rng.permutation(n)
    y_shuffled = y[perm]

    y_hat = oof_ridge_predictions(X, y_shuffled, alpha, n_folds, fold_seed)
    panel = local_r2_panel(y_shuffled, y_hat, neighbour_idx, min_finite)

    controls = np.column_stack(
        [log_knn_radius, panel["local_label_variance"], panel["local_evaluation_count"]]
    )
    h_field = np.asarray(h_field, dtype=np.float64)
    finite = np.isfinite(panel["r2"])
    controlled = controlled_partial(h_field[finite], panel["r2"][finite], controls[finite])

    return {
        "controlled_partial": float(controlled),
        "local_label_variance": panel["local_label_variance"],
        "local_evaluation_count": panel["local_evaluation_count"],
        "n_masked_anchors": panel["n_masked_anchors"],
    }


def per_d_verdict(rho: float, p_fwer: float, fwer_alpha: float) -> str:
    """Returns the first entry of :data:`PER_D_VERDICT_VALUES` when ``rho < 0.0`` AND
    ``p_fwer < fwer_alpha`` using STRICT comparisons on both (a rho of exactly 0.0 is not
    negative; a p exactly equal to the level does not clear), otherwise the second. Raises
    ``ValueError`` on a non-finite ``rho``."""
    if not np.isfinite(rho):
        raise ValueError(f"per_d_verdict: rho={rho} is not finite.")
    if rho < 0.0 and p_fwer < fwer_alpha:
        return PER_D_VERDICT_VALUES[0]
    return PER_D_VERDICT_VALUES[1]


def phase_verdict(per_d_map: Dict[int, str]) -> str:
    """Every ``d`` fired gives ``VERDICT_VALUES[0]`` (the "every d" value), at least one but not
    all gives ``VERDICT_VALUES[1]`` (the "subset of d" value), none gives ``VERDICT_VALUES[2]``
    (the "does not replicate" value), and an empty map gives ``VERDICT_VALUES[2]`` rather than
    absent. Never prints or returns a pooled headline number."""
    if not per_d_map:
        return VERDICT_VALUES[2]
    fired_flag = PER_D_VERDICT_VALUES[0]
    entries = list(per_d_map.values())
    fired = [v for v in entries if v == fired_flag]
    if len(fired) == len(entries):
        return VERDICT_VALUES[0]
    if len(fired) > 0:
        return VERDICT_VALUES[1]
    return VERDICT_VALUES[2]


def combine_seed_verdicts(seed_verdicts: Any) -> str:
    """Raises ``ValueError`` unless given exactly three entries; returns the shared value on
    unanimity; returns ``"SPLIT ACROSS SEEDS"`` otherwise. Never averages, never upgrades a
    2-of-3. Mirrors ``05-03-DECISION.md``'s one-way ratification."""
    verdicts = list(seed_verdicts)
    if len(verdicts) != 3:
        raise ValueError(
            f"combine_seed_verdicts: expected exactly three seed verdict entries; got "
            f"{len(verdicts)}."
        )
    if len(set(verdicts)) == 1:
        return verdicts[0]
    return "SPLIT ACROSS SEEDS"


def verdict_sentence(
    instrument: str,
    d_values: Any,
    colleague_rho: float,
    colleague_d: int,
    fwer_p_display: str,
    stratified_p_display: str,
    instrument_fidelity_ranges: Dict[int, Any],
    neighbourhood_ratio: str,
) -> str:
    """Assembles D9-10's caveat-bearing sentence naming the instrument, the ``d`` values, the
    colleague's ``-0.240`` at his ``d=16``, both nulls, the instrument-fidelity ranges, and the
    neighbourhood n-ratio. Reads :data:`VERDICT_SENTENCE_RULE`."""
    if _is_unset(VERDICT_SENTENCE_RULE):
        raise RuntimeError(
            "verdict_sentence: VERDICT_SENTENCE_RULE is UNSET; the freeze (09-05) must fill it "
            "before a verdict sentence can be assembled."
        )
    return (
        f"Instrument {instrument} at d={list(d_values)}, against the colleague's "
        f"{colleague_rho:.3f} at his d={colleague_d}: Freedman-Lane FWER p={fwer_p_display}, "
        f"density-stratified null p={stratified_p_display}; instrument fidelity ranges "
        f"{instrument_fidelity_ranges}; neighbourhood ratio {neighbourhood_ratio}."
    )
