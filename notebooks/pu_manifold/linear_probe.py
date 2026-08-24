"""Phase 5 curvature-conditioned linear decodability: probe fit/score, seed pooling, bucketing
and verdict functions, plus the pre-registration constants block and its guard.

(a) **D5-03's own citation is corrected here, not silently followed.** D5-03 names
``notebooks/pu_manifold/decoder_curvature.py`` as the decoder-side curvature source. That
module's own docstring states it is ``chart_curvature.py`` with the
``chart_decoders[chart_idx]`` two-hop composition removed, built for Phase 02.6's
no-chart-index substrates (a plain autoencoder and a ``PlainAutoEncoder`` trained under
``topoae.train_topoae``, both decoding through one smooth MLP with no chart index at all).
``ChartAutoEncoder`` is chart-routed and has no bare single-hop decode entry point matching
that module's signature. The function this phase actually uses, matching the CAE Phase 3
built, is ``chart_curvature.chart_curvature_field(model, x, mode="reverse")`` -- frozen as
the named constant ``CURVATURE_SOURCE_FUNCTION`` (unset here, filled at the 05-04 freeze) so
the correction is auditable rather than buried in a comment. This module imports nothing
from ``decoder_curvature``, and the runner does not either.

(b) **The pooled design is REMOVED from this module, not merely left unused.**
``05-CONTEXT.md`` D5-04 said to pool the three cached CAE seeds into one averaged ``||H||``
field. That was put to the developer at the ``05-03`` Task 1 blocking checkpoint and REJECTED
-- one-way, per ``05-03-DECISION.md``, which SUPERSEDES D5-04. The evidence: measured at
``05-02`` over all 10,000 PU points, the three seeds' fields are mutually anti-correlated on
rank (pairwise Spearman on ``H_norm`` -0.1402, +0.2019, -0.2725 -- sign-inconsistent, two of
three negative) and directionally orthogonal (median cosine of unit ``H_vec`` 0.0007 to 0.0039,
with 46 to 48 percent of points anti-aligned between any pair), with seeds 20260814 and
20260815 taking 4 and 3 effective distinct levels (see the correction below) at a metric
determinant around ``1e-166``, roughly 100 orders of magnitude from seed 20260813's continuous
field. Any pooled field would not be a consensus: it would be seed 20260813's structure plus
two step-like functions that disagree with it and with each other. Rejected alongside the raw
mean: per-seed median-divide then average (``05-RESEARCH.md``'s own recommendation), per-seed
percentile-rank then average, and halting the phase. :func:`pool_seed_fields` is RETAINED as
tested but unused code, per CLAUDE.md's additive-only rule -- Phase 5 calls it nowhere.

**Correction to 05-02-SUMMARY.md.** That summary reported seeds 20260814/15 as "not literally
piecewise-constant -- 5,301 / 9,852 exact distinct ``H_norm`` values (not 3-4)". That claim is
WRONG: those counts are float noise in the last ULPs. Measured directly from the cached fields
at RELATIVE precision, seed 20260814 has 4 distinct levels and seed 20260815 has 3, stable from
rel 1e-9 through rel 1e-3. ``05-RESEARCH.md`` Pitfall 2 and ``03-09-SUMMARY.md``'s original
measurement were both correct.

(c) **D5-11's accepted gap.** The field this phase splits on has no demonstrated relationship
to true curvature: the sealed ``d=20`` decoder row is
``rank_spearman_rho = -0.015106571347065712`` against the only analytic-curvature control
that tests it (spike-findings-effdim, ``high-d-curvature-feasibility.md``), and direction is
near a coin flip (52-75 percent of points anti-aligned). A Swiss roll / low-``d`` anchor was
offered and declined for this phase. Any relationship Phase 5 measures between ``||H||`` and
probe residual therefore cannot be attributed to curvature by anything in this phase --
stated here, in this module's own words, and not only by cross-reference.

(d) **D5-12's inherited chain.** The CAE underlying every field this module consumes failed
its own validity gate (``CAE_VERDICT = FAIL``, Phase 02.2); Phase 3 ran on a deliberate
override of that gate; Phase 03.1 found the pullback metric fully repaired by a ``scale``
prior while the curvature ordering only partially and non-seed-consistently moved. Every
number this module's functions eventually help produce inherits that chain.

No file I/O happens in this module -- every function is pure, operates on arrays the caller
already has in memory, and every pre-registered value is passed in by the caller with no
default (mirroring ``region_partition.py``'s own stated convention: a default is how a
pre-registered value gets inherited by accident instead of by an explicit call-site choice).
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score


# --- Pre-registration (D5-09, FROZEN at plan 05-04 Task 2, this commit) --------------------
#
# FROZEN. Every constant below, and VERDICT_RULE's and SEED_VERDICT_COMBINATION_RULE's full
# text, were ratified at plan 05-04's Task 1 blocking decision checkpoint (the protocol) and
# at plan 05-03's Task 1 blocking checkpoint (the seed-handling rule, 05-03-DECISION.md) --
# BOTH before any PU probe number existed anywhere in this repository. Amending any of them
# after a PU probe number has been computed invalidates the phase: a rule chosen after seeing
# the numbers is a rationalization, not a pre-registration. From this commit forward,
# `notebooks/pu_manifold/linear_probe.py` is closed -- a later edit is a recorded pre-
# registration BREACH, written up in `05-FINDINGS.md`/`05-VERIFICATION.md` with the diff and
# the reason, never a silent fix. See
# `.planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md` for
# the full committed record, including both checkpoints' ratification notes.
#
# `05-CONTEXT.md` D5-04's pooled-field design -- `POOLING_METHOD` (a required normalization
# method name) and `BUCKET_EDGES` (one flat tuple of edges cut over a pooled field) -- is
# SUPERSEDED by `05-03-DECISION.md`. Both constants were REMOVED at `05-03` rather than left
# unused, so the pooled path cannot be re-entered by assigning them. In their place:
# `SEED_HANDLING_RULE` (the ratified no-pooling decision), `BUCKET_EDGES_PER_SEED` (three
# per-seed edge tuples, one per `SEED_STEMS` entry, never one pooled tuple),
# `SEED_VERDICT_COMBINATION_RULE` and `PHASE_VERDICT_VALUES` (how three per-seed verdicts
# combine into one phase read-out, including the terminal "SPLIT ACROSS SEEDS" outcome). See
# this module's docstring paragraph (b) for the measured evidence the rejection was made on.

TRAIN_FRACTION = 0.7
SPLIT_SEED = 20260824
SPLIT_RULE = (
    "One permutation of np.arange(10000) under np.random.default_rng(SPLIT_SEED); the first "
    "7,000 indices of the permutation are train, the last 3,000 are test; both index arrays "
    "are returned sorted ascending. ONE split, shared by all three seeds' bucketings, so the "
    "three per-seed verdicts differ only in how the same held-out residuals are bucketed and "
    "never in which points were held out. Deliberately NOT stratified by bucket -- "
    "stratifying would manufacture the equal per-bucket test counts D5-08 exists to check for "
    "rather than assume. Implemented by train_test_split_indices."
)
RIDGE_ALPHA_GRID = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
RIDGE_SELECTION_RULE = (
    "scikit-learn RidgeCV's efficient generalized leave-one-out cross-validation, evaluated "
    "on the training split alone, selecting ONE alpha from RIDGE_ALPHA_GRID. The GRID and the "
    "SELECTION RULE are frozen; the selected alpha is an OUTPUT and is reported at 05-05, "
    "never pre-specified. If the design matrix is well-conditioned in practice, the rule "
    "selects a near-zero alpha and the fit degrades gracefully to OLS, so RESEARCH A2 cannot "
    "silently distort the result."
)
ALPHA_PER_TARGET = False
FIT_INTERCEPT = True
EMBEDDING_PREPROCESSING = (
    "raw_as_cached -- both modalities are already L2-normalized upstream (every row norm "
    "equals 1.0 to float64 rounding in the resolved npz), so re-normalizing would be a no-op "
    "dressed as a decision."
)
RESIDUAL_METRIC = "squared_l2_per_point"
R2_MULTIOUTPUT = "variance_weighted"
N_BUCKETS = 3
BUCKET_RULE = (
    "Equal-frequency rank partition of ONE SEED's ||H|| field over all 10,000 points by "
    "stable argsort and np.array_split, applied INDEPENDENTLY to each of the three seeds; a "
    "value equal to an edge lands in the HIGHER bucket (assign_buckets' documented tie rule). "
    "Tertiles (N_BUCKETS=3) rather than quartiles because at a 70/30 split three buckets "
    "leave roughly 1,000 test points each, supporting a percentile bootstrap without the CI "
    "collapsing; quartiles would leave roughly 750 and buy nothing."
)
BUCKET_EDGES_PER_SEED = (
    (1225.4263017421292, 1538.3597929379368),
    (49062.2351870738, 66977.54374981482),
    (51694.86079512253, 75252.52609688243),
)
SEED_HANDLING_RULE = "no_pooling_per_seed_verdicts"
SEED_STEMS = (20260813, 20260814, 20260815)
N_CHARTS = 4
CURVATURE_MODE = "reverse"
CURVATURE_CONVENTION = "trace"
CURVATURE_SOURCE_FUNCTION = "chart_curvature.chart_curvature_field"
SIZE_MATCH_RULE = (
    "Per seed, subsample every bucket down to the smallest REALIZED TEST-SPLIT bucket count "
    "FOR THAT SEED (never the full-field count, never a count borrowed from another seed), "
    "and re-run that seed's highest-versus-lowest comparison SIZE_MATCH_N_REPEATS times under "
    "SIZE_MATCH_SEED. This is the exact artifact that undercut Phase 4's HOLDS verdict, built "
    "into the protocol from the start."
)
SIZE_MATCH_N_REPEATS = 200
SIZE_MATCH_SEED = 20260824
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260824
CONFIDENCE_LEVEL = 0.95
K_DENSITY = 30
FIELD_D = 20
VERDICT_RULE = """D5-09 per-seed VERDICT_RULE -- ratified at plan 05-04's Task 1 blocking
checkpoint, before any PU probe number existed.

Per seed, the headline comparison is that seed's highest-||H|| bucket (of N_BUCKETS = 3
tertiles) against its lowest, on mean per-point squared L2 residual over the ONE shared 70/30
test split (TRAIN_FRACTION, SPLIT_SEED), under that seed's own frozen BUCKET_EDGES_PER_SEED
entry.

That seed's verdict is HOLDS if and only if ALL three of:
  (a) the highest and lowest bucket's CONFIDENCE_LEVEL (0.95) percentile bootstrap CIs on
      mean per-point squared L2 residual are disjoint;
  (b) the highest bucket's mean residual strictly exceeds the lowest bucket's; AND
  (c) the sign survives that seed's SIZE_MATCH_RULE re-check (subsampled to that seed's
      realized test-split bucket counts) with CIs disjoint in at least half of
      SIZE_MATCH_N_REPEATS = 200 repeats.

NO DETECTABLE RELATIONSHIP is that seed's verdict whenever any one of (a)/(b)/(c) fails. It is
a complete, valid, TERMINAL per-seed outcome -- never a phase failure, never escalated by the
continuous statistic, and never re-decided by trying a different N_BUCKETS.

The three per-seed verdicts (HOLDS / NO DETECTABLE RELATIONSHIP) then combine under
SEED_VERDICT_COMBINATION_RULE into exactly one of PHASE_VERDICT_VALUES, including the
terminal outcome SPLIT ACROSS SEEDS -- see that rule's own text for the full mapping and for
why a split is not partial support.

The continuous Spearman between that seed's curvature magnitude and per-point residual on the
test split is reported per seed alongside the verdict as SENSITIVITY ONLY; it can neither
establish nor overturn any verdict at either the per-seed or the phase level.

D5-11 CAVEAT, carried in this rule's own text rather than only alongside it: the field this
rule buckets on has no demonstrated relationship to true curvature. The sealed d=20 decoder
row is rank_spearman_rho = -0.015106571347065712 against the only analytic-curvature control
that tests it, essentially zero, with 52 to 75 percent of points anti-aligned in direction. A
Swiss roll / low-d anchor was offered and declined for this phase. No verdict produced under
this rule can be attributed to curvature by anything in this phase. The mitigating context --
the sealed saddle control sets a constant analytic Hessian, so its ||H|| varies only through
the pullback metric, which may make that fixture structurally unable to show ordering at all
-- is reported and is explicitly NOT used to upgrade any result produced under this rule; the
question is open and it is not for autonomous action.

D5-12 CAVEAT, carried in this rule's own text: the CAE supplying every decoder this rule reads
curvature from failed its own validity gate (CAE_VERDICT = FAIL, Phase 02.2); Phase 3 ran on a
deliberate override of that gate; Phase 03.1 found the pullback metric repaired by the scale
prior while the curvature ordering only partially and non-seed-consistently moved. Every
verdict this rule produces inherits that chain.

D5-13 NOTE: the per-seed density Spearman (spearman(density, ||H||)) is reported alongside
every verdict as a disclosure only; it is not a gate under this rule.
"""
SEED_VERDICT_COMBINATION_RULE = """D5-09 SEED_VERDICT_COMBINATION_RULE -- ratified at plan
05-04's Task 1 blocking checkpoint, before any PU probe number existed. Supersedes
05-CONTEXT.md D5-04's pooled-field design per 05-03-DECISION.md.

The probe is scored once per seed under the IDENTICAL protocol (the identical TRAIN_FRACTION
70/30 split, shared across all three seeds' bucketings via the one SPLIT_SEED) and the
IDENTICAL VERDICT_RULE, producing exactly one per-seed terminal verdict per seed: HOLDS or
NO DETECTABLE RELATIONSHIP.

The three per-seed verdicts combine into exactly one PHASE_VERDICT_VALUES member by counting
the HOLDS outcomes:
  * three of three HOLDS  -> "HOLDS IN ALL THREE SEEDS"
  * zero of three HOLDS   -> "NO DETECTABLE RELATIONSHIP IN ANY SEED"
  * one or two of three   -> "SPLIT ACROSS SEEDS"

SPLIT ACROSS SEEDS is a COMPLETE TERMINAL OUTCOME and is NOT partial support for the
hypothesis. The three seed fields were measured at 05-02 to be mutually anti-correlated on
rank (pairwise Spearman on H_norm -0.1402, +0.2019, -0.2725 -- sign-inconsistent, two of
three negative) and directionally orthogonal (median cosine of unit H_vec 0.0007 to 0.0039,
with 46 to 48 percent of points anti-aligned between any pair), so a relationship that appears
in one or two of three seeds' fields and not the third is a property of that individual
decoder fit, not of the manifold, and does not license the claim that decodability degrades
with curvature.

A split is NEVER upgraded to HOLDS IN ALL THREE SEEDS by majority vote, by the continuous
Spearman statistic, by a non-headline bucket, or by trying a different N_BUCKETS; and it is
NEVER downgraded to NO DETECTABLE RELATIONSHIP IN ANY SEED either -- it is reported exactly as
SPLIT ACROSS SEEDS, with all three per-seed verdicts and their supporting numbers beside it.

Because one split is shared across all three seeds' bucketings, the three per-seed verdicts
are NOT statistically independent -- they score the same held-out residuals under three
different bucketings -- which isolates the field as the only thing that differs between them,
but must be stated in 05-FINDINGS.md rather than left implicit.
"""
PHASE_VERDICT_VALUES = (
    "HOLDS IN ALL THREE SEEDS",
    "SPLIT ACROSS SEEDS",
    "NO DETECTABLE RELATIONSHIP IN ANY SEED",
)
PREREGISTRATION_PATH = (
    ".planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md"
)


def assert_preregistered() -> None:
    """Raise ``RuntimeError`` unless the pre-registration is intact. Checks, in order, one
    check per constant, raising on the FIRST failing check (``region_partition.py``'s own
    idiom): ``VERDICT_RULE`` is a non-empty string naming ``N_BUCKETS`` and naming the
    ``"SPLIT ACROSS SEEDS"`` outcome; ``N_BUCKETS`` is a positive int; ``TRAIN_FRACTION`` is a
    float strictly inside ``(0, 1)``; ``SPLIT_SEED`` is a positive int; ``RIDGE_ALPHA_GRID`` is
    a non-empty tuple of positive floats; ``SEED_HANDLING_RULE`` equals the exact ratified
    string ``"no_pooling_per_seed_verdicts"``; ``SEED_STEMS`` is a tuple of three positive
    ints; ``BUCKET_EDGES_PER_SEED`` is a tuple of ``len(SEED_STEMS)`` per-seed tuples, each of
    ``N_BUCKETS - 1`` finite floats in strictly ascending order; ``SEED_VERDICT_COMBINATION_
    RULE`` is a non-empty string naming the ``"SPLIT ACROSS SEEDS"`` outcome;
    ``PHASE_VERDICT_VALUES`` is a tuple of exactly three distinct non-empty strings and every
    outcome :func:`combine_seed_verdicts` can produce under the frozen rule is a member of it;
    ``CURVATURE_CONVENTION`` equals ``"trace"``; ``CURVATURE_SOURCE_FUNCTION`` is a non-empty
    string. Called at the top of the runner's ``--mode bucketed`` branch so that path fails
    loudly rather than computing anything when the pre-registration is absent or malformed.

    ``SEED_HANDLING_RULE`` is checked by EQUALITY, not by a non-empty-string check: a future
    edit that assigns it a pooling-method name (re-entering the design ``05-03-DECISION.md``
    ratified as rejected) must fail this guard rather than pass it.
    """
    if not isinstance(VERDICT_RULE, str) or not VERDICT_RULE.strip():
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} is empty or not a string."
        )
    if "N_BUCKETS" not in VERDICT_RULE:
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} does not name N_BUCKETS."
        )
    if "SPLIT ACROSS SEEDS" not in VERDICT_RULE:
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} does not name the "
            '"SPLIT ACROSS SEEDS" outcome.'
        )
    if not isinstance(N_BUCKETS, int) or isinstance(N_BUCKETS, bool) or N_BUCKETS <= 0:
        raise RuntimeError(f"assert_preregistered: N_BUCKETS={N_BUCKETS!r} is not a positive int.")
    if not isinstance(TRAIN_FRACTION, float) or not (0.0 < TRAIN_FRACTION < 1.0):
        raise RuntimeError(
            f"assert_preregistered: TRAIN_FRACTION={TRAIN_FRACTION!r} is not a float strictly "
            "inside (0, 1)."
        )
    if not isinstance(SPLIT_SEED, int) or isinstance(SPLIT_SEED, bool) or SPLIT_SEED <= 0:
        raise RuntimeError(
            f"assert_preregistered: SPLIT_SEED={SPLIT_SEED!r} is not a positive int."
        )
    if (
        not isinstance(RIDGE_ALPHA_GRID, tuple)
        or len(RIDGE_ALPHA_GRID) == 0
        or not all(isinstance(v, float) and v > 0.0 for v in RIDGE_ALPHA_GRID)
    ):
        raise RuntimeError(
            f"assert_preregistered: RIDGE_ALPHA_GRID={RIDGE_ALPHA_GRID!r} is not a non-empty "
            "tuple of positive floats."
        )
    if SEED_HANDLING_RULE != "no_pooling_per_seed_verdicts":
        raise RuntimeError(
            f"assert_preregistered: SEED_HANDLING_RULE={SEED_HANDLING_RULE!r} does not equal "
            '"no_pooling_per_seed_verdicts" -- the ratified no-pooling decision '
            "(05-03-DECISION.md)."
        )
    if (
        not isinstance(SEED_STEMS, tuple)
        or len(SEED_STEMS) != 3
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in SEED_STEMS)
    ):
        raise RuntimeError(
            f"assert_preregistered: SEED_STEMS={SEED_STEMS!r} is not a tuple of three positive "
            "ints."
        )
    if not isinstance(BUCKET_EDGES_PER_SEED, tuple) or len(BUCKET_EDGES_PER_SEED) != len(SEED_STEMS):
        raise RuntimeError(
            f"assert_preregistered: BUCKET_EDGES_PER_SEED={BUCKET_EDGES_PER_SEED!r} is not a "
            f"tuple of {len(SEED_STEMS)} per-seed edge tuples (one per SEED_STEMS entry, never "
            "one pooled tuple)."
        )
    for _seed_idx, _edges in enumerate(BUCKET_EDGES_PER_SEED):
        if (
            not isinstance(_edges, tuple)
            or len(_edges) != N_BUCKETS - 1
            or not all(isinstance(v, float) and np.isfinite(v) for v in _edges)
            or list(_edges) != sorted(_edges)
            or len(set(_edges)) != len(_edges)
        ):
            raise RuntimeError(
                f"assert_preregistered: BUCKET_EDGES_PER_SEED[{_seed_idx}]={_edges!r} is not a "
                f"tuple of {N_BUCKETS - 1} finite floats in strictly ascending order -- the "
                "shape a pooled design would produce instead."
            )
    if (
        not isinstance(SEED_VERDICT_COMBINATION_RULE, str)
        or not SEED_VERDICT_COMBINATION_RULE.strip()
        or "SPLIT ACROSS SEEDS" not in SEED_VERDICT_COMBINATION_RULE
    ):
        raise RuntimeError(
            f"assert_preregistered: SEED_VERDICT_COMBINATION_RULE="
            f"{SEED_VERDICT_COMBINATION_RULE!r} is empty or does not name the "
            '"SPLIT ACROSS SEEDS" outcome.'
        )
    if (
        not isinstance(PHASE_VERDICT_VALUES, tuple)
        or len(PHASE_VERDICT_VALUES) != 3
        or not all(isinstance(v, str) and v for v in PHASE_VERDICT_VALUES)
        or len(set(PHASE_VERDICT_VALUES)) != 3
    ):
        raise RuntimeError(
            f"assert_preregistered: PHASE_VERDICT_VALUES={PHASE_VERDICT_VALUES!r} is not a "
            "tuple of exactly three distinct non-empty strings."
        )
    _canonical_trios = (
        {s: "HOLDS" for s in SEED_STEMS},
        {s: "NO DETECTABLE RELATIONSHIP" for s in SEED_STEMS},
        {
            SEED_STEMS[0]: "HOLDS",
            SEED_STEMS[1]: "NO DETECTABLE RELATIONSHIP",
            SEED_STEMS[2]: "NO DETECTABLE RELATIONSHIP",
        },
    )
    for _trio in _canonical_trios:
        _result = combine_seed_verdicts(_trio, SEED_VERDICT_COMBINATION_RULE)
        if _result["phase_verdict"] not in PHASE_VERDICT_VALUES:
            raise RuntimeError(
                f"assert_preregistered: combine_seed_verdicts produced "
                f"{_result['phase_verdict']!r}, which is not a member of "
                f"PHASE_VERDICT_VALUES={PHASE_VERDICT_VALUES!r}."
            )
    if CURVATURE_CONVENTION != "trace":
        raise RuntimeError(
            f"assert_preregistered: CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} does not "
            'equal "trace".'
        )
    if not isinstance(CURVATURE_SOURCE_FUNCTION, str) or not CURVATURE_SOURCE_FUNCTION.strip():
        raise RuntimeError(
            f"assert_preregistered: CURVATURE_SOURCE_FUNCTION={CURVATURE_SOURCE_FUNCTION!r} "
            "is empty or not a string."
        )


# --- D5-01/D5-02: the probe itself -----------------------------------------------------------


def train_test_split_indices(
    n: int, train_fraction: float, split_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """One permutation of ``np.arange(n)`` under ``np.random.default_rng(split_seed)``; the
    first ``round(n * train_fraction)`` indices (of the permutation) are train, the rest test.
    Both returned index arrays are sorted ascending so downstream row alignment is readable.
    ``train_fraction`` and ``split_seed`` are required arguments with no default.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"train_test_split_indices: n={n} must be an int >= 2.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_test_split_indices: train_fraction={train_fraction} must be in (0, 1)."
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(int(n))
    n_train = int(round(n * train_fraction))
    n_train = min(max(n_train, 1), n - 1)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    return train_idx, test_idx


def fit_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha_grid: Any,
    alpha_per_target: bool,
    fit_intercept: bool,
) -> Dict[str, Any]:
    """Wraps ``sklearn.linear_model.RidgeCV(alphas=alpha_grid,
    alpha_per_target=alpha_per_target, fit_intercept=fit_intercept)``. Never hand-rolls a CV
    loop or a least-squares solver. Returns a flat dict carrying the fitted estimator and
    everything a caller or test might want to inspect without refitting.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError(f"fit_probe: X_train must be two-dimensional, got shape {X_train.shape}.")
    if Y_train.ndim != 2:
        raise ValueError(f"fit_probe: Y_train must be two-dimensional, got shape {Y_train.shape}.")
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"fit_probe: X_train has {X_train.shape[0]} rows but Y_train has "
            f"{Y_train.shape[0]} rows."
        )
    if not np.all(np.isfinite(X_train)):
        raise ValueError("fit_probe: X_train contains a non-finite value.")
    if not np.all(np.isfinite(Y_train)):
        raise ValueError("fit_probe: Y_train contains a non-finite value.")
    alpha_grid = tuple(float(a) for a in alpha_grid)
    if len(alpha_grid) == 0:
        raise ValueError("fit_probe: alpha_grid must be non-empty.")

    estimator = RidgeCV(
        alphas=alpha_grid, alpha_per_target=alpha_per_target, fit_intercept=fit_intercept
    )
    estimator.fit(X_train, Y_train)

    return {
        "estimator": estimator,
        "coef_shape": tuple(np.asarray(estimator.coef_).shape),
        "intercept_shape": tuple(np.asarray(estimator.intercept_).shape),
        "alpha_": estimator.alpha_,
        "alpha_grid": alpha_grid,
        "alpha_per_target": bool(alpha_per_target),
        "fit_intercept": bool(fit_intercept),
        "n_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_targets": int(Y_train.shape[1]),
    }


def predict_probe(fit: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """``fit["estimator"].predict(X)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"predict_probe: X must be two-dimensional, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("predict_probe: X contains a non-finite value.")
    return fit["estimator"].predict(X)


def per_point_residuals(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """The length-``n`` array ``np.sum((Y_true - Y_pred) ** 2, axis=1)``, the per-point
    squared L2 residual. Pairing this with :func:`aggregate_r2` is what satisfies
    05-CONTEXT.md's "R-squared and per-point residual derivable from one underlying quantity"
    constraint.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"per_point_residuals: Y_true has shape {Y_true.shape} but Y_pred has shape "
            f"{Y_pred.shape}."
        )
    if Y_true.ndim != 2:
        raise ValueError(
            f"per_point_residuals: Y_true/Y_pred must be two-dimensional, got shape "
            f"{Y_true.shape}."
        )
    if not np.all(np.isfinite(Y_true)) or not np.all(np.isfinite(Y_pred)):
        raise ValueError("per_point_residuals: Y_true or Y_pred contains a non-finite value.")
    return np.sum((Y_true - Y_pred) ** 2, axis=1)


def aggregate_r2(Y_true: np.ndarray, Y_pred: np.ndarray, multioutput: str) -> float:
    """``sklearn.metrics.r2_score(Y_true, Y_pred, multioutput=multioutput)``. RESEARCH A3
    claims this equals ``1 - per_point_residuals(...).sum() / sum((Y_true - Y_true.mean(0))
    ** 2)`` exactly when ``multioutput="variance_weighted"`` -- this plan's test file pins
    that identity numerically rather than trusting the citation.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"aggregate_r2: Y_true has shape {Y_true.shape} but Y_pred has shape {Y_pred.shape}."
        )
    if not np.all(np.isfinite(Y_true)) or not np.all(np.isfinite(Y_pred)):
        raise ValueError("aggregate_r2: Y_true or Y_pred contains a non-finite value.")
    return float(r2_score(Y_true, Y_pred, multioutput=multioutput))


# --- D5-04: seed pooling -----------------------------------------------------------------------


def pool_seed_fields(fields: Dict[Any, np.ndarray], method: str) -> Dict[str, Any]:
    """Pool multiple seeds' length-``n`` ``||H||`` fields into one field. ``method`` is
    REQUIRED with no default -- see this module's docstring (b) for why a raw average is not
    a fair pooling rule on this milestone's measured seed spread. Supports exactly two
    methods, raising ``ValueError`` naming the offending value on anything else:

    * ``"per_seed_median_divide"`` -- divide each seed's field by its own median, then average
      the normalized fields across seeds.
    * ``"per_seed_rank_uniform"`` -- map each seed's field to its own percentile rank on
      ``[0, 1]`` via a stable argsort, then average the ranks across seeds.

    Returns a dict with ``pooled``, ``per_seed_normalized`` (dict keyed by seed),
    ``per_seed_median`` (dict keyed by seed, the RAW per-seed median before normalization),
    ``method``, ``n_seeds``, ``n``.
    """
    if not isinstance(fields, dict) or len(fields) == 0:
        raise ValueError("pool_seed_fields: fields must be a non-empty dict mapping seed to an array.")
    seeds = sorted(fields.keys())
    arrays = {s: np.asarray(fields[s], dtype=np.float64) for s in seeds}
    n = arrays[seeds[0]].shape[0]
    for s in seeds:
        a = arrays[s]
        if a.ndim != 1:
            raise ValueError(
                f"pool_seed_fields: field for seed {s} must be one-dimensional, got shape "
                f"{a.shape}."
            )
        if a.shape[0] != n:
            raise ValueError(
                f"pool_seed_fields: field for seed {s} has {a.shape[0]} rows, expected {n}."
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"pool_seed_fields: field for seed {s} contains a non-finite value.")

    per_seed_median: Dict[Any, float] = {s: float(np.median(arrays[s])) for s in seeds}
    per_seed_normalized: Dict[Any, np.ndarray] = {}

    if method == "per_seed_median_divide":
        for s in seeds:
            med = per_seed_median[s]
            per_seed_normalized[s] = arrays[s] / max(med, 1e-12)
    elif method == "per_seed_rank_uniform":
        denom = max(n - 1, 1)
        for s in seeds:
            order = np.argsort(arrays[s], kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            per_seed_normalized[s] = ranks / denom
    else:
        raise ValueError(
            f"pool_seed_fields: method={method!r} is not recognized; must be "
            '"per_seed_median_divide" or "per_seed_rank_uniform".'
        )

    stacked = np.stack([per_seed_normalized[s] for s in seeds], axis=0)
    pooled = stacked.mean(axis=0)

    return {
        "pooled": pooled,
        "per_seed_normalized": per_seed_normalized,
        "per_seed_median": per_seed_median,
        "method": method,
        "n_seeds": len(seeds),
        "n": int(n),
    }


# --- D5-07/D5-09: bucketing --------------------------------------------------------------------


def bucket_edges_from_field(values: np.ndarray, n_buckets: int) -> Tuple[float, ...]:
    """Equal-frequency rank partition over ALL supplied values: stable-argsort, then
    ``np.array_split`` into ``n_buckets`` groups. The edge between adjacent groups is the
    value of the first element of the later group. Returns ``n_buckets - 1`` edges as a tuple
    of floats.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"bucket_edges_from_field: values must be one-dimensional, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("bucket_edges_from_field: values contains a non-finite value.")
    if not isinstance(n_buckets, (int, np.integer)) or isinstance(n_buckets, bool) or n_buckets < 2:
        raise ValueError(f"bucket_edges_from_field: n_buckets={n_buckets} must be an int >= 2.")
    if values.shape[0] < n_buckets:
        raise ValueError(
            f"bucket_edges_from_field: values has {values.shape[0]} points, fewer than "
            f"n_buckets={n_buckets}."
        )
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, n_buckets)
    return tuple(float(values[g[0]]) for g in groups[1:])


def assign_buckets(values: np.ndarray, edges: Any) -> np.ndarray:
    """``np.searchsorted(edges, values, side="right")`` -- a value exactly equal to an edge
    lands in the HIGHER bucket. Stated, deterministic tie rule.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"assign_buckets: values must be one-dimensional, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("assign_buckets: values contains a non-finite value.")
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def bucket_by_field(values: np.ndarray, n_buckets: int) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Composes :func:`bucket_edges_from_field` and :func:`assign_buckets`. Returns
    ``(labels, edges)``.
    """
    edges = bucket_edges_from_field(values, n_buckets)
    labels = assign_buckets(values, edges)
    return labels, edges


def bucket_counts(labels: np.ndarray, n_buckets: int) -> Dict[str, Any]:
    """``np.bincount(labels, minlength=n_buckets)`` as plain ints plus fractions, plus
    ``n_total``. The counts sum exactly to ``labels.shape[0]``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"bucket_counts: labels must be one-dimensional, got shape {labels.shape}.")
    counts = np.bincount(labels, minlength=n_buckets).astype(np.int64)
    n_total = int(labels.shape[0])
    fractions = counts.astype(np.float64) / n_total if n_total > 0 else counts.astype(np.float64)
    return {
        "counts": counts,
        "fractions": fractions,
        "n_total": n_total,
        "n_buckets": int(n_buckets),
    }


# --- D5-08: bootstrap CI and the size-matched check ---------------------------------------------


def bucket_residual_ci(
    residuals: np.ndarray, n_resamples: int, seed: int, confidence_level: float
) -> Dict[str, Any]:
    """Copies ``mknn.bootstrap_ci``'s shape: required args with no default,
    ``scipy.stats.bootstrap((residuals,), np.mean, method="percentile",
    n_resamples=n_resamples, confidence_level=confidence_level,
    rng=np.random.default_rng(seed))``, flat result dict including ``degenerate``.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 1 or residuals.shape[0] < 2:
        raise ValueError(
            f"bucket_residual_ci: residuals must be one-dimensional with at least 2 points, "
            f"got shape {residuals.shape}."
        )
    if not np.all(np.isfinite(residuals)):
        raise ValueError("bucket_residual_ci: residuals contains a non-finite value.")

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (residuals,),
        np.mean,
        method="percentile",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)

    return {
        "score": float(residuals.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "degenerate": bool(ci_low == ci_high),
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n": int(residuals.shape[0]),
    }


_SIZE_MATCH_CI_N_RESAMPLES = 300
"""Internal CI resample count used inside :func:`size_matched_check`'s per-draw disjointness
check. Not itself a pre-registered constant -- it is an implementation detail of how the
size-match diagnostic is computed, not a value that changes what question is asked."""


def size_matched_check(
    residuals: np.ndarray, labels: np.ndarray, n_repeats: int, seed: int, confidence_level: float
) -> Dict[str, Any]:
    """D5-08 / RESEARCH Pitfall 4. ``residuals`` and ``labels`` MUST be the REALIZED
    TEST-SPLIT arrays -- never the full-field bucket assignment. ``n_match`` is
    ``min(bucket_counts(labels, ...))`` computed over the REALIZED test-split counts, never
    over the full-field bucket counts: the edges are cut on all points, so equal full-field
    counts say nothing about the counts a plain random train/test split leaves in each
    bucket, and using the full-field count here is exactly the artifact that undercut Phase
    4's verdict.

    For each of ``n_repeats`` draws under one ``np.random.default_rng(seed)``, samples
    ``n_match`` residuals without replacement from the lowest and the highest bucket, records
    the highest-minus-lowest mean difference and whether the two buckets' percentile CIs at
    ``confidence_level`` are disjoint in that draw. Returns ``n_match``,
    ``realized_bucket_counts`` (a tuple, one count per bucket label 0..n_buckets-1),
    ``median_diff``, ``sign_stable`` (all repeats share the sign of ``median_diff``),
    ``ci_disjoint_fraction``, ``n_repeats``, ``seed``.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    labels = np.asarray(labels)
    if residuals.shape[0] != labels.shape[0]:
        raise ValueError(
            f"size_matched_check: residuals has {residuals.shape[0]} rows but labels has "
            f"{labels.shape[0]} rows."
        )
    if not np.all(np.isfinite(residuals)):
        raise ValueError("size_matched_check: residuals contains a non-finite value.")
    if residuals.shape[0] == 0:
        raise ValueError("size_matched_check: residuals/labels must be non-empty.")

    n_buckets = int(labels.max()) + 1
    counts_info = bucket_counts(labels, n_buckets)
    realized_counts = tuple(int(c) for c in counts_info["counts"])
    n_match = int(min(realized_counts))
    if n_match < 1:
        raise ValueError(
            f"size_matched_check: smallest realized test-split bucket count is {n_match}, "
            "need at least 1."
        )

    bucket_indices = {b: np.flatnonzero(labels == b) for b in range(n_buckets)}
    lowest, highest = 0, n_buckets - 1

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_repeats, dtype=np.float64)
    disjoint = 0
    for i in range(n_repeats):
        low_sample = rng.choice(bucket_indices[lowest], size=n_match, replace=False)
        high_sample = rng.choice(bucket_indices[highest], size=n_match, replace=False)
        low_res = residuals[low_sample]
        high_res = residuals[high_sample]
        diffs[i] = float(high_res.mean() - low_res.mean())

        low_ci = bucket_residual_ci(
            low_res,
            n_resamples=_SIZE_MATCH_CI_N_RESAMPLES,
            seed=int(rng.integers(0, 2**31 - 1)),
            confidence_level=confidence_level,
        )
        high_ci = bucket_residual_ci(
            high_res,
            n_resamples=_SIZE_MATCH_CI_N_RESAMPLES,
            seed=int(rng.integers(0, 2**31 - 1)),
            confidence_level=confidence_level,
        )
        if low_ci["ci_high"] < high_ci["ci_low"] or high_ci["ci_high"] < low_ci["ci_low"]:
            disjoint += 1

    median_diff = float(np.median(diffs))
    sign_stable = bool(np.all(diffs > 0) or np.all(diffs < 0))

    return {
        "n_match": n_match,
        "realized_bucket_counts": realized_counts,
        "median_diff": median_diff,
        "sign_stable": sign_stable,
        "ci_disjoint_fraction": float(disjoint) / n_repeats if n_repeats > 0 else 0.0,
        "n_repeats": int(n_repeats),
        "seed": int(seed),
    }


# --- D5-09/D5-10: the verdict --------------------------------------------------------------


def apply_verdict_rule(
    bucket_stats: Any, size_match: Dict[str, Any], verdict_rule: str
) -> Dict[str, Any]:
    """Applies the frozen rule mechanically. ``bucket_stats`` is a sequence of per-bucket
    stat dicts in bucket-index order (each carrying at least ``score``, ``ci_low``,
    ``ci_high`` -- :func:`bucket_residual_ci`'s own return shape). ``size_match`` is
    :func:`size_matched_check`'s return dict. Raises ``RuntimeError`` if ``verdict_rule`` is
    empty, so this cannot run before the freeze. The two terminal verdicts are the strings
    ``"HOLDS"`` and ``"NO DETECTABLE RELATIONSHIP"``; there is no third outcome and no
    near-miss: HOLDS requires ALL of -- the lowest and highest bucket's CIs are disjoint, the
    highest bucket's mean residual exceeds the lowest bucket's, and the size-matched check's
    sign is stable across repeats.
    """
    if not isinstance(verdict_rule, str) or not verdict_rule.strip():
        raise RuntimeError(
            "apply_verdict_rule: verdict_rule is empty; cannot apply before the "
            "pre-registration freeze."
        )
    if bucket_stats is None or len(bucket_stats) < 2:
        raise ValueError("apply_verdict_rule: bucket_stats must carry at least 2 buckets.")

    lowest = bucket_stats[0]
    highest = bucket_stats[-1]
    ci_disjoint = bool(
        lowest["ci_high"] < highest["ci_low"] or highest["ci_high"] < lowest["ci_low"]
    )
    residual_higher_at_high_curvature = bool(highest["score"] > lowest["score"])
    sign_stable = bool(size_match.get("sign_stable", False))

    holds = ci_disjoint and residual_higher_at_high_curvature and sign_stable
    verdict = "HOLDS" if holds else "NO DETECTABLE RELATIONSHIP"

    return {
        "verdict": verdict,
        "criteria": {
            "ci_disjoint": ci_disjoint,
            "residual_higher_at_high_curvature": residual_higher_at_high_curvature,
            "size_match_sign_stable": sign_stable,
        },
    }


_VALID_PER_SEED_VERDICTS = ("HOLDS", "NO DETECTABLE RELATIONSHIP")
"""The two terminal per-seed verdict strings :func:`apply_verdict_rule` can produce -- the
only values :func:`combine_seed_verdicts` accepts as input."""


def combine_seed_verdicts(per_seed_verdicts: Dict[int, str], rule: str) -> Dict[str, Any]:
    """Maps exactly three per-seed terminal verdicts onto one of the three terminal
    phase-level outcomes -- the promotion this plan makes structural (05-03-DECISION.md): the
    verdict-bearing noun is now ``(seed, field)``, not ``field``, and no phase-level row can
    exist without three per-seed rows beside it.

    ``per_seed_verdicts`` maps seed int to that seed's terminal verdict string, each one of
    :data:`_VALID_PER_SEED_VERDICTS` (:func:`apply_verdict_rule`'s own output). ``rule`` is the
    frozen ``SEED_VERDICT_COMBINATION_RULE`` text. Counts the ``"HOLDS"`` outcomes:

    * three -- ``"HOLDS IN ALL THREE SEEDS"``
    * zero -- ``"NO DETECTABLE RELATIONSHIP IN ANY SEED"``
    * one or two -- ``"SPLIT ACROSS SEEDS"`` -- a complete, terminal, non-supportive outcome,
      not a near-miss awaiting a tie-break

    Computes no statistic, applies no threshold, takes no numeric argument -- the arithmetic
    already happened once per seed in :func:`apply_verdict_rule`. Raises ``RuntimeError`` when
    ``rule`` is empty or whitespace-only, mirroring :func:`apply_verdict_rule`'s pre-freeze
    guard, so this function cannot run before the ``05-04`` freeze. Raises ``ValueError`` when
    ``per_seed_verdicts`` does not hold exactly three entries (naming the count actually
    supplied), and ``ValueError`` when any value is not a member of
    :data:`_VALID_PER_SEED_VERDICTS` (naming the offending value).
    """
    if not isinstance(rule, str) or not rule.strip():
        raise RuntimeError(
            "combine_seed_verdicts: rule is empty; cannot run before the pre-registration "
            "freeze."
        )
    if not isinstance(per_seed_verdicts, dict) or len(per_seed_verdicts) != 3:
        n_seeds = len(per_seed_verdicts) if isinstance(per_seed_verdicts, dict) else None
        raise ValueError(
            "combine_seed_verdicts: per_seed_verdicts must hold exactly three seeds, got "
            f"{n_seeds if n_seeds is not None else per_seed_verdicts!r}."
        )
    for seed, verdict in per_seed_verdicts.items():
        if verdict not in _VALID_PER_SEED_VERDICTS:
            raise ValueError(
                f"combine_seed_verdicts: per-seed verdict for seed {seed} is {verdict!r}, not "
                f"one of {_VALID_PER_SEED_VERDICTS}."
            )

    sorted_seeds = sorted(per_seed_verdicts.keys())
    n_holds = sum(1 for s in sorted_seeds if per_seed_verdicts[s] == "HOLDS")
    if n_holds == 3:
        phase_verdict = "HOLDS IN ALL THREE SEEDS"
    elif n_holds == 0:
        phase_verdict = "NO DETECTABLE RELATIONSHIP IN ANY SEED"
    else:
        phase_verdict = "SPLIT ACROSS SEEDS"

    return {
        "phase_verdict": phase_verdict,
        "n_holds": n_holds,
        "n_seeds": len(sorted_seeds),
        "per_seed_verdicts": {s: per_seed_verdicts[s] for s in sorted_seeds},
        "rule": rule,
    }
