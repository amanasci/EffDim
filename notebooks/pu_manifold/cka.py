"""Phase 8 CKA estimator: the Song et al. (2012) unbiased-HSIC estimator, linear and RBF Gram
builders, and the pre-registration freeze machinery for Phase 8's curvature-conditioned CKA
alignment work.

**This module adds; it does not edit.** ``notebooks/pu_manifold/density_stratified_null.py``
(Phase 07.1) and ``notebooks/pu_manifold/crossmodal_curvature.py`` (Phase 7, sealed by D7-05)
are never imported for a gating VALUE here -- every constant this module needs is re-declared as
a fresh top-level literal, even where a later value might coincide with either sealed module's
own. D7-05/D8-23 sealed those modules as import-never-edit, and a gating constant imported
ACROSS that freeze boundary would not be covered by this module's own
``assert_preregistered()`` or by this phase's own git-ancestry proof.

**The constants below are UNSET in this commit.** Every name in ``_REQUIRED_CONSTANTS`` is
declared with its UNSET sentinel (``None`` for scalars, ``()`` for tuples, ``""`` for rule
strings) -- ``KERNELS = ()``, and so on down the block. They are filled, all of them, in ONE
later commit: Phase 8's single freeze commit (D8-22), which must be a strict git ancestor of
every commit that computes a Phase 8 number. A later edit to any of them after a Phase 8 number
exists anywhere in the tree is a pre-registration BREACH: the only remedy is a fresh freeze and
a fresh run, never a silent fix (mirrors D7-06's discipline, carried into this phase's own
constants exactly as ``density_stratified_null.py`` carried it into 07.1's).

**This plan (08-01) produces NO Phase 8 number.** ``--mode selfcheck`` in the accompanying
runner drives the estimator through D8-16's invariance ladder on synthetic pairs whose CKA
answer is known in closed form -- it never calls :func:`assert_preregistered`, because it opens
no PU file and computes no number this phase will ever claim.

**Supersession, not an edit.** ``crossmodal_curvature.py`` line 109 freezes
``ALIGNMENT_METRIC = "mknn"`` under D7-07 ("CKA is out of scope and not implemented anywhere in
this codebase"). Phase 8 supersedes that scope decision BY PHASE DECISION, taken by the
developer on 2026-08-27 and recorded in ``08-CONTEXT.md`` -- never by patching the sealed
module. ``SUPERSEDES`` (filled at the freeze) names ``crossmodal_curvature.ALIGNMENT_METRIC``
as a positive, checkable fact; ``SUPERSESSION_RULE`` (also filled at the freeze) states the
supersession in prose. Phase 7's own ``ALIGNMENT_METRIC = "mknn"`` remains true of Phase 7's own
record rows and is not reinterpreted.

**The Swiss roll standing rule (CLAUDE.md) does not apply here, by decision (D8-17).** CKA has
no decoder and no representation map -- it is a statistic computed over two representations that
already exist. The rule's purpose (telling a broken implementation apart from a real FAIL on
data with no known answer) is served instead by D8-16's invariance ladder, whose answers are
known in closed form. ``SWISS_ROLL_APPLICABILITY_RULE`` (filled at the freeze) carries this
declaration as a checkable fact, not only as this docstring's prose.

No file I/O happens anywhere in this module, following ``crossmodal_curvature.py``'s and
``density_stratified_null.py``'s stated convention: a default is how a pre-registered value gets
inherited by accident instead of by an explicit call-site choice. Every pure function below
takes its parameters as explicit arguments -- ``sigma`` in particular has no default anywhere in
this module, so no call site that only sees a subset of the full point cloud can silently supply
a per-subset bandwidth (D8-03's named confound).
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

# =============================================================================================
# Frozen constants block -- ALL FOURTEEN UNSET IN THIS COMMIT. Filled, together, in the single
# 08-04 freeze commit (D8-22). Never filled piecemeal, never filled here.
# =============================================================================================

KERNELS = ()
"""At the freeze: the tuple of kernel names this phase computes CKA for, e.g.
``("linear", "rbf")`` (D8-01)."""

SIGMA_MULTIPLIERS = ()
"""At the freeze: the RBF bandwidth sensitivity ladder, e.g. ``(0.5, 1.0, 2.0)`` (D8-04)."""

SIGMA_HSC = None
"""At the freeze: the frozen RBF bandwidth for the HSC modality -- the median pairwise Euclidean
distance over all 10,000 HSC points, computed once, before any subset exists (D8-03)."""

SIGMA_LEGACYSURVEY = None
"""At the freeze: the frozen RBF bandwidth for the Legacy Survey modality, computed the same
way, independently, over all 10,000 Legacy Survey points (D8-03)."""

GRAM_DTYPE = ""
"""At the freeze: the storage dtype for the precomputed Gram matrices, e.g. ``"float32"``
(discretion decision, RESEARCH.md A3's memory argument)."""

HSIC_ESTIMATOR_RULE = ""
"""At the freeze: the prose rule naming the Song et al. (2012) unbiased-HSIC form and the
double-centering trap it must never fall into (D8-02)."""

SIGMA_FREEZE_RULE = ""
"""At the freeze: the prose rule stating sigma is computed once, globally, per modality, before
any subset exists, and reused unchanged for every subset/d/seed/S (D8-03)."""

ALIGNMENT_METRIC = ""
"""At the freeze: this phase's own alignment-metric name, e.g. ``"cka"`` -- Phase 8's own
checkable fact, distinct from and not overwriting ``crossmodal_curvature.ALIGNMENT_METRIC``."""

SUPERSEDES = ()
"""At the freeze: names the sealed constant this phase supersedes by decision --
``crossmodal_curvature.ALIGNMENT_METRIC`` -- as a positive, checkable fact (see module
docstring's "Supersession, not an edit" section)."""

SUPERSESSION_RULE = ""
"""At the freeze: the prose rule stating that Phase 8 supersedes D7-07's CKA-out-of-scope
decision by phase decision, never by editing the sealed module."""

SWISS_ROLL_APPLICABILITY_RULE = ""
"""At the freeze: the prose rule recording D8-17's declaration that the CLAUDE.md Swiss roll
standing rule is NOT APPLICABLE to Phase 8, on purpose (see module docstring)."""

RBF_IS_NON_GATING = None
"""At the freeze: ``True`` -- RBF CKA is reported as robustness and gates nothing; linear CKA
alone carries the headline verdict (D8-01)."""

SIGMA_LADDER_IS_NON_GATING = None
"""At the freeze: ``True`` -- the 0.5x/2x sigma sensitivity rungs are diagnostics only and gate
nothing; only the ``sigma`` rung itself feeds the headline (D8-04)."""

DIAGNOSTICS_ARE_NON_GATING = None
"""At the freeze: ``True`` -- the D7-03 non-gating-diagnostic pattern, carried into this phase
for every diagnostic quantity it reports beside a verdict."""

# --- 08-02 additions: the within-density-stratum tertile split (D8-05/06/07/08) --------------

S_GRID = ()
"""At the freeze: the threshold grid of stratum counts ``S`` this phase's tertile split and null
are computed at, e.g. ``(10, 20, 50)`` -- a grid of THRESHOLDS, not a headline value (D8-08). See
``SENSITIVITY_GRID_RULE`` below for what a reader may and may not do with it."""

N_TERTILES = None
"""At the freeze: ``3`` -- the number of ``||H||``-magnitude buckets the within-stratum split
produces (D8-05). Not a discretion value: Phase 8's whole design is built on three tertiles."""

DENSITY_K = None
"""At the freeze: the ``k`` used by ``curvature_probe.local_density_weights`` to build the
per-point density field this phase stratifies on -- re-declared fresh, inherited unchanged from
``crossmodal_curvature.py``'s own ``DENSITY_K = 30`` (D8-07), never imported across the freeze
boundary."""

DENSITY_FIELD_D = None
"""At the freeze: the ambient dimension the density field is computed at -- re-declared fresh,
inherited unchanged from ``crossmodal_curvature.py``'s own ``DENSITY_FIELD_D = 20`` (D8-07)."""

DENSITY_INPUT = ""
"""At the freeze: which modality's embedding the density field is computed over, e.g.
``"legacysurvey_ambient_768"`` -- re-declared fresh from ``crossmodal_curvature.py``'s own
``DENSITY_INPUT`` (D8-07)."""

DENSITY_SIGN_CONVENTION = ""
"""At the freeze: the prose rule stating D8-07's sign convention --
``curvature_probe.local_density_weights`` returns the per-point INVERSE density ``w``,
mean-normalized to 1; the density used throughout this phase is the RELATIVE density
``1.0 / w``, matching Phase 4's printed convention (``region_partition_mknn_run.py`` REGN-01) so
Phase 4 / 7 / 07.1 / 8 density numbers stay comparable rather than needing translation."""

STRATIFICATION_RULE = ""
"""At the freeze: the prose rule naming ``density_stratified_null.density_strata``'s exact
binning convention this phase reuses (equal-count quantile bins on density RANK, stable-sort
tie-breaking, remainder-to-last-stratum), PLUS D8-06's semantic consequence: the tertiles this
phase computes rank DENSITY-RESIDUALIZED CURVATURE, not raw ``||H||``."""

SENSITIVITY_GRID_RULE = ""
"""At the freeze: the prose rule stating D8-08/D8-09's grid semantics -- ``S_GRID`` is a grid of
THRESHOLDS, not point estimates; there is NO headline ``S``; clearance is required at EVERY grid
point; an ``S``-dependent gap is self-reporting as an artifact rather than something a reader has
to notice."""


_REQUIRED_CONSTANTS = (
    "KERNELS",
    "SIGMA_MULTIPLIERS",
    "SIGMA_HSC",
    "SIGMA_LEGACYSURVEY",
    "GRAM_DTYPE",
    "HSIC_ESTIMATOR_RULE",
    "SIGMA_FREEZE_RULE",
    "ALIGNMENT_METRIC",
    "SUPERSEDES",
    "SUPERSESSION_RULE",
    "SWISS_ROLL_APPLICABILITY_RULE",
    "RBF_IS_NON_GATING",
    "SIGMA_LADDER_IS_NON_GATING",
    "DIAGNOSTICS_ARE_NON_GATING",
    "S_GRID",
    "N_TERTILES",
    "DENSITY_K",
    "DENSITY_FIELD_D",
    "DENSITY_INPUT",
    "DENSITY_SIGN_CONVENTION",
    "STRATIFICATION_RULE",
    "SENSITIVITY_GRID_RULE",
)
"""Every gating constant this module declares, in declaration order. A constant added later
without a guard entry here fails the parametrized rejection sweep in
``tests/test_cka.py::test_assert_preregistered_rejects_unset_constant`` -- that is the mechanism
this tuple exists to serve."""


def assert_preregistered() -> None:
    """Refuse to proceed while any pre-registered Phase 8 constant is UNSET.

    One check per name in :data:`_REQUIRED_CONSTANTS`, in declaration order, raising
    ``RuntimeError`` on the FIRST failure. A value is UNSET if it is ``None``, an empty tuple,
    or an empty-or-whitespace-only string -- the three UNSET sentinels this module's own
    constants block uses. In THIS commit every one of the fourteen constants is UNSET, so this
    function raises on ``KERNELS`` (the first name in declaration order) -- that is the intended
    state; the 08-04 freeze commit is the single commit that fills all fourteen at once.
    """
    g = globals()
    for name in _REQUIRED_CONSTANTS:
        value = g.get(name, None)
        is_unset = (
            value is None
            or (isinstance(value, tuple) and len(value) == 0)
            or (isinstance(value, str) and not value.strip())
        )
        if is_unset:
            raise RuntimeError(
                f"assert_preregistered: {name}={value!r} is UNSET. Every Phase 8 gating "
                "constant must be filled by the single 08-04 freeze commit (D8-22) before any "
                "Phase 8 number may be computed. A later edit to a filled constant after any "
                "Phase 8 number exists is a pre-registration breach -- the only remedy is a "
                "fresh freeze and a fresh run."
            )


# =============================================================================================
# Estimator functions -- pure numpy, no file I/O, no module-level default parameters that could
# be inherited silently across a call site.
# =============================================================================================


def _zero_diag(K: np.ndarray) -> np.ndarray:
    """Copy of `K` with the diagonal set to 0.0. Never mutates the caller's array."""
    K = np.asarray(K).copy()
    np.fill_diagonal(K, 0.0)
    return K


def unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """The Song et al. (2012) unbiased HSIC estimator, computed on RAW Gram matrices with only
    the diagonal zeroed.

    ``HSIC_1(K, L) = 1/(n(n-3)) * [ tr(K~L~) + (1'K~1)(1'L~1)/((n-1)(n-2)) - (2/(n-2))*1'K~L~1 ]``
    where ``K~``/``L~`` are `K`/`L` with the diagonal zeroed.

    **CRITICAL: `K` and `L` must be the RAW Gram matrices, only zero-diagonalized -- never
    double-centered (`H K H`) first.** The `1/(n(n-3))` correction terms above already perform
    the debiasing; applying them to a pre-centered matrix silently reproduces (a scaled variant
    of) the *biased* estimator under this unbiased formula's name. This is the exact trap D8-02
    exists to avoid; ``tests/test_cka.py::test_double_centering_changes_the_answer`` pins it
    behaviorally.

    Raises ``ValueError`` on non-square, shape-mismatched, or non-finite input before doing any
    arithmetic, and on ``n <= 3`` (the estimator's own floor -- `(n-1)(n-2)` and `(n-2)` in the
    denominators must be non-zero and positive).
    """
    K = np.asarray(K)
    L = np.asarray(L)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"unbiased_hsic: K has shape {K.shape}; must be a square 2D array.")
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"unbiased_hsic: L has shape {L.shape}; must be a square 2D array.")
    if K.shape != L.shape:
        raise ValueError(
            f"unbiased_hsic: K has shape {K.shape} but L has shape {L.shape}; they must match."
        )
    if not np.all(np.isfinite(K)):
        raise ValueError("unbiased_hsic: K contains non-finite values.")
    if not np.all(np.isfinite(L)):
        raise ValueError("unbiased_hsic: L contains non-finite values.")
    n = K.shape[0]
    if n <= 3:
        raise ValueError(f"unbiased_hsic: n={n} must exceed 3 (Song et al. 2012 floor).")
    Kt, Lt = _zero_diag(K), _zero_diag(L)
    ones = np.ones(n)
    term1 = np.trace(Kt @ Lt)
    term2 = (ones @ Kt @ ones) * (ones @ Lt @ ones) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ Kt @ Lt @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """Centered Kernel Alignment, composed from the unbiased HSIC estimator above:
    ``CKA(K, L) = HSIC_1(K, L) / sqrt(HSIC_1(K, K) * HSIC_1(L, L))``.

    `K` and `L` are RAW Gram matrices (see :func:`unbiased_hsic`'s critical note); this function
    never centers them itself.
    """
    hsic_kl = unbiased_hsic(K, L)
    hsic_kk = unbiased_hsic(K, K)
    hsic_ll = unbiased_hsic(L, L)
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


def linear_gram(X: np.ndarray, dtype: Any) -> np.ndarray:
    """Linear kernel Gram matrix, ``X @ X.T``, cast to `dtype`. `dtype` is a required, explicit
    call-site argument -- never a module-level default -- so a caller can never silently inherit
    a stale precision choice."""
    X = np.asarray(X)
    return (X @ X.T).astype(dtype)


def median_pairwise_distance(X: np.ndarray) -> float:
    """D8-03's sigma: the median Euclidean pairwise distance over ALL rows of `X`. In
    production this means all 10,000 points of one modality, computed once, before any subset
    (tertile, stratum, permutation) ever exists. Never call this on a subset -- :func:`rbf_gram`
    requires `sigma` explicitly and has no default specifically so this mistake cannot happen
    silently at a call site that only has access to a subset (D8-03's named confound)."""
    X = np.asarray(X)
    return float(np.median(pdist(X, metric="euclidean")))


def rbf_gram(X: np.ndarray, sigma: float, dtype: Any) -> np.ndarray:
    """RBF/Gaussian kernel Gram matrix at a REQUIRED, explicit bandwidth `sigma` -- no default
    value, by design. A call site that sees only a subset of the full point cloud can never
    silently compute and use a per-subset bandwidth, because there is nothing to fall back on if
    `sigma` is omitted (D8-03's named confound, restated as an interface property).

    Raises ``ValueError`` when `sigma` is non-finite or `sigma <= 0`.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"rbf_gram: sigma={sigma!r} must be finite and > 0.")
    X = np.asarray(X)
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))
    K = np.exp(-sq_dists / (2.0 * sigma ** 2))
    return K.astype(dtype)


def cka_on_subset(K_full: np.ndarray, L_full: np.ndarray, idx: np.ndarray) -> float:
    """CKA on the subset named by `idx`, computed via submatrix indexing into already-built
    full Gram matrices: ``K_full[np.ix_(idx, idx)]`` / ``L_full[np.ix_(idx, idx)]`` then
    :func:`cka`. This is EXACT, not an approximation -- a kernel value `K(x_i, x_j)` depends
    only on the pair `(x_i, x_j)`, never on which other points are present in the batch. This is
    the Gram-matrix-once/index-many architecture this phase's entire runtime budget depends on
    (08-RESEARCH.md's Runtime/Cost Model)."""
    idx = np.asarray(idx)
    K_sub = K_full[np.ix_(idx, idx)]
    L_sub = L_full[np.ix_(idx, idx)]
    return cka(K_sub, L_sub)


# =============================================================================================
# 08-02 additions: the within-density-stratum tertile split and the realized-contrast
# diagnostic (D8-05/06/07/08). ``strata`` is always an array already produced by
# ``density_stratified_null.density_strata(density, S)`` at some call site upstream of these
# functions -- imported there as a pure function only; no gating value ever crosses the freeze
# boundary. These functions never call ``density_strata`` themselves; they only ever consume the
# stratum-id array it produces, exactly as D8-06's split is specified to be built ON TOP of it,
# never a reimplementation of it.
# =============================================================================================


def tertile_split_within_strata(
    h: np.ndarray, strata: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D8-06's within-density-stratum ``||H||`` tertile split.

    For each unique stratum id in `strata`, rank that stratum's points by `h` using a stable
    argsort (ascending), then cut into three contiguous rank blocks of size ``n_s // 3`` with the
    ``n_s % 3`` remainder going to the LAST (highest-``h``) block -- the same remainder-to-last
    convention ``density_stratified_null.density_strata`` itself uses when dividing `n` points
    into `S` strata, so the two binning rules agree rather than each inventing their own. The
    per-stratum blocks are pooled across strata into three global index arrays, each returned
    sorted ascending.

    Because the split is computed WITHIN each stratum independently, tertile 3 holds the
    highest-``h`` third within every stratum, never the globally highest third -- this is what
    makes the three returned subsets' density-stratum marginals identical by construction
    (D8-06), up to each stratum's own ``n_s % 3`` remainder.

    Raises ``ValueError`` when `h` and `strata` have different lengths, when `h` contains a
    non-finite value, or when any stratum holds fewer than 3 points (naming the offending
    stratum and its size) -- a stratum that small cannot support a three-way split at all.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    strata = np.asarray(strata).ravel()
    if h.shape[0] != strata.shape[0]:
        raise ValueError(
            f"tertile_split_within_strata: h has {h.shape[0]} entries but strata has "
            f"{strata.shape[0]}; they must be row-aligned."
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("tertile_split_within_strata: h contains non-finite values.")

    tertile_blocks: Tuple[list, list, list] = ([], [], [])
    for stratum_id in np.unique(strata):
        idx = np.where(strata == stratum_id)[0]
        n_s = idx.shape[0]
        if n_s < 3:
            raise ValueError(
                f"tertile_split_within_strata: stratum {stratum_id!r} holds {n_s} point(s), "
                "below the 3-point floor a within-stratum tertile split requires."
            )
        order = idx[np.argsort(h[idx], kind="stable")]
        bin_size = n_s // 3
        tertile_blocks[0].append(order[:bin_size])
        tertile_blocks[1].append(order[bin_size:2 * bin_size])
        tertile_blocks[2].append(order[2 * bin_size:])  # remainder -> last (highest-h) block

    return tuple(np.sort(np.concatenate(blocks)) for blocks in tertile_blocks)


def realized_h_contrast(h: np.ndarray, tertiles: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """D8-21's mandatory "realized ``||H||`` contrast per `S`" row: the tertile-3 median of `h`
    over the tertile-1 median, strictly greater than 1.0 whenever `h` is non-constant. This is
    the number that makes D8-18's planted effect calibratable against PU's measured ~1.5x
    spread. Reported, never gated on."""
    h = np.asarray(h, dtype=np.float64).ravel()
    t1, _t2, t3 = tertiles
    return float(np.median(h[t3]) / np.median(h[t1]))
