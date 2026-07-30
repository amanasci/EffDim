"""
Seeded, row-alignment-safe subsampling of the ``legacysurvey_dinov3_vitb16`` config of
``UniverseTBD/pu-embeddings``.

There is no ``object_id`` anywhere in this dataset (see ``PROJECT.md``); the *only* join
between the ``dinov3_vitb16_hsc`` and ``dinov3_vitb16_legacysurvey`` columns is that they
are already row-aligned in the upstream parquet. ``load_subsample`` therefore reads both
columns off exactly one sorted, seeded index array in a single indexing pass
(:func:`draw_row_indices` + one ``ds.with_format("numpy")[row_indices]`` call) -- never two
independent selections, which would silently break the alignment
(``ARCHITECTURE.md`` Anti-Pattern 3). :func:`assert_alignment` is the runtime proof that the
alignment held: a structural check (shapes, finiteness, row-index hash) plus a statistical
smoke test (a z-score against a permuted null) that a genuine off-by-one misalignment fails.

Both embedding columns are L2-normalized here, at cache-write time (D-05/D-06), so the
cached ``subsample_*.npz`` artifact never mixes a normalized and a raw array. The raw
pre-normalization norms are retained separately (``hsc_norms``/``ls_norms``) so the DATA-04
norm-distribution histogram stays reproducible without re-streaming the parquet.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import cache

# Feature width of both embedding columns.
N_FEATURES = 768

# Row count of the legacysurvey_dinov3_vitb16 config, per PROJECT.md. load_subsample
# asserts the loaded config reports exactly this many rows (T-01-02 mitigation: catches a
# silently changed upstream file).
EXPECTED_N_TOTAL = 101_725

# A dense geodesic distance matrix over the full EXPECTED_N_TOTAL rows would be roughly
# 83 GB (101_725**2 float64 entries). This cap keeps every Isomap fit in this milestone
# tractable on a single machine (T-01-05 mitigation).
MAX_N_ROWS = 20_000

# Strict z-score margin for the D-08 statistical row-alignment smoke test. A z of exactly
# ALIGNMENT_MARGIN_Z is treated as insufficient and FAILS -- the comparison in
# assert_alignment is `>`, not `>=`.
ALIGNMENT_MARGIN_Z = 5.0

# Number of independent seeded permutations used to estimate the null mean/std for the
# alignment smoke test.
ALIGNMENT_N_PERMUTATIONS = 50


def draw_row_indices(n_total: int, n_rows: int, seed: int) -> np.ndarray:
    """
    Draw a deterministic, sorted, duplicate-free sample of row indices.

    Parameters:
    -----------
    n_total : int
        Total number of rows available to sample from.
    n_rows : int
        Number of rows to draw.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns:
    --------
    np.ndarray
        ``np.sort(np.random.default_rng(seed).choice(n_total, n_rows, replace=False))``.
        ``replace=False`` admits no equal elements and ``np.sort`` fixes one deterministic
        total order (DATA-03). Both paired columns must be read off this single array in
        one indexing pass, never via two independent shuffle/select calls.

    Raises:
    -------
    ValueError
        If ``n_rows < 2``, if ``n_rows > MAX_N_ROWS`` (a dense-geodesic fit at the full
        dataset size would need ~83 GB), or if ``n_total < n_rows``.
    """
    if n_rows < 2:
        raise ValueError(f"n_rows must be at least 2, got {n_rows}.")
    if n_rows > MAX_N_ROWS:
        raise ValueError(
            f"n_rows={n_rows} exceeds MAX_N_ROWS={MAX_N_ROWS}. A dense geodesic distance "
            f"matrix over the full {EXPECTED_N_TOTAL} rows would be roughly 83 GB "
            f"({EXPECTED_N_TOTAL}**2 float64 entries); this cap keeps the Isomap fit "
            f"tractable on a single machine."
        )
    if n_total < n_rows:
        raise ValueError(
            f"n_total={n_total} is smaller than n_rows={n_rows}; cannot draw {n_rows} "
            f"rows without replacement from only {n_total}."
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, n_rows, replace=False))


def l2_normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    L2-normalize each row of a 2D array, returning the unit rows and the raw norms.

    Parameters:
    -----------
    x : np.ndarray
        A ``(n_rows, n_features)`` array.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        ``(x / norms[:, None], norms)`` where ``norms = np.linalg.norm(x, axis=1)``.

    Raises:
    -------
    ValueError
        If any row has a norm of exactly zero (would divide by zero).
    """
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            "l2_normalize received at least one zero-norm row; cannot normalize a "
            "zero vector to the unit sphere."
        )
    return x / norms[:, None], norms


def row_indices_sha256(row_indices: np.ndarray) -> str:
    """
    Compute a sha256 hash over a row-index array's C-contiguous int64 bytes.

    Parameters:
    -----------
    row_indices : np.ndarray
        The row-index array to hash.

    Returns:
    --------
    str
        Hex digest of the sha256 hash of ``row_indices``, cast to ``int64`` and made
        C-contiguous first so the hash is independent of dtype/layout quirks.
    """
    contiguous = np.ascontiguousarray(row_indices, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def assert_structural_alignment(
    hsc: np.ndarray,
    legacysurvey: np.ndarray,
    row_indices: np.ndarray,
    expected_sha256: Optional[str] = None,
) -> str:
    """
    Raise unless the two paired arrays and the row-index array are structurally consistent.

    Parameters:
    -----------
    hsc : np.ndarray
        The HSC embedding array, shape ``(n_rows, N_FEATURES)``.
    legacysurvey : np.ndarray
        The Legacy Survey embedding array, shape ``(n_rows, N_FEATURES)``.
    row_indices : np.ndarray
        The row-index array both arrays were selected with.
    expected_sha256 : Optional[str]
        If given, the recomputed hash of ``row_indices`` must match this value.

    Returns:
    --------
    str
        The (recomputed) sha256 hash of ``row_indices``.

    Raises:
    -------
    ValueError
        If ``hsc.shape != legacysurvey.shape``, if ``hsc.shape[0] != row_indices.shape[0]``,
        if ``hsc.shape[1] != N_FEATURES``, if either array contains a non-finite value, or
        if ``expected_sha256`` is given and does not match the recomputed hash.
    """
    if hsc.shape != legacysurvey.shape:
        raise ValueError(
            f"hsc.shape {hsc.shape} != legacysurvey.shape {legacysurvey.shape}; the two "
            f"paired columns must have identical shape."
        )
    if hsc.shape[0] != row_indices.shape[0]:
        raise ValueError(
            f"hsc.shape[0]={hsc.shape[0]} != row_indices.shape[0]={row_indices.shape[0]}."
        )
    if hsc.shape[1] != N_FEATURES:
        raise ValueError(f"hsc.shape[1]={hsc.shape[1]} != N_FEATURES={N_FEATURES}.")
    if not np.all(np.isfinite(hsc)):
        raise ValueError("hsc contains non-finite (NaN or infinite) values.")
    if not np.all(np.isfinite(legacysurvey)):
        raise ValueError("legacysurvey contains non-finite (NaN or infinite) values.")

    computed_sha256 = row_indices_sha256(row_indices)
    if expected_sha256 is not None and computed_sha256 != expected_sha256:
        raise ValueError(
            f"row_indices sha256 mismatch: computed {computed_sha256} != "
            f"expected {expected_sha256}. The row ordering used to build hsc/legacysurvey "
            f"does not match the row ordering this check was told to expect."
        )
    return computed_sha256


def alignment_smoke_test(
    hsc: np.ndarray,
    legacysurvey: np.ndarray,
    seed: int,
    n_permutations: int = ALIGNMENT_N_PERMUTATIONS,
) -> Dict[str, Any]:
    """
    Statistical row-alignment smoke test: a scale-free z-score against a permuted null.

    Parameters:
    -----------
    hsc : np.ndarray
        L2-normalized HSC embedding array, shape ``(n_rows, N_FEATURES)``.
    legacysurvey : np.ndarray
        L2-normalized Legacy Survey embedding array, shape ``(n_rows, N_FEATURES)``.
    seed : int
        Seed for the permutation draws (``np.random.default_rng(seed)``).
    n_permutations : int
        Number of independent seeded permutations of ``legacysurvey`` to draw for the null.

    Returns:
    --------
    Dict[str, Any]
        ``{"s_true", "mu_perm", "sd_perm", "z", "margin_z", "n_permutations"}``.
        ``s_true`` is the mean per-row cosine (plain dot product, since inputs are already
        L2-normalized) between the true pairing. ``mu_perm``/``sd_perm`` are the mean and
        standard deviation of that same statistic over ``n_permutations`` independent
        permutations of ``legacysurvey``. ``z = (s_true - mu_perm) / sd_perm``.

    Raises:
    -------
    ValueError
        If ``sd_perm`` is exactly zero (the null has no spread to compare against).

    Notes:
    ------
    Scale-free by construction: the origin paper (arXiv:2509.19453) reports Legacy Survey
    crossmodal MKNN at only 0.4-2%, so an absolute-cosine margin would risk rejecting a
    genuinely correct but weak pairing. A z-score against the permuted null adapts to
    whatever the true-pair cosine actually is. The null's spread is a standard error over
    ``n_rows`` rows, so a correct alignment -- even a weak one -- puts ``s_true`` many
    standard errors above ``mu_perm``; a gross misalignment (off-by-one, independent
    re-sort) makes ``s_true`` itself a draw from the permuted distribution, so ``z`` lands
    near 0.
    """
    s_true = float(np.mean(np.sum(hsc * legacysurvey, axis=1)))

    rng = np.random.default_rng(seed)
    n_rows = hsc.shape[0]
    perm_means = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        permuted = rng.permutation(n_rows)
        perm_means[i] = np.mean(np.sum(hsc * legacysurvey[permuted], axis=1))

    mu_perm = float(np.mean(perm_means))
    sd_perm = float(np.std(perm_means))
    if sd_perm == 0.0:
        raise ValueError(
            "Permutation null has zero spread (sd_perm == 0); cannot compute a z-score."
        )
    z = (s_true - mu_perm) / sd_perm

    return {
        "s_true": s_true,
        "mu_perm": mu_perm,
        "sd_perm": sd_perm,
        "z": z,
        "margin_z": ALIGNMENT_MARGIN_Z,
        "n_permutations": n_permutations,
    }


def assert_alignment(
    hsc: np.ndarray,
    legacysurvey: np.ndarray,
    row_indices: np.ndarray,
    seed: int,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the structural check then the statistical smoke test; raise unless both pass.

    Parameters:
    -----------
    hsc : np.ndarray
        L2-normalized HSC embedding array.
    legacysurvey : np.ndarray
        L2-normalized Legacy Survey embedding array.
    row_indices : np.ndarray
        The row-index array both arrays were selected with.
    seed : int
        Seed for the permutation-null draws.
    expected_sha256 : Optional[str]
        If given, forwarded to :func:`assert_structural_alignment`.

    Returns:
    --------
    Dict[str, Any]
        The stats dict from :func:`alignment_smoke_test`, with ``row_indices_sha256`` added.

    Raises:
    -------
    ValueError
        If the structural check fails, or if ``stats["z"] <= ALIGNMENT_MARGIN_Z`` -- the
        comparison is strict, so a z of exactly ``ALIGNMENT_MARGIN_Z`` is insufficient and
        raises. This check must never be weakened, skipped, or downgraded to a warning
        (DATA-03 prohibition): there is no ``object_id`` in this dataset, so this assertion
        is the only correctness invariant the whole milestone rests on.
    """
    computed_sha256 = assert_structural_alignment(
        hsc, legacysurvey, row_indices, expected_sha256
    )
    stats = alignment_smoke_test(hsc, legacysurvey, seed)
    if not stats["z"] > ALIGNMENT_MARGIN_Z:
        raise ValueError(
            f"Row-alignment smoke test failed: z={stats['z']:.4f} is not > "
            f"ALIGNMENT_MARGIN_Z={ALIGNMENT_MARGIN_Z}. s_true={stats['s_true']:.6f}, "
            f"mu_perm={stats['mu_perm']:.6f}, sd_perm={stats['sd_perm']:.6f}, "
            f"n_permutations={stats['n_permutations']}. Halting rather than proceeding "
            f"with a possibly misaligned pairing."
        )
    stats["row_indices_sha256"] = computed_sha256
    return stats


def load_subsample(cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Load (or compute-and-cache) a seeded, L2-normalized subsample of the paired columns.

    Parameters:
    -----------
    cfg : Dict[str, Any]
        Must contain ``"dataset"`` (the HF config name), ``"seed"``, and ``"n_rows"``.
        Extra keys (e.g. ``n_neighbors``, ``n_components``) are ignored here -- see Notes.

    Returns:
    --------
    Dict[str, np.ndarray]
        ``{"hsc", "legacysurvey", "hsc_norms", "ls_norms", "row_indices"}``. ``hsc`` and
        ``legacysurvey`` are L2-normalized (D-05/D-06); ``hsc_norms``/``ls_norms`` are the
        raw pre-normalization norms, retained so the DATA-04 norm histogram is reproducible
        without re-streaming; ``row_indices`` is the sorted seeded index array both columns
        were read off.

    Raises:
    -------
    ValueError
        If ``cfg["n_rows"] > MAX_N_ROWS`` (checked before anything else, including before
        the ``datasets`` import). Propagated from :func:`draw_row_indices` for the other
        degenerate cases, and from the loaded-row-count assertion below if the upstream
        config no longer reports ``EXPECTED_N_TOTAL`` rows.

    Notes:
    ------
    ``datasets`` is imported lazily, inside this function, never at module top level, so
    this module imports cleanly with only numpy and joblib present.

    The cache key this function computes is deliberately **narrower** than the full ``cfg``
    it is called with: only ``dataset``, ``seed``, ``n_rows``, ``normalize``, the installed
    ``datasets`` version, and the installed ``numpy`` version participate. This is a stated
    refinement of D-14 (see the Phase 1 plan's "Cache-key refinement" note): the subsample
    artifact must not be invalidated by changing a fit-only parameter such as
    ``n_neighbors`` or ``n_components``, which would otherwise force an unnecessary
    re-download and re-subsample. The Isomap *fit* cache (built by the notebook, not this
    function) uses the full D-14 field set instead.
    """
    if cfg["n_rows"] > MAX_N_ROWS:
        raise ValueError(
            f"cfg['n_rows']={cfg['n_rows']} exceeds MAX_N_ROWS={MAX_N_ROWS}; see "
            f"draw_row_indices for the ~83 GB dense-geodesic-matrix rationale."
        )

    import datasets as hf_datasets  # lazy: keep this module importable with numpy+joblib only

    subsample_cfg = {
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "n_rows": cfg["n_rows"],
        "normalize": cfg["normalize"],
        "datasets_version": hf_datasets.__version__,
        "numpy_version": np.__version__,
    }

    def _compute() -> Dict[str, np.ndarray]:
        ds = hf_datasets.load_dataset(
            "UniverseTBD/pu-embeddings", name=cfg["dataset"], split="train"
        )
        if ds.num_rows != EXPECTED_N_TOTAL:
            raise ValueError(
                f"Loaded config '{cfg['dataset']}' reports {ds.num_rows} rows; expected "
                f"EXPECTED_N_TOTAL={EXPECTED_N_TOTAL}. The upstream dataset may have "
                f"changed -- refusing to proceed on an unverified row count."
            )
        row_indices = draw_row_indices(ds.num_rows, cfg["n_rows"], cfg["seed"])

        ds_numpy = ds.with_format("numpy")
        selected = ds_numpy[row_indices]
        hsc_raw = np.asarray(selected["dinov3_vitb16_hsc"], dtype=np.float64)
        ls_raw = np.asarray(selected["dinov3_vitb16_legacysurvey"], dtype=np.float64)

        hsc, hsc_norms = l2_normalize(hsc_raw)
        legacysurvey, ls_norms = l2_normalize(ls_raw)

        return {
            "hsc": hsc,
            "legacysurvey": legacysurvey,
            "hsc_norms": hsc_norms,
            "ls_norms": ls_norms,
            "row_indices": row_indices,
        }

    stem = f"subsample_{cfg['seed']}_{cache.config_key(subsample_cfg)}"
    return cache.npz_cache(stem, subsample_cfg, _compute)
