"""Seeded, row-alignment-safe subsampling of ``UniverseTBD/pu-embeddings``.

No object_id exists in this dataset — row order is the only join between the paired
columns, so both are read off ONE sorted seeded index array in a single indexing pass;
two independent selections would silently break alignment. ``assert_alignment`` is the
runtime proof (structural check + permuted-null z-score). Columns are L2-normalized at
cache-write time (D-05/D-06); raw norms kept separately so the DATA-04 histogram stays
reproducible without re-streaming.
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
    """Deterministic sorted duplicate-free sample (DATA-03). Both paired columns must
    be read off this single array in one pass. Raises ValueError on degenerate sizes."""
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
    """L2-normalize each row of a ``(n_rows, n_features)`` array, returning
    ``(x / norms[:, None], norms)``. Raises ValueError on a zero-norm row."""
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            "l2_normalize received at least one zero-norm row; cannot normalize a "
            "zero vector to the unit sphere."
        )
    return x / norms[:, None], norms


def row_indices_sha256(row_indices: np.ndarray) -> str:
    """Sha256 hex digest of row_indices, cast to int64 and made C-contiguous first so
    the hash is independent of dtype/layout quirks."""
    contiguous = np.ascontiguousarray(row_indices, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def assert_structural_alignment(
    hsc: np.ndarray,
    legacysurvey: np.ndarray,
    row_indices: np.ndarray,
    expected_sha256: Optional[str] = None,
) -> str:
    """Raise ValueError unless hsc/legacysurvey/row_indices are structurally consistent
    (matching shapes, N_FEATURES width, finiteness, and -- if expected_sha256 is given --
    a matching row_indices hash). Returns the (recomputed) sha256 hash of row_indices."""
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
    """Row-alignment z-score against a permuted null. s_true = mean per-row cosine of
    the true pairing; z = (s_true - mu_perm) / sd_perm. Scale-free by construction: the
    origin paper reports crossmodal MKNN at only 0.4-2%, so an absolute-cosine margin
    would reject a correct-but-weak pairing, while a gross misalignment makes s_true a
    draw from the null (z near 0). Raises on sd_perm == 0."""
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
    """Structural check + statistical smoke test; raises unless both pass (strictly
    z > ALIGNMENT_MARGIN_Z). Never weaken or skip (DATA-03): with no object_id, this is
    the only correctness invariant the milestone rests on."""
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
    """Load (or compute-and-cache) the seeded, L2-normalized subsample. Returns
    {"hsc", "legacysurvey", "hsc_norms", "ls_norms", "row_indices"}. ``datasets`` is
    imported lazily so the module works with numpy+joblib only.

    The cache key is deliberately NARROWER than cfg (dataset/seed/n_rows/normalize +
    library versions, D-14 refinement): a fit-only parameter like n_neighbors must not
    invalidate this artifact and force a re-download. The Isomap fit cache uses the full
    field set."""
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
