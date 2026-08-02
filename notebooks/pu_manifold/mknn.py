"""Phase 4 regional crossmodal MKNN stubs.

Metric (Chechik et al. 2010, adopted verbatim by arXiv:2509.19453):

    MKNN(z1, z2) = k^-1 * |N_k(z1) intersect N_k(z2)|

with N_k(z) the k-NN index set within z's own embedding space. Label-free and
training-free — works on this dataset's row-aligned, join-key-free HSC/Legacy pairing.

Known caveat (MKNN-08): k-NN alignment metrics are hubness-sensitive in high
dimensions; state this alongside any MKNN result. No module-level faiss import.
"""

from typing import Any


def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    """Mean MKNN score over all points. Caller guarantees row alignment — there is
    no object_id in this dataset to catch a mismatch."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-01)")


def permutation_null(z1: Any, z2: Any, k: Any, n_permutations: Any, seed: Any) -> Any:
    """Permutation-null MKNN distribution, drawn *within* the region's own index set —
    a global null would not control for the region's local density."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-04)")


def bootstrap_ci(z1: Any, z2: Any, k: Any, n_resamples: Any, seed: Any) -> Any:
    """Bootstrap (low, high) CI on the regional MKNN score, resampling within region."""
    raise NotImplementedError("Implemented in Phase 4 (MKNN-05)")
