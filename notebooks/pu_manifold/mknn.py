"""
Phase 4 helpers for regional crossmodal MKNN alignment, stubbed here per D-02 so the
eventual four-module ``pu_manifold`` package shape is visible from the start.

Metric definition (Chechik et al. 2010, adopted verbatim by arXiv:2509.19453, the
origin paper this milestone extends):

    MKNN(z1, z2) = k^-1 * |N_k(z1) intersect N_k(z2)|

where ``N_k(z)`` is the set of indices of the k nearest neighbours of point ``z`` within
its own embedding space. It is label-free and training-free: it compares the k-NN sets
of two embeddings of the *same* object, which is exactly what this milestone's
row-aligned (but label-free, join-key-free) HSC/Legacy Survey pairing supports.

k-NN-based alignment metrics are known to be hubness-sensitive in high dimensions (a
small number of points recur disproportionately often in many other points' k-NN sets,
inflating or distorting the apparent alignment). MKNN-08 requires that this caveat be
stated alongside any MKNN result this module produces -- it is a known open problem in
the alignment literature, not solved here.

Neither this module nor its functions import faiss at module level -- Phase 4's faiss
usage must not become an import-time requirement for notebook 01 or for this package's
Phase 1 test suite, both of which run with only numpy and joblib present.
"""

from typing import Any


def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    """
    Compute the mutual k-nearest-neighbour alignment score between two embeddings.

    Parameters:
    -----------
    z1 : Any
        The first embedding array, e.g. the HSC-modality points for one region.
    z2 : Any
        The second embedding array, e.g. the Legacy-Survey-modality points for the
        same region.
    k : Any
        Number of nearest neighbours to compare per point.

    Returns:
    --------
    Any
        ``k^-1 * |N_k(z1) intersect N_k(z2)|`` averaged over all points (Chechik et
        al. 2010).

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 4 (MKNN-01). Both inputs must come from the
        same ``row_indices`` ordering and the same normalization convention, since
        there is no ``object_id`` in this dataset to catch a mismatch -- the caller
        is responsible for that invariant, this function trusts its inputs are
        already row-aligned.
    """
    raise NotImplementedError("Implemented in Phase 4 (MKNN-01)")


def permutation_null(z1: Any, z2: Any, k: Any, n_permutations: Any, seed: Any) -> Any:
    """
    Compute a permutation-null distribution for the MKNN score, within one region.

    Parameters:
    -----------
    z1 : Any
        The first embedding array for one region.
    z2 : Any
        The second embedding array for the same region.
    k : Any
        Number of nearest neighbours to compare per point.
    n_permutations : Any
        Number of independent seeded permutations to draw.
    seed : Any
        Seed for the permutation draws.

    Returns:
    --------
    Any
        The distribution of MKNN scores under random pairing within this region.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 4 (MKNN-04). The permutation is drawn
        **within** a region's own index set, never reused from a global null: a
        global null would not control for the region's own local density and
        would misrepresent the significance of a regional MKNN score.
    """
    raise NotImplementedError("Implemented in Phase 4 (MKNN-04)")


def bootstrap_ci(z1: Any, z2: Any, k: Any, n_resamples: Any, seed: Any) -> Any:
    """
    Compute a bootstrap confidence interval for the MKNN score, within one region.

    Parameters:
    -----------
    z1 : Any
        The first embedding array for one region.
    z2 : Any
        The second embedding array for the same region.
    k : Any
        Number of nearest neighbours to compare per point.
    n_resamples : Any
        Number of bootstrap resamples to draw.
    seed : Any
        Seed for the resampling draws.

    Returns:
    --------
    Any
        A confidence interval (e.g. a ``(low, high)`` pair) on the regional MKNN
        score, from resampling with replacement within the region.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 4 (MKNN-05). A reader gets bootstrap
        confidence intervals on every regional MKNN score.
    """
    raise NotImplementedError("Implemented in Phase 4 (MKNN-05)")
