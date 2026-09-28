"""Thin the counterfactual's anchors to pairwise-disjoint neighbourhoods, for a sign test with independent units.

The 512 anchors of ``09_physics_normal_scaling_run.py`` have overlapping 2,048-neighbourhoods on 86,471 points, so a
sign test over all of them is anticonservative. This script recomputes the same anchors and k-NN panel (deterministic:
same seeds, same X), computes the full pairwise neighbourhood-overlap matrix, and keeps anchors greedily in anchor order whenever their
overlap with every kept anchor is at most a threshold fraction of k (thresholds 0, 0.01, 0.02, 0.05, 0.10; exact
disjointness from the union of kept sets keeps only 6-7 anchors, since a few kept sets already cover ~14% of the data). The sign tests themselves are computed offline from the mask and the per-anchor arrays.

NOT PRE-REGISTERED, GATES NOTHING.

Usage:
    python curvature-experiment/runners/09_physics_normal_scaling_thin_run.py --parquet-path <...> --embedding-column vit_base_galaxies \\
        --arrays-npz <09_physics_normal_scaling_<enc>_d<d>.npz> --out <..._thin.npz> --threads 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAGNOSTICS_ROOT.parent))
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402


def load_embeddings(path: str, col: str) -> np.ndarray:
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=[col])
    raw = np.stack([np.asarray(v, dtype=np.float64) for v in tbl.column(col).to_pylist()])
    return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)


THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10)


def overlap_matrix(neigh: np.ndarray) -> np.ndarray:
    sets = [set(r.tolist()) for r in neigh]; b = len(sets)
    ov = np.zeros((b, b))
    for i in range(b):
        for j in range(i + 1, b):
            ov[i, j] = ov[j, i] = len(sets[i] & sets[j]) / neigh.shape[1]
    return ov


def greedy_independent(ov: np.ndarray, thr: float) -> np.ndarray:
    keep = np.zeros(ov.shape[0], dtype=bool)
    for i in range(ov.shape[0]):
        if not keep.any() or ov[i, keep].max() <= thr:
            keep[i] = True
    return keep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parquet-path", required=True); p.add_argument("--embedding-column", required=True)
    p.add_argument("--arrays-npz", required=True, help="per-anchor arrays of the counterfactual run (anchor order check)")
    p.add_argument("--out", required=True); p.add_argument("--threads", type=int, default=8)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    X = load_embeddings(args.parquet_path, args.embedding_column)
    n = X.shape[0]
    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, pcp.N_ANCHORS, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    stored = np.load(args.arrays_npz)["anchor_idx"]
    assert np.array_equal(stored, a), "anchor draw differs from the stored arrays"
    panel = pcp.knn_panel(X, a, pcp.K_NEIGHBOURS)
    ov = overlap_matrix(panel["indices"])
    keeps = {f"keep_{thr:g}": greedy_independent(ov, thr) for thr in THRESHOLDS}
    iu = np.triu_indices(ov.shape[0], 1)
    np.savez_compressed(args.out, anchor_idx=a, overlap=ov.astype(np.float32), k=pcp.K_NEIGHBOURS, thresholds=np.array(THRESHOLDS), **keeps)
    print(f"anchors {len(a)}; pairwise overlap median {np.median(ov[iu]):.4f} p90 {np.percentile(ov[iu], 90):.3f} max {ov[iu].max():.3f}; kept at thresholds "
          + ", ".join(f"{thr:g}: {int(m.sum())}" for thr, m in zip(THRESHOLDS, keeps.values())) + f"; -> {args.out}")


if __name__ == "__main__":
    main()
