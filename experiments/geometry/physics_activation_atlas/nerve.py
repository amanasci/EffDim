"""Exploratory chart-overlap nerve complex and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def overlap_mass_matrix(
    membership_idx: np.ndarray,
    membership_w: np.ndarray,
    n_charts: int,
) -> np.ndarray:
    """Soft pairwise overlap mass M[c,c'] = sum_i min(w_ic, w_ic')."""
    n, r = membership_idx.shape
    W = np.zeros((n, n_charts), dtype=np.float64)
    for i in range(n):
        for j in range(r):
            c = int(membership_idx[i, j])
            if 0 <= c < n_charts:
                W[i, c] = membership_w[i, j]
    M = np.zeros((n_charts, n_charts), dtype=np.float64)
    for a in range(n_charts):
        for b in range(a + 1, n_charts):
            m = float(np.minimum(W[:, a], W[:, b]).sum())
            M[a, b] = M[b, a] = m
    return M, W


def triple_overlap_mass(W: np.ndarray, a: int, b: int, c: int) -> float:
    return float(np.minimum(np.minimum(W[:, a], W[:, b]), W[:, c]).sum())


def build_nerve(
    M: np.ndarray,
    W: np.ndarray,
    *,
    threshold: float,
    maxdim: int = 2,
) -> dict:
    """Downward-closed clique nerve from pairwise/triple soft overlap masses."""
    C = M.shape[0]
    nodes = list(range(C))
    edges = []
    for a in range(C):
        for b in range(a + 1, C):
            if M[a, b] >= threshold:
                edges.append((a, b))
    triangles = []
    if maxdim >= 2:
        edge_set = {frozenset(e) for e in edges}
        for i, (a, b) in enumerate(edges):
            for c in range(C):
                if c == a or c == b:
                    continue
                trip = sorted((a, b, c))
                if trip[0] == a and trip[1] == b:  # enumerate each triangle once via a<b
                    pass
                faces = [
                    frozenset((a, b)),
                    frozenset((a, c)),
                    frozenset((b, c)),
                ]
                if all(f in edge_set for f in faces):
                    mass3 = triple_overlap_mass(W, a, b, c)
                    if mass3 >= threshold:
                        t = tuple(sorted((a, b, c)))
                        if t not in triangles:
                            triangles.append(t)
        triangles = sorted(set(triangles))
    return {
        "threshold": float(threshold),
        "n_nodes": C,
        "n_edges": len(edges),
        "n_triangles": len(triangles),
        "edges": edges,
        "triangles": triangles,
    }


def nerve_persistence(filtration: list[dict], maxdim: int = 2) -> dict:
    """Build GUDHI simplex tree along decreasing thresholds; return Betti curves."""
    try:
        import gudhi
    except ImportError:
        return {"error": "gudhi_missing", "betti_curves": {}}

    # filtration values: birth when simplex appears as threshold decreases
    # Use -log(mass) style: assign filtration value = -mass so stronger overlaps appear earlier
    st = gudhi.SimplexTree()
    # collect all simplices with their mass
    masses = {}
    for step in filtration:
        thr = step["threshold"]
        for n in range(step["n_nodes"]):
            masses.setdefault((n,), thr)  # nodes always present; use max thr
        for a, b in step["edges"]:
            key = tuple(sorted((a, b)))
            masses[key] = max(masses.get(key, 0.0), thr)
        for t in step.get("triangles", []):
            key = tuple(sorted(t))
            masses[key] = max(masses.get(key, 0.0), thr)

    # insert with filtration = -mass (stronger first), then compute persistence
    # Ensure nodes first
    C = filtration[0]["n_nodes"] if filtration else 0
    for n in range(C):
        st.insert([n], filtration=-1e9)  # always present
    for key, mass in sorted(masses.items(), key=lambda kv: -kv[1]):
        if len(key) == 1:
            continue
        st.insert(list(key), filtration=-float(mass))
    st.make_filtration_non_decreasing()
    st.compute_persistence()
    betti = {}
    dgms = {}
    for dim in range(maxdim + 1):
        iv = np.asarray(st.persistence_intervals_in_dimension(dim), dtype=np.float64)
        dgms[str(dim)] = iv.tolist()
        # count finite bars with persistence > small eps
        if iv.size == 0:
            betti[str(dim)] = 0
        else:
            pers = iv[:, 1] - iv[:, 0]
            finite = np.isfinite(pers)
            betti[str(dim)] = int(np.sum((pers > 1e-6) & finite) + np.sum(~np.isfinite(iv[:, 1])))
    # Betti curves over thresholds
    curves = []
    for step in filtration:
        curves.append(
            {
                "threshold": step["threshold"],
                "n_edges": step["n_edges"],
                "n_triangles": step["n_triangles"],
                "H0_components_proxy": step["n_nodes"] - step["n_edges"],  # tree proxy only
            }
        )
    return {"betti_counts": betti, "diagrams": dgms, "filtration_curve": curves, "note": "exploratory_nerve"}


def shuffled_control_masses(W: np.ndarray, seed: int) -> np.ndarray:
    """Permute memberships within charts preserving approximate column masses."""
    rng = np.random.default_rng(seed)
    Wn = W.copy()
    for c in range(W.shape[1]):
        Wn[:, c] = rng.permutation(W[:, c])
    # renormalize soft? keep as control of mass structure
    M = np.zeros((W.shape[1], W.shape[1]))
    for a in range(W.shape[1]):
        for b in range(a + 1, W.shape[1]):
            m = float(np.minimum(Wn[:, a], Wn[:, b]).sum())
            M[a, b] = M[b, a] = m
    return M


def run_nerve_analysis(
    membership_idx: np.ndarray,
    membership_w: np.ndarray,
    n_charts: int,
    *,
    thresholds: list[float] | None = None,
    maxdim: int = 2,
    seed: int = 0,
) -> dict:
    M, W = overlap_mass_matrix(membership_idx, membership_w, n_charts)
    pos = M[np.triu_indices(n_charts, 1)]
    pos = pos[pos > 0]
    if thresholds is None:
        if len(pos) == 0:
            thresholds = [0.0]
        else:
            thresholds = sorted(
                {
                    float(np.quantile(pos, q))
                    for q in [0.2, 0.4, 0.6, 0.8, 0.9]
                },
                reverse=True,
            )
    filtration = [build_nerve(M, W, threshold=t, maxdim=maxdim) for t in thresholds]
    pers = nerve_persistence(filtration, maxdim=maxdim)
    # shuffled controls
    ctrl = []
    for s in range(3):
        Ms = shuffled_control_masses(W, seed + s)
        f0 = build_nerve(Ms, W, threshold=thresholds[len(thresholds) // 2], maxdim=maxdim)
        ctrl.append({"seed": seed + s, "n_edges": f0["n_edges"], "n_triangles": f0["n_triangles"]})
    return {
        "overlap_definition": "sum_i min(w_ic, w_ic') soft intersection; triangles need triple min mass",
        "pairwise_mass_summary": {
            "mean": float(pos.mean()) if len(pos) else 0.0,
            "p50": float(np.median(pos)) if len(pos) else 0.0,
            "p90": float(np.percentile(pos, 90)) if len(pos) else 0.0,
            "max": float(pos.max()) if len(pos) else 0.0,
        },
        "filtration": filtration,
        "persistence": pers,
        "shuffled_controls": ctrl,
        "label": "exploratory_nerve",
    }


def save_nerve(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "nerve.json").write_text(json.dumps(result, indent=2, default=str))
