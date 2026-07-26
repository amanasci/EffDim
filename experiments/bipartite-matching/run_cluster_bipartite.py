#!/usr/bin/env python3
"""Cluster the bipartite feature graph of the shared-basis map W.

Spectral co-clustering on the top-k-per-row |W| graph. Per cluster:
  - sizes and |W| mass concentration
  - cross-model agreement: corr(src-side activation, dst-side activation)
  - physics alignment: corr of cluster activation with each property

Usage:
  python experiments/bipartite-matching/run_cluster_bipartite.py \\
      --src dinov3 --dst vit_base --platonic-root ~/platonic-universe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import SpectralCoclustering
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    DEFAULT_PROPERTIES,
    LABELS_NPZ,
    MODELS,
    RidgeMap,
    load_col,
    load_model_codes,
    platonic_root,
    resolve_path,
    topk_rows,
)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="dinov3", choices=sorted(MODELS))
    p.add_argument("--dst", default="vit_base", choices=sorted(MODELS))
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--topk-row", type=int, default=64)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--properties", default=",".join(DEFAULT_PROPERTIES))
    p.add_argument("--labels", default=LABELS_NPZ)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir", default="outputs/bipartite_matching/cluster_bipartite"
    )
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    props = [x.strip() for x in args.properties.split(",") if x.strip()]
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    parquet0, col0 = MODELS[args.src]
    n_full = len(load_col(root / parquet0, col0))
    n = min(args.max_n, n_full) if args.max_n else n_full
    sel = np.sort(rng.choice(n_full, size=n, replace=False))

    _, C_src = load_model_codes(root, args.src, sel, device)
    _, C_dst = load_model_codes(root, args.dst, sel, device)
    labels = dict(np.load(root / args.labels))
    Y = np.stack([labels[p_][sel].astype(np.float64) for p_ in props], 1)
    print(f"encoded ({time.time()-t0:.0f}s)", flush=True)

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    tr, te = np.sort(tr), np.sort(te)

    rm = RidgeMap(C_src, C_dst, tr, alpha=args.ridge_alpha)
    Ak = np.abs(topk_rows(rm.W, args.topk_row))

    live_r = np.where(Ak.sum(1) > 1e-8)[0]
    live_c = np.where(Ak.sum(0) > 1e-8)[0]
    Ak_live = Ak[np.ix_(live_r, live_c)]
    print(f"live features: src={len(live_r)} dst={len(live_c)}", flush=True)

    cocluster = SpectralCoclustering(
        n_clusters=args.n_clusters, random_state=args.seed, n_init=10
    )
    cocluster.fit(Ak_live + 1e-12)
    row_lab = cocluster.row_labels_
    col_lab = cocluster.column_labels_

    C_src_te, C_dst_te, Y_te = C_src[te], C_dst[te], Y[te]
    finite = np.all(np.isfinite(Y_te), axis=1)

    clusters = []
    total_mass = Ak_live.sum()
    for c in range(args.n_clusters):
        r_idx = live_r[row_lab == c]
        c_idx = live_c[col_lab == c]
        if len(r_idx) == 0 or len(c_idx) == 0:
            continue
        block_mass = float(Ak[np.ix_(r_idx, c_idx)].sum())
        row_total = float(Ak[r_idx].sum())
        act_src = C_src_te[:, r_idx].sum(1)
        act_dst = C_dst_te[:, c_idx].sum(1)
        phys = {}
        for j, prop in enumerate(props):
            m = finite & np.isfinite(Y_te[:, j])
            phys[prop] = safe_corr(act_dst[m], Y_te[m, j])
        best_prop = max(
            phys, key=lambda q: abs(phys[q]) if np.isfinite(phys[q]) else 0
        )
        clusters.append(
            {
                "cluster": int(c),
                "n_src": int(len(r_idx)),
                "n_dst": int(len(c_idx)),
                "block_mass_frac_of_rows": block_mass / max(row_total, 1e-12),
                "mass_frac_of_total": block_mass / total_mass,
                "cross_model_corr": safe_corr(act_src, act_dst),
                "phys_corr": phys,
                "best_prop": best_prop,
                "best_prop_corr": phys[best_prop],
                "src_features": r_idx.tolist(),
                "dst_features": c_idx.tolist(),
            }
        )

    diag_mass = sum(c["mass_frac_of_total"] for c in clusters)
    clusters.sort(
        key=lambda c: -abs(
            c["best_prop_corr"] if np.isfinite(c["best_prop_corr"]) else 0
        )
    )

    print(f"\nblock-diagonal mass fraction: {diag_mass:.3f}", flush=True)
    for c in clusters[:15]:
        print(
            f"  clu {c['cluster']:>3} n=({c['n_src']},{c['n_dst']}) "
            f"blk={100*c['block_mass_frac_of_rows']:.0f}% "
            f"xcorr={c['cross_model_corr']:.2f} "
            f"{c['best_prop']}={c['best_prop_corr']:.2f}",
            flush=True,
        )

    out_dir = resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.src}__{args.dst}"
    (out_dir / f"cluster_bipartite_{tag}.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "block_diag_mass": diag_mass,
                "clusters": clusters,
                "elapsed_s": time.time() - t0,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {out_dir}/cluster_bipartite_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
