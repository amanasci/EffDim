#!/usr/bin/env python3
"""Structure of the shared-basis map W for one model pair.

Combines the rank sweep and the sparsification sweep:
  - truncated-SVD rank r vs held-out mKNN (is W low-rank?)
  - top-k-per-row / Hungarian 1-to-1 / random 1-to-1 vs mKNN (is W sparse?)

Usage:
  python experiments/bipartite-matching/run_sparsity_rank.py \\
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
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    MODELS,
    RidgeMap,
    idf_weights,
    knn_graph,
    load_col,
    load_model_codes,
    platonic_root,
    rank_truncate,
    resolve_path,
    topk_rows,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="dinov3", choices=sorted(MODELS))
    p.add_argument("--dst", default="vit_base", choices=sorted(MODELS))
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--ranks", default="4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--row-ks", default="1,2,4,8,16,64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs/bipartite_matching/sparsity_rank")
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    K = args.k
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    parquet0, col0 = MODELS[args.src]
    n_full = len(load_col(root / parquet0, col0))
    n = min(args.max_n, n_full) if args.max_n else n_full
    sel = np.sort(rng.choice(n_full, size=n, replace=False))

    _, C_src = load_model_codes(root, args.src, sel, device)
    _, C_dst = load_model_codes(root, args.dst, sel, device)
    print(f"encoded ({time.time()-t0:.0f}s)", flush=True)

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    tr, te = np.sort(tr), np.sort(te)

    rm = RidgeMap(C_src, C_dst, tr, alpha=args.ridge_alpha)
    W = rm.W
    w_idf = idf_weights(C_dst[tr])
    g_true = knn_graph(C_dst[te] * w_idf[None], K)

    def ev(Wv: np.ndarray) -> float:
        return rm.eval_mknn(te, g_true, w_idf, K, Wv)

    results: dict = {
        "pair": f"{args.src}->{args.dst}",
        "stable_rank": rm.stable_rank(),
        "mknn_full": ev(W),
        "rank": {},
        "topk": {},
    }
    print(
        f"stable_rank={results['stable_rank']:.1f} full={results['mknn_full']:.4f}",
        flush=True,
    )

    S = np.linalg.svd(W, compute_uv=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    for r in [int(x) for x in args.ranks.split(",")]:
        r = min(r, W.shape[0])
        m = ev(rank_truncate(W, r))
        results["rank"][str(r)] = {"mknn": m, "energy": float(energy[r - 1])}
        print(f"  rank={r:>5} mknn={m:.4f} energy={energy[r-1]:.3f}", flush=True)

    for k_row in [int(x) for x in args.row_ks.split(",")]:
        Wk = topk_rows(W, k_row)
        m = ev(Wk)
        results["topk"][str(k_row)] = {
            "mknn": m,
            "nnz_frac": float((Wk != 0).mean()),
            "mass_frac": float(np.abs(Wk).sum() / np.abs(W).sum()),
        }
        print(f"  topk={k_row:>3} mknn={m:.4f}", flush=True)

    # one-to-one assignments
    ri, ci = linear_sum_assignment(-np.abs(W))
    W_h = np.zeros_like(W)
    W_h[ri, ci] = W[ri, ci]
    results["hungarian_1to1"] = {
        "mknn": ev(W_h),
        "agrees_row_argmax": float((np.abs(W).argmax(axis=1)[ri] == ci).mean()),
    }
    perm = rng.permutation(W.shape[1])
    W_r = np.zeros_like(W)
    W_r[np.arange(W.shape[0]), perm] = W[np.arange(W.shape[0]), perm]
    results["random_1to1"] = {"mknn": ev(W_r)}
    print(
        f"  hungarian={results['hungarian_1to1']['mknn']:.4f} "
        f"random={results['random_1to1']['mknn']:.4f}",
        flush=True,
    )

    results["elapsed_s"] = time.time() - t0
    out_dir = resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.src}__{args.dst}"
    (out_dir / f"sparsity_rank_{tag}.json").write_text(
        json.dumps({"args": vars(args), **results}, indent=2)
    )
    print(f"Wrote {out_dir}/sparsity_rank_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
