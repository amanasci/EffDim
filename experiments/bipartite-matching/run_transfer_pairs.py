#!/usr/bin/env python3
"""Shared-basis transfer matrix: all model pairs with trained SAEs.

Per pair:
  - baselines: dense cosine / SAE cosine / SAE IDF mKNN
  - Ridge affine shared basis, both directions, IDF mKNN (headline)
  - W structure: stable rank, top-64-per-row and top-1-per-row mKNN

Usage:
  python experiments/bipartite-matching/run_transfer_pairs.py \\
      --platonic-root ~/platonic-universe --max-n 16384
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _shared import (  # noqa: E402
    MODELS,
    RidgeMap,
    idf_weights,
    knn_graph,
    l2n,
    load_col,
    load_model_codes,
    mknn,
    platonic_root,
    resolve_path,
    sae_dir,
    topk_rows,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", default=",".join(sorted(MODELS)))
    p.add_argument("--output-dir", default="outputs/bipartite_matching/transfer_pairs")
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    K = args.k
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    available = [
        m for m in wanted if (sae_dir(root, m) / "model.pt").is_file()
    ]
    skipped = sorted(set(wanted) - set(available))
    if skipped:
        print(f"SKIP (no SAE): {skipped}", flush=True)
    print(f"models: {available}", flush=True)

    parquet0, col0 = MODELS[available[0]]
    n_full = len(load_col(root / parquet0, col0))
    n = min(args.max_n, n_full) if args.max_n else n_full
    sel = np.sort(rng.choice(n_full, size=n, replace=False))

    data = {}
    for m in available:
        X, C = load_model_codes(root, m, sel, device)
        data[m] = {"X": X, "C": C}
        print(f"loaded {m} ({time.time()-t0:.0f}s)", flush=True)

    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    tr, te = np.sort(tr), np.sort(te)

    for m, d in data.items():
        d["idf"] = idf_weights(d["C"][tr])
        d["g_dense"] = knn_graph(l2n(d["X"][te]), K)
        d["g_sae"] = knn_graph(d["C"][te], K)
        d["g_idf"] = knn_graph(d["C"][te] * d["idf"][None], K)

    def map_and_eval(src: str, dst: str) -> dict:
        rm = RidgeMap(data[src]["C"], data[dst]["C"], tr, alpha=args.ridge_alpha)
        g_true = data[dst]["g_idf"]
        w = data[dst]["idf"]
        return {
            "mknn_full": rm.eval_mknn(te, g_true, w, K),
            "mknn_top64": rm.eval_mknn(te, g_true, w, K, topk_rows(rm.W, 64)),
            "mknn_top1": rm.eval_mknn(te, g_true, w, K, topk_rows(rm.W, 1)),
            "stable_rank": rm.stable_rank(),
        }

    results = {}
    for a, b in itertools.combinations(available, 2):
        key = f"{a}__{b}"
        res = {
            "dense_cosine": mknn(data[a]["g_dense"], data[b]["g_dense"]),
            "sae_cosine": mknn(data[a]["g_sae"], data[b]["g_sae"]),
            "sae_idf": mknn(data[a]["g_idf"], data[b]["g_idf"]),
            f"{b}_to_{a}": map_and_eval(b, a),
            f"{a}_to_{b}": map_and_eval(a, b),
        }
        results[key] = res
        best = max(res[f"{b}_to_{a}"]["mknn_full"], res[f"{a}_to_{b}"]["mknn_full"])
        print(
            f"{key:<28} dense={res['dense_cosine']:.3f} idf={res['sae_idf']:.3f} "
            f"shared_best={best:.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    out_dir = resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_pairs.json").write_text(
        json.dumps(
            {"results": results, "n": n, "k": K, "args": vars(args),
             "elapsed_s": time.time() - t0},
            indent=2,
        )
    )

    lines = [
        "# Shared-basis transfer matrix",
        "",
        f"- n={n}, k={K}, test_size={args.test_size}, alpha={args.ridge_alpha}",
        "",
        "| pair | dense | SAE IDF | shared best | top64 | top1 | stable rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in results.items():
        dirs = [v for v in r.values() if isinstance(v, dict)]
        bd = max(dirs, key=lambda d: d["mknn_full"])
        lines.append(
            f"| {key} | {r['dense_cosine']:.3f} | {r['sae_idf']:.3f} | "
            f"{bd['mknn_full']:.3f} | {bd['mknn_top64']:.3f} | "
            f"{bd['mknn_top1']:.3f} | {bd['stable_rank']:.0f} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "transfer_pairs_report.md").write_text(report)
    print(report, flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
