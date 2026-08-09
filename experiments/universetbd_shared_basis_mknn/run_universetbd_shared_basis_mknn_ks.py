#!/usr/bin/env python3
"""UniverseTBD SAE shared-basis mKNN with k-scaling.

Applies Ridge affine SAE shared-basis + IDF across UniverseTBD embedding pairs,
evaluating mutual kNN at:

    k ∈ {10, 20, 50, 100}

Dense baseline matches Platonic Universe (arXiv:2509.19453) Table 2 protocol:
ambient cosine MKNN on the **full catalog** (self-excluded), no learned map.
See ``paper_protocol_fulln_dense_mknn.py``.

Learned methods (SAE / shared maps / IDF) are fit on train only, then scored as
**held-out queries with full-catalog galleries** (average MKNN over test rows;
neighbours searched in all N). A matching ``dense_cosine_heldout`` uses the
same query set so lift is apples-to-apples; ``dense_cosine`` remains the
paper-comparable absolute baseline.

No new representation learning: reuses existing TopK SAE checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import load_aligned_pair, platonic_root, resolve_path  # noqa: E402
from sae_affine_basis_mknn_gpu import (  # noqa: E402
    encode,
    fit_affine_express_in_basis,
    idf_np,
    knn_cos,
    load_sae,
    mknn,
)

DEFAULT_KS = (10, 20, 50, 100)
SAE_TAG_PREFER = (
    "F2048_k20_seed0",
    "F2048_k22_seed0",
    "F2048_k19_seed0",
    "F2048_k64_seed0",
    "F2048_k32_seed0",
    "F2048_k128_seed0",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument(
        "--pairs-yaml",
        default="experiments/universetbd_shared_basis_mknn/compatible_pairs.yaml",
    )
    p.add_argument(
        "--pairs",
        default="",
        help="Comma pair names (default: all ready pairs, or smoke subset)",
    )
    p.add_argument(
        "--surveys",
        default="",
        help="Comma survey filters: physics,jwst,desi,legacy,cosmosweb",
    )
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--ks", default="10,20,50,100")
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--allow-truncate", action="store_true")
    p.add_argument(
        "--out-dir",
        default="outputs/universetbd_shared_basis_mknn_ks",
    )
    p.add_argument("--run-tag", default="")
    return p.parse_args()


def load_pairs(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def resolve_sae_dir(root: Path, parquet_rel: str, col: str) -> Path | None:
    stem = Path(parquet_rel).stem
    base = root / "outputs/sae" / stem / col
    if not base.is_dir():
        return None
    tags = {
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / "model.pt").is_file()
    }
    for tag in SAE_TAG_PREFER:
        if tag in tags:
            return base / tag
    # fallback: any checkpoint
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "model.pt").is_file():
            return p
    return None


def build_manifest(
    root: Path, pairs: dict, selected: list[str]
) -> pd.DataFrame:
    rows = []
    for name in selected:
        cfg = pairs[name]
        sae1 = resolve_sae_dir(root, cfg["parquet1"], cfg["col1"])
        sae2 = resolve_sae_dir(root, cfg["parquet2"], cfg["col2"])
        pq1 = resolve_path(root, cfg["parquet1"])
        pq2 = resolve_path(root, cfg["parquet2"])
        ok = pq1.is_file() and pq2.is_file() and sae1 is not None and sae2 is not None
        reason = ""
        if not pq1.is_file() or not pq2.is_file():
            reason = "missing_parquet"
        elif sae1 is None or sae2 is None:
            reason = "missing_sae"
        rows.append(
            {
                "pair": name,
                "kind": cfg.get("kind", ""),
                "survey": cfg.get("survey", ""),
                "parquet1": cfg["parquet1"],
                "col1": cfg["col1"],
                "parquet2": cfg["parquet2"],
                "col2": cfg["col2"],
                "sae1": str(sae1) if sae1 else "",
                "sae2": str(sae2) if sae2 else "",
                "sae1_tag": sae1.name if sae1 else "",
                "sae2_tag": sae2.name if sae2 else "",
                "default_max_n": int(cfg.get("default_max_n", 0) or 0),
                "included": bool(ok),
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def score_mknn_full(
    A: torch.Tensor, B: torch.Tensor, k: int, row_batch: int
) -> float:
    """Paper-protocol MKNN: neighbour search over the full catalog."""
    return mknn(knn_cos(A, k, row_batch), knn_cos(B, k, row_batch), k)


@torch.inference_mode()
def score_mknn_queries_in_gallery(
    A: torch.Tensor,
    B: torch.Tensor,
    query_idx: torch.Tensor,
    k: int,
    row_batch: int,
) -> float:
    """Diagnostic: average MKNN only on query rows, but kNN over full gallery."""
    # Build full knn once, then restrict the overlap average to query rows.
    nn_a = knn_cos(A, k, row_batch)
    nn_b = knn_cos(B, k, row_batch)
    return mknn(nn_a[query_idx], nn_b[query_idx], k)


# Paper Table 2 (arXiv:2509.19453) same-architecture cross-survey fractions.
PAPER_TABLE2_DENSE_K10 = {
    "jwst_cross_vit": 0.0675,
    "jwst_cross_convnext": 0.0719,
    "jwst_cross_astropt95": 0.0670,  # AstroPTv2 Base ≈ 95M
    "legacy_cross_vit": 0.0089,
    "legacy_cross_dino": 0.0155,
    "legacy_cross_convnext": 0.0173,
    "legacy_cross_vitlarge": 0.0225,
}


def eval_pair(
    root: Path,
    name: str,
    cfg: dict,
    sae1_path: Path,
    sae2_path: Path,
    *,
    ks: list[int],
    max_n: int,
    test_size: float,
    alpha: float,
    seed: int,
    device: torch.device,
    row_batch: int,
    allow_truncate: bool,
) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    pq1 = resolve_path(root, cfg["parquet1"])
    pq2 = resolve_path(root, cfg["parquet2"])
    X1, X2 = load_aligned_pair(
        pq1, cfg["col1"], pq2, cfg["col2"], allow_truncate=allow_truncate
    )
    n_full = len(X1)
    n_cap = int(cfg.get("default_max_n", 0) or 0)
    n_use = max_n
    if n_cap > 0:
        n_use = min(n_use, n_cap) if n_use > 0 else n_cap
    rng = np.random.default_rng(seed)
    if n_use and n_full > n_use:
        sel = np.sort(rng.choice(n_full, size=n_use, replace=False))
        X1, X2 = X1[sel], X2[sel]
    n = len(X1)
    print(f"[{name}] n={n} (full={n_full})", flush=True)

    b1 = load_sae(sae1_path, device)
    b2 = load_sae(sae2_path, device)
    C1 = encode(b1, X1, device)
    C2 = encode(b2, X2, device)

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    # side1 basis ← map side2; side2 basis ← map side1
    map_2_in_1 = fit_affine_express_in_basis(
        C1, C2, train_idx=train_idx, test_idx=test_idx, alpha=alpha
    )
    map_1_in_2 = fit_affine_express_in_basis(
        C2, C1, train_idx=train_idx, test_idx=test_idx, alpha=alpha
    )

    fit_rows = []
    for direction, block in (
        ("2_in_1", map_2_in_1),
        ("1_in_2", map_1_in_2),
    ):
        te = block["test"]
        fit_rows.append(
            {
                "pair": name,
                "survey": cfg.get("survey", ""),
                "kind": cfg.get("kind", ""),
                "direction": direction,
                "mse": te["mse"],
                "r2": te["r2"],
                "cosine": te["cosine"],
                "jaccard_at_k": te["binary"]["jaccard_at_k"],
                "effective_rank_95": block["effective_rank_95"],
                "coef_nuclear": block["coef_nuclear"],
                "n": n,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "sae1_tag": sae1_path.name,
                "sae2_tag": sae2_path.name,
            }
        )

    Z1 = torch.as_tensor(X1, device=device)
    Z2 = torch.as_tensor(X2, device=device)
    Z1n = Z1 / Z1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Z2n = Z2 / Z2.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C1_t = torch.as_tensor(C1, device=device)
    C2_t = torch.as_tensor(C2, device=device)
    M2in1 = torch.as_tensor(map_2_in_1["y_hat_all"], device=device)
    M1in2 = torch.as_tensor(map_1_in_2["y_hat_all"], device=device)
    te = torch.as_tensor(test_idx, device=device, dtype=torch.long)

    idf1 = torch.as_tensor(idf_np(C1[train_idx]), device=device)
    idf2 = torch.as_tensor(idf_np(C2[train_idx]), device=device)

    # Paper absolute baseline: full-catalog dense (no learned map).
    paper_methods = {"dense_cosine": (Z1n, Z2n)}
    # Fair held-out comparison: test queries, neighbours in full catalog.
    heldout_methods = {
        "dense_cosine_heldout": (Z1n, Z2n),
        "sae_codes_cosine": (C1_t, C2_t),
        "sae_idf_cosine": (C1_t * idf1[None, :], C2_t * idf2[None, :]),
        "shared_side1_basis_cosine": (C1_t, M2in1),
        "shared_side1_basis_idf": (C1_t * idf1[None, :], M2in1 * idf1[None, :]),
        "shared_side2_basis_cosine": (C2_t, M1in2),
        "shared_side2_basis_idf": (C2_t * idf2[None, :], M1in2 * idf2[None, :]),
    }

    mknn_rows = []
    for k in ks:
        if k >= n or k >= len(test_idx):
            print(f"[{name}] skip k={k} (n={n}, test={len(test_idx)})", flush=True)
            continue

        for method, (A, B) in paper_methods.items():
            t1 = time.time()
            score = score_mknn_full(A, B, k, row_batch)
            row = {
                "pair": name,
                "survey": cfg.get("survey", ""),
                "kind": cfg.get("kind", ""),
                "col1": cfg["col1"],
                "col2": cfg["col2"],
                "k": int(k),
                "method": method,
                "mknn": float(score),
                "n": n,
                "n_test": int(len(test_idx)),
                "protocol": "paper_full_catalog",
                "sae1_tag": sae1_path.name,
                "sae2_tag": sae2_path.name,
                "sec": time.time() - t1,
            }
            if k == 10 and name in PAPER_TABLE2_DENSE_K10:
                row["paper_table2_k10"] = PAPER_TABLE2_DENSE_K10[name]
                row["dense_minus_paper"] = float(score) - PAPER_TABLE2_DENSE_K10[name]
            mknn_rows.append(row)
            print(f"[{name}] k={k:<3} {method:<28} {score:.4f}  [paper]", flush=True)

        for method, (A, B) in heldout_methods.items():
            t1 = time.time()
            score = score_mknn_queries_in_gallery(A, B, te, k, row_batch)
            mknn_rows.append(
                {
                    "pair": name,
                    "survey": cfg.get("survey", ""),
                    "kind": cfg.get("kind", ""),
                    "col1": cfg["col1"],
                    "col2": cfg["col2"],
                    "k": int(k),
                    "method": method,
                    "mknn": float(score),
                    "n": n,
                    "n_test": int(len(test_idx)),
                    "protocol": "heldout_query_full_gallery",
                    "sae1_tag": sae1_path.name,
                    "sae2_tag": sae2_path.name,
                    "sec": time.time() - t1,
                }
            )
            print(f"[{name}] k={k:<3} {method:<28} {score:.4f}", flush=True)

        # Diagnostic: old inflated protocol (kNN among test only)
        t1 = time.time()
        dense_test_only = mknn(
            knn_cos(Z1n[te], k, row_batch),
            knn_cos(Z2n[te], k, row_batch),
            k,
        )
        mknn_rows.append(
            {
                "pair": name,
                "survey": cfg.get("survey", ""),
                "kind": cfg.get("kind", ""),
                "col1": cfg["col1"],
                "col2": cfg["col2"],
                "k": int(k),
                "method": "dense_cosine_test_subset",
                "mknn": float(dense_test_only),
                "n": n,
                "n_test": int(len(test_idx)),
                "protocol": "test_subset_mknn",
                "sae1_tag": sae1_path.name,
                "sae2_tag": sae2_path.name,
                "sec": time.time() - t1,
            }
        )

        s1 = next(
            r["mknn"]
            for r in mknn_rows
            if r["k"] == k and r["method"] == "shared_side1_basis_idf"
        )
        s2 = next(
            r["mknn"]
            for r in mknn_rows
            if r["k"] == k and r["method"] == "shared_side2_basis_idf"
        )
        mknn_rows.append(
            {
                "pair": name,
                "survey": cfg.get("survey", ""),
                "kind": cfg.get("kind", ""),
                "col1": cfg["col1"],
                "col2": cfg["col2"],
                "k": int(k),
                "method": "shared_best_basis_idf",
                "mknn": float(max(s1, s2)),
                "n": n,
                "n_test": int(len(test_idx)),
                "protocol": "heldout_query_full_gallery",
                "sae1_tag": sae1_path.name,
                "sae2_tag": sae2_path.name,
                "sec": 0.0,
            }
        )

    print(f"[{name}] done in {time.time() - t0:.1f}s", flush=True)
    return mknn_rows, fit_rows


def make_figures(df: pd.DataFrame, figdir: Path) -> None:
    if df.empty:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    focus_methods = [
        "dense_cosine",  # paper full-catalog baseline
        "dense_cosine_heldout",
        "sae_idf_cosine",
        "shared_best_basis_idf",
    ]
    # Aggregate mean across pairs vs k
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in focus_methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        g = sub.groupby("k")["mknn"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", label=method, capsize=3)
    ax.set_xlabel("mKNN k")
    ax.set_ylabel("mean mKNN overlap")
    ax.set_title("UniverseTBD SAE shared-basis: mKNN vs k")
    ax.legend(fontsize=8)
    ax.set_xticks(sorted(df["k"].unique()))
    fig.tight_layout()
    fig.savefig(figdir / "mknn_vs_k_aggregate.png", dpi=140)
    plt.close(fig)

    # Per survey
    for survey, sdf in df.groupby("survey"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method in focus_methods:
            sub = sdf[sdf["method"] == method]
            if sub.empty:
                continue
            g = sub.groupby("k")["mknn"].mean()
            ax.plot(g.index, g.values, "o-", label=method)
        ax.set_xlabel("mKNN k")
        ax.set_ylabel("mean mKNN")
        ax.set_title(f"mKNN vs k ({survey})")
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(sdf["k"].unique()))
        fig.tight_layout()
        fig.savefig(figdir / f"mknn_vs_k_{survey}.png", dpi=140)
        plt.close(fig)

    # Per-pair curves for shared_best
    sub = df[df["method"] == "shared_best_basis_idf"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for pair, g in sub.groupby("pair"):
            g = g.sort_values("k")
            ax.plot(g["k"], g["mknn"], "o-", alpha=0.75, label=pair)
        ax.set_xlabel("mKNN k")
        ax.set_ylabel("shared_best_basis_idf mKNN")
        ax.set_title("Per-pair shared-basis mKNN vs k")
        ax.legend(fontsize=6, ncol=2)
        ax.set_xticks(sorted(sub["k"].unique()))
        fig.tight_layout()
        fig.savefig(figdir / "mknn_vs_k_per_pair_shared.png", dpi=140)
        plt.close(fig)


def write_report(
    out: Path,
    manifest: pd.DataFrame,
    mdf: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# UniverseTBD SAE shared-basis mKNN (k-scaling)",
        "",
        "Ridge affine SAE shared-basis + IDF on UniverseTBD activation pairs.",
        "",
        "**Paper dense baseline** ([arXiv:2509.19453](https://arxiv.org/pdf/2509.19453) "
        "Table 2): ambient cosine MKNN on the full catalog (`dense_cosine`).",
        "Learned methods use train-fit maps/IDF and are scored as held-out queries "
        "with full-catalog galleries; lift is vs `dense_cosine_heldout`.",
        "",
        f"- mode: `{args.mode}`",
        f"- ks: `{sorted(mdf['k'].unique().tolist()) if not mdf.empty else []}`",
        f"- pairs included: {int(manifest['included'].sum())} / {len(manifest)}",
        "",
        "## Aggregate mean mKNN by method and k",
        "",
    ]
    if mdf.empty:
        lines.append("_No results._")
    else:
        primary = mdf[mdf["method"] != "dense_cosine_test_subset"]
        pivot = (
            primary.groupby(["method", "k"])["mknn"]
            .mean()
            .unstack("k")
            .sort_index()
        )
        lines.append("```")
        lines.append(pivot.to_string())
        lines.append("```")
        lines += [
            "",
            "## Shared-basis lift vs held-out dense (fair protocol)",
            "",
        ]
        for k in sorted(primary["k"].unique()):
            sub = primary[primary["k"] == k]
            paper_d = sub[sub["method"] == "dense_cosine"]["mknn"].mean()
            dense = sub[sub["method"] == "dense_cosine_heldout"]["mknn"].mean()
            shared = sub[sub["method"] == "shared_best_basis_idf"]["mknn"].mean()
            sae = sub[sub["method"] == "sae_idf_cosine"]["mknn"].mean()
            lines.append(
                f"- k={k}: paper_dense={paper_d:.4f}, dense_heldout={dense:.4f}, "
                f"sae_idf={sae:.4f}, shared_best_idf={shared:.4f}, "
                f"Δ(shared−heldout)={shared - dense:+.4f}"
            )
        lines += ["", "## Scaling notes", ""]
        best = primary[primary["method"] == "shared_best_basis_idf"].groupby("k")["mknn"].mean()
        dens = primary[primary["method"] == "dense_cosine_heldout"].groupby("k")["mknn"].mean()
        if len(best) >= 2 and len(dens) >= 2:
            ks = sorted(set(best.index) & set(dens.index))
            lift = [float(best[k] - dens[k]) for k in ks]
            lines.append(
                f"- shared−heldout_dense lift by k: "
                + ", ".join(f"k={k}:{v:+.4f}" for k, v in zip(ks, lift))
            )
            lines.append(
                "- lift increases with k"
                if lift[-1] > lift[0] + 0.005
                else (
                    "- lift decreases with k"
                    if lift[-1] < lift[0] - 0.005
                    else "- lift roughly stable across k"
                )
            )

        paper_rows = mdf[
            (mdf["method"] == "dense_cosine")
            & (mdf["k"] == 10)
            & mdf["pair"].isin(PAPER_TABLE2_DENSE_K10)
        ]
        if not paper_rows.empty:
            lines += [
                "",
                "## Dense vs paper Table 2 (k=10, overlapping pairs)",
                "",
                "Absolute baseline check. Residual gaps usually mean local parquet "
                "≠ official HF dump and/or N subsample (Legacy).",
                "",
                "```",
            ]
            for _, r in paper_rows.sort_values("pair").iterrows():
                paper = PAPER_TABLE2_DENSE_K10[r["pair"]]
                lines.append(
                    f"{r['pair']}: ours={r['mknn']:.4f}  paper={paper:.4f}  "
                    f"Δ={r['mknn'] - paper:+.4f}  n={int(r['n'])}"
                )
            lines.append("```")

        test_only = mdf[mdf["method"] == "dense_cosine_test_subset"]
        if not test_only.empty:
            lines += [
                "",
                "## Diagnostic: test-subset-only dense (inflated; not the baseline)",
                "",
            ]
            for k in sorted(test_only["k"].unique()):
                m = test_only[test_only["k"] == k]["mknn"].mean()
                d = primary[
                    (primary["k"] == k) & (primary["method"] == "dense_cosine")
                ]["mknn"].mean()
                lines.append(
                    f"- k={k}: dense_test_subset={m:.4f} vs paper_dense={d:.4f} "
                    f"(inflation {m - d:+.4f})"
                )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = platonic_root(args.platonic_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA unavailable; running on CPU", flush=True)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    pairs = load_pairs(resolve_path(root, args.pairs_yaml))

    if args.pairs.strip():
        selected = [x.strip() for x in args.pairs.split(",") if x.strip()]
    elif args.mode == "smoke":
        # Representative cross-model + cross-survey pairs
        selected = [
            "physics_vit_dino",
            "jwst_cross_vit",
            "desi_cross_vit",
            "legacy_cross_vit",
            "legacy_ls_vit_dinov3",
        ]
        selected = [p for p in selected if p in pairs]
    else:
        selected = list(pairs.keys())

    if args.surveys.strip():
        surveys = {s.strip() for s in args.surveys.split(",") if s.strip()}
        selected = [p for p in selected if pairs[p].get("survey") in surveys]

    tag = args.run_tag or args.mode
    out = resolve_path(root, args.out_dir) / tag
    figdir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root, pairs, selected)
    manifest.to_csv(out / "pair_manifest.csv", index=False)
    print(manifest.to_string(index=False), flush=True)

    included = manifest[manifest["included"]]
    if included.empty:
        raise RuntimeError("no includable pairs")

    config = {
        "args": vars(args),
        "ks": ks,
        "n_pairs_requested": len(selected),
        "n_pairs_included": int(len(included)),
        "device": str(device),
        "protocol": (
            "dense_cosine = paper full-catalog ambient cosine MKNN; "
            "learned methods + dense_cosine_heldout = held-out queries, full gallery; "
            "maps/IDF fit on train only"
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    all_mknn: list[dict] = []
    all_fit: list[dict] = []
    for _, row in included.iterrows():
        name = row["pair"]
        cfg = pairs[name]
        m_rows, f_rows = eval_pair(
            root,
            name,
            cfg,
            Path(row["sae1"]),
            Path(row["sae2"]),
            ks=ks,
            max_n=args.max_n,
            test_size=args.test_size,
            alpha=args.alpha,
            seed=args.seed,
            device=device,
            row_batch=args.row_batch,
            allow_truncate=args.allow_truncate or cfg.get("kind") == "cross_survey",
        )
        all_mknn.extend(m_rows)
        all_fit.extend(f_rows)

    mdf = pd.DataFrame(all_mknn)
    fdf = pd.DataFrame(all_fit)
    mdf.to_parquet(out / "mknn_by_k.parquet", index=False)
    fdf.to_parquet(out / "pair_fit_metrics.parquet", index=False)

    # aggregate summary
    if not mdf.empty:
        agg_src = (
            mdf[mdf["protocol"] != "test_subset_mknn"]
            if "protocol" in mdf.columns
            else mdf[mdf["method"] != "dense_cosine_test_subset"]
        )
        agg = (
            agg_src.groupby(["method", "k"])["mknn"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.to_csv(out / "aggregate_summary.csv", index=False)
    make_figures(
        mdf[mdf["method"] != "dense_cosine_test_subset"] if not mdf.empty else mdf,
        figdir,
    )
    write_report(out, manifest, mdf, args)
    print(f"[done] → {out}", flush=True)
    if not mdf.empty:
        print(mdf.groupby(["method", "k"])["mknn"].mean().unstack("k").to_string())


if __name__ == "__main__":
    main()
