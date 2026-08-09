#!/usr/bin/env python3
"""Is the sparse fringe of the embedding manifold more curved than dense regions?

Stage A (label-free) measures local curvature per test point and stratifies it by
local density, for each model and at several neighbourhood scales.

The headline metric is `kappa_ratio` = kappa_jet / kappa_null: a local
second-fundamental-form magnitude divided by the magnitude the identical fit
produces after the tangent/normal pairing is destroyed by permutation. Naive
alternatives (`rf_k`, `kappa_jet`, `kappa_naive_ratio`) are computed too, but
only as negative controls -- on a synthetic manifold that is exactly flat with
non-uniform density they reach |rho(d_k, metric)| ~ 0.99, so any conclusion drawn
from them would be an artifact of neighbourhood radius rather than geometry.

Every real number is reported beside the same number computed on a FLAT surrogate
built from the same embeddings, which is what makes the real column readable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must precede numpy. Each point's matrices are small, so multithreaded BLAS
# costs ~20x through thread oversubscription; parallelism is taken over points
# instead (--n-jobs), and worker processes inherit these.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import load_embeddings, platonic_root  # noqa: E402
from curvature_core import (  # noqa: E402
    build_knn,
    compute_curvature_suite,
    flatten_null,
    synthetic_manifold,
)
from density_stats import (  # noqa: E402
    QUARTILE_NAMES,
    density_quartile_stats,
    density_quartiles,
    epsilon_ball_feasibility,
    partial_spearman,
)
from multiscale_curvature_probe import estimate_global_id  # noqa: E402

HEADLINE = "kappa_ratio"
DIAGNOSTICS = ["kappa_jet", "kappa_null", "kappa_z", "rf_k", "kappa_naive_ratio",
               "kappa_slope", "noise_floor", "R_med", "r2_quad"]
PLOT_METRICS = ["kappa_ratio", "kappa_jet", "rf_k", "kappa_naive_ratio"]

# Categorical slots 1 and 2 of the reference palette; validated as a pair
# (adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface).
C_REAL, C_NULL = "#2a78d6", "#eb6834"
SEQ_RAMP = ["#bcd7f4", "#7fb0e8", "#3f88db", "#1c4f8f"]  # Q1 -> Q4, light -> dark
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"


# ---------------------------------------------------------------------------
# One dataset (real embeddings, a flat surrogate, or a synthetic control)
# ---------------------------------------------------------------------------

def run_dataset(
    name: str,
    X: np.ndarray,
    *,
    k_t: int,
    K_list: list[int],
    args: argparse.Namespace,
) -> dict:
    """Build one kNN graph and evaluate the curvature suite at every scale."""
    K_max = max(K_list)
    t0 = time.time()
    print(f"  [{name}] kNN graph (n={len(X)}, K={K_max})...", flush=True)
    dists, idx = build_knn(X, K_max)
    d_k = dists[:, args.k_density - 1].astype(np.float64)

    n_quad_cols = 1 + k_t + args.p_quad * (args.p_quad + 1) // 2
    scales: dict[int, dict[str, np.ndarray]] = {}
    for K in K_list:
        if K < args.min_k_factor * n_quad_cols:
            print(
                f"  [{name}] SKIP K={K}: design has {n_quad_cols} columns and "
                f"K < {args.min_k_factor}x that, so the null absorbs the signal.",
                flush=True,
            )
            continue
        print(f"  [{name}] curvature suite at K={K}...", flush=True)
        scales[K] = compute_curvature_suite(
            X, idx[:, :K], k_t,
            p_quad=args.p_quad, m_norm=args.m_norm, n_perm=args.n_perm,
            seed=args.seed, progress_every=args.progress_every, n_jobs=args.n_jobs,
        )
    if not scales:
        raise SystemExit(
            f"No scale in {K_list} satisfies K >= {args.min_k_factor} x {n_quad_cols} "
            f"design columns (k_t={k_t}, p_quad={args.p_quad}). Raise --k-ladder, or "
            f"lower --min-k-factor knowing the null will absorb some real signal."
        )
    print(f"  [{name}] done in {time.time() - t0:.0f}s", flush=True)
    return {"d_k": d_k, "scales": scales, "k_t": k_t, "n": int(len(X))}


def summarise(bundle: dict, args: argparse.Namespace) -> dict:
    """Quartile stats for every metric at every scale, plus derived kappa_ms."""
    d_k = bundle["d_k"]
    out: dict = {}
    for K, suite in bundle["scales"].items():
        metrics = dict(suite)
        Ks = sorted(bundle["scales"])
        if len(Ks) >= 2 and K == Ks[-1]:
            # kappa_ms = rf(K_large) - rf(K_small), deliberately UNCLIPPED:
            # multiscale_curvature_probe clips at 0, which ties ~half the points
            # and corrupts Spearman.
            metrics["kappa_ms"] = suite["rf_k"] - bundle["scales"][Ks[0]]["rf_k"]
        out[str(K)] = {
            m: density_quartile_stats(v, d_k, n_boot=args.n_boot, seed=args.seed)
            for m, v in metrics.items()
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def plot_quartiles(real: dict, null: dict, K: int, tag: str, path: Path) -> None:
    """Median curvature by density quartile, real vs flat surrogate."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    x = np.arange(4)
    for ax, metric in zip(axes.ravel(), PLOT_METRICS):
        for series, stats, color in (("Real", real, C_REAL), ("Flat surrogate", null, C_NULL)):
            q = stats.get(str(K), {}).get(metric, {}).get("quartiles")
            if not q:
                continue
            med = np.array([q[n]["median"] for n in QUARTILE_NAMES])
            lo = np.array([q[n]["ci"][0] for n in QUARTILE_NAMES])
            hi = np.array([q[n]["ci"][1] for n in QUARTILE_NAMES])
            ax.errorbar(x, med, yerr=[med - lo, hi - med], color=color, marker="o",
                        markersize=7, linewidth=2, capsize=4, label=series)
        _style(ax)
        ax.set_xticks(x, ["Q1\ndensest", "Q2", "Q3", "Q4\nsparsest"])
        ax.set_title(metric, fontsize=10, color=INK, loc="left")
        if metric == HEADLINE:
            ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    fig.suptitle(f"Curvature by density quartile — {tag} (K={K})",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_scale_ladder(bundle: dict, tag: str, path: Path) -> None:
    """Median kappa_ratio against the actual physical neighbourhood radius.

    Each quartile traces its own x-range, so this reads off directly whether
    sparse points are more curved *at a matched physical scale* rather than
    merely at a matched neighbour count.
    """
    import matplotlib.pyplot as plt

    q = density_quartiles(bundle["d_k"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for j, name in enumerate(QUARTILE_NAMES):
        xs, ys = [], []
        for K in sorted(bundle["scales"]):
            suite = bundle["scales"][K]
            sel = (q == j) & np.isfinite(suite["kappa_ratio"])
            if sel.sum() < 10:
                continue
            xs.append(np.median(suite["R_med"][sel]))
            ys.append(np.median(suite["kappa_ratio"][sel]))
        if xs:
            ax.plot(xs, ys, color=SEQ_RAMP[j], marker="o", markersize=7, linewidth=2,
                    label=f"{name}{' densest' if j == 0 else ' sparsest' if j == 3 else ''}")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)
    _style(ax)
    ax.set_xlabel("median neighbourhood radius", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("median kappa_ratio", fontsize=9, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, title="density quartile",
              title_fontsize=9)
    ax.set_title(f"Curvature vs physical scale — {tag}", fontsize=12, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_null_calibration(bundle: dict, K: int, tag: str, path: Path) -> None:
    """kappa_jet against its own permutation null, coloured by density."""
    import matplotlib.pyplot as plt

    suite = bundle["scales"][K]
    ok = np.isfinite(suite["kappa_jet"]) & np.isfinite(suite["kappa_null"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sc = ax.scatter(suite["kappa_null"][ok], suite["kappa_jet"][ok],
                    c=np.log10(bundle["d_k"][ok]), s=7, cmap="viridis", alpha=0.65,
                    linewidths=0)
    lim = [0, float(np.nanpercentile(suite["kappa_jet"][ok], 99.5))]
    ax.plot(lim, lim, color=INK_MUTED, linewidth=1, linestyle=":")
    _style(ax)
    ax.set_xlabel("kappa_null (permutation)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("kappa_jet (observed)", fontsize=9, color=INK_MUTED)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10 d_k  (larger = sparser)", fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"Null calibration — {tag} (K={K})", fontsize=12, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v, spec="+.3f"):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def quartile_table(stats: dict, metric: str, label: str) -> list[str]:
    q = stats.get(metric, {}).get("quartiles")
    if not q:
        return []
    med = " | ".join(_fmt(q[n]["median"], ".4f") for n in QUARTILE_NAMES)
    mwu = stats[metric].get("mwu", {})
    sp = stats[metric].get("spearman_dk", {})
    return [f"| {label} | {med} | {_fmt(mwu.get('rank_biserial'))} | "
            f"{_fmt(sp.get('rho'))} | [{_fmt(sp.get('ci', [np.nan, np.nan])[0])}, "
            f"{_fmt(sp.get('ci', [np.nan, np.nan])[1])}] |"]


def build_markdown(payload: dict, args: argparse.Namespace) -> str:
    L = [
        "# Is the sparse fringe more curved than dense regions?",
        "",
        f"- n_max={args.max_n}, test_size={args.test_size}, seed={args.seed}",
        f"- K ladder={args.k_ladder}, density proxy d_k at K={args.k_density}",
        f"- p_quad={args.p_quad}, m_norm={args.m_norm}, n_perm={args.n_perm}, "
        f"min_k_factor={args.min_k_factor}",
        "",
        "Q1 = densest quartile, Q4 = sparsest. `rank_biserial` compares Q4 vs Q1: "
        "positive means the sparsest quartile has larger values. `rho` is Spearman "
        "against d_k (larger d_k = sparser).",
        "",
        "## 1. Fixed-radius feasibility gate",
        "",
        "Why fixed-k neighbourhoods were used rather than fixed-radius ones.",
        "",
        "| model | k_t | median d_k Q1 | median d_k Q4 | ratio | predicted Q4 neighbours at Q1 eps | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in payload["per_model"]:
        g = m["epsilon_ball_gate"]
        L.append(
            f"| {m['model']} | {g['k_t']:.0f} | {g['median_d_k_Q1']:.4f} | "
            f"{g['median_d_k_Q4']:.4f} | {g['dk_ratio_Q4_Q1']:.3f} | "
            f"{g['predicted_Q4_neighbours_at_Q1_eps']:.2g} | {g['verdict']} |"
        )

    L += [
        "",
        "## 2. Methodology validation (synthetic ground truth)",
        "",
        "Manifolds with known curvature and deliberately non-uniform density. A "
        "usable metric must read ~1.0 with rho ~ 0 on the flat manifolds, and must "
        "still detect the sphere. rho far from zero on a FLAT manifold means the "
        "metric is measuring neighbourhood radius, not geometry.",
        "",
        "| control | true kappa | K | metric | median | rho(d_k, metric) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for c in payload["synthetic_controls"]:
        for K, per_metric in sorted(c["summary"].items(), key=lambda kv: int(kv[0])):
            for metric in [HEADLINE, "kappa_jet", "rf_k", "kappa_naive_ratio"]:
                st = per_metric.get(metric, {})
                q = st.get("quartiles")
                if not q:
                    continue
                allmed = np.median([q[n]["median"] for n in QUARTILE_NAMES])
                L.append(
                    f"| {c['name']} | {c['true_kappa']:.2f} | {K} | {metric} | "
                    f"{allmed:.4f} | {_fmt(st.get('spearman_dk', {}).get('rho'))} |"
                )

    for m in payload["per_model"]:
        for K in sorted(m["summary"], key=int):
            L += [
                "",
                f"## 3. Headline — {m['model']}, K={K}",
                "",
                "| series | Q1 | Q2 | Q3 | Q4 | rank_biserial Q4-Q1 | rho(d_k) | rho 95% CI |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
            L += quartile_table(m["summary"][K], HEADLINE, "**Real**")
            for null_name, null_sum in m["null_summaries"].items():
                if K in null_sum:
                    L += quartile_table(null_sum[K], HEADLINE, f"Flat surrogate ({null_name})")
            # The synthetic flat manifold is a different dataset, so only its rho
            # column is comparable -- but it is the method's own bias floor and
            # belongs beside the real number, not buried in section 2.
            for c in payload["synthetic_controls"]:
                if "flat" in c["name"] and K in c["summary"]:
                    L += quartile_table(c["summary"][K], HEADLINE,
                                        "_method bias floor (synthetic flat)_")

            L += [
                "",
                f"### 4. Diagnostics and negative controls — {m['model']}, K={K}",
                "",
                "These are reported for completeness. Section 2 shows they carry a "
                "large density trend on manifolds that are exactly flat, so they "
                "cannot support a conclusion in either direction.",
                "",
                "| metric | series | Q1 | Q2 | Q3 | Q4 | rank_biserial | rho(d_k) | rho 95% CI |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
            extra = ["kappa_ms"] if "kappa_ms" in m["summary"][K] else []
            for metric in ["kappa_jet", "kappa_null", "rf_k", "kappa_naive_ratio",
                           "kappa_slope", "noise_floor", "R_med"] + extra:
                for row in quartile_table(m["summary"][K], metric, "real"):
                    L.append(f"| {metric} " + row)
                for null_name, null_sum in m["null_summaries"].items():
                    if K in null_sum:
                        for row in quartile_table(null_sum[K], metric, f"null:{null_name}"):
                            L.append(f"| {metric} " + row)

    L += ["", "## 5. Scale ladder", "",
          "See `<model>_scale_ladder.png`: median kappa_ratio against the actual "
          "median neighbourhood radius, one line per density quartile. Because each "
          "quartile spans its own radius range, overlapping x-values compare "
          "curvature at a matched physical scale.", ""]

    if payload.get("connectback"):
        L += build_connectback_markdown(payload)

    L += ["", "## Limitations", "",
          "- The permutation null assumes normal residuals are exchangeable across "
          "neighbours once the linear term is removed. Direction-dependent "
          "heteroscedastic thickness would misspecify it.",
          "- `p_quad` selects the top-p_quad *local-variance* tangent directions, and "
          "which directions those are is itself mildly density-dependent.",
          "- The flat surrogate bounds the artifact under the null hypothesis only. "
          "It does not bound residual scale sensitivity under a genuinely curved "
          "alternative.",
          ""]
    return "\n".join(L) + "\n"


def build_connectback_markdown(payload: dict) -> list[str]:
    L = ["", "## 6. Connect-back: does curvature explain probe / SAE failure?", "",
         "`partial rho(curv, target | d_k)` asks whether curvature adds anything "
         "beyond raw density; `partial rho(d_k, target | curv)` asks the reverse. "
         "`rho(curv, n_valid)` is the label-availability confound.", ""]
    for cb in payload["connectback"]:
        L += [f"### {cb['model']}, K={cb['K']}", "",
              "| curvature | target | rho | partial rho given d_k | partial rho given n_valid | rho(curv, n_valid) |",
              "|---|---|---:|---:|---:|---:|"]
        for r in cb["rows"]:
            L.append(
                f"| {r['curv']} | {r['target']} | {_fmt(r['rho'])} | "
                f"{_fmt(r['partial_given_dk'])} | {_fmt(r['partial_given_nvalid'])} | "
                f"{_fmt(r['rho_vs_nvalid'])} |"
            )
        L.append("")
    if payload.get("probe_health"):
        ph = payload["probe_health"]
        L += ["### 7. Probe health", "",
              f"- {ph['n_good']} of {ph['n_total']} probes have r2_cv > {ph['r2_min']}.",
              f"- Legacy target `mean_residual_all`: median {ph['all_median']:.3f}, "
              f"mean {ph['all_mean']:.3f} (a standardised squared residual should be ~1).",
              f"- Cleaned target `mean_residual_good`: median {ph['good_median']:.3f}, "
              f"mean {ph['good_mean']:.3f}.",
              f"- Spearman(mean_residual_all, n_valid_probes) = {ph['all_vs_nvalid']:+.3f} "
              "— the legacy target partly encodes which galaxies have rare labels.",
              ""]
    return L


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--model-a", default="vit_base")
    p.add_argument("--model-b", default="dinov3_vitb16")
    p.add_argument("--dataset", default="physics")
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)

    # At the real data's intrinsic dimension (k_t ~ 19) the design has ~26
    # columns, and calibration on synthetic manifolds shows the estimator needs
    # K/cols >~ 6 before the flat-manifold median settles near 1.0. That puts the
    # K=50 scale of the d_k density proxy out of reach for the jet fit, so d_k is
    # used only to define density, never to fit curvature.
    p.add_argument("--k-ladder", default="200,300,400")
    p.add_argument("--k-density", type=int, default=50,
                   help="K used for the d_k density proxy and the quartile split")
    p.add_argument("--k-tangent", type=int, default=0,
                   help="p_lin; 0 means use the Two-NN intrinsic dimension estimate")
    p.add_argument("--p-quad", type=int, default=3)
    p.add_argument("--m-norm", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=16)
    p.add_argument("--min-k-factor", type=int, default=6,
                   help="Require K >= this many times the design column count. "
                        "Below ~6 the permutation null starts absorbing real signal.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="Processes for the per-point loop; BLAS is pinned to 1 thread each")

    p.add_argument("--nulls", default="flat_gauss,flat_shuffle,synthetic")
    p.add_argument("--synthetic-n", type=int, default=4000)
    p.add_argument("--synthetic-curvature", type=float, default=0.3)

    p.add_argument("--skip-probes", action="store_true",
                   help="Stage A only: no labels, no HF download, no SAE join")
    p.add_argument("--probes", default="independent")
    p.add_argument("--probe-r2-min", type=float, default=0.1)
    p.add_argument("--sae-npz-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = platonic_root(args.platonic_root)
    K_list = sorted(int(k) for k in args.k_ladder.split(","))
    nulls = [n for n in args.nulls.split(",") if n] if args.nulls else []

    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "experiments" / "physics-probe-subspace" / "outputs"
        / f"run_{time.strftime('%Y%m%d_%H%M%S')}" / "density_curvature"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}", flush=True)

    parquet_dir = root / "data_hf" / args.dataset
    model_cfgs = [
        (args.model_a, parquet_dir / f"{args.model_a}_test.parquet", f"{args.model_a}_galaxies"),
        (args.model_b, parquet_dir / f"{args.model_b}_test.parquet", f"{args.model_b}_galaxies"),
    ]

    from sklearn.model_selection import train_test_split

    per_model, k_t_ref = [], None
    for model_name, parquet_path, col in model_cfgs:
        print(f"\n{'=' * 64}\n  {model_name}\n{'=' * 64}", flush=True)
        Z = load_embeddings(parquet_path, col=col)
        n_total = min(args.max_n, len(Z))
        Z = Z[:n_total]
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

        # Identical split convention to sae_curvature_probe, so idx_test lines up
        # with the previously saved SAE outputs.
        idx_train, idx_test = train_test_split(
            np.arange(n_total), test_size=args.test_size, random_state=args.seed
        )
        Z_test = Z[idx_test]

        k_t = args.k_tangent or max(2, int(round(estimate_global_id(Z[idx_train]))))
        k_t_ref = k_t_ref or k_t
        print(f"  intrinsic dimension k_t = {k_t}  (n_test={len(Z_test)})", flush=True)

        real = run_dataset("real", Z_test, k_t=k_t, K_list=K_list, args=args)
        null_bundles = {
            mode.replace("flat_", ""): run_dataset(
                mode, flatten_null(Z_test, k_t, mode.replace("flat_", ""), args.seed),
                k_t=k_t, K_list=K_list, args=args)
            for mode in nulls if mode.startswith("flat_")
        }

        entry = {
            "model": model_name,
            "col": col,
            "k_t": k_t,
            "n_test": int(len(Z_test)),
            "epsilon_ball_gate": epsilon_ball_feasibility(real["d_k"], k_t, args.k_density),
            "summary": summarise(real, args),
            "null_summaries": {n: summarise(b, args) for n, b in null_bundles.items()},
        }
        per_model.append(entry)

        tag = model_name.replace("/", "_")
        for K, suite in real["scales"].items():
            np.savez_compressed(
                out_dir / f"{tag}_curvature_K{K}.npz",
                d_k=real["d_k"], idx_test=idx_test, k_t=k_t, K=K,
                p_quad=args.p_quad, p_lin=k_t, **suite,
            )
        for n, b in null_bundles.items():
            for K, suite in b["scales"].items():
                np.savez_compressed(out_dir / f"{tag}_null_{n}_K{K}.npz",
                                    d_k=b["d_k"], K=K, **suite)

        if real["scales"]:
            K_plot = max(real["scales"])
            first_null = next(iter(entry["null_summaries"].values()), {})
            plot_quartiles(entry["summary"], first_null, K_plot, tag,
                           out_dir / f"{tag}_quartiles.png")
            plot_scale_ladder(real, tag, out_dir / f"{tag}_scale_ladder.png")
            plot_null_calibration(real, K_plot, tag, out_dir / f"{tag}_null_calibration.png")
        # Kept in memory for Stage B only; both keys are stripped before the
        # payload is serialised.
        entry["_real"] = real
        entry["_data"] = {"Z": Z, "idx_train": idx_train, "idx_test": idx_test,
                          "n_total": n_total}

    synthetic_controls = []
    if "synthetic" in nulls:
        print(f"\n{'=' * 64}\n  synthetic controls (d={k_t_ref})\n{'=' * 64}", flush=True)
        for label, kind, curv in [("synthetic flat", "flat", 0.0),
                                  ("synthetic sphere", "sphere", args.synthetic_curvature)]:
            X, true_k = synthetic_manifold(
                args.synthetic_n, 768, k_t_ref, kind=kind,
                curvature=curv or 0.3, noise=0.02, seed=args.seed)
            b = run_dataset(label, X, k_t=k_t_ref, K_list=K_list, args=args)
            synthetic_controls.append(
                {"name": label, "true_kappa": true_k, "summary": summarise(b, args)})
            for K, suite in b["scales"].items():
                np.savez_compressed(
                    out_dir / f"{label.replace(' ', '_')}_K{K}.npz",
                    d_k=b["d_k"], true_kappa=true_k, K=K, **suite)

    payload = {
        "experiment": "density_curvature_probe",
        "args": vars(args),
        "synthetic_controls": synthetic_controls,
        "per_model": per_model,
    }

    if not args.skip_probes:
        import stage_b_connectback

        stage_b_connectback.attach(payload, args, root, out_dir)

    for m in payload["per_model"]:
        m.pop("_real", None)
        m.pop("_data", None)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    (out_dir / "results.md").write_text(build_markdown(payload, args))
    print(f"\nResults written to {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
