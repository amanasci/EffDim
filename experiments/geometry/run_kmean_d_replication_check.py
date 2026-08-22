#!/usr/bin/env python3
"""Quick d∈{8,12,16} K_mean vs OOF global-probe geography check (vit_base screen)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.confirmatory_object_curvature import (  # noqa: E402
    _fit_neighborhood,
)
from geometry.physics_activation_atlas.curvature_probe_alignment import traceless_B0  # noqa: E402
from geometry.physics_activation_atlas.curvature_probe_screen import (  # noqa: E402
    partial_spearman,
    spearman_dict,
)
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import (  # noqa: E402
    EPS,
    load_model_X,
)
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument("--model", default="vit_base")
    p.add_argument("--target", default="mag_r_desi")
    p.add_argument("--dims", default="8,12,16")
    p.add_argument("--scales", default="256,512,1024,2048")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--anchor-mode",
        choices=("screen", "all", "holdout384"),
        default="screen",
        help="screen=96 quadratic screen; all=all prepared anchors; holdout384=IDs from prior magnitude run",
    )
    p.add_argument(
        "--holdout-metrics",
        default="outputs/geometry/physics_global_probe_curvature_magnitude/anchor_target_curvature_metrics.parquet",
    )
    p.add_argument("--tag", default="", help="Optional output subdirectory suffix")
    args = p.parse_args(argv)

    root = platonic_root()
    out = resolve_path(root, args.output_dir)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    model = args.model
    target = args.target

    anchors_sid = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
    if args.anchor_mode == "screen":
        qcv = pd.read_parquet(out / "quadratic_dimension_cv.parquet")
        use_sids = sorted(int(x) for x in qcv.sample_id.unique())
        tag = args.tag or "screen96"
    elif args.anchor_mode == "all":
        use_sids = sorted(int(x) for x in anchors_sid)
        tag = args.tag or f"all{len(use_sids)}"
    else:
        hold = pd.read_parquet(resolve_path(root, args.holdout_metrics))
        prior_sids = set(int(x) for x in hold.sample_id.unique())
        use_sids = sorted(sid for sid in anchors_sid if int(sid) in prior_sids)
        if len(use_sids) < 50:
            # fall back: use prior IDs that exist in current selection via geo table
            geo0 = pd.read_parquet(out / "local_probe_fields.parquet")
            avail = set(int(x) for x in geo0.sample_id.unique())
            use_sids = sorted(prior_sids & avail)
        tag = args.tag or f"holdout{len(use_sids)}"
        print(f"[d-check] holdout384 overlap n={len(use_sids)} / prior={len(prior_sids)}", flush=True)

    pack = np.load(out / "model_neighbourhoods" / f"{model}_kmax{max(scales)}.npz")
    # neighbourhood rows are aligned to prepare anchors.npz order
    pack_sids = anchors_sid
    X = load_model_X(out, model)
    geo = pd.read_parquet(out / "local_probe_fields.parquet")
    gp = pd.read_parquet(out / "graph_dimension_prior.parquet")

    sub_out = out / f"d_replication_check_{tag}"
    sub_out.mkdir(exist_ok=True)
    (sub_out / "anchor_ids.json").write_text(
        json.dumps({"anchor_mode": args.anchor_mode, "n": len(use_sids), "sample_ids": use_sids}, indent=2)
    )

    sid_to_ai = {int(s): i for i, s in enumerate(pack_sids)}
    feat_rows = []
    t0 = time.time()
    for k in scales:
        for d in dims:
            n_ok = 0
            for sid in use_sids:
                ai = sid_to_ai.get(int(sid))
                if ai is None:
                    continue
                N = pack["neigh"][ai, :k]
                rho = float(pack["dists"][ai, k - 1])
                chart, _, info, _, _, reason = _fit_neighborhood(
                    X, N, d, seed=args.seed + ai + 17 * k + d
                )
                if chart is None:
                    continue
                B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
                d_eff = chart.J.shape[1]
                K_mean = float(np.linalg.norm(H))
                K_tr = float(np.linalg.norm(B0))
                feat_rows.append(
                    {
                        "model": model,
                        "sample_id": int(sid),
                        "scale_k": int(k),
                        "d": int(d),
                        "d_eff": int(d_eff),
                        "K_mean": K_mean,
                        "K_traceless": K_tr,
                        "recon_error": float(info.get("val_E_TRS", np.nan)),
                        "knn_radius": rho,
                        "log_knn_radius": float(np.log(max(rho, EPS))),
                        "fit_reason": reason,
                    }
                )
                n_ok += 1
            print(f"[d-check] {model} k={k} d={d} n={n_ok}/{len(use_sids)}", flush=True)

    feats = pd.DataFrame(feat_rows)
    feats.to_parquet(sub_out / "curvature_features_d_sweep.parquet", index=False)

    assoc_rows = []
    ggeo = geo[
        (geo.model == model)
        & (geo.target == target)
        & (geo.neighbourhood == "model")
        & (geo.scale_k.isin(scales))
    ]
    for k in scales:
        for d in dims:
            f = feats[(feats.scale_k == k) & (feats.d == d)]
            g = ggeo[ggeo.scale_k == k].merge(
                f, on=["sample_id", "scale_k"], how="inner", suffixes=("", "_feat")
            )
            g = g.merge(
                gp[(gp.model == model) & (gp.scale_k == k)][
                    ["sample_id", "graph_support_turnover", "graph_boundary_imbalance"]
                ],
                on="sample_id",
                how="left",
            )
            if len(g) < 20:
                continue
            log_r = (
                g["log_knn_radius_feat"].to_numpy(float)
                if "log_knn_radius_feat" in g.columns
                else g["log_knn_radius"].to_numpy(float)
            )
            Km = g.K_mean.to_numpy(float)
            y = g.local_r2.to_numpy(float)
            C0 = np.column_stack(
                [
                    log_r,
                    g.local_label_variance.to_numpy(float),
                    g.recon_error.to_numpy(float),
                    g.local_evaluation_count.to_numpy(float),
                ]
            )
            C6 = np.column_stack(
                [
                    C0,
                    g.graph_support_turnover.fillna(0).to_numpy(float),
                    g.graph_boundary_imbalance.fillna(0).to_numpy(float),
                ]
            )
            raw = spearman_dict(Km, y)
            p0 = partial_spearman(Km, y, C0)
            p6 = partial_spearman(Km, y, C6)
            assoc_rows.append(
                {
                    "model": model,
                    "target": target,
                    "scale_k": k,
                    "d": d,
                    "n": int(len(g)),
                    "raw_rho_K_mean": raw["rho"],
                    "partial_C0": p0["rho"],
                    "p_partial_C0": p0["pvalue"],
                    "partial_C6": p6["rho"],
                    "p_partial_C6": p6["pvalue"],
                }
            )

    assoc = pd.DataFrame(assoc_rows)
    assoc.to_parquet(sub_out / "kmean_associations_d_sweep.parquet", index=False)
    assoc.to_csv(sub_out / "kmean_associations_d_sweep.csv", index=False)

    prior = {
        "note": "Prior magnitude confirmatory: vit_base mag_r k=2048 d=8 n=384",
        "raw_K_mean": -0.330396,
        "partial_K_mean_C0": -0.314380,
    }
    report = [
        "# K_mean d-replication check (OOF global probes)",
        "",
        f"Model={model}, target={target}, anchor_mode={args.anchor_mode}, n_anchors={len(use_sids)}",
        f"dims={dims}, scales={scales}",
        f"elapsed_s={time.time()-t0:.1f}",
        "",
        "## Associations (local OOF R²)",
        "",
        assoc.to_string(index=False),
        "",
        "## Prior reference",
        json.dumps(prior, indent=2),
        "",
        "## Reading",
        "- Compare d=8 / k=2048 partial_C0 to prior −0.314.",
        "- If d=8 recovers negativity but d=16 does not, tangent-dimension mismatch explains the multimodel d_star result.",
        "- all/holdout modes test whether the 96-anchor screen was underpowered or compositionally different.",
    ]
    (sub_out / "REPORT.md").write_text("\n".join(report))
    print(assoc.to_string(index=False), flush=True)
    print(f"[d-check] wrote {sub_out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
