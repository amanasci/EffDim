#!/usr/bin/env python3
"""CLI for geometry-only atlas ablation (no priors / topology / labels)."""

from __future__ import annotations

import argparse

from .geometry_ablation import STAGES, AblationConfig, run_ablation


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Geometry-only Physics atlas ablation")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_activation_atlas_geometry_ablation",
    )
    p.add_argument("--n-charts-grid", type=int, nargs="+", default=[6, 12, 24, 48])
    p.add_argument("--local-dims", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--charts-per-sample", type=int, default=3)
    p.add_argument("--top-k-mlp", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curvature-anchors", type=int, default=4)
    p.add_argument("--fd-autodiff-anchors", type=int, default=2)
    p.add_argument("--max-decoder-train-samples", type=int, default=1536)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = AblationConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        n_charts_grid=list(args.n_charts_grid),
        local_dims=list(args.local_dims),
        charts_per_sample=args.charts_per_sample,
        top_k_mlp=args.top_k_mlp,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        curvature_anchors=args.curvature_anchors,
        fd_autodiff_anchors=args.fd_autodiff_anchors,
        max_decoder_train_samples=args.max_decoder_train_samples,
        force=args.force,
    )
    run_ablation(cfg)


if __name__ == "__main__":
    main()
