#!/usr/bin/env python3
"""CLI for Physics activation atlas experiment."""

from __future__ import annotations

import argparse

from .pipeline import STAGES, AtlasConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overlapping local atlas on dense Physics ViT activations")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument("--output-dir", default="outputs/geometry/physics_activation_atlas")
    p.add_argument("--parquet", default="data_hf/physics/vit_base_test.parquet")
    p.add_argument("--column", default="vit_base_galaxies")
    p.add_argument(
        "--selection-path",
        default="outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz",
    )
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-charts", type=int, default=12)
    p.add_argument("--charts-per-sample", type=int, default=3)
    p.add_argument("--chart-selection", default="fps", choices=["fps", "density_stratified", "stratified"])
    p.add_argument("--chart-bandwidth-policy", default="median_knn")
    p.add_argument("--min-chart-samples", type=int, default=40)
    p.add_argument("--max-chart-samples", type=int, default=None)
    p.add_argument("--candidate-dims", type=int, nargs="+", default=[8, 16])
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--decoder-hidden-dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--decoder-activation", default="softplus", choices=["softplus", "tanh"])
    p.add_argument("--decoder-residual-scale", type=float, default=0.01)
    p.add_argument("--decoder-output-normalization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-decoder-train-samples", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prior-family", default="all")
    p.add_argument("--mixture-components", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--cf-frequency-count", type=int, default=256)
    p.add_argument("--cf-frequency-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--cf-loss-weight", type=float, default=1.0)
    p.add_argument("--prior-learning-rate", type=float, default=0.02)
    p.add_argument("--prior-epochs", type=int, default=40)
    p.add_argument("--prior-patience", type=int, default=8)
    p.add_argument("--patch-max-points", type=int, default=400)
    p.add_argument("--nerve-maxdim", type=int, default=2)
    p.add_argument("--curvature-anchors", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--synth-n", type=int, default=400)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = AtlasConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        parquet=args.parquet,
        column=args.column,
        selection_path=args.selection_path,
        max_n=args.max_n,
        global_seed=args.global_seed,
        seed=args.seed,
        n_charts=args.n_charts,
        charts_per_sample=args.charts_per_sample,
        chart_selection=args.chart_selection,
        chart_bandwidth_policy=args.chart_bandwidth_policy,
        min_chart_samples=args.min_chart_samples,
        max_chart_samples=args.max_chart_samples,
        candidate_dims=list(args.candidate_dims),
        latent_dim=args.latent_dim,
        decoder_hidden_dims=list(args.decoder_hidden_dims),
        decoder_activation=args.decoder_activation,
        decoder_residual_scale=args.decoder_residual_scale,
        decoder_output_normalization=args.decoder_output_normalization,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_decoder_train_samples=args.max_decoder_train_samples,
        device=args.device,
        prior_family=args.prior_family,
        mixture_components=list(args.mixture_components),
        cf_frequency_count=args.cf_frequency_count,
        cf_frequency_scales=list(args.cf_frequency_scales),
        cf_loss_weight=args.cf_loss_weight,
        prior_learning_rate=args.prior_learning_rate,
        prior_epochs=args.prior_epochs,
        prior_patience=args.prior_patience,
        patch_max_points=args.patch_max_points,
        nerve_maxdim=args.nerve_maxdim,
        curvature_anchors=args.curvature_anchors,
        n_seeds=args.n_seeds,
        synth_n=args.synth_n,
        smoke=args.smoke,
        force=args.force,
    )
    if cfg.smoke and cfg.output_dir == "outputs/geometry/physics_activation_atlas":
        cfg.output_dir = "outputs/geometry/physics_activation_atlas_smoke"
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
