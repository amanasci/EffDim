#!/usr/bin/env python3
"""CLI: confirmatory multi-seed Δ_S + frozen object-level curvature features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.confirmatory_object_curvature import (  # noqa: E402
    STAGES,
    ConfirmatoryConfig,
    run_confirmatory_object,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Confirmatory sphere-normal Δ_S + object curvature features"
    )
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_quadratic_atlas_sphere_normal",
    )
    p.add_argument(
        "--structure-dir",
        default="outputs/geometry/physics_quadratic_atlas_structure",
    )
    p.add_argument("--n-charts", type=int, default=6)
    p.add_argument("--local-dim", type=int, default=8)
    p.add_argument("--atlas-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--n-anchors", type=int, default=384)
    p.add_argument("--knn-scales", nargs="+", type=int, default=[512, 1024, 2048])
    p.add_argument("--n-feature-bootstrap", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = ConfirmatoryConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        structure_dir=args.structure_dir,
        n_charts=args.n_charts,
        local_dim=args.local_dim,
        atlas_seeds=list(args.atlas_seeds),
        n_bootstrap=args.n_bootstrap,
        n_null=args.n_null,
        n_anchors=args.n_anchors,
        knn_scales=list(args.knn_scales),
        n_feature_bootstrap=args.n_feature_bootstrap,
        seed=args.seed,
        force=args.force,
    )
    run_confirmatory_object(cfg)


if __name__ == "__main__":
    main()
