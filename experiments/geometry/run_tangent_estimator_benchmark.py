#!/usr/bin/env python3
"""CLI: ground-truth tangent-estimator benchmark + frozen Gauss-map Physics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.tangent_estimator_benchmark import (  # noqa: E402
    BenchmarkConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_tangent_estimator_benchmark",
    )
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument("--sae-dir", default="outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0")
    p.add_argument("--stage", default="all", help="Comma stages or 'all'")
    p.add_argument("--dims", default="8,12,16")
    p.add_argument("--primary-d", type=int, default=16)
    p.add_argument("--k-tier1", default="128,256,512,1024")
    p.add_argument("--k-tier2", default="2048")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-seconds", type=float, default=14400.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-synth-train", type=int, default=4096)
    p.add_argument("--n-synth-anchors", type=int, default=48)
    args = p.parse_args(argv)
    cfg = BenchmarkConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        sae_dir=args.sae_dir,
        stage=args.stage,
        dims=[int(x) for x in args.dims.split(",") if x.strip()],
        primary_d=args.primary_d,
        k_tier1=[int(x) for x in args.k_tier1.split(",") if x.strip()],
        k_tier2=[int(x) for x in args.k_tier2.split(",") if x.strip()],
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
        max_seconds=args.max_seconds,
        force=args.force,
        n_synth_train=args.n_synth_train,
        n_synth_anchors=args.n_synth_anchors,
    )
    run(cfg)


if __name__ == "__main__":
    main()
