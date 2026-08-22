#!/usr/bin/env python3
"""CLI: SAE tangent benchmark v2 (code-manifold pushforward hypothesis)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.sae_tangent_benchmark import (  # noqa: E402
    SAETangentConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_sae_tangent_benchmark_v2",
    )
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument(
        "--physics-sae-dir",
        default="outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--dims", default="8,12,16")
    p.add_argument("--primary-d", type=int, default=12)
    p.add_argument("--k-list", default="256,512,1024,2048")
    p.add_argument("--n-global", type=int, default=16384)
    p.add_argument("--n-anchors", type=int, default=48)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=14400.0)
    args = p.parse_args(argv)
    cfg = SAETangentConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        physics_sae_dir=args.physics_sae_dir,
        stage=args.stage,
        dims=[int(x) for x in args.dims.split(",") if x.strip()],
        primary_d=args.primary_d,
        k_list=[int(x) for x in args.k_list.split(",") if x.strip()],
        n_global=args.n_global,
        n_anchors=args.n_anchors,
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
        force=args.force,
        max_seconds=args.max_seconds,
    )
    run(cfg)


if __name__ == "__main__":
    main()
