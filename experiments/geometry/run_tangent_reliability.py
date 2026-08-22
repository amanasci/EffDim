#!/usr/bin/env python3
"""CLI: tangent-reliability falsification for Physics mean-curvature result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.tangent_reliability import (  # noqa: E402
    TangentReliabilityConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_tangent_reliability")
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--dims", default="8,12,16")
    p.add_argument("--primary-d", type=int, default=16)
    p.add_argument("--k-fit", default="512,1024,2048,3072")
    p.add_argument("--k-fit-curvature", default="1024,2048,3072")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-n-anchors", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=14400.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = TangentReliabilityConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        stage=args.stage,
        dims=[int(x) for x in args.dims.split(",") if x.strip()],
        primary_d=args.primary_d,
        k_fit=[int(x) for x in args.k_fit.split(",") if x.strip()],
        k_fit_curvature=[int(x) for x in args.k_fit_curvature.split(",") if x.strip()],
        device=args.device,
        seed=args.seed,
        smoke_n_anchors=args.smoke_n_anchors,
        max_seconds=args.max_seconds,
        force=args.force,
    )
    run(cfg)


if __name__ == "__main__":
    main()
