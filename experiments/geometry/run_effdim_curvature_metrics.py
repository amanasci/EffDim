#!/usr/bin/env python3
"""CLI: graph-effective-dimension curvature metrics (Smith42/Physics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.effdim_curvature_metrics import (  # noqa: E402
    EffDimCurvatureConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_effdim_curvature_metrics",
    )
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=36000.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-gauss", action="store_true", default=True)
    args = p.parse_args(argv)
    cfg = EffDimCurvatureConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
        skip_gauss=args.skip_gauss,
    )
    run(cfg)


if __name__ == "__main__":
    main()
