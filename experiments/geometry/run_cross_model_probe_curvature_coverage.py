#!/usr/bin/env python3
"""CLI: Smith42 Physics cross-model × all-probe curvature coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.cross_model_probe_curvature_coverage import (  # noqa: E402
    CoverageConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_cross_model_probe_curvature_coverage",
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
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated subset (default: all physics models)",
    )
    p.add_argument(
        "--targets",
        default="",
        help="Comma-separated subset (default: all label keys incl. sfr)",
    )
    args = p.parse_args(argv)
    cfg = CoverageConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
    )
    if args.models.strip():
        cfg.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.targets.strip():
        cfg.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    run(cfg)


if __name__ == "__main__":
    main()
