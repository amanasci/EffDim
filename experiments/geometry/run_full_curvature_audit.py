#!/usr/bin/env python3
"""CLI: full curvature audit (reuses completed 512×10 split-half artifacts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.full_curvature_audit import (  # noqa: E402
    FullCurvatureAuditConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_full_curvature_audit",
    )
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument(
        "--split-half-dir",
        default="outputs/geometry/physics_split_half_curvature_reliability",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=36000.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-dinov3", action="store_true")
    args = p.parse_args(argv)
    cfg = FullCurvatureAuditConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        split_half_dir=args.split_half_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
        skip_dinov3=args.skip_dinov3,
    )
    run(cfg)


if __name__ == "__main__":
    main()
