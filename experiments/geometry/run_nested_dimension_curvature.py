#!/usr/bin/env python3
"""CLI: nested-dimension curvature diagnostic (ViT-B mag_r d=12 vs d=16)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.nested_dimension_curvature import (  # noqa: E402
    NestedDimConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_nested_dimension_curvature",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=36000.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-replication", action="store_true")
    args = p.parse_args(argv)
    cfg = NestedDimConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
        skip_replication=args.skip_replication,
    )
    run(cfg)


if __name__ == "__main__":
    main()
