#!/usr/bin/env python3
"""CLI: sphere-normal curvature component predictive decomposition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_curvature_component_predictive_decomposition.config import ExpConfig  # noqa: E402
from geometry.physics_curvature_component_predictive_decomposition.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_curvature_component_predictive_decomposition",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args(argv)
    default_out = "outputs/geometry/physics_curvature_component_predictive_decomposition"
    if args.smoke and args.output_dir == default_out:
        args.output_dir = default_out + "/smoke"
    run(
        ExpConfig(
            output_dir=args.output_dir,
            smoke=args.smoke,
            force=args.force,
            skip_probes=args.skip_probes,
            stage=args.stage,
            seed=args.seed,
            n_anchors_override=args.n_anchors,
            models_override=args.models,
            n_workers=args.n_workers,
        )
    )


if __name__ == "__main__":
    main()
