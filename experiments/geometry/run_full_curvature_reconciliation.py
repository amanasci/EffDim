#!/usr/bin/env python3
"""CLI: full sphere-normal curvature reconciliation across five encoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_cross_model_full_curvature_reconciliation.config import ExpConfig  # noqa: E402
from geometry.physics_cross_model_full_curvature_reconciliation.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_cross_model_full_curvature_reconciliation",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-geometry", action="store_true")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args(argv)
    run(
        ExpConfig(
            output_dir=args.output_dir,
            smoke=args.smoke,
            force=args.force,
            skip_geometry=args.skip_geometry,
            stage=args.stage,
            seed=args.seed,
            device=args.device,
            n_anchors_override=args.n_anchors,
            models_override=args.models,
        )
    )


if __name__ == "__main__":
    main()
