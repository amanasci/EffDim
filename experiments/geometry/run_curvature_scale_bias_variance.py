#!/usr/bin/env python3
"""CLI: separate neighbourhood radius from curvature-fit sample count.

Does not write into preserved geometry or manuscript trees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_curvature_scale_bias_variance.config import ExpConfig  # noqa: E402
from geometry.physics_curvature_scale_bias_variance.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_curvature_scale_bias_variance")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-replicates", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-secondary", action="store_true")
    p.add_argument("--extended-radius", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    cfg = ExpConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        n_replicates=args.n_replicates,
        smoke=args.smoke,
        force=args.force,
        skip_secondary=args.skip_secondary,
        extended_radius=args.extended_radius,
        device=args.device,
    )
    run(cfg)


if __name__ == "__main__":
    main()
