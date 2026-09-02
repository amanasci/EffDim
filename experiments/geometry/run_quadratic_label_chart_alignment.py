#!/usr/bin/env python3
"""CLI: quadratic label structure vs fitted activation-chart curvature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_quadratic_label_chart_alignment.config import ExpConfig  # noqa: E402
from geometry.physics_quadratic_label_chart_alignment.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_quadratic_label_chart_alignment")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-anchors", type=int, default=None)
    args = p.parse_args(argv)
    run(
        ExpConfig(
            output_dir=args.output_dir,
            smoke=args.smoke,
            force=args.force,
            seed=args.seed,
            n_anchors_override=args.n_anchors,
        )
    )


if __name__ == "__main__":
    main()
