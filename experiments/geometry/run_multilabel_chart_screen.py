#!/usr/bin/env python3
"""CLI: frozen-chart curvature / quadratic screen across eligible physics labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_multilabel_chart_screen.config import ScreenConfig  # noqa: E402
from geometry.physics_multilabel_chart_screen.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_multilabel_chart_screen")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-quadratic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--labels", default=None, help="comma-separated subset of eligible fields")
    args = p.parse_args(argv)
    labels = None
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    cfg = ScreenConfig(
        output_dir=args.output_dir,
        smoke=args.smoke,
        force=args.force,
        seed=args.seed,
        n_anchors_override=args.n_anchors,
        skip_quadratic=args.skip_quadratic,
    )
    if labels is not None:
        cfg.labels = labels
    run(cfg)


if __name__ == "__main__":
    main()
