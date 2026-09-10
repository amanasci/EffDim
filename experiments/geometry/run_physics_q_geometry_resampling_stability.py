#!/usr/bin/env python3
"""CLI: ViT-B Q geometry-resampling stability (45 min, no decoder/probe refits)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_q_geometry_resampling_stability.config import OUT_REL, WALL_S, ExpConfig
from geometry.physics_q_geometry_resampling_stability.pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-workers", type=int, default=12)
    p.add_argument("--wall-s", type=float, default=WALL_S)
    p.add_argument("--force-reps", type=int, default=None)
    args = p.parse_args(argv)
    cfg = ExpConfig(
        output_dir=args.output_dir,
        device=args.device,
        n_workers=int(args.n_workers),
        wall_s=float(args.wall_s),
        force_reps=args.force_reps,
    )
    run(cfg)


if __name__ == "__main__":
    main()
