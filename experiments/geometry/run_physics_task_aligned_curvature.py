#!/usr/bin/env python3
"""CLI: bounded ViT-B task-aligned curvature (60 min, no new autoencoders)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_task_aligned_curvature.config import OUT_REL, WALL_S, ExpConfig
from geometry.physics_task_aligned_curvature.pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hessian-device", default="cpu")
    p.add_argument("--wall-s", type=float, default=WALL_S)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    run(
        ExpConfig(
            output_dir=args.output_dir,
            device=args.device,
            hessian_device=args.hessian_device,
            wall_s=float(args.wall_s),
            smoke=bool(args.smoke),
        )
    )


if __name__ == "__main__":
    main()
