#!/usr/bin/env python3
"""CLI: cross-model task-aligned curvature (90 min, no new autoencoders)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_cross_model_task_aligned_curvature.config import OUT_REL, WALL_S, ExpConfig
from geometry.physics_cross_model_task_aligned_curvature.pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hessian-device", default="cpu")
    p.add_argument("--wall-s", type=float, default=WALL_S)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--models", default="")
    args = p.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    run(
        ExpConfig(
            output_dir=args.output_dir,
            device=args.device,
            hessian_device=args.hessian_device,
            wall_s=float(args.wall_s),
            smoke=bool(args.smoke),
            skip_decoder=bool(args.skip_decoder),
            models_override=models,
        )
    )


if __name__ == "__main__":
    main()
