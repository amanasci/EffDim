#!/usr/bin/env python3
"""CLI: bounded colleague decoder-curvature reproduction (60 min, max 4 new decoders)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.pointwise_decoder_curvature_reproduction.config import OUT_REL, WALL_S
from geometry.pointwise_decoder_curvature_reproduction.pipeline import ExpConfig, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--wall-s", type=float, default=WALL_S)
    args = p.parse_args(argv)
    cfg = ExpConfig(output_dir=args.output_dir, device=args.device, wall_s=float(args.wall_s))
    run(cfg)


if __name__ == "__main__":
    main()
