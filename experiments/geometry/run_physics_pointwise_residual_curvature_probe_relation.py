#!/usr/bin/env python3
"""CLI: bounded D-residual vs frozen ViT-B G/P probes (75 min, 3 decoders)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_pointwise_residual_curvature_probe_relation.config import OUT_REL, WALL_S, ExpConfig
from geometry.physics_pointwise_residual_curvature_probe_relation.pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hessian-device", default="cpu")
    p.add_argument("--wall-s", type=float, default=WALL_S)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    cfg = ExpConfig(
        output_dir=args.output_dir,
        device=args.device,
        hessian_device=args.hessian_device,
        wall_s=float(args.wall_s),
        smoke=bool(args.smoke),
    )
    run(cfg)


if __name__ == "__main__":
    main()
