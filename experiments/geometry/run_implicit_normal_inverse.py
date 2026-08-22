#!/usr/bin/env python3
"""CLI: implicit normal-space inverse for the physics activation point cloud.

ViT-B only. Does not write into completed geometry directories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_implicit_normal_inverse.pipeline import (  # noqa: E402
    ImplicitNormalConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_implicit_normal_inverse",
    )
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=36000.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--model", default="vit_base")
    args = p.parse_args(argv)
    cfg = ImplicitNormalConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
        smoke=args.smoke,
        n_anchors=args.n_anchors,
        model=args.model,
    )
    run(cfg)


if __name__ == "__main__":
    main()
