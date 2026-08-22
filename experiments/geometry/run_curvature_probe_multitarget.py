#!/usr/bin/env python3
"""CLI: GPU-batched multi-target curvature–probe alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.curvature_probe_multitarget_gpu import (  # noqa: E402
    MultiTargetConfig,
    run_multitarget,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Multi-target curvature probe alignment (GPU)")
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_curvature_probe_multitarget",
    )
    p.add_argument("--batch-anchors", type=int, default=16)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permute", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = MultiTargetConfig(
        output_dir=args.output_dir,
        batch_anchors=args.batch_anchors,
        n_bootstrap=args.n_bootstrap,
        n_permute=args.n_permute,
        device=args.device,
        seed=args.seed,
        force=args.force,
    )
    run_multitarget(cfg)


if __name__ == "__main__":
    main()
