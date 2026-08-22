#!/usr/bin/env python3
"""CLI: adaptive per-dataset curvature–physics-label probe.

Does not write into completed geometry directories.
Does not run cross-model encoder replication.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_adaptive_dataset_curvature_probe.pipeline import AdaptiveProbeConfig  # noqa: E402
from geometry.physics_adaptive_dataset_curvature_probe.stages import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_adaptive_dataset_curvature_probe")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=36000.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--skip-desi", action="store_true")
    args = p.parse_args(argv)
    cfg = AdaptiveProbeConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
        device=args.device,
        smoke=args.smoke,
        n_anchors=args.n_anchors,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        skip_desi=args.skip_desi,
    )
    run(cfg)


if __name__ == "__main__":
    main()
