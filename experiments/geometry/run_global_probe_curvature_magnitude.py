#!/usr/bin/env python3
"""CLI: fixed global probe × curvature magnitude/alignment analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.global_probe_curvature_magnitude import (  # noqa: E402
    MagnitudeConfig,
    run_magnitude_analysis,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_global_probe_curvature_magnitude",
    )
    p.add_argument(
        "--align-dir",
        default="outputs/geometry/physics_global_probe_curvature_alignment",
    )
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permute", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = MagnitudeConfig(
        output_dir=args.output_dir,
        align_dir=args.align_dir,
        n_bootstrap=args.n_bootstrap,
        n_permute=args.n_permute,
        seed=args.seed,
        force=args.force,
    )
    run_magnitude_analysis(cfg)


if __name__ == "__main__":
    main()
