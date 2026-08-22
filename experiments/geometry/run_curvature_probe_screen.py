#!/usr/bin/env python3
"""CLI: screening association between frozen curvature and local mag_r_desi probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.curvature_probe_screen import (  # noqa: E402
    ScreenConfig,
    run_screen,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Curvature–probe screening analysis")
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_curvature_probe_screen",
    )
    p.add_argument(
        "--curvature-path",
        default=(
            "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
            "object_curvature_features_aggregated.parquet"
        ),
    )
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = ScreenConfig(
        output_dir=args.output_dir,
        curvature_path=args.curvature_path,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        force=args.force,
    )
    run_screen(cfg)


if __name__ == "__main__":
    main()
