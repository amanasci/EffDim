#!/usr/bin/env python3
"""CLI: sphere-normal nested quadratic atlas test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.sphere_normal_quadratic import (  # noqa: E402
    STAGES,
    SphereNormalConfig,
    run_sphere_normal,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sphere-normal quadratic residual test")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_quadratic_atlas_sphere_normal",
    )
    p.add_argument(
        "--structure-dir",
        default="outputs/geometry/physics_quadratic_atlas_structure",
    )
    p.add_argument("--primary", default="n6_d8")
    p.add_argument(
        "--configs",
        nargs="+",
        default=["n6_d8", "n4_d8", "n8_d8", "n6_d6", "n6_d10", "n6_d12"],
    )
    p.add_argument("--n-bootstrap", type=int, default=40)
    p.add_argument("--n-null", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = SphereNormalConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        structure_dir=args.structure_dir,
        primary=args.primary,
        configs=list(args.configs),
        n_bootstrap=args.n_bootstrap,
        n_null=args.n_null,
        seed=args.seed,
        force=args.force,
    )
    run_sphere_normal(cfg)


if __name__ == "__main__":
    main()
