#!/usr/bin/env python3
"""CLI: submission validation for the frozen curvature–probe claim.

Does not write into completed geometry directories.
Does not launch a geometry refit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_curvature_probe_submission_validation.pipeline import ValConfig  # noqa: E402
from geometry.physics_curvature_probe_submission_validation.stages import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_curvature_probe_submission_validation")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    cfg = ValConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        force=args.force,
        smoke=args.smoke,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
    )
    run(cfg)


if __name__ == "__main__":
    main()
