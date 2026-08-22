#!/usr/bin/env python3
"""CLI: curvature–probe raw/conditional association + subspace alignment follow-up."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.curvature_probe_alignment import (  # noqa: E402
    AlignmentConfig,
    run_alignment,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Curvature–probe alignment follow-up")
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_curvature_probe_alignment",
    )
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permute", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = AlignmentConfig(
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        n_permute=args.n_permute,
        seed=args.seed,
        force=args.force,
    )
    run_alignment(cfg)


if __name__ == "__main__":
    main()
