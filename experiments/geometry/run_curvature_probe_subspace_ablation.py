#!/usr/bin/env python3
"""CLI: confirmatory curvature-subspace probe ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.curvature_probe_subspace_ablation import (  # noqa: E402
    AblationConfig,
    run_ablation,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Curvature-subspace probe ablation")
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_curvature_probe_subspace_ablation",
    )
    p.add_argument("--n-random-controls", type=int, default=50)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-disjoint-anchors", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = AblationConfig(
        output_dir=args.output_dir,
        n_random_controls=args.n_random_controls,
        n_bootstrap=args.n_bootstrap,
        n_disjoint_anchors=args.n_disjoint_anchors,
        seed=args.seed,
        force=args.force,
    )
    run_ablation(cfg)


if __name__ == "__main__":
    main()
