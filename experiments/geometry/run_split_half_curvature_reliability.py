#!/usr/bin/env python3
"""CLI: split-half curvature reliability audit (fixed PCA tangent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.split_half_curvature_reliability import (  # noqa: E402
    SplitHalfConfig,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_split_half_curvature_reliability",
    )
    p.add_argument(
        "--multimodel-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument("--stage", default="all", help="parity,smoke,full,analyze or all")
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n-anchors-smoke", type=int, default=128)
    p.add_argument("--n-splits-smoke", type=int, default=5)
    p.add_argument("--n-anchors-full", type=int, default=512)
    p.add_argument("--n-splits-full", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=7200.0)
    p.add_argument("--smoke-only", action="store_true", help="parity+smoke+analyze only")
    args = p.parse_args(argv)
    stage = "parity,smoke,analyze" if args.smoke_only else args.stage
    cfg = SplitHalfConfig(
        output_dir=args.output_dir,
        multimodel_dir=args.multimodel_dir,
        stage=stage,
        d=args.d,
        k=args.k,
        n_anchors_smoke=args.n_anchors_smoke,
        n_splits_smoke=args.n_splits_smoke,
        n_anchors_full=args.n_anchors_full,
        n_splits_full=args.n_splits_full,
        seed=args.seed,
        force=args.force,
        max_seconds=args.max_seconds,
    )
    run(cfg)


if __name__ == "__main__":
    main()
