#!/usr/bin/env python3
"""CLI: quadratic atlas structure analysis (rank, nulls, retrieval links)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m geometry.run_quadratic_atlas_structure` and direct path execution.
_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.quadratic_structure import (  # noqa: E402
    STAGES,
    StructureConfig,
    run_structure,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quadratic atlas structure follow-up")
    p.add_argument("--stage", default="all", choices=["all", *STAGES])
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_quadratic_atlas_structure",
    )
    p.add_argument(
        "--ablation-dir",
        default="outputs/geometry/physics_activation_atlas_geometry_ablation",
    )
    p.add_argument(
        "--retrieval-dir",
        default="outputs/retrieval_information_geometry/smoke",
    )
    p.add_argument("--n-bootstrap", type=int, default=50)
    p.add_argument("--n-null", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--configs",
        nargs="+",
        default=["4,8", "6,8", "8,8", "6,6", "6,10", "6,12"],
        help="n_charts,d pairs",
    )
    args = p.parse_args(argv)
    configs = []
    for tok in args.configs:
        n, d = tok.split(",")
        configs.append((int(n), int(d)))
    cfg = StructureConfig(
        stage=args.stage,
        output_dir=args.output_dir,
        ablation_dir=args.ablation_dir,
        retrieval_dir=args.retrieval_dir,
        n_bootstrap=args.n_bootstrap,
        n_null=args.n_null,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        force=args.force,
        configs=configs,
    )
    run_structure(cfg)


if __name__ == "__main__":
    main()
