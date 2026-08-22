#!/usr/bin/env python3
"""CLI: staged falsification of global-probe curvature associations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.global_probe_curvature_falsification import (  # noqa: E402
    FalsificationConfig,
    run_falsification,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_global_probe_curvature_falsification",
    )
    p.add_argument("--smoke-anchors", type=int, default=96)
    p.add_argument("--n-null", type=int, default=40)
    p.add_argument("--n-permute", type=int, default=300)
    p.add_argument("--stages", default="all", help="all or comma list e.g. 1,2,3")
    p.add_argument("--no-confirm-full", action="store_true", help="skip 384-anchor dim confirm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = FalsificationConfig(
        output_dir=args.output_dir,
        smoke_anchors=args.smoke_anchors,
        n_null=args.n_null,
        n_permute=args.n_permute,
        stages=args.stages,
        confirm_full=not args.no_confirm_full,
        seed=args.seed,
        force=args.force,
    )
    run_falsification(cfg)


if __name__ == "__main__":
    main()
