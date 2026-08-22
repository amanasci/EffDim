#!/usr/bin/env python3
"""CLI: curvature vs advantage of OOF patch probes over the frozen global probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_local_probe_adaptation.config import ExpConfig  # noqa: E402
from geometry.physics_local_probe_adaptation.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_local_probe_adaptation")
    p.add_argument("--stage", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-tangent", action="store_true")
    p.add_argument("--nested-alpha", action="store_true", help="secondary nested α sensitivity")
    p.add_argument("--skip-shuffle", action="store_true")
    args = p.parse_args(argv)
    cfg = ExpConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        smoke=args.smoke,
        force=args.force,
        skip_tangent=args.skip_tangent,
        skip_nested_alpha=not args.nested_alpha,
        skip_shuffle=args.skip_shuffle,
    )
    run(cfg)


if __name__ == "__main__":
    main()
