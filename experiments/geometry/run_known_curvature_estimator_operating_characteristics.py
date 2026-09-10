#!/usr/bin/env python3
"""CLI: bounded D-full / Q operating-characteristics audit (45 min, max 8 AEs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.known_curvature_estimator_operating_characteristics.config import OUT_REL, WALL_S, ExpConfig
from geometry.known_curvature_estimator_operating_characteristics.pipeline import run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--q-device", default="cpu")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--wall-s", type=float, default=WALL_S)
    p.add_argument("--skip-repeats", action="store_true")
    p.add_argument("--skip-oracles", action="store_true")
    args = p.parse_args(argv)
    cfg = ExpConfig(
        output_dir=args.output_dir,
        device=args.device,
        q_device=args.q_device,
        n_workers=int(args.n_workers),
        wall_s=float(args.wall_s),
        skip_repeats=bool(args.skip_repeats),
        skip_oracles=bool(args.skip_oracles),
    )
    run(cfg)


if __name__ == "__main__":
    main()
