#!/usr/bin/env python3
"""CLI: bounded Q/D failure localization (45 min, no new decoders/oracles)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.known_curvature_instrument_failure_localization.config import (  # noqa: E402
    OUT_REL,
    ExpConfig,
)
from geometry.known_curvature_instrument_failure_localization.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--audit-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--wall-s", type=float, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = ExpConfig(output_dir=args.output_dir, device=args.device, n_workers=args.n_workers, force=args.force)
    if args.audit_dir:
        cfg.audit_dir = args.audit_dir
    if args.wall_s is not None:
        cfg.wall_s = float(args.wall_s)
    run(cfg)


if __name__ == "__main__":
    main()
