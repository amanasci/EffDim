#!/usr/bin/env python3
"""CLI: known-answer pointwise vs finite-patch curvature fixture audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.known_curvature_point_patch_fixture_audit.config import (  # noqa: E402
    OUT_REL,
    ExpConfig,
)
from geometry.known_curvature_point_patch_fixture_audit.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_REL)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--factorial-smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stage", default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--n-points", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--skip-n86471", action="store_true")
    p.add_argument("--skip-suite-f", action="store_true")
    p.add_argument("--bounded", action="store_true", help="Suites A–E on n=16384 only; no 86k, no Suite F")
    p.add_argument("--n-workers", type=int, default=12)
    p.add_argument("--suites", nargs="*", default=None)
    args = p.parse_args(argv)
    out = args.output_dir
    if args.smoke and args.output_dir == OUT_REL:
        out = OUT_REL + "/smoke"
    elif args.factorial_smoke and args.output_dir == OUT_REL:
        out = OUT_REL + "/factorial_smoke"
    run(
        ExpConfig(
            output_dir=out,
            smoke=args.smoke,
            factorial_smoke=args.factorial_smoke,
            force=args.force,
            stage=args.stage,
            device=args.device,
            n_anchors_override=args.n_anchors,
            n_points_override=args.n_points,
            k_override=args.k,
            epochs_override=args.epochs,
            skip_decoder=args.skip_decoder,
            skip_oracle=args.skip_oracle,
            skip_n86471=args.skip_n86471 or args.bounded or args.stage == "finalize",
            skip_suite_f=args.skip_suite_f or args.bounded or args.stage == "finalize",
            bounded=args.bounded or args.stage == "finalize",
            n_workers=args.n_workers,
            suites=args.suites,
        )
    )


if __name__ == "__main__":
    main()
