#!/usr/bin/env python3
"""CLI: audit frozen quadratic-label chart alignment (read-only original trees)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_quadratic_label_chart_alignment_audit.config import AuditConfig  # noqa: E402
from geometry.physics_quadratic_label_chart_alignment_audit.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_quadratic_label_chart_alignment_audit",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-anchors", type=int, default=None)
    p.add_argument("--skip-truncated", action="store_true")
    args = p.parse_args(argv)
    run(
        AuditConfig(
            output_dir=args.output_dir,
            smoke=args.smoke,
            force=args.force,
            seed=args.seed,
            n_anchors_override=args.n_anchors,
            skip_truncated=args.skip_truncated,
        )
    )


if __name__ == "__main__":
    main()
