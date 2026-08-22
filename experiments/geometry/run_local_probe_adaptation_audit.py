#!/usr/bin/env python3
"""CLI: final audit of local-probe-adaptation result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_local_probe_adaptation_audit.config import AuditConfig  # noqa: E402
from geometry.physics_local_probe_adaptation_audit.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/geometry/physics_local_probe_adaptation_audit")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-shuffle", action="store_true")
    args = p.parse_args(argv)
    run(
        AuditConfig(
            output_dir=args.output_dir,
            smoke=args.smoke,
            force=args.force,
            skip_shuffle=args.skip_shuffle,
        )
    )


if __name__ == "__main__":
    main()
