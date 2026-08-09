#!/usr/bin/env python3
"""Rebuild results.md from a finished results.json.

The curvature run is the expensive part; the markdown is not. This lets the
report be reworded or re-tabulated without recomputing anything.

    python render_report.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from density_curvature_probe import build_markdown, parse_args  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="directory containing results.json")
    ns = ap.parse_args()

    run_dir = Path(ns.run_dir)
    payload = json.loads((run_dir / "results.json").read_text())

    # Rehydrate the Namespace the report formatter expects from the recorded args.
    args = parse_args([])
    for k, v in payload.get("args", {}).items():
        setattr(args, k, v)

    (run_dir / "results.md").write_text(build_markdown(payload, args))
    print(f"Rewrote {run_dir / 'results.md'}")


if __name__ == "__main__":
    main()
