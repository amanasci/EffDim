#!/usr/bin/env python3
"""Dispatcher for curvature handoff CLIs. PYTHONPATH must include experiments/."""

from __future__ import annotations

import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.curvature.cli import main

if __name__ == "__main__":
    main()
