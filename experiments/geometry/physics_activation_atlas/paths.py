"""Path helpers — reuse SAE-shared-basis / topology loaders."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parents[1]
_SAE = _EXP / "SAE-shared-basis"
_TOPO = _EXP / "topology" / "physics_activation_density_ph"
for p in (_EXP, _SAE, _TOPO.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from topology.physics_activation_density_ph.paths import (  # noqa: E402
    load_col,
    platonic_root,
    resolve_path,
)

PKG_DIR = _HERE
__all__ = ["load_col", "platonic_root", "resolve_path", "PKG_DIR"]
