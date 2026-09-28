"""
Synthetic-fixture tests for the ``pu_manifold.cae`` module (Phase 02.2 Chart Auto-Encoder,
arXiv:1912.10094).

No HuggingFace access, no frozen cache reads -- torch is required (this module needs it),
unlike its sibling ``test_geometry_probes.py``. Not collected by the core `effdim` test
suite (``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run
explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_cae.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pu_manifold import cae as c


# --- Plan 02.2-03 Task 3: matched-capacity baseline trainers for the CAE-03 gate --------


def test_plain_autoencoder_matches_eq22_shape():
    model = c.PlainAutoEncoder(768, 20, hidden=(250, 250, 250), activation="silu")
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 8
    assert model.encoder[-1].out_features == 20
