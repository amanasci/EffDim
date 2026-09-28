"""Attributes the paper runners overwrite at runtime must survive trimming."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402


def test_monkeypatch_targets_exist():
    for mod, name in ((pl, "load_physics_embeddings"), (pl, "load_label_table"),
                      (pcp, "ALPHA_RIDGE"), (pcp, "TORCH_INIT_SEED"), (pcp, "AE_HIDDEN"),
                      (pl, "LABEL_REPO"), (pl, "LABEL_REVISION"), (pl, "LABEL_SPLIT"),
                      (pl, "LABEL_N_SHARDS")):
        assert hasattr(mod, name), f"{mod.__name__}.{name}"
