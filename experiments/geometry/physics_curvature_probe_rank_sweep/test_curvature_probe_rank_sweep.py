"""Unit tests for the curvature–probe rank sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.effdim_curvature_metrics import metric_scalars
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_curvature_probe_rank_sweep.classify import DEFAULT_THRESHOLDS, primary_label
from geometry.physics_curvature_probe_rank_sweep.inference import (
    associate,
    crossing_d,
    freedman_lane_y,
    permutation_curves,
)
from geometry.physics_curvature_probe_rank_sweep.pipeline import (
    PRESERVED,
    SOURCE_NDC,
    SOURCE_QPD,
    _done,
    assert_not_preserved,
    hash_select,
    kh_trace_identity,
)
from geometry.physics_curvature_probe_rank_sweep.synthetics import make_wide


def test_trace_matches_metric_scalars():
    rng = np.random.default_rng(0)
    d, D = 6, 20
    B = rng.normal(size=(D, d, d))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (B[:, a, b] + B[:, b, a]))
    flat = np.stack(cols, axis=1)
    assert abs(kh_trace_identity(flat, d) - metric_scalars(flat, d)["K_H"]) < 1e-10


def test_hash_select_is_probe_blind_and_deterministic():
    a = hash_select(list(range(200)), 128, seed=0)
    b = hash_select(list(range(200)), 128, seed=0)
    c = hash_select(list(range(200)), 128, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 128


def test_crossing_not_reached():
    ds = np.arange(8, 21)
    r2 = np.linspace(0.5, 0.84, len(ds))
    assert crossing_d(ds, r2, 0.85) == "not_reached"
    r2[-1] = 0.86
    assert crossing_d(ds, r2, 0.85) == 20


def test_permutation_null_not_always_12():
    ds = list(range(12, 21))
    wide = make_wide(180, ds, seed=3, kind="planted16")
    perm = permutation_curves(wide, ds, ycol="local_r2", x_prefix="KH", n_perm=80, seed=4, controlled=False)
    peak = max(ds, key=lambda d: abs(perm["obs"][d]["raw"]))
    assert peak == 16
    null = permutation_curves(wide, ds, ycol="local_r2", x_prefix="KH", n_perm=80, seed=5, controlled=True)
    # planted is not a confounder; controlled should still see d=16
    assert abs(null["obs"][16]["controlled"]) > abs(null["obs"][12]["controlled"])


def test_confound_removed_by_controls():
    ds = list(range(12, 21))
    wide = make_wide(200, ds, seed=6, kind="confound")
    raw = associate(wide.KH16.to_numpy(), wide.local_r2.to_numpy(), None)
    from geometry.physics_curvature_probe_rank_sweep.inference import control_matrix

    ctl = associate(wide.KH16.to_numpy(), wide.local_r2.to_numpy(), control_matrix(wide))
    assert abs(raw["raw"]) > 0.3
    assert abs(ctl["controlled"]) < abs(raw["raw"])


def test_freedman_lane_preserves_length():
    rng = np.random.default_rng(7)
    y = rng.normal(size=40)
    Z = rng.normal(size=(40, 3))
    y2 = freedman_lane_y(y, Z, rng)
    assert y2.shape == y.shape


def test_labels_require_fwer():
    lab = primary_label(
        fwer_hits=[],
        reliable={d: True for d in range(12, 21)},
        tracks_rel=False,
        scale_stable=True,
        missing_ok=True,
        thr=DEFAULT_THRESHOLDS,
    )
    assert lab == "curvature_probe_association_not_familywise_supported"
    lab2 = primary_label(
        fwer_hits=[16, 17, 18],
        reliable={d: True for d in range(12, 21)},
        tracks_rel=False,
        scale_stable=True,
        missing_ok=True,
        thr=DEFAULT_THRESHOLDS,
    )
    assert lab2 == "curvature_probe_association_rank_robust"


def test_preserved_guard(tmp_path):
    p = tmp_path / "m.json"
    assert _done(p, False) is False
    p.write_text("{}")
    assert _done(p, False) is True
    root = platonic_root()
    dest = resolve_path(root, SOURCE_NDC)
    try:
        assert_not_preserved(dest, root)
        raise AssertionError("should refuse")
    except RuntimeError:
        pass
    names = " ".join(PRESERVED)
    assert "physics_nested_dimension_curvature" in names
    assert "physics_quadratic_predictive_dimension" in names
    assert "physics_implicit_normal_inverse" in names
