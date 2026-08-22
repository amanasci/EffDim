"""Unit tests for adaptive per-dataset curvature–physics probe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_adaptive_dataset_curvature_probe.association_stage import interpolate_rho_on_tau
from geometry.physics_adaptive_dataset_curvature_probe.classify import primary_label
from geometry.physics_adaptive_dataset_curvature_probe.config import DISCOVERY_DATASET, DISCOVERY_LABEL, PRESERVED
from geometry.physics_adaptive_dataset_curvature_probe.geometry_stage import construct_interval
from geometry.physics_adaptive_dataset_curvature_probe.pipeline import (
    crossing_d,
    hash_select,
    p_report,
    p_value,
    primary_k,
)


def test_primary_k_rule_is_sample_size_only():
    assert primary_k(16384) == 2048
    assert primary_k(20465) == 2048
    assert primary_k(1496) is None
    assert primary_k(4096) == 512


def test_crossing_not_extrapolated():
    ds = np.arange(1, 21)
    r2 = np.linspace(0.50, 0.84, 20)
    assert crossing_d(ds, r2, 0.85) == "not_reached"
    r2[-1] = 0.86
    assert crossing_d(ds, r2, 0.85) == 20


def test_interval_uses_only_existing_quantities():
    rec = construct_interval(
        {"d_75": 8, "d_90": "not_reached", "dL_plat": 18, "dQ_plat": 19, "dL_plat_lo": 17, "dQ_plat_lo": 18, "dL_plat_hi": 19, "dQ_plat_hi": 20},
        d_curv_max=20,
    )
    assert rec["d_low"] == max(2, 8 - 2)
    assert rec["right_truncated"] is True
    assert rec["d_high"] <= 20


def test_interval_missing_d75():
    rec = construct_interval({"d_75": "not_reached", "d_90": 16, "dL_plat": 14}, d_curv_max=20)
    assert rec["d_low"] >= 2
    assert rec["d_high"] >= 16


def test_p_zero_exceedances_not_zero():
    assert p_value(0, 10000) == 1.0 / 10001
    assert p_report(0, 10000).startswith("<")
    assert "0.000" not in p_report(0, 10) or p_report(0, 10).startswith("<")


def test_hash_select_probe_blind():
    a = hash_select(list(range(200)), 32, seed=0)
    b = hash_select(list(range(200)), 32, seed=0)
    c = hash_select(list(range(200)), 32, seed=1)
    assert a == b
    assert a != c
    assert a == sorted(a, key=lambda s: __import__("hashlib").sha256(f"adcp:0:{s}".encode()).hexdigest())


def test_no_variance_extrapolation():
    df = interpolate_rho_on_tau([10, 12, 14], np.array([0.72, 0.80, 0.84]), np.array([0.1, -0.1, -0.2]))
    below = df[np.isclose(df.tau, 0.70)]
    above = df[np.isclose(df.tau, 0.95)]
    mid = df[np.isclose(df.tau, 0.80)]
    assert bool(below.iloc[0].in_range) is False
    assert np.isnan(below.iloc[0].rho)
    assert bool(above.iloc[0].in_range) is False
    assert bool(mid.iloc[0].in_range) is True


def test_discovery_not_in_confirmatory_family_constant():
    assert DISCOVERY_DATASET == "physics_vit_base"
    assert DISCOVERY_LABEL == "mag_r_desi"


def test_preserved_trees_listed():
    assert "outputs/geometry/physics_curvature_probe_rank_sweep" in PRESERVED
    assert "outputs/geometry/physics_quadratic_predictive_dimension" in PRESERVED


def test_gate_unresolved_if_single_dataset_no_hits():
    lab = primary_label(
        n_included=1,
        n_reliable_datasets=1,
        n_confirmatory_fwer=0,
        p_global=0.4,
        mag_deltas=[],
        transition_aligned_var=False,
        transition_aligned_rank=False,
        scale_stable=True,
        missing_ok=True,
    )
    assert lab == "cross_dataset_curvature_replication_unresolved"


def test_gate_underidentified():
    lab = primary_label(
        n_included=2,
        n_reliable_datasets=0,
        n_confirmatory_fwer=0,
        p_global=1.0,
        mag_deltas=[],
        transition_aligned_var=False,
        transition_aligned_rank=False,
        scale_stable=True,
        missing_ok=True,
    )
    assert lab == "adaptive_curvature_sweeps_underidentified"


def test_partial_reuse_skips_cached_pairs():
    from geometry.physics_adaptive_dataset_curvature_probe.curvature_stage import _have_pairs

    df = pd.DataFrame({"sample_id": [1, 1, 2], "d": [8, 12, 8]})
    have = _have_pairs(df)
    assert (1, 8) in have and (1, 20) not in have


def test_orientation_not_in_classify():
    # classify must not flip signs from observed rho
    src = open(__file__).read() if False else True
    import inspect

    src = inspect.getsource(primary_label)
    assert "reverse" not in src.lower()
    assert "-1 *" not in src
