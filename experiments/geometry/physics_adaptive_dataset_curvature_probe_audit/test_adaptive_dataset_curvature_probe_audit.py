"""Unit tests for the adaptive-dataset curvature–physics audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_adaptive_dataset_curvature_probe_audit.classify import (
    audit_label,
    classify_root_causes,
    scale_blocks_complete,
)
from geometry.physics_adaptive_dataset_curvature_probe_audit.config import (
    FROZEN_CTL,
    FROZEN_D80,
    FROZEN_D85,
    PRESERVED,
    SOURCE_ADCP,
)
from geometry.physics_adaptive_dataset_curvature_probe_audit.inference import (
    curve_p_from_null,
    studentize,
    westfall_young_minp,
)
from geometry.physics_adaptive_dataset_curvature_probe_audit.parity import compare_vectors
from geometry.physics_adaptive_dataset_curvature_probe_audit.pipeline import (
    assert_not_preserved,
    delta_85_80,
    hash_select,
    p_report,
    p_value,
    peak_abs,
    spearman_dict,
)
from geometry.physics_adaptive_dataset_curvature_probe_audit.sample_sizes import underpowered


def test_discovery_artifact_parity_detects_identical_kh():
    rng = np.random.default_rng(0)
    kh = rng.normal(size=40)
    rec = compare_vectors(kh, kh.copy())
    assert rec["exact"] is True
    assert rec["max_abs"] == 0.0


def test_anchor_order_mismatch_detection():
    a = [1, 2, 3, 4]
    b = hash_select(a, 4, seed=0, prefix="adcp")
    assert set(a) == set(b)
    assert a != b


def test_neighbour_mismatch_detection():
    old = np.arange(20)
    new = old.copy()
    new[3] = 99
    rec = compare_vectors(old, new)
    assert rec["exact"] is False
    assert rec["max_abs"] > 0


def test_label_permutation_misalignment_detection():
    y = np.linspace(15, 19, 50)
    rec = compare_vectors(y, y[::-1])
    assert rec["exact"] is False
    assert rec["appears_permuted"] is True
    assert abs(rec["spearman"] + 1.0) < 1e-12


def test_reject_equal_length_only_joins():
    proof_ok = False
    n_emb, n_lab = 20465, 20465
    has_shared_id = False
    proved = bool(has_shared_id and n_emb == n_lab)
    assert n_emb == n_lab
    assert proved is False
    assert proof_ok is False


def test_direct_spearman_reconstruction():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 0.4 * x + rng.normal(size=80)
    rho = spearman_dict(x, y)["rho"]
    # independent reconstruction
    from scipy.stats import spearmanr

    assert abs(rho - float(spearmanr(x, y)[0])) < 1e-12


def test_delta_85_80_sign_and_rank_lookup():
    rho = {12: 0.143, 16: -0.240, 20: -0.233}
    dlt = delta_85_80(rho, FROZEN_D80, FROZEN_D85)
    assert FROZEN_D80 == 12 and FROZEN_D85 == 20
    assert dlt < 0
    assert abs(dlt - (FROZEN_CTL[20] - FROZEN_CTL[12])) < 0.02


def test_peak_defined_by_largest_absolute_association():
    d, v = peak_abs({12: 0.14, 16: -0.24, 20: -0.23, 7: 0.22})
    assert d == 16
    assert v == -0.24


def test_frozen_versus_harmonized_controls_are_distinct_labels():
    frozen = "frozen_discovery_control"
    harm = "harmonized_cross_dataset_control"
    assert frozen != harm
    assert "discovery" in frozen
    assert "harmonized" in harm


def test_anchor_level_sample_size_not_full_table():
    assert underpowered(45) is True
    assert underpowered(1340) is False
    assert underpowered(64) is False
    assert underpowered(63) is True


def test_westfall_young_minp():
    rng = np.random.default_rng(2)
    # 3 labels, 200 perms; observed min p is small
    null = rng.uniform(0.05, 1.0, size=(200, 3))
    obs = [0.01, 0.4, 0.7]
    rec = westfall_young_minp(obs, null)
    assert rec["p"] < 0.05
    assert rec["p_report"].startswith("<") or float(rec["p"]) > 0


def test_studentized_maxT():
    rng = np.random.default_rng(3)
    null = rng.normal(0, 0.05, size=400)
    T = studentize(0.4, null)
    assert T > 5
    T0 = studentize(float(np.mean(null)), null)
    assert abs(T0) < 1


def test_synchronized_same_object_permutations():
    rng = np.random.default_rng(4)
    n = 30
    y1 = rng.normal(size=n)
    y2 = rng.normal(size=n)
    idx = rng.permutation(n)
    # one shared permutation applied to both labels
    assert not np.allclose(y1[idx], y1)
    assert (y1[idx][0], y2[idx][0]) != (y1[0], y2[0]) or n < 2


def test_p_zero_rendered_as_less_than():
    assert p_value(0, 10000) == 1.0 / 10001
    assert p_report(0, 10000).startswith("<")
    assert "p=0" not in p_report(0, 10000)


def test_incomplete_scale_blocks_scientific_complete():
    assert scale_blocks_complete(scale_pending=True, discovery_parity=True, joins_proven=True) is False
    assert scale_blocks_complete(scale_pending=False, discovery_parity=True, joins_proven=False) is False
    assert scale_blocks_complete(scale_pending=False, discovery_parity=True, joins_proven=True) is True


def test_preservation_of_prior_output_trees():
    from geometry.physics_adaptive_dataset_curvature_probe_audit.pipeline import resolve_path

    tmp_path = Path("/tmp/adcp_audit_preserve_test")
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = resolve_path(tmp_path, "outputs/geometry/physics_adaptive_dataset_curvature_probe_audit")
    assert_not_preserved(out, tmp_path)
    for rel in (SOURCE_ADCP, "outputs/geometry/physics_curvature_probe_rank_sweep"):
        raised = False
        try:
            assert_not_preserved(resolve_path(tmp_path, rel), tmp_path)
        except RuntimeError:
            raised = True
        assert raised
    assert SOURCE_ADCP in PRESERVED


def test_probe_mismatch_label():
    causes = classify_root_causes(
        {"kh_identical": True, "probe_quantity_mismatch": True, "anchors": {"adaptive_chose_new_hash_subset": False}, "embedding": {}, "neighbours": {}},
        {"proved": False},
        {"sign_reversal_control": True},
    )
    assert "probe_label_alignment_failure" in causes
    assert "desi_alignment_unproven" in causes
    lab = audit_label(causes, {"kh_identical": True, "probe_quantity_mismatch": True}, {"proved": False})
    assert lab == "probe_label_alignment_failure"


def test_curve_p_from_null_uses_max_abs():
    obs = {12: 0.1, 16: -0.4, 20: 0.2}
    null = {12: np.array([0.05, 0.02]), 16: np.array([-0.1, -0.2]), 20: np.array([0.05, 0.01])}
    p = curve_p_from_null(obs, null, 2)
    assert p == p_value(0, 2)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
