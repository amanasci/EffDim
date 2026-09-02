"""Unit tests for the multi-label frozen-chart screen."""

from __future__ import annotations

import numpy as np

from geometry.physics_multilabel_chart_screen.metrics import finite_enough, neighbourhood_metrics
from geometry.physics_multilabel_chart_screen.inventory import (
    assert_not_desi_resurrected,
    eligible_fields,
    exclusion_table,
    record_for,
)


def test_eligible_set_excludes_desi_and_sfr():
    fields = eligible_fields()
    assert fields == ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass")
    excluded = {r["field"] for r in exclusion_table()}
    assert "sfr" in excluded
    assert "desi_spec_z" in excluded
    assert "desi_mag_r" in excluded


def test_record_terminology():
    assert record_for("mag_r_desi")["family"] == "photometric_magnitude"
    assert record_for("photo_z")["family"] == "photometric_redshift"
    assert "physical property" not in record_for("mag_r_desi")["display"]


def test_refuse_unproven_desi():
    try:
        assert_not_desi_resurrected("desi_spec_z")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for desi_spec_z")
    assert_not_desi_resurrected("mag_r_desi")


def test_neighbourhood_metrics_not_catalog():
    y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    yhat = np.array([10.1, 11.2, 11.8, 13.1, 13.7])
    rec = neighbourhood_metrics(y, yhat, np.arange(5))
    assert rec["mse_G"] > 0
    assert rec["r2_G"] < 1
    # R² is a probe-performance number, not the catalog mean.
    assert abs(rec["r2_G"] - float(np.mean(y))) > 1


def test_finite_neighbour_gate():
    y = np.full(80, np.nan)
    y[:30] = 1.0
    assert finite_enough(y, np.arange(80)) is False
    y[:70] = 1.0
    assert finite_enough(y, np.arange(80)) is True


if __name__ == "__main__":
    test_eligible_set_excludes_desi_and_sfr()
    test_record_terminology()
    test_refuse_unproven_desi()
    test_neighbourhood_metrics_not_catalog()
    test_finite_neighbour_gate()
    print("ok")
