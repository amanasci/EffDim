"""Pytest wrapper for the reconciliation unit/synthetic suite."""

from .tests_unit import run_unit_tests


def test_all_unit_synthetic():
    result = run_unit_tests()
    failed = [r["name"] for r in result["rows"] if not r["ok"]]
    assert result["ok"], failed
