"""
Known-answer and boundary tests for ``pu_manifold.linear_probe`` (Phase 5's probe fit/score,
seed pooling, bucketing and verdict functions).

No HuggingFace access, no CAE checkpoint, no read of ``notebooks/.cache/``, and no PU
embedding -- every fixture is generated in-test from a fixed ``np.random.default_rng`` seed.
Not collected by the core `effdim` test suite (``pyproject.toml``'s ``testpaths = ["tests"]``
excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q

This is a permitted new test file: ``05-VALIDATION.md``'s Wave 0 Requirements name it
explicitly and ``05-RESEARCH.md``'s Validation Architecture lists it as the sole test-file gap
for this phase.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from pu_manifold import linear_probe as lp


# --- Test 4: convention agreement, frozen at the 05-04 freeze ---------------------------------


def test_curvature_convention_matches_sealed_modules():
    """D5-06: linear_probe.CURVATURE_CONVENTION equals chart_curvature.CURVATURE_CONVENTION
    equals curvature_probe.CURVATURE_CONVENTION equals the string "trace". This test was
    written RED before the 05-04 freeze; the freeze tripwire (the xfail marker that formerly
    decorated this test) was removed by 05-04 Task 2 once the constant was set, and the test
    now passes for real."""
    from pu_manifold import chart_curvature, curvature_probe

    assert (
        lp.CURVATURE_CONVENTION
        == chart_curvature.CURVATURE_CONVENTION
        == curvature_probe.CURVATURE_CONVENTION
        == "trace"
    )
