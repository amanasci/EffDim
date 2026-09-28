# Removed from notebooks/pu_manifold/tests/test_curvature_probe.py in the paper-closure
# Stage 2: it imports pu_manifold.curvature (archived stubs). Verbatim.
def test_phase3_curvature_stubs_remain_unimplemented():
    """The executable form of OQ-1's resolution -- phase 02.5 builds parallel machinery in
    curvature_probe.py (and chart_curvature.py) and does NOT deliver Phase 3's
    CURV-01..04 ahead of schedule. This test is NOT coverage of curvature.py; it only
    pins that its four stubs still raise NotImplementedError, unedited by this phase."""
    from pu_manifold import curvature as curv

    with pytest.raises(NotImplementedError):
        curv.first_fundamental_form(None)
    with pytest.raises(NotImplementedError):
        curv.second_fundamental_form(None, None)
    with pytest.raises(NotImplementedError):
        curv.mean_curvature_vector(None, None)
    with pytest.raises(NotImplementedError):
        curv.metric_condition_number(None)
