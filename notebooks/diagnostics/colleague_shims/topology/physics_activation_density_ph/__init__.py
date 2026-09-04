"""Placeholder for `topology.physics_activation_density_ph`, a sibling package the colleague's
`origin/curvature-experiments` branch (commit 97efb2eb) imports at module level but does not
contain on any ref. His branch is not self-contained; the ten names below are stubbed here so
his estimator modules import UNCHANGED, and every stub RAISES if called -- none of them is on
the `nested_pca_frame` + `_fit_rank` path `09_colleague_estimator_run.py` uses (verified by
running that path against these raising stubs). Nothing here computes anything."""

_MESSAGE = "shim: absent dependency of origin/curvature-experiments -- {name} is a placeholder and must never be called on the K_H^cross estimator path"


def _placeholder(name: str):
    def _stub(*args, **kwargs):
        raise NotImplementedError(_MESSAGE.format(name=name))

    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub
