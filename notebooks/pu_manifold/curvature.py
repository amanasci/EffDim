"""
Phase 3 (CURV-01..08) mean-curvature helpers, stubbed here per D-02 so the eventual
four-module ``pu_manifold`` package shape is visible from the start.

These functions will operate on the trained decoder ``f: R^d -> R^768`` that Phase 3
fits (a torch MLP with a C2-smooth activation), using ``torch.func``'s ``jacrev``/
``hessian`` to get the Jacobian and Hessian of ``f`` at a batch of Isomap coordinates.
The reportable scalar this module ultimately produces is ``||H||``, the norm of the
**mean curvature vector** -- never Gaussian curvature and never the principal
curvatures, both of which are category errors at codimension ``768 - d`` (a
2-dimensional intrinsic manifold embedded in 768-dimensional ambient space has no single
scalar "the" curvature the way a surface in R^3 does; only the mean curvature vector and
the second fundamental form are well-defined at this codimension).

Because Phase 1 L2-normalizes both embedding columns (D-05), the ambient space these
decoders reconstruct into is the unit sphere S^767, which is itself intrinsically
curved. This is the reason this stub exists in Phase 1 rather than being created fresh
in Phase 3: Phase 3's synthetic-control falsification manifolds (CURV-06) must be
matched **on the sphere**, not against a flat plane in R^768, or the control compares
the fitted decoder's curvature against the wrong baseline and CURV-07's
data-vs-decoder-artifact question becomes unanswerable. This note must not be lost
between phases.

None of these functions import torch at module level -- Phase 3's torch dependency must
not become an import-time requirement for notebook 01 or for this package's Phase 1
test suite, both of which run with only numpy and joblib present.
"""

from typing import Any


def first_fundamental_form(jacobian: Any) -> Any:
    """
    Compute the first fundamental form (pullback metric) from the decoder Jacobian.

    Parameters:
    -----------
    jacobian : Any
        The decoder's ambient-space Jacobian, shape ``(768, d)``, at one or more
        Isomap coordinates (Phase 3 will type this as a ``torch.Tensor``).

    Returns:
    --------
    Any
        ``g = J^T J``, the ``(d, d)`` symmetric metric tensor induced on the manifold
        by the ambient embedding.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 3 (CURV-01). A reader can compute the first
        fundamental form from the decoder Jacobian via ``torch.func`` autodiff,
        batched over all points rather than looped.
    """
    raise NotImplementedError("Implemented in Phase 3 (CURV-01)")


def second_fundamental_form(jacobian: Any, hessian: Any) -> Any:
    """
    Compute the second fundamental form as the normal-projected ambient Hessian.

    Parameters:
    -----------
    jacobian : Any
        The decoder's ambient-space Jacobian, shape ``(768, d)``.
    hessian : Any
        The decoder's ambient-space Hessian, shape ``(768, d, d)``.

    Returns:
    --------
    Any
        ``(I - J g^-1 J^T)`` applied to the ``(768, d, d)`` Hessian, projecting each
        ambient second-derivative slice onto the normal space of the manifold at that
        point.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 3 (CURV-02). A reader can compute the second
        fundamental form as the normal-projected ambient Hessian of the decoder.
    """
    raise NotImplementedError("Implemented in Phase 3 (CURV-02)")


def mean_curvature_vector(jacobian: Any, hessian: Any) -> Any:
    """
    Compute the mean curvature vector field and its norm.

    Parameters:
    -----------
    jacobian : Any
        The decoder's ambient-space Jacobian, shape ``(768, d)``.
    hessian : Any
        The decoder's ambient-space Hessian, shape ``(768, d, d)``.

    Returns:
    --------
    Any
        The ``g``-trace of the second fundamental form: a ``(768,)`` vector lying in
        the normal space of the manifold at that point, plus its norm ``||H||``. The
        output must be labelled as a vector norm and never as Gaussian or principal
        curvature (CURV-03) -- both are category errors at codimension 768-d.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 3 (CURV-03). A reader gets the mean curvature
        vector field and its norm, with the output labelled as a vector norm and
        never as Gaussian or principal curvature.
    """
    raise NotImplementedError("Implemented in Phase 3 (CURV-03)")


def metric_condition_number(jacobian: Any) -> Any:
    """
    Compute the conditioning of the pullback metric tensor, per point.

    Parameters:
    -----------
    jacobian : Any
        The decoder's ambient-space Jacobian, shape ``(768, d)``, at one or more
        Isomap coordinates.

    Returns:
    --------
    Any
        ``cond(g)`` per point, where ``g = J^T J`` is the first fundamental form, so
        near-singular non-immersion points (where the decoder fails to be locally
        injective) can be flagged rather than silently corrupting the curvature field.

    Raises:
    -------
    NotImplementedError
        Always -- implemented in Phase 3 (CURV-04). A reader can see the conditioning
        of the metric tensor, with near-singular points flagged, so a non-immersion
        point cannot silently corrupt the field.
    """
    raise NotImplementedError("Implemented in Phase 3 (CURV-04)")
