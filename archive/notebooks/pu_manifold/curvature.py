"""Phase 3 (CURV-01..08) mean-curvature stubs.

Will operate on the trained decoder f: R^d -> R^768 via torch.func jacrev/hessian.
The reportable scalar is ||H||, the norm of the mean curvature *vector* — never
Gaussian or principal curvature, which are category errors at codimension 768-d.

Carried note (must not be lost between phases): Phase 1 L2-normalizes both columns
(D-05), so the ambient space is the unit sphere S^767 — CURV-06's synthetic controls
must be matched on the sphere, not against a flat plane in R^768, or CURV-07's
data-vs-artifact question becomes unanswerable.

No module-level torch import: this package's Phase 1 users run with numpy/joblib only.
"""

from typing import Any


def first_fundamental_form(jacobian: Any) -> Any:
    """g = J^T J, the (d, d) pullback metric from the (768, d) decoder Jacobian."""
    raise NotImplementedError("Implemented in Phase 3 (CURV-01)")


def second_fundamental_form(jacobian: Any, hessian: Any) -> Any:
    """Normal projection (I - J g^-1 J^T) of the (768, d, d) ambient Hessian."""
    raise NotImplementedError("Implemented in Phase 3 (CURV-02)")


def mean_curvature_vector(jacobian: Any, hessian: Any) -> Any:
    """g-trace of the second fundamental form: a (768,) normal-space vector plus ||H||."""
    raise NotImplementedError("Implemented in Phase 3 (CURV-03)")


def metric_condition_number(jacobian: Any) -> Any:
    """cond(g) per point, flagging near-singular non-immersion points."""
    raise NotImplementedError("Implemented in Phase 3 (CURV-04)")
