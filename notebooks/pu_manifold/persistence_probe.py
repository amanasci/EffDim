"""
Phase 02.6 persistent-homology agreement instrument (SC-3; D-01, D-03, D-04, D-05, D-06).
Turns any point cloud into a persistence diagram, and any (model diagram, reference
diagram) pair into bottleneck and Wasserstein distances normalised by the reference
diagram's own largest bar -- the criterion ratified in
``02.6-SCREENING-RULE-02.md``, replacing the pre-ratification quick task's own
now-superseded normalisation convention. Everything the phase ranks candidates on flows
through this one module.

Arrays in, dicts and arrays out -- no file I/O, no cache handling; the runners under
``notebooks/diagnostics/`` own paths and cache stems.

``ripser`` and ``persim`` have been used exactly once before in this project --
``notebooks/quick_topoae_vs_cae_persistence.ipynb``, notebook-only, untested, unpromoted.
That quick task proved the library API works in this venv at these sample sizes and
surfaced two hazards that transfer directly onto this module's design; both are pinned as
regression tests in this module's test file rather than left as prose warnings. This
module promotes that quick task's helpers into a tested sibling, generalised over any
(model diagram, reference diagram) pair rather than the quick task's TopoAE-versus-CAE
framing.

Persistent homology needs no C2 smoothness guard, no autodiff and no chart-index care -- a
diagram is built from point coordinates alone. No smoothness guard of any kind is ever
invoked in this module, and the chart auto-encoder -- excluded from the halted curvature
axis entirely because its chart-fragmented decoder cannot be safely differentiated --
needs no special casing here.

Like ``cae.py``, ``topoae.py``, ``chart_curvature.py``, ``curvature_probe.py``,
``decoder_curvature.py`` and ``analytic_param.py``, this module imports ``torch`` at
module level (``topoae.pairwise_distances_f64`` and ``topoae.latent_unit_scale`` both take
torch tensors). For the same reason those modules are excluded from
``pu_manifold/__init__.py``'s eager imports, this module is deliberately NOT re-exported
there either.

**The reproducibility gap -- inherited, not resolved here.** ``ripser`` and ``persim`` are
installed into this repository's ``.venv`` only. They are NOT declared in
``pyproject.toml``, because CLAUDE.md bars editing that file for the whole v1.1 milestone.
A clean checkout of this repository does not reproduce this module's behaviour without a
manual install; the import guard below raises with the exact command rather than a bare
``ImportError``, and presence is confirmed by import, never by shelling out to a package
manager.

**This module measures and applies no bar.** No PASS/FAIL, no verdict, no threshold, no
composite or averaged score. :data:`READOUT_CELLS`' sixteen cells are reported
separately and never collapsed -- the criterion is ratified explicitly non-gating (D-09)
and the phase promotes no substrate (D-10).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from . import topoae

try:
    import persim
    from ripser import ripser
except ImportError as _e:  # pragma: no cover -- environment, not logic
    raise ImportError(
        "persistence_probe requires ripser and persim. They are NOT declared in "
        "pyproject.toml -- CLAUDE.md bars editing it for the whole v1.1 milestone -- so "
        "they must already be installed into this repository's venv by hand:\n\n"
        "    .venv/bin/pip install ripser persim\n\n"
        "This is a known reproducibility gap, already recorded against "
        "notebooks/quick_topoae_vs_cae_persistence.ipynb's own section 12, and it is "
        "inherited here, not resolved."
    ) from _e


# --- the fixed maximum homology degree ----------------------------------------------------

PH_MAXDIM = 1
"""H0 and H1 only. ROADMAP.md records no power analysis for the second Betti number, so an
H2 read-out could neither support nor refute a candidate (D-04). Every internal ``ripser``
call in this module passes this constant; no call site raises it."""


# --- the sixteen canonical read-out cells (D-03, D-04, D-05, D-06) -------------------------

READOUT_CELLS: Tuple[str, ...] = (
    "latent|intrinsic|H0|bottleneck_norm",
    "latent|intrinsic|H0|wasserstein_norm",
    "latent|intrinsic|H1|bottleneck_norm",
    "latent|intrinsic|H1|wasserstein_norm",
    "latent|ambient|H0|bottleneck_norm",
    "latent|ambient|H0|wasserstein_norm",
    "latent|ambient|H1|bottleneck_norm",
    "latent|ambient|H1|wasserstein_norm",
    "decoder_image|intrinsic|H0|bottleneck_norm",
    "decoder_image|intrinsic|H0|wasserstein_norm",
    "decoder_image|intrinsic|H1|bottleneck_norm",
    "decoder_image|intrinsic|H1|wasserstein_norm",
    "decoder_image|ambient|H0|bottleneck_norm",
    "decoder_image|ambient|H0|wasserstein_norm",
    "decoder_image|ambient|H1|bottleneck_norm",
    "decoder_image|ambient|H1|wasserstein_norm",
)
"""The one place the 16 canonical labels are written down in code, exactly matching
``02.6-SCREENING-RULE-02.md``'s ratified list and order: the cross product of space in
(``latent``, ``decoder_image``), reference in (``intrinsic``, ``ambient``), degree in
(``H0``, ``H1``) and distance in (``bottleneck``, ``wasserstein``), joined by a pipe with a
``_norm`` suffix on the distance component. The runner, the notebooks and the findings
document all key off this one tuple."""

_SPACES: Tuple[str, ...] = ("latent", "decoder_image")
_REFERENCES: Tuple[str, ...] = ("intrinsic", "ambient")
_DEGREES: Tuple[str, ...] = ("H0", "H1")
_DISTANCES: Tuple[str, ...] = ("bottleneck", "wasserstein")


# --- diagram construction -------------------------------------------------------------------


def finite_pairs(dgm: np.ndarray) -> np.ndarray:
    """The finite birth/death rows of one ``ripser`` per-degree diagram, as a contiguous
    float64 ``(m, 2)`` array -- empty ``(0, 2)`` if none. Promoted verbatim (in behaviour)
    from the quick task's own helper: filtering the H0 diagram's single infinite death is
    required for correctness, not hygiene, because ``persim``'s distances cannot consume
    an infinite value and an unfiltered largest bar is infinite."""
    d = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if d.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.ascontiguousarray(d[np.isfinite(d[:, 1])])


def persistence_diagram(D: np.ndarray, maxdim: int = PH_MAXDIM) -> List[np.ndarray]:
    """One point cloud's persistence diagrams, in RAW units, from its square symmetric
    float64 distance matrix ``D``. Returns a list of length ``maxdim + 1`` -- index ``k``
    holds the filtered (via :func:`finite_pairs`) degree-``k`` diagram.

    Normalisation is a caller-side division (see :func:`ph_agreement`), never baked in
    here, so the same raw diagram serves D-05's rule and any future convention without
    recomputing ``ripser``."""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"persistence_diagram: D must be a square 2-d distance matrix, got shape "
            f"{D.shape!r}"
        )
    if D.shape[0] == 0:
        raise ValueError("persistence_diagram: D is empty; nothing to compute persistence on.")
    if not np.allclose(D, D.T):
        raise ValueError("persistence_diagram: D must be symmetric")

    dgms = ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]
    return [finite_pairs(dgm) for dgm in dgms]


# --- D-05's normalisation denominator and the saturation travelling caveat -----------------


def max_persistence(dgm: np.ndarray) -> float:
    """The reference diagram's own scale per D-05: the longest life (death minus birth) in
    ``dgm``, ``0.0`` for an empty diagram -- never ``nan``, never an exception.

    Thin denominator: for the intrinsic-plane reference's ``(H1, intrinsic)`` cell this
    value can be small relative to other cells' denominators -- ``02.6-SCREENING-RULE-02.md``'s
    travelling caveat on the thin ``(H1, intrinsic)`` denominator names the measured gap and
    its resolution (P1: report raw, denominator and normalised quotient together, no floor).
    """
    d = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    if d.shape[0] == 0:
        return 0.0
    lives = d[:, 1] - d[:, 0]
    return float(lives.max())


def saturation_value(ref_dgm: np.ndarray) -> float:
    """Half of ``ref_dgm``'s own longest life: the value a bottleneck distance takes and
    cannot exceed once nothing in the model diagram is close enough to match that bar.
    ``02.6-SCREENING-RULE-02.md``'s bottleneck-saturation travelling caveat names this
    value and its consequence (a saturated cell is reported with its flag and never read
    as a measurement)."""
    return 0.5 * max_persistence(ref_dgm)


def is_saturated(value: float, ref_dgm: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether a bottleneck ``value`` sits on :func:`saturation_value` of ``ref_dgm``, to
    within ``tol``."""
    return bool(abs(float(value) - saturation_value(ref_dgm)) <= tol)


# --- the one policy-free agreement computation ----------------------------------------------


def ph_agreement(model_dgm: np.ndarray, ref_dgm: np.ndarray) -> Dict[str, Any]:
    """Bottleneck and Wasserstein distance between one model diagram and one reference
    diagram, both raw and normalised by :func:`max_persistence` of ``ref_dgm``.

    Returns ``{"bottleneck", "wasserstein", "ref_max_persistence", "bottleneck_norm",
    "wasserstein_norm", "saturated"}``. Both normalised fields are the raw distance
    divided by ``ref_max_persistence``, or ``nan`` when that denominator is not strictly
    positive -- never ``inf``, never a substituted fallback. This function is
    policy-free by design: whichever of option group P was ratified (P1) is applied by
    the caller and is the only place it is written down (``02.6-SCREENING-RULE-02.md``).
    """
    model_dgm = np.asarray(model_dgm, dtype=np.float64).reshape(-1, 2)
    ref_dgm = np.asarray(ref_dgm, dtype=np.float64).reshape(-1, 2)

    bottleneck_raw = float(persim.bottleneck(model_dgm, ref_dgm))
    wasserstein_raw = float(persim.wasserstein(model_dgm, ref_dgm))
    ref_max = max_persistence(ref_dgm)

    if ref_max > 0.0:
        bottleneck_norm = bottleneck_raw / ref_max
        wasserstein_norm = wasserstein_raw / ref_max
    else:
        bottleneck_norm = float("nan")
        wasserstein_norm = float("nan")

    return {
        "bottleneck": bottleneck_raw,
        "wasserstein": wasserstein_raw,
        "ref_max_persistence": ref_max,
        "bottleneck_norm": bottleneck_norm,
        "wasserstein_norm": wasserstein_norm,
        "saturated": is_saturated(bottleneck_raw, ref_dgm),
    }


# --- cloud to distance matrix, with Q1's symmetric optional pre-scaling --------------------


def cloud_distance_matrix(points: Any, prescale: bool) -> Tuple[np.ndarray, float]:
    """The one entry point that turns a point cloud into a distance matrix.

    ``prescale`` is a REQUIRED positional argument with no default, so no call site can
    silently inherit a convention. When ``True`` it multiplies the cloud by
    ``topoae.latent_unit_scale(cloud)`` first and returns the applied scalar alongside the
    matrix; when ``False`` it returns ``1.0``. Distances go through
    ``topoae.pairwise_distances_f64`` unchanged -- the float64 discipline is sealed, tested
    code and is not re-derived here.

    ``topoae.latent_unit_scale`` is a single global isotropic scalar, so multiplying by it
    is an isometry up to uniform scale and cannot reorder neighbours -- which is why (Q1)
    it is safe to apply symmetrically to model clouds and reference clouds alike.
    """
    pts = torch.as_tensor(np.asarray(points, dtype=np.float64), dtype=torch.float64)
    if pts.ndim != 2:
        raise ValueError(
            f"cloud_distance_matrix: points must be a 2-d (n, d) array; got shape "
            f"{tuple(pts.shape)}."
        )

    if prescale:
        scale = topoae.latent_unit_scale(pts)
        pts = pts * scale
    else:
        scale = 1.0

    D = topoae.pairwise_distances_f64(pts).numpy()
    return D, float(scale)


# --- the phase's reporting unit --------------------------------------------------------------


def readout_matrix(
    spaces: Dict[str, Any], references: Dict[str, Any], prescale: bool
) -> Dict[str, Any]:
    """The phase's reporting unit: exactly one diagram per cloud (four ``ripser`` calls
    per candidate per seed -- ``spaces["latent"]``, ``spaces["decoder_image"]``,
    ``references["intrinsic"]``, ``references["ambient"]``), decomposed into the sixteen
    :data:`READOUT_CELLS` numbers plus their provenance companions. The sixteen numbers
    are a reporting decomposition of these four diagrams, never sixteen separate
    ``ripser`` calls.

    ``spaces`` maps ``"latent"`` and ``"decoder_image"`` to point clouds; ``references``
    maps ``"intrinsic"`` and ``"ambient"`` to point clouds; ``prescale`` is again required
    with no default, and (Q1) applies identically to every one of the four clouds.

    Returns one flat dict carrying:

    - every key in :data:`READOUT_CELLS` (the normalised value, or ``nan`` when its
      denominator is not strictly positive);
    - for each cell, its raw distance under a ``{space}|{reference}|H{k}|{distance}_raw``
      key;
    - ``{reference}|H{k}|ref_max_persistence`` for each (reference, degree) pair;
    - ``{space}|{reference}|H{k}|saturated`` for each bottleneck cell;
    - ``{cloud_name}|applied_scale`` for each of the four clouds.

    It returns NO key that weights, sums, averages or ranks cells together -- there is no
    composite figure anywhere in this phase.
    """
    for name in _SPACES:
        if name not in spaces:
            raise ValueError(f"readout_matrix: spaces is missing required key {name!r}")
    for name in _REFERENCES:
        if name not in references:
            raise ValueError(f"readout_matrix: references is missing required key {name!r}")

    clouds: Dict[str, Any] = {}
    clouds.update({name: spaces[name] for name in _SPACES})
    clouds.update({name: references[name] for name in _REFERENCES})

    diagrams: Dict[str, List[np.ndarray]] = {}
    scales: Dict[str, float] = {}
    for name, cloud in clouds.items():
        D, scale = cloud_distance_matrix(cloud, prescale)
        diagrams[name] = persistence_diagram(D)
        scales[name] = scale

    result: Dict[str, Any] = {}
    for space in _SPACES:
        for reference in _REFERENCES:
            for k, hk in enumerate(_DEGREES):
                model_dgm = diagrams[space][k]
                ref_dgm = diagrams[reference][k]
                agreement = ph_agreement(model_dgm, ref_dgm)

                for distance in _DISTANCES:
                    cell = f"{space}|{reference}|{hk}|{distance}_norm"
                    result[cell] = agreement[f"{distance}_norm"]
                    result[f"{space}|{reference}|{hk}|{distance}_raw"] = agreement[distance]

                result[f"{reference}|{hk}|ref_max_persistence"] = agreement["ref_max_persistence"]
                result[f"{space}|{reference}|{hk}|saturated"] = agreement["saturated"]

    for name, scale in scales.items():
        result[f"{name}|applied_scale"] = scale

    return result
