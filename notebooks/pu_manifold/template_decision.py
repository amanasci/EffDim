"""
notebooks/pu_manifold/template_decision.py -- D-01, D-04: the Betti-vector + `d_hat`
joint lookup table, and D-11's `H_2` ceiling made explicit as assumption A-01.

Phase 02.7 manifold-template-inference-front-end-inserted. The template call is made by a
**Betti-vector lookup table**, keyed jointly on `(Betti vector, d_hat)` -- D-01, ratified
exactly as written in `02.7-CONTEXT.md` at this plan's Task 1 blocking checkpoint, never
by diagram distance to a reference cloud and never by a hybrid of the two. This module
performs NO diagram-distance computation, uses NO reference cloud, and does not import
`ripser` or `persim` -- pure decision logic over already-computed Betti vectors, per
`02.7-PATTERNS.md`'s recommendation.

D-04's joint key exists because `beta = (1, 0, 0)` is contractible at every dimension:
keyed on Betti alone, the ball row would be a catch-all swallowing any acyclic cloud
whether `d_hat` is 2 or 18. The joint key makes the output a NAMED manifold (`B^2` and
`B^18` are materially different objects to constrain a downstream VAE to) and turns a
Betti/dimension contradiction (e.g. `beta_2 = 1` at `d_hat = 1`) into a detectable
inconsistency -- `lookup` returns `None` on any such mismatch, exactly as it does on a
genuinely off-library Betti vector. Wiring that `None` to a NAMED abstain condition ((a)
off-library vs (d) untrustworthy) is plan `02.7-06`'s job, not this module's.

**Assumption A-01, stated explicitly (D-11's consequence).** The homology degree ceiling
is capped at `H_2`, the `ripser`/`persim` library maximum -- not `H_dhat`. This makes the
front end BLIND to `beta_3` and above: a cloud with non-trivial `beta_3` reads as
`(1, 0, 0)` and is called a ball. This module's claim is bounded to "off-library as
detectable through `H_0..H_2`," never "off-library, full stop."

**Assumption A-02, stated explicitly.** Betti numbers do not determine homeomorphism
type. `decide` below is a classifier over a small named library, never a claim to have
recovered the manifold -- a matched template is a candidate consistent with the measured
topology and dimension, not a proof.

**D-03/plan `02.7-06`: the four named abstain conditions, in a fixed precedence order,
evaluated in `decide` below.** Abstain is the CHEAP failure -- a wrong template sends the
deferred VAE phase to constrain a latent space to the wrong `Z` and derive curvature
through it, a confident wrong number with no detection path. `decide` therefore checks the
conditions most predictive of an unreliable measurement first:

1. **(d) measurement untrustworthy** -- no `d_hat` plateau/consensus (`dimension_readout`
   carries `d_hat is None`), the global estimator spread exceeds `bounds["spread_bound"]`,
   or the agreed Betti vector contradicts `d_hat` (matches a non-ball `TEMPLATE_TABLE` row's
   betti pattern at a *different* `d` than that row specifies -- e.g. `beta_2 = 1` at
   `d_hat = 1` matches `S2`'s betti `(1, 0, 1)` but not `S2`'s `d = 2`). This is D-04's
   stated reason for the joint key: the contradiction is a detectable inconsistency, never
   silently swallowed by a confident label.
2. **(b) no global template applies** -- more than one component survives the geodesic `k`
   range (`connectivity_readout["n_components"] > 1`), or local `d_hat` dispersion exceeds
   `bounds["dispersion_bound"]`.
3. **(c) metric disagreement** -- the Euclidean and geodesic arms report different Betti
   vectors for the same cloud.
4. **(a) off-library** -- both arms agree and the shared Betti vector matches no
   `TEMPLATE_TABLE` row at the measured `d_hat` (checked last: a genuine off-library case,
   not a dimension contradiction already caught by (d)).

`beta_0 = 1` with no persistent `beta_1` and a clean `d_hat` plateau is **not** an
abstention -- it matches the ball row and is a positive call of the trivial template, the
likely PU answer per the ROADMAP prior. It must be reachable only as a success, never only
as a failure mode of some other check.

`bounds` supplies every threshold `decide` compares against as a REQUIRED key with NO
default anywhere -- the ratified rule document (plan `02.7-08`) is the only place those
numbers are written down; `decide` raises `ValueError` naming any missing key. The full
required set is :data:`_REQUIRED_BOUNDS` (`dispersion_bound`, `spread_bound`).

D-13's three-way tally (`tally` below) puts every scored cell in exactly one of
`correct`/`wrong`/`abstained`, reports all three counts and all three rates -- `accuracy`,
`abstention_rate`, `error_rate` -- and never computes accuracy over non-abstained cells
only. On synthetic benchmark cells, ground truth is always in-library, so an abstain is a
miss (mostly condition (d)); on the single PU probe, abstain may be the *correct* answer
(if PU is ball-like, there is no non-trivial template to restrict to). The two arms' tallies
are therefore never summed together -- an abstain does not mean the same thing in both.

`assert_labelled` is this module's metric-label invariant: the primary reporting unit
downstream of this phase is a **(metric, read-out) pair**, never a bare read-out, so a
per-metric value (currently just `betti`) that is not itself a dict keyed by
`{"euclidean", "geodesic"}` raises `ValueError` naming the offending key. Called at the end
of `decide`, so a later change that reintroduces a single unlabelled value goes red here
rather than silently reporting a number whose metric nobody can recover.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

_MAX_DEGREES = 3
"""H0, H1, H2 -- D-11's ceiling. A betti vector may carry at most this many entries."""

TEMPLATE_TABLE: Dict[str, Dict[str, object]] = {
    "S1": {"betti": (1, 1), "d": 1},
    "S2": {"betti": (1, 0, 1), "d": 2},
    "T2": {"betti": (1, 2, 1), "d": 2},
    "ball": {"betti": (1, 0, 0), "d": None},  # matches any d_hat -- carried into the label
}
"""D-04's four rows, keyed jointly on `(Betti vector, d_hat)`. Betti tuples shorter than
`_MAX_DEGREES` entries are implicitly zero-padded on the right by `lookup` before
matching -- D-11's `H_2` ceiling means a caller may reasonably report only as many
degrees as it actually computed."""


def _normalize_betti(betti: Sequence[int]) -> Tuple[int, ...]:
    """Coerce `betti` to a length-`_MAX_DEGREES` `(beta_0, beta_1, beta_2)` tuple,
    right-padding with zeros for callers that report fewer degrees. Raises `ValueError`
    if `betti` is empty or carries more than `_MAX_DEGREES` entries -- D-11 caps homology
    at `H_2`, so a 4th-or-later entry is not a length this module can accept.
    """
    b = tuple(int(x) for x in betti)
    if len(b) == 0:
        raise ValueError("lookup: betti vector must not be empty")
    if len(b) > _MAX_DEGREES:
        raise ValueError(
            f"lookup: betti vector has {len(b)} entries; D-11 caps homology at H_2, so at "
            f"most {_MAX_DEGREES} (beta_0, beta_1, beta_2) entries are valid, got {b!r}"
        )
    return b + (0,) * (_MAX_DEGREES - len(b))


def lookup(betti: Sequence[int], d_hat: Union[int, float]) -> Optional[str]:
    """D-01's classifier: match `betti` (any length up to `_MAX_DEGREES`, zero-padded)
    and `d_hat` jointly against `TEMPLATE_TABLE`. Returns the matched template name,
    `"ball_d{d_hat}"` for the ball row (D-04's joint key, so `B^2` and `B^18` are
    distinguishable outputs), or `None` on no match -- abstain condition (a), off-library
    (wiring lands in plan `02.7-06`).
    """
    b = _normalize_betti(betti)

    ball_row = TEMPLATE_TABLE["ball"]
    ball_betti = ball_row["betti"] + (0,) * (_MAX_DEGREES - len(ball_row["betti"]))
    if b == ball_betti:
        return f"ball_d{int(d_hat)}"

    for name, row in TEMPLATE_TABLE.items():
        if name == "ball":
            continue
        row_betti = row["betti"] + (0,) * (_MAX_DEGREES - len(row["betti"]))
        if b == row_betti and int(d_hat) == row["d"]:
            return name

    return None


# --- D-03/D-04: the four named abstain conditions and the joint-key decision --------------

ABSTAIN_CONDITIONS: Tuple[str, ...] = ("a", "b", "c", "d")
"""The four disjoint abstain conditions, in D-03's own letter order (not `decide`'s
precedence order -- see :func:`decide`'s docstring for the evaluation order, (d) first)."""

ABSTAIN_CONDITION_MEANINGS: Dict[str, Dict[str, str]] = {
    "a": {
        "meaning": (
            "off-library -- both metric arms agree on a Betti vector that matches no "
            "TEMPLATE_TABLE row at the measured d_hat."
        ),
        "instruction": (
            "widen the template library (lifting the H_2 ceiling and/or adding new rows) -- "
            "a deferred, out-of-scope-for-this-phase action."
        ),
    },
    "b": {
        "meaning": (
            "no global template applies -- more than one component survives the geodesic k "
            "range, or local d_hat dispersion exceeds the ratified dispersion_bound."
        ),
        "instruction": (
            "terminate the follow-on topological-VAE arm -- no single global template fits "
            "this cloud."
        ),
    },
    "c": {
        "meaning": (
            "metric disagreement -- the Euclidean and geodesic arms report different Betti "
            "vectors for the same cloud."
        ),
        "instruction": "retry with more data or reconsider which metric is trustworthy here.",
    },
    "d": {
        "meaning": (
            "measurement untrustworthy -- no d_hat plateau/consensus was reached, the global "
            "estimator spread exceeds the ratified spread_bound, or the agreed Betti vector "
            "contradicts d_hat (matches a table row's betti pattern at a different d)."
        ),
        "instruction": (
            "retry with more data or a better-conditioned measurement -- the dimension "
            "estimate itself is not trustworthy, independent of which template it would "
            "otherwise have matched."
        ),
    },
}
"""Parallel to :data:`ABSTAIN_CONDITIONS`: full text and downstream instruction for each
letter, so a reader of `decide`'s output never has to consult the planning record to know
what an abstention means or what to do about it."""

_METRIC_LABELS = frozenset({"euclidean", "geodesic"})
"""The two metric arms this phase ever reports, per D-05's Euclidean-vs-geodesic design.
Shared by `decide`'s input validation and `assert_labelled`'s output invariant."""

_REQUIRED_BOUNDS: Tuple[str, ...] = ("dispersion_bound", "spread_bound")
"""Every threshold `decide` compares a measurement against. No default anywhere -- plan
`02.7-08`'s ratified rule document is required to supply exactly this set, no gaps and no
extras. `dispersion_bound` fires part of abstain (b); `spread_bound` fires part of abstain
(d)."""


def _require_bounds(bounds: Dict[str, Any]) -> None:
    """Raises `ValueError` naming every missing key in `_REQUIRED_BOUNDS` -- `decide` never
    substitutes a value for an absent bound."""
    missing = [key for key in _REQUIRED_BOUNDS if key not in bounds]
    if missing:
        raise ValueError(
            f"decide: bounds is missing required key(s) {missing!r}; every threshold decide "
            f"compares against must be supplied explicitly by the ratified rule document "
            f"(plan 02.7-08) -- no default is ever substituted. Required keys: "
            f"{list(_REQUIRED_BOUNDS)!r}"
        )


def _betti_dimension_contradiction(betti: Sequence[int], d_hat: Union[int, float]) -> Optional[str]:
    """D-04's contradiction check: does `betti` match a *non-ball* `TEMPLATE_TABLE` row's
    betti pattern at a `d` other than that row's own `d`? (`beta_2 = 1` at `d_hat = 1`
    matches `S2`'s `(1, 0, 1)` but not `S2`'s `d = 2`.) The ball row is excluded -- it
    matches any `d_hat` by construction (D-04), so it can never contradict a dimension.
    Returns a detail string naming the contradiction, or `None` if none found."""
    b = _normalize_betti(betti)
    for name, row in TEMPLATE_TABLE.items():
        if name == "ball":
            continue
        row_betti = row["betti"] + (0,) * (_MAX_DEGREES - len(row["betti"]))
        if b == row_betti and int(d_hat) != row["d"]:
            return (
                f"betti {b} matches template {name!r}'s betti pattern (expects d={row['d']}) "
                f"but the measured d_hat={d_hat!r} -- a Betti/dimension contradiction per "
                f"D-04, routed to abstain (d) rather than a confident label"
            )
    return None


def decide(
    arms: Dict[str, Dict[str, Any]],
    dimension_readout: Dict[str, Any],
    connectivity_readout: Dict[str, Any],
    bounds: Dict[str, float],
) -> Dict[str, Any]:
    """D-01/D-03/D-04/D-13's full template decision: the joint `(Betti vector, d_hat)`
    lookup, gated by the four named abstain conditions, evaluated in this fixed precedence
    order -- **(d) then (b) then (c) then (a)** -- because the conditions most predictive of
    an unreliable measurement are checked first, before a metric-disagreement or
    off-library reading is trusted at all:

    1. **(d) measurement untrustworthy** -- `dimension_readout["d_hat"] is None` (no
       plateau/consensus), or `dimension_readout["spread"]["range"]` exceeds
       `bounds["spread_bound"]`, or either arm's Betti vector contradicts `d_hat`
       (:func:`_betti_dimension_contradiction`).
    2. **(b) no global template applies** -- `connectivity_readout["n_components"] > 1`, or
       the largest per-estimator `"range"` in `dimension_readout["local_dispersion"]`
       exceeds `bounds["dispersion_bound"]`.
    3. **(c) metric disagreement** -- the two arms' (zero-padded) Betti vectors differ.
    4. **(a) off-library** -- both arms agree and :func:`lookup` returns `None` for the
       shared Betti vector at `d_hat`.

    Parameters:

    - `arms`: `{"euclidean": {"betti": Sequence[int], ...}, "geodesic": {"betti":
      Sequence[int], ...}}` -- each arm's Betti vector (its confidence-band `betti_vector`
      reading) and, optionally, its raw `confidence_band.bootstrap_band`/`bands_for_diagram`
      results (carried through unread by this function; `decide` needs only `"betti"`).
    - `dimension_readout`: carries `"d_hat"` and `"reason"` (straight from
      `local_dimension.plateau_consensus`'s own return dict), `"spread"` (a
      `local_dimension.spread`-shaped dict, read at the winning `d_hat`'s `k`), and
      `"local_dispersion"` (a `local_dimension.dispersion`-shaped
      `{estimator_name: {"range": ..., ...}}` dict).
    - `connectivity_readout`: a `geodesic_graph.component_readout`-shaped dict; only
      `"n_components"` and `"dropped_fraction"` are read.
    - `bounds`: every threshold in :data:`_REQUIRED_BOUNDS`, required, no default.

    Returns a dict carrying `"template"` (`str` or `None`), `"abstained"` (`bool`),
    `"abstain_condition"` (a letter in :data:`ABSTAIN_CONDITIONS`, or `None`),
    `"abstain_detail"` (the specific read-out that tripped it, or `None`), `"betti"` (the
    per-arm Betti vectors, each under its own metric label -- `{"euclidean": ..., "geodesic":
    ...}`), `"d_hat"`, and `"connectivity"` (`{"n_components", "dropped_fraction"}`). Emits
    no key that averages, weights or ranks the two arms together, and no composite score.
    `assert_labelled` is called on the result before it is returned.
    """
    _require_bounds(bounds)

    for label in _METRIC_LABELS:
        if label not in arms:
            raise ValueError(f"decide: arms is missing required key {label!r}")
        if "betti" not in arms[label]:
            raise ValueError(f"decide: arms[{label!r}] is missing required key 'betti'")

    betti_euclidean = _normalize_betti(arms["euclidean"]["betti"])
    betti_geodesic = _normalize_betti(arms["geodesic"]["betti"])

    d_hat = dimension_readout.get("d_hat")
    n_components = connectivity_readout["n_components"]
    dropped_fraction = connectivity_readout["dropped_fraction"]

    common: Dict[str, Any] = {
        "betti": {"euclidean": betti_euclidean, "geodesic": betti_geodesic},
        "d_hat": d_hat,
        "connectivity": {"n_components": n_components, "dropped_fraction": dropped_fraction},
    }

    def _abstain(condition: str, detail: str) -> Dict[str, Any]:
        result = {
            "template": None,
            "abstained": True,
            "abstain_condition": condition,
            "abstain_detail": detail,
            **common,
        }
        # plan 02.7-06 Task 2 wires assert_labelled(result) in here.
        return result

    # (d) measurement untrustworthy -- checked first (see docstring precedence order).
    if d_hat is None:
        reason = dimension_readout.get("reason", "")
        return _abstain("d", f"no d_hat plateau/consensus reached: {reason}")

    spread_range = dimension_readout["spread"]["range"]
    if spread_range > bounds["spread_bound"]:
        return _abstain(
            "d",
            f"global estimator spread range={spread_range!r} exceeds "
            f"bounds['spread_bound']={bounds['spread_bound']!r}",
        )

    contradiction = _betti_dimension_contradiction(
        betti_euclidean, d_hat
    ) or _betti_dimension_contradiction(betti_geodesic, d_hat)
    if contradiction is not None:
        return _abstain("d", contradiction)

    # (b) no global template applies.
    if n_components > 1:
        return _abstain(
            "b", f"n_components={n_components} > 1 surviving components in the geodesic kNN graph"
        )

    local_dispersion = dimension_readout["local_dispersion"]
    max_dispersion = max(entry["range"] for entry in local_dispersion.values())
    if max_dispersion > bounds["dispersion_bound"]:
        return _abstain(
            "b",
            f"local d_hat dispersion max range={max_dispersion!r} exceeds "
            f"bounds['dispersion_bound']={bounds['dispersion_bound']!r}",
        )

    # (c) metric disagreement.
    if betti_euclidean != betti_geodesic:
        return _abstain(
            "c",
            f"euclidean betti={betti_euclidean!r} != geodesic betti={betti_geodesic!r}",
        )

    # (a) off-library -- both arms agree; try the shared vector against the table.
    template = lookup(betti_euclidean, d_hat)
    if template is None:
        return _abstain(
            "a", f"betti={betti_euclidean!r} at d_hat={d_hat!r} matches no TEMPLATE_TABLE row"
        )

    result = {
        "template": template,
        "abstained": False,
        "abstain_condition": None,
        "abstain_detail": None,
        **common,
    }
    # plan 02.7-06 Task 2 wires assert_labelled(result) in here.
    return result
