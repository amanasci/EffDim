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
"""

from typing import Dict, Optional, Sequence, Tuple, Union

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
