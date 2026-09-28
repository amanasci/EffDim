"""Phase 7 curvature-conditioned crossmodal alignment: the pre-registration constants block,
its guard, and the two-tailed, three-`d`, positive-control-gated verdict rule.

**This module adds; it does not edit.** Five sealed modules are imported unchanged and never
modified: ``mknn`` (the source paper's headline probe -- ``mknn_score``, ``permutation_null``,
``bootstrap_ci``, ``chance_floor``, ``hubness_skewness``), ``cae`` (``PlainAutoEncoder``,
``train_plain_ae``, ``reconstruction_stats`` -- the validated instrument, D7-01),
``decoder_curvature`` (``plain_decoder_curvature``, which differentiates ``model.decode``
alone, never the encoder-composed round trip), ``curvature_probe`` (``permutation_null``,
``local_density_weights`` -- D7-03's density statistic), and ``cross_split_curvature``
(``partial_spearman`` -- D7-03's density partial). Two more sealed modules are named here
even though this file imports nothing from them for a constant: ``linear_probe.py`` (Phase 5)
and ``pointcloud_probe.py`` (Phase 6) are both frozen artifacts of prior phases and carry no
constant this phase inherits -- Phase 7 promotes the point, not the region/bucket, as its unit
of observation (D7-04), so nothing in either module's bucketed vocabulary applies here. This
file re-declares its own constants as plain literals rather than importing from any of the
five or the two above, so a same-named constant in a sealed module can never collide with or
shadow this phase's own (D7-05).

**The constants below are FROZEN.** They were committed in this file, in this commit, before
any PU number existed anywhere in the tree. A later edit to any of them is a recorded
pre-registration BREACH, never a silent fix (D7-06) -- the failure mode this freeze exists to
prevent is exactly the one ``02.6-FINDINGS.md`` Section 4 already documented once.

**What each pre-registered decision governs, by ID:**

- **D7-01** -- the curvature field and the `d`-sweep. The instrument is
  ``cae.PlainAutoEncoder`` trained by ``cae.train_plain_ae``, curvature from
  ``decoder_curvature.plain_decoder_curvature(model, model.encode(x))``. The headline
  correlation is measured and reported at every ``d`` in ``D_SWEEP = (20, 25, 32)`` -- never at
  one `d` alone, because PU's own reconstruction sweep shows no plateau through `d=48`
  (07-CONTEXT.md Section 5), so a single-`d` fit is a truncated approximation and cannot be
  defended as the whole answer.
- **D7-02** -- the positive control. Not optional: a curvature-MKNN relationship is planted at
  PU's own realized ``||H||`` dynamic range and the test must recover it, or the phase may not
  report a null.
- **D7-03** -- density and hubness. Reported alongside the headline result; gates nothing.
- **D7-04** -- the per-point statistic. ``mknn.mknn_score`` computes a per-point array before
  it is averaged away; this phase retains it, so the unit of observation is one of 10,000
  paired points, never a bucket or region (the promote decision recorded in this plan's
  ``assumption_delta_decision``).
- **D7-05** -- additive-only. Nothing under ``src/effdim/`` and none of the seven sealed
  ``notebooks/pu_manifold/*.py`` modules named above are edited by this phase.
- **D7-06** -- the freeze itself. ``assert_preregistered()`` is the gate every number-producing
  code path calls first; the commit that adds this file is the strict git ancestor every later
  PU number must be proven against.
- **D7-07** -- the alignment-metric scope. CKA is out of scope and not implemented anywhere in
  this codebase (07-CONTEXT.md Section 3). ``ALIGNMENT_METRIC = "mknn"`` freezes that scope as
  a checkable constant carried on every record row, so the exclusion is a positive, checkable
  fact rather than a claim made only in prose.

No file I/O happens in this module, following ``linear_probe.py``'s and
``region_partition.py``'s stated convention: a default is how a pre-registered value gets
inherited by accident instead of by an explicit call-site choice. This file defines no
computable defaults either -- only flat literals.
"""


# =============================================================================================
# Field and instrument (D7-01).
# =============================================================================================

D_SWEEP = (20, 25, 32)
"""The only latent dimensions this phase ever fits or reports. A runner MUST refuse a `d`
outside this tuple rather than silently fitting it. No plateau through d=48 in PU's own
reconstruction sweep (07-CONTEXT.md Section 5) is why one `d` cannot be defended alone."""

AE_IN_DIM = 768
AE_HIDDEN = (250, 250, 250)
AE_ACTIVATION = "silu"
CURVATURE_SOURCE_FUNCTION = "decoder_curvature.plain_decoder_curvature"
CURVATURE_CONVENTION = "trace"

# =============================================================================================
# Fit protocol -- re-declared from the measured 07_pu_plain_ae_fit_run.py spike (D7-01).
# =============================================================================================

MAX_EPOCHS = 600
TORCH_INIT_SEED = 0
SPLIT_SEED = 20260813
HOLDOUT_FRACTION = 0.2
TRAIN_CFG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "early_stop_patience": MAX_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "wallclock_ceiling_s": float("inf"),
}
"""early_stop_patience > MAX_EPOCHS deliberately disables total-loss early stopping, per
03-08-DEFECTS-01.md defect 2: an earlier phase's early-stop plateaued on a penalty term, not
on reconstruction, and ended training prematurely. This phase does not repeat that."""


# =============================================================================================
# Significance (D7-04).
# =============================================================================================

N_PERMUTATIONS = 1000
PERMUTATION_SEED = 20260825


# =============================================================================================
# Positive control (D7-02) -- shares the freeze commit; not tunable after seeing the real
# number.
# =============================================================================================

POSITIVE_CONTROL_TARGET_RHOS = (0.02, 0.05, 0.10, 0.20)
"""Strictly increasing (asserted below), so 'the smallest planted effect the test recovers'
is well defined by tuple order."""
POSITIVE_CONTROL_SEED = 20260825
POSITIVE_CONTROL_RULE = (
    "The planted relationship is built on PU's own realized d=20 ||H|| field, so the "
    "planting happens at PU's actual dynamic range -- Phase 6's rng.random(n) selfcheck (a "
    "~20x-spread field against PU's own order-2x) explicitly does not serve as a substitute. "
    "The same permutation machinery and the same n as the headline test are used. The "
    "reported quantity is the SMALLEST entry of POSITIVE_CONTROL_TARGET_RHOS at which either "
    "tail (per SIGNIFICANCE_TAIL_RULE) clears. Mechanism, spelled out in full because a "
    "planting mechanism chosen after seeing the real number is exactly the tuning this "
    "freeze exists to prevent: rank-transform the real field to u = (rankdata(h) - 0.5) / n; "
    "set p = clip(0.5 + slope * (u - 0.5), 0.0, 1.0); draw j ~ Binomial(k, p) from "
    "np.random.default_rng(POSITIVE_CONTROL_SEED) and take the planted per-point value as "
    "j / k, so it carries the same j/k discretization the real statistic has; find slope by "
    "40 iterations of bisection on the bracket [0.0, 2.0] against the achieved Spearman, "
    "re-seeding the generator to POSITIVE_CONTROL_SEED at every trial so the search is "
    "deterministic. The achieved Spearman is recorded beside every target, never silently "
    "substituted for it."
)

# =============================================================================================
# Provenance and record (D7-05, D7-06).
# =============================================================================================

SEED_HANDLING_RULE = "single_seed_across_d_sweep"
"""ACCEPTED LIMITATION, not a silent stability assumption: inherited from Phase 5's measured
seed-instability of decoder curvature fields (three seeds' fields mutually anti-correlated on
rank, 05-03-DECISION.md). Three seeds x three d would be ~6 hours of field computation alone;
this phase runs one seed per d and names the gap explicitly wherever a verdict is reported."""
RECORD_STEM = "07_crossmodal_curvature"
RECORD_LOCATION_RULE = (
    "The frozen record is written under cache.CACHE_DIR via cache.cache_path, which routes "
    "every write through cache._assert_inside_cache's containment guard -- this phase's one "
    "real security mitigation (T-07-01). It is written to "
    "notebooks/.cache/07_crossmodal_curvature.jsonl and is distinct from the nine "
    "pre-existing notebooks/diagnostics/07_*.jsonl spike outputs, which are informational "
    "inputs only, carry no assert_preregistered guard, predate this freeze, and satisfy "
    "nothing of this phase's own pre-registration."
)
PREREGISTRATION_FREEZE_RULE = (
    "The freeze commit -- the commit that adds this file -- must be a STRICT ancestor of the "
    "commit that first produces a PU number. `git merge-base --is-ancestor <freeze> HEAD` "
    "alone is insufficient because a commit is its own ancestor and would pass even if a "
    "number were produced in the freeze commit itself; `git rev-list --count <freeze>..HEAD` "
    "must also be at least 1."
)

VERDICT_RULE = """D7-08 VERDICT_RULE -- frozen in committed source before any Phase 7 probe
number existed (D7-06). Ratified at this plan's Task 1 blocking checkpoint, ratify-all.

The headline statistic at each d in D_SWEEP is the Spearman rank correlation between the
per-point curvature magnitude ||H||_i and the per-point MKNN score MKNN_i (mknn.mknn_score),
over all 10,000 rows, at HEADLINE_K = 20 (D7-04). The per-point unit of observation is a
promote, not an add-alongside: this phase runs no bucketed arm at all.

Significance at each d follows SIGNIFICANCE_TAIL_RULE in full: curvature_probe.permutation_null
is called TWICE -- once on (H, MKNN), once on (-H, MKNN) -- each at
NULL_QUANTILE_PER_TAIL = 0.975, because the research hypothesis predicts a NEGATIVE association
and the underlying test is one-sided in the wrong direction as written. "Clears" at a given d
means either tail cleared at that d.

The three-d outcome maps onto VERDICT_VALUES as follows:
  (a) all three d in D_SWEEP agree the field clears (association detected at every d)
      -> ASSOCIATION DETECTED;
  (b) all three d in D_SWEEP agree the field does not clear at either tail
      -> NO DETECTABLE RELATIONSHIP, subject to the D7-02 override below;
  (c) any disagreement across the three d -- some clear, some do not -- is a complete,
      valid, TERMINAL outcome in its own right, never a stall and never resolved by a
      majority vote: SPLIT ACROSS d, mirroring Phase 5's SPLIT ACROSS SEEDS precedent.

D7-02 OVERRIDE: NO DETECTABLE RELATIONSHIP may only be reported if the positive control
(POSITIVE_CONTROL_RULE) cleared at some entry of POSITIVE_CONTROL_TARGET_RHOS. If the positive
control recovers NOTHING at the pre-registered effect-size grid, the phase may not report a
null at all -- the verdict is UNDERPOWERED -- NO CLAIM instead. Without this check a null
cannot be distinguished from an underpowered test, and a null is the likely outcome absent it.

Density and hubness (D7-03: spearman(1.0 / w, ||H||) under DENSITY_SIGN_CONVENTION,
mknn.hubness_skewness) are reported alongside every verdict above and gate NONE of it
(DIAGNOSTICS_ARE_NON_GATING = True).

CAVEATS, carried in this rule's own text and not only in surrounding prose:

- No ground truth for PU curvature exists anywhere in this record. The validated instrument
  fidelity is the RANGE in INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99), never a point estimate --
  no verdict produced under this rule may quote a single number as the instrument's accuracy.
- Phase 4's HOLDS is NOT evidence of a curvature-alignment association and must not be cited
  as one under this rule: its split axis correlated 0.82 with density and its raw gap was
  mostly a region-size artifact.
- This milestone runs at n = 10,000, where the k/n chance floor is roughly ten times the
  source paper's own n = 101,725 -- ratio-over-chance readings must be read against that
  floor, not against the source paper's.
- The field is evaluated on all 10,000 rows, including the 8,000 rows the decoder itself
  trained on (FIELD_EVALUATED_ON) -- this is not a fresh-holdout evaluation of the field.
- SEED_HANDLING_RULE = "single_seed_across_d_sweep" is an ACCEPTED LIMITATION inherited from
  Phase 5's measured seed-instability of decoder curvature fields, not a silent assumption
  that a single seed is representative.
"""

VERDICT_VALUES = (
    "ASSOCIATION DETECTED",
    "NO DETECTABLE RELATIONSHIP",
    "SPLIT ACROSS d",
    "UNDERPOWERED -- NO CLAIM",
)
"""The four terminal outcomes. SPLIT ACROSS d is a complete result, not a stall (mirrors
Phase 5's SPLIT ACROSS SEEDS). UNDERPOWERED -- NO CLAIM makes D7-02's power requirement
mechanical: NO DETECTABLE RELATIONSHIP may not be reported without it."""


# =============================================================================================
# Compute functions (plan 07-02, Task 1). Everything above this line is the frozen
# pre-registration -- nothing above it is touched by this addition. Pure functions only: no
# file I/O, and no defaults on any pre-registered parameter, matching pointcloud_probe.py's
# stated discipline about how a pre-registered value gets inherited by accident instead of
# chosen explicitly at every call site.
# =============================================================================================

# Deliberately below the freeze, not merged into the module's original
# ``from typing import Any, Dict`` line above, so that line is never touched.
from typing import Tuple  # noqa: E402

import numpy as np  # noqa: E402


def split_indices(n: int, split_seed: int, holdout_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Re-declares ``curvature_field_pu_run._split``'s algorithm rather than importing a
    diagnostics module (D7-05 adjacency: this file imports only sealed ``pu_manifold``
    modules, never a ``notebooks/diagnostics`` script): ``np.random.default_rng(split_seed)
    .permutation(n)``, the first ``round(n * holdout_fraction)`` entries are holdout, the
    rest train. Returns ``(train_idx, holdout_idx)``."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return train_idx, holdout_idx
