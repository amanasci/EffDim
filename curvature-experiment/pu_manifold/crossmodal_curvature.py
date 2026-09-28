"""Phase 7 curvature-conditioned crossmodal alignment: the pre-registration constants block,
its guard, and the two-tailed, three-`d`, positive-control-gated verdict rule.

**Closure note.** Only ``split_indices`` and the two split constants its tests use
(``SPLIT_SEED``, ``HOLDOUT_FRACTION``) remain in this file; the rest of the constants block and
``VERDICT_RULE`` described below are archived verbatim in
``archive/pu_manifold_trimmed/crossmodal_curvature.py``.

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
# Fit protocol -- re-declared from the measured 07_pu_plain_ae_fit_run.py spike (D7-01).
# =============================================================================================

SPLIT_SEED = 20260813
HOLDOUT_FRACTION = 0.2


# =============================================================================================
# Compute functions (plan 07-02, Task 1). Everything above this line is the frozen
# pre-registration -- nothing above it is touched by this addition. Pure functions only: no
# file I/O, and no defaults on any pre-registered parameter, matching pointcloud_probe.py's
# stated discipline about how a pre-registered value gets inherited by accident instead of
# chosen explicitly at every call site.
# =============================================================================================

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
