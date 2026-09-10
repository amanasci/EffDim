"""Frozen decision gates. Do not overwrite prior Q labels."""

from __future__ import annotations

from .associations import adapt_pass, global_pass
from .config import MIN_VALID_REPS, PRESERVED_LABELS


def decide(stab_a: dict, stab_b: dict, n_a: int, n_b: int, *, parity_ok: bool, unexpected_ab: bool = False) -> dict:
    if (not parity_ok) or n_a < MIN_VALID_REPS or n_b < MIN_VALID_REPS:
        label = "q_geometry_resampling_unresolved"
        return _pack(label, stab_a, stab_b, n_a, n_b, False, False, False, False)

    ga, aa = global_pass(stab_a), adapt_pass(stab_a)
    gb, ab = global_pass(stab_b), adapt_pass(stab_b)
    cond_ok = ga and aa
    obj_ok = gb and ab

    if unexpected_ab and (not ga) and (not aa) and gb and ab:
        # unstable A, stable B — do not interpret mechanically
        label = "q_geometry_resampling_unresolved"
    elif ga and aa and gb and ab:
        label = "q_global_and_adaptation_associations_geometry_robust"
    elif ga and gb and (not aa or not ab):
        label = "q_global_association_robust_adaptation_fragile"
    elif aa and ab and (not ga or not gb):
        label = "q_adaptation_robust_global_association_fragile"
    elif cond_ok and not obj_ok:
        label = "q_associations_conditionally_stable_support_sensitive"
    elif (not ga) or (not aa):
        label = "q_associations_partition_or_fit_sensitive"
    elif (not ga and not aa) and (not gb and not ab):
        label = "q_associations_geometry_fragile"
    else:
        label = "q_associations_geometry_fragile"

    return _pack(label, stab_a, stab_b, n_a, n_b, ga, aa, gb, ab)


def _pack(label, stab_a, stab_b, n_a, n_b, ga, aa, gb, ab) -> dict:
    return {
        "summary_label": label,
        "prior_labels_not_overwritten": list(PRESERVED_LABELS),
        "global_pass_conditional": ga,
        "adapt_pass_conditional": aa,
        "global_pass_object_support": gb,
        "adapt_pass_object_support": ab,
        "n_valid_conditional": int(n_a),
        "n_valid_object_support": int(n_b),
        "notes": {
            "conditional": {k: stab_a.get(k, {}) for k in ("r2_G", "mse_G", "delta_adapt")},
            "object_support": {k: stab_b.get(k, {}) for k in ("r2_G", "mse_G", "delta_adapt")},
        },
        "distinction": (
            "Prior labels answered exact-recovery, operating characteristics, or chart-link "
            "questions. This label answers only whether the frozen ViT-B Q–probe associations "
            "survive resampling of the geometry used to estimate Q."
        ),
    }
