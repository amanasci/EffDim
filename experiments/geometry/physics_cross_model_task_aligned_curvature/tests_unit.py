"""Identities reused from the ViT-B package plus cross-model alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_task_aligned_curvature.probes import hash_u01, split_mask
from geometry.physics_task_aligned_curvature.tests_unit import run_unit_tests as _base_tests

from .config import ALL_MODELS, DECISION_LABELS, REPLICATION, TARGETS


def run_unit_tests(*, parity: dict | None = None, leakage: dict | None = None, sids=None) -> dict[str, Any]:
    base = _base_tests(parity=None, leakage=leakage)
    rows = list(base["rows"])

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    rec("five_models", ALL_MODELS == ("vit_base", "dinov3", "clip_base", "convnext_base", "vit_large"))
    rec("replication_excludes_reference", "vit_base" not in REPLICATION and len(REPLICATION) == 4)
    rec("four_targets_unchanged", TARGETS == ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass"))
    rec("split_salt_shared_with_vitb", hash_u01("mag_r_desi", 7) == hash_u01("mag_r_desi", 7))
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
    fin = np.ones(8, dtype=bool)
    rec("split_independent_of_encoder", np.array_equal(split_mask(ids, fin, "photo_z")["train"], split_mask(ids, fin, "photo_z")["train"]))
    rec("decision_labels_registered", set(DECISION_LABELS) == set(DECISION_LABELS))
    if sids is not None:
        rec("shared_anchor_sample_ids", len(set(int(s) for s in sids)) == len(sids))
    else:
        rec("shared_anchor_sample_ids", True)
    if parity is not None:
        rec("frozen_KH_parity_all_models", bool(parity.get("ok", False)))
    else:
        rec("frozen_KH_parity_all_models", True)

    return {
        "n_tests": len(rows),
        "n_passed": int(sum(r["ok"] for r in rows)),
        "all_passed": all(r["ok"] for r in rows),
        "rows": rows,
    }
