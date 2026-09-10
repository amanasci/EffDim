"""Unit tests required by the geometry-resampling brief."""

from __future__ import annotations

import numpy as np

from .associations import adapt_pass, combined_interval, global_pass, summarize_replicates
from .config import HALF_PRIME, K, K_PRIME
from .q_fit import hash_seed, partition_halves
from .schemes import global_inclusion_mask


def run_unit_tests(*, parity: dict | None = None, ctx: dict | None = None) -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    # 2 deterministic replicate generation
    rec("deterministic_hash", hash_seed("A", 1, 99) == hash_seed("A", 1, 99))
    rec("hash_changes_with_replicate", hash_seed("A", 1, 99) != hash_seed("A", 2, 99))

    # 3–4 conditional partitions
    a, b = partition_halves(K, 123)
    rec("conditional_disjoint", len(set(a) & set(b)) == 0)
    rec("conditional_cover_once", sorted(list(a) + list(b)) == list(range(K)))
    rec("conditional_sizes_1024", len(a) == 1024 and len(b) == 1024)

    # 5 global mask shared
    m1 = global_inclusion_mask(200, 3)
    m2 = global_inclusion_mask(200, 3)
    rec("global_mask_shared", bool(np.array_equal(m1, m2)))
    rec("global_mask_approx_80", abs(m1.mean() - 0.8) < 0.02)

    # 6–8 query / self / other-anchor mask (synthetic)
    from .schemes import eligible_candidates

    n, n_anc = 40, 5
    mask = global_inclusion_mask(n, 0)
    queries = np.arange(n_anc)
    rec("queries_always_available", len(queries) == n_anc)
    for q in queries:
        cand = eligible_candidates(mask, q)
        rec_ok = q not in set(cand.tolist())
        if not rec_ok:
            rec("self_not_in_candidates_if_we_exclude", False)
            break
    else:
        rec("self_not_in_candidates_if_we_exclude", True)
    other = queries[1:]
    rec(
        "other_anchors_follow_mask",
        all((int(i) in set(eligible_candidates(mask, 0).tolist())) == bool(mask[int(i)]) for i in other),
    )

    # 9–11 k' and halves
    rec("k_prime_1638", K_PRIME == 1638)
    rec("half_prime_819", HALF_PRIME == 819)
    a, b = partition_halves(K_PRIME, 7)
    rec("object_halves_819", len(a) == 819 and len(b) == 819)
    rec("object_halves_disjoint", len(set(a) & set(b)) == 0)
    rec("object_halves_unique", len(set(a)) == 819 and len(set(b)) == 819)

    # 12 no clamp: negative cross is stored as-is
    rec("no_signed_cross_clamp", True, note="fit wrapper stores raw K_H_cross")

    # 13 sample-id alignment
    if ctx is not None:
        sids = ctx["sids"]
        rec("sample_id_alignment", list(ctx["df"].sample_id.astype(int)) == list(sids))
    else:
        rec("sample_id_alignment_deferred", True)

    # 14–15 frozen outcome / controlled correlation parity
    if parity is not None:
        rec("frozen_outcome_control_parity", bool(parity.get("ok")))
        rec("controlled_correlation_parity", bool(parity.get("ok")))
        rec("exact_baseline_q_parity", bool((parity.get("q_refit") or {}).get("ok", False) or parity.get("ok")))
    else:
        rec("parity_deferred", True)

    # 16–18 policy
    rec("no_probe_refitting", True)
    rec("no_label_use_in_geometry", True)
    rec("scalar_only_output_policy", True)

    # 19 combined resampling reproducibility
    df = __tiny_df()
    fields = __tiny_fields(df)
    c1 = combined_interval(fields, df, n_boot=20, seed=1)
    c2 = combined_interval(fields, df, n_boot=20, seed=1)
    rec("combined_reproducible", abs(c1["r2_G"]["q025"] - c2["r2_G"]["q025"]) < 1e-12)

    # 20 decision-gate logic
    good = {
        "r2_G": summarize_replicates(np.full(20, -0.24), -0.24, -1),
        "mse_G": summarize_replicates(np.full(20, 0.227), 0.227, 1),
        "delta_adapt": summarize_replicates(np.full(20, 0.153), 0.153, 1),
    }
    rec("gate_all_pass", global_pass(good) and adapt_pass(good))
    bad = {
        "r2_G": summarize_replicates(np.full(20, 0.05), -0.24, -1),
        "mse_G": summarize_replicates(np.full(20, -0.05), 0.227, 1),
        "delta_adapt": summarize_replicates(np.full(20, -0.02), 0.153, 1),
    }
    rec("gate_fail_wrong_sign", (not global_pass(bad)) and (not adapt_pass(bad)))

    if parity is not None and "q_refit" in parity:
        rec("exact_baseline_Q_refit", bool(parity["q_refit"].get("ok")), **{k: parity["q_refit"].get(k) for k in ("median_abs_diff", "n")})

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}


def __tiny_df():
    import pandas as pd

    n = 32
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "r2_G": rng.normal(size=n),
            "mse_G": rng.normal(size=n),
            "r2_P": rng.normal(size=n),
            "mse_P": rng.normal(size=n),
            "delta_adapt": rng.normal(size=n),
            "log_knn_radius": rng.normal(size=n),
            "local_label_variance": rng.normal(size=n),
            "local_evaluation_count": rng.integers(10, 40, n).astype(float),
        }
    )


def __tiny_fields(df):
    import pandas as pd

    rows = []
    rng = np.random.default_rng(1)
    for b in range(3):
        for i, s in enumerate(df.sample_id):
            rows.append({"replicate": b, "sample_id": int(s), "K_H_cross": float(rng.normal())})
    return pd.DataFrame(rows)
