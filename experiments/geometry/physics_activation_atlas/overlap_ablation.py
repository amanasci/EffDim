"""Overlap diagnostics with explicit failure-reason breakdown (no prior gate)."""

from __future__ import annotations

import numpy as np

from .overlaps import fit_weighted_affine, soft_intersection, tangent_disagreement


FAILURE_REASONS = [
    "insufficient_overlap",
    "reconstruction_disagreement",
    "transition_failure",
    "tangent_disagreement",
    "rank_failure",
    "conditioning_failure",
    "extrapolation",
]


def evaluate_overlaps_ablation(
    membership_idx: np.ndarray,
    membership_w: np.ndarray,
    coords: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
    recon: dict[int, np.ndarray],
    jacobians_ok: dict[int, dict] | None = None,
    *,
    min_overlap_mass: float = 5.0,
    min_overlap_n: int = 8,
    max_pairs: int = 300,
    recon_thresh: float = 0.5,
    transition_thresh: float = 2.0,
    tangent_frac: float = 0.5,
    cond_thresh: float = 1e4,
) -> dict:
    """
    No prior/extrapolation gate by default; extrapolation flagged only if
    jacobians_ok provides per-chart extrapolative fraction > 0.5 (optional).
    """
    n, r = membership_idx.shape
    charts = sorted({int(c) for c in membership_idx.ravel().tolist() if c >= 0})
    W = {c: np.zeros(n, dtype=np.float64) for c in charts}
    for i in range(n):
        for j in range(r):
            c = int(membership_idx[i, j])
            if c >= 0:
                W[c][i] = membership_w[i, j]

    # all pairs for insufficient_overlap counting
    n_possible = len(charts) * (len(charts) - 1) // 2
    n_insufficient = 0
    pairs_mass = []
    for ia, ca in enumerate(charts):
        for cb in charts[ia + 1 :]:
            mass = soft_intersection(W[ca], W[cb])
            if mass < min_overlap_mass:
                n_insufficient += 1
            else:
                pairs_mass.append((ca, cb, mass))
    pairs_mass.sort(key=lambda t: -t[2])
    pairs_mass = pairs_mass[:max_pairs]

    rows = []
    reason_counts = {k: 0 for k in FAILURE_REASONS}
    reason_counts["insufficient_overlap"] = n_insufficient

    for ca, cb, mass in pairs_mass:
        mask = np.minimum(W[ca], W[cb]) > 1e-6
        reasons = []
        if mask.sum() < min_overlap_n:
            reasons.append("insufficient_overlap")
            reason_counts["insufficient_overlap"] += 1
            rows.append(
                {
                    "chart_a": int(ca),
                    "chart_b": int(cb),
                    "overlap_mass": float(mass),
                    "n_overlap": int(mask.sum()),
                    "valid": False,
                    "failure_reasons": reasons,
                }
            )
            continue
        w = np.minimum(W[ca], W[cb])[mask]
        Ua = coords[ca][mask]
        Ub = coords[cb][mask]
        ok = np.isfinite(Ua).all(axis=1) & np.isfinite(Ub).all(axis=1)
        if ok.sum() < min_overlap_n:
            reasons.append("insufficient_overlap")
            reason_counts["insufficient_overlap"] += 1
            rows.append(
                {
                    "chart_a": int(ca),
                    "chart_b": int(cb),
                    "overlap_mass": float(mass),
                    "n_overlap": int(ok.sum()),
                    "valid": False,
                    "failure_reasons": reasons,
                }
            )
            continue
        Ua, Ub, w = Ua[ok], Ub[ok], w[ok]
        n_ov = len(w)
        n_tr = max(4, int(0.7 * n_ov))
        A, b = fit_weighted_affine(Ua[:n_tr], Ub[:n_tr], w[:n_tr])
        pred = Ua[n_tr:] @ A + b if n_ov > n_tr else Ua @ A + b
        Ub_te = Ub[n_tr:] if n_ov > n_tr else Ub
        w_te = w[n_tr:] if n_ov > n_tr else w
        mse = float(np.average(np.sum((pred - Ub_te) ** 2, axis=1), weights=w_te))
        Ap, bp = fit_weighted_affine(Ub[:n_tr], Ua[:n_tr], w[:n_tr])
        cyc = (Ua[:n_tr] @ A + b) @ Ap + bp
        cycle = float(np.average(np.sum((cyc - Ua[:n_tr]) ** 2, axis=1), weights=w[:n_tr]))
        ra = recon[ca][mask][ok]
        rb = recon[cb][mask][ok]
        recon_dis = float(np.average(np.linalg.norm(ra - rb, axis=1), weights=w))
        d_tan = tangent_disagreement(bases[ca], bases[cb])
        cond = float(np.linalg.cond(A))
        d = bases[ca].shape[1]

        # rank / conditioning from optional per-chart jacobian summary
        rank_fail = False
        cond_fail = False
        extrap = False
        if jacobians_ok is not None:
            for c in (ca, cb):
                js = jacobians_ok.get(c, {})
                if js.get("frac_full_rank", 1.0) < 0.8:
                    rank_fail = True
                if js.get("median_condition", 0.0) > cond_thresh:
                    cond_fail = True
                if js.get("frac_extrapolative", 0.0) > 0.5:
                    extrap = True

        if recon_dis >= recon_thresh:
            reasons.append("reconstruction_disagreement")
        if (not np.isfinite(mse)) or mse >= transition_thresh or cycle >= transition_thresh:
            reasons.append("transition_failure")
        if d_tan >= tangent_frac * d:
            reasons.append("tangent_disagreement")
        if rank_fail:
            reasons.append("rank_failure")
        if cond_fail or cond >= cond_thresh:
            reasons.append("conditioning_failure")
        if extrap:
            reasons.append("extrapolation")

        for rsn in reasons:
            reason_counts[rsn] += 1

        valid = len(reasons) == 0
        rows.append(
            {
                "chart_a": int(ca),
                "chart_b": int(cb),
                "overlap_mass": float(mass),
                "n_overlap": int(ok.sum()),
                "transition_mse": mse,
                "cycle_mse": cycle,
                "recon_disagreement": recon_dis,
                "tangent_disagreement": d_tan,
                "affine_cond": cond,
                "valid": valid,
                "failure_reasons": reasons,
            }
        )

    n_eval = len(rows)
    n_valid = sum(1 for r in rows if r["valid"])
    return {
        "n_possible_pairs": n_possible,
        "n_pairs_evaluated": n_eval,
        "n_valid": n_valid,
        "valid_frac": float(n_valid / max(n_eval, 1)),
        "failure_counts": reason_counts,
        "pairs": rows,
        "thresholds": {
            "min_overlap_mass": min_overlap_mass,
            "recon_thresh": recon_thresh,
            "transition_thresh": transition_thresh,
            "tangent_frac": tangent_frac,
            "cond_thresh": cond_thresh,
            "prior_gate": False,
        },
        "mean_recon_disagreement": float(np.nanmean([r.get("recon_disagreement", np.nan) for r in rows])),
        "mean_transition_mse": float(np.nanmean([r.get("transition_mse", np.nan) for r in rows])),
        "mean_tangent_disagreement": float(np.nanmean([r.get("tangent_disagreement", np.nan) for r in rows])),
    }
