"""Experiments 1 and 2 on the sealed Phase 9 anchor tables. NOT PRE-REGISTERED. GATES NOTHING.

Runs on the cached anchor tables alone (no embeddings, no refit):

  1. How much of local R^2 a density model explains (linear log r, decile dummies, plus label
     variance and count), and the curvature partial after residualising on the DECILE model in
     place of the frozen linear log-r control. Freedman-Lane p on the decile design.
  2. Matched-radius anchor pairs: adjacent anchors in log r closer than EPS; sign concordance of
     delta-curvature vs delta-R^2, with a sign-flip permutation p.

Both instruments: decoder ||H_tan|| (Amendment 01 tables, d in 16/20/25/32) and the colleague's
K_H_cross (chart rank 12/16/20). All four labels.

Usage:
    python notebooks/diagnostics/09_density_anchor_tables_run.py \\
        --amend01-root notebooks/.cache/09-amend01 --colleague-root notebooks/.cache \\
        --record-path notebooks/.cache/09_density_anchor_tables.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

LABELS = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")
D_OURS = (16, 20, 25, 32)
D_HIS = (12, 16, 20)
EPS = 0.02
S_DECILES = 10


def resid_on(v, design):
    A = np.column_stack([np.ones(len(v))] + design)
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def decile_design(r, others, S):
    q = np.floor(rankdata(r) / (len(r) + 1) * S).astype(int)
    return [(q == j).astype(float) for j in range(1, S)] + [rankdata(c) for c in others]


def partial(x, y, design):
    return float(spearmanr(resid_on(rankdata(x), design), resid_on(rankdata(y), design)).statistic)


def perm_p(x, y, design, obs, rng, n_perm):
    xr, yr = rankdata(x), rankdata(y)
    A = np.column_stack([np.ones(len(x))] + design)
    fit = A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    res = yr - fit
    rx = resid_on(xr, design)
    cnt = sum(abs(spearmanr(rx, resid_on(fit + rng.permutation(res), design)).statistic) >= abs(obs) for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def r2_of_model(y, design):
    yr = rankdata(y)
    return float(1 - resid_on(yr, design).var() / yr.var())


def matched_pairs(curv, r2, logr, eps, rng, n_perm):
    order = np.argsort(logr)
    c, y, r = curv[order], r2[order], logr[order]
    keep = np.diff(r) < eps
    dc, dy = np.diff(c)[keep], np.diff(y)[keep]
    ok = (dc != 0) & (dy != 0)
    dc, dy = dc[ok], dy[ok]
    tau = float(2 * np.mean(np.sign(dc) == np.sign(dy)) - 1)
    null = np.array([2 * np.mean(np.sign(dc * rng.choice([-1, 1], size=dc.size)) == np.sign(dy)) - 1 for _ in range(n_perm)])
    p = (1 + int(np.sum(np.abs(null) >= abs(tau)))) / (n_perm + 1)
    return {"n_pairs": int(ok.sum()), "median_gap": float(np.median(np.diff(r)[keep])), "tau": tau, "p": float(p)}


def one(instr, d, lab, x, y, r, lv, cnt, rng, n_perm, out):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, r, lv, cnt = x[m], y[m], r[m], lv[m], cnt[m].astype(float)
    lin = [rankdata(r), rankdata(lv), rankdata(cnt)]
    dec = decile_design(r, [lv, cnt], S_DECILES)
    p_lin, p_dec = partial(x, y, lin), partial(x, y, dec)
    rec = {"experiment": 1, "instrument": instr, "d": d, "label": lab, "n": int(m.sum()),
           "rho_r2_logr": float(spearmanr(y, r).statistic), "rho_curv_logr": float(spearmanr(x, r).statistic),
           "r2_variance_explained_by": {
               "linear_logr": r2_of_model(y, [rankdata(r)]),
               "decile_logr": r2_of_model(y, decile_design(r, [], S_DECILES)),
               "decile_logr+labelvar": r2_of_model(y, decile_design(r, [lv], S_DECILES)),
               "decile_logr+labelvar+count": r2_of_model(y, dec)},
           "partial_linear3": p_lin, "partial_decile3": p_dec, "p_decile3": perm_p(x, y, dec, p_dec, rng, n_perm)}
    out.append(rec)
    out.append({"experiment": 2, "instrument": instr, "d": d, "label": lab, **matched_pairs(x, y, r, EPS, rng, n_perm)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amend01-root", default="notebooks/.cache/09-amend01")
    p.add_argument("--colleague-root", default="notebooks/.cache")
    p.add_argument("--record-path", default="notebooks/.cache/09_density_anchor_tables.json")
    p.add_argument("--n-permutations", type=int, default=5000)
    a = p.parse_args()
    rng = np.random.default_rng(20260910)
    out = []
    for d in D_OURS:
        for lab in LABELS:
            z = np.load(Path(a.amend01_root) / f"09_anchor_table_d{d}_{lab}.npz")
            one("decoder_H_tan", d, lab, z["H_tan_norm"], z["r2"], z["log_knn_radius"], z["local_label_variance"],
                z["local_evaluation_count"], rng, a.n_permutations, out)
    for d in D_HIS:
        z = np.load(Path(a.colleague_root) / f"09_colleague_anchor_table_d{d}.npz")
        for lab in LABELS:
            one("colleague_K_H_cross", d, lab, z["K_H_cross"], z[f"r2_{lab}"], z["log_knn_radius"],
                z[f"local_label_variance_{lab}"], z[f"local_evaluation_count_{lab}"], rng, a.n_permutations, out)
    Path(a.record_path).write_text(json.dumps({"note": "NOT PRE-REGISTERED; GATES NOTHING", "eps": EPS, "deciles": S_DECILES,
                                              "n_permutations": a.n_permutations, "rows": out}, indent=1))
    print("EXP1 instrument d label | rho(R2,logr) rho(curv,logr) | partial lin3 -> decile3 (p)")
    for r in out:
        if r["experiment"] == 1:
            print(f"  {r['instrument'][:8]} {r['d']:2d} {r['label']:15s} | {r['rho_r2_logr']:+.3f} {r['rho_curv_logr']:+.3f} | "
                  f"{r['partial_linear3']:+.3f} -> {r['partial_decile3']:+.3f} (p={r['p_decile3']:.4f})")
    print("EXP2 matched-radius pairs: n, tau, p")
    for r in out:
        if r["experiment"] == 2:
            print(f"  {r['instrument'][:8]} {r['d']:2d} {r['label']:15s} | n={r['n_pairs']} tau={r['tau']:+.3f} p={r['p']:.4f}")
    print(f"record -> {a.record_path}")


if __name__ == "__main__":
    main()
