"""Plan 08-07 Task 4 — an exact permutation p-value for every rho on the record.

DIAGNOSTIC ONLY. Gates nothing. Touches no frozen constant, appends no row to any sealed record,
reopens no verdict.

Why the record has none. The per-point MKNN statistic is `j/k` for integer `j`, so at
`HEADLINE_K = 20` it takes at most 21 distinct values across 10,000 points -- measured 15.
`scipy.stats.spearmanr`'s asymptotic p assumes no ties and is invalid here, which is why Phase 7
froze a permutation route instead: threshold clearance at `NULL_QUANTILE_PER_TAIL = 0.975` read on
both tails, the Bonferroni equivalent of one two-sided 0.05 test. The null DRAWS are not stored,
so an exact p is not recoverable from the record and has to be recomputed. Threshold clearance is
a correct answer to a reviewer but not a sufficient one.

Two nulls, matched to the statistic. Getting this wrong is the failure mode this runner exists to
avoid:

  * PLAIN unrestricted permutation -- for the raw and diagnostic rho: `spearman(||H||, MKNN)` at
    every `d`, the three 07.1 seed fields, `spearman(density, ||H||)`, `spearman(density, MKNN)`,
    and the `MKNN_K_GRID` sensitivity grid.
  * WITHIN-DENSITY-STRATUM permutation -- 07.1's own `PERMUTATION_SCHEME_RULE`, `h` and `m`
    permuted independently inside each stratum -- for `partial_rho_density_controlled`. A plain
    permutation p for the partial would ignore the density structure the statistic exists to
    control for, and would be WRONG. This runner refuses to compute one.

Ranks are permuted directly rather than recomputed inside the loop:
`rankdata(x)[perm] == rankdata(x[perm])`, already pinned by `density_stratified_null`'s own
`test_rank_permutation_equivariance`.

Reporting rule. `p = (1 + #{null at least as extreme}) / (N + 1)`. Where zero draws reach the
observed value that is the RESOLUTION FLOOR, not a measurement: every such row carries
`p_is_floor: true` and must be reported as `< 1/(N+1)`, never as `=`.

Two self-validations, both printed, both failing loudly:
  1. plain arm -- the recomputed observed rho equals `07_crossmodal_curvature.jsonl`'s
     `observed_rho` to 1e-9 at every `d`.
  2. stratified arm -- at `density_stratified_null.PERMUTATION_SEED` and `N_PERMUTATIONS`, this
     file's replicated null band is compared against the SEALED
     `density_stratified_null.stratified_partial_null`'s own `null_low`/`null_high`. The check is
     a TOLERANCE check, not bit-equality: the two loops consume their RNG in different orders, so
     the draws differ even at an identical seed, and at `N = 1000` a 2.5% quantile is itself noisy.
     What is being established is that the replicated scheme has the same null distribution as the
     sealed one -- which is what licenses reading a p off it.

Usage:
    python notebooks/diagnostics/08_exact_permutation_p_run.py
    python notebooks/diagnostics/08_exact_permutation_p_run.py --threads 4 --n-plain 20000
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


_THREADS = _flag_value_from_argv("--threads", sys.argv)
if _THREADS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(int(_THREADS))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pu_manifold import cache  # noqa: E402
from pu_manifold import cross_split_curvature as csc  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe as cp  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402

RECORD_STEM = "08_exact_permutation_p"
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
S_GRID = (10, 20, 50)
"""The stratum counts 07.1 reports its partial at. Re-declared here rather than imported, per this
milestone's freeze-boundary convention -- this file gates nothing, so nothing crosses either way."""

BAND_TOLERANCE = 0.25
"""Relative tolerance on the self-validation band comparison. Generous by design: at
`N_PERMUTATIONS = 1000` an empirical 2.5% quantile carries real sampling error, and the check is
for distributional agreement, not for a reproduced draw sequence."""


def _run_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def plain_permutation_p(x, y, n_perm, seed, tail, batch=4000):
    """Exact one-sided permutation p for Spearman under UNRESTRICTED permutation.

    `tail="neg"` counts `null <= observed` (the research hypothesis's direction for every
    curvature-alignment statistic); `tail="pos"` counts `null >= observed`.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    xc = rx - rx.mean()
    yc = ry - ry.mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    observed = float(xc @ yc / den)
    n = xc.shape[0]
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        idx = np.argsort(rng.random((b, n)), axis=1)
        draws = (xc[idx] @ yc) / den
        count += int((draws <= observed).sum() if tail == "neg" else (draws >= observed).sum())
        done += b
    return observed, (1 + count) / (n_perm + 1), count


def stratified_partial_p(h, m, density, n_strata, n_perm, seed, quantile=0.975, batch=1000):
    """Exact one-sided (negative-tail) p for `partial_rho_density_controlled` under 07.1's
    WITHIN-STRATUM scheme, plus the replicated band for the self-validation.

    Mirrors `density_stratified_null.stratified_partial_null`'s documented construction: strata
    from the sealed :func:`density_strata`; `rankdata` on `h`, `m` and `density` once outside the
    loop; the design matrix `[1, rank(density)]` FIXED across every resample; `h` and `m` each
    permuted independently within each stratum; the residual-Pearson statistic taken directly from
    the permuted rank vectors. The observed value comes from the sealed
    `cross_split_curvature.partial_spearman`, called not reimplemented, so it can never diverge
    from what every other call site reports.
    """
    labels = dsn.density_strata(density, n_strata)
    blocks = [np.flatnonzero(labels == s) for s in np.unique(labels)]
    rh = rankdata(h)
    rm = rankdata(m)
    rd = rankdata(density)
    n = rh.shape[0]
    A = np.column_stack([np.ones(n), rd])
    pinv = np.linalg.pinv(A)

    def residualise(R):
        return R - (R @ pinv.T) @ A.T

    observed = float(csc.partial_spearman(h, m, controls=density))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        Ph = np.tile(rh, (b, 1))
        Pm = np.tile(rm, (b, 1))
        for idx in blocks:
            s = idx.size
            Ph[:, idx] = rh[idx][np.argsort(rng.random((b, s)), axis=1)]
            Pm[:, idx] = rm[idx][np.argsort(rng.random((b, s)), axis=1)]
        eh = residualise(Ph)
        em = residualise(Pm)
        eh -= eh.mean(1, keepdims=True)
        em -= em.mean(1, keepdims=True)
        draws[done:done + b] = ((eh * em).sum(1)
                                / (np.linalg.norm(eh, axis=1) * np.linalg.norm(em, axis=1)))
        done += b
    count = int((draws <= observed).sum())
    return (observed, (1 + count) / (n_perm + 1), count,
            float(np.quantile(draws, 1.0 - quantile)), float(np.quantile(draws, quantile)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--n-plain", type=int, default=100_000)
    ap.add_argument("--n-strat", type=int, default=20_000)
    ap.add_argument("--record-path", default=None)
    args = ap.parse_args()

    record_path = (args.record_path if args.record_path is not None
                   else str(cache.cache_path(RECORD_STEM, "jsonl")))
    t_start = time.time()

    sub = np.load(cache.cache_path(SUBSAMPLE_STEM, "npz"))
    fields = np.load(cache.cache_path("07_crossmodal_curvature_fields", "npz"))
    seed_fields = np.load(cache.cache_path("07.1_seed_fields_d25", "npz"))
    density = 1.0 / cp.local_density_weights(sub["legacysurvey"], cc.DENSITY_K, cc.DENSITY_FIELD_D)
    mknn = {k: cc.per_point_mknn(sub["hsc"], sub["legacysurvey"], k) for k in cc.MKNN_K_GRID}
    print(f"per-point MKNN built for k in {tuple(cc.MKNN_K_GRID)}; "
          f"distinct at HEADLINE_K={cc.HEADLINE_K}: "
          f"{cc._relative_precision_distinct_count(mknn[cc.HEADLINE_K])}", flush=True)

    sealed = {}
    for line in open(cache.cache_path("07_crossmodal_curvature", "jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("row_kind") == "sweep" and r.get("d") is not None:
            sealed[int(r["d"])] = r

    run_commit = _run_commit()
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []

    def emit(**kw):
        kw.update(gates_nothing=True, run_commit=run_commit, timestamp=stamp)
        kw["p_is_floor"] = (kw.get("n_more_extreme") == 0)
        rows.append(kw)
        return kw

    def show(label, rho, p, floor_n):
        flag = "<" if p <= 1.0 / (floor_n + 1) + 1e-15 else "="
        print(f"  {label:<34} rho={rho:+.6f}  p {flag} {p:.2e}", flush=True)

    # ---- SELF-VALIDATION 1: the plain arm reproduces the sealed observed_rho -------------------
    print("\n== self-validation 1: recomputed rho vs the sealed record ==", flush=True)
    for d in cc.D_SWEEP:
        got = float(spearmanr(fields[f"h_norm_{d}"], mknn[cc.HEADLINE_K]).statistic)
        want = sealed[d]["observed_rho"]
        assert abs(got - want) < 1e-9, f"d={d}: recomputed {got!r} != sealed {want!r}"
        print(f"  d={d:<3} {got:+.9f} == sealed {want:+.9f}   OK", flush=True)

    # ---- SELF-VALIDATION 2: the replicated stratified band vs the sealed function --------------
    print(f"\n== self-validation 2: replicated band vs sealed stratified_partial_null "
          f"(N={dsn.N_PERMUTATIONS}, seed={dsn.PERMUTATION_SEED}, tolerance {BAND_TOLERANCE:.0%}) ==",
          flush=True)
    d_ref, s_ref = 20, 20
    ref = dsn.stratified_partial_null(
        fields[f"h_norm_{d_ref}"], mknn[cc.HEADLINE_K], density, s_ref,
        dsn.N_PERMUTATIONS, dsn.PERMUTATION_SEED, dsn.NULL_QUANTILE_PER_TAIL)
    _o, _p, _c, lo, hi = stratified_partial_p(
        fields[f"h_norm_{d_ref}"], mknn[cc.HEADLINE_K], density, s_ref,
        dsn.N_PERMUTATIONS, dsn.PERMUTATION_SEED, dsn.NULL_QUANTILE_PER_TAIL)
    for name, mine, theirs in (("null_low", lo, ref["null_low"]), ("null_high", hi, ref["null_high"])):
        rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
        print(f"  {name:<10} replicated {mine:+.6f}  sealed {theirs:+.6f}  rel diff {rel:.1%}",
              flush=True)
        assert rel < BAND_TOLERANCE, f"{name}: replicated {mine} vs sealed {theirs}, {rel:.1%} apart"
    assert abs(_o - ref["observed"]) < 1e-12, (_o, ref["observed"])
    print("  observed statistic identical (both call cross_split_curvature.partial_spearman)",
          flush=True)
    emit(row_kind="self_validation", stat="stratified_band_check", d=d_ref, S=s_ref,
         replicated_null_low=lo, replicated_null_high=hi,
         sealed_null_low=ref["null_low"], sealed_null_high=ref["null_high"],
         N=int(dsn.N_PERMUTATIONS), seed=int(dsn.PERMUTATION_SEED), n_more_extreme=None)

    NP, NS = args.n_plain, args.n_strat

    print(f"\n== A. raw spearman(||H||, MKNN) at HEADLINE_K -- PLAIN null, N={NP} ==", flush=True)
    for d in cc.D_SWEEP:
        rho, p, c = plain_permutation_p(fields[f"h_norm_{d}"], mknn[cc.HEADLINE_K], NP, 1, "neg")
        emit(row_kind="stat", stat="raw_spearman_h_mknn", d=int(d), k=int(cc.HEADLINE_K),
             rho=rho, p=p, N=NP, null="plain", tail="neg", n_more_extreme=c)
        show(f"d={d}", rho, p, NP)

    print(f"\n== B. 07.1 seed fields -- PLAIN null, N={NP} ==", flush=True)
    for name in sorted(f for f in seed_fields.files if f.startswith("h_norm_25_seed")):
        rho, p, c = plain_permutation_p(seed_fields[name], mknn[cc.HEADLINE_K], NP, 2, "neg")
        emit(row_kind="stat", stat="raw_spearman_h_mknn_seed", field=name, d=25,
             rho=rho, p=p, N=NP, null="plain", tail="neg", n_more_extreme=c)
        show(name, rho, p, NP)

    print(f"\n== C. density diagnostics -- PLAIN null, N={NP} ==", flush=True)
    for d in cc.D_SWEEP:
        rho, p, c = plain_permutation_p(density, fields[f"h_norm_{d}"], NP, 3, "pos")
        emit(row_kind="stat", stat="spearman_density_h", d=int(d),
             rho=rho, p=p, N=NP, null="plain", tail="pos", n_more_extreme=c)
        show(f"spearman(density, ||H||) d={d}", rho, p, NP)
    rho, p, c = plain_permutation_p(density, mknn[cc.HEADLINE_K], NP, 4, "neg")
    emit(row_kind="stat", stat="spearman_density_mknn", k=int(cc.HEADLINE_K),
         rho=rho, p=p, N=NP, null="plain", tail="neg", n_more_extreme=c)
    show("spearman(density, MKNN)", rho, p, NP)

    print(f"\n== D. MKNN_K_GRID sensitivity -- PLAIN null, N={NP} (non-gating) ==", flush=True)
    for d in cc.D_SWEEP:
        for k in cc.MKNN_K_GRID:
            rho, p, c = plain_permutation_p(fields[f"h_norm_{d}"], mknn[k], NP, 5, "neg")
            emit(row_kind="stat", stat="sensitivity_k", d=int(d), k=int(k),
                 rho=rho, p=p, N=NP, null="plain", tail="neg", n_more_extreme=c)
            show(f"d={d} k={k}", rho, p, NP)

    print(f"\n== E. partial_rho_density_controlled -- WITHIN-STRATUM null, N={NS} ==", flush=True)
    for d in cc.D_SWEEP:
        for S in S_GRID:
            rho, p, c, lo, hi = stratified_partial_p(
                fields[f"h_norm_{d}"], mknn[cc.HEADLINE_K], density, S, NS, dsn.PERMUTATION_SEED)
            emit(row_kind="stat", stat="partial_rho_density_controlled", d=int(d), S=int(S),
                 k=int(cc.HEADLINE_K), rho=rho, p=p, N=NS, null="within_density_stratum",
                 tail="neg", band_low=lo, band_high=hi, n_more_extreme=c)
            print(f"  d={d:<3} S={S:<3} partial={rho:+.6f}  p={p:.2e}  "
                  f"band=[{lo:+.6f},{hi:+.6f}]", flush=True)

    with open(record_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    n_floor = sum(1 for r in rows if r.get("p_is_floor"))
    print(f"\nwrote {len(rows)} rows to {record_path}  ({time.time() - t_start:.0f}s)")
    print(f"resolution floors: plain p >= {1/(NP+1):.2e}, stratified p >= {1/(NS+1):.2e}")
    print(f"{n_floor} rows sit AT the floor and carry p_is_floor=true -- report those as "
          f"'< floor', never as '= floor'")
    print("EXACT PERMUTATION P COMPLETE -- gates nothing, no frozen constant changed")


if __name__ == "__main__":
    main()
