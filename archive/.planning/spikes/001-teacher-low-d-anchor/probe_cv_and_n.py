"""Spike 001, follow-up probe -- isolates the two anchor failures.

Failure 1: saddle magnitude-ratio CV = 5.42 (d=2) / 7.62 (d=4) with a median ratio of ~1.00.
  Hypothesis: the mixed-sign saddle's trace passes through zero, so `||H_est|| / ||H_true||`
  divides by a near-zero denominator on the cancellation set. The scorer excludes only
  `||H_true|| <= MIN_TRUE_NORM`, and n_excluded was 0 -- so the whole cancellation set is in
  the statistic. If true, CV falls sharply as a quantile floor on `||H_true||` rises, while
  the median ratio barely moves. That would make CV a property of THIS fixture, not of the
  teacher, and the Swiss roll (whose H never changes sign) is the control: its CV is 0.009.

Failure 2: saddle d=4 rank rho = 0.738, below the 0.90 band.
  Hypothesis: it is `n`, not `d`. The anchor ran n=3000; the sealed protocol is n=10000. Under
  `r/R ~ (k/n)^(1/d)` (02.5-NOTE §1), tripling n at d=4 shrinks r/R by (1/3.33)^(1/4) = 0.76 --
  a real 24% reduction, unlike d=20 where the same change buys 11%. If rho climbs at n=10000,
  the anchor was undersampled rather than the teacher being wrong at d=4.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks" / "diagnostics"))

import synthetic_control_run as scr  # noqa: E402
from pu_manifold import curvature_probe, synthetic_controls  # noqa: E402

K = 30
SEED = 20260816
FLOORS = [0.0, 0.05, 0.10, 0.25, 0.50]


def run(d, D, n):
    fx = synthetic_controls.make_saddle_control(n=n, d=d, D=D, seed=SEED)
    X, H_true = fx["X"], fx["H_vec"]
    out = curvature_probe.quadric_mean_curvature(X, k=K, d=d)
    return X, H_true, out["H_vec"]


def report(tag, H_est, H_true):
    ne = np.linalg.norm(H_est, axis=-1)
    nt = np.linalg.norm(H_true, axis=-1)
    ratio = ne / nt
    rho = float(curvature_probe.spearman_gate_statistic(ne, nt))
    mre = float(curvature_probe.median_relative_error(ne, nt))

    print(f"\n--- {tag}")
    print(f"  ||H_true||: min={nt.min():.3e} p01={np.quantile(nt,0.01):.3e} "
          f"p10={np.quantile(nt,0.10):.3e} median={np.median(nt):.3e} max={nt.max():.3e}")
    print(f"  MIN_TRUE_NORM = {scr.MIN_TRUE_NORM:.3e}  -> excluded by scorer: "
          f"{int((nt <= scr.MIN_TRUE_NORM).sum())} of {nt.size}")
    print(f"  rank rho (all points) = {rho:.6f}   median relative error = {mre:.6f}")
    print("  ratio statistics as a quantile floor on ||H_true|| rises:")
    print(f"    {'floor':>6} {'kept':>6} {'median':>10} {'CV':>10} {'p99 ratio':>12}")
    for q in FLOORS:
        thr = np.quantile(nt, q)
        keep = nt >= thr
        r = ratio[keep]
        cv = float(np.std(r, ddof=1) / np.mean(r))
        print(f"    {q:>6.2f} {int(keep.sum()):>6d} {np.median(r):>10.4f} "
              f"{cv:>10.4f} {np.quantile(r, 0.99):>12.4f}")
    return rho


print("=" * 78)
print("PROBE A -- is the saddle CV the cancellation set?")
print("=" * 78)
X, Ht, He = run(d=2, D=8, n=3000)
report("saddle d=2, n=3000, k=30", He, Ht)
X, Ht, He = run(d=4, D=12, n=3000)
report("saddle d=4, n=3000, k=30", He, Ht)

print("\n" + "=" * 78)
print("PROBE B -- is d=4's rho about n, not d?")
print("=" * 78)
for n in (3000, 10000, 30000):
    X, Ht, He = run(d=4, D=12, n=n)
    rho = report(f"saddle d=4, n={n}, k=30", He, Ht)
    r_over_R = (K / n) ** (1.0 / 4)
    print(f"  predicted r/R ~ (k/n)^(1/d) = {r_over_R:.4f}")

print("\n" + "=" * 78)
print("PROBE C -- d=2 at the sealed n, for the same comparison")
print("=" * 78)
X, Ht, He = run(d=2, D=8, n=10000)
report("saddle d=2, n=10000, k=30", He, Ht)
