"""Given a fixture with real, dimension-stable curvature, can the estimator rank it at
``d = 20`` once the neighbourhood is genuinely LOCAL?

**Why this is the decisive cell.** Three separate things have been conflated in every
``d = 20`` dead end this milestone has recorded, and the ridge fixture plus this sweep
separates the last two:

    1. the FIXTURE has no rankable curvature -- true of every sealed control (``flat``:
       ``II = 0``; ``sphere``: ``||H||`` constant; ``saddle``: ``II`` constant so ``||H||``
       moves only via the metric), and true at high ``d`` of ANY separable surface, whose
       ``||D^2 f||_F`` concentrates like ``1/sqrt(d)``.
       ``varying_ii_controls.make_ridge_graph_control`` removes this: its Hessian is rank
       one, so curvature variation is dimension-independent (measured CV ~ 0.48, flat from
       ``d = 2`` to ``d = 40``).

    2. the NEIGHBOURHOOD is not local -- ``r / R >> 1``, so the quadratic model's ``O(r^3)``
       truncation error is order one and the estimator fits the wrong thing.

    3. the ESTIMATOR genuinely cannot see curvature at ``d = 20`` at any locality.

**The lever, and why it is clean.** For ``f(x) = A sin(w . x)`` with ``|w| = 1`` the radius of
curvature is ``R ~ 1 / (A freq^2)``, so frequency sets ``R`` directly. Meanwhile
``Var(w . x) = |w|^2 L^2 / 3 = L^2 / 3`` is INDEPENDENT OF ``d``, so lowering the frequency
buys locality WITHOUT shrinking the fixture's dynamic range the way every previous attempt
did. Amplitude and frequency scale ``||H||`` by a constant and rank statistics are invariant
to that, so no part of the comparison is flattered by the tuning.

At ``d = 20``, ``k = 30``, ``n = 5000``, ``L = 3``: ``r ~ L (k/n)^(1/d) ~ 2.32``, so

    freq 1.0  ->  R ~ 1.00,  r/R ~ 2.32   (neighbourhood dwarfs the curvature scale)
    freq 0.5  ->  R ~ 4.00,  r/R ~ 0.58
    freq 0.3  ->  R ~ 11.1,  r/R ~ 0.21   (genuinely local)
    freq 0.2  ->  R ~ 25.0,  r/R ~ 0.09

**How to read the result.** If ``rho`` climbs monotonically as frequency falls, reading (3) is
refuted: the estimator CAN rank curvature at ``d = 20`` and every previous zero was readings
(1) and (2) compounded. If ``rho`` stays at zero even at ``r/R ~ 0.1`` on a fixture with ~38x
dynamic range, reading (3) survives and ``d = 20`` is out of reach for this estimator family
-- which is what the published dimension-dependent minimax rates predict.

Both estimators run, for the same reason as in ``varying_ii_teacher_sweep_run.py``:
``centroid_mean_curvature`` estimates only ``H``'s trace (one unknown from ``k`` samples)
while ``quadric_mean_curvature`` fits ``d(d+1)/2 = 210`` coefficients, so their sample
complexities differ by two orders of magnitude and neither alone separates "unrankable" from
"undersampled".

    python notebooks/diagnostics/ridge_frequency_sweep_run.py --smoke
    python notebooks/diagnostics/ridge_frequency_sweep_run.py --d 20 --freqs 1.0 0.5 0.3 0.2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from pu_manifold import curvature_probe
from pu_manifold import varying_ii_controls as vic

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "03.2_ridge_frequency_sweep.jsonl"


def _knn_radius(X: np.ndarray, k: int) -> float:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    return float(np.median(dist[:, -1]))


def _axes(H_est_vec: np.ndarray, fx: Dict[str, Any]) -> Dict[str, float]:
    h_est = np.linalg.norm(H_est_vec, axis=-1)
    h_true = np.asarray(fx["H_norm"], dtype=np.float64)
    Hv = np.asarray(fx["H_vec"], dtype=np.float64)
    num = (H_est_vec * Hv).sum(1)
    den = np.maximum(np.linalg.norm(H_est_vec, axis=1) * np.linalg.norm(Hv, axis=1), 1e-12)
    return {
        "rho": float(spearmanr(h_est, h_true).statistic),
        "median_cosine": float(np.median(num / den)),
        "median_relative_error": float(curvature_probe.median_relative_error(h_est, h_true)),
    }


def run_cell(freq: float, n: int, k: int, d: int, D: int, seed: int,
             amplitude: float, domain_radius: float, run_quadric: bool,
             phase: float = 0.0) -> Dict[str, Any]:
    t0 = time.monotonic()
    fx = vic.make_ridge_graph_control(
        n=n, d=d, D=D, seed=seed, amplitude=amplitude,
        frequency=freq, domain_radius=domain_radius, phase=phase,
    )
    h_true = np.asarray(fx["H_norm"], dtype=np.float64)
    p05, p95 = float(np.percentile(h_true, 5)), float(np.percentile(h_true, 95))

    # r/R measured in the fixture's own (scaled) coordinates, so it is comparable to the
    # r/R figures spike 002 recorded rather than to raw generator units.
    r = _knn_radius(np.asarray(fx["X"]), k)
    R_curv = 1.0 / max(np.median(h_true), 1e-12)

    cent = curvature_probe.centroid_mean_curvature(fx["X"], k=k, d=d)
    out: Dict[str, Any] = {
        "kind": "ridge_frequency",
        "frequency": freq, "amplitude": amplitude, "domain_radius": domain_radius,
        "phase": float(phase),
        "n": n, "k": k, "d": d, "D": D, "seed": seed,
        "ii_cv": fx["ii_variation"]["hess_fro_cv"],
        "h_true_spread": float(p95 / p05) if p05 > 0 else float("inf"),
        "h_true_median": float(np.median(h_true)),
        "knn_radius": r,
        "R_curvature": R_curv,
        "r_over_R": r / R_curv,
        "centroid": _axes(cent, fx),
        "curvature_convention": vic.CURVATURE_CONVENTION,
    }
    if run_quadric:
        q = curvature_probe.quadric_mean_curvature(fx["X"], k=k, d=d)
        out["quadric"] = _axes(q["H_vec"], fx)
        out["quadric_coefficient_deficit"] = int(q["coefficient_deficit"])
    out["wallclock_s"] = time.monotonic() - t0
    return out


def _header(with_quadric: bool) -> None:
    head = (f"{'freq':>6} {'IIcv':>6} {'spread':>8} {'r':>7} {'R':>7} {'r/R':>7} | "
            f"{'cent rho':>9} {'cMRE':>8} {'ccos':>7}")
    if with_quadric:
        head += f" | {'quad rho':>9} {'qcos':>7}"
    print(head)


def _row(r: Dict[str, Any]) -> None:
    c = r["centroid"]
    line = (f"{r['frequency']:>6.2f} {r['ii_cv']:>6.3f} {r['h_true_spread']:>8.1f} "
            f"{r['knn_radius']:>7.3f} {r['R_curvature']:>7.3f} {r['r_over_R']:>7.3f} | "
            f"{c['rho']:>+9.4f} {c['median_relative_error']:>8.3f} {c['median_cosine']:>+7.3f}")
    if "quadric" in r:
        q = r["quadric"]
        line += f" | {q['rho']:>+9.4f} {q['median_cosine']:>+7.3f}"
    print(line)


def summarize(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("READ-OUT")
    print("=" * 78)
    best = max(records, key=lambda r: r["centroid"]["rho"])
    most_local = min(records, key=lambda r: r["r_over_R"])
    print(f"  best centroid rho   = {best['centroid']['rho']:+.4f} at freq {best['frequency']}"
          f" (r/R {best['r_over_R']:.3f})")
    print(f"  most local cell     = r/R {most_local['r_over_R']:.3f} at freq"
          f" {most_local['frequency']}, rho {most_local['centroid']['rho']:+.4f}")
    rhos = [r["centroid"]["rho"] for r in records]
    rr = [r["r_over_R"] for r in records]
    if len(records) >= 3:
        trend = float(spearmanr(rr, rhos).statistic)
        print(f"  rank corr(r/R, rho) = {trend:+.4f}  (strongly negative => locality is the lever)")
    print()
    if max(rhos) > 0.5:
        print("  ESTIMATOR VINDICATED AT d=20. On a fixture with dimension-stable curvature and")
        print("  a genuinely local neighbourhood, curvature RANKS at d=20. Every previous zero")
        print("  at this dimension was the fixture and the locality compounded, not a hard")
        print("  limit of curvature estimation. Spike 002's open question closes.")
    elif max(rhos) > 0.25:
        print("  PARTIAL. Ranking recovers well above the sealed controls' zero but not to a")
        print("  usable level. Locality helps and is not sufficient; report as a bound.")
    else:
        print("  DEAD END SURVIVES. Even with dimension-stable curvature and r/R well below 1,")
        print("  ranking does not recover at d=20. That is consistent with the published")
        print("  dimension-dependent minimax rates and points at sample complexity as the")
        print("  binding constraint rather than fixture design.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--ambient", type=int, default=28)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--freqs", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1])
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--domain-radius", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--phase", type=float, default=0.0,
                   help="pi/2 centres curvature near its maximum, making spread tunable by "
                        "frequency*domain_radius. See make_ridge_graph_control.")
    p.add_argument("--with-quadric", action="store_true",
                   help="also run the local quadric fit (210 coefficients at d=20; slow)")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    if a.smoke:
        print("SMOKE: n=600, d=6, k=30 -- proves the path runs, measures nothing.\n")
        _header(False)
        for f in [1.0, 0.3]:
            _row(run_cell(f, 600, 30, 6, 14, a.seed, a.amplitude, a.domain_radius, False, a.phase))
        return

    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Ridge frequency sweep -- d={a.d}, D={a.ambient}, n={a.n}, k={a.k}")
    print("=" * 78)
    print("Lowering frequency raises the radius of curvature R, buying locality WITHOUT")
    print("shrinking dynamic range: Var(w.x) = L^2/3 is independent of d and of frequency.")
    print(f"record_path = {record_path}\n")

    _header(a.with_quadric)
    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for f in a.freqs:
            r = run_cell(f, a.n, a.k, a.d, a.ambient, a.seed,
                         a.amplitude, a.domain_radius, a.with_quadric, a.phase)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            records.append(r)
            _row(r)

    summarize(records)


if __name__ == "__main__":
    main()
