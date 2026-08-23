"""Can ANY estimator rank curvature at ``d=20``, on a fixture that has curvature to rank?

**The question this settles.** Spike 002 measured the local-polynomial teacher at ``rho ~ 0``
on the ``d=20`` saddle and filed the result PARTIAL, because the saddle might not be able to
show ordering at all. That doubt was never resolved: on a graph ``M = {(x, f(x))}`` the mean
curvature is ``H = tr_g(II)``, and the saddle's ``II = diag(signs)`` is CONSTANT, so its
``||H||`` varies only through the metric tilt ``1/(1 + |grad|^2)``. Every sealed control has
the same defect or worse -- ``flat`` has ``II = 0``, ``sphere`` has ``||H||`` constant and its
sealed ``d=4`` record literally stores ``rho: None`` because rank is undefined there.

So a ``rho ~ 0`` at ``d=20`` has always had two readings that no measurement could separate:
the estimator cannot see curvature at that dimension, or the fixture has no rankable curvature
in it. This runner separates them by holding the estimator, ``d``, ``D``, ``n`` and ``k``
fixed and varying ONLY the surface, across
:data:`pu_manifold.varying_ii_controls.FAMILIES`:

    quadratic_saddle  minimal,     ``II`` constant   -- the sealed fixture, reproduced exactly
    quadratic_bowl    NON-minimal, ``II`` constant   -- tests "is minimality the problem?"
    quadratic_aniso   NON-minimal, ``II`` constant   -- same, with a wide eigenvalue spread
    cubic             non-minimal, ``II`` VARIES
    sine              non-minimal, ``II`` VARIES, and bounded

The three quadratics all score ``hess_fro_cv == 0`` exactly and differ only in trace and
anisotropy, so they isolate minimality. The last two isolate varying ``II``. If ``rho``
recovers on ``cubic``/``sine`` while staying at zero across all three quadratics, then the
``d=20`` dead end is a property of the CONTROL SUITE and not of curvature estimation, and
spike 002's open question closes.

**Both estimators are run.** ``centroid_mean_curvature`` is D-05's gating estimator (it
estimates only ``H``'s trace -- one unknown from ``k`` samples) and ``quadric_mean_curvature``
is the non-gating local quadric fit (``d(d+1)/2 = 210`` coefficients at ``d=20``). Their
sample complexities differ by two orders of magnitude, so reporting one without the other
would confound "this surface is unrankable" with "this estimator is undersampled".

``D=28`` rather than 768 throughout: spike 002 Part A measured the teacher's four fidelity
axes to be D-invariant to ``1.288e-14`` with a 204x speedup, and zero-padding is totally
geodesic so the manifold's own geometry is unchanged. That licence applies to POINT-CLOUD
estimators only -- it does NOT transfer to a decoder, whose input layer depends on ``D``.

    python notebooks/diagnostics/varying_ii_teacher_sweep_run.py --smoke
    python notebooks/diagnostics/varying_ii_teacher_sweep_run.py --n 5000 --k 30 231
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

from pu_manifold import curvature_probe
from pu_manifold import varying_ii_controls as vic

CACHE_DIR = NOTEBOOK_ROOT / ".cache"
DEFAULT_RECORD = CACHE_DIR / "03.2_varying_ii_teacher_sweep.jsonl"

D_INTRINSIC = 20
AMBIENT = 28
FIXTURE_SEED = 20260816


def _axes(H_est_vec: np.ndarray, fx: Dict[str, Any]) -> Dict[str, float]:
    h_est = np.linalg.norm(H_est_vec, axis=-1)
    h_true = np.asarray(fx["H_norm"], dtype=np.float64)
    H_true_vec = np.asarray(fx["H_vec"], dtype=np.float64)
    num = (H_est_vec * H_true_vec).sum(1)
    den = np.maximum(np.linalg.norm(H_est_vec, axis=1) * np.linalg.norm(H_true_vec, axis=1), 1e-12)
    return {
        "rho": float(spearmanr(h_est, h_true).statistic),
        "median_cosine": float(np.median(num / den)),
        "median_relative_error": float(curvature_probe.median_relative_error(h_est, h_true)),
        "median_ratio": float(np.median(h_est / np.maximum(h_true, 1e-12))),
    }


def run_cell(name: str, n: int, k: int, d: int, D: int, seed: int) -> Dict[str, Any]:
    t0 = time.monotonic()
    fx = vic.FAMILIES[name](n, d, D, seed)
    h_true = np.asarray(fx["H_norm"], dtype=np.float64)
    p05, p95 = float(np.percentile(h_true, 5)), float(np.percentile(h_true, 95))

    t1 = time.monotonic()
    H_cent = curvature_probe.centroid_mean_curvature(fx["X"], k=k, d=d)
    t_cent = time.monotonic() - t1

    t2 = time.monotonic()
    quad = curvature_probe.quadric_mean_curvature(fx["X"], k=k, d=d)
    t_quad = time.monotonic() - t2

    return {
        "kind": "varying_ii_teacher",
        "family": name,
        "n": n, "k": k, "d": d, "D": D, "seed": seed,
        "ii_variation": fx["ii_variation"],
        "trace": fx.get("trace"),
        "h_true_p05": p05, "h_true_p50": float(np.percentile(h_true, 50)), "h_true_p95": p95,
        "h_true_spread": float(p95 / p05) if p05 > 0 else float("inf"),
        "centroid": _axes(H_cent, fx),
        "quadric": _axes(quad["H_vec"], fx),
        "quadric_underdetermined": bool(quad["underdetermined"]),
        "quadric_coefficient_deficit": int(quad["coefficient_deficit"]),
        "n_coefficients": int(quad["n_coefficients"]),
        "t_centroid_s": t_cent, "t_quadric_s": t_quad,
        "wallclock_s": time.monotonic() - t0,
        "curvature_convention": vic.CURVATURE_CONVENTION,
    }


def _header() -> None:
    print(f"{'family':>17} {'IIcv':>6} {'spread':>8} | {'cent rho':>9} {'cMRE':>8} | "
          f"{'quad rho':>9} {'qMRE':>9} {'qcos':>7} | {'def':>4} {'s':>5}")


def _row(r: Dict[str, Any]) -> None:
    c, q = r["centroid"], r["quadric"]
    print(f"{r['family']:>17} {r['ii_variation']['hess_fro_cv']:>6.3f} "
          f"{r['h_true_spread']:>8.2f} | {c['rho']:>+9.4f} {c['median_relative_error']:>8.3f} | "
          f"{q['rho']:>+9.4f} {q['median_relative_error']:>9.3f} {q['median_cosine']:>+7.3f} | "
          f"{r['quadric_coefficient_deficit']:>4} {r['wallclock_s']:>5.0f}")


def summarize(records: List[Dict[str, Any]]) -> None:
    const = [r for r in records if r["ii_variation"]["hess_fro_cv"] < 1e-9]
    vary = [r for r in records if r["ii_variation"]["hess_fro_cv"] >= 1e-9]
    print("\n" + "=" * 78)
    print("READ-OUT")
    print("=" * 78)
    for label, group in (("constant II (every quadratic)", const), ("varying II", vary)):
        if not group:
            continue
        cr = [r["centroid"]["rho"] for r in group]
        qr = [r["quadric"]["rho"] for r in group]
        print(f"  {label}:")
        print(f"      centroid rho  min {min(cr):+.4f}  max {max(cr):+.4f}  mean {np.mean(cr):+.4f}")
        print(f"      quadric  rho  min {min(qr):+.4f}  max {max(qr):+.4f}  mean {np.mean(qr):+.4f}")
    if const and vary:
        best_const = max(max(abs(r["centroid"]["rho"]) for r in const),
                         max(abs(r["quadric"]["rho"]) for r in const))
        best_vary = max(max(r["centroid"]["rho"] for r in vary),
                        max(r["quadric"]["rho"] for r in vary))
        print()
        if best_vary > 0.5 and best_const < 0.3:
            print("  FIXTURE, NOT ESTIMATOR. At the same d, D, n and k, curvature ranks on a")
            print("  surface whose second fundamental form varies and does not rank on any")
            print("  surface where it is constant. The d=20 dead end recorded against the")
            print("  sealed saddle is a property of the CONTROL SUITE: none of flat, sphere or")
            print("  saddle has rankable curvature, so none of them could ever have produced a")
            print("  positive result, whatever the estimator did.")
        elif best_vary < 0.3:
            print("  NOT THE FIXTURE. Even a surface with genuinely varying curvature fails to")
            print("  rank at this d, so the dead end survives the fixture explanation and is")
            print("  about sample complexity at d=20.")
        else:
            print(f"  MIXED. best |rho| constant-II {best_const:+.4f}, varying-II {best_vary:+.4f}.")
            print("  Neither reading is clean; report both and do not collapse them.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--k", type=int, nargs="+", default=[30, 231])
    p.add_argument("--d", type=int, default=D_INTRINSIC)
    p.add_argument("--ambient", type=int, default=AMBIENT)
    p.add_argument("--seed", type=int, default=FIXTURE_SEED)
    p.add_argument("--families", type=str, nargs="+", default=list(vic.FAMILIES))
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoke:
        print("SMOKE: n=400, k=30, d=6 -- proves the path runs, measures nothing.\n")
        _header()
        for name in vic.FAMILIES:
            _row(run_cell(name, 400, 30, 6, 10, args.seed))
        return

    record_path = Path(args.record_path) if args.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Varying-II control sweep -- d={args.d}, D={args.ambient}, n={args.n}")
    print("=" * 78)
    print(f"quadric fit needs d(d+1)/2 = {args.d * (args.d + 1) // 2} coefficients")
    print(f"record_path = {record_path}\n")

    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for k in args.k:
            print(f"--- k = {k}")
            _header()
            for name in args.families:
                r = run_cell(name, args.n, k, args.d, args.ambient, args.seed)
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                records.append(r)
                _row(r)
            print()

    summarize(records)


if __name__ == "__main__":
    main()
