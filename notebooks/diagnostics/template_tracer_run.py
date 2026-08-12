"""
notebooks/diagnostics/template_tracer_run.py -- Phase 02.7 Task 2: the whole front end,
one path, one cloud, both metrics.

One synthetic S^1 cloud, lifted to production dimensionality D=768, measured under BOTH
distance metrics (Euclidean and graph geodesic), read through `ripser` at `maxdim=2`
(D-11's cap), turned into a per-metric significant Betti vector (D-02's bootstrap band),
and classified by D-01's Betti-vector lookup (D-01, ratified exactly as written in
`02.7-CONTEXT.md` at this plan's Task 1 blocking checkpoint). One template, one k, one
draw, no sweep, no grid, no tally.

**This runner's own result is counted in NO D-13 tally and sets NO ratified constant
except compute feasibility.** Every constant below is named `PROVISIONAL_` precisely
because none of them is ratified -- plan `02.7-08` ratifies the real values against this
phase's own compute-cost research; this runner exists to prove the pipeline has no
architectural dead end, not to produce a scored result. `d_hat` is likewise a placeholder
local value standing in for the real estimator route, which plan `02.7-03` supplies.

Invoke:
    .venv/bin/python notebooks/diagnostics/template_tracer_run.py --smoke
    .venv/bin/python notebooks/diagnostics/template_tracer_run.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import (  # noqa: E402
    confidence_band,
    geodesic_graph,
    persistence_probe,
    template_decision,
    template_immersion,
)

PROVISIONAL_N_PH = 300
PROVISIONAL_B = 10
PROVISIONAL_ALPHA = 0.05
PROVISIONAL_K = 15
TRACER_SEED = 20260811

AMBIENT_D = 768
"""Production dimensionality -- D-15's required grid level, and this plan's own
production-dimensionality acceptance criterion."""

TRACER_TEMPLATE = "S1"
TRACER_NOISE = 0.02
TRACER_WARP_PARAMS = {"strength": 0.05, "freq": 2.0, "seed": TRACER_SEED}


def _run_metric_arm(name: str, D: np.ndarray, b_boot: int) -> dict:
    """One metric's full read: persistence diagram at maxdim=2, one bootstrap band per
    degree, the significant Betti vector, and the template call. Prints wall-clock per
    `ripser` call as it goes.
    """
    t0 = time.monotonic()
    dgms = persistence_probe.persistence_diagram(D, maxdim=2)
    dgm_wallclock_s = time.monotonic() - t0
    print(f"  [{name}] persistence_diagram(maxdim=2) wallclock_s={dgm_wallclock_s:.2f}")

    bands = []
    for degree in range(3):
        t1 = time.monotonic()
        band_info = confidence_band.bootstrap_band(
            D, degree, b_boot, PROVISIONAL_ALPHA, TRACER_SEED
        )
        band_wallclock_s = time.monotonic() - t1
        per_call_s = band_wallclock_s / (b_boot + 1)
        bands.append(band_info["band"])
        print(
            f"    [{name}] degree=H{degree} bootstrap_band wallclock_s={band_wallclock_s:.2f} "
            f"({b_boot + 1} ripser calls, {per_call_s:.3f}s/call)  "
            f"c_alpha={band_info['c_alpha']:.6f}  band={band_info['band']:.6f}"
        )

    betti = confidence_band.betti_vector(dgms, bands)

    # Runner constant standing in for the real d_hat estimator route -- plan 02.7-03
    # supplies it. TRACER_TEMPLATE's own d_true == 1, so this is the value the tracer's
    # own cloud should produce once the real estimator is wired in.
    d_hat_placeholder = 1
    label = template_decision.lookup(betti, d_hat_placeholder)

    print(
        f"  metric={name}  betti={betti}  d_hat={d_hat_placeholder}  "
        f"bands={[f'{b:.4f}' for b in bands]}  label={label!r}"
    )

    return {
        "metric": name,
        "betti": betti,
        "d_hat": d_hat_placeholder,
        "bands": bands,
        "label": label,
        "dgm_wallclock_s": dgm_wallclock_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: n=120, B=3 (D unchanged at 768) -- exercises every code path "
            "(immersion, both metric arms, ripser at maxdim=2, bootstrap bands, the "
            "lookup) in well under 90 seconds. This is the runner's automated <verify>."
        ),
    )
    args = parser.parse_args()

    n_ph = PROVISIONAL_N_PH
    b_boot = PROVISIONAL_B
    if args.smoke:
        n_ph = 120
        b_boot = 3

    print("=" * 78)
    print("Phase 02.7 Task 2 tracer: one S^1 cloud, both metrics, one template label")
    print(
        "This runner's own result is counted in no D-13 tally and sets no ratified "
        "constant except compute feasibility -- every PROVISIONAL_ constant here is "
        "exactly that: provisional, ratified for real in plan 02.7-08."
    )
    print(
        f"n_ph={n_ph}  D={AMBIENT_D}  k={PROVISIONAL_K}  B={b_boot}  "
        f"alpha={PROVISIONAL_ALPHA}  seed={TRACER_SEED}  smoke={args.smoke}"
    )
    print("=" * 78)

    t_immerse0 = time.monotonic()
    cloud = template_immersion.immerse(
        TRACER_TEMPLATE, n_ph, AMBIENT_D, TRACER_NOISE, TRACER_SEED, TRACER_WARP_PARAMS
    )
    immerse_wallclock_s = time.monotonic() - t_immerse0
    points = cloud["points"]
    print(
        f"immersed cloud: template={cloud['template']}  d_true={cloud['d_true']}  "
        f"points.shape={points.shape}  wallclock_s={immerse_wallclock_s:.2f}"
    )

    D_euclidean, _ = persistence_probe.cloud_distance_matrix(points, prescale=False)

    t_geo0 = time.monotonic()
    D_geodesic, geo_readout = geodesic_graph.geodesic_distance_matrix(points, PROVISIONAL_K)
    geo_wallclock_s = time.monotonic() - t_geo0
    print(
        f"geodesic graph: n_components={geo_readout['n_components']}  "
        f"largest_size={geo_readout['largest_size']}  "
        f"dropped_fraction={geo_readout['dropped_fraction']:.4f}  "
        f"build_wallclock_s={geo_wallclock_s:.2f}"
    )

    print()
    print("euclidean arm:")
    result_euclidean = _run_metric_arm("euclidean", D_euclidean, b_boot)

    print()
    print("geodesic arm:")
    result_geodesic = _run_metric_arm("geodesic", D_geodesic, b_boot)

    print()
    print("=" * 78)
    print("Summary (no D-13 tally -- this is an engineering integration check only):")
    for result in (result_euclidean, result_geodesic):
        print(
            f"  metric={result['metric']}  betti={result['betti']}  "
            f"d_hat={result['d_hat']}  label={result['label']!r}"
        )
    print("=" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
