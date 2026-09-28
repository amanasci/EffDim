"""
notebooks/diagnostics/ph_budget_calibration_run.py -- Phase 02.7 Plan 07: the H_2
compute-feasibility calibration at the real D = 768.

**This script measures and prints; it applies no bar, scores no cell and computes no
tally.** It answers one question with numbers instead of an estimate: does an unthresholded
`ripser(..., maxdim=2)` call, at this phase's real production dimensionality (D=768) with the
real random-orthogonal-lift-plus-warp immersion and real per-template noise, cost what
`02.7-RESEARCH.md`'s own D=3 measurements suggested it would -- and does the candidate
`n_ph` carry enough statistical power to resolve `H_2` at all? `02.7-RESEARCH.md`'s Pitfall 1
table and Open Question 2's worked ~8.25-hour grid arithmetic were both taken at D=3 on flat
fixtures; assumption A2 flags the D=768-plus-warp interaction as explicitly unverified. This
runner is that verification.

Every number below is printed with its template, metric, degree and `n` label attached, and
no figure is ever averaged across templates -- template-dependence (S^2 the expensive one,
per RESEARCH) is one of the findings under measurement here, not noise to smooth over.

This runner never imports the module that turns a Betti vector into a template label or a
three-way tally -- it has no access to that decision layer at all, so it structurally cannot
score a cell. Plan 02.7-08 ratifies `n_ph`, `B`, the grid levels, the dispersion bound and the
spread bound against these numbers, blind, in its own separate document.

Invoke:
    .venv/bin/python notebooks/diagnostics/ph_budget_calibration_run.py --smoke
    .venv/bin/python notebooks/diagnostics/ph_budget_calibration_run.py
"""

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import confidence_band, geodesic_graph, persistence_probe, template_immersion  # noqa: E402

# =============================================================================================
# Constants. Every one of these is an UNRATIFIED, this-runner-only measurement setting -- none
# of them is a D-15 grid level, none carries a "default" reading, and plan 02.7-08 is the sole
# place any of these numbers becomes a ratified constant. Where a value doubles as a stand-in
# for a not-yet-ratified level (WORST_NOISE, WARP_PARAMS, BALL_D, PROVISIONAL_K,
# CANDIDATE_N_PH, CANDIDATE_B), that is stated explicitly at its own definition.
# =============================================================================================

AMBIENT_D = 768
"""Production dimensionality -- the whole point of this calibration (02.7-RESEARCH.md's own
numbers were taken at D=3). Fixed, never varied by --smoke."""

CALL_TIMEOUT_S = 90.0
"""Hard per-call timeout ceiling for every `ripser`-touching call this runner makes, wrapped
via a forked-subprocess timeout (see `run_timed` below) so a cell that will not finish is
recorded as a timeout at this known bound rather than hanging silently -- the exact warning
sign 02.7-RESEARCH.md Pitfall 1 names. Matches RESEARCH's own D=3 alarm precedent; a deliberate
reuse, not a re-derivation, so the two are directly comparable."""

N_LADDER: Tuple[int, ...] = (150, 200, 250, 300, 400)
"""The plan's own minimum required ladder. Once a (template, metric) sequence hits a timeout
at some n, larger n values in the same sequence are SKIPPED (not attempted) and filled in by
extrapolation instead -- 02.7-07-PLAN.md critical note #11 explicitly prefers a smaller
measured ladder plus a stated extrapolation over an incomplete table or a timed-out run."""

EXTRAPOLATE_TO_N: Tuple[int, ...] = (500,)
"""Never measured directly in this run -- power-law-extrapolated only, from the fitted
log-log exponent of each (template, metric)'s own measured ladder points. Reported so
RESEARCH's own D=3 finding (S^2 did not finish at n=500 inside 90s) has a D=768 analogue to
compare against, without spending the wall-clock a full measurement at n=500 would cost if it
also times out."""

TEMPLATES: Tuple[str, ...] = ("S1", "S2", "T2", "ball")

BALL_D = 18
"""The 'ball' template's own intrinsic dimension for this calibration run only -- an
unratified proxy, not a D-15 grid level. Chosen to match this project's own PU d_hat estimate
range (~18-25, see PROJECT.md/STATE.md), so the calibration's ball cell is at least in the
right neighbourhood of the real regime rather than an arbitrary small d."""

WORST_NOISE = 0.05
"""The noise level this calibration uses for every cloud. Matches 02.7-RESEARCH.md's own
Pitfall 1 table noise level exactly (noise=0.05) so the D=768 numbers below are directly
comparable to the D=3 numbers they are being checked against -- not independently re-derived
as 'the worst noise this phase's grid will use', since D-15's own noise LEVELS are not yet
ratified. If plan 02.7-08 ratifies a higher noise level, this is not automatically the
worst case; stated here as a limitation, not hidden."""

WARP_PARAMS: Dict[str, float] = {"strength": 0.05, "freq": 1.0}
"""The exact (strength, freq) pair 02.7-05-SUMMARY.md measured all four templates PASS the
immersion Jacobian rank check at, at D=768 -- reused verbatim rather than re-swept, since this
calibration's job is compute cost and H_2 power, not re-verifying the immersion is a genuine
immersion (02.7-05 already did that)."""

CALIBRATION_SEED = 20260812

PROVISIONAL_K = 15
"""The same unratified provisional k the tracer (02.7-01) and every other phase runner uses
for the geodesic arm's kNN graph -- not D-06's ratified k, which plan 02.7-08 fixes."""

CANDIDATE_N_PH = 300
"""02.7-RESEARCH.md's own primary recommendation (Summary: 'Pin n_ph = 300') -- the value
this calibration's band-cost, whole-cell and H_2-power sections size against. Not ratified
here; plan 02.7-08 may choose differently in light of these numbers."""

CANDIDATE_B = 10
"""Matches the tracer's own PROVISIONAL_B (02.7-01) -- the B this calibration extrapolates
band cost and the H_2 power check against. Not ratified here."""

CANDIDATE_ALPHA = 0.05

BAND_SMALL_B = 3
"""The small B this calibration measures a real bootstrap_band call at, before extrapolating
linearly to CANDIDATE_B and to other candidate B values -- kept deliberately small so the
measurement itself is cheap; the extrapolation, not a second large-B measurement, is what
projects the candidate cost."""

BAND_FULL_TEMPLATES: Tuple[str, ...] = ("S1", "S2")
"""The two templates this calibration measures band cost for across ALL THREE degrees
(H0, H1, H2) -- the cheap and the RESEARCH-identified worst-case bookends. Running all four
templates at all three degrees would roughly double this section's own wall-clock for two
degrees (H0, H1) 02.7-04-SUMMARY.md and the tracer already show are comparatively cheap and
much less template-sensitive than H2 (the template-dependent cost driver per RESEARCH Pitfall
1); T2 and 'ball' still get their own H2-only measurement below (BAND_H2_ONLY_TEMPLATES), so
every template's dominant cost term is measured, just not every template's full 3-degree
sweep."""

BAND_H2_ONLY_TEMPLATES: Tuple[str, ...] = ("T2", "ball")
"""The remaining two templates, measured at H2 only (the template-sensitive degree) to keep
this section's wall-clock bounded while still reporting all four templates' H2 band cost --
never averaged across templates, per the plan's own instruction."""

H2_POWER_EXTRA_BALL_DIM = 18
"""The 'B^{d-2}' factor RESEARCH's Open Question 1 names for a known-H_2 fixture,
'S^2 x B^{d-2}' -- chosen so the fixture's total intrinsic dimension (2 + 18 = 20) sits in
this project's own ~18-25 PU d_hat estimate range, i.e. a genuinely PU-regime-shaped power
check, not an easy low-dimensional toy that would answer the open power question by
assumption rather than by measurement."""

H2_POWER_EXTRA_BALL_DIM_TEMPLATE_NATIVE = 0
"""A second, cheaper H2 power arm at the templates' own native intrinsic dimension (a plain
S^2 shell, d_true=2, vs a plain solid B^3, no product factor at all) -- added after this
session's first attempt at H2_POWER_EXTRA_BALL_DIM=18 (the PU-regime-shaped fixture) timed
out on BOTH arms even at a doubled ceiling (see the printed run). RESEARCH's Open Question 1
is specifically about whether the candidate n_ph resolves H_2 on THIS phase's own S^2/T^2
templates -- which are 2-dimensional, not 20-dimensional -- so this cheaper arm answers the
question the phase actually needs answered, while the d=20 arm's timeout is reported
separately as its own, genuinely harder finding, not discarded."""

H2_POWER_N = CANDIDATE_N_PH
H2_POWER_B = CANDIDATE_B

GRID_SIZE_OPTIONS: Tuple[Dict[str, Any], ...] = (
    {
        "label": "one template, one (D,noise) cell, 1 draw",
        "combinations": 1,
        "draws": 1,
    },
    {
        "label": "02.7-RESEARCH.md Open Question 2's own worked example: "
        "15 cells x 4 templates, 3 draws",
        "combinations": 60,
        "draws": 3,
    },
    {
        "label": "same cell count, 1 draw only (draw-count reduction, stated not silent)",
        "combinations": 60,
        "draws": 1,
    },
)
"""Illustrative grid sizes only -- NOT a decision. Plan 02.7-08 chooses the real grid size;
this list exists so the projected whole-grid wall clock is shown at more than one candidate
size, per the plan's own acceptance criteria."""


# =============================================================================================
# Timeout + peak-memory isolation: every ripser-touching measurement below runs in a forked
# child process so (a) a hard wall-clock ceiling can be enforced by terminating the child, and
# (b) the child's own peak resident memory (never the parent's) is what gets reported --
# 02.7-RESEARCH.md records an early, less-controlled version of this exact measurement growing
# past 7GB resident before being killed, so isolating each call's peak from the long-lived
# parent process is deliberate, not incidental.
# =============================================================================================


def _child_worker(fn: Callable, args: tuple, conn) -> None:
    try:
        t0 = time.monotonic()
        result = fn(*args)
        wallclock_s = time.monotonic() - t0
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        conn.send({"status": "ok", "wallclock_s": wallclock_s, "peak_rss_gb": peak_rss_gb, "result": result})
    except Exception as exc:  # pragma: no cover -- defensive; reported, not swallowed
        conn.send({"status": "error", "error": repr(exc)})
    finally:
        conn.close()


def run_timed(fn: Callable, args: tuple, timeout_s: float) -> Dict[str, Any]:
    """Run `fn(*args)` in a forked child process with a hard `timeout_s` ceiling. Returns
    `{"status": "ok"|"timeout"|"error"|"crashed", "wallclock_s", "peak_rss_gb", "result",
    "timeout_ceiling_s"}` -- a cell that does not finish inside `timeout_s` is recorded as a
    `"timeout"` at that known bound, never silently retried at a smaller n and reported as if
    it were the requested one.
    """
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child_worker, args=(fn, args, child_conn))
    proc.start()
    child_conn.close()

    t0 = time.monotonic()
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = {"status": "crashed"}
        proc.join(timeout=5)
    else:
        result = {"status": "timeout", "timeout_ceiling_s": timeout_s}
        proc.terminate()
        proc.join(timeout=5)

    if proc.is_alive():
        proc.kill()
        proc.join()

    result.setdefault("wallclock_s", time.monotonic() - t0)
    result["exitcode"] = proc.exitcode
    return result


def _measure_baseline_rss() -> float:
    """A trivial no-op forked child's own peak RSS (GB) -- the Python/numpy/torch import
    baseline every measurement below inherits via copy-on-write, reported once so every
    per-cell peak_rss_gb figure can be read against it rather than mistaken for the call's
    own incremental cost alone."""

    def _noop() -> int:
        return 0

    result = run_timed(_noop, (), timeout_s=30.0)
    return float(result.get("peak_rss_gb", 0.0))


# =============================================================================================
# Cost grid: one persistence_diagram(D, maxdim=2) call per (template, metric, n).
# =============================================================================================


def _generate_cloud(template: str, n: int, seed: int) -> Dict[str, Any]:
    return template_immersion.immerse(
        template,
        n,
        AMBIENT_D,
        WORST_NOISE,
        1.0,  # density=1.0 -- uniform baseline; D-15's density arm is a separate question
        seed,
        WARP_PARAMS,
        d=BALL_D,
    )


def _euclidean_matrix(points: np.ndarray) -> np.ndarray:
    D, _ = persistence_probe.cloud_distance_matrix(points, prescale=False)
    return D


def _geodesic_matrix(points: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        D_geo, readout = geodesic_graph.geodesic_distance_matrix(points, PROVISIONAL_K)
        return D_geo, readout
    except ValueError as exc:
        return None, {"error": str(exc)}


def measure_cost_grid(templates: Sequence[str], n_ladder: Sequence[int], smoke: bool) -> List[Dict[str, Any]]:
    """For each (template, metric) sequence, walk n_ladder in increasing order, timing one
    `persistence_diagram(D, maxdim=2)` call per cell. Once a cell records anything other than
    `"ok"`, remaining larger-n cells in that same sequence are skipped (never attempted) and
    marked for extrapolation -- bounding this section's own worst-case wall clock."""
    rows: List[Dict[str, Any]] = []

    for template in templates:
        cloud_by_n: Dict[int, Dict[str, Any]] = {}
        for n in n_ladder:
            cloud_by_n[n] = _generate_cloud(template, n, CALIBRATION_SEED)

        for metric in ("euclidean", "geodesic"):
            short_circuited = False
            for n in n_ladder:
                cloud = cloud_by_n[n]
                points = cloud["points"]

                if short_circuited:
                    rows.append(
                        {
                            "template": template,
                            "metric": metric,
                            "n": n,
                            "status": "skipped_after_timeout",
                        }
                    )
                    print(f"  [{template}] n={n} metric={metric} maxdim=2 SKIPPED (prior n in this sequence timed out)")
                    continue

                if metric == "euclidean":
                    D_mat = _euclidean_matrix(points)
                    geo_readout: Dict[str, Any] = {}
                else:
                    D_mat, geo_readout = _geodesic_matrix(points)
                    if D_mat is None:
                        rows.append(
                            {
                                "template": template,
                                "metric": metric,
                                "n": n,
                                "status": "geodesic_graph_error",
                                "geo_readout": geo_readout,
                            }
                        )
                        print(f"  [{template}] n={n} metric={metric} maxdim=2 GEODESIC GRAPH ERROR: {geo_readout}")
                        continue

                timed = run_timed(persistence_probe.persistence_diagram, (D_mat, 2), CALL_TIMEOUT_S)
                row: Dict[str, Any] = {
                    "template": template,
                    "metric": metric,
                    "n": n,
                    "status": timed["status"],
                    "wallclock_s": timed.get("wallclock_s"),
                    "peak_rss_gb": timed.get("peak_rss_gb"),
                    "timeout_ceiling_s": timed.get("timeout_ceiling_s"),
                    "geo_dropped_fraction": geo_readout.get("dropped_fraction"),
                }
                if timed["status"] == "ok":
                    dgms = timed["result"]
                    row["n_bars_h0"] = int(dgms[0].shape[0])
                    row["n_bars_h1"] = int(dgms[1].shape[0])
                    row["n_bars_h2"] = int(dgms[2].shape[0])
                    print(
                        f"  [{template}] n={n} metric={metric} maxdim=2 wallclock_s={row['wallclock_s']:.2f} "
                        f"peak_rss_gb={row['peak_rss_gb']:.3f} "
                        f"n_bars(h0,h1,h2)=({row['n_bars_h0']},{row['n_bars_h1']},{row['n_bars_h2']})"
                    )
                else:
                    short_circuited = True
                    print(
                        f"  [{template}] n={n} metric={metric} maxdim=2 status={timed['status']!r} "
                        f"(timeout ceiling {CALL_TIMEOUT_S}s) -- remaining larger n SKIPPED, extrapolated instead"
                    )
                rows.append(row)

            if smoke:
                break  # --smoke: one metric arm is plenty for the automated <verify>
        if smoke:
            break  # --smoke: one template only

    return rows


def fit_power_law(ns: Sequence[int], ts: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Log-log linear fit `log(t) = a*log(n) + b` over the measured (n, t) pairs. Returns
    `(a, b)` (the fitted exponent and intercept), or `None` if fewer than 2 points are
    available. Used ONLY to extrapolate to n values this run does not measure directly
    (EXTRAPOLATE_TO_N, and any ladder point skipped after a timeout) -- per plan critical
    note #11, an honest extrapolation with its basis shown, clearly labelled as such."""
    if len(ns) < 2:
        return None
    log_n = np.log(np.asarray(ns, dtype=np.float64))
    log_t = np.log(np.asarray(ts, dtype=np.float64))
    a, b = np.polyfit(log_n, log_t, 1)
    return float(a), float(b)


def extrapolate(fit: Tuple[float, float], n: int) -> float:
    a, b = fit
    return float(np.exp(a * np.log(n) + b))


def annotate_extrapolations(cost_rows: List[Dict[str, Any]], extra_ns: Sequence[int]) -> List[Dict[str, Any]]:
    """For each (template, metric) sequence in cost_rows, fit a power law to its own measured
    ("ok") points and append extrapolated rows for `extra_ns` plus any ladder n this run
    skipped after a timeout. Every extrapolated row is marked status="extrapolated" and
    carries the fitted exponent -- never presented as a measurement."""
    extra_rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in cost_rows:
        by_key.setdefault((row["template"], row["metric"]), []).append(row)

    for (template, metric), rows in by_key.items():
        measured = [r for r in rows if r.get("status") == "ok"]
        if len(measured) < 2:
            print(f"  [{template}/{metric}] fewer than 2 measured points -- no extrapolation possible")
            continue
        ns = [r["n"] for r in measured]
        ts = [r["wallclock_s"] for r in measured]
        fit = fit_power_law(ns, ts)
        if fit is None:
            continue
        a, b = fit
        print(f"  [{template}/{metric}] fitted power-law exponent a={a:.3f} (t ~ n^{a:.2f}) over measured n={ns}")

        already_present = {r["n"] for r in rows}
        skipped_ns = {r["n"] for r in rows if r.get("status") == "skipped_after_timeout"}
        targets = sorted(set(extra_ns) | skipped_ns)
        for n in targets:
            if n in already_present and n not in skipped_ns:
                continue
            predicted = extrapolate(fit, n)
            extra_rows.append(
                {
                    "template": template,
                    "metric": metric,
                    "n": n,
                    "status": "extrapolated",
                    "wallclock_s": predicted,
                    "fit_exponent_a": a,
                    "fit_intercept_b": b,
                }
            )
            print(f"    [{template}/{metric}] n={n} EXTRAPOLATED wallclock_s={predicted:.2f} (not measured)")

    return cost_rows + extra_rows


# =============================================================================================
# Band cost: bootstrap_band at a small B, extrapolated linearly to CANDIDATE_B.
# =============================================================================================


def measure_band_cost(smoke: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if smoke:
        return rows  # --smoke skips band cost entirely; the cost grid alone is the automated <verify>

    plan: List[Tuple[str, Tuple[int, ...]]] = [(t, (0, 1, 2)) for t in BAND_FULL_TEMPLATES]
    plan += [(t, (2,)) for t in BAND_H2_ONLY_TEMPLATES]

    for template, degrees in plan:
        cloud = _generate_cloud(template, CANDIDATE_N_PH, CALIBRATION_SEED)
        points = cloud["points"]
        D_euclidean = _euclidean_matrix(points)
        D_geodesic, _ = _geodesic_matrix(points)

        metric_matrices = [("euclidean", D_euclidean)]
        if D_geodesic is not None:
            metric_matrices.append(("geodesic", D_geodesic))

        for metric, D_mat in metric_matrices:
            for degree in degrees:
                timed = run_timed(
                    confidence_band.bootstrap_band,
                    (D_mat, degree, BAND_SMALL_B, CANDIDATE_ALPHA, CALIBRATION_SEED),
                    CALL_TIMEOUT_S,
                )
                row: Dict[str, Any] = {
                    "template": template,
                    "metric": metric,
                    "degree": degree,
                    "n": CANDIDATE_N_PH,
                    "B_measured": BAND_SMALL_B,
                    "status": timed["status"],
                    "wallclock_s": timed.get("wallclock_s"),
                    "peak_rss_gb": timed.get("peak_rss_gb"),
                }
                if timed["status"] == "ok":
                    per_call_s = row["wallclock_s"] / (BAND_SMALL_B + 1)
                    extrapolated_full_s = per_call_s * (CANDIDATE_B + 1)
                    row["per_call_s"] = per_call_s
                    row["extrapolated_at_candidate_B_s"] = extrapolated_full_s
                    print(
                        f"  [{template}] n={CANDIDATE_N_PH} metric={metric} degree=H{degree} "
                        f"MEASURED B={BAND_SMALL_B} wallclock_s={row['wallclock_s']:.2f} "
                        f"per_call_s={per_call_s:.3f} peak_rss_gb={row['peak_rss_gb']:.3f}  "
                        f"EXTRAPOLATED (linear in B) at candidate B={CANDIDATE_B}: "
                        f"{extrapolated_full_s:.2f}s -- kept separate from the measured figure"
                    )
                else:
                    print(
                        f"  [{template}] n={CANDIDATE_N_PH} metric={metric} degree=H{degree} "
                        f"status={timed['status']!r} (timeout ceiling {CALL_TIMEOUT_S}s)"
                    )
                rows.append(row)

    return rows


# =============================================================================================
# H_2 power check: a known-H_2 fixture (S^2 x B^k) against a matched, void-filled control
# (solid B^3 x B^k), both immersed through the SAME lift/warp/noise pipeline as the four
# named templates. This fixture is generic -- never one of the four benchmark templates'
# own scored clouds -- so measuring it does not violate ratify-blind.
# =============================================================================================


def _h2_power_fixture(void: bool, n: int, D: int, extra_ball_dim: int, seed: int) -> np.ndarray:
    seeds = np.random.SeedSequence(seed).spawn(4)
    rng_shell = np.random.default_rng(seeds[0])
    rng_ball = np.random.default_rng(seeds[1])
    rng_lift = np.random.default_rng(seeds[2])
    rng_noise = np.random.default_rng(seeds[3])

    if void:
        shell_points, _ = template_immersion.canonical_sample("S2", n, rng_shell)
    else:
        shell_points, _ = template_immersion.canonical_sample("ball", n, rng_shell, d=3)

    if extra_ball_dim > 0:
        ball_points, _ = template_immersion.canonical_sample("ball", n, rng_ball, d=extra_ball_dim)
        product_points = np.concatenate([shell_points, ball_points], axis=1)
    else:
        product_points = shell_points  # extra_ball_dim=0 -- no product factor, template-native

    d_from = product_points.shape[1]
    lift = template_immersion.random_orthogonal_lift(d_from, D, rng_lift)
    lifted = product_points @ lift.T
    warped = template_immersion.smooth_warp(lifted, WARP_PARAMS["strength"], WARP_PARAMS["freq"], seed)

    if WORST_NOISE > 0.0:
        warped = warped + rng_noise.normal(scale=WORST_NOISE, size=warped.shape)

    return np.asarray(warped, dtype=np.float64)


def _measure_h2_power_regime(regime_label: str, extra_ball_dim: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": H2_POWER_N,
        "D": AMBIENT_D,
        "extra_ball_dim": extra_ball_dim,
        "intrinsic_dim": 2 + extra_ball_dim,
        "B": H2_POWER_B,
        "alpha": CANDIDATE_ALPHA,
        "arms": {},
    }

    for label, void in ((f"void_{regime_label}", True), (f"filled_{regime_label}", False)):
        points = _h2_power_fixture(void, H2_POWER_N, AMBIENT_D, extra_ball_dim, CALIBRATION_SEED)
        D_euclidean = _euclidean_matrix(points)

        t0 = time.monotonic()
        timed = run_timed(
            confidence_band.bootstrap_band,
            (D_euclidean, 2, H2_POWER_B, CANDIDATE_ALPHA, CALIBRATION_SEED),
            CALL_TIMEOUT_S * 2,  # H2 at B=10 legitimately needs more than one call's ceiling
        )
        wallclock_s = time.monotonic() - t0

        arm: Dict[str, Any] = {
            "void": void,
            "status": timed["status"],
            "wallclock_s": wallclock_s,
            "peak_rss_gb": timed.get("peak_rss_gb"),
        }
        if timed["status"] == "ok":
            band_info = timed["result"]
            dgms = persistence_probe.persistence_diagram(D_euclidean, maxdim=2)
            n_h2_bars = int(dgms[2].shape[0])
            n_significant_h2 = int(np.sum(confidence_band.significant_bars(dgms[2], band_info["band"])))
            arm["band"] = band_info["band"]
            arm["c_alpha"] = band_info["c_alpha"]
            arm["n_h2_bars_total"] = n_h2_bars
            arm["n_h2_bars_significant"] = n_significant_h2
            arm["h2_resolved"] = n_significant_h2 > 0
            print(
                f"  [{label}] n={H2_POWER_N} D={AMBIENT_D} intrinsic_dim={result['intrinsic_dim']} "
                f"B={H2_POWER_B} wallclock_s={wallclock_s:.2f} peak_rss_gb={arm['peak_rss_gb']:.3f} "
                f"band={band_info['band']:.6f} n_h2_bars_total={n_h2_bars} "
                f"n_h2_bars_significant={n_significant_h2}  h2_resolved={arm['h2_resolved']}"
            )
        else:
            arm["h2_resolved"] = None
            print(f"  [{label}] status={timed['status']!r} (timeout ceiling {CALL_TIMEOUT_S * 2}s) -- H2 power result UNKNOWN for this arm")

        result["arms"][label] = arm

    void_arm = result["arms"][f"void_{regime_label}"]
    filled_arm = result["arms"][f"filled_{regime_label}"]
    if void_arm.get("h2_resolved") is not None and filled_arm.get("h2_resolved") is not None:
        result["power_check_clean"] = bool(void_arm["h2_resolved"] and not filled_arm["h2_resolved"])
    else:
        result["power_check_clean"] = None

    return result


def measure_h2_power(smoke: bool) -> Dict[str, Any]:
    if smoke:
        return {"skipped": True, "reason": "--smoke skips the H2 power check"}

    return {
        "template_native": _measure_h2_power_regime("S2_vs_ball", H2_POWER_EXTRA_BALL_DIM_TEMPLATE_NATIVE),
        "pu_regime": _measure_h2_power_regime("S2xball18_vs_ballxball18", H2_POWER_EXTRA_BALL_DIM),
    }


# =============================================================================================
# Whole-cell / whole-grid arithmetic -- composed from the measurements above, no new ripser
# calls. One complete grid cell = both metrics x 3 degrees x (1 + candidate_B) calls, using
# each (template, metric, degree)'s own measured per-call cost where available.
# =============================================================================================


def compose_whole_cell_costs(band_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_call: Dict[Tuple[str, str, int], float] = {}
    for row in band_rows:
        if row.get("status") == "ok":
            per_call[(row["template"], row["metric"], row["degree"])] = row["per_call_s"]

    # H0/H1 fallback: templates measured at H2-only borrow the H0/H1 per-call figure from the
    # mean of BAND_FULL_TEMPLATES' own measured H0/H1 costs, since H0/H1 are the
    # comparatively cheap, comparatively template-insensitive legs (RESEARCH Pitfall 1: the
    # template-dependent cost driver is H2). This is an explicit, stated approximation for
    # those two legs only -- H2 itself is always the template's own measurement, never
    # borrowed.
    h0_h1_fallback: Dict[Tuple[str, int], float] = {}
    for metric in ("euclidean", "geodesic"):
        for degree in (0, 1):
            vals = [
                per_call[(t, metric, degree)]
                for t in BAND_FULL_TEMPLATES
                if (t, metric, degree) in per_call
            ]
            if vals:
                h0_h1_fallback[(metric, degree)] = float(np.mean(vals))

    cell_costs: Dict[str, Any] = {}
    for template in TEMPLATES:
        total_s = 0.0
        complete = True
        breakdown = []
        for metric in ("euclidean", "geodesic"):
            for degree in (0, 1, 2):
                key = (template, metric, degree)
                if key in per_call:
                    cost = per_call[key]
                    measured = True
                elif (metric, degree) in h0_h1_fallback:
                    cost = h0_h1_fallback[(metric, degree)]
                    measured = False
                else:
                    complete = False
                    continue
                calls = 1 + CANDIDATE_B
                contribution = calls * cost
                total_s += contribution
                breakdown.append(
                    {
                        "metric": metric,
                        "degree": degree,
                        "per_call_s": cost,
                        "measured": measured,
                        "calls": calls,
                        "contribution_s": contribution,
                    }
                )
        cell_costs[template] = {
            "total_s": total_s if complete else None,
            "complete": complete,
            "breakdown": breakdown,
        }

    return cell_costs


def project_whole_grid(cell_costs: Dict[str, Any]) -> List[Dict[str, Any]]:
    complete_totals = [v["total_s"] for v in cell_costs.values() if v["complete"]]
    if not complete_totals:
        return []
    mean_cell_s = float(np.mean(complete_totals))
    max_cell_s = float(np.max(complete_totals))  # the worst template, per RESEARCH's own sizing advice

    projections = []
    for option in GRID_SIZE_OPTIONS:
        combos = option["combinations"]
        draws = option["draws"]
        n_cells = combos * draws
        projections.append(
            {
                "label": option["label"],
                "combinations": combos,
                "draws": draws,
                "n_cells": n_cells,
                "projected_hours_mean_template": n_cells * mean_cell_s / 3600.0,
                "projected_hours_worst_template": n_cells * max_cell_s / 3600.0,
            }
        )
    return projections


# =============================================================================================
# main
# =============================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: one template (S1), one metric arm, smallest ladder n only -- "
            "exercises the cost-grid code path (immersion, distance matrix, timed "
            "persistence_diagram at maxdim=2) in well under CALL_TIMEOUT_S. Skips band "
            "cost and the H2 power check entirely. This is the runner's automated <verify>."
        ),
    )
    args = parser.parse_args()

    print("=" * 88)
    print("Phase 02.7 Plan 07: H_2 budget calibration at D=768 -- measures and prints; no bar,")
    print("no tally, no verdict. Every constant below is unratified; plan 02.7-08 ratifies.")
    print(
        f"AMBIENT_D={AMBIENT_D}  CALL_TIMEOUT_S={CALL_TIMEOUT_S}  WORST_NOISE={WORST_NOISE}  "
        f"WARP_PARAMS={WARP_PARAMS}  smoke={args.smoke}"
    )
    print("=" * 88)

    baseline_rss_gb = _measure_baseline_rss()
    print(f"baseline forked-child peak RSS (Python/numpy/torch import overhead): {baseline_rss_gb:.3f} GB")
    print("Every peak_rss_gb figure below includes this baseline via copy-on-write inheritance.")

    print()
    print("-" * 88)
    print("SECTION 1 -- cost grid: persistence_diagram(D, maxdim=2), all four templates, both")
    print(f"metrics, n in {N_LADDER}, at D={AMBIENT_D}, noise={WORST_NOISE}")
    print("-" * 88)
    cost_rows = measure_cost_grid(TEMPLATES, N_LADDER, args.smoke)

    if not args.smoke:
        print()
        print("Fitted power-law extrapolations (n=500 and any timed-out ladder n):")
        cost_rows = annotate_extrapolations(cost_rows, EXTRAPOLATE_TO_N)

    print()
    print("-" * 88)
    print(
        f"SECTION 2 -- band cost: bootstrap_band at n={CANDIDATE_N_PH}, B_measured={BAND_SMALL_B}, "
        f"extrapolated (linear in B) to candidate B={CANDIDATE_B}"
    )
    print("-" * 88)
    band_rows = measure_band_cost(args.smoke)

    print()
    print("-" * 88)
    print(
        f"SECTION 3 -- H_2 power check, TWO regimes: (a) template-native, plain S^2 (known void)"
    )
    print(
        f"vs solid B^3 (filled control); (b) PU-regime, S^2 x B^{H2_POWER_EXTRA_BALL_DIM} vs "
        f"solid-ball x B^{H2_POWER_EXTRA_BALL_DIM}. n={H2_POWER_N}, B={H2_POWER_B} both regimes."
    )
    print("-" * 88)
    h2_power = measure_h2_power(args.smoke)

    print()
    print("-" * 88)
    print("SECTION 4 -- one complete grid cell, composed from measured per-call costs (no new")
    print(f"ripser calls): both metrics x 3 degrees x (1 + candidate_B={CANDIDATE_B}) calls")
    print("-" * 88)
    if args.smoke:
        cell_costs: Dict[str, Any] = {}
        grid_projections: List[Dict[str, Any]] = []
        print("  --smoke: skipped (needs band-cost data from Section 2)")
    else:
        cell_costs = compose_whole_cell_costs(band_rows)
        for template, info in cell_costs.items():
            if info["complete"]:
                print(f"  [{template}] one complete grid cell (both metrics, all 3 degrees): {info['total_s']:.1f}s")
            else:
                print(f"  [{template}] INCOMPLETE -- missing per-call data for at least one (metric,degree) leg")

        print()
        print("Projected whole-grid wall clock at several candidate grid sizes (options, not a decision):")
        grid_projections = project_whole_grid(cell_costs)
        for proj in grid_projections:
            print(
                f"  {proj['label']}: n_cells={proj['n_cells']} "
                f"-> mean-template {proj['projected_hours_mean_template']:.2f}h, "
                f"worst-template {proj['projected_hours_worst_template']:.2f}h"
            )

    print()
    print("=" * 88)
    print("Done. Full structured results follow as one JSON line (RESULTS_JSON:) for exact")
    print("transcription into 02.7-BUDGET-CALIBRATION.md -- this script itself computes no")
    print("tally and reaches no decision layer.")
    print("=" * 88)

    results_blob = {
        "ambient_D": AMBIENT_D,
        "call_timeout_s": CALL_TIMEOUT_S,
        "worst_noise": WORST_NOISE,
        "warp_params": WARP_PARAMS,
        "baseline_rss_gb": baseline_rss_gb,
        "smoke": args.smoke,
        "cost_grid": cost_rows,
        "band_cost": band_rows,
        "h2_power": h2_power,
        "cell_costs": cell_costs,
        "grid_projections": grid_projections,
    }
    print("RESULTS_JSON:" + json.dumps(results_blob, default=str))


if __name__ == "__main__":
    main()
