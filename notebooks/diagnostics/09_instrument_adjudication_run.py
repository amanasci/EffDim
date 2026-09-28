"""Known-answer validation of the Phase 9 curvature instrument on identical points.

PURPOSE. Phase 9 measured opposite signs for curvature-vs-local-decodability under two
instruments on the same anchors: our plain-autoencoder decoder curvature ``H_tan`` (Amendment
01, sphere-projected decoder, ``09_physics_curvature_run.fit_and_field_at_anchors``). This
instrument has no known-answer validation in the regime where the two instruments disagreed:
unit sphere in R^768, chart rank d=16, k=2048 neighbourhoods, n=86,471, noisy samples.
This runner supplies that known answer, on the same points and the same anchors, and scores
our instrument against it. It follows the spike-findings-effdim validation protocol: anchor at
low d first (``--mode swiss-roll``), state the pass regime in r/R, write the decision rule
before the numbers, score with the sealed four axes where they apply.

THE FIXTURE (``--mode sphere-fixture``). An explicit in-sphere generator

    G(z) = normalize([stereo(z); a * bumps(z); 0, ..., 0]) @ Q^T,   z in R^d, G(z) in S^(D-1)

with d=16, D=768, a=0.8, stereo the inverse stereographic map R^d -> S^d, four Gaussian bumps
(centres seeded, widths (0.7, 0.9, 0.8, 1.0), amplitudes (1.0, -0.8, 0.6, -0.5)) and Q a fixed
seeded rotation of R^D. Latents are a scale mixture z ~ N(0, s^2 I), s in {0.4, 0.6, 0.9} with
probabilities {0.2, 0.5, 0.3}, so the k-NN radius varies about 3x across points (a density
gradient the estimators could couple to). ``--noise 0`` uses X = G(z); ``--noise patch`` uses
X = normalize(G(z) + eps) with eps isotropic Gaussian in R^D scaled so its median displacement
is 25% of the median k=2048 patch radius of the noiseless cloud.

WHY THE TRUTH IS EXACT. G is an explicit ``torch.nn.Module`` with a ``.decode`` method, so the
sealed ``decoder_curvature.plain_decoder_curvature`` (float64 autodiff Jacobian + Hessian,
trace convention H = tr_g(II)) differentiates it exactly at each anchor's OWN latent z. G maps
into the unit sphere, so the radial component is H_rad = -d identically and the tangential
remainder H_tan is the sphere-intrinsic mean curvature vector of the image manifold; the run
asserts max|H_rad + d| < 1e-8 as its exactness check. The truth is a property of the generator,
never of any fit.

WHAT IS AND IS NOT VALIDATED. This validates our instrument AS A CURVATURE ESTIMATOR in the
Phase 9 regime: does it recover the ordering and the direction of a known sphere-intrinsic mean
curvature field from n=86,471 samples at k=2048 and d=16, with and without sample noise, on
identical anchors? It does NOT validate or reinterpret the Physics result, the Phase 9 verdict,
or the pipeline's statistics; it touches no Phase 9 production record.

SCORING. ``synthetic_control_run._fidelity_axes`` (the sealed four axes: direction median
cosine, magnitude median ratio and CV, calibration slope/intercept/R^2, rank Spearman) of the
estimated ``H_tan`` vector against the truth ``H_tan`` vector, both projected to the tangent of
their own sphere image, plus Spearman with ``log_knn_radius`` (density coupling) beside the
truth's own Spearman with log radius.

DECISION RULES (fixed here, before any number is printed):
  swiss-roll   : rank rho vs analytic truth >= 0.5           -> SWISS ROLL PASS (coarse anchor)
  sphere-fixture: rank rho vs truth >= 0.7, and direction median cosine >= 0.8
                 -> "validated in regime" per noise level.

NOT PRE-REGISTERED, GATES NOTHING. This is a diagnostic, additive to the record; no sealed
constant is reinterpreted by it and no Phase 9 verdict depends on it.

Usage:
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode smoke
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode swiss-roll
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode sphere-fixture --noise 0 --threads 16
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode sphere-fixture --noise patch --threads 16
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_RUNNER_PATH = DIAGNOSTICS_ROOT / "09_physics_curvature_run.py"

# Load the production runner FIRST, before numpy/torch are imported anywhere in this process:
# its module-level code applies the `--threads` cap (OMP/MKL/NUMEXPR env vars, then
# `torch.set_num_threads`) from `sys.argv`, and puts `notebooks/` and `notebooks/diagnostics/`
# on `sys.path` for the imports below.
_spec = importlib.util.spec_from_file_location("physics_curvature_run", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import chart_curvature  # noqa: E402
from pu_manifold import curvature_probe  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

# --- scorer copied verbatim from synthetic_control_run.py (archived); definitions unchanged ---
CONSTANT_TRUTH_REL_TOL = 1e-9

MIN_TRUE_NORM = 1e-12


def _fidelity_axes(H_est: np.ndarray, H_true: np.ndarray) -> Dict[str, Any]:
    """The four axes, kept as four. ``curvature_fidelity_report`` supplies direction,
    magnitude and calibration; ``spearman_gate_statistic`` supplies rank;
    ``median_relative_error`` is context. Nothing here combines them.

    **The flat control has no fidelity axes, and this is mathematics rather than a special
    case.** Its analytic ``H`` is the exactly-zero vector at every point, so cosine similarity
    against it is undefined, the magnitude ratio divides by zero, and a calibration regression
    on a constant-zero predictor has no slope. ``curvature_fidelity_report`` refuses such an
    input outright ("A fidelity report over fewer than two points is not a measurement"), and
    that refusal is correct. What the flat fixture measures instead is the pipeline's NOISE
    FLOOR: every value it reports is artifact, recorded under ``noise_floor``. Returning a
    not-applicable marker here keeps that distinction visible rather than manufacturing four
    numbers that would each be a division by zero wearing a float."""
    h_true_norm_all = np.linalg.norm(H_true, axis=-1)
    if not np.any(h_true_norm_all > MIN_TRUE_NORM):
        return {
            "applicable": False,
            "reason": (
                "analytic ||H|| is exactly zero at every point: cosine, magnitude ratio and "
                "calibration slope are all undefined against a zero field. This fixture "
                "measures the pipeline's noise floor instead -- see the noise_floor key."
            ),
            "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
        }
    report = chart_curvature.curvature_fidelity_report(H_est, H_true)
    h_est_norm = np.linalg.norm(H_est, axis=-1)
    h_true_norm = np.linalg.norm(H_true, axis=-1)
    mre = curvature_probe.median_relative_error(h_est_norm, h_true_norm)

    # A constant analytic field leaves rank and calibration undefined -- see
    # CONSTANT_TRUTH_REL_TOL. Direction and magnitude remain well posed.
    scale = max(abs(float(np.median(h_true_norm))), MIN_TRUE_NORM)
    constant_truth = bool(float(np.ptp(h_true_norm)) / scale <= CONSTANT_TRUTH_REL_TOL)
    if constant_truth:
        rank_value: Any = None
        calib: Dict[str, Any] = {
            "calibration_slope": None,
            "calibration_intercept": None,
            "calibration_r2": None,
        }
    else:
        rank_value = float(curvature_probe.spearman_gate_statistic(h_est_norm, h_true_norm))
        calib = {
            "calibration_slope": report["calibration_slope"],
            "calibration_intercept": report["calibration_intercept"],
            "calibration_r2": report["calibration_r2"],
        }
    return {
        "applicable": True,
        "constant_truth": constant_truth,
        "rank_calibration_applicable": not constant_truth,
        "rank_calibration_note": (
            "analytic ||H|| is constant over this fixture: Spearman rho has no ranks to "
            "correlate and the calibration regression has a zero-variance predictor, so both "
            "are UNDEFINED here and are reported as null rather than as numbers. Direction "
            "and magnitude remain well posed."
        ) if constant_truth else "",
        "direction_median_cosine": report["median_cosine_similarity"],
        "magnitude_median_ratio": report["median_magnitude_ratio"],
        "magnitude_ratio_cv": report["magnitude_ratio_cv"],
        **calib,
        "rank_spearman_rho": rank_value,
        "median_relative_error": float(mre),
        "n_points": report["n_points"],
        "n_excluded": report["n_excluded"],
        "curvature_convention": report["curvature_convention"],
    }

EXPERIMENT = "instrument-adjudication"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_instrument_adjudication.jsonl"
PRODUCTION_STEMS = ("09_physics_curvature",)

# Decision rules -- fixed in source before any run.
REGIME_RANK_RHO_PASS = 0.7
REGIME_DIRECTION_COS_PASS = 0.8

# The fixture.
FIXTURE = {
    "d": 16, "D": 768, "n": 86471, "a": 0.8,
    "bump_widths": (0.7, 0.9, 0.8, 1.0), "bump_amps": (1.0, -0.8, 0.6, -0.5),
    "scale_choices": (0.4, 0.6, 0.9), "scale_probs": (0.2, 0.5, 0.3),
    "k": pcp.K_NEIGHBOURS, "n_anchors": pcp.N_ANCHORS,
}
NOISE_FRACTION_OF_PATCH_RADIUS = 0.25

SMOKE = {**FIXTURE, "d": 4, "D": 64, "n": 4000, "k": 128, "n_anchors": 64}
SMOKE_EPOCHS = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_pair(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], int(m.sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    x, y, n = _finite_pair(np.asarray(a, float), np.asarray(b, float))
    if n < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _scalar_calibration(est: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Least-squares est ~ slope * truth + intercept, with R^2, on finite pairs."""
    t, e, n = _finite_pair(np.asarray(truth, float), np.asarray(est, float))
    if n < 3 or np.ptp(t) == 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    A = np.stack([t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ coef
    sst = float(((e - e.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan")
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2}


def _tangential(H_vec: np.ndarray, image: np.ndarray) -> np.ndarray:
    """H_tan vector: H minus its component along the image's own radial unit vector, the vector
    form of the sealed ``decompose_radial_tangential`` formula."""
    u = image / np.linalg.norm(image, axis=1, keepdims=True)
    H_rad = np.einsum("ij,ij->i", H_vec, u)
    return H_vec - H_rad[:, None] * u


def measure_r_over_R(X: np.ndarray, kth_distances: np.ndarray) -> Dict[str, float]:
    """The skill's pass-regime statistic: median k-th neighbour distance at the anchors over the
    median distance of every point to the cloud centroid."""
    r_knn = float(np.median(kth_distances))
    R = float(np.median(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    return {"r_knn": r_knn, "R": R, "r_over_R": r_knn / R}


def _refuse_production_record(path: Path) -> None:
    for stem in PRODUCTION_STEMS:
        if path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {path}")


def _append(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _git_head(cwd: Path) -> Optional[str]:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


# --- the generator ------------------------------------------------------------------------


def stereo(z: torch.Tensor) -> torch.Tensor:
    s = (z * z).sum(-1, keepdim=True)
    return torch.cat([2 * z, s - 1], dim=-1) / (1 + s)


class InSphereGenerator(torch.nn.Module):
    """G(z) = normalize([stereo(z); a * bumps(z); 0...]) @ Q^T, float64, image in S^(D-1).
    Has no parameters and no activation modules, so ``assert_c2_decoder`` reports
    ``no-activation-modules`` and the float64 guard passes on the buffers' dtype."""

    def __init__(self, d: int, D: int, a: float, widths: Sequence[float], amps: Sequence[float], seed: int):
        super().__init__()
        if D < d + 2:
            raise ValueError(f"D={D} must be at least d+2={d + 2}")
        self.d, self.D, self.a = d, D, a
        gen = torch.Generator().manual_seed(seed)
        A = torch.randn(D, D, generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        self.register_buffer("Q", Q)
        self.register_buffer("centres", torch.stack([torch.randn(d, generator=gen, dtype=torch.float64) * 0.5 for _ in widths]))
        self.register_buffer("widths", torch.tensor(widths, dtype=torch.float64))
        self.register_buffer("amps", torch.tensor(amps, dtype=torch.float64))

    def bumps(self, z: torch.Tensor) -> torch.Tensor:
        sq = ((z[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)  # (b, n_bumps)
        return (torch.exp(-sq / (2 * self.widths ** 2)) * self.amps).sum(-1, keepdim=True)

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        v = torch.cat([stereo(z), self.a * self.bumps(z), torch.zeros(b, self.D - self.d - 2, dtype=z.dtype)], -1)
        return v @ self.Q.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        F = self.raw(z)
        return F / F.norm(dim=-1, keepdim=True)


def draw_latents(n: int, d: int, scale_choices, scale_probs, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.choice(np.asarray(scale_choices, float), size=n, p=np.asarray(scale_probs, float))
    return rng.standard_normal((n, d)) * s[:, None]


def generate_points(G: InSphereGenerator, z: np.ndarray, batch: int = 8192) -> np.ndarray:
    out = np.empty((z.shape[0], G.D), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, z.shape[0], batch):
            zb = torch.as_tensor(z[start:start + batch], dtype=torch.float64)
            out[start:start + batch] = G.decode(zb).numpy()
    return out


def exact_truth_at_anchors(G: InSphereGenerator, z_anchor: np.ndarray) -> Dict[str, np.ndarray]:
    """Sealed autodiff curvature of the explicit generator at the anchors' OWN latents, then the
    sealed radial/tangential decomposition. Asserts the in-sphere exactness H_rad == -d."""
    zt = torch.as_tensor(z_anchor, dtype=torch.float64)
    field = decoder_curvature.plain_decoder_curvature(G, zt)
    with torch.no_grad():
        image = G.decode(zt).numpy()
    H_vec = field["H_vec"].numpy()
    dec = pcp.decompose_radial_tangential(H_vec, image, pcp.MIN_IMAGE_NORM)
    max_dev = float(np.max(np.abs(dec["H_rad"] + G.d)))
    if not max_dev < 1e-8:
        raise AssertionError(f"exactness check failed: max|H_rad + d| = {max_dev:.3e} (expected < 1e-8)")
    return {
        "H_vec": H_vec, "image": image, "H_tan_vec": _tangential(H_vec, image),
        "H_tan_norm": dec["H_tan_norm"], "H_rad": dec["H_rad"], "max_abs_H_rad_plus_d": max_dev,
        "metric_condition_number": field["metric_condition_number"].numpy(),
    }


# --- scoring ---------------------------------------------------------------------------------


def score_ours(H_tan_est: np.ndarray, H_tan_true: np.ndarray, log_knn_radius: np.ndarray) -> Dict[str, Any]:
    axes = _fidelity_axes(H_tan_est, H_tan_true)
    est_norm = np.linalg.norm(H_tan_est, axis=1)
    axes["rho_vs_log_knn_radius"] = _spearman(est_norm, log_knn_radius)
    return axes


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_table(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> None:
    widths = [max(len(k), *(len(_fmt(r.get(k))) for r in rows)) for k in keys]
    print("  ".join(k.ljust(w) for k, w in zip(keys, widths)))
    for r in rows:
        print("  ".join(_fmt(r.get(k)).ljust(w) for k, w in zip(keys, widths)))


def _environment_row(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
        "repo_head": _git_head(NOTEBOOK_ROOT.parent),
        "threads": args.threads, "device": args.device, "seed": args.seed,
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
        "decoder_image_projection": pcp.DECODER_IMAGE_PROJECTION, "curvature_convention": decoder_curvature.CURVATURE_CONVENTION,
        "pre_registered": False, "gates": "nothing",
    }


# --- modes: sphere-fixture and smoke -------------------------------------------------------


def run_fixture(cfg: Dict[str, Any], noise: str, max_epochs: int, args: argparse.Namespace,
                record_path: Path, mode: str) -> Dict[str, Any]:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print("\n" + "=" * 78 + f"\n{mode}: in-sphere fixture d={d} D={D} n={n} k={k} anchors={n_anchors} noise={noise}\n" + "=" * 78)
    print(f"DECISION RULE (fixed before the numbers): validated in regime iff rank rho vs truth >= {REGIME_RANK_RHO_PASS}"
          f" and, for ours, direction median cosine >= {REGIME_DIRECTION_COS_PASS}.")
    assert pcp.DECODER_IMAGE_PROJECTION == "sphere", pcp.DECODER_IMAGE_PROJECTION

    G = InSphereGenerator(d, D, cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=args.seed)
    z = draw_latents(n, d, cfg["scale_choices"], cfg["scale_probs"], seed=args.seed + 1)
    t0 = time.monotonic()
    X0 = generate_points(G, z)
    print(f"generated X0 {X0.shape} in {time.monotonic() - t0:.1f}s; row norms max|.-1|={np.max(np.abs(np.linalg.norm(X0, axis=1) - 1)):.2e}")

    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    anchor_idx = split["anchor_idx"]
    panel0 = pcp.knn_panel(X0, anchor_idx, k)
    r_med0 = float(np.median(panel0["distances"][:, -1]))
    rr = measure_r_over_R(X0, panel0["distances"][:, -1])
    print(f"noiseless k-NN radius at anchors: median={r_med0:.4f}, spread p05/p95={np.percentile(panel0['distances'][:, -1], 5):.4f}/"
          f"{np.percentile(panel0['distances'][:, -1], 95):.4f}; r/R={rr['r_over_R']:.4f}")

    noise_info: Dict[str, Any] = {"sigma": 0.0, "median_displacement": 0.0, "noiseless_median_patch_radius": r_med0}
    if noise == "patch":
        target = NOISE_FRACTION_OF_PATCH_RADIUS * r_med0
        sigma = target / np.sqrt(D)
        eps = np.random.default_rng(args.seed + 2).standard_normal((n, D)) * sigma
        disp = float(np.median(np.linalg.norm(eps, axis=1)))
        X = X0 + eps
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        noise_info.update({"sigma": float(sigma), "median_displacement": disp, "target_displacement": float(target)})
        print(f"noise=patch: sigma per coord={sigma:.5f}, median ||eps||={disp:.4f} (target {target:.4f} = "
              f"{NOISE_FRACTION_OF_PATCH_RADIUS:.2f} x median patch radius {r_med0:.4f}); X re-normalised to the sphere")
        del eps
    else:
        X = X0
    panel = pcp.knn_panel(X, anchor_idx, k)
    log_r = panel["log_knn_radius"]

    t0 = time.monotonic()
    truth = exact_truth_at_anchors(G, z[anchor_idx])
    print(f"truth at {n_anchors} anchors in {time.monotonic() - t0:.1f}s: max|H_rad+d|={truth['max_abs_H_rad_plus_d']:.2e} (<1e-8 OK), "
          f"||H_tan|| median={np.median(truth['H_tan_norm']):.4f} p05/p95={np.percentile(truth['H_tan_norm'], 5):.4f}/"
          f"{np.percentile(truth['H_tan_norm'], 95):.4f}, spread p95/p05={np.percentile(truth['H_tan_norm'], 95) / np.percentile(truth['H_tan_norm'], 5):.2f}")
    truth_rho_radius = _spearman(truth["H_tan_norm"], log_r)

    print(f"[ours] fit_and_field_at_anchors d={d} in_dim={D} max_epochs={max_epochs} (frozen {pcp.MAX_EPOCHS}) ...", flush=True)
    ours = runner.fit_and_field_at_anchors(
        X, d, anchor_idx, in_dim=D, hidden=pcp.AE_HIDDEN, activation=pcp.AE_ACTIVATION, train_cfg=pcp.TRAIN_CFG,
        max_epochs=max_epochs, torch_init_seed=pcp.TORCH_INIT_SEED, split_seed=pcp.SPLIT_SEED, holdout_fraction=pcp.HOLDOUT_FRACTION,
    )
    H_tan_est = _tangential(ours["H_vec"], ours["image"])
    ours_scores = score_ours(H_tan_est, truth["H_tan_vec"], log_r)
    print(f"[ours] var_explained={ours['var_explained']:.4f} fit {ours['wallclock_fit_s']:.1f}s field {ours['wallclock_field_s']:.1f}s")

    rows = [
        {"instrument": "ours_H_tan", **{k_: ours_scores.get(k_) for k_ in (
            "rank_spearman_rho", "direction_median_cosine", "magnitude_median_ratio", "magnitude_ratio_cv",
            "calibration_slope", "calibration_intercept", "calibration_r2", "rho_vs_log_knn_radius")}},
        {"instrument": "truth_H_tan", "rho_vs_log_knn_radius": truth_rho_radius},
    ]
    print()
    _print_table(rows, ["instrument", "rank_spearman_rho", "direction_median_cosine", "magnitude_median_ratio", "magnitude_ratio_cv",
                        "calibration_slope", "calibration_intercept", "calibration_r2", "rho_vs_log_knn_radius"])
    print(f"AE var_explained={ours['var_explained']:.4f}   "
          f"truth rho(||H_tan||, log r)={truth_rho_radius:.4f}\n")

    def _ok(v: Any, thr: float) -> bool:
        return v is not None and np.isfinite(v) and v >= thr

    ours_pass = _ok(ours_scores["rank_spearman_rho"], REGIME_RANK_RHO_PASS) and _ok(ours_scores["direction_median_cosine"], REGIME_DIRECTION_COS_PASS)
    print(f"ours: rank rho={_fmt(ours_scores['rank_spearman_rho'])} >= {REGIME_RANK_RHO_PASS} and direction cos="
          f"{_fmt(ours_scores['direction_median_cosine'])} >= {REGIME_DIRECTION_COS_PASS}: {'PASS' if ours_pass else 'FAIL'} [noise={noise}]")

    base = {"experiment": EXPERIMENT, "mode": mode, "noise": noise, "timestamp": _utc_now(), "d": d, "D": D, "n": n, "k": k,
            "n_anchors": n_anchors, "fixture": {k_: v for k_, v in cfg.items() if k_ not in ("d", "D", "n", "k", "n_anchors")},
            "noise_info": noise_info, **{f"regime_{k_}": v for k_, v in rr.items()},
            "truth_max_abs_H_rad_plus_d": truth["max_abs_H_rad_plus_d"], "truth_rho_vs_log_knn_radius": truth_rho_radius,
            "truth_H_tan_norm_p05_p50_p95": [float(np.percentile(truth["H_tan_norm"], q)) for q in (5, 50, 95)],
            "decision_rule": f"rank_rho >= {REGIME_RANK_RHO_PASS}; ours also direction_median_cosine >= {REGIME_DIRECTION_COS_PASS}"}
    _append({**base, "instrument": "ours_H_tan", "max_epochs": max_epochs, "var_explained": ours["var_explained"],
             "wallclock_fit_s": ours["wallclock_fit_s"], "scores": ours_scores, "verdict": "PASS" if ours_pass else "FAIL"}, record_path)
    return {"ours": ours_scores, "ours_pass": ours_pass, "var_explained": ours["var_explained"], "truth": truth}


def run_smoke(args: argparse.Namespace, record_path: Path) -> bool:
    print("\n" + "=" * 78 + "\nSMOKE: tiny fixture, both noise levels. Verdict lines are informational;\n"
          "SMOKE PASS means the path ran end to end with finite scores and the exactness check held.\n" + "=" * 78)
    ok = True
    for noise in ("0", "patch"):
        res = run_fixture(SMOKE, noise, SMOKE_EPOCHS, args, record_path, mode="smoke")
        finite = all(np.isfinite(res["ours"][k_]) for k_ in ("rank_spearman_rho", "direction_median_cosine"))
        print(f"stage=smoke_noise_{noise} finite_scores={finite} exactness={res['truth']['max_abs_H_rad_plus_d']:.2e} "
              f"{'PASS' if finite else 'FAIL'}")
        ok = ok and finite
    return ok


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "sphere-fixture"], required=True)
    p.add_argument("--noise", choices=["0", "patch"], default="0", help="sphere-fixture only")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=20260905, help="fixture seed (generator, latents, noise)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help=f"AE epoch budget; default frozen MAX_EPOCHS={pcp.MAX_EPOCHS} for sphere-fixture")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    record_path = Path(args.record_path).resolve()
    _refuse_production_record(record_path)
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.")

    _append(_environment_row(args), record_path)

    if args.mode == "smoke":
        ok = run_smoke(args, record_path)
        print("SMOKE PASS" if ok else "SMOKE FAIL")
        sys.exit(0 if ok else 1)
    max_epochs = args.max_epochs if args.max_epochs is not None else pcp.MAX_EPOCHS
    run_fixture(FIXTURE, args.noise, max_epochs, args, record_path, mode="sphere-fixture")
    sys.exit(0)


if __name__ == "__main__":
    main()
