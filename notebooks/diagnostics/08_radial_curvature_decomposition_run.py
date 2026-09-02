"""Plan 08-07 Task 2 — how much of ||H|| is the unit sphere?

DIAGNOSTIC ONLY. Gates nothing. Touches no frozen constant, appends no row to
`notebooks/.cache/08_cka_alignment.jsonl`, reopens no sealed verdict, and does not overwrite
`07_crossmodal_curvature_fields.npz` or any other Phase 7 artifact.

The claim under test. `subsample.l2_normalize` puts every row on the unit sphere -- verified,
`norm min/med/max = 1.000000` for both modalities. For an exact `d`-dimensional submanifold of the
unit sphere, the mean curvature vector under this milestone's `H = tr_g(II)` convention has a
RADIAL component of exactly `-d`, pointing at the origin. That term is a constant of the ambient
normalization: it carries nothing about the manifold's own shape, but it enters `||H||` in full.

Why it matters. The frozen `||H||` medians are 37.19 / 41.41 / 47.03 at `d` = 20 / 25 / 32.
Removing a radial term of magnitude `d` in quadrature leaves 31.36 / 33.02 / 34.46 -- a spread of
10% where the raw field's is 26%. If that arithmetic holds, a large constant is inflating `||H||`
and compressing its dynamic range, which is the milestone's thinnest number: PU's `||H||` p95/p05
spread of ~1.5, and Phase 8's `realized_h_contrast` of 1.16.

The decision-relevant output is `spearman(||H||, ||H_tangential||)`. Rank-based conclusions --
which is every conclusion Phase 7 and Phase 8 draw -- are invariant under a constant offset. If
that correlation is near 1 the radial term changes nothing and this is a one-paragraph limitation.
If it is not, the `||H||` field means something other than what Phases 7 and 8 assumed.

The fit path mirrors `07_crossmodal_curvature_run.fit_and_field` exactly -- same architecture,
same `TORCH_INIT_SEED`, `SPLIT_SEED`, `HOLDOUT_FRACTION`, `TRAIN_CFG` and `MAX_EPOCHS` -- but keeps
`H_vec` and the decoder image, which that function discards after taking the norm. The re-fit is
not guaranteed bit-identical to Phase 7's, so `spearman(refit ||H||, frozen h_norm_<d>)` is
recorded: it separates re-fit drift from the decomposition's own effect.

Usage:
    python notebooks/diagnostics/08_radial_curvature_decomposition_run.py
    python notebooks/diagnostics/08_radial_curvature_decomposition_run.py --threads 4 --d 20
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Value passed for `flag`, accepting both `--flag value` and `--flag=value` (CR-03)."""
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
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pu_manifold import cache  # noqa: E402
from pu_manifold import cae  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe as cp  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402

RECORD_STEM = "08_radial_curvature_decomposition"
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
PU_FIELD_COLUMN = cc.PU_FIELD_COLUMN


def _run_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def fit_and_decompose(X, d):
    """`fit_and_field`'s path, keeping `H_vec` and the decoder image `F(z)` rather than
    discarding them after the norm.

    Returns the raw pieces; all statistics are taken by the caller so this function stays a
    faithful mirror of the sealed fit path and nothing else.
    """
    if d not in cc.D_SWEEP:
        raise ValueError(f"fit_and_decompose: d={d} is not in D_SWEEP={cc.D_SWEEP}.")

    torch.manual_seed(cc.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(
        in_dim=X.shape[1], latent_dim=d, hidden=cc.AE_HIDDEN, activation=cc.AE_ACTIVATION
    )
    train_idx, holdout_idx = cc.split_indices(X.shape[0], cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]

    train_cfg = dict(cc.TRAIN_CFG)
    train_cfg["max_epochs"] = cc.MAX_EPOCHS
    t0 = time.monotonic()
    cae.train_plain_ae(model, x_train32, train_cfg)
    fit_s = time.monotonic() - t0

    model.eval().double()
    with torch.no_grad():
        z_full = model.encode(x64)
        y_holdout = model(x_holdout64)["y"]
        image = model.decode(z_full).detach().cpu().numpy()  # F(z), the point on the manifold
    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    var_explained = 1.0 - recon["mse_total"] / sig

    field = decoder_curvature.plain_decoder_curvature(model, z_full)
    H_vec = field["H_vec"].detach().cpu().numpy()
    cond = field["metric_condition_number"].detach().cpu().numpy()
    return dict(H_vec=H_vec, image=image, cond=cond,
                var_explained=float(var_explained), fit_s=fit_s)


def decompose(H_vec, image):
    """Split each mean-curvature vector into its radial and sphere-tangential parts.

    `u_i = F(z_i) / ||F(z_i)||` is the outward radial direction at the decoder image point.
    `H_rad,i = <H_i, u_i>` is signed -- for a submanifold of the UNIT sphere the exact value is
    `-d`, negative because mean curvature points at the centre of curvature. The tangential
    residual `H_tan,i = H_i - H_rad,i * u_i` is what remains once the ambient normalization's
    contribution is removed.
    """
    img_norm = np.linalg.norm(image, axis=1)
    u = image / img_norm[:, None]
    H_rad = np.einsum("ij,ij->i", H_vec, u)
    H_tan = H_vec - H_rad[:, None] * u
    return H_rad, np.linalg.norm(H_tan, axis=1), img_norm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--d", type=int, default=None,
                    help="run a single d (default: every d in crossmodal_curvature.D_SWEEP)")
    ap.add_argument("--record-path", default=None)
    args = ap.parse_args()

    record_path = (args.record_path if args.record_path is not None
                   else str(cache.cache_path(RECORD_STEM, "jsonl")))
    d_values = (args.d,) if args.d is not None else cc.D_SWEEP

    sub = np.load(cache.cache_path(SUBSAMPLE_STEM, "npz"))
    X = sub[PU_FIELD_COLUMN]
    row_norm = np.linalg.norm(X, axis=1)
    print(f"loaded {PU_FIELD_COLUMN} {X.shape}; row norm min/med/max = "
          f"{row_norm.min():.6f} / {np.median(row_norm):.6f} / {row_norm.max():.6f}", flush=True)
    print(f"[premise] the sphere premise holds only to the extent these are 1.0", flush=True)

    frozen = np.load(cache.cache_path("07_crossmodal_curvature_fields", "npz"))
    mknn = cc.per_point_mknn(sub["hsc"], sub["legacysurvey"], cc.HEADLINE_K)
    dens = 1.0 / cp.local_density_weights(sub["legacysurvey"], cc.DENSITY_K, cc.DENSITY_FIELD_D)
    run_commit = _run_commit()
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []

    for d in d_values:
        print(f"\n=== d={d} ===", flush=True)
        out = fit_and_decompose(X, d)
        H_rad, h_tan, img_norm = decompose(out["H_vec"], out["image"])
        h_norm = np.linalg.norm(out["H_vec"], axis=1)
        h_frozen = frozen[f"h_norm_{d}"]

        def spread(a):
            p5, p95 = np.percentile(a, [5, 95])
            return float(p95 / p5)

        row = dict(
            row_kind="per_d", d=int(d), gates_nothing=True,
            # --- the premise -----------------------------------------------------------------
            decoder_image_norm_p05=float(np.percentile(img_norm, 5)),
            decoder_image_norm_p50=float(np.median(img_norm)),
            decoder_image_norm_p95=float(np.percentile(img_norm, 95)),
            # --- the decomposition -----------------------------------------------------------
            h_rad_median=float(np.median(H_rad)),
            h_rad_p05=float(np.percentile(H_rad, 5)),
            h_rad_p95=float(np.percentile(H_rad, 95)),
            radial_over_minus_d=float(np.median(H_rad) / (-d)),
            h_tan_median=float(np.median(h_tan)),
            h_tan_p05=float(np.percentile(h_tan, 5)),
            h_tan_p95=float(np.percentile(h_tan, 95)),
            h_norm_median=float(np.median(h_norm)),
            h_norm_p05=float(np.percentile(h_norm, 5)),
            h_norm_p95=float(np.percentile(h_norm, 95)),
            spread_h_norm=spread(h_norm), spread_h_tan=spread(h_tan),
            # --- the decision-relevant number ------------------------------------------------
            spearman_h_vs_htan=float(spearmanr(h_norm, h_tan).statistic),
            # --- does substituting the tangential field move the headline? --------------------
            spearman_hnorm_mknn=float(spearmanr(h_norm, mknn).statistic),
            spearman_htan_mknn=float(spearmanr(h_tan, mknn).statistic),
            partial_hnorm_mknn=float(cc_partial(h_norm, mknn, dens)),
            partial_htan_mknn=float(cc_partial(h_tan, mknn, dens)),
            # --- re-fit drift, so it is separable from the decomposition ----------------------
            spearman_refit_vs_frozen=float(spearmanr(h_norm, h_frozen).statistic),
            frozen_h_norm_median=float(np.median(h_frozen)),
            var_explained=out["var_explained"], cond_g_median=float(np.median(out["cond"])),
            fit_wallclock_s=round(out["fit_s"], 1),
            max_epochs=int(cc.MAX_EPOCHS), torch_init_seed=int(cc.TORCH_INIT_SEED),
            run_commit=run_commit, timestamp=stamp,
        )
        rows.append(row)
        print(f"  decoder image ||F(z)|| p50 = {row['decoder_image_norm_p50']:.6f} "
              f"(sphere premise holds iff ~1.0)", flush=True)
        print(f"  H_rad median = {row['h_rad_median']:+.4f}   exact-sphere value = {-d}   "
              f"ratio = {row['radial_over_minus_d']:.4f}", flush=True)
        print(f"  ||H|| median {row['h_norm_median']:.4f} (frozen {row['frozen_h_norm_median']:.4f}, "
              f"refit-vs-frozen rho {row['spearman_refit_vs_frozen']:+.4f}, "
              f"var_explained {row['var_explained']:.5f})", flush=True)
        print(f"  ||H_tan|| median {row['h_tan_median']:.4f}   "
              f"spread {row['spread_h_norm']:.4f} -> {row['spread_h_tan']:.4f}", flush=True)
        print(f"  *** spearman(||H||, ||H_tan||) = {row['spearman_h_vs_htan']:+.6f} ***",
              flush=True)
        print(f"  headline: rho(||H||,MKNN) {row['spearman_hnorm_mknn']:+.6f} -> "
              f"rho(||H_tan||,MKNN) {row['spearman_htan_mknn']:+.6f}   "
              f"partial {row['partial_hnorm_mknn']:+.6f} -> {row['partial_htan_mknn']:+.6f}",
              flush=True)

    with open(record_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} rows to {record_path}")
    print("RADIAL CURVATURE DECOMPOSITION COMPLETE -- gates nothing, no frozen constant changed, "
          "no Phase 7 artifact overwritten")


def cc_partial(x, y, density):
    """Phase 7's own density partial, called not reimplemented."""
    from pu_manifold import cross_split_curvature as csc
    return csc.partial_spearman(x, y, controls=density)


if __name__ == "__main__":
    main()
