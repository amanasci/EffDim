"""II rank: in how many independent normal directions does the decoder manifold bend?

PURPOSE. At each anchor the second fundamental form is a linear map from symmetric 2-tensors on the
tangent space (dimension m = d(d+1)/2, 136 at d = 16) into the normal space. A linear probe's
restriction to the manifold has Hessian <w_N, II>, so the set of Hessians a probe can produce at an
anchor is the image of II's transpose. If II has full rank m, some normal direction reproduces ANY
label Hessian there, and "the probe bends toward the label" is a matter of expressivity. If II's
spectrum is concentrated in a few directions, the manifold can only bend in a few ways, and whether
those ways match physical labels is an open question (the planned probe-free capacity test).

WHAT IS COMPUTED, per anchor, from the stored split-runner geometry (J, Hess, image at 512 anchors):
II = Hess - J g^-1 J^T Hess; an orthonormal tangent frame from J = QR (II_on = II(R^-1., R^-1.));
the in-sphere part II^S = II_on minus its radial component (on the unit sphere that component is
-delta_ij x_hat, checked and reported as radial_dev); II^S flattened to a D x m matrix whose
Frobenius norm equals the tensor's (off-diagonal entries times sqrt 2); its singular values.
Summaries: entropy effective rank, participation ratio, k90 / k99 (directions holding 90% / 99% of
the squared spectrum), condition number; and the same metrics for a Gaussian D x m matrix.

DECISION RULE (fixed here, before any number is printed), on the median over anchors per encoder:
  - "full":    k99 >= 0.9 m  -> capacity is trivially full; the probe-free capacity test is uninformative.
  - "low":     k90 <= m / 4  -> the manifold bends in few directions; the capacity test is informative.
  - otherwise "partial".

NOT PRE-REGISTERED FOR THE PAPER, GATES NOTHING.

Usage:
    python curvature-experiment/runners/12_ii_rank_run.py --root /mnt/ssd-cluster/EffDim/sweep-out \\
        --encoders vit_base,clip_base --d 16 --out /mnt/ssd-cluster/EffDim/ii-rank --threads 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _set_threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)


if __name__ == "__main__":                       # before numpy loads its BLAS
    _t = sys.argv.index("--threads") + 1 if "--threads" in sys.argv else None
    _set_threads(int(sys.argv[_t]) if _t else 4)

import numpy as np  # noqa: E402

FULL_FRAC = 0.9          # k99 >= FULL_FRAC * m  -> "full"
LOW_FRAC = 0.25          # k90 <= LOW_FRAC * m   -> "low"
CHUNK = 32


def sym_flatten(T: np.ndarray) -> np.ndarray:
    """(..., d, d) symmetric -> (..., d(d+1)/2), off-diagonals times sqrt 2 (Frobenius-preserving)."""
    d = T.shape[-1]
    iu, ju = np.triu_indices(d)
    w = np.where(iu == ju, 1.0, np.sqrt(2.0))
    return T[..., iu, ju] * w


def spectrum_metrics(s: np.ndarray) -> dict:
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    e = s ** 2
    tot = e.sum()
    p = e / tot
    nz = p[p > 0]
    cum = np.cumsum(p)
    return {"erank": float(np.exp(-(nz * np.log(nz)).sum())), "pr": float(1.0 / (p ** 2).sum()),
            "k90": int(np.searchsorted(cum, 0.9 - 1e-12) + 1), "k99": int(np.searchsorted(cum, 0.99 - 1e-12) + 1),
            "cond": float(s[0] / s[-1]) if s[-1] > 0 else float("inf")}


def ii_spectra(J: np.ndarray, Hess: np.ndarray, image: np.ndarray) -> dict:
    """Singular values of the in-sphere and full II per anchor; J (b,D,d), Hess (b,D,d,d), image (b,D)."""
    J = J.astype(np.float64); Hess = Hess.astype(np.float64); image = image.astype(np.float64)
    d = J.shape[-1]
    g = np.einsum("bai,baj->bij", J, J)
    ginv = np.linalg.inv(g)
    JtH = np.einsum("bai,bajk->bijk", J, Hess)
    II = Hess - np.einsum("bai,bij,bjkl->bakl", J, ginv, JtH)
    _, R = np.linalg.qr(J)                                    # J = Q R, orthonormal coords u = R z
    Rinv = np.linalg.inv(R)
    II_on = np.einsum("baij,bip,bjq->bapq", II, Rinv, Rinv)
    xh = image / np.linalg.norm(image, axis=1, keepdims=True)
    radial = np.einsum("ba,bapq->bpq", xh, II_on)
    radial_dev = np.abs(radial + np.eye(d)[None]).max(axis=(1, 2))
    II_S = II_on - xh[:, :, None, None] * radial[:, None]
    s_in = np.linalg.svd(sym_flatten(II_S), compute_uv=False)
    s_full = np.linalg.svd(sym_flatten(II_on), compute_uv=False)
    return {"s_insphere": s_in, "s_full": s_full, "radial_dev": radial_dev}


def random_reference(D: int, m: int, seed: int = 0) -> dict:
    s = np.linalg.svd(np.random.default_rng(seed).standard_normal((D, m)), compute_uv=False)
    return spectrum_metrics(s)


def verdict(med_k90: float, med_k99: float, m: int) -> str:
    if med_k99 >= FULL_FRAC * m:
        return "full"
    if med_k90 <= LOW_FRAC * m:
        return "low"
    return "partial"


def run_encoder(npz_path: Path) -> dict:
    z = np.load(npz_path)
    J, Hess, image = z["J"], z["Hess"], z["image"]
    n, D, d = J.shape
    m = d * (d + 1) // 2
    s_in, s_full, dev = [], [], []
    for a in range(0, n, CHUNK):
        out = ii_spectra(J[a:a + CHUNK], Hess[a:a + CHUNK], image[a:a + CHUNK])
        s_in.append(out["s_insphere"]); s_full.append(out["s_full"]); dev.append(out["radial_dev"])
    s_in = np.concatenate(s_in); s_full = np.concatenate(s_full); dev = np.concatenate(dev)
    per = [spectrum_metrics(s) for s in s_in]
    per_full = [spectrum_metrics(s) for s in s_full]

    def q(rows, key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return [float(x) for x in np.percentile(v, [25, 50, 75])]

    summ = {k: q(per, k) for k in ("erank", "pr", "k90", "k99", "cond")}
    summ_full = {k: q(per_full, k) for k in ("erank", "pr", "k90", "k99")}
    norm_spec = s_in / s_in[:, :1]
    return {"n_anchors": int(n), "D": int(D), "d": int(d), "m": int(m),
            "insphere_p25_p50_p75": summ, "full_p25_p50_p75": summ_full,
            "radial_dev_max": float(dev.max()), "radial_dev_median": float(np.median(dev)),
            "median_normalised_spectrum": [float(x) for x in np.median(norm_spec, axis=0)],
            "random_reference": random_reference(D, m),
            "verdict": verdict(summ["k90"][1], summ["k99"][1], m)}, s_in


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="sweep output root holding geometry/<encoder>/")
    p.add_argument("--encoders", required=True)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--seed-tag", default="seed0")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for enc in a.encoders.split(","):
        npz = Path(a.root) / "geometry" / enc / f"09_probe_facing_geometry_d{a.d}_{a.seed_tag}.npz"
        t0 = time.monotonic()
        res, s_in = run_encoder(npz)
        res.update({"encoder": enc, "geometry_npz": str(npz), "numpy": np.__version__, "wall_s": time.monotonic() - t0})
        (out / f"ii_rank_{enc}_d{a.d}.json").write_text(json.dumps(res, indent=1) + "\n")
        np.savez_compressed(out / f"ii_rank_{enc}_d{a.d}_spectra.npz", s_insphere=s_in.astype(np.float32))
        s = res["insphere_p25_p50_p75"]; r = res["random_reference"]
        print(f"{enc}: D {res['D']} m {res['m']} | median erank {s['erank'][1]:.1f} pr {s['pr'][1]:.1f} "
              f"k90 {s['k90'][1]:.0f} k99 {s['k99'][1]:.0f} | random erank {r['erank']:.1f} k99 {r['k99']} | "
              f"radial dev max {res['radial_dev_max']:.1e} | {res['verdict']} ({res['wall_s']:.0f}s)", flush=True)
    print("II_RANK_DONE", flush=True)


if __name__ == "__main__":
    main()
