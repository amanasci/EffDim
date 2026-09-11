"""Probe-facing curvature on the Physics anchors: <w_N, II> from both instruments against local R^2.

PURPOSE. Supplement 05 showed, on the adjudication fixture with exact geometry, that the
curvature a global linear probe pays for is the second fundamental form seen from the probe's
normal direction, ``<w_N, II>`` (the intrinsic Hessian of ``w.x`` restricted to the manifold), and
that its sealed three-control partial against local R^2 is stable at -0.74 to -0.80 across
samplings where ``||H_tan||``'s partial swings from -0.29 to +0.26. Neither production instrument
outputs that quantity. This runner computes it on the real Physics anchors from both:

  decoder    : II = P_N D^2F at the anchor's latent code, by autodiff of the Amendment 01
               sphere-projected decoder (the frozen fit protocol, one fit per d);
  colleague  : his fitted sphere-normal quadratic B^S (his code, unchanged), averaged over the
               two halves and three splits, with his own `project_normal` for w_N and his own
               `probe_facing_scalar` reported beside our Frobenius contraction.

Probe weights ``w`` are a whole-data ridge at the frozen alpha on the finite rows of each label;
the outcome, anchors, k-NN panel, out-of-fold local R^2 and the three sealed controls are the
production pipeline's own calls. Beside the sealed controls, the multi-scale radius control of
Supplement 04 (log r_k at k in {16, 64, 256, 1024, 2048}, read from the same k=2048 panel) is
reported, since it removed the colleague's ||H_tan||-based association there.

COLUMNS per anchor: ``H_tan_norm`` (sealed verdict field, reference), ``pf_curv_dec`` =
|<w_N, II>|_g, ``pf_trace_tan_dec`` = <w_N, H_tan>, ``pf_trace_rad_dec`` = H_rad <w_N, x>,
``bias_sq`` = (y - yhat_oof)^2 at the anchor, and for the colleague ``K_H_cross_col``
(reference), ``pf_curv_col`` = |<w_N, B^S>|_F, ``K_w_dir_col`` = his probe-facing scalar times
|w_N|. Each column gets raw Spearman, the sealed partial, the multi-scale partial, and its
coupling with log radius and with ||H_tan||.

NOT PRE-REGISTERED, GATES NOTHING. Writes only to its own record and to
``<output root>/probe-facing/`` (per-d npz of J, D^2F and image at the anchors, float32).

Usage:
    python notebooks/diagnostics/09_physics_probe_facing_run.py --mode smoke --skip-colleague --threads 8
    EFFDIM_09_OUTPUT_ROOT=... HF_HOME=... python notebooks/diagnostics/09_physics_probe_facing_run.py \\
        --mode physics --d-values 16,20 --colleague-root <root> --threads 16
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_ADJ_PATH = DIAGNOSTICS_ROOT / "09_instrument_adjudication_run.py"
_spec = importlib.util.spec_from_file_location("instrument_adjudication_run", _ADJ_PATH)
adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adj)          # loads the colleague runner, which loads the production runner (threads cap)
colleague, runner = adj.colleague, adj.runner

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from torch.func import hessian, jacrev, vmap  # noqa: E402

from pu_manifold import cae, chart_curvature, crossmodal_curvature, decoder_curvature  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402

EXPERIMENT = "physics-probe-facing"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_physics_probe_facing.jsonl"
PRODUCTION_STEMS = ("09_physics_curvature", "09_colleague_estimator", "09_instrument_adjudication")
MULTISCALE_KS = (16, 64, 256, 1024, 2048)
DEC_COLUMNS = ("H_tan_norm", "pf_curv_dec", "pf_trace_tan_dec", "pf_trace_rad_dec", "bias_sq")
COL_COLUMNS = ("K_H_cross_col", "pf_curv_col", "K_w_dir_col")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def _spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else float("nan")


def partial_row(x: np.ndarray, r2: np.ndarray, Z: np.ndarray, n_perm: int) -> Dict[str, Any]:
    x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(r2) & np.all(np.isfinite(Z), axis=1)
    out: Dict[str, Any] = {"n_finite": int(m.sum())}
    try:
        fw = pcp.permutation_fwer({0: x[m]}, r2[m], Z[m], n_perm, pcp.PERMUTATION_SEED)
        out.update({"partial": fw["per_d"][0]["observed_rho"], "p": fw["per_d"][0]["p"], "undefined": False})
    except ValueError as e:
        out.update({"partial": float("nan"), "p": float("nan"), "undefined": True, "reason": str(e)})
    return out


# --- decoder: fit, then J and D^2F at the anchors ---------------------------------------------


def fit_decoder(X: np.ndarray, d: int, in_dim: int, max_epochs: int) -> Dict[str, Any]:
    """`fit_and_field_at_anchors`'s fit steps, returning the model so the Hessian can be taken."""
    torch.manual_seed(pcp.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(in_dim=in_dim, latent_dim=d, hidden=pcp.AE_HIDDEN, activation=pcp.AE_ACTIVATION)
    train_idx, holdout_idx = crossmodal_curvature.split_indices(X.shape[0], pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    cfg = dict(pcp.TRAIN_CFG); cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x32[torch.as_tensor(train_idx, dtype=torch.long)], cfg)
    wall = time.monotonic() - t0
    model.eval().double()
    x_hold = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]
    with torch.no_grad():
        y_hold = model(x_hold)["y"]
    recon = cae.reconstruction_stats(x_hold, y_hold)
    var_explained = 1.0 - recon["mse_total"] / float((torch.linalg.norm(x_hold, dim=1) ** 2).mean())
    curvature_model = runner.SphereProjectedDecoder(model).eval() if pcp.DECODER_IMAGE_PROJECTION == "sphere" else model
    return {"model": model, "curvature_model": curvature_model, "x64": x64, "var_explained": float(var_explained), "wallclock_fit_s": wall}


def decoder_geometry(curvature_model: torch.nn.Module, z: torch.Tensor) -> Dict[str, np.ndarray]:
    """J (b,D,d), Hess (b,D,d,d), image (b,D) of the sphere-projected decoder at latent codes z,
    by torch.func in the sealed VMAP_CHUNK width; then g, ginv, II = Hess - J Gamma, H = tr_g II."""
    decode_one = decoder_curvature.plain_decoder_map(curvature_model)
    Js: List[np.ndarray] = []; Hs: List[np.ndarray] = []
    for start in range(0, z.shape[0], chart_curvature.VMAP_CHUNK):
        real = z[start:start + chart_curvature.VMAP_CHUNK]
        chunk = chart_curvature._pad_to_chunk(real)
        J = vmap(jacrev(decode_one))(chunk)[: real.shape[0]]
        Hc = vmap(hessian(decode_one))(chunk)[: real.shape[0]]
        Js.append(J.detach().cpu().numpy()); Hs.append(Hc.detach().cpu().numpy())
    J = np.concatenate(Js); Hess = np.concatenate(Hs)
    with torch.no_grad():
        image = curvature_model.decode(z).detach().cpu().numpy()
    g = np.einsum("bai,baj->bij", J, J)
    ginv = np.linalg.inv(g)
    Gamma = np.einsum("bkl,bal,baij->bkij", ginv, J, Hess)
    II = Hess - np.einsum("bak,bkij->baij", J, Gamma)
    H = np.einsum("bij,baij->ba", ginv, II)
    return {"J": J, "Hess": Hess, "image": image, "g": g, "ginv": ginv, "II": II, "H": H,
            "cond_g": np.linalg.cond(g)}


def metric_norms(M: np.ndarray, ginv: np.ndarray) -> Dict[str, np.ndarray]:
    tr = np.einsum("bij,bji->b", ginv, M)
    fro2 = np.einsum("bij,bjk,bkl,bli->b", ginv, M, ginv, M)
    return {"tr": tr, "fro": np.sqrt(np.maximum(fro2, 0.0))}


def decoder_probe_facing(geo: Dict[str, np.ndarray], w: np.ndarray, d: int) -> Dict[str, np.ndarray]:
    J, ginv, II, H, x = geo["J"], geo["ginv"], geo["II"], geo["H"], geo["image"]
    wT = np.einsum("bai,a->bi", J, w)
    w_N = w[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, wT)
    b = np.einsum("baij,ba->bij", II, w_N)
    nm = metric_norms(b, ginv)
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    H_rad = np.einsum("ba,ba->b", H, u)
    H_tan = H - H_rad[:, None] * u
    return {"pf_curv_dec": nm["fro"], "pf_trace_dec": nm["tr"],
            "pf_trace_tan_dec": np.einsum("ba,ba->b", w_N, H_tan),
            "pf_trace_rad_dec": H_rad * np.einsum("ba,ba->b", w_N, u),
            "H_tan_norm_from_geo": np.linalg.norm(H_tan, axis=1), "H_rad_from_geo": H_rad,
            "w_N_norm": np.linalg.norm(w_N, axis=1)}


# --- colleague: B^S at the anchors -----------------------------------------------------------


def colleague_BS_at_anchors(X: np.ndarray, neigh: np.ndarray, d: int, est: Dict[str, Any], device: torch.device,
                            n_splits: int, seed: int) -> Dict[str, Any]:
    """His `nested_pca_frame` + `_fit_rank` per anchor (unchanged), keeping the fitted B^S of both
    halves of every split and averaging them (his H_mean is the same average of the halves).
    Returns per-anchor x0 (b,D), J (b,D,d), BS_flat mean (b,D,q), K_H_cross (b), n_splits_ok."""
    nested_pca_frame, _fit_rank, _rows_from_fits = est["nested_pca_frame"], est["_fit_rank"], est["_rows_from_fits"]
    n_anchors, k = neigh.shape
    D = X.shape[1]; q = d * (d + 1) // 2
    x0s = np.zeros((n_anchors, D)); Js = np.zeros((n_anchors, D, d)); BS = np.full((n_anchors, D, q), np.nan)
    kh = np.full(n_anchors, np.nan); ok = np.zeros(n_anchors, dtype=int)
    t0 = time.monotonic()
    for ai in range(n_anchors):
        Xloc = X[neigh[ai, :k]].astype(np.float64)
        x0, J, _ev, _diag = nested_pca_frame(Xloc, d, device)
        fits = _fit_rank(Xloc, x0, J, d, k, n_splits, seed, ai)
        x0s[ai] = x0; Js[ai] = J[:, :d]
        if fits:
            rec = _rows_from_fits(ai, d, k, fits)
            kh[ai] = float(rec["K_H_cross"]); ok[ai] = int(rec.get("n_splits_ok", len(fits)))
            BS[ai] = np.mean([0.5 * (f["BS_flat_A"] + f["BS_flat_B"]) for f in fits], axis=0)
        if (ai + 1) % 64 == 0 or ai + 1 == n_anchors:
            print(f"[colleague] {ai + 1}/{n_anchors} anchors, {time.monotonic() - t0:.0f}s", flush=True)
    return {"x0": x0s, "J": Js, "BS_flat": BS, "K_H_cross": kh, "n_splits_ok": ok, "wallclock_s": time.monotonic() - t0}


def colleague_probe_facing(cb: Dict[str, Any], w: np.ndarray, d: int, est_mod: Dict[str, Any]) -> Dict[str, np.ndarray]:
    unpack, probe_facing_scalar, project_normal = est_mod["unpack_BS_symmetric"], est_mod["probe_facing_scalar"], est_mod["project_normal"]
    n = cb["BS_flat"].shape[0]
    pf = np.full(n, np.nan); kw = np.full(n, np.nan); wn = np.full(n, np.nan)
    for ai in range(n):
        if not np.all(np.isfinite(cb["BS_flat"][ai])):
            continue
        wn_unit, wn_norm = project_normal(w, cb["x0"][ai], cb["J"][ai])       # his: orthogonal to span(x0, J)
        B = unpack(cb["BS_flat"][ai], d)                                       # (D, d, d)
        b = np.einsum("a,aij->ij", wn_norm * wn_unit, B)
        pf[ai] = float(np.linalg.norm(b))                                      # his J is orthonormal: plain Frobenius
        kw[ai] = float(probe_facing_scalar(cb["BS_flat"][ai], d, wn_unit)["K_w_dir"]) * wn_norm
        wn[ai] = wn_norm
    return {"pf_curv_col": pf, "K_w_dir_col": kw, "w_N_norm_col": wn}


# --- data ----------------------------------------------------------------------------------------


def load_physics(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.monotonic()
    emb = pl.load_physics_embeddings()
    X, n_rows = emb["X"], emb["n_rows"]
    print(f"[load] physics embeddings n_rows={n_rows} {time.monotonic() - t0:.1f}s", flush=True)
    table = pl.load_label_table(columns=list(pl.LABEL_COLUMN_MAP.values()))
    offset_perm = pl.shifted_pairing(n_rows, pl.ALIGNMENT_ASSUMED_OFFSET)
    labels = {name: pl.canonical_label(table, name, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)[offset_perm]
              for name in args.labels.split(",")}
    return {"X": np.asarray(X, dtype=np.float64), "labels": labels, "in_dim": pcp.AE_IN_DIM}


def load_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = adj.SMOKE
    G = adj.InSphereGenerator(cfg["d"], cfg["D"], cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=args.seed)
    z = adj.draw_latents(cfg["n"], cfg["d"], cfg["scale_choices"], cfg["scale_probs"], seed=args.seed + 1)
    X = adj.generate_points(G, z)
    rng = np.random.default_rng(args.seed + 7)
    a1 = rng.standard_normal(cfg["d"]); a1 /= np.linalg.norm(a1)
    y1 = z @ a1; y2 = np.sin(2 * y1) + (z @ rng.standard_normal(cfg["d"])) ** 2
    y2[rng.random(cfg["n"]) < 0.05] = np.nan                                    # exercise the finite-row path
    return {"X": X, "labels": {"lin": y1, "nonlin_with_nan": y2}, "in_dim": cfg["D"], "smoke_cfg": cfg}


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "physics"], required=True)
    p.add_argument("--d-values", type=str, default="16,20")
    p.add_argument("--labels", type=str, default=",".join((pl.PRIMARY_LABEL,) + pl.SECONDARY_LABELS))
    p.add_argument("--colleague-root", type=str, default=None)
    p.add_argument("--skip-colleague", action="store_true")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=20260905, help="smoke fixture seed only")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--n-permutations", type=int, default=None)
    args = p.parse_args()
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    if not args.skip_colleague and args.colleague_root is None:
        raise SystemExit("--colleague-root is required unless --skip-colleague")
    record_path = Path(args.record_path).resolve()
    for stem in PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")
    n_perm = args.n_permutations if args.n_permutations is not None else (200 if args.mode == "smoke" else 2000)
    max_epochs = args.max_epochs if args.max_epochs is not None else (adj.SMOKE_EPOCHS if args.mode == "smoke" else pcp.MAX_EPOCHS)
    d_values = [int(v) for v in args.d_values.split(",")] if args.mode == "physics" else [adj.SMOKE["d"]]
    out_root = pcp.resolve_output_root() / "probe-facing"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} d={d_values} max_epochs={max_epochs} n_perm={n_perm} out={out_root}")

    est = est_mod = None
    if not args.skip_colleague:
        est = colleague.load_colleague_estimator(args.colleague_root)
        from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric  # noqa: E402
        from geometry.physics_activation_atlas.effdim_curvature_metrics import probe_facing_scalar, project_normal  # noqa: E402
        est_mod = {"unpack_BS_symmetric": unpack_BS_symmetric, "probe_facing_scalar": probe_facing_scalar, "project_normal": project_normal}
        print(f"colleague checkout HEAD={est['colleague_head']} (expected {colleague.COLLEAGUE_COMMIT}); topology shim={est['topology_is_shim']}")

    data = load_physics(args) if args.mode == "physics" else load_smoke(args)
    X, labels, in_dim = data["X"], data["labels"], data["in_dim"]
    n = X.shape[0]
    k = pcp.K_NEIGHBOURS if args.mode == "physics" else adj.SMOKE["k"]
    n_anchors = pcp.N_ANCHORS if args.mode == "physics" else adj.SMOKE["n_anchors"]
    ks = [kk for kk in MULTISCALE_KS if kk <= k]

    _append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
             "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "colleague_head": est["colleague_head"] if est else None,
             "threads": args.threads, "device": args.device, "d_values": d_values, "labels": list(labels), "n": n, "k": k,
             "n_anchors": n_anchors, "multiscale_ks": ks, "n_permutations": n_perm, "max_epochs": max_epochs,
             "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
             "decoder_image_projection": pcp.DECODER_IMAGE_PROJECTION, "curvature_convention": decoder_curvature.CURVATURE_CONVENTION,
             "pre_registered": False, "gates": "nothing"}, record_path)

    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    t0 = time.monotonic()
    panel = pcp.knn_panel(X, a, k)
    log_r = panel["log_knn_radius"]
    log_r_multi = np.column_stack([np.log(panel["distances"][:, kk - 1]) for kk in ks])
    print(f"[knn] k={k} anchors={n_anchors} {time.monotonic() - t0:.1f}s; multiscale ks={ks}", flush=True)

    # per label: outcome, controls, probe weight, bias at the anchor
    per_label: Dict[str, Dict[str, Any]] = {}
    for name, y in labels.items():
        y = np.asarray(y, dtype=np.float64)
        y_hat = runner._oof_predictions_for_label(X, y, pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
        loc = pcp.local_r2_panel(y, y_hat, panel["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        Z_sealed = np.column_stack([log_r, loc["local_label_variance"], loc["local_evaluation_count"]])
        Z_multi = np.column_stack([log_r_multi, loc["local_label_variance"], loc["local_evaluation_count"]])
        fin = np.isfinite(y)
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X[fin], y[fin])
        per_label[name] = {"r2": loc["r2"], "Z_sealed": Z_sealed, "Z_multi": Z_multi, "w": ridge.coef_.astype(np.float64),
                           "bias_sq": (y[a] - y_hat[a]) ** 2, "n_masked": int(loc["n_masked_anchors"]),
                           "global_oof_r2": 1.0 - float(np.nansum((y - y_hat) ** 2) / np.nansum((y - np.nanmean(y)) ** 2))}
        print(f"[probe] {name}: global OOF R2 {per_label[name]['global_oof_r2']:.3f}, masked anchors {loc['n_masked_anchors']}, "
              f"local R2 p05/p50 {np.nanpercentile(loc['r2'], 5):.3f}/{np.nanpercentile(loc['r2'], 50):.3f}", flush=True)

    neigh = None
    if est is not None:
        neigh = colleague.colleague_neighbourhoods(X, a, k)["neigh"]

    for d in d_values:
        print("\n" + "=" * 78 + f"\nd={d}\n" + "=" * 78, flush=True)
        fit = fit_decoder(X, d, in_dim, max_epochs)
        with torch.no_grad():
            z_anchor = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
        t0 = time.monotonic()
        geo = decoder_geometry(fit["curvature_model"], z_anchor)
        t_geo = time.monotonic() - t0
        sealed = decoder_curvature.plain_decoder_curvature(fit["curvature_model"], z_anchor)
        H_sealed = sealed["H_vec"].detach().cpu().numpy()
        cos_check = float(np.median(np.einsum("ba,ba->b", geo["H"], H_sealed) / (np.linalg.norm(geo["H"], axis=1) * np.linalg.norm(H_sealed, axis=1))))
        dec = pcp.decompose_radial_tangential(H_sealed, geo["image"], pcp.MIN_IMAGE_NORM)
        print(f"[decoder] var_explained={fit['var_explained']:.5f} fit {fit['wallclock_fit_s']:.0f}s; geometry {t_geo:.0f}s; "
              f"median cos(H_geo, H_sealed)={cos_check:.9f}; H_rad median {np.nanmedian(dec['H_rad']):.4f}; cond(g) p50/p95 "
              f"{np.percentile(geo['cond_g'], 50):.2f}/{np.percentile(geo['cond_g'], 95):.2f}", flush=True)
        np.savez_compressed(out_root / f"09_probe_facing_geometry_d{d}.npz", anchor_idx=a, J=geo["J"].astype(np.float32),
                            Hess=geo["Hess"].astype(np.float32), image=geo["image"].astype(np.float32), z_anchor=z_anchor.detach().cpu().numpy())

        cb = None
        if est is not None:
            cb = colleague_BS_at_anchors(X, neigh, d, est, torch.device(args.device), colleague.COLLEAGUE_N_SPLITS, colleague.COLLEAGUE_SEED)
            print(f"[colleague] K_H_cross finite {int(np.isfinite(cb['K_H_cross']).sum())}/{n_anchors}, {cb['wallclock_s']:.0f}s; "
                  f"rank(K_H_cross, H_tan_norm) = {_spearman(cb['K_H_cross'], dec['H_tan_norm']):+.3f}", flush=True)

        _append({"experiment": EXPERIMENT, "row": "fit", "mode": args.mode, "d": d, "timestamp": _utc_now(),
                 "var_explained": fit["var_explained"], "wallclock_fit_s": fit["wallclock_fit_s"], "wallclock_geometry_s": t_geo,
                 "median_cos_H_geo_vs_sealed": cos_check, "H_rad_median": float(np.nanmedian(dec["H_rad"])),
                 "cond_g_p50_p95": [float(np.percentile(geo["cond_g"], q)) for q in (50, 95)],
                 "colleague_wallclock_s": cb["wallclock_s"] if cb else None,
                 "colleague_rank_vs_H_tan_norm": _spearman(cb["K_H_cross"], dec["H_tan_norm"]) if cb else None}, record_path)

        for name, L in per_label.items():
            cols: Dict[str, np.ndarray] = {"H_tan_norm": dec["H_tan_norm"]}
            dpf = decoder_probe_facing(geo, L["w"], d)
            cols.update({c: dpf[c] for c in ("pf_curv_dec", "pf_trace_tan_dec", "pf_trace_rad_dec")})
            cols["bias_sq"] = L["bias_sq"]
            if cb is not None:
                cpf = colleague_probe_facing(cb, L["w"], d, est_mod)
                cols["K_H_cross_col"] = cb["K_H_cross"]; cols.update({c: cpf[c] for c in ("pf_curv_col", "K_w_dir_col")})
            r2 = L["r2"]
            rows = {}
            print(f"\n[d={d}] {name}: global OOF R2 {L['global_oof_r2']:.3f}; |w_N|/|w| median (decoder frame) "
                  f"{np.median(dpf['w_N_norm']) / np.linalg.norm(L['w']):.3f}")
            print(f"{'column':18s} {'raw rho':>8s} {'sealed':>8s} {'p':>7s} {'multi':>8s} {'p':>7s} | {'vs H_tan':>9s} {'vs log r':>9s} {'vs pf_dec':>10s}")
            for c, x in cols.items():
                ps = partial_row(x, r2, L["Z_sealed"], n_perm); pm = partial_row(x, r2, L["Z_multi"], n_perm)
                rows[c] = {"raw_rho": _spearman(x, r2), "sealed": ps, "multiscale": pm,
                           "rho_vs_H_tan_norm": _spearman(x, dec["H_tan_norm"]), "rho_vs_log_r": _spearman(x, log_r),
                           "rho_vs_pf_curv_dec": _spearman(x, cols["pf_curv_dec"]), "median": float(np.nanmedian(x))}
                print(f"{c:18s} {rows[c]['raw_rho']:+8.3f} {ps['partial']:+8.3f} {ps['p']:7.4f} {pm['partial']:+8.3f} {pm['p']:7.4f} | "
                      f"{rows[c]['rho_vs_H_tan_norm']:+9.3f} {rows[c]['rho_vs_log_r']:+9.3f} {rows[c]['rho_vs_pf_curv_dec']:+10.3f}", flush=True)
            _append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "d": d, "label": name, "timestamp": _utc_now(),
                     "global_oof_r2": L["global_oof_r2"], "n_masked_anchors": L["n_masked"],
                     "local_r2_p05_p50_p95": [float(v) for v in np.nanpercentile(r2, [5, 50, 95])],
                     "w_N_fraction_median_decoder": float(np.median(dpf["w_N_norm"]) / np.linalg.norm(L["w"])),
                     "columns": rows}, record_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
