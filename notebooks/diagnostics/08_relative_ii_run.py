"""Relative second fundamental form between two embeddings of the same galaxies, against per-point MKNN.

PURPOSE. Phases 7-8 conditioned per-point cross-survey agreement (MKNN between the HSC and Legacy
Survey DINOv3 embeddings) on ONE embedding's mean-curvature magnitude and found nothing free of
density. The second-order expansion of a linear alignment map A between the two embeddings says the
obstruction to linear alignment is the RELATIVE second fundamental form,

    II_rel(u,u) = II_G(L u, L u) - P_N^G [A II_F(u,u)],      A J_F ~ J_G L,

with F, G the two decoders, L the tangent map A induces between their latent charts and P_N^G the
projector onto G's normal space. This runner computes it pointwise at a seeded set of anchors from
two sphere-projected plain auto-encoders (one per embedding, the Phase 7 fit protocol) and a global
ridge map A from x_F to x_G, and correlates its norm with the per-point MKNN under density
controls in both ambient spaces. Beside it: ||H_tan|| of each decoder (the Phase 7 fields),
||II_F||, ||II_G||, the first-order obstruction ||A J_F - J_G L||/||A J_F||, an in-sphere variant
(both II with the sphere's radial part removed), a variant with L fitted locally in latent space,
and a decoder-free estimate: the quadratic coefficient of a local regression of the alignment
residual x_G - (A x_F + b) on F's tangent-projected coordinates, projected onto G's normal space.

NOT PRE-REGISTERED, GATES NOTHING. Writes only to its own record.

Usage:
    python notebooks/diagnostics/08_relative_ii_run.py --mode smoke --threads 8
    python notebooks/diagnostics/08_relative_ii_run.py --mode full --d 20 --n-anchors 2048 --threads 16 \\
        --pair-npz notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
sys.path.insert(0, str(NOTEBOOK_ROOT))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTICS_ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r7 = _load("crossmodal_curvature_run", "07_crossmodal_curvature_run.py")      # threads cap from --threads
r9 = _load("physics_curvature_run", "09_physics_curvature_run.py")            # SphereProjectedDecoder
adj = _load("instrument_adjudication_run", "09_instrument_adjudication_run.py")  # smoke generator
cc = r7.cc

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from torch.func import hessian, jacrev, vmap  # noqa: E402

from pu_manifold import cae, chart_curvature, decoder_curvature  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "relative-ii"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "08_relative_ii.jsonl"
DENSITY_KS = (10, 30, 100, 300)
COLUMNS = ("H_tan_F", "H_tan_G", "II_F", "II_G", "II_S_F", "II_S_G", "tan_resid", "II_rel", "II_rel_S",
           "II_rel_loc", "II_rel_emp", "align_resid")
ANCHOR_SEED = 20260912


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


def partial_row(x: np.ndarray, y: np.ndarray, Z: np.ndarray, n_perm: int) -> Dict[str, Any]:
    x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    try:
        fw = pcp.permutation_fwer({0: x[m]}, y[m], Z[m], n_perm, pcp.PERMUTATION_SEED)
        return {"partial": fw["per_d"][0]["observed_rho"], "p": fw["per_d"][0]["p"], "n": int(m.sum())}
    except ValueError as e:
        return {"partial": float("nan"), "p": float("nan"), "n": int(m.sum()), "reason": str(e)}


# --- decoders ---------------------------------------------------------------------------------


def fit_decoder(X: np.ndarray, d: int, max_epochs: int) -> Dict[str, Any]:
    """`r7.fit_and_field`'s fit steps, returning the model (sphere-projected for differentiation)."""
    torch.manual_seed(cc.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(in_dim=X.shape[1], latent_dim=d, hidden=cc.AE_HIDDEN, activation=cc.AE_ACTIVATION)
    train_idx, holdout_idx = cc.split_indices(X.shape[0], cc.SPLIT_SEED, cc.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32); x64 = torch.tensor(X, dtype=torch.float64)
    cfg = dict(cc.TRAIN_CFG); cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x32[torch.as_tensor(train_idx, dtype=torch.long)], cfg)
    wall = time.monotonic() - t0
    model.eval().double()
    with torch.no_grad():
        z = model.encode(x64)
        y_hold = model(x64[torch.as_tensor(holdout_idx, dtype=torch.long)])["y"]
    recon = cae.reconstruction_stats(x64[torch.as_tensor(holdout_idx, dtype=torch.long)], y_hold)
    var_explained = 1.0 - recon["mse_total"] / float((torch.linalg.norm(x64[torch.as_tensor(holdout_idx, dtype=torch.long)], dim=1) ** 2).mean())
    return {"model": model, "curv_model": r9.SphereProjectedDecoder(model).eval(), "z": z, "train_idx": train_idx,
            "holdout_idx": holdout_idx, "var_explained": float(var_explained), "wallclock_fit_s": wall}


def decoder_geometry(curv_model: torch.nn.Module, z: torch.Tensor) -> Dict[str, np.ndarray]:
    decode_one = decoder_curvature.plain_decoder_map(curv_model)
    Js: List[np.ndarray] = []; Hs: List[np.ndarray] = []
    for start in range(0, z.shape[0], chart_curvature.VMAP_CHUNK):
        real = z[start:start + chart_curvature.VMAP_CHUNK]
        chunk = chart_curvature._pad_to_chunk(real)
        Js.append(vmap(jacrev(decode_one))(chunk)[: real.shape[0]].detach().cpu().numpy())
        Hs.append(vmap(hessian(decode_one))(chunk)[: real.shape[0]].detach().cpu().numpy())
    J = np.concatenate(Js); Hess = np.concatenate(Hs)
    with torch.no_grad():
        image = curv_model.decode(z).detach().cpu().numpy()
    g = np.einsum("bai,baj->bij", J, J); ginv = np.linalg.inv(g)
    Gamma = np.einsum("bkl,bal,baij->bkij", ginv, J, Hess)
    II = Hess - np.einsum("bak,bkij->baij", J, Gamma)
    H = np.einsum("bij,baij->ba", ginv, II)
    xhat = image / np.linalg.norm(image, axis=1, keepdims=True)
    II_rad = np.einsum("baij,ba->bij", II, xhat)
    II_S = II - np.einsum("ba,bij->baij", xhat, II_rad)
    H_rad = np.einsum("ba,ba->b", H, xhat)
    H_tan = H - H_rad[:, None] * xhat
    return {"J": J, "g": g, "ginv": ginv, "II": II, "II_S": II_S, "H_tan_norm": np.linalg.norm(H_tan, axis=1),
            "H_rad": H_rad, "xhat": xhat, "cond_g": np.linalg.cond(g)}


def fro_g(T: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """Frobenius norm of a (b, D, d, d) normal-valued 2-tensor: sum over ambient, g^{-1} on both slots."""
    return np.sqrt(np.maximum(np.einsum("baij,bjk,bakl,bli->b", T, ginv, T, ginv), 0.0))


def quad_design(u: np.ndarray) -> np.ndarray:
    n, d = u.shape
    iu, ju = np.triu_indices(d)
    quad = u[:, iu] * u[:, ju]; quad[:, iu == ju] *= 0.5
    return np.concatenate([np.ones((n, 1)), u, quad], axis=1)


def unpack_sym_batch(coef: np.ndarray, d: int) -> np.ndarray:
    """coef (q, D) -> (D, d, d)."""
    iu, ju = np.triu_indices(d)
    B = np.zeros((coef.shape[1], d, d)); B[:, iu, ju] = coef.T; B[:, ju, iu] = coef.T
    return B


def relative_columns(gF: Dict[str, np.ndarray], gG: Dict[str, np.ndarray], A: np.ndarray, b: np.ndarray,
                     XF: np.ndarray, XG: np.ndarray, zF: np.ndarray, zG: np.ndarray, anchors: np.ndarray,
                     neigh_align: np.ndarray, neigh_emp: np.ndarray, d: int) -> Dict[str, np.ndarray]:
    JF, JG, ginvF, ginvG = gF["J"], gG["J"], gF["ginv"], gG["ginv"]
    n = JF.shape[0]
    # tangent map induced by A, and the first-order obstruction
    AJF = np.einsum("ab,cbi->cai", A, JF)                                              # (b, D, d): A J_F
    JG_pinv = np.einsum("bij,baj->bia", ginvG, JG)                                     # (b, d, D): g_G^{-1} J_G^T
    L = np.einsum("bia,baj->bij", JG_pinv, AJF)                                        # (b, d, d)
    JGL = np.einsum("bak,bkj->baj", JG, L)
    tan_resid = np.linalg.norm((AJF - JGL).reshape(n, -1), axis=1) / np.maximum(np.linalg.norm(AJF.reshape(n, -1), axis=1), 1e-300)
    PN_G = lambda T: T - np.einsum("bak,bkc,bcij->baij", JG, JG_pinv, T)                # project onto N_G

    def rel(IIF: np.ndarray, IIG: np.ndarray, Lmap: np.ndarray) -> np.ndarray:
        lift = np.einsum("bakl,bki,blj->baij", IIG, Lmap, Lmap)                        # II_G(Lu, Lu)
        push = np.einsum("ab,cbij->caij", A, IIF)                                       # A II_F(u,u)
        return fro_g(lift - PN_G(push), ginvF)

    II_rel = rel(gF["II"], gG["II"], L)
    II_rel_S = rel(gF["II_S"], gG["II_S"], L)
    # locally fitted latent map: dz_G ~ L_loc dz_F over the anchor's neighbours (in F ambient)
    L_loc = np.zeros_like(L)
    for i in range(n):
        idx = neigh_align[i]
        dF = zF[idx] - zF[anchors[i]][None, :]; dG = zG[idx] - zG[anchors[i]][None, :]
        L_loc[i] = np.linalg.lstsq(dF, dG, rcond=None)[0].T                            # dG ~ dF L^T
    II_rel_loc = rel(gF["II"], gG["II"], L_loc)
    # decoder-free: quadratic coefficient of the alignment residual on F's tangent-projected coordinates
    II_rel_emp = np.full(n, np.nan); align_resid = np.full(n, np.nan)
    for i in range(n):
        idx = neigh_emp[i]
        dx = XF[idx] - XF[anchors[i]][None, :]
        u = dx @ JF[i] @ ginvF[i]
        r = XG[idx] - (XF[idx] @ A.T + b[None, :])
        coef = np.linalg.lstsq(quad_design(u), r, rcond=None)[0]                        # (q, D)
        B = unpack_sym_batch(coef[1 + d:], d)[None]                                    # (1, D, d, d)
        II_rel_emp[i] = fro_g((B[0] - np.einsum("ak,kc,cij->aij", JG[i], JG_pinv[i], B[0]))[None], ginvF[i:i + 1])[0]
        align_resid[i] = float(np.sqrt((r ** 2).sum(axis=1).mean()))
    return {"H_tan_F": gF["H_tan_norm"], "H_tan_G": gG["H_tan_norm"],
            "II_F": fro_g(gF["II"], ginvF), "II_G": fro_g(gG["II"], ginvG),
            "II_S_F": fro_g(gF["II_S"], ginvF), "II_S_G": fro_g(gG["II_S"], ginvG),
            "tan_resid": tan_resid, "II_rel": II_rel, "II_rel_S": II_rel_S, "II_rel_loc": II_rel_loc,
            "II_rel_emp": II_rel_emp, "align_resid": align_resid,
            "_L_vs_Lloc": np.linalg.norm((L - L_loc).reshape(n, -1), axis=1) / np.maximum(np.linalg.norm(L.reshape(n, -1), axis=1), 1e-300)}


# --- data ---------------------------------------------------------------------------------------


def load_pair(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "smoke":
        cfg = adj.SMOKE
        G = adj.InSphereGenerator(cfg["d"], cfg["D"], cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=args.seed)
        z = adj.draw_latents(cfg["n"], cfg["d"], cfg["scale_choices"], cfg["scale_probs"], seed=args.seed + 1)
        XF = adj.generate_points(G, z)
        rng = np.random.default_rng(args.seed + 3)
        Q, _ = np.linalg.qr(rng.standard_normal((cfg["D"], cfg["D"])))
        XG = XF @ Q.T + 0.01 * rng.standard_normal(XF.shape)                          # a rotation plus noise: II_rel ~ 0
        XG /= np.linalg.norm(XG, axis=1, keepdims=True)
        return {"XF": XF, "XG": XG, "source": "smoke: in-sphere generator, G = rotation + 1% noise", "sha256": None}
    if args.pair_npz:
        with np.load(args.pair_npz) as z:
            XF = np.asarray(z[cc.PU_COLUMN_A], dtype=np.float64); XG = np.asarray(z[cc.PU_COLUMN_B], dtype=np.float64)
        src = args.pair_npz
    else:
        XF, XG, src = r7.load_pu_pair(cc.PU_COLUMN_A, cc.PU_COLUMN_B)
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    return {"XF": XF, "XG": XG, "source": src, "sha256": sha}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--n-anchors", type=int, default=2048)
    p.add_argument("--k-align", type=int, default=256)
    p.add_argument("--k-emp", type=int, default=1024)
    p.add_argument("--alpha-align", type=float, default=1.0)
    p.add_argument("--pair-npz", type=str, default=None)
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260905, help="smoke generator seed")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--n-permutations", type=int, default=None)
    args = p.parse_args()
    assert r7._THREADS == args.threads, (r7._THREADS, args.threads)
    record_path = Path(args.record_path).resolve()
    if record_path.name.startswith(("07_crossmodal_curvature", "08_cka", "09_physics_curvature")):
        raise SystemExit(f"refusing to write to a production record path: {record_path}")
    n_perm = args.n_permutations if args.n_permutations is not None else (200 if args.mode == "smoke" else 2000)
    max_epochs = args.max_epochs if args.max_epochs is not None else (30 if args.mode == "smoke" else cc.MAX_EPOCHS)
    d = adj.SMOKE["d"] if args.mode == "smoke" else args.d
    n_anchors = 128 if args.mode == "smoke" else args.n_anchors
    k_align = min(args.k_align, 64) if args.mode == "smoke" else args.k_align
    k_emp = min(args.k_emp, 256) if args.mode == "smoke" else args.k_emp
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} d={d} anchors={n_anchors} k_align={k_align} k_emp={k_emp} n_perm={n_perm} epochs={max_epochs}")

    data = load_pair(args)
    XF, XG = data["XF"], data["XG"]
    n = XF.shape[0]
    normsF, normsG = np.linalg.norm(XF, axis=1), np.linalg.norm(XG, axis=1)
    print(f"[data] n={n} D={XF.shape[1]} |x_F| {normsF.min():.4f}-{normsF.max():.4f} |x_G| {normsG.min():.4f}-{normsG.max():.4f} from {data['source']}")
    XF = XF / normsF[:, None]; XG = XG / normsG[:, None]                             # unit sphere, as the Phase 7 fields assumed
    _append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
             "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "d": d, "n": n, "n_anchors": n_anchors,
             "k_align": k_align, "k_emp": k_emp, "alpha_align": args.alpha_align, "density_ks": list(DENSITY_KS), "mknn_k": [cc.HEADLINE_K, 50],
             "n_permutations": n_perm, "max_epochs": max_epochs, "pair_source": data["source"], "pair_sha256": data["sha256"],
             "row_norm_range_F": [float(normsF.min()), float(normsF.max())], "row_norm_range_G": [float(normsG.min()), float(normsG.max())],
             "anchor_seed": ANCHOR_SEED, "columns": COLUMNS, "torch": torch.__version__, "numpy": np.__version__,
             "python": sys.version.split()[0], "pre_registered": False, "gates": "nothing"}, record_path)

    # agreement and density
    mknn = {k: cc.per_point_mknn(XF, XG, k) for k in (cc.HEADLINE_K, 50)}
    print(f"[mknn] k=20 mean {mknn[cc.HEADLINE_K].mean():.4f} (chance {cc.HEADLINE_K / n:.4f}); k=50 mean {mknn[50].mean():.4f}", flush=True)
    kmax = max(max(DENSITY_KS), k_emp, k_align) + 1
    nnF = NearestNeighbors(n_neighbors=kmax).fit(XF); dF, iF = nnF.kneighbors(XF)
    nnG = NearestNeighbors(n_neighbors=kmax).fit(XG); dG, _ = nnG.kneighbors(XG)
    logr = {("F", k): np.log(dF[:, k]) for k in DENSITY_KS}; logr.update({("G", k): np.log(dG[:, k]) for k in DENSITY_KS})

    # decoders
    fits = {}
    for name, X in (("F", XF), ("G", XG)):
        fits[name] = fit_decoder(X, d, max_epochs)
        print(f"[decoder {name}] var_explained={fits[name]['var_explained']:.4f} fit {fits[name]['wallclock_fit_s']:.0f}s", flush=True)
    # global linear alignment A: x_G ~ A x_F + b, fit on F's training rows
    tr, ho = fits["F"]["train_idx"], fits["F"]["holdout_idx"]
    ridge = Ridge(alpha=args.alpha_align).fit(XF[tr], XG[tr])
    A, b = ridge.coef_.astype(np.float64), ridge.intercept_.astype(np.float64)
    def r2(idx):
        pred = XF[idx] @ A.T + b
        return 1.0 - float(((XG[idx] - pred) ** 2).sum() / ((XG[idx] - XG[idx].mean(axis=0)) ** 2).sum())
    print(f"[align] ridge alpha={args.alpha_align}: R2 train {r2(tr):.4f} holdout {r2(ho):.4f}", flush=True)

    rng = np.random.default_rng(ANCHOR_SEED)
    anchors = np.sort(rng.choice(n, size=n_anchors, replace=False))
    zF_all = fits["F"]["z"].detach().cpu().numpy(); zG_all = fits["G"]["z"].detach().cpu().numpy()
    t0 = time.monotonic()
    gF = decoder_geometry(fits["F"]["curv_model"], fits["F"]["z"][torch.as_tensor(anchors, dtype=torch.long)])
    gG = decoder_geometry(fits["G"]["curv_model"], fits["G"]["z"][torch.as_tensor(anchors, dtype=torch.long)])
    print(f"[geometry] {n_anchors} anchors x 2 decoders in {time.monotonic() - t0:.0f}s; H_rad median F {np.median(gF['H_rad']):.3f} G {np.median(gG['H_rad']):.3f}; "
          f"cond(g) p50 F {np.median(gF['cond_g']):.1f} G {np.median(gG['cond_g']):.1f}", flush=True)
    t0 = time.monotonic()
    cols = relative_columns(gF, gG, A, b, XF, XG, zF_all, zG_all, anchors, iF[anchors, 1:k_align + 1], iF[anchors, 1:k_emp + 1], d)
    print(f"[relative] columns in {time.monotonic() - t0:.0f}s; medians " + ", ".join(f"{c}={np.nanmedian(cols[c]):.3g}" for c in COLUMNS)
          + f"; |L-L_loc|/|L| p50 {np.nanmedian(cols['_L_vs_Lloc']):.2f}", flush=True)

    Z_sealed = np.column_stack([logr[("F", 30)][anchors], logr[("G", 30)][anchors]])
    Z_multi = np.column_stack([logr[(s, k)][anchors] for s in ("F", "G") for k in DENSITY_KS])
    for kk, m_all in mknn.items():
        m = m_all[anchors]
        print(f"\n[mknn k={kk}] {'column':12s} {'raw rho':>8s} {'sealed':>8s} {'p':>7s} {'multi':>8s} {'p':>7s} | {'vs logr_F':>9s} {'vs logr_G':>9s} {'vs II_F':>8s} {'median':>9s}")
        rows = {}
        for c in COLUMNS:
            x = cols[c]
            ps = partial_row(x, m, Z_sealed, n_perm); pm = partial_row(x, m, Z_multi, n_perm)
            rows[c] = {"raw_rho": _spearman(x, m), "sealed": ps, "multiscale": pm,
                       "rho_vs_logr_F": _spearman(x, logr[("F", 30)][anchors]), "rho_vs_logr_G": _spearman(x, logr[("G", 30)][anchors]),
                       "rho_vs_II_F": _spearman(x, cols["II_F"]), "median": float(np.nanmedian(x))}
            print(f"{'':13s}{c:12s} {rows[c]['raw_rho']:+8.3f} {ps['partial']:+8.3f} {ps['p']:7.4f} {pm['partial']:+8.3f} {pm['p']:7.4f} | "
                  f"{rows[c]['rho_vs_logr_F']:+9.3f} {rows[c]['rho_vs_logr_G']:+9.3f} {rows[c]['rho_vs_II_F']:+8.3f} {rows[c]['median']:9.3g}", flush=True)
        _append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "d": d, "mknn_k": kk, "timestamp": _utc_now(),
                 "n_anchors": n_anchors, "mknn_mean_all": float(m_all.mean()), "chance_floor": float(kk / n),
                 "var_explained_F": fits["F"]["var_explained"], "var_explained_G": fits["G"]["var_explained"],
                 "align_r2_train": r2(tr), "align_r2_holdout": r2(ho),
                 "rho_mknn_logr_F": _spearman(m, logr[("F", 30)][anchors]), "rho_mknn_logr_G": _spearman(m, logr[("G", 30)][anchors]),
                 "rho_II_rel_vs_II_rel_emp": _spearman(cols["II_rel"], cols["II_rel_emp"]), "rho_II_rel_vs_II_rel_S": _spearman(cols["II_rel"], cols["II_rel_S"]),
                 "rho_II_rel_vs_II_rel_loc": _spearman(cols["II_rel"], cols["II_rel_loc"]), "L_vs_Lloc_p50": float(np.nanmedian(cols["_L_vs_Lloc"])),
                 "H_rad_median": [float(np.median(gF["H_rad"])), float(np.median(gG["H_rad"]))], "columns": rows}, record_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
