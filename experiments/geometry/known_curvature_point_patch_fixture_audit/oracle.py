"""Finite-patch population-optimal quadratic targets T2 (uniform volume) and T3 (sampling)."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .config import TOL_ORACLE_MED_REL
from .fixtures import GENERATORS_NP, autodiff_geometry
from .geometry import curvature_from_B, hess_from_bs_flat, sphere_normal_projectors


def _quad_phi(U: np.ndarray) -> np.ndarray:
    n, d = U.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(U[:, a] * U[:, b])
    return np.stack(cols, axis=1)


def _fit_population_quadratic(
    Y: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    weights: np.ndarray,
    ridge: float = 1e-8,
    U: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted LS of sphere-normal residual onto Phi(u).

    ``U`` should be generator-chart coordinates (exact latent offsets when
    available). Otherwise ``u = J^+ (Y-x0)``. Returns Hessian-convention B.
    """
    g, PT, PNS, Gh = sphere_normal_projectors(x0, J)
    d = J.shape[1]
    if U is None:
        try:
            ginv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            ginv = np.linalg.pinv(g)
        U = (Y - x0[None, :]) @ J @ ginv
    lin = x0[None, :] + U @ J.T
    resid = Y - lin
    resid = resid @ PNS.T
    Phi = _quad_phi(U)
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-18, None)
    sw = np.sqrt(w)
    Pw = Phi * sw[:, None]
    Rw = resid * sw[:, None]
    Gmat = Pw.T @ Pw + ridge * np.eye(Phi.shape[1])
    S = np.linalg.solve(Gmat, Pw.T @ Rw).T  # (D, q) Phi coefficients
    S = PNS @ S
    return hess_from_bs_flat(S, d)


def _latent_ball_samples(
    z0: np.ndarray,
    n: int,
    radius_lat: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sobol directions with linear radius (high-d U^{1/d} hugs the shell)."""
    n = int(max(n, 2))
    d = z0.shape[0]
    m = max(1, int(np.ceil(np.log2(n))))
    seed = int(rng.integers(1, 2**31))
    directions = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m)[:n]
    from scipy.special import erfinv

    gauss = np.sqrt(2.0) * erfinv(np.clip(2.0 * directions - 1.0, -1 + 1e-12, 1 - 1e-12))
    gauss /= np.clip(np.linalg.norm(gauss, axis=1, keepdims=True), 1e-15, None)
    rad_u = qmc.Sobol(d=1, scramble=True, seed=seed + 1).random_base2(m)[:n]
    out = z0[None, :] + gauss * (radius_lat * rad_u)
    out[0] = z0
    return out


def _physical_ball(
    name: str,
    z0: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    n: int,
    radius_lat: float,
    r_phys: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    zcand = _latent_ball_samples(z0, n, radius_lat, rng)
    Y = GENERATORS_NP[name](zcand, Q)
    keep = np.linalg.norm(Y - G[None, :], axis=1) <= r_phys
    return zcand[keep], Y[keep]


def _sqrt_det_g(z_k: np.ndarray, name: str, Q: np.ndarray, h: float = 1e-5, chunk: int = 512) -> np.ndarray:
    """sqrt(det g) via central-difference J, chunked so (chunk, D, d) stays ~50 MiB."""
    fn = lambda u: GENERATORS_NP[name](u, Q)
    n, d = z_k.shape
    Damb = int(fn(z_k[:1]).shape[1])
    w = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, chunk):
        sl = z_k[i0 : i0 + chunk]
        Jb = np.empty((len(sl), Damb, d), dtype=np.float64)
        for a in range(d):
            zp, zm = sl.copy(), sl.copy()
            zp[:, a] += h
            zm[:, a] -= h
            Jb[:, :, a] = (fn(zp) - fn(zm)) / (2.0 * h)
        g = np.einsum("ndi,ndj->nij", Jb, Jb)
        w[i0 : i0 + len(sl)] = np.sqrt(np.clip(np.linalg.det(g), 1e-18, None))
    return w


def patch_oracle(
    name: str,
    z0: np.ndarray,
    Q: np.ndarray,
    r_phys: float,
    n_qmc: int,
    sampling_logp_fn=None,
    seed: int = 0,
) -> dict:
    """Population quadratic at physical radius r_phys.

    T2: weights = sqrt(det g) (uniform volume in the generator chart).
    T3: weights = exp(log p) of the sampling density, if provided.
    """
    rng = np.random.default_rng(seed)
    geo0 = autodiff_geometry(name, z0, Q)
    J, G = geo0["J"], geo0["G"]
    svals = np.linalg.svd(J, compute_uv=False)
    r_lat = float(r_phys / max(np.median(svals), 1e-8))
    n_need = max(int(n_qmc), 32)
    z_k, Y_k = _physical_ball(name, z0, Q, G, max(2 * n_need, 256), r_lat * 1.4, r_phys, rng)
    if len(z_k) < max(64, n_need // 4):
        z_k, Y_k = _physical_ball(name, z0, Q, G, max(4 * n_need, 512), r_lat * 1.15, r_phys, rng)
    if len(z_k) < 32:
        return {"ok": False, "reason": "empty_patch", "n_keep": int(len(z_k))}
    if len(z_k) > n_need:
        take = rng.choice(len(z_k), size=n_need, replace=False)
        z_k, Y_k = z_k[take], Y_k[take]
    w2 = _sqrt_det_g(z_k, name, Q)
    if sampling_logp_fn is None:
        w3 = w2.copy()
    else:
        w3 = np.exp(np.asarray(sampling_logp_fn(z_k), dtype=np.float64))
        w3 = w3 / (np.mean(w3) + 1e-18)
    U_exact = z_k - z0[None, :]
    B2 = _fit_population_quadratic(Y_k, G, J, w2, U=U_exact)
    B3 = _fit_population_quadratic(Y_k, G, J, w3, U=U_exact)
    c2 = curvature_from_B(B2, geo0["g"])
    c3 = curvature_from_B(B3, geo0["g"])
    return {
        "ok": True,
        "n_keep": int(len(z_k)),
        "r_phys": float(r_phys),
        "r_lat": float(r_lat),
        "B_T2": B2,
        "B_T3": B3,
        "H_T2": c2["H"],
        "H_T3": c3["H"],
        "K_dir_T2": c2["K_dir"],
        "K_dir_T3": c3["K_dir"],
        "K_tf_T2": c2["K_tf"],
        "K_tf_T3": c3["K_tf"],
        "K_H_T2": c2["K_H"],
        "K_H_T3": c3["K_H"],
    }


_ORACLE_SHARED: dict = {}


def _cpu_worker_init() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"


def _oracle_pool_init(Q: np.ndarray, z_obs, logp_obs) -> None:
    _cpu_worker_init()
    shared: dict = {"Q": np.asarray(Q, dtype=np.float64)}
    if z_obs is not None and logp_obs is not None:
        from scipy.spatial import cKDTree

        shared["tree"] = cKDTree(np.asarray(z_obs, dtype=np.float64))
        shared["logp"] = np.asarray(logp_obs, dtype=np.float64)
    globals()["_ORACLE_SHARED"] = shared


def _oracle_worker(payload: dict) -> dict:
    """Picklable worker for ProcessPoolExecutor."""
    Q = payload.get("Q", _ORACLE_SHARED.get("Q"))
    logp_fn = None
    tree = _ORACLE_SHARED.get("tree")
    if tree is not None:
        p = _ORACLE_SHARED["logp"]

        def logp_fn(zz, tree=tree, p=p):
            _, j = tree.query(np.asarray(zz, dtype=np.float64), k=1)
            return p[np.asarray(j, dtype=np.intp)]

    try:
        orc = patch_oracle(
            payload["name"],
            payload["z0"],
            Q,
            r_phys=payload["r_phys"],
            n_qmc=payload["n_qmc"],
            sampling_logp_fn=logp_fn,
            seed=payload["seed"],
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "sample_id": payload["sample_id"], "error": str(e)}
    if not orc.get("ok"):
        return {"ok": False, "sample_id": payload["sample_id"], **{k: orc[k] for k in orc if k != "B_T2" and k != "B_T3"}}
    return {
        "ok": True,
        "sample_id": payload["sample_id"],
        "K_dir_T2": orc["K_dir_T2"],
        "K_dir_T3": orc["K_dir_T3"],
        "K_H_T2": orc["K_H_T2"],
        "K_H_T3": orc["K_H_T3"],
        "H_norm_T2": float(np.linalg.norm(orc["H_T2"])),
        "H_norm_T3": float(np.linalg.norm(orc["H_T3"])),
    }


def run_oracles(payloads: list[dict], n_workers: int) -> list[dict]:
    if not payloads:
        return []
    Q = payloads[0]["Q"]
    z_obs = payloads[0].get("z_obs")
    logp_obs = payloads[0].get("logp_obs")
    slim = [{k: v for k, v in p.items() if k not in ("Q", "z_obs", "logp_obs")} for p in payloads]
    workers = max(1, int(n_workers))
    if workers == 1 or len(slim) == 1:
        _oracle_pool_init(Q, z_obs, logp_obs)
        return [_oracle_worker(p) for p in slim]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(slim)),
        mp_context=ctx,
        initializer=_oracle_pool_init,
        initargs=(Q, z_obs, logp_obs),
    ) as ex:
        return list(ex.map(_oracle_worker, slim, chunksize=1))


def oracle_convergence(
    name: str, z0: np.ndarray, Q: np.ndarray, r_phys: float, seed: int = 0
) -> dict:
    """Double n_qmc until median tensor change <1% (single-anchor diagnostic)."""
    ns = [128, 256, 512, 1024]
    prev = None
    rows = []
    ok = False
    for n in ns:
        cur = patch_oracle(name, z0, Q, r_phys, n_qmc=n, seed=seed)
        if not cur.get("ok"):
            rows.append({"n": n, "ok": False})
            continue
        rec = {"n": n, "K_dir_T2": cur["K_dir_T2"], "K_dir_T3": cur["K_dir_T3"]}
        if prev is not None:
            rec["B_rel"] = float(
                np.linalg.norm(cur["B_T2"] - prev["B_T2"]) / max(np.linalg.norm(prev["B_T2"]), 1e-15)
            )
            rec["ok_double"] = bool(rec["B_rel"] < TOL_ORACLE_MED_REL)
            if rec["ok_double"]:
                ok = True
        rows.append(rec)
        prev = cur
        if ok:
            break
    return {"ok": ok, "rows": rows}
