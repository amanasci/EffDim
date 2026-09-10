"""End-to-end known-answer audit. Writes only into the new output tree."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .config import (
    ANCHOR_HASH_SEED,
    D_LAT,
    DIRECTION_SEED,
    ETA_LEVELS,
    ExpConfig,
    FROZEN_ESTIMATOR_PATHS,
    K_GRID_SMALL_N,
    N_COLLEAGUE,
    N_SPLITS_Q,
    N_TANGENT_DIRS,
    OUT_REL,
    PRIMARY_ETA,
    PRIMARY_K,
    PRIMARY_N,
    SPLIT_SEED,
)
from .decision import decide
from .estimator_d import (
    decoder_curvature_at_latents,
    encoder_latents,
    reconstruction_stats,
    split_indices,
    train_decoder,
)
from .estimator_q import fit_anchors_parallel, knn_fixed_radius, knn_indices
from .fixtures import (
    GENERATORS_NP,
    analytic_geometry,
    ambient_rotation,
    autodiff_geometry,
    batch_autodiff_geometry,
    bump_field,
    fixture_definitions,
    invariance_checks,
    make_torch_map,
    validate_point,
)
from .figures import make_figures
from .geometry import cosine, directional_tensor_cos, kdir_from_pair, rel_err, whiten_B
from .io_util import (
    assert_not_preserved,
    hash_stable_order,
    peak_rss_mb,
    platonic_root,
    resolve_path,
    sha256_file16,
    write_df,
    write_json,
    write_text,
)
from .noise import apply_noise
from .oracle import oracle_convergence, patch_oracle, run_oracles
from .reports import write_reports
from .sampling import sample_condition, sample_latent_S0
from .scoring import partial_spearman, rank_cal, vector_recovery
from .tests_unit import run_unit_tests


def _device(cfg: ExpConfig) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.device)
    return torch.device("cpu")


def _hash_estimators(root: Path) -> dict:
    out = {}
    for rel in FROZEN_ESTIMATOR_PATHS:
        p = root / rel
        out[rel] = sha256_file16(p) if p.exists() else "MISSING"
    return out


def _k_proxy_fn(name: str):
    if name in ("F4", "F5"):
        from .config import F4_WIDTHS, F5_WIDTHS

        w = F4_WIDTHS if name == "F4" else F5_WIDTHS
        return lambda z: np.abs(bump_field(z, w))
    return lambda z: np.zeros(len(z), dtype=np.float64)


def _batch_J(name: str, Z: np.ndarray, Q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    fn = lambda u: GENERATORS_NP[name](u, Q)
    cols = []
    for a in range(Z.shape[1]):
        zp, zm = Z.copy(), Z.copy()
        zp[:, a] += h
        zm[:, a] -= h
        cols.append((fn(zp) - fn(zm)) / (2.0 * h))
    return np.stack(cols, axis=-1)


def _seeded_dirs(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _median_knn_radius(X: np.ndarray, k: int, n_probe: int = 16) -> float:
    """Median primary-neighbourhood radius from a small clean probe set."""
    n = len(X)
    kk = int(min(max(k, 8), n - 1))
    step = max(1, n // max(n_probe, 1))
    probe = X[::step][:n_probe]
    device = torch.device("cpu")
    idx = knn_indices(X, probe, k=kk, device=device)
    rad = []
    for i in range(len(probe)):
        rad.append(float(np.max(np.linalg.norm(X[idx[i]] - probe[i], axis=1))))
    return float(np.median(rad)) if rad else 0.25


def build_observed(
    name: str,
    sampling: str,
    noise: str,
    n: int,
    eta: float,
    seed: int,
    Q: np.ndarray,
    r_med_hint: float | None,
    k_for_eta: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    samp = sample_condition(name, sampling, n, rng, Q, k_true_fn=_k_proxy_fn(name))
    z = samp["z"]
    X_clean = GENERATORS_NP[name](z, Q)
    sample_ids = np.arange(n, dtype=np.int64)
    J_all = _batch_J(name, z, Q)
    if r_med_hint is not None:
        r_med = float(r_med_hint)
    else:
        r_med = _median_knn_radius(X_clean, k=int(k_for_eta or 256))
    k_proxy = _k_proxy_fn(name)(z)
    nname = noise
    if noise == "N5n":
        nname = "N5n"
    noisy = apply_noise(
        name=name,
        noise=nname if nname != "N5" else "N5",
        X_clean=X_clean,
        z=z,
        Q=Q,
        generator=GENERATORS_NP[name],
        jacobian_fn=lambda zz: _batch_J(name, zz, Q),
        eta=eta,
        r_med=r_med,
        rng=rng,
        k_true=k_proxy,
    )
    return {
        "z": z,
        "X_clean": X_clean,
        "X": noisy["X"],
        "sample_ids": sample_ids,
        "p_select": samp["p_select"],
        "log_density": samp["log_density"],
        "sampling_meta": {k: samp[k] for k in samp if k not in ("z", "p_select", "log_density")},
        "noise_diag": noisy["diagnostics"],
        "J_all": J_all,
    }


def select_anchors(sample_ids: np.ndarray, holdout_idx: np.ndarray, n_anc: int) -> np.ndarray:
    ho_ids = sample_ids[holdout_idx]
    ordered = hash_stable_order(ho_ids, ANCHOR_HASH_SEED)
    return ordered[:n_anc]


def run_cell(
    cfg: ExpConfig,
    out: Path,
    *,
    suite: str,
    name: str,
    sampling: str,
    noise: str,
    n: int,
    k: int,
    eta: float,
    Q: np.ndarray,
    device: torch.device,
    do_decoder: bool,
    do_oracle: bool,
    r_fixed: float | None = None,
    adaptive_k: bool = False,
    weighted_q: bool = False,
) -> dict:
    t1 = time.time()
    tag = f"{suite}_{name}_{sampling}_{noise}_n{n}_k{k}_eta{eta:.2f}"
    if r_fixed is not None:
        tag += f"_r{r_fixed:.4f}"
    if adaptive_k:
        tag += "_adaptk"
    if weighted_q:
        tag += "_invw"
    cell_dir = out / "cells" / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    done = cell_dir / "cell.json"
    if done.exists() and not cfg.force:
        return json.loads(done.read_text())

    seed = cfg.seed + 17 * (sum(map(ord, tag)) % 10007)
    cloud = build_observed(
        name, sampling, noise, n, eta, seed, Q, r_med_hint=None, k_for_eta=k
    )
    X, z = cloud["X"], cloud["z"]
    train_idx, holdout_idx = split_indices(n, SPLIT_SEED)
    anc_ids = select_anchors(cloud["sample_ids"], holdout_idx, cfg.n_anc())
    # map sample_id -> row
    id_to_row = {int(s): i for i, s in enumerate(cloud["sample_ids"])}
    anc_rows = np.array([id_to_row[int(s)] for s in anc_ids], dtype=np.int64)
    z_anc = z[anc_rows]
    X_anc = X[anc_rows]

    # physical radius from a probe kNN on a subset to set noise scale already done;
    # compute neighbour packs
    neigh = knn_indices(X, X_anc, k=min(k, n - 1), device=device)
    radii = np.array(
        [float(np.max(np.linalg.norm(X[neigh[i]] - X_anc[i], axis=1))) for i in range(len(anc_rows))]
    )
    r_med = float(np.median(radii))
    if r_fixed is not None:
        neigh_list = []
        k_used = []
        for i in range(len(anc_rows)):
            idx = knn_fixed_radius(X, X_anc[i], r_fixed)
            if len(idx) < 40:
                idx = neigh[i]
            neigh_list.append(idx)
            k_used.append(len(idx))
        radii = np.array(
            [float(np.max(np.linalg.norm(X[idx] - X_anc[i], axis=1))) for i, idx in enumerate(neigh_list)]
        )
    elif adaptive_k:
        # choose per-anchor k so max radius ≈ median radius of the global k pack
        target_r = r_med
        neigh_list = []
        k_used = []
        for i in range(len(anc_rows)):
            d = np.linalg.norm(X - X_anc[i], axis=1)
            idx = np.argsort(d)
            # smallest m with d[m] >= target_r
            m = int(np.searchsorted(d[idx], target_r))
            m = max(64, min(m, n - 1))
            neigh_list.append(idx[:m])
            k_used.append(m)
        radii = np.full(len(anc_rows), target_r)
    else:
        neigh_list = [neigh[i] for i in range(len(anc_rows))]
        k_used = [int(min(k, n - 1))] * len(anc_rows)

    # T1 truth at anchors (clean generator, even in noise conditions)
    geos = batch_autodiff_geometry(name, z_anc, Q, chunk=8 if (cfg.smoke or cfg.factorial_smoke) else 16)
    dirs = _seeded_dirs(D_LAT, N_TANGENT_DIRS, DIRECTION_SEED)
    if sampling == "S1":
        rho_s1 = spearmanr(
            cloud["log_density"][anc_rows],
            np.array([g["K_dir"] for g in geos], dtype=np.float64),
        ).correlation
        cloud["sampling_meta"]["rho_logp_Kdir_anchors"] = (
            float(rho_s1) if rho_s1 is not None else float("nan")
        )

    # Oracle T2/T3 at subset of anchors. Constant-curvature S0: T2=T3=T1 exactly.
    oracle_rows = []
    constant_clean = name in ("F0", "F1", "F2", "F3")
    n_oracle = min(len(anc_rows), 8 if (cfg.smoke or cfg.factorial_smoke) else (256 if do_oracle else 0))
    if constant_clean:
        for i in range(len(anc_rows)):
            geo = geos[i]
            oracle_rows.append(
                {
                    "sample_id": int(anc_ids[i]),
                    "K_dir_T2": geo["K_dir"],
                    "K_dir_T3": geo["K_dir"],
                    "K_H_T2": geo["K_H"],
                    "K_H_T3": geo["K_H"],
                    "H_norm_T2": geo["H_norm"],
                    "H_norm_T3": geo["H_norm"],
                    "oracle_note": "constant_curvature_T2_eq_T1",
                }
            )
    elif do_oracle and not cfg.skip_oracle:
        payloads = []
        z_obs = z if sampling != "S0" else None
        logp_obs = cloud["log_density"] if sampling != "S0" else None
        for i in range(n_oracle):
            payloads.append(
                {
                    "name": name,
                    "z0": z_anc[i],
                    "Q": Q,
                    "r_phys": max(float(radii[i]), 1e-3),
                    "n_qmc": cfg.oracle_qmc(),
                    "sampling_logp_fn": None,
                    "seed": seed + i,
                    "sample_id": int(anc_ids[i]),
                    "z_obs": z_obs,
                    "logp_obs": logp_obs,
                }
            )
        print(f"[cell] oracle n={len(payloads)} workers={cfg.n_workers}", flush=True)
        for orc in run_oracles(payloads, cfg.n_workers):
            if orc.get("ok"):
                oracle_rows.append(
                    {
                        "sample_id": int(orc["sample_id"]),
                        "K_dir_T2": orc["K_dir_T2"],
                        "K_dir_T3": orc["K_dir_T3"],
                        "K_H_T2": orc["K_H_T2"],
                        "K_H_T3": orc["K_H_T3"],
                        "H_norm_T2": orc["H_norm_T2"],
                        "H_norm_T3": orc["H_norm_T3"],
                    }
                )

    # Estimator Q (CPU process pool; CUDA is not safe across fit threads)
    n_splits = 1 if (cfg.smoke or cfg.factorial_smoke or suite != "A") else N_SPLITS_Q
    q_payloads = []
    for i, ai in enumerate(anc_rows):
        w = None
        if weighted_q:
            w = 1.0 / np.clip(np.exp(cloud["log_density"][neigh_list[i]]), 1e-8, None)
        q_payloads.append(
            {
                "Xloc": X[neigh_list[i]],
                "d": D_LAT,
                "n_splits": n_splits,
                "seed": seed,
                "ai": int(ai),
                "weights": w,
            }
        )
    print(f"[cell] quadratic n={len(q_payloads)} splits={n_splits} workers={cfg.n_workers}", flush=True)
    fits = fit_anchors_parallel(q_payloads, cfg.n_workers)
    q_rows = []
    for i, ai in enumerate(anc_rows):
        fit = fits[i] if i < len(fits) else {"ok": False}
        geo = geos[i]
        rec = {
            "sample_id": int(anc_ids[i]),
            "row": int(ai),
            "fixture": name,
            "sampling": sampling,
            "noise": noise,
            "suite": suite,
            "n": n,
            "k": int(k_used[i]),
            "k_request": int(k),
            "radius": float(radii[i]),
            "eta": eta,
            "H_norm_T1": geo["H_norm"],
            "K_dir_T1": geo["K_dir"],
            "K_tf_T1": geo["K_tf"],
            "K_H_T1": geo["K_H"],
        }
        if fit.get("ok"):
            rec.update({f"Q_{k}": v for k, v in fit["agg"].items()})
            rec["Q_split_R_BS"] = float(np.mean([s.get("R_BS", np.nan) for s in fit["splits"]]))
            rec["Q_split_R_H"] = float(np.mean([s.get("R_H", np.nan) for s in fit["splits"]]))
            if fit.get("Hess_A") is not None:
                Bw = whiten_B(geo["B"], geo["g"])
                Bhat = 0.5 * (fit["Hess_A"] + fit["Hess_B"]) if fit.get("Hess_B") is not None else fit["Hess_A"]
                # transport Bhat into truth whitened coords via J_est -> J_true Procrustes
                Jt = geo["J"]
                Je = fit["J"][:, : D_LAT]
                # principal vectors: polar alignment
                M = Je.T @ Jt
                U, _, Vt = np.linalg.svd(M)
                R = U @ Vt
                Bhat_w = whiten_B(Bhat, Je.T @ Je)
                Bhat_al = np.einsum("Dab,ai,bj->Dij", Bhat_w, R, R)
                rec["Q_tensor_dir_cos"] = directional_tensor_cos(Bhat_al, Bw, dirs)
                rec["Q_H_cosine"] = cosine(fit["Hess_A"].trace(axis1=1, axis2=2) / D_LAT, geo["H"])
        q_rows.append(rec)

    qdf = pd.DataFrame(q_rows)
    if oracle_rows:
        qdf = qdf.merge(pd.DataFrame(oracle_rows), on="sample_id", how="left")

    # Estimator D
    d_seed_fields = []
    recon = {}
    if do_decoder and not cfg.skip_decoder:
        try:
            seeds = cfg.seeds() if suite == "A" else (cfg.seeds()[0],)
            for sd in seeds:
                model, info = train_decoder(X, train_idx, seed=int(sd), epochs=cfg.epochs(), device=str(device))
                recon[str(sd)] = {
                    "train": reconstruction_stats(model, X, train_idx, str(device)),
                    "holdout": reconstruction_stats(model, X, holdout_idx, str(device)),
                    "fit": {k: info.get(k) for k in ("best_loss", "epochs") if k in info} if isinstance(info, dict) else {},
                }
                z_lat = encoder_latents(model, X_anc)
                geos_d = decoder_curvature_at_latents(model, z_lat)
                field = []
                for i, gd in enumerate(geos_d):
                    field.append(
                        {
                            "sample_id": int(anc_ids[i]),
                            "seed": int(sd),
                            "H_norm_D": gd["H_norm"],
                            "K_dir_D": gd["K_dir"],
                            "K_tf_D": gd["K_tf"],
                            "K_H_D": gd["K_H"],
                            "D_H_cosine": cosine(gd["H"], geos[i]["H"]),
                            "D_tensor_dir_cos": directional_tensor_cos(
                                whiten_B(gd["B"], gd["g"]), whiten_B(geos[i]["B"], geos[i]["g"]), dirs
                            ),
                        }
                    )
                d_seed_fields.append(pd.DataFrame(field))
            ddf = pd.concat(d_seed_fields, ignore_index=True)
            write_df(cell_dir / "decoder_seeds.parquet", ddf, force=True)
            dmean = ddf.groupby("sample_id")[["H_norm_D", "K_dir_D", "K_tf_D", "K_H_D", "D_H_cosine", "D_tensor_dir_cos"]].mean()
            qdf = qdf.merge(dmean.reset_index(), on="sample_id", how="left")
        except Exception as e:  # noqa: BLE001
            recon["error"] = str(e)
            print(f"[cell] decoder failed {tag}: {e}", flush=True)

    write_df(cell_dir / "anchors.parquet", qdf, force=True)
    write_json(cell_dir / "noise_diag.json", cloud["noise_diag"], force=True)
    write_json(cell_dir / "sampling_meta.json", cloud["sampling_meta"], force=True)

    scores = _score_cell(qdf, name)
    cell = {
        "tag": tag,
        "suite": suite,
        "fixture": name,
        "sampling": sampling,
        "noise": noise,
        "n": n,
        "k": k,
        "eta": eta,
        "r_med": r_med,
        "n_anchors": int(len(anc_rows)),
        "n_oracle": int(len(oracle_rows)),
        "scores": scores,
        "recon": recon,
        "runtime_s": time.time() - t1,
        "holdout_fraction_check": len(set(train_idx).intersection(set(holdout_idx))) == 0,
    }
    write_json(done, cell, force=True)
    print(f"[cell] {tag} {scores.get('rho_D_T1')} Q_T1={scores.get('rho_Q_T1')} t={cell['runtime_s']:.1f}s", flush=True)
    return cell


def _score_cell(df: pd.DataFrame, name: str) -> dict:
    out = {}
    if "K_dir_D" in df.columns:
        out["rho_D_T1"] = rank_cal(df["K_dir_D"].values, df["K_dir_T1"].values)["rho"]
        out["rho_D_H"] = rank_cal(df["H_norm_D"].values, df["H_norm_T1"].values)["rho"]
        out["median_cos_H_D"] = float(np.nanmedian(df["D_H_cosine"])) if "D_H_cosine" in df.columns else float("nan")
        out["median_tensor_cos_D"] = float(np.nanmedian(df["D_tensor_dir_cos"])) if "D_tensor_dir_cos" in df.columns else float("nan")
    if "Q_K_dir_cross" in df.columns:
        # split-cross vs squared truth
        out["rho_Q_T1"] = rank_cal(df["Q_K_dir_cross"].values, df["K_dir_T1"].values ** 2)["rho"]
        out["rho_Q_Hcross_T1"] = rank_cal(df["Q_K_H_cross"].values, df["K_H_T1"].values ** 2)["rho"] if "Q_K_H_cross" in df.columns else float("nan")
        out["Q_split_R"] = float(np.nanmean(df["Q_split_R_BS"])) if "Q_split_R_BS" in df.columns else float("nan")
        out["rho_Q_log_r"] = rank_cal(df["Q_K_dir_cross"].values, np.log(np.clip(df["radius"].values, 1e-12, None)))["rho"]
        out["rho_Q_T1_given_r"] = partial_spearman(
            df["Q_K_dir_cross"].values, df["K_dir_T1"].values ** 2, np.log(np.clip(df["radius"].values, 1e-12, None))
        )
    if "K_dir_T2" in df.columns:
        m = df["K_dir_T2"].notna()
        if m.sum() >= 8 and "Q_K_dir_cross" in df.columns:
            out["rho_Q_T2"] = rank_cal(df.loc[m, "Q_K_dir_cross"].values, df.loc[m, "K_dir_T2"].values ** 2)["rho"]
            out["rho_Q_T3"] = rank_cal(df.loc[m, "Q_K_dir_cross"].values, df.loc[m, "K_dir_T3"].values ** 2)["rho"]
            if "K_dir_D" in df.columns:
                out["rho_D_T2"] = rank_cal(df.loc[m, "K_dir_D"].values, df.loc[m, "K_dir_T2"].values)["rho"]
    out["false_Kdir_T1_mean"] = float(np.nanmean(df["K_dir_T1"]))
    if name == "F0":
        out["F0_Q_Kdir"] = float(np.nanmean(np.abs(df["Q_K_dir_cross"]))) if "Q_K_dir_cross" in df.columns else float("nan")
        out["F0_D_Kdir"] = float(np.nanmean(np.abs(df["K_dir_D"]))) if "K_dir_D" in df.columns else float("nan")
    if name == "F1":
        out["F1_tf_T1"] = float(np.nanmean(df["K_tf_T1"] / np.clip(df["K_dir_T1"], 1e-12, None)))
        out["F1_D_tf"] = float(np.nanmean(df["K_tf_D"] / np.clip(df["K_dir_D"], 1e-12, None))) if "K_tf_D" in df.columns else float("nan")
    if name == "F2":
        out["F2_H_T1"] = float(np.nanmean(df["H_norm_T1"]))
        out["F2_D_H"] = float(np.nanmean(df["H_norm_D"])) if "H_norm_D" in df.columns else float("nan")
        out["F2_Kdir_T1"] = float(np.nanmean(df["K_dir_T1"]))
        if "K_dir_D" in df.columns:
            out["F2_D_Kdir"] = float(np.nanmean(df["K_dir_D"]))
        if "Q_K_dir_cross" in df.columns:
            out["F2_Q_Kdir"] = float(np.nanmean(np.abs(df["Q_K_dir_cross"])))
    if "K_dir_D" in df.columns:
        t = np.clip(df["K_dir_T1"].values, 1e-12, None)
        out["rel_D_Kdir"] = float(np.nanmedian(np.abs(df["K_dir_D"].values - df["K_dir_T1"].values) / t))
        out["rel_D_H"] = float(
            np.nanmedian(np.abs(df["H_norm_D"].values - df["H_norm_T1"].values) / np.clip(df["H_norm_T1"].values, 1e-12, None))
        )
    if "Q_K_dir_cross" in df.columns:
        t2 = df["K_dir_T1"].values ** 2
        out["rel_Q_Kdir_T1"] = float(np.nanmedian(np.abs(df["Q_K_dir_cross"].values - t2) / np.clip(np.abs(t2) + np.abs(df["Q_K_dir_cross"].values), 1e-12, None)))
    return out


def _suite_plan(cfg: ExpConfig) -> list[dict]:
    n = cfg.n_points()
    k = cfg.k_primary()
    smoke = cfg.smoke
    plan = []
    # Suite A
    for fx in ("F0", "F1", "F2", "F3", "F4", "F5"):
        plan.append(dict(suite="A", name=fx, sampling="S0", noise="N0", n=n, k=k, eta=0.0, do_decoder=True, do_oracle=True))
    if smoke:
        return plan[:3] + [plan[4]]  # F0,F1,F2,F4
    if cfg.factorial_smoke:
        # 128-anchor density/noise factorial: keep the matrix small and frozen.
        for fx in ("F0", "F2", "F4"):
            for s in ("S1", "S4"):
                plan.append(dict(suite="B", name=fx, sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=fx == "F4", do_oracle=s == "S1"))
            plan.append(dict(suite="C", name=fx, sampling="S0", noise="N2", n=n, k=k, eta=PRIMARY_ETA, do_decoder=fx == "F4", do_oracle=False))
        return plan
    # Suite B
    for fx in ("F0", "F2", "F4"):
        for s in ("S0", "S1", "S2", "S3", "S4", "S5"):
            if s == "S0":
                continue
            plan.append(dict(suite="B", name=fx, sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=fx == "F4", do_oracle=s in ("S1", "S2", "S3")))
    # Suite C
    for fx in ("F0", "F2", "F4"):
        for noise, eta in (
            ("N1", PRIMARY_ETA),
            ("N2", 0.05),
            ("N2", PRIMARY_ETA),
            ("N2", 0.25),
            ("N3", PRIMARY_ETA),
            ("N4", PRIMARY_ETA),
            ("N5", PRIMARY_ETA),
            ("N5n", PRIMARY_ETA),
        ):
            plan.append(dict(suite="C", name=fx, sampling="S0", noise=noise, n=n, k=k, eta=eta, do_decoder=fx == "F4" and eta == PRIMARY_ETA, do_oracle=False))
    # Suite D
    for s in ("S1", "S2", "S3"):
        for noise, eta in (("N0", 0.0), ("N2", PRIMARY_ETA), ("N4", PRIMARY_ETA)):
            plan.append(dict(suite="D", name="F4", sampling=s, noise=noise, n=n, k=k, eta=eta, do_decoder=noise == "N0", do_oracle=True))
    # Suite E
    for fx in ("F4", "F5"):
        for kk in K_GRID_SMALL_N:
            plan.append(dict(suite="E", name=fx, sampling="S0", noise="N0", n=PRIMARY_N, k=kk, eta=0.0, do_decoder=kk == PRIMARY_K, do_oracle=kk in (256, PRIMARY_K)))
    if not (cfg.bounded or cfg.skip_n86471):
        k_frac = int(round(N_COLLEAGUE / 8))
        # physical-radius match: k such that k/n ≈ 2048/16384 = 1/8, already k_frac
        plan.append(dict(suite="E", name="F4", sampling="S0", noise="N0", n=N_COLLEAGUE, k=2048, eta=0.0, do_decoder=True, do_oracle=False))
        plan.append(dict(suite="E", name="F4", sampling="S0", noise="N0", n=N_COLLEAGUE, k=k_frac, eta=0.0, do_decoder=False, do_oracle=False))
    # Suite F — neighbourhood-definition variants; dropped in the bounded fixture check
    if not (cfg.bounded or cfg.skip_suite_f):
        for s in ("S1", "S2"):
            plan.append(dict(suite="F", name="F4", sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=False, do_oracle=True))
            plan.append(dict(suite="F", name="F4", sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=False, do_oracle=True, r_fixed="primary"))
            plan.append(dict(suite="F", name="F4", sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=False, do_oracle=True, adaptive_k=True))
            plan.append(dict(suite="F", name="F4", sampling=s, noise="N0", n=n, k=k, eta=0.0, do_decoder=False, do_oracle=True, weighted_q=True))
    if cfg.suites:
        plan = [p for p in plan if p["suite"] in cfg.suites]
    return plan


def _aggregate(cells: list[dict]) -> dict:
    def mean_score(pred, key):
        vals = [c["scores"][key] for c in cells if pred(c) and key in c["scores"] and np.isfinite(c["scores"][key])]
        return float(np.mean(vals)) if vals else float("nan")

    varying = lambda c: c["suite"] == "A" and c.get("fixture") in ("F4", "F5") and c.get("noise") == "N0"
    decoder_pointwise = {
        "rho_H": mean_score(varying, "rho_D_H"),
        "rho_Kdir": mean_score(varying, "rho_D_T1"),
        "median_cosine_H": mean_score(varying, "median_cos_H_D"),
        "median_tensor_cos": mean_score(varying, "median_tensor_cos_D"),
        "rel_Kdir_F1": mean_score(lambda c: c.get("fixture") == "F1" and c["suite"] == "A", "rel_D_Kdir"),
        "rel_H_F1": mean_score(lambda c: c.get("fixture") == "F1" and c["suite"] == "A", "rel_D_H"),
        "rel_Kdir_F4F5": mean_score(varying, "rel_D_Kdir"),
    }
    quadratic_patch = {
        "rho_Kdir_T2": mean_score(varying, "rho_Q_T2"),
        "rho_Kdir_T3": mean_score(varying, "rho_Q_T3"),
        "rho_Kdir_T1": mean_score(varying, "rho_Q_T1"),
        "rel_Kdir_T1_const": mean_score(
            lambda c: c["suite"] == "A" and c.get("fixture") in ("F1", "F2", "F3"), "rel_Q_Kdir_T1"
        ),
    }
    e_cells = [c for c in cells if c["suite"] == "E" and c["fixture"] == "F4"]
    e_cells_sorted = sorted(e_cells, key=lambda c: c["k"])
    shrink = {
        "rho_Kdir_T1_smallest_k": e_cells_sorted[0]["scores"].get("rho_Q_T1", float("nan")) if e_cells_sorted else float("nan"),
        "rho_Kdir_T1_largest_k": e_cells_sorted[-1]["scores"].get("rho_Q_T1", float("nan")) if e_cells_sorted else float("nan"),
        "rho_improves": float("nan"),
    }
    if e_cells_sorted and np.isfinite(shrink["rho_Kdir_T1_smallest_k"]) and np.isfinite(shrink["rho_Kdir_T1_largest_k"]):
        shrink["rho_improves"] = float(shrink["rho_Kdir_T1_smallest_k"] - shrink["rho_Kdir_T1_largest_k"])
    # seed stability from Suite A F4 recon/fields stored in cells
    seed_rho = []
    for c in cells:
        if c["suite"] == "A" and "recon" in c and len(c.get("recon", {})) >= 2:
            seed_rho.append(np.nan)
    decoder_seeds = {"spearman_Kdir": float("nan")}
    # false curvature
    f0 = [c for c in cells if c["fixture"] == "F0" and c["sampling"] == "S0" and c["noise"] == "N0"]
    f1 = [c for c in cells if c["fixture"] == "F1"]
    f2 = [c for c in cells if c["fixture"] == "F2" and c["sampling"] == "S0" and c["noise"] == "N0"]
    false = {
        "F0_decoder_Kdir": mean_score(lambda c: c["fixture"] == "F0" and c["suite"] == "A", "F0_D_Kdir"),
        "F0_quadratic_Kdir": mean_score(lambda c: c["fixture"] == "F0" and c["suite"] == "A", "F0_Q_Kdir"),
        "F1_decoder_tf_frac": mean_score(lambda c: c["fixture"] == "F1", "F1_D_tf"),
        "F2_decoder_H_frac": float("nan"),
    }
    if f2:
        h = f2[0]["scores"].get("F2_D_H", np.nan)
        k = f2[0]["scores"].get("F2_Kdir_T1", np.nan)
        false["F2_decoder_H_frac"] = float(h / max(np.sqrt(max(k, 0.0)), 1e-8)) if np.isfinite(h) else float("nan")
    # robustness drops
    a4 = [c for c in cells if c["suite"] == "A" and c["fixture"] == "F4"]
    base = a4[0]["scores"].get("rho_Q_T1", np.nan) if a4 else np.nan
    drops_s, drops_n = [], []
    t3_rescue = False
    thick = False
    for c in cells:
        if c["suite"] == "B" and c["fixture"] == "F4":
            dlt = float(base - c["scores"].get("rho_Q_T1", np.nan)) if np.isfinite(base) else np.nan
            drops_s.append(dlt)
            if c["scores"].get("rho_Q_T3", 0) > c["scores"].get("rho_Q_T1", 0) + 0.1:
                t3_rescue = True
        if c["suite"] == "C" and c["fixture"] == "F4":
            dlt = float(base - c["scores"].get("rho_Q_T1", np.nan)) if np.isfinite(base) else np.nan
            drops_n.append(dlt)
            if c["noise"] in ("N2", "N4", "N5", "N5n") and np.isfinite(dlt) and dlt > 0.2:
                thick = True
    sampling_robust = {"max_rho_drop": float(np.nanmax(drops_s)) if drops_s else 0.0, "T3_rescues_Q": t3_rescue}
    noise_robust = {"max_rho_drop": float(np.nanmax(drops_n)) if drops_n else 0.0, "thickness_drives_Q": thick}
    # mean vs full: trigger only if *recovery* of H and of K_dir disagree materially
    d_h = decoder_pointwise["rho_H"]
    d_k = decoder_pointwise["rho_Kdir"]
    q_h = mean_score(varying, "rho_Q_Hcross_T1")
    q_k = quadratic_patch["rho_Kdir_T1"]
    mean_gap = False
    if np.isfinite(d_h) and np.isfinite(d_k):
        mean_gap = mean_gap or abs(d_h - d_k) > 0.30
    if np.isfinite(q_h) and np.isfinite(q_k):
        mean_gap = mean_gap or abs(q_h - q_k) > 0.30
    f2_d_h = mean_score(lambda c: c.get("fixture") == "F2" and c["suite"] == "A", "F2_D_H")
    f2_d_k = mean_score(lambda c: c.get("fixture") == "F2" and c["suite"] == "A", "F2_D_Kdir")
    if np.isfinite(f2_d_h) and np.isfinite(f2_d_k):
        # decoder invents mean on the minimal Clifford, or misses traceless energy
        mean_gap = mean_gap or (f2_d_h > 0.05 and f2_d_k > 1e-4) or (f2_d_h < 1e-3 and f2_d_k < 1e-4)
    mean_vs_full = {
        "divergence": bool(mean_gap),
        "rho_D_H": d_h,
        "rho_D_Kdir": d_k,
        "rho_Q_H": q_h,
        "rho_Q_Kdir": q_k,
        "F2_D_H": f2_d_h,
        "F2_D_Kdir": f2_d_k,
    }
    return {
        "decoder_pointwise": decoder_pointwise,
        "quadratic_patch": quadratic_patch,
        "quadratic_shrink": shrink,
        "decoder_seeds": decoder_seeds,
        "false_curvature": false,
        "sampling_robust": sampling_robust,
        "noise_robust": noise_robust,
        "mean_vs_full": mean_vs_full,
    }


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    reuse = _hash_estimators(root)
    write_json(out / "reuse_manifest.json", {"frozen_hashes": reuse, "preserved": True}, force=True)
    write_json(out / "fixture_definitions.json", fixture_definitions(), force=True)

    if cfg.stage in ("all", "unit", "tests"):
        print("[audit] unit tests", flush=True)
        tests = run_unit_tests()
        write_json(out / "unit_tests.json", tests, force=True)
        if cfg.stage == "unit":
            return tests
    else:
        p = out / "unit_tests.json"
        tests = json.loads(p.read_text()) if p.exists() else {"all_passed": False, "n_passed": 0, "n_tests": 0}

    Q = ambient_rotation()
    z_chk = 0.12 * np.ones(D_LAT) / np.sqrt(D_LAT)
    truth_val = {}
    for fx in ("F0", "F1", "F2", "F3", "F4", "F5"):
        try:
            truth_val[fx] = validate_point(fx, z_chk, Q)
            if fx in ("F0", "F1", "F2", "F3"):
                inv = invariance_checks(fx, z_chk, Q)
                truth_val[fx]["invariance"] = {k: inv[k] for k in inv if k != "ok"}
                truth_val[fx]["invariance_ok"] = inv["ok"]
        except Exception as e:  # noqa: BLE001
            truth_val[fx] = {"checks": {"ok": False}, "error": str(e), "trace": traceback.format_exc()}
    write_json(out / "truth_validation.json", truth_val, force=True)
    truth_ok = all(truth_val[fx].get("checks", {}).get("ok") for fx in ("F0", "F1", "F2", "F3"))
    tests_ok = bool(tests.get("all_passed"))

    write_json(
        out / "CONFIG.json",
        {
            "smoke": cfg.smoke,
            "factorial_smoke": cfg.factorial_smoke,
            "n_points": cfg.n_points(),
            "n_anchors": cfg.n_anc(),
            "k": cfg.k_primary(),
            "epochs": cfg.epochs(),
            "seeds": list(cfg.seeds()),
            "device": str(_device(cfg)),
        },
        force=True,
    )

    if cfg.stage in ("unit", "truth"):
        return {"tests": tests, "truth": truth_val}

    device = _device(cfg)
    if cfg.stage == "finalize":
        cells = []
        for p in sorted((out / "cells").glob("*/cell.json")):
            cells.append(json.loads(p.read_text()))
        print(f"[audit] finalize n_cells={len(cells)}", flush=True)
        plan = []
    else:
        plan = _suite_plan(cfg)
    # 64-anchor smoke is cfg.smoke with n_anc=64 via n_anc()
    if cfg.stage != "finalize":
        cells = []
    for i, spec in enumerate(plan):
        print(f"[audit] cell {i+1}/{len(plan)} {spec}", flush=True)
        r_fixed = None
        if spec.get("r_fixed") == "primary":
            # use median radius from a cheap probe: kNN radius of k/n * typical sphere chord ~ 0.2
            r_fixed = 0.20
        try:
            cell = run_cell(
                cfg,
                out,
                suite=spec["suite"],
                name=spec["name"],
                sampling=spec["sampling"],
                noise=spec["noise"],
                n=spec["n"],
                k=spec["k"],
                eta=spec["eta"],
                Q=Q,
                device=device,
                do_decoder=spec["do_decoder"],
                do_oracle=spec["do_oracle"],
                r_fixed=r_fixed,
                adaptive_k=bool(spec.get("adaptive_k")),
                weighted_q=bool(spec.get("weighted_q")),
            )
            cells.append(cell)
        except Exception as e:  # noqa: BLE001
            err = {"tag": str(spec), "error": str(e), "trace": traceback.format_exc()}
            write_json(out / "cells" / f"FAILED_{i}.json", err, force=True)
            print(f"[audit] FAILED {spec}: {e}", flush=True)
            cells.append({"suite": spec["suite"], "fixture": spec["name"], "scores": {}, "error": str(e), "sampling": spec["sampling"], "noise": spec["noise"], "k": spec["k"], "n": spec["n"]})

    write_json(out / "cells_index.json", cells, force=True)

    # Concatenate anchor tables
    frames = []
    for p in sorted((out / "cells").glob("*/anchors.parquet")):
        df = pd.read_parquet(p)
        frames.append(df)
    anchors = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(anchors):
        write_df(out / "tables" / "per_anchor.parquet", anchors, force=True)

    agg = _aggregate(cells)
    # decoder seed stability from parquet
    seed_files = list((out / "cells").glob("*/decoder_seeds.parquet"))
    seed_rho = []
    if seed_files:
        ddf = pd.concat([pd.read_parquet(p) for p in seed_files], ignore_index=True)
        write_df(out / "tables" / "decoder_seed_stability.parquet", ddf, force=True)
        if "seed" in ddf.columns and ddf["seed"].nunique() >= 2:
            piv = ddf.pivot_table(index="sample_id", columns="seed", values="K_dir_D")
            cols = list(piv.columns)
            if len(cols) >= 2:
                r = spearmanr(piv[cols[0]], piv[cols[1]], nan_policy="omit").correlation
                seed_rho.append(float(r) if r is not None else float("nan"))
            if len(cols) >= 3:
                r = spearmanr(piv[cols[0]], piv[cols[2]], nan_policy="omit").correlation
                seed_rho.append(float(r) if r is not None else float("nan"))
    agg["decoder_seeds"]["spearman_Kdir"] = float(np.nanmean(seed_rho)) if seed_rho else float("nan")

    bundle = dict(agg)
    bundle["tests_ok"] = tests_ok
    bundle["truth_ok"] = truth_ok
    decision = decide(bundle)
    write_json(out / "decision.json", decision, force=True)

    # robustness table
    rob = []
    for c in cells:
        if c.get("suite") in ("B", "C", "D"):
            rob.append(
                {
                    "kind": "sampling" if c["suite"] == "B" else ("noise" if c["suite"] == "C" else "density_noise"),
                    "condition": f"{c.get('fixture')}_{c.get('sampling')}_{c.get('noise')}",
                    "rho_Kdir": c.get("scores", {}).get("rho_Q_T1", np.nan),
                    "rho_D": c.get("scores", {}).get("rho_D_T1", np.nan),
                }
            )
    robust = pd.DataFrame(rob)
    if len(robust):
        write_df(out / "tables" / "density_noise_robustness.parquet", robust, force=True)
    scale_rows = []
    for c in cells:
        if c.get("suite") == "E":
            scale_rows.append(
                {
                    "fixture": c.get("fixture"),
                    "n": c.get("n"),
                    "k": c.get("k"),
                    "k_over_n": (c.get("k") or np.nan) / max(c.get("n") or 1, 1),
                    "radius_median": c.get("r_med", np.nan),
                    "rho_Q_T1": c.get("scores", {}).get("rho_Q_T1"),
                    "rho_Q_T2": c.get("scores", {}).get("rho_Q_T2"),
                    "rho_Q_T3": c.get("scores", {}).get("rho_Q_T3"),
                    "rho_D_T1": c.get("scores", {}).get("rho_D_T1"),
                }
            )
    scale = pd.DataFrame(scale_rows)
    if len(scale):
        write_df(out / "tables" / "scale_convergence.parquet", scale, force=True)
    if len(anchors):
        keep = [
            c
            for c in (
                "suite",
                "fixture",
                "sampling",
                "noise",
                "n",
                "k",
                "radius",
                "K_dir_T1",
                "K_dir_T2",
                "K_dir_T3",
                "H_norm_T1",
                "K_H_T1",
                "K_tf_T1",
                "K_dir_D",
                "H_norm_D",
                "K_tf_D",
                "Q_K_dir_cross",
                "Q_K_H_cross",
                "Q_K_tf_cross",
                "D_H_cosine",
                "D_tensor_dir_cos",
                "Q_tensor_dir_cos",
            )
            if c in anchors.columns
        ]
        write_df(out / "tables" / "pointwise_versus_patch.parquet", anchors[keep], force=True)
    stab = pd.DataFrame(
        [
            {
                "q_split_R": c.get("scores", {}).get("Q_split_R", np.nan),
                "d_seed_rho": agg["decoder_seeds"]["spearman_Kdir"],
                "fixture": c.get("fixture"),
            }
            for c in cells
            if c.get("suite") == "A"
        ]
    )
    if len(stab):
        write_df(out / "tables" / "quadratic_split_reliability.parquet", stab, force=True)

    figs = make_figures(
        out,
        {
            "anchors": anchors if len(anchors) else None,
            "scale": scale if len(scale) else None,
            "robust": robust if len(robust) else None,
            "stability": stab if len(stab) else None,
        },
    )

    runtime = time.time() - t0
    bounded = bool(cfg.bounded or cfg.stage == "finalize")
    skipped = []
    if bounded or cfg.skip_n86471:
        skipped.append("n=86471")
    if bounded or cfg.skip_suite_f:
        skipped.append("suite_F")
    have_e5_k2048 = any(
        c.get("suite") == "E" and c.get("fixture") == "F5" and int(c.get("k") or 0) == 2048 for c in cells
    )
    if not have_e5_k2048:
        skipped.append("E_F5_k2048")
    summary = {
        "decision": decision["label"],
        "reason": decision["reason"],
        "tests_ok": tests_ok,
        "truth_ok": truth_ok,
        "bounded": bounded,
        "skipped": skipped,
        "n_cells": len(cells),
        "n_cells_ok": sum(1 for c in cells if "error" not in c),
        "decoder_pointwise_H": agg["decoder_pointwise"],
        "decoder_pointwise_Kdir": agg["decoder_pointwise"],
        "decoder_seed_stability": agg["decoder_seeds"],
        "quadratic_shrink": agg["quadratic_shrink"],
        "quadratic_patch": agg["quadratic_patch"],
        "false_F0": agg["false_curvature"],
        "runtime_s": runtime,
        "peak_rss_mb": peak_rss_mb(),
        "output_dir": str(out),
        "figures": figs,
        "device": str(device),
        "smoke": cfg.smoke,
    }
    write_json(out / "summary.json", summary, force=True)
    write_json(out / "sampling_manifest.json", {"regimes": ["S0", "S1", "S2", "S3", "S4", "S5"], "s1_rho_max": 0.05}, force=True)
    write_json(
        out / "noise_manifest.json",
        {"regimes": ["N0", "N1", "N2", "N3", "N4", "N5", "N5n"], "eta_primary": PRIMARY_ETA, "eta_levels": list(ETA_LEVELS)},
        force=True,
    )
    write_reports(out, summary=summary, decision=decision, fixtures=fixture_definitions(), tests=tests)
    n_ok = sum(1 for c in cells if "error" not in c)
    if (
        tests_ok
        and truth_ok
        and n_ok == len(cells)
        and n_ok > 0
        and not cfg.smoke
        and not cfg.factorial_smoke
        and (bounded or n_ok == len(_suite_plan(cfg)))
    ):
        write_json(
            out / "COMPLETE.json",
            {
                "ok": True,
                "bounded": bounded,
                "skipped": skipped,
                "decision": decision["label"],
                "runtime_s": runtime,
                "n_cells": len(cells),
            },
            force=True,
        )
    elif cfg.smoke:
        write_json(out / "smoke_complete.json", {"ok": tests_ok and truth_ok, "runtime_s": runtime}, force=True)
    elif cfg.factorial_smoke:
        write_json(out / "factorial_smoke_complete.json", {"ok": tests_ok and truth_ok, "runtime_s": runtime}, force=True)
    print(f"[audit] done label={decision['label']} t={runtime:.1f}s tests={tests.get('n_passed')}/{tests.get('n_tests')}", flush=True)
    return summary
