"""Follow-up: raw vs radius-conditional association + probe/subspace alignment."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .confirmatory_object_curvature import (
    _fit_neighborhood,
    decompose_BS,
    select_anchors,
    unpack_BS_symmetric,
)
from .curvature_probe_screen import (
    EXPECTED_HASH,
    PRIMARY_CURV,
    LOCAL_DIM,
    ScreenConfig,
    bootstrap_spearman,
    load_frozen_curvature,
    load_labels_for_selection,
    partial_spearman,
    spearman_dict,
)
from .data import load_prepare
from .paths import platonic_root, resolve_path

SCALES = (512, 1024, 2048)
PROBE_ALPHA = 100.0
EPS = 1e-12


@dataclass
class AlignmentConfig:
    output_dir: str = "outputs/geometry/physics_curvature_probe_alignment"
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    screen_dir: str = "outputs/geometry/physics_curvature_probe_screen"
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    expected_hash: str = EXPECTED_HASH
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    probe_alpha: float = PROBE_ALPHA
    n_folds: int = 5
    n_bootstrap: int = 1000
    n_permute: int = 500
    seed: int = 0
    force: bool = False
    rematerialize_tol: float = 0.15  # relative |B°|_F match vs frozen aggregate

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def fit_ridge_direction(X_tr, y_tr, *, alpha: float) -> tuple[np.ndarray, float]:
    """Return ambient probe direction w and train R² proxy (always fit-only)."""
    m = np.isfinite(y_tr)
    if m.sum() < 8:
        return np.zeros(X_tr.shape[1]), float("nan")
    xs = StandardScaler().fit(X_tr[m])
    ys = StandardScaler().fit(y_tr[m].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(X_tr[m]), ys.transform(y_tr[m].reshape(-1, 1)).ravel())
    # ambient direction: coef on standardized X → divide by feature scales
    w = (model.coef_ / np.maximum(xs.scale_, EPS)).astype(np.float64)
    return w, float(np.linalg.norm(w))


def eval_ridge_r2(X_tr, y_tr, X_te, y_te, *, alpha: float) -> float:
    m_tr = np.isfinite(y_tr)
    m_te = np.isfinite(y_te)
    if m_tr.sum() < 8 or m_te.sum() < 4:
        return float("nan")
    xs = StandardScaler().fit(X_tr[m_tr])
    ys = StandardScaler().fit(y_tr[m_tr].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(X_tr[m_tr]), ys.transform(y_tr[m_tr].reshape(-1, 1)).ravel())
    pred = ys.inverse_transform(model.predict(xs.transform(X_te[m_te])).reshape(-1, 1)).ravel()
    return float(r2_score(y_te[m_te], pred))


def traceless_B0(BS_flat: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    B = unpack_BS_symmetric(BS_flat, d)
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    B0 = B.copy()
    for a in range(d):
        B0[:, a, a] = B[:, a, a] - H
    return B0, H


def B0_flat_for_svd(B0: np.ndarray, d: int) -> np.ndarray:
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B0[:, a, a] if a == b else (np.sqrt(2.0) * B0[:, a, b]))
    return np.stack(cols, axis=1)


def alignment_from_w(
    w: np.ndarray, x0: np.ndarray, J: np.ndarray, B0: np.ndarray, rho: float
) -> dict[str, float]:
    wn = float(np.linalg.norm(w))
    if wn < EPS:
        return {k: float("nan") for k in ("A_T", "A_N", "A_B", "C_w", "w_norm")}
    w_hat = w / wn
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    # P_T = J J^T, P_R = x0 x0^T
    w_T = J @ (J.T @ w_hat)
    w_R = x0u * float(np.dot(x0u, w_hat))
    w_N = w_hat - w_T - w_R
    A_T = float(np.dot(w_T, w_T))
    A_N = float(np.dot(w_N, w_N))
    nN = float(np.linalg.norm(w_N))
    d = J.shape[1]
    Bflat = B0_flat_for_svd(B0, d)
    # left singular projector of full curvature span
    U, s, _ = np.linalg.svd(Bflat, full_matrices=False)
    keep = s > (1e-8 * (s[0] if len(s) else 1.0))
    if not np.any(keep) or nN < EPS:
        A_B = float("nan")
        C_w = float("nan")
    else:
        Ub = U[:, keep]
        proj = Ub @ (Ub.T @ w_N)
        A_B = float(np.dot(proj, proj) / (nN**2 + EPS))
        # probe-visible curvature
        wN_hat = w_N / (nN + EPS)
        acc = 0.0
        for a in range(d):
            for b in range(a, d):
                acc += float(np.dot(wN_hat, B0[:, a, b])) ** 2
        C_w = float(rho * np.sqrt(acc))
    return {
        "A_T": A_T,
        "A_N": A_N,
        "A_B": A_B,
        "C_w": C_w,
        "w_norm": wn,
        "normal_mass": A_N,
    }


def permute_pvalue(real: float, nulls: np.ndarray, *, alternative: str = "two-sided") -> tuple[float, int]:
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[np.isfinite(nulls)]
    B = len(nulls)
    if B == 0 or not np.isfinite(real):
        return float("nan"), 0
    if alternative == "greater":
        return float((1 + np.sum(nulls >= real)) / (B + 1)), B
    if alternative == "less":
        return float((1 + np.sum(nulls <= real)) / (B + 1)), B
    # two-sided on |ρ|
    return float((1 + np.sum(np.abs(nulls) >= abs(real))) / (B + 1)), B


def residual_variance_after_radius(c: np.ndarray, log_r: np.ndarray) -> dict:
    m = np.isfinite(c) & np.isfinite(log_r)
    if m.sum() < 12:
        return {"residual_var_frac": float("nan"), "vif": float("nan"), "cond": float("nan")}
    x = rankdata(log_r[m]).astype(np.float64)
    y = rankdata(c[m]).astype(np.float64)
    A = np.column_stack([np.ones(m.sum()), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    var_y = float(np.var(y))
    var_r = float(np.var(resid))
    # VIF for C° ~ radius via R^2 of rank regression
    r2 = 1.0 - var_r / max(var_y, EPS)
    vif = float(1.0 / max(1.0 - r2, EPS))
    # condition of [1, log_r, C]
    M = np.column_stack([np.ones(m.sum()), log_r[m], c[m]])
    s = np.linalg.svd(M, compute_uv=False)
    cond = float(s[0] / max(s[-1], EPS))
    return {"residual_var_frac": float(var_r / max(var_y, EPS)), "vif_C_vs_radius": vif, "cond_1_logr_C": cond, "r2_C_on_radius_rank": float(r2)}


def within_decile_correlations(c: np.ndarray, y: np.ndarray, log_r: np.ndarray) -> list[dict]:
    m = np.isfinite(c) & np.isfinite(y) & np.isfinite(log_r)
    c, y, log_r = c[m], y[m], log_r[m]
    if len(c) < 30:
        return []
    # deciles of radius
    qs = np.quantile(log_r, np.linspace(0, 1, 11))
    rows = []
    for d in range(10):
        lo, hi = qs[d], qs[d + 1]
        if d < 9:
            sel = (log_r >= lo) & (log_r < hi)
        else:
            sel = (log_r >= lo) & (log_r <= hi)
        st = spearman_dict(c[sel], y[sel])
        rows.append({"decile": d, "n": st["n"], "rho": st["rho"], "pvalue": st["pvalue"], "log_r_lo": float(lo), "log_r_hi": float(hi)})
    return rows


def conditional_block(df_k: pd.DataFrame, cfg: AlignmentConfig, role: str) -> dict:
    g = df_k[df_k["probe_ok"] & df_k["valid"]].copy()
    C = g[PRIMARY_CURV].to_numpy(float)
    K = g["B_traceless_fro"].to_numpy(float)
    logK = np.log(np.maximum(K, EPS))
    R2 = g["probe_r2"].to_numpy(float)
    log_r = np.log(np.maximum(g["knn_radius"].to_numpy(float), EPS))
    labvar = g["local_label_variance"].to_numpy(float)
    recon = g["reconstruction_error"].to_numpy(float)

    def corr_pack(x, y, seed):
        raw = spearman_dict(x, y)
        boot = bootstrap_spearman(x, y, n_boot=cfg.n_bootstrap, seed=seed)
        # permutation null
        rng = np.random.default_rng(seed + 7)
        nulls = []
        idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
        for _ in range(cfg.n_permute):
            yp = y.copy()
            yp[idx] = y[idx][rng.permutation(len(idx))]
            nulls.append(spearman_dict(x, yp)["rho"])
        p_perm, B = permute_pvalue(raw["rho"], np.asarray(nulls))
        return {**raw, "ci95": boot["ci95"], "p_perm": p_perm, "B_perm": B}

    out = {
        "role": role,
        "scale_k": int(g["scale_k"].iloc[0]) if len(g) else -1,
        "n": int(len(g)),
        "corr_C_R2": corr_pack(C, R2, cfg.seed + 11),
        "corr_C_log_radius": corr_pack(C, log_r, cfg.seed + 13),
        "corr_R2_log_radius": corr_pack(R2, log_r, cfg.seed + 17),
        "corr_K_R2": corr_pack(K, R2, cfg.seed + 19),
        "corr_logK_R2": corr_pack(logK, R2, cfg.seed + 23),
        "partial_radius_only": partial_spearman(C, R2, log_r.reshape(-1, 1)),
        "partial_radius_labvar": partial_spearman(C, R2, np.column_stack([log_r, labvar])),
        "partial_full": partial_spearman(C, R2, np.column_stack([log_r, labvar, recon])),
        "partial_K_radius": partial_spearman(K, R2, log_r.reshape(-1, 1)),
        "partial_logK_radius": partial_spearman(logK, R2, log_r.reshape(-1, 1)),
        "residual_diagnostics": residual_variance_after_radius(C, log_r),
        "within_decile": within_decile_correlations(C, R2, log_r),
        "within_decile_K": within_decile_correlations(K, R2, log_r),
    }
    # bootstrap CIs for partials
    out["boot_partial_radius"] = bootstrap_spearman(
        C, R2, n_boot=cfg.n_bootstrap, seed=cfg.seed + 29, partial_Z=log_r.reshape(-1, 1)
    )
    out["boot_partial_full"] = bootstrap_spearman(
        C, R2, n_boot=cfg.n_bootstrap, seed=cfg.seed + 31, partial_Z=np.column_stack([log_r, labvar, recon])
    )
    return out


def nested_cv_incremental(df_k: pd.DataFrame, feature_sets: dict[str, list[str]], *, seed: int, n_folds: int = 5) -> list[dict]:
    """Rank-regression CV: predict ranked probe_r2 from ranked features; report held-out R²."""
    g = df_k[df_k["probe_ok"] & df_k["valid"]].copy()
    y = g["probe_r2"].to_numpy(float)
    m = np.isfinite(y)
    for cols in feature_sets.values():
        for c in cols:
            m &= np.isfinite(g[c].to_numpy(float))
    g = g.loc[m]
    y = g["probe_r2"].to_numpy(float)
    n = len(g)
    if n < 40:
        return [{"model": k, "cv_r2": float("nan"), "n": n} for k in feature_sets]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    rows = []
    for name, cols in feature_sets.items():
        X = g[cols].to_numpy(float)
        preds = np.full(n, np.nan)
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
            # rank within train, apply to test via train ranks approx: rank all then mask is biased;
            # use train-only rank mapping via scipy on concatenated with test held as nan — simpler: rank globally is leak.
            # Proper: rank-transform train; for test use mid-ranks via searchsorted on train values.
            y_tr = y[tr]
            X_tr = X[tr]
            y_te = y[te]
            X_te = X[te]
            # rank y train
            y_tr_r = rankdata(y_tr).astype(np.float64)
            # for each feature, map test to train empirical rank
            X_tr_r = np.column_stack([rankdata(X_tr[:, j]) for j in range(X_tr.shape[1])])
            X_te_r = np.zeros_like(X_te)
            for j in range(X.shape[1]):
                order = np.argsort(X_tr[:, j])
                sorted_x = X_tr[order, j]
                # fractional rank
                ranks = np.arange(1, len(sorted_x) + 1, dtype=np.float64)
                X_te_r[:, j] = np.interp(X_te[:, j], sorted_x, ranks, left=ranks[0], right=ranks[-1])
            y_te_r = np.interp(y_te, np.sort(y_tr), np.arange(1, len(y_tr) + 1, dtype=np.float64), left=1.0, right=float(len(y_tr)))
            A = np.column_stack([np.ones(len(tr)), X_tr_r])
            beta, *_ = np.linalg.lstsq(A, y_tr_r, rcond=None)
            Ate = np.column_stack([np.ones(len(te)), X_te_r])
            preds[te] = Ate @ beta
            # store target ranks for scoring
            if fi == 0:
                pass
        # score: correlate predicted ranks with true ranks on all (approx)
        y_r_all = rankdata(y).astype(np.float64)
        ss_res = float(np.sum((y_r_all - preds) ** 2))
        ss_tot = float(np.sum((y_r_all - y_r_all.mean()) ** 2))
        cv_r2 = 1.0 - ss_res / max(ss_tot, EPS)
        rows.append({"model": name, "cv_r2": float(cv_r2), "n": int(n), "features": ",".join(cols)})
    return rows


def rematerialize_and_align(root: Path, cfg: AlignmentConfig, curv: pd.DataFrame) -> pd.DataFrame:
    """Rematerialize frozen J,B° and compute fold-wise probe alignment features."""
    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    scfg_lab = ScreenConfig(
        labels_path=cfg.labels_path,
        prepare_dir=cfg.prepare_dir,
        expected_hash=cfg.expected_hash,
    )
    y = load_labels_for_selection(root, scfg_lab, data["sample_ids"])
    train = np.asarray(data["train_local"])
    anchors = select_anchors(data, 384)
    # map sample_id -> ai
    sid_to_ai = {int(data["sample_ids"][a]): i for i, a in enumerate(anchors)}
    k_max = max(cfg.scales)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train])
    dists, inds = nn.kneighbors(X[anchors])

    # join frozen scalars
    rows = []
    mismatch = 0
    for _, crow in curv.iterrows():
        sid = int(crow["sample_id"])
        k = int(crow["scale_k"])
        if sid not in sid_to_ai:
            continue
        ai = sid_to_ai[sid]
        a_local = int(anchors[ai])
        rho = float(dists[ai, k - 1])
        neigh = train[inds[ai, :k]]
        neigh = neigh[neigh != a_local]
        seed_fit = cfg.seed + 17 * ai + k
        chart, chart_RS, info, Uloc, glob, reason = _fit_neighborhood(
            X, neigh, LOCAL_DIM, seed=seed_fit
        )
        if chart is None:
            rows.append(
                {
                    "sample_id": sid,
                    "scale_k": k,
                    "align_ok": False,
                    "failure_reason": reason or "rematerialize_failed",
                }
            )
            continue
        dec = decompose_BS(chart.BS_flat, chart.J.shape[1])
        # verify against frozen aggregate
        frozen_B0 = float(crow["B_traceless_fro"])
        rel = abs(dec["B_traceless_fro"] - frozen_B0) / max(frozen_B0, EPS)
        if rel > cfg.rematerialize_tol:
            mismatch += 1
        B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
        x0, J = chart.x0, chart.J

        # K-fold probe alignment on neighborhood (global indices in glob order)
        n = len(glob)
        rng = np.random.default_rng(cfg.seed + 101 * ai + k)
        order = np.arange(n)
        rng.shuffle(order)
        folds = np.array_split(order, cfg.n_folds)
        fold_stats = []
        w_list = []
        for fi, te_loc in enumerate(folds):
            tr_loc = np.concatenate([folds[j] for j in range(cfg.n_folds) if j != fi])
            if len(tr_loc) < 16 or len(te_loc) < 4:
                continue
            X_tr, y_tr = X[glob[tr_loc]], y[glob[tr_loc]]
            X_te, y_te = X[glob[te_loc]], y[glob[te_loc]]
            w, _ = fit_ridge_direction(X_tr, y_tr, alpha=cfg.probe_alpha)
            r2 = eval_ridge_r2(X_tr, y_tr, X_te, y_te, alpha=cfg.probe_alpha)
            al = alignment_from_w(w, x0, J, B0, rho)
            al["fold"] = fi
            al["fold_r2"] = r2
            fold_stats.append(al)
            if np.isfinite(al["w_norm"]) and al["w_norm"] > 0:
                w_list.append(w / al["w_norm"])
        if not fold_stats:
            rows.append(
                {
                    "sample_id": sid,
                    "scale_k": k,
                    "align_ok": False,
                    "failure_reason": "fold_failed",
                    "rematerialize_rel_err": rel,
                }
            )
            continue
        # direction stability: mean pairwise |cos|
        stab = float("nan")
        if len(w_list) >= 2:
            sims = []
            for i in range(len(w_list)):
                for j in range(i + 1, len(w_list)):
                    sims.append(abs(float(np.dot(w_list[i], w_list[j]))))
            stab = float(np.mean(sims))
        def mean_key(key):
            vals = [f[key] for f in fold_stats if np.isfinite(f.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        rows.append(
            {
                "sample_id": sid,
                "scale_k": k,
                "local_index": a_local,
                "align_ok": True,
                "failure_reason": "",
                "rematerialize_rel_err": float(rel),
                "rematerialize_B0": float(dec["B_traceless_fro"]),
                "frozen_B0": frozen_B0,
                "A_T": mean_key("A_T"),
                "A_N": mean_key("A_N"),
                "A_B": mean_key("A_B"),
                "C_w": mean_key("C_w"),
                "fold_r2_mean": mean_key("fold_r2"),
                "probe_dir_stability": stab,
                "n_folds_ok": int(len(fold_stats)),
                "ridge_A": float(info.get("ridge_A", np.nan)),
                "ridge_BS": float(info.get("ridge_BS", np.nan)),
            }
        )
        if (len(rows) % 64) == 0:
            print(f"[align] rematerialized {len(rows)} rows rss-ok", flush=True)

    print(f"[align] rematerialize mismatches (>{cfg.rematerialize_tol}): {mismatch}", flush=True)
    return pd.DataFrame(rows)


def association_with_align(df: pd.DataFrame, cfg: AlignmentConfig) -> pd.DataFrame:
    rows = []
    for k in cfg.scales:
        g = df[(df.scale_k == k) & df.probe_ok & df.valid & df.align_ok]
        role = "primary" if k == 2048 else ("comparison" if k == 1024 else "exploratory")
        for feat in ["A_T", "A_N", "A_B", "C_w", PRIMARY_CURV, "B_traceless_fro", "log_K"]:
            if feat == "log_K":
                x = np.log(np.maximum(g["B_traceless_fro"].to_numpy(float), EPS))
            else:
                x = g[feat].to_numpy(float)
            y = g["probe_r2"].to_numpy(float)
            st = spearman_dict(x, y)
            boot = bootstrap_spearman(x, y, n_boot=cfg.n_bootstrap, seed=cfg.seed + k + hash(feat) % 1000)
            rng = np.random.default_rng(cfg.seed + 100 * k + (hash(feat) % 997))
            nulls = []
            idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
            for _ in range(cfg.n_permute):
                yp = y.copy()
                yp[idx] = y[idx][rng.permutation(len(idx))]
                nulls.append(spearman_dict(x, yp)["rho"])
            p_perm, B = permute_pvalue(st["rho"], np.asarray(nulls))
            # partial controlling radius
            log_r = np.log(np.maximum(g["knn_radius"].to_numpy(float), EPS))
            part = partial_spearman(x, y, log_r.reshape(-1, 1))
            rows.append(
                {
                    "scale_k": k,
                    "role": role,
                    "feature": feat,
                    "rho": st["rho"],
                    "p_spearman": st["pvalue"],
                    "p_perm": p_perm,
                    "B_perm": B,
                    "ci95_lo": boot["ci95"][0],
                    "ci95_hi": boot["ci95"][1],
                    "rho_partial_radius": part["rho"],
                    "p_partial_radius": part["pvalue"],
                    "n": st["n"],
                }
            )
    return pd.DataFrame(rows)


def choose_labels(cond_by_k: dict, align_assoc: pd.DataFrame, cv_rows: pd.DataFrame) -> tuple[list[str], str]:
    labels = []
    notes = []
    c2048 = cond_by_k[2048]
    c1024 = cond_by_k[1024]
    raw_neg = c2048["corr_C_R2"]["rho"] < -0.2 and c2048["corr_C_R2"]["p_perm"] <= 0.05
    rad_med = (
        abs(c2048["corr_C_log_radius"]["rho"]) > 0.5
        and c2048["partial_radius_only"]["rho"] > c2048["corr_C_R2"]["rho"] + 0.15
    )
    if raw_neg and rad_med:
        labels.append("total_negative_radius_mediated")
        notes.append("Raw C°–R² negative at k=2048 with strong C°–radius coupling; radius-only partial flips/attenuates.")

    # residual magnitude: unscaled K or partial after radius
    k_part = c2048["partial_K_radius"]["rho"]
    c_part = c2048["partial_radius_only"]["rho"]
    if (abs(k_part) >= 0.1 and c2048["partial_K_radius"]["pvalue"] <= 0.05) or (
        abs(c_part) >= 0.1 and c2048["partial_radius_only"]["pvalue"] <= 0.05
    ):
        labels.append("residual_curvature_magnitude_association")
        notes.append("Unscaled/partial curvature magnitude remains associated with R² after radius control.")

    # alignment
    a2048 = align_assoc[(align_assoc.scale_k == 2048)]
    def feat_sig(name):
        r = a2048[a2048.feature == name]
        if len(r) == 0:
            return False
        row = r.iloc[0]
        return abs(row["rho_partial_radius"]) >= 0.1 and row["p_partial_radius"] <= 0.05

    if feat_sig("A_T") and not feat_sig("A_B"):
        labels.append("tangent_alignment_explains_association")
        notes.append("Partial association dominated by tangent probe alignment A_T.")
    if feat_sig("A_B"):
        labels.append("curvature_alignment_explains_association")
        notes.append("Probe–curvature subspace alignment A_B associated with R² after radius control.")
    if feat_sig("C_w"):
        labels.append("probe_visible_curvature_association")
        notes.append("Probe-visible curvature C_w associated with R².")

    # CV incremental
    cv2048 = cv_rows[cv_rows["scale_k"] == 2048] if "scale_k" in cv_rows.columns else cv_rows
    if len(cv2048):
        base = float(cv2048.loc[cv2048.model == "controls", "cv_r2"].iloc[0]) if (cv2048.model == "controls").any() else float("nan")
        for m in ["controls+curvature_magnitude", "controls+tangent_normal", "controls+curvature_subspace", "controls+probe_visible"]:
            if (cv2048.model == m).any():
                d = float(cv2048.loc[cv2048.model == m, "cv_r2"].iloc[0]) - base
                notes.append(f"CV ΔR²({m})={d:.4f}")

    # mechanism difference 1024 vs 2048
    if abs(c1024["corr_C_R2"]["rho"] - c2048["corr_C_R2"]["rho"]) < 0.1:
        if np.sign(c1024["partial_radius_only"]["rho"]) != np.sign(c2048["partial_radius_only"]["rho"]) or abs(
            c1024["partial_radius_only"]["rho"] - c2048["partial_radius_only"]["rho"]
        ) > 0.1:
            notes.append(
                "Similar raw ρ at k=1024 and k=2048 conceal different conditional mechanisms "
                f"(partial_radius ρ: {c1024['partial_radius_only']['rho']:.3f} vs {c2048['partial_radius_only']['rho']:.3f})."
            )

    if not labels:
        # suppression unresolved if raw strong but partial/align unclear
        if abs(c2048["corr_C_R2"]["rho"]) > 0.3 and abs(c2048["partial_full"]["rho"]) < 0.1 and not feat_sig("A_B"):
            labels.append("suppression_unresolved")
            notes.append("Strong raw association with weak residual/alignment signal; mediation not fully resolved.")
        else:
            labels.append("no_robust_association")
            notes.append("No robust residual or alignment association after controls.")
    return labels, " ".join(notes)


def make_plots(joined: pd.DataFrame, cond_by_k: dict, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # raw vs partial by scale
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ks = list(SCALES)
    raw = [cond_by_k[k]["corr_C_R2"]["rho"] for k in ks]
    part = [cond_by_k[k]["partial_radius_only"]["rho"] for k in ks]
    partf = [cond_by_k[k]["partial_full"]["rho"] for k in ks]
    ax.plot(ks, raw, "o-", label="raw C°–R²")
    ax.plot(ks, part, "s--", label="partial (radius)")
    ax.plot(ks, partf, "^:", label="partial (full)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Raw vs conditional association")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_vs_partial_by_scale.png", dpi=140)
    plt.close(fig)

    for k in SCALES:
        g = joined[(joined.scale_k == k) & joined.probe_ok & joined.valid & joined.align_ok]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        axes[0].scatter(g[PRIMARY_CURV], g["probe_r2"], s=10, alpha=0.5)
        axes[0].set_xlabel("C°")
        axes[0].set_ylabel("R²")
        axes[0].set_title(f"k={k} raw")
        axes[1].scatter(g["A_B"], g["probe_r2"], s=10, alpha=0.5, c="#b85c38")
        axes[1].set_xlabel("A_B")
        axes[1].set_title("curvature alignment")
        axes[2].scatter(g["C_w"], g["probe_r2"], s=10, alpha=0.5, c="#2a6f4e")
        axes[2].set_xlabel("C_w")
        axes[2].set_title("probe-visible")
        fig.suptitle(f"Alignment diagnostics k={k}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"alignment_k{k}.png", dpi=140)
        plt.close(fig)


def write_report(
    out: Path,
    cfg: AlignmentConfig,
    cond_by_k: dict,
    align_assoc: pd.DataFrame,
    cv_df: pd.DataFrame,
    labels: list[str],
    narrative: str,
    remat_summary: dict,
) -> None:
    def fmt_cond(k):
        c = cond_by_k[k]
        return (
            f"### k={k} ({c['role']})\n"
            f"- corr(C°,R²)={c['corr_C_R2']['rho']:.4f}  p_perm={c['corr_C_R2']['p_perm']:.4g}  "
            f"CI={c['corr_C_R2']['ci95']}\n"
            f"- corr(C°,log ρ)={c['corr_C_log_radius']['rho']:.4f}\n"
            f"- corr(R²,log ρ)={c['corr_R2_log_radius']['rho']:.4f}\n"
            f"- partial(radius)={c['partial_radius_only']['rho']:.4f} (p={c['partial_radius_only']['pvalue']:.4g}) "
            f"CI={c['boot_partial_radius']['ci95']}\n"
            f"- partial(+labvar)={c['partial_radius_labvar']['rho']:.4f}\n"
            f"- partial(full)={c['partial_full']['rho']:.4f} CI={c['boot_partial_full']['ci95']}\n"
            f"- corr(K°,R²)={c['corr_K_R2']['rho']:.4f}; partial(K°|radius)={c['partial_K_radius']['rho']:.4f}\n"
            f"- corr(log K°,R²)={c['corr_logK_R2']['rho']:.4f}; partial(logK|radius)={c['partial_logK_radius']['rho']:.4f}\n"
            f"- residual var frac(C°|radius)={c['residual_diagnostics'].get('residual_var_frac')}; "
            f"VIF={c['residual_diagnostics'].get('vif_C_vs_radius')}\n"
        )

    report = f"""# Curvature–probe alignment follow-up

Frozen config_hash `{cfg.expected_hash}` verified. Probe α={cfg.probe_alpha}. No curvature refit/retune.

## Rematerialization

{json.dumps(remat_summary, indent=2)}

## 1. Raw versus conditional association

Interpret raw ρ as total observational association; partial ρ as conditional association.
Do **not** claim fully radius-driven unless unscaled K° and within-decile analyses are also null.

{fmt_cond(2048)}
{fmt_cond(1024)}
{fmt_cond(512)}

Within-radius-decile tables: `within_radius_decile_correlations.csv`.

## 2. Probe / subspace alignment

Fold-aggregated features: A_T, A_N, A_B (full curvature span), C_w.
Associations (Spearman + radius-partial):

{align_assoc.to_string(index=False)}

## 3. Nested CV incremental value (rank regression)

{cv_df.to_string(index=False)}

## Mechanism note (k=1024 vs k=2048)

{narrative}

## Conservative labels

{chr(10).join(f"- `{lab}`" for lab in labels)}

## Exact next command (not run)

Do not launch retrieval / Fisher / JS / new curvature estimation from this report.
"""
    (out / "REPORT.md").write_text(report)


def run_alignment(cfg: AlignmentConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()

    # load frozen curvature + screening join
    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    curv = load_frozen_curvature(root, scfg)
    screen_join = pd.read_parquet(resolve_path(root, cfg.screen_dir) / "joined_curvature_probe.parquet")

    align_path = out / "alignment_features.parquet"
    if _done(align_path, cfg.force):
        align = pd.read_parquet(align_path)
    else:
        print("[align] rematerializing frozen J/B° + fold probes", flush=True)
        align = rematerialize_and_align(root, cfg, curv)
        align.to_parquet(align_path, index=False)

    joined = screen_join.merge(align, on=["sample_id", "scale_k"], how="left", suffixes=("", "_al"))
    # ensure log_knn
    if "log_knn_radius" not in joined.columns:
        joined["log_knn_radius"] = np.log(np.maximum(joined["knn_radius"].astype(float), EPS))
    joined_path = out / "joined_alignment.parquet"
    joined.to_parquet(joined_path, index=False)

    remat_ok = joined.loc[joined.align_ok == True, "rematerialize_rel_err"] if "rematerialize_rel_err" in joined.columns else pd.Series(dtype=float)
    remat_summary = {
        "n_align_ok": int((joined.get("align_ok") == True).sum()) if "align_ok" in joined.columns else 0,
        "median_rel_err": float(remat_ok.median()) if len(remat_ok) else float("nan"),
        "max_rel_err": float(remat_ok.max()) if len(remat_ok) else float("nan"),
        "frac_within_tol": float((remat_ok <= cfg.rematerialize_tol).mean()) if len(remat_ok) else float("nan"),
    }

    # conditional association per scale
    cond_by_k = {}
    cond_rows = []
    decile_rows = []
    for k in cfg.scales:
        role = "primary" if k == 2048 else ("comparison" if k == 1024 else "exploratory")
        block = conditional_block(joined[joined.scale_k == k], cfg, role)
        block["role"] = role
        cond_by_k[k] = block
        cond_rows.append(
            {
                "scale_k": k,
                "role": role,
                "n": block["n"],
                "rho_C_R2": block["corr_C_R2"]["rho"],
                "p_perm_C_R2": block["corr_C_R2"]["p_perm"],
                "ci95_C_R2_lo": block["corr_C_R2"]["ci95"][0],
                "ci95_C_R2_hi": block["corr_C_R2"]["ci95"][1],
                "rho_C_log_radius": block["corr_C_log_radius"]["rho"],
                "rho_R2_log_radius": block["corr_R2_log_radius"]["rho"],
                "rho_partial_radius": block["partial_radius_only"]["rho"],
                "p_partial_radius": block["partial_radius_only"]["pvalue"],
                "ci95_partial_radius_lo": block["boot_partial_radius"]["ci95"][0],
                "ci95_partial_radius_hi": block["boot_partial_radius"]["ci95"][1],
                "rho_partial_radius_labvar": block["partial_radius_labvar"]["rho"],
                "rho_partial_full": block["partial_full"]["rho"],
                "p_partial_full": block["partial_full"]["pvalue"],
                "ci95_partial_full_lo": block["boot_partial_full"]["ci95"][0],
                "ci95_partial_full_hi": block["boot_partial_full"]["ci95"][1],
                "rho_K_R2": block["corr_K_R2"]["rho"],
                "rho_partial_K_radius": block["partial_K_radius"]["rho"],
                "rho_logK_R2": block["corr_logK_R2"]["rho"],
                "rho_partial_logK_radius": block["partial_logK_radius"]["rho"],
                **{f"diag_{kk}": vv for kk, vv in block["residual_diagnostics"].items()},
            }
        )
        for drow in block["within_decile"]:
            decile_rows.append({"scale_k": k, "metric": "C", **drow})
        for drow in block["within_decile_K"]:
            decile_rows.append({"scale_k": k, "metric": "K", **drow})

    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(out / "conditional_association_by_scale.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(out / "within_radius_decile_correlations.csv", index=False)
    (out / "conditional_association_full.json").write_text(json.dumps(cond_by_k, indent=2, default=str))

    # alignment associations
    align_assoc = association_with_align(joined, cfg)
    align_assoc.to_csv(out / "alignment_associations.csv", index=False)

    # nested CV models per scale
    cv_all = []
    for k in cfg.scales:
        g = joined[joined.scale_k == k].copy()
        feature_sets = {
            "controls": ["log_knn_radius", "local_label_variance", "reconstruction_error"],
            "controls+curvature_magnitude": [
                "log_knn_radius",
                "local_label_variance",
                "reconstruction_error",
                "B_traceless_fro",
                PRIMARY_CURV,
            ],
            "controls+tangent_normal": [
                "log_knn_radius",
                "local_label_variance",
                "reconstruction_error",
                "A_T",
                "A_N",
            ],
            "controls+curvature_subspace": [
                "log_knn_radius",
                "local_label_variance",
                "reconstruction_error",
                "A_B",
            ],
            "controls+probe_visible": [
                "log_knn_radius",
                "local_label_variance",
                "reconstruction_error",
                "C_w",
            ],
        }
        # drop models if columns missing
        feature_sets = {
            name: cols
            for name, cols in feature_sets.items()
            if all(c in g.columns for c in cols)
        }
        rows = nested_cv_incremental(g, feature_sets, seed=cfg.seed + k)
        for r in rows:
            r["scale_k"] = k
        cv_all.extend(rows)
    cv_df = pd.DataFrame(cv_all)
    cv_df.to_csv(out / "nested_cv_incremental.csv", index=False)

    labels, narrative = choose_labels(cond_by_k, align_assoc, cv_df)
    make_plots(joined, cond_by_k, out)
    write_report(out, cfg, cond_by_k, align_assoc, cv_df, labels, narrative, remat_summary)

    analysis = {
        "labels": labels,
        "narrative": narrative,
        "conditional": cond_rows,
        "rematerialize": remat_summary,
        "seconds": time.time() - t0,
        "config_hash": cfg.expected_hash,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[align] labels={labels}", flush=True)
    return analysis
