"""Quadratic predictive dimension: config, freeze, parity, and local fitting.

Never writes into completed geometry output directories.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    angular_radii,
    rms_tangent_radius,
    row_l2_status,
    sphere_log_map,
)

from .algebra import (
    EPS,
    closest_point_project,
    component_r2,
    geodesic_error,
    n_quad_features,
    nmse,
    phi2,
    predict_f,
    project_B_normal,
    r2_from_nmse,
    remove_radial_cols,
    remove_radial_rows,
    ridge_df,
    ridge_fit,
    ridge_grid_from_gram,
    scale_phi_train,
)
from .classify import DEFAULT_THRESHOLDS

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_NDC = "outputs/geometry/physics_nested_dimension_curvature"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_STD = "outputs/geometry/physics_stable_tangent_dimension"
SOURCE_OSG = "outputs/geometry/physics_order_stratified_geometry"
SOURCE_INI = "outputs/geometry/physics_implicit_normal_inverse"

PRESERVED = [SOURCE_MM, SOURCE_EDM, SOURCE_NDC, SOURCE_COV, SOURCE_STD, SOURCE_OSG, SOURCE_INI]
PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03
PARITY_E4_R2 = 0.15
PARITY_E4_TOL = 0.08
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"
K_CANDIDATES = [512, 768, 1024, 1536, 2048]


@dataclass
class QuadPredConfig:
    output_dir: str = "outputs/geometry/physics_quadratic_predictive_dimension"
    multimodel_dir: str = SOURCE_MM
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    R: int = 20
    d_min: int = 4
    d_max: int = 20
    d_core: int = 12
    d_ref: int = 16
    n_parity_anchors: int = 32
    n_synth_cal: int = 4
    n_synth_eval: int = 4
    n_scale_anchors: int = 128
    n_inner_cp: int = 96
    ks: list[int] = field(default_factory=list)
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 400.0
    smoke: bool = False
    n_anchors: int | None = None

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)

    def ds(self) -> list[int]:
        return list(range(int(self.d_min), int(self.d_max) + 1))


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: QuadPredConfig, reserve: bool = False) -> bool:
    return (cfg.max_seconds - (time.time() - t0)) > (cfg.analyze_reserve_seconds if reserve else 20.0)


def _sha16(payload: Any) -> str:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


def assert_not_preserved(out: Path, root: Path) -> None:
    resolved = out.resolve()
    for rel in PRESERVED:
        pres = resolve_path(root, rel).resolve()
        if resolved == pres or pres in resolved.parents:
            raise RuntimeError(f"refusing to write into preserved geometry dir {rel}")


def atomic_replace(tmp: Path, dest: Path, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and force:
        ts = time.strftime("%Y%m%dT%H%M%S")
        dest.rename(dest.with_name(f"{dest.stem}.superseded.{ts}{dest.suffix}"))
    tmp.replace(dest)


def write_df(path: Path, df: pd.DataFrame, *, force: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    atomic_replace(tmp, path, force=force)


def hash_select(sids: list[int], n: int, *, seed: int) -> list[int]:
    scored = []
    for s in sids:
        h = hashlib.sha256(f"qpd:{seed}:{int(s)}".encode()).hexdigest()
        scored.append((h, int(s)))
    scored.sort()
    return [s for _, s in scored[: min(n, len(scored))]]


def resolve_k_grid(k_max: int, *, smoke: bool, primary_k: int) -> list[int]:
    if smoke:
        grid = [k for k in [64, 96, 128, 192] if k <= k_max]
        return grid[:4] if grid else [min(k_max, 64)]
    grid = [k for k in K_CANDIDATES if k <= k_max]
    if primary_k <= k_max and primary_k not in grid:
        grid.append(int(primary_k))
    return sorted(set(grid))


def load_ctx(root: Path, cfg: QuadPredConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(aid.read_text())["sample_ids"] if aid.exists() else [int(s) for s in anchors_sid]
    if cfg.n_anchors is not None:
        use_sids = use_sids[: int(cfg.n_anchors)]
    elif cfg.smoke:
        use_sids = use_sids[:8]
        cfg.d_max = min(int(cfg.d_max), 8)
        cfg.d_min = min(int(cfg.d_min), 4)
        cfg.n_synth_cal = min(cfg.n_synth_cal, 2)
        cfg.n_synth_eval = min(cfg.n_synth_eval, 2)
        cfg.n_scale_anchors = min(cfg.n_scale_anchors, 8)
        cfg.n_inner_cp = 32
        cfg.n_parity_anchors = min(cfg.n_parity_anchors, 8)
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[(geo.model == cfg.model) & (geo.target == cfg.target) & (geo.neighbourhood == "model")]
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    freeze_p = resolve_path(root, SOURCE_EDM) / "dimension_freeze.json"
    freeze = json.loads(freeze_p.read_text()) if freeze_p.exists() else {}
    k_max = int(pack["neigh"].shape[1])
    ks = list(cfg.ks) if cfg.ks else resolve_k_grid(min(k_max, cfg.primary_k), smoke=cfg.smoke, primary_k=cfg.primary_k)
    X = load_model_X(mm, cfg.model)
    scale_sids = hash_select(use_sids, cfg.n_scale_anchors, seed=cfg.seed)
    return {
        "mm": mm,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "scale_sids": scale_sids,
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "pack": pack,
        "freeze": freeze,
        "X": X,
        "ks": ks,
        "k_max": k_max,
        "l2": row_l2_status(X),
        "std": resolve_path(root, SOURCE_STD),
        "ndc": resolve_path(root, SOURCE_NDC),
        "edm": resolve_path(root, SOURCE_EDM),
        "cov": resolve_path(root, SOURCE_COV),
        "osg": resolve_path(root, SOURCE_OSG),
        "ini": resolve_path(root, SOURCE_INI),
    }


def ensure_neigh(ctx: dict, ai: int, k: int) -> np.ndarray:
    return ctx["pack"]["neigh"][ai, : min(k, ctx["pack"]["neigh"].shape[1])]


def frozen_J_path(ctx: dict, cfg: QuadPredConfig, sid: int, k: int) -> Path | None:
    for base in (ctx["osg"], ctx["ini"]):
        p = base / "J" / f"{cfg.model}_{int(sid)}_k{int(k)}.npz"
        if p.exists():
            return p
    return None


def load_frozen_J(ctx: dict, cfg: QuadPredConfig, sid: int, k: int) -> np.ndarray | None:
    p = frozen_J_path(ctx, cfg, sid, k)
    if p is None:
        return None
    J = np.load(p)["J"]
    return J


def local_pack(ctx: dict, cfg: QuadPredConfig, sid: int, k: int):
    ai = ctx["sid_to_ai"][int(sid)]
    X = ctx["X"]
    N = ensure_neigh(ctx, ai, k)
    Xloc = X[N].astype(np.float64)
    x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
    Z = sphere_log_map(x0, Xloc)
    Z = remove_radial_rows(Z, x0)
    th = angular_radii(x0, Xloc)
    return x0, Xloc, Z, th, N


def _pick_ridge(cands: list[dict[str, float]]) -> dict[str, float]:
    losses = [c["inner_nmse"] for c in cands]
    best = float(np.nanmin(losses))
    finite = [c for c in cands if np.isfinite(c["inner_nmse"])]
    if not finite:
        return cands[0]
    se = float(np.nanstd(losses) / max(np.sqrt(len(losses)), 1.0))
    thresh = best + se
    # prefer more regularized among 1-SE
    ok = [c for c in finite if c["inner_nmse"] <= thresh]
    ok.sort(key=lambda c: (c["lam"], c["inner_nmse"]))
    return ok[-1] if ok else min(finite, key=lambda c: c["inner_nmse"])


def fit_neighbourhood(
    Z: np.ndarray,
    radii: np.ndarray,
    x0: np.ndarray,
    *,
    ds: list[int],
    thr: dict[str, Any],
    seed: int,
    frozen_J: np.ndarray | None,
    d_core: int,
    d_ref: int,
    R: int,
    n_inner_cp: int,
    device=None,
) -> list[dict[str, Any]]:
    rows = []
    Aidx, Bidx = radial_stratified_halves(radii, seed)
    folds = [("A", Aidx, Bidx), ("B", Bidx, Aidx)]
    d_max = max(ds)
    u_q = float(thr.get("u_bound_q", 0.99))
    max_iter = int(thr.get("gn_max_iter", 8))
    damp = float(thr.get("gn_damp", 1e-4))
    n_grid = int(thr.get("ridge_n_grid", 11))
    for fold_name, tr, te in folds:
        if min(len(tr), len(te)) < d_max + 8:
            continue
        Jall, _ = nested_uncentred_svd(Z[tr], d_max, device=device, centre=False)
        if Jall.shape[1] < min(ds):
            continue
        subA, subB = radial_stratified_halves(radii[tr], seed + 19)
        if min(len(subA), len(subB)) < 8:
            cut = max(len(tr) // 5, 8)
            inner_tr, inner_va = tr[cut:], tr[:cut]
        else:
            inner_tr, inner_va = tr[subA], tr[subB]
        for d in ds:
            if Jall.shape[1] < d:
                continue
            J = Jall[:, :d]
            Utr = Z[inner_tr] @ J
            Phi_raw = phi2(Utr)
            Phi, rms = scale_phi_train(Phi_raw)
            Ytr = Z[inner_tr] - Utr @ J.T
            G = Phi.T @ Phi
            C = Phi.T @ Ytr
            grid = ridge_grid_from_gram(G, len(inner_tr), n_grid)
            Uva = Z[inner_va] @ J
            Phi_va = phi2(Uva) / rms[None, :]
            n_cp = min(int(n_inner_cp), len(inner_va))
            va_cp = inner_va[:n_cp]
            cands = []
            for lam in grid:
                Bsc = ridge_fit(Phi, Ytr, float(lam), G=G, C=C)
                B = Bsc / rms[None, :]
                B = remove_radial_cols(B, x0)
                pack = closest_point_project(
                    Z[va_cp],
                    J,
                    B,
                    Z[va_cp] @ J,
                    u_max=float(np.quantile(np.linalg.norm(Utr, axis=1), u_q)) if len(Utr) else 1.0,
                    max_iter=max_iter,
                    damp=damp,
                    x_anchor=x0,
                    device=device,
                )
                cands.append(
                    {
                        "lam": float(lam),
                        "inner_nmse": pack["close_nmse"],
                        "inner_fixed": pack["fixed_nmse"],
                        "df": ridge_df(G, float(lam)),
                    }
                )
            picked = _pick_ridge(cands)
            lam = picked["lam"]
            # refit on full outer train
            U_full = Z[tr] @ J
            Phi_f, rms_f = scale_phi_train(phi2(U_full))
            Y_full = Z[tr] - U_full @ J.T
            Gf = Phi_f.T @ Phi_f
            Bsc = ridge_fit(Phi_f, Y_full, lam, G=Gf, C=Phi_f.T @ Y_full)
            B = remove_radial_cols(Bsc / rms_f[None, :], x0)
            BN = project_B_normal(B, J)
            u_max = float(np.quantile(np.linalg.norm(U_full, axis=1), u_q))
            Ute = Z[te] @ J
            # linear
            Zlin = Ute @ J.T
            lin_nmse = nmse(Z[te], Zlin)
            # unrestricted closest + fixed
            cpu = closest_point_project(
                Z[te], J, B, Ute, u_max=u_max, max_iter=max_iter, damp=damp, x_anchor=x0, device=device
            )
            cpn = closest_point_project(
                Z[te], J, BN, Ute, u_max=u_max, max_iter=max_iter, damp=damp, x_anchor=x0, device=device
            )
            Zfix = predict_f(Ute, J, B)
            sse_close = float(np.sum((Z[te] - cpu["Zhat"]) ** 2))
            energy = float(np.sum(Z[te] * Z[te]))
            rec: dict[str, Any] = {
                "fold": fold_name,
                "d": int(d),
                "n_tr": int(len(tr)),
                "n_te": int(len(te)),
                "n_features": int(n_quad_features(d)),
                "lam": lam,
                "df": picked["df"],
                "df_frac": float(picked["df"] / max(n_quad_features(d), 1)),
                "train_nmse_fixed": nmse(Z[tr], predict_f(U_full, J, B)),
                "lin_nmse": lin_nmse,
                "lin_r2": r2_from_nmse(lin_nmse),
                "quad_fixed_nmse": cpu["fixed_nmse"],
                "quad_close_nmse": cpu["close_nmse"],
                "quad_fixed_r2": cpu["fixed_r2"],
                "quad_close_r2": cpu["close_r2"],
                "quadN_fixed_nmse": cpn["fixed_nmse"],
                "quadN_close_nmse": cpn["close_nmse"],
                "quadN_close_r2": cpn["close_r2"],
                "mean_euclid": cpu["mean_euclid"],
                "median_euclid": cpu["median_euclid"],
                "boundary_frac": float(np.mean(cpu["boundary"])),
                "improved_frac": float(np.mean(cpu["improved"])),
                "mean_n_iter": float(np.mean(cpu["n_iter"])),
                "u_max": u_max,
                "inner_nmse": picked["inner_nmse"],
                "test_energy": energy,
                "test_sse_close": sse_close,
                "test_sse_lin": float(np.sum((Z[te] - Zlin) ** 2)),
                "coord_mae": float(np.mean(np.abs(Z[te] - cpu["Zhat"]))),
            }
            rec["gap"] = rec["quad_close_nmse"] - rec["train_nmse_fixed"]
            if frozen_J is not None and frozen_J.shape[1] >= min(R, frozen_J.shape[1]):
                Jf = frozen_J
                T12 = Jf[:, : min(d_core, Jf.shape[1])]
                E4 = Jf[:, d_core:d_ref] if Jf.shape[1] >= d_ref else Jf[:, d_core:]
                U4 = Jf[:, d_ref:R] if Jf.shape[1] >= R else Jf[:, d_ref:]
                U8 = Jf[:, d_core:R] if Jf.shape[1] >= R else Jf[:, d_core:]
                rec["r2_T12"] = component_r2(Z[te], cpu["Zhat"], T12)
                rec["r2_E4"] = component_r2(Z[te], cpu["Zhat"], E4)
                rec["r2_U4"] = component_r2(Z[te], cpu["Zhat"], U4)
                rec["r2_U8"] = component_r2(Z[te], cpu["Zhat"], U8)
                rec["r2_E4_fixed"] = component_r2(Z[te], Zfix, E4)
                rec["r2_E4_normal"] = component_r2(Z[te], cpn["Zhat"], E4)
                rec["r2_outside"] = _r2_complement(Z[te], cpu["Zhat"], Jf[:, : min(R, Jf.shape[1])])
                rec["r2_T12_lin"] = component_r2(Z[te], Zlin, T12)
                rec["r2_E4_lin"] = component_r2(Z[te], Zlin, E4)
                rec["r2_U8_lin"] = component_r2(Z[te], Zlin, U8)
            if d in (12, 16, max(ds)):
                rec["geo_err"] = geodesic_error(x0, Z[te][:64], cpu["Zhat"][:64])
            else:
                rec["geo_err"] = float("nan")
            rows.append(rec)
    return rows


def _r2_complement(Z: np.ndarray, Zhat: np.ndarray, JR: np.ndarray) -> float:
    """Held-out R^2 in the orthogonal complement of span(JR)."""
    if JR.size == 0:
        return float("nan")
    Pz = Z @ JR
    Pr = (Z - Zhat) @ JR
    tot = float(np.sum(Z * Z) - np.sum(Pz * Pz))
    num = float(np.sum((Z - Zhat) ** 2) - np.sum(Pr * Pr))
    if tot < EPS:
        return float("nan")
    return float(1.0 - num / tot)


def _outside_basis(J: np.ndarray, R: int, D: int) -> np.ndarray:
    if J.shape[1] < R:
        P = np.eye(D) - J @ J.T
    else:
        P = np.eye(D) - J[:, :R] @ J[:, :R].T
    U, s, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, s > 0.5][:, : min(8, int(np.sum(s > 0.5)))]


def stage_prepare(root: Path, cfg: QuadPredConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    assert_not_preserved(out, root)
    for sub in ("cache", "figures", "logs", "synth"):
        (out / sub).mkdir(exist_ok=True)
    mm = ctx["mm"]
    x_path = mm / "prepare" / "models" / f"{cfg.model}.npz"
    pack_path = mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    probes = mm / "global_probes" / "oof_predictions"
    probe_hash = _sha16(sorted(p.name for p in probes.glob("*")) if probes.exists() else "missing")
    meta = {
        "config": asdict(cfg),
        "protocol": "quadratic_predictive_dimension_v1",
        "preserved": PRESERVED,
        "ks": ctx["ks"],
        "ds": cfg.ds(),
        "n_anchors": len(ctx["use_sids"]),
        "n_scale_anchors": len(ctx["scale_sids"]),
        "scale_anchor_ids": ctx["scale_sids"],
        "scale_anchor_rule": "sha256(qpd:{seed}:{sample_id}) lexicographic prefix, disclosed before fitting",
        "primary_k": cfg.primary_k,
        "l2_status": ctx["l2"],
        "no_probe_selection": True,
        "correction_prior_E4": "order-stratified R2_E4≈0.15 means 15% explained, 85% unexplained",
        "software": {"numpy": np.__version__, "torch": torch.__version__, "pandas": pd.__version__},
        "hashes": {
            "activations": _file_sha(x_path) if x_path.exists() else None,
            "knn_pack": _file_sha(pack_path) if pack_path.exists() else None,
            "oof_probes": probe_hash,
            "freeze": ctx["freeze"].get("dimension_config_hash"),
        },
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
        "seeds": {"analysis": cfg.seed},
        "thresholds_frozen_from": "synthetic_calibration",
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "logs" / "prepare.log").write_text(
        f"ks={ctx['ks']} ds={cfg.ds()} n={len(ctx['use_sids'])} scale={len(ctx['scale_sids'])}\n"
    )
    print(
        f"[qpd] prepare ks={ctx['ks']} ds={cfg.ds()} n={len(ctx['use_sids'])} "
        f"scale_subset={len(ctx['scale_sids'])}",
        flush=True,
    )
    return meta


from .stages_run import run  # noqa: E402,F401
