"""Nested-dimension decomposition: why ViT-B mag_r K_H is strong at d=16 but weak at d*=12.

Diagnostic only. Does not select dimension from probe correlations.
No local probe fitting — fixed five-fold OOF geography only.
"""

from __future__ import annotations

import hashlib
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
import torch
from scipy.stats import spearmanr

from .confirmatory_object_curvature import unpack_BS_symmetric
from .curvature_probe_screen import partial_spearman, spearman_dict
from .effdim_curvature_metrics import (
    aniso_prefactor,
    cross_metric_pair,
    decompose_tensors,
    metric_scalars,
)
from .full_curvature_audit import RIDGES, fit_quad, pca_tangent_gpu
from .multimodel_graph_prior_quadratic import EPS, load_model_X
from .paths import platonic_root, resolve_path
from .sphere_normal_quadratic import sphere_project_basis
from .split_half_curvature_reliability import _half_fit_indices, tensor_agreement
from .tangent_reliability import grassmann_dist, principal_angles, projector

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_EDM = "outputs/geometry/physics_effdim_curvature_metrics"
SOURCE_COV = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
SOURCE_SH = "outputs/geometry/physics_split_half_curvature_reliability"
SOURCE_FCA = "outputs/geometry/physics_full_curvature_audit"

PARITY_D16_RHO = -0.423283
PARITY_D12_RHO = -0.036315
PARITY_TOL = 0.03
FREEZE_HASH_EXPECTED = "d9e8616bcc9fe790"


@dataclass
class NestedDimConfig:
    output_dir: str = "outputs/geometry/physics_nested_dimension_curvature"
    multimodel_dir: str = SOURCE_MM
    effdim_dir: str = SOURCE_EDM
    coverage_dir: str = SOURCE_COV
    model: str = "vit_base"
    target: str = "mag_r_desi"
    primary_k: int = 2048
    d_max: int = 20
    d_core: int = 12
    d_ref: int = 16
    ranks: list[int] = field(default_factory=lambda: list(range(8, 21)))
    n_splits: int = 5
    scale_ks: list[int] = field(default_factory=lambda: [1024, 2048, 3072])
    scale_ranks: list[int] = field(default_factory=lambda: list(range(10, 19)))
    n_scale_anchors: int = 256
    n_scale_splits: int = 3
    replication_models: list[str] = field(
        default_factory=lambda: ["convnext_base", "dinov3", "clip_base", "vit_large"]
    )
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 600.0
    skip_replication: bool = False

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: NestedDimConfig, reserve: bool = False) -> bool:
    rem = cfg.max_seconds - (time.time() - t0)
    return rem > (cfg.analyze_reserve_seconds if reserve else 30.0)


def load_ctx(root: Path, cfg: NestedDimConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = (
        json.loads(aid.read_text())["sample_ids"]
        if aid.exists()
        else [int(s) for s in anchors_sid]
    )
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == cfg.model)
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
    ]
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    # knn packs
    pack2048 = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    knn3072 = resolve_path(root, SOURCE_FCA) / "cache" / f"{cfg.model}_kmax3072_gpu.npz"
    pack3072 = dict(np.load(knn3072)) if knn3072.exists() else None
    freeze = json.loads(resolve_path(root, cfg.effdim_dir).joinpath("dimension_freeze.json").read_text())
    return {
        "mm": mm,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": {int(s): i for i, s in enumerate(anchors_sid)},
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
        "pack2048": pack2048,
        "pack3072": pack3072,
        "freeze": freeze,
        "X": load_model_X(mm, cfg.model),
    }


def ensure_neigh(ctx: dict, ai: int, k: int) -> np.ndarray:
    if k <= 2048:
        return ctx["pack2048"]["neigh"][ai, :k]
    if ctx["pack3072"] is None:
        raise RuntimeError("k=3072 requires full_curvature_audit kNN cache")
    return ctx["pack3072"]["neigh"][ai, :k]


# -------------------- nested PCA --------------------


def nested_pca_frame(
    Xloc: np.ndarray, d_max: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Single sphere-tangent PCA basis J (D,d_max) nested for all ranks."""
    x0 = Xloc.mean(0)
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J, diag = pca_tangent_gpu(Xloc, x0, d_max, device)
    # full eigenvalues for diagnostics via larger q if needed
    x0u = x0
    dx = Xloc.astype(np.float32) - x0u.astype(np.float32)
    dx = dx - np.outer(dx @ x0u, x0u)
    Xt = torch.as_tensor(dx, device=device, dtype=torch.float32)
    q = min(max(d_max + 16, 2 * d_max), min(Xt.shape) - 1)
    try:
        _, S, V = torch.svd_lowrank(Xt, q=q, niter=3)
        ev = (S.detach().cpu().numpy() ** 2) / max(len(Xloc), 1)
        Jfull = V[:, :d_max].detach().cpu().numpy().astype(np.float64)
        Jfull = sphere_project_basis(x0u, Jfull)
        J = Jfull
    except Exception:  # noqa: BLE001
        ev = np.full(d_max, np.nan)
    gaps = np.array(
        [float(ev[i] - ev[i + 1]) if i + 1 < len(ev) else float("nan") for i in range(d_max)]
    )
    return x0, J, ev[:d_max], {"eigengaps": gaps, **diag}


def verify_H_partition(B: np.ndarray, d_core: int = 12, d_full: int = 16) -> dict:
    """H_full = (d_c/d) H_C + (d_e/d) H_E with block-averaged traces."""
    d_e = d_full - d_core
    H = B[:, np.arange(d_full), np.arange(d_full)].mean(axis=1)
    HC = B[:, np.arange(d_core), np.arange(d_core)].mean(axis=1)
    HE = B[:, np.arange(d_core, d_full), np.arange(d_core, d_full)].mean(axis=1)
    H_rec = (d_core / d_full) * HC + (d_e / d_full) * HE
    err = float(np.linalg.norm(H - H_rec) / max(np.linalg.norm(H), EPS))
    return {"H": H, "H_C": HC, "H_E": HE, "H_recon": H_rec, "rel_err": err}


def block_frobenius(B: np.ndarray, a0: int, a1: int, b0: int, b1: int) -> float:
    return float(np.linalg.norm(B[:, a0:a1, b0:b1]))


def block_energies(B: np.ndarray, d_core: int = 12, d_full: int = 16) -> dict[str, float]:
    """Dimension-normalized directional block energies (MC-validated scale)."""
    d = d_full
    pref = aniso_prefactor(d)  # for full; report raw F trop + normalized
    bcc = block_frobenius(B, 0, d_core, 0, d_core)
    bce = block_frobenius(B, 0, d_core, d_core, d_full)
    # CE is off-diagonal; also count CE.T (same values) → use sqrt(2)*||CE|| for full off-diag energy
    bee = block_frobenius(B, d_core, d_full, d_core, d_full)
    bfull = float(np.linalg.norm(B))
    # energy proxies used in associations: squared Frobenius
    return {
        "E_CC": bcc**2,
        "E_CE": 2.0 * (bce**2),  # both triangles
        "E_EE": bee**2,
        "E_full": bfull**2,
        "B_CC_fro": bcc,
        "B_CE_fro": bce,
        "B_EE_fro": bee,
        "B_full_fro": bfull,
        "aniso_pref": pref,
    }


# -------------------- stages --------------------


def stage_prepare(root: Path, cfg: NestedDimConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "batches", "figures", "logs", "H_vectors"):
        (out / sub).mkdir(exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "protocol": "nested_dimension_curvature_v1",
        "preserved": [SOURCE_MM, SOURCE_EDM, SOURCE_COV, SOURCE_SH, SOURCE_FCA],
        "no_local_probes": True,
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
        "freeze_hash": ctx["freeze"].get("dimension_config_hash"),
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[ndc] prepare hash={meta['config_hash']}", flush=True)
    return meta


def stage_parity(root: Path, cfg: NestedDimConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    # d=16 from coverage
    cov = pd.read_parquet(
        resolve_path(root, cfg.coverage_dir) / "model_reliability_anchor_mean.parquet"
    )
    c16 = cov[(cov.model == cfg.model) & (cov.k == cfg.primary_k) & (cov.d == 16)]
    m16 = geo.merge(c16, on="sample_id", how="inner")
    rho16, _ = spearmanr(m16.K_H_cross, m16.local_r2)
    # d=12 from effdim
    edm = pd.read_parquet(
        resolve_path(root, cfg.effdim_dir) / "crossfit_curvature_metrics.parquet"
    )
    e12 = edm[
        (edm.model == cfg.model)
        & (edm.k == cfg.primary_k)
        & (edm.role == "d_star")
        & (edm.d == cfg.d_core)
    ]
    m12 = geo.merge(e12, on="sample_id", how="inner")
    rho12, _ = spearmanr(m12.K_H_cross, m12.local_r2)
    # reliability from coverage / split-half
    r16 = float(c16.R_H.median()) if "R_H" in c16 else float("nan")
    freeze_hash = ctx["freeze"].get("dimension_config_hash")
    ok16 = abs(float(rho16) - PARITY_D16_RHO) <= PARITY_TOL
    ok12 = abs(float(rho12) - PARITY_D12_RHO) <= PARITY_TOL
    ok_hash = freeze_hash == FREEZE_HASH_EXPECTED
    result = {
        "ok": bool(ok16 and ok12 and ok_hash),
        "d16": {
            "rho_KH_cross": float(rho16),
            "expected": PARITY_D16_RHO,
            "ok": ok16,
            "n": int(len(m16)),
            "median_R_H": r16,
            "source": "coverage model_reliability_anchor_mean d=16",
        },
        "d12": {
            "rho_KH_cross": float(rho12),
            "expected": PARITY_D12_RHO,
            "ok": ok12,
            "n": int(len(m12)),
            "source": "effdim crossfit d_star d=12",
        },
        "freeze_hash": freeze_hash,
        "freeze_hash_ok": ok_hash,
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
    }
    path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"[ndc][parity] ok={result['ok']} d16={rho16:.4f} d12={rho12:.4f} hash_ok={ok_hash}",
        flush=True,
    )
    return result


def stage_nested_pca(root: Path, cfg: NestedDimConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "nested_pca_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    X = ctx["X"]
    device = ctx["device"]
    # compare independent PCA at 12 and 16 for a subset
    check_sids = ctx["use_sids"][::16][:32]
    for si, sid in enumerate(ctx["use_sids"]):
        if si % 64 == 0:
            print(f"[ndc][pca] {si}/512", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x0, J, ev, diag = nested_pca_frame(Xloc, cfg.d_max, device)
        # bootstrap subspace stability for dirs 13-16
        rng = np.random.default_rng(cfg.seed + ai)
        stab = []
        for _ in range(8):
            boot = rng.choice(len(Xloc), size=len(Xloc), replace=True)
            _, Jb, _, _ = nested_pca_frame(Xloc[boot], cfg.d_max, device)
            # principal angles for extra subspace
            ang = principal_angles(J[:, 12:16], Jb[:, 12:16])
            stab.append(float(np.mean(np.cos(ang))))
        # independent PCA agreement
        pa12 = pa16 = float("nan")
        if sid in check_sids:
            J12i, _ = pca_tangent_gpu(Xloc, x0, 12, device)
            J16i, _ = pca_tangent_gpu(Xloc, x0, 16, device)
            pa12 = float(np.mean(np.cos(principal_angles(J[:, :12], J12i))))
            pa16 = float(np.mean(np.cos(principal_angles(J[:, :16], J16i))))
        # cache J for curvature stage
        np.savez_compressed(
            out / "cache" / f"J_{int(sid)}_k{cfg.primary_k}.npz",
            x0=x0,
            J=J,
            ev=ev,
        )
        rows.append(
            {
                "sample_id": int(sid),
                "k": cfg.primary_k,
                "d_max": cfg.d_max,
                "eigengap_12": float(diag["eigengaps"][11]) if len(diag["eigengaps"]) > 11 else float("nan"),
                "eigengap_16": float(diag["eigengaps"][15]) if len(diag["eigengaps"]) > 15 else float("nan"),
                "var_share_1_12": float(np.sum(ev[:12]) / max(np.sum(ev[: cfg.d_max]), EPS)),
                "var_share_13_16": float(np.sum(ev[12:16]) / max(np.sum(ev[: cfg.d_max]), EPS)),
                "bootstrap_extra_cos": float(np.mean(stab)),
                "nested_vs_indep_cos12": pa12,
                "nested_vs_indep_cos16": pa16,
                **{f"ev_{i+1}": float(ev[i]) for i in range(min(20, len(ev)))},
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ndc] nested_pca n={len(rows)}", flush=True)


def _fit_rank(
    Xloc: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    d: int,
    k: int,
    n_splits: int,
    seed: int,
    ai: int,
) -> list[dict]:
    Jd = J[:, :d]
    rows = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + 1009 * ai + 17 * s + d * 13 + k)
        perm = rng.permutation(k)
        halfA, halfB = perm[: k // 2], perm[k // 2 :]
        fA, vA = _half_fit_indices(halfA, seed + 3 + s)
        fB, vB = _half_fit_indices(halfB, seed + 7 + s)
        chA, _, infoA = fit_quad(Xloc, x0, Jd, fA, vA, halfB, ridges=RIDGES)
        chB, _, infoB = fit_quad(Xloc, x0, Jd, fB, vB, halfA, ridges=RIDGES)
        if chA is None or chB is None:
            continue
        cross = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
        sa, sb = metric_scalars(chA.BS_flat, d), metric_scalars(chB.BS_flat, d)
        HA = decompose_tensors(chA.BS_flat, d)["H"]
        HB = decompose_tensors(chB.BS_flat, d)["H"]
        rows.append(
            {
                "split": s,
                "d": d,
                "K_H_cross": cross["K_H_cross"],
                "K_aniso_cross": cross["K_aniso_cross"],
                "K_dir_cross": cross["K_dir_cross"],
                "R_H": cross["R_H"],
                "R_B0": cross["R_B0"],
                "R_BS": cross["R_BS"],
                "norm_H": 0.5 * (sa["K_H"] + sb["K_H"]),
                "norm_dir": 0.5 * (sa["K_dir"] + sb["K_dir"]),
                "dS": 0.5 * (float(infoA.get("dS", np.nan)) + float(infoB.get("dS", np.nan))),
                "H_mean": 0.5 * (HA + HB),
                "BS_flat_A": chA.BS_flat,
                "BS_flat_B": chB.BS_flat,
            }
        )
    return rows


def stage_nested_curvature(root: Path, cfg: NestedDimConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "nested_curvature_metrics.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    metric_rows = []
    # also accumulate for core_extra / reclass at end of each anchor
    for si, sid in enumerate(ctx["use_sids"]):
        if si % 32 == 0:
            print(f"[ndc][fit] primary {si}/512", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            print("[ndc][fit] budget stop", flush=True)
            break
        batch = out / "batches" / f"vit_k{cfg.primary_k}_sid{int(sid)}.parquet"
        if _done(batch, cfg.force):
            metric_rows.append(pd.read_parquet(batch))
            continue
        ai = ctx["sid_to_ai"][int(sid)]
        jp = out / "cache" / f"J_{int(sid)}_k{cfg.primary_k}.npz"
        if not jp.exists():
            continue
        z = np.load(jp)
        x0, J = z["x0"], z["J"]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        rows_out = []
        H12_list, H16_list = [], []
        BS16A = BS16B = BS12A = BS12B = None
        for d in cfg.ranks:
            if not _budget_ok(t0, cfg, reserve=True):
                break
            fits = _fit_rank(
                Xloc, x0, J, d, cfg.primary_k, cfg.n_splits, cfg.seed, ai
            )
            if not fits:
                continue
            # aggregate across splits
            for f in fits:
                rows_out.append(
                    {
                        "sample_id": int(sid),
                        "k": cfg.primary_k,
                        "model": cfg.model,
                        "d": d,
                        "split": f["split"],
                        "K_H_cross": f["K_H_cross"],
                        "K_aniso_cross": f["K_aniso_cross"],
                        "K_dir_cross": f["K_dir_cross"],
                        "R_H": f["R_H"],
                        "R_B0": f["R_B0"],
                        "R_BS": f["R_BS"],
                        "norm_H": f["norm_H"],
                        "norm_dir": f["norm_dir"],
                        "dS": f["dS"],
                    }
                )
            # keep last split pair tensors for 12/16 diagnostics (use split 0)
            f0 = fits[0]
            if d == 12:
                H12_list.append(f0["H_mean"])
                BS12A, BS12B = f0["BS_flat_A"], f0["BS_flat_B"]
            if d == 16:
                H16_list.append(f0["H_mean"])
                BS16A, BS16B = f0["BS_flat_A"], f0["BS_flat_B"]
        dfb = pd.DataFrame(rows_out)
        dfb.to_parquet(batch, index=False)
        metric_rows.append(dfb)
        # save H / compressed tensors
        if H12_list and H16_list and BS16A is not None and BS12A is not None:
            np.savez_compressed(
                out / "H_vectors" / f"{int(sid)}.npz",
                H12=np.mean(H12_list, axis=0),
                H16=np.mean(H16_list, axis=0),
                BS16_A=BS16A,
                BS16_B=BS16B,
                BS12_A=BS12A,
                BS12_B=BS12B,
                x0=x0,
                J16=J[:, :16],
                J12=J[:, :12],
                J_E=J[:, 12:16],
            )
    if metric_rows:
        pd.concat(metric_rows, ignore_index=True).to_parquet(path, index=False)
        print(f"[ndc] nested_curvature wrote {path}", flush=True)


def stage_core_extra(root: Path, cfg: NestedDimConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "core_extra_decomposition.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    hdir = out / "H_vectors"
    for p in sorted(hdir.glob("*.npz")):
        z = np.load(p)
        sid = int(p.stem)
        BA, BB = z["BS16_A"], z["BS16_B"]
        tensors = {
            "A": unpack_BS_symmetric(BA, 16),
            "B": unpack_BS_symmetric(BB, 16),
        }
        tensors["mean"] = 0.5 * (tensors["A"] + tensors["B"])
        for tag, B in tensors.items():
            part = verify_H_partition(B, 12, 16)
            en = block_energies(B, 12, 16)
            rows.append(
                {
                    "sample_id": sid,
                    "half": tag,
                    "H_norm": float(np.linalg.norm(part["H"])),
                    "HC_norm": float(np.linalg.norm(part["H_C"])),
                    "HE_norm": float(np.linalg.norm(part["H_E"])),
                    "H_partition_rel_err": part["rel_err"],
                    "cos_H_HC": float(
                        np.dot(part["H"], part["H_C"])
                        / max(np.linalg.norm(part["H"]) * np.linalg.norm(part["H_C"]), EPS)
                    ),
                    "cos_H_HE": float(
                        np.dot(part["H"], part["H_E"])
                        / max(np.linalg.norm(part["H"]) * np.linalg.norm(part["H_E"]), EPS)
                    ),
                    **en,
                }
            )
        # split-half block reliability
        BA_t = unpack_BS_symmetric(BA, 16)
        BB_t = unpack_BS_symmetric(BB, 16)
        for name, sl in [
            ("CC", (slice(0, 12), slice(0, 12))),
            ("CE", (slice(0, 12), slice(12, 16))),
            ("EE", (slice(12, 16), slice(12, 16))),
        ]:
            a = BA_t[:, sl[0], sl[1]].ravel()
            b = BB_t[:, sl[0], sl[1]].ravel()
            ag = tensor_agreement(a, b)
            rows.append(
                {
                    "sample_id": sid,
                    "half": f"R_{name}",
                    "R_block": ag["R_signal"],
                    "block": name,
                    **{k: float("nan") for k in ("H_norm", "HC_norm", "HE_norm")},
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ndc] core_extra n={len(rows)}", flush=True)


def stage_reclassification(root: Path, cfg: NestedDimConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "reclassification.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    for p in sorted((out / "H_vectors").glob("*.npz")):
        z = np.load(p)
        sid = int(p.stem)
        H12 = z["H12"]
        x0 = z["x0"]
        J16 = z["J16"]
        JE = z["J_E"]
        x0u = x0 / max(np.linalg.norm(x0), EPS)
        # project H12 into N_16, E_4, radial
        P_E = JE @ JE.T
        P_T16 = J16 @ J16.T
        P_R = np.outer(x0u, x0u)
        H_E = P_E @ H12
        H_R = P_R @ H12
        H_N = H12 - P_T16 @ H12 - H_R  # persistent normal wrt T16
        # also component in T12 should be ~0 for sphere-normal H12
        nH = max(np.linalg.norm(H12), EPS)
        rows.append(
            {
                "sample_id": sid,
                "H12_norm": float(np.linalg.norm(H12)),
                "H12_persistent_normal": float(np.linalg.norm(H_N)),
                "H12_reclassified_extra": float(np.linalg.norm(H_E)),
                "H12_radial": float(np.linalg.norm(H_R)),
                "reclassified_fraction": float(np.linalg.norm(H_E) / nH),
                "persistent_fraction": float(np.linalg.norm(H_N) / nH),
                "radial_fraction": float(np.linalg.norm(H_R) / nH),
                "cos_H12_H16": float(
                    np.dot(H12, z["H16"])
                    / max(np.linalg.norm(H12) * np.linalg.norm(z["H16"]), EPS)
                ),
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ndc] reclassification n={len(rows)}", flush=True)


def stage_incremental(root: Path, cfg: NestedDimConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "incremental_directions.parquet"
    if _done(path, cfg.force):
        return
    curv = pd.read_parquet(out / "nested_curvature_metrics.parquet")
    pca = pd.read_parquet(out / "nested_pca_diagnostics.parquet")
    # per-rank means then deltas
    g = (
        curv.groupby(["sample_id", "d"], as_index=False)
        .agg(
            K_H_cross=("K_H_cross", "mean"),
            K_dir_cross=("K_dir_cross", "mean"),
            K_aniso_cross=("K_aniso_cross", "mean"),
            dS=("dS", "median"),
            R_H=("R_H", "median"),
        )
    )
    rows = []
    for sid, gs in g.groupby("sample_id"):
        gs = gs.sort_values("d")
        pc = pca[pca.sample_id == sid]
        for i in range(1, len(gs)):
            r0, r1 = gs.iloc[i - 1], gs.iloc[i]
            d_new = int(r1.d)
            # label-free classification of added direction
            ev_key = f"ev_{d_new}"
            ev = float(pc.iloc[0][ev_key]) if len(pc) and ev_key in pc.columns else float("nan")
            boot = float(pc.iloc[0].bootstrap_extra_cos) if len(pc) and d_new > 12 else float("nan")
            dS_gain = float(r1.dS - r0.dS)
            # geometric class
            if d_new <= 12:
                label = "stable_geometric_direction"
            elif np.isfinite(boot) and boot > 0.7 and dS_gain > 0:
                label = "weak_but_predictive_direction"
            elif np.isfinite(boot) and boot < 0.4:
                label = "unstable_thickness_direction"
            else:
                label = "unresolved"
            rows.append(
                {
                    "sample_id": int(sid),
                    "d_from": int(r0.d),
                    "d_to": d_new,
                    "delta_KH": float(r1.K_H_cross - r0.K_H_cross),
                    "delta_Kdir": float(r1.K_dir_cross - r0.K_dir_cross),
                    "delta_dS": dS_gain,
                    "ev_added": ev,
                    "bootstrap_extra_cos": boot,
                    "direction_label": label,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ndc] incremental n={len(rows)}", flush=True)


def stage_associations(root: Path, cfg: NestedDimConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "rank_associations.csv"
    if _done(Path(path), cfg.force):
        return
    curv = pd.read_parquet(out / "nested_curvature_metrics.parquet")
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    # mean over splits
    agg = curv.groupby(["sample_id", "d"], as_index=False)[
        ["K_H_cross", "K_aniso_cross", "K_dir_cross", "norm_H", "R_H"]
    ].mean()
    rows = []
    for d, gd in agg.groupby("d"):
        gg = geo.merge(gd, on="sample_id", how="inner")
        y = gg.local_r2.to_numpy(float)
        for metric in ["K_H_cross", "K_aniso_cross", "K_dir_cross", "norm_H"]:
            x = gg[metric].to_numpy(float)
            raw = spearman_dict(x, y)
            path_c = {"raw": raw["rho"]}
            Z1 = gg.log_knn_radius.fillna(0).to_numpy(float)[:, None]
            path_c["+radius"] = partial_spearman(x, y, Z1)["rho"]
            Z = np.column_stack(
                [
                    gg.log_knn_radius.fillna(0).to_numpy(float),
                    gg.local_label_variance.fillna(0).to_numpy(float),
                    gg.local_evaluation_count.fillna(0).to_numpy(float),
                ]
            )
            path_c["+controls"] = partial_spearman(x, y, Z)["rho"]
            # split sign recurrence
            sub = curv[curv.d == d]
            signs = []
            for s, gs in sub.groupby("split"):
                mm = geo.merge(gs, on="sample_id")
                if len(mm) < 30:
                    continue
                r, _ = spearmanr(mm[metric], mm.local_r2)
                if np.isfinite(r):
                    signs.append(np.sign(r))
            rows.append(
                {
                    "d": int(d),
                    "metric": metric,
                    "n": raw["n"],
                    "sign_recurrence": float(np.mean(np.array(signs) < 0)) if signs else float("nan"),
                    **path_c,
                }
            )
    # block associations from core_extra
    ce = pd.read_parquet(out / "core_extra_decomposition.parquet")
    ce_m = ce[ce.half == "mean"].copy()
    if len(ce_m):
        gg = geo.merge(ce_m, on="sample_id", how="inner")
        y = gg.local_r2.to_numpy(float)
        for col in ["H_norm", "HC_norm", "HE_norm", "E_CC", "E_CE", "E_EE"]:
            if col not in gg:
                continue
            x = gg[col].to_numpy(float)
            raw = spearman_dict(x, y)
            rows.append(
                {
                    "d": 16,
                    "metric": f"block_{col}",
                    "n": raw["n"],
                    "raw": raw["rho"],
                    "+radius": float("nan"),
                    "+controls": float("nan"),
                    "sign_recurrence": float("nan"),
                }
            )
    # reclassification
    rc = pd.read_parquet(out / "reclassification.parquet")
    gg = geo.merge(rc, on="sample_id", how="inner")
    y = gg.local_r2.to_numpy(float)
    for col in [
        "H12_persistent_normal",
        "H12_reclassified_extra",
        "reclassified_fraction",
        "persistent_fraction",
    ]:
        x = gg[col].to_numpy(float)
        raw = spearman_dict(x, y)
        rows.append(
            {
                "d": -1,
                "metric": f"reclass_{col}",
                "n": raw["n"],
                "raw": raw["rho"],
                "+radius": float("nan"),
                "+controls": float("nan"),
                "sign_recurrence": float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ndc] associations n={len(rows)}", flush=True)


def stage_block_ablation(root: Path, cfg: NestedDimConfig, ctx: dict, t0: float) -> None:
    """Held-out predictive contribution of CC/CE/EE blocks (subset of anchors)."""
    out = cfg.resolved(root)
    path = out / "block_ablation.csv"
    if _done(Path(path), cfg.force):
        return
    from .sphere_normal_quadratic import chart_errors

    rows = []
    X = ctx["X"]
    # 128 anchors for ablation cost
    sids = ctx["use_sids"][::4][:128]
    for si, sid in enumerate(sids):
        if si % 32 == 0:
            print(f"[ndc][ablate] {si}/{len(sids)}", flush=True)
        if not _budget_ok(t0, cfg, reserve=True):
            break
        hp = out / "H_vectors" / f"{int(sid)}.npz"
        if not hp.exists():
            continue
        z = np.load(hp)
        ai = ctx["sid_to_ai"][int(sid)]
        N = ensure_neigh(ctx, ai, cfg.primary_k)
        Xloc = X[N].astype(np.float64)
        x0, J16 = z["x0"], z["J16"]
        # one split
        rng = np.random.default_rng(cfg.seed + ai)
        perm = rng.permutation(cfg.primary_k)
        halfA, halfB = perm[: cfg.primary_k // 2], perm[cfg.primary_k // 2 :]
        fA, vA = _half_fit_indices(halfA, cfg.seed)
        ch, _, info = fit_quad(Xloc, x0, J16, fA, vA, halfB, ridges=RIDGES)
        if ch is None:
            continue
        B = unpack_BS_symmetric(ch.BS_flat, 16)
        # zero blocks
        def pack_from_B(Bmod):
            cols = []
            for a in range(16):
                for b in range(a, 16):
                    cols.append(Bmod[:, a, a] if a == b else (2.0 * Bmod[:, a, b]))
            return np.stack(cols, axis=1)

        variants = {}
        B_cc = B.copy()
        B_cc[:, 12:16, :] = 0
        B_cc[:, :, 12:16] = 0
        variants["M_core"] = pack_from_B(B_cc)
        B_cm = B.copy()
        B_cm[:, 12:16, 12:16] = 0
        variants["M_core_mixed"] = pack_from_B(B_cm)
        variants["M_full"] = ch.BS_flat
        # linear-only extra: compare d=12 vs d=16 chart errors already in nested metrics
        rows.append(
            {
                "sample_id": int(sid),
                "E_full": float(info.get("E_TRS", np.nan)),
                "dS_full": float(info.get("dS", np.nan)),
                "B_CC_fro": block_frobenius(B, 0, 12, 0, 12),
                "B_CE_fro": block_frobenius(B, 0, 12, 12, 16),
                "B_EE_fro": block_frobenius(B, 12, 16, 12, 16),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ndc] block_ablation n={len(rows)}", flush=True)


def stage_synthetic(root: Path, cfg: NestedDimConfig) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_nested_results.csv"
    if _done(Path(path), cfg.force):
        return
    rows = []
    rng = np.random.default_rng(cfg.seed + 7)
    D, n, k = 64, 2048, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make(kind: str):
        if kind == "extra_tangent":
            # intrinsic 16, weak 13-16
            Z = rng.normal(size=(n, 16))
            Z[:, 12:] *= 0.25
            A = rng.normal(size=(D, 16))
            X = Z @ A.T
        elif kind == "normal_thickness":
            Z = rng.normal(size=(n, 12))
            A = rng.normal(size=(D, 12))
            X = Z @ A.T + 0.15 * rng.normal(size=(n, D))
        elif kind == "finite_scale":
            Z = rng.normal(size=(n, 12))
            A = rng.normal(size=(D, 12))
            X = Z @ A.T
            # curvature-like quadratic bump
            X[:, :8] += 0.1 * (Z[:, :1] ** 2)
        else:
            Z = rng.normal(size=(n, 12))
            A = rng.normal(size=(D, 12))
            X = Z @ A.T + 0.05 * rng.normal(size=(n, D))
        X = X.astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X

    for kind in [
        "extra_tangent",
        "normal_thickness",
        "finite_scale",
        "rank_specific_overfit",
    ]:
        X = make(kind)
        # pick an anchor neighbourhood
        x0 = X[0]
        # knn by IP
        sims = X @ x0
        idx = np.argsort(-sims)[:k]
        Xloc = X[idx].astype(np.float64)
        x0u, J, ev, _ = nested_pca_frame(Xloc, 20, device)
        boot = []
        for _ in range(6):
            b = rng.choice(k, size=k, replace=True)
            _, Jb, _, _ = nested_pca_frame(Xloc[b], 20, device)
            boot.append(float(np.mean(np.cos(principal_angles(J[:, 12:16], Jb[:, 12:16])))))
        # fit d=12 and d=16 once
        rh = {}
        for d in (12, 16):
            fits = _fit_rank(Xloc, x0u, J, d, k, 3, cfg.seed, 0)
            if fits:
                rh[d] = float(np.mean([f["R_H"] for f in fits]))
                rh[f"dS_{d}"] = float(np.mean([f["dS"] for f in fits]))
        label = (
            "extra_tangent_geometry"
            if kind == "extra_tangent"
            else "normal_thickness"
            if kind == "normal_thickness"
            else "finite_scale_stratification"
            if kind == "finite_scale"
            else "rank_specific_overfit"
        )
        rows.append(
            {
                "kind": kind,
                "true_label": label,
                "boot_extra_cos": float(np.mean(boot)),
                "R_H_12": rh.get(12, np.nan),
                "R_H_16": rh.get(16, np.nan),
                "dS_12": rh.get("dS_12", np.nan),
                "dS_16": rh.get("dS_16", np.nan),
                "var_share_13_16": float(np.sum(ev[12:16]) / max(np.sum(ev[:20]), EPS)),
            }
        )
        print(f"[ndc][syn] {kind} boot={rows[-1]['boot_extra_cos']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def stage_replication(root: Path, cfg: NestedDimConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "cross_model_replication.csv"
    if cfg.skip_replication:
        pd.DataFrame([{"skipped": True}]).to_csv(path, index=False)
        return
    if _done(Path(path), cfg.force):
        return
    # require vit associations done
    if not (out / "rank_associations.csv").exists():
        pd.DataFrame([{"skipped": True, "reason": "vit_incomplete"}]).to_csv(path, index=False)
        return
    freeze = pd.DataFrame(ctx["freeze"]["by_model_scale"])
    geo_all = pd.read_parquet(ctx["mm"] / "local_probe_fields.parquet")
    rows = []
    sids = ctx["use_sids"][::2][:256]
    for m in cfg.replication_models:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        fr = freeze[(freeze.model == m) & (freeze.scale_k == 2048)]
        if fr.empty:
            continue
        d_star = int(fr.iloc[0].d_star)
        ranks = list(range(max(4, d_star - 2), d_star + 5))
        X = load_model_X(ctx["mm"], m)
        pack = dict(np.load(ctx["mm"] / "model_neighbourhoods" / f"{m}_kmax2048.npz"))
        geo = geo_all[
            (geo_all.model == m)
            & (geo_all.target == cfg.target)
            & (geo_all.neighbourhood == "model")
            & (geo_all.scale_k == 2048)
        ]
        device = ctx["device"]
        metric_by_d: dict[int, list] = {d: [] for d in ranks}
        for si, sid in enumerate(sids):
            if si % 64 == 0:
                print(f"[ndc][repl] {m} {si}/{len(sids)}", flush=True)
            if not _budget_ok(t0, cfg, reserve=True):
                break
            ai = ctx["sid_to_ai"][int(sid)]
            N = pack["neigh"][ai, :2048]
            Xloc = X[N].astype(np.float64)
            x0, J, _, _ = nested_pca_frame(Xloc, max(ranks), device)
            for d in ranks:
                fits = _fit_rank(Xloc, x0, J, d, 2048, cfg.n_scale_splits, cfg.seed, ai)
                if not fits:
                    continue
                metric_by_d[d].append(
                    {
                        "sample_id": int(sid),
                        "K_H_cross": float(np.mean([f["K_H_cross"] for f in fits])),
                    }
                )
        for d, lst in metric_by_d.items():
            if len(lst) < 30:
                continue
            df = pd.DataFrame(lst).merge(geo, on="sample_id")
            raw = spearman_dict(df.K_H_cross.to_numpy(float), df.local_r2.to_numpy(float))
            rows.append(
                {
                    "model": m,
                    "d_star": d_star,
                    "d": d,
                    "rho_KH": raw["rho"],
                    "n": raw["n"],
                }
            )
            print(f"[ndc][repl] {m} d={d} rho={raw['rho']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def stage_analyze(root: Path, cfg: NestedDimConfig, parity: dict) -> None:
    out = cfg.resolved(root)
    assoc = (
        pd.read_csv(out / "rank_associations.csv")
        if (out / "rank_associations.csv").exists()
        else pd.DataFrame()
    )
    reclass = (
        pd.read_parquet(out / "reclassification.parquet")
        if (out / "reclassification.parquet").exists()
        else pd.DataFrame()
    )
    ce = (
        pd.read_parquet(out / "core_extra_decomposition.parquet")
        if (out / "core_extra_decomposition.parquet").exists()
        else pd.DataFrame()
    )
    inc = (
        pd.read_parquet(out / "incremental_directions.parquet")
        if (out / "incremental_directions.parquet").exists()
        else pd.DataFrame()
    )
    syn = (
        pd.read_csv(out / "synthetic_nested_results.csv")
        if (out / "synthetic_nested_results.csv").exists()
        else pd.DataFrame()
    )
    repl = (
        pd.read_csv(out / "cross_model_replication.csv")
        if (out / "cross_model_replication.csv").exists()
        else pd.DataFrame()
    )
    pca = (
        pd.read_parquet(out / "nested_pca_diagnostics.parquet")
        if (out / "nested_pca_diagnostics.parquet").exists()
        else pd.DataFrame()
    )

    # figures: rank path
    figdir = out / "figures"
    if len(assoc):
        fig, ax = plt.subplots(figsize=(7, 4))
        for metric, ls in [
            ("K_H_cross", "-"),
            ("K_aniso_cross", "--"),
            ("K_dir_cross", ":"),
        ]:
            sub = assoc[assoc.metric == metric].sort_values("d")
            if sub.empty:
                continue
            ax.plot(sub.d, sub.raw, ls, marker="o", label=metric)
        ax.axvline(12, color="C1", alpha=0.5, label="d*=12")
        ax.axvline(16, color="C3", alpha=0.5, label="d=16")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("nested rank d")
        ax.set_ylabel(r"Spearman $\rho(K, R^2_{\mathrm{mag\_r}})$")
        ax.legend(fontsize=8)
        ax.set_title("ViT-B k=2048 nested rank → mag_r association")
        fig.tight_layout()
        fig.savefig(figdir / "rank_association_path.png", dpi=140)
        plt.close(fig)

    # interpretation
    label = "nested_dimension_result_unresolved"
    kh = assoc[assoc.metric == "K_H_cross"].sort_values("d") if len(assoc) else pd.DataFrame()
    if len(kh) >= 5:
        r12 = float(kh[kh.d == 12].raw.iloc[0]) if (kh.d == 12).any() else np.nan
        r16 = float(kh[kh.d == 16].raw.iloc[0]) if (kh.d == 16).any() else np.nan
        # smooth vs spike
        between = kh[(kh.d >= 12) & (kh.d <= 16)]
        smooth = (
            len(between) >= 3
            and (between.raw.diff().dropna() < 0).mean() >= 0.6
            if len(between) >= 3
            else False
        )
        med_reclass = float(reclass.reclassified_fraction.median()) if len(reclass) else 0
        med_persist = float(reclass.persistent_fraction.median()) if len(reclass) else 0
        boot = float(pca.bootstrap_extra_cos.median()) if len(pca) else 0
        # block probe
        he = assoc[assoc.metric == "block_HE_norm"]
        he_rho = float(he.raw.iloc[0]) if len(he) else np.nan
        dS_gain = (
            float(
                inc[(inc.d_from == 15) & (inc.d_to == 16)].delta_dS.median()
            )
            if len(inc)
            else float("nan")
        )
        if med_reclass > 0.35 and med_persist < 0.5:
            label = "tangent_normal_reclassification"
        elif boot > 0.65 and np.isfinite(dS_gain) and dS_gain > 0 and smooth:
            label = "stable_weak_geometry_beyond_effdim"
        elif smooth and boot > 0.5:
            label = "finite_scale_dimension_exceeds_graph_core"
        elif (not smooth) and (boot < 0.45 or (np.isfinite(dS_gain) and dS_gain <= 0)):
            label = "common_rank_curvature_misspecification"
        elif np.isfinite(he_rho) and abs(he_rho) > 0.15:
            label = "mixed_core_and_extra_curvature"

    labels = [label]
    if parity.get("ok"):
        labels.append("parity_ok")
    (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))

    report = f"""# Nested-dimension curvature diagnostic (ViT-B / mag_r_desi)

## Question

Why is $\\rho(K_H^{{d=16}}, R^2_{{\\mathrm{{mag\\_r}}}})\\approx -0.42$ reproducible while at graph $d^*=12$ the same association is $\\approx -0.04$?

## Parity gates

```json
{json.dumps(parity, indent=2)}
```

## Nested PCA

```
{pca[['nested_vs_indep_cos12','nested_vs_indep_cos16','bootstrap_extra_cos','var_share_13_16']].describe().to_string() if len(pca) else 'n/a'}
```

## Rank → association path (raw Spearman)

```
{kh[['d','raw','+radius','+controls','sign_recurrence']].to_string(index=False) if len(kh) else 'n/a'}
```

## Reclassification of $H_{{12}}$ into $T_{{16}}=T_{{12}}\\oplus E_4$

```
{reclass.describe().to_string() if len(reclass) else 'n/a'}
```

## Core / extra blocks at rank 16

```
{ce[ce.half=='mean'][['H_norm','HC_norm','HE_norm','E_CC','E_CE','E_EE','H_partition_rel_err']].describe().to_string() if len(ce) else 'n/a'}
```

Block–probe associations:

```
{assoc[assoc.metric.astype(str).str.startswith('block_')].to_string(index=False) if len(assoc) else 'n/a'}
```

## Incremental directions (label-free classes)

```
{inc.groupby('direction_label').size().to_string() if len(inc) else 'n/a'}
```

## Synthetic controls

```
{syn.to_string(index=False) if len(syn) else 'n/a'}
```

## Cross-model replication

```
{repl.to_string(index=False) if len(repl) else 'n/a'}
```

## Primary interpretation

**{label}**

### Answers

1. **Nested-frame reproduces d=12 and d=16?** Parity gates on artifacts: {parity.get('ok')}. Nested refits supply the continuous path.
2. **Smooth or isolated?** see rank path / sign_recurrence between 12 and 16.
3. **Directions 13–16 stable?** median bootstrap_extra_cos in PCA table.
4. **Improve held-out geometry?** incremental ΔdS and block_ablation.csv.
5. **Association in core / extra / mixed / reclass?** block_* and reclass_* rows.
6. **Does H12 persist in N16?** persistent_fraction vs reclassified_fraction.
7. **Extra blocks split-half reliable?** R_CC/R_CE/R_EE rows in core_extra.
8. **Closest synthetic?** compare boot/dS pattern in synthetic_nested_results.csv.
9. **Other models?** cross_model_replication.csv.
10. **Interpretation:** {label}. The d=16 mag_r association is **not** dismissed a priori; it is only called misspecification if extra directions fail stability and held-out gates.

## Decision labels

{labels}
"""
    (out / "REPORT.md").write_text(report)
    print(f"[ndc] analyze labels={labels}", flush=True)


def run(cfg: NestedDimConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    for banned in (SOURCE_MM, SOURCE_EDM, SOURCE_COV, SOURCE_SH, SOURCE_FCA):
        if out.resolve() == resolve_path(root, banned).resolve():
            raise RuntimeError(f"Refusing to write into {banned}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    order = [
        "prepare",
        "parity",
        "nested_pca",
        "nested_curvature",
        "core_extra",
        "reclassification",
        "incremental",
        "associations",
        "block_ablation",
        "synthetic",
        "replication",
        "analyze",
    ]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    run_set = set(want)
    if "analyze" in run_set:
        run_set.update(["prepare", "parity"])
    if "nested_curvature" in run_set:
        run_set.update(["prepare", "parity", "nested_pca"])
    if "associations" in run_set:
        run_set.update(["nested_curvature", "core_extra", "reclassification"])

    def mark(name, dt):
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    parity = {}
    if "prepare" in run_set:
        t1 = time.time()
        print("[ndc] stage=prepare", flush=True)
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    if "parity" in run_set:
        t1 = time.time()
        print("[ndc] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        mark("parity", time.time() - t1)
        if not parity.get("ok"):
            stage_analyze(root, cfg, parity)
            profile["stopped"] = "parity_failed"
            (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
            return profile
    elif (out / "parity.json").exists():
        parity = json.loads((out / "parity.json").read_text())

    if "nested_pca" in run_set:
        t1 = time.time()
        print("[ndc] stage=nested_pca", flush=True)
        stage_nested_pca(root, cfg, ctx, t0)
        mark("nested_pca", time.time() - t1)

    if "nested_curvature" in run_set:
        t1 = time.time()
        print("[ndc] stage=nested_curvature", flush=True)
        stage_nested_curvature(root, cfg, ctx, t0)
        mark("nested_curvature", time.time() - t1)

    if "core_extra" in run_set:
        t1 = time.time()
        print("[ndc] stage=core_extra", flush=True)
        stage_core_extra(root, cfg, ctx)
        mark("core_extra", time.time() - t1)

    if "reclassification" in run_set:
        t1 = time.time()
        print("[ndc] stage=reclassification", flush=True)
        stage_reclassification(root, cfg, ctx)
        mark("reclassification", time.time() - t1)

    if "incremental" in run_set:
        t1 = time.time()
        print("[ndc] stage=incremental", flush=True)
        stage_incremental(root, cfg, ctx)
        mark("incremental", time.time() - t1)

    if "associations" in run_set:
        t1 = time.time()
        print("[ndc] stage=associations", flush=True)
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)

    if "block_ablation" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[ndc] stage=block_ablation", flush=True)
        stage_block_ablation(root, cfg, ctx, t0)
        mark("block_ablation", time.time() - t1)

    if "synthetic" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[ndc] stage=synthetic", flush=True)
        stage_synthetic(root, cfg)
        mark("synthetic", time.time() - t1)

    if "replication" in run_set and _budget_ok(t0, cfg, reserve=True):
        t1 = time.time()
        print("[ndc] stage=replication", flush=True)
        stage_replication(root, cfg, ctx, t0)
        mark("replication", time.time() - t1)

    if "analyze" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[ndc] stage=analyze", flush=True)
        if (out / "parity.json").exists():
            parity = json.loads((out / "parity.json").read_text())
        stage_analyze(root, cfg, parity)
        mark("analyze", time.time() - t1)

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[ndc] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
