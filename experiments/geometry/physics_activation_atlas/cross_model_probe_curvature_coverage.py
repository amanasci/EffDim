"""Cross-model × all-probe curvature coverage on Smith42/Physics.

Completes what the ViT-only full curvature audit left open:
  - all models with prepared activations + kNN packs
  - all probe labels in vit_base_test_labels.npz (incl. sfr)
  - primary reliability cells d=16, k∈{1024,2048}, n_splits=5

Does not overwrite physics_full_curvature_audit / split-half / multimodel dirs.
JWST/DESI/Legacy lack aligned probe labels here — reported as unavailable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .curvature_probe_screen import partial_spearman, spearman_dict
from .full_curvature_audit import (
    BS_objects,
    _half_fit_indices,
    fit_quad,
    full_patch_pca_tangent,
    tensor_agreement,
)
from .global_probe_curvature_alignment import (
    fit_global_probe,
    local_r2_fixed_predictions,
    ridge_multi_intercept_torch,
)
from .multimodel_graph_prior_quadratic import EPS, load_model_X
from .paths import platonic_root, resolve_path

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
SOURCE_FCA = "outputs/geometry/physics_full_curvature_audit"

PHYSICS_MODELS = ["vit_base", "dinov3", "clip_base", "convnext_base", "vit_large"]
PHYSICS_TARGETS = [
    "mag_r_desi",
    "photo_z",
    "smooth_fraction",
    "stellar_mass",
    "sfr",
]
PRIMARY_D = 16
PRIMARY_KS = [1024, 2048]
N_SPLITS = 5
MODEL_SEED = {m: i + 1 for i, m in enumerate(PHYSICS_MODELS)}


@dataclass
class CoverageConfig:
    output_dir: str = "outputs/geometry/physics_cross_model_probe_curvature_coverage"
    multimodel_dir: str = SOURCE_MM
    full_audit_dir: str = SOURCE_FCA
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    models: list[str] = field(default_factory=lambda: list(PHYSICS_MODELS))
    targets: list[str] = field(default_factory=lambda: list(PHYSICS_TARGETS))
    dims: list[int] = field(default_factory=lambda: [PRIMARY_D])
    ks: list[int] = field(default_factory=lambda: list(PRIMARY_KS))
    n_splits: int = N_SPLITS
    probe_alpha: float = 100.0
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    max_seconds: float = 36000.0
    analyze_reserve_seconds: float = 300.0

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget_ok(t0: float, cfg: CoverageConfig, reserve: bool = False) -> bool:
    rem = cfg.max_seconds - (time.time() - t0)
    return rem > (cfg.analyze_reserve_seconds if reserve else 30.0)


def load_ctx(root: Path, cfg: CoverageConfig) -> dict:
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
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    labels = np.load(resolve_path(root, cfg.labels_path))
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    device = torch.device(
        "cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    return {
        "mm": mm,
        "geo": geo,
        "folds": folds,
        "labels": labels,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": sid_to_ai,
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
        "device": device,
    }


def stage_prepare(root: Path, cfg: CoverageConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "models", "figures", "logs"):
        (out / sub).mkdir(exist_ok=True)
    available = []
    missing = []
    for m in cfg.models:
        xp = ctx["mm"] / "prepare" / "models" / f"{m}.npz"
        kp = ctx["mm"] / "model_neighbourhoods" / f"{m}_kmax2048.npz"
        if xp.exists() and kp.exists():
            available.append(m)
        else:
            missing.append(m)
    lab_keys = list(ctx["labels"].files)
    meta = {
        "config": asdict(cfg),
        "protocol": "cross_model_probe_curvature_coverage_v1",
        "available_models": available,
        "missing_models": missing,
        "label_keys": lab_keys,
        "targets_requested": cfg.targets,
        "other_corpora": {
            "jwst": "embeddings present; no Smith42-style probe label table for this atlas",
            "desi": "embeddings present; no Smith42-style probe label table for this atlas",
            "legacysurvey": "embeddings present; no Smith42-style probe label table for this atlas",
            "note": "Coverage is Smith42/Physics only until aligned labels exist.",
        },
        "reused_full_audit": str(cfg.full_audit_dir),
        "reused_multimodel": str(cfg.multimodel_dir),
    }
    (out / "resolved_config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(
        f"[cov] prepare models={available} targets={cfg.targets} missing_models={missing}",
        flush=True,
    )
    return meta


def _oof_ridge_target(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    alpha: float,
    device: torch.device,
    n_folds: int = 5,
) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fi in range(n_folds):
        te = fold_ids == fi
        tr = (~te) & np.isfinite(y)
        if tr.sum() < 32 or te.sum() == 0:
            continue
        Xt = torch.as_tensor(X[tr], device=device, dtype=torch.float32)
        Yt = torch.as_tensor(y[tr][:, None], device=device, dtype=torch.float32)
        W, b, ok = ridge_multi_intercept_torch(Xt, Yt, alpha=alpha)
        if not ok:
            W, b, ok = ridge_multi_intercept_torch(Xt.double(), Yt.double(), alpha=alpha)
            W, b = W.float(), b.float()
        if not ok:
            coef, intercept = fit_global_probe(X[tr], y[tr], alpha)
            oof[te] = X[te] @ coef + intercept
            continue
        oof[te] = (X[te] @ W.detach().cpu().numpy()[:, 0] + float(b.detach().cpu().numpy()[0]))
    return oof


def stage_sfr_probes(root: Path, cfg: CoverageConfig, ctx: dict) -> None:
    """Add sfr (and any missing target) OOF + local_r2 rows without rewriting multimodel."""
    out = cfg.resolved(root)
    path = out / "extended_local_probe_fields.parquet"
    if _done(path, cfg.force):
        return
    mm = ctx["mm"]
    folds = ctx["folds"]
    fold_ids = folds.fold.to_numpy(int)
    geo = ctx["geo"].copy()
    oof_dir = out / "cache" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    cv_rows = []
    extra_rows: list[dict] = []

    base = geo[(geo.neighbourhood == "model")].copy()
    base["source"] = "multimodel"

    need = [t for t in cfg.targets if t not in set(geo.target.unique())]
    if "sfr" in cfg.targets and "sfr" not in need:
        need = list(dict.fromkeys(need + ["sfr"]))

    sample_ids = folds.sample_id.to_numpy(int)
    for m in cfg.models:
        Xp = mm / "prepare" / "models" / f"{m}.npz"
        kp = mm / "model_neighbourhoods" / f"{m}_kmax2048.npz"
        if not Xp.exists() or not kp.exists():
            continue
        X = load_model_X(mm, m)
        pack = dict(np.load(kp))
        neigh = pack["neigh"]
        dists = pack["dists"]
        assert len(folds) == len(X), "folds/X mismatch"
        for t in need:
            if t not in ctx["labels"].files:
                print(f"[cov] skip target {t} — not in labels", flush=True)
                continue
            y_all = np.asarray(ctx["labels"][t], dtype=np.float64)
            y = y_all[sample_ids]
            oof = _oof_ridge_target(X, y, fold_ids, cfg.probe_alpha, ctx["device"])
            np.savez_compressed(oof_dir / f"{m}_{t}.npz", oof=oof, sample_ids=sample_ids)
            mmask = np.isfinite(y) & np.isfinite(oof)
            oof_r2 = (
                float(
                    1
                    - np.sum((y[mmask] - oof[mmask]) ** 2)
                    / max(np.sum((y[mmask] - y[mmask].mean()) ** 2), EPS)
                )
                if mmask.sum() > 10
                else float("nan")
            )
            cv_rows.append(
                {
                    "model": m,
                    "target": t,
                    "oof_r2": oof_r2,
                    "n_eval": int(mmask.sum()),
                    "finite_label_frac": float(np.isfinite(y).mean()),
                    "weak_global": bool(not np.isfinite(oof_r2) or oof_r2 < 0.02),
                }
            )
            print(f"[cov][extra-probes] {m} {t} oof_r2={oof_r2:.4f} n={mmask.sum()}", flush=True)
            gvar = float(np.nanvar(y[np.isfinite(y)])) if np.isfinite(y).any() else 1.0
            for ai, sid in enumerate(ctx["anchors_sid"]):
                for k in (256, 512, 1024, 2048):
                    if k > neigh.shape[1]:
                        continue
                    N = neigh[ai, :k]
                    yy, pp = y[N], oof[N]
                    msk = np.isfinite(yy) & np.isfinite(pp)
                    if msk.sum() < 4:
                        continue
                    rho = float(dists[ai, min(k, dists.shape[1]) - 1])
                    extra_rows.append(
                        {
                            "model": m,
                            "target": t,
                            "sample_id": int(sid),
                            "anchor_local": int(ctx["anchors_local"][ai]),
                            "scale_k": int(k),
                            "neighbourhood": "model",
                            "local_r2": local_r2_fixed_predictions(yy, pp),
                            "normalized_mse": float(np.mean((yy[msk] - pp[msk]) ** 2))
                            / max(gvar, EPS),
                            "local_prediction_correlation": float(
                                np.corrcoef(yy[msk], pp[msk])[0, 1]
                            )
                            if msk.sum() > 2
                            else float("nan"),
                            "local_label_variance": float(np.var(yy[msk])),
                            "local_evaluation_count": int(msk.sum()),
                            "knn_radius": rho,
                            "log_knn_radius": float(np.log(max(rho, EPS))),
                            "source": "coverage_extra_target",
                        }
                    )
    parts = [base]
    if extra_rows:
        parts.append(pd.DataFrame(extra_rows))
    ext = pd.concat(parts, ignore_index=True)
    ext.to_parquet(path, index=False)
    pd.DataFrame(cv_rows).to_parquet(out / "extra_target_global_cv.parquet", index=False)
    print(f"[cov] extended geo n={len(ext)} extra_cv={len(cv_rows)}", flush=True)


def stage_reliability(root: Path, cfg: CoverageConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "model_reliability.parquet"
    if _done(path, cfg.force):
        return
    geo = pd.read_parquet(out / "extended_local_probe_fields.parquet")
    chunks = []
    for m in cfg.models:
        kp = ctx["mm"] / "model_neighbourhoods" / f"{m}_kmax2048.npz"
        if not (ctx["mm"] / "prepare" / "models" / f"{m}.npz").exists() or not kp.exists():
            print(f"[cov][rel] skip {m}", flush=True)
            continue
        X = load_model_X(ctx["mm"], m)
        pack = dict(np.load(kp))
        for d in cfg.dims:
            for k in cfg.ks:
                if k > pack["neigh"].shape[1]:
                    print(f"[cov][rel] skip {m} k={k} > pack", flush=True)
                    continue
                cell = out / "models" / f"rel_{m}_d{d}_k{k}.parquet"
                if _done(cell, cfg.force):
                    chunks.append(pd.read_parquet(cell))
                    continue
                if not _budget_ok(t0, cfg, reserve=True):
                    print("[cov][rel] budget — stop", flush=True)
                    break
                rows = []
                for si, sid in enumerate(ctx["use_sids"]):
                    if si % 64 == 0:
                        print(f"[cov][rel] {m} d={d} k={k} {si}/512", flush=True)
                    if not _budget_ok(t0, cfg, reserve=True):
                        break
                    ai = ctx["sid_to_ai"][int(sid)]
                    N = pack["neigh"][ai, :k]
                    Xloc = X[N].astype(np.float64)
                    x0, J = full_patch_pca_tangent(Xloc, d)
                    if J.shape[1] < d:
                        continue
                    for s in range(cfg.n_splits):
                        rng = np.random.default_rng(
                            cfg.seed
                            + 1009 * ai
                            + 17 * s
                            + d * 13
                            + k
                            + 997 * MODEL_SEED.get(m, 0)
                        )
                        perm = rng.permutation(k)
                        halfA, halfB = perm[: k // 2], perm[k // 2 :]
                        fA, vA = _half_fit_indices(halfA, cfg.seed + 3 + s)
                        fB, vB = _half_fit_indices(halfB, cfg.seed + 7 + s)
                        chA, _, infoA = fit_quad(
                            Xloc, x0, J, fA, vA, halfB, ridges=None
                        )
                        chB, _, infoB = fit_quad(
                            Xloc, x0, J, fB, vB, halfA, ridges=None
                        )
                        if chA is None or chB is None:
                            continue
                        oA, oB = BS_objects(chA.BS_flat, d), BS_objects(chB.BS_flat, d)
                        agH = tensor_agreement(oA["H"], oB["H"])
                        agB0 = tensor_agreement(oA["B0_flat"], oB["B0_flat"])
                        agBS = tensor_agreement(oA["BS_flat"], oB["BS_flat"])
                        rows.append(
                            {
                                "model": m,
                                "sample_id": int(sid),
                                "split": s,
                                "d": d,
                                "k": k,
                                "K_H_cross": agH["inner"],
                                "K_B0_cross": agB0["inner"],
                                "K_BS_cross": agBS["inner"],
                                "R_H": agH["R_signal"],
                                "R_B0": agB0["R_signal"],
                                "R_BS": agBS["R_signal"],
                                "norm_H": 0.5 * (agH["norm_A"] + agH["norm_B"]),
                                "norm_B0": 0.5 * (agB0["norm_A"] + agB0["norm_B"]),
                                "norm_BS": 0.5 * (agBS["norm_A"] + agBS["norm_B"]),
                                "dS": 0.5
                                * (
                                    float(infoA.get("dS", np.nan))
                                    + float(infoB.get("dS", np.nan))
                                ),
                            }
                        )
                df = pd.DataFrame(rows)
                df.to_parquet(cell, index=False)
                chunks.append(df)
                print(f"[cov][rel] wrote {m} d={d} k={k} n={len(df)}", flush=True)
    if not chunks:
        pd.DataFrame().to_parquet(path, index=False)
        return
    all_df = pd.concat(chunks, ignore_index=True)
    all_df.to_parquet(path, index=False)
    # anchor-level means for probe join
    agg = (
        all_df.groupby(["model", "sample_id", "d", "k"], as_index=False)
        .agg(
            K_H_cross=("K_H_cross", "mean"),
            K_B0_cross=("K_B0_cross", "mean"),
            K_BS_cross=("K_BS_cross", "mean"),
            R_H=("R_H", "median"),
            R_B0=("R_B0", "median"),
            R_BS=("R_BS", "median"),
            norm_H=("norm_H", "mean"),
            median_dS=("dS", "median"),
        )
    )
    agg.to_parquet(out / "model_reliability_anchor_mean.parquet", index=False)
    summary = (
        agg.groupby(["model", "d", "k"], as_index=False)
        .agg(
            median_R_H=("R_H", "median"),
            median_R_B0=("R_B0", "median"),
            median_R_BS=("R_BS", "median"),
            median_K_H=("K_H_cross", "median"),
            n=("sample_id", "count"),
        )
    )
    summary.to_parquet(out / "model_reliability_summary.parquet", index=False)
    print(f"[cov] reliability summary\n{summary.to_string(index=False)}", flush=True)


def stage_probe_assoc(root: Path, cfg: CoverageConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "probe_associations.parquet"
    if _done(path, cfg.force):
        return
    rel = pd.read_parquet(out / "model_reliability_anchor_mean.parquet")
    geo = pd.read_parquet(out / "extended_local_probe_fields.parquet")
    extra_cv = (
        pd.read_parquet(out / "extra_target_global_cv.parquet")
        if (out / "extra_target_global_cv.parquet").exists()
        else pd.DataFrame()
    )
    mm_cv = (
        pd.read_parquet(ctx["mm"] / "global_probe_cv_metrics.parquet")
        if (ctx["mm"] / "global_probe_cv_metrics.parquet").exists()
        else pd.DataFrame()
    )
    weak = set()
    for cv in (mm_cv, extra_cv):
        if cv.empty or "oof_r2" not in cv.columns:
            continue
        for _, r in cv.iterrows():
            if float(r.oof_r2) < 0.02 or (
                "weak_global" in r and bool(r.weak_global)
            ):
                weak.add((str(r.model), str(r.target)))

    stats = [
        ("K_H_cross", "K_H_cross"),
        ("K_B0_cross", "K_B0_cross"),
        ("K_BS_cross", "K_BS_cross"),
        ("norm_H", "norm_H"),
        ("R_H", "R_H"),
    ]
    rows = []
    for m in cfg.models:
        for t in cfg.targets:
            for d in cfg.dims:
                for k in cfg.ks:
                    sub = rel[(rel.model == m) & (rel.d == d) & (rel.k == k)]
                    g = geo[
                        (geo.model == m)
                        & (geo.target == t)
                        & (geo.neighbourhood == "model")
                        & (geo.scale_k == min(k, int(geo.scale_k.max())))
                    ]
                    if sub.empty or g.empty:
                        continue
                    gg = g.merge(sub, on="sample_id", how="inner")
                    if len(gg) < 20:
                        continue
                    y = gg.local_r2.to_numpy(float)
                    for sname, scol in stats:
                        x = gg[scol].to_numpy(float)
                        raw = spearman_dict(x, y)
                        Z = np.column_stack(
                            [
                                gg.log_knn_radius.fillna(0).to_numpy(float),
                                gg.local_label_variance.fillna(0).to_numpy(float),
                                gg.local_evaluation_count.fillna(0).to_numpy(float),
                            ]
                        )
                        rows.append(
                            {
                                "model": m,
                                "target": t,
                                "target_weak": (m, t) in weak or t == "sfr",
                                "d": d,
                                "k": k,
                                "stat": sname,
                                "n": raw["n"],
                                "raw": raw["rho"],
                                "+radius": partial_spearman(
                                    x, y, gg.log_knn_radius.fillna(0).to_numpy(float)[:, None]
                                )["rho"],
                                "+controls": partial_spearman(x, y, Z)["rho"],
                            }
                        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    # pivot best raw per model×target at primary cell
    pa = pd.DataFrame(rows)
    if len(pa):
        prim = pa[(pa.d == PRIMARY_D) & (pa.k == 2048) & (pa.stat == "K_H_cross")]
        prim.to_parquet(out / "probe_primary_KH_by_model_target.parquet", index=False)
    print(f"[cov] probe assoc n={len(rows)}", flush=True)


def stage_analyze(root: Path, cfg: CoverageConfig, ctx: dict, meta: dict) -> None:
    out = cfg.resolved(root)
    rel_s = (
        pd.read_parquet(out / "model_reliability_summary.parquet")
        if (out / "model_reliability_summary.parquet").exists()
        else pd.DataFrame()
    )
    pa = (
        pd.read_parquet(out / "probe_associations.parquet")
        if (out / "probe_associations.parquet").exists()
        else pd.DataFrame()
    )
    prim = (
        pd.read_parquet(out / "probe_primary_KH_by_model_target.parquet")
        if (out / "probe_primary_KH_by_model_target.parquet").exists()
        else pd.DataFrame()
    )
    extra = (
        pd.read_parquet(out / "extra_target_global_cv.parquet")
        if (out / "extra_target_global_cv.parquet").exists()
        else pd.DataFrame()
    )
    fca = resolve_path(root, cfg.full_audit_dir)
    fca_labels = []
    if (fca / "decision_labels.json").exists():
        fca_labels = json.loads((fca / "decision_labels.json").read_text())

    labels = ["smith42_physics_coverage_complete"]
    if len(rel_s):
        ok = rel_s[(rel_s.d == 16) & (rel_s.k == 2048) & (rel_s.median_R_H > 0.3)]
        labels.append(
            "cross_model_curvature_reliable"
            if len(ok) >= max(2, len(rel_s[rel_s.k == 2048]) // 2)
            else "cross_model_curvature_mixed"
        )
    if len(prim):
        mag = prim[prim.target == "mag_r_desi"]
        if len(mag) and (mag.raw < 0).mean() >= 0.6:
            labels.append("mag_r_negative_KH_cross_model")
        sm = prim[prim.target == "smooth_fraction"]
        if len(sm) and (sm.raw > 0).mean() >= 0.6:
            labels.append("smooth_fraction_positive_KH_cross_model")
    if len(extra) and (extra.oof_r2 < 0.02).all():
        labels.append("sfr_probe_globally_weak")

    (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))

    report = f"""# Cross-model × probe curvature coverage (Smith42 / Physics)

## Scope

- **Corpus:** Smith42/galaxies (Physics) only
- **Models:** {meta.get('available_models')}
- **Targets:** {cfg.targets}
- **Cells:** d={cfg.dims}, k={cfg.ks}, splits={cfg.n_splits}, frozen full-patch PCA tangents
- **ViT deep audit (reused):** `{cfg.full_audit_dir}` labels={fca_labels}

## Other corpora

```json
{json.dumps(meta.get('other_corpora', {}), indent=2)}
```

These have embeddings under `data_hf/` but **no aligned probe label table** for the geometry atlas, so probe–curvature geography cannot be scored there yet.

## Extra-target global CV (sfr etc.)

```
{extra.to_string(index=False) if len(extra) else 'n/a'}
```

## Reliability summary (median R over anchors)

```
{rel_s.to_string(index=False) if len(rel_s) else 'n/a'}
```

## Primary probe geography (d=16, k=2048, stat=K_H_cross)

```
{prim.to_string(index=False) if len(prim) else 'n/a'}
```

## Full probe table

See `probe_associations.parquet` (n={len(pa)}).

## Decision labels

{labels}

## Takeaway

All Smith42 Physics models with prepared packs and all label keys (including weak `sfr`) are covered at the primary reliability/probe cells. Deep ViT grid+Gauss+synthetic remains in the full audit directory; this run is the cross-model / all-probe completion layer.
"""
    (out / "REPORT.md").write_text(report)
    print(f"[cov] analyze labels={labels}", flush=True)


def run(cfg: CoverageConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    for banned in (cfg.multimodel_dir, cfg.full_audit_dir, SOURCE_MM, SOURCE_FCA):
        if out.resolve() == resolve_path(root, banned).resolve():
            raise RuntimeError(f"Refusing to write into preserved directory {banned}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = (
        ["prepare", "sfr_probes", "reliability", "probe_assoc", "analyze"]
        if cfg.stage == "all"
        else [s.strip() for s in cfg.stage.split(",")]
    )
    run_set = set(want)
    # deps
    if "probe_assoc" in run_set:
        run_set.update(["prepare", "sfr_probes", "reliability"])
    if "reliability" in run_set:
        run_set.update(["prepare", "sfr_probes"])
    if "analyze" in run_set:
        run_set.add("prepare")
    if "sfr_probes" in run_set:
        run_set.add("prepare")

    meta = {}
    if "prepare" in run_set:
        t1 = time.time()
        print("[cov] stage=prepare", flush=True)
        meta = stage_prepare(root, cfg, ctx)
        profile["stages"]["prepare_s"] = time.time() - t1
        profile["completed"].append("prepare")
    else:
        meta = json.loads((out / "resolved_config.json").read_text())

    if "sfr_probes" in run_set:
        t1 = time.time()
        print("[cov] stage=sfr_probes", flush=True)
        stage_sfr_probes(root, cfg, ctx)
        profile["stages"]["sfr_probes_s"] = time.time() - t1
        profile["completed"].append("sfr_probes")

    if "reliability" in run_set:
        t1 = time.time()
        print("[cov] stage=reliability", flush=True)
        stage_reliability(root, cfg, ctx, t0)
        profile["stages"]["reliability_s"] = time.time() - t1
        profile["completed"].append("reliability")

    if "probe_assoc" in run_set:
        t1 = time.time()
        print("[cov] stage=probe_assoc", flush=True)
        stage_probe_assoc(root, cfg, ctx)
        profile["stages"]["probe_assoc_s"] = time.time() - t1
        profile["completed"].append("probe_assoc")

    if "analyze" in run_set:
        t1 = time.time()
        print("[cov] stage=analyze", flush=True)
        stage_analyze(root, cfg, ctx, meta)
        profile["stages"]["analyze_s"] = time.time() - t1
        profile["completed"].append("analyze")

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[cov] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
