"""Split-half curvature reliability audit with fixed PCA tangent.

Protocol:
  1) Parity gate: reproduce all-512 d=16 k=2048 K_mean↔OOF local_r2 cell
     (raw ≈ -0.38, partial_C0 ≈ -0.15) via the same _fit_neighborhood path.
  2) Freeze full-patch PCA tangent J per anchor; split neighbours; refit B^S only.
  3) Report H^S / B° / B^S stability, held-out Δ_S, and per-split probe correlations.

Does not refit kNN, PCA (after freeze), SAE, or global probes.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .confirmatory_object_curvature import _fit_neighborhood
from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import partial_spearman, spearman_dict
from .multimodel_graph_prior_quadratic import EPS, load_model_X
from .paths import platonic_root, resolve_path
from .sphere_normal_quadratic import sphere_project_basis
from .tangent_reliability import fit_nested_fixed_tangent, pca_tangent

SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
# Parity targets from d_replication_check_all512
PARITY_RAW = -0.380543
PARITY_PARTIAL_C0 = -0.148145
PARITY_TOL = 0.03  # absolute tolerance on rho


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


@dataclass
class SplitHalfConfig:
    output_dir: str = "outputs/geometry/physics_split_half_curvature_reliability"
    multimodel_dir: str = SOURCE_MM
    model: str = "vit_base"
    target: str = "mag_r_desi"
    d: int = 16
    k: int = 2048
    n_anchors_smoke: int = 128
    n_splits_smoke: int = 5
    n_anchors_full: int = 512
    n_splits_full: int = 10
    seed: int = 0
    force: bool = False
    stage: str = "all"  # parity,smoke,full,analyze,all
    max_seconds: float = 7200.0
    # same ridge grid as fit_nested_chart
    ridges: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0]
    )
    sensitivity_ks: list[int] = field(default_factory=list)
    sensitivity_d: list[int] = field(default_factory=list)

    def resolved(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


def load_context(root: Path, cfg: SplitHalfConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    all512 = mm / "d_replication_check_all512" / "anchor_ids.json"
    if all512.exists():
        use_sids = json.loads(all512.read_text())["sample_ids"]
    else:
        use_sids = [int(s) for s in anchors_sid]
    pack = dict(np.load(mm / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"))
    X = load_model_X(mm, cfg.model)
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == cfg.model)
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
        & (geo.scale_k == cfg.k)
    ]
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    return {
        "mm": mm,
        "X": X,
        "pack": pack,
        "geo": geo,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": sid_to_ai,
        "anchors_local": anchors_local,
        "anchors_sid": anchors_sid,
    }


def full_patch_pca_tangent(Xloc: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA tangent on the full neighbourhood (frozen for all splits)."""
    x0 = Xloc.mean(0)
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J, _, _ = pca_tangent(Xloc, x0, d)
    J = sphere_project_basis(x0, J)
    return x0, J


def _half_fit_indices(half: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Within a half: 80% fit / 20% val for ridge selection."""
    rng = np.random.default_rng(seed)
    h = half.copy()
    rng.shuffle(h)
    n_fit = max(20, int(0.8 * len(h)))
    if n_fit >= len(h) - 8:
        n_fit = max(20, len(h) - 8)
    return h[:n_fit], h[n_fit:]


def tensor_agreement(HA: np.ndarray, HB: np.ndarray) -> dict[str, float]:
    na = float(np.linalg.norm(HA))
    nb = float(np.linalg.norm(HB))
    inp = float(np.dot(HA.ravel(), HB.ravel()))
    r_dir = inp / max(na * nb, EPS)
    R = (2.0 * inp) / max(na**2 + nb**2, EPS)
    return {
        "norm_A": na,
        "norm_B": nb,
        "inner": inp,  # K_cross^2 for vectors / Frobenius for tensors
        "r_dir": r_dir,
        "R_signal": R,
    }


def BS_objects(BS_flat: np.ndarray, d: int) -> dict[str, np.ndarray]:
    B0, H = traceless_B0(BS_flat, d)
    B0f = B0_flat_for_svd(B0, d)
    return {"H": H, "B0_flat": B0f, "BS_flat": BS_flat}


# -------------------- parity --------------------


def stage_parity(root: Path, cfg: SplitHalfConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity_gate.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    X, pack, use_sids, sid_to_ai = ctx["X"], ctx["pack"], ctx["use_sids"], ctx["sid_to_ai"]
    geo = ctx["geo"]
    rows = []
    t0 = time.time()
    for sid in use_sids:
        ai = sid_to_ai[int(sid)]
        N = pack["neigh"][ai, : cfg.k]
        rho = float(pack["dists"][ai, cfg.k - 1])
        chart, _, info, _, _, reason = _fit_neighborhood(
            X, N, cfg.d, seed=cfg.seed + ai + 17 * cfg.k + cfg.d
        )
        if chart is None:
            continue
        _, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
        rows.append(
            {
                "sample_id": int(sid),
                "K_mean": float(np.linalg.norm(H)),
                "recon_error": float(info.get("val_E_TRS", np.nan)),
                "knn_radius": rho,
                "log_knn_radius": float(np.log(max(rho, EPS))),
                "fit_reason": reason,
            }
        )
        if len(rows) % 64 == 0:
            print(f"[split][parity] {len(rows)}/{len(use_sids)}", flush=True)
    feats = pd.DataFrame(rows)
    feats.to_parquet(out / "parity_full_fit_features.parquet", index=False)
    # Prefer feature-side radius/recon (same as d_replication_check merge)
    g = geo.merge(feats, on="sample_id", how="inner", suffixes=("_geo", ""))
    Km = g.K_mean.to_numpy(float)
    y = g.local_r2.to_numpy(float)
    log_r = g["log_knn_radius"].to_numpy(float)
    C0 = np.column_stack(
        [
            log_r,
            g.local_label_variance.to_numpy(float),
            g.recon_error.to_numpy(float),
            g.local_evaluation_count.to_numpy(float),
        ]
    )
    raw = spearman_dict(Km, y)
    p0 = partial_spearman(Km, y, C0)
    ok = bool(
        abs(raw["rho"] - PARITY_RAW) <= PARITY_TOL
        and abs(p0["rho"] - PARITY_PARTIAL_C0) <= PARITY_TOL
        and int(len(g)) >= 500
    )
    result = {
        "ok": ok,
        "n": int(len(g)),
        "raw_rho_K_mean": raw["rho"],
        "raw_p": raw["pvalue"],
        "partial_C0": p0["rho"],
        "partial_C0_p": p0["pvalue"],
        "expected_raw": PARITY_RAW,
        "expected_partial_C0": PARITY_PARTIAL_C0,
        "tol": PARITY_TOL,
        "seconds": time.time() - t0,
        "protocol": "same as run_kmean_d_replication_check --anchor-mode all d=16 k=2048",
    }
    path.write_text(json.dumps(result, indent=2))
    print(
        f"[split][parity] n={result['n']} raw={result['raw_rho_K_mean']:.4f} "
        f"partial_C0={result['partial_C0']:.4f} ok={ok}",
        flush=True,
    )
    if not ok:
        print(
            "[split][parity] PROTOCOL DRIFT — stopping before split-half. "
            f"Expected raw≈{PARITY_RAW}, partial_C0≈{PARITY_PARTIAL_C0}",
            flush=True,
        )
    return result


# -------------------- split-half core --------------------


def run_split_half_audit(
    root: Path,
    cfg: SplitHalfConfig,
    ctx: dict,
    *,
    n_anchors: int,
    n_splits: int,
    tag: str,
) -> pd.DataFrame:
    out = cfg.resolved(root)
    path = out / f"split_half_{tag}.parquet"
    if _done(path, cfg.force):
        return pd.read_parquet(path)

    parity_p = out / "parity_gate.json"
    if parity_p.exists() and not json.loads(parity_p.read_text()).get("ok", False):
        raise RuntimeError("Parity gate failed — refuse to run split-half")

    X, pack = ctx["X"], ctx["pack"]
    use_sids = ctx["use_sids"][:n_anchors]
    sid_to_ai = ctx["sid_to_ai"]
    geo = ctx["geo"].set_index("sample_id")
    rows = []
    t0 = time.time()
    for si, sid in enumerate(use_sids):
        if si % 16 == 0:
            print(f"[split][{tag}] anchor {si}/{len(use_sids)}", flush=True)
        ai = sid_to_ai[int(sid)]
        N = pack["neigh"][ai, : cfg.k]
        Xloc = X[N].astype(np.float64)
        x0, J = full_patch_pca_tangent(Xloc, cfg.d)
        if J.shape[1] < cfg.d:
            continue
        if int(sid) not in geo.index:
            continue
        local_r2 = float(geo.loc[int(sid), "local_r2"])
        log_r = float(geo.loc[int(sid), "log_knn_radius"])
        lab_var = float(geo.loc[int(sid), "local_label_variance"])
        eval_n = float(geo.loc[int(sid), "local_evaluation_count"])
        for s in range(n_splits):
            if time.time() - t0 > cfg.max_seconds:
                print("[split] time budget", flush=True)
                break
            rng = np.random.default_rng(cfg.seed + 1009 * ai + 17 * s)
            perm = rng.permutation(cfg.k)
            halfA, halfB = perm[: cfg.k // 2], perm[cfg.k // 2 :]
            fitA, valA = _half_fit_indices(halfA, cfg.seed + 3 + s)
            fitB, valB = _half_fit_indices(halfB, cfg.seed + 7 + s)
            # Fit A → hold out B; Fit B → hold out A
            chA, _, infoA = fit_nested_fixed_tangent(
                Xloc, x0, J, fitA, valA, halfB, ridges=cfg.ridges
            )
            chB, _, infoB = fit_nested_fixed_tangent(
                Xloc, x0, J, fitB, valB, halfA, ridges=cfg.ridges
            )
            if chA is None or chB is None:
                continue
            objA = BS_objects(chA.BS_flat, cfg.d)
            objB = BS_objects(chB.BS_flat, cfg.d)
            agH = tensor_agreement(objA["H"], objB["H"])
            agB0 = tensor_agreement(objA["B0_flat"], objB["B0_flat"])
            agBS = tensor_agreement(objA["BS_flat"], objB["BS_flat"])
            rows.append(
                {
                    "sample_id": int(sid),
                    "split": s,
                    "tag": tag,
                    "d": cfg.d,
                    "k": cfg.k,
                    "local_r2": local_r2,
                    "log_knn_radius": log_r,
                    "local_label_variance": lab_var,
                    "local_evaluation_count": eval_n,
                    "norm_HA": agH["norm_A"],
                    "norm_HB": agH["norm_B"],
                    "r_H_dir": agH["r_dir"],
                    "K_H_cross": agH["inner"],
                    "R_H": agH["R_signal"],
                    "norm_B0A": agB0["norm_A"],
                    "norm_B0B": agB0["norm_B"],
                    "r_B0_dir": agB0["r_dir"],
                    "K_B0_cross": agB0["inner"],
                    "R_B0": agB0["R_signal"],
                    "norm_BSA": agBS["norm_A"],
                    "norm_BSB": agBS["norm_B"],
                    "r_BS_dir": agBS["r_dir"],
                    "K_BS_cross": agBS["inner"],
                    "R_BS": agBS["R_signal"],
                    "dS_A": float(infoA.get("dS", np.nan)),
                    "dS_B": float(infoB.get("dS", np.nan)),
                    "E_TR_A": float(infoA.get("E_TR", np.nan)),
                    "E_TRS_A": float(infoA.get("E_TRS", np.nan)),
                    "E_TR_B": float(infoB.get("E_TR", np.nan)),
                    "E_TRS_B": float(infoB.get("E_TRS", np.nan)),
                    "recon_A": float(infoA.get("recon_error", np.nan)),
                    "recon_B": float(infoB.get("recon_error", np.nan)),
                }
            )
        if time.time() - t0 > cfg.max_seconds:
            break
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[split][{tag}] wrote n_rows={len(df)}", flush=True)
    return df


def summarize_splits(df: pd.DataFrame, tag: str) -> dict[str, Any]:
    if df.empty:
        return {"tag": tag, "empty": True}
    # within-anchor medians then across anchors
    per = df.groupby("sample_id").agg(
        r_H_dir=("r_H_dir", "median"),
        R_H=("R_H", "median"),
        r_B0_dir=("r_B0_dir", "median"),
        R_B0=("R_B0", "median"),
        r_BS_dir=("r_BS_dir", "median"),
        R_BS=("R_BS", "median"),
        dS_A=("dS_A", "median"),
        dS_B=("dS_B", "median"),
        norm_HA=("norm_HA", "median"),
        norm_HB=("norm_HB", "median"),
        K_H_cross=("K_H_cross", "median"),
    )
    # cross-split Spearman of magnitudes (pool A vs B medians)
    sp_H = spearman_dict(per.norm_HA.to_numpy(), per.norm_HB.to_numpy())
    # predictive Δ_S stability
    dS = 0.5 * (df.dS_A + df.dS_B)
    summary = {
        "tag": tag,
        "n_anchors": int(df.sample_id.nunique()),
        "n_splits": int(df.split.nunique()),
        "n_rows": int(len(df)),
        "median_r_H_dir": float(per.r_H_dir.median()),
        "median_R_H": float(per.R_H.median()),
        "median_r_B0_dir": float(per.r_B0_dir.median()),
        "median_R_B0": float(per.R_B0.median()),
        "median_r_BS_dir": float(per.r_BS_dir.median()),
        "median_R_BS": float(per.R_BS.median()),
        "spearman_normH_A_vs_B": sp_H["rho"],
        "median_dS": float(np.nanmedian(dS)),
        "frac_dS_positive": float(np.mean(dS > 0)),
        "dS_iqr": float(np.nanpercentile(dS, 75) - np.nanpercentile(dS, 25)),
    }
    # interpretation heuristics
    dS_ok = summary["frac_dS_positive"] > 0.7 and summary["median_dS"] > 0
    B0_ok = summary["median_R_B0"] > 0.3 and summary["median_r_B0_dir"] > 0.3
    H_ok = summary["median_R_H"] > 0.3 and summary["median_r_H_dir"] > 0.3
    BS_ok = summary["median_R_BS"] > 0.3
    if dS_ok and B0_ok and not H_ok:
        label = "dS_and_B0_reliable_H_weak"
    elif dS_ok and not BS_ok:
        label = "dS_stable_BS_unidentifiable"
    elif H_ok and B0_ok and dS_ok:
        label = "everything_stable_instability_elsewhere"
    elif not dS_ok and not BS_ok:
        label = "everything_unstable"
    elif B0_ok and not H_ok:
        label = "B0_stable_H_unstable"
    else:
        label = "mixed_partial_reliability"
    summary["reliability_label"] = label
    return summary


def probe_by_split(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per-split Corr(||H_A||, local_r2) and ||H_B|| — raw then radius-only partial."""
    rows = []
    for s, g in df.groupby("split"):
        # aggregate to anchors (one row per anchor in this split)
        for side, ncol in (("A", "norm_HA"), ("B", "norm_HB")):
            x = g[ncol].to_numpy(float)
            y = g.local_r2.to_numpy(float)
            raw = spearman_dict(x, y)
            C = g.log_knn_radius.to_numpy(float)[:, None]
            pr = partial_spearman(x, y, C)
            rows.append(
                {
                    "tag": tag,
                    "split": int(s),
                    "side": side,
                    "n": raw["n"],
                    "raw_rho": raw["rho"],
                    "raw_p": raw["pvalue"],
                    "radius_only_rho": pr["rho"],
                    "radius_only_p": pr["pvalue"],
                }
            )
    return pd.DataFrame(rows)


def summarize_probe(probe: pd.DataFrame) -> dict:
    if probe.empty:
        return {}
    raw = probe.raw_rho.to_numpy(float)
    return {
        "median_raw_rho": float(np.nanmedian(raw)),
        "min_raw_rho": float(np.nanmin(raw)),
        "max_raw_rho": float(np.nanmax(raw)),
        "frac_negative_raw": float(np.mean(raw < 0)),
        "sign_recurrent": bool(np.mean(raw < 0) >= 0.8 or np.mean(raw > 0) >= 0.8),
        "median_radius_only": float(np.nanmedian(probe.radius_only_rho)),
        "min_radius_only": float(np.nanmin(probe.radius_only_rho)),
        "max_radius_only": float(np.nanmax(probe.radius_only_rho)),
        "frac_negative_radius_only": float(np.mean(probe.radius_only_rho < 0)),
    }


def stage_analyze(root: Path, cfg: SplitHalfConfig) -> None:
    out = cfg.resolved(root)
    parity = json.loads((out / "parity_gate.json").read_text()) if (out / "parity_gate.json").exists() else {}
    summaries = []
    probe_summaries = []
    for tag in ("smoke", "full"):
        p = out / f"split_half_{tag}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        sm = summarize_splits(df, tag)
        summaries.append(sm)
        (out / f"summary_{tag}.json").write_text(json.dumps(sm, indent=2))
        pr = probe_by_split(df, tag)
        pr.to_parquet(out / f"probe_by_split_{tag}.parquet", index=False)
        ps = summarize_probe(pr)
        ps["tag"] = tag
        probe_summaries.append(ps)
        (out / f"probe_summary_{tag}.json").write_text(json.dumps(ps, indent=2))
        # plot
        if len(df):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df.groupby("sample_id").r_H_dir.median(), bins=30, alpha=0.7, label="r_H")
            ax.hist(df.groupby("sample_id").r_B0_dir.median(), bins=30, alpha=0.7, label="r_B0")
            ax.legend()
            ax.set_title(f"Directional agreement ({tag})")
            fig.tight_layout()
            fig.savefig(out / "figures" / f"r_dir_{tag}.png", dpi=140)
            plt.close(fig)

    report = f"""# Split-half curvature reliability audit

## Parity gate (full-fit, same protocol as d_replication all512)

```json
{json.dumps(parity, indent=2)}
```

Gate {'PASSED' if parity.get('ok') else 'FAILED / missing'}.

## Split-half (fixed full-patch PCA tangent J)

Because J is frozen, disagreement is quadratic estimation noise — not tangent estimation.

### Summaries

```json
{json.dumps(summaries, indent=2)}
```

### Probe correlations by split (raw first, radius-only second)

```json
{json.dumps(probe_summaries, indent=2)}
```

If the sign of ρ changes across quadratic splits, there is **no reliable curvature–probe result**
regardless of nominal p-values on any single fit.

## Strongest defensible reading

Look at `reliability_label` in the summaries. The historically expected pattern is
`dS_and_B0_reliable_H_weak`: quadratic predictive gain is real, traceless bending is
moderately stable, mean curvature is a noisy trace cancellation.
"""
    (out / "REPORT.md").write_text(report)
    print(f"[split] analyze done labels={[s.get('reliability_label') for s in summaries]}", flush=True)


def run(cfg: SplitHalfConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "resolved_config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "config_hash": hashlib.sha256(
                    json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
                ).hexdigest()[:16],
            },
            indent=2,
            default=str,
        )
    )
    ctx = load_context(root, cfg)
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}
    stages_wanted = (
        ["parity", "smoke", "full", "analyze"]
        if cfg.stage == "all"
        else [s.strip() for s in cfg.stage.split(",")]
    )

    if "parity" in stages_wanted or cfg.stage == "all":
        t1 = time.time()
        print("[split] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        profile["stages"]["parity_s"] = time.time() - t1
        if not parity.get("ok"):
            (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
            (out / "REPORT.md").write_text(
                "# Split-half audit\n\n**STOPPED: parity gate failed (protocol drift).**\n\n"
                + json.dumps(parity, indent=2)
            )
            return {"parity": parity, "stopped": True}

    if "smoke" in stages_wanted or cfg.stage == "all":
        t1 = time.time()
        print("[split] stage=smoke", flush=True)
        run_split_half_audit(
            root,
            cfg,
            ctx,
            n_anchors=cfg.n_anchors_smoke,
            n_splits=cfg.n_splits_smoke,
            tag="smoke",
        )
        profile["stages"]["smoke_s"] = time.time() - t1

    # auto-expand to full if smoke exists and stage asks
    if "full" in stages_wanted or cfg.stage == "all":
        t1 = time.time()
        print("[split] stage=full", flush=True)
        run_split_half_audit(
            root,
            cfg,
            ctx,
            n_anchors=cfg.n_anchors_full,
            n_splits=cfg.n_splits_full,
            tag="full",
        )
        profile["stages"]["full_s"] = time.time() - t1

    if "analyze" in stages_wanted or cfg.stage == "all":
        t1 = time.time()
        print("[split] stage=analyze", flush=True)
        stage_analyze(root, cfg)
        profile["stages"]["analyze_s"] = time.time() - t1

    profile.update({"total_seconds": time.time() - t0, "peak_rss_mb": _rss()})
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[split] done in {profile['total_seconds']:.1f}s", flush=True)
    return profile
