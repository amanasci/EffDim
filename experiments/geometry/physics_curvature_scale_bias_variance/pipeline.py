"""Orchestrate parity → synthetic → factorial → analysis. No manuscript edits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .analysis import (
    contrast_stats,
    decide,
    drift_table,
    manuscript_action,
    reliability_table,
    summarize_cells,
)
from .config import CELLS, PRIMARY_D as PRIMARY_D, SECONDARY_DS as SECONDARY_DS, ExpConfig
from .data import load_bundle as load_bundle
from .factorial import run_factorial as run_factorial
from .figures import write_figures as write_figures
from .io_util import (
    assert_not_preserved as assert_not_preserved,
    platonic_root as platonic_root,
    resolve_path as resolve_path,
    write_df,
    write_json,
)
from .parity import run_parity as run_parity
from .synthetic import run_synthetic as run_synthetic


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "r2_k2048": "r2_k2048",
            "mse_k2048": "mse_k2048",
            "sst_k2048": "sst_k2048",
            "var_k2048": "sst_k2048_var",
            "r2_matched": "r2_matched",
            "mse_matched": "mse_matched",
            "sst_matched": "sst_matched",
        }
    )


def _peak_rss_mb() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def _flat_contrast(metric: str, kind: str, block: dict, full: dict) -> dict:
    return {
        "metric": metric,
        "kind": kind,
        "name": block["name"],
        "estimate": block["estimate"],
        "fisher_z_delta": block.get("fisher_z_delta"),
        "ci95_lo": block["ci95"][0],
        "ci95_hi": block["ci95"][1],
        "p_mc": block["p_mc"],
        "p_holm": block["p_holm"],
        "rho_R2048_m2048": full.get("rho_R2048_m2048"),
        "rho_R2048_m1024": full.get("rho_R2048_m1024"),
        "rho_R1024_m1024": full.get("rho_R1024_m1024"),
        "rho_R1536_m1024": full.get("rho_R1536_m1024"),
    }


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=True)

    if cfg.stage != "analyze":
        synth = run_synthetic(seed=cfg.seed)
        write_df(out / "synthetic_results.csv", synth, force=True)

    if cfg.stage == "synthetic":
        return {"stage": "synthetic"}

    if cfg.stage == "analyze":
        parquet = out / "replicate_curvature.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"analyze stage requires {parquet}")
        parity = json.loads((out / "parity.json").read_text()) if (out / "parity.json").exists() else {"ok": None}
        df = pd.read_parquet(parquet)
        return _analyze(cfg, out, df, parity, t0)

    bundle = load_bundle(cfg)
    parity = run_parity(bundle, cfg, out)
    if cfg.stage == "parity":
        return parity

    ds_primary = [PRIMARY_D]
    df = run_factorial(bundle, cfg, out, ds=ds_primary)
    if not cfg.skip_secondary and cfg.stage in ("all", "factorial"):
        df2 = run_factorial(bundle, cfg, out, ds=list(SECONDARY_DS))
        df = pd.concat([df, df2], ignore_index=True)
    df = _rename(df)
    write_df(out / "replicate_curvature.parquet", df, force=True)
    write_df(out / "factorial_cells.csv", pd.DataFrame([{"R": R, "m": m} for R, m in CELLS]), force=True)
    return _analyze(cfg, out, df, parity, t0)


def _analyze(cfg: ExpConfig, out: Path, df: pd.DataFrame, parity: dict, t0: float) -> dict:
    assoc = summarize_cells(df)
    write_df(out / "association_by_cell.csv", assoc, force=True)
    rel = reliability_table(df)
    write_df(out / "reliability_by_cell.csv", rel, force=True)
    drift = drift_table(out, df)
    write_df(out / "curvature_drift.csv", drift, force=True)

    n_boot = 200 if cfg.smoke else cfg.n_boot
    n_perm = 200 if cfg.smoke else cfg.n_perm
    d16 = df[df.d == PRIMARY_D]
    c_r2 = contrast_stats(d16, ycol="r2_k2048", n_boot=n_boot, n_perm=n_perm, seed=cfg.seed)
    c_mse = contrast_stats(d16, ycol="mse_k2048", n_boot=n_boot, n_perm=n_perm, seed=cfg.seed + 1)

    write_df(
        out / "primary_contrasts.csv",
        pd.DataFrame(
            [
                _flat_contrast("r2_k2048", "count", c_r2["delta_count"], c_r2),
                _flat_contrast("r2_k2048", "radius", c_r2["delta_radius"], c_r2),
                _flat_contrast("mse_k2048", "count", c_mse["delta_count"], c_mse),
                _flat_contrast("mse_k2048", "radius", c_mse["delta_radius"], c_mse),
            ]
        ),
        force=True,
    )
    decision = decide(c_r2, rel, drift)
    decision["mse_contrasts"] = c_mse
    decision["manuscript"] = manuscript_action(decision["label"])
    decision["extended_radius_run"] = False
    if cfg.extended_radius:
        decision["extended_radius_note"] = (
            "extended-radius requested but skipped: frozen kmax pack is 2048; "
            "R∈{3072,4096} would require a new ordered-neighbour pack without embedding refit. "
            "Washout beyond R=2048 remains possible but untested."
        )
    else:
        decision["extended_radius_note"] = (
            "Washout beyond R=2048 remains possible but untested; --extended-radius was not enabled."
        )
    decision["peak_rss_mb"] = _peak_rss_mb()
    write_json(out / "decision.json", decision, force=True)
    write_figures(out, assoc, rel, drift)
    _write_methods(out)
    _write_report(out, parity, decision, assoc, rel, t0, cfg)
    write_json(
        out / "COMPLETE.json",
        {"ok": True, "label": decision["label"], "seconds": time.time() - t0, "smoke": cfg.smoke, "stage": cfg.stage},
        force=True,
    )
    print(f"[sbv] done label={decision['label']} s={time.time()-t0:.1f}", flush=True)
    return decision


def _write_methods(out: Path) -> None:
    (out / "METHODS.md").write_text(
        """# METHODS

Curvature statistic: cross-split sphere-normal mean-curvature inner product
K_H_cross = <H^(A), H^(B)>. This is a split-half estimator of mean-curvature energy,
not the mean-curvature vector.

Geometric support R = first R ordered neighbours. Fit sample count m ≤ R is a
deterministic hash subset of that pool. A/B halves are an independent hash split
of the m points. Nested PCA and the frozen ridge quadratic (same λ grid, sphere-normal
projection, radial subtraction, metric whitening) are fit on those m points.

Primary probe geography is always the canonical k=2048 neighbourhood of the frozen
global OOF ridge. Matched-R probe geography is secondary and tabulated separately.

Cell associations are summarized as the median across sampling/split replicates of the
controlled Spearman. Primary contrasts Δ_count and Δ_radius are differences of those
replicate-median associations, with paired anchor bootstraps and label permutations.

R_H is split-half concordance of H, 2<H_A,H_B> / (|H_A|^2+|H_B|^2). It is not a
classical test-retest reliability coefficient for K_H_cross. Repeat Spearman of
K_H_cross across seeds is used for the attenuation diagnostic. Correlations are
never disattenuated.
"""
    )


def _write_report(out: Path, parity: dict, decision: dict, assoc, rel, t0, cfg) -> None:
    ms = decision["manuscript"]
    c = decision["contrasts"]
    (out / "REPORT.md").write_text(
        f"""# REPORT

## Mechanism label

`{decision['label']}`

## Parity

ok={parity.get('ok')} common-128 d=16 cells and frozen n=512 k=2048 ρ=-0.240 are recorded in parity.json.
n=128 estimates are not significance-tested against n=512.

## Fixed probe geography (primary)

All primary associations use `mag_r_desi_local_oof_r2` / OOF MSE / SST evaluated on the
canonical k=2048 neighbourhood while curvature varies in (R, m).

## Matched-scale probe geography (secondary)

Fields `r2_matched` / `mse_matched` in replicate_curvature.parquet and `r2match_*` /
`msematch_*` in association_by_cell.csv evaluate the same frozen OOF predictions on the
first R neighbours. They are not mixed into the primary contrast table.

## Fixed-support sample-count contrast (R=2048)

Δ_count = ρ(m=2048) − ρ(m=1024) = {c['delta_count']['estimate']:+.3f}
(Fisher-z Δ = {c['delta_count'].get('fisher_z_delta')})
95% CI {c['delta_count']['ci95']}, p_MC={c['delta_count']['p_mc']:.4g}, Holm={c['delta_count']['p_holm']:.4g}.
Cell ρs (replicate medians): m=2048 → {c.get('rho_R2048_m2048')}; m=1024 → {c.get('rho_R2048_m1024')}.
Negative Δ_count means more samples strengthen the expected negative association.

## Fixed-count radius contrast (m=1024)

Δ_radius = ρ(R=2048) − ρ(R=1024) = {c['delta_radius']['estimate']:+.3f}
(Fisher-z Δ = {c['delta_radius'].get('fisher_z_delta')})
95% CI {c['delta_radius']['ci95']}, p_MC={c['delta_radius']['p_mc']:.4g}, Holm={c['delta_radius']['p_holm']:.4g}.
Cell ρs: R=2048 → {c.get('rho_R2048_m1024')}; R=1536 → {c.get('rho_R1536_m1024')}; R=1024 → {c.get('rho_R1024_m1024')}.
Positive Δ_radius means larger support moves a negative association toward zero (washout).
Negative Δ_radius means the association strengthens over the larger support (anti-washout in range).

## Reliability

See reliability_by_cell.csv. R_H is not classical reliability of K_H_cross; repeat Spearman is.
reliability_rises_with_m={decision.get('reliability_rises_with_m')}
{decision.get('note_R_H','')}

## Direct MSE vs local R²

Primary contrasts were also computed for OOF MSE (primary_contrasts.csv). Signs should oppose R² if the association is error-driven.

## Curvature-vector drift

median cosine={decision.get('median_cross_radius_cosine')}
Low cosine plus growing outer-shell residuals would support spatial heterogeneity. Unreliable cells are not interpreted as drift.

## Synthetic validation

synthetic_results.csv: constant-curvature family should not wash out with R at fixed m; heterogeneous family should.

## Optional extension

{decision.get('extended_radius_note')}

## Manuscript recommendation (not applied)

action: `{ms['action']}`

Proposed sentence:

> {ms['sentence']}

Do not edit submissions/neurreps_2026 until author review.

## Runtime

smoke={cfg.smoke} stage={cfg.stage} wall_seconds≈{time.time()-t0:.1f}
"""
    )
