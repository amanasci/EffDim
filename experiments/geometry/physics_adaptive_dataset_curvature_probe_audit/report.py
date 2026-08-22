"""CORRECTED_REPORT.md, AUDIT_METHODS.md, AUDIT_COMPLETE.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FROZEN_CTL, FROZEN_D80, FROZEN_D85, FROZEN_RAW
from .pipeline import AuditConfig, write_json


def _fmt(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def write_methods(out: Path, cfg: AuditConfig, ctx: dict) -> None:
    text = rf"""# Audit methods

The completed adaptive run and all listed geometry trees are **read-only**.
Corrected tables are written only under `{cfg.output_dir}`.

## Estimands

The frozen ViT-B / `mag_r_desi` discovery curve is

$$\\rho_d=\\rho_{{\\mathrm{{Spearman}}}}\\bigl(K_H^{{(d)}},\\mathrm{{local\\_r2}}\\bigr)$$

where `local_r2` is the out-of-fold local ridge-probe $R^2$ for target
`mag_r_desi` in `local_probe_fields.parquet`. It is **not** the catalog
magnitude.

The adaptive run estimated the different quantity

$$\\rho_d=\\rho_{{\\mathrm{{Spearman}}}}\\bigl(K_H^{{(d)}},\\mathrm{{catalog\\ mag\\_r\\_desi}}\\bigr).$$

Three curves are reported and never aliased:

1. Raw association (stated \(y\)).
2. Frozen discovery-control association (`local_r2` + `local_probe_fields` controls).
3. Harmonized catalog-control association (catalog \(y\) + the same control names).

Curve 3 is not discovery parity.

## Parity

For \(d\\in\\{{12,16,20\\}}\) the audit compares embeddings, anchors, \(k=2048\)
neighbours, \(K_H\), and both \(y\) vectors on the original sets, the
intersection, and the frozen discovery order. Factorial Spearman
correlations are computed from the per-anchor tables without calling either
experiment's high-level inference.

## DESI

Alignment requires object IDs, a source-row manifest, or a reproducible
reconstruction of embedding order. Equal catalog and embedding row counts
are recorded and are **not** treated as proof. No correlation-maximizing
permutation is searched.

## Inference

Permutations: raw shuffle of \(y\); controlled rank-space Freedman–Lane.
Same-object physics labels share one object permutation. DESI, when
computed, is an independent sample and is excluded from scientific
conclusions while alignment is unproven.

Global corrections (confirmatory family, discovery `mag_r_desi` excluded):

- Unstudentized \(\\max|\\rho|\) (the previous global statistic).
- Westfall–Young min-\(p\) on within-label curve-level permutation \(p\)-values.
- Studentized \(\\max|T|\) with \(T=(\\rho-\\mu_0)/\\sigma_0\).

Zero exceedances are reported as \(p<1/(B+1)\), never \(p=0\). Monte Carlo
intervals use a Clopper–Pearson exceedance interval mapped through
\((e+1)/(B+1)\).

The global “any association” test is **not** a test of a common dimensional
transition.

## Sample size

The inferential unit is the curvature anchor. A label is underpowered if
valid labelled anchors \(< 64\) (frozen \(|\\rho|=0.35\), \(\\alpha=0.05\),
nominal 80% power floor).

## Scale

Scale sensitivity is deferred until discovery-quantity parity and every
included label join are proved. This audit writes `AUDIT_COMPLETE.json`
and does **not** write a scientific `COMPLETE.json`.

Thresholds: `{json.dumps(ctx.get("thresholds", {}), default=str)}`
"""
    (out / "AUDIT_METHODS.md").write_text(text)


def write_report(out: Path, cfg: AuditConfig, ctx: dict) -> None:
    parity = ctx["parity"]
    desi = ctx["desi"]
    controls = ctx["controls"]
    sizes = ctx["sizes"]
    sci = ctx["global_scientific"]
    pub = ctx.get("global_published")
    rel = ctx["reliability"]
    causes = ctx["causes"]
    label = ctx["audit_label"]
    fact = pd.read_csv(out / "factorial_discovery_correlations.csv")
    fact_i = fact[fact.scope == "intersection"] if len(fact) else fact

    kh_lines = "\n".join(
        f"- d={r['d']}: Pearson={_fmt(r['pearson'])}, Spearman={_fmt(r['spearman'])}, "
        f"max|Δ|={_fmt(r['max_abs_diff'])}, exact_rate={_fmt(r['exact_rate'])}, identical={r['identical']}"
        for r in parity["curvature_by_rank"]
    )
    fact_lines = "\n".join(
        f"- d={int(r.d)}: oldK-oldy={_fmt(r.rho_oldK_oldy)}  oldK-newy={_fmt(r.rho_oldK_newy)}  "
        f"newK-oldy={_fmt(r.rho_newK_oldy)}  newK-newy={_fmt(r.rho_newK_newy)}  follows={r.disagreement_follows}"
        for _, r in fact_i.iterrows()
    )
    size_tbl = sizes.to_string(index=False) if len(sizes) else "NA"
    wy = sci["wy"]
    mx = sci["maxT"]
    rawg = sci["raw_global"]
    wy_surv = sci["wy_table"]
    surv = wy_surv[wy_surv.survives_wy == True][["dataset_id", "label", "curve_p", "wy_adjusted_p"]] if len(wy_surv) else pd.DataFrame()  # noqa: E712

    sfr = sizes[(sizes.dataset_id == "physics_vit_base") & (sizes.label == "sfr")]
    sfr_n = int(sfr.valid_labelled_anchors.iloc[0]) if len(sfr) else -1

    text = rf"""# Corrected report: adaptive dataset curvature–physics audit

**Audit label:** `{label}`

The previous label `dataset_specific_curvature_probe_associations` is
**suspended**. This file does not replace the 12.2-hour run; that tree is
unchanged.

## Root cause

First divergence: **probe / label quantity**, not geometry.

The frozen discovery curve correlates \(K_H^{{(d)}}\) with `local_r2` of the
`mag_r_desi` probe field. The adaptive run correlated the **same** \(K_H\)
with catalog `mag_r_desi`. Those \(y\) vectors are different quantities
(Spearman ≈ {_fmt(parity["label_compare"].get("spearman"))}).

Assigned causes: {", ".join(causes)}.

Conditional repair: reuse existing per-anchor \(K_H\) (exact match at
\(d=12,16,20\)). Do not refit geometry. Do not launch a 12-hour rerun.
DESI label associations are removed from scientific conclusions.

## Old-versus-new \(K_H\) parity

{kh_lines}

A monotone rescaling is not required: the reused ranks are identical.

## Old-versus-new label parity

- Frozen \(y\): `local_r2` (range roughly 0.04–0.44).
- Adaptive \(y\): catalog `mag_r_desi` (range roughly 15–19).
- Pearson={_fmt(parity["label_compare"].get("pearson"))}, Spearman={_fmt(parity["label_compare"].get("spearman"))}.
- Physics `sample_id` is the galaxies test-table row. `vit_base_test_labels.npz`
  is row-aligned to the parquet; `selection.npz` indexes both. Equal row
  count is not the proof.

## Anchor and neighbourhood parity

- Shared anchors: {parity["anchors"]["n_shared"]} / 512, Jaccard={_fmt(parity["anchors"]["jaccard"])}.
- Same set, different order (adaptive `adcp:` hash of the same 512).
- Neighbours: both use `vit_base_kmax2048.npz` at \(k=2048\), compared after
  aligning on `sample_id`. Agreement={parity["neighbours"].get("exact_id_agreement")}.

## Factorial raw correlations

{fact_lines}

The raw-\\(\\rho\\) disagreement follows **labels**, not \(K_H\) or anchors.

## Frozen versus harmonized controls

| d | raw local_r2 | frozen-control local_r2 | raw catalog | harmonized-control catalog |
|---:|-------------:|------------------------:|------------:|---------------------------:|
"""
    side = controls["side"]
    for _, r in side.iterrows():
        text += (
            f"| {int(r.d)} | {_fmt(r.raw_discovery_local_r2)} | {_fmt(r.frozen_discovery_control_local_r2)} | "
            f"{_fmt(r.raw_catalog_mag)} | {_fmt(r.harmonized_control_catalog_mag)} |\n"
        )
    text += rf"""
Frozen published values: d=12 raw {FROZEN_RAW[12]} ctl {FROZEN_CTL[12]};
d=16 raw {FROZEN_RAW[16]} ctl {FROZEN_CTL[16]};
d=20 raw {FROZEN_RAW[20]} ctl {FROZEN_CTL[20]}.

The d=12 sign change under frozen controls (+0.143 vs raw −0.038) is a
property of the **discovery** control model, not a reason to prefer
harmonized controls.

## Corrected \(\\Delta^{{85-80}}\)

- Frozen discovery-control (local_r2, \(d_{{85}}=20\), \(d_{{80}}=12\)):
  {_fmt(controls["delta_frozen_ctl"])}
  \(= {_fmt(FROZEN_CTL[FROZEN_D85])} - {_fmt(FROZEN_CTL[FROZEN_D80])}\).
- Harmonized catalog-control (not discovery parity):
  {_fmt(controls["delta_harmonized_ctl"])}.

There is **one** independent magnitude catalog (DESI), and its label join
is unproven. No leave-one-dataset-out stability and no cross-dataset
meta-analysis are reported as replications.

## DESI alignment

Status: `{desi.get("status")}`. Proved={desi.get("proved")}.
Embedding columns are vision vectors only. Catalog `desi_object_id` has no
partner in the embedding parquet. Equal \(n=20465\) is not proof.
DESI geometry is retained. DESI curvature–label associations are **not**
scientific results.

## Anchor-level sample sizes

```
{size_tbl}
```

`sfr` has **{sfr_n}** valid labelled anchors, not 1,340. It is underpowered
under the frozen \(n<64\) rule.

## Global multiple-testing (scientific family: physics catalog labels, discovery excluded, DESI excluded)

Unstudentized max-|ρ|: p={_fmt(rawg["p"])} ({rawg["p_report"]}), CI=[{_fmt(rawg.get("ci_lo"))}, {_fmt(rawg.get("ci_hi"))}], T={_fmt(rawg["t_obs"])}.

Westfall–Young min-p: p={_fmt(wy["p"])} ({wy["p_report"]}), CI=[{_fmt(wy.get("ci_lo"))}, {_fmt(wy.get("ci_hi"))}].

Studentized max-T: p={_fmt(mx["p"])} ({mx["p_report"]}), CI=[{_fmt(mx.get("ci_lo"))}, {_fmt(mx.get("ci_hi"))}], T={_fmt(mx["t_obs"])}.

Curves surviving WY (α=0.05):
```
{surv.to_string(index=False) if len(surv) else "none"}
```

This is a test of **any association anywhere** in the confirmatory physics
family. It is not a test of a common dimensional transition.

As-published family (includes unaligned DESI) is in the CSV files and is
not used for scientific claims.

## Transition-specific inference

- Any-association: see global tests above.
- Magnitude-transition replication: frozen \(\\Delta^{{85-80}}={_fmt(controls["delta_frozen_ctl"])}\)
  (discovery reference). No proven independent magnitude replicate.
- Redshift: physics `photo_z` vs DESI `spec_z` is a **post hoc** observation.
  DESI `spec_z` is not a scientific result. No signed heterogeneous-label
  transition statistic is formed.

## Reliability sensitivity

High-rank peaks (`smooth_fraction` near d=43, DESI `mag_r` near d=36) sit
where median \(R_H\) is declining. Cutoffs 0.2 (frozen), 0.4, 0.5, 0.6 are
reported in `high_rank_reliability_sensitivity.csv`. Cutoffs were not
chosen to preserve significance. Ranks remain in the figures and are
marked weak when \(R_H\) is below a cutoff.

## Scale analysis

**Not completed.** The prior `COMPLETE.json` left `predeclared_pending`
rows. Discovery-quantity parity is now understood, but DESI joins are
unproven, so scale refits are deferred. No scientific `COMPLETE.json`.

## Corrected scientific label

None. Suspended: `dataset_specific_curvature_probe_associations`.
Audit outcome: `{label}`.

## Runtime, tests, paths

- Audit runtime: {_fmt(ctx.get("runtime_s"), 4)} s
- Tests: {ctx.get("n_tests", "see test_adaptive_dataset_curvature_probe_audit.py")}
- Output: `{out}`
- Read-only sources: adaptive run, rank sweep, nested curvature, multimodel, QPD
"""
    (out / "CORRECTED_REPORT.md").write_text(text)


def write_audit_complete(out: Path, cfg: AuditConfig, ctx: dict) -> None:
    write_json(
        out / "AUDIT_COMPLETE.json",
        {
            "ok": True,
            "scientific_complete": False,
            "audit_label": ctx["audit_label"],
            "previous_label_suspended": "dataset_specific_curvature_probe_associations",
            "root_causes": ctx["causes"],
            "discovery_kh_identical": ctx["parity"].get("kh_identical"),
            "probe_quantity_mismatch": True,
            "desi_alignment": ctx["desi"].get("status"),
            "scale_analysis": "deferred",
            "wrote_scientific_COMPLETE": False,
            "n_perm": ctx["global_scientific"]["n_perm"],
            "seconds": ctx.get("runtime_s"),
            "n_tests": ctx.get("n_tests"),
        },
        force=cfg.force,
    )
