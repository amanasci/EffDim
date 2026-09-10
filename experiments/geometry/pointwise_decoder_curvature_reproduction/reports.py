"""Phase-0 protocol docs, formula audit, reproduction report, and two figures."""

from __future__ import annotations

import json
from pathlib import Path

from geometry.pointwise_decoder_curvature_reproduction.config import (
    COLLEAGUE,
    CURVATURE_EXPERIMENTS_SHA,
    F_DIFFERENTIATED_AFTER_NORMALIZE,
    FIXTURE_VALIDITY_AUDIT_SHA,
    H_IS_AVERAGED,
    HISTORICAL,
    II_REMOVES_SPHERE_RADIAL,
    TOL,
)


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def protocol_manifest() -> dict:
    return {
        "colleague": COLLEAGUE,
        "curvature_experiments_sha": CURVATURE_EXPERIMENTS_SHA,
        "fixture_validity_audit_sha": FIXTURE_VALIDITY_AUDIT_SHA,
        "named_sources": {
            "02.6-FINDINGS": {
                "path": ".planning/phases/02.6-decoder-substrate-screening/02.6-FINDINGS.md",
                "on_curvature_experiments": True,
                "note": "150-epoch Swiss-roll decoder screening. Not the R1 300-epoch numbers.",
            },
            "Supp. 02 brief": {
                "path": "notebooks/.planning or papers: 09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md",
                "on_branch": "origin/fixture-validity-audit",
                "sha": FIXTURE_VALIDITY_AUDIT_SHA,
                "note": "R1 table 0.553 / 0.9998 / 1.046 labeled ours H_tan; swiss-roll runner scores ambient H_vec.",
            },
            "09-FIXTURE-FIDELITY-D16": {
                "path": "09-FIXTURE-FIDELITY-D16.md",
                "on_branch": "origin/fixture-validity-audit",
                "sha": FIXTURE_VALIDITY_AUDIT_SHA,
                "note": "Exact R2–R5 plain-decoder table.",
            },
            "07_plain_decoder_sweep": {
                "path": "notebooks/diagnostics/07_instrument_fixture_sweep_run.py",
                "on_branch": "origin/fixture-validity-audit",
                "default_d": 20,
                "d16_flag": "--d 16",
            },
            "07_low_spread_control": {
                "path": "notebooks/diagnostics/07_instrument_low_spread_control_run.py",
                "on_branch": "origin/fixture-validity-audit",
            },
            "Supp. 03": {
                "path": "03-08-SUPPLEMENT-03.md",
                "note": "Exists; not the cubic/ridge d=16 table.",
            },
        },
        "estimator_files": {
            "plain_decoder_curvature": "notebooks/pu_manifold/decoder_curvature.py",
            "decoder_curvature_commit_note": "Present on both branches; Austin Lutterbach authorship (577c3e3).",
            "fixtures_R2_R4": (
                "experiments/geometry/pointwise_decoder_curvature_reproduction/"
                "vendor_varying_ii_controls.py"
            ),
            "fixtures_source": "origin/fixture-validity-audit:notebooks/pu_manifold/varying_ii_controls.py",
            "vendor_change": "import rewritten from relative to `from pu_manifold import curvature_probe, synthetic_controls`",
            "training": "notebooks/pu_manifold/cae.py::train_plain_ae / PlainAutoEncoder",
            "truth_graphs": "notebooks/pu_manifold/curvature_probe.py::graph_mean_curvature",
            "truth_swiss": "notebooks/pu_manifold/decoder_curvature.py::swiss_roll_analytic_H_vector",
        },
        "fixture_generator": {
            "R1": {
                "fn": "sklearn.datasets.make_swiss_roll",
                "n": 3000,
                "noise": 0.0,
                "random_state": 0,
                "normalization": "centre, then divide by one global std (scalar)",
            },
            "R2_R5": {
                "fn": "varying_ii_controls.FAMILIES[cubic|ridge](n=5000, d=16, D, seed=20260816)",
                "sampling": "uniform box in parameter domain (cubic radius 1.5, ridge as module default)",
                "embed": "graph assembly then synthetic_controls.rotate_and_pad",
                "normalization": "rotate_and_pad's own global_std (single scalar)",
            },
        },
        "split": {
            "R1": {
                "algorithm": "np.random.default_rng(20260813).permutation(n); first round(n*0.2) holdout",
                "train_on": "train split only",
                "eval_anchors": "256 holdout rows via subsample.draw_row_indices(..., seed=20260902), then sort",
            },
            "R2_R4": {
                "train_on": "all n=5000 points",
                "eval_on": "all n=5000 points",
            },
        },
        "architecture": {
            "encoder_decoder": "cae.PlainAutoEncoder",
            "hidden": [250, 250, 250],
            "activation": "silu",
            "latent_dim": {"R1": 2, "R2_R4": 16},
            "output_normalization": "none on R1–R4 (raw decode)",
        },
        "training": {
            "loss": "mean of per-row ||x-y||^2 (sum over dim, mean over batch)",
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch": 128,
            "epochs": {"R1": 300, "R2_R4": 400},
            "early_stop": "disabled (patience > max_epochs)",
            "seeds": {
                "R1_torch_init": 0,
                "R1_train_cfg_seed": 0,
                "R2_R4_torch_init": 0,
                "R2_R4_train_cfg_seed": 20260816,
                "R2_R4_data": 20260816,
            },
            "checkpoint": "in-memory last epoch; no .pt written by the historical runners",
        },
        "autodiff": {
            "precision": "float64 required (_assert_float64)",
            "map": "model.decode only, never forward",
            "jacobian": "torch.func.vmap(jacrev) — J shape (D, d), columns are coordinate partials",
            "hessian": "torch.func.vmap(hessian) — Q shape (D, d, d)",
            "chunk": 32,
        },
        "formulas": {
            "I": "g = J^T J",
            "P_T": "J g^{-1} J^T",
            "II": "(I - P_T) D^2 F   i.e. full Euclidean second fundamental form",
            "II_removes_sphere_radial": II_REMOVES_SPHERE_RADIAL,
            "H": "g^{ab} II_ab  (UNNORMALIZED trace; no 1/d)",
            "H_is_averaged_1_over_d": H_IS_AVERAGED,
            "F_differentiated_after_normalization": F_DIFFERENTIATED_AFTER_NORMALIZE,
            "implementation_trick": "trace-first-then-project: raw = g^{jk} Hess_jk, then subtract tangent component",
        },
        "explicit_answers": {
            "reported_H_equals": "g^{ab} II_ab   (NOT (1/d) g^{ab} II_ab)",
            "II_equals": "(I - P_T) D^2 F   (NOT (I - xx^T - P_T) D^2 F)",
            "F_differentiated": "before output normalization (raw decode) on R1–R4",
        },
        "metrics": {
            "variance_explained": "1 - mean(||x-y||^2) / mean(||x||^2)  — NOT centered R^2",
            "rho": "Spearman of ||H_est|| vs ||H_true|| (scipy.stats.spearmanr .statistic)",
            "cosine": "median over points of <H_est,H_true> / (||H_est|| ||H_true||)",
            "ratio": "median over points of ||H_est|| / ||H_true||",
            "R1_scorer": (
                "09 swiss-roll runner uses synthetic_control_run._fidelity_axes "
                "(chart_curvature.curvature_fidelity_report + spearman_gate_statistic). "
                "On the Swiss roll this coincides with the 07 axes() formulas up to "
                "excluding ||H_true||<=1e-12, which does not occur."
            ),
            "true_H_spread": "not a reported 07 column; ii_variation.hess_fro_cv is the fixture II-variation diagnostic",
            "aggregation": "single seed; median over anchors/points; no multi-seed mean",
        },
        "tolerances_frozen": TOL,
        "historical_targets": HISTORICAL,
    }


def write_protocol_docs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "protocol_manifest.json", protocol_manifest())
    (out / "PROTOCOL_AUDIT.md").write_text(PROTOCOL_AUDIT)
    (out / "FORMULA_AUDIT.md").write_text(FORMULA_AUDIT)


def write_all_reports(
    out: Path,
    *,
    blocked: bool,
    tests: dict,
    env: dict,
    decision: dict | None = None,
    cells: list | None = None,
    parity: list | None = None,
    skipped: list | None = None,
    runtime_s: float | None = None,
) -> None:
    write_protocol_docs(out)
    if blocked:
        (out / "REPRODUCTION_REPORT.md").write_text(
            "# REPRODUCTION_REPORT\n\nBlocked before training. See `unit_test_results.json`.\n"
        )
        return
    decision = decision or {}
    skipped = skipped or []
    cells = cells or []
    parity = parity or []
    lines = [
        "# REPRODUCTION_REPORT",
        "",
        "Bounded reproduction of Austin Lutterbach's pointwise decoder-curvature pipeline.",
        "No manuscript edits. No local-quadratic Q estimator. No n=86471 sphere training.",
        "",
        f"- curvature-experiments SHA: `{CURVATURE_EXPERIMENTS_SHA}`",
        f"- fixture-validity-audit SHA (tables + fixtures): `{FIXTURE_VALIDITY_AUDIT_SHA}`",
        f"- primary label: `{decision.get('primary_reproduction_label')}`",
        f"- estimand label: `{decision.get('secondary_estimand_label')}`",
        f"- radial label: `{decision.get('radial_baseline_label')}`",
        f"- runtime_s: {runtime_s}",
        f"- skipped: {skipped}",
        f"- unit tests: {tests.get('n_passed')}/{tests.get('n_tests')}",
        "",
        "## Recovered formula",
        "",
        "Reported H is the unnormalized metric trace `H = g^{ab} II_ab` of the full Euclidean",
        "second fundamental form `II = (I - P_T) D²F`, differentiated through raw `model.decode`.",
        "The factor `1/d` is **not** used. Sphere-radial projection is **not** applied on R1–R4.",
        "",
        "## Cell parity",
        "",
    ]
    for row in parity:
        lines.append(
            f"- {row['cell']} {row['fixture']} d={row['d']} D={row['D']}: "
            f"ΔVE={row.get('abs_diff_var_explained')} "
            f"Δρ={row.get('abs_diff_rho')} "
            f"Δcos={row.get('abs_diff_cosine')} "
            f"Δratio={row.get('abs_diff_ratio')} "
            f"pass={row.get('pass_all_available')}"
        )
    lines += [
        "",
        "## Radial baseline",
        "",
        "Not applicable: none of R1–R4 constrain the decoder image to the unit sphere.",
        "Cached n=86471 D=768 sphere weights were not found.",
        "",
        "## What this does not claim",
        "",
        "Numerical parity does not mean the learned decoder recovered the data Hessian.",
        "High reconstruction R² does not validate second derivatives.",
        "This run does not adjudicate the finite-patch quadratic estimator.",
        "",
    ]
    (out / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n")
    _maybe_figures(out, parity)


def _maybe_figures(out: Path, parity: list) -> None:
    if not parity:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    cells = [r for r in parity if r.get("cell") in {"R1", "R2", "R3", "R4"}]
    if not cells:
        return
    metrics = [
        ("rho", "historical_rho", "new_rho"),
        ("cosine", "historical_cosine", "new_cosine"),
        ("ratio", "historical_ratio", "new_ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    names = [r["cell"] for r in cells]
    x = np.arange(len(names))
    for ax, (title, hkey, nkey) in zip(axes, metrics):
        hist = [r.get(hkey) for r in cells]
        new = [r.get(nkey) for r in cells]
        ax.plot(x, hist, "o-", label="historical")
        ax.plot(x, new, "s--", label="reproduced")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_historical_vs_reproduced.png", dpi=120)
    plt.close(fig)

    # Figure 2: radial baseline is N/A; write a one-panel note rather than a fake comparison.
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.text(
        0.5,
        0.5,
        "Radial-only vs full vs residual: not applicable.\n"
        "R1–R4 decoder outputs are not constrained to the unit sphere.\n"
        "No cached n=86471 sphere weights were evaluated.",
        ha="center",
        va="center",
        wrap=True,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "fig2_radial_baseline_not_applicable.png", dpi=120)
    plt.close(fig)


PROTOCOL_AUDIT = f"""# PROTOCOL_AUDIT — pointwise decoder-curvature reproduction

Read-only recovery. Colleague whose results are reproduced: **Austin Lutterbach**
(`rmhslutterbach@gmail.com`), author of `notebooks/pu_manifold/decoder_curvature.py`
and of the d=16 cubic/ridge table on `origin/fixture-validity-audit`.

The current worktree is on `curvature-experiments` at `{CURVATURE_EXPERIMENTS_SHA}`.
That SHA equals `refs/heads/curvature-experiments` and
`refs/remotes/origin/curvature-experiments`. It was inspected in place (already
checked out). The d=16 table and `varying_ii_controls.py` live on
`origin/fixture-validity-audit` at `{FIXTURE_VALIDITY_AUDIT_SHA}` and are **not**
ancestors of HEAD. They were read with `git show` / vendored; that branch was
not checked out, merged, or rebased.

Angus's nested-chart `K_H^cross` / local-quadratic estimator on this branch is a
**different** instrument and was not run.

## Named sources

| Name | Where recovered | Role |
| --- | --- | --- |
| 02.6-FINDINGS | `.planning/phases/02.6-decoder-substrate-screening/02.6-FINDINGS.md` on HEAD | Early Swiss-roll decoder screen (150 epochs). **Not** the R1 numbers. |
| Supp. 02 brief | `09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md` on fixture-validity-audit | Quotes R1 0.553 / 0.9998 / 1.046 as `ours H_tan`. |
| 09-FIXTURE-FIDELITY-D16 | same branch, `09-FIXTURE-FIDELITY-D16.md` | Exact R2–R5 plain-decoder cells. |
| 07_plain_decoder_sweep | `notebooks/diagnostics/07_instrument_fixture_sweep_run.py` | Implementation of R2–R5. Default `--d 20`; d=16 is `--d 16`. |
| 07_low_spread_control | `notebooks/diagnostics/07_instrument_low_spread_control_run.py` | Control arm; not a primary cell. |
| Supp. 03 | `03-08-SUPPLEMENT-03.md` | Present; not the cubic/ridge table. |

Hardcoded path in the 07 runner (`/home/akagi/Documents/Projects/EffDim/notebooks`)
is Austin's machine. Compatibility fix: this package inserts **this** repo's
`notebooks/` on `sys.path`. No other colleague code was edited.

## Fixture generators

**R1 Swiss roll.** `sklearn.datasets.make_swiss_roll(n_samples=3000, noise=0.0,
random_state=0)`. Centre by the coordinate-wise mean, divide by one global
standard deviation (a single scalar). Latent dimension 2.

**R2–R4 (optional R5).** `varying_ii_controls.FAMILIES["cubic"|"ridge"](5000, 16,
D, 20260816)`. Parameter samples are uniform in a box (cubic `domain_radius=1.5`).
The graph `(x, f(x))` is assembled, then `synthetic_controls.rotate_and_pad`
applies a seeded rotation into ambient `D` and a single global std. Analytic
`H_vec` is `curvature_probe.graph_mean_curvature` (same unnormalized trace)
rotated with the cloud.

Sampling distribution for cubic/ridge is the fixture generator's uniform
parameter box, not a Gaussian on the sphere.

## Train / evaluation split

- R1: `split_indices(n, seed=20260813, holdout_fraction=0.2)` — permutation,
  first `round(n*0.2)` holdout, remainder train. Train the AE on the train
  split only. Evaluate curvature at 256 holdout anchors drawn by
  `subsample.draw_row_indices(len(holdout), 256, seed=20260902)` then sorted.
- R2–R4: **no split**. Train and evaluate on all 5000 points.

## Normalization

- Input: as above (global std). No per-feature standardization after that.
- Decoder output: **raw** `model.decode`. No sphere projection on R1–R4.
  `SphereProjectedDecoder` is the Amendment-01 Physics path and is not used
  by `07_instrument_fixture_sweep_run` or the swiss-roll ambient arm.

## Architecture, loss, optimizer

`cae.PlainAutoEncoder(in_dim=D, latent_dim=d, hidden=(250,250,250),
activation="silu")`. SiLU is C², required by `assert_c2_decoder`.

Reconstruction loss inside `cae._train_decoder_protocol`: batch mean of the
per-row sum of squares `||x-y||^2`. Optimizer AdamW, `lr=1e-3`,
`weight_decay=1e-4`, `batch=128`. Early stopping disabled
(`early_stop_patience > max_epochs`, `min_delta=1e-9`).

- R1: 300 epochs. `torch.manual_seed(0)` at construction. Train cfg seed
  defaults to 0 (pcp.TRAIN_CFG has no `seed` key).
- R2–R4: 400 epochs. `torch.manual_seed(0)` at construction, then
  `cfg["seed"]=20260816` reseeds the training shuffle.

Checkpoint: last in-memory state. Historical runners wrote JSONL metrics,
not `.pt` weights. No frozen-weight selection rule.

## Autodiff conventions

`plain_decoder_curvature` requires float64. Derivatives use
`torch.func.vmap(jacrev)` and `vmap(hessian)` in chunks of 32.

Jacobian convention: `J.shape = (D, d)` with `J[:, a] = ∂_a F`.
Hessian convention: `Q.shape = (D, d, d)` with `Q[:, a, b] = ∂_a ∂_b F`.
First fundamental form: `g = J^T J`. Inverse via `torch.linalg.solve`.

The map differentiated is **raw `model.decode`**, never `forward` (which would
compose the encoder) and never a sphere-normalized wrap on these cells.

## Second fundamental form and mean curvature

Documented and implemented as:

```
J    = DF
g    = J^T J
P_N  = I - J g^{{-1}} J^T
II   = P_N D²F
H    = tr_g(II) = g^{{jk}} II_jk
```

Implementation is trace-first-then-project (`raw = g^{{jk}} Hess_jk`, then
subtract the tangent component). Algebraically identical to projecting each
Hessian slice and then tracing.

**Is reported H equal to `g^{{ab}} II_ab` or `(1/d) g^{{ab}} II_ab`?**
`g^{{ab}} II_ab`. `CURVATURE_CONVENTION = "trace"`. A unit d-sphere has
`||H|| = d`, not 1.

**Is II equal to `(I-P_T) D²F` or `(I-xx^T-P_T) D²F`?**
`(I-P_T) D²F`. The normal projector removes the tangent space only.

**Is F differentiated before or after output normalization?**
Before, on R1–R4.

## Metric definitions (traced, not inferred from column names)

- **variance explained** = `1 - mse_total / mean(||X||^2)` where `mse_total`
  is the mean over rows of `||x-y||^2` from `cae.reconstruction_stats`.
  This is **not** centered variance R².
- **ρ** = Spearman correlation of `||H_est||` with `||H_true||`
  (`scipy.stats.spearmanr(...).statistic`).
- **cosine** = median over evaluation points of the pointwise cosine of the
  ambient vectors `H_est` and `H_true`.
- **ratio** = median of `||H_est|| / ||H_true||`.
- **true ||H|| spread** is not a 07 output column. The fixture diagnostic
  `ii_variation.hess_fro_cv` is the CV of `||D²f||_F` across points.

R1 Supp. 02 labels the row `ours H_tan`. The swiss-roll runner stores
`instrument: ours_H_ambient` and scores `H_vec` against
`swiss_roll_analytic_H_vector`. The label in the brief is a naming
discrepancy; this reproduction preserves the **runner** definition.

## Analytic truth

- Swiss roll: plane-curve curvature vector of the Archimedean spiral in the
  x–z plane, y-component zero, scaled by `global_std`.
- Cubic/ridge: `graph_mean_curvature` on the closed-form grad/hess, then the
  same `rotate_and_pad` as the cloud.

## Aggregation

Single frozen seed per cell. No multi-seed mean. Medians over the evaluation
set (256 anchors for R1; 5000 points for R2–R4).

## Cached sphere fixtures

Searched this worktree for `*86471*.pt` and `notebooks/.cache/**/*.pt`.
None found. Missing sphere checkpoints are not a failure of the four-cell
reproduction. Label: `cached_sphere_evaluation_skipped_missing_weights`.

## Blockers

Estimator, truth, and metric code were recovered. This audit does **not**
assign `reproduction_blocked_by_missing_definition_or_code`.
"""


FORMULA_AUDIT = f"""# FORMULA_AUDIT

Source of record: `notebooks/pu_manifold/decoder_curvature.py::plain_decoder_curvature`
on `{CURVATURE_EXPERIMENTS_SHA}`, with `CURVATURE_CONVENTION = "trace"`.

## Reported H

```
H = g^{{ab}} II_ab
```

Not `(1/d) g^{{ab}} II_ab`. `H_IS_AVERAGED = {H_IS_AVERAGED}`.

The implementation traces first (`raw = g^{{jk}} Hess_jk`) then removes the
tangent component of that ambient vector. That equals the g-trace of
`II = (I-P_T) D²F`.

## Reported II

```
II = (I - P_T) D²F
P_T = J g^{{-1}} J^T
g = J^T J
```

Not `(I - xx^T - P_T) D²F`. `II_REMOVES_SPHERE_RADIAL = {II_REMOVES_SPHERE_RADIAL}`.

## Differentiated map

Raw `model.decode`. `F_DIFFERENTIATED_AFTER_NORMALIZE = {F_DIFFERENTIATED_AFTER_NORMALIZE}`.

On a unit-sphere immersion the Euclidean identity `II^E = -g ⊗ x + B^S` still
holds mathematically, but the historical estimator reports `H^E` (full
Euclidean), not `H^S`. R1–R4 are not sphere-constrained, so the radial
decomposition is not applied to those cells.

## Secondary estimand

`reported_quantity_is_full_euclidean_curvature`

## Propositions kept separate

1. Full `II^E` is the Euclidean curvature of the decoder image.
2. `B^S` is the second fundamental form of that image inside the unit sphere.
3. A large full-curvature score may be dominated by a normalization sphere.
4. A small residual-energy fraction does not make residual curvature unreal.
5. Numerical reproduction does not establish recovery of the data Hessian.
6. High reconstruction R² does not validate second derivatives.
7. This experiment does not adjudicate the finite-patch quadratic estimator.
"""
