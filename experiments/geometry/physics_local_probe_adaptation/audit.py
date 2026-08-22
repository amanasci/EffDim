"""Write ACCIDENTAL_RESULT_AUDIT.md from known historical screen protocol."""

from __future__ import annotations

from pathlib import Path

from .config import SOURCE_SCREEN
from .io_util import platonic_root, resolve_path


AUDIT_TEXT = """# ACCIDENTAL_RESULT_AUDIT

Status of the historical number: **`historical_invalid_or_unverified`** for scientific
inference under the present protocol. The number may be reproducible as a screen
artifact but does **not** pass strict global-fold OOF patch evaluation.

## Source

Implementation: `experiments/geometry/physics_activation_atlas/curvature_probe_screen.py`

Artifacts (read-only): `{screen}/` (`SCREENING_REPORT.md`, `primary_correlations.csv`,
`probe_metrics.parquet`).

Reported association (k=2048, sphere-normal screening curvature × local Ridge R²):

* raw Spearman ≈ **−0.55**
* partial (radius / label-variance controls) ≈ **+0.220**

That **partial positive** is the accidental observation motivating this experiment.

## Protocol of the accidental fit

| Item | Historical screen |
|------|-------------------|
| Features | Ambient ViT-B L2 embeddings (768-D); **StandardScaler** on X and y within the local train split |
| Neighbourhood | k ∈ {{512,1024,2048}}; anchors ≈ 384 (screen), not the frozen 512 |
| Train/test | Within-neighbourhood shuffle: ~70% train / 30% eval (`eval_frac=0.3`) |
| Fold assignment | **Not** the frozen global five-fold OOF; local random split per anchor |
| Test in train? | Eval slice excluded from that local fit; anchor excluded from neighbour list |
| Ridge α | Grid including 100; selected on ≤48 probe-only anchors (**mild HP leakage** relative to full screen) |
| Loss | sklearn `Ridge` = sum of squares + α‖w‖² (after scaling) |
| R² SST | Local held-out SST (`sklearn.r2_score`) |
| Association outcome | Local **probe R²** (performance), not raw catalog magnitude as the y-variable |
| Curvature | Earlier sphere-normal atlas features (not necessarily identical to frozen K_H^cross panel) |

## Why it is not used for inference here

1. Local random splits are **not** the frozen global OOF fold assignment.
2. StandardScaler + α selection differ from the frozen global probe convention (α=100,
   unscaled ambient features, sum-of-squares).
3. A positive *partial* correlation after controlling radius can reflect residual
   confounding rather than genuine local direction adaptation.
4. In-sample or weakly held-out local fits can be optimistically biased.

## Reproduction policy

This experiment may recompute a screen-style in-sample / local-split score **only** as
an explicitly invalid comparison column. Scientific claims use strict global-fold OOF
patch probes (models G/I/C/P/T) on the frozen 512 anchors, k=2048, d=16, and frozen
`K_H_cross`.

## Frozen objects used instead

* Global five-fold OOF ridge (α=100) predictions
* `sample_folds.parquet` fold IDs
* `per_anchor_rank_curve.parquet` `K_H_cross` at d=16
* Neighbour packs `vit_base_kmax2048.npz`
* Controls: log kNN radius, local target variance, local evaluation count
"""


def write_audit(out: Path) -> None:
    root = platonic_root()
    screen = resolve_path(root, SOURCE_SCREEN)
    text = AUDIT_TEXT.format(screen=str(screen))
    (out / "ACCIDENTAL_RESULT_AUDIT.md").write_text(text)
