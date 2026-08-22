"""Write validation report, claims, reproducibility, and the methods file copy."""

from __future__ import annotations

import json
from pathlib import Path

from .config import FROZEN_CTL, FROZEN_RAW, N_BOOT, N_PERM
from .pipeline import ValConfig, file_sha_full, write_json
from .schema import PRIMARY


METHODS_TEXT = r"""# METHODS_FOR_PAPER

Extracted from the implementation used for the frozen ViT-B / physics analysis. Not reconstructed from memory.

## 1. Local coordinates and centring

File: `experiments/geometry/physics_activation_atlas/full_curvature_audit.py` (`pca_tangent_gpu`, `fit_nested_fixed_tangent_gpu`).

Embeddings are row-ℓ₂ unit-normalised in the multimodel prepare stage (`load_model_X`). At an anchor \(x_0\), the neighbourhood \(X_{\mathrm{loc}}\) is sphere-centred:

- \(x_0 \leftarrow x_0/\|x_0\|\)
- \(\Delta X \leftarrow (X_{\mathrm{loc}}-x_0) - ((X_{\mathrm{loc}}-x_0)x_0)\,x_0^\top\) (radial component removed before PCA)
- nested tangent basis \(J\) from PCA/SVD of \(\Delta X\), then `sphere_project_basis`: \(J \leftarrow \mathrm{orth}((I-x_0 x_0^\top)J)\).

Local coordinates: \(U = (X-x_0)J\). Coordinate RMS whitening: \(s_a = \sqrt{\mathrm{mean}(U_{\mathrm{fit},a}^2)}\) (`fit_nested_fixed_tangent_gpu`).

Identity test: after `sphere_project_basis`, \(x_0^\top J \approx 0\) and \(J^\top J \approx I\).

## 2. Nested coordinate basis

File: `experiments/geometry/physics_activation_atlas/nested_dimension_curvature.py` (`nested_pca_frame`, `_fit_rank`).

A single nested PCA frame \(J_{:,1:d_{\max}}\) is computed; rank-\(d\) charts use \(J_{:,1:d}\). Split-half curvature uses three random splits of the \(k\)-neighbourhood; each half is itself split 80/20 for ridge selection (`_half_fit_indices`).

## 3. Quadratic-map convention and factors of 1/2

File: `experiments/geometry/physics_activation_atlas/quadratic.py` (`quadratic_features`); packing in `confirmatory_object_curvature.py` (`unpack_BS_symmetric`).

Features are \(\phi_{ab}=u_a u_b\) for \(a\le b\) (no \(\sqrt{2}\) and no \(1/2\) in the feature map). Off-diagonal packed coefficients store \(2B_{ab}\), so unpacking divides off-diagonals by 2. The decode residual is \(B_{\mathrm{flat}}\phi\), which equals \(\sum_a B_{aa}u_a^2 + 2\sum_{a<b}B_{ab}u_a u_b\).

QPD uses a different, metric-aware map `phi2` with off-diagonal \(\sqrt{2}\,u_a u_b\) (`physics_quadratic_predictive_dimension/algebra.py`). Curvature \(K_H\) does **not** use `phi2`.

## 4. Regularisation and fitting

File: `full_curvature_audit.py` `fit_nested_fixed_tangent_gpu`.

Two-stage ridge on quadratic features \(\Phi\): (i) tangential warp \(A\) selected by validation MSE after row-ℓ₂ decode; (ii) sphere-normal \(B^S\) on residuals after the warp, again λ-selected on validation. Decode is row-ℓ₂ renormalised. Forced-sphere radial residual is formed by scaling targets by \(\|L\|\) before the tangential/normal residual (radius normalisation).

## 5. Metric whitening

Coordinate scales \(s_a\) as above. Downstream SVD analyses rescale \(B^S\) columns by \(s_a s_b\) (`sphere_normal_quadratic.py` `fit_config_nested`). Production \(K_H\) uses packed \(B^S\) after the sphere-normal projection, without a further ambient metric.

## 6. Tangent and sphere-normal projectors

File: `sphere_normal_quadratic.py`.

- `sphere_project_basis(x0,J)`: \(J\leftarrow\mathrm{orth}((I-x_0 x_0^\top)J)\).
- `normal_projector_apply(V,x0,J)`: \(P_{N,S}V = V - QQ^\top V\) where \(Q=\mathrm{qr}[x_0\,J]\).
- After fitting, \(B^S \leftarrow B^S - Q(Q^\top B^S)\) so columns of \(B^S\) lie in \(\mathrm{im}\,P_{N,S}\).

Identities: \(P_{N,S}^2=P_{N,S}\), \(P_{N,S}x_0=0\), \(P_{N,S}J=0\).

## 7. Subtraction of the forced sphere-radial component

In `fit_nested_fixed_tangent_gpu`, linear decode \(L = x_0 + U J^\top\) is used as the sphere-radial carrier. Residuals for \(B^S\) are projected orthogonal to \(\mathrm{span}(x_0,J)\) before the ridge solve, so the fitted second fundamental form is sphere-normal rather than ambient.

## 8. \(B_S\), \(H_S\), \(K_H\), radius normalisation

Files: `effdim_curvature_metrics.py` (`decompose_tensors`, `cross_metric_pair`); production scalar in `_fit_rank`.

Unpack \(B^S\) to a symmetric \(d\times d\) matrix per ambient coordinate. Mean curvature vector \(H_a = d^{-1}\sum_i B^S_{a,ii}\). Split-half production statistic:

\[
K_H^{\mathrm{cross}} = \langle H^{(A)}, H^{(B)}\rangle
\]

(`cross_metric_pair`: `khx = dot(H_A, H_B)`). Reliability \(R_H = 2\langle H_A,H_B\rangle / (\|H_A\|^2+\|H_B\|^2)\) (`tensor_agreement`). Radius normalisation is the row-ℓ₂ decode above, not a separate scalar \(r\) in the \(K_H\) formula.

Note: `metric_scalars()["K_H"]` is \(\|H\|\), used in identity tests. The frozen discovery curve uses \(K_H^{\mathrm{cross}}\), not \(\|H\|\).

## 9. OOF probe construction

File: `multimodel_graph_prior_quadratic.py` `stage_global_probes`.

Five-fold ridge (\(\alpha=100\)) from embeddings to labels. Fold \(f\) is predicted from weights fit on all other folds. Artifact: `global_probes/oof_predictions/vit_base_mag_r_desi.npz` key `oof`.

## 10. Exact local \(R^2\)

File: `global_probe_curvature_alignment.py` `local_r2_fixed_predictions` / `weighted_r2`.

On neighbours with finite \(y\) and \(\hat y\), uniform weights \(w_i=1/n\):

\[
R^2_{\mathrm{local}} = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.
\]

This is `mag_r_desi_local_oof_r2`. Catalog `mag_r_desi` is `mag_r_desi_catalog_value` and is never substituted.

## 11. Controls, bootstrap, permutation

File: `physics_curvature_probe_rank_sweep/inference.py`.

Controls: `log_knn_radius`, `local_label_variance`, `local_evaluation_count`. Partial Spearman on ranks. Permutation: \(B=10^4\); raw shuffles \(y\); controlled uses rank-space Freedman–Lane. FWER from the max-\(|\rho|\) envelope across ranks. Bootstrap: \(B=2000\) paired anchor resamples; simultaneous bands from the 95th percentile of \(\max_d|\hat\rho_d^\ast-\hat\rho_d|\). Never report \(p=0\); use \(p<1/(B+1)\).

## 12. Dimensionality-range construction

Held-out linear/quadratic NMSE and pooled \(R^2_L\) from QPD (`aggregate_risk_curves.csv`, \(k=2048\)). Variance crossings: \(\tau=0.80\to d=12\), \(\tau=0.85\to d=20\) (post hoc). Predeclared chart positions for scale mapping: lower \(d=12\), middle \(d=16\), upper \(d=20\). These are geometry-only and are not chosen by maximising correlation. Rank analysis over \(d=8,\ldots,20\) is exploratory / not preregistered. Linear charts continue to improve beyond this family (\(d_L^{\mathrm{plat}}=115\) in the adaptive interval); the paper reports a plausible finite-scale range, not an intrinsic dimension.
"""


def write_reports(root, cfg: ValConfig, parity, probe, scale, decision) -> None:
    out = cfg.resolved(root)
    (out / "METHODS_FOR_PAPER.md").write_text(METHODS_TEXT)
    claims = "\n".join([
        "# CLAIMS",
        "",
        f"Decision label: `{decision['label']}`",
        "",
        "Permitted: rank-conditioned sphere-normal mean curvature is associated with local OOF linear-probe performance across a geometrically adequate dimensional range — only if the label is `submission_claim_supported` or, with a scale caveat, `claim_supported_but_scale_dependent`.",
        "",
        "Forbidden: exact intrinsic dimension; causality; intrinsic Riemannian curvature of an unknown manifold; cross-dataset generality; independent replication; DESI label associations.",
        "",
        f"Primary target: `{PRIMARY.value}` (never `mag_r_desi_catalog_value`).",
        "",
        f"Frozen controlled ρ: d=12 {FROZEN_CTL[12]:+.3f}; d=16 {FROZEN_CTL[16]:+.3f}; d=20 {FROZEN_CTL[20]:+.3f}.",
        f"Frozen raw ρ at d=16: {FROZEN_RAW[16]:+.3f}.",
        f"Reasons: {decision.get('reasons')}",
        "",
    ])
    (out / "CLAIMS.md").write_text(claims)
    repro = "\n".join([
        "# REPRODUCIBILITY",
        "",
        "```",
        "cd ~/platonic-universe && source .venv/bin/activate && PYTHONPATH=experiments \\",
        "  python experiments/geometry/run_curvature_probe_submission_validation.py",
        "```",
        "",
        f"n_perm={N_PERM}, n_boot={N_BOOT}, seed=0, k=2048.",
        "No geometry refit. Reads frozen QPD, rank-sweep, multimodel, and nested-curvature artifacts.",
        "Smoke: add `--smoke`.",
        "",
    ])
    (out / "REPRODUCIBILITY.md").write_text(repro)
    (out / "VALIDATION_REPORT.md").write_text(
        "# VALIDATION_REPORT\n\n"
        + json.dumps({"parity": parity, "probe": {k: probe[k] for k in probe if k != "shuffle"}, "scale": scale, "decision": decision}, indent=2, default=str)
        + "\n"
    )
    write_json(
        out / "ARTIFACT_MANIFEST.json",
        {
            "output_dir": str(out),
            "parity": str(out / "parity_report.json"),
            "decision": str(out / "decision.json"),
            "figures": [str(out / "figures" / "figure1_dimension.pdf"), str(out / "figures" / "figure2_curvature_probe.pdf")],
            "hashes": {
                "rank_curve": file_sha_full(root / "outputs/geometry/physics_curvature_probe_rank_sweep/per_anchor_rank_curve.parquet"),
            },
        },
        force=True,
    )
