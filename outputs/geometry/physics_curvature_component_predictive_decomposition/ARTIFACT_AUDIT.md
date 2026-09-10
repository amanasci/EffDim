# Artifact audit

Coordinate convention: orthonormal PCA chart; Euclidean Frobenius is metric-correct.

- `cmcla_manifest`: exists=True sha16=25db01da9fea4cbd path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/common_anchor_manifest.json`
- `qlca_risks`: exists=True sha16=1ca838b13e974077 path=`/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment/anchor_risks.csv`
- `qlca_primary`: exists=True sha16=b421381d6304d739 path=`/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment/primary_inference.json`
- `qlca_align`: exists=True sha16=717571ac8d65ab43 path=`/home/angus/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment/alignment_summary.json`
- `ndc_example`: exists=True sha16=76f6c46fa722ae24 path=`/home/angus/platonic-universe/outputs/geometry/physics_nested_dimension_curvature/H_vectors/0.npz`
- `fcr_vit_base`: exists=True sha16=a4ecc11a5da672e2 path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/vit_base_per_anchor_curvature.parquet`
- `cmcla_vit_base`: exists=True sha16=140c1ae0b7ce7c76 path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/probes/vit_base_anchor_metrics.parquet`
- `fcr_dinov3`: exists=True sha16=085d60d1efc6383e path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/dinov3_per_anchor_curvature.parquet`
- `cmcla_dinov3`: exists=True sha16=24d86fd723d6027d path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/probes/dinov3_anchor_metrics.parquet`
- `fcr_clip_base`: exists=True sha16=d2703ef0a405522c path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/clip_base_per_anchor_curvature.parquet`
- `cmcla_clip_base`: exists=True sha16=bac8f7fd9cdca106 path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/probes/clip_base_anchor_metrics.parquet`
- `fcr_convnext_base`: exists=True sha16=98885721bcf99ed5 path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/convnext_base_per_anchor_curvature.parquet`
- `cmcla_convnext_base`: exists=True sha16=ce99c0b4a3bba70a path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/probes/convnext_base_anchor_metrics.parquet`
- `fcr_vit_large`: exists=True sha16=b788f753e20a58ce path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_full_curvature_reconciliation/tables/vit_large_per_anchor_curvature.parquet`
- `cmcla_vit_large`: exists=True sha16=0edade6af0d19097 path=`/home/angus/platonic-universe/outputs/geometry/physics_cross_model_curvature_local_adaptation/probes/vit_large_anchor_metrics.parquet`

Parity ok: True
FCR aggregate: {'rho_Kdir_R2G': -0.024605928348267934, 'rho_Kdir_R2P': -0.06047751728064453, 'rho_Kdir_Dadapt': -0.07318655156727435}
QLCA: {'median_delta_Q': 0.02058161760162215, 'rho_KH_delta_Q': 0.111248619551161, 'A_B_median': 2.4271836244410787, 'all_stable': True}

All artifacts aligned by `sample_id`.
