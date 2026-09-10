# Methods — Q geometry-resampling stability

This audit resamples the observations used to estimate the frozen ViT-B finite-patch statistic K_H^cross, then recomputes the scientific associations with frozen probe outcomes and frozen primary controls. It is not an ordinary anchor bootstrap: those treat the curvature field as fixed.

Production estimator: nested_dimension_curvature._fit_rank → fit_quad (RIDGES [1e-4 … 3]) → unpacked cross_metric_pair. Negative cross-products are not clamped. Frozen frames (x0, J) and d=16 are reused. No decoder, probe, or label-model refit.

Scheme A keeps the frozen k=2048 neighbour IDs and repartitions them into disjoint 1024/1024 halves with a hash of (experiment seed, replicate, sample_id).

Scheme B draws one shared 80% inclusion mask over the embedding table, recomputes k'=1638 neighbours, and splits into 819/819. Query anchors remain queries; they are excluded from their own neighbour set; other anchors follow the mask.

Primary associations use the frozen controlled-rank residualization (log kNN radius, local label variance, evaluation count). Replicate-specific radius is a marked sensitivity only.

Pilot then adaptive replicate count. Runtime: {'runtime_s': 1678.8757109642029, 'wall_s': 2700.0, 'pilot_n': 4, 'selected_n_per_scheme': 32, 'completed_A': 32, 'completed_B': 32, 'median_t_A_s': 26.978554129600525, 'median_t_B_s': 19.51921844482422, 'proj_32_s': 1504.4214339256287, 'proj_64_s': 2992.3501563072205, 'projection_rule': '64 if total<40min else 32 else max fitting under 40min', 'rss_mb': 1790.41015625, 'stages': ['disk', 'reuse', 'parity', 'pilot', 'scheme_A', 'scheme_B'], 'skipped': [], 'n_ae': 0, 'figures': ['/home/angus/platonic-universe/outputs/geometry/physics_q_geometry_resampling_stability/fig01_association_distributions.png', '/home/angus/platonic-universe/outputs/geometry/physics_q_geometry_resampling_stability/fig02_reliability_vs_overlap.png', '/home/angus/platonic-universe/outputs/geometry/physics_q_geometry_resampling_stability/fig03_uncertainty_intervals.png']}.
