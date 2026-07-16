//! Pure Rust compute core — preprocess + spectral metrics + shared k-NN (Phase 4).

pub mod knn;
pub mod metrics;
pub mod preprocess;

use ndarray::Array2;

use metrics::{
    geometric_mean_eff_dimensionality, participation_ratio, pca_explained_variance,
    renyi_eff_dimensionality, shannon_entropy,
};
use preprocess::{ensure_centered, singular_values_exact};

/// Identity over an `f64` slice — keeps the path dependency live from the PyO3 stub.
pub fn identity_f64_slice(xs: &[f64]) -> Vec<f64> {
    xs.to_vec()
}

/// Spectral-only result bundle (D-01, D-02 — no geometry fields).
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralResults {
    pub pca_explained_variance_95: u32,
    pub participation_ratio: f64,
    pub shannon_entropy: f64,
    pub renyi_eff_dimensionality_alpha_2: f64,
    pub renyi_eff_dimensionality_alpha_3: f64,
    pub renyi_eff_dimensionality_alpha_4: f64,
    pub renyi_eff_dimensionality_alpha_5: f64,
    pub geometric_mean_eff_dimensionality: f64,
}

/// Error from spectral orchestration (SVD failure).
#[derive(Debug)]
pub enum SpectralError {
    Svd(faer::linalg::svd::SvdError),
    Renyi(String),
}

impl From<faer::linalg::svd::SvdError> for SpectralError {
    fn from(e: faer::linalg::svd::SvdError) -> Self {
        SpectralError::Svd(e)
    }
}

impl std::fmt::Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectralError::Svd(e) => write!(f, "SVD failed: {e:?}"),
            SpectralError::Renyi(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SpectralError {}

/// Center → exact SVD → eigenvalues `s²/(n-1)` → probabilities → eight spectral keys.
///
/// Geometry keys are intentionally omitted (D-02). Geo-mean uses probabilities (api.py fidelity).
pub fn compute_spectral(data: &Array2<f64>) -> Result<SpectralResults, SpectralError> {
    let centered = ensure_centered(data.clone(), 1e-5);
    let n_samples = centered.nrows();
    let s = singular_values_exact(&centered)?;

    let denom = (n_samples.saturating_sub(1)) as f64;
    let eigenvalues: Vec<f64> = s.iter().map(|&si| (si * si) / denom).collect();

    let total_variance: f64 = eigenvalues.iter().sum();
    let probabilities: Vec<f64> = if total_variance == 0.0 {
        vec![0.0; eigenvalues.len()]
    } else {
        eigenvalues.iter().map(|&e| e / total_variance).collect()
    };

    Ok(SpectralResults {
        pca_explained_variance_95: pca_explained_variance(&eigenvalues, 0.95),
        participation_ratio: participation_ratio(&eigenvalues),
        shannon_entropy: shannon_entropy(&probabilities),
        renyi_eff_dimensionality_alpha_2: renyi_eff_dimensionality(&probabilities, 2.0)
            .map_err(SpectralError::Renyi)?,
        renyi_eff_dimensionality_alpha_3: renyi_eff_dimensionality(&probabilities, 3.0)
            .map_err(SpectralError::Renyi)?,
        renyi_eff_dimensionality_alpha_4: renyi_eff_dimensionality(&probabilities, 4.0)
            .map_err(SpectralError::Renyi)?,
        renyi_eff_dimensionality_alpha_5: renyi_eff_dimensionality(&probabilities, 5.0)
            .map_err(SpectralError::Renyi)?,
        geometric_mean_eff_dimensionality: geometric_mean_eff_dimensionality(&probabilities),
    })
}

#[cfg(test)]
mod compute_spectral_inventory_b {
    //! Inventory B: spectral-only mirrors of compute_dim pytest SETUPs (D-14).
    //! Geometry keys (MLE/TwoNN/…) are intentionally never asserted (D-02).

    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array2, Axis};

    /// Deterministic standard-normal fills for SETUP parity (not bit-identical to NumPy).
    fn seeded_randn(seed: u64, nrows: usize, ncols: usize) -> Array2<f64> {
        let mut state = seed.max(1);
        let mut next_u64 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            state
        };
        let mut next_unit = || (next_u64() as f64) / (u64::MAX as f64);
        let mut out = Array2::<f64>::zeros((nrows, ncols));
        let mut i = 0;
        let total = nrows * ncols;
        while i < total {
            let u1 = next_unit().max(1e-12);
            let u2 = next_unit();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let z0 = r * theta.cos();
            let z1 = r * theta.sin();
            out.as_slice_mut().unwrap()[i] = z0;
            i += 1;
            if i < total {
                out.as_slice_mut().unwrap()[i] = z1;
                i += 1;
            }
        }
        out
    }

    fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        a.dot(b)
    }

    fn assert_spectral_finite_nonneg(r: &SpectralResults) {
        assert!(r.pca_explained_variance_95 >= 1 || r.pca_explained_variance_95 == 0);
        for v in [
            r.participation_ratio,
            r.shannon_entropy,
            r.renyi_eff_dimensionality_alpha_2,
            r.renyi_eff_dimensionality_alpha_3,
            r.renyi_eff_dimensionality_alpha_4,
            r.renyi_eff_dimensionality_alpha_5,
            r.geometric_mean_eff_dimensionality,
        ] {
            assert!(v.is_finite() && v >= 0.0, "non-finite or negative spectral value: {v}");
        }
    }

    // --- tests/test_api.py ---

    /// SETUP: test_api::test_compute_dim_small_data — seed=42, shape=(100, 10)
    #[test]
    fn test_compute_dim_small_data_spectral_keys() {
        let data = seeded_randn(42, 100, 10);
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio.is_finite());
        assert!(r.shannon_entropy.is_finite());
        assert!(r.geometric_mean_eff_dimensionality.is_finite());
        assert!(r.renyi_eff_dimensionality_alpha_2.is_finite());
        assert!(r.renyi_eff_dimensionality_alpha_3.is_finite());
        assert!(r.renyi_eff_dimensionality_alpha_4.is_finite());
        assert!(r.renyi_eff_dimensionality_alpha_5.is_finite());
        assert!(r.pca_explained_variance_95 >= 1);
    }

    /// SETUP: test_api::test_compute_dim_list_input — five chunks (10,5), seed=42
    #[test]
    fn test_compute_dim_list_input_spectral() {
        let mut chunks = Vec::new();
        let mut seed = 42u64;
        for _ in 0..5 {
            chunks.push(seeded_randn(seed, 10, 5));
            seed = seed.wrapping_add(1);
        }
        let data = ndarray::concatenate(
            Axis(0),
            &chunks.iter().map(|c| c.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio > 0.0);
    }

    /// SETUP: test_api::test_compute_dim_centered — seed=42, shape=(50,5), +100 mean
    #[test]
    fn test_compute_dim_centered_pr_positive() {
        let data = seeded_randn(42, 50, 5) + 100.0;
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio > 0.0);
    }

    // --- tests/test_public_api.py ---

    /// SETUP: TestRenyiDimensionalitiesInComputeDim::test_renyi_keys_alpha_2_through_5
    /// seed=0, shape=(50, 5)
    #[test]
    fn test_renyi_keys_alpha_2_through_5() {
        let data = seeded_randn(0, 50, 5);
        let r = compute_spectral(&data).unwrap();
        for v in [
            r.renyi_eff_dimensionality_alpha_2,
            r.renyi_eff_dimensionality_alpha_3,
            r.renyi_eff_dimensionality_alpha_4,
            r.renyi_eff_dimensionality_alpha_5,
        ] {
            assert!(v.is_finite() && v > 0.0, "renyi={v}");
        }
    }

    /// SETUP: TestRenyiDimensionalitiesInComputeDim::test_renyi_ordering_in_compute_dim
    /// seed=0, shape=(100, 5), scales [5,3,2,1,0.5]
    #[test]
    fn test_renyi_ordering_in_compute_dim() {
        let scales = Array2::from_shape_vec(
            (1, 5),
            vec![5.0, 3.0, 2.0, 1.0, 0.5],
        )
        .unwrap();
        let data = seeded_randn(0, 100, 5) * &scales;
        let r = compute_spectral(&data).unwrap();
        let values = [
            r.renyi_eff_dimensionality_alpha_2,
            r.renyi_eff_dimensionality_alpha_3,
            r.renyi_eff_dimensionality_alpha_4,
            r.renyi_eff_dimensionality_alpha_5,
        ];
        for i in 0..values.len() - 1 {
            assert!(
                values[i] >= values[i + 1] - 1e-6,
                "Rényi not non-increasing: {} then {}",
                values[i],
                values[i + 1]
            );
        }
    }

    /// SETUP: TestRenyiDimensionalitiesInComputeDim::test_renyi_alpha_2_matches_participation_ratio
    /// seed=0, shape=(50, 5)
    #[test]
    fn test_renyi_alpha_2_matches_participation_ratio() {
        let data = seeded_randn(0, 50, 5);
        let r = compute_spectral(&data).unwrap();
        assert_relative_eq!(
            r.renyi_eff_dimensionality_alpha_2,
            r.participation_ratio,
            max_relative = 1e-6
        );
    }

    /// SETUP: TestReproducibility::test_same_input_same_output — seed=7, shape=(80, 6)
    #[test]
    fn test_same_input_same_spectral_output() {
        let data = seeded_randn(7, 80, 6);
        let r1 = compute_spectral(&data).unwrap();
        let r2 = compute_spectral(&data).unwrap();
        assert_eq!(r1, r2);
    }

    // --- tests/test_input_validation.py (compute_dim cases) ---

    /// SETUP: TestComputeDimInputValidation::test_list_of_arrays_input — three (20,5), seed=0
    #[test]
    fn test_list_of_arrays_input_spectral() {
        let c0 = seeded_randn(0, 20, 5);
        let c1 = seeded_randn(1, 20, 5);
        let c2 = seeded_randn(2, 20, 5);
        let data = ndarray::concatenate(Axis(0), &[c0.view(), c1.view(), c2.view()]).unwrap();
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio > 0.0);
    }

    /// SETUP: TestComputeDimInputValidation::test_result_contains_all_expected_keys
    /// seed=0, shape=(50, 5) — spectral subset only
    #[test]
    fn test_spectral_key_inventory_present() {
        let data = seeded_randn(0, 50, 5);
        let r = compute_spectral(&data).unwrap();
        assert_spectral_finite_nonneg(&r);
        assert!(r.pca_explained_variance_95 >= 1);
        assert!(r.participation_ratio > 0.0);
        assert!(r.shannon_entropy > 0.0);
        assert!(r.geometric_mean_eff_dimensionality > 0.0);
    }

    /// SETUP: TestComputeDimInputValidation::test_single_feature_column — seed=0, shape=(50, 1)
    #[test]
    fn test_single_feature_column() {
        let data = seeded_randn(0, 50, 1);
        let r = compute_spectral(&data).unwrap();
        assert_eq!(r.pca_explained_variance_95, 1);
        assert!(r.participation_ratio > 0.0);
    }

    /// SETUP: TestComputeDimInputValidation::test_square_matrix — seed=0, shape=(20, 20)
    #[test]
    fn test_square_matrix() {
        let data = seeded_randn(0, 20, 20);
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio > 0.0);
    }

    /// SETUP: TestComputeDimInputValidation::test_uncentered_data_handled — seed=42, +1000
    #[test]
    fn test_uncentered_data_handled_spectral() {
        let data = seeded_randn(42, 50, 5) + 1000.0;
        let r = compute_spectral(&data).unwrap();
        assert_spectral_finite_nonneg(&r);
    }

    // --- tests/test_numerical_stability.py::TestComputeDimIntegration ---

    /// SETUP: TestComputeDimIntegration::test_low_rank_data — seed=42, (100,50) rank-5
    #[test]
    fn test_low_rank_data() {
        let n = 100usize;
        let p = 50usize;
        let k = 5usize;
        let a = seeded_randn(42, n, k);
        let b = seeded_randn(43, p, k);
        let data = matmul(&a, &b.t().to_owned());
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.participation_ratio < (k as f64) + 2.0,
            "PR={}",
            r.participation_ratio
        );
        assert!(
            r.shannon_entropy < (k as f64) + 2.0,
            "Shannon={}",
            r.shannon_entropy
        );
        assert!(
            r.pca_explained_variance_95 <= (k as u32) + 1,
            "PCA={}",
            r.pca_explained_variance_95
        );
    }

    /// SETUP: TestComputeDimIntegration::test_noisy_low_rank_data — seed=42, noise=0.1
    #[test]
    fn test_noisy_low_rank_data() {
        let n = 100usize;
        let p = 50usize;
        let k = 5usize;
        let a = seeded_randn(42, n, k);
        let b = seeded_randn(43, p, k);
        let noise = seeded_randn(44, n, p) * 0.1;
        let data = matmul(&a, &b.t().to_owned()) + noise;
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.participation_ratio < (k as f64) + 5.0,
            "PR={}",
            r.participation_ratio
        );
    }

    /// SETUP: TestComputeDimIntegration::test_isotropic_gaussian — seed=42, shape=(100, 10)
    #[test]
    fn test_isotropic_gaussian() {
        let p = 10usize;
        let data = seeded_randn(42, 100, p);
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.participation_ratio > 0.7 * (p as f64),
            "PR={}",
            r.participation_ratio
        );
        assert!(
            r.shannon_entropy > 0.7 * (p as f64),
            "Shannon={}",
            r.shannon_entropy
        );
    }

    /// SETUP: TestComputeDimIntegration::test_all_results_finite — seed=42, shape=(50, 10)
    #[test]
    fn test_all_spectral_results_finite() {
        let data = seeded_randn(42, 50, 10);
        let r = compute_spectral(&data).unwrap();
        assert_spectral_finite_nonneg(&r);
    }

    /// SETUP: TestComputeDimIntegration::test_centered_vs_uncentered — seed=42, shift=+100
    #[test]
    fn test_centered_vs_uncentered_spectral() {
        let data_centered = seeded_randn(42, 50, 10);
        let data_shifted = &data_centered + 100.0;
        let r0 = compute_spectral(&data_centered).unwrap();
        let r1 = compute_spectral(&data_shifted).unwrap();
        assert_eq!(r0.pca_explained_variance_95, r1.pca_explained_variance_95);
        assert_relative_eq!(
            r0.participation_ratio,
            r1.participation_ratio,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            r0.shannon_entropy,
            r1.shannon_entropy,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            r0.renyi_eff_dimensionality_alpha_2,
            r1.renyi_eff_dimensionality_alpha_2,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            r0.geometric_mean_eff_dimensionality,
            r1.geometric_mean_eff_dimensionality,
            max_relative = 1e-10
        );
    }

    /// SETUP: TestNumericalStability::test_very_large_data_values — seed=42, scale=1e6
    #[test]
    fn test_very_large_data_values_spectral() {
        let data = seeded_randn(42, 50, 10) * 1e6;
        let r = compute_spectral(&data).unwrap();
        assert_spectral_finite_nonneg(&r);
    }

    /// SETUP: TestNumericalStability::test_very_small_data_values — seed=42, scale=1e-6
    #[test]
    fn test_very_small_data_values_spectral() {
        let data = seeded_randn(42, 50, 10) * 1e-6;
        let r = compute_spectral(&data).unwrap();
        assert_spectral_finite_nonneg(&r);
    }

    // --- tests/test_known_dimensionalities.py (spectral asserts only) ---

    /// SETUP: TestKnownDimensionalities::test_random_noise_10d — seed=42, shape=(500, 10)
    #[test]
    fn test_random_noise_10d_pr() {
        let data = seeded_randn(42, 500, 10);
        let r = compute_spectral(&data).unwrap();
        assert!(r.participation_ratio > 7.0, "PR={}", r.participation_ratio);
    }

    /// SETUP: TestKnownDimensionalities::test_swiss_roll_2d_manifold
    /// Embedded float64 from sklearn.datasets.make_swiss_roll(n=1000, noise=0.01, random_state=42)
    #[test]
    fn test_swiss_roll_2d_manifold_pca() {
        const BYTES: &[u8] =
            include_bytes!("fixtures/swiss_roll_n1000_noise001_rs42.f64bin");
        assert_eq!(BYTES.len(), 1000 * 3 * 8);
        let mut vals = Vec::with_capacity(3000);
        for chunk in BYTES.chunks_exact(8) {
            vals.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let data = Array2::from_shape_vec((1000, 3), vals).unwrap();
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.pca_explained_variance_95 >= 2,
            "PCA={}",
            r.pca_explained_variance_95
        );
    }

    /// SETUP: TestKnownDimensionalities::test_linear_subspace_rank3_in_10d
    /// seed=42, (500,3)@(3,10) + 1e-6 noise
    #[test]
    fn test_linear_subspace_rank3_in_10d() {
        let a = seeded_randn(42, 500, 3);
        let b = seeded_randn(43, 3, 10);
        let noise = seeded_randn(44, 500, 10) * 1e-6;
        let data = matmul(&a, &b) + noise;
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.pca_explained_variance_95 <= 4,
            "PCA={}",
            r.pca_explained_variance_95
        );
        assert!(r.participation_ratio < 5.0, "PR={}", r.participation_ratio);
    }

    /// SETUP: TestKnownDimensionalities::test_2d_plane_in_5d — seed=42, (400,2)@(2,5)
    #[test]
    fn test_2d_plane_in_5d_pca() {
        let coords = seeded_randn(42, 400, 2);
        let emb = seeded_randn(43, 2, 5);
        let noise = seeded_randn(44, 400, 5) * 1e-6;
        let data = matmul(&coords, &emb) + noise;
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.pca_explained_variance_95 <= 3,
            "PCA={}",
            r.pca_explained_variance_95
        );
    }

    /// SETUP: TestKnownDimensionalities::test_isotropic_gaussian_spectral — seed=42, (400, 8)
    #[test]
    fn test_isotropic_gaussian_spectral() {
        let d = 8usize;
        let data = seeded_randn(42, 400, d);
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.participation_ratio > 0.6 * (d as f64),
            "PR={}",
            r.participation_ratio
        );
        assert!(
            r.shannon_entropy > 0.6 * (d as f64),
            "Shannon={}",
            r.shannon_entropy
        );
    }

    /// SETUP: TestEstimatorConsistency::test_low_dim_data_all_estimators_low
    /// seed=42, rank-1 (200,1)@(1,10) + 1e-6 noise
    #[test]
    fn test_low_dim_data_spectral_low() {
        let t = seeded_randn(42, 200, 1);
        let emb = seeded_randn(43, 1, 10);
        let noise = seeded_randn(44, 200, 10) * 1e-6;
        let data = matmul(&t, &emb) + noise;
        let r = compute_spectral(&data).unwrap();
        assert!(
            r.pca_explained_variance_95 <= 2,
            "PCA={}",
            r.pca_explained_variance_95
        );
        assert!(r.participation_ratio < 3.0, "PR={}", r.participation_ratio);
    }
}
