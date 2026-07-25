//! Full `compute_dim` orchestration — spectral + shared k-NN + geometry (RUST-01).

use ndarray::Array2;

use crate::geometry::{
    danco_dimensionality, ess_dimensionality, gmst_dimensionality, mind_mli_dimensionality,
    mind_mlk_dimensionality, mle_dimensionality, tle_dimensionality, two_nn_dimensionality,
};
use crate::knn::exact_knn_l2_sq;
use crate::preprocess::ensure_centered;
use crate::{compute_spectral_centered, SpectralError, SpectralResults};

/// Full 16-key result bundle (8 spectral + 8 geometry).
#[derive(Debug, Clone, PartialEq)]
pub struct ComputeDimResults {
    pub pca_explained_variance_95: u32,
    pub participation_ratio: f64,
    pub shannon_entropy: f64,
    pub renyi_eff_dimensionality_alpha_2: f64,
    pub renyi_eff_dimensionality_alpha_3: f64,
    pub renyi_eff_dimensionality_alpha_4: f64,
    pub renyi_eff_dimensionality_alpha_5: f64,
    pub geometric_mean_eff_dimensionality: f64,
    pub mle_dimensionality: f64,
    pub two_nn_dimensionality: f64,
    pub danco_dimensionality: f64,
    pub mind_mli_dimensionality: f64,
    pub mind_mlk_dimensionality: f64,
    pub ess_dimensionality: f64,
    pub tle_dimensionality: f64,
    pub gmst_dimensionality: f64,
}

fn from_spectral(s: SpectralResults) -> ComputeDimResults {
    ComputeDimResults {
        pca_explained_variance_95: s.pca_explained_variance_95,
        participation_ratio: s.participation_ratio,
        shannon_entropy: s.shannon_entropy,
        renyi_eff_dimensionality_alpha_2: s.renyi_eff_dimensionality_alpha_2,
        renyi_eff_dimensionality_alpha_3: s.renyi_eff_dimensionality_alpha_3,
        renyi_eff_dimensionality_alpha_4: s.renyi_eff_dimensionality_alpha_4,
        renyi_eff_dimensionality_alpha_5: s.renyi_eff_dimensionality_alpha_5,
        geometric_mean_eff_dimensionality: s.geometric_mean_eff_dimensionality,
        mle_dimensionality: 0.0,
        two_nn_dimensionality: 0.0,
        danco_dimensionality: 0.0,
        mind_mli_dimensionality: 0.0,
        mind_mlk_dimensionality: 0.0,
        ess_dimensionality: 0.0,
        tle_dimensionality: 0.0,
        gmst_dimensionality: 0.0,
    }
}

/// Center → SVD/spectral → float32 shared k-NN → geometry → full 16-field results.
///
/// Geometry soft-fails return `0.0` (never panic on tiny `n`). Euclidean GMST only (D-11).
pub fn compute_dim(data: &Array2<f64>) -> Result<ComputeDimResults, SpectralError> {
    // Center once; spectral and geometry paths share the same centered matrix
    // (centering is idempotent, so single-center matches api.py's duplicate-center D-04).
    let centered = ensure_centered(data.clone(), 1e-5);
    let spectral = compute_spectral_centered(&centered)?;
    let mut results = from_spectral(spectral);

    let data_f32 = centered.mapv(|x| x as f32);

    let (dist_sq, indices) = exact_knn_l2_sq(&data_f32, 10);

    results.mle_dimensionality = mle_dimensionality(&data_f32, 10, Some(&dist_sq));
    results.two_nn_dimensionality = two_nn_dimensionality(&data_f32, Some(&dist_sq));
    results.danco_dimensionality =
        danco_dimensionality(&data_f32, 10, Some(&dist_sq), Some(&indices));
    results.mind_mli_dimensionality = mind_mli_dimensionality(&data_f32, Some(&dist_sq));
    results.mind_mlk_dimensionality = mind_mlk_dimensionality(&data_f32, 10, Some(&dist_sq));
    results.ess_dimensionality =
        ess_dimensionality(&data_f32, 10, Some(&dist_sq), Some(&indices));
    results.tle_dimensionality = tle_dimensionality(&data_f32, 10, Some(&dist_sq));
    // Euclidean GMST only inside compute_dim (D-11); geodesic remains on geometry::gmst_dimensionality.
    results.gmst_dimensionality = gmst_dimensionality(&data_f32, false, 42);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic standard-normal fills (same helper style as Inventory B).
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

    /// SETUP: smoke seed=42, shape=(50, 5) — all 16 fields finite and geometry >= 0.
    #[test]
    fn test_compute_dim_smoke_50x5() {
        let data = seeded_randn(42, 50, 5);
        let r = compute_dim(&data).unwrap();

        assert!(r.pca_explained_variance_95 >= 1);
        for v in [
            r.participation_ratio,
            r.shannon_entropy,
            r.renyi_eff_dimensionality_alpha_2,
            r.renyi_eff_dimensionality_alpha_3,
            r.renyi_eff_dimensionality_alpha_4,
            r.renyi_eff_dimensionality_alpha_5,
            r.geometric_mean_eff_dimensionality,
            r.mle_dimensionality,
            r.two_nn_dimensionality,
            r.danco_dimensionality,
            r.mind_mli_dimensionality,
            r.mind_mlk_dimensionality,
            r.ess_dimensionality,
            r.tle_dimensionality,
            r.gmst_dimensionality,
        ] {
            assert!(v.is_finite() && v >= 0.0, "non-finite or negative: {v}");
        }
    }
}
