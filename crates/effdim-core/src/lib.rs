//! Pure Rust compute core — preprocess + spectral metrics (Phase 3).

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
