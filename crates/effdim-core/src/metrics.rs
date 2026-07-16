//! Spectral effective-dimension metrics — 1:1 ports of `src/effdim/metrics.py`.

/// Number of principal components needed to explain `threshold` of variance.
///
/// Uses searchsorted-equivalent (left) + 1, matching NumPy.
pub fn pca_explained_variance(spectrum: &[f64], threshold: f64) -> u32 {
    let total_variance: f64 = spectrum.iter().sum();
    let mut cumulative = 0.0f64;
    let mut ratios = Vec::with_capacity(spectrum.len());
    for &v in spectrum {
        cumulative += v;
        ratios.push(cumulative / total_variance);
    }
    // NumPy searchsorted(..., side='left'): first index where ratio >= threshold
    let idx = ratios.partition_point(|&r| r < threshold);
    (idx + 1) as u32
}

/// Participation ratio: `(sum λ)² / sum(λ²)`, or `0.0` if denominator is zero.
pub fn participation_ratio(spectrum: &[f64]) -> f64 {
    let sum: f64 = spectrum.iter().sum();
    let denom: f64 = spectrum.iter().map(|x| x * x).sum();
    if denom == 0.0 {
        return 0.0;
    }
    (sum * sum) / denom
}

/// Shannon effective dimension: `exp(-Σ p log p)` over positive probabilities.
pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
    let mut entropy = 0.0f64;
    for &p in probabilities {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

/// Rényi effective dimensionality for `alpha > 0`, `alpha != 1`.
pub fn renyi_eff_dimensionality(probabilities: &[f64], alpha: f64) -> Result<f64, String> {
    if alpha <= 0.0 || (alpha - 1.0).abs() < f64::EPSILON {
        return Err("Alpha must be greater than 0 and not equal to 1.".to_string());
    }
    let sum_probs_alpha: f64 = probabilities.iter().map(|&p| p.powf(alpha)).sum();
    if sum_probs_alpha == 0.0 {
        return Ok(0.0);
    }
    Ok(sum_probs_alpha.powf(1.0 / (1.0 - alpha)))
}

/// Geometric-mean effective dimensionality on positive entries: `am / gm`, else `0.0`.
///
/// Call site in `compute_spectral` passes probabilities (not raw eigenvalues).
pub fn geometric_mean_eff_dimensionality(spectrum: &[f64]) -> f64 {
    let positive: Vec<f64> = spectrum.iter().copied().filter(|&x| x > 0.0).collect();
    if positive.is_empty() {
        return 0.0;
    }
    let n = positive.len() as f64;
    let am: f64 = positive.iter().sum::<f64>() / n;
    let log_mean: f64 = positive.iter().map(|x| x.ln()).sum::<f64>() / n;
    let gm = log_mean.exp();
    am / gm
}

#[cfg(test)]
mod tests {
    //! Inventory A mirrors (D-14): identical setups + asserts as Python
    //! `TestMetricsInputEdgeCases` + `TestSpectralEstimatorsEdgeCases`.

    use super::*;
    use approx::assert_relative_eq;

    fn normalize(spectrum: &[f64]) -> Vec<f64> {
        let total: f64 = spectrum.iter().sum();
        spectrum.iter().map(|&x| x / total).collect()
    }

    // --- TestMetricsInputEdgeCases ---

    /// SETUP: TestMetricsInputEdgeCases::test_participation_ratio_single_nonzero
    #[test]
    fn test_participation_ratio_single_nonzero() {
        let spectrum = [5.0, 0.0, 0.0];
        let pr = participation_ratio(&spectrum);
        assert_relative_eq!(pr, 1.0, epsilon = 1e-12);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_participation_ratio_uniform
    #[test]
    fn test_participation_ratio_uniform() {
        let d = 7;
        let spectrum = vec![1.0; d];
        let pr = participation_ratio(&spectrum);
        assert_relative_eq!(pr, d as f64, epsilon = 1e-12);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_shannon_entropy_uniform
    #[test]
    fn test_shannon_entropy_uniform() {
        let d = 6;
        let probs = vec![1.0 / d as f64; d];
        let ed = shannon_entropy(&probs);
        assert_relative_eq!(ed, d as f64, max_relative = 1e-6);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_shannon_entropy_concentrated
    #[test]
    fn test_shannon_entropy_concentrated() {
        let probs = [0.999, 0.0005, 0.0005];
        let ed = shannon_entropy(&probs);
        assert!(1.0 < ed && ed < 1.5, "expected 1 < ed < 1.5, got {ed}");
    }

    /// SETUP: TestMetricsInputEdgeCases::test_renyi_valid_alpha_half
    #[test]
    fn test_renyi_valid_alpha_half() {
        let probs = [0.4, 0.3, 0.2, 0.1];
        let result = renyi_eff_dimensionality(&probs, 0.5).unwrap();
        assert!(result.is_finite() && result > 0.0);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_renyi_valid_integer_alphas
    #[test]
    fn test_renyi_valid_integer_alphas() {
        let probs = [0.4, 0.3, 0.2, 0.1];
        for alpha in [2.0, 3.0, 4.0, 5.0] {
            let result = renyi_eff_dimensionality(&probs, alpha).unwrap();
            assert!(
                result.is_finite() && result > 0.0,
                "alpha={alpha} returned {result}"
            );
        }
    }

    /// SETUP: TestMetricsInputEdgeCases::test_renyi_ordering
    #[test]
    fn test_renyi_ordering() {
        let probs = [0.5, 0.3, 0.15, 0.05];
        let alphas = [2.0, 3.0, 4.0, 5.0];
        let values: Vec<f64> = alphas
            .iter()
            .map(|&a| renyi_eff_dimensionality(&probs, a).unwrap())
            .collect();
        for i in 0..values.len() - 1 {
            assert!(
                values[i] >= values[i + 1],
                "Rényi not non-increasing: alpha={} gave {}, alpha={} gave {}",
                alphas[i],
                values[i],
                alphas[i + 1],
                values[i + 1]
            );
        }
    }

    /// SETUP: TestMetricsInputEdgeCases::test_geometric_mean_known_value
    #[test]
    fn test_geometric_mean_known_value() {
        let spectrum = [4.0, 1.0];
        // am = 2.5, gm = 2.0, ratio = 1.25
        let result = geometric_mean_eff_dimensionality(&spectrum);
        let expected = 2.5 / 2.0;
        assert_relative_eq!(result, expected, max_relative = 1e-6);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_geometric_mean_equal_values
    #[test]
    fn test_geometric_mean_equal_values() {
        let spectrum = [3.0, 3.0, 3.0];
        let result = geometric_mean_eff_dimensionality(&spectrum);
        assert_relative_eq!(result, 1.0, max_relative = 1e-6);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_pca_variance_threshold_50pct
    #[test]
    fn test_pca_variance_threshold_50pct() {
        let spectrum = [4.0, 2.0, 1.0, 1.0];
        let result = pca_explained_variance(&spectrum, 0.5);
        assert_eq!(result, 1);
    }

    /// SETUP: TestMetricsInputEdgeCases::test_pca_variance_threshold_75pct
    #[test]
    fn test_pca_variance_threshold_75pct() {
        let spectrum = [4.0, 2.0, 1.0, 1.0];
        let result = pca_explained_variance(&spectrum, 0.75);
        assert_eq!(result, 2);
    }

    // --- TestSpectralEstimatorsEdgeCases ---

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_single_dominant_eigenvalue
    #[test]
    fn test_single_dominant_eigenvalue() {
        let spectrum = [100.0, 0.01, 0.01, 0.01];
        let pr = participation_ratio(&spectrum);
        assert!(1.0 < pr && pr < 1.5, "PR should be close to 1, got {pr}");
        let probs = normalize(&spectrum);
        let shannon = shannon_entropy(&probs);
        assert!(
            1.0 < shannon && shannon < 1.5,
            "Shannon ED should be close to 1, got {shannon}"
        );
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_equal_eigenvalues
    #[test]
    fn test_equal_eigenvalues() {
        let d = 10;
        let spectrum = vec![1.0; d];
        let pr = participation_ratio(&spectrum);
        assert_relative_eq!(pr, d as f64, epsilon = 1e-12);
        let probs = normalize(&spectrum);
        let shannon = shannon_entropy(&probs);
        assert_relative_eq!(shannon, d as f64, epsilon = 1e-12);
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_zero_eigenvalues
    #[test]
    fn test_zero_eigenvalues() {
        let spectrum = [10.0, 5.0, 1.0, 0.0, 0.0];
        let pr = participation_ratio(&spectrum);
        assert!(pr > 0.0, "PR should handle zero eigenvalues");
        let probs = normalize(&spectrum);
        let shannon = shannon_entropy(&probs);
        assert!(shannon.is_finite(), "Shannon ED should handle zero probabilities");
        let gm = geometric_mean_eff_dimensionality(&spectrum);
        assert!(gm > 0.0, "Geometric mean should handle zero eigenvalues");
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_all_zero_spectrum
    #[test]
    fn test_all_zero_spectrum() {
        let spectrum = [0.0; 5];
        assert_eq!(participation_ratio(&spectrum), 0.0);
        assert_eq!(geometric_mean_eff_dimensionality(&spectrum), 0.0);
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_very_small_eigenvalues
    #[test]
    fn test_very_small_eigenvalues() {
        let spectrum = [1.0, 1e-8, 1e-10, 1e-12];
        let pr = participation_ratio(&spectrum);
        assert!(pr.is_finite() && pr > 0.0, "PR should handle very small eigenvalues");
        let probs = normalize(&spectrum);
        let shannon = shannon_entropy(&probs);
        assert!(shannon.is_finite(), "Shannon should handle very small probabilities");
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_very_large_eigenvalue_range
    #[test]
    fn test_very_large_eigenvalue_range() {
        let spectrum = [1e10, 1e5, 1e0, 1e-5];
        let pr = participation_ratio(&spectrum);
        assert!(pr.is_finite(), "PR should handle large eigenvalue range");
        let probs = normalize(&spectrum);
        let shannon = shannon_entropy(&probs);
        assert!(shannon.is_finite(), "Shannon should handle large eigenvalue range");
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_renyi_alpha_2_equals_pr
    #[test]
    fn test_renyi_alpha_2_equals_pr() {
        let spectrum = [4.0, 3.0, 2.0, 1.0];
        let probs = normalize(&spectrum);
        let pr = participation_ratio(&spectrum);
        let renyi_2 = renyi_eff_dimensionality(&probs, 2.0).unwrap();
        assert_relative_eq!(pr, renyi_2, epsilon = 1e-12);
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_renyi_invalid_alpha
    #[test]
    fn test_renyi_invalid_alpha() {
        let probs = [0.4, 0.3, 0.2, 0.1];
        assert!(renyi_eff_dimensionality(&probs, 1.0).is_err());
        assert!(renyi_eff_dimensionality(&probs, 0.0).is_err());
        assert!(renyi_eff_dimensionality(&probs, -1.0).is_err());
    }

    /// SETUP: TestSpectralEstimatorsEdgeCases::test_pca_threshold_edge_cases
    #[test]
    fn test_pca_threshold_edge_cases() {
        let spectrum = [4.0, 3.0, 2.0, 1.0];
        assert!(pca_explained_variance(&spectrum, 0.0) >= 1);
        assert_eq!(
            pca_explained_variance(&spectrum, 1.0),
            spectrum.len() as u32
        );
        let ratio_first = 4.0 / 10.0;
        assert_eq!(
            pca_explained_variance(&spectrum, ratio_first + 0.01),
            2
        );
    }
}
