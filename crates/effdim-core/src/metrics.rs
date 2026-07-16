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
