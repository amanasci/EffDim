//! Geometric intrinsic-dimension estimators (1:1 ports of `geometry.py`).
//!
//! Soft-return `0.0` on undersized inputs (D-13). Shared k-NN distances/indices
//! avoid a second search for DANCo/ESS (D-08, D-09).

use ndarray::Array2;

use crate::knn::exact_knn_l2_sq;

const EPS: f64 = 1e-10;

/// Resolve squared k-NN distances: use precomputed or compute via exact float32 L2.
fn resolve_dist_sq(
    data: &Array2<f32>,
    k: usize,
    precomputed: Option<&Array2<f32>>,
) -> Array2<f32> {
    if let Some(d) = precomputed {
        d.clone()
    } else {
        exact_knn_l2_sq(data, k).0
    }
}

/// Resolve neighbor indices: use shared precomputed or run exact k-NN.
fn resolve_indices(
    data: &Array2<f32>,
    k: usize,
    precomputed_indices: Option<&Array2<usize>>,
) -> Array2<usize> {
    if let Some(idx) = precomputed_indices {
        idx.clone()
    } else {
        exact_knn_l2_sq(data, k).1
    }
}

/// Levina-Bickel MLE intrinsic dimensionality.
pub fn mle_dimensionality(
    data: &Array2<f32>,
    k: usize,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 2 {
        return 0.0;
    }

    let (dist_sq, k_eff) = if let Some(pre) = precomputed_knn_dist_sq {
        let k_eff = pre.ncols();
        if k_eff < 2 {
            return 0.0;
        }
        (pre.clone(), k_eff)
    } else {
        let k_eff = k.min(n_samples - 1);
        if k_eff < 2 {
            return 0.0;
        }
        (exact_knn_l2_sq(data, k_eff).0, k_eff)
    };

    let mut sum_estimates = 0.0f64;
    for i in 0..n_samples {
        let r_k = (dist_sq[[i, k_eff - 1]] as f64).sqrt() + EPS;
        let log_r_k = r_k.ln();
        let mut sum_log_ratios = 0.0f64;
        for j in 0..(k_eff - 1) {
            let r_j = (dist_sq[[i, j]] as f64).sqrt() + EPS;
            sum_log_ratios += log_r_k - r_j.ln();
        }
        // errstate(divide/invalid='ignore'): non-finite estimates still enter the mean
        sum_estimates += (k_eff - 1) as f64 / (sum_log_ratios + EPS);
    }
    sum_estimates / n_samples as f64
}

/// Two-NN intrinsic dimensionality.
pub fn two_nn_dimensionality(
    data: &Array2<f32>,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 3 {
        return 0.0;
    }

    let dist_sq = resolve_dist_sq(data, 2, precomputed_knn_dist_sq);
    if dist_sq.ncols() < 2 {
        return 0.0;
    }

    let mut mu: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let r1 = (dist_sq[[i, 0]] as f64).sqrt() + EPS;
        let r2 = (dist_sq[[i, 1]] as f64).sqrt() + EPS;
        mu.push(r2 / r1);
    }
    mu.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Drop last to avoid F(mu)=1 → log(0)
    mu.pop();
    let n_fit = mu.len();
    if n_fit == 0 {
        return 0.0;
    }

    let mut x_dot_x = 0.0f64;
    let mut x_dot_y = 0.0f64;
    for (idx, &m) in mu.iter().enumerate() {
        let x = m.ln();
        let i = (idx + 1) as f64;
        let y = -(1.0 - i / n_samples as f64).ln();
        x_dot_x += x * x;
        x_dot_y += x * y;
    }
    if x_dot_x == 0.0 {
        return 0.0;
    }
    x_dot_y / x_dot_x
}

/// DANCo (angle-only as in current Python — D-13).
pub fn danco_dimensionality(
    data: &Array2<f32>,
    k: usize,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
    precomputed_indices: Option<&Array2<usize>>,
) -> f64 {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    if n_samples < 3 {
        return 0.0;
    }

    let mut k_eff = if let Some(pre) = precomputed_knn_dist_sq {
        pre.ncols()
    } else {
        k
    };
    k_eff = k_eff.min(n_samples - 1);
    if k_eff < 2 {
        return 0.0;
    }

    let neighbor_indices = resolve_indices(data, k_eff, precomputed_indices);
    let k_actual = neighbor_indices.ncols();
    if k_actual < 2 {
        return 0.0;
    }

    let mut sum_cos_sq = 0.0f64;
    let mut n_cos = 0usize;

    for i in 0..n_samples {
        // unit vectors to neighbors
        let mut units: Vec<Vec<f64>> = Vec::with_capacity(k_actual);
        for r in 0..k_actual {
            let j = neighbor_indices[[i, r]];
            let mut v = Vec::with_capacity(n_features);
            let mut norm_sq = 0.0f64;
            for t in 0..n_features {
                let d = data[[j, t]] as f64 - data[[i, t]] as f64;
                v.push(d);
                norm_sq += d * d;
            }
            let norm = norm_sq.sqrt() + EPS;
            for x in &mut v {
                *x /= norm;
            }
            units.push(v);
        }

        for a in 0..k_actual {
            for b in (a + 1)..k_actual {
                let mut cos = 0.0f64;
                for t in 0..n_features {
                    cos += units[a][t] * units[b][t];
                }
                sum_cos_sq += cos * cos;
                n_cos += 1;
            }
        }
    }

    if n_cos == 0 {
        return 0.0;
    }
    let mean_cos_sq = sum_cos_sq / n_cos as f64;
    if mean_cos_sq < EPS {
        return 0.0;
    }
    1.0 / mean_cos_sq
}

/// MiND-MLi (single nearest-neighbor MLE).
pub fn mind_mli_dimensionality(
    data: &Array2<f32>,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 3 {
        return 0.0;
    }

    let dist_sq = resolve_dist_sq(data, 1, precomputed_knn_dist_sq);
    if dist_sq.ncols() < 1 {
        return 0.0;
    }

    let mut dist: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        dist.push((dist_sq[[i, 0]] as f64).sqrt() + EPS);
    }

    let r_max = dist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let first = dist[0];
    if dist.iter().all(|&d| (d - first).abs() < EPS) {
        return 0.0;
    }

    let sum_log_ratios: f64 = dist.iter().map(|&r| (r_max / r).ln()).sum();
    if sum_log_ratios < EPS {
        return 0.0;
    }
    n_samples as f64 / sum_log_ratios
}

/// MiND-MLk (median of per-point MLE estimates).
pub fn mind_mlk_dimensionality(
    data: &Array2<f32>,
    k: usize,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 2 {
        return 0.0;
    }

    let (dist_sq, k_eff) = if let Some(pre) = precomputed_knn_dist_sq {
        let k_eff = pre.ncols();
        if k_eff < 2 {
            return 0.0;
        }
        (pre.clone(), k_eff)
    } else {
        let k_eff = k.min(n_samples - 1);
        if k_eff < 2 {
            return 0.0;
        }
        (exact_knn_l2_sq(data, k_eff).0, k_eff)
    };

    let mut estimates: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let r_k = (dist_sq[[i, k_eff - 1]] as f64).sqrt() + EPS;
        let log_r_k = r_k.ln();
        let mut sum_log_ratios = 0.0f64;
        for j in 0..(k_eff - 1) {
            let r_j = (dist_sq[[i, j]] as f64).sqrt() + EPS;
            sum_log_ratios += log_r_k - r_j.ln();
        }
        let est = (k_eff - 1) as f64 / (sum_log_ratios + EPS);
        estimates.push(est);
    }

    estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // NumPy median: average of two middle for even length
    let n = estimates.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        estimates[n / 2]
    } else {
        0.5 * (estimates[n / 2 - 1] + estimates[n / 2])
    }
}

/// ESS (Expected Simplex Skewness).
pub fn ess_dimensionality(
    data: &Array2<f32>,
    k: usize,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
    precomputed_indices: Option<&Array2<usize>>,
) -> f64 {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    if n_samples < 3 {
        return 0.0;
    }

    let mut k_eff = if let Some(pre) = precomputed_knn_dist_sq {
        pre.ncols()
    } else {
        k
    };
    k_eff = k_eff.min(n_samples - 1);
    if k_eff < 1 {
        return 0.0;
    }

    let neighbor_indices = resolve_indices(data, k_eff, precomputed_indices);
    let k_actual = neighbor_indices.ncols();
    if k_actual < 1 {
        return 0.0;
    }

    let mut sum_s = 0.0f64;
    for i in 0..n_samples {
        let mut centroid = vec![0.0f64; n_features];
        for r in 0..k_actual {
            let j = neighbor_indices[[i, r]];
            let mut v = vec![0.0f64; n_features];
            let mut norm_sq = 0.0f64;
            for t in 0..n_features {
                let d = data[[j, t]] as f64 - data[[i, t]] as f64;
                v[t] = d;
                norm_sq += d * d;
            }
            let norm = norm_sq.sqrt() + EPS;
            for t in 0..n_features {
                centroid[t] += v[t] / norm;
            }
        }
        for t in 0..n_features {
            centroid[t] /= k_actual as f64;
        }
        let s: f64 = centroid.iter().map(|c| c * c).sum();
        sum_s += s;
    }

    let s_avg = sum_s / n_samples as f64;
    if s_avg < EPS {
        return 0.0;
    }
    1.0 / (k_actual as f64 * s_avg)
}

/// TLE≈MLE (port current Python formula as-is — D-13).
pub fn tle_dimensionality(
    data: &Array2<f32>,
    k: usize,
    precomputed_knn_dist_sq: Option<&Array2<f32>>,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 2 {
        return 0.0;
    }

    let (dist_sq, k_eff) = if let Some(pre) = precomputed_knn_dist_sq {
        let k_eff = pre.ncols();
        if k_eff < 2 {
            return 0.0;
        }
        (pre.clone(), k_eff)
    } else {
        let k_eff = k.min(n_samples - 1);
        if k_eff < 2 {
            return 0.0;
        }
        (exact_knn_l2_sq(data, k_eff).0, k_eff)
    };

    let mut sum_estimates = 0.0f64;
    for i in 0..n_samples {
        let r_k = (dist_sq[[i, k_eff - 1]] as f64).sqrt() + EPS;
        let mut neg_sum_log_u = 0.0f64;
        for j in 0..(k_eff - 1) {
            let r_j = (dist_sq[[i, j]] as f64).sqrt() + EPS;
            let u_j = r_j / r_k;
            neg_sum_log_u += -u_j.ln();
        }
        sum_estimates += (k_eff - 1) as f64 / (neg_sum_log_u + EPS);
    }
    sum_estimates / n_samples as f64
}

#[cfg(test)]
mod soft_zero_tests {
    //! Soft-zero sentinels matching test_input_validation.py geometric edge cases.

    use super::*;
    use ndarray::array;

    #[test]
    fn test_mle_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(mle_dimensionality(&data, 1, None), 0.0);
    }

    #[test]
    fn test_two_nn_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(two_nn_dimensionality(&data, None), 0.0);
    }

    #[test]
    fn test_danco_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(danco_dimensionality(&data, 10, None, None), 0.0);
    }

    #[test]
    fn test_mind_mli_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(mind_mli_dimensionality(&data, None), 0.0);
    }

    #[test]
    fn test_mind_mlk_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(mind_mlk_dimensionality(&data, 10, None), 0.0);
    }

    #[test]
    fn test_ess_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(ess_dimensionality(&data, 10, None, None), 0.0);
    }

    #[test]
    fn test_tle_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(tle_dimensionality(&data, 10, None), 0.0);
    }

    /// Smoke: TLE≈MLE on identical precomputed distances (D-13).
    #[test]
    fn test_tle_matches_mle_approx_on_precomputed() {
        let data = array![
            [0.0f32, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let (dist_sq, _) = exact_knn_l2_sq(&data, 3);
        let mle = mle_dimensionality(&data, 3, Some(&dist_sq));
        let tle = tle_dimensionality(&data, 3, Some(&dist_sq));
        assert!(mle.is_finite() && mle > 0.0);
        assert!(tle.is_finite() && tle > 0.0);
        // TLE≈MLE: same algebraic form on these inputs → relative band
        let rel = (mle - tle).abs() / mle.max(tle);
        assert!(rel < 0.05, "mle={mle} tle={tle} rel={rel}");
    }
}
