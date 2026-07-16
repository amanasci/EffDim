//! Centering and exact SVD preprocess (D-09, D-10, D-12).

use faer::Mat;
use ndarray::Array2;

/// Ensure column means are near zero; subtract means only if any `|mean| >= tol`.
///
/// Mirrors Python `_ensure_centered` (`tol=1e-5` by default).
pub fn ensure_centered(mut data: Array2<f64>, tol: f64) -> Array2<f64> {
    let ncols = data.ncols();
    let nrows = data.nrows();
    if nrows == 0 || ncols == 0 {
        return data;
    }

    let mut means = vec![0.0f64; ncols];
    for row in data.rows() {
        for (j, &v) in row.iter().enumerate() {
            means[j] += v;
        }
    }
    let n = nrows as f64;
    for m in &mut means {
        *m /= n;
    }

    let needs_center = means.iter().any(|&m| m.abs() >= tol);
    if needs_center {
        for mut row in data.rows_mut() {
            for (j, v) in row.iter_mut().enumerate() {
                *v -= means[j];
            }
        }
    }
    data
}

/// Exact singular values via faer (nonincreasing, nonnegative). Float64 only (D-10).
pub fn singular_values_exact(data: &Array2<f64>) -> Result<Vec<f64>, faer::linalg::svd::SvdError> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    let mat = Mat::<f64>::from_fn(nrows, ncols, |i, j| data[(i, j)]);
    mat.singular_values()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    /// SETUP support: centering invariance used by TestComputeDimIntegration::test_centered_vs_uncentered
    #[test]
    fn ensure_centered_subtracts_large_mean() {
        let data = array![[100.0, 200.0], [102.0, 198.0], [101.0, 201.0]];
        let centered = ensure_centered(data, 1e-5);
        let means: Vec<f64> = (0..2)
            .map(|j| centered.column(j).mean().unwrap())
            .collect();
        assert_relative_eq!(means[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(means[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ensure_centered_skips_when_within_tol() {
        let data = array![[1e-6, -1e-6], [-1e-6, 1e-6]];
        let centered = ensure_centered(data.clone(), 1e-5);
        assert_eq!(centered, data);
    }
}
