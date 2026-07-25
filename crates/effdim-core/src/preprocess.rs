//! Centering and exact SVD preprocess (D-09, D-10, D-12).

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use ndarray::{Array1, Array2, ArrayView2, Axis};

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

/// Accumulate a covariance matrix in row chunks using the parallel-variance
/// (Chan) merge, then return its nonnegative eigenvalues in descending order.
pub fn covariance_eigenvalues_streaming(
    data: ArrayView2<'_, f64>,
    chunk_size: usize,
) -> Result<Vec<f64>, faer::linalg::svd::SvdError> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    if nrows < 2 || ncols == 0 {
        return Ok(vec![0.0; ncols]);
    }

    let mut count = 0usize;
    let mut mean = Array1::<f64>::zeros(ncols);
    let mut m2 = Array2::<f64>::zeros((ncols, ncols));

    for chunk in data.axis_chunks_iter(Axis(0), chunk_size.max(1)) {
        let chunk_count = chunk.nrows();
        let chunk_mean = chunk.mean_axis(Axis(0)).expect("chunk is non-empty");
        let centered = &chunk - &chunk_mean;
        let chunk_m2 = centered.t().dot(&centered);

        if count == 0 {
            mean.assign(&chunk_mean);
            m2.assign(&chunk_m2);
            count = chunk_count;
            continue;
        }

        let combined = count + chunk_count;
        let delta = &chunk_mean - &mean;
        let correction_scale = (count as f64 * chunk_count as f64) / combined as f64;
        m2 += &chunk_m2;
        for i in 0..ncols {
            for j in 0..ncols {
                m2[[i, j]] += correction_scale * delta[i] * delta[j];
            }
        }
        mean += &(delta * (chunk_count as f64 / combined as f64));
        count = combined;
    }

    let covariance = m2 / (count - 1) as f64;
    singular_values_exact(&covariance)
}

/// Streaming covariance using faer's parallel GEMM for each chunk.
pub fn covariance_eigenvalues_streaming_faer(
    data: ArrayView2<'_, f64>,
    chunk_size: usize,
    threads: usize,
) -> Result<Vec<f64>, faer::linalg::svd::SvdError> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    if nrows < 2 || ncols == 0 {
        return Ok(vec![0.0; ncols]);
    }

    let parallelism = if threads == 1 {
        Par::Seq
    } else {
        Par::rayon(threads)
    };
    let mut count = 0usize;
    let mut mean = vec![0.0f64; ncols];
    let mut m2 = Mat::<f64>::zeros(ncols, ncols);

    for chunk in data.axis_chunks_iter(Axis(0), chunk_size.max(1)) {
        let chunk_count = chunk.nrows();
        let mut chunk_mean = vec![0.0f64; ncols];
        for row in chunk.rows() {
            for (column, &value) in row.iter().enumerate() {
                chunk_mean[column] += value;
            }
        }
        for value in &mut chunk_mean {
            *value /= chunk_count as f64;
        }

        let centered = Mat::<f64>::from_fn(chunk_count, ncols, |row, column| {
            chunk[[row, column]] - chunk_mean[column]
        });
        let mut chunk_m2 = Mat::<f64>::zeros(ncols, ncols);
        matmul(
            &mut chunk_m2,
            Accum::Replace,
            centered.transpose(),
            &centered,
            1.0,
            parallelism,
        );

        if count == 0 {
            mean.copy_from_slice(&chunk_mean);
            for i in 0..ncols {
                for j in 0..ncols {
                    m2[(i, j)] = chunk_m2[(i, j)];
                }
            }
            count = chunk_count;
            continue;
        }

        let combined = count + chunk_count;
        let correction_scale = (count as f64 * chunk_count as f64) / combined as f64;
        let delta: Vec<f64> = chunk_mean
            .iter()
            .zip(&mean)
            .map(|(&chunk_value, &global_value)| chunk_value - global_value)
            .collect();
        for i in 0..ncols {
            for j in 0..ncols {
                m2[(i, j)] += chunk_m2[(i, j)] + correction_scale * delta[i] * delta[j];
            }
            mean[i] += delta[i] * (chunk_count as f64 / combined as f64);
        }
        count = combined;
    }

    let scale = 1.0 / (count - 1) as f64;
    let covariance = Mat::<f64>::from_fn(ncols, ncols, |i, j| m2[(i, j)] * scale);
    covariance.singular_values()
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

    #[test]
    fn faer_streaming_matches_ndarray_streaming() {
        let data = array![
            [1.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
            [4.0, 3.0, 2.0],
            [5.0, 7.0, 1.0],
            [8.0, 6.0, 9.0],
        ];
        let expected = covariance_eigenvalues_streaming(data.view(), 2).unwrap();
        let actual = covariance_eigenvalues_streaming_faer(data.view(), 2, 1).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (&left, &right) in expected.iter().zip(&actual) {
            assert_relative_eq!(left, right, epsilon = 1e-12);
        }
    }
}
