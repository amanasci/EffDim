use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray1};
use ndarray::Axis;
use std::collections::HashSet;
use rayon::prelude::*;

/// Calculate squared Euclidean distance between two points
#[inline]
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Find k nearest neighbors using optimized brute-force (parallel)
fn find_k_nearest_brute(query: &[f64], all_points: &[Vec<f64>], k: usize) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = all_points.par_iter()
        .enumerate()
        .map(|(idx, point)| (idx, squared_distance(query, point)))
        .collect();
    
    // Partial sort to find k smallest
    distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k + 1);
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    distances
}

/// MLE (Levina-Bickel) dimensionality estimation - Brute Force
#[pyfunction]
fn mle_dimensionality(
    _py: Python,
    data: PyReadonlyArray2<f64>,
    k: usize,
) -> PyResult<f64> {
    let data_array = data.as_array();
    let n_samples = data_array.nrows();
    
    // Safety check
    let k = k.min(n_samples - 1);
    if k < 2 {
        return Ok(0.0);
    }
    
    // Convert to Vec<Vec<f64>> for easier processing
    let points: Vec<Vec<f64>> = data_array.axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();
    
    // Parallel processing of all points
    let inv_dim_estimates: Vec<f64> = points.par_iter()
        .filter_map(|point| {
            // Find k+1 nearest neighbors (including self)
            let neighbors = find_k_nearest_brute(point, &points, k);
            
            // Extract distances (skip first which is self with distance ~0)
            let distances: Vec<f64> = neighbors.iter()
                .skip(1)
                .map(|(_, d)| d.sqrt() + 1e-10)  // Add epsilon to avoid log(0)
                .collect();
            
            if distances.len() < k {
                return None;
            }
            
            let r_k = distances[k - 1];
            let r_j = &distances[..k - 1];
            
            // Calculate sum of log ratios
            let sum_log_ratios: f64 = r_j.iter()
                .map(|&r| (r_k / r).ln())
                .sum();
            
            // Inverse dimension estimate for this point
            let inv_dim = (k - 1) as f64 / (sum_log_ratios + 1e-10);
            Some(inv_dim)
        })
        .collect();
    
    // Return mean of inverse dimension estimates
    if inv_dim_estimates.is_empty() {
        Ok(0.0)
    } else {
        Ok(inv_dim_estimates.iter().sum::<f64>() / inv_dim_estimates.len() as f64)
    }
}

/// Two-NN dimensionality estimation (Facco et al.) - Brute Force
#[pyfunction]
fn two_nn_dimensionality(
    _py: Python,
    data: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let data_array = data.as_array();
    let n_samples = data_array.nrows();
    
    if n_samples < 3 {
        return Ok(0.0);
    }
    
    // Convert to Vec<Vec<f64>> for easier processing
    let points: Vec<Vec<f64>> = data_array.axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();
    
    // Parallel processing of all points
    let mu_values: Vec<f64> = points.par_iter()
        .filter_map(|point| {
            // Find 3 nearest neighbors (self + 2 neighbors)
            let neighbors = find_k_nearest_brute(point, &points, 2);
            
            if neighbors.len() < 3 {
                return None;
            }
            
            // Extract distances (skip self)
            let r1 = neighbors[1].1.sqrt() + 1e-10;
            let r2 = neighbors[2].1.sqrt() + 1e-10;
            
            let mu = r2 / r1;
            Some(mu)
        })
        .collect();
    
    if mu_values.is_empty() {
        return Ok(0.0);
    }
    
    // Sort mu values
    let mut mu_sorted = mu_values;
    mu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Drop last point to avoid log(0)
    mu_sorted.pop();
    
    let n_fit = mu_sorted.len();
    if n_fit == 0 {
        return Ok(0.0);
    }
    
    let mut x_values = Vec::with_capacity(n_fit);
    let mut y_values = Vec::with_capacity(n_fit);
    
    for (i, &mu) in mu_sorted.iter().enumerate() {
        let x = mu.ln();
        let y = -(1.0 - (i + 1) as f64 / n_samples as f64).ln();
        x_values.push(x);
        y_values.push(y);
    }
    
    // Linear regression: y = d * x (through origin)
    let x_dot_y: f64 = x_values.iter().zip(y_values.iter())
        .map(|(x, y)| x * y)
        .sum();
    let x_dot_x: f64 = x_values.iter()
        .map(|x| x * x)
        .sum();
    
    if x_dot_x == 0.0 {
        return Ok(0.0);
    }
    
    let d = x_dot_y / x_dot_x;
    Ok(d)
}

/// Box-counting dimensionality estimation
#[pyfunction]
fn box_counting_dimensionality(
    _py: Python,
    data: PyReadonlyArray2<f64>,
    box_sizes: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let data = data.as_array();
    let box_sizes = box_sizes.as_array();
    
    // Compute min bounds once
    let min_bounds: Vec<f64> = (0..data.ncols())
        .map(|col| data.column(col).iter().cloned().fold(f64::INFINITY, f64::min))
        .collect();
    
    let mut counts = Vec::with_capacity(box_sizes.len());
    
    for &box_size in box_sizes.iter() {
        let mut unique_boxes = HashSet::new();
        
        for row in data.axis_iter(Axis(0)) {
            let box_indices: Vec<i64> = row.iter()
                .zip(min_bounds.iter())
                .map(|(&val, &min_val)| ((val - min_val) / box_size).floor() as i64)
                .collect();
            
            unique_boxes.insert(box_indices);
        }
        
        counts.push(unique_boxes.len() as f64);
    }
    
    // Fit line: log(N) = -d * log(epsilon) + C
    let log_box_sizes: Vec<f64> = box_sizes.iter().map(|x| x.ln()).collect();
    let log_counts: Vec<f64> = counts.iter().map(|x| x.ln()).collect();
    
    let n = log_box_sizes.len() as f64;
    let sum_x: f64 = log_box_sizes.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xy: f64 = log_box_sizes.iter().zip(log_counts.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_xx: f64 = log_box_sizes.iter().map(|x| x * x).sum();
    
    // Linear regression slope
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator == 0.0 {
        return Ok(0.0);
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    
    Ok(-slope)
}

/// Python module
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mle_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(two_nn_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(box_counting_dimensionality, m)?)?;
    Ok(())
}
