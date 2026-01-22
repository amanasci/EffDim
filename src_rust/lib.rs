use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray1};
use ndarray::Axis;
use std::collections::HashSet;
use std::num::NonZero;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;

fn mle_impl<const K: usize>(data_array: ndarray::ArrayView2<f64>, k_neighbors: usize, n_samples: usize) -> PyResult<f64> {
    // Convert data to fixed-size arrays for kd-tree
    let items: Vec<[f64; K]> = data_array.axis_iter(Axis(0))
        .map(|row| {
            let mut point = [0.0f64; K];
            for (i, &val) in row.iter().enumerate().take(K) {
                point[i] = val;
            }
            point
        })
        .collect();
    
    // Build immutable k-d tree
    let tree: ImmutableKdTree<f64, u32, K, 32> = ImmutableKdTree::new_from_slice(&items);
    
    let mut inv_dim_estimates = Vec::with_capacity(n_samples);
    let k_plus_1 = NonZero::new(k_neighbors + 1).unwrap();
    
    for point in &items {
        // Find k+1 nearest neighbors (including self)
        let neighbors = tree.nearest_n::<SquaredEuclidean>(point, k_plus_1);
        
        // Extract distances (skip first which is self with distance ~0)
        let distances: Vec<f64> = neighbors.iter()
            .skip(1)
            .map(|n| (n.distance.sqrt() + 1e-10))  // Add epsilon to avoid log(0)
            .collect();
        
        if distances.len() < k_neighbors {
            continue;
        }
        
        let r_k = distances[k_neighbors - 1];
        let r_j = &distances[..k_neighbors - 1];
        
        // Calculate sum of log ratios
        let sum_log_ratios: f64 = r_j.iter()
            .map(|&r| (r_k / r).ln())
            .sum();
        
        // Inverse dimension estimate for this point
        let inv_dim = (k_neighbors - 1) as f64 / (sum_log_ratios + 1e-10);
        inv_dim_estimates.push(inv_dim);
    }
    
    // Return mean of inverse dimension estimates
    if inv_dim_estimates.is_empty() {
        Ok(0.0)
    } else {
        Ok(inv_dim_estimates.iter().sum::<f64>() / inv_dim_estimates.len() as f64)
    }
}

fn two_nn_impl<const K: usize>(data_array: ndarray::ArrayView2<f64>, n_samples: usize) -> PyResult<f64> {
    // Convert data to fixed-size arrays for kd-tree
    let items: Vec<[f64; K]> = data_array.axis_iter(Axis(0))
        .map(|row| {
            let mut point = [0.0f64; K];
            for (i, &val) in row.iter().enumerate().take(K) {
                point[i] = val;
            }
            point
        })
        .collect();
    
    // Build immutable k-d tree
    let tree: ImmutableKdTree<f64, u32, K, 32> = ImmutableKdTree::new_from_slice(&items);
    
    let mut mu_values: Vec<f64> = Vec::with_capacity(n_samples);
    let three = NonZero::new(3usize).unwrap();
    
    for point in &items {
        // Find 3 nearest neighbors (self + 2 neighbors)
        let neighbors = tree.nearest_n::<SquaredEuclidean>(point, three);
        
        if neighbors.len() < 3 {
            continue;
        }
        
        // Extract distances (skip self)
        let r1 = neighbors[1].distance.sqrt() + 1e-10;
        let r2 = neighbors[2].distance.sqrt() + 1e-10;
        
        let mu = r2 / r1;
        mu_values.push(mu);
    }
    
    if mu_values.is_empty() {
        return Ok(0.0);
    }
    
    // Sort mu values
    mu_values.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    
    // Drop last point to avoid log(0)
    mu_values.pop();
    
    let n_fit = mu_values.len();
    if n_fit == 0 {
        return Ok(0.0);
    }
    
    let mut x_values = Vec::with_capacity(n_fit);
    let mut y_values = Vec::with_capacity(n_fit);
    
    for (i, &mu) in mu_values.iter().enumerate() {
        let x: f64 = mu.ln();
        let y: f64 = -(1.0 - (i + 1) as f64 / n_samples as f64).ln();
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

/// MLE (Levina-Bickel) dimensionality estimation
#[pyfunction]
fn mle_dimensionality(
    _py: Python,
    data: PyReadonlyArray2<f64>,
    k: usize,
) -> PyResult<f64> {
    let data_array = data.as_array();
    let n_samples = data_array.nrows();
    let n_features = data_array.ncols();
    
    // Safety check
    let k = k.min(n_samples - 1);
    if k < 2 {
        return Ok(0.0);
    }
    
    // Dispatch to the correct dimension-specific implementation
    match n_features {
        1..=16 => mle_impl::<16>(data_array, k, n_samples),
        17..=32 => mle_impl::<32>(data_array, k, n_samples),
        33..=64 => mle_impl::<64>(data_array, k, n_samples),
        65..=128 => mle_impl::<128>(data_array, k, n_samples),
        129..=256 => mle_impl::<256>(data_array, k, n_samples),
        257..=512 => mle_impl::<512>(data_array, k, n_samples),
        513..=1024 => mle_impl::<1024>(data_array, k, n_samples),
        _ => mle_impl::<2048>(data_array, k, n_samples),
    }
}

/// Two-NN dimensionality estimation (Facco et al.)
#[pyfunction]
fn two_nn_dimensionality(
    _py: Python,
    data: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let data_array = data.as_array();
    let n_samples = data_array.nrows();
    let n_features = data_array.ncols();
    
    if n_samples < 3 {
        return Ok(0.0);
    }
    
    // Dispatch to the correct dimension-specific implementation
    match n_features {
        1..=16 => two_nn_impl::<16>(data_array, n_samples),
        17..=32 => two_nn_impl::<32>(data_array, n_samples),
        33..=64 => two_nn_impl::<64>(data_array, n_samples),
        65..=128 => two_nn_impl::<128>(data_array, n_samples),
        129..=256 => two_nn_impl::<256>(data_array, n_samples),
        257..=512 => two_nn_impl::<512>(data_array, n_samples),
        513..=1024 => two_nn_impl::<1024>(data_array, n_samples),
        _ => two_nn_impl::<2048>(data_array, n_samples),
    }
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
    let log_box_sizes: Vec<f64> = box_sizes.iter().map(|&x: &f64| x.ln()).collect();
    let log_counts: Vec<f64> = counts.iter().map(|&x: &f64| x.ln()).collect();
    
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
