//! PyO3 bindings for `effdim._native` (round-trip, compute_dim, granular estimators/metrics).

extern crate openblas_src;

#[link(name = "openblas")]
unsafe extern "C" {}

use effdim_core::geometry;
use effdim_core::knn;
use effdim_core::metrics;
use effdim_core::{
    compute_dim as core_compute_dim, compute_spectral as core_compute_spectral, identity_f64_slice,
    spectral_eigenvalues_exact as core_spectral_eigenvalues_exact,
    spectral_eigenvalues_streaming as core_spectral_eigenvalues_streaming,
    spectral_eigenvalues_streaming_faer as core_spectral_eigenvalues_streaming_faer,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Borrow a float64 2-D NumPy array and return an owned copy with the same shape/dtype.
#[pyfunction]
fn roundtrip_array<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let owned = data.as_array().to_owned();
    let shape = owned.raw_dim();
    // Invoke core so the path dependency is linked into the extension.
    let flat = identity_f64_slice(owned.as_slice().expect("owned array is contiguous"));
    ndarray::Array2::from_shape_vec(shape, flat)
        .expect("shape matches flat len")
        .into_pyarray(py)
}

/// Full 16-key flat dict (spectral + geometry). PCA key is a Python int.
#[pyfunction]
fn compute_dim<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned = data.as_array().to_owned();
    let results = py
        .detach(|| core_compute_dim(&owned))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "pca_explained_variance_95",
        results.pca_explained_variance_95 as i64,
    )?;
    dict.set_item("participation_ratio", results.participation_ratio)?;
    dict.set_item("shannon_entropy", results.shannon_entropy)?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_2",
        results.renyi_eff_dimensionality_alpha_2,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_3",
        results.renyi_eff_dimensionality_alpha_3,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_4",
        results.renyi_eff_dimensionality_alpha_4,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_5",
        results.renyi_eff_dimensionality_alpha_5,
    )?;
    dict.set_item(
        "geometric_mean_eff_dimensionality",
        results.geometric_mean_eff_dimensionality,
    )?;
    dict.set_item("mle_dimensionality", results.mle_dimensionality)?;
    dict.set_item("two_nn_dimensionality", results.two_nn_dimensionality)?;
    dict.set_item("danco_dimensionality", results.danco_dimensionality)?;
    dict.set_item("mind_mli_dimensionality", results.mind_mli_dimensionality)?;
    dict.set_item("mind_mlk_dimensionality", results.mind_mlk_dimensionality)?;
    dict.set_item("ess_dimensionality", results.ess_dimensionality)?;
    dict.set_item("tle_dimensionality", results.tle_dimensionality)?;
    dict.set_item("gmst_dimensionality", results.gmst_dimensionality)?;
    Ok(dict)
}

/// Spectral-only dimensionality metrics, without the quadratic k-NN/geometry path.
#[pyfunction]
fn compute_spectral<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned = data.as_array().to_owned();
    let results = py
        .detach(|| core_compute_spectral(&owned))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "pca_explained_variance_95",
        results.pca_explained_variance_95 as i64,
    )?;
    dict.set_item("participation_ratio", results.participation_ratio)?;
    dict.set_item("shannon_entropy", results.shannon_entropy)?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_2",
        results.renyi_eff_dimensionality_alpha_2,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_3",
        results.renyi_eff_dimensionality_alpha_3,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_4",
        results.renyi_eff_dimensionality_alpha_4,
    )?;
    dict.set_item(
        "renyi_eff_dimensionality_alpha_5",
        results.renyi_eff_dimensionality_alpha_5,
    )?;
    dict.set_item(
        "geometric_mean_eff_dimensionality",
        results.geometric_mean_eff_dimensionality,
    )?;
    Ok(dict)
}

/// Covariance eigenvalues from the regular centered exact-SVD path.
#[pyfunction]
fn spectral_eigenvalues_exact<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let owned = data.as_array().to_owned();
    let eigenvalues = py
        .detach(|| core_spectral_eigenvalues_exact(&owned))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(eigenvalues.into_pyarray(py))
}

/// Covariance eigenvalues from chunked streaming covariance accumulation.
#[pyfunction]
#[pyo3(signature = (data, chunk_size=4096))]
fn spectral_eigenvalues_streaming<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    chunk_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let eigenvalues = core_spectral_eigenvalues_streaming(data.as_array(), chunk_size)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(eigenvalues.into_pyarray(py))
}

/// Streaming covariance using faer's parallel matrix multiplication.
#[pyfunction]
#[pyo3(signature = (data, chunk_size=4096, threads=0))]
fn spectral_eigenvalues_streaming_faer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    chunk_size: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let eigenvalues =
        core_spectral_eigenvalues_streaming_faer(data.as_array(), chunk_size, threads)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(eigenvalues.into_pyarray(py))
}

/// Seven k-NN-based geometry metrics using externally computed neighbors.
#[pyfunction]
fn compute_geometry_precomputed<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    knn_dist_sq: PyReadonlyArray2<'py, f32>,
    knn_indices: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let data_f32 = data_f32(data);
    let dist_sq = knn_dist_sq.as_array().to_owned();
    let indices_i64 = knn_indices.as_array();
    if dist_sq.nrows() != data_f32.nrows()
        || indices_i64.nrows() != data_f32.nrows()
        || indices_i64.raw_dim() != dist_sq.raw_dim()
    {
        return Err(PyValueError::new_err(
            "precomputed distances and indices must have shape (n_samples, k)",
        ));
    }
    if indices_i64.iter().any(|&index| index < 0) {
        return Err(PyValueError::new_err(
            "precomputed neighbor indices must be non-negative",
        ));
    }
    let indices = indices_i64.mapv(|index| index as usize);
    let k = dist_sq.ncols();

    let (mle, two_nn, danco, mind_mli, mind_mlk, ess, tle) = py.detach(|| {
        (
            geometry::mle_dimensionality(&data_f32, k, Some(&dist_sq)),
            geometry::two_nn_dimensionality(&data_f32, Some(&dist_sq)),
            geometry::danco_dimensionality(&data_f32, k, Some(&dist_sq), Some(&indices)),
            geometry::mind_mli_dimensionality(&data_f32, Some(&dist_sq)),
            geometry::mind_mlk_dimensionality(&data_f32, k, Some(&dist_sq)),
            geometry::ess_dimensionality(&data_f32, k, Some(&dist_sq), Some(&indices)),
            geometry::tle_dimensionality(&data_f32, k, Some(&dist_sq)),
        )
    });

    let dict = PyDict::new(py);
    dict.set_item("mle_dimensionality", mle)?;
    dict.set_item("two_nn_dimensionality", two_nn)?;
    dict.set_item("danco_dimensionality", danco)?;
    dict.set_item("mind_mli_dimensionality", mind_mli)?;
    dict.set_item("mind_mlk_dimensionality", mind_mlk)?;
    dict.set_item("ess_dimensionality", ess)?;
    dict.set_item("tle_dimensionality", tle)?;
    Ok(dict)
}

fn owned_f64_1d(arr: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn data_f32(data: PyReadonlyArray2<'_, f64>) -> ndarray::Array2<f32> {
    data.as_array().mapv(|x| x as f32)
}

fn precomputed_f32(
    precomputed: Option<PyReadonlyArray2<'_, f32>>,
) -> Option<ndarray::Array2<f32>> {
    precomputed.map(|a| a.as_array().to_owned())
}

fn precomputed_indices_usize(
    precomputed: Option<PyReadonlyArray2<'_, i64>>,
) -> Option<ndarray::Array2<usize>> {
    precomputed.map(|a| a.as_array().mapv(|x| x as usize))
}

// --- Metrics (1-D f64 spectra / probabilities) ---

#[pyfunction]
#[pyo3(signature = (spectrum, threshold=0.95))]
fn pca_explained_variance(
    py: Python<'_>,
    spectrum: PyReadonlyArray1<'_, f64>,
    threshold: f64,
) -> u32 {
    let owned = owned_f64_1d(spectrum);
    py.detach(|| metrics::pca_explained_variance(&owned, threshold))
}

#[pyfunction]
fn participation_ratio(py: Python<'_>, spectrum: PyReadonlyArray1<'_, f64>) -> f64 {
    let owned = owned_f64_1d(spectrum);
    py.detach(|| metrics::participation_ratio(&owned))
}

#[pyfunction]
fn shannon_entropy(py: Python<'_>, probabilities: PyReadonlyArray1<'_, f64>) -> f64 {
    let owned = owned_f64_1d(probabilities);
    py.detach(|| metrics::shannon_entropy(&owned))
}

#[pyfunction]
fn renyi_eff_dimensionality(
    py: Python<'_>,
    probabilities: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<f64> {
    let owned = owned_f64_1d(probabilities);
    py.detach(|| metrics::renyi_eff_dimensionality(&owned, alpha))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn geometric_mean_eff_dimensionality(py: Python<'_>, spectrum: PyReadonlyArray1<'_, f64>) -> f64 {
    let owned = owned_f64_1d(spectrum);
    py.detach(|| metrics::geometric_mean_eff_dimensionality(&owned))
}

// --- Geometry / k-NN ---

#[pyfunction]
fn compute_knn_distances<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> Bound<'py, PyArray2<f32>> {
    let data_f32 = data_f32(data);
    let dist = py.detach(|| knn::exact_knn_l2_sq(&data_f32, k).0);
    dist.into_pyarray(py)
}

/// Squared k-NN distances **and** neighbor indices (int64) in one pass.
/// Feed both to `danco_dimensionality` / `ess_dimensionality` to skip
/// their internal k-NN recompute.
#[pyfunction]
fn compute_knn<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>) {
    let data_f32 = data_f32(data);
    let (dist, idx) = py.detach(|| knn::exact_knn_l2_sq(&data_f32, k));
    let idx_i64 = idx.mapv(|x| x as i64);
    (dist.into_pyarray(py), idx_i64.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, k=10, precomputed_knn_dist_sq=None))]
fn mle_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    py.detach(|| geometry::mle_dimensionality(&data_f32, k, pre.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, precomputed_knn_dist_sq=None))]
fn two_nn_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    py.detach(|| geometry::two_nn_dimensionality(&data_f32, pre.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, k=10, precomputed_knn_dist_sq=None, precomputed_indices=None))]
fn danco_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
    precomputed_indices: Option<PyReadonlyArray2<'_, i64>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    let pre_idx = precomputed_indices_usize(precomputed_indices);
    py.detach(|| geometry::danco_dimensionality(&data_f32, k, pre.as_ref(), pre_idx.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, precomputed_knn_dist_sq=None))]
fn mind_mli_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    py.detach(|| geometry::mind_mli_dimensionality(&data_f32, pre.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, k=10, precomputed_knn_dist_sq=None))]
fn mind_mlk_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    py.detach(|| geometry::mind_mlk_dimensionality(&data_f32, k, pre.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, k=10, precomputed_knn_dist_sq=None, precomputed_indices=None))]
fn ess_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
    precomputed_indices: Option<PyReadonlyArray2<'_, i64>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    let pre_idx = precomputed_indices_usize(precomputed_indices);
    py.detach(|| geometry::ess_dimensionality(&data_f32, k, pre.as_ref(), pre_idx.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, k=10, precomputed_knn_dist_sq=None))]
fn tle_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    k: usize,
    precomputed_knn_dist_sq: Option<PyReadonlyArray2<'_, f32>>,
) -> f64 {
    let data_f32 = data_f32(data);
    let pre = precomputed_f32(precomputed_knn_dist_sq);
    py.detach(|| geometry::tle_dimensionality(&data_f32, k, pre.as_ref()))
}

#[pyfunction]
#[pyo3(signature = (data, geodesic=false, random_state=42))]
fn gmst_dimensionality(
    py: Python<'_>,
    data: PyReadonlyArray2<'_, f64>,
    geodesic: bool,
    random_state: u64,
) -> f64 {
    let data_f32 = data_f32(data);
    py.detach(|| geometry::gmst_dimensionality(&data_f32, geodesic, random_state))
}

#[pymodule]
#[pyo3(name = "_native")]
fn effdim_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dim, m)?)?;
    m.add_function(wrap_pyfunction!(compute_spectral, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_eigenvalues_exact, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_eigenvalues_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_eigenvalues_streaming_faer, m)?)?;
    m.add_function(wrap_pyfunction!(compute_geometry_precomputed, m)?)?;
    m.add_function(wrap_pyfunction!(pca_explained_variance, m)?)?;
    m.add_function(wrap_pyfunction!(participation_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(renyi_eff_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_mean_eff_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(compute_knn_distances, m)?)?;
    m.add_function(wrap_pyfunction!(compute_knn, m)?)?;
    m.add_function(wrap_pyfunction!(mle_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(two_nn_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(danco_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(mind_mli_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(mind_mlk_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(ess_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(tle_dimensionality, m)?)?;
    m.add_function(wrap_pyfunction!(gmst_dimensionality, m)?)?;
    Ok(())
}
