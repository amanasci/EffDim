//! PyO3 bindings for `effdim._native` (round-trip stub + full compute_dim dict).

use effdim_core::{compute_dim as core_compute_dim, identity_f64_slice};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
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

#[pymodule]
#[pyo3(name = "_native")]
fn effdim_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip_array, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dim, m)?)?;
    Ok(())
}
