//! PyO3 bindings for `effdim._native` (Phase 2 NumPy round-trip stub).

use effdim_core::identity_f64_slice;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

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

#[pymodule]
#[pyo3(name = "_native")]
fn effdim_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(roundtrip_array, m)?)?;
    Ok(())
}
