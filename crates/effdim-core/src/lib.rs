//! Pure Rust compute core (placeholder for Phase 3+ ports).

/// Identity over an `f64` slice — keeps the path dependency live from the PyO3 stub.
pub fn identity_f64_slice(xs: &[f64]) -> Vec<f64> {
    xs.to_vec()
}
