//! Geometric intrinsic-dimension estimators (1:1 ports of `geometry.py`).
//!
//! Soft-return `0.0` on undersized inputs (D-13). Shared k-NN distances/indices
//! avoid a second search for DANCo/ESS (D-08, D-09).

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;

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

/// Dense pairwise Euclidean distance matrix (squareform(pdist)).
fn pairwise_euclidean(data: &Array2<f32>) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut dist = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0f64;
            for t in 0..d {
                let diff = data[[i, t]] as f64 - data[[j, t]] as f64;
                s += diff * diff;
            }
            let e = s.sqrt();
            dist[[i, j]] = e;
            dist[[j, i]] = e;
        }
    }
    dist
}

/// k-NN graph in distance mode, then sum-symmetrize (graph + graph.T) — D-11/D-13.
fn knn_graph_sum_symmetrize(data: &Array2<f32>, k_geo: usize) -> Array2<f64> {
    let n = data.nrows();
    let d = data.ncols();
    let mut graph = Array2::<f64>::zeros((n, n));
    if n == 0 || k_geo == 0 {
        return graph;
    }
    let (_dist_sq, indices) = exact_knn_l2_sq(data, k_geo);
    for i in 0..n {
        for r in 0..indices.ncols() {
            let j = indices[[i, r]];
            let mut s = 0.0f64;
            for t in 0..d {
                let diff = data[[i, t]] as f64 - data[[j, t]] as f64;
                s += diff * diff;
            }
            graph[[i, j]] = s.sqrt();
        }
    }
    // Sum-symmetrize: graph + graph.T (not min)
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = graph[[i, j]] + graph[[j, i]];
        }
    }
    sym
}

/// Floyd–Warshall on a dense graph. Missing edges (0 off-diagonal) treated as ∞.
fn floyd_warshall(graph: &Array2<f64>) -> Array2<f64> {
    let n = graph.nrows();
    let mut dist = Array2::<f64>::from_elem((n, n), f64::INFINITY);
    for i in 0..n {
        dist[[i, i]] = 0.0;
        for j in 0..n {
            if i != j && graph[[i, j]] > 0.0 {
                dist[[i, j]] = graph[[i, j]];
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = dist[[i, k]];
            if !dik.is_finite() {
                continue;
            }
            for j in 0..n {
                let cand = dik + dist[[k, j]];
                if cand < dist[[i, j]] {
                    dist[[i, j]] = cand;
                }
            }
        }
    }
    // Replace remaining inf with max_finite * 10 (Python geodesic path)
    let mut max_finite = 0.0f64;
    let mut any_finite = false;
    for i in 0..n {
        for j in 0..n {
            let v = dist[[i, j]];
            if v.is_finite() {
                any_finite = true;
                if v > max_finite {
                    max_finite = v;
                }
            }
        }
    }
    let fill = if any_finite { max_finite * 10.0 } else { 1.0 * 10.0 };
    for i in 0..n {
        for j in 0..n {
            if !dist[[i, j]].is_finite() {
                dist[[i, j]] = fill;
            }
        }
    }
    dist
}

/// Dense Prim MST total edge weight (matches SciPy minimum_spanning_tree(...).sum()).
fn prim_mst_length(dist: &Array2<f64>) -> f64 {
    let n = dist.nrows();
    if n < 2 {
        return 0.0;
    }
    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    min_edge[0] = 0.0;
    let mut total = 0.0f64;

    for _ in 0..n {
        let mut u = None;
        let mut best = f64::INFINITY;
        for i in 0..n {
            if !in_tree[i] && min_edge[i] < best {
                best = min_edge[i];
                u = Some(i);
            }
        }
        let u = match u {
            Some(u) => u,
            None => break,
        };
        in_tree[u] = true;
        if best.is_finite() {
            total += best;
        }
        for v in 0..n {
            if !in_tree[v] {
                let w = dist[[u, v]];
                if w < min_edge[v] {
                    min_edge[v] = w;
                }
            }
        }
    }
    total
}

/// GMST intrinsic dimensionality (Euclidean or geodesic).
///
/// `random_state` knobs match Python (default 42); bands are the gate (D-14).
pub fn gmst_dimensionality(
    data: &Array2<f32>,
    geodesic: bool,
    random_state: u64,
) -> f64 {
    let n_samples = data.nrows();
    if n_samples < 10 {
        return 0.0;
    }

    let mut sizes: Vec<usize> = [
        (n_samples / 8).max(4),
        (n_samples / 4).max(4),
        (n_samples / 2).max(4),
        n_samples,
    ]
    .into_iter()
    .collect();
    sizes.sort_unstable();
    sizes.dedup();

    if sizes.len() < 2 {
        return 0.0;
    }

    let mut rng = StdRng::seed_from_u64(random_state);
    let mut log_n_list: Vec<f64> = Vec::new();
    let mut log_l_list: Vec<f64> = Vec::new();

    for &size_raw in &sizes {
        let size = size_raw.min(n_samples);
        let idx: Vec<usize> = if size == n_samples {
            (0..n_samples).collect()
        } else {
            sample(&mut rng, n_samples, size).into_vec()
        };

        let mut subsample = Array2::<f32>::zeros((size, data.ncols()));
        for (row, &i) in idx.iter().enumerate() {
            for c in 0..data.ncols() {
                subsample[[row, c]] = data[[i, c]];
            }
        }

        let dist_matrix = if geodesic {
            let k_geo = 10.min(size.saturating_sub(1));
            let graph = knn_graph_sum_symmetrize(&subsample, k_geo);
            floyd_warshall(&graph)
        } else {
            pairwise_euclidean(&subsample)
        };

        let l = prim_mst_length(&dist_matrix);
        if l > 0.0 {
            log_n_list.push((size as f64).ln());
            log_l_list.push(l.ln());
        }
    }

    if log_n_list.len() < 2 {
        return 0.0;
    }

    let m = log_n_list.len() as f64;
    let mean_x: f64 = log_n_list.iter().sum::<f64>() / m;
    let mean_y: f64 = log_l_list.iter().sum::<f64>() / m;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..log_n_list.len() {
        let dx = log_n_list[i] - mean_x;
        let dy = log_l_list[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    let alpha = num / (den + EPS);

    if (1.0 - alpha).abs() < EPS {
        return 0.0;
    }
    1.0 / (1.0 - alpha)
}

#[cfg(test)]
mod inventory_a_tests {
    //! Inventory A: direct geometry #[test] mirrors (D-12). SETUP comments cite pytest.

    use super::*;
    use ndarray::{array, Array2};

    /// Deterministic standard-normal fills for SETUP parity (not bit-identical to NumPy).
    fn seeded_randn(seed: u64, nrows: usize, ncols: usize) -> Array2<f32> {
        let mut state = seed.max(1);
        let mut next_u64 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            state
        };
        let mut next_unit = || (next_u64() as f64) / (u64::MAX as f64);
        let mut out = Array2::<f32>::zeros((nrows, ncols));
        let mut i = 0;
        let total = nrows * ncols;
        while i < total {
            let u1 = next_unit().max(1e-12);
            let u2 = next_unit();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let z0 = (r * theta.cos()) as f32;
            let z1 = (r * theta.sin()) as f32;
            out.as_slice_mut().unwrap()[i] = z0;
            i += 1;
            if i < total {
                out.as_slice_mut().unwrap()[i] = z1;
                i += 1;
            }
        }
        out
    }

    /// Uniform [lo, hi) via same LCG stream as seeded_randn unit draws.
    fn seeded_uniform(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut state = seed.max(1);
        let mut next_u64 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            state
        };
        let span = hi - lo;
        (0..n)
            .map(|_| {
                let u = (next_u64() as f64) / (u64::MAX as f64);
                lo + span * (u as f32)
            })
            .collect()
    }

    // --- tests/test_input_validation.py::TestGeometricEstimatorsInputEdgeCases ---

    /// SETUP: test_input_validation::test_mle_single_sample_returns_zero — shape=(1,3), k=1
    #[test]
    fn test_mle_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(mle_dimensionality(&data, 1, None), 0.0);
    }

    /// SETUP: test_input_validation::test_two_nn_two_samples_returns_zero — shape=(2,2)
    #[test]
    fn test_two_nn_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(two_nn_dimensionality(&data, None), 0.0);
    }

    /// SETUP: test_input_validation::test_danco_two_samples_returns_zero — shape=(2,2)
    #[test]
    fn test_danco_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(danco_dimensionality(&data, 10, None, None), 0.0);
    }

    /// SETUP: test_input_validation::test_mind_mli_two_samples_returns_zero — shape=(2,2)
    #[test]
    fn test_mind_mli_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(mind_mli_dimensionality(&data, None), 0.0);
    }

    /// SETUP: test_input_validation::test_mind_mlk_single_sample_returns_zero — shape=(1,3)
    #[test]
    fn test_mind_mlk_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(mind_mlk_dimensionality(&data, 10, None), 0.0);
    }

    /// SETUP: test_input_validation::test_ess_two_samples_returns_zero — shape=(2,2)
    #[test]
    fn test_ess_two_samples_returns_zero() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(ess_dimensionality(&data, 10, None, None), 0.0);
    }

    /// SETUP: test_input_validation::test_tle_single_sample_returns_zero — shape=(1,3)
    #[test]
    fn test_tle_single_sample_returns_zero() {
        let data = array![[1.0f32, 2.0, 3.0]];
        assert_eq!(tle_dimensionality(&data, 10, None), 0.0);
    }

    /// SETUP: test_input_validation::test_gmst_small_dataset_returns_zero — n=9, seed=0
    #[test]
    fn test_gmst_small_dataset_returns_zero() {
        let data = seeded_randn(0, 9, 3);
        assert_eq!(gmst_dimensionality(&data, false, 42), 0.0);
    }

    /// SETUP: test_input_validation::test_mle_positive_for_reasonable_data — seed=0, (100,5), k=5
    #[test]
    fn test_mle_positive_for_reasonable_data() {
        let data = seeded_randn(0, 100, 5);
        assert!(mle_dimensionality(&data, 5, None) > 0.0);
    }

    /// SETUP: test_input_validation::test_two_nn_positive_for_reasonable_data — seed=0, (100,5)
    #[test]
    fn test_two_nn_positive_for_reasonable_data() {
        let data = seeded_randn(0, 100, 5);
        assert!(two_nn_dimensionality(&data, None) > 0.0);
    }

    // --- tests/test_new_geometric_estimators.py::TestDANCo ---

    /// SETUP: TestDANCo::test_high_dimensional_gaussian — seed=42, shape=(200,10), k=10
    #[test]
    fn test_danco_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let dim = danco_dimensionality(&data, 10, None, None);
        assert!(dim.is_finite() && dim > 0.0, "DANCo returned {dim}");
    }

    /// SETUP: TestDANCo::test_low_dimensional_manifold — seed=42, n=200 line in 3D, noise=1e-6, k=5
    #[test]
    fn test_danco_low_dimensional_manifold() {
        let t = seeded_uniform(42, 200, 0.0, 10.0);
        let noise = seeded_randn(43, 200, 3);
        let mut data = Array2::<f32>::zeros((200, 3));
        for i in 0..200 {
            data[[i, 0]] = t[i] + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = 2.0 * t[i] + 1e-6 * noise[[i, 1]];
            data[[i, 2]] = 3.0 * t[i] + 1e-6 * noise[[i, 2]];
        }
        let dim = danco_dimensionality(&data, 5, None, None);
        assert!(dim.is_finite() && dim > 0.0, "DANCo returned {dim}");
    }

    /// SETUP: TestDANCo::test_small_dataset — fixed shape=(2,5), exact == 0.0
    #[test]
    fn test_danco_small_dataset() {
        let data = array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        assert_eq!(danco_dimensionality(&data, 10, None, None), 0.0);
    }

    /// SETUP: TestDANCo::test_with_precomputed_knn — seed=42, shape=(50,5), k=5
    #[test]
    fn test_danco_with_precomputed_knn() {
        let data = seeded_randn(42, 50, 5);
        let (dist_sq, indices) = exact_knn_l2_sq(&data, 5);
        let dim = danco_dimensionality(&data, 5, Some(&dist_sq), Some(&indices));
        assert!(dim.is_finite() && dim > 0.0);
    }

    /// SETUP: TestDANCo::test_duplicate_points — fixed (6,2), k=2
    #[test]
    fn test_danco_duplicate_points() {
        let data = array![
            [1.0f32, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        let dim = danco_dimensionality(&data, 2, None, None);
        assert!(dim.is_finite());
    }

    // --- TestMiNDMLi ---

    /// SETUP: TestMiNDMLi::test_high_dimensional_gaussian — seed=42, shape=(200,10)
    #[test]
    fn test_mind_mli_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let dim = mind_mli_dimensionality(&data, None);
        assert!(dim.is_finite() && dim > 0.0, "MiND-MLi returned {dim}");
    }

    /// SETUP: TestMiNDMLi::test_1d_manifold — seed=42, shape=(200,2) linspace line, noise=1e-6
    #[test]
    fn test_mind_mli_1d_manifold() {
        let noise = seeded_randn(42, 200, 2);
        let mut data = Array2::<f32>::zeros((200, 2));
        for i in 0..200 {
            let t = 10.0 * (i as f32) / 199.0;
            data[[i, 0]] = t + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = 2.0 * t + 1e-6 * noise[[i, 1]];
        }
        let dim = mind_mli_dimensionality(&data, None);
        assert!(dim.is_finite() && dim > 0.0, "MiND-MLi got {dim} for 1D data");
    }

    /// SETUP: TestMiNDMLi::test_small_dataset — fixed shape=(2,5), exact == 0.0
    #[test]
    fn test_mind_mli_small_dataset() {
        let data = array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        assert_eq!(mind_mli_dimensionality(&data, None), 0.0);
    }

    /// SETUP: TestMiNDMLi::test_identical_distances — n=50 circle
    #[test]
    fn test_mind_mli_identical_distances() {
        let n = 50usize;
        let mut data = Array2::<f32>::zeros((n, 2));
        for i in 0..n {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
            data[[i, 0]] = theta.cos();
            data[[i, 1]] = theta.sin();
        }
        assert!(mind_mli_dimensionality(&data, None).is_finite());
    }

    /// SETUP: TestMiNDMLi::test_with_precomputed_knn — seed=42, shape=(50,5)
    #[test]
    fn test_mind_mli_with_precomputed_knn() {
        let data = seeded_randn(42, 50, 5);
        let (dist_sq, _) = exact_knn_l2_sq(&data, 5);
        let dim = mind_mli_dimensionality(&data, Some(&dist_sq));
        assert!(dim.is_finite() && dim > 0.0);
    }

    // --- TestMiNDMLk ---

    /// SETUP: TestMiNDMLk::test_high_dimensional_gaussian — seed=42, shape=(200,10), k=10
    #[test]
    fn test_mind_mlk_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let dim = mind_mlk_dimensionality(&data, 10, None);
        assert!(5.0 < dim && dim < 15.0, "MiND-MLk got {dim}, expected ~10");
    }

    /// SETUP: TestMiNDMLk::test_2d_manifold — seed=42, n=200 plane z=x+y, noise=1e-6, k=5
    #[test]
    fn test_mind_mlk_2d_manifold() {
        let x = seeded_uniform(42, 200, -5.0, 5.0);
        let y = seeded_uniform(43, 200, -5.0, 5.0);
        let noise = seeded_randn(44, 200, 3);
        let mut data = Array2::<f32>::zeros((200, 3));
        for i in 0..200 {
            data[[i, 0]] = x[i] + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = y[i] + 1e-6 * noise[[i, 1]];
            data[[i, 2]] = x[i] + y[i] + 1e-6 * noise[[i, 2]];
        }
        let dim = mind_mlk_dimensionality(&data, 5, None);
        assert!(1.0 < dim && dim < 3.5, "MiND-MLk got {dim} for 2D manifold");
    }

    /// SETUP: TestMiNDMLk::test_small_dataset — fixed shape=(1,5), exact == 0.0
    #[test]
    fn test_mind_mlk_small_dataset() {
        let data = array![[1.0f32, 2.0, 3.0, 4.0, 5.0]];
        assert_eq!(mind_mlk_dimensionality(&data, 10, None), 0.0);
    }

    /// SETUP: TestMiNDMLk::test_robustness_vs_mle — seed=42, shape=(200,5), k=10
    #[test]
    fn test_mind_mlk_robustness_vs_mle() {
        let data = seeded_randn(42, 200, 5);
        let mle = mle_dimensionality(&data, 10, None);
        let mlk = mind_mlk_dimensionality(&data, 10, None);
        assert!((mle - mlk).abs() < 5.0, "MLE={mle}, MLk={mlk}");
    }

    // --- TestESS ---

    /// SETUP: TestESS::test_high_dimensional_gaussian — seed=42, shape=(200,10), k=10
    #[test]
    fn test_ess_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let dim = ess_dimensionality(&data, 10, None, None);
        assert!(dim.is_finite() && dim > 0.0, "ESS returned {dim}");
    }

    /// SETUP: TestESS::test_low_dimensional_structure — seed=42, shape=(200,2) line, noise=1e-6, k=5
    #[test]
    fn test_ess_low_dimensional_structure() {
        let noise = seeded_randn(42, 200, 2);
        let mut data = Array2::<f32>::zeros((200, 2));
        for i in 0..200 {
            let t = 10.0 * (i as f32) / 199.0;
            data[[i, 0]] = t + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = 2.0 * t + 1e-6 * noise[[i, 1]];
        }
        let dim = ess_dimensionality(&data, 5, None, None);
        assert!(dim.is_finite() && dim > 0.0);
    }

    /// SETUP: TestESS::test_small_dataset — fixed shape=(2,5), exact == 0.0
    #[test]
    fn test_ess_small_dataset() {
        let data = array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        assert_eq!(ess_dimensionality(&data, 10, None, None), 0.0);
    }

    /// SETUP: TestESS::test_duplicate_points — fixed (6,2), k=2
    #[test]
    fn test_ess_duplicate_points() {
        let data = array![
            [1.0f32, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        assert!(ess_dimensionality(&data, 2, None, None).is_finite());
    }

    // --- TestTLE ---

    /// SETUP: TestTLE::test_high_dimensional_gaussian — seed=42, shape=(200,10), k=10
    #[test]
    fn test_tle_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let dim = tle_dimensionality(&data, 10, None);
        assert!(5.0 < dim && dim < 15.0, "TLE got {dim}, expected ~10");
    }

    /// SETUP: TestTLE::test_2d_manifold — seed=42, n=200 plane z=x+y, noise=1e-6, k=5
    #[test]
    fn test_tle_2d_manifold() {
        let x = seeded_uniform(42, 200, -5.0, 5.0);
        let y = seeded_uniform(43, 200, -5.0, 5.0);
        let noise = seeded_randn(44, 200, 3);
        let mut data = Array2::<f32>::zeros((200, 3));
        for i in 0..200 {
            data[[i, 0]] = x[i] + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = y[i] + 1e-6 * noise[[i, 1]];
            data[[i, 2]] = x[i] + y[i] + 1e-6 * noise[[i, 2]];
        }
        let dim = tle_dimensionality(&data, 5, None);
        assert!(1.0 < dim && dim < 3.5, "TLE got {dim} for 2D manifold");
    }

    /// SETUP: TestTLE::test_small_dataset — fixed shape=(1,5), exact == 0.0
    #[test]
    fn test_tle_small_dataset() {
        let data = array![[1.0f32, 2.0, 3.0, 4.0, 5.0]];
        assert_eq!(tle_dimensionality(&data, 10, None), 0.0);
    }

    /// SETUP: TestTLE::test_equivalent_to_mle — seed=42, shape=(200,5), k=10
    #[test]
    fn test_tle_equivalent_to_mle() {
        let data = seeded_randn(42, 200, 5);
        let mle = mle_dimensionality(&data, 10, None);
        let tle = tle_dimensionality(&data, 10, None);
        assert!((mle - tle).abs() < 1.0, "MLE={mle}, TLE={tle}");
    }

    // --- TestGMST (Euclidean units + Rust geodesic smoke; pytest geodesic stays Inventory C) ---

    /// SETUP: TestGMST::test_high_dimensional_gaussian — seed=42, shape=(100,5)
    #[test]
    fn test_gmst_high_dimensional_gaussian() {
        let data = seeded_randn(42, 100, 5);
        let dim = gmst_dimensionality(&data, false, 42);
        assert!(dim.is_finite() && dim > 0.0, "GMST returned {dim}");
    }

    /// SETUP: TestGMST::test_2d_data — seed=42, shape=(100,2)
    #[test]
    fn test_gmst_2d_data() {
        let data = seeded_randn(42, 100, 2);
        let dim = gmst_dimensionality(&data, false, 42);
        assert!(dim.is_finite() && dim > 0.0, "GMST returned {dim}");
    }

    /// SETUP: TestGMST::test_small_dataset — fixed shape=(5,3), exact == 0.0
    #[test]
    fn test_gmst_small_dataset_fixed() {
        let data = array![
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        assert_eq!(gmst_dimensionality(&data, false, 42), 0.0);
    }

    /// SETUP: TestGMST::test_geodesic_mode — seed=42, shape=(50,5), geodesic=true (D-11 Rust smoke)
    #[test]
    fn test_gmst_geodesic_mode() {
        let data = seeded_randn(42, 50, 5);
        let dim = gmst_dimensionality(&data, true, 42);
        assert!(dim.is_finite(), "GMST geodesic returned {dim}");
    }

    /// SETUP: TestGMST::test_geodesic_vs_euclidean — seed=42, shape=(50,3)
    #[test]
    fn test_gmst_geodesic_vs_euclidean() {
        let data = seeded_randn(42, 50, 3);
        let dim_e = gmst_dimensionality(&data, false, 42);
        let dim_g = gmst_dimensionality(&data, true, 42);
        assert!(dim_e.is_finite() && dim_g.is_finite());
    }

    // --- tests/test_numerical_stability.py::TestGeometricEstimatorsEdgeCases ---

    /// SETUP: test_perfect_line_2d — seed=42, shape=(100,2) y=2x, noise=1e-8, mle k=5
    #[test]
    fn test_perfect_line_2d() {
        let noise = seeded_randn(42, 100, 2);
        let mut data = Array2::<f32>::zeros((100, 2));
        for i in 0..100 {
            let t = 10.0 * (i as f32) / 99.0;
            data[[i, 0]] = t + 1e-8 * noise[[i, 0]];
            data[[i, 1]] = 2.0 * t + 1e-8 * noise[[i, 1]];
        }
        let mle_dim = mle_dimensionality(&data, 5, None);
        let two_nn_dim = two_nn_dimensionality(&data, None);
        assert!(0.5 < mle_dim && mle_dim < 2.0, "MLE got {mle_dim}");
        assert!(two_nn_dim > 0.0, "Two-NN got {two_nn_dim}");
    }

    /// SETUP: test_perfect_plane_3d — seed=42, n=100 plane z=x+y, noise=1e-8, mle k=5
    #[test]
    fn test_perfect_plane_3d() {
        let x = seeded_uniform(42, 100, -5.0, 5.0);
        let y = seeded_uniform(43, 100, -5.0, 5.0);
        let noise = seeded_randn(44, 100, 3);
        let mut data = Array2::<f32>::zeros((100, 3));
        for i in 0..100 {
            data[[i, 0]] = x[i] + 1e-8 * noise[[i, 0]];
            data[[i, 1]] = y[i] + 1e-8 * noise[[i, 1]];
            data[[i, 2]] = x[i] + y[i] + 1e-8 * noise[[i, 2]];
        }
        let mle_dim = mle_dimensionality(&data, 5, None);
        let two_nn_dim = two_nn_dimensionality(&data, None);
        assert!(1.0 < mle_dim && mle_dim < 3.5, "MLE got {mle_dim}");
        assert!(1.0 < two_nn_dim && two_nn_dim < 3.5, "Two-NN got {two_nn_dim}");
    }

    /// SETUP: test_very_small_dataset — fixed (2,5)==0.0; seed=42 shape=(3,5) n=3 path
    #[test]
    fn test_very_small_dataset() {
        let data2 = array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        assert_eq!(mle_dimensionality(&data2, 5, None), 0.0);
        assert_eq!(two_nn_dimensionality(&data2, None), 0.0);
        let data3 = seeded_randn(42, 3, 5);
        assert!(mle_dimensionality(&data3, 2, None) >= 0.0);
        assert!(two_nn_dimensionality(&data3, None) >= 0.0);
    }

    /// SETUP: test_identical_points — fixed (6,2) duplicates, k=2
    #[test]
    fn test_identical_points() {
        let data = array![
            [1.0f32, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        assert!(mle_dimensionality(&data, 2, None).is_finite());
        assert!(two_nn_dimensionality(&data, None).is_finite());
    }

    /// SETUP: test_high_dimensional_gaussian — seed=42, shape=(200,10), mle k=10
    #[test]
    fn test_stability_high_dimensional_gaussian() {
        let data = seeded_randn(42, 200, 10);
        let mle_dim = mle_dimensionality(&data, 10, None);
        let two_nn_dim = two_nn_dimensionality(&data, None);
        assert!(5.0 < mle_dim && mle_dim < 15.0, "MLE={mle_dim}");
        assert!(5.0 < two_nn_dim && two_nn_dim < 15.0, "TwoNN={two_nn_dim}");
    }

    // --- tests/test_known_dimensionalities.py direct geometry ---

    /// SETUP: TestKnownDimensionalities::test_1d_curve_in_3d — helix n=500, noise=1e-6, mle k=5
    #[test]
    fn test_1d_curve_in_3d_helix() {
        let noise = seeded_randn(42, 500, 3);
        let mut data = Array2::<f32>::zeros((500, 3));
        for i in 0..500 {
            let t = 4.0 * std::f32::consts::PI * (i as f32) / 499.0;
            data[[i, 0]] = t.cos() + 1e-6 * noise[[i, 0]];
            data[[i, 1]] = t.sin() + 1e-6 * noise[[i, 1]];
            data[[i, 2]] = t / (4.0 * std::f32::consts::PI) + 1e-6 * noise[[i, 2]];
        }
        let mle = mle_dimensionality(&data, 5, None);
        let two_nn = two_nn_dimensionality(&data, None);
        assert!(0.5 < mle && mle < 2.5, "MLE got {mle} for 1D helix");
        assert!(two_nn > 0.0, "Two-NN got {two_nn} for 1D helix");
    }
}
