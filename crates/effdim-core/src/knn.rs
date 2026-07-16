//! Exact float32 squared-L2 k-NN (FAISS IndexFlatL2-compatible contract).
//!
//! Returns squared distances and neighbor indices, excluding self (D-08..D-10).
//! Equal-distance ties prefer the smaller index (D-10).

use ndarray::Array2;

/// Exact brute-force k-NN with squared L2 distances.
///
/// For each query row `i`, neighbors are other rows `j != i` ranked by
/// `(dist_sq ascending, index ascending)`. Effective `k` is capped at
/// `min(k, n_samples - 1)`. Output shape is `(n, k_eff)` for both matrices.
///
/// Does **not** return Euclidean (sqrt) distances.
pub fn exact_knn_l2_sq(data: &Array2<f32>, k: usize) -> (Array2<f32>, Array2<usize>) {
    let n = data.nrows();
    let dims = data.ncols();
    let k_eff = k.min(n.saturating_sub(1));

    if n == 0 || k_eff == 0 {
        return (
            Array2::<f32>::zeros((n, 0)),
            Array2::<usize>::zeros((n, 0)),
        );
    }

    let mut dist_out = Array2::<f32>::zeros((n, k_eff));
    let mut idx_out = Array2::<usize>::zeros((n, k_eff));

    for i in 0..n {
        // (dist_sq, neighbor_index) — exclude self
        let mut neighbors: Vec<(f32, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut dsq = 0.0f32;
            for t in 0..dims {
                let diff = data[[i, t]] - data[[j, t]];
                dsq += diff * diff;
            }
            neighbors.push((dsq, j));
        }

        // Ascending dist_sq; on ties prefer smaller index (D-10)
        neighbors.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        for (rank, &(dsq, j)) in neighbors.iter().take(k_eff).enumerate() {
            dist_out[[i, rank]] = dsq;
            idx_out[[i, rank]] = j;
        }
    }

    (dist_out, idx_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    /// SETUP: three 2-D points — hand-computed sum((x-y)^2) and neighbor order.
    ///
    /// Points: A=(0,0), B=(1,0), C=(0,1)
    /// A→B = 1, A→C = 1 → tie → smaller index B=1 before C=2
    /// B→A = 1, B→C = 2
    /// C→A = 1, C→B = 2
    #[test]
    fn test_exact_knn_hand_computed_squared_l2() {
        let data = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let k = 2usize;
        let (dist_sq, indices) = exact_knn_l2_sq(&data, k);

        assert_eq!(dist_sq.shape(), &[3, 2]);
        assert_eq!(indices.shape(), &[3, 2]);

        // Point 0: neighbors 1 then 2 (equal dist 1.0 — smaller index first)
        assert_relative_eq!(dist_sq[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dist_sq[[0, 1]], 1.0, epsilon = 1e-6);
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(indices[[0, 1]], 2);

        // Point 1: neighbor 0 (dist 1), then 2 (dist 2)
        assert_relative_eq!(dist_sq[[1, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dist_sq[[1, 1]], 2.0, epsilon = 1e-6);
        assert_eq!(indices[[1, 0]], 0);
        assert_eq!(indices[[1, 1]], 2);

        // Point 2: neighbor 0 (dist 1), then 1 (dist 2)
        assert_relative_eq!(dist_sq[[2, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dist_sq[[2, 1]], 2.0, epsilon = 1e-6);
        assert_eq!(indices[[2, 0]], 0);
        assert_eq!(indices[[2, 1]], 1);

        // Self never among indices
        for i in 0..3 {
            for r in 0..k {
                assert_ne!(indices[[i, r]], i);
            }
        }
    }

    /// SETUP: equal-distance tie — four collinear points where query has two
    /// neighbors at identical squared distance; smaller index must win (D-10).
    #[test]
    fn test_equal_distance_tie_prefers_smaller_index() {
        // Query at origin; neighbors at (±1, 0) share dist_sq=1; also (2,0) farther
        let data = array![
            [0.0f32, 0.0], // 0 — query
            [1.0, 0.0],    // 1 — dist 1
            [-1.0, 0.0],   // 2 — dist 1 (tie with 1)
            [2.0, 0.0],    // 3 — dist 4
        ];
        let (dist_sq, indices) = exact_knn_l2_sq(&data, 2);

        // For query 0: first two neighbors at dist 1 must be indices 1 then 2
        assert_relative_eq!(dist_sq[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dist_sq[[0, 1]], 1.0, epsilon = 1e-6);
        assert_eq!(indices[[0, 0]], 1);
        assert_eq!(indices[[0, 1]], 2);
    }

    /// SETUP: shape / self-exclusion / k cap when k >= n-1.
    #[test]
    fn test_shape_excludes_self_and_caps_k() {
        let data = array![[0.0f32, 0.0], [3.0, 4.0], [1.0, 0.0]];
        // Request k=10 but only 2 other points exist
        let (dist_sq, indices) = exact_knn_l2_sq(&data, 10);
        assert_eq!(dist_sq.shape(), &[3, 2]);
        assert_eq!(indices.shape(), &[3, 2]);

        for i in 0..3 {
            for r in 0..2 {
                assert_ne!(indices[[i, r]], i, "self must not appear in neighbor list");
            }
        }

        // Point 0 → (1,0): dist_sq to point 2 = 1; to point 1 = 25
        assert_relative_eq!(dist_sq[[0, 0]], 1.0, epsilon = 1e-6);
        assert_eq!(indices[[0, 0]], 2);
        assert_relative_eq!(dist_sq[[0, 1]], 25.0, epsilon = 1e-6);
        assert_eq!(indices[[0, 1]], 1);
    }
}
