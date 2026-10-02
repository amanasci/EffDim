import numpy as np

from sweep import intrinsic_dim as idm


def test_choose_d_rounds_median_and_caps():
    assert idm.choose_d({"mle": 15.0, "two_nn": 16.2, "tle": 16.6, "mind_mlk": 30.0}) == (16, 16)
    assert idm.choose_d({"mle": 23.0, "two_nn": 23.4, "tle": 23.8, "mind_mlk": 40.0}) == (24, 20)
    assert idm.choose_d({"mle": 6.4, "two_nn": 6.5, "tle": 6.7, "mind_mlk": 7.0}) == (7, 7)


def test_estimate_uses_four_estimators_and_one_knn(monkeypatch):
    calls = []
    real = idm.compute_knn_distances
    monkeypatch.setattr(idm, "compute_knn_distances", lambda X, k: calls.append(k) or real(X, k))
    est = idm.estimate(idm.sphere(1500, 3, 20, 0), n_sub=1000)
    assert tuple(est) == idm.ESTIMATORS and calls == [idm.K]
    assert est == idm.estimate(idm.sphere(1500, 3, 20, 0), n_sub=1000)          # seeded subsample


def test_sphere_and_synthetic_check():
    X = idm.sphere(500, 4, 32, 1)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0)
    assert X.shape == (500, 32) and np.linalg.matrix_rank(X) == 5
    res = idm.synthetic_check(true_dims=(4,), Ds=(32,), n=2000)
    (row,) = res["rows"]
    assert (row["true_d"], row["D"], res["n"]) == (4, 32, 2000)
    assert abs(np.median(list(row["estimates"].values())) - 4) < 1.0
    assert row["bias"]["mle"] == row["estimates"]["mle"] - 4
