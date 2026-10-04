"""II rank: how many independent normal directions the decoder manifold bends in."""
import importlib.util
from pathlib import Path

import numpy as np

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load():
    spec = importlib.util.spec_from_file_location("ii_rank", RUNNERS / "12_ii_rank_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ir = _load()


def _orthonormal(D, k, rng):
    q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return q


def _fixture(D=40, d=4, r=3, seed=0, scale_chart=True):
    """One anchor on the unit sphere: II^S built from r normal directions; radial part -delta x."""
    rng = np.random.default_rng(seed)
    basis = _orthonormal(D, 1 + d + r, rng)
    x, T, N = basis[:, 0], basis[:, 1:1 + d], basis[:, 1 + d:]
    II_on = -np.einsum("a,ij->aij", x, np.eye(d))
    for k in range(r):
        S = rng.standard_normal((d, d)); S = S + S.T
        II_on += np.einsum("a,ij->aij", N[:, k], S)
    R = rng.standard_normal((d, d)) + 3 * np.eye(d) if scale_chart else np.eye(d)
    J = T @ R                                       # a non-orthonormal chart: u = R z
    # Hess in latent coords = II_on pulled back by R, plus a tangent part that II must remove.
    Hess = np.einsum("aij,ip,jq->apq", II_on, R, R) + np.einsum("ak,kpq->apq", T, rng.standard_normal((d, d, d)))
    return J[None], Hess[None], x[None]


def test_sym_flatten_keeps_frobenius_norm():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((5, 5)); S = S + S.T
    v = ir.sym_flatten(S[None, None])[0, 0]
    assert v.shape == (15,)
    assert np.isclose(np.linalg.norm(v), np.linalg.norm(S))


def test_spectrum_metrics_on_known_spectra():
    flat = ir.spectrum_metrics(np.ones(10))
    assert np.isclose(flat["erank"], 10) and np.isclose(flat["pr"], 10) and flat["k90"] == 9 and flat["k99"] == 10
    one = ir.spectrum_metrics(np.array([1.0, 0, 0, 0]))
    assert np.isclose(one["erank"], 1) and np.isclose(one["pr"], 1) and one["k90"] == 1


def test_recovers_rank_and_radial_part_in_a_skewed_chart():
    J, Hess, x = _fixture(r=3)
    out = ir.ii_spectra(J, Hess, x)
    s = out["s_insphere"][0]
    assert s.shape == (10,)                         # d(d+1)/2 for d = 4
    assert (s > 1e-8 * s[0]).sum() == 3             # exactly the r bending directions
    assert out["radial_dev"][0] < 1e-10             # x . II_on = -I on the unit sphere
    s_full = out["s_full"][0]
    assert (s_full > 1e-8 * s_full[0]).sum() == 4   # plus the radial identity direction


def test_full_rank_when_enough_directions():
    J, Hess, x = _fixture(D=60, d=4, r=12, seed=3)
    s = ir.ii_spectra(J, Hess, x)["s_insphere"][0]
    assert (s > 1e-8 * s[0]).sum() == 10


def test_random_reference_is_near_full():
    ref = ir.random_reference(D=768, m=136, seed=0)
    assert ref["erank"] > 100 and ref["k99"] > 120
