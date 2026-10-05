"""Does the tangent-projected local quadratic fit put II-shaped Hessian into labels that have none?

Toy: d=4 graph manifold x = (z, phi(z)) in R^D, phi(z) = 0.5*N(z,z) + (1/6)*C(z,z,z), anchor at z=0, so the true
tangent is span(e_1..e_4), Gamma = 0 there and the covariant Hessian of a label is its plain z-Hessian.
II at the anchor is N (normal-valued). Labels:
  L0: y = a.z                (true intrinsic Hessian 0)
  L1: y = a.z + 0.5 z'Hz     (H random symmetric, isotropic in Sym^2)
Estimator: same algebra as 09_physics_probe_facing_split_run.local_quadratics (u = dx J g^-1, quad_design, lstsq),
split-half cross products for F_k. Perturbations: (i) the 'decoder' tangent J tilted into the normal space by angle
eps, (ii) off-manifold Gaussian noise, (iii) a density gradient (skewed sampling) around the anchor.
"""
import numpy as np

rng = np.random.default_rng(1)
d, D = 4, 60
m = d * (d + 1) // 2
iu, ju = np.triu_indices(d)
wf = np.where(iu == ju, 1.0, np.sqrt(2.0))


def quad_design(u):
    quad = u[:, iu] * u[:, ju]
    quad[:, iu == ju] *= 0.5
    return np.concatenate([np.ones((len(u), 1)), u, quad], axis=1)


def flat(T):
    return T[..., iu, ju] * wf


def unpack(c):
    B = np.zeros((d, d)); B[iu, ju] = c; B[ju, iu] = c
    return B


# II with a decaying spectrum: N[a, i, j] for normal coordinate a
Nn = D - d
Mflat = rng.standard_normal((Nn, m)) * (0.6 ** np.arange(m))[None, :]   # decaying column scales
Uq, sq, Vq = np.linalg.svd(Mflat, full_matrices=False)
Vq = Vq @ np.linalg.qr(rng.standard_normal((m, m)))[0]                    # random right singular vectors
sq = 3.0 * sq / sq[0]
Mflat = Uq @ np.diag(sq) @ Vq
# unflatten into N (Nn, d, d)
Nt = np.zeros((Nn, d, d))
for a_ in range(Nn):
    c = Mflat[a_] / wf
    Nt[a_] = unpack(c)
C3 = 0.3 * rng.standard_normal((Nn, d, d, d))
C3 = (C3 + C3.transpose(0, 2, 1, 3) + C3.transpose(0, 3, 2, 1) + C3.transpose(0, 1, 3, 2)) / 4
Mtrue = np.stack([flat(Nt[a_]) for a_ in range(Nn)])          # (Nn, m)
_, s, Vt = np.linalg.svd(Mtrue, full_matrices=False)
K = 3
Pk = Vt[:K].T @ Vt[:K]
F_lin_iso = (s[:K] ** 2).sum() / (s ** 2).sum()


def sample(n, skew, sig_off, rng):
    z = rng.standard_normal((n, d)) * 0.35
    if skew:
        keep = rng.random(n) < 1 / (1 + np.exp(-skew * z[:, 0] / 0.35))
        z = z[keep]
    nor = 0.5 * np.einsum("aij,ni,nj->na", Nt, z, z, optimize=True) + np.einsum("aijk,ni,nj,nk->na", C3, z, z, z, optimize=True) / 6
    X = np.concatenate([z, nor], axis=1)
    X = X + sig_off * rng.standard_normal(X.shape)
    return z, X


def tilted_J(eps, rng):
    E = rng.standard_normal((Nn, d)); E *= eps / np.linalg.norm(E, 2)
    return np.concatenate([np.eye(d), E], axis=0)


def fit_h(X, y, idx, J):
    g = J.T @ J; ginv = np.linalg.inv(g)
    u = X[idx] @ J @ ginv                                  # anchor at origin
    coef, *_ = np.linalg.lstsq(quad_design(u), y[idx], rcond=None)
    # convert to orthonormal tangent coords (J = QR)
    _, R = np.linalg.qr(J); Ri = np.linalg.inv(R)
    return flat(Ri.T @ unpack(coef[1 + d:]) @ Ri)


def run(eps, sig_off, skew, label, n=60000, k=2048, reps=24, a_norm=1.0, h_norm=1.0):
    rows = []
    for r in range(reps):
        rr = np.random.default_rng(100 + r)
        z, X = sample(n, skew, sig_off, rr)
        x0 = np.zeros(D)
        idx = np.argsort(((X - x0) ** 2).sum(1))[:k]
        a = rr.standard_normal(d); a *= a_norm / np.linalg.norm(a)
        Hm = rr.standard_normal((d, d)); Hm = (Hm + Hm.T) / 2; Hm *= h_norm / np.linalg.norm(Hm)
        y = z @ a + (0.5 * np.einsum("ni,ij,nj->n", z, Hm, z) if label == "L1" else 0.0) + 0.05 * rr.standard_normal(len(z))
        J = tilted_J(eps, rr)
        perm = rr.permutation(k); A, B = idx[perm[: k // 2]], idx[perm[k // 2:]]
        hA, hB = fit_h(X, y, A, J), fit_h(X, y, B, J)
        h_true = flat(Hm) if label == "L1" else np.zeros(m)
        bias = 0.5 * (hA + hB) - h_true
        rows.append((hA @ Pk @ hB, hA @ hB, bias @ Pk @ bias, bias @ bias, np.linalg.norm(bias), np.linalg.norm(h_true)))
    R_ = np.array(rows)
    F = R_[:, 0].sum() / R_[:, 1].sum()
    Fb = R_[:, 2].sum() / R_[:, 3].sum()
    return F, Fb, np.median(R_[:, 4]), np.median(R_[:, 5])


print(f"m={m}, k={K}: N0 k/m = {K/m:.3f}; isotropic-linear-label F = {F_lin_iso:.3f}; s = {np.round(s[:6],2)}")
for label in ("L0", "L1"):
    for eps, sig, skew in [(0, 0, 0), (0, 0.03, 0), (0, 0, 2.0), (0.1, 0, 0), (0.2, 0, 0), (0.2, 0.03, 2.0), (0.3, 0.03, 0)]:
        F, Fb, nb, nh = run(eps, sig, skew, label)
        print(f"{label} eps={eps:.2f} sig_off={sig:.2f} skew={skew:.1f}: F_k(split-half)={F:.3f}  F_k(bias only)={Fb:.3f}  "
              f"|bias| med {nb:.3f}  |h_true| {nh:.2f}")
