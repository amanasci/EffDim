"""Noise covariance of the split-half OLS Hessian (quad_design) for a kNN-ball design in d = 16:
how much of the estimation noise sits on the trace (identity) direction of Sym^2?"""
import numpy as np
rng = np.random.default_rng(0)
d = 16; m = d * (d + 1) // 2; n = 1024
iu, ju = np.triu_indices(d)
wf = np.where(iu == ju, 1.0, np.sqrt(2.0))
def quad_design(u):
    q = u[:, iu] * u[:, ju]; q[:, iu == ju] *= 0.5
    return np.concatenate([np.ones((len(u), 1)), u, q], 1)
for name, gen in [("uniform ball", lambda: (lambda g: g / np.linalg.norm(g, axis=1, keepdims=True) * rng.random((n, 1)) ** (1 / d))(rng.standard_normal((n, d)))),
                  ("gaussian (no ball)", lambda: rng.standard_normal((n, d)) / np.sqrt(d))]:
    covs = []
    for _ in range(20):
        A = quad_design(gen())
        C = np.linalg.inv(A.T @ A)[1 + d:, 1 + d:]
        # coefficient c (on design columns) -> flattened Frobenius coords: h_flat = c * wf  (B_ij = c_ij)
        Wd = np.diag(wf); covs.append(Wd @ C @ Wd)
    C = np.mean(covs, 0)
    eI = (iu == ju).astype(float) / np.sqrt(d)
    tr_var = eI @ C @ eI; avg = np.trace(C) / m
    ev = np.linalg.eigvalsh(C)
    print(f"{name}: noise variance along trace direction / average per direction = {tr_var/avg:.1f}; "
          f"share of total noise on trace = {tr_var/np.trace(C):.3f}; top eigen share {ev[-1]/ev.sum():.3f}")
