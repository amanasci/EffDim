"""Toy check of the corrected proposition: t* identities for variants S (data-side) and S_model (model quadratic)."""
import numpy as np
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
n, d, D = 20000, 2, 12
z = rng.uniform(-1, 1, (n, d))
B = rng.standard_normal((D - d, d, d)) * 0.5
X = np.concatenate([z, 0.5 * np.einsum("aij,ni,nj->na", B, z, z)], 1) + 0.05 * rng.standard_normal((n, D))
y = np.sin(2 * z[:, 0]) + z[:, 1] ** 2 + 0.2 * rng.standard_normal(n)
def tstars(Xn, yn, w, x0, J, II):
    ginv = np.linalg.inv(J.T @ J); P = J @ ginv @ J.T
    wT = P @ w; wS = w - wT
    u = (Xn - x0) @ J @ ginv
    q = 0.5 * np.einsum("ki,ij,kj->k", u, np.einsum("aij,a->ij", II, wS), u)
    e0 = yn - Xn @ wT; e0 -= e0.mean(); p = Xn @ wS; p -= p.mean(); q -= q.mean()
    r = yn - Xn @ w; r -= r.mean()
    return e0 @ p / (p @ p), e0 @ q / (q @ q), 1 + (r + p - q) @ q / (q @ q), 1 + r @ p / (p @ p)
# anchor at origin: tangent = first d coords, II = [0; B]
J = np.zeros((D, d)); J[:d, :d] = np.eye(d); II = np.concatenate([np.zeros((d, d, d)), B], 0)
nb = np.argsort(np.linalg.norm(z, axis=1))[:800]
x0 = np.zeros(D)
# local OLS
loc = Ridge(alpha=1e-10).fit(X[nb], y[nb]); w = loc.coef_
print("local LS:  t*_S %.6f  t*_model %.4f  identity check %.4f" % tstars(X[nb], y[nb], w, x0, J, II)[:3])
for a in (1e-6, 1.0, 100.0):
    g = Ridge(alpha=a).fit(X, y); w = g.coef_
    tS, tM, tMid, tSid = tstars(X[nb], y[nb], w, x0, J, II)
    wS = w - J @ np.linalg.inv(J.T @ J) @ J.T @ w; Xc = X - X.mean(0); r = y - g.predict(X)
    pooled = 1 + (r @ (Xc @ wS)) / ((Xc @ wS) @ (Xc @ wS)); pred = 1 + a * (wS @ wS) / ((Xc @ wS) @ (Xc @ wS))
    print(f"global ridge a={a:g}: per-anchor t*_S {tS:.3f} t*_model {tM:.3f} | pooled t*_S {pooled:.4f} vs 1+a|wS|^2/|XwS|^2 {pred:.4f}")
