"""Noise-free version of sim_leak: the estimator's systematic (shared-by-both-halves) Hessian error and its F_k."""
import numpy as np
exec(open("sim_leak.py").read().split("def run(")[0])   # reuse setup and helpers

def bias_run(eps, sig_off, skew, a_norm, n=40000, k=2048, reps=16, c3=True):
    out = []
    for r in range(reps):
        rr = np.random.default_rng(500 + r)
        z, X = sample(n, skew, sig_off, rr)
        if not c3:
            X[:, d:] = 0.5 * np.einsum("aij,ni,nj->na", Nt, z, z, optimize=True) + sig_off * rr.standard_normal((len(z), Nn))
        idx = np.argsort((X ** 2).sum(1))[:k]
        a = rr.standard_normal(d); a *= a_norm / np.linalg.norm(a)
        y = z @ a                                      # true intrinsic Hessian at the anchor = 0
        J = tilted_J(eps, rr)
        h = fit_h(X, y, idx, J)
        out.append((h @ Pk @ h, h @ h))
    o = np.array(out)
    return o[:, 0].sum() / o[:, 1].sum(), np.sqrt(np.median(o[:, 1]))

print(f"N0 k/m = {K/m:.3f}; isotropic linear label F = {F_lin_iso:.3f}; II top singular value s1 = {s[0]:.2f}")
print("label y = a.z, true Hessian 0; |a| = 1; report F_k of the estimated (pure bias) Hessian and its norm")
for eps, sig, skew, c3 in [(0, 0, 0, True), (0, 0, 2.0, True), (0, 0.03, 0, True), (0, 0.03, 2.0, True),
                           (0.05, 0, 0, False), (0.1, 0, 0, False), (0.2, 0, 0, False), (0.1, 0, 0, True), (0.2, 0.03, 2.0, True)]:
    F, nb = bias_run(eps, sig, skew, 1.0, c3=c3)
    print(f"eps={eps:.2f} sig_off={sig:.2f} skew={skew:.1f} cubic={c3}: F_k(bias)={F:.3f}  |bias| med {nb:.3f}  (eps*s1 = {eps*s[0]:.2f})")
