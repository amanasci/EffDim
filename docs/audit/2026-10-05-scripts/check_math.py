"""Independent checks of label-Hessian fit, emp-mismatch linearity, counterfactual SSE identity, sphere split."""
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/akagi/Documents/Projects/EffDim/curvature-experiment")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "runners"))
def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / "runners" / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pfs = load("09_physics_probe_facing_split_run.py", "pfs")
ns = load("09_physics_normal_scaling_run.py", "ns")
rng = np.random.default_rng(0)

# ---- 1. label Hessian on an exact second-order graph surface in R^D (flat ambient, no sphere) -------------
d, D, k = 3, 10, 2000
Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
T, N = Q[:, :d], Q[:, d:]                      # orthonormal tangent / normal frames
B = rng.standard_normal((D - d, d, d)); B = 0.5 * (B + B.transpose(0, 2, 1))  # II components in normal frame
def embed(u):  # x(u) = T u + 1/2 N B(u,u)   (Monge chart)
    return u @ T.T + 0.5 * (np.einsum("aij,ki,kj->ka", B, u, u) @ N.T)
u = rng.standard_normal((k, d)) * 0.05
X = np.vstack([np.zeros((1, D)), embed(u)])
Hy = rng.standard_normal((d, d)); Hy = 0.5 * (Hy + Hy.T); gy = rng.standard_normal(d)
uu = np.vstack([np.zeros((1, d)), u])
y = 1.0 + uu @ gy + 0.5 * np.einsum("ki,ij,kj->k", uu, Hy, uu)      # intrinsic Hessian at 0 is Hy (Christoffel = 0 at Monge origin)
J = T.copy(); g = J.T @ J
geo = {"J": J[None], "g": g[None], "ginv": np.linalg.inv(g)[None]}
neigh = np.arange(k + 1)[None]
w = rng.standard_normal(D)
lq = pfs.local_quadratics(X, np.array([0]), neigh, geo, {"y": y, "p": X @ w, "r": y - X @ w}, 32)
print("1a label Hessian rel err (Monge chart, exact quadratic label):",
      np.linalg.norm(lq["hess"]["y"][0] - Hy) / np.linalg.norm(Hy))
# probe Hessian on M should equal <w_N, II> = sum_a (N^T w)_a B_a ; up to O(u^2) higher-order (x is exactly quadratic in u, but u_hat = J^T x != u? here J^T x = u exactly)
K = np.einsum("a,aij->ij", N.T @ w, B)
print("1b probe-Hessian (emp) vs <w_N,II> rel err:", np.linalg.norm(lq["hess"]["p"][0] - K) / np.linalg.norm(K))
print("1c emp mismatch == quad coef of residual y - Xw (same mask):",
      np.abs((lq["hess"]["y"][0] - lq["hess"]["p"][0]) - lq["hess"]["r"][0]).max())

# ---- 2. counterfactual SSE identity: r2_curve equals a brute-force local-intercept refit for each t ---------------
d, D, k = 4, 30, 400
J = rng.standard_normal((D, d)); g = J.T @ J; ginv = np.linalg.inv(g)
x0 = rng.standard_normal(D); x0 /= np.linalg.norm(x0)
J -= np.outer(x0, x0 @ J)                     # tangent orthogonal to x_hat (sphere-like)
g = J.T @ J; ginv = np.linalg.inv(g)
II = rng.standard_normal((D, d, d)); II = 0.5 * (II + II.transpose(0, 2, 1))
II -= np.einsum("ab,bij->aij", J @ ginv @ J.T, II)       # make II normal
Xn = x0 + rng.standard_normal((k, D)) * 0.05; yn = rng.standard_normal(k)
wv = rng.standard_normal(D)
sc = ns.scaling_at_anchor(Xn, yn, x0, wv, J, g, ginv, II, x0, np.random.default_rng(1))
wT = J @ (ginv @ (J.T @ wv)); wN = wv - wT; w_rad = (wN @ x0) * x0; wS = wN - w_rad
u_ = (Xn - x0) @ J @ ginv
II_tan = II - np.einsum("a,ij->aij", x0, np.einsum("aij,a->ij", II, x0))
qS = 0.5 * np.einsum("ki,ij,kj->k", u_, np.einsum("aij,a->ij", II_tan, wS), u_)
sst = ((yn - yn.mean()) ** 2).sum(); worst = 0
for j, t in enumerate(ns.T_GRID):
    pred = Xn @ (wT + w_rad) + t * qS; c = (yn - pred).mean(); r2 = 1 - ((yn - pred - c) ** 2).sum() / sst
    worst = max(worst, abs(r2 - sc["S_model"]["r2_curve"][j]))
print("2a S_model r2_curve vs brute-force refit, max |diff|:", worst)
ts = sc["S_model"]["t_star"]; dR2 = sc["S_model"]["dR2"]; cv = sc["S_model"]["r2_curve"]
print("2b help<=>t*>1/2:", (cv[4] > cv[2]) == (ts > 0.5), " hurt<=>t*>-1/2:", (cv[0] < cv[2]) == (ts > -0.5))
# S variant at t=1 equals the global readout with a local intercept
pred = Xn @ wv; c = (yn - pred).mean(); r2g = 1 - ((yn - pred - c) ** 2).sum() / sst
print("2c variant S at t=1 == global readout + local intercept:", abs(r2g - sc["S"]["r2_curve"][4]))

# ---- 3. tilted-tangent leak: linear label (zero intrinsic Hessian), decoder tangent tilted by eps ---------------
d, D, k = 3, 10, 4000
Q, _ = np.linalg.qr(rng.standard_normal((D, D))); T, N = Q[:, :d], Q[:, d:]
B = rng.standard_normal((D - d, d, d)); B = 0.5 * (B + B.transpose(0, 2, 1))
u = rng.standard_normal((k, d)) * 0.05
X = np.vstack([np.zeros((1, D)), u @ T.T + 0.5 * (np.einsum("aij,ki,kj->ka", B, u, u) @ N.T)])
uu = np.vstack([np.zeros((1, d)), u]); a = rng.standard_normal(d); a /= np.linalg.norm(a)
y = uu @ a
for eps in (0.0, 0.05, 0.1, 0.2):
    Jt = T + eps * N[:, :d]
    gt = Jt.T @ Jt
    geo = {"J": Jt[None], "g": gt[None], "ginv": np.linalg.inv(gt)[None]}
    h = pfs.local_quadratics(X, np.array([0]), np.arange(k + 1)[None], geo, {"y": y}, 32)["hess"]["y"][0]
    print(f"3 tilt eps={eps}: |Hess_est| (truth 0) = {np.linalg.norm(h):.4f}")
