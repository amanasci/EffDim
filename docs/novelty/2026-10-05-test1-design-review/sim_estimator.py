"""Toy simulation of the split-half F_k estimator, using the real per-anchor II spectra (vit_base, d=16).

Per anchor, work in the basis of II's right singular vectors, so P_k h = first k coordinates.
Signal h = mixture of a 'linear' part (M^T v, coords s_j xi_j, v isotropic) and an isotropic residual.
Two halves h_A = h + e_A, h_B = h + e_B, e isotropic (or structured), noise level set so the
median split-half cosine is about 0.27.  Anchors are clustered (32 blocks) with shared signal and
partly shared noise to mimic neighbourhood overlap (each point is in ~12 neighbourhoods).
"""
import numpy as np

rng = np.random.default_rng(0)
S = np.load("/home/akagi/Documents/Projects/EffDim/curvature-experiment/results/ii-rank/ii_rank_vit_base_d16_spectra.npz")["s_insphere"].astype(float)
n, m = S.shape
K = 32
G = 32  # blocks


def draw(w_lin, noise_rel, rng, cluster=True, rho_sig=0.6, rho_noise=0.4, bias=None):
    blocks = rng.integers(0, G, n)
    def field(gen):
        base = gen((G, m))[blocks]
        ind = gen((n, m))
        return np.sqrt(rho_sig) * base + np.sqrt(1 - rho_sig) * ind if cluster else ind
    xi = field(rng.standard_normal)
    lin = S * xi
    lin /= np.sqrt((lin ** 2).sum(1, keepdims=True).mean())
    res = field(rng.standard_normal)
    res /= np.sqrt((res ** 2).sum(1, keepdims=True).mean())
    h = np.sqrt(w_lin) * lin + np.sqrt(1 - w_lin) * res
    scale = np.exp(0.8 * rng.standard_normal(n))  # heavy-tailed Hessian norms across anchors
    h = h * scale[:, None]
    hn2 = (h ** 2).sum(1)
    sig2 = noise_rel * np.median(hn2) / m  # per-coordinate noise variance, constant across anchors
    def noise():
        if cluster:
            base = rng.standard_normal((G, m))[blocks]
            e = np.sqrt(rho_noise) * base + np.sqrt(1 - rho_noise) * rng.standard_normal((n, m))
        else:
            e = rng.standard_normal((n, m))
        return e * np.sqrt(sig2)
    hA, hB = h + noise(), h + noise()
    if bias is not None:
        hA = hA + bias; hB = hB + bias
    return h, hA, hB, blocks


def estimators(h, hA, hB):
    P = lambda x: x[:, :K]
    num = (P(hA) * P(hB)).sum(1); den = (hA * hB).sum(1)
    cos = den / np.sqrt((hA ** 2).sum(1) * (hB ** 2).sum(1))
    return {
        "truth": (P(h) ** 2).sum() / (h ** 2).sum(),
        "ros": num.sum() / den.sum(),
        "mor": np.median(num / den),
        "naive": (P(hA) ** 2).sum() / (hA ** 2).sum(),
        "frac_den_neg": float((den < 0).mean()),
        "cos_med": float(np.median(cos)),
    }


if __name__ == "__main__":
    for noise_rel in (2.7, 4.0):
        for w in (0.0, 0.3, 0.6, 0.9):
            for cl in (False, True):
                out = [estimators(*draw(w, noise_rel, rng, cluster=cl)[:3]) for _ in range(400)]
                g = lambda k: np.array([o[k] for o in out])
                print(f"noise/|h|^2={noise_rel} w_lin={w:.1f} clustered={cl}: cos_med {g('cos_med').mean():.2f} "
                      f"den<0 {g('frac_den_neg').mean():.2f} | truth {g('truth').mean():.3f} (sd {g('truth').std():.3f}) "
                      f"ros {g('ros').mean():.3f} (sd {g('ros').std():.3f}) mor {g('mor').mean():.3f} (sd {g('mor').std():.3f}) "
                      f"naive {g('naive').mean():.3f}")
    # power: SD of the difference between two independent labels' ros estimates, clustered
    d = []
    for _ in range(400):
        a = estimators(*draw(0.5, 2.7, rng, cluster=True)[:3])["ros"]
        b = estimators(*draw(0.5, 2.7, rng, cluster=True)[:3])["ros"]
        d.append(a - b)
    print("sd of difference between two labels' F_32 (clustered, w=0.5):", np.std(d).round(3), "-> 2*1.96*sd/sqrt2 approx MDE", (2.8 * np.std(d)).round(3))
