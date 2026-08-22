"""Learnable sampleable local priors: Gaussian / diagonal GMM / CF-matched GMM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class DiagGMM:
    weights: np.ndarray  # (K,)
    means: np.ndarray  # (K, d)
    variances: np.ndarray  # (K, d) diagonal
    family: str

    def log_prob(self, U: np.ndarray) -> np.ndarray:
        # stable log-sum-exp
        K, d = self.means.shape
        logs = []
        for j in range(K):
            v = np.maximum(self.variances[j], 1e-6)
            diff = U - self.means[j]
            quad = (diff**2 / v).sum(axis=1)
            log_det = np.log(v).sum()
            logs.append(
                np.log(max(self.weights[j], 1e-12))
                - 0.5 * (d * np.log(2 * np.pi) + log_det + quad)
            )
        L = np.stack(logs, axis=1)
        m = L.max(axis=1, keepdims=True)
        return (m.squeeze(1) + np.log(np.exp(L - m).sum(axis=1)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comps = rng.choice(len(self.weights), size=n, p=self.weights / self.weights.sum())
        out = np.zeros((n, self.means.shape[1]), dtype=np.float64)
        for j in range(len(self.weights)):
            m = comps == j
            if not np.any(m):
                continue
            out[m] = rng.normal(
                self.means[j], np.sqrt(np.maximum(self.variances[j], 1e-6)), size=(m.sum(), self.means.shape[1])
            )
        return out.astype(np.float32)

    def characteristic_function(self, T: np.ndarray) -> np.ndarray:
        """phi(T) complex, T shape (M, d)."""
        # sum_j alpha exp(i t mu - 0.5 sum sigma^2 t^2)
        out = np.zeros(len(T), dtype=np.complex128)
        for j in range(len(self.weights)):
            phase = T @ self.means[j]
            damp = 0.5 * (T**2 * self.variances[j]).sum(axis=1)
            out += self.weights[j] * np.exp(1j * phase - damp)
        return out


def fit_standard_gaussian(d: int) -> DiagGMM:
    return DiagGMM(
        weights=np.array([1.0]),
        means=np.zeros((1, d)),
        variances=np.ones((1, d)),
        family="standard_gaussian",
    )


def fit_mle_gmm(
    U: np.ndarray,
    w: np.ndarray,
    *,
    n_components: int,
    seed: int,
    var_floor: float = 1e-4,
) -> DiagGMM:
    mask = w > 1e-6
    Uu = U[mask]
    ww = w[mask]
    if len(Uu) < n_components + 2:
        n_components = 1
    # sklearn GMM doesn't take weights directly — importance resample
    rng = np.random.default_rng(seed)
    p = ww / ww.sum()
    n_fit = min(len(Uu), max(200, 20 * n_components))
    idx = rng.choice(len(Uu), size=n_fit, replace=True, p=p)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=var_floor,
        random_state=seed,
        max_iter=200,
        init_params="kmeans",
    )
    gmm.fit(Uu[idx])
    # enforce min weight
    alpha = np.maximum(gmm.weights_, 1e-3)
    alpha = alpha / alpha.sum()
    return DiagGMM(
        weights=alpha.astype(np.float64),
        means=gmm.means_.astype(np.float64),
        variances=np.maximum(gmm.covariances_.astype(np.float64), var_floor),
        family=f"mle_gmm_K{n_components}",
    )


def empirical_cf(U: np.ndarray, w: np.ndarray, T: np.ndarray) -> np.ndarray:
    ww = w / max(w.sum(), 1e-12)
    # (N, M) via batch
    phase = U @ T.T  # (N, M)
    return (ww[:, None] * np.exp(1j * phase)).sum(axis=0)


def sample_frequencies(d: int, n: int, scales: list[float], rng: np.random.Generator) -> np.ndarray:
    out = []
    per = max(1, n // max(len(scales), 1))
    for s in scales:
        dirs = rng.standard_normal((per, d))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        rad = rng.normal(0, s, size=per)
        out.append(dirs * rad[:, None])
    T = np.concatenate(out, axis=0)[:n]
    return T.astype(np.float64)


def cf_loss(model: DiagGMM, U: np.ndarray, w: np.ndarray, T: np.ndarray) -> float:
    emp = empirical_cf(U, w, T)
    th = model.characteristic_function(T)
    err = emp - th
    return float(np.mean(err.real**2 + err.imag**2))


def fine_tune_gmm_cf(
    model: DiagGMM,
    U: np.ndarray,
    w: np.ndarray,
    *,
    n_freq: int,
    scales: list[float],
    steps: int,
    lr: float,
    seed: int,
) -> tuple[DiagGMM, dict]:
    """Simple coordinate-descent style CF fine-tune on means/variances/weights."""
    rng = np.random.default_rng(seed)
    T = sample_frequencies(model.means.shape[1], n_freq, scales, rng)
    m = DiagGMM(
        weights=model.weights.copy(),
        means=model.means.copy(),
        variances=model.variances.copy(),
        family=model.family.replace("mle", "cf") if "mle" in model.family else "cf_gmm",
    )
    hist = [cf_loss(m, U, w, T)]
    for _ in range(steps):
        # finite-difference / random coordinate steps
        for param in ("means", "variances", "weights"):
            base = getattr(m, param).copy()
            noise = rng.normal(0, lr, size=base.shape)
            trial = base + noise
            if param == "variances":
                trial = np.maximum(trial, 1e-4)
            if param == "weights":
                trial = np.maximum(trial, 1e-3)
                trial = trial / trial.sum()
            setattr(m, param, trial)
            loss = cf_loss(m, U, w, T)
            if loss <= hist[-1]:
                hist.append(loss)
            else:
                setattr(m, param, base)
                hist.append(hist[-1])
    return m, {"cf_loss_init": hist[0], "cf_loss_final": hist[-1], "steps": steps}


def weighted_loglik(model: DiagGMM, U: np.ndarray, w: np.ndarray) -> float:
    if w.sum() <= 0 or len(U) == 0:
        return float("nan")
    lp = model.log_prob(U)
    return float(np.sum(w * lp) / np.sum(w))


def bic_score(model: DiagGMM, U: np.ndarray, w: np.ndarray) -> float:
    # approximate BIC with effective N
    n_eff = float((w.sum() ** 2) / max((w**2).sum(), 1e-12))
    ll = weighted_loglik(model, U, w) * w.sum()
    K, d = model.means.shape
    n_params = (K - 1) + K * d + K * d
    return float(n_params * np.log(max(n_eff, 2)) - 2 * ll)


def select_prior(
    U_tr: np.ndarray,
    w_tr: np.ndarray,
    U_va: np.ndarray,
    w_va: np.ndarray,
    *,
    ks: list[int],
    use_cf: bool,
    n_freq: int,
    scales: list[float],
    seed: int,
) -> dict:
    d = U_tr.shape[1]
    candidates = []
    # standard gaussian
    g0 = fit_standard_gaussian(d)
    candidates.append(
        {
            "model": g0,
            "val_ll": weighted_loglik(g0, U_va, w_va),
            "val_cf": cf_loss(g0, U_va, w_va, sample_frequencies(d, n_freq, scales, np.random.default_rng(seed))),
            "bic": bic_score(g0, U_tr, w_tr),
            "name": g0.family,
        }
    )
    for K in ks:
        m = fit_mle_gmm(U_tr, w_tr, n_components=K, seed=seed + K)
        row = {
            "model": m,
            "val_ll": weighted_loglik(m, U_va, w_va),
            "val_cf": cf_loss(
                m, U_va, w_va, sample_frequencies(d, n_freq, scales, np.random.default_rng(seed + K))
            ),
            "bic": bic_score(m, U_tr, w_tr),
            "name": m.family,
        }
        candidates.append(row)
        if use_cf and K >= 1:
            m2, info = fine_tune_gmm_cf(
                m, U_tr, w_tr, n_freq=n_freq, scales=scales, steps=40, lr=0.02, seed=seed + 100 + K
            )
            candidates.append(
                {
                    "model": m2,
                    "val_ll": weighted_loglik(m2, U_va, w_va),
                    "val_cf": cf_loss(
                        m2,
                        U_va,
                        w_va,
                        sample_frequencies(d, n_freq, scales, np.random.default_rng(seed + 200 + K)),
                    ),
                    "bic": bic_score(m2, U_tr, w_tr),
                    "name": m2.family,
                    "cf_tune": info,
                }
            )
    # prefer highest val_ll; one-SE rule vs best
    lls = np.array([c["val_ll"] for c in candidates], dtype=np.float64)
    best = float(np.nanmax(lls))
    # complexity: prefer fewer components among near-best
    order = np.argsort(-lls)
    chosen = candidates[int(order[0])]
    for i in order:
        c = candidates[int(i)]
        if c["val_ll"] >= best - 0.05:  # rough 1-SE proxy
            if "gmm_K" in c["name"]:
                k = int(c["name"].split("K")[-1]) if "K" in c["name"] else 99
            else:
                k = 0
            ck = 0
            if "gmm_K" in chosen["name"]:
                ck = int(chosen["name"].split("K")[-1])
            if k < ck or (k == ck and "cf" not in c["name"] and "cf" in chosen["name"]):
                chosen = c
    # serialize models
    def ser(m: DiagGMM):
        return {
            "weights": m.weights.tolist(),
            "means": m.means.tolist(),
            "variances": m.variances.tolist(),
            "family": m.family,
        }

    return {
        "chosen": {**{k: v for k, v in chosen.items() if k != "model"}, "model": ser(chosen["model"])},
        "candidates": [
            {**{k: v for k, v in c.items() if k != "model"}, "model": ser(c["model"])}
            for c in candidates
        ],
    }


def save_priors(out: Path, table: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "prior_selection.json").write_text(json.dumps(table, indent=2))
