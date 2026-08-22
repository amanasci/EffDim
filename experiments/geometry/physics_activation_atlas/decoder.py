"""Smooth residual chart decoders f_c(u) = Normalize(mu + W u + r(u))."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "softplus":
        return nn.Softplus()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Need C^2 activation, got {name}")


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        d: int,
        ambient: int,
        mu: np.ndarray,
        basis: np.ndarray,  # (D, d)
        hidden: list[int],
        activation: str = "softplus",
        residual_scale: float = 0.01,
        output_normalize: bool = True,
    ):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))
        self.output_normalize = output_normalize
        layers: list[nn.Module] = []
        prev = d
        act = _activation(activation)
        for h in hidden:
            layers += [nn.Linear(prev, h), act]
            prev = h
        layers.append(nn.Linear(prev, ambient))
        self.mlp = nn.Sequential(*layers)
        # near-zero residual init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, std=1e-3)
        self.residual_scale = residual_scale

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u standardized; undo via caller or absorb in basis scales — coords already std
        linear = self.mu + u @ self.basis.T
        resid = self.residual_scale * self.mlp(u)
        y = linear + resid
        if self.output_normalize:
            n = torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-8)
            y = y / n
        return y


def pca_reconstruct(X_mu_basis: dict, U: np.ndarray) -> np.ndarray:
    # U standardized → unstandardize
    U_raw = U * X_mu_basis["coord_std"]
    Y = X_mu_basis["mu"] + U_raw @ X_mu_basis["basis"].T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, 1e-8)).astype(np.float32)


@torch.no_grad()
def decode_np(model: ResidualDecoder, U: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    t = torch.tensor(U, dtype=torch.float32, device=device)
    for i0 in range(0, len(t), 2048):
        out.append(model(t[i0 : i0 + 2048]).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_chart_decoder(
    pca: dict,
    U_tr: np.ndarray,
    X_tr: np.ndarray,
    w_tr: np.ndarray,
    U_va: np.ndarray,
    X_va: np.ndarray,
    w_va: np.ndarray,
    *,
    hidden: list[int],
    activation: str,
    residual_scale: float,
    output_normalize: bool,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    max_train: int | None = None,
    seed: int = 0,
) -> tuple[ResidualDecoder, dict]:
    rng = np.random.default_rng(seed)
    mask = w_tr > 1e-6
    idx = np.where(mask)[0]
    if max_train is not None and len(idx) > max_train:
        # weighted subsample
        p = w_tr[idx] / w_tr[idx].sum()
        idx = rng.choice(idx, size=max_train, replace=False, p=p)
    Ut = U_tr[idx]
    Xt = X_tr[idx]
    wt = w_tr[idx]
    wt = wt / max(wt.sum(), 1e-12)

    model = ResidualDecoder(
        d=Ut.shape[1],
        ambient=Xt.shape[1],
        mu=pca["mu"],
        basis=pca["basis"] * pca["coord_std"],  # absorb std into basis
        hidden=hidden,
        activation=activation,
        residual_scale=residual_scale,
        output_normalize=output_normalize,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(
        torch.tensor(Ut, dtype=torch.float32),
        torch.tensor(Xt, dtype=torch.float32),
        torch.tensor(wt, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    def eval_loss(U, X, w):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(U, dtype=torch.float32, device=device))
            x = torch.tensor(X, dtype=torch.float32, device=device)
            ww = torch.tensor(w, dtype=torch.float32, device=device)
            err = ((pred - x) ** 2).sum(dim=-1)
            return float((ww * err).sum() / ww.sum().clamp_min(1e-12))

    best = float("inf")
    best_state = None
    bad = 0
    hist = []
    for ep in range(epochs):
        model.train()
        for ub, xb, wb in loader:
            ub, xb, wb = ub.to(device), xb.to(device), wb.to(device)
            pred = model(ub)
            loss = (wb * ((pred - xb) ** 2).sum(dim=-1)).sum() / wb.sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = eval_loss(U_va, X_va, np.maximum(w_va, 0))
        if np.sum(w_va) <= 0:
            va = eval_loss(Ut, Xt, wt)
        hist.append(va)
        if va < best - 1e-6:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # metrics
    pred_va = decode_np(model, U_va, device) if len(U_va) else np.zeros((0, Xt.shape[1]))
    pca_va = pca_reconstruct(pca, U_va) if len(U_va) else pred_va
    def wmean_mse(pred, X, w):
        if len(X) == 0 or w.sum() <= 0:
            return float("nan")
        e = ((pred - X) ** 2).sum(axis=1)
        return float(np.sum(w * e) / np.sum(w))
    def wmean_cos(pred, X, w):
        if len(X) == 0 or w.sum() <= 0:
            return float("nan")
        c = (pred * X).sum(axis=1)
        return float(np.sum(w * c) / np.sum(w))

    metrics = {
        "best_val_mse": best,
        "epochs_ran": len(hist),
        "val_mse_decoder": wmean_mse(pred_va, X_va, w_va),
        "val_mse_pca": wmean_mse(pca_va, X_va, w_va),
        "val_cos_decoder": wmean_cos(pred_va, X_va, w_va),
        "val_cos_pca": wmean_cos(pca_va, X_va, w_va),
        "improvement_vs_pca": (
            wmean_mse(pca_va, X_va, w_va) - wmean_mse(pred_va, X_va, w_va)
            if len(X_va)
            else float("nan")
        ),
    }
    return model, metrics


def jacobian_stats(
    model: ResidualDecoder, u0: np.ndarray, device: str
) -> dict:
    model.eval()
    u = torch.tensor(u0, dtype=torch.float32, device=device)

    def f(uu):
        return model(uu.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, u)  # (D, d)
    d = int(u.numel())
    svals = torch.linalg.svdvals(J).detach().cpu().numpy()
    eps = 1e-6 * float(svals.max()) if svals.size else 0.0
    rank = int(np.sum(svals > eps))
    cond = float(svals.max() / max(svals.min(), 1e-12)) if svals.size else float("inf")
    return {
        "singular_values": svals.astype(np.float64).tolist(),
        "rank": rank,
        "full_rank": bool(rank >= d),
        "condition": cond,
        "d": d,
    }


def save_decoder(path: Path, model: ResidualDecoder, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)
    (path.with_suffix(".json")).write_text(json.dumps(metrics, indent=2))
