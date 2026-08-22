"""Acosta-style extrinsic mean curvature for valid chart anchors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .decoder import ResidualDecoder


CURVATURE_CONVENTION = (
    "Averaged mean-curvature vector H = (1/d) sum_{i,j} g^{ij} B_ij, "
    "where B_ij = P_perp partial_ij f, P_perp v = v - J g^{-1} J^T v. "
    "For the unit sphere in R^{n+1} with the standard embedding, |H| ≈ 1/R "
    "under this averaged convention (for S^2 of radius R, |H|=1/R)."
)


def _jacobian(model: ResidualDecoder, u: torch.Tensor) -> torch.Tensor:
    """J shape (D, d) at single point u (d,). Prefer forward-mode (cheap in d)."""

    def f(uu):
        return model(uu.unsqueeze(0)).squeeze(0)

    try:
        from torch.func import jacfwd

        return jacfwd(f)(u)
    except Exception:  # noqa: BLE001
        return torch.autograd.functional.jacobian(f, u)


def _second_derivs_fd(
    model: ResidualDecoder, u: torch.Tensor, J0: torch.Tensor, *, eps: float = 1e-3
) -> torch.Tensor:
    """Finite-difference ∂_j J (≈ ∂_i∂_j f), shape (D, d, d). Avoids D×d×d autograd."""
    d = u.numel()
    D = J0.shape[0]
    J2 = torch.zeros(D, d, d, device=u.device, dtype=u.dtype)
    for j in range(d):
        uj = u.detach().clone()
        uj[j] = uj[j] + eps
        Jj = _jacobian(model, uj)
        J2[:, :, j] = (Jj - J0) / eps
    # Symmetrize in (i,j) for numerical stability
    J2 = 0.5 * (J2 + J2.transpose(1, 2))
    return J2


def mean_curvature_vector(
    model: ResidualDecoder,
    u0: np.ndarray,
    device: str,
) -> dict:
    """Compute H and diagnostics at one latent point."""
    model.eval()
    u = torch.tensor(u0, dtype=torch.float32, device=device)
    with torch.enable_grad():
        J = _jacobian(model, u)  # (D, d)
    d = J.shape[1]
    g = J.T @ J  # (d, d)
    eye = torch.eye(d, device=device, dtype=J.dtype)
    try:
        g_inv = torch.linalg.solve(g + 1e-6 * eye, eye)
    except RuntimeError:
        g_inv = torch.linalg.pinv(g)
    J2 = _second_derivs_fd(model, u, J)
    H = torch.zeros(J.shape[0], device=device, dtype=J.dtype)
    for i in range(d):
        for j in range(d):
            fij = J2[:, i, j]
            # P_perp fij = fij - J g^{-1} J^T fij  (no D×D projector)
            Bij = fij - J @ (g_inv @ (J.T @ fij))
            H = H + g_inv[i, j] * Bij
    H = H / float(d)
    svals = torch.linalg.svdvals(J).detach().cpu().numpy()
    eps = 1e-6 * float(svals.max()) if svals.size else 0.0
    rank = int(np.sum(svals > eps))
    cond = float(svals.max() / max(svals.min(), 1e-12)) if svals.size else float("inf")
    Hn = H.detach().cpu().numpy()
    return {
        "H": Hn.astype(np.float64),
        "H_norm": float(np.linalg.norm(Hn)),
        "jacobian_rank": rank,
        "jacobian_full_rank": bool(rank >= d),
        "jacobian_condition": cond,
        "d": d,
        "valid_geometry": bool(rank >= d and cond < 1e4 and np.isfinite(np.linalg.norm(Hn))),
        "second_deriv": "central_fd_on_jacobian",
    }


def analytic_circle_curvature_test(radius: float = 1.0, device: str = "cpu") -> dict:
    """Unit test: f(theta) = R (cos theta, sin theta, 0,...,0) in ambient>=2 — use 1D latent."""
    # Build a tiny decoder that is exactly the circle embedding via frozen MLP ≈ identity residual 0
    # Direct analytic H for circle in R^2: H = -n / R with |H|=1/R for mean curvature of curve?
    # For a curve, mean curvature vector magnitude is 1/R.
    # We test our formula on an analytic torch map.

    class Circle(torch.nn.Module):
        def __init__(self, R):
            super().__init__()
            self.R = R

        def forward(self, u):
            # u: (B, 1)
            th = u[..., 0]
            x = self.R * torch.cos(th)
            y = self.R * torch.sin(th)
            return torch.stack([x, y], dim=-1)

    model = Circle(radius).to(device)
    # monkey-patch interface: wrap as callable with same jacobian code
    u0 = np.array([0.3], dtype=np.float32)

    def _mc_analytic():
        u = torch.tensor(u0, dtype=torch.float32, device=device, requires_grad=True)

        def f(uu):
            return model(uu.unsqueeze(0)).squeeze(0)

        J = torch.autograd.functional.jacobian(f, u, create_graph=True)
        d = 1
        g = J.T @ J
        g_inv = torch.linalg.solve(g + 1e-8 * torch.eye(d, device=device), torch.eye(d, device=device))
        J2 = torch.autograd.functional.jacobian(
            lambda uu: torch.autograd.functional.jacobian(f, uu, create_graph=True),
            u,
            create_graph=False,
        )
        H = torch.zeros(J.shape[0], device=device)
        for i in range(d):
            for j in range(d):
                fij = J2[:, i, j]
                Bij = fij - J @ (g_inv @ (J.T @ fij))
                H = H + g_inv[i, j] * Bij
        H = H / float(d)
        return float(torch.linalg.vector_norm(H).cpu())

    hn = _mc_analytic()
    return {
        "case": "circle",
        "radius": radius,
        "H_norm": hn,
        "expected": 1.0 / radius,
        "rel_err": abs(hn - 1.0 / radius) / (1.0 / radius),
        "pass": abs(hn - 1.0 / radius) < 0.05,
    }


def analytic_plane_curvature_test(device: str = "cpu") -> dict:
    class Plane(torch.nn.Module):
        def forward(self, u):
            # u (B,2) -> (x,y,0)
            z = torch.zeros(u.shape[0], 1, device=u.device, dtype=u.dtype)
            return torch.cat([u, z], dim=-1)

    model = Plane().to(device)
    u0 = np.array([0.1, -0.2], dtype=np.float32)
    u = torch.tensor(u0, device=device, requires_grad=True)

    def f(uu):
        return model(uu.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, u, create_graph=True)
    d = 2
    g = J.T @ J
    g_inv = torch.linalg.inv(g + 1e-8 * torch.eye(d, device=device))
    J2 = torch.autograd.functional.jacobian(
        lambda uu: torch.autograd.functional.jacobian(f, uu, create_graph=True),
        u,
        create_graph=False,
    )
    H = torch.zeros(J.shape[0], device=device)
    for i in range(d):
        for j in range(d):
            fij = J2[:, i, j]
            Bij = fij - J @ (g_inv @ (J.T @ fij))
            H = H + g_inv[i, j] * Bij
    H = H / float(d)
    hn = float(torch.linalg.vector_norm(H).cpu())
    return {"case": "plane", "H_norm": hn, "expected": 0.0, "pass": hn < 1e-5}


def analytic_sphere_curvature_test(radius: float = 1.0, device: str = "cpu") -> dict:
    """Stereographic-like local chart for S^2: not exact |H|=1/R everywhere;
    use standard spherical coords chart."""

    class SphereChart(torch.nn.Module):
        def __init__(self, R):
            super().__init__()
            self.R = R

        def forward(self, u):
            # u=(theta, phi) near equator
            th = u[..., 0]
            ph = u[..., 1]
            x = self.R * torch.sin(th) * torch.cos(ph)
            y = self.R * torch.sin(th) * torch.sin(ph)
            z = self.R * torch.cos(th)
            return torch.stack([x, y, z], dim=-1)

    model = SphereChart(radius).to(device)
    u0 = np.array([np.pi / 2, 0.2], dtype=np.float32)
    u = torch.tensor(u0, device=device, requires_grad=True)

    def f(uu):
        return model(uu.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, u, create_graph=True)
    d = 2
    g = J.T @ J
    g_inv = torch.linalg.solve(g + 1e-8 * torch.eye(d, device=device), torch.eye(d, device=device))
    J2 = torch.autograd.functional.jacobian(
        lambda uu: torch.autograd.functional.jacobian(f, uu, create_graph=True),
        u,
        create_graph=False,
    )
    H = torch.zeros(J.shape[0], device=device)
    for i in range(d):
        for j in range(d):
            fij = J2[:, i, j]
            Bij = fij - J @ (g_inv @ (J.T @ fij))
            H = H + g_inv[i, j] * Bij
    H = H / float(d)
    hn = float(torch.linalg.vector_norm(H).cpu())
    # For surface in R3, mean curvature of sphere is 1/R (or 2/R depending on convention).
    # Our averaged H = (1/d) sum g^{ij} B_ij for sphere equals -n/R (vector), |H|=1/R.
    return {
        "case": "sphere",
        "radius": radius,
        "H_norm": hn,
        "expected": 1.0 / radius,
        "rel_err": abs(hn - 1.0 / radius) / (1.0 / radius),
        "pass": abs(hn - 1.0 / radius) < 0.1,
        "convention": CURVATURE_CONVENTION,
    }


def run_curvature_unit_tests(device: str = "cpu") -> dict:
    tests = [
        analytic_plane_curvature_test(device=device),
        analytic_circle_curvature_test(1.0, device=device),
        analytic_sphere_curvature_test(1.0, device=device),
    ]
    return {
        "convention": CURVATURE_CONVENTION,
        "tests": tests,
        "all_pass": all(t["pass"] for t in tests),
    }


def evaluate_chart_curvature(
    model: ResidualDecoder,
    U: np.ndarray,
    w: np.ndarray,
    *,
    n_anchors: int,
    device: str,
    seed: int,
    prior_logp: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    mask = (w > 1e-4) & np.isfinite(U).all(axis=1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"anchors": [], "n_valid": 0, "n_excluded": 0}
    if len(idx) > n_anchors:
        p = w[idx] / w[idx].sum()
        idx = rng.choice(idx, size=n_anchors, replace=False, p=p)
    anchors = []
    n_valid = 0
    for i in idx:
        # skip extrapolative under prior if provided
        if prior_logp is not None and prior_logp[i] < np.percentile(prior_logp[mask], 5):
            anchors.append({"index": int(i), "excluded": "low_prior_density"})
            continue
        try:
            out = mean_curvature_vector(model, U[i], device)
        except Exception as e:  # noqa: BLE001
            anchors.append({"index": int(i), "excluded": f"curvature_error:{e}"})
            continue
        if not out["valid_geometry"]:
            anchors.append(
                {
                    "index": int(i),
                    "excluded": "invalid_jacobian",
                    "jacobian_condition": out["jacobian_condition"],
                    "jacobian_rank": out["jacobian_rank"],
                }
            )
            continue
        n_valid += 1
        anchors.append(
            {
                "index": int(i),
                "H_norm": out["H_norm"],
                "H": out["H"].tolist(),
                "jacobian_condition": out["jacobian_condition"],
                "jacobian_rank": out["jacobian_rank"],
                "prior_logp": float(prior_logp[i]) if prior_logp is not None else None,
            }
        )
    return {
        "anchors": anchors,
        "n_valid": n_valid,
        "n_excluded": len(idx) - n_valid,
        "mean_H_norm_valid": float(np.mean([a["H_norm"] for a in anchors if "H_norm" in a]))
        if n_valid
        else float("nan"),
    }


def overlap_curvature_agreement(curv_a: dict, curv_b: dict) -> dict:
    """Compare H vectors for shared indices present in both charts."""
    Ha = {a["index"]: np.asarray(a["H"]) for a in curv_a["anchors"] if "H" in a}
    Hb = {a["index"]: np.asarray(a["H"]) for a in curv_b["anchors"] if "H" in a}
    common = sorted(set(Ha) & set(Hb))
    if not common:
        return {"n_common": 0}
    cos, rel, euc = [], [], []
    for i in common:
        a, b = Ha[i], Hb[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos.append(float(np.dot(a, b) / max(na * nb, 1e-12)))
        rel.append(float(abs(na - nb) / max(na, nb, 1e-12)))
        euc.append(float(np.linalg.norm(a - b)))
    return {
        "n_common": len(common),
        "cosine_mean": float(np.mean(cos)),
        "rel_norm_diff_mean": float(np.mean(rel)),
        "euclid_diff_mean": float(np.mean(euc)),
        "stable": bool(np.mean(cos) > 0.5),
    }


def save_curvature(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "curvature.json").write_text(json.dumps(payload, indent=2, default=str))


def sphere_tangent_decompose(H: np.ndarray, x: np.ndarray) -> dict:
    """Split Euclidean mean-curvature into radial and sphere-tangent parts at unit x."""
    H = np.asarray(H, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    xn = x / max(np.linalg.norm(x), 1e-12)
    H_rad = float(np.dot(H, xn)) * xn
    H_sph = H - H_rad
    hn = float(np.linalg.norm(H))
    return {
        "H_rad": H_rad,
        "H_sphere": H_sph,
        "H_rad_norm": float(np.linalg.norm(H_rad)),
        "H_sphere_norm": float(np.linalg.norm(H_sph)),
        "radial_fraction": float(np.linalg.norm(H_rad) / max(hn, 1e-12)),
        "H_norm": hn,
    }


def mean_curvature_from_J_J2(J: np.ndarray, J2: np.ndarray) -> np.ndarray:
    """H from Jacobian (D,d) and second derivatives (D,d,d)."""
    d = J.shape[1]
    g = J.T @ J
    try:
        g_inv = np.linalg.solve(g + 1e-6 * np.eye(d), np.eye(d))
    except np.linalg.LinAlgError:
        g_inv = np.linalg.pinv(g)
    H = np.zeros(J.shape[0], dtype=np.float64)
    for i in range(d):
        for j in range(d):
            fij = J2[:, i, j]
            Bij = fij - J @ (g_inv @ (J.T @ fij))
            H = H + g_inv[i, j] * Bij
    return H / float(d)


def mean_curvature_callable_fd(
    decode_fn,
    u0: np.ndarray,
    *,
    h: float = 1e-3,
) -> dict:
    """FD curvature for any decode_fn: R^d -> R^D (numpy)."""
    u0 = np.asarray(u0, dtype=np.float64)
    d = u0.shape[0]
    y0 = np.asarray(decode_fn(u0), dtype=np.float64)
    D = y0.shape[0]
    J = np.zeros((D, d), dtype=np.float64)
    for k in range(d):
        up = u0.copy()
        um = u0.copy()
        up[k] += h
        um[k] -= h
        J[:, k] = (decode_fn(up) - decode_fn(um)) / (2 * h)
    J2 = np.zeros((D, d, d), dtype=np.float64)
    for j in range(d):
        up = u0.copy()
        um = u0.copy()
        up[j] += h
        um[j] -= h
        Jp = np.zeros((D, d))
        Jm = np.zeros((D, d))
        for k in range(d):
            upp = up.copy()
            upm = up.copy()
            ump = um.copy()
            umm = um.copy()
            upp[k] += h
            upm[k] -= h
            ump[k] += h
            umm[k] -= h
            Jp[:, k] = (decode_fn(upp) - decode_fn(upm)) / (2 * h)
            Jm[:, k] = (decode_fn(ump) - decode_fn(umm)) / (2 * h)
        J2[:, :, j] = (Jp - Jm) / (2 * h)
    J2 = 0.5 * (J2 + J2.transpose(0, 2, 1))
    H = mean_curvature_from_J_J2(J, J2)
    decomp = sphere_tangent_decompose(H, y0)
    svals = np.linalg.svd(J, compute_uv=False)
    eps = 1e-6 * float(svals.max()) if svals.size else 0.0
    return {
        "H": H,
        "x": y0,
        **decomp,
        "jacobian_rank": int(np.sum(svals > eps)),
        "jacobian_condition": float(svals.max() / max(svals.min(), 1e-12)),
        "h": h,
    }


def mean_curvature_torch_autodiff(model: ResidualDecoder, u0: np.ndarray, device: str) -> dict:
    """Exact nested autodiff curvature (slow; for validation only)."""
    model.eval()
    u = torch.tensor(u0, dtype=torch.float32, device=device, requires_grad=True)

    def f(uu):
        return model(uu.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, u, create_graph=True)
    J2 = torch.autograd.functional.jacobian(
        lambda uu: torch.autograd.functional.jacobian(f, uu, create_graph=True),
        u,
        create_graph=False,
    )
    H = mean_curvature_from_J_J2(
        J.detach().cpu().numpy(),
        J2.detach().cpu().numpy(),
    )
    with torch.no_grad():
        x = f(u).detach().cpu().numpy()
    decomp = sphere_tangent_decompose(H, x)
    return {"H": H, "x": x, **decomp, "method": "autodiff"}


def validate_fd_vs_autodiff(
    model: ResidualDecoder,
    anchors_U: list[np.ndarray],
    *,
    device: str = "cpu",
    hs: list[float] | None = None,
) -> dict:
    hs = hs or [1e-2, 3e-3, 1e-3]
    rows = []
    for ai, u0 in enumerate(anchors_U):
        try:
            exact = mean_curvature_torch_autodiff(model, u0, device)
        except Exception as e:  # noqa: BLE001
            rows.append({"anchor": ai, "error": str(e)})
            continue

        def decode_fn(u, _m=model, _dev=device):
            with torch.no_grad():
                t = torch.tensor(u, dtype=torch.float32, device=_dev)
                return _m(t.unsqueeze(0)).squeeze(0).cpu().numpy()

        for h in hs:
            fd = mean_curvature_callable_fd(decode_fn, u0, h=h)
            err_full = float(np.linalg.norm(fd["H"] - exact["H"]))
            err_sph = float(np.linalg.norm(fd["H_sphere"] - exact["H_sphere"]))
            err_norm = abs(fd["H_norm"] - exact["H_norm"])
            rows.append(
                {
                    "anchor": ai,
                    "h": h,
                    "full_vector_error": err_full,
                    "sphere_tangent_vector_error": err_sph,
                    "norm_error": err_norm,
                    "exact_radial_fraction": exact["radial_fraction"],
                    "fd_radial_fraction": fd["radial_fraction"],
                }
            )
    # step-size sensitivity: variance of H_sphere across h per anchor
    sens = []
    for ai in range(len(anchors_U)):
        norms = [r["sphere_tangent_vector_error"] for r in rows if r.get("anchor") == ai and "h" in r]
        if norms:
            sens.append({"anchor": ai, "sph_err_range": float(max(norms) - min(norms))})
    return {
        "comparisons": rows,
        "step_size_sensitivity": sens,
        "mean_full_error_h1e3": float(
            np.nanmean([r["full_vector_error"] for r in rows if r.get("h") == 1e-3])
        ),
        "mean_sphere_error_h1e3": float(
            np.nanmean([r["sphere_tangent_vector_error"] for r in rows if r.get("h") == 1e-3])
        ),
    }


def overlap_curvature_agreement_sphere(
    Ha: dict[int, np.ndarray],
    Hb: dict[int, np.ndarray],
    Xa: dict[int, np.ndarray],
    Xb: dict[int, np.ndarray],
) -> dict:
    """Agreement for full H and sphere-tangent H at shared indices."""
    common = sorted(set(Ha) & set(Hb))
    if not common:
        return {"n_common": 0}
    full_cos, sph_cos, rad_frac, sph_rel = [], [], [], []
    for i in common:
        da = sphere_tangent_decompose(Ha[i], Xa[i])
        db = sphere_tangent_decompose(Hb[i], Xb[i])
        na, nb = da["H_norm"], db["H_norm"]
        full_cos.append(float(np.dot(Ha[i], Hb[i]) / max(na * nb, 1e-12)))
        nsa, nsb = da["H_sphere_norm"], db["H_sphere_norm"]
        if nsa < 1e-8 and nsb < 1e-8:
            sph_cos.append(1.0)
        else:
            sph_cos.append(float(np.dot(da["H_sphere"], db["H_sphere"]) / max(nsa * nsb, 1e-12)))
        rad_frac.append(0.5 * (da["radial_fraction"] + db["radial_fraction"]))
        sph_rel.append(float(abs(nsa - nsb) / max(nsa, nsb, 1e-12)))
    return {
        "n_common": len(common),
        "full_H_cosine_mean": float(np.mean(full_cos)),
        "radial_fraction_mean": float(np.mean(rad_frac)),
        "H_sphere_cosine_mean": float(np.mean(sph_cos)),
        "H_sphere_rel_norm_diff_mean": float(np.mean(sph_rel)),
        "radial_dominated": bool(np.mean(rad_frac) > 0.8),
        "sphere_tangent_stable": bool(np.mean(sph_cos) > 0.5),
    }
