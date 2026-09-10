#!/usr/bin/env python3
"""Scale-match the Phase 9 AE H_tan field to the frozen k=2048 local chart.

Trains the sealed PlainAutoEncoder protocol (d=16, hidden 250^3, SiLU, 600 epochs,
sphere-projected decode) on the frozen ViT-B 16,384-row Physics subset. Evaluates
H_tan at every row, then averages / split-crosses it on the same 2048 neighbours
used for K_H^cross. Reports raw and radius-residualized Spearman with K_H and
local R^2.

This is their *instrument* on our *cloud and neighbourhoods*, not a bit-identical
replay of the 86,471-row Phase 9 fit. Writes only into
outputs/geometry/physics_ae_local_patch_scale_match/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
_NB = _ROOT / "notebooks"
for p in (_EXP, _NB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pu_manifold import cae, decoder_curvature  # noqa: E402
from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix  # noqa: E402
from geometry.physics_curvature_component_predictive_decomposition.data import (  # noqa: E402
    load_shared,
    merge_components,
)
from geometry.physics_curvature_component_predictive_decomposition.config import ExpConfig  # noqa: E402
from geometry.physics_curvature_component_predictive_decomposition.io_util import (  # noqa: E402
    platonic_root,
    write_df,
    write_json,
    write_text,
)
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X  # noqa: E402

D = 16
K = 2048
HIDDEN = (250, 250, 250)
ACTIVATION = "silu"
MAX_EPOCHS = 600
TORCH_INIT_SEED = 0
SPLIT_SEED = 20260813
HOLDOUT_FRACTION = 0.2
TRAIN_CFG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch": 128,
    "lip_weight": 0.0,
    "fps_pretrain_epochs": 0,
    "early_stop_patience": MAX_EPOCHS + 1,
    "early_stop_min_delta": 1e-9,
    "wallclock_ceiling_s": float("inf"),
    "seed": 0,
}
OUT_REL = "outputs/geometry/physics_ae_local_patch_scale_match"


class SphereProjectedDecoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.decoder = model.decoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        F = self.model.decode(z)
        return F / torch.linalg.norm(F, dim=-1, keepdim=True)


def split_indices(n: int, split_seed: int, holdout_fraction: float):
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    return perm[n_holdout:], perm[:n_holdout]


def decompose_H(H_vec: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
    img_norm = np.linalg.norm(image, axis=1, keepdims=True)
    u = image / np.clip(img_norm, 1e-12, None)
    H_rad = np.einsum("ij,ij->i", H_vec, u)
    H_tan = H_vec - H_rad[:, None] * u
    return {
        "H_rad": H_rad,
        "H_tan": H_tan,
        "H_tan_norm": np.linalg.norm(H_tan, axis=1),
        "H_norm": np.linalg.norm(H_vec, axis=1),
        "image_norm": img_norm.reshape(-1),
    }


def patch_stats(H_tan: np.ndarray, neigh: np.ndarray, ais: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ai in ais:
        N = np.asarray(neigh[int(ai), :K], dtype=int)
        Ht = H_tan[N]
        norms = np.linalg.norm(Ht, axis=1)
        perm = rng.permutation(len(N))
        a, b = perm[: len(N) // 2], perm[len(N) // 2 :]
        ha, hb = Ht[a].mean(0), Ht[b].mean(0)
        rows.append(
            {
                "row_index": int(ai),
                "H_tan_point": float(np.linalg.norm(H_tan[int(ai)])),
                "H_tan_patch_mean": float(np.mean(norms)),
                "H_tan_patch_median": float(np.median(norms)),
                "H_tan_cross": float(np.dot(ha, hb)),
                "H_tan_ab_cos": float(
                    np.dot(ha, hb) / max(np.linalg.norm(ha) * np.linalg.norm(hb), 1e-18)
                ),
            }
        )
    return pd.DataFrame(rows)


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(x)), z])
    m = np.isfinite(x) & np.all(np.isfinite(A), axis=1)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if int(m.sum()) < 8:
        return out
    coef, *_ = np.linalg.lstsq(A[m], x[m], rcond=None)
    out[m] = x[m] - A[m] @ coef
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    t0 = time.time()
    root = platonic_root()
    out = root / OUT_REL
    if args.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    epochs = args.epochs if args.epochs is not None else (20 if args.smoke else MAX_EPOCHS)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[scale] root={root} device={device} epochs={epochs}", flush=True)

    cfg = ExpConfig(smoke=False, n_anchors_override=None)
    shared = load_shared(cfg)
    sids = list(shared["sids"])
    geo = merge_components(shared, "vit_base", sids)
    mm = shared["mm"]
    X = np.asarray(load_model_X(mm, "vit_base"), dtype=np.float64)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    anchors = np.load(mm / "prepare" / "anchors.npz")
    sid_all = [int(s) for s in anchors["anchors_sample_id"]]
    sid_to_ai = {int(s): i for i, s in enumerate(sid_all)}
    neigh = np.asarray(np.load(mm / "model_neighbourhoods" / f"vit_base_kmax{K}.npz")["neigh"], dtype=np.int64)
    ais = np.array([sid_to_ai[int(s)] for s in geo.sample_id.astype(int)], dtype=int)
    print(f"[scale] X={X.shape} anchors={len(ais)} mean|x|={float(np.linalg.norm(X, axis=1).mean()):.6f}", flush=True)

    train_idx, holdout_idx = split_indices(X.shape[0], SPLIT_SEED, HOLDOUT_FRACTION)
    torch.manual_seed(TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(in_dim=X.shape[1], latent_dim=D, hidden=HIDDEN, activation=ACTIVATION).to(device)
    x_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    tcfg = dict(TRAIN_CFG)
    tcfg["max_epochs"] = int(epochs)
    print(f"[scale] training on {len(train_idx)} rows…", flush=True)
    fit = cae.train_plain_ae(model, x_train, tcfg)
    print(f"[scale] trained epochs={fit['epochs_run']} wall={fit['wallclock_s']:.1f}s", flush=True)

    model.eval().double().to(device)
    curv = SphereProjectedDecoder(model).eval()
    with torch.no_grad():
        x64 = torch.tensor(X, dtype=torch.float64, device=device)
        z_all = model.encode(x64)
        image = curv.decode(z_all).detach().cpu().numpy()
        y_hold = model(x64[torch.as_tensor(holdout_idx, device=device)])["y"]
        xh = x64[torch.as_tensor(holdout_idx, device=device)]
        var_explained = float(1.0 - ((xh - y_hold) ** 2).sum(-1).mean() / (xh.pow(2).sum(-1).mean()))
    print(f"[scale] holdout var_explained={var_explained:.4f}", flush=True)

    print("[scale] autodiff field on all rows…", flush=True)
    t1 = time.time()
    field = decoder_curvature.plain_decoder_curvature(curv, z_all)
    H_vec = field["H_vec"].detach().cpu().numpy()
    print(f"[scale] field wall={time.time()-t1:.1f}s", flush=True)
    dec = decompose_H(H_vec, image)
    print(
        f"[scale] median H_rad={float(np.median(dec['H_rad'])):.4f} (expect {-D}) "
        f"median H_tan={float(np.median(dec['H_tan_norm'])):.4f}",
        flush=True,
    )

    patch = patch_stats(dec["H_tan"], neigh, ais, seed=0)
    patch["sample_id"] = geo.sample_id.to_numpy()
    df = geo.merge(patch, on="sample_id", how="inner")
    df["H_tan_point"] = dec["H_tan_norm"][ais]
    write_df(out / "tables" / "anchor_scale_match.parquet", df, force=True)

    radius = df["log_knn_radius"].to_numpy(float)
    Z = control_matrix(df)
    pairs = [
        ("H_tan_point", "K_H_cross"),
        ("H_tan_patch_mean", "K_H_cross"),
        ("H_tan_cross", "K_H_cross"),
        ("H_tan_point", "r2_G"),
        ("H_tan_patch_mean", "r2_G"),
        ("H_tan_cross", "r2_G"),
        ("K_H_cross", "r2_G"),
        ("H_tan_point", "log_knn_radius"),
        ("H_tan_patch_mean", "log_knn_radius"),
        ("K_H_cross", "log_knn_radius"),
        ("H_tan_point", "H_tan_patch_mean"),
        ("H_tan_point", "H_tan_cross"),
        ("H_tan_patch_mean", "H_tan_cross"),
    ]
    rows = []
    for xcol, ycol in pairs:
        x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
        rec = {
            "x": xcol,
            "y": ycol,
            **associate(x, y, Z),
            "rho_radius_residualized": float(
                associate(residualize(x, radius[:, None]), residualize(y, radius[:, None]), None)["raw"]
            ),
        }
        rec["rho_vs_radius_only"] = float(associate(x, y, radius[:, None])["controlled"])
        rows.append(rec)
    assoc_df = pd.DataFrame(rows)
    write_df(out / "tables" / "scale_match_associations.csv", assoc_df, force=True)

    def _get(x, y, key="controlled"):
        hit = assoc_df[(assoc_df.x == x) & (assoc_df.y == y)]
        return float(hit.iloc[0][key]) if len(hit) else float("nan")

    summary = {
        "n_rows": int(X.shape[0]),
        "n_anchors": int(len(df)),
        "epochs": int(epochs),
        "var_explained": var_explained,
        "median_H_rad": float(np.median(dec["H_rad"])),
        "median_H_tan": float(np.median(dec["H_tan_norm"])),
        "rho_point_KH": _get("H_tan_point", "K_H_cross"),
        "rho_patch_KH": _get("H_tan_patch_mean", "K_H_cross"),
        "rho_cross_KH": _get("H_tan_cross", "K_H_cross"),
        "rho_point_KH_radius": _get("H_tan_point", "K_H_cross", "rho_vs_radius_only"),
        "rho_patch_KH_radius": _get("H_tan_patch_mean", "K_H_cross", "rho_vs_radius_only"),
        "rho_cross_KH_radius": _get("H_tan_cross", "K_H_cross", "rho_vs_radius_only"),
        "rho_point_R2": _get("H_tan_point", "r2_G"),
        "rho_patch_R2": _get("H_tan_patch_mean", "r2_G"),
        "rho_cross_R2": _get("H_tan_cross", "r2_G"),
        "rho_KH_R2": _get("K_H_cross", "r2_G"),
        "rho_point_radius": _get("H_tan_point", "log_knn_radius", "raw"),
        "rho_patch_radius": _get("H_tan_patch_mean", "log_knn_radius", "raw"),
        "rho_KH_radius": _get("K_H_cross", "log_knn_radius", "raw"),
        "rho_point_vs_patch": _get("H_tan_point", "H_tan_patch_mean", "raw"),
        "runtime_s": time.time() - t0,
        "device": str(device),
        "smoke": bool(args.smoke),
    }
    write_json(out / "summary.json", summary, force=True)
    write_text(
        out / "REPORT.md",
        f"""# AE field averaged on the frozen k=2048 patch

Instrument: Phase 9 `PlainAutoEncoder` + sphere-projected `H_tan` (Amendment 01).
Cloud: frozen ViT-B 16,384-row subset, same 512 anchors and k=2048 neighbours as `K_H`.
Not a replay of the 86,471-row Phase 9 fit.

Holdout variance explained: {var_explained:.4f}. Median `H_rad`={summary['median_H_rad']:.4f} (want −16).

## Do the instruments agree after scale matching?

Controlled Spearman (frozen 3 controls):

| | vs `K_H` | vs `K_H` | radius only | vs local `R_G^2` |
|---|---:|---:|---:|---:|
| pointwise `‖H_tan‖` | {summary['rho_point_KH']:.3f} | {summary['rho_point_KH_radius']:.3f} | {summary['rho_point_radius']:.3f} | {summary['rho_point_R2']:.3f} |
| patch-mean `‖H_tan‖` | {summary['rho_patch_KH']:.3f} | {summary['rho_patch_KH_radius']:.3f} | {summary['rho_patch_radius']:.3f} | {summary['rho_patch_R2']:.3f} |
| split-cross `⟨H̄_A, H̄_B⟩` | {summary['rho_cross_KH']:.3f} | {summary['rho_cross_KH_radius']:.3f} |  | {summary['rho_cross_R2']:.3f} |
| frozen `K_H` | 1 |  | {summary['rho_KH_radius']:.3f} | {summary['rho_KH_R2']:.3f} |

Pointwise vs patch-mean `‖H_tan‖`: raw ρ = {summary['rho_point_vs_patch']:.3f}.
Runtime {summary['runtime_s']/60:.1f} min.
""",
        force=True,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[scale] wrote {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
