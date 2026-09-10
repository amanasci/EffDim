"""Model-specific sphere-normal K_H^cross at frozen d=16, k=2048."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import POSITIVE_CONTROL, PRIMARY_D, PRIMARY_K, R_H_FAIL, SOURCE_NDC, ExpConfig
from .data import fill_controls_from_pack, geo_controls, load_model_bundle, vitb_frozen_kh
from .io_util import resolve_path, write_df


def _device(cfg: ExpConfig):
    import torch

    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_model_kh(
    shared: dict,
    model: str,
    sids: list[int],
    cfg: ExpConfig,
    out,
) -> pd.DataFrame:
    """Return per-anchor K_H_cross / R_H. ViT-B reuses the frozen rank-sweep panel."""
    cache_dir = out / "geometry" / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "anchor_curvature.parquet"
    if path.exists() and not cfg.force:
        return pd.read_parquet(path)

    if model == POSITIVE_CONTROL and not cfg.smoke:
        kh = vitb_frozen_kh(shared, sids)
        ctrl = geo_controls(shared, model, sids)
        df = fill_controls_from_pack(shared, model, sids, kh.merge(ctrl, on=["sample_id", "model"], how="left"))
        write_df(path, df, force=True)
        return df

    from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank, nested_pca_frame

    bundle = load_model_bundle(shared, model)
    X = bundle["X"]
    neigh = bundle["neigh"]
    device = _device(cfg)
    n_splits = cfg.n_splits_kh()
    rows = []
    for i, sid in enumerate(sids):
        jp = cache_dir / f"J_{int(sid)}.npz"
        ai = shared["sid_to_ai"][int(sid)]
        N = neigh[ai, :PRIMARY_K]
        Xloc = np.asarray(X[N], dtype=np.float64)
        if jp.exists() and not cfg.force:
            z = np.load(jp)
            x0, J = z["x0"], z["J"]
        else:
            x0, J, ev, _ = nested_pca_frame(Xloc, PRIMARY_D, device)
            np.savez(jp, x0=x0, J=J, ev=ev)
        fits = _fit_rank(Xloc, x0, J, PRIMARY_D, PRIMARY_K, n_splits, cfg.seed, ai)
        if not fits:
            rows.append(
                {
                    "sample_id": int(sid),
                    "model": model,
                    "K_H_cross": float("nan"),
                    "R_H": float("nan"),
                    "n_splits": 0,
                    "source": "computed",
                }
            )
            continue
        rows.append(
            {
                "sample_id": int(sid),
                "model": model,
                "K_H_cross": float(np.mean([f["K_H_cross"] for f in fits])),
                "R_H": float(np.mean([f["R_H"] for f in fits])),
                "n_splits": int(len(fits)),
                "source": "computed",
            }
        )
        if (i + 1) % 16 == 0:
            print(f"[cmcla][kh] {model} {i+1}/{len(sids)}", flush=True)
    kh = pd.DataFrame(rows)
    ctrl = geo_controls(shared, model, sids)
    df = fill_controls_from_pack(shared, model, sids, kh.merge(ctrl, on=["sample_id", "model"], how="left"))
    write_df(path, df, force=True)
    return df


def reliability_row(model: str, df: pd.DataFrame) -> dict[str, Any]:
    rh = df.R_H.to_numpy(float)
    med = float(np.nanmedian(rh))
    fail = bool(not np.isfinite(med) or med <= R_H_FAIL)
    return {
        "model": model,
        "n": int(np.isfinite(df.K_H_cross).sum()),
        "R_H_median": med,
        "R_H_fail_threshold": R_H_FAIL,
        "geometry_unreliable": fail,
        "reason": "median_R_H_leq_threshold" if fail else "ok",
    }


def copy_vitb_tangent_cache(shared: dict, sids: list[int], out) -> None:
    """Reuse frozen NDC J for ViT-B rotation; do not write into NDC."""
    ndc = resolve_path(shared["root"], SOURCE_NDC) / "cache"
    dest = out / "geometry" / POSITIVE_CONTROL
    dest.mkdir(parents=True, exist_ok=True)
    for sid in sids:
        src = ndc / f"J_{int(sid)}_k{PRIMARY_K}.npz"
        dst = dest / f"J_{int(sid)}.npz"
        if src.exists() and not dst.exists():
            z = np.load(src)
            np.savez(dst, **{k: z[k] for k in z.files})
