"""Recompute full sphere-normal scalars from frozen frames via _fit_rank."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import POSITIVE_CONTROL, PRIMARY_D, PRIMARY_K, R_H_FAIL, ExpConfig
from .data import load_frame, load_model_bundle, load_ndc_full_vitb
from .io_util import write_df
from .metrics import kb_from_kh_kaniso, orthogonality_residuals, split_scalars, unpack_BS_symmetric


def _device(cfg: ExpConfig):
    import torch

    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mean_rows(fits: list[dict]) -> dict[str, float]:
    keys = [
        "K_H_cross",
        "K_aniso_cross",
        "K_dir_cross",
        "K_B_cross",
        "R_H",
        "R_B0",
        "R_BS",
        "trace_energy_mean",
        "traceless_energy_mean",
        "full_energy_mean",
        "B_fro_A",
        "B_fro_B",
        "B0_fro_A",
        "B0_fro_B",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(f[k]) for f in fits if k in f and np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_splits"] = float(len(fits))
    full = out["full_energy_mean"]
    out["traceless_fraction"] = float(out["traceless_energy_mean"] / full) if full and np.isfinite(full) and full > 0 else float("nan")
    out["trace_fraction"] = float(out["trace_energy_mean"] / full) if full and np.isfinite(full) and full > 0 else float("nan")
    return out


def _fit_one(Xloc, x0, J, n_splits: int, seed: int, ai: int) -> tuple[list[dict], dict[str, float]]:
    from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank

    fits_raw = _fit_rank(Xloc, x0, J, PRIMARY_D, PRIMARY_K, n_splits, seed, ai)
    rows = []
    ortho = []
    for f in fits_raw:
        sc = split_scalars(f["BS_flat_A"], f["BS_flat_B"], PRIMARY_D)
        rec = {**f, **sc}
        rec.pop("H_mean", None)
        BA = unpack_BS_symmetric(f["BS_flat_A"], PRIMARY_D)
        ortho.append(orthogonality_residuals(BA, x0, J[:, :PRIMARY_D]))
        rec.pop("BS_flat_A", None)
        rec.pop("BS_flat_B", None)
        rows.append(rec)
    if ortho:
        ortho_mean = {
            "max_abs_x0_dot": float(np.max([o["max_abs_x0_dot"] for o in ortho])),
            "max_abs_J_dot": float(np.max([o["max_abs_J_dot"] for o in ortho])),
        }
    else:
        ortho_mean = {"max_abs_x0_dot": float("nan"), "max_abs_J_dot": float("nan")}
    return rows, ortho_mean


def fit_model_full(
    shared: dict,
    model: str,
    sids: list[int],
    cfg: ExpConfig,
    out,
) -> pd.DataFrame:
    cache_dir = out / "geometry" / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "anchor_curvature.parquet"
    if path.exists() and not cfg.force:
        return pd.read_parquet(path)

    if model == POSITIVE_CONTROL and not cfg.smoke:
        ndc = load_ndc_full_vitb(sids, shared)
        ndc["K_B_cross"] = [
            kb_from_kh_kaniso(float(r.K_H_cross), float(r.K_aniso_cross), PRIMARY_D)
            for r in ndc.itertuples()
        ]
        ndc["source"] = "ndc_reuse_plus_KB_identity"
        write_df(path, ndc, force=True)
        return ndc

    bundle = load_model_bundle(shared, model)
    X = bundle["X"]
    neigh = bundle["neigh"]
    rows = []
    for i, sid in enumerate(sids):
        frame = load_frame(shared, model, int(sid))
        ai = shared["sid_to_ai"][int(sid)]
        N = neigh[ai, :PRIMARY_K]
        Xloc = np.asarray(X[N], dtype=np.float64)
        if frame is None:
            from geometry.physics_activation_atlas.nested_dimension_curvature import nested_pca_frame

            x0, J, ev, _ = nested_pca_frame(Xloc, PRIMARY_D, _device(cfg))
        else:
            x0, J = frame
        fits, ortho = _fit_one(Xloc, x0, J, cfg.n_splits(), cfg.seed, ai)
        rec = {
            "sample_id": int(sid),
            "model": model,
            "source": "fit_rank_frozen_frame",
            **_mean_rows(fits),
            **ortho,
        }
        rows.append(rec)
        if (i + 1) % 16 == 0:
            print(f"[fcrr][geo] {model} {i + 1}/{len(sids)}", flush=True)
    df = pd.DataFrame(rows)
    write_df(path, df, force=True)
    return df


def vitb_refit_parity(
    shared: dict,
    sids: list[int],
    cfg: ExpConfig,
    n_check: int = 8,
) -> dict[str, Any]:
    """Recompute a few ViT-B anchors and match stored NDC K_dir / K_H."""
    stored = load_ndc_full_vitb(sids, shared).set_index("sample_id")
    bundle = load_model_bundle(shared, POSITIVE_CONTROL)
    check = [s for s in sids if int(s) in stored.index][:n_check]
    diffs = []
    for sid in check:
        frame = load_frame(shared, POSITIVE_CONTROL, int(sid))
        ai = shared["sid_to_ai"][int(sid)]
        N = bundle["neigh"][ai, :PRIMARY_K]
        Xloc = np.asarray(bundle["X"][N], dtype=np.float64)
        if frame is None:
            continue
        x0, J = frame
        fits, _ = _fit_one(Xloc, x0, J, 5 if not cfg.smoke else 1, cfg.seed, ai)
        got = _mean_rows(fits)
        st = stored.loc[int(sid)]
        diffs.append(
            {
                "sample_id": int(sid),
                "dK_dir": float(got["K_dir_cross"] - st.K_dir_cross),
                "dK_H": float(got["K_H_cross"] - st.K_H_cross),
                "dK_aniso": float(got["K_aniso_cross"] - st.K_aniso_cross),
            }
        )
    if not diffs:
        return {"ok": False, "reason": "no_vitb_parity_anchors", "n": 0}
    max_dir = float(np.max(np.abs([d["dK_dir"] for d in diffs])))
    max_h = float(np.max(np.abs([d["dK_H"] for d in diffs])))
    return {
        "ok": bool(max_dir < 1e-7 and max_h < 1e-7),
        "n": len(diffs),
        "max_abs_dK_dir": max_dir,
        "max_abs_dK_H": max_h,
        "rows": diffs,
    }


def reliability_table(by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in by_model.items():
        rec = {"model": model, "n": int(len(df))}
        for col, name in (("R_H", "trace"), ("R_B0", "traceless"), ("R_BS", "full_tensor")):
            if col in df.columns:
                rec[f"{name}_median_R"] = float(np.nanmedian(df[col].to_numpy(float)))
            else:
                rec[f"{name}_median_R"] = float("nan")
        rec["geometry_unreliable"] = bool(
            not np.isfinite(rec["trace_median_R"]) or rec["trace_median_R"] <= R_H_FAIL
        )
        rows.append(rec)
    return pd.DataFrame(rows)
