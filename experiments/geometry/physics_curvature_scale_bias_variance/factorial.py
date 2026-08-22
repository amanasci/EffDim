"""Six-cell (R, m) curvature refit with checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CELLS, PRIMARY_D, PRIMARY_K, ExpConfig
from .curvature import fit_cell
from .data import control_row
from .io_util import write_df
from .probe import neighbourhood_metrics


def _cell_path(out: Path, R: int, m: int, d: int) -> Path:
    return out / "cells" / f"R{R}_m{m}_d{d}.parquet"


def run_factorial(bundle: dict[str, Any], cfg: ExpConfig, out: Path, *, ds: list[int]) -> pd.DataFrame:
    (out / "cells").mkdir(parents=True, exist_ok=True)
    X = bundle["X"]
    neigh = bundle["neigh"]
    sid_to_ai = bundle["sid_to_ai"]
    y, yhat = bundle["y"], bundle["yhat"]
    geo = bundle["geo"]
    sids = bundle["scale_sids"]
    frames = []
    for R, m in CELLS:
        for d in ds:
            nrep = cfg.n_rep() if int(d) == PRIMARY_D else cfg.n_rep_sec()
            dest = _cell_path(out, int(R), int(m), int(d))
            if dest.exists() and not cfg.force:
                frames.append(pd.read_parquet(dest))
                print(f"[sbv] resume {dest.name}", flush=True)
                continue
            t0 = time.time()
            recs = []
            Hs = []
            ambient = int(X.shape[1])
            for sid in sids:
                ai = sid_to_ai[int(sid)]
                pool_idx = neigh[ai, : int(R)]
                Xpool = X[pool_idx]
                met_fix = neighbourhood_metrics(y, yhat, neigh[ai, :PRIMARY_K])
                met_match = neighbourhood_metrics(y, yhat, neigh[ai, : int(R)])
                ctrl_fix = control_row(geo, int(sid), PRIMARY_K)
                try:
                    ctrl_match = control_row(geo, int(sid), int(R))
                except KeyError:
                    ctrl_match = {
                        "log_knn_radius": float("nan"),
                        "local_label_variance": met_match["local_target_var"],
                        "local_evaluation_count": met_match["n_eval"],
                    }
                for r in range(nrep):
                    fit = fit_cell(
                        Xpool,
                        R=int(R),
                        m=int(m),
                        d=int(d),
                        seed=int(cfg.seed) + r,
                        sample_id=int(sid),
                        device=cfg.device,
                    )
                    recs.append(
                        {
                            "sample_id": int(sid),
                            "R": int(R),
                            "m": int(m),
                            "d": int(d),
                            "replicate": int(r),
                            "K_H_cross": fit.get("K_H_cross", float("nan")),
                            "R_H": fit.get("R_H", float("nan")),
                            "ok": bool(fit.get("ok", False)),
                            "reason": fit.get("reason", ""),
                            "select_hash": fit.get("select_hash", ""),
                            "split_hash": fit.get("split_hash", ""),
                            "r2_k2048": met_fix["r2_local"],
                            "mse_k2048": met_fix["oof_mse"],
                            "sst_k2048": met_fix["local_sst"],
                            "var_k2048": met_fix["local_target_var"],
                            "r2_matched": met_match["r2_local"],
                            "mse_matched": met_match["oof_mse"],
                            "sst_matched": met_match["local_sst"],
                            "var_matched": met_match["local_target_var"],
                            **{f"ctl_{k}": v for k, v in ctrl_fix.items()},
                            **{f"ctlR_{k}": v for k, v in ctrl_match.items()},
                            "outer_inner_ratio": (fit.get("outer_resid") or {}).get("outer_inner_ratio", float("nan")),
                        }
                    )
                    h = fit.get("H")
                    Hs.append(np.asarray(h, dtype=float) if h is not None else np.full(ambient, np.nan))
            df = pd.DataFrame(recs)
            write_df(dest, df, force=True)
            if Hs:
                np.savez_compressed(
                    out / "cells" / f"H_R{R}_m{m}_d{d}.npz",
                    sample_id=df.sample_id.to_numpy(),
                    replicate=df.replicate.to_numpy(),
                    H=np.stack(Hs),
                )
            print(f"[sbv] cell R={R} m={m} d={d} n={len(df)} s={time.time()-t0:.1f}", flush=True)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


run_factorial = run_factorial
