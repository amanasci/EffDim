"""Record readers for the scaling sweep, ported from the manuscript's appendix generator (appendix_gen.py)
so the sweep tables read records exactly the way the paper's appendix does. Ported, not imported: the
experiment code never depends on the manuscript tree."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

LABELS = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")


def read_rows(path) -> List[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def split_cells(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["label"], c): v["multiscale"] for r in rows if r.get("row") == "result"
            for c, v in r["columns"].items() if isinstance(v, dict) and "multiscale" in v}


def xfit_cells(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["label"], c): v for r in rows if r.get("row") == "xfit" for c, v in r["columns"].items()}


def xfit_hess_cos_p50(rows: List[dict]) -> Dict[str, float]:
    """Median split-half Hessian cosine per label (appendix_gen reads `hessian_split_half_cos_p25_p50_p75[1]`)."""
    out = {}
    for r in rows:
        if r.get("row") != "xfit":
            continue
        if "hessian_split_half_cos_p25_p50_p75" in r:
            out[r["label"]] = float(r["hessian_split_half_cos_p25_p50_p75"][1])
        elif "hessian_split_half_cos_p50" in r:
            out[r["label"]] = float(r["hessian_split_half_cos_p50"])
    return out


def global_r2(rows: List[dict]) -> Dict[str, float]:
    return {r["label"]: r["global_oof_r2"] for r in rows if r.get("row") == "result" and "global_oof_r2" in r}


def var_explained(rows: List[dict]):
    fits = [r["var_explained"] for r in rows if r.get("row") == "fit"]
    return fits[0] if fits else None


def devices(rows: List[dict]) -> List[str]:
    """Devices named by the environment rows (records written before the device flag carry none)."""
    return sorted({str(r["device"]) for r in rows if r.get("row") == "environment" and "device" in r})


def cf_summary(npz_path, labels=LABELS) -> Dict[str, Dict[str, dict]]:
    z = np.load(npz_path)
    out: Dict[str, Dict[str, dict]] = {}
    for lab in labels:
        if f"{lab}:S_model:eq" not in z.files:
            continue
        out[lab] = {}
        for var in ("S_model", "random_qmatched"):
            eq, qq, cv = z[f"{lab}:{var}:eq"], z[f"{lab}:{var}:qq"], z[f"{lab}:{var}:r2_curve"]
            m = np.isfinite(eq); dp = cv[m, 4] - cv[m, 2]; dm = cv[m, 0] - cv[m, 2]
            v = {"help": float(np.mean(dp > 0)), "hurt": float(np.mean(dm < 0)),
                 "d_r2_plus": float(np.median(dp)), "d_r2_minus": float(np.median(dm))}
            if var == "S_model":
                v["t_star"] = float(np.median(eq[m] / np.maximum(qq[m], 1e-300)))
            out[lab][var] = v
    return out


def _indep(ov: np.ndarray, thr: float) -> np.ndarray:
    A = ov > thr; np.fill_diagonal(A, False); alive = np.ones(ov.shape[0], bool); keep = np.zeros(ov.shape[0], bool)
    while alive.any():
        deg = (A & alive[None, :]).sum(1); deg[~alive] = 10**9
        i = int(np.argmin(deg)); keep[i] = True; alive[i] = False; alive[A[i]] = False
    return keep


def _ptex(p):
    e = int(np.floor(np.log10(p))); c = int(np.ceil(p / 10 ** e - 1e-9))   # upper bound c x 10^e with c in 1..9
    if c == 10: c, e = 1, e + 1
    return ("%d\\times 10^{%d}" % (c, e)) if c > 1 else "10^{%d}" % e


def sign_test(cf_npz, thin_npz, thr: float = 0.05, labels=LABELS) -> Dict[str, dict]:
    # Ported from appendix_gen's thinned-anchor block: the mask is `keep & isfinite(r2_curve[:, 0])`
    # (not `isfinite(eq)`), and help/hurt compare r2_curve columns directly (4 > 2, 0 < 2).
    z = np.load(cf_npz); keep = _indep(np.load(thin_npz)["overlap"].astype(float), thr)
    out = {}
    for lab in labels:
        if f"{lab}:S_model:r2_curve" not in z.files:
            continue
        cv = z[f"{lab}:S_model:r2_curve"]; m = keep & np.isfinite(cv[:, 0]); n = int(m.sum())
        kh = int((cv[m, 4] > cv[m, 2]).sum()); ku = int((cv[m, 0] < cv[m, 2]).sum())
        out[lab] = {"n": n, "help": kh / n if n else float("nan"), "hurt": ku / n if n else float("nan"),
                    "p_help": float(binomtest(kh, n, 0.5, alternative="greater").pvalue) if n else float("nan"),
                    "p_hurt": float(binomtest(ku, n, 0.5, alternative="greater").pvalue) if n else float("nan")}
    return out
