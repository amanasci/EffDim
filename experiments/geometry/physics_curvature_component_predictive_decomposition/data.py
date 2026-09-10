"""Read-only frozen FCR, CMCLA, QLCA, and NDC artifacts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CONTROLS,
    MODELS,
    PRIMARY_D,
    SOURCE_CMCLA,
    SOURCE_FCR,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QLCA,
    ExpConfig,
)
from .io_util import platonic_root, resolve_path


def load_shared(cfg: ExpConfig) -> dict[str, Any]:
    root = platonic_root()
    fcr = resolve_path(root, SOURCE_FCR)
    cmcla = resolve_path(root, SOURCE_CMCLA)
    qlca = resolve_path(root, SOURCE_QLCA)
    ndc = resolve_path(root, SOURCE_NDC)
    mm = resolve_path(root, SOURCE_MM)
    man = json.loads((cmcla / "common_anchor_manifest.json").read_text())
    sids = [int(s) for s in man["sample_ids"]][: cfg.n_anc()]
    models = list(cfg.models_override) if cfg.models_override else list(MODELS)
    return {
        "root": root,
        "fcr": fcr,
        "cmcla": cmcla,
        "qlca": qlca,
        "ndc": ndc,
        "mm": mm,
        "sids": sids,
        "models": models,
    }


def load_fcr_anchor(shared: dict, model: str, sids: list[int]) -> pd.DataFrame:
    path = shared["fcr"] / "tables" / f"{model}_per_anchor_curvature.parquet"
    df = pd.read_parquet(path)
    df = df[df.sample_id.astype(int).isin(sids)].drop_duplicates("sample_id").copy()
    df["K_TF_cross"] = df["K_aniso_cross"]
    df["K_dir_check"] = df["K_H_cross"] + df["K_TF_cross"]
    return df.reset_index(drop=True)


def load_qlca_risks(shared: dict, sids: list[int]) -> pd.DataFrame:
    df = pd.read_csv(shared["qlca"] / "anchor_risks.csv")
    return df[df.sample_id.astype(int).isin(sids)].drop_duplicates("sample_id").reset_index(drop=True)


def attach_within_split_energies(shared: dict, model: str, df: pd.DataFrame) -> pd.DataFrame:
    """Read-only FCR geometry cache, or NDC tensors for ViT-B."""
    out = df.copy()
    geo_p = shared["fcr"] / "geometry" / model / "anchor_curvature.parquet"
    if geo_p.exists():
        g = pd.read_parquet(geo_p)
        extra = [
            c
            for c in (
                "trace_energy_mean",
                "traceless_energy_mean",
                "full_energy_mean",
                "trace_fraction",
                "traceless_fraction",
            )
            if c in g.columns and c not in out.columns
        ]
        if extra:
            out = out.merge(g[["sample_id"] + extra], on="sample_id", how="left")
    if model == "vit_base" and "trace_energy_mean" not in out.columns:
        from .config import PRIMARY_D
        from .decompose import split_flat

        rows = []
        for sid in out.sample_id.astype(int):
            p = shared["ndc"] / "H_vectors" / f"{int(sid)}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            a = split_flat(np.asarray(z["BS16_A"], dtype=np.float64), PRIMARY_D)
            b = split_flat(np.asarray(z["BS16_B"], dtype=np.float64), PRIMARY_D)
            rows.append(
                {
                    "sample_id": int(sid),
                    "E_H_A": float(a["E_H"]),
                    "E_H_B": float(b["E_H"]),
                    "E_TF_A": float(a["E_TF"]),
                    "E_TF_B": float(b["E_TF"]),
                    "C_trace_A": float(a["C_trace"]),
                    "C_trace_B": float(b["C_trace"]),
                    "trace_energy_mean": 0.5 * PRIMARY_D * (float(a["E_H"]) + float(b["E_H"])),
                    "traceless_energy_mean": 0.5
                    * (
                        float(np.linalg.norm(a["BTF"]) ** 2)
                        + float(np.linalg.norm(b["BTF"]) ** 2)
                    ),
                    "full_energy_mean": 0.5
                    * (float(np.linalg.norm(a["B"]) ** 2) + float(np.linalg.norm(b["B"]) ** 2)),
                    "C_trace_mean": 0.5 * (float(a["C_trace"]) + float(b["C_trace"])),
                }
            )
        if rows:
            out = out.merge(pd.DataFrame(rows), on="sample_id", how="left")
    return out


def merge_components(shared: dict, model: str, sids: list[int]) -> pd.DataFrame:
    geo = load_fcr_anchor(shared, model, sids)
    keep_geo = [
        "sample_id",
        "K_H_cross",
        "K_TF_cross",
        "K_dir_cross",
        "K_B_cross",
        "K_aniso_cross",
        "R_H",
        "R_B0",
        "R_BS",
        "traceless_fraction",
        "trace_fraction",
        "aniso_share_of_Kdir",
        "C_trace_mean",
    ]
    keep_geo = [c for c in keep_geo if c in geo.columns]
    geo = geo[keep_geo]
    probes = pd.read_parquet(shared["cmcla"] / "probes" / f"{model}_anchor_metrics.parquet")
    probes = probes[probes.sample_id.astype(int).isin(sids)].drop_duplicates("sample_id")
    drop = [c for c in ("K_H_cross", "R_H") if c in probes.columns]
    probes = probes.drop(columns=drop)
    df = geo.merge(probes, on="sample_id", how="inner")
    for c in CONTROLS:
        if c not in df.columns:
            df[c] = np.nan
    df = attach_within_split_energies(shared, model, df)
    return df.sort_values("sample_id").reset_index(drop=True)
