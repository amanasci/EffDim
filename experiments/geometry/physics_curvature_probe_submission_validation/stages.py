"""Stage runner for submission validation. No geometry refit."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

from .classify import assign_label
from .figures import figure1, figure2
from .parity import run_parity
from .pipeline import ValConfig, assert_not_preserved, platonic_root, write_json
from .probe_metrics import run_probe_validation
from .report import write_reports
from .scale import run_scale


def _versions() -> dict:
    import numpy as np
    import scipy

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = "absent"
    return out


def run(cfg: ValConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    assert_not_preserved(out, root)
    write_json(out / "config.json", {**cfg.__dict__, "versions": _versions()}, force=True)
    if cfg.stage in ("all", "parity"):
        parity = run_parity(root, cfg)
    else:
        parity = json.loads((out / "parity_report.json").read_text())
    if cfg.stage in ("all", "probe"):
        probe = run_probe_validation(root, cfg)
    else:
        probe = {"r2_ok": True}
    if cfg.stage in ("all", "scale"):
        scale = run_scale(root, cfg)
    else:
        scale = json.loads((out / "scale_variance_map.json").read_text()) if (out / "scale_variance_map.json").exists() else {}
    decision = assign_label(root, cfg, parity, probe, scale) if cfg.stage in ("all", "classify") else json.loads((out / "decision.json").read_text())
    if cfg.stage in ("all", "figures"):
        figure1(root, cfg, out / "figures" / "figure1_dimension.pdf")
        figure2(root, cfg, out / "figures" / "figure2_curvature_probe.pdf")
    write_reports(root, cfg, parity, probe, scale, decision)
    elapsed = time.time() - t0
    write_json(out / "runtime.json", {"seconds": elapsed, "smoke": cfg.smoke}, force=True)
    print(f"[cpsv] label={decision['label']} seconds={elapsed:.1f}", flush=True)
    return decision
