import json
from pathlib import Path

import numpy as np

from sweep.aggregate import aggregate
from sweep.extract import LABELS
from sweep.manifest import load_manifest

COLS = ("hess_mismatch_emp", "align_cos_tan", "hess_mismatch_dec", "pf_rad")


def _split(path, partial, p=0.001):
    rows = [{"row": "environment", "device": "cuda"}, {"row": "fit", "var_explained": 0.95}]
    for lab in LABELS:
        rows.append({"row": "result", "label": lab, "global_oof_r2": 0.5,
                     "columns": {c: {"multiscale": {"partial": partial, "p": p}} for c in COLS}})
        rows.append({"row": "xfit", "label": lab, "hessian_split_half_cos_p50": 0.9,
                     "columns": {c: {"fitA_scoreB": {"partial": partial, "p": p}, "fitB_scoreA": {"partial": partial, "p": p}} for c in COLS}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _cf(path, n=40, seed=0):
    rng = np.random.default_rng(seed); d = {}
    for lab in LABELS:
        for var in ("S_model", "random_qmatched"):
            cv = rng.normal(size=(n, 5)); cv[:, 4] = cv[:, 2] + 0.1; cv[:, 0] = cv[:, 2] - 0.1
            d[f"{lab}:{var}:eq"] = rng.uniform(1, 2, n); d[f"{lab}:{var}:qq"] = np.ones(n); d[f"{lab}:{var}:r2_curve"] = cv
    path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, **d)


def _thin(path, n=40):
    np.savez(path, overlap=np.zeros((n, n)))


def _fixture(tmp_path, complete=("vit_base", "clip_base"), partial=("dinov3_vitb16",)):
    rec, arr = tmp_path / "records", tmp_path / "arrays"
    for e in complete:
        for s in ("main_xfit", "main", "seed1", "seed2", "w400", "alpha1", "d20"):
            _split(rec / f"scaling__{e}__{s}.jsonl", -0.3 if s != "seed2" else 0.1)
        _cf(arr / f"scaling__{e}__cf.npz"); _thin(arr / f"scaling__{e}__thin.npz")
    for e in partial:
        _split(rec / f"scaling__{e}__main_xfit.jsonl", -0.2)
    return rec, arr


def test_aggregate_writes_all_outputs(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    for f in ("tab_scaling_main.tex", "tab_scaling_xfit.tex", "tab_scaling_cf.tex", "tab_scaling_robust.tex",
              "fig_scaling_partials.pdf", "fig_scaling_cf.png", "fig_scaling_robust.pdf", "SCALING_REPORT.md"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f


def test_aggregate_with_missing_jobs(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    main = (out / "tab_scaling_main.tex").read_text()
    assert "--" in main                                   # encoders with no records
    rep = (out / "SCALING_REPORT.md").read_text()
    assert "3 of 31 encoders have records" in rep
    assert "2 of 31 encoders complete all 9 jobs" in rep


def test_robust_counts_sign_changes(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rob = (out / "tab_scaling_robust.tex").read_text()
    assert "vit_base" in rob.replace("\\_", "_") and "1" in rob   # seed2 flips the sign once


def test_aggregate_is_deterministic(tmp_path):
    rec, arr = _fixture(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    aggregate(load_manifest(), rec, arr, a); aggregate(load_manifest(), rec, arr, b)
    for f in ("tab_scaling_main.tex", "tab_scaling_cf.tex", "SCALING_REPORT.md", "fig_scaling_cf.png"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
