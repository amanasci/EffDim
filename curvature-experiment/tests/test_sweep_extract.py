import os
import re
from pathlib import Path

import pytest

from sweep.extract import cf_summary, read_rows, split_cells

REC = Path(os.environ.get("EFFDIM_RECORDS", "/home/akagi/Documents/Projects/EffDim/notebooks/.cache"))
MAIN_TEX = Path(__file__).resolve().parents[2] / "paper" / "latex" / "main.tex"
needs = pytest.mark.skipif(not (REC / "09_physics_probe_facing_split_clip_base.jsonl").exists(), reason="paper records absent")


def _fmt(v):
    s = f"${v['partial']:+.2f}" + ("^{*}" if v["p"] > 0.05 else "") + "$"
    return s


@needs
def test_split_cells_match_tab_xenc():
    cells = split_cells(read_rows(REC / "09_physics_probe_facing_split_clip_base.jsonl"))
    tex = MAIN_TEX.read_text()
    block = tex[tex.index(r"\label{tab:xenc}") - 6000: tex.index(r"\label{tab:xenc}")]
    for lab in ("mag_r", "photo_z"):
        s = _fmt(cells[(lab, "hess_mismatch_emp")])
        assert s in block, (lab, s)


@needs
def test_cf_summary_matches_tab_cf():
    z = REC / "09_physics_normal_scaling_clip_base_d16.npz"
    cf = cf_summary(z)
    tex = MAIN_TEX.read_text()
    block = tex[tex.index(r"\label{tab:cf}") - 8000: tex.index(r"\label{tab:cf}")]
    v = cf["mag_r"]["S_model"]
    assert f"{v['help']:.2f}" in block and f"{v['t_star']:.1f}" in block


import numpy as np

from sweep.extract import sign_test


def test_readers_take_labels(tmp_path):
    rng = np.random.default_rng(0); n = 30; d = {}
    for var in ("S_model", "random_qmatched"):
        cv = rng.normal(size=(n, 5)); cv[:, 4] = cv[:, 2] + 0.1; cv[:, 0] = cv[:, 2] - 0.1
        d[f"gap:{var}:eq"] = rng.uniform(1, 2, n); d[f"gap:{var}:qq"] = np.ones(n); d[f"gap:{var}:r2_curve"] = cv
    cf = tmp_path / "cf.npz"; np.savez(cf, **d)
    th = tmp_path / "th.npz"; np.savez(th, overlap=np.zeros((n, n)))
    assert cf_summary(cf) == {} and sign_test(cf, th) == {}          # galaxy defaults read no molecule key
    s = cf_summary(cf, labels=("gap", "mu"))
    assert set(s) == {"gap"} and s["gap"]["S_model"]["help"] == 1.0 and s["gap"]["S_model"]["hurt"] == 1.0
    t = sign_test(cf, th, labels=("gap",))
    assert t["gap"]["n"] == n and t["gap"]["help"] == 1.0
