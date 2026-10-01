import json
from pathlib import Path

import numpy as np

from sweep.aggregate import aggregate
from sweep.extract import LABELS
from sweep.manifest import load_manifest

COLS = ("hess_mismatch_emp", "align_cos_tan", "hess_mismatch_dec", "pf_rad")


def _split(path, partial, p=0.001):
    rows = [{"row": "environment", "device": "cuda"}, {"row": "fit", "var_explained": 0.95, "geometry_npz_sha256": "sha0"}]
    for lab in LABELS:
        rows.append({"row": "result", "label": lab, "global_oof_r2": 0.5,
                     "columns": {c: {"multiscale": {"partial": partial, "p": p}} for c in COLS}})
        rows.append({"row": "xfit", "label": lab, "hessian_split_half_cos_p50": 0.9,
                     "columns": {c: {"fitA_scoreB": {"partial": partial, "p": p}, "fitB_scoreA": {"partial": partial, "p": p}} for c in COLS}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _robust(path, sha="sha0"):
    rows = [{"row": "environment", "geometry_sha256": sha}]
    for lab in LABELS:
        for mode in ("published", "tuned"):
            rows.append({"row": "result", "label": lab, "alpha_mode": mode, "alpha": 100.0 if mode == "published" else 0.1,
                         "partials": {"published_controls": {c: {"partial": -0.4, "p": 0.001} for c in ("hess_mismatch_emp", "align_cos_tan")}}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _cf_record(path, sha="sha0"):
    path.write_text(json.dumps({"row": "environment", "geometry_sha256": sha, "threads": 3}) + "\n")


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
        for s in ("main_xfit", "seed1", "seed2"):
            _split(rec / f"scaling__{e}__{s}.jsonl", -0.3 if s != "seed2" else 0.1)
        _cf(arr / f"scaling__{e}__cf.npz"); _thin(arr / f"scaling__{e}__thin.npz")
        _robust(rec / f"scaling__{e}__robust.jsonl"); _cf_record(rec / f"scaling__{e}__cf.jsonl")
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
    assert "2 of 31 encoders complete all 6 jobs" in rep


def test_robust_counts_sign_changes(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rob = (out / "tab_scaling_robust.tex").read_text()
    row = next(l for l in rob.splitlines() if l.startswith("vit\\_base & mag\\_r"))
    assert row == "vit\\_base & mag\\_r & $[-0.30, +0.10]$ & 1 & $[-0.30, +0.10]$ & 1 \\\\"   # seed2 flips the sign once


def test_robust_marks_partial_coverage(tmp_path):
    rec, arr = _fixture(tmp_path)
    (rec / "scaling__vit_base__seed1.jsonl").unlink()
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rob = (out / "tab_scaling_robust.tex").read_text()
    row = next(l for l in rob.splitlines() if l.startswith("vit\\_base & mag\\_r"))
    assert "$[-0.30, +0.10]$ ($n=2$) & 1" in row


def test_report_reversal_hurts(tmp_path):
    rec, arr = _fixture(tmp_path)
    z = dict(np.load(arr / "scaling__vit_base__cf.npz"))
    for lab in LABELS:                                     # random direction never hurts on vit_base
        cv = z[f"{lab}:random_qmatched:r2_curve"]; cv[:, 0] = cv[:, 2] + 0.1
    np.savez(arr / "scaling__vit_base__cf.npz", **z)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rep = (out / "SCALING_REPORT.md").read_text()
    assert "- mag_r: hurt > 0.5 and hurt > random hurt in 1 of 2" in rep
    assert "- (c') hurt > 0.5 and hurt > random hurt, mag_r: clip_base" in rep
    assert "mag_r and photo_z positive and significant; stellar_mass non-significant" in rep
    assert "- stellar_mass: 3 negative-significant / 0 positive-significant / 0 non-significant (of 3); paper claim: non-significant" in rep
    assert "- (b) alignment non-significant, stellar_mass: clip_base, dinov3_vitb16, vit_base" in rep
    assert "- smooth_fraction: 3 negative-significant / 0 positive-significant / 0 non-significant (of 3); no paper claim" in rep
    assert "(b) alignment positive and significant, smooth_fraction" not in rep
    assert not any(l.startswith("- (b)") and "smooth_fraction" in l for l in rep.splitlines())
    assert "- (b) alignment positive and significant, mag_r: clip_base, dinov3_vitb16, vit_base" in rep


def test_aggregate_is_deterministic(tmp_path):
    rec, arr = _fixture(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    aggregate(load_manifest(), rec, arr, a); aggregate(load_manifest(), rec, arr, b)
    for f in ("tab_scaling_main.tex", "tab_scaling_cf.tex", "SCALING_REPORT.md", "fig_scaling_cf.png"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_report_flags_stale_geometry(tmp_path):
    rec, arr = _fixture(tmp_path)
    _robust(rec / "scaling__clip_base__robust.jsonl", sha="sha1")        # robust read a newer geometry than cf
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rep = (out / "SCALING_REPORT.md").read_text()
    assert "## Stale encoders" in rep and "- clip_base: robust geometry sha differs from main_xfit's" in rep
    assert "- vit_base:" not in rep.split("## Stale encoders")[1].split("##")[0]


def test_aggregate_encoder_filter(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out, encoders=["vit_base", "clip_base", "dinov3_vitb16"])
    rep = (out / "SCALING_REPORT.md").read_text()
    assert rep.startswith("3 of 3 encoders have records\n2 of 3 encoders complete all 6 jobs")


from sweep.aggregate import compare_cell


def test_comparison_borderline_and_missing():
    assert compare_cell(None, {"partial": -0.3, "p": 0.001}, 0.05) == "missing"
    assert compare_cell({"partial": -0.3, "p": 0.001}, None, 0.05) == "missing"
    assert compare_cell({"partial": -0.30, "p": 0.001}, {"partial": -0.28, "p": 0.002}, 0.05) == "agree"
    assert compare_cell({"partial": -0.30, "p": 0.001}, {"partial": -0.10, "p": 0.002}, 0.05) == "disagree"   # |d| > tol
    assert compare_cell({"partial": -0.30, "p": 0.001}, {"partial": +0.30, "p": 0.001}, 1.0) == "disagree"    # sign
    assert compare_cell({"partial": -0.10, "p": 0.04}, {"partial": -0.09, "p": 0.07}, 0.05) == "borderline"


def _published(tmp_path):
    pub = tmp_path / "published"; pub.mkdir()
    for name, part in (("09_physics_probe_facing_split.jsonl", -0.30), ("09_physics_probe_facing_split_seed1.jsonl", -0.26),
                       ("09_physics_probe_facing_split_seed2.jsonl", -0.34), ("09_physics_probe_facing_split_clip_base.jsonl", -0.31)):
        rows = [{"row": "environment"}]
        for d in (16, 20):
            for lab in LABELS:
                rows.append({"row": "result", "d": d, "label": lab, "global_oof_r2": 0.5,
                             "columns": {c: {"multiscale": {"partial": part if d == 16 else 0.9, "p": 0.001}} for c in COLS}})
        (pub / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for e in ("vit_base", "clip_base"):
        _cf(pub / f"09_physics_normal_scaling_{e}_d16.npz"); _thin(pub / f"09_physics_normal_scaling_{e}_d16_thin.npz")
    return pub


def test_report_published_comparison_and_ladder(tmp_path):
    rec, arr = _fixture(tmp_path)
    pub = _published(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out, encoders=["vit_base", "clip_base", "dinov3_vitb16"], published_dir=pub)
    rep = (out / "SCALING_REPORT.md").read_text()
    sec = rep.split("## Published five: sweep GPU versus published CPU")[1].split("\n## ")[0]
    assert "tolerance = ViT-B's published seed spread" in sec
    assert "| vit_base | mag_r | hess_mismatch_emp | -0.30 | -0.30 | agree |" in sec
    assert "OOF R2 identity: vit_base max |diff| 0" in sec
    assert "| dinov3_vitb16 | mag_r | hess_mismatch_emp | -- |" in sec           # no published file in the fixture
    assert "| vit_base | mag_r | y/y | y/y |" in sec                              # counterfactual yes/no agreement
    lad = rep.split("## DINOv3 size ladder")[1]
    assert "descriptive" in lad and "| dinov3_vitb16 |" in lad
