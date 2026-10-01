from pathlib import Path

import pytest

from sweep.jobs import JOB_SUFFIXES, Layout, build_jobs
from sweep.manifest import load_manifest

RUN = Path("/runners")
TEN = ["vit_base", "clip_base", "convnext_base", "vit_large", "dinov3_vits16", "dinov3_vits16plus",
       "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16"]


def _jobs(tmp_path, encoders=None):
    return build_jobs(load_manifest(), Layout(tmp_path), "/py", RUN, threads=3, encoders=encoders)


def test_battery_and_counts(tmp_path):
    assert JOB_SUFFIXES == ("main_xfit", "seed1", "seed2", "cf", "thin", "robust")
    allj = _jobs(tmp_path)
    assert len(allj) == 31 * 6 and len({j.id for j in allj}) == len(allj)
    ten = _jobs(tmp_path, TEN)
    assert len(ten) == 60 and {j.encoder for j in ten} == set(TEN)


def test_build_jobs_unknown_encoder_raises(tmp_path):
    with pytest.raises(ValueError, match="dinov3_vit7b"):
        _jobs(tmp_path, ["vit_base", "dinov3_vit7b"])


def test_dependencies(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path, TEN)}
    assert by["vit_base__cf"].deps == ("vit_base__main_xfit",)
    assert by["vit_base__thin"].deps == ("vit_base__cf",)
    assert by["vit_base__robust"].deps == ("vit_base__main_xfit", "vit_base__cf")
    assert all(by[f"vit_base__{s}"].deps == () for s in ("main_xfit", "seed1", "seed2"))


def test_split_argv_flags(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path, TEN)}
    a = by["clip_base__main_xfit"].argv
    assert a[:2] == ("/py", str(RUN / "09_physics_probe_facing_split_run.py"))
    for flag in ("--mode", "--fit-seed", "--hessian-xfit", "--geometry-out", "--parquet-path",
                 "--embedding-column", "--label-table", "--device", "--deterministic", "--record-path", "--threads"):
        assert flag in a, flag
    assert a[a.index("--embedding-column") + 1] == "clip_base_galaxies"
    for s, seed in (("seed1", "1"), ("seed2", "2")):
        v = by[f"clip_base__{s}"].argv
        assert v[v.index("--fit-seed") + 1] == seed and "--hessian-xfit" not in v and "--geometry-out" not in v
        assert v[v.index("--d-values") + 1] == "16"


def test_cf_and_thin_chain_paths(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path, TEN)}
    lay = Layout(tmp_path)
    geo = lay.geometry / "clip_base" / "09_probe_facing_geometry_d16_seed0.npz"
    cf = by["clip_base__cf"].argv
    assert cf[cf.index("--geometry-npz") + 1] == str(geo)
    arrays = cf[cf.index("--arrays-out") + 1]
    th = by["clip_base__thin"].argv
    assert th[th.index("--arrays-npz") + 1] == arrays
    assert "--device" not in th and "--device" not in cf


def test_robust_argv(tmp_path):
    m = load_manifest(); lay = Layout(tmp_path)
    j = {x.id: x for x in _jobs(tmp_path, TEN)}["dinov3_vitl16__robust"]
    a = j.argv
    def val(f): return a[a.index(f) + 1]
    assert a[1] == str(RUN / "11_review_robustness_run.py")
    assert val("--encoder") == "dinov3_vitl16"
    assert val("--geometry-npz") == str(lay.geometry / "dinov3_vitl16" / "09_probe_facing_geometry_d16_seed0.npz")
    assert val("--parquet-path") == str(lay.hf_parquet(next(e for e in m.encoders if e.name == "dinov3_vitl16")))
    assert val("--embedding-column") == "dinov3_vitl16_galaxies"
    assert val("--label-table") == m.label_table and val("--label-table-sha256") == m.label_table_sha256
    assert val("--published-split") == str(lay.records / "scaling__dinov3_vitl16__main_xfit.jsonl")
    assert val("--published-cf") == str(lay.arrays / "scaling__dinov3_vitl16__cf.npz")
    assert val("--published-cf-record") == str(lay.records / "scaling__dinov3_vitl16__cf.jsonl")
    assert val("--guard") == "exact" and val("--threads") == "3"
    assert val("--record-path") == str(lay.records / "scaling__dinov3_vitl16__robust.jsonl")
    assert j.outputs == (val("--record-path"),) and j.kind == "robust"
    assert m.label_table_sha256 == "60f2f82e64e4036eb4eff9a448b15365dd1e974e52779fe7c1323db0d9dabfd9"


def test_record_names_avoid_production_stems(tmp_path):
    for j in _jobs(tmp_path):
        assert not Path(j.outputs[0]).name.startswith(("09_physics_curvature", "09_instrument_adjudication"))
