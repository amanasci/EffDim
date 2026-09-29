from pathlib import Path

from sweep.jobs import JOB_SUFFIXES, Layout, build_jobs
from sweep.manifest import load_manifest

RUN = Path("/runners")


def _jobs(tmp_path):
    return build_jobs(load_manifest(), Layout(tmp_path), "/py", RUN, threads=3)


def test_279_jobs_unique_ids(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 31 * 9
    assert len({j.id for j in jobs}) == len(jobs)
    assert {j.suffix for j in jobs} == set(JOB_SUFFIXES)


def test_dependencies(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    assert by["vit_base__cf"].deps == ("vit_base__main_xfit",)
    assert by["vit_base__thin"].deps == ("vit_base__cf",)
    assert all(by[f"vit_base__{s}"].deps == () for s in JOB_SUFFIXES[:7])


def test_split_argv_flags(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    a = by["clip_base__main_xfit"].argv
    assert a[:2] == ("/py", str(RUN / "09_physics_probe_facing_split_run.py"))
    for flag in ("--mode", "--fit-seed", "--hessian-xfit", "--geometry-out", "--parquet-path",
                 "--embedding-column", "--label-table", "--device", "--deterministic", "--record-path", "--threads"):
        assert flag in a, flag
    assert a[a.index("--embedding-column") + 1] == "clip_base_galaxies"
    assert a[a.index("--device") + 1] == "cuda"
    assert "--hessian-xfit" not in by["clip_base__main"].argv
    assert by["clip_base__seed1"].argv[by["clip_base__seed1"].argv.index("--fit-seed") + 1] == "1"
    assert "400,400,400" in by["clip_base__w400"].argv
    assert by["clip_base__alpha1"].argv[by["clip_base__alpha1"].argv.index("--alpha") + 1] == "1"
    d20 = by["clip_base__d20"].argv
    assert d20[d20.index("--d-values") + 1] == "20"
    for s in JOB_SUFFIXES[:7]:
        v = by[f"clip_base__{s}"].argv
        assert v[v.index("--d-values") + 1] == ("20" if s == "d20" else "16")


def test_cf_and_thin_chain_paths(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    lay = Layout(tmp_path)
    geo = lay.geometry / "clip_base" / "09_probe_facing_geometry_d16_seed0.npz"
    cf = by["clip_base__cf"].argv
    assert cf[cf.index("--geometry-npz") + 1] == str(geo)
    arrays = cf[cf.index("--arrays-out") + 1]
    th = by["clip_base__thin"].argv
    assert th[th.index("--arrays-npz") + 1] == arrays
    assert "--device" not in th and "--device" not in cf


def test_record_names_avoid_production_stems(tmp_path):
    for j in _jobs(tmp_path):
        assert not Path(j.outputs[0]).name.startswith(("09_physics_curvature", "09_instrument_adjudication"))
