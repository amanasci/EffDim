"""Paper invocations still parse, and the production-runner names the kept runners use exist.

Guards trimming: full physics mode is not exercised by smoke, so this pins the CLI surface and
the cross-runner attributes that physics mode reads.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


PAPER_ARGS = {
    "09_physics_probe_facing_split_run.py": [
        ["--mode", "physics", "--d-values", "16,20", "--label-table", "x.parquet"],
        ["--mode", "physics", "--fit-seed", "1"],
        ["--mode", "physics", "--fit-seed", "0", "--hidden", "400,400,400"],
        ["--mode", "physics", "--hessian-xfit"],
        ["--mode", "physics", "--alpha", "1"],
        ["--mode", "physics", "--fit-seed", "0", "--hessian-xfit", "--parquet-path", "p.parquet",
         "--embedding-column", "clip_base_galaxies", "--geometry-out", "g.npz"],
    ],
    "09_physics_normal_scaling_run.py": [
        ["--mode", "physics", "--geometry-npz", "g.npz", "--arrays-out", "a.npz"],
    ],
    "09_physics_normal_scaling_thin_run.py": [
        ["--parquet-path", "p.parquet", "--embedding-column", "c", "--arrays-npz", "a.npz", "--out", "o.npz"],
    ],
    "09_fixture_probe_facing_run.py": [["--mode", "full"]],
    "09_fixture_probe_facing_split_run.py": [["--mode", "full"]],
    "09_fixture_probe_decodability_run.py": [["--mode", "full", "--gammas", "-1,0,1"], ["--mode", "full", "--gammas", "0.4,0.6,0.8"]],
    "09_instrument_adjudication_run.py": [["--mode", "sphere-fixture", "--noise", "0"]],
}


@pytest.mark.parametrize("runner_file,argv", [(r, a) for r, v in PAPER_ARGS.items() for a in v])
def test_paper_invocation_parses(runner_file, argv):
    mod = _load(runner_file)
    builder = getattr(mod, "build_parser", None) or getattr(mod, "build_arg_parser", None)
    parser = builder() if builder is not None else None
    if parser is None:
        pytest.skip(f"{runner_file} builds its parser inside main(); covered by the gate's smoke run")
    parser.parse_args(argv)


def test_production_runner_names_used_by_kept_runners_exist():
    adj = _load("09_instrument_adjudication_run.py")
    for name in ("_THREADS", "fit_and_field_at_anchors", "SphereProjectedDecoder", "_oof_predictions_for_label"):
        assert hasattr(adj.runner, name), name
