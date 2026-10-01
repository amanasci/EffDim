"""Second-domain flags on the split, counterfactual and robustness runners; the galaxy defaults change nothing."""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


split = _load("09_physics_probe_facing_split_run.py")
ns = _load("09_physics_normal_scaling_run.py")
rr = _load("11_review_robustness_run.py")
pl = split.ppf.pl
PL_NAMES = ("EXPECTED_N_PHYSICS_ROWS", "LABEL_COLUMN_MAP", "SENTINEL_VALUES", "PHYSICS_PARQUET_PATH", "PHYSICS_COLUMN",
            "load_physics_embeddings", "load_label_table")


@pytest.fixture
def restore_pl(monkeypatch):
    """The runners assign physics_labels globals; registering them with monkeypatch undoes each test's changes."""
    for name in PL_NAMES:
        monkeypatch.setattr(pl, name, getattr(pl, name))
    return pl


def test_domain_flags_default_is_noop(restore_pl):
    before = (pl.EXPECTED_N_PHYSICS_ROWS, dict(pl.LABEL_COLUMN_MAP), pl.SENTINEL_VALUES)
    args = split.build_parser().parse_args(["--mode", "physics"])
    assert (args.expected_rows, args.label_map, args.d_file_sha256) == (None, "galaxy", None)
    split.apply_domain_flags(args)
    assert (pl.EXPECTED_N_PHYSICS_ROWS, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES) == before
    assert split.domain_env(args) == {}


def test_identity_map_and_expected_rows(restore_pl):
    args = split.build_parser().parse_args(["--mode", "physics", "--labels", "gap,mu", "--label-map", "identity",
                                            "--expected-rows", "50", "--d-file-sha256", "ab" * 32])
    split.apply_domain_flags(args)
    assert pl.EXPECTED_N_PHYSICS_ROWS == 50 and pl.LABEL_COLUMN_MAP == {"gap": "gap", "mu": "mu"} and pl.SENTINEL_VALUES == ()
    assert split.domain_env(args) == {"expected_rows": 50, "label_map": "identity", "d_file_sha256": "ab" * 32}


def _toy_files(tmp_path, n=50, D=8):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, D)).astype(np.float32)
    e = tmp_path / "e.parquet"
    pq.write_table(pa.table({"toy_molecules": pa.array(X.tolist(), type=pa.list_(pa.float32()))}), e)
    t = tmp_path / "t.parquet"
    pd.DataFrame({"qm9_index": np.arange(1, n + 1), "gap": rng.standard_normal(n), "mu": np.full(n, -99.0)}).to_parquet(t, index=False)
    return X, e, t


def _robust_args(tmp_path, rows):
    return rr.build_parser().parse_args(["--record-path", str(tmp_path / "r.jsonl"), "--labels", "gap,mu",
                                         "--label-map", "identity", "--expected-rows", str(rows)])


def test_molecule_loaders_through_the_shims(restore_pl, tmp_path):
    X, e, t = _toy_files(tmp_path)
    args = _robust_args(tmp_path, 50)
    rr.pfs.apply_domain_flags(args)
    rr.install_shims(str(e), "toy_molecules", str(t))
    data = rr.ppf.load_physics(argparse.Namespace(labels=args.labels))
    assert data["X"].shape == (50, 8)
    np.testing.assert_allclose(np.linalg.norm(data["X"], axis=1), 1.0)
    frame = pd.read_parquet(t)
    np.testing.assert_array_equal(data["labels"]["gap"], frame["gap"].to_numpy())
    np.testing.assert_array_equal(data["labels"]["mu"], frame["mu"].to_numpy())     # -99 is a value here, not a sentinel


def test_wrong_expected_rows_refuses(restore_pl, tmp_path):
    _, e, t = _toy_files(tmp_path)
    rr.pfs.apply_domain_flags(_robust_args(tmp_path, 51))
    rr.install_shims(str(e), "toy_molecules", str(t))
    with pytest.raises(RuntimeError, match="expected 51"):
        rr.ppf.load_physics(argparse.Namespace(labels="gap,mu"))


def test_counterfactual_parser_takes_domain_flags():
    a = ns.build_parser().parse_args(["--mode", "physics", "--label-map", "identity", "--expected-rows", "7", "--d-file-sha256", "x"])
    assert (a.label_map, a.expected_rows, a.d_file_sha256) == ("identity", 7, "x")


def test_robust_labels_and_d():
    a = rr.build_parser().parse_args(["--record-path", "r", "--labels", "gap,mu", "--d", "20"])
    assert a.labels == "gap,mu" and a.d == 20
    b = rr.build_parser().parse_args(["--record-path", "r"])
    assert b.labels == "mag_r,photo_z,smooth_fraction,stellar_mass" and b.d == 16 and b.label_map == "galaxy"
