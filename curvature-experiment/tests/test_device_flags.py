"""--device / --deterministic on the split runner; default CPU path unchanged."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_split_parser_device_defaults():
    split = _load("09_physics_probe_facing_split_run.py")
    args = split.build_parser().parse_args(["--mode", "smoke"])
    assert args.device == "cpu" and args.deterministic is False
    args = split.build_parser().parse_args(["--mode", "physics", "--device", "cuda", "--deterministic"])
    assert args.device == "cuda" and args.deterministic is True


def test_fit_decoder_cpu_device_matches_default():
    ppf = _load("09_physics_probe_facing_run.py")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = ppf.fit_decoder(X, 2, 12, 3)
    b = ppf.fit_decoder(X, 2, 12, 3, device="cpu")
    for pa, pb in zip(a["model"].state_dict().values(), b["model"].state_dict().values()):
        assert torch.equal(pa, pb)
    assert a["var_explained"] == b["var_explained"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU refusal")
def test_cuda_without_gpu_refuses(tmp_path, monkeypatch):
    split = _load("09_physics_probe_facing_split_run.py")
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "smoke", "--device", "cuda", "--threads", "8",
                                      "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="CUDA is not available"):
        split.main()


def test_load_physics_rejects_wrong_row_count(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ppf = _load("09_physics_probe_facing_run.py")
    p = tmp_path / "e.parquet"
    pq.write_table(pa.table({"m_galaxies": [[0.1, 0.2, 0.3]] * 5}), p)
    with pytest.raises(Exception, match="86471|rows"):
        ppf.pl.load_physics_embeddings(parquet_path=str(p), column="m_galaxies")


def test_decoder_geometry_out_chunk_matches_unchunked():
    ppf = _load("09_physics_probe_facing_run.py")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 12)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    fit = ppf.fit_decoder(X, 3, 12, 3)
    with torch.no_grad():
        z = fit["model"].encode(fit["x64"][:40])
    ref = ppf.decoder_geometry(fit["curvature_model"], z)
    got = ppf.decoder_geometry(fit["curvature_model"], z, out_chunk=5)
    for k in ("J", "Hess", "image", "g", "II", "H"):
        assert got[k].shape == ref[k].shape, k
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-12, atol=1e-12, err_msg=k)


def test_geometry_out_chunk_only_off_cpu():
    split = _load("09_physics_probe_facing_split_run.py")
    assert split.geometry_out_chunk("cpu") is None
    assert split.geometry_out_chunk("cuda") == split.GPU_GEOMETRY_OUT_CHUNK > 0
