from pathlib import Path

import pytest
import yaml

from sweep.manifest import load_manifest

PAPER = {"vit_base", "vit_large", "clip_base", "convnext_base", "dinov3_vitb16"}


def test_manifest_has_31_unique_encoders():
    m = load_manifest()
    names = [e.name for e in m.encoders]
    assert len(names) == 31 and len(set(names)) == 31
    assert {e.name for e in m.encoders if e.in_paper} == PAPER
    assert m.snapshot == "bc081f8a5db4767edcd958653d96efde9137de0b"
    assert m.n_rows == 86471


def test_every_encoder_has_dim_params_and_source():
    for e in load_manifest().encoders:
        assert e.dim > 0 and e.params > 0, e.name
        assert e.params_source.startswith("https://"), e.name
        assert e.parquet_file == f"physics/{e.name}_test.parquet"
        assert e.column == f"{e.name}_galaxies"


def test_manifest_rejects_duplicate_names(tmp_path):
    src = yaml.safe_load(Path(load_manifest.__globals__["DEFAULT_PATH"]).read_text())
    src["encoders"].append(dict(src["encoders"][0]))
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(src))
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)
