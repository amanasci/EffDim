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
from sweep.manifest import MOLECULES_PATH, update_manifest
from qm9_fixtures import MOL_MANIFEST

SPEC = {
    "chemberta_5m_mtr": ("DeepChem/ChemBERTa-5M-MTR", "a642f2611383af9c37550b805ca0ecc279079914", 384),
    "chemberta_10m_mtr": ("DeepChem/ChemBERTa-10M-MTR", "b65d0a6af3156071d9519e867d695aa265bb393f", 384),
    "chemberta_77m_mtr": ("DeepChem/ChemBERTa-77M-MTR", "66b895cab8adebea0cb59a8effa66b2020f204ca", 384),
    "chemberta_10m_mlm": ("DeepChem/ChemBERTa-10M-MLM", "a5cfe173103cff3149e7322130342a4880010cba", 384),
    "chemberta_77m_mlm": ("DeepChem/ChemBERTa-77M-MLM", "ed8a5374f2024ec8da53760af91a33fb8f6a15ff", 384),
    "molformer_xl": ("ibm-research/MoLFormer-XL-both-10pct", "361063d0ad524ef77cf39b08469f6be770dc550f", 768),
    "chemfm_1b": ("ChemFM/ChemFM-1B", "f99dc2e89726539bb9cf31b2e2b4360650bac6a8", 2048),
    "chemfm_3b": ("ChemFM/ChemFM-3B", "c1464dd3c51643d2d6926a71db98558be97e1ecf", 3072),
}


def test_molecule_manifest():
    assert MOLECULES_PATH == MOL_MANIFEST
    m = load_manifest(MOLECULES_PATH)
    assert m.labels == ("gap", "mu", "alpha", "cv") and m.label_map == "identity"
    assert m.n_rows == 130744 and m.expected_rows == 130744
    assert m.repo == "yairschiff/qm9" and m.snapshot == "fe76afe05ba82b29b2f6b2d4da0cdad8d82e7d0e"
    assert m.label_table == "/mnt/ssd-cluster/EffDim/repo/curvature-experiment/data/qm9/qm9_molecules.parquet"
    assert {e.name: (e.hf_id, e.revision, e.dim) for e in m.encoders} == SPEC
    for e in m.encoders:
        assert e.parquet_file == f"molecules/{e.name}.parquet" and e.column == f"{e.name}_molecules" and e.in_paper is False
    params = {e.name: e.params for e in m.encoders}
    assert (params["molformer_xl"], params["chemfm_1b"], params["chemfm_3b"]) == (46805760, 970287104, 3004357632)


def test_galaxy_manifest_has_no_molecule_fields():
    m = load_manifest()
    assert m.labels is None and m.label_map is None and m.expected_rows is None
    assert all(e.parquet_sha256 is None and e.hf_id is None and e.revision is None for e in m.encoders)


def test_update_manifest_keeps_header_and_sets_fields(tmp_path):
    p = tmp_path / "m.yaml"; p.write_text(MOLECULES_PATH.read_text())
    update_manifest(p, top={"label_table_sha256": "ab" * 32}, encoders={"molformer_xl": {"parquet_sha256": "cd" * 32}})
    assert p.read_text().startswith("# QM9 molecule manifest")
    m = load_manifest(p)
    assert m.label_table_sha256 == "ab" * 32 and m.labels == ("gap", "mu", "alpha", "cv")
    assert next(e for e in m.encoders if e.name == "molformer_xl").parquet_sha256 == "cd" * 32
    with pytest.raises(ValueError, match="nope"):
        update_manifest(p, encoders={"nope": {"params": 1}})
