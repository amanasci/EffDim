"""The QM9 molecule table: index rule, exclusion by index, canonicalisation, dedup keeping the lowest index, units."""
import pandas as pd
import pytest

from sweep import qm9_prepare as qp

UNCHAR = """List of molecules among the 133885 GDB9 molecules for which the Corina generated Cartesian coordinates and
different SMILES and InChI strings when parsed through the program Openbabel.

==============================================================================================================
#   Index                           GDB17 SMILES                         SMILES for B3LYP XYZ    D_IJ
==============================================================================================================
      6                               CC                                    CC                  [CH3][CH3]      1.000
==============================================================================================================
"""


def _source():
    return pd.DataFrame({"smiles": ["C", "CO", "OC", "C1CC1", "not_a_smiles", "CC", "OCC"],
                         "gap": [1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "mu": [0.1] * 7, "alpha": [2.0] * 7, "cv": [3.0] * 7})


def test_read_uncharacterized(tmp_path):
    p = tmp_path / "u.txt"; p.write_text(UNCHAR)
    assert qp.read_uncharacterized(p) == {6: "CC"}


def test_prepare_index_exclusion_canonical_dedup_units(tmp_path):
    p = tmp_path / "u.txt"; p.write_text(UNCHAR)
    table, s = qp.prepare(_source(), qp.read_uncharacterized(p))
    assert list(table.columns) == list(qp.OUT_COLUMNS)
    assert table["qm9_index"].tolist() == [1, 2, 4, 7]                 # 3 duplicates 2 ("OC" == "CO"), 5 unparsable, 6 excluded
    assert table["smiles_canonical"].tolist() == ["C", "CO", "C1CC1", "CCO"]
    assert table["n_heavy_atoms"].tolist() == [1, 2, 3, 3]
    assert table.loc[0, "gap"] == pytest.approx(qp.HARTREE_TO_EV) and qp.HARTREE_TO_EV == 27.211386245988
    assert table.loc[0, "mu"] == 0.1 and table.loc[0, "alpha"] == 2.0 and table.loc[0, "cv"] == 3.0
    assert s == {"n_source": 7, "n_uncharacterized": 1, "n_uncharacterized_smiles_mismatch": 0, "n_parse_failed": 1,
                 "n_duplicate_rows": 1, "n_duplicate_groups": 1, "n": 4}


def test_prepare_counts_a_smiles_mismatch_at_an_excluded_index():
    _, s = qp.prepare(_source(), {6: "CCC"})
    assert s["n_uncharacterized_smiles_mismatch"] == 1


def test_prepare_rejects_an_index_beyond_the_source():
    with pytest.raises(ValueError, match="99"):
        qp.prepare(_source(), {99: "C"})


def test_check_counts_passes_on_the_expected_counts():
    qp.check_counts(dict(qp.EXPECTED_COUNTS))


def test_check_counts_names_every_differing_key():
    bad = dict(qp.EXPECTED_COUNTS, n=130745, n_parse_failed=2)
    with pytest.raises(SystemExit) as e:
        qp.check_counts(bad)
    msg = str(e.value)
    assert "n_parse_failed" in msg and "n: expected 130744, got 130745" in msg and "n_source" not in msg
