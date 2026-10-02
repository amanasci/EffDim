"""Exact duplicate rows in the molecule embeddings."""
import hashlib
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sweep import embedding_duplicates as dup
from sweep.manifest import MOLECULES_PATH, load_manifest


def test_duplicate_counts():
    E = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4], [1, 2], [7, 8]], dtype=np.float32)
    assert dup.duplicate_counts(E) == {"n_rows": 7, "n_unique": 4, "rows_in_duplicate_groups": 5,
                                       "duplicate_groups": 2, "largest_group": 3}
    assert dup.duplicate_counts(np.eye(3, dtype=np.float32)) == {"n_rows": 3, "n_unique": 3, "rows_in_duplicate_groups": 0,
                                                                  "duplicate_groups": 0, "largest_group": 1}


def _write(path, column, E):
    n, D = E.shape
    col = pa.ListArray.from_arrays(pa.array(np.arange(0, n * D + 1, D, dtype=np.int32)), pa.array(E.ravel()))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: col}), path)


def test_main_writes_json(tmp_path, monkeypatch, capsys):
    m = load_manifest(MOLECULES_PATH)
    for i, e in enumerate(m.encoders):
        E = np.arange(12, dtype=np.float32).reshape(4, 3) + i
        if e.name == "chemberta_5m_mtr":
            E[2] = E[0]
        _write(tmp_path / "hf" / e.parquet_file, e.column, E)
    out = tmp_path / "dup.json"
    monkeypatch.setattr(sys, "argv", ["x", "--manifest", str(MOLECULES_PATH), "--root", str(tmp_path), "--out", str(out)])
    dup.main()
    res = json.loads(out.read_text())
    assert res["numpy"] == np.__version__
    assert set(res) == {"numpy"} | {e.name for e in m.encoders}
    c = res["chemberta_5m_mtr"]
    assert (c["n_rows"], c["n_unique"], c["rows_in_duplicate_groups"], c["duplicate_groups"], c["largest_group"]) == (4, 3, 2, 1, 2)
    assert res["molformer_xl"]["rows_in_duplicate_groups"] == 0
    p = tmp_path / "hf" / m.encoders[0].parquet_file
    assert res[m.encoders[0].name]["parquet_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(capsys.readouterr().out.strip().splitlines()) == len(m.encoders)
