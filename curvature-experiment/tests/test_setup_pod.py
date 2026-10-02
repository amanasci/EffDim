"""setup_pod.sh: the embedded verification script, run locally against a tiny molecule manifest, and the shell syntax."""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "sweep" / "setup_pod.sh"
ROOT = Path(__file__).resolve().parents[1]


def _heredoc() -> str:
    m = re.search(r"<<'EOF'\n(.*?)\nEOF\n", SCRIPT.read_text(), re.S)
    assert m
    return m.group(1)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _run(tmp_path, manifest):
    code = tmp_path / "embedded.py"
    code.write_text(_heredoc())
    repo = ROOT.parent
    return subprocess.run([sys.executable, str(code), str(repo), str(tmp_path / "sweep-out"), "snap",
                           str(manifest), str(tmp_path / "qm9-out")], capture_output=True, text=True)


def _manifest(tmp_path, table_sha):
    table = tmp_path / "table.parquet"
    table.write_bytes(b"table")
    a, b, c = b"aaa", b"bbb", b"ccc"
    hf = tmp_path / "qm9-out" / "hf"
    hf.mkdir(parents=True)
    (hf / "a.parquet").write_bytes(a)
    (hf / "b.parquet").write_bytes(b)
    enc = lambda n, sha: (f"  - {{name: {n}, family: f, dim: 4, params: 1, params_source: s, in_paper: false, "
                          f"parquet_file: {n}.parquet, column: {n}_col" + (f", parquet_sha256: {sha}" if sha else "") + "}\n")
    p = tmp_path / "m.yaml"
    p.write_text(f"repo: r\nsnapshot: s\nn_rows: 3\nlabel_table: {table}\nlabel_table_sha256: {table_sha}\nlabels: [gap]\n"
                 "encoders:\n" + enc("a", _sha(a)) + enc("b", None) + enc("c", None))
    return p


def test_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_molecule_manifest_reports_verified_unpinned_missing(tmp_path):
    r = _run(tmp_path, _manifest(tmp_path, _sha(b"table")))
    assert r.returncode == 0, r.stderr
    assert "embeddings verified 1 of 3; present but not yet pinned: b; missing: c" in r.stdout


def test_molecule_table_sha_mismatch_fails(tmp_path):
    r = _run(tmp_path, _manifest(tmp_path, "0" * 64))
    assert r.returncode != 0 and "molecule table sha256" in r.stderr


def test_embedding_sha_mismatch_fails(tmp_path):
    m = _manifest(tmp_path, _sha(b"table"))
    (tmp_path / "qm9-out" / "hf" / "a.parquet").write_bytes(b"corrupt")
    r = _run(tmp_path, m)
    assert r.returncode != 0 and "embedding sha256 mismatch for a" in r.stderr


def test_manifest_flag_and_effdim_install_present():
    s = SCRIPT.read_text()
    assert "--manifest) MANIFEST=$2" in s and "--no-deps -e \"$REPO\"" in s
    assert "MANIFEST=curvature-experiment/encoders.yaml" in s
