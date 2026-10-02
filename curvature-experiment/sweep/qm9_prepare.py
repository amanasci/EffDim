"""Build the QM9 molecule table: yairschiff/qm9 at a pinned revision, the 3,054 uncharacterized molecules dropped,
RDKit-canonical SMILES deduplicated (lowest QM9 index kept), gap in eV. Local, CPU.

QM9 index = HF row + 1 (the dataset has no index column; checked against every GDB17 SMILES in the uncharacterized
list). Outputs data/qm9/qm9_molecules.parquet (sorted by qm9_index) and qm9_molecules.summary.json, and writes the
table's sha256 into molecules.yaml (label_table_sha256). The table is both the label table and the SMILES source."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sweep.manifest import MOLECULES_PATH, update_manifest

HERE = Path(__file__).resolve().parents[1]
REPO = "yairschiff/qm9"
REVISION = "fe76afe05ba82b29b2f6b2d4da0cdad8d82e7d0e"
PARQUET = "data/train-00000-of-00001-baa918c342229731.parquet"
PARQUET_SHA256 = "8d526730480ec9db6be5cc30439c77b644c809352bd6c451efba1285cf239c2f"
UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"
UNCHARACTERIZED_SHA256 = "3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24"
HARTREE_TO_EV = 27.211386245988          # CODATA 2018
SOURCE_COLUMNS = ("smiles", "gap", "mu", "alpha", "cv")
OUT_COLUMNS = ("qm9_index", "smiles_canonical", "n_heavy_atoms", "gap", "mu", "alpha", "cv")
OUT_DIR = HERE / "data" / "qm9"
EXPECTED_COUNTS = {"n_source": 133885, "n_uncharacterized": 3054, "n_uncharacterized_smiles_mismatch": 0,
                   "n_parse_failed": 0, "n_duplicate_rows": 87, "n_duplicate_groups": 87, "n": 130744}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_uncharacterized(path) -> Dict[int, str]:
    """QM9 index -> GDB17 SMILES for each data line of uncharacterized.txt (a line whose first field is an integer)."""
    out: Dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 2 and f[0].isdigit():
            out[int(f[0])] = f[1]
    return out


def canonical(smiles: str) -> Optional[Tuple[str, int]]:
    """(RDKit canonical SMILES, heavy-atom count), or None when RDKit cannot parse the string."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol), int(mol.GetNumHeavyAtoms())


def prepare(source: pd.DataFrame, uncharacterized: Dict[int, str]) -> Tuple[pd.DataFrame, dict]:
    df = source.loc[:, list(SOURCE_COLUMNS)].copy()
    df.insert(0, "qm9_index", np.arange(1, len(df) + 1, dtype=np.int64))
    by_idx = df.set_index("qm9_index")["smiles"]
    beyond = sorted(i for i in uncharacterized if i not in by_idx.index)
    if beyond:
        raise ValueError(f"uncharacterized indices beyond the source: {beyond[:5]}")
    mismatch = sum(1 for i, s in uncharacterized.items() if by_idx[i] != s)
    df = df[~df["qm9_index"].isin(list(uncharacterized))]
    parsed = [canonical(s) for s in df["smiles"]]
    ok = np.array([p is not None for p in parsed], dtype=bool)
    df = df[ok].copy()
    df["smiles_canonical"] = [p[0] for p in parsed if p is not None]
    df["n_heavy_atoms"] = np.array([p[1] for p in parsed if p is not None], dtype=np.int64)
    df = df.sort_values("qm9_index", kind="stable")
    dup = df.duplicated("smiles_canonical", keep="first")
    n_groups = int(df.loc[dup, "smiles_canonical"].nunique())
    df = df[~dup].copy()
    df["gap"] = df["gap"] * HARTREE_TO_EV
    out = df.loc[:, list(OUT_COLUMNS)].reset_index(drop=True)
    summary = {"n_source": int(len(source)), "n_uncharacterized": len(uncharacterized),
               "n_uncharacterized_smiles_mismatch": int(mismatch), "n_parse_failed": int((~ok).sum()),
               "n_duplicate_rows": int(dup.sum()), "n_duplicate_groups": n_groups, "n": int(len(out))}
    return out, summary


def check_counts(summary: dict, expected: dict = EXPECTED_COUNTS) -> None:
    """Refuse (SystemExit) unless every expected count matches; names each differing key."""
    bad = [f"{k}: expected {v!r}, got {summary.get(k)!r}" for k, v in expected.items() if summary.get(k) != v]
    if bad:
        raise SystemExit("molecule table counts differ from the expected ones: " + "; ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=HERE / ".cache" / "qm9" / "source", help="download directory")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--manifest", type=Path, default=MOLECULES_PATH)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import rdkit
    a.work.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(REPO, PARQUET, repo_type="dataset", revision=REVISION, local_dir=str(a.work)))
    if sha256_file(src) != PARQUET_SHA256:
        raise SystemExit(f"{src}: sha256 {sha256_file(src)} != pinned {PARQUET_SHA256}")
    unc = a.work / "uncharacterized.txt"
    if not unc.exists():
        req = urllib.request.Request(UNCHARACTERIZED_URL, headers={"User-Agent": "effdim-qm9-prepare"})
        with urllib.request.urlopen(req) as r:
            unc.write_bytes(r.read())
    if sha256_file(unc) != UNCHARACTERIZED_SHA256:
        got = sha256_file(unc)
        unc.unlink()
        raise SystemExit(f"{unc}: sha256 {got} != pinned {UNCHARACTERIZED_SHA256} (file deleted)")
    table, summary = prepare(pd.read_parquet(src, columns=list(SOURCE_COLUMNS)), read_uncharacterized(unc))
    check_counts(summary)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "qm9_molecules.parquet"
    table.to_parquet(out, index=False)
    sha = sha256_file(out)
    summary.update({"repo": REPO, "revision": REVISION, "source_parquet": PARQUET, "source_parquet_sha256": PARQUET_SHA256,
                    "uncharacterized_url": UNCHARACTERIZED_URL, "uncharacterized_sha256": UNCHARACTERIZED_SHA256,
                    "hartree_to_ev": HARTREE_TO_EV, "rdkit": rdkit.__version__, "pandas": pd.__version__, "table_sha256": sha})
    (a.out_dir / "qm9_molecules.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    update_manifest(a.manifest, top={"label_table_sha256": sha})
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
