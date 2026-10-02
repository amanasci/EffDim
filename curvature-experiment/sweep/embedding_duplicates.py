"""Exact duplicate rows in each molecule encoder's embeddings (pod CPU, after embed).

A ChemBERTa-2 tokenizer maps some distinct molecules to one token sequence, so they share an embedding. This counts,
per encoder, the rows whose embedding equals another row's exactly (np.unique over rows) and writes the counts that
the QM9 report quotes.

Usage:
    python -m sweep.embedding_duplicates --manifest molecules.yaml --root <qm9-out> --out data/qm9/embedding_duplicates.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np

from sweep.intrinsic_dim import load_embeddings
from sweep.jobs import Layout
from sweep.manifest import MOLECULES_PATH, load_manifest


def duplicate_counts(E: np.ndarray) -> Dict[str, int]:
    _, counts = np.unique(np.asarray(E), axis=0, return_counts=True)
    dup = counts[counts > 1]
    return {"n_rows": int(E.shape[0]), "n_unique": int(counts.size), "rows_in_duplicate_groups": int(dup.sum()),
            "duplicate_groups": int(dup.size), "largest_group": int(counts.max())}


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(MOLECULES_PATH))
    ap.add_argument("--root", required=True, help="output root holding hf/<parquet_file>")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m = load_manifest(a.manifest); lay = Layout(Path(a.root))
    out: dict = {"numpy": np.__version__}
    for e in m.encoders:
        p = lay.hf_parquet(e)
        c = duplicate_counts(load_embeddings(p, e.column))
        out[e.name] = {**c, "parquet_sha256": _sha256_file(p)}
        print(f"{e.name}: {c['rows_in_duplicate_groups']} of {c['n_rows']} rows in {c['duplicate_groups']} duplicate groups "
              f"(largest {c['largest_group']})", flush=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
