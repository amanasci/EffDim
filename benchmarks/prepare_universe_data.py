#!/usr/bin/env python3
"""Stream selected UniverseTBD Parquet embedding columns into NumPy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


DATASETS = {
    "jwst_dinov3_vitl16": ("jwst/jwst_dinov3_vitl16.parquet", "_jwst"),
    "desi_dinov3_small_vitl16": (
        "desi/desi_dinov3_small_vitl16.parquet",
        "_desi",
    ),
    "legacysurvey_dinov3_vitl16": (
        "legacysurvey/legacysurvey_dinov3_vitl16.parquet",
        "_legacysurvey",
    ),
}
BASE_URL = "https://huggingface.co/datasets/UniverseTBD/pu-embeddings/resolve/main"


def _embedding_column(
    connection: duckdb.DuckDBPyConnection, url: str, target_suffix: str
) -> str:
    rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [url]
    ).fetchall()
    candidates = [
        name
        for name, data_type, *_ in rows
        if "[]" in data_type or data_type.startswith("FLOAT[") or data_type.startswith("DOUBLE[")
    ]
    matching = [name for name in candidates if name.endswith(target_suffix)]
    if len(matching) == 1:
        return matching[0]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one embedding column, found {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    metadata = []
    for name, (relative_path, target_suffix) in DATASETS.items():
        url = f"{BASE_URL}/{relative_path}"
        print(f"Reading {name}...", flush=True)
        column = _embedding_column(connection, url, target_suffix)
        escaped_column = column.replace('"', '""')
        table = connection.execute(
            f'SELECT "{escaped_column}" FROM read_parquet(?) LIMIT ?',
            [url, args.limit],
        ).fetch_arrow_table()
        values = table.column(0).combine_chunks()
        width = len(values[0].as_py())
        matrix = values.values.to_numpy(zero_copy_only=False).reshape(len(values), width)
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        output = args.output_dir / f"{name}.npy"
        np.save(output, matrix)
        item = {
            "name": name,
            "source": f"UniverseTBD/pu-embeddings:{relative_path}",
            "column": column,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "path": str(output.resolve()),
        }
        metadata.append(item)
        print(f"  {matrix.shape} -> {output}", flush=True)

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
