"""Encoder manifests for the sweep: the 31-encoder galaxy manifest (encoders.yaml) and the QM9 molecule manifest
(molecules.yaml). Molecule-only fields are optional; the galaxy manifest loads unchanged."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "encoders.yaml"
MOLECULES_PATH = Path(__file__).resolve().parents[1] / "molecules.yaml"


@dataclass(frozen=True)
class Encoder:
    name: str
    family: str
    dim: int
    params: int
    params_source: str
    in_paper: bool
    parquet_file: Optional[str] = None     # None -> the galaxy physics/<name>_test.parquet
    column: Optional[str] = None           # None -> the galaxy <name>_galaxies
    parquet_sha256: Optional[str] = None
    hf_id: Optional[str] = None
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        if self.parquet_file is None:
            object.__setattr__(self, "parquet_file", f"physics/{self.name}_test.parquet")
        if self.column is None:
            object.__setattr__(self, "column", f"{self.name}_galaxies")


@dataclass(frozen=True)
class Manifest:
    repo: str
    snapshot: str
    n_rows: int
    label_table: str
    encoders: Tuple[Encoder, ...]
    label_table_sha256: Optional[str] = None
    labels: Optional[Tuple[str, ...]] = None
    label_map: Optional[str] = None
    expected_rows: Optional[int] = None


def load_manifest(path: Union[str, Path] = DEFAULT_PATH) -> Manifest:
    src = yaml.safe_load(Path(path).read_text())
    encs = tuple(Encoder(**e) for e in src["encoders"])
    names = [e.name for e in encs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate encoder names in {path}: {dup}")
    return Manifest(repo=src["repo"], snapshot=src["snapshot"], n_rows=int(src["n_rows"]),
                    label_table=src["label_table"], encoders=encs,
                    label_table_sha256=src.get("label_table_sha256"),
                    labels=tuple(src["labels"]) if src.get("labels") else None,
                    label_map=src.get("label_map"),
                    expected_rows=int(src["expected_rows"]) if src.get("expected_rows") is not None else None)


def update_manifest(path: Union[str, Path], top: Optional[dict] = None,
                    encoders: Optional[Dict[str, dict]] = None) -> None:
    """Set top-level keys and per-encoder keys in a manifest YAML, keeping its leading comment block.
    Used by qm9_prepare (label_table_sha256) and `embed --pin` (parquet_sha256, ChemBERTa-2 params)."""
    text = Path(path).read_text()
    head = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            head.append(line)
        else:
            break
    src = yaml.safe_load(text)
    src.update(top or {})
    by = {e["name"]: e for e in src["encoders"]}
    unknown = sorted(set(encoders or {}) - set(by))
    if unknown:
        raise ValueError(f"unknown encoders: {', '.join(unknown)}")
    for name, kv in (encoders or {}).items():
        by[name].update(kv)
    body = yaml.safe_dump(src, sort_keys=False, default_flow_style=False)
    Path(path).write_text(("\n".join(head) + "\n" if head else "") + body)
