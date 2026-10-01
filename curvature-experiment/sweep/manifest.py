"""The 31-encoder manifest for the encoder-scaling sweep (encoders.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "encoders.yaml"


@dataclass(frozen=True)
class Encoder:
    name: str
    family: str
    dim: int
    params: int
    params_source: str
    in_paper: bool

    @property
    def parquet_file(self) -> str:
        return f"physics/{self.name}_test.parquet"

    @property
    def column(self) -> str:
        return f"{self.name}_galaxies"


@dataclass(frozen=True)
class Manifest:
    repo: str
    snapshot: str
    n_rows: int
    label_table: str
    encoders: Tuple[Encoder, ...]
    label_table_sha256: Optional[str] = None


def load_manifest(path: Union[str, Path] = DEFAULT_PATH) -> Manifest:
    src = yaml.safe_load(Path(path).read_text())
    encs = tuple(Encoder(**e) for e in src["encoders"])
    names = [e.name for e in encs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate encoder names in {path}: {dup}")
    return Manifest(repo=src["repo"], snapshot=src["snapshot"], n_rows=int(src["n_rows"]),
                    label_table=src["label_table"], encoders=encs,
                    label_table_sha256=src.get("label_table_sha256"))
