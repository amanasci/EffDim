"""Dense Physics activation loading — thin wrapper over topology prepare."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from topology.physics_activation_density_ph.data import (  # noqa: E402
    PreparedActivations,
    effective_rank_from_cov,
    l2_normalize,
    prepare_activations,
    summarize_population,
)

from .paths import resolve_path


def prepare_atlas_data(root: Path, **kwargs) -> PreparedActivations:
    return prepare_activations(root, **kwargs)


def save_prepare(out: Path, prep: PreparedActivations, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "activations_l2.npz",
        X=prep.X_l2,
        sample_ids=prep.sample_ids,
        train_local=prep.train_local,
        test_local=prep.test_local,
        holdout_local=prep.holdout_local,
    )
    np.savez_compressed(
        out / "ids_and_splits.npz",
        sample_ids=prep.sample_ids,
        train_local=prep.train_local,
        test_local=prep.test_local,
        holdout_local=prep.holdout_local,
    )
    (out / "population_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "input_schema.json").write_text(
        json.dumps(
            {
                "parquet": prep.parquet,
                "column": prep.column,
                "selection_path": prep.selection_path,
                "preprocess": prep.preprocess,
                "metric": prep.metric,
                "ambient_dim": prep.ambient_dim,
                "n_selected": int(len(prep.X_l2)),
                "schema_version": 1,
            },
            indent=2,
        )
    )


def load_prepare(out: Path) -> dict:
    z = np.load(out / "activations_l2.npz")
    return {
        "X": z["X"].astype(np.float32),
        "sample_ids": z["sample_ids"].astype(np.int64),
        "train_local": z["train_local"].astype(np.int64),
        "test_local": z["test_local"].astype(np.int64),
        "holdout_local": z["holdout_local"].astype(np.int64),
        "summary": json.loads((out / "population_summary.json").read_text()),
    }


__all__ = [
    "PreparedActivations",
    "effective_rank_from_cov",
    "l2_normalize",
    "prepare_atlas_data",
    "summarize_population",
    "save_prepare",
    "load_prepare",
    "resolve_path",
]
