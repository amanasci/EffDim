"""Shared utilities for Physics Probe Subspace experiments."""

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset


def platonic_root(cli_value: str | None = None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path.home() / "GitHub" / "platonic-universe",
        Path.home() / "platonic-universe",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return candidates[0]


def load_embeddings(path: Path, col: str = "embeddings", hf_repo: str = "UniverseTBD/pu-embeddings") -> np.ndarray:
    """Load embedding column as float32 matrix. Fetches from HF if local path is missing."""
    if not path.exists():
        from huggingface_hub import hf_hub_download
        print(f"File {path} not found locally. Downloading from HF {hf_repo}...")
        # Reconstruct repo path: e.g. physics/vit_base_test.parquet
        rel_path = f"{path.parent.name}/{path.name}"
        local_path = hf_hub_download(repo_id=hf_repo, filename=rel_path, repo_type="dataset")
        path = Path(local_path)
        
    table = pq.read_table(path, columns=[col])
    X = np.vstack(table.column(0).to_pylist()).astype(np.float32)
    return X


def load_physics_labels(
    n: int, 
    split: str = "test", 
    hf_token: str | None = None
) -> dict[str, np.ndarray]:
    """Stream Smith42/galaxies v2.0 labels and compute derived properties."""
    ds = load_dataset(
        "Smith42/galaxies",
        revision="v2.0",
        split=split,
        streaming=True,
        token=hf_token
    )
    
    # Pre-allocate dictionary of lists
    raw_data = {k: [] for k in ALL_PROBES.values() if k in METADATA_COLUMNS}
    
    count = 0
    for row in ds:
        if count >= n:
            break
        for k in raw_data.keys():
            raw_data[k].append(row.get(k, float('nan')))
        count += 1
        
    if count < n:
        print(f"Warning: Only {count} samples found in split '{split}', requested {n}")

    # Convert to numpy arrays
    data = {k: np.array(v, dtype=np.float32) for k, v in raw_data.items()}
    
    # Compute derived properties
    def add_derived(key: str, val: np.ndarray):
        if key in ALL_PROBES.values():
            data[key] = val

    if "mag_g" in data and "mag_r" in data:
        add_derived("mag_g - mag_r", data["mag_g"] - data["mag_r"])
    if "mag_g" in data and "mag_z" in data:
        add_derived("mag_g - mag_z", data["mag_g"] - data["mag_z"])
    if "mag_r" in data and "mag_z" in data:
        add_derived("mag_r - mag_z", data["mag_r"] - data["mag_z"])
        
    if "petro_th50" in data and "petro_th90" in data:
        add_derived("petro_th90 / petro_th50", data["petro_th90"] / np.maximum(data["petro_th50"], 1e-6))
        
    if "petro_th50" in data:
        add_derived("log10(petro_th50)", np.log10(np.maximum(data["petro_th50"], 1e-6)))
    if "petro_th90" in data:
        add_derived("log10(petro_th90)", np.log10(np.maximum(data["petro_th90"], 1e-6)))
    if "sersic_n" in data:
        add_derived("log10(sersic_n + 1)", np.log10(np.maximum(data["sersic_n"] + 1.0, 1e-6)))

    if "smooth-or-featured_smooth_fraction" in data and "smooth-or-featured_featured-or-disk_fraction" in data:
        add_derived(
            "smooth - featured_disk", 
            data["smooth-or-featured_smooth_fraction"] - data["smooth-or-featured_featured-or-disk_fraction"]
        )
        
    if "bar_strong_fraction" in data and "bar_weak_fraction" in data:
        add_derived("bar_strong + bar_weak", data["bar_strong_fraction"] + data["bar_weak_fraction"])
        
    if "merging_merger_fraction" in data and "merging_major-disturbance_fraction" in data and "merging_minor-disturbance_fraction" in data:
        add_derived(
            "merging + major_disturbance + minor_disturbance",
            data["merging_merger_fraction"] + data["merging_major-disturbance_fraction"] + data["merging_minor-disturbance_fraction"]
        )
        
    if "bulge-size_dominant_fraction" in data and "bulge-size_large_fraction" in data:
        add_derived(
            "bulge_dominant + bulge_large",
            data["bulge-size_dominant_fraction"] + data["bulge-size_large_fraction"]
        )

    # Sanitize sentinel values (-99.0 is common for missing physical quantities)
    for k in data.keys():
        invalid = (data[k] == -99.0) | np.isinf(data[k])
        data[k][invalid] = np.nan

    return data


METADATA_COLUMNS = [
    # Morphology
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "has-spiral-arms_yes_fraction",
    "has-spiral-arms_no_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    "bulge-size_dominant_fraction",
    "bulge-size_large_fraction",
    "bulge-size_moderate_fraction",
    "bulge-size_small_fraction",
    "bulge-size_none_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "merging_merger_fraction",
    "merging_major-disturbance_fraction",
    "merging_minor-disturbance_fraction",
    "merging_none_fraction",
    # Photometry
    "mag_r",
    "mag_g",
    "mag_z",
    "u_minus_r",
    # Structure
    "sersic_n",
    "sersic_ba",
    "petro_th50",
    "petro_th90",
    "elpetro_ba",
    "elpetro_theta_r",
    # Physical
    "elpetro_mass_log",
    "redshift",
    # Star formation
    "total_sfr_median",
    "total_ssfr_median",
]

# Map from short key to column/formula
ALL_PROBES = {
    # Morphology GZ
    "smooth": "smooth-or-featured_smooth_fraction",
    "featured_disk": "smooth-or-featured_featured-or-disk_fraction",
    "artifact": "smooth-or-featured_artifact_fraction",
    "has_spiral_arms": "has-spiral-arms_yes_fraction",
    "no_spiral_arms": "has-spiral-arms_no_fraction",
    "bar_strong": "bar_strong_fraction",
    "bar_weak": "bar_weak_fraction",
    "bar_none": "bar_no_fraction",
    "bulge_dominant": "bulge-size_dominant_fraction",
    "bulge_large": "bulge-size_large_fraction",
    "bulge_moderate": "bulge-size_moderate_fraction",
    "bulge_small": "bulge-size_small_fraction",
    "bulge_none": "bulge-size_none_fraction",
    "edge_on": "disk-edge-on_yes_fraction",
    "not_edge_on": "disk-edge-on_no_fraction",
    "merging": "merging_merger_fraction",
    "major_disturbance": "merging_major-disturbance_fraction",
    "minor_disturbance": "merging_minor-disturbance_fraction",
    "no_merger": "merging_none_fraction",
    # Photometry
    "mag_r": "mag_r",
    "mag_g": "mag_g",
    "mag_z": "mag_z",
    "u_minus_r": "u_minus_r",
    # Structure
    "sersic_n": "sersic_n",
    "sersic_ba": "sersic_ba",
    "petro_th50": "petro_th50",
    "petro_th90": "petro_th90",
    "elpetro_ba": "elpetro_ba",
    "elpetro_theta": "elpetro_theta_r",
    # Physical
    "stellar_mass": "elpetro_mass_log",
    "redshift": "redshift",
    # Star formation
    "sfr": "total_sfr_median",
    "ssfr": "total_ssfr_median",
    # Derived
    "g_minus_r": "mag_g - mag_r",
    "g_minus_z": "mag_g - mag_z",
    "r_minus_z": "mag_r - mag_z",
    "concentration": "petro_th90 / petro_th50",
    "log_petro_th50": "log10(petro_th50)",
    "log_petro_th90": "log10(petro_th90)",
    "log_sersic_n": "log10(sersic_n + 1)",
    "smooth_minus_disk": "smooth - featured_disk",
    "bar_signal": "bar_strong + bar_weak",
    "total_merger": "merging + major_disturbance + minor_disturbance",
    "bulge_total": "bulge_dominant + bulge_large",
}

# The deduplicated set excluding one member of each GZ sum-to-1 group
INDEPENDENT_PROBES = [
    # Morphology (13 independent)
    "smooth", "featured_disk", 
    "has_spiral_arms",
    "bar_strong", "bar_weak",
    "bulge_dominant", "bulge_large", "bulge_moderate", "bulge_small",
    "edge_on",
    "merging", "major_disturbance", "minor_disturbance",
    # Photometry (4)
    "mag_r", "mag_g", "mag_z", "u_minus_r",
    # Structure (6)
    "sersic_n", "sersic_ba", "petro_th50", "petro_th90", "elpetro_ba", "elpetro_theta",
    # Physical (2)
    "stellar_mass", "redshift",
    # Star formation (2)
    "sfr", "ssfr",
    # Derived (~11)
    "g_minus_r", "g_minus_z", "r_minus_z",
    "concentration", "log_petro_th50", "log_petro_th90", "log_sersic_n",
    "smooth_minus_disk", "bar_signal", "total_merger", "bulge_total"
]

DEFAULT_11_PROBES = [
    "redshift", "stellar_mass", "g_minus_r", "u_minus_r",
    "sersic_n", "sersic_ba", "petro_th50", 
    "smooth", "featured_disk", "merging", "has_spiral_arms"
]
