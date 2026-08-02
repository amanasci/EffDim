"""Shared utilities for Physics Probe Subspace experiments."""

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def platonic_root(cli_value: str | None = None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Default to the EffDim repository root (two directories up from this file)
    return Path(__file__).resolve().parents[2]


def ensure_sae_import() -> Path:
    """Put the local sae/ vendored copy on sys.path; return the chosen dir."""
    candidates = [
        Path(__file__).resolve().parent / "sae",
        Path(__file__).resolve().parents[1] / "SAE-shared-basis" / "sae",
    ]
    for p in candidates:
        if (p / "sae_model.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
    raise FileNotFoundError(
        "sae_model.py not found. Expected vendored copy at "
        f"{candidates[0]} (shipped with this package) or SAE-shared-basis/sae/"
    )


def load_embeddings(path: Path, col: str = "embeddings", hf_repo: str = "UniverseTBD/pu-embeddings") -> np.ndarray:
    """Load embedding column as float32 matrix. Fetches from HF if local path is missing."""
    if not path.exists():
        from huggingface_hub import hf_hub_download
        import shutil
        print(f"File {path} not found locally. Downloading from HF {hf_repo}...")
        # Reconstruct repo path: e.g. physics/vit_base_test.parquet
        rel_path = f"{path.parent.name}/{path.name}"
        local_path = hf_hub_download(repo_id=hf_repo, filename=rel_path, repo_type="dataset")
        
        print(f"Moving downloaded file to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, path)
        
    table = pq.read_table(path, columns=[col])
    X = np.vstack(table.column(0).to_pylist()).astype(np.float32)
    return X


def load_physics_labels(
    n: int, 
    split: str = "test", 
    hf_token: str | None = None
) -> dict[str, np.ndarray]:
    """Load Smith42/galaxies v2.0 labels and compute derived properties.
    
    Uses local HF dataset caching for fast, robust loads; falls back to streaming if needed.
    """
    ds = None
    try:
        # Try standard download + local HF cache (fast & robust against streaming timeouts)
        full_ds = load_dataset(
            "Smith42/galaxies",
            revision="v2.0",
            split=split,
            streaming=False,
            token=hf_token
        )
        ds = full_ds.select(range(min(n, len(full_ds))))
    except Exception as e:
        print(f"Non-streaming dataset load failed/timed out ({e}). Falling back to streaming mode...")
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

    mapped_data = {}
    for short_name, long_name in ALL_PROBES.items():
        if long_name in data:
            mapped_data[short_name] = data[long_name]

    return mapped_data


def train_probes(Z: np.ndarray, y_dict: dict[str, np.ndarray], probe_keys: list[str]) -> tuple[np.ndarray, dict]:
    """Train linear probes and return weight matrix W (D x M) and diagnostic stats.

    Each column of W is the unit-free coefficient vector for one probe property.
    """
    D = Z.shape[1]
    M = len(probe_keys)
    W = np.zeros((D, M), dtype=np.float32)
    stats = {}
    
    for m, key in enumerate(probe_keys):
        y = y_dict[key]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            print(f"Warning: Probe '{key}' has less than 10 valid samples, skipping.")
            stats[key] = {"r2_train": float('nan'), "r2_cv": float('nan'), "n_valid": int(valid.sum())}
            continue
            
        Z_valid = Z[valid]
        y_valid = y[valid]
        
        # Standardize target
        y_mean = y_valid.mean()
        y_std = y_valid.std() + 1e-12
        y_valid_std = (y_valid - y_mean) / y_std
        
        # Fit on full train valid set
        model = LinearRegression(fit_intercept=True)
        model.fit(Z_valid, y_valid_std)
        w = model.coef_
        W[:, m] = w
        
        r2_train = r2_score(y_valid_std, model.predict(Z_valid))
        
        # 5-fold CV
        cv_scores = []
        kf = KFold(n_splits=min(5, len(y_valid)), shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(Z_valid):
            m_cv = LinearRegression(fit_intercept=True)
            m_cv.fit(Z_valid[train_idx], y_valid_std[train_idx])
            pred = m_cv.predict(Z_valid[test_idx])
            cv_scores.append(r2_score(y_valid_std[test_idx], pred))
            
        stats[key] = {
            "r2_train": float(r2_train),
            "r2_cv": float(np.mean(cv_scores)),
            "n_valid": int(valid.sum())
        }
    
    return W, stats


def compute_probe_residuals(
    Z_test: np.ndarray,
    y_test: dict[str, np.ndarray],
    W: np.ndarray,
    probe_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point, per-probe standardized squared residuals.

    Returns
    -------
    residuals : (n_test, M) array of squared residuals (standardized targets)
    mean_residual : (n_test,) mean across valid probes per point
    """
    n = Z_test.shape[0]
    M = len(probe_keys)
    residuals = np.full((n, M), np.nan, dtype=np.float64)

    for m, key in enumerate(probe_keys):
        if key not in y_test:
            continue
        y = y_test[key]
        valid = ~np.isnan(y)
        if valid.sum() < 5:
            continue
        y_std = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        # predict = Z_test @ w_m  (linear probe, intercept not saved; fit was standardized so intercept ≈ 0)
        w_m = W[:, m]
        y_hat = Z_test @ w_m
        sq = (y_std - y_hat) ** 2
        sq[~valid] = np.nan
        residuals[:, m] = sq

    mean_residual = np.nanmean(residuals, axis=1)
    return residuals.astype(np.float32), mean_residual.astype(np.float32)


def correlation_analysis(
    curvature_dict: dict[str, np.ndarray],
    mean_residual: np.ndarray,
    residuals: np.ndarray,
    probe_keys: list[str],
    output_dir: Path,
    tag: str = "",
) -> dict:
    """Spearman ρ, binned box-plots, logistic AUC, per-probe breakdown.

    Parameters
    ----------
    curvature_dict : mapping metric_name -> (n_test,) curvature values
    mean_residual  : (n_test,) mean probe squared residual per point
    residuals      : (n_test, M) per-probe squared residuals
    probe_keys     : list of probe names corresponding to residuals columns
    output_dir     : directory to write plots and JSON
    tag            : prefix for output filenames (e.g. 'model_a')
    """
    import json
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = np.isfinite(mean_residual)
    hard_label = (mean_residual > np.nanmedian(mean_residual)).astype(int)

    summary = {}

    # ---- Spearman + logistic AUC per metric ----
    spearman_rows = []
    for metric_name, curv in curvature_dict.items():
        curv = np.asarray(curv, dtype=np.float64)
        both_valid = valid_mask & np.isfinite(curv)
        if both_valid.sum() < 20:
            continue
        rho, pval = spearmanr(curv[both_valid], mean_residual[both_valid])
        # Logistic AUC
        x = curv[both_valid].reshape(-1, 1)
        y_lab = hard_label[both_valid]
        if len(np.unique(y_lab)) < 2:
            auc = float("nan")
        else:
            sc = StandardScaler()
            x_sc = sc.fit_transform(x)
            lr = LogisticRegression(max_iter=300, random_state=0)
            lr.fit(x_sc, y_lab)
            auc = roc_auc_score(y_lab, lr.predict_proba(x_sc)[:, 1])
        row = {
            "metric": metric_name,
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "logistic_auc": float(auc),
            "n_valid": int(both_valid.sum()),
        }
        spearman_rows.append(row)
        print(f"  [{tag}] {metric_name:<35} ρ={rho:+.3f} (p={pval:.2e})  AUC={auc:.3f}", flush=True)
    summary["spearman"] = spearman_rows

    # ---- Binned box-plots ----
    fig, axes = plt.subplots(1, len(curvature_dict), figsize=(5 * len(curvature_dict), 5), squeeze=False)
    for ax, (metric_name, curv) in zip(axes[0], curvature_dict.items()):
        curv = np.asarray(curv, dtype=np.float64)
        both_valid = valid_mask & np.isfinite(curv)
        c_v = curv[both_valid]
        r_v = mean_residual[both_valid]
        q33, q67 = np.quantile(c_v, [1/3, 2/3])
        groups = [
            r_v[c_v <= q33],
            r_v[(c_v > q33) & (c_v <= q67)],
            r_v[c_v > q67],
        ]
        ax.boxplot(groups, labels=["Low", "Mid", "High"])
        ax.set_title(f"{metric_name}\n({tag})", fontsize=9)
        ax.set_xlabel("Curvature tercile")
        ax.set_ylabel("Mean probe ε²")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Curvature vs Probe Error ({tag})", fontsize=11)
    fig.tight_layout()
    boxplot_path = output_dir / f"{tag}_boxplot.png"
    fig.savefig(boxplot_path, dpi=120)
    plt.close(fig)
    summary["boxplot_path"] = str(boxplot_path)

    # ---- Per-probe breakdown (top-5 |ρ| for each curvature metric) ----
    per_probe = {}
    for metric_name, curv in curvature_dict.items():
        curv = np.asarray(curv, dtype=np.float64)
        both_valid = valid_mask & np.isfinite(curv)
        probe_rows = []
        for m, key in enumerate(probe_keys):
            col = residuals[:, m].astype(np.float64)
            mask = both_valid & np.isfinite(col)
            if mask.sum() < 20:
                continue
            rho, pval = spearmanr(curv[mask], col[mask])
            probe_rows.append({"probe": key, "rho": float(rho), "pval": float(pval)})
        probe_rows.sort(key=lambda r: abs(r["rho"]), reverse=True)
        per_probe[metric_name] = probe_rows[:5]
    summary["per_probe_top5"] = per_probe

    return summary


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
