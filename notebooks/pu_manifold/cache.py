"""
Config-hash-keyed cache helpers for the notebook-scoped ``pu_manifold`` package.

Every artifact this package persists (subsamples, sweep results, Isomap fits) is
written under :data:`CACHE_DIR` at a filename the caller composes from a stem plus a
:func:`config_key`, and every load re-verifies a sidecar manifest against the cfg dict
the caller is currently asking for -- a filename match alone is never trusted (see
``PITFALLS.md`` Pitfall 10 and threat T-01-03 in the phase 1 plan). ``joblib_cache``
only ever loads a path this module composed itself from :data:`CACHE_DIR`; no helper
here accepts a caller-supplied absolute path, because ``joblib.load`` is pickle
deserialization (threat T-01-01).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from joblib import dump as joblib_dump
from joblib import load as joblib_load

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 16 hex characters (64 bits) rather than the 8 (32 bits) used in ARCHITECTURE.md's
# illustrative example. This is a deliberate, stated deviation: it is the collision
# margin for threat T-01-03.
KEY_LEN = 16


def config_key(cfg: Dict[str, Any]) -> str:
    """
    Compute a stable, order-independent hash key for a config dict.

    Parameters:
    -----------
    cfg : Dict[str, Any]
        The analysis-parameter dict to key on. Must be JSON-serializable.

    Returns:
    --------
    str
        The first ``KEY_LEN`` hex characters of the sha256 digest of
        ``json.dumps(cfg, sort_keys=True)``. ``sort_keys=True`` is required so key
        stability does not depend on dict insertion order.
    """
    serialized = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:KEY_LEN]


def _assert_inside_cache(path: Path) -> None:
    """
    Raise ``ValueError`` unless ``path`` resolves to a location inside ``CACHE_DIR``.

    Parameters:
    -----------
    path : Path
        The path to check.

    Returns:
    --------
    None

    This is the T-01-01 mitigation: every load path is one this module composed
    itself from ``CACHE_DIR`` plus a ``config_key`` it just computed, and this guard
    makes a ``..`` segment in a stem unable to escape ``CACHE_DIR``.
    """
    resolved = path.resolve()
    resolved_cache_dir = CACHE_DIR.resolve()
    if resolved_cache_dir not in resolved.parents and resolved != resolved_cache_dir:
        raise ValueError(
            f"Refusing to use path outside CACHE_DIR: {resolved} is not inside "
            f"{resolved_cache_dir}."
        )


def cache_path(stem: str, ext: str) -> Path:
    """
    Compose a containment-checked path under ``CACHE_DIR`` for a given stem and extension.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem, e.g. ``f"subsample_{seed}_{subsample_key}"``.
    ext : str
        The file extension without a leading dot, e.g. ``"npz"``.

    Returns:
    --------
    Path
        ``CACHE_DIR / f"{stem}.{ext}"``, guaranteed to resolve inside ``CACHE_DIR``.
    """
    path = CACHE_DIR / f"{stem}.{ext}"
    _assert_inside_cache(path)
    return path


def _manifest_path(stem: str) -> Path:
    """
    Compose the sidecar manifest path for a given stem.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem the artifact was written under.

    Returns:
    --------
    Path
        ``CACHE_DIR / f"{stem}.meta.json"``, guaranteed to resolve inside ``CACHE_DIR``.
    """
    return cache_path(stem, "meta.json")


def _manifest_matches(stem: str, cfg: Dict[str, Any]) -> bool:
    """
    Check whether an existing sidecar manifest's cfg equals the passed cfg.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem.
    cfg : Dict[str, Any]
        The cfg dict the caller currently wants to load or write.

    Returns:
    --------
    bool
        True if a manifest exists and its stored cfg equals ``cfg`` exactly.

    Raises:
    -------
    ValueError
        If a manifest exists but its stored cfg differs from ``cfg``, naming both
        dicts. A filename-key match alone is not sufficient trust (PITFALLS Pitfall 10).
    """
    manifest_path = _manifest_path(stem)
    if not manifest_path.exists():
        return False
    stored_cfg = json.loads(manifest_path.read_text())
    if stored_cfg != cfg:
        raise ValueError(
            f"Cache manifest mismatch for stem '{stem}': stored cfg {stored_cfg!r} "
            f"does not equal requested cfg {cfg!r}. Refusing to silently reuse a "
            f"stale or incompatible artifact."
        )
    return True


def _write_manifest(stem: str, cfg: Dict[str, Any]) -> None:
    """
    Write the sidecar manifest recording the full cfg dict for a stem.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem.
    cfg : Dict[str, Any]
        The cfg dict to record.

    Returns:
    --------
    None
    """
    manifest_path = _manifest_path(stem)
    manifest_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def npz_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Load-or-compute an npz-backed artifact, keyed by a sidecar manifest.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem (caller composes it, e.g.
        ``f"subsample_{seed}_{subsample_key}"``).
    cfg : Dict[str, Any]
        The cfg dict this artifact is keyed on. Recorded in the sidecar manifest.
    compute_fn : Callable[[], Dict[str, np.ndarray]]
        Zero-argument callable returning the dict of arrays to cache on a miss.

    Returns:
    --------
    Dict[str, np.ndarray]
        The cached or freshly computed dict of arrays.
    """
    path = cache_path(stem, "npz")
    if path.exists() and _manifest_matches(stem, cfg):
        return dict(np.load(path))
    arrays = compute_fn()
    np.savez(path, **arrays)
    _write_manifest(stem, cfg)
    return arrays


def joblib_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Any]) -> Any:
    """
    Load-or-compute a joblib-pickled artifact, keyed by a sidecar manifest.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem (caller composes it, e.g. ``f"isomap_{fit_key}"``).
    cfg : Dict[str, Any]
        The cfg dict this artifact is keyed on. Recorded in the sidecar manifest.
    compute_fn : Callable[[], Any]
        Zero-argument callable returning the object to cache on a miss.

    Returns:
    --------
    Any
        The cached or freshly computed object.

    This function only ever loads a path it composed itself from ``CACHE_DIR`` plus a
    caller-supplied stem (validated by ``_assert_inside_cache``). Do not add a helper
    that loads a caller-supplied absolute path -- ``joblib.load`` is pickle
    deserialization (threat T-01-01), and never loading a path this module did not
    build is the only defence.
    """
    path = cache_path(stem, "joblib")
    if path.exists() and _manifest_matches(stem, cfg):
        return joblib_load(path)
    obj = compute_fn()
    joblib_dump(obj, path)
    _write_manifest(stem, cfg)
    return obj


def json_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load-or-compute a json-backed artifact, keyed by a sidecar manifest.

    Parameters:
    -----------
    stem : str
        The fully-formed filename stem (caller composes it, e.g.
        ``f"effdim_panel_{seed}_{subsample_key}"``).
    cfg : Dict[str, Any]
        The cfg dict this artifact is keyed on. Recorded in the sidecar manifest.
    compute_fn : Callable[[], Dict[str, Any]]
        Zero-argument callable returning the dict to cache on a miss.

    Returns:
    --------
    Dict[str, Any]
        The cached or freshly computed dict.
    """
    path = cache_path(stem, "json")
    if path.exists() and _manifest_matches(stem, cfg):
        return json.loads(path.read_text())
    result = compute_fn()
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_manifest(stem, cfg)
    return result
