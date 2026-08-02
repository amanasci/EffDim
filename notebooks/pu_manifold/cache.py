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
    """First KEY_LEN hex chars of sha256(json.dumps(cfg, sort_keys=True)). sort_keys=True
    so the key does not depend on dict insertion order."""
    serialized = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:KEY_LEN]


def _assert_inside_cache(path: Path) -> None:
    """Raise ValueError unless path resolves inside CACHE_DIR (T-01-01 mitigation: every
    load path this module composes must not let a '..' segment in a stem escape
    CACHE_DIR)."""
    resolved = path.resolve()
    resolved_cache_dir = CACHE_DIR.resolve()
    if resolved_cache_dir not in resolved.parents and resolved != resolved_cache_dir:
        raise ValueError(
            f"Refusing to use path outside CACHE_DIR: {resolved} is not inside "
            f"{resolved_cache_dir}."
        )


def cache_path(stem: str, ext: str) -> Path:
    """Containment-checked CACHE_DIR / f"{stem}.{ext}"."""
    path = CACHE_DIR / f"{stem}.{ext}"
    _assert_inside_cache(path)
    return path


def _manifest_path(stem: str) -> Path:
    """Sidecar manifest path for a stem: CACHE_DIR / f"{stem}.meta.json"."""
    return cache_path(stem, "meta.json")


def _manifest_matches(stem: str, cfg: Dict[str, Any]) -> bool:
    """True if a manifest exists and its stored cfg equals cfg exactly. Raises ValueError
    on a mismatch rather than returning False -- a filename-key match alone is not
    sufficient trust (PITFALLS Pitfall 10)."""
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
    """Write the sidecar manifest recording the full cfg dict for a stem."""
    manifest_path = _manifest_path(stem)
    manifest_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def npz_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Load-or-compute an npz-backed artifact, keyed by a sidecar manifest (cfg is
    recorded and re-verified on load; compute_fn runs only on a cache miss)."""
    path = cache_path(stem, "npz")
    if path.exists() and _manifest_matches(stem, cfg):
        return dict(np.load(path))
    arrays = compute_fn()
    np.savez(path, **arrays)
    _write_manifest(stem, cfg)
    return arrays


def joblib_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Any]) -> Any:
    """Load-or-compute a joblib-pickled artifact, keyed by a sidecar manifest. Only ever
    loads a path this module composed itself from CACHE_DIR (validated by
    _assert_inside_cache) -- do not add a helper that loads a caller-supplied absolute
    path, since joblib.load is pickle deserialization (threat T-01-01)."""
    path = cache_path(stem, "joblib")
    if path.exists() and _manifest_matches(stem, cfg):
        return joblib_load(path)
    obj = compute_fn()
    joblib_dump(obj, path)
    _write_manifest(stem, cfg)
    return obj


def json_cache(stem: str, cfg: Dict[str, Any], compute_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Load-or-compute a json-backed artifact, keyed by a sidecar manifest."""
    path = cache_path(stem, "json")
    if path.exists() and _manifest_matches(stem, cfg):
        return json.loads(path.read_text())
    result = compute_fn()
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_manifest(stem, cfg)
    return result
