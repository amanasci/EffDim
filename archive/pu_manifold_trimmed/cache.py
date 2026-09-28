# Final whole-branch review (after fa139de): removed from curvature-experiment/pu_manifold/cache.py -- reachable only
# through a string constant, an __init__ re-export or its own tests. Line numbers at fa139de. Verbatim, original order.


# --- removed from cache.py:100-122 ---
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
