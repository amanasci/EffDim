"""CACHE_DIR honours EFFDIM_CACHE_DIR and otherwise sits next to pu_manifold."""
import importlib
import sys
from pathlib import Path

PKG_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_PARENT))


def _reload_cache():
    import pu_manifold.cache as c
    return importlib.reload(c)


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFDIM_CACHE_DIR", str(tmp_path / "records"))
    assert _reload_cache().CACHE_DIR == (tmp_path / "records").resolve()
    monkeypatch.delenv("EFFDIM_CACHE_DIR")
    _reload_cache()


def test_cache_dir_default(monkeypatch):
    monkeypatch.delenv("EFFDIM_CACHE_DIR", raising=False)
    assert _reload_cache().CACHE_DIR == (PKG_PARENT / ".cache").resolve()
