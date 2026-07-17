"""
Packaging metadata gate (RUST-05 / D-05 / D-06).

Asserts installed Requires-Dist is NumPy-only — no FAISS, SciPy,
scikit-learn, or ipykernel reintroduced into runtime metadata.
"""
from __future__ import annotations

import importlib.metadata
import re

import pytest

FORBIDDEN_RUNTIME_TOKENS = ("faiss", "scipy", "scikit-learn", "sklearn", "ipykernel")


def _requirement_name(req: str) -> str:
    """Extract the distribution name from a Requirement string."""
    # Strip environment markers / extras: "numpy>=1.0; python_version>='3.10'"
    base = req.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)", base)
    assert match is not None, f"Could not parse requirement name from {req!r}"
    return match.group(1).lower().replace("_", "-")


class TestPackagingMetadata:
    """Requires-Dist must be NumPy-only after the thin-shell cutover."""

    def test_requires_dist_includes_numpy(self):
        reqs = importlib.metadata.requires("effdim")
        assert reqs is not None, "effdim package metadata has no Requires-Dist"
        names = [_requirement_name(r) for r in reqs]
        assert any(n == "numpy" for n in names), (
            f"Expected a numpy requirement in Requires-Dist, got: {reqs}"
        )

    def test_requires_dist_excludes_removed_compute_deps(self):
        reqs = importlib.metadata.requires("effdim")
        assert reqs is not None, "effdim package metadata has no Requires-Dist"
        for req in reqs:
            name = _requirement_name(req)
            for token in FORBIDDEN_RUNTIME_TOKENS:
                assert token not in name, (
                    f"Forbidden runtime dependency {token!r} found in Requires-Dist: {req!r}"
                )
            # Also catch token smuggling in the raw requirement string (extras/markers)
            lower = req.lower()
            for token in FORBIDDEN_RUNTIME_TOKENS:
                # Match as a dependency name token, not a random substring in markers
                if re.search(rf"(^|[\s,;\[]){re.escape(token)}([\s,;=<>!~\]]|$)", lower):
                    pytest.fail(
                        f"Forbidden runtime dependency token {token!r} in Requires-Dist: {req!r}"
                    )
