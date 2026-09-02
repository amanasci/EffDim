"""D9-18's cross-cutting import-purity regression test for Phase 9's two new modules.

Copies the mechanism `test_cka_import_purity.py` already invented for the same
D8-23/D9-18-class requirement ("importing the new module(s) must never mutate any sealed
`pu_manifold` module, regardless of import order") rather than reinventing it. This is a
Phase-9-scoped SIBLING test file -- `test_cka_import_purity.py` itself is never edited (D9-18,
additive only); Phase 8's own `SEALED_MODULES` tuple and `cka` gain no new phase-9 entries.

This suite loads no PU or Physics data, trains nothing, and opens no on-disk cached array -- it
exercises only the import machinery and each module's own ``vars()``.

Load-bearing tests: ``test_import_purity_holds_under_every_order`` (parametrized over at least
four distinct import orders, each run in its own subprocess so `sys.modules` caching in a single
process cannot mask a mutation) and ``test_snapshot_detects_a_planted_mutation`` (proves the
comparison mechanism itself can fail -- without this, a passing import-purity suite would be
uninformative).
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pytest

NOTEBOOK_ROOT = Path(__file__).resolve().parents[2]

SEALED_MODULES: Tuple[str, ...] = (
    "mknn",
    "cae",
    "decoder_curvature",
    "curvature_probe",
    "cross_split_curvature",
    "linear_probe",
    "pointcloud_probe",
    "crossmodal_curvature",
    "density_stratified_null",
    "cka",
)
"""The nine sealed modules `test_cka_import_purity.py` already names, plus `cka` itself (Phase
8's own new module, sealed as of Phase 8's completion) -- the full set of modules that must
never be mutated by importing either of Phase 9's two new modules, in any order."""

NEW_MODULES: Tuple[str, ...] = ("physics_labels", "physics_curvature_probe")
"""Phase 9's own two new modules -- importing either, in any position/order, must leave every
one of SEALED_MODULES' `vars()` snapshots unchanged."""


def _snapshot_module_state(module) -> Dict[str, str]:
    """A stable textual fingerprint of every public (non-dunder) name in `vars(module)`. Mirrors
    `test_cka_import_purity.py`'s own snapshot function exactly (self-contained: embedded via
    `inspect.getsource` into each subprocess script rather than imported, so this logic lives in
    one place per test file)."""
    import hashlib

    import numpy as np

    snapshot: Dict[str, str] = {}
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            snapshot[name] = (
                f"ndarray:shape={value.shape}:dtype={value.dtype}:"
                f"sha256={hashlib.sha256(value.tobytes()).hexdigest()}"
            )
        elif isinstance(value, np.generic):
            snapshot[name] = f"npscalar:dtype={value.dtype}:{value!r}"
        elif isinstance(value, (set, frozenset)):
            snapshot[name] = f"{type(value).__name__}:{sorted(repr(x) for x in value)}"
        elif callable(value) or isinstance(value, type) or type(value).__name__ == "module":
            qualname = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
            snapshot[name] = f"{type(value).__name__}:{getattr(value, '__module__', '')}.{qualname}"
        else:
            snapshot[name] = repr(value)
    return snapshot


def _import_in_order(order: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Runs `order` (a sequence of names, each either a `SEALED_MODULES` entry or a
    `NEW_MODULES` entry) as a sequence of imports inside a fresh subprocess, then returns
    `_snapshot_module_state` for every one of `SEALED_MODULES` as they stand once every import
    in `order` has completed. `order` must name every one of `SEALED_MODULES` at least once."""
    import inspect

    snapshot_src = inspect.getsource(_snapshot_module_state)
    order_literal = json.dumps(list(order))
    sealed_literal = json.dumps(list(SEALED_MODULES))
    script = f"""
import importlib
import json
import sys

sys.path.insert(0, {str(NOTEBOOK_ROOT)!r})

{snapshot_src}

_order = json.loads({order_literal!r})
_sealed = json.loads({sealed_literal!r})

_imported = {{}}
for _name in _order:
    _mod = importlib.import_module(f"pu_manifold.{{_name}}")
    _imported[_name] = _mod

_snapshots = {{name: _snapshot_module_state(_imported[name]) for name in _sealed}}
print(json.dumps(_snapshots))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"_import_in_order: subprocess for order {list(order)!r} exited "
            f"{result.returncode}. stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


# --- import purity across orders -------------------------------------------------------------

_ORDER_NEW_FIRST = tuple(NEW_MODULES) + SEALED_MODULES
_ORDER_NEW_LAST = SEALED_MODULES + tuple(NEW_MODULES)
_ORDER_NEW_INTERLEAVED = (
    SEALED_MODULES[:3] + (NEW_MODULES[0],) + SEALED_MODULES[3:7] + (NEW_MODULES[1],) + SEALED_MODULES[7:]
)
_ORDER_NEW_ALONE_THEN_REVERSED = tuple(NEW_MODULES) + tuple(reversed(SEALED_MODULES))
"""A fresh interpreter that has imported ONLY physics_labels/physics_curvature_probe up to that
point, followed by the sealed modules in REVERSED order -- distinct from `_ORDER_NEW_FIRST`
(which imports the sealed set in its own declared order)."""

_ORDERS = {
    "new_first": _ORDER_NEW_FIRST,
    "new_last": _ORDER_NEW_LAST,
    "new_interleaved": _ORDER_NEW_INTERLEAVED,
    "new_alone_then_sealed_reversed": _ORDER_NEW_ALONE_THEN_REVERSED,
}


def _baseline_snapshots() -> Dict[str, Dict[str, str]]:
    """The sealed modules' own state, imported in a subprocess that never imports either of
    Phase 9's new modules at all -- the reference every order below is compared against."""
    return _import_in_order(SEALED_MODULES)


def test_import_physics_modules_does_not_mutate_sealed_modules():
    """Importing both Phase 9 modules before any of the sealed set leaves every one of their
    `vars()` snapshots identical to the physics-free baseline."""
    baseline = _baseline_snapshots()
    with_new = _import_in_order(_ORDER_NEW_FIRST)
    assert with_new == baseline


@pytest.mark.parametrize("order_name", sorted(_ORDERS))
def test_import_purity_holds_under_every_order(order_name):
    """Parametrized over four distinct import orders -- each case runs `_import_in_order` in
    its own subprocess, and every one must match the physics-free baseline exactly."""
    baseline = _baseline_snapshots()
    observed = _import_in_order(_ORDERS[order_name])
    assert observed == baseline, (
        f"order {order_name!r} ({_ORDERS[order_name]!r}) mutated at least one sealed module's "
        "state relative to the physics-free baseline."
    )


def test_snapshot_detects_a_planted_mutation():
    """Proves the snapshot comparison can fail: a subprocess imports one sealed module,
    snapshots it, plants a new module-level attribute, then snapshots it again. The two
    snapshots must differ -- without this test, a suite of only passing import-purity checks
    would be uninformative."""
    import inspect

    snapshot_src = inspect.getsource(_snapshot_module_state)
    script = f"""
import importlib
import json
import sys

sys.path.insert(0, {str(NOTEBOOK_ROOT)!r})

{snapshot_src}

_mod = importlib.import_module("pu_manifold.crossmodal_curvature")
_before = _snapshot_module_state(_mod)
_mod._PLANTED_MUTATION_FOR_TEST = "this attribute must not exist on a real import"
_after = _snapshot_module_state(_mod)
print(json.dumps({{"before": _before, "after": _after}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"] != payload["after"], (
        "the planted mutation was not detected -- the snapshot comparison cannot fail, which "
        "makes every other test in this suite uninformative."
    )
