"""D8-23's cross-cutting import-purity regression test.

**No in-repo precedent.** A repo-wide grep of ``notebooks/pu_manifold/tests/`` (recorded in
``08-PATTERNS.md``'s "No Analog Found" section) found only ordinary ``monkeypatch.setattr``
usages that patch a module's OWN constants for test isolation -- none of them assert that
*importing* one module leaves another module's ``vars()`` unmutated. This suite invents that
pattern from scratch for D8-23's cross-cutting constraint: importing ``pu_manifold.cka`` must
never mutate module-level state in any of nine sealed ``pu_manifold`` modules, regardless of
import order.

This suite loads no PU data, trains nothing, and opens no on-disk cached array -- it exercises
only the import machinery and each module's own ``vars()``.

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
)
"""The nine sealed `pu_manifold` modules D8-23 names verbatim. Importing `pu_manifold.cka` must
never mutate any of these -- no monkeypatching, no attribute assignment onto any one of them,
regardless of import order."""


def _snapshot_module_state(module) -> Dict[str, str]:
    """A stable textual fingerprint of every public (non-dunder) name in `vars(module)`.

    `repr(value)` for scalars, strings, tuples and other immutables; for a numpy array, its
    shape, dtype and a sha256 of `value.tobytes()`; for a `set`/`frozenset`, a SORTED list of
    each element's own `repr` (a bare `repr(frozenset(...))` is not reproducible across separate
    interpreters -- CPython's string hash is randomized per process by default, so a
    frozenset-of-strings' iteration order, and therefore its repr, differs between two
    subprocesses that never touched each other's state at all; sorting the element reprs removes
    that false positive); for a function, class or module, its qualified name. This keeps the
    snapshot deterministic across runs without depending on object identity (a fresh interpreter
    reimports every module from scratch each time this runs, so identity-based comparison would
    never match even absent any real mutation).

    Self-contained on purpose (its own local imports, no reliance on this test file's top-level
    imports): `_import_in_order` embeds this function's own source, via `inspect.getsource`,
    into a subprocess script, so the fingerprint logic lives in exactly one place rather than
    being duplicated in a script string.
    """
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
    """Runs `order` (a sequence of names, each either a `SEALED_MODULES` entry or the literal
    string `"cka"`) as a sequence of imports inside a **fresh subprocess**, then returns
    `_snapshot_module_state` for every one of the nine `SEALED_MODULES` as they stand once every
    import in `order` has completed.

    Each order runs in its own subprocess via `subprocess.run([sys.executable, "-c", script])`,
    because `sys.modules` caching within a single process would hide exactly the mutation this
    suite hunts for -- a module already resident in `sys.modules` from an earlier import in the
    same process could mask a later import's side effect.

    `order` must name every one of `SEALED_MODULES` at least once (any position/repetition is
    fine) so that a snapshot exists for each of the nine at the end; `"cka"` may appear zero or
    more times, at any position.
    """
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
    if _name == "cka":
        from pu_manifold import cka as _mod
    else:
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

# The four orders D8-23's plan names explicitly. Each is a full sequence naming every one of the
# nine SEALED_MODULES plus "cka" exactly once; only the relative position of "cka" (and, for the
# fourth case, the order of the nine sealed modules themselves) differs between orders, so a
# comparison against the CKA-FREE baseline below isolates whether cka's import position changes
# anything about the sealed modules' final state.
_ORDER_CKA_FIRST = ("cka",) + SEALED_MODULES
_ORDER_CKA_LAST = SEALED_MODULES + ("cka",)
_ORDER_CKA_MIDDLE = SEALED_MODULES[:8] + ("cka",) + SEALED_MODULES[8:]
_ORDER_CKA_ALONE_THEN_NINE_REVERSED = ("cka",) + tuple(reversed(SEALED_MODULES))
"""A fresh interpreter that has imported ONLY `cka` up to that point, followed by the nine
sealed modules in REVERSED order -- distinct from `_ORDER_CKA_FIRST` (which imports the nine in
`SEALED_MODULES`'s own declared order), so this order is not a duplicate of the first case while
still satisfying "cka alone, followed by the nine"."""

_ORDERS = {
    "cka_first": _ORDER_CKA_FIRST,
    "cka_last": _ORDER_CKA_LAST,
    "cka_after_crossmodal_before_density": _ORDER_CKA_MIDDLE,
    "cka_alone_then_nine_reversed": _ORDER_CKA_ALONE_THEN_NINE_REVERSED,
}


def _baseline_snapshots() -> Dict[str, Dict[str, str]]:
    """The nine sealed modules' own state, imported in a subprocess that never imports `cka` at
    all. The reference every cka-including order is compared against."""
    return _import_in_order(SEALED_MODULES)


def test_import_cka_does_not_mutate_sealed_modules():
    """Importing `pu_manifold.cka` before any of the nine sealed modules leaves every one of
    their `vars()` snapshots identical to the cka-free baseline."""
    baseline = _baseline_snapshots()
    with_cka = _import_in_order(_ORDER_CKA_FIRST)
    assert with_cka == baseline


@pytest.mark.parametrize("order_name", sorted(_ORDERS))
def test_import_purity_holds_under_every_order(order_name):
    """Parametrized over four distinct import orders (`cka` first, last, in the middle, and a
    fresh interpreter that imports only `cka` before the nine in reversed order) -- each case
    runs `_import_in_order` in its own subprocess (`sys.executable` above), and every one must
    match the cka-free baseline exactly."""
    baseline = _baseline_snapshots()
    observed = _import_in_order(_ORDERS[order_name])
    assert observed == baseline, (
        f"order {order_name!r} ({_ORDERS[order_name]!r}) mutated at least one sealed module's "
        "state relative to the cka-free baseline."
    )


def test_snapshot_detects_a_planted_mutation():
    """Proves the snapshot comparison can fail: a subprocess imports one sealed module, snapshots
    it, plants a new module-level attribute, then snapshots it again. The two snapshots must
    differ -- without this test, a suite of only passing import-purity checks would be
    uninformative (it could pass merely because the comparison itself never fires)."""
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
