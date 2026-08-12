"""
notebooks/diagnostics/check_swiss_roll_contract.py -- re-runnable CLAUDE.md Swiss-roll
sanity-notebook contract check.

Phase 02.7-manifold-template-inference-front-end-inserted, plan `02.7-09`. Every model that
maps data to a lower-dimensional representation, or claims to recover manifold structure,
gets a Swiss roll notebook per CLAUDE.md's "Swiss roll sanity check" section -- this script
is the machine-checkable subset of that contract: cell-count ceiling, executed-with-outputs,
no-cache, no-verdict-artifact, and `make_swiss_roll` presence. It does NOT check content
correctness (whether the roll reads as `ball`, whether the controls read as themselves) --
that is the blocking `checkpoint:human-verify` task's own job, reading the notebook's printed
read-out directly.

Structural checks only, so this script has no opinion on what the notebook's front end
concluded -- it is generic across any Swiss roll notebook this project produces, not specific
to plan `02.7-09`'s own template-inference front end.

Invoke:
    .venv/bin/python notebooks/diagnostics/check_swiss_roll_contract.py <notebook.ipynb>

Exits 0 iff every check passes; prints PASS/FAIL per check and exits 1 on the first failure
category encountered (all checks still run and print before exit).
"""

import json
import re
import sys
from pathlib import Path

CELL_CEILING = 16
"""CLAUDE.md's own "~15 cells" ceiling, with one cell of slack for a notebook that lands at
exactly the boundary -- matches this plan's own acceptance criteria ("at most 16 cells")."""

FORBIDDEN_TERMS_PATTERN = r"VERDICT|pre-registration|merge-base"
"""Gated-milestone-run apparatus CLAUDE.md bars from a sanity notebook: no verdict artifact,
no pre-registration, no git-ancestry proof. A sanity notebook measures and reports; it does
not seal a result the way a ratified milestone run does."""


def _cell_source_text(cell: dict) -> str:
    """A cell's `source` field is a list of strings (or, rarely, a single string) --
    normalise to one joined string for substring/regex checks."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def check(notebook_path: Path) -> bool:
    all_ok = True

    def fail(message: str) -> None:
        nonlocal all_ok
        all_ok = False
        print(f"FAIL: {message}")

    def ok(message: str) -> None:
        print(f"PASS: {message}")

    if not notebook_path.exists():
        fail(f"{notebook_path} does not exist")
        return False

    raw_text = notebook_path.read_text()
    nb = json.loads(raw_text)
    cells = nb.get("cells", [])

    # --- 1. Cell-count ceiling --------------------------------------------------------
    n_cells = len(cells)
    if n_cells <= CELL_CEILING:
        ok(f"cell count {n_cells} <= {CELL_CEILING} (CLAUDE.md's ~15-cell ceiling)")
    else:
        fail(f"cell count {n_cells} exceeds the {CELL_CEILING}-cell ceiling")

    # --- 2. Every code cell executed, with a non-null execution_count -----------------
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if len(code_cells) == 0:
        fail("no code cells found -- nothing was executed")
    else:
        unexecuted = [i for i, c in enumerate(code_cells) if c.get("execution_count") is None]
        if unexecuted:
            fail(f"code cell(s) at index {unexecuted} carry a null execution_count (not run)")
        else:
            ok(f"all {len(code_cells)} code cells carry a non-null execution_count")

        no_output = [
            i for i, c in enumerate(code_cells)
            if not c.get("outputs") and _cell_source_text(c).strip()
        ]
        # A code cell with only a function/class definition and no print/plot legitimately
        # produces zero outputs -- this is informational, not a failure, matching this
        # notebook's own run_front_end definition cell.
        if no_output:
            print(
                f"INFO: {len(no_output)} executed code cell(s) produced no outputs "
                f"(expected for definition-only cells): {no_output}"
            )

    # --- 3. make_swiss_roll present -----------------------------------------------------
    swiss_roll_count = raw_text.count("make_swiss_roll")
    if swiss_roll_count >= 1:
        ok(f'"make_swiss_roll" appears {swiss_roll_count} time(s)')
    else:
        fail('"make_swiss_roll" does not appear anywhere in the notebook')

    # --- 4. No cache read or write -------------------------------------------------------
    cache_count = raw_text.count(".cache")
    if cache_count == 0:
        ok('no ".cache" reference anywhere (CLAUDE.md: never read from or write to the cache)')
    else:
        fail(f'".cache" appears {cache_count} time(s) -- a sanity notebook must touch no cache')

    # --- 5. No gated-milestone-run apparatus ---------------------------------------------
    forbidden_matches = re.findall(FORBIDDEN_TERMS_PATTERN, raw_text)
    if len(forbidden_matches) == 0:
        ok("no VERDICT / pre-registration / merge-base apparatus present")
    else:
        fail(f"forbidden gated-run term(s) found: {sorted(set(forbidden_matches))!r}")

    # --- 6. At least one printed stream output exists (the pass/fail / read-out block) --
    has_stream_output = any(
        out.get("output_type") == "stream"
        for c in code_cells
        for out in c.get("outputs", [])
    )
    if has_stream_output:
        ok("at least one printed (stream) output exists")
    else:
        fail("no printed (stream) output found anywhere -- expected the pass/fail read-out")

    return all_ok


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <notebook.ipynb>", file=sys.stderr)
        sys.exit(2)

    notebook_path = Path(sys.argv[1])
    print(f"== check_swiss_roll_contract.py: {notebook_path} ==")
    passed = check(notebook_path)
    print("== all checks passed ==" if passed else "== one or more checks FAILED ==")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
