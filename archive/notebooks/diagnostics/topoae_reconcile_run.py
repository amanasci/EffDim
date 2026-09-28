"""
Phase 02.4's R7 reconciliation step -- the one runner in this codebase that writes
tracked ``.planning/`` files rather than the gitignored ``notebooks/.cache/``. Every
prior runner in this milestone (``topoae_train_run.py``, ``topoae_evaluate_run.py``,
``cae_evaluate_run.py``, ...) writes only cache artifacts; this module is new, careful
code precisely because it does not.

Conditional by design. It reads ``topoae_verdict_{fit_key}.json`` through
``load_verdict`` -- it never recomputes a verdict of its own -- and, only on a recorded
``TOPOAE_VERDICT == "PASS"``, writes a brand-new dated amendment revisiting Phase 02.1's
falsifier *reading* (``02.1-AMENDMENT-03.md``) plus a marker-guarded traceability note
in ``REQUIREMENTS.md`` reinstating the DEC and CURV requirements a coordinate-producing
win implies. On any other verdict -- in particular the sealed
``TOPOAE_VERDICT = FAIL`` -- it is a reported no-op: no amendment is written, no DEC or
CURV requirement is reinstated, and Phase 02.1's graph-native recommendation stands
entirely untouched. A FAIL is a complete, reportable outcome of this phase, never an
error to retry, downgrade, or suppress.

It never edits, retracts, or overwrites any sealed Phase 02.1 measurement, artifact, or
prior amendment. The amendment it may write is always a brand-new dated file --
``write_amendment_if_absent`` opens it in exclusive-create ("x") mode, so an existing
file could never be clobbered even if the existence guard were bypassed -- and no other
path under Phase 02.1's directory is ever opened for writing. The ``REQUIREMENTS.md``
change is strictly append-only, bounded by a fresh
``<!-- topoae-reconciliation:start -->`` / ``<!-- topoae-reconciliation:end -->`` marker
pair, guarded by an exact string search for the end marker rather than any diff or
content heuristic, so the idempotency check is exact and testable. Every step is
existence-guarded: running this twice produces exactly one amendment file and exactly
one ``REQUIREMENTS.md`` change, and the second run reports both steps already written.

Every path this module writes is parameterised by a ``planning_root`` argument
(defaulting to the repository's real ``.planning/``), so tests can drive it against a
``tmp_path``-scoped fake tree without ever touching the real files. Module-level code
never runs the real reconciliation on import -- it only runs inside the
``if __name__ == "__main__":`` guard below -- so importing this module for its four
functions (``load_verdict``, ``write_amendment_if_absent``,
``apply_traceability_update_if_absent``, ``reconcile``) is always test-safe.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/topoae_reconcile_run.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from pu_manifold import cache

# Verbatim-copied from 02.4-PREREGISTRATION.md Section 4 -- not re-derived, not
# imported from a heavy runner module (importing topoae_evaluate_run would re-execute
# its own full sixteen-fit reload-and-recompute pass as a side effect of import).
FIT_KEY = "43cf438bc944c509"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNING_ROOT = REPO_ROOT / ".planning"

AMENDMENT_DIR_REL = Path("phases") / "02.1-geometry-representation-research"
AMENDMENT_FILENAME = "02.1-AMENDMENT-03.md"
REQUIREMENTS_FILENAME = "REQUIREMENTS.md"

TRACEABILITY_START_MARKER = "<!-- topoae-reconciliation:start -->"
TRACEABILITY_END_MARKER = "<!-- topoae-reconciliation:end -->"

# The full DEC/CURV set 02.1's graph-native branch dropped or rewrote
# (02.1-AMENDMENT-02.md Section 4) and that a coordinate-producing PASS reinstates.
REINSTATED_REQUIREMENTS = (
    "DEC-01",
    "DEC-02",
    "DEC-03",
    "DEC-04",
    "DEC-05",
    "CURV-01",
    "CURV-02",
    "CURV-03",
    "CURV-04",
    "CURV-05",
    "CURV-06",
    "CURV-07",
    "CURV-08",
)


def load_verdict(fit_key: str) -> Dict[str, Any]:
    """Read ``topoae_verdict_{fit_key}.json`` through ``cache.cache_path``. Raises
    ``FileNotFoundError`` naming the path when absent -- this runner never assumes an
    outcome and never recomputes a verdict of its own."""
    path = cache.cache_path(f"topoae_verdict_{fit_key}", "json")
    if not path.exists():
        raise FileNotFoundError(
            f"load_verdict: no verdict artifact at {path} -- the reconciliation step "
            "never assumes an outcome and never recomputes a verdict of its own; run "
            "the evaluation runner (topoae_evaluate_run.py) first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _amendment_content(verdict_payload: Dict[str, Any], today: str) -> str:
    """The amendment's prose content -- revisits Phase 02.1's falsifier *reading* only,
    modelled on 02.1-AMENDMENT-02.md's structure. Never asserts the falsifier's literal
    DISTORTION condition was met: TopoAE's gate measures topological signature
    agreement, a different statistic from 02.1's ~0.0796 distance-preservation wall, and
    that difference is the thing this document asks a reader to weigh."""
    metrics = verdict_payload.get("metrics", {})
    thresholds = verdict_payload.get("thresholds", {})
    fit_key = verdict_payload.get("fit_key", "unknown")
    t1 = metrics.get("t1_topo_fidelity")
    t2 = metrics.get("t2_recon_margin")
    t3 = metrics.get("t3_rank_structure")
    thresh_t1 = thresholds.get("t1_topo_fidelity")
    thresh_t2 = thresholds.get("t2_recon_margin")
    thresh_t3 = thresholds.get("t3_rank_structure")
    reinstated = ", ".join(REINSTATED_REQUIREMENTS)

    return f"""---
status: amendment
phase: 02.1-geometry-representation-research
amends: .planning/phases/02.1-geometry-representation-research/02.1-AMENDMENT-02.md (falsifier reading revisited; that document is not overwritten)
created: {today}
trigger: Phase 02.4's Topological Auto-Encoder gate (fit_key={fit_key}) returned
  TOPOAE_VERDICT=PASS -- a coordinate-producing representation clearing its own
  pre-registered validity gate, close to but not identical to
  02.1-PREREGISTRATION.md's symmetric falsifier condition
---

# Amendment 03 -- Falsifier reading revisited: Phase 02.4's TopoAE gate returned PASS

**Written by Phase 02.4's reconciliation step (`notebooks/diagnostics/topoae_reconcile_run.py`),
existence-guarded so a second invocation is a reported no-op.** This document argues
about the *reading* of Phase 02.1's falsifier, not its retraction -- no sealed Phase
02.1 measurement, artifact, or prior amendment is edited, retracted, or overwritten by
this file. This is a brand-new, separately-dated file, opened in exclusive-create mode
so it can never silently clobber a prior version of itself.

## 1. What Phase 02.4 measured

`TOPOAE_VERDICT = PASS`, fit_key `{fit_key}`, sealed in
`notebooks/.cache/topoae_verdict_{fit_key}.json`:

| Gate | Value | Threshold |
|---|---|---|
| `t1_topo_fidelity` | {t1} | < {thresh_t1} |
| `t2_recon_margin` | {t2} | < {thresh_t2} |
| `t3_rank_structure` | {t3} | < {thresh_t3} |

## 2. The falsifier this touches

`02.1-PREREGISTRATION.md` carries a symmetric falsifier: the graph-native verdict is
overturned if a coordinate candidate's `DISTORTION` drops materially below the classical
failure (~0.0796) at a tractable working dimension. That falsifier already fired once,
against the coordinate-producing branch itself (`02.1-AMENDMENT-02.md`).

## 3. Why this is a reading question, not a literal match

**Phase 02.4's gate is not that statistic.** It measures topological signature
agreement (0-dimensional persistence-pair matching between input and latent space),
never distance preservation -- `02.4-PREREGISTRATION.md` Section 1 states this
explicitly, and the gate's own `context_metrics` reports `DISTORTION` only for
comparability, never as a gating quantity. 02.1's ~0.0796 wall was measured on
`DISTORTION = median(|d^2_rep - d^2_geo|/d^2_geo)`, a distance-preservation statistic
every arm that hit that wall was optimising in some form. TopoAE does not optimise it
and does not claim it.

So this amendment does not claim the falsifier's literal condition -- a `DISTORTION`
drop -- was met. It argues about the *reading*: a coordinate-producing representation
reaching a validity gate through a different, legitimately pre-registered route sits
close in spirit to what the falsifier exists to detect, and a reader must weigh that
difference explicitly rather than have it pass silently. That is what this amendment
asks a reader to weigh -- not a claim that 02.1's own statistic was cleared.

## 4. What this reinstates

A coordinate-producing win reinstates the following DEC and CURV requirements, dropped
or rewritten by 02.1's graph-native branch (`02.1-AMENDMENT-02.md` Section 4):
{reinstated}.

## 5. What this does not do

No sealed Phase 02.1 measurement, artifact, or prior amendment is retracted, edited, or
overwritten by this file. `02.1-AMENDMENT-02.md`'s falsifier-firing record stands
exactly as written. `02.1-RECOMMENDATION.md`'s graph-native recommendation is not
deleted -- it remains the phase's sealed record; this amendment adds a new, separately
weighed reading alongside it for Phase 3 to consider, not a replacement.
"""


def write_amendment_if_absent(
    planning_root: Path, verdict_payload: Dict[str, Any], today: Optional[str] = None
) -> Dict[str, Any]:
    """PASS-only. When the recorded verdict is not ``"PASS"``, returns a result
    reporting the no-op and its reason, touching nothing. When it is ``"PASS"`` and
    ``02.1-AMENDMENT-03.md`` already exists under
    ``planning_root/phases/02.1-geometry-representation-research/``, returns
    "already_written" and touches nothing. Otherwise writes that new dated file,
    opened in exclusive-create mode so an existing file could never be clobbered even
    if the existence guard above were bypassed. Opens no other path under that
    directory for writing."""
    verdict = verdict_payload.get("TOPOAE_VERDICT")
    if verdict != "PASS":
        return {
            "action": "skipped",
            "reason": f"TOPOAE_VERDICT={verdict!r} is not PASS -- amendment is a no-op by design",
            "path": None,
        }

    today = today or date.today().isoformat()
    amendment_dir = Path(planning_root) / AMENDMENT_DIR_REL
    amendment_path = amendment_dir / AMENDMENT_FILENAME

    if amendment_path.exists():
        return {
            "action": "already_written",
            "reason": f"{amendment_path} already exists",
            "path": str(amendment_path),
        }

    amendment_dir.mkdir(parents=True, exist_ok=True)
    content = _amendment_content(verdict_payload, today)
    with amendment_path.open("x", encoding="utf-8") as f:
        f.write(content)

    return {
        "action": "written",
        "reason": "PASS verdict, no prior amendment at this path",
        "path": str(amendment_path),
    }


def _traceability_block(verdict_payload: Dict[str, Any], today: str) -> str:
    fit_key = verdict_payload.get("fit_key", "unknown")
    reinstated = ", ".join(REINSTATED_REQUIREMENTS)
    return (
        f"\n{TRACEABILITY_START_MARKER}\n"
        f"**{today}**: Phase 02.4's Topological Auto-Encoder gate PASSed (fit_key="
        f"{fit_key}). The DEC and CURV requirements a coordinate-producing "
        f"representation reinstates are back in scope for Phase 3: {reinstated}. See "
        f"`.planning/phases/02.1-geometry-representation-research/{AMENDMENT_FILENAME}` "
        "for the full reading.\n"
        f"{TRACEABILITY_END_MARKER}\n"
    )


def apply_traceability_update_if_absent(
    planning_root: Path, verdict_payload: Dict[str, Any], today: Optional[str] = None
) -> Dict[str, Any]:
    """PASS-only, guarded by a string search for ``<!-- topoae-reconciliation:end -->``
    in ``REQUIREMENTS.md``. When the marker is present, returns "already_written" and
    touches nothing -- the guard is a marker string search, deliberately never a diff
    or content heuristic, so the idempotency check is exact and testable. Otherwise
    appends, inside a fresh marker pair, a dated note recording the PASS and the
    reinstated DEC/CURV requirement IDs. Read, append, write -- never rewrites
    unrelated content, never renumbers a row, never deletes a line."""
    verdict = verdict_payload.get("TOPOAE_VERDICT")
    if verdict != "PASS":
        return {
            "action": "skipped",
            "reason": f"TOPOAE_VERDICT={verdict!r} is not PASS -- traceability update is a no-op by design",
        }

    today = today or date.today().isoformat()
    req_path = Path(planning_root) / REQUIREMENTS_FILENAME
    existing = req_path.read_text(encoding="utf-8") if req_path.exists() else ""

    if TRACEABILITY_END_MARKER in existing:
        return {
            "action": "already_written",
            "reason": f"{TRACEABILITY_END_MARKER} already present in {req_path}",
            "path": str(req_path),
        }

    block = _traceability_block(verdict_payload, today)
    req_path.parent.mkdir(parents=True, exist_ok=True)
    with req_path.open("a", encoding="utf-8") as f:
        f.write(block)

    return {
        "action": "written",
        "reason": "PASS verdict, no prior traceability marker",
        "path": str(req_path),
    }


def reconcile(
    fit_key: str, planning_root: Optional[Path] = None, today: Optional[str] = None
) -> Dict[str, Any]:
    """Compose ``load_verdict`` + ``write_amendment_if_absent`` +
    ``apply_traceability_update_if_absent``, print one clear line per step (written,
    already_written, or skipped, and why), and return a structured result carrying the
    verdict, both step outcomes, and the paths involved. On a non-PASS verdict, prints
    that the reconciliation is a no-op by design, that 02.1's graph-native
    recommendation stands untouched, and that a FAIL is a complete, reportable outcome
    of this phase rather than an error to retry, downgrade, or suppress."""
    planning_root = Path(planning_root) if planning_root is not None else DEFAULT_PLANNING_ROOT
    verdict_payload = load_verdict(fit_key)
    verdict = verdict_payload.get("TOPOAE_VERDICT")

    print(f"reconcile: fit_key={fit_key} TOPOAE_VERDICT={verdict}")

    amendment_result = write_amendment_if_absent(planning_root, verdict_payload, today)
    print(f"  amendment ({AMENDMENT_FILENAME}): {amendment_result['action']} -- {amendment_result['reason']}")

    traceability_result = apply_traceability_update_if_absent(planning_root, verdict_payload, today)
    print(
        f"  traceability ({REQUIREMENTS_FILENAME}): {traceability_result['action']} -- "
        f"{traceability_result['reason']}"
    )

    if verdict != "PASS":
        print(
            "  Reconciliation is a no-op BY DESIGN on a non-PASS verdict. Phase 02.1's "
            "graph-native recommendation stands entirely untouched. A FAIL is a "
            "complete, reportable outcome of this phase, never an error to retry, "
            "downgrade, or suppress."
        )

    return {
        "fit_key": fit_key,
        "verdict": verdict,
        "amendment": amendment_result,
        "traceability": traceability_result,
        "planning_root": str(planning_root),
    }


if __name__ == "__main__":
    _result = reconcile(FIT_KEY)
    print()
    print(
        f"Done. TOPOAE_VERDICT={_result['verdict']}  "
        f"amendment={_result['amendment']['action']}  "
        f"traceability={_result['traceability']['action']}"
    )
