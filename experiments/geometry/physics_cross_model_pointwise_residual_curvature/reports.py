"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION. No manuscript edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_util import write_text


def write_markdowns(out: Path, *, decision: dict, runtime: dict, tests: dict, parity: dict, extra: str = "") -> None:
    write_text(
        out / "METHODS.md",
        r"""# Methods

Pointwise sphere-residual decoder curvature \(C_H=\|H^S\|\) with averaged
\(H^S=(1/d)g^{ab}B^S_{ab}\), differentiating through output normalization.
`PlainAutoEncoder` \(d=16\), hidden \((250,250,250)\) SiLU, 400 epochs, label-blind
reconstruction, evaluation anchors excluded, neighbours retained. Native ambient
dimension per encoder. Frozen G/P outcomes and controls from the existing
cross-model probe tables. Consensus \(C_H\) is equal-weight median rank across
seeds `{0,1,2}` only if both seed-reliability gates pass. ViT-B is the frozen
reference and is excluded from the H1–H3 equal-model-weight aggregate.
Synchronized Freedman–Lane permutations and synchronized anchor bootstraps.
Optional Q resampling is resource-contingent and does not change the D-residual label.
""",
        force=True,
    )
    write_text(
        out / "REPORT.md",
        f"""# Report

Primary label: `{decision.get("label")}`.
Optional Q label: `{decision.get("q_label")}`.
Reason: {decision.get("reason")}.
Tests: {tests.get("n_passed")}/{tests.get("n_tests")} passed.
Parity ok: {parity.get("ok")}.
Runtime s: {runtime.get("wall_s")}.
New decoders: {runtime.get("n_new_decoders")}.
{extra}
""",
        force=True,
    )
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        rf"""# Manuscript recommendation

Do not automatically rewrite the manuscript.

Recommended one-sentence addition if the primary label is a replication or
heterogeneity result: report fixture-validated pointwise residual curvature
\(C_H\) separately from the finite-patch Q statistic, and state the H1–H3
aggregate with Holm-corrected p-values.

Frozen label: `{decision.get("label")}`.
Do not overwrite `pointwise_residual_probe_relation_unresolved`,
`representation_specific_effect`, or `q_global_and_adaptation_associations_geometry_robust`.
""",
        force=True,
    )
