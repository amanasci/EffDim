"""METHODS / REPORT / MANUSCRIPT_RECOMMENDATION. No manuscript edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_util import write_text


def write_markdowns(out: Path, *, decision: dict, runtime: dict, tests: dict, parity: dict, extra: str = "") -> None:
    write_text(
        out / "METHODS.md",
        r"""# Methods

Task-aligned curvature is the covariant Hessian of a linear probe restricted
to a representation: \(b_{ab}=\langle w_N,II_{ab}\rangle\).

Primary D statistic: invariant energy \(E_D^S=\|b_D^S\|_g^2\) of the
sphere-normal residual decoder tensor, consensus = equal-weight median rank
across seeds `{0,1,2}` if seed-reliable. No best-seed selection.

Primary Q statistic: signed split-half cross energy
\(E_Q^{S,\mathrm{cross}}=\langle b_{Q,A}^S,b_{Q,B}^S\rangle_g\) from the
frozen \(d=16\), \(k=2048\) Q frames. Mean of split-wise cross energies;
never clamp; never average \(B_A,B_B\) then square.

Analysis H uses historical full-data \(w\) and is descriptive.
Analysis C fits Ridge \(\alpha=100\) on a hash-stable 60/40 split and
scores only `probe_eval` neighbours (minimum 128). Primary outcome: held-out MSE.
Controls: log kNN radius, local evaluation-label variance, evaluation count.
P1/P2 are equal-target-weight means, one-sided positive, Holm, synchronized permutations.
""",
        force=True,
    )
    write_text(
        out / "REPORT.md",
        f"""# Report

Primary label: `{decision.get("label")}`.
Reason: {decision.get("reason")}.
Tests: {tests.get("n_passed")}/{tests.get("n_tests")} passed.
Parity ok: {parity.get("ok")}.
Runtime s: {runtime.get("wall_s")}.
{extra}
""",
        force=True,
    )
    write_text(
        out / "MANUSCRIPT_RECOMMENDATION.md",
        f"""# Manuscript recommendation

Do not automatically rewrite the manuscript.

If the confirmatory label is a supported or instrument-specific effect, add a
sentence that task-aligned sphere-normal energy \(E^S\) is a different estimand
from generic \(\|H\|\) and from finite-patch \(K_H\).

Frozen label: `{decision.get("label")}`.
Do not overwrite prior Q or D-residual decision labels.
""",
        force=True,
    )
