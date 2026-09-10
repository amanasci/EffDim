"""METHODS and REPORT. Does not edit manuscripts."""

from __future__ import annotations

from pathlib import Path


METHODS = """# METHODS — known-curvature instrument failure localization

Bounded 64-anchor diagnostic. No new decoder training. No new finite-patch
oracle integrations. Previous audit trees are read-only.

## Panel

F0, F1, F2, F4 at S0/N0; plus F4 S1/N0 and F4 S0/N2 η=0.10.
d=16, D=768, n=16384, 64 hash-stable holdout anchors, k∈{512,2048}.
q=d(d+1)/2=136. At k=2048 each production split has 1024 observations;
at k=512 each has 256. k=256 is not run (128 half-samples < 136 coefficients).

## Q ladder

- Q0: cached T2/T3 scalars (tensors only where T2=T1, i.e. F0–F2). F4 T2
  tensors were not stored; those comparisons are scalar-only or marked unavailable.
- Q1: exact generator frame, exact latent offsets, full neighbourhood, thin-SVD
  least squares, machine-eps pseudoinverse cutoff. Sphere-normal residual is
  applied without materialising I_D.
- Q2: Q1 plus production RIDGES selected on a 20% neighbourhood hold-out.
- Q3: sphere-tangent local PCA (same convention as nested_pca_frame; numpy SVD
  on the CPU). Chart coordinates are J-projected, then Procrustes-aligned to
  the true frame before tensor comparison.
- Q4: PCA frame plus production ridge.
- Q5: frozen `fit_anchor_quadratic` A/B split (exact production path, including
  torch `nested_pca_frame` / `fit_quad`).

Reducing k is a bias–variance probe, not an asymptotic consistency test:
r∝(k/n)^{1/d}; (512/2048)^{1/16}≈0.917.

## Decoder

No `.pt` weights exist in the audit cells. D0/D3 use cached
`decoder_seeds.parquet` / `anchors.parquet` and reconstruction R². D1 (true
projector) and D2 (decoder finite difference) are unavailable without weights.
"""


def write_reports(out: Path, *, summary: dict, decision: dict, tests: dict) -> None:
    label = decision.get("label", "bounded_failure_localization_unresolved")
    report = f"""# REPORT — instrument failure localization

## Decision

**{label}**

{decision.get("reason", "")}

See `decision.json` for frozen thresholds. This run localizes failure stages;
it is not a new 62-cell benchmark and does not edit manuscripts.

## Runtime and completeness

- wall-clock: {summary.get("runtime_s")} s
- tests: {tests.get("n_passed")}/{tests.get("n_tests")}
- stages: {summary.get("stages_done")}
- skipped: {summary.get("skipped")}
"""
    (out / "METHODS.md").write_text(METHODS)
    (out / "REPORT.md").write_text(report)


def maybe_ablation_plot(out: Path, qtab) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if qtab is None or not len(qtab):
        return
    m = qtab[(qtab.sampling == "S0") & (qtab.noise == "N0") & (qtab.k == 2048)]
    if not len(m):
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    stages = [s for s in ("Q1", "Q2", "Q3", "Q4", "Q5") if s in set(m.stage)]
    xs = range(len(stages))
    for fx, col in (("F0", "C0"), ("F1", "C1"), ("F2", "C2"), ("F4", "C3")):
        ys = [float(m[(m.stage == s) & (m.fixture == fx)]["rel_Kdir_T1"].median()) for s in stages]
        ax.plot(list(xs), ys, marker="o", label=fx, color=col)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(stages)
    ax.set_ylabel("median rel. $K_{dir}$ vs T1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "q_ablation.png", dpi=120)
    plt.close(fig)
