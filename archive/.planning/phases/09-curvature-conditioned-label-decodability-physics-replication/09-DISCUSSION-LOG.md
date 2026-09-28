# Phase 9: Curvature-Conditioned Label Decodability (Physics Replication) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-02
**Phase:** 09-curvature-conditioned-label-decodability-physics-replication
**Areas discussed:** Sample, neighbourhood and anchor scale; Row-alignment proof; Replication verdict rule; Probe and control construction

---

## Sample, neighbourhood and anchor scale

| Option | Description | Selected |
|--------|-------------|----------|
| Full 86,471 rows | Whole test split; AE ~8x Phase 7 cost; k=2048 is 1/42 of n | ✓ |
| 16,384-row subset | Match his host's size so k/n = 1/8 | |
| 10,000-row subsample | Milestone standard; k=2048 is 1/5 of n | |

| Option | Description | Selected |
|--------|-------------|----------|
| k=2048 headline + k grid | His absolute k gating, non-gating grid (512..n/8) | |
| k=2048 only | Exact match, no sensitivity read-out | ✓ |
| k = n/8 ≈ 10,816 | Match his ratio, not his absolute k | |

| Option | Description | Selected |
|--------|-------------|----------|
| 512 anchors, seeded uniform | Match his n=512 | ✓ |
| 2048 anchors | Tighter bands, not power-matched | |
| All 86,471 rows as anchors | Per-point design, ~18h curvature | |

| Option | Description | Selected |
|--------|-------------|----------|
| All 86,471 rows as anchor pool | Phase 7's FIELD_EVALUATED_ON convention | |
| AE holdout rows only (~17k) | Curvature only where decoder never trained; departs from Phase 7 | ✓ |

**User's choice:** full rows; k=2048 only; 512 seeded anchors from AE holdout rows.
**Notes:** User asked what local R² is (statistics R² of the OOF ridge probe over the k neighbours, not AE reconstruction loss) and what anchors are for (the sample points where curvature and local R² are both measured; anchor count is the n of the final Spearman).

---

## Row-alignment proof

| Option | Description | Selected |
|--------|-------------|----------|
| Statistical: probe R² vs shifted rows | R²(shift 0) vs shifted/permuted label rows | ✓ |
| Re-embed a few galaxies | Run ViT-B on ~20 test images, match nearest rows | |
| Both | Statistical gates, re-embedding corroborates | |

| Option | Description | Selected |
|--------|-------------|----------|
| Pass rule + fail = halt, no Physics number | Match his DESI treatment | |
| Pass rule + fail = search for the true offset | Adopt a passing shift; post-hoc step | ✓ |
| You decide | | |

| Option | Description | Selected |
|--------|-------------|----------|
| mag_r only; shifts ±1..±10, ±100, ±1000 + 20 permutations | Frozen list, sharpest label | ✓ |
| All four labels, same shift set | Four confirmations; stellar_mass needs a mask | |
| You decide | | |

**User's choice:** statistical shifted-row check on mag_r; search for the offset on failure.
**Notes:** User asked whether the colleague set a standard. Answer recorded in CONTEXT D9-05: a principle ("equal row count is not the proof", DESI struck) but no method; his Physics join is a documented convention with no test on the branch. User asked for the prediction setup to be explained (Smith42/galaxies = images + catalog labels; pu-embeddings = ViT-B vectors of those images, no ids; response = `mag_r` via OOF ridge, scored locally as R² over 2048 neighbours per anchor). User confirmed the colleague assumes row i of pu = row i of galaxies.

---

## Replication verdict rule

| Option | Description | Selected |
|--------|-------------|----------|
| 3-control rank-partial Spearman (his statistic) | Direct comparison to his -0.240 | ✓ |
| 07.1 stratified partial | House statistic | |
| Raw Spearman | Uncontrolled | |

| Option | Description | Selected |
|--------|-------------|----------|
| Negative AND clears Freedman–Lane/FWER null at any d | No magnitude threshold | ✓ |
| Negative and significant at every d | Stricter; his own sweep flips at d=12 | |
| Sign + magnitude band | Require overlap with his band | |

| Option | Description | Selected |
|--------|-------------|----------|
| ‖H_tan‖ carries; ‖H‖ beside | Like-for-like with his radial-removed estimator | ✓ |
| ‖H‖ carries; ‖H_tan‖ beside | Milestone convention | |
| Both must agree | Conservative | |

| Option | Description | Selected |
|--------|-------------|----------|
| Verdict names instrument and d; his d=12 non-comparable | D8-21 pattern, sweep unchanged | |
| Add d=16 to the sweep | D_SWEEP=(16,20,25,32) for a same-d match | ✓ |
| You decide | | |

**User's choice:** his 3-control partial; negative + null at any d; ‖H_tan‖ carries; d=16 added.

---

## Probe and control construction

| Option | Description | Selected |
|--------|-------------|----------|
| alpha=100 fixed | His METHODS §9 value | ✓ |
| Inner-CV grid selection | Phase 5's RIDGE_ALPHA_GRID | |
| alpha=100 headline + grid sensitivity | Both | |

| Option | Description | Selected |
|--------|-------------|----------|
| Curvature side, Phase 7's rank-plant | Real R² kept, synthetic curvature planted | ✓ |
| R² side, degrade predictions in high-curvature regions | Fakes the mechanism; new code | |
| Both | | |

| Option | Description | Selected |
|--------|-------------|----------|
| Secondaries reported, non-gating | Same pipeline, own nulls, mag_r decides | ✓ |
| mag_r only | | |
| Part of the verdict | | |

| Option | Description | Selected |
|--------|-------------|----------|
| Single seed across all d | Phase 7's rule | |
| Three seeds at every d | Twelve fits | |
| Single-seed sweep, then 3 seeds at any surviving d | Two waves, unanimity, never pool | ✓ |

**User's choice:** alpha=100; curvature-side plant; secondaries non-gating; two-wave seeds.
**Notes:** User asked what a positive control is (a manufactured known-size effect run through the same pipeline to establish the detection floor) before choosing the curvature side.

---

## Claude's Discretion

Alignment margin value and offset ratification; fixture fidelity at d=16; shuffled-label calibration shape; permutation and bootstrap counts; OOF/anchor/control seeds and the control-radius k; positive-control target grid; module naming, runner layout, wave decomposition, budget; how 07.1's stratified null attaches (radius rank vs density weights).

## Deferred Ideas

Per-anchor comparison against his K_H (needs `selection.npz`, not on branch; developer's call to request it); k sensitivity grid on our data; R²-side positive control; re-embedding as alignment proof; fixture fidelity at d=32.
