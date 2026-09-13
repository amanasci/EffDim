# 09-SUPPLEMENT-08 — relative second fundamental form between the HSC and Legacy embeddings: pilot

**Status:** post-hoc, pilot. **Not pre-registered. Feeds no verdict.** **Written:** 2026-09-13 UTC

## Design

Runner `notebooks/diagnostics/08_relative_ii_run.py` (commit 7b98b70), pod `universetbd-0`, 16 threads.
Data: the Phase 7 10,000 paired rows (`subsample_20260729_a79b3460b838fd0a.npz`, sha256 b80ee1d0…),
unit-normalized. Two sphere-projected plain AEs (Phase 7 protocol, 600 epochs) at d=20: HSC (F)
var_explained 0.974, Legacy (G) 0.982. Global ridge alignment A: x_G ≈ A x_F + b on the 8,000 training
rows (α=1), R² 0.656 train / 0.636 holdout. 2,048 seeded anchors (seed 20260912). Per anchor: J, II of both
decoders; tangent map L = J_G⁺ A J_F; `tan_resid = ‖A J_F − J_G L‖/‖A J_F‖`; `II_rel = ‖II_G(L·,L·) − P_N^G A II_F‖_{g_F}`;
in-sphere variant; `II_rel_loc` with L fitted on 256 neighbours' latent codes; `II_rel_emp` = G-normal part
of the quadratic coefficient of the alignment residual regressed on F's tangent coordinates (1,024
neighbours). Outcome: per-point MKNN (k=20, 50). Controls: log kNN radius in both ambient spaces (k=30;
multi-scale k∈{10,30,100,300}); Freedman–Lane 2,000 draws. Record `notebooks/.cache/08_relative_ii_d20.jsonl`
(sha256 48b3322dd39f1b1a…). Wall clock 43,991 s, of which 40,923 s was a naive einsum in the column stage
(fixed in principle, not re-run). d=25 launched and killed at the same stage; not reported.

## Results (multi-scale control; k=20 / k=50)

| column | partial | p |
|---|---|---|
| `H_tan_F` (HSC decoder ‖H_tan‖) | −0.15 / −0.17 | <0.001 |
| `H_tan_G` (Legacy decoder, the Phase 7 field) | +0.04* / +0.02* | n.s. |
| `II_F`, `II_G` | +0.02* / +0.01*, +0.04* / +0.03* | n.s. |
| `tan_resid` (first-order obstruction) | −0.18 / −0.21 | <0.001 |
| **`II_rel`** (A-induced tangent map) | +0.04* / +0.05 | 0.10 / 0.03 |
| `II_rel_S` | +0.02* / +0.02* | n.s. |
| `II_rel_loc` (locally fitted L) | −0.18 / −0.24 | <0.001 |
| `II_rel_emp` (decoder-free) | +0.13 / +0.13 | <0.001 |
| `align_resid` (local RMS of the linear residual) | −0.26 / −0.27 | <0.001 |

Cross-column facts: median `tan_resid` 0.44; median ‖L − L_loc‖/‖L‖ 1.12 (the global map's tangent map and
the local chart correspondence disagree completely); ρ(`II_rel`, `II_rel_loc`) 0.54; ρ(`II_rel`, `II_rel_emp`) 0.43;
every curvature column couples to log radius at −0.25 to −0.60.

## Reading

1. The global linear map explains 64 % of the Legacy embedding and carries HSC tangent frames into Legacy
   with 44 % residual: on this pair the obstruction to linear alignment is already **first order**.
2. The theory's quantity built on that map, `II_rel`, is null against MKNN under density control.
3. Alternatives disagree in sign (`II_rel_loc` −0.18…−0.24; `II_rel_emp` +0.13). `II_rel_loc` inherits the
   scale of L_loc (a density-ratio between charts); `II_rel_emp` is a 231-coefficient fit on a 10 %-of-data
   neighbourhood whose norm scales with the residual noise and inversely with radius. Neither is trusted.
4. The HSC decoder's ‖H_tan‖ reads −0.15/−0.17 under the two-space multi-scale control, where the Legacy
   field (Phase 7) reads null: the one-embedding trace statistic is again estimator/embedding dependent.

**Conclusion:** the relative curvature is not yet an instrument-stable quantity on this pair; the pilot
does not support or refute curvature-concentrated convergence. Manuscript: one sentence in the Discussion.

---
*Phase: 09 (cross-phase follow-up to Phases 7–8). Not pre-registered, feeds no verdict.*
