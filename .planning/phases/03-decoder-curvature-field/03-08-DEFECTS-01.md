# 03-08 Defects 1 — the PU grid measured three artifacts, not the CAE

**Found:** 2026-08-14, after the nine-cell CPU grid completed (12 records, `03_curvature_field_pu.jsonl`)
**Status:** grid result NOT usable; no `n_charts` selection may be read from it
**Raised by:** the developer, from theory — the CAE is proven in its source paper to beat a plain
autoencoder on reconstruction, so the observed 3.5x loss had to be an experimental fault rather than
a property of the model. All three defects below were found by following that objection.

---

## Summary

The grid ran to completion and the pre-declared selection rule was applied faithfully. The rule
selected `n_charts=16`. **That selection must not be used.** Every axis it ranked on is corrupted by
one of three defects, none of which concern chart auto-encoders.

| # | Defect | What it corrupts |
|---|--------|------------------|
| 1 | Control autoencoder has 2x the CAE's effective bottleneck | reconstruction axis; D-12 trigger (recon leg) |
| 2 | Training protocol far more aggressive than the roll's; 5/9 cells stopped at 7 epochs | reconstruction axis (confounded with training length) |
| 3 | PH normalization leaves distance scale proportional to sqrt(d); latent 40-dim vs ambient 768-dim | all 8 `latent\|*` PH cells; selection axis 3; D-12 trigger (PH leg) |

---

## Defect 1 — the "matched" control is not matched

`_run_control_cell` builds `cae.PlainAutoEncoder(AMBIENT_DIM, PU_EMBED_DIM, ...)` with
`PU_EMBED_DIM = 40`. `PlainAutoEncoder`'s second positional argument is `latent_dim` — the
bottleneck. So the control reconstructs through a **40-dimensional** bottleneck.

The CAE's information path is `encode -> z (embed_dim=40) -> chart_coords -> chart coordinate
(chart_dim=20) -> chart_decoder -> embedding_decoder -> 768`. Its narrowest point is
**`chart_dim = 20`**.

The control therefore solves a strictly easier problem with double the latent capacity. The observed
result — control `mse_per_dim ~3.5e-05` vs best CAE `1.25e-04`, a 3.5x advantage in 1/70th the wall
clock — is what an unmatched bottleneck produces on its own and is not evidence about atlases.

`PlainAutoEncoder`'s own docstring states the question it exists to answer: *"whether the atlas
bought anything over one chart at matched capacity."* At 40 vs 20, capacity is not matched.

The Swiss roll runner does this correctly — `cae.PlainAutoEncoder(3, CHART_DIM, ...)` — and CLAUDE.md
states the rule explicitly: *"same width, depth, and training protocol, at the same 2-D bottleneck."*

**Fix:** `cae.PlainAutoEncoder(AMBIENT_DIM, PU_CHART_DIM, ...)`. Three control cells, ~30 s each.
Optionally retain the 40-dim control as a separate, clearly-labelled capacity reference.

## Defect 2 — the PU training protocol is far more aggressive than the roll's

| parameter | Swiss roll | PU grid |
|---|---|---|
| `max_epochs` | 300 | **40** |
| `early_stop_patience` | 25 | **5** |
| `fps_pretrain_epochs` | 20 | 5 |
| `lip_weight` | 1e-3 | **1e-2** (10x) |

`train_cae` early-stops on **total** loss — reconstruction plus the Lipschitz penalty — and that
penalty is now 10x heavier, so a plateau in the regularizer can stop training while reconstruction is
still improving. Observed `epochs_run` across the nine grid cells: `[7, 7, 7, 7, 7, 34, 34, 40, 40]`.
All three controls ran the full 40 without early stopping, i.e. were still improving at the cap.

Consequence for the selection: the tie on axis 1 (`cond(g)`, within a factor of 2) passed the
decision to axis 2 (`mse_per_dim`), where `n_charts=16` won having trained 34-40 epochs and 2000+ s
against `n_charts=4`'s 7 epochs and 110 s. **The winning axis substantially measures how long each
cell happened to train before early stopping**, which is not a property of `n_charts`.

Note also that 02.2's per-fit training anchor is 1,941.2 s; the `nc=4` cells used 110 s.

## Defect 3 — PH compares across an ambient-dimension gap the normalization does not close

`cloud_distance_matrix(points, prescale=True)` scales each cloud by
`topoae.latent_unit_scale(cloud) = 1 / sqrt(mean(var(cloud, axis=0)))`, giving **mean per-dimension
variance 1**. That is isotropic and neighbour-preserving — correct for what it was designed for — but
it leaves the *distance* scale growing as `sqrt(d)`. Bottleneck distance is an absolute metric on
diagrams, so comparing a 40-dim latent against a 768-dim reference compares diagrams living at scales
`sqrt(768/40) ~ 4.4x` apart. Every reference feature then matches to the diagonal and the value pins
at `saturation_value = 0.5 * max_persistence(ref)`.

**Measured directly.** Identical intrinsic structure (300 points drawn in 10-d, isometrically
embedded into each ambient dimension), each cloud prescaled by the current normalizer:

```
 ambient dim  applied_scale  H1 max_pers  median dist
          10         1.0079       0.5510       4.3278
          40         2.0159       1.1020       8.6555
         128         3.6061       1.9714      15.4835
         768         8.8331       4.8288      37.9267
```

Observed persistence ratios `1, 2.00, 3.578, 8.763` match `sqrt(d/10)` = `1, 2, 3.578, 8.764`
exactly. Bottleneck between the 40-dim and 768-dim diagrams of *the same structure* equals the
saturation ceiling exactly (`15.2054 == 15.2054`), i.e. **saturated**.

The PH axis is therefore measuring **ambient dimension, not topology preservation**.

**Internal control confirming the mechanism.** Within a single grid record:
`decoder_image|ambient|H0|bottleneck_norm = 0.4486, saturated = False` — the decoder image is 768-dim,
matching the reference. Every `latent|*` cell is `0.5, saturated = True` — 40-dim against 768-dim.
Matched dimension yields a real measurement; mismatched dimension always saturates. No property of
the CAE can produce that pattern.

**Fix, verified.** Normalize by a distance-scale statistic, which is dimension-invariant by
construction. Same structure at d=40 vs d=768, ideal bottleneck 0:

```
current: mean per-dim variance   bottleneck=  2.4144  ceiling=  2.4144  saturated=True
median pairwise distance         bottleneck=  0.0000  ceiling=  0.0637  saturated=False
diameter (max distance)          bottleneck=  0.0000  ceiling=  0.0308  saturated=False
```

Both distance-scale normalizers give exact agreement.

### Cross-phase implication — this is not confined to Phase 3

`02.6-SCREENING-RULE-02.md` already carries a "bottleneck-saturation travelling caveat", recording
that a saturated cell is flagged and never read as a measurement. **The symptom was observed and
quarantined; the cause was never diagnosed.** Consequently:

- Saturation was treated as an occasional nuisance rather than as the deterministic consequence of
  comparing clouds across an ambient-dimension gap.
- Any prior phase comparing a latent cloud to an ambient reference across such a gap carries the same
  artifact — `decoder_substrate_ph_screen_run.py` and `template_benchmark_run.py` both use this
  probe and should be audited before their PH conclusions are relied on further.

This should be checked before it propagates into Phase 3's findings or any downstream milestone.

---

## What is NOT in question

- The runner behaved correctly and did not fail silently.
- The selection rule was declared before any PU number existed and was applied unchanged.
- The `n_charts=4` disqualification on occupancy (median 1 distinct chart) is sound and unaffected.
- Every diagnostic was kept uncollapsed, with no weighted composite. **This is the only reason the
  three defects are legible at all** — a single composite score would have hidden all of them behind
  one number that looked plausible.
- `03-08-DECLARATION-01.md` (CPU grid primary, GPU grid replication) stands. It governs which grid is
  read, not whether this grid is valid.

## Consequence for the phase

No `n_charts` may be selected from this grid, and `03-09` must not compute a curvature field from a
fit chosen by it. The grid must be re-run after defects 1 and 2 are fixed and defect 3 is either
fixed or its axis explicitly withdrawn from the selection rule.

Withdrawing an axis after seeing the data would normally be a forking-paths move. It is defensible
here only because the axis is **provably constant** — it takes the value 0.5 in 12 of 12 records
including controls, and is demonstrably a function of ambient dimension rather than of any fitted
model. That reasoning must be recorded in whatever amendment authorizes the re-run, not assumed.
