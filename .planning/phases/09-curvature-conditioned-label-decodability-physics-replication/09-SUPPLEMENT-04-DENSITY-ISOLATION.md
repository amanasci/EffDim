# 09-SUPPLEMENT-04 — where density enters the local probe, and what is left of the curvature partial when it is removed

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** Nothing here changes
`09-WAVE-A-RESULTS.md`, its Amendment 01 re-run, Supplements 01 to 03, or the phase verdict.
**Written:** 2026-09-11 UTC. Experiment 7 (fixture gamma refinement) is running as this is written;
its section is marked pending and will be filled from its record.

## The question

Supplement 03 showed that on a known surface the pre-registered curvature-vs-local-$R^2$ partial
takes the sign of the sample's density–curvature coupling, and that the three frozen controls do not
remove it. That leaves open (a) *where* density enters a global linear probe scored locally, (b)
whether a stronger density control or a density-robust probe design removes the dependence, and
(c) what, if anything, remains of a curvature association on the real Physics data once it is
removed. Six experiments, all on the sealed Phase 9 objects, all post-hoc.

## Provenance

- Experiments 1 and 2: `notebooks/diagnostics/09_density_anchor_tables_run.py`, commit `bdd32ff`,
  run locally on the sealed anchor tables (Amendment 01 tables for the decoder, colleague tables for
  `K_H^cross`). Record `notebooks/.cache/09_density_anchor_tables.json`. 5,000 permutations.
- Experiments 4, 5, 6 and the multi-scale control: `notebooks/diagnostics/09_density_probe_run.py`,
  commit `918f4cf`, run on the execution host of `09-EXECUTION-HOST.md` (host label `pod128`, 16
  threads, CPU only) under the developer's instruction to test the density angle. Guide digest
  verified unchanged before any command. Wall clock 3,296 s of which 1,474 s was reading label
  shards over the network mount. Record `notebooks/.cache/09_density_probe/09_density_probe.jsonl`,
  163 rows, sha256 `57b4376c…cfbeafb`, verified on both sides of the transfer. Pod copy
  `/mnt/ssd-cluster/effdim/density-probe/`.
- Scale-coupling table (§2): computed locally from the cached normalized Physics embeddings and the
  sealed anchor tables; `log r_2048` reproduced against the tables to 6e-15.
- Experiment 7: `09_fixture_probe_decodability_run.py` at `--gammas 0.4,0.6,0.8`, same host, record
  `09_fixture_probe_decodability_gamma2.jsonl`. Pending.

Every variant below reproduces the sealed global probe first: max $|\Delta R^2|$ against the
Amendment 01 anchor tables is 5.6e-15 (mag\_r), 4.4e-16, 4.9e-15, 1.0e-15.

## 1. Design

Same 86,471 rows, same 512 anchors, same $k=2048$ neighbourhoods, same ridge $\alpha = 100$ and
5-fold OOF split. Curvature columns are read from the sealed tables and never refit.

| variant | what changes |
|---|---|
| `global` | the sealed pipeline, reproduced |
| `weighted_half` | global probe refit with sample weights $w \propto r_{k=30}^{d/2}$, $d=20$, clipped at the 1st/99th percentile |
| `weighted_full` | same with $w \propto r_{k=30}^{d}$; weights span 2e-7 to 0.36 (mean 1), so the fit rests on the sparsest ~1 % of rows and global $R^2$ falls to 0.29 (mag\_r) and −0.16 (smooth\_fraction). Stress case only |
| `local` | a ridge probe fit inside each 2048-patch, scored 5-fold OOF within the patch |
| `fixed_radius` | global probe scored in a ball of the median 2048-radius ($R_0 = 0.467$); counts run 8 to 16,085, anchors under 32 masked (48 to 53 of 512) |

Two control sets for every partial: the frozen three ($\log r_{2048}$, local label variance,
evaluation count) and a multi-scale set ($\log r_k$ at $k \in \{16, 64, 256, 1024, 2048\}$ plus the
same two). Freedman–Lane $p$ with 2,000 draws. A matched-radius pair test (adjacent anchors in
$\log r_{2048}$ closer than 0.02; median gap 0.0009) as a regression-free check.

## 2. Both curvature fields couple to density at every scale

$\rho(\text{field}, \log r_k)$ at the anchors:

| field | $k=16$ | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|---|
| decoder $\|H_{\rm tan}\|$, $d=16$ | −0.44 | −0.46 | −0.50 | −0.56 | −0.60 |
| decoder, $d=20$ | −0.30 | −0.33 | −0.38 | −0.45 | −0.50 |
| decoder, $d=25$ | −0.41 | −0.45 | −0.50 | −0.57 | −0.62 |
| decoder, $d=32$ | −0.42 | −0.44 | −0.49 | −0.55 | −0.60 |
| colleague $K_H^{\rm cross}$, rank 12 | +0.39 | +0.44 | +0.51 | +0.59 | +0.64 |
| colleague, rank 16 | +0.49 | +0.53 | +0.59 | +0.66 | +0.70 |
| colleague, rank 20 | +0.58 | +0.62 | +0.67 | +0.73 | +0.77 |
| $\rho(\log r_k, \log r_{2048})$ | 0.88 | 0.91 | 0.95 | 0.99 | 1.00 |

Opposite signs, both monotone in scale, both strongest at the patch scale. The five radii are
collinear (0.88 to 1.00 with the patch radius), so the multi-scale control adds information only
through the differences between scales: the density *profile* inside the patch, not its level.

## 3. Experiment 1: a decile density control changes nothing

Replacing the linear $\log r_{2048}$ term by ten decile dummies moves no partial by more than 0.05.
Decoder $d=16$ mag\_r +0.332 → +0.330; photo\_z +0.354 → +0.357; colleague rank 16 mag\_r −0.121 →
−0.122; rank 20 −0.229 → −0.227. Radius at the patch scale explains under 11 % of local $R^2$
variance for any label (decile model: 7 %, 5 %, 11 %, 6 %); with label variance and count, 18 %,
6 %, 53 %, 35 %.

## 4. Experiment 2: matched-radius pairs keep each instrument's sign

502 adjacent pairs at essentially identical patch radius. Sign concordance $\tau$ of
$\Delta$curvature against $\Delta R^2$:

| | mag\_r | photo\_z | smooth\_fraction | stellar\_mass |
|---|---|---|---|---|
| decoder $d=16$ | +0.22 | +0.20 | +0.21 | +0.04 (n.s.) |
| decoder $d=25$ | +0.12 | +0.27 | +0.35 | 0.00 |
| colleague rank 16 | −0.17 | −0.09 | −0.17 | +0.06 (n.s.) |
| colleague rank 20 | −0.22 | −0.02 (n.s.) | −0.11 | 0.00 |

Holding the patch radius fixed pairwise does not reconcile the instruments. Whatever separates them
is not the $k=2048$ radius.

## 5. Where density enters the probe: two opposing mechanisms

$\rho(\text{local } R^2, \log r_k)$ per variant, at $k=16$ and $k=2048$:

| label | global | weighted\_half | local | fixed\_radius |
|---|---|---|---|---|
| mag\_r | −0.00 / −0.23 | +0.29 / +0.11 | +0.33 / +0.28 | −0.46 / −0.72 |
| photo\_z | +0.40 / +0.23 | +0.56 / +0.45 | +0.52 / +0.49 | −0.33 / −0.58 |
| smooth\_fraction | +0.45 / +0.29 | +0.74 / +0.67 | +0.69 / +0.63 | −0.24 / −0.50 |
| stellar\_mass | −0.06 / −0.20 | +0.51 / +0.50 | +0.19 / +0.12 | −0.52 / −0.72 |

Two mechanisms with opposite sign, and the label decides which wins.

- **Fit side.** A global probe fit by mean squared error fits dense regions better. Reweighting the
  fit toward sparse rows (`weighted_half`) or fitting inside each patch (`local`) flips mag\_r from
  −0.23 to +0.11 and +0.28 and stellar\_mass from −0.20 to +0.50 and +0.12. For mag\_r the fit-side
  effect grows with patch scale (−0.00 at $k=16$ to −0.23 at 2048): it is a patch-size effect, not a
  pointwise density effect.
- **Scoring side.** At fixed $k$, a sparse anchor's patch spans more of the manifold, so local label
  variance is larger and a linear map explains a larger fraction of it. Under `local` and
  `weighted_half`, $\rho(R^2, \text{label variance})$ runs +0.53 to +0.88. This is why photo\_z and
  smooth\_fraction read *better* in sparse regions even under the sealed global probe.
- **Fixed radius** removes the scoring-side mechanism and exposes count: $\rho(R^2, \text{count})$ is
  +0.50 to +0.71, and $R^2$ falls steeply with radius for every label (−0.50 to −0.72). It trades one
  confound for another and masks the sparsest tenth of anchors.

So "density hurts the probe" is not a statement this data supports in general. The sealed global
probe is better in dense regions for two labels and worse for two, and every probe design moves
the sign.

## 6. What is left of the curvature partial

Cells are the controlled partial; `*` $p<0.01$, `.` $p<0.05$ (Freedman–Lane, 2,000 draws).

**Sealed statistic (global probe, three controls), for reference:**

| | d16 | d20 | d25 | d32 | c12 | c16 | c20 |
|---|---|---|---|---|---|---|---|
| mag\_r | +0.33\* | +0.02 | +0.03 | −0.01 | −0.10. | −0.15\* | −0.24\* |
| photo\_z | +0.36\* | +0.31\* | +0.37\* | +0.42\* | −0.10. | −0.14\* | −0.09. |
| smooth\_fraction | +0.34\* | +0.33\* | +0.34\* | +0.42\* | −0.25\* | −0.16\* | −0.15\* |
| stellar\_mass | +0.07 | +0.12\* | +0.22\* | +0.26\* | +0.01 | +0.03 | −0.04 |

**Global probe, multi-scale control:**

| | d16 | d20 | d25 | d32 | c12 | c16 | c20 |
|---|---|---|---|---|---|---|---|
| mag\_r | +0.25\* | −0.17\* | −0.15\* | −0.17\* | +0.10. | −0.01 | −0.13\* |
| photo\_z | +0.28\* | +0.13\* | +0.20\* | +0.27\* | +0.10. | −0.01 | +0.01 |
| smooth\_fraction | +0.21\* | +0.15\* | +0.16\* | +0.29\* | −0.08 | −0.03 | −0.02 |
| stellar\_mass | −0.02 | −0.00 | +0.11. | +0.16\* | +0.10. | +0.11. | +0.02 |

**Local probe, multi-scale control (the most density-robust cell):**

| | d16 | d20 | d25 | d32 | c12 | c16 | c20 |
|---|---|---|---|---|---|---|---|
| mag\_r | +0.08 | −0.34\* | −0.26\* | −0.19\* | +0.32\* | +0.24\* | +0.07 |
| photo\_z | +0.16\* | −0.01 | +0.02 | +0.12\* | +0.22\* | +0.10. | +0.05 |
| smooth\_fraction | +0.10. | −0.00 | +0.01 | +0.13\* | +0.05 | +0.03 | −0.02 |
| stellar\_mass | −0.03 | −0.05 | −0.04 | +0.09. | +0.09. | +0.04 | +0.01 |

**Local probe, matched-radius $\tau$:**

| | d16 | d20 | d25 | d32 | c12 | c16 | c20 |
|---|---|---|---|---|---|---|---|
| mag\_r | +0.09 | +0.04 | +0.11. | +0.13\* | +0.12\* | −0.04 | −0.03 |
| photo\_z | +0.07 | +0.04 | +0.01 | +0.11. | +0.04 | −0.00 | +0.04 |
| smooth\_fraction | +0.16\* | +0.27\* | +0.30\* | +0.24\* | −0.17\* | −0.13\* | −0.13\* |
| stellar\_mass | +0.01 | −0.22\* | −0.18\* | −0.12\* | +0.11. | +0.10. | +0.04 |

## 7. Reading

1. **The colleague's negative association is a within-patch density-profile effect.** The
   multi-scale control alone takes his rank-16 partials to −0.01, −0.01, −0.03 for the three labels
   that had carried it, and his rank-20 mag\_r from −0.24 to −0.13. Fit the probe locally and the
   sign reverses: +0.32 and +0.24 for mag\_r at ranks 12 and 16. A quadratic fit over a 2048-patch
   reads the density profile across that patch; the sealed three controls see only its outer radius.
2. **The decoder field's positive sign is partly the same thing and partly not.** Multi-scale
   control reduces the $d=16$ mag\_r partial from +0.33 to +0.25 and the photo\_z and
   smooth\_fraction partials at every $d$ by a third to a half, and they remain positive and
   significant under the global probe. Under the local probe they mostly vanish for $d \in \{20,
   25\}$ and persist weakly at $d \in \{16, 32\}$.
3. **One regularity appears only when density is removed on both sides:** decoder $d \in \{20, 25,
   32\}$ against mag\_r reads negative under every density-robust combination (global probe with
   multi-scale control −0.17, −0.15, −0.17; `weighted_half` −0.13, −0.11, −0.16; local probe three
   controls −0.24, −0.17, −0.12; local probe multi-scale −0.34, −0.26, −0.19), all at $p<0.05$ and
   most at $p<0.01$. That is the hypothesis's sign. $d=16$ does not show it (+0.08, n.s.), no other
   label shows it, and the matched-radius $\tau$ for the same cells is positive (+0.04 to +0.13).
   A partial that appears only after conditioning on five collinear radii can be a suppression
   artefact as easily as an unmasked effect, and nothing here separates the two.
4. **No design yields a sign that is stable across instrument, label and $d$.** The sealed statistic
   is label-dependent in magnitude and instrument-dependent in sign; the density-robust variants
   are $d$-dependent in sign for the gating label. This is the same conclusion as Supplement 03,
   reached from the real data instead of the fixture: the pre-registered statistic does not
   measure curvature, and neither does any single re-analysis of it.

## 8. Experiment 7: the fixture partial at zero coupling is not zero

Record `notebooks/.cache/09_fixture_probe_decodability_gamma2.jsonl`, sha256
`ce3863f9…2adbb7dd`, verified both sides; wall clock 6,503 s; same generator, seed, anchors and
pipeline as Supplement 03, $\gamma \in \{0.4, 0.6, 0.8\}$ added to its $\{-1, 0, +1\}$. Label
linear in intrinsic coordinates; sealed three-control partial.

| $\gamma$ | $\rho(\text{exact } \|H_{\rm tan}\|, \log r)$ | exact pointwise | exact patch mean | decoder | colleague |
|---|---|---|---|---|---|
| −1.0 | +0.80 | −0.29 | −0.49 | −0.29 | −0.26 |
| 0.0 | +0.51 | −0.00 | +0.04 | +0.01 | +0.19 |
| +0.4 | +0.14 | +0.19 | +0.21 | +0.19 | +0.22 |
| **+0.6** | **+0.05** | **+0.22** | **+0.25** | **+0.25** | **+0.21** |
| +0.8 | −0.16 | +0.26 | +0.26 | +0.28 | +0.28 |
| +1.0 | −0.15 | +0.21 | +0.22 | +0.22 | +0.21 |

Every non-zero cell at the permutation floor ($p \le 4 \times 10^{-4}$). The decoder's own
coupling at $\gamma = 0.6$ is −0.016, the colleague's +0.074.

The curve is not linear in coupling. It sits on a plateau of about +0.21 to +0.28 for coupling
below +0.15, then falls steeply through zero near coupling +0.5 to −0.29 at +0.80. **At zero
coupling the sealed statistic reads +0.22 with exact curvature**, and every instrument column
agrees to within 0.04. On this surface, with this label, higher curvature goes with *better* local
decodability once density is uncoupled. That is the opposite of the hypothesis sign, and it is not
an instrument artefact.

Two consequences.

- Supplement 03's rule ("the partial takes the opposite sign of the coupling") holds only on the
  steep part of the curve. The zero-coupling value is a property of the surface and the label, and
  it is positive here. The real-data readings still lie on the curve: the decoder at coupling −0.60
  read +0.33 (plateau); the colleague at +0.70 read −0.24 (steep part). Neither is a curvature
  finding, for the same reason as before, but the fixture no longer supports reading the decoder's
  positive sign as *only* density.
- The colleague's estimator ranks 0.82, 0.80, 0.77 against the exact truth at $\gamma = 0.4, 0.6,
  0.8$, above Supplement 02's 0.7 bar, with split-half reliability 0.93 to 0.97. Its noiseless FAIL
  at 0.62 is specific to the uniform sampling. The noisy-arm inversion (rank −0.30 at 25 % noise)
  was not re-run and stands as measured.

The ambient-linear null label reads −0.07 to −0.16 near zero coupling and $\approx 0$ at the ends;
the nonlinear label reads 0 to +0.16 with no pattern in $\gamma$. Neither is a clean leakage probe
at this scale, as Supplement 03 found.

Why an intrinsic-linear label would read positive at zero coupling is not resolved here. Local
label variance is one of the three controls, so it is not that alone. A candidate: on this
generator the bumps that carry curvature are also where the stereographic map's stretch is most
uniform within a patch, so a global linear map approximates $a \cdot z$ better there. Testing that
needs the per-anchor arrays the fixture runner does not save.

## 9. What this settles and what it does not

**Settles.** Density enters a globally fit, locally scored linear probe through two opposing
channels whose net sign depends on the label. The colleague's negative curvature–decodability
association on this data is removed by a within-patch density control and reversed by a locally
fit probe. The decoder's positive association survives the same controls for two secondary labels
under the global probe and mostly not under the local one.

**Also settles (Experiment 7).** On the known surface the sealed statistic at zero
density–curvature coupling is +0.22 with exact curvature, not zero. Coupling moves it along a
curve from that plateau to −0.29; the sign is density-driven only where coupling exceeds roughly
+0.5. Neither real-data number can be read as "only density" any longer, and neither can be read
as curvature.

**Does not settle.** Whether the negative mag\_r partial at $d \geq 20$ under density-robust
designs is curvature or suppression. Why the fixture's zero-coupling value is positive. The density-robust variants have not been calibrated on the
known surface (the fixture runner does not save per-anchor arrays; adding that is a code change).
Whether a fixed-radius design with a count control, rather than a mask, behaves differently. The
full-exponent weighting is degenerate and should not be quoted.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 04 — post-hoc, not pre-registered, feeds no verdict. Runner commits `bdd32ff`, `918f4cf`;
density-probe record sha256 `57b4376c9b1404a05834548d83c2a8a9b1e215508a9bec205df2bf8a4cfbeafb`.*
