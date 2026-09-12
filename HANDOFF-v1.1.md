# EffDim v1.1 handoff

**Written 2026-09-02, on branch `fixture-validity-audit`, for a reader starting fresh.**

This records what the previous attempt did across sixteen
phases and 398 planning documents. It does not tell you which conclusions to keep.

**How to read the claims here.** Every phase in this milestone froze a decision rule in committed
source before it computed a number, then applied that rule and recorded a verdict string. A verdict
string is the output of a rule, not a fact about galaxies. Where this document says
`ASSOCIATION DETECTED`, read it as "the pre-registered rule returned that token on these inputs."
Several verdicts on the record rest on inputs that later phases showed to be confounded, and the
project chose not to retract them. Section 6 lists the ones you should treat as open.

---

## 1. The question

Duraphe et al. report that foundation models trained on astronomical images from different surveys
converge toward similar representations, measuring 0.4 to 2 percent crossmodal mutual-k-nearest-
neighbour alignment on Legacy Survey data. The v1.1 milestone asks a follow-on question: does that
alignment vary with the local curvature of the representation manifold?

The hypothesis under test predicts a negative association. Regions of higher local mean curvature
should show worse crossmodal alignment.

## 2. The data

`UniverseTBD/pu-embeddings`, config `legacysurvey_dinov3_vitb16`. A 10,000-row subsample, seeded and
row-aligned, pairing HSC and Legacy Survey embeddings of the same objects. Each embedding has 768
dimensions. `subsample.l2_normalize` divides every row by its norm, which places all 10,000 points
on the unit sphere in 768 dimensions. That normalization turns out to matter for curvature (§5.4).

Cached artifacts live in the gitignored `notebooks/.cache/`. The frozen subsample is
`subsample_20260729_a79b3460b838fd0a.npz`.

## 3. What got built

`src/effdim/` ships `compute_dim`, an effective-dimensionality panel from v1.0. The v1.1 milestone
never modified it, by standing rule.

All milestone code sits in `notebooks/pu_manifold/`, a notebook-scoped package of 29 modules
imported relatively. The ones a fresh reader needs first:

| Module | What it does |
|---|---|
| `subsample.py` | Loads and L2-normalizes the row-aligned pair |
| `cae.py` | `PlainAutoEncoder`, `ChartAutoEncoder`, training loops |
| `decoder_curvature.py` | `plain_decoder_curvature`, autodiff through a fitted decoder, convention `H = tr_g(II)` |
| `curvature_probe.py` | `centroid_mean_curvature` (training-free, point-cloud), `local_density_weights` |
| `mknn.py` | Mutual-k-nearest-neighbour alignment, computed in ambient 768-D |
| `cka.py` | Centered kernel alignment, unbiased HSIC, also ambient |
| `density_stratified_null.py` | Within-stratum permutation nulls |
| `varying_ii_controls.py`, `synthetic_controls.py`, `analytic_param.py` | Analytic fixtures with known closed-form curvature |
| `geodesic_graph.py` | k-NN graph construction and component read-out |

22 notebooks carry their outputs and 49 diagnostic runner scripts sit in `notebooks/diagnostics/`.

A standing project rule requires that every new manifold-learning model ship a Swiss roll notebook
importing the model code unedited, testing it on a manifold whose answer is known. Eleven such
notebooks exist. That rule caught real defects more than once (§7).

## 4. What each phase did

### Phases 1 and 2: reconstruct and gate

Phase 1 loaded the subsample and fit Isomap, checking k-NN graph connectivity and stability across
`n_neighbors` values.

Phase 2 audited the classical-MDS eigenspectrum by hand behind a pre-registered gate. It recorded
`GATE_VERDICT = FAIL` at `r = 0.052419`, `m = 0.412071`, and froze `d_frozen = 5`. The FAIL means
the Isomap coordinates are not trustworthy as a Euclidean embedding. The frozen `d = 5` should not
be inherited downstream; later phases used `d` in `{20, 25, 32}` instead.

Phase 02.1 responded by selecting a graph-native representation, Ollivier-Ricci curvature, on the
grounds that it needs no embedding. The pre-registered falsifier fired and overturned the
coordinate-producing alternative. Read the caveat: the distance criterion that ranking used is
itself open to question, and the continuum-limit claim behind Ollivier-Ricci (arXiv:2307.02378) was
recorded as cited rather than verified.

### Phase 2 stage: four model families, four FAILs

The project tried four ways to obtain a smooth map whose derivatives give curvature.

- **Chart Auto-Encoder** (Schonsheck, Chen, Lai). Phase 02.2 recorded `CAE_VERDICT = FAIL`:
  geodesic distortion 0.296981 against a 0.15 bar, held-out reconstruction margin 3.586350 against
  a 0.90 bar. The chart-transition cycle test passed.
- **Topological Auto-Encoder** (Moor et al.). Phase 02.4 recorded `TOPOAE_VERDICT = FAIL` on both
  global-scoped gates; the local-scoped gate passed. TopoAE never produced a curvature field and
  handed nothing downstream.
- **Local curvature feasibility** (Phase 02.5) recorded `CURVATURE_VERDICT = FAIL` under a 5-seed
  amendment, then paused at a checkpoint that stayed open.
- **Decoder substrate screening** (Phase 02.6) halted, replanned onto persistent-homology
  agreement, and finished promoting no substrate and eliminating none. Its ranking axis carried two
  named confounds.

On 2026-08-12 the developer put the Phase 2 stage on hold and chose the CAE by decision. Phase 3
therefore started on a deliberate override of its own PASS precondition. No PASS existed.

### Phase 3 and 03.1: the CAE decoder field, and its repair

Phase 3 differentiated through the CAE chart decoder to get a per-point mean-curvature field. The
field failed its own three-seed spread check: a 52-fold range, with two of three fields piecewise
constant on collapsed metrics. The training objective constrains no decoder derivative at any order,
and `cond(g)` is scale-invariant, so it cannot detect that collapse.

Phase 03.1 added scale and Christoffel priors. The `scale` prior repaired the metric
(`log10_det_g` moved from -83.9 to +0.037) and moved ordering part of the way (`rho` -0.122 to
+0.116), inconsistently across seeds. The `christoffel` prior did not demonstrate its own mechanism.
The phase closed as necessary but not sufficient.

### Phases 4, 5, 6: three attempts at a curvature-conditioned effect

- **Phase 4** partitioned into high and low curvature regions and compared crossmodal MKNN.
  `VERDICT_RULE` returned HOLDS at every `k`. The same document records that the split correlates
  0.82 with density and that the gap is mostly a region-size artifact. Treat HOLDS as a rule output,
  not as a curvature finding.
- **Phase 5** tested curvature-conditioned linear decodability. Verdict `SPLIT ACROSS SEEDS`,
  2 of 3.
- **Phase 6** repeated Phase 5's design with the training-free point-cloud curvature estimator at
  `K_FROZEN = 500` in place of the decoder field. Verdict `NO DETECTABLE RELATIONSHIP`. The same
  3,000 residuals with only the curvature field swapped produced a different verdict. The two fields
  correlate below |0.12| with inconsistent signs.

That instrument dependence is one of the milestone's clearest results and one of its least resolved.
The developer ruled on 2026-08-30 that it should not be framed as a threat to the milestone, on the
reasoning that finding one tool that estimates true mean curvature well makes disagreement with a
worse tool uninformative. Whether the plain-AE decoder is that tool rests on §5.3.

### Phase 7 and 07.1: the plain autoencoder, and density

Phase 7 dropped the CAE and fit `cae.PlainAutoEncoder` at `d` in `{20, 25, 32}`, differentiating
the decoder alone. Held-out `var_explained` reached 0.98194 / 0.98432 / 0.98647.

Headline statistic `spearman(‖H‖, MKNN)` measured **-0.112181 / -0.127891 / -0.023726**, negative at
every `d`, matching the hypothesis direction. The frozen rule returned `ASSOCIATION DETECTED`.

Phase 7 also measured what undercuts that. Local k-NN density correlates with `‖H‖` at
**+0.4281 / +0.3150 / +0.0118** and with MKNN at **-0.2121**. After residualizing both sides against
density, the partial drops to **-0.024189 / -0.065835 / -0.021719**. Density accounts for roughly
78, 49 and 8 percent of the raw association at the three `d`.

Phase 07.1 gave the partial its own calibrated null, permuting `h` and `m` inside quantile density
strata rather than borrowing the raw statistic's thresholds. Verdict `SURVIVES AT SUBSET OF d`,
with `{20: False, 25: True, 32: False}`, and `SEED STABLE AT d=25` across three seeds. Both verdicts
carry a flag saying no human has read and ratified them.

### Phase 8: the CKA replication

Phase 8 swapped the alignment probe from MKNN to centered kernel alignment and ran the same
curvature conditioning on density-residualized tertiles. Production runs finished 2026-08-30 against
freeze `f023c8fa`, 211 rows.

- Sweep: `d=20` and `d=25` clear at every stratum count, `d=32` does not clear, 3 of 3 seeds clear at
  `d=25`.
- Negative control: 1 clearance in 60 cells, 0.017 against a nominal 0.05.
- Positive control: **`POSITIVE CONTROL INVALID`**. The `magnitude = 0.0` no-injection anchor cleared
  its null at all six cells, so the detection-floor power curve the phase intended does not exist.
  **Phase 8 has no power estimate.**

## 5. Post-freeze diagnostics (plan 08-07, 2026-09-01 to 09-02)

Four diagnostics ran after the Phase 8 freeze. All four gate nothing, and none changed a verdict.

### 5.1 Density measured on the manifold instead of in ambient space

A curved manifold's ambient chord runs shorter than its geodesic, more so where curvature is higher,
so ambient k-NN density could read high where curvature is high and the density control could be
removing real signal.

The measurement rejects that. `spearman(geodesic/ambient radius ratio, ‖H‖)` reads
**+0.023 / +0.024 / +0.013**, near zero at all three `d`. Geodesic density gives the same confound
(+0.409 against ambient's +0.428 at `d=20`), and the geodesic partial is larger in magnitude than
the ambient one, meaning a switch would flatter the result. The developer ratified keeping ambient
density.

One structural fact worth carrying forward: `local_density_weights` fixes `d = 20` in its volume
exponent, which makes density a strictly monotone function of the k-NN radius. Measured
`spearman(density, -r) = 0.9999999999999999`. Every rank-based control in this milestone therefore
controls for raw radius, and the `d=20` exponent cannot affect any of them.

### 5.2 Exact permutation p-values

The frozen record carried no p for any rho, by design: the per-point MKNN statistic is `j/k` for
integer `j` and takes 15 distinct values at `k=20` across 10,000 points, so `spearmanr`'s asymptotic
p assumes no ties and does not apply.

Plan 08-07 computed exact permutation p under the null matched to each statistic, with two
self-validations against the sealed record passing first.

| `d` | raw rho | p | partial | p at S = 10 / 20 / 50 |
|---|---|---|---|---|
| 20 | -0.112181 | < 1e-5 | -0.024189 | 0.057 / 0.070 / 0.069 |
| 25 | -0.127891 | < 1e-5 | -0.065835 | < 5e-5 at all three |
| 32 | -0.023726 | 0.0090 | -0.021719 | 0.099 / 0.171 / 0.168 |

Only `d=25` survives density control, reproducing 07.1's verdict by a second route. At n = 10,000 a
raw |rho| of 0.11 clears significance without effort, so the raw p-values carry little weight.

### 5.3 Instrument fidelity per `d`

`INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)` was measured on analytic fixtures at `d=20` alone. Plan
08-07 measured `d=25`:

| fixture | `D` | `d=20` | `d=25` |
|---|---|---|---|
| cubic | 28 | +0.8688 | +0.7760 |
| cubic | 768 | +0.5253 | **+0.1713** |
| ridge | 28 | +0.9823 | +0.9637 |
| ridge | 768 | +0.9745 | +0.9698 |

Fidelity spans (0.53, 0.98) at `d=20` and (0.17, 0.97) at `d=25`. The ceiling holds; the floor drops
threefold, all of it in one cell. Nothing on the record says which fixture the PU manifold resembles.

`d=32` has **no fidelity measurement**. The sweep aborts:
`ValueError: rotate_and_pad: D=28 must be >= local width m=33`. The ambient grid is the hard literal
`(28, 768)`, graph fixtures have local width `m = d + 1`, and `rotate_and_pad` requires `D >= m`, so
the small-ambient arm caps at `d=27`. The developer ratified recording that as a fixture-design
finding and deferring the measurement. At `d=32`, a dying instrument and a vanishing effect remain
indistinguishable.

### 5.4 The unit sphere and radial curvature

Because `l2_normalize` puts every row on the unit sphere, a `d`-dimensional submanifold carries a
radial mean-curvature component of exactly `-d` under the `tr_g(II)` convention. That term says
nothing about the manifold's own shape and enters `‖H‖` in full.

The measurement confirms the geometry. Decoder image norms sit at 0.993 / 0.996 / 0.996, and the
measured radial component lands within 3.5 percent of `-d` at every `d`. Predicted `‖H_tan‖` median
from `sqrt(‖H‖² - d²)` was 31.36 at `d=20`; measured 31.37.

Substituting the sphere-tangential residual for `‖H‖` does something different at each `d`:

| `d` | partial with `‖H‖` | partial with `‖H_tan‖` | effect |
|---|---|---|---|
| 20 | -0.02258 | -0.02525 | strengthens 1.12x |
| 25 | **-0.06591** | **-0.02326** | **collapses 2.8x** |
| 32 | -0.02686 | +0.05639 | **sign flips** |

`d=25` is the only `d` surviving density control and the only partial at p < 5e-5. Removing the
radial term drops it into the range of the two `d` that fail, while the raw rho does not move
(-0.1274 to -0.1278).

Plan 08-07 had pre-registered `spearman(‖H‖, ‖H_tan‖)` as the decision rule, with a value near 1
meaning the radial term acts as a constant offset. It measured 0.961 / 0.918 / 0.888, which passes
that rule while the partial says otherwise. A rank correlation is the wrong sufficient statistic for
a partial correlation: two fields agreeing on 90-plus percent of the ranking still disagree about a
partial of magnitude 0.02 to 0.07, because that quantity lives in the residual they do not share.

The developer ruled this a one-paragraph limitation rather than its own phase. Read the ruling as
settling how much weight the finding carries, not whether it happened. The permutation p for the
tangential partial at `d=25`, which would separate "collapsed to noise" from "collapsed but still
significant", remains unmeasured.

## 6. What a fresh reader should treat as open

1. **Whether the curvature-alignment effect exists.** The strongest surviving number is `d=25`'s
   density-controlled partial of -0.0659 at p < 5e-5. It does not survive substituting the
   sphere-tangential curvature field. `d=20` and `d=32` do not clear density control. No phase in
   this milestone has produced a positive curvature-alignment finding that is free of a named
   confound.
2. **Which curvature estimator to trust.** Phase 6 showed the verdict changes when only the
   estimator changes. The plain-AE decoder scored 0.53 to 0.99 against analytic fixtures at `d=20`
   and 0.17 to 0.97 at `d=25`. High reconstruction does not predict fidelity: `cubic@768` reached
   99.70 percent reconstruction at rho 0.525, `ridge@768` reached 99.88 percent at rho 0.975.
3. **Whether Phase 8's result means anything without a power estimate.** The positive control is
   invalid. The negative control gives a false-positive rate of 0.017 in 60 cells, which answers a
   different question.
4. **Whether the phase structure earned its cost.** 398 planning documents, 16 phases, four model
   families rejected, three phases halted or put on hold. The pre-registration discipline caught
   real problems (the density confound, the metric collapse, the invalid anchor). It also produced
   verdict strings that read stronger than the evidence supports, which every phase then spent
   prose walking back.
5. **The `d` question.** PU reconstruction never plateaus, so no single bottleneck width is
   defensible. The milestone swept `{20, 25, 32}` and got different answers at each.

## 7. Practices worth keeping, whatever you rebuild

- **Swiss roll checks.** Test every new manifold model on a 2-D sheet curled into 3-D, importing the
  model code unedited. A FAIL on real data has two causes, no structure or a broken implementation,
  and only a manifold with a known answer separates them. This caught an unfaithful translation, a
  check with no baseline, and a score confounding model with estimator.
- **Freeze the decision rule before computing the number**, and prove the freeze commit is a strict
  git ancestor of the run commit. Check `git rev-list --count >= 1` alongside
  `merge-base --is-ancestor`, since a commit is its own ancestor.
- **Match the null to the statistic.** A plain permutation p for a density-controlled partial
  ignores the structure the statistic exists to control for.
- **Report `p < 1/(N+1)`, never `=`,** when no permutation draw reaches the observed value.
- **Measure acceptance criteria at production dimensionality.** One phase passed every check at toy
  scale and failed at `d=20` with three defects.
- **Calibrate a decision rule on the quantity it decides.** §5.4 shows a rank-correlation rule
  passing a result that the partial correlation it was standing in for contradicts.

## 8. Papers

**Load-bearing, used directly:**

- Duraphe et al., *The Platonic Universe*, NeurIPS 2025 ML4PS, [arXiv:2509.19453](https://arxiv.org/abs/2509.19453). The origin experiment. Supplies the MKNN alignment probe and the 0.4 to 2 percent crossmodal figure this milestone conditions on.
- Schonsheck, Chen, Lai, *Chart Auto-Encoders for Manifold Structured Data*, [arXiv:1912.10094](https://arxiv.org/abs/1912.10094). The CAE, tested in 02.2 and used in Phase 3.
- Moor, Horn, Rieck, Borgwardt, *Topological Autoencoders*, ICML 2020, [arXiv:1906.00722](https://arxiv.org/abs/1906.00722). Tested in 02.4, FAILed, produced no curvature field.
- Kornblith et al., 2019. Centered kernel alignment and its invariances, the Phase 8 probe.
- Tenenbaum, de Silva, Langford, 2000. Isomap, Phase 1.

**Cited for method or grounding, not independently verified:**

- Ollivier-Ricci continuum-limit claim, [arXiv:2307.02378](https://arxiv.org/abs/2307.02378). The record flags this grounding as thinner than 02.1 stated. Read the paper before leaning on it.
- Aamari, Levrard, *Non-Asymptotic Rates for Manifold, Tangent Space, and Curvature Estimation*, [arXiv:1705.00989](https://arxiv.org/abs/1705.00989).
- *Efficient Mean Curvature Computation on High-Dimensional Data Manifolds*, [arXiv:2606.06329](https://arxiv.org/abs/2606.06329).
- Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh, *Confidence sets for persistence diagrams*, Ann. Statist. 42(6), [arXiv:1303.7117](https://arxiv.org/abs/1303.7117).
- *Gaussian curvature in codimension > 1*, [arXiv:1312.2554](https://arxiv.org/abs/1312.2554). Grounds why Gaussian curvature is not canonically defined above codimension 1.
- Curvature-based geometric data analysis roadmap, [arXiv:2510.22599](https://arxiv.org/abs/2510.22599), and intrinsic dimension survey, [arXiv:2509.15517](https://arxiv.org/abs/2509.15517). Both recorded as under-extracted.
- [arXiv:2511.02873](https://arxiv.org/abs/2511.02873), on bias increasing with dimension in curvature estimation.

**Considered and not adopted:**

- Complexity Decoupled Chart Autoencoders, [arXiv:2208.10570](https://arxiv.org/abs/2208.10570).
- RTD-AE, ICLR 2023, [arXiv:2302.00136](https://arxiv.org/abs/2302.00136), with the H1 extension at [arXiv:2502.20215](https://arxiv.org/abs/2502.20215).
- GRAE, [arXiv:2007.07142](https://arxiv.org/abs/2007.07142).
- Neuc-MDS, NeurIPS 2024, [arXiv:2411.10889](https://arxiv.org/abs/2411.10889).
- Conformal regularization of decoders for scalar curvature, [arXiv:2508.20413](https://arxiv.org/abs/2508.20413).

## 9. Where the artifacts are

| What | Where |
|---|---|
| Phase records, one directory per phase | `.planning/phases/` |
| Current position and history | `.planning/STATE.md` |
| Phase 8 post-freeze diagnostics | `.planning/phases/08-curvature-conditioned-cka-alignment/08-DIAGNOSTICS.md` |
| Concept-by-concept inventory of Phases 7, 07.1, 8 | `.planning/07-08-CONCEPT-INVENTORY.md` (untracked) |
| Frozen numeric records | `notebooks/.cache/*.jsonl` (gitignored) |
| Milestone code | `notebooks/pu_manifold/` |
| Diagnostic runners | `notebooks/diagnostics/` |
| Standing project rules | `CLAUDE.md` |

Phase 8 has one plan outstanding, `08-06`, covering the reporting notebook, `08-FINDINGS.md` and
`08-VALIDATION.md`. Its `must_haves` were amended on 2026-09-02 to require the positive control's
invalidity in the verdict sentence.
