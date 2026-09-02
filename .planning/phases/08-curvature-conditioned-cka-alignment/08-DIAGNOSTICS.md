# 08-DIAGNOSTICS — post-freeze diagnostics for Phase 8

**Plan:** 08-07 · **Status:** Tasks 1, 2 and 4 complete. Task 3 PARTIAL — `d=25` measured,
`d=32` halted on the pre-registered stop condition (§3). Task 5 checkpoint NOT YET PRESENTED;
all five decisions stand at `NOT RATIFIED`.

Every number in this document is **non-gating**. No verdict in `notebooks/.cache/08_cka_alignment.jsonl`
was recomputed, revised or reinterpreted; no row was appended to it (verified: 66 positive-control,
66 negative-control, 79 sweep rows, unchanged). No constant in `notebooks/pu_manifold/` was edited.
Phase 7's `ASSOCIATION DETECTED`, 07.1's `SURVIVES AT SUBSET OF d` and `SEED STABLE AT d=25`, and
Phase 8's per-`d` and per-seed verdicts all stand exactly as measured.

---

## 1. Density control — ambient versus graph-geodesic

**Runner:** `notebooks/diagnostics/08_density_control_diagnostic_run.py` · **Cache:**
`notebooks/.cache/08_density_control_diagnostic.jsonl` · **Commit:** `294c5dd`

### The hypothesis under test

The density field is a k-NN estimate in ambient 768-D LegacySurvey space. A curved manifold's
ambient chord runs shorter than its geodesic, and more so where curvature is higher, so ambient
density could read artificially high in high-curvature regions. If that were happening, controlling
on ambient density would be removing real curvature signal rather than a nuisance covariate, and the
partial rho would be an underestimate.

### What was measured

Ambient radius to the 30th neighbour, against a graph-geodesic radius to the 30th neighbour on the
symmetric k-NN graph at Phase 2's frozen `k* = 15`, Dijkstra in source chunks. Graph read-out:
`n_components = 1`, `largest_size = 10000`, `dropped_fraction = 0.000000` — connected, nothing
dropped.

| `d` | `rho(density_ambient, ‖H‖)` | `rho(density_geodesic, ‖H‖)` | `rho(geo/amb ratio, ‖H‖)` |
|----|----|----|----|
| 20 | +0.428088 | +0.408753 | **+0.023405** |
| 25 | +0.315019 | +0.301114 | **+0.024331** |
| 32 | +0.011798 | +0.021981 | **+0.012516** |

| `d` | raw `rho(‖H‖, MKNN)` | partial \| ambient | partial \| geodesic | partial \| both |
|----|----|----|----|----|
| 20 | -0.112181 | -0.024189 | -0.044764 | -0.025698 |
| 25 | -0.127891 | -0.065835 | -0.079767 | -0.067087 |
| 32 | -0.023726 | -0.021719 | -0.020176 | -0.024028 |

Supporting: `spearman(density_ambient, density_geodesic) = 0.940034`; geodesic/ambient radius ratio
p05/p50/p95 = 1.0664 / 1.5412 / 1.6389; `spearman(density_ambient, MKNN) = -0.212148`.

### Reading

**The hypothesis is rejected.** The contamination it predicts would show up as a correlation between
the geodesic/ambient radius ratio and `‖H‖` — a chord that shortens where curvature is high. That
correlation is +0.023 / +0.024 / +0.013, indistinguishable from zero at all three `d`. The geodesic
metric gives the *same* confound, not a smaller one: the density-curvature correlation barely moves
(+0.428 to +0.409 at `d=20`), and the geodesic partial is *larger* in magnitude than the ambient one,
which is the direction that would flatter the result rather than protect it.

The `DENSITY_FIELD_D = 20` exponent and the gamma-function ball volume cannot affect any of this.
Density is a strictly decreasing monotone function of radius at fixed `d` — measured
`spearman(density_ambient, -r_ambient) = 0.9999999999999999` — so for any rank-based statistic the
density ranks are exactly the reverse radius ranks, and the volume constant cancels.

Note the `d=32` row separately: `rho(density, ‖H‖) = +0.0118` is **not significant**
(p = 0.121, §4). At `d=32` there is no density-curvature coupling to control for, which is why the
partial barely moves there — `d=32`'s null is about the effect, not about the control.

**Verdict impact: changes no verdict.** The ambient density field stays as frozen. Recommendation carried to
checkpoint (a): keep it. MKNN and CKA are both computed in ambient 768-D space
(`mknn._membership_matrix` uses brute-force `NearestNeighbors` on raw arrays; `cka.linear_gram` is
`X @ X.T`), so ambient density is the matched covariate. Adopting a manifold-metric density would
change `DENSITY_INPUT`, which sits in `cka._REQUIRED_CONSTANTS` — a D8-22 pre-registration breach
costing a fresh pre-registration, a ~20 h Phase 8 re-run, and Phase 7 and 07.1 re-runs to stay
comparable — in exchange for a control the evidence says is no better.

---

## 2. Radial curvature decomposition

**Runner:** `notebooks/diagnostics/08_radial_curvature_decomposition_run.py` · **Cache:**
`notebooks/.cache/08_radial_curvature_decomposition.jsonl`

### The claim under test

`subsample.l2_normalize` puts every row on the unit sphere — verified, `norm min/med/max =
1.000000 / 1.000000 / 1.000000`. For an exact `d`-dimensional submanifold of the unit sphere, the
mean curvature vector under this milestone's `H = tr_g(II)` convention has a radial component of
exactly `-d`, pointing at the origin. That term carries no information about the manifold's own
shape but enters `||H||` in full.

### Was the premise sound

Yes, and precisely so. The decoder image sits on the sphere to within half a percent
(`||F(z)||` p50 = 0.9933 / 0.9956 / 0.9957), and the measured radial component lands within 3.5% of
the exact `-d` at every `d`. The re-fit reproduces the frozen field almost exactly
(`spearman(refit ||H||, frozen) >= 0.9994`, `var_explained` 0.98192 / 0.98433 / 0.98647 against
Phase 7's 0.98194 / 0.98432 / 0.98647), so nothing below is re-fit noise.

| `d` | `||F(z)||` p50 | `H_rad` med | exact `-d` | ratio | `||H||` med | `||H_tan||` med | spread `||H||` | spread `||H_tan||` | `rho(||H||,||H_tan||)` | refit-vs-frozen |
|----|----|----|----|----|----|----|----|----|----|----|
| 20 | 0.993341 | -19.7660 | -20 | 0.9883 | 37.2332 | 31.3674 | 1.4116 | 1.5028 | **0.960736** | 0.9994 |
| 25 | 0.995561 | -25.5600 | -25 | 1.0224 | 41.4236 | 32.3893 | 1.3111 | 1.3782 | **0.917973** | 0.9999 |
| 32 | 0.995686 | -33.1042 | -32 | 1.0345 | 46.9977 | 33.2195 | 1.2267 | 1.2691 | **0.887753** | 0.9995 |

The pre-plan arithmetic predicted `||H_tan||` medians of 31.36 / 33.02 / 34.46 from
`sqrt(||H||^2 - d^2)`. Measured: 31.37 / 32.39 / 33.22. The `d=20` prediction is exact; the two
larger `d` come in slightly low, consistent with `H_rad` exceeding `-d` there.

### What changes when the tangential field is substituted

| `d` | `rho(||H||,MKNN)` | `rho(||H_tan||,MKNN)` | partial `||H||` | partial `||H_tan||` | change in the partial |
|----|----|----|----|----|----|
| 20 | -0.111430 | -0.140801 | -0.022580 | -0.025253 | **strengthens 1.12x** |
| 25 | -0.127416 | -0.127796 | -0.065909 | -0.023256 | **collapses 2.8x** |
| 32 | -0.029793 | +0.017495 | -0.026858 | +0.056385 | **sign flips** |

### Reading — this one does not resolve cleanly, and the checkpoint must decide it

**`spearman(||H||, ||H_tan||)` is 0.961 / 0.918 / 0.888 — high, and high is not sufficient.** Plan
08-07 named this quantity as "the single most decision-relevant number in this task", with the rule
that a value near 1 makes the radial term a constant offset and the whole matter a one-paragraph
limitation. **That rule would have passed this result, and the partial says it should not.** The
rank correlation is the wrong sufficient statistic: three fields that agree on 90+% of the ranking
can still disagree about a partial correlation of magnitude 0.02-0.07, because that partial lives
in exactly the residual the two fields do not share.

What the second table shows is that substituting the tangential field does something **different at
every `d`**, and the differences are not small relative to the effects being measured:

- At `d=20` the partial strengthens slightly. Both values are non-significant (§4: p ~ 0.06).
- At `d=25` — **the only `d` that survives density control, and the number §4 puts at p < 5e-5** —
  the partial collapses from -0.0659 to -0.0233, into the same range as the two `d` that fail. The
  raw `rho` meanwhile does not move at all (-0.1274 to -0.1278), so this is specific to the
  density-controlled statistic.
- At `d=32` the partial inverts sign, from -0.0269 to +0.0564, and the tangential value is larger
  in magnitude than any raw partial on the record.

Two readings are available and this document does not choose between them:

1. The surviving `d=25` signal is carried substantially by the radial component — a term fixed by
   the sphere embedding at `-d`, carrying no information about the manifold's own shape. On this
   reading the milestone's one surviving result is materially an artifact of L2 normalization.
2. `H_rad`'s deviation from exactly `-d` encodes something real — local intrinsic dimension, or
   decoder image fidelity — that the tangential projection discards along with the constant. Note
   `H_rad` is not constant: its p05/p95 at `d=20` is -23.74 / -16.60, a range of 7.1 around a
   median of -19.77.

**Verdict impact: changes no verdict**, because this runner gates nothing and every Phase 7, 07.1
and 8 verdict was computed on the frozen `||H||` field, which is untouched. But it bears directly on
what those verdicts *mean*, which is a different question and the one the checkpoint answers.

**Escalation, per plan 08-07's own terms.** The plan states that if the decomposition shows the
`||H||` field means something different from what Phases 7 and 8 assumed, "that is a finding large
enough to warrant its own phase — **not** something to absorb into Phase 8's write-up." The `d=25`
collapse and the `d=32` sign flip meet that description. Option (b)-as-its-own-phase is live.

**What would sharpen the decision, not yet run:** an exact permutation p for
`partial_htan_mknn` at `d=25` under the same within-density-stratum null §4 used. The Task 4 runner
can be pointed at the tangential field. That is scope beyond 08-07 as written and is not being taken
unilaterally — it is put to the developer at the checkpoint.

## 3. Per-`d` instrument fidelity — PARTIAL, `d=32` HALTED

**Runner:** `notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d 25` · **Cache:**
`notebooks/.cache/07_plain_decoder_sweep.jsonl` (pre-merge copy kept at `.jsonl.pre0807`)

`INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)` is frozen and **does not change**. It was measured on
analytic fixtures at `d=20` only. The `d=25` numbers below are reported *beside* it, never merged
into it.

**Provenance, stated so the rows are not presented as one sweep.** The four `d=20` rows were
measured in Phase 7 on 2026-08-25 and are byte-identical after the merge (asserted before and after
writing). The four `d=25` rows were measured by plan 08-07 on 2026-09-02, by the same script with an
additive `--d` argument — the diff removes 3 lines, all displaced by the argparse insertion, and
`--d` defaults to 20 so a bare invocation reproduces the sealed behaviour. No fixture, seed, epoch
count, `D` or `K` was altered.

### Decoder-arm Spearman against analytic `H`

| fixture | `D` | `d=20` | `d=25` | change |
|----|----|----|----|----|
| cubic | 28  | +0.8688 | +0.7760 | -0.093 |
| cubic | 768 | +0.5253 | **+0.1713** | **-0.354** |
| ridge | 28  | +0.9823 | +0.9637 | -0.019 |
| ridge | 768 | +0.9745 | +0.9698 | -0.005 |

Supporting columns at `d=25`: reconstruction 99.92 / 99.64 / 99.92 / 99.78%; `cond(g)` median
2.76 / 8.38 / 2.23 / 3.18; decoder magnitude ratio 1.05 / 1.46 / 0.96 / 0.99; the point-cloud arm
on the identical clouds reads +0.5894 (cubic) and +0.3676 (ridge). The decoder arm beats the
point-cloud arm on rank in 3 of 4 cells, as at `d=20`.

### Reading

**Fidelity spans (0.53, 0.98) at `d=20` and (0.17, 0.97) at `d=25`.** The ceiling is intact — `ridge`
at `D=768`, the cell closest to PU's actual ambient dimension on the better-behaved surface, reads
+0.9698 against +0.9745, a change of half a percent. The floor drops by a factor of three, and the
entire drop is one cell: `cubic` at `D=768`, where `cond(g)` also rises from 7.79 to 8.38 and the
magnitude ratio degrades from 1.46.

So this is **not** "the instrument dies at `d=25`". It is that the instrument's *worst case* at the
ambient dimension PU occupies gets substantially worse with `d`, while its best case does not. The
two fixtures already disagreed by 0.45 at `d=20`, and nothing on the record says which of them the
real PU manifold resembles. That was the point of running two.

**Verdict impact: changes no verdict.** `INSTRUMENT_FIDELITY_RANGE` is untouched, and this sweep
gates nothing.

### `d=32` HALTED — the pre-registered stop condition fired

`--d 32` aborts on its first cell:

```
ValueError: rotate_and_pad: D=28 must be >= local width m=33
```

`D` is the hard-coded literal tuple `(28, 768)` at line 62 of the sweep. A `d`-dimensional graph
fixture has local width `m = d + 1`, so the `D=28` arm admits at most `d=27`. It was chosen at
`d=20`, where `m=21`, as the small-ambient cell. It cannot represent `d=32` at all.

Task 3's action states: *"If a fixture generator turns out to be hard-coded to `d=20`, stop and
report it at the checkpoint rather than generalizing it. Rewriting a fixture to admit a new `d`
changes what the `d=20` numbers mean and would put the existing `INSTRUMENT_FIDELITY_RANGE` in
question — a much larger act than this task, and not one to take unilaterally."* **That condition
fired and the run was stopped. No fixture, no `D` grid and no guard was modified.**

Note what remains available: the `D=768` cell at `d=32` is not blocked — 768 >= 33 comfortably — and
it is the cell that matters, since PU's ambient dimension is 768. The script aborts on `cubic D=28`
before reaching it, so nothing was produced. Obtaining it needs a way to run a subset of the `D`
grid. That would touch no fixture, no seed and no existing number, but it is still a change made to
route around a pre-registered stop, so it is **put to the developer at the checkpoint rather than
taken here**.

**Consequence for the `d=32` reading, unchanged by this plan.** Phase 7 (after density control) and
Phase 8 both lose the signal at `d=32`, and a dying instrument and a vanishing effect remain
indistinguishable there. §1 adds one relevant fact: at `d=32` `rho(density, ||H||)` is +0.0118 with
p = 0.121, so there is no density-curvature coupling to control for and the `d=32` null is about the
effect rather than about the control. That is suggestive, not a substitute for the fixture
measurement.

## 4. Exact permutation p-values

**Runner:** `notebooks/diagnostics/08_exact_permutation_p_run.py` · **Cache:**
`notebooks/.cache/08_exact_permutation_p.jsonl` · **Commit:** `4c378a1`

### Why the record had none

The per-point MKNN statistic is `j/k` for integer `j` and takes at most 21 distinct values across
10,000 points — measured 15 at `HEADLINE_K = 20`. `spearmanr`'s asymptotic p assumes no ties and is
invalid here. The frozen significance route is threshold clearance at
`NULL_QUANTILE_PER_TAIL = 0.975` on both tails, Bonferroni-equivalent to one two-sided 0.05 test,
and the null draws are not stored, so an exact p is not recoverable from the record and had to be
recomputed.

### Self-validation, run before any new number is emitted

1. **Plain arm** — recomputed rho against `07_crossmodal_curvature.jsonl`'s `observed_rho`:
   `d=20` -0.112180716, `d=25` -0.127891147, `d=32` -0.023725706. All match to 9 decimals. **PASS.**
2. **Stratified arm** — replicated null at `PERMUTATION_SEED`, `N = 1000`, against
   `stratified_partial_null`'s sealed band: `null_low` -0.028823 vs sealed -0.027944 (3.1%);
   `null_high` +0.010131 vs sealed +0.009261 (9.4%). **PASS** at the 25% tolerance.

### Nulls, matched to the statistic

Plain unrestricted label permutation (N = 100,000) for raw and diagnostic rho. Within-density-stratum
permutation (N = 20,000), 07.1's own `PERMUTATION_SCHEME_RULE` with `h` and `m` permuted
independently inside each stratum, for the density-controlled partial. A plain permutation p for the
partial would ignore the density structure the statistic exists to control for; the runner refuses
to compute one. Tail direction is read from `SIGNIFICANCE_TAIL_RULE`, not assumed.

`p = (1 + #{null at least as extreme}) / (N + 1)`. Rows where no draw reached the observed value
carry `p_is_floor = true` and are reported as a strict inequality against the floor.

### Results

| `d` | raw `rho(‖H‖, MKNN)` | p (plain, N=100k) | partial \| density | p (S=10) | p (S=20) | p (S=50) |
|----|----|----|----|----|----|----|
| 20 | -0.112181 | **< 1e-5** | -0.024189 | 0.057 | 0.070 | 0.069 |
| 25 | -0.127891 | **< 1e-5** | -0.065835 | **< 5e-5** | **< 5e-5** | **< 5e-5** |
| 32 | -0.023726 | 0.0090 | -0.021719 | 0.099 | 0.171 | 0.168 |

Diagnostic rho, plain null: `rho(density, ‖H‖)` = +0.428088 (< 1e-5) / +0.315019 (< 1e-5) /
**+0.011798 (p = 0.121, not significant)**; `rho(density, MKNN)` = -0.212148 (< 1e-5).
07.1 seed fields at `d=25`: all three p < 1e-5. `MKNN_K_GRID` sensitivity: p < 1e-5 at every `k` for
`d` = 20 and 25.

### Reading

**Only `d=25` survives density control.** This independently reproduces 07.1's
`SURVIVES AT SUBSET OF d` verdict by a different route — a permutation p rather than threshold
clearance — and the two agree. `d=20`'s partial sits just outside 0.05 at every stratum count;
`d=32`'s is not close at any.

The raw p-values are **not the claim**. At `n = 10,000` a raw `|rho|` of 0.11 is trivially
significant, and a tiny raw p is not the confidence that carries the milestone. The load-bearing
numbers are the partial's p under the stratified null and Phase 8's independent CKA clearance.

**Verdict impact: changes no verdict.** Every row carries `gates_nothing: true`.

---

## 5. Developer checkpoint (Task 5)

**NOT YET PRESENTED.** Decisions (a) through (e) are pending the completion of §2 and §3.

- (a) density control — **NOT RATIFIED**
- (b) radial decomposition — **NOT RATIFIED**
- (c) per-`d` instrument fidelity — **NOT RATIFIED**
- (d) the invalid positive control's place in the verdict sentence — **NOT RATIFIED**
- (e) p-value reporting discipline — **NOT RATIFIED**

A standing "keep working" instruction is not an answer to this checkpoint and must not be recorded
as one.
