# 08-DIAGNOSTICS — post-freeze diagnostics for Phase 8

**Plan:** 08-07 · **Status:** Tasks 1 and 4 complete; Tasks 2 and 3 in progress; Task 5 checkpoint
NOT YET PRESENTED.

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

**STATUS: RUNNING.** Numbers pending. This section is a placeholder and must be filled before the
Task 5 checkpoint is presented.

### The claim under test

`subsample.l2_normalize` puts every row on the unit sphere — verified, `norm min/med/max =
1.000000 / 1.000000 / 1.000000` for both modalities. For an exact `d`-dimensional submanifold of the
unit sphere, the mean curvature vector under this milestone's `H = tr_g(II)` convention has a radial
component of exactly `-d`, pointing at the origin. That term carries no information about the
manifold's own shape but enters `‖H‖` in full.

Frozen `‖H‖` medians are 37.19 / 41.41 / 47.03 at `d` = 20 / 25 / 32. Removing a radial term of
magnitude `d` in quadrature leaves 31.36 / 33.02 / 34.46 — a spread of 10% where the raw field's is
26%. If that arithmetic holds, the `‖H‖` field is largely a constant plus a small residual, which
would explain both PU's ~1.5 p95/p05 spread and Phase 8's `realized_h_contrast` of 1.16.

The decision-relevant number is `spearman(‖H‖, ‖H_tan‖)`.

---

## 3. Per-`d` instrument fidelity

**Runner:** `notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d {25,32}` · **Cache:**
`notebooks/.cache/07_plain_decoder_sweep.jsonl`

**STATUS: RUNNING.** Numbers pending. This section is a placeholder and must be filled before the
Task 5 checkpoint is presented.

`INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)` is frozen and **does not change**. It was measured on
analytic fixtures at `d=20` only. The new `d=25` and `d=32` numbers are reported *beside* it, never
merged into it.

**Provenance, stated explicitly so the three are not presented as one sweep:** the `d=20` rows were
measured in Phase 7 (`notebooks/.cache/07_plain_decoder_sweep.jsonl`, 2026-08-25). The `d=25` and
`d=32` rows were measured by plan 08-07 on 2026-09-01, using the same script with an additive `--d`
argument that leaves the default at 20 and alters no fixture, seed, epoch count or `D`.

---

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
