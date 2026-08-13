# Phase 3: Decoder & Curvature Field - Research

**Researched:** 2026-08-13
**Domain:** Riemannian differential geometry (mean curvature of an immersed submanifold) computed by `torch.func` autodiff through a trained neural chart decoder, at intrinsic dimension `d≈20`, ambient `D=768`.
**Confidence:** HIGH for the mathematics and the existing code's correctness (verified by reading the sealed, tested implementation and its regression tests). MEDIUM for forward-mode `torch.func` composition risk (grounded in official docs plus in-repo precedent, but genuinely untested at this exact composition). LOW/ASSUMED only for the synthetic-control fixture design, which is new code this phase must write.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

D-01 through D-15, verbatim intent (full text in `03-CONTEXT.md`, read there for the complete rationale of each — summarized here for traceability):

- **D-01:** Step-1 statistic is the median `rho_chart` over ≥5 torch seeds, full spread reported, never a single draw.
- **D-02:** Gate is an absolute floor: median `rho_chart > 0.65`. The `0.6712` raw-point centroid baseline is reported as context only and gates nothing. Swiss roll only — PU has no analytic `H`.
- **D-03:** Only the exact `g`-trace path may gate. The Hutchinson randomized-trace estimator stays non-gating (report `K=4,8,16` as context if run at all).
- **D-04:** Under an `n_charts` sweep the floor applies to the best config — passes if median `rho_chart > 0.65` at *any* swept `n_charts`, full sweep table printed, multiple-comparisons caveat named in the read-out.
- **D-05a:** If the roll cannot clear 0.65 at any swept config, Phase 3 stops and reports — a complete outcome, not a failure to work around.
- **D-05:** `n_charts` is an in-scope Phase 3 hyperparameter (explicit user scope ruling overriding 02.3's on-hold status for this knob only). The ruling must be recorded in the plan's own artifacts.
- **D-06:** Nothing from the roll constrains the PU fit. PU gets its own independent `n_charts` sweep selected on model-side diagnostics alone.
- **D-07:** PU-side `n_charts` selection keys on four model-side diagnostics needing no ground truth: `max cond(g)`, argmax chart occupancy, held-out reconstruction error, persistent-homology agreement restricted to H0 and H1 (H2 deliberately excluded — no power analysis exists for it).
- **D-08:** Add a forward-mode toggle to `chart_curvature.py` (`chart_mean_curvature`, `chart_curvature_field`). Default stays reverse (bit-identical to today); forward is opt-in. Scope is `chart_curvature.py` only — not `decoder_curvature.py`, not `derivative_bridge.py`. The `~38×` operation-count reduction is a ceiling, not a measured speedup; the toggle exists so it can be abandoned cheaply.
- **D-09:** Forward/reverse equivalence must be proved (float64 round-off agreement), in the shape of `test_chart_curvature_dxd_solve_matches_explicit_projector`. Existing sphere known-answer test and shape assertions must still pass. `chart_curvature.py` has never been edited before this phase; the reverse path must reproduce `−0.0604` bit-identically.
- **D-10:** Fresh fits only for the PU field. The sealed 02.2 PU fit (`D_CHART=20, L_EMBED=40, N_CHARTS_INIT=16` all-16-survived, SiLU, width 250, 8000/2000 split, `SPLIT_SEED=20260803`) plays no part.
- **D-11:** `chart_dim = 20`, justification restated in this phase's own artifacts (TwoNN 19.5, local-PCA median 25.0, median of 8 geometric estimators 18). `d_frozen = 5` explicitly rejected, not inherited.
- **D-12:** Escalate to a `d` sweep only if the best `d=20` config loses to a matched plain-AE control on held-out reconstruction and PH H0/H1 agreement. Known limitation: reconstruction/topology do not predict curvature quality. The trigger is expected to fire (CAE measured ~5× worse H0 retention than a plain AE at matched dimension using ~8× the parameters).
- **D-13:** PU sweep budget: 3 `n_charts` values × 3 seeds = 9 fits, ~3–5h. Anchor: 1,941.2s/fit training-only at `n_charts=16, d=20, width 250, 8000 train` (predates the forward-mode toggle); curvature at `d=20, D=768` has never been timed. A timing probe before committing the sweep is left to the planner.
- **D-14:** `derivative_bridge` runs at PU scale; WR-01/02/03 close inside Phase 3 (full text and fixes in Common Pitfalls below).
- **D-15:** Gate machinery: the roll floor is the only declared bar. No `PREREGISTRATION.md`, no ratification commit, no git-ancestry proof script, no verdict JSON artifact, no threshold table. The 0.65 floor is written into the plan before anything runs; that is sufficient.

### Claude's Discretion

- Exact `n_charts` values in each sweep (roll and PU). Roll should span the measured monotone range (something like 2/3/5/8); PU picks 3 values under D-13's budget.
- Whether to run a timing probe before committing the PU sweep (D-13 notes it is sensible).
- Deliverable shape — milestone pattern is a runner script under `notebooks/diagnostics/` for the expensive PU grid plus a presentation notebook, and CLAUDE.md **mandates** a new `notebooks/03_swiss_roll_*_check.ipynb`. Additive only: `02.5_swiss_roll_chart_curvature_check.ipynb` is not rewritten.
- How the four D-07 diagnostics combine into a single selection (weighted, lexicographic, or a printed table plus a stated rule).

### Deferred Ideas (OUT OF SCOPE)

- PH H2 agreement as a selection term — excluded until a power analysis exists.
- Resuming Phases 02.3 / 02.5 / 02.7 — remains on hold; D-05 opens `n_charts` to Phase 3 and nothing else.
- The 5 open windows in `.planning/WINDOWS.md`; not Phase 3's work unless it touches one.
- `VERIFICATION.md` missing on Phases 02, 02.1, 02.6 — not Phase 3's work.
- 02.7's two Swiss roll defects — do not block Phase 3.
- Curvature-based partitioning / MKNN — that is Phase 4, not Phase 3.
</user_constraints>

<phase_requirements>
## Phase Requirements

DEC-01..05 and CURV-01..08 are **stale** (written against Isomap coordinates and a global chart) and must be **re-minted**, not re-pointed. This table maps each stale requirement to what changes and what supports it, as input to the planner's re-minting — it does not itself mint the IDs.

| Stale ID | Original text (Isomap-era) | What changes for the CAE chart-decoder / local-scope reality | Research support |
|----------|------------------------------|---------------------------------------------------------------|-------------------|
| DEC-01 | Train a decoder mapping Isomap coordinates to the 768-d embedding, C2-smooth activation throughout | Decoder is per-chart (`cae.ChartAutoEncoder`'s `chart_decoders[i]` + shared `embedding_decoder`), not a single global map from Isomap coordinates; C2-smooth (SiLU) requirement carries over unchanged, enforced by `assert_c2_activation`/`assert_c2_decoder` | `chart_curvature.py` (existing, sealed math); `cae.py` `ChartAutoEncoder` architecture |
| DEC-02 | Verify no ReLU-family activation | Same intent, already implemented as a hard-raising guard (`ZERO_SECOND_DERIVATIVE_ACTIVATIONS` frozenset), not merely "verified" — carries forward unchanged | `chart_curvature.assert_c2_activation`, `decoder_curvature.assert_c2_decoder` |
| DEC-03/04 | Held-out reconstruction, aggregate + per-dimension | Same intent, now scoped per `n_charts` config in the PU sweep (D-07's reconstruction term) rather than a single fit | `cae.reconstruction_stats` (existing) |
| DEC-05 | Reproducible from a recorded torch seed | Same intent, now explicitly multi-seed (D-01 roll: ≥5 seeds; D-13 PU: 3 seeds) rather than one seed | D-01, D-13 |
| CURV-01 | First fundamental form from decoder Jacobian via `torch.func`, batched | Same intent, already implemented (`g = J^T J`, batched via `vmap(jacrev(...))`/`vmap(jacfwd(...))`) — extend with D-08's forward-mode toggle | `chart_curvature.chart_mean_curvature` |
| CURV-02 | Second fundamental form as normal-projected ambient Hessian | Same intent, but the code **never materializes** an explicit `(D,D)` projector or full `II` tensor — it g-traces first, projects via a `d×d` solve (proved equal by `test_chart_curvature_dxd_solve_matches_explicit_projector`). The re-minted text should describe the trace-first-then-project form, not the textbook projector-then-trace form, so a future reader does not "fix" the optimization back to the slow path. | `chart_curvature.chart_mean_curvature` docstring, math below |
| CURV-03 | Mean curvature vector + norm, labelled vector-norm, never Gaussian/principal | Unchanged intent; the codebase additionally pins `CURVATURE_CONVENTION = "trace"` (`H = tr_g(II)`, **not** `H = (1/d)tr_g(II)`) — re-minted text should state the convention explicitly, since this is the single most expensive thing in this file to get wrong (see Common Pitfalls) | `chart_curvature.CURVATURE_CONVENTION`, `02.5-NOTE-randomized-trace.md` |
| CURV-04 | Metric conditioning shown, near-singular points flagged | Unchanged intent, already implemented (`metric_condition_number` returned per point via `torch.linalg.cond(g)`); re-minted text should add D-13's guidance to **flag, not average in**, and name a concrete threshold policy (see Numerical Hazards below) | `chart_curvature.chart_mean_curvature` return dict |
| CURV-05 | Second derivatives verified non-zero and finite away from training nodes | Unchanged intent; extend with D-14's finite-difference bridge as the independent verification instrument, at PU scale, not just a shape/finiteness assertion | `derivative_bridge.py`, D-14 |
| CURV-06 | Compared against same architecture fitted to flat/sphere/saddle at matched `d`, ambient size | Scope now explicitly local: the synthetic control is fitted at PU's actual `d≈20, D=768` (not the frozen `d=5`), and the re-minted text must state plainly that this control **cannot** detect parameterization damage (a synthetic manifold that trains cleanly never reproduces the CAE's own atlas-fragmentation pathology) | See Synthetic Control Manifolds below |
| CURV-07 | Whether curvature is data property or decoder artifact, on CURV-06's evidence | Unchanged intent, but re-minted text must carry the override caveat forward: no PASS exists anywhere upstream in this milestone, so CURV-07's answer is conditioned on that override, never presented as if the parameterization were independently validated | `03-CONTEXT.md` §override, `02.4-FINDINGS.md` |
| CURV-08 | Curvature only evaluated at/near actual Isomap coordinates, never extrapolated | Same intent, "Isomap coordinates" -> "each point's own chart coordinate as assigned by `model.chart_probs(z).argmax(dim=1)`" — already how `chart_curvature_field` works (each row is evaluated in its own assigned chart, never off-manifold) | `chart_curvature.chart_curvature_field` |

**Recommendation for the planner:** re-mint under the SAME `DEC-`/`CURV-` ID namespace (the Traceability table in `REQUIREMENTS.md` already maps `DEC-01..05`/`CURV-01..08` to Phase 3) with rewritten descriptions per the table above, rather than inventing a new prefix — this keeps `REQUIREMENTS.md`'s existing Traceability section structurally valid and satisfies "re-mint, do not re-point." This is a recommendation, not a locked decision; the planner makes the final call.
</phase_requirements>

## Summary

The core mathematics this phase needs is **already implemented, sealed, and tested** in `notebooks/pu_manifold/chart_curvature.py` (02.5-08) — a chart decoder's mean curvature vector `H = tr_g(II)` computed via `torch.func` `jacrev`/`hessian` under `vmap`, with the second fundamental form's `(D,D)`-normal-projector eliminated in favour of a `d×d` solve (proved equal to the textbook form by `test_chart_curvature_dxd_solve_matches_explicit_projector`), chunked at a fixed `VMAP_CHUNK=32` for bit-reproducibility, and guarded against ReLU-family activations that would silently return an identically-zero second fundamental form. This phase's job is **not** to derive new differential geometry — it is to (1) reproduce and then try to beat the sealed `−0.0604` Swiss-roll measurement under a `n_charts` sweep, (2) add a forward-mode toggle to the same module and prove it equivalent to float64 round-off, (3) run the field on PU at a properly-justified `d=20` (not the rejected `d_frozen=5`), (4) close three named code-review defects in the finite-difference bridge, and (5) build new synthetic-control fixtures (flat/sphere/saddle) at PU's actual scale — none of which requires new external libraries.

The single most consequential correctness risk is the **trace-vs-averaged curvature convention**: every external source on this topic (including the "official" differential-geometry literature and any AI-generated code a future editor might consult) states mean curvature as `H = (1/d)tr_g(II)`, while this codebase's `CURVATURE_CONVENTION = "trace"` pins `H = tr_g(II)` — a factor-of-`d` (20, at PU's scale) error that this project has already shipped once and fixed once. The repo's own convention wins; external sources are wrong for this codebase's purposes and must be re-derived, never transcribed, whenever this phase's math touches new code (the synthetic controls especially, since they are new fixtures with no existing regression test).

The second most consequential risk is genuinely unresolved by research and needs an early empirical spike: whether `torch.func.jacfwd(torch.func.jacfwd(f))` — the forward-over-forward Hessian composition D-08's cost table proposes — actually executes on this exact SiLU-MLP decoder architecture without hitting an "unimplemented batching rule" error under `vmap`. Official PyTorch guidance (`M > N` outputs-vs-inputs → prefer `jacfwd`) confirms the *direction* of D-08's proposal is mathematically the textbook-correct choice for a `d=20 → D=768` map, and this exact codebase already runs `vmap`-composed `jvp` successfully in the randomized-trace estimator (a related but not identical composition) — evidence in favour, not proof.

**Primary recommendation:** treat `chart_curvature.py`'s existing implementation as the reference to extend, never to rederive; add the forward-mode toggle as a parallel code path behind an explicit `mode` parameter (default `"reverse"`, bit-identical to today) sharing the same downstream `d×d`-solve/g-trace code, proved equal by a test mirroring `test_chart_curvature_dxd_solve_matches_explicit_projector`'s structure; fix WR-01/02/03 in `derivative_bridge.py` using the exact fixes already specified in `02.6-REVIEW.md`; and build synthetic controls as a **new** phase-scoped module (not an edit to sealed `curvature_probe.py`) that reuses `curvature_probe.graph_mean_curvature` — already tested — for the sphere and saddle constructions.

## Architectural Responsibility Map

This project has no web-app tiers; the equivalent boundary is the milestone's own established layering between differentiable math modules, expensive-computation runners, and presentation notebooks. Mapping capabilities onto that layering surfaces misplacement risk the same way a browser/API split would in a web project — in particular, the risk of putting expensive sweep computation inside a notebook that CLAUDE.md caps at "under two minutes" and "~15 cells."

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Chart-decoder Jacobian/Hessian/g-trace math (D-08, D-09) | `notebooks/pu_manifold/chart_curvature.py` (library) | — | Sealed, tested differentiable math; the phase's one editable module per D-08 |
| Finite-difference bridge fixes (D-14, WR-01/02/03) | `notebooks/pu_manifold/derivative_bridge.py` (library) | — | Independent verification instrument; needs no chart routing |
| Synthetic-control fixture construction (flat/sphere/saddle) | New `notebooks/pu_manifold/synthetic_controls.py` (library) | `curvature_probe.py` (reused, unedited) | New math (embedding + analytic `H`), same "duplicate rather than edit sealed" pattern `decoder_curvature.py` already used against `chart_curvature.py` |
| Swiss-roll gate: n_charts × 5-seed sweep, floor decision (D-01..D-05a) | `notebooks/diagnostics/` runner script | `notebooks/03_swiss_roll_*_check.ipynb` (presentation) | 4 configs × 5 seeds ≈ 20 fits at ~60-90s each exceeds CLAUDE.md's 2-minute sanity-check budget; the milestone's own precedent is "expensive grids are runner scripts" |
| CLAUDE.md-mandated sanity check (single seed, chart_dim=2, ≤15 cells, <2 min) | `notebooks/03_swiss_roll_*_check.ipynb` | — | Distinct deliverable from the gate above — see Architecture Patterns |
| PU curvature field: descriptive `‖H‖` + cond(g) (step 2) | `notebooks/diagnostics/` runner script | presentation notebook | One fit, one seed at `d=20, D=768`; 1,941s/fit training alone, curvature cost unmeasured — budget it as a runner, not an inline notebook cell |
| PU sweep: 3 `n_charts` × 3 seeds, D-07 diagnostics table (steps 2-3) | `notebooks/diagnostics/` runner script | presentation notebook | D-13's own budget (~3-5h) is explicitly framed against the milestone's resumable-runner pattern (`template_benchmark_run.py` precedent) |
| Persistent homology H0/H1 diagnostic (D-07) | `notebooks/pu_manifold/persistence_probe.py` (reused, unedited) | runner script (invoker) | Sealed 02.6 module; no chart-index or smoothness concerns; called from the PU sweep runner |
| Synthetic control fits + curvature (step 4) | `notebooks/diagnostics/` runner script | presentation notebook | 3+ additional CAE fits at `D=768`; same cost profile as the PU sweep |
| Cached expensive-fit artifacts | `notebooks/.cache/` (gitignored) | — | Milestone convention; the Swiss roll sanity notebook must **never** touch this (CLAUDE.md), the sweep runners **may** |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | 2.13.0+cpu [VERIFIED: local `.venv`] | `torch.func` (`jacrev`, `jacfwd`, `hessian`, `vmap`, `jvp`) autodiff of the decoder | Already the sole autodiff engine for every curvature computation in this milestone; no alternative under consideration |
| `numpy` | 2.5.1 [VERIFIED: local `.venv`] | Array plumbing, analytic-fixture construction | Already used throughout `curvature_probe.py`, `persistence_probe.py` |
| `scipy` | 1.18.0 [VERIFIED: local `.venv`] | `scipy.stats.spearmanr` (the roll gate statistic), `scipy.special.gammaln` (density weights, not used this phase) | Already the gate-statistic library (`curvature_probe.spearman_gate_statistic`) |
| `scikit-learn` | 1.9.0 [VERIFIED: local `.venv`] | `NearestNeighbors` (not needed for the decoder arm itself; used by the raw-point baseline this phase reuses read-only) | Already in use, no new surface added |
| `ripser` + `persim` | present, unversioned in `pyproject.toml` [VERIFIED: `import` succeeds in local `.venv`] | Persistent homology (D-07's H0/H1 diagnostic) | Already installed and used by `persistence_probe.py` (02.6); **known reproducibility gap** — not declared in `pyproject.toml` because `CLAUDE.md` bars editing it this milestone; a clean checkout needs `.venv/bin/pip install ripser persim` by hand |

### Supporting

None — every function this phase needs beyond the Core table already exists in a sealed, imported module (`cae.py`, `curvature_probe.py`, `chart_curvature.py`, `decoder_curvature.py`, `derivative_bridge.py`, `persistence_probe.py`).

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.func` autodiff for the FD bridge's comparison Hessian | A third-party finite-difference library (e.g. `numdifftools`) | Rejected by precedent: `02.6-PATTERNS.md` records "No Analog Found" for finite-difference code in this project, and the existing hand-rolled stencil in `derivative_bridge.py` is already calibrated against this project's own decoder shape (`calibrate_fd_step`) — a third-party library would need its own calibration and does not obviously improve on a ~250-line already-tested module |
| `ripser`/`persim` for PH | `giotto-tda`, `gudhi` | Rejected: switching libraries mid-milestone for a non-gating diagnostic reintroduces the exact reproducibility-gap risk `persistence_probe.py`'s docstring already names and accepts for `ripser`/`persim`; no reason to pay that cost twice |

**Installation:**
No new packages required. If a clean checkout is missing `ripser`/`persim` (the one gap that is NOT covered by an existing venv):
```bash
.venv/bin/pip install ripser persim
```

**Version verification:** confirmed directly against the project's own `.venv` (not training-data recall):
```
torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0
```
matching `02-FINDINGS.md`'s own recorded versions for numpy/scipy/scikit-learn, so no drift since Phase 2's fit.

## Package Legitimacy Audit

No new external packages are installed this phase. All packages this phase's code will import (`torch`, `numpy`, `scipy`, `scikit-learn`, `ripser`, `persim`) are already installed in the project's `.venv` and already imported by sealed, tested modules from Phases 02.2/02.5/02.6. No `gsd-tools query package-legitimacy check` run was needed — there is no new package name to check.

| Package | Registry | Status | Disposition |
|---------|----------|--------|-------------|
| torch, numpy, scipy, scikit-learn | PyPI, declared in `pyproject.toml` | Pre-existing, in production use across 6+ prior phases | Approved (no action) |
| ripser, persim | PyPI, **not** declared in `pyproject.toml` (deliberate, per CLAUDE.md's frozen-`pyproject.toml` rule) | Pre-existing in `.venv`, in production use since Phase 02.6 | Approved (no action); the reproducibility gap is a known, already-documented limitation, not a new risk this phase introduces |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  cae.ChartAutoEncoder  (sealed, 02.2 — imported, never    │
                 │  edited)                                                  │
                 │    encode -> chart_coords -> chart_probs (argmax)         │
                 │    chart_decoders[i]  ->  embedding_decoder                │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  z_chart  (batch, chart_dim)
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │ chart_curvature.chart_decoder_map(model, chart_idx)                 │
     │   decode_one: R^chart_dim -> R^out_dim   (the differentiated map)   │
     └───────────────┬───────────────────────────────────────────────────┘
                      │
        reverse mode  │  forward mode (D-08 toggle, NEW this phase)
   J = vmap(jacrev)   │  J = vmap(jacfwd)      <- same (batch,out_dim,chart_dim)
   Hess = vmap(       │  Hess = vmap(
     hessian)         │    jacfwd(jacfwd))     <- same (batch,out_dim,d,d), UNVERIFIED
                      ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │  g = J^T J   (pullback metric, d x d)                              │
     │  g-trace FIRST:  raw = einsum(g_inv, Hess)                         │
     │  normal-project SECOND via d x d solve (never a (D,D) projector)   │
     │  --  chart_mean_curvature (UNCHANGED sealed math, D-09 proves      │
     │      forward == reverse to float64 round-off)                     │
     └───────────┬───────────────────────────────┬─────────────────────┘
                 │ H_vec, H_norm,                 │ metric_condition_number
                 │ per point                      │ per point (cond(g))
                 ▼                                ▼
     ┌───────────────────────┐        ┌─────────────────────────────┐
     │ Step 1: Swiss roll     │        │ Step 3: near-singular flag,  │
     │ gate — n_charts sweep, │        │ NEVER averaged in            │
     │ median rho over        │        └─────────────────────────────┘
     │ >=5 seeds, floor 0.65  │
     └───────────┬───────────┘
                 │ n_charts sweep table (context only, D-06: does not
                 │ constrain PU)
                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │ Step 2/3: PU field.  D-07 selection on 4 diagnostics, NONE needing │
     │ ground truth:  max cond(g)  |  argmax chart occupancy  |           │
     │ held-out reconstruction (cae.reconstruction_stats)  |              │
     │ PH H0/H1 agreement (persistence_probe.readout_matrix, H2 excluded) │
     └───────────┬───────────────────────────────────────────────────────┘
                 │ selected n_charts, 3-seed spread reported
                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │ Step 4: Synthetic control.  SAME architecture+protocol fitted to  │
     │ flat / sphere / saddle at d=20, D=768 (NEW: synthetic_controls.py)│
     │ -- states what PU numbers can/cannot mean; CANNOT detect          │
     │ parameterization damage (a clean fit never reproduces atlas       │
     │ fragmentation)                                                     │
     └───────────────────────────────────────────────────────────────────┘

     ┌───────────────────────────────────────────────────────────────────┐
     │ derivative_bridge.py (D-14): independent finite-difference check   │
     │ on the SAME trained decoder's Jacobian/Hessian, no ground truth    │
     │ needed -- catches bugs shared by both autodiff paths that D-09's   │
     │ forward-vs-reverse comparison structurally cannot see              │
     └───────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
notebooks/pu_manifold/
├── chart_curvature.py         # EDITED (D-08 forward-mode toggle; D-09 equivalence proof)
├── derivative_bridge.py       # EDITED (D-14: WR-01/02/03 fixes; run at PU scale)
├── synthetic_controls.py      # NEW: flat/sphere/saddle fixtures at matched d=20, D=768,
│                               #      reusing curvature_probe.graph_mean_curvature
├── cae.py                     # UNCHANGED, sealed — import only
├── decoder_curvature.py       # UNCHANGED — only relevant if a PlainAutoEncoder-shaped
│                               #             control needs curvature (D-12's escalation)
├── curvature_probe.py         # UNCHANGED, sealed 02.5 — reused: swiss_roll_analytic_H_scaled,
│                               #                          centroid_mean_curvature (raw-point
│                               #                          baseline), graph_mean_curvature
├── persistence_probe.py       # UNCHANGED, sealed 02.6 — reused for D-07's H0/H1 term
└── tests/
    ├── test_curvature_probe.py    # EDITED — chart_curvature tests already live here (not a
    │                               #          separate test_chart_curvature.py); add D-09's
    │                               #          forward/reverse equivalence tests alongside
    │                               #          test_chart_curvature_dxd_solve_matches_...
    ├── test_derivative_bridge.py  # EDITED — WR-01/02/03 regression tests
    └── test_synthetic_controls.py # NEW

notebooks/diagnostics/
├── swiss_roll_curvature_sweep_run.py   # NEW — D-01..D-05a's n_charts x 5-seed gate sweep
├── curvature_field_pu_run.py           # NEW — D-07/D-13's 9-fit PU sweep, resumable
└── synthetic_control_run.py            # NEW — step 4's flat/sphere/saddle fits

notebooks/
├── 03_swiss_roll_chart_curvature_field_check.ipynb  # NEW, CLAUDE.md-mandated: single seed,
│                                                     #      chart_dim=2, <=15 cells, <2 min,
│                                                     #      no gate machinery (D-15)
├── 03_pu_curvature_field.ipynb                      # NEW: presentation of the sweep runner's
│                                                     #      results (steps 2-3)
└── 03_synthetic_control.ipynb                       # NEW: presentation of step 4
```

### Pattern 1: Trace-first-then-project (already implemented — preserve exactly)

**What:** `H = tr_g(II)` is computed by g-tracing the ambient Hessian *before* applying the normal projector, because the g-trace (contracts only the two chart-space indices) and the normal projection (acts only on the ambient index) commute. This avoids ever materializing a `(D,D)` projector matrix or the full `(D,D,d,d)`... actually `(batch, out_dim, chart_dim, chart_dim)` second-fundamental-form tensor.

**When to use:** Any codimension-`(D-d)` curvature computation where `D >> d` (here `D=768, d=20` — a `(D,D)` projector per point in a 32-point chunk is 151 MB float64; a full `II` tensor is another 78 MB).

**Example (verbatim from the sealed module):**
```python
# Source: notebooks/pu_manifold/chart_curvature.py, chart_mean_curvature
J = vmap(jacrev(decode_one))(chunk)     # (batch, out_dim, chart_dim)
Hess = vmap(hessian(decode_one))(chunk)  # (batch, out_dim, chart_dim, chart_dim)

g = torch.einsum("boi,boj->bij", J, J)                       # (batch, chart_dim, chart_dim)
g_inv = torch.linalg.solve(g, eye_d)

raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)               # g-trace FIRST -> (batch, out_dim)
alpha = torch.linalg.solve(g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)).squeeze(-1)
H_vec = raw - torch.einsum("boi,bi->bo", J, alpha)             # normal-project SECOND
```
This is proved — not merely asserted — equal to the textbook projector-then-trace form by `test_chart_curvature_dxd_solve_matches_explicit_projector` (`notebooks/pu_manifold/tests/test_curvature_probe.py:1503`), which reimplements the slow form verbatim and checks agreement to `rtol=1e-9, atol=1e-12`. **Any new curvature code this phase writes (synthetic controls, forward-mode) must reuse this identity, not re-derive its own projector.**

### Pattern 2: `mode` toggle sharing one downstream code path (D-08's recommended shape)

**What:** Add `mode: str = "reverse"` to `chart_mean_curvature`/`chart_curvature_field`. Dispatch only the Jacobian/Hessian construction on `mode`; everything downstream (the `g`-trace, the `d×d` solve, the return dict) stays one shared code path, so equivalence reduces to "does `(J, Hess)` agree between modes," not "do two independently-written functions agree."

```python
# Recommended shape, not existing code -- grounded in the module's own D-09-cited precedent
# (derivative_bridge.reduce_to_H_vec mirrors chart_mean_curvature's algebra rather than
# re-deriving it, and is pinned equal by test_reduce_to_H_vec_pins_plain_decoder_curvature)
def _jacobian_hessian(decode_one, chunk, mode: str):
    if mode == "reverse":
        J = vmap(jacrev(decode_one))(chunk)
        Hess = vmap(hessian(decode_one))(chunk)          # hessian == jacfwd(jacrev(f))
    elif mode == "forward":
        J = vmap(jacfwd(decode_one))(chunk)
        Hess = vmap(jacfwd(jacfwd(decode_one)))(chunk)   # UNTESTED at this composition -- spike first
    else:
        raise ValueError(f"chart_mean_curvature: unknown mode {mode!r}")
    return J, Hess
```

Both `jacrev` and `jacfwd` return the **same** `(out_dim, chart_dim)` Jacobian shape — mode changes only the number of internal passes (`~D=768` for `jacrev`'s per-row VJPs vs `~d=20` for `jacfwd`'s per-column JVPs), never the output shape or value [CITED: docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html]. `torch.func.hessian` for `f: R^n -> R^m` returns shape `(m, n, n)` [CITED: same source] — matching the existing `(out_dim, chart_dim, chart_dim)` shape assertion, so `jacfwd(jacfwd(f))` must be checked against the **same** shape assertion, not assumed compatible.

### Pattern 3: Duplicate-and-pin-by-test, never edit sealed code (already the milestone's pattern)

**What:** `decoder_curvature.py` (02.6) is a near-verbatim copy of `chart_curvature.py`'s body with the chart-routing removed, explicitly documented as "a strict simplification of already-reviewed code, not a new derivation." `derivative_bridge.reduce_to_H_vec` similarly mirrors `chart_mean_curvature`'s algebra and is pinned equal by a dedicated test rather than trusted by inspection.

**When to use:** Building `synthetic_controls.py`. Do not edit `curvature_probe.py` (a sealed 02.5 artifact with no phase-3 edit authorization in `03-CONTEXT.md`'s canonical refs). Instead, import `curvature_probe.graph_mean_curvature` (already tested, general `(d,D)` graph-of-function curvature) unmodified, and write only the new fixture-construction code (point sampling + embedding), following the same "new module, sealed module imported not edited" split `decoder_curvature.py` established.

### Anti-Patterns to Avoid

- **Rewriting `chart_curvature.py`'s d×d solve "for clarity" while adding the forward-mode toggle.** The existing implementation's non-obvious form (trace-first) is load-bearing for memory at `D=768`; a well-meaning refactor toward the textbook projector-then-trace form would materialize a 151 MB `(D,D)` projector per 32-row chunk and defeat the entire point of the optimization Pattern 1 documents.
- **Conflating CLAUDE.md's mandatory sanity-check notebook with the phase's actual Step-1 gate.** These are two different deliverables with two different budgets (see Validation Architecture below) — putting the 5-seed × n_charts sweep inside the `<2 minute, ≤15 cells` sanity notebook either blows the budget or forces cutting seeds/configs that D-01/D-04 require.
- **Editing `curvature_probe.py` to add sphere/saddle fixtures.** It is a sealed 02.5 artifact; `03-CONTEXT.md`'s canonical refs list only `chart_curvature.py` as this phase's editable curvature module (plus `derivative_bridge.py` for D-14).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Second-fundamental-form / mean-curvature computation | A new autodiff Jacobian/Hessian pipeline | `chart_curvature.chart_mean_curvature` / `chart_curvature_field` | Already sealed, tested against a closed-form toy decoder (`test_chart_curvature_matches_analytic_on_toy_decoder`) and a projector-form cross-check; a rewrite reintroduces every shape/index trap the existing tests already caught (Pitfall 5 in the code: `jacrev(jacrev(f))` and `hessian(f)` are not drop-in interchangeable under an outer `vmap`) |
| Finite-difference verification of second derivatives | A new central-difference stencil | `derivative_bridge.finite_difference_hessian` (fix WR-01/02/03, do not rewrite) | The stencil is already calibrated against this project's own `_SphereDecoder` fixture (`4.5e-8` max abs error at `h=1e-4`); a new stencil needs its own calibration from scratch |
| Near-singular metric detection | A custom eigenvalue-ratio check | `torch.linalg.cond(g)`, already computed and returned by `chart_mean_curvature`/`plain_decoder_curvature` | `cond(g)` is the standard conditioning diagnostic for a Gram matrix; the existing return dict already carries it per point, so "flag near-singular points" is a threshold decision on data already produced, not new computation |
| Persistent homology diagrams / distances | A hand-rolled Vietoris-Rips or bottleneck-distance implementation | `persistence_probe.persistence_diagram` / `ph_agreement` (via `ripser`/`persim`) | Sealed 02.6 module already handles the H0-infinite-death filtering hazard, the bottleneck-saturation hazard, and the `torch.quantile` `2**24`-element cap fix; re-deriving any of these from scratch reopens defects this milestone already closed once |
| Sphere / flat-plane known-answer curvature fixtures | A new closed-form derivation from first principles | The pattern already used (and tested) in `test_decoder_curvature.py`'s `_SphereDecoder` (inverse stereographic map, `‖H‖=d/R` exactly) and `test_curvature_probe.py`'s `_flat_plane_fixture`/`_sample_sphere` (pad-and-rotate into `R^D`) | These constructions are already validated to machine precision (`1e-12` sphere agreement) at small scale; generalizing them to `d=20, D=768` is a parameter change, not a new derivation |

**Key insight:** every piece of mathematics this phase needs has already been built, reviewed, and pinned by a regression test somewhere in Phases 02.5/02.6. The actual net-new work is (1) a forward-mode code path sharing the existing downstream math, (2) three named bug fixes with fixes already specified in a code review, and (3) parameter-generalizing two already-tested toy fixtures (sphere, flat plane) plus reusing a third already-tested general function (`graph_mean_curvature`) for the saddle. Writing any of this from scratch is strictly worse than extending what exists.

## Mean Curvature Mathematics (this codebase's convention)

For the chart decoder as an immersion `F: R^d → R^D` (`chart_decoder_map`, `d=chart_dim=20`, `D=out_dim=768`):

```
J    = D F(z)                                (D, d)     Jacobian
g    = J^T J                                 (d, d)     pullback (first fundamental form)
P_N  = I_D - J g^{-1} J^T                    (D, D)     normal projector -- NEVER materialized
II   = P_N D^2F(z)                           (D, d, d)  second fundamental form
H    = tr_g(II) = sum_{jk} g^{jk} II_jk      (D,)       mean curvature VECTOR, trace convention
```

`CURVATURE_CONVENTION = "trace"` [VERIFIED: `chart_curvature.py` module constant, regression-guarded by `test_chart_curvature_uses_trace_convention_not_averaged`]. This is **not** the convention most differential-geometry references state. Standard references (e.g. do Carmo) define mean curvature as the *average* of principal curvatures, `H = (1/d) tr_g(II)` (equivalently `H = (κ_1+...+κ_d)/d`); under that convention a unit `d`-sphere has `H=1`. Under **this codebase's** convention, a unit `d`-sphere has `‖H‖ = d` [VERIFIED: `test_plain_decoder_curvature_sphere_known_answer`, `_sample_sphere` docstring in `test_curvature_probe.py`]. The Laplace-Beltrami identity used to derive the raw-point baseline reads `Δ_g F = H` under the trace convention, **not** `Δ_g F = dH`.

**Why this matters for this phase specifically:** the synthetic control fixtures (flat/sphere/saddle) are new code. Any derivation copied from an external differential-geometry reference, or generated by an LLM consulting training knowledge, will very likely use the averaged convention and be off by exactly `d=20` at PU's working dimension. `02.5-NOTE-randomized-trace.md` records this codebase has already shipped and fixed exactly one such bug (`2*(d+2)/r2` corrected to `2*d/r2` in `centroid_mean_curvature`). Every new analytic-`H` formula this phase writes needs its own `test_..._uses_trace_convention_not_averaged`-shaped regression test, following the existing pattern, not a visual inspection.

The `d×d`-solve optimization (Pattern 1 above) is mathematically exact — it is a reordering of the same contraction, not an approximation — and is the form to extend for both the forward-mode toggle and any new curvature computation on synthetic-control fixtures that goes through `chart_curvature`-style code (as opposed to `curvature_probe.graph_mean_curvature`, which uses its own equivalent-but-independently-derived `(d, d)`-metric form already suited to closed-form fixtures).

## Batched `torch.func` Shapes and the Forward-Mode Toggle

### Shape reference (both modes agree)

For `decode_one: R^chart_dim → R^out_dim`, batched via an outer `vmap`:

| Quantity | Shape | Reverse-mode call | Forward-mode call |
|---|---|---|---|
| Jacobian `J` | `(batch, out_dim, chart_dim)` | `vmap(jacrev(decode_one))(chunk)` | `vmap(jacfwd(decode_one))(chunk)` |
| Hessian `Hess` | `(batch, out_dim, chart_dim, chart_dim)` | `vmap(hessian(decode_one))(chunk)` — `hessian = jacfwd(jacrev(f))` | `vmap(jacfwd(jacfwd(decode_one)))(chunk)` — **new composition, not yet exercised anywhere in this codebase** |

`torch.func.hessian(f)` for `f: R^n → R^m` returns `(m, n, n)` [CITED: docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html] — matching the existing `(out_dim, chart_dim, chart_dim)` per-point shape. This shape is identical regardless of composition order (`jacfwd(jacrev)`, `jacfwd(jacfwd)`, `jacrev(jacrev)` all compute the same mathematical Hessian and therefore the same shape) — **but the existing code's own Pitfall 5 comment is explicit that a wrong composition "still runs" and silently returns a Jacobian-shaped result instead of raising**, so D-09's shape assertions (already present in `chart_mean_curvature`, checking `tuple(Hess.shape) == (VMAP_CHUNK, out_dim, chart_dim, chart_dim)`) are not optional scaffolding to remove — they are the only thing standing between a wrong composition and a silently-wrong curvature field.

### Which composition to use, and why (official guidance, confirmed applicable)

> "if you're computing the jacobian of an `R^N → R^M` function, and there are many more outputs than inputs (for example, `M > N`) then `jacfwd` is preferred, otherwise use `jacrev`." [CITED: docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html]
>
> "depending on your model, you may also want to use `jacfwd(jacfwd(f))` or `jacrev(jacrev(f))` instead [of the default `hessian(f) = jacfwd(jacrev(f))`]... leveraging the rule of thumb above." [CITED: same source]

For the chart decoder, `N = chart_dim = 20`, `M = out_dim = 768`, so `M >> N` — the *official* PyTorch guidance independently confirms D-08's proposed direction (forward mode is the textbook-correct choice here), not merely this project's own operation-count arithmetic. This is a genuinely useful cross-check: D-08's `~38×` ceiling table (`d=20` vs `d·D=15,360` for Hessian) is arithmetic this project derived itself; the PyTorch tutorial's independent rule of thumb agrees with its *direction* without having seen this project's numbers.

### The genuine open risk: forward-over-forward may not be a no-op swap

`jacfwd(jacfwd(f))` differentiates twice under forward-mode (dual-number) autodiff — a strictly less battle-tested composition than `jacfwd(jacrev(f))` (today's default) or `jacrev(jacrev(f))`. PyTorch's official docs name exactly this tradeoff: `jacrev(jacrev(f))` "has better operator coverage" precisely because forward-mode AD does not implement a `vmap` batching rule for every ATen operator [CITED: same tutorial page]; the general symptom is a `RuntimeError` of the shape `"vmap: We do not yet support ... batching rule not implemented for aten::..."`, confirmed as a live, current-PyTorch class of issue by multiple open GitHub issues (e.g. `aten::_make_dual` unbatched, issue #138800) [CITED: github.com/pytorch/pytorch/issues/138800]. `torch.func`'s own "UX Limitations" page documents this class of gap directly [CITED: docs.pytorch.org/docs/stable/func.ux_limitations.html].

**Grounds for cautious optimism, not certainty:** this exact codebase already runs a `vmap`-composed forward-mode-adjacent computation successfully on this exact decoder architecture (SiLU-activated `Linear` stack) — `chart_curvature.directional_second_derivative`, used by the (non-gating) randomized-trace estimator, nests `jvp(df, (zz,), (vv,))` inside an outer `vmap`, and is exercised by `test_chart_curvature_randomized_trace_converges_to_exact`. That is evidence the operators in this specific decoder (Linear + SiLU) are forward-mode-`vmap`-compatible for a *single* level of forward differentiation. It is **not** proof that a *second, nested* level (`jacfwd(jacfwd(f))`, i.e. forward-over-forward rather than forward-once) succeeds — dual-number-of-dual-number composition can exercise different, less-covered code paths than a single `jvp`.

**Recommendation for the planner:** treat this as an empirical spike, not a research-answerable question. Before committing D-09's full equivalence-test suite, run a 2-line smoke test — `vmap(jacfwd(jacfwd(decode_one)))(small_batch)` against the sealed `cae.ChartAutoEncoder`'s actual chart-decoder shape (SiLU, the real hidden widths) — early in the plan's task ordering, so a batching-rule failure is discovered before test-suite design effort is spent on it. **Fallback if it fails:** `jacfwd(jacrev(f))` — forward Jacobian, reverse-composed Hessian — still gets the cheap `~d=20`-pass Jacobian (most of D-08's win, since the Jacobian is computed and reused for the metric `g`) while leaving Hessian cost at today's `~d·D` rather than the hoped-for `~d²`; this is a real, useful, lesser win rather than an all-or-nothing toggle. D-08's own text already anticipates exactly this failure mode ("forward-mode `vmap` may hit an unimplemented-batching-rule error on some decoder op and not run at all. That is precisely why the toggle defaults to reverse") — this research corroborates that caution rather than resolving it.

### Chunking discipline extends to the forward-mode path

`VMAP_CHUNK=32` exists for two independent reasons documented in the sealed module: (1) peak memory (`VMAP_CHUNK * out_dim * chart_dim^2 * 8 bytes` = `32*768*400*8` = 78.6 MB at the sealed architecture — **unchanged by mode**, since it bounds the output tensor, not the compute path), and (2) bit-reproducibility (`vmap(hessian(f))` and, separately confirmed, `vmap(jacrev(f))` are measurably **not** bit-identical across differing batch widths — up to `~1e-14` on a decoder-shaped map). Whether `vmap(jacfwd(f))`/`vmap(jacfwd(jacfwd(f)))` share this non-reproducibility-across-batch-width property is untested; **the safe default is to keep the identical `VMAP_CHUNK`-width chunking discipline for the forward-mode path** rather than assume it is unnecessary, so `test_chart_curvature_field_reassembles_in_row_order`'s bit-identical-across-batch-size guarantee extends to forward mode without a separate investigation.

## Numerical Hazards at `d≈20, D=768`

### Near-singular metric: detect and flag, never average in

`cond(g)` is already computed per point by both `chart_mean_curvature` and `plain_decoder_curvature`. The sealed Swiss-roll measurement (`02.5-09-SUMMARY.md`) gives a concrete, in-repo reference scale for what "well conditioned" looks like at this codebase's own decoder shapes: `max cond(g)` tracked chart count in lockstep — `3.26` (3 charts), `7.64` (5 charts), `63.19` (8 charts, seed 0), `122.22` (8 charts, seed 1) — and D-05's own finding is that a rising `max cond(g)` is the *symptom* of atlas fragmentation (more charts → more artificial seams → less well-conditioned local parameterizations), not an independent numerical artifact to threshold away from the geometric finding. **This phase should report `cond(g)`'s distribution (not just its max) alongside the D-07 selection table**, since D-07 already keys `n_charts` selection on `max cond(g)` as one of its four diagnostics — a per-point histogram, not only the scalar max, lets a reader distinguish "one bad point" from "the whole field is marginal."

There is no single universally-correct numeric threshold for "near-singular" in this codebase (none is pre-registered anywhere in 02.5/02.6) — the existing pattern is comparative (compare `max cond(g)` across configs, as D-05/D-07 both do), not absolute. **Recommendation:** the planner should pick a comparative or percentile-based flag (e.g., "points above the 99th percentile of `cond(g)` within a config are flagged, not averaged into the reported `‖H‖` distribution's summary statistics") rather than inventing an absolute numeric bound with no precedent to justify it — consistent with the milestone-wide rule against fixed absolute thresholds (`REQUIREMENTS.md`'s Out-of-Scope table explicitly rules out "fixed absolute curvature threshold" for the same structural reason: scale depends on the specific fit).

### Distinguishing genuine near-zero curvature from a degenerate metric

This is exactly what the recommended saddle synthetic control (see below) is designed to test: a point where the mean curvature vector is genuinely (analytically) near-zero because positive and negative principal curvatures cancel in the trace, versus a point where `cond(g)` is large because the decoder's differential is nearly rank-deficient (a near-non-immersion point). These produce the same symptom (`‖H‖` small) for different reasons, and only `cond(g)` — reported alongside `‖H‖`, never substituted for it — distinguishes them. `chart_mean_curvature` already returns both quantities per point; the phase's job is to *report* the joint distribution, not compute anything new.

### Memory and time cost, unmeasured at `D=768`

Everything the milestone has timed so far is either Swiss roll (`d=2, D=3`, curvature 1.2s over 3,000 points per `02.5-09-SUMMARY.md`) or PU **training only** (1,941.2s/fit, `02.2-FINDINGS.md`, predates the forward-mode toggle and measures nothing about curvature cost). Peak memory per chunk is knowable from the existing formula (`VMAP_CHUNK * out_dim * chart_dim^2 * 8B` = 78.6 MB at `chart_dim=20, out_dim=768` — unchanged from the sealed architecture's own documented figure, since `chart_dim=20` was already the D-11-justified value used when that formula was written), but **wall-clock time for curvature at `d=20, D=768` over the full 10,000-row PU cloud has never been measured**. D-13 explicitly flags this and leaves a timing probe to the planner's discretion; given the total unknown, this research recommends treating it as *not* optional — a timing probe (even a small one, e.g. 200 points) before committing the full 9-fit `n_charts × seed` sweep protects the ~3-5h budget from a silent multiple-times overrun, following the exact precedent of `cae.timing_probe` (used by `cae_train_run.py`'s D-07 compute-contingency check, already in this codebase).

## Synthetic Control Manifolds at `d≈20, D=768`

This is the more consequential of the two gray areas the discussion left open. **What it can and cannot establish must be stated plainly in the phase's own artifacts, not left implicit.**

### What the control CANNOT do (state this explicitly in the deliverable)

A synthetic manifold sampled cleanly and fitted under the same protocol **will train to a clean, unfragmented atlas** — it has none of the pathology (`02.5-09`'s measured chart-count-driven fragmentation, `max cond(g)` climbing from 3.26 to 122.22, second derivatives banding along artificial chart seams) that the milestone's own override is worried about on real PU data. **A control that passes therefore cannot rule out parameterization damage on PU** — it establishes only that the decoder-curvature *pipeline* (autodiff mechanics, the trace-convention math, the guard functions) is correct on a manifold that is easy for the CAE to fit, which is a necessary but not sufficient condition for trusting the PU field. This caveat is required by CURV-06/07's stale-but-intent-preserved text and must survive re-minting.

### Constructions (three fixture types, all reusing already-tested code)

All three follow the same "pad with zeros to `D=768`, apply a fixed random orthogonal rotation `Q`, so no coordinate axis is privileged" pattern already used by `test_curvature_probe.py`'s `_flat_plane_fixture` and `curvature_probe.make_graph_of_function_fixture`'s own `apply_rotation` step — this is not new design, it is the existing tested pattern generalized to `d=20`.

**1. Flat plane, `d=20`, `D=768`.** Direct generalization of `test_curvature_probe._flat_plane_fixture`: sample `n` points uniformly in `[-1,1]^20`, zero-pad to `R^768`, rotate by a fixed random orthogonal `Q ∈ R^{768×768}`. Analytic `‖H‖ = 0` **exactly**, at every point, by construction (a linear embedding has zero second fundamental form identically) [VERIFIED: `test_centroid_estimator_known_curvature`'s flat-plane assertion pattern, generalizable without modification since the construction has no `d`-dependence in its correctness argument].

**2. Sphere, `d=20`, `D=768`.** Direct generalization of `test_curvature_probe._sample_sphere`: sample `n` points uniformly on the unit `d`-sphere in `R^{d+1}` (normalized Gaussian construction), embedded in `R^{21}`, then zero-padded to `R^{768}` and rotated by fixed `Q`. Under this codebase's **trace** convention, `‖H‖ = d/R` exactly at every point [VERIFIED: `_SphereDecoder`'s docstring and `test_plain_decoder_curvature_sphere_known_answer`, generalizable to any `d` — the `‖H‖=d/R` identity is `d`-parametric, not specific to `d=2`]. At `R=1`, `d=20`: analytic `‖H‖ = 20` exactly — a useful, large, unambiguous target value, far from both zero and any float64 noise floor.

**3. Saddle, `d=20`, `D=768` (new construction, no existing exact analogue — build on `graph_mean_curvature`, do not invent a new curvature formula).** Recommended construction: a quadratic graph `f: R^20 → R^1`, `f(x) = 0.5 x^T Q x` with `Q = diag(s_1,...,s_20)`, `s_i ∈ {+1,-1}` mixed-sign (e.g. 10 positive, 10 negative, for the sharpest possible near-cancellation test) — embedded as `M = {(x, f(x), 0, ..., 0)} ⊂ R^{21}`, zero-padded to `R^{768}` and rotated. `grad f(x) = Qx` (linear, not constant — so the metric `g = I + (grad f)(grad f)^T` genuinely varies across the domain, unlike a pure quadric where curvature would be constant only at the origin), `hess f(x) = Q` (exactly constant). This is a **direct input** to `curvature_probe.graph_mean_curvature(grad, hess)` — already tested, already trace-convention-pinned, no new curvature math is written, only new `grad`/`hess` arrays computed by hand (`grad = x @ Q`, `hess = Q` broadcast over the batch). With `tr(Q)=0` (equal positive/negative eigenvalue count), the trace of the raw ambient Hessian is exactly zero at every point that lies exactly on the flat tangent-only submanifold `x=0`... more precisely, this construction gives a spatially-varying, sign-mixed curvature field whose value is analytically known at every sampled point via `graph_mean_curvature`, satisfying the "known-geometry" requirement without inventing new differential geometry.

**Why this design over alternatives:** a single fixed non-degenerate quadratic form (rather than the existing `make_graph_of_function_fixture`'s Gaussian-bump family) gives every point a well-defined, non-decaying, spatially-varying analytic answer with a genuine positive/negative principal-curvature mix — a materially different test than the sphere (constant, uniformly positive) or the flat plane (exactly zero everywhere). It is the recommended shape, not a locked decision; `make_graph_of_function_fixture` with a large `n_bumps` and mixed-sign amplitudes is an available alternative already fully implemented and tested (no new code at all), at the cost of the field decaying to near-zero far from the bump centres rather than the saddle staying genuinely curved everywhere.

**[ASSUMED]** — this saddle construction has no existing regression test in the codebase (unlike the flat plane and sphere, which generalize already-tested small-`d` code). The planner should add a dedicated `test_saddle_fixture_matches_graph_mean_curvature`-shaped test (finite-difference cross-check, following `curvature_probe.py`'s own precedent of pinning `graph_mean_curvature` against an independent finite-difference computation) before trusting it as ground truth.

### Fitting protocol (CURV-06's "same architecture and protocol")

Whatever `n_charts` and `chart_dim` the PU field settles on (D-07's selection), the synthetic controls should be fitted with the **same** architecture (width, depth, activation, `n_charts`, training hyperparameters) — this is what "matched" means in CURV-06's intent, and it is why this is a genuinely new set of CAE fits (3+ fits: one each for flat/sphere/saddle, potentially at more than one `n_charts` if the PU-side sweep did not converge on a single clear winner), not a reuse of any existing fit.

## Persistent Homology H0/H1 Agreement (D-07's fourth diagnostic)

`persistence_probe.py` (sealed 02.6) is a complete, tested, policy-free instrument: `persistence_diagram` builds `ripser` diagrams (finite pairs only — the H0 diagram's one infinite death is filtered, a **correctness** requirement, not hygiene, since `persim`'s distances cannot consume an infinite value), `ph_agreement` computes bottleneck/Wasserstein distance normalized by the reference diagram's own longest life, and `readout_matrix` produces 16 cells (`{latent, decoder_image} × {intrinsic, ambient} × {H0, H1} × {bottleneck, wasserstein}`) — **never** collapsed into a composite score, matching this milestone's consistent "never collapse multi-axis evidence" pattern (also seen in `curvature_fidelity_report`'s direction/magnitude/calibration split).

`PH_MAXDIM = 1` is a module-level constant [VERIFIED: `persistence_probe.py`] — H2 is structurally excluded by construction, not merely by convention, matching D-07's explicit exclusion.

**For Phase 3's D-07 use:** the relevant cells are almost certainly the `latent|intrinsic|H{0,1}|*` and `latent|ambient|H{0,1}|*` rows (the CAE's chart-coordinate space against a reference), not the `decoder_image` rows (which compare the *reconstruction* back in ambient space — a reconstruction-quality question, already covered by D-07's separate held-out-reconstruction term). The planner should decide which of the 8 relevant `latent|*` cells (of the 16 total) actually feeds the `n_charts` selection rule, and state that choice explicitly, since `readout_matrix` computes all 16 by default and D-07's own text says only "H0 and H1" without specifying `intrinsic` vs `ambient` reference or `bottleneck` vs `wasserstein` distance — **this is an open question the phase's plan must resolve**, not something `persistence_probe.py` resolves for it.

**Reproducibility gap already documented, inherited unchanged:** `ripser`/`persim` are not in `pyproject.toml` (CLAUDE.md bars editing it); a clean checkout needs a manual `pip install`. This phase inherits, not resolves, that gap.

## Finite-Difference Bridge at PU Scale (D-14, WR-01/02/03)

`derivative_bridge.py` (sealed 02.6, this phase's second editable module per D-14) already contains everything needed for the bridge itself — a calibrated central-difference stencil (`FD_STENCIL_CONVENTION`, diagonal + four-point mixed), a chunking budget (`MAX_FD_ROWS=8192`, derived from the `1 + 2d + 4d(d-1)/2` decoder-evaluations-per-point count — `3,201` evaluations/point at `d=40`; at Phase 3's `d=20` this is `1 + 40 + 4·190 = 801` evaluations/point, roughly a quarter of the `d=40` cost the constant was sized against, so the existing `MAX_FD_ROWS=8192` budget has headroom, not less), and a reporting function (`derivative_agreement`) that never applies a threshold — matching D-18's "report, never gate" discipline.

**The three fixes this phase must apply, with fixes already specified (verified by reading `02.6-REVIEW.md` directly, not paraphrased from memory):**

### WR-01 — the float64 guard silently no-ops on the FD side

`finite_difference_jacobian`, `finite_difference_hessian`, and `calibrate_fd_step` each call `chart_curvature._assert_float64(decode_batch, z)`, where `decode_batch` is a bound method or closure — `_assert_float64` reads `getattr(model, "parameters", None)`, which is `None`/non-callable on a bound method, so the per-parameter dtype check is silently skipped (only the `z.dtype` half of the check actually runs). **Not currently corrupting any recorded PU number** (every production call site pre-casts with `.double()`), but a public-API trap for any future caller that does not.

**Fix [CITED: `02.6-REVIEW.md` WR-01 fix, verbatim]:**
```python
def _assert_decode_batch_float64(decode_batch, z_chart):
    if z_chart.dtype != torch.float64:
        raise ValueError(f"...; got z_chart.dtype={z_chart.dtype}. Pass z_chart.double().")
    with torch.no_grad():
        probe_dtype = decode_batch(z_chart[:1]).dtype
    if probe_dtype != torch.float64:
        raise ValueError(
            f"derivative_bridge: decode_batch produced {probe_dtype} output from a float64 "
            "input -- the underlying model is not float64. Call model.double() first."
        )
```
Replace the three `_assert_float64(decode_batch, z)` call sites (`derivative_bridge.py:156, 205, 308`) with this probe-based check, and add a regression test constructing a float32 model + float64 `z`, asserting a friendly `ValueError` (not the bare `RuntimeError` — `"mat1 and mat2 must have the same dtype, but got Double and Float"` — that currently surfaces).

### WR-02 — relative-error columns inflated by near-zero (not exactly-zero) references

`_agreement_stats`'s `clamp_min(1e-300)` prevents exact division-by-zero but not division-by-near-zero; a genuinely tiny (but nonzero) autodiff Hessian entry near a locally-flat axis pair produces an enormous relative error at that one entry, and `max_abs_relative` reports it as if representative. **Already visible in the recorded PU table**: `full_hess_max_abs_rel = 1.1351e+00` (>100%) for one chart of `cae_seed20260804` [CITED: `02.6-FINDINGS-02.md` §8b, via `02.6-REVIEW.md`].

**Fix [CITED: `02.6-REVIEW.md` WR-02 fix]:** either (a) a one-sentence docstring caveat next to `max_abs_relative` (D-18 already forbids applying any threshold here, so this is a reporting-clarity fix, not a math fix — mirroring `persistence_probe.max_persistence`'s existing "thin denominator" caveat style), or (b) an additional diagnostic — the count/fraction of entries where `reference.abs() < floor`, or a version of `max_abs_relative` computed only above a floor — so a table reader can tell "genuine large disagreement" from "one near-zero reference entry." The review does not mandate which; either satisfies D-18 (report, never gate).

### WR-03 — `calibrate_fd_step`'s autodiff Hessian is computed unchunked

`autodiff_hess = vmap(hessian(decode_one))(z)` runs the whole batch in one `vmap` call, unlike every other Hessian call site in the codebase (all chunked at `VMAP_CHUNK=32`, both for peak memory and bit-reproducibility). Currently masked because `derivative_bridge_run.py`'s `BRIDGE_N_POINTS=32` happens to equal `VMAP_CHUNK` — a numeric coincidence, not an enforced invariant. **This is exactly the "passes at toy scale, breaks at real scale" defect class `02.6-FINDINGS-02.md` §12 already names twice** (the `torch.quantile` `2**24` cap and the training-budget asymmetry) — a third instance of the same species, and D-14's own text explicitly names this precedent.

**Fix [CITED: `02.6-REVIEW.md` WR-03 fix]:** factor a `_chunked_vmap_hessian(decode_one, z)` helper shared between `calibrate_fd_step` and `derivative_agreement`'s existing chunking loop, so both compute their comparison Hessian under the identical discipline. At minimum, assert `z.shape[0] <= VMAP_CHUNK` or chunk defensively.

**Why this closes now, not later:** D-09's forward-vs-reverse equivalence test compares two *autodiff* paths and cannot catch a bug shared by both (e.g. a shape transposition both paths happen to make identically). The finite-difference bridge is independent of `torch.func` entirely, so it is the one check that *can* catch such a shared bug — and this phase is the first to edit `chart_curvature.py` at all, making this the first point such a shared bug could actually be introduced.

## Common Pitfalls

### Pitfall 1: Transcribing the averaged curvature convention from an external or LLM-generated source
**What goes wrong:** Any new curvature formula (most likely: the saddle synthetic-control fixture) copied from a textbook, paper, or LLM training knowledge will very likely use `H = (1/d)tr_g(II)`.
**Why it happens:** the averaged convention is genuinely more common in the external literature; this codebase's `"trace"` pin is a deliberate, documented departure.
**How to avoid:** every new analytic-`H` formula gets its own regression test in the shape of `test_chart_curvature_uses_trace_convention_not_averaged` / `test_curvature_convention_is_trace_not_averaged`, asserting the result is **not** `1/d`, `d`, or `(d+2)/d` times a plausible alternative reading.
**Warning signs:** a computed curvature exactly `d=20` times smaller than expected against a known-answer fixture; a sphere test that "almost" passes at a suspiciously round ratio.

### Pitfall 2: A wrong `torch.func` transform composition that still runs
**What goes wrong:** `jacrev(jacrev(f))`, `hessian(f)`, and (new this phase) `jacfwd(jacfwd(f))` can each silently return a differently-shaped or transposed tensor under an outer `vmap` without raising — the existing code's own comment names this ("A Jacobian-shaped result here is RESEARCH Pitfall 5's exact warning sign").
**Why it happens:** `torch.func`'s composition machinery does not validate that the caller composed transforms in the mathematically-intended order.
**How to avoid:** keep every existing shape assertion (`tuple(J.shape) == (VMAP_CHUNK, out_dim, chart_dim)`, `tuple(Hess.shape) == (VMAP_CHUNK, out_dim, chart_dim, chart_dim)`) in place for the forward-mode path too — do not remove them as "redundant" once forward mode is added.
**Warning signs:** a curvature field that runs without error but disagrees wildly with a known-answer fixture; shapes that pass `len(shape)==4` but not the exact tuple check.

### Pitfall 3: `jacfwd(jacfwd(f))` hitting an unimplemented `vmap` batching rule
**What goes wrong:** a `RuntimeError` naming an unbatched ATen op (e.g. `aten::_make_dual`) partway through what should be a routine forward-mode Hessian call.
**Why it happens:** forward-mode `vmap` batching rules are not implemented for every operator; nested (nested-dual-number) forward-mode composition exercises less-common code paths than a single level of forward differentiation.
**How to avoid:** spike the exact composition on the real decoder architecture (SiLU + Linear, the real hidden widths) before committing to the full D-09 test suite; keep the D-08-anticipated fallback (`jacfwd(jacrev(f))`, a real but smaller win) ready.
**Warning signs:** the error surfaces only at the real `cae.ChartAutoEncoder` architecture, not at a toy 2-layer decoder — always spike against the real architecture, not a simplified stand-in.

### Pitfall 4: Conflating the CLAUDE.md sanity-check notebook with the phase's real Step-1 gate
**What goes wrong:** either the mandatory `<2 minute, ≤15 cell` sanity notebook is stretched to hold a 5-seed × n_charts sweep (blowing the budget, or forcing seed/config cuts that violate D-01/D-04), or the sanity notebook is skipped because "the gate covers it" (violating CLAUDE.md, which is unconditional per new model).
**Why it happens:** both deliverables involve "Swiss roll + chart decoder curvature," making them look like the same artifact.
**How to avoid:** build both, explicitly: the sanity notebook (single seed, `chart_dim=2`, no gate machinery per D-15, imports `chart_curvature` unchanged) and a separate `notebooks/diagnostics/` runner for the real sweep (D-01..D-05a), following the milestone's own established split.
**Warning signs:** a notebook with more than ~20 executed cells, or one that takes more than a couple of minutes to run end to end, is a sign the two deliverables have merged.

### Pitfall 5: Trusting `chart_survival` for the D-07 occupancy term
**What goes wrong:** `cae.chart_survival` thresholds a *ratio* to the largest chart's weight mass, so decoupled decay shrinks live and dead charts together and cancels — measured returning `8/8` in all 24 fits of a quick task while argmax occupancy actually fell to `6/8`.
**Why it happens:** the function is a plausible-looking existing helper with the right-sounding name.
**How to avoid:** use argmax chart occupancy (`torch.unique(model.chart_probs(z).argmax(dim=1))`, already exactly what `chart_curvature_field` computes internally as `assignment`) for D-07's occupancy term, never `cae.chart_survival`.
**Warning signs:** an occupancy number that never changes across configurations that visibly differ in reconstruction quality or `cond(g)`.

### Pitfall 6: Reading `0.6712` as a validated Swiss-roll target
**What goes wrong:** treating the raw-point centroid baseline's `0.6712` Spearman as a proven-good number to beat.
**Why it happens:** it is the only concrete number in the record, and D-02's `0.65` floor is deliberately set just below it.
**How to avoid:** `02.5-09-SUMMARY.md` itself records that `0.6712` **missed** its own notebook's `>0.90` sanity bar (passed 3 of 4 read-out lines, not 4) — it is a baseline that *works*, not one that is validated as correct. D-02 already demotes it to context-only for exactly this reason; the phase's own artifacts should repeat this caveat rather than presenting `0.65`/`0.6712` as more solid than they are.
**Warning signs:** phase text describing `0.6712` as "the validated benchmark" rather than "the context baseline."

## Code Examples

### Recommended D-09 equivalence test shape (mirrors an existing, sealed test)

```python
# Recommended shape for the forward/reverse equivalence test, mirroring
# test_chart_curvature_dxd_solve_matches_explicit_projector's structure
# (notebooks/pu_manifold/tests/test_curvature_probe.py:1503)
def test_chart_curvature_forward_mode_matches_reverse_to_float64_round_off():
    model = _small_cae("silu", seed=3)
    chart_idx = 1
    z_chart = torch.rand(6, model.chart_dim, dtype=torch.float64)

    out_reverse = cc.chart_mean_curvature(model, z_chart, chart_idx, mode="reverse")
    out_forward = cc.chart_mean_curvature(model, z_chart, chart_idx, mode="forward")

    torch.testing.assert_close(
        out_forward["H_vec"], out_reverse["H_vec"], rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        out_forward["metric_condition_number"], out_reverse["metric_condition_number"],
        rtol=1e-9, atol=1e-12,
    )
    # existing shape assertions must ALSO pass under forward mode -- do not relax them
    assert out_forward["jacobian_shape"] == out_reverse["jacobian_shape"]
    assert out_forward["hessian_shape"] == out_reverse["hessian_shape"]
```

### Recommended saddle fixture, built on the already-tested `graph_mean_curvature`

```python
# Source: builds on notebooks/pu_manifold/curvature_probe.py's graph_mean_curvature
# (already tested; this snippet is new fixture-construction code, not new curvature math)
def make_saddle_fixture(n: int, d: int, D: int, seed: int, domain_radius: float = 2.0) -> dict:
    rng = np.random.default_rng(seed)
    signs = np.array([1.0] * (d // 2) + [-1.0] * (d - d // 2))
    rng.shuffle(signs)                                   # mixed-sign eigenvalues, tr(Q) == 0
    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))

    grad = (x * signs)[:, None, :]                        # (n, 1, d): grad f = Q x, Q = diag(signs)
    hess = np.broadcast_to(np.diag(signs), (n, 1, d, d))   # (n, 1, d, d): hess f = Q, constant

    H_vec_local = curvature_probe.graph_mean_curvature(grad, hess)   # (n, d+1) -- ALREADY TESTED
    # ... zero-pad to (n, D), apply fixed random rotation Q_rot, rescale by global_std,
    #     following make_graph_of_function_fixture's existing pattern exactly.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Reverse-mode-only Jacobian/Hessian (`jacrev`, `hessian=jacfwd(jacrev)`) throughout `chart_curvature.py` | Opt-in forward-mode toggle (`jacfwd`, `jacfwd(jacfwd)`) for the `d << D` regime | This phase (D-08), never previously needed because every prior execution was Swiss roll at `d=2, D=3` where reverse (cost 3) and forward (cost 2) are nearly equal | Up to `~38×` fewer autodiff passes in the theoretical operation-count ceiling for the Hessian at `d=20, D=768`; **unmeasured as an actual wall-clock speedup** — PyTorch's forward-mode path is documented as less optimized than reverse, and `vmap` over dual numbers has its own overhead |
| `derivative_bridge.py` exercised only on Swiss roll / small toy fixtures | Run at PU scale (`d=20`+, `D=768`) for the first time | This phase (D-14) | Surfaces WR-01/02/03 as live risks rather than latent code-review notes; `torch.quantile`'s `2**24`-element cap was already once found this way (`02.6-14`'s post-completion repair) — the same class of "passes at toy scale, breaks at real scale" defect this bridge run is specifically positioned to catch |

**Deprecated/outdated:**
- The frozen `d_frozen=5` embedding dimension (Phase 2) is explicitly rejected for Phase 3's working dimension (D-11) — superseded by the `d=20` justification (TwoNN 19.5, local-PCA median 25.0, median-of-8-estimators 18) recorded in `02-FINDINGS.md` §6.3/6.4.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(decode_one)))` executes without hitting an unimplemented batching-rule error on the real `cae.ChartAutoEncoder` chart-decoder architecture (SiLU + Linear stack) | Batched `torch.func` Shapes / Pitfall 3 | If wrong, D-08's Hessian-side speedup (`~d²` vs `~d·D`) is unavailable; the Jacobian-side speedup (`~d` vs `~D`, via `jacfwd(jacrev(f))`) still holds as a fallback, so the risk is a *reduced* win, not a blocked phase — but the plan should budget an early spike task rather than discovering this mid-D-09 |
| A2 | The recommended `saddle` fixture construction (quadratic graph, mixed-sign diagonal Hessian, fed through the already-tested `graph_mean_curvature`) generalizes correctly to `d=20` with no new edge case | Synthetic Control Manifolds, construction 3 | If wrong (e.g. a sign/shape error in the hand-computed `grad`/`hess` arrays), the "known-answer" control would silently carry a wrong ground truth — mitigated by the recommendation to add a dedicated finite-difference cross-check test before trusting it, following `curvature_probe.py`'s own established pattern for validating new analytic fixtures |
| A3 | The `MAX_FD_ROWS=8192` budget (sized against `d=40`'s `3,201` evaluations/point) has adequate headroom at Phase 3's `d=20` (`801` evaluations/point) without adjustment | Finite-Difference Bridge at PU Scale | Low risk — this is arithmetic on an already-documented formula, not new measurement, but the actual `BRIDGE_N_POINTS` chosen by the planner interacts with this budget and should be re-checked against the real chunk math rather than assumed safe by analogy to `d=40` |
| A4 | The `ripser`/`persim` installation in the current session's `.venv` persists for the whole of this phase's execution | Standard Stack / Persistent Homology | If the venv is rebuilt mid-phase without re-installing these two packages, `persistence_probe.py`'s import guard raises with an actionable message (not a silent failure) — low risk, already mitigated by the existing module's own error message |

**If this table is empty:** N/A — see entries above; every entry is a genuinely open technical risk this research could not close by reading code or citing documentation, and each one has a stated mitigation or fallback.

## Open Questions

1. **Will `jacfwd(jacfwd(f))` actually run on the real decoder without an unimplemented-batching-rule error?**
   - What we know: official PyTorch guidance confirms the *direction* of D-08's proposal is correct for `M>>N`; this exact codebase already runs a related (single-level) forward-mode `jvp`-under-`vmap` composition successfully on this exact architecture.
   - What's unclear: whether the *nested* (forward-over-forward) composition specifically hits an operator gap.
   - Recommendation: an early empirical spike task in the plan, before D-09's full test suite is designed; `jacfwd(jacrev(f))` as a documented fallback if it fails.

2. **Which of `persistence_probe.readout_matrix`'s 16 cells feed D-07's "PH agreement restricted to H0 and H1" selection term?**
   - What we know: D-07's text says "H0 and H1," and `PH_MAXDIM=1` structurally excludes H2 already.
   - What's unclear: whether the selection uses `latent|intrinsic`, `latent|ambient`, both, and whether `bottleneck` or `wasserstein` (or both) — the module computes all combinations but does not choose among them.
   - Recommendation: the plan should state this choice explicitly as one of its own decisions, most likely `latent|intrinsic|{H0,H1}|bottleneck_norm` as the primary read (chart-coordinate space against the sampled manifold's own intrinsic reference, the "does the atlas preserve the actual topology" question D-07 seems aimed at), with the other 6 relevant cells reported as context.

3. **How do the four D-07 diagnostics combine into an `n_charts` selection?** (Claude's Discretion, explicitly left open by CONTEXT.md)
   - What we know: this milestone consistently refuses to collapse multi-axis evidence into one score (`curvature_fidelity_report`'s three separate axes, `persistence_probe`'s 16 uncollapsed cells, `curvature_fidelity_report`'s direction/magnitude/calibration split).
   - What's unclear: whether the planner should follow that precedent (printed table + a stated, simple rule — e.g., a hard-filter-then-rank scheme, or a lexicographic order) versus a weighted composite.
   - Recommendation: printed table + a stated non-composite rule, matching the milestone's own established pattern; a weighted composite would be a departure from precedent that should be named as such if chosen.

4. **Exact `n_charts` sweep values for both Swiss roll and PU** (Claude's Discretion).
   - What we know: the roll's monotone-in-charts mechanism (D-05) was measured at `{3, 5, 8}` giving `{0.8665, 0.4250, -0.0604}`; D-04 wants "the measured monotone range (something like 2/3/5/8)."
   - What's unclear: whether to include `n_charts=16` (the sealed 02.2 architecture's value) in either sweep for continuity with prior fits, or keep both sweeps deliberately independent per D-06.
   - Recommendation: roll sweep `{2, 3, 5, 8}` (spans the measured monotone range plus one untested low value); PU sweep of 3 values under D-13's budget, one of which could reasonably include `16` for continuity even though D-06 forbids using the roll's *winner* to pick it — nothing forbids `16` appearing in the PU sweep on its own model-side merits.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `torch` (CPU build) | All curvature autodiff | ✓ | 2.13.0+cpu [VERIFIED: local `.venv`] | — |
| `numpy` | Fixture construction, array plumbing | ✓ | 2.5.1 [VERIFIED] | — |
| `scipy` | `spearmanr` (roll gate statistic) | ✓ | 1.18.0 [VERIFIED] | — |
| `scikit-learn` | Raw-point baseline (`NearestNeighbors`), reused read-only | ✓ | 1.9.0 [VERIFIED] | — |
| `ripser` / `persim` | D-07's PH H0/H1 diagnostic | ✓ | present, unpinned, not in `pyproject.toml` [VERIFIED: `import` succeeds] | `.venv/bin/pip install ripser persim` on a clean checkout — already the documented fallback for this known gap |
| GPU / CUDA | — | ✗ (CPU-only `.venv`) | — | None needed — every existing timing figure in this milestone (69s Swiss roll notebook wall-clock, 1,941.2s/PU-fit) is already CPU-measured; forward-mode's cost-ceiling arithmetic is dimensionless (pass-count) and applies identically on CPU, though the *actual* speedup (unmeasured either way) may differ from a GPU's |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** none currently missing — `ripser`/`persim` confirmed present in this session's `.venv`; the fallback above is documented in case a clean checkout lacks them.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest`, existing suite (296+ tests as of 02.5-09, `notebooks/pu_manifold/tests/`) |
| Config file | none discovered — tests run via `.venv/bin/python -m pytest notebooks/pu_manifold/tests/` (existing convention, e.g. `02.5-09-SUMMARY.md`'s "Full suite: 286 passed") |
| Quick run command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py -x -k chart_curvature` (existing chart-curvature tests, not a separate `test_chart_curvature.py` file) |
| Full suite command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/` |

### Phase step → evidence map

This phase's four steps are explicitly staged so each gates or informs the next; the evidence needed to trust each step differs in kind, not just degree.

| Step | Behavior | What would make it trustworthy | Sampling density |
|------|----------|-------------------------------|-------------------|
| **1 — Swiss roll gate** | Median `rho_chart > 0.65` at best swept `n_charts`, ≥5 seeds | (a) the sweep table shows the measured monotone-in-charts pattern reproducing `02.5-09`'s direction (fewer charts → higher rho) rather than noise; (b) `chart_curvature.py`'s existing shape/dtype/C2 guards still fire correctly (unit tests, not just the sweep itself); (c) the raw-point `0.6712` context number is reproduced unchanged (regression check that nothing in `chart_curvature.py`'s edits touched the raw-point path) | Full: every swept `n_charts` × every seed (no subsampling — this is the gate itself, D-15 explicitly forbids extra ceremony but not the sweep's own completeness) |
| **2 — PU field, one fit, one seed** | `‖H‖` + `cond(g)` per point, descriptive only | (a) shape assertions pass at real `D=768` scale (not just Swiss roll's `D=3`); (b) `cond(g)`'s distribution is reported, not just its extremes; (c) a wall-clock timing figure is recorded (currently entirely missing at this scale) | One fit is definitionally the whole of this step; per-task validation should still run the existing unit-test suite before and after touching `chart_curvature.py` |
| **3 — Seeds and sanity** | ≥3-seed spread, near-singular flags, finite/nonzero second derivatives, no extrapolation | (a) `derivative_agreement`'s bridge output (WR-01/02/03 fixed) run at PU scale, both `full_hessian_agreement` and `reduced_mean_curvature_agreement` reported; (b) every point evaluated is confirmed to be an actual chart-assigned coordinate (`chart_curvature_field`'s existing `assignment` machinery), never an off-manifold grid point (CURV-08's intent) | 3 seeds × 3 `n_charts` = 9 fits per D-13; bridge run on a representative subsample per config (existing `BRIDGE_N_POINTS`-style budget, re-derived for `d=20`, not assumed equal to the `d=40` precedent) |
| **4 — Synthetic control** | Same architecture/protocol fitted to flat/sphere/saddle at matched `d=20, D=768` | (a) each fixture's own finite-difference cross-check passes (especially the new saddle construction, per Assumption A2); (b) the fitted decoder's curvature is compared against the closed-form answer using the SAME `curvature_fidelity_report`-style direction/magnitude/calibration split already used for the roll, not a new ad-hoc comparison; (c) the parameterization-damage caveat is printed alongside the numbers, not only in prose elsewhere | One fit per fixture type at the PU-matched config is the minimum; if D-12's `d`-sweep escalation trigger fires, the synthetic controls should be re-fit at whatever `d` the escalation lands on, not left stale at `d=20` |

### Sampling Rate

- **Per task commit:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py notebooks/pu_manifold/tests/test_derivative_bridge.py -x` (the two edited modules' own test files)
- **Per wave merge:** full suite (`.venv/bin/python -m pytest notebooks/pu_manifold/tests/`)
- **Phase gate:** full suite green, plus the Step 1 Swiss-roll gate table printed and the D-15-mandated floor decision recorded in the plan's own artifacts, before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `notebooks/pu_manifold/tests/test_curvature_probe.py` — add D-09's forward/reverse equivalence test (mirrors `test_chart_curvature_dxd_solve_matches_explicit_projector`'s structure; chart-curvature tests already live in this file, not a separate one)
- [ ] `notebooks/pu_manifold/tests/test_derivative_bridge.py` — add WR-01/02/03 regression tests (float32-model-raises-friendly-error for WR-01; a near-zero-reference relative-error diagnostic assertion for WR-02; a `z.shape[0] > VMAP_CHUNK` chunking assertion for WR-03)
- [ ] `notebooks/pu_manifold/tests/test_synthetic_controls.py` — new file; flat-plane exact-zero test, sphere exact `d/R` test, saddle finite-difference cross-check test (Assumption A2)
- [ ] Framework install: none — `pytest` already in use, no new framework needed

## Security Domain

`security_enforcement` is absent from `.planning/config.json`, treated as enabled per instructions; the finding, consistent with `02.6-REVIEW.md`'s own prior security assessment of this exact codebase area, is that almost none of the standard ASVS categories apply.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface anywhere in this milestone's code — local research notebooks/scripts only |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Partial | Internal contract checks only (not user input): `_assert_float64`/`_assert_decode_batch_float64` (dtype guards), `assert_c2_activation`/`assert_c2_decoder` (activation-family guards), shape assertions on every `torch.func` transform composition — these exist to catch programmer error and silent-wrong-answer failure modes, not adversarial input, since there is no external input path |
| V6 Cryptography | No | N/A — no secrets, no crypto anywhere in this codebase area |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silent wrong-answer from a ReLU-family (zero second derivative) decoder | Tampering (data integrity, not adversarial) | `assert_c2_activation`/`assert_c2_decoder` raise rather than warn — already implemented, must be preserved for the forward-mode path (D-08's toggle must still call the guard before differentiating) |
| Silent wrong-answer from a wrong `torch.func` transform composition | Tampering (data integrity) | Shape assertions on every Jacobian/Hessian call — see Pitfall 2 |
| Reproducibility gap from an undeclared dependency (`ripser`/`persim`) | Repudiation (a result cannot be reproduced from a clean checkout without the manual install step) | Already documented in `persistence_probe.py`'s own module docstring and import-guard error message; this phase inherits, does not resolve, the gap |

No network surface, no untrusted input path, no persistence layer of consequence beyond gitignored local caches, and no secrets anywhere in this phase's code — matching `02.6-REVIEW.md`'s own explicit finding for the same codebase area ("there is nothing to report on this axis, and I am not manufacturing a finding to fill it").

## Sources

### Primary (HIGH confidence)
- `notebooks/pu_manifold/chart_curvature.py` (read in full) — the sealed, tested curvature computation this phase extends
- `notebooks/pu_manifold/decoder_curvature.py` (read in full) — the "duplicate rather than edit sealed" pattern this phase's `synthetic_controls.py` should follow
- `notebooks/pu_manifold/derivative_bridge.py` (read in full) — the finite-difference bridge D-14 closes three defects in
- `notebooks/pu_manifold/curvature_probe.py` (read through line 987 of 1867; `graph_mean_curvature`, `centroid_mean_curvature`, `swiss_roll_analytic_H_scaled`, the sphere/flat-plane fixture math) — the sealed 02.5 module this phase reuses read-only
- `notebooks/pu_manifold/persistence_probe.py` (read in full) — the sealed 02.6 PH instrument D-07 reuses
- `notebooks/pu_manifold/cae.py` (architecture read via targeted grep — `ChartAutoEncoder`, `PlainAutoEncoder`, `timing_probe`) — the sealed model this phase differentiates, never edits
- `notebooks/pu_manifold/tests/test_curvature_probe.py` (chart-curvature test section read in full) — existing test patterns to mirror for D-09
- `notebooks/pu_manifold/tests/test_decoder_curvature.py` (`_SphereDecoder`, `_LinearDecoder` read) — the known-answer fixture pattern to generalize
- `.planning/phases/02.6-decoder-substrate-screening/02.6-REVIEW.md` (WR-01/02/03 sections read in full, including their prescribed fixes)
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-NOTE-randomized-trace.md`, `02.5-NOTE-high-d-curvature-approaches.md`, `02.5-NOTE-substrate-selection.md`, `02.5-09-SUMMARY.md` (all read in full) — the measured mechanism, convention traps, and seed-sensitivity evidence this phase's Step 1 must reproduce
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md` §6.3 (read) — the dimension-estimate evidence behind D-11
- docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html [CITED, fetched directly] — official `jacfwd`/`jacrev` selection rule of thumb (`M>N` → `jacfwd`) and `hessian(f)` return shape `(m,n,n)`

### Secondary (MEDIUM confidence)
- docs.pytorch.org/docs/stable/func.ux_limitations.html [CITED, via search] — `torch.func`'s own documentation of `vmap` batching-rule coverage gaps
- github.com/pytorch/pytorch/issues/138800 [CITED, via search] — a current, open example of a forward-mode `vmap` batching-rule gap (`aten::_make_dual`), corroborating the class of risk named in D-08's own text

### Tertiary (LOW confidence)
- None used as load-bearing — every claim in this document is either grounded in a file read directly from this repository or a directly-fetched/cited official PyTorch documentation page.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every package is already installed and version-verified in the project's own `.venv`; no new dependency decisions
- Architecture / mathematics: HIGH — the entire computation this phase extends already exists, is sealed, and is pinned by regression tests read directly
- Forward-mode composition risk (D-08/D-09): MEDIUM — grounded in official docs and in-repo precedent for a *related* composition, but the exact `jacfwd(jacfwd(f))` composition on this exact architecture is genuinely untested and flagged as requiring an empirical spike
- Synthetic control fixture design: LOW/ASSUMED for the saddle construction specifically (new code, no existing test); MEDIUM-HIGH for the flat-plane and sphere constructions (direct generalizations of already-tested small-scale code)
- Pitfalls: HIGH — every named pitfall is either a documented, already-fixed defect in this exact codebase (WR-01/02/03, the factor-of-`d` convention bug) or a directly-cited, currently-open class of PyTorch issue

**Research date:** 2026-08-13
**Valid until:** 30 days (stable domain — the sealed code this phase extends has not changed in the interim, and `torch`/`numpy`/`scipy`/`scikit-learn` versions are pinned to this session's `.venv`; re-verify package versions if execution is deferred past that window)
