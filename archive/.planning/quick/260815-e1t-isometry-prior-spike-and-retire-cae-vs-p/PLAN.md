---
phase: quick-260815-e1t
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: false
requirements: [PART-A-ISOMETRY-SPIKE, PART-B-D12-RETIREMENT]
files_modified:
  - notebooks/pu_manifold/decoder_priors.py
  - notebooks/pu_manifold/tests/test_decoder_priors.py
  - notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py
  - .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md
  - .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md
  - .planning/STATE.md

must_haves:
  truths:
    - "An isometry prior on the CAE chart decoder can be switched on for a Swiss roll fit without editing notebooks/pu_manifold/cae.py, and with the prior weight at 0 the fit is bit-identical to the fit cae.train_cae produces on its own"
    - "The Swiss roll anchor rho_chart = -0.06041003026778113 at n_charts=8, seed=0, n_points=3000 still reproduces exactly through the new code path, proving the new runner's fixture, protocol and metric call sequence are a faithful copy and not a re-derivation"
    - "The weight ladder reports, per weight and per seed, cond(g) median and max, rho_chart against analytic H, held-out mse_per_dim, median_magnitude_ratio and calibration_slope — never collapsed into a composite score"
    - "The bias check is decided and printed either way: rho_chart rising while median_magnitude_ratio drifts systematically below 1 is reported as FLATTENING SUSPECTED and blocks any adoption recommendation, and a null or negative result is reported as plainly as a positive one"
    - "The mechanism check gates interpretation: if cond(g) does not fall as the prior weight rises, the penalty is not doing what it claims and no other column in the table is read as evidence"
    - "Every cell in the ladder trains for exactly the same number of epochs, so no column of the table can be a measurement of training length (03-08-DEFECTS-01.md defect 2)"
    - "A reader of .planning/ can see that D-12's escalation trigger is retired, that it FIRED on the corrected grid on both legs before being retired, and that the reason for retirement is that the comparison was the wrong instrument and not that it returned an unfavourable answer"
    - "The C0/C2 argument and the disjoint-regularizer finding are on the record in the planning docs, with the measured cond(g) evidence that motivated them"
    - "No sealed prior-phase finding, summary, verdict, notebook or runner is edited or rewritten; notebooks/pu_manifold/cae.py is byte-for-byte unchanged; the test suite is still green"
  artifacts:
    - "notebooks/pu_manifold/decoder_priors.py — isometry and conformal metric-deviation priors on the chart decoder, opt-in, default off"
    - "notebooks/pu_manifold/tests/test_decoder_priors.py — known-answer and shim-hygiene tests, additive to the existing suite"
    - "notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py — the weight-ladder runner, with --anchor-check, --probe, --dry-run and a resumable JSONL record"
    - ".planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md — the spike result with the full table inline"
    - ".planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md — the C0/C2 argument, the disjoint-regularizer finding, D-12's retirement, and the replacement criterion"
  key_links:
    - "decoder_priors.isometry_penalty <-> chart_curvature.chart_decoder_map (the prior must constrain the SAME map that curvature differentiates; if it constrains chart_encoders instead, it reproduces the exact defect this spike exists to fix)"
    - "the ladder's weight=0 arm <-> cae.train_cae's untouched loop (the control must be the identical architecture and protocol, so the only difference between arms is the prior)"
    - "median_magnitude_ratio and calibration_slope <-> the adoption recommendation (these are the only columns that can see amplitude attenuation; rho_chart is rank-based and is exactly blind to it)"
    - "fixed-length training with early stopping disabled <-> train_cae's total-loss early-stopping rule (the prior is added to total, so leaving early stopping on would let the prior weight change the training length and confound every column)"
---

<objective>
Two developer-directed pieces of work in one quick task.

**Part A — spike an isometry prior on the CAE chart decoder, on the Swiss roll only.** Measured
this session: `cae.lipschitz_penalty` regularizes `model.chart_encoders`, while curvature
differentiates `chart_curvature.chart_decoder_map` (the chart decoders plus the shared embedding
decoder). Those two object sets are **disjoint**, so nothing in the training objective constrains
the decoder's derivatives at any order. The consequence is measured: `cond(g)` reaches `4.886e7`
on the corrected PU grid against the Swiss roll's `1.4`–`122`, which destroys roughly seven digits
of float64 in the `g^-1` contraction inside `H = sum_jk g^jk II_jk`. Add a **first-order** prior
on the chart decoder's Jacobian, `|| J^T J - I ||_F^2` (and its conformal variant
`|| g - c I ||_F^2`), sweep a small weight ladder on the roll, and report whether it works and
whether it biases the estimand.

**Part B — record the decision to retire the CAE-vs-plain-AE comparison**, and with it D-12's
escalation trigger, replacing it with a direct two-part C0/C2 criterion.

Purpose: Part A tests one specific, falsifiable hypothesis about the measured `cond(g)`
pathology, on a manifold whose curvature is known in closed form, before any PU compute is spent
on it. Part B puts a decision and its reasoning on the record so a later reader cannot mistake a
retired instrument for a dropped criterion.

Output: one new library module and its tests, one new diagnostics runner, two additive planning
notes, and an additive STATE.md paragraph. No existing source file, notebook, runner, finding,
summary or verdict is edited.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/phases/03-decoder-curvature-field/03-08-DEFECTS-01.md
@notebooks/pu_manifold/chart_curvature.py
@notebooks/pu_manifold/cae.py
@notebooks/diagnostics/swiss_roll_curvature_sweep_run.py
</context>

<measured_facts>
These were measured during planning, directly from the repository, and are the ground the plan
stands on. The executor does not need to re-derive them, but every one of them is re-checkable.

**1. The disjoint-regularizer finding.** `cae.train_cae` line 483 calls
`lipschitz_penalty(model.chart_encoders, lip_weight)`. `chart_curvature.chart_decoder_map`
(line 183) composes `model.chart_decoders[i]` with `model.embedding_decoder`. The regularized
set and the differentiated set share no parameter. `cae.chart_loss` constrains the decoder only
through reconstruction, which is a C0 quantity.

**2. The PU `cond(g)` pathology, from `notebooks/.cache/03_curvature_field_pu.jsonl`
(the corrected 15-record grid, run 2026-08-15).**

| config | cond median | cond max |
|---|---|---|
| nc4_seed20260813 | 9.758e6 | **4.886e7** |
| nc8_seed20260814 | 4.805e6 | 2.137e7 |
| nc16_seed20260814 | 7.682e6 | 4.202e7 |

Swiss roll, same machinery, `n_charts=8`, `n=12000`: `cond_median` 1.44–1.94, `cond_max`
2.6–8.3. Five to seven orders of magnitude apart.

**3. D-12 fired on the corrected grid, on both legs.** Verbatim from
`.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --select-only`:

```
Selected n_charts: 4
best d=20 CAE (n_charts=4) mse_per_dim=0.000173309 vs control mse_per_dim=3.58866e-05 -> loses_reconstruction=True
best H0/H1 bottleneck_norm=(0.6217,0.8451) vs control=(0.2144,0.8247) -> loses_ph_agreement=True
TRIGGER FIRES = True
```

Held-out `mse_per_dim` across the nine CAE grid cells: 8.899e-05 … 2.438e-04. The matched
(`latent_dim=20`) control: 3.585e-05 … 3.612e-05. Mean residual L2 norm (`mean_norm`): CAE
0.2468 … 0.4112, control 0.1537 … 0.1564.

**4. The Swiss roll baseline the ladder starts from**, `n_charts=8`, `n_points=12000`, from
`notebooks/.cache/03_swiss_roll_curvature_sweep_n12000.jsonl`:

| seed | rho_chart | median_magnitude_ratio | calibration_slope | cond_median | cond_max |
|---|---|---|---|---|---|
| 0 | 0.2877 | 0.9636 | 0.4209 | 1.442 | 8.333 |
| 1 | 0.9377 | 0.9942 | 0.9085 | 1.503 | 2.630 |
| 2 | 0.9743 | 0.9947 | 0.9703 | 1.939 | 4.540 |
| 3 | 0.8234 | 0.9969 | 0.8551 | 1.512 | 2.746 |
| 4 | 0.7758 | 1.0015 | 0.7123 | 1.461 | 3.488 |

These are context for choosing the ladder's seeds. They are **not** the ladder's comparison
baseline — the ladder runs its own `weight = 0` arm under its own fixed-length protocol.

**5. Architecture fact with a design consequence.** `cae.ChartEncoder` ends in a sigmoid, so
chart coordinates live in the open unit cube `(0,1)^d`. A decoder whose domain has extent 1 and
whose image must span a manifold of extent L needs `||J|| ~ L`, i.e. `g ~ L^2 I`, not `g = I`.
The strict isometry penalty therefore contains a **scale** term that fights reconstruction
directly, while the measured pathology — `cond(g) = lambda_max / lambda_min` — is
scale-invariant and is exactly what the **conformal** variant penalizes. This prediction is
written into the runner's source before any number exists, and the conformal arm is the
pre-declared branch if the isometry arm fails its mechanism check. It is recorded here so the
branch cannot be read as a post-hoc axis switch.

**6. The plain-AE fit does not affect `rho_chart`.** In
`swiss_roll_curvature_sweep_run.run_cell`, the CAE is constructed and trained to completion
before `plain` is constructed, and `torch.manual_seed(seed)` is called again before the plain
fit. Omitting the plain AE from the new runner therefore leaves the CAE's parameters, and every
curvature number derived from them, bit-identical. This is what makes the anchor reproduction in
Task 2 valid without fitting a baseline the new runner has no use for.
</measured_facts>

<design_decisions_resolved_here>
**How the prior is wired without editing `cae.py`.** `cae.train_cae`'s main loop resolves
`chart_loss` from `cae`'s module globals at call time (line 479). The prior is therefore
installed by a **scoped context manager** in the new module that temporarily rebinds
`cae.chart_loss` to a wrapper adding the penalty to `total`, and restores the original binding in
a `finally` block. The training loop that runs is `cae.train_cae` itself, unmodified — no forked
copy, so there is no translation to drift. This is deliberately chosen over duplicating the
training loop into the new module: an unfaithful translation of a reference implementation is a
failure mode this project has already paid for once, and a fork of an RNG-order-dependent loop is
the sharpest form of it.

At `weight == 0` the context manager **installs nothing at all**, so the zero-weight arm is
structurally, not merely numerically, the untouched code path.

The permanent form of this seam is a four-line optional `extra_loss` callable parameter on
`cae.train_cae`, defaulting to `None`. That edit is **proposed to the developer at Task 5's
checkpoint and is not applied by this plan.**

**Why the spike script suffices and the notebook does not land now.** CLAUDE.md's Swiss roll
mandate is triggered by a new manifold-learning or representation-learning **model**. This is not
a new model: it is one additive, default-off loss term on the existing CAE, whose Swiss roll
notebook already exists (`notebooks/02.2_swiss_roll_cae_check.ipynb`) and whose chart-decoder
curvature notebook also already exists (`notebooks/03_swiss_roll_chart_curvature_field_check.ipynb`).
The spike does run on the Swiss roll, does set the chart dimension to 2, and does compare against
a matched control — the `weight = 0` arm, which is a *strictly better* matched control than a
plain AE because it is the identical architecture, the identical protocol and the identical
seeds. What the spike additionally needs — a weight ladder, a threshold table, a JSONL record and
a resumable runner — is precisely what CLAUDE.md forbids inside a sanity-check notebook.

So: **spike script now; the notebook becomes mandatory at adoption.** If the developer adopts the
prior, `notebooks/03_swiss_roll_isometry_prior_check.ipynb` must exist and pass before any PU fit
runs with a non-zero prior weight. That obligation is written into the spike note and confirmed
at Task 5's checkpoint.
</design_decisions_resolved_here>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: decoder_priors.py — the prior, end to end, on a real CAE</name>
  <precondition>`.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` reports `289 passed, 1 skipped` before any edit. If it does not, halt and report — the baseline is not what this plan was written against.</precondition>
  <files>notebooks/pu_manifold/decoder_priors.py, notebooks/pu_manifold/tests/test_decoder_priors.py</files>
  <behavior>
    - `metric_deviation(g, mode="isometry")` on a batch of exact identity metrics returns exactly 0.0 per point.
    - `metric_deviation(4*I, mode="isometry")` at chart_dim=2 returns exactly 18.0 (`||4I - I||_F^2 = 2 * 9`).
    - `metric_deviation(4*I, mode="conformal")` returns exactly 0.0 — a uniformly scaled metric has no conformal deviation, which is the whole distinction between the two modes.
    - `metric_deviation(diag(1,4), mode="conformal")` returns exactly 4.5 (`c = 2.5`, `||diag(-1.5, 1.5)||_F^2`).
    - `chart_decoder_jacobian` on a duck-typed linear decoder with known orthonormal columns returns that exact matrix for every row, to float round-off.
    - The returned Jacobian carries a live autograd graph: `metric_deviation(...).sum().backward()` populates a non-zero `.grad` on at least one `chart_decoders[i]` parameter AND on at least one `embedding_decoder` parameter.
    - Inside `decoder_prior_active(model, weight=0.0)`, `cae.chart_loss` is the original function object.
    - Inside `decoder_prior_active(model, weight=1e-2)`, `cae.chart_loss` is not the original; on exit — including exit by exception — it is the original again.
    - With the prior active at `weight=w`, the wrapper's `total` equals the unpatched `total` plus `isometry_penalty(model, x, w, mode)` on the same batch, to float round-off, and `recon`/`xent` are untouched.
    - A 3-epoch `cae.train_cae` on a tiny roll sample inside `decoder_prior_active(..., weight=1e-2)` completes and produces parameters that differ from the same 3-epoch fit at `weight=0.0` — the prior actually moves the optimizer.
    - The same 3-epoch fit at `weight=0.0` produces parameters bit-identical (`torch.equal` on every tensor in the state dict) to a plain `cae.train_cae` call at the same seed and cfg.
  </behavior>
  <action>
Create `notebooks/pu_manifold/decoder_priors.py`. Like `cae.py` and `chart_curvature.py` it
imports torch at module level, so it is **not** added to `pu_manifold/__init__.py` — do not touch
that file.

Module docstring states, in this order: that the prior exists because the regularized objects
(`model.chart_encoders`, via `cae.lipschitz_penalty`) and the differentiated objects
(`chart_curvature.chart_decoder_map`) are disjoint so nothing constrains the decoder's
derivatives; the measured `cond(g)` evidence from measured fact 2; that the prior is FIRST order
and constrains only the PARAMETERIZATION; and — the single most important paragraph in the file —
**why a curvature or Hessian-norm penalty must never be substituted for it.** Penalizing
`||D^2 F||` or `||H||` biases the estimand: it pushes the decoder toward flat surfaces and then
measures flatness, making the curvature field a function of the regularizer weight. The isometry
prior is safe precisely because `H = tr_g(II)` is parameterization-invariant — a property of the
decoder's image alone — so constraining the parameterization cannot move it. If a second-order
term is ever added it must be the TANGENTIAL part `P_T D^2 F` only, which is pure
Christoffel/parameterization content; never the normal part `P_N D^2 F = II`, which is the
geometry itself.

Four public objects:

`PRIOR_MODES = ("isometry", "conformal")` — module constant, with an unknown mode raising a
`ValueError` naming the offending string in the refuse-and-name-the-fix style of
`chart_curvature._assert_float64`. Never a silent fall-through to a default.

`metric_deviation(g, mode)` -> `(batch,)`. `isometry`: `||g - I||_F^2` per point. `conformal`:
`c = trace(g)/d` per point (the mean eigenvalue), then `||g - c I||_F^2`. Pure tensor function,
no model, no dtype guard — this is the piece with exact known answers.

`chart_decoder_jacobian(model, z_chart, chart_idx)` -> `(batch, out_dim, chart_dim)`,
**differentiable with respect to the model's parameters**. Obtains the map by calling
`chart_curvature.chart_decoder_map(model, chart_idx)` — reuse it, do not re-derive the two-hop
decode — and applies `vmap(jacfwd(decode_one))`. Do NOT call
`chart_curvature._chunked_jacobian`: it detaches, which is correct for measurement and fatal for
a training term. Do NOT call `chart_mean_curvature` or anything else in `chart_curvature` that
runs `_assert_float64` — the model is float32 during training and that guard would refuse,
correctly, since this is a first-order quantity where float32 is fine. If the autograd-graph
behaviour test fails under `jacfwd`, fall back to `jacrev` (out_dim is 3 on the roll, so reverse
mode is not expensive here) and record which one was used in a source comment and in the SUMMARY.

`isometry_penalty(model, x, weight, mode="isometry")` -> scalar tensor. Encodes `x`, takes
`model.chart_coords(z)` and the argmax chart from `model.chart_probs(z)` — the **same assignment
`chart_curvature.chart_curvature_field` uses**, so the prior constrains the decoder at the
coordinates curvature will actually be measured at. The assignment is an argmax and is inherently
non-differentiable; take it under `torch.no_grad()` and state in a comment that this is
deliberate. Loop over the charts present in the batch, call `chart_decoder_jacobian` on that
chart's own rows, form `g = J^T J` by `torch.einsum("boi,boj->bij", J, J)` (the same contraction
`chart_mean_curvature` uses), and return `weight` times the batch mean of
`metric_deviation(g, mode)`. Returns a differentiable scalar the optimizer can trade off against
reconstruction, exactly as `lipschitz_penalty` does.

`decoder_prior_active(model, weight, mode="isometry")` — a `@contextmanager`. When
`weight == 0.0` it yields without touching anything, so the zero-weight path is structurally
untouched. Otherwise it rebinds `cae.chart_loss` to a wrapper that calls the original, copies the
returned dict, adds `isometry_penalty(model, x, weight, mode)` to `total` only, and returns it;
`recon` and `xent` stay the base values so `train_cae`'s history keeps reporting the
reconstruction term unpolluted. Restores the original binding in `finally`. Docstring states
plainly that this is a scoped, restore-on-exit shim chosen over forking `train_cae`, and why: the
loop that runs must be the one that will actually run, and `cae.py` is the sealed 02.2
architecture with RNG-order-dependent anchors.

Then create `notebooks/pu_manifold/tests/test_decoder_priors.py` covering every line of
`<behavior>`. Keep the training-loop tests tiny (a few hundred roll points, 3 epochs, chart_dim=2,
hidden [16,16]) so the suite stays fast. Add nothing to any existing test file.
  </action>
  <verify>
    <automated>.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_decoder_priors.py -q && .venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q && git diff --quiet notebooks/pu_manifold/cae.py && git diff --quiet notebooks/pu_manifold/chart_curvature.py && git diff --quiet notebooks/pu_manifold/__init__.py && echo GUARDS_OK</automated>
  </verify>
  <done>`decoder_priors.py` and its test file exist; every `<behavior>` line has a passing test; the full suite reports 289 + N passed with the pre-existing 1 skipped; `cae.py`, `chart_curvature.py` and `__init__.py` are byte-for-byte unchanged.</done>
  <reversibility rating="reversible">A new, default-off module and a new test file; deleting both restores the tree exactly.</reversibility>
</task>

<task type="auto">
  <name>Task 2: the weight-ladder runner, with the anchor reproduction as its first gate</name>
  <files>notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py</files>
  <action>
Create `notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py`. Copy the shape, the
fixture, the protocol and the metric call sequence from
`notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` — that file is **not edited**.

Import it for its constants rather than restating them:
`import swiss_roll_curvature_sweep_run as sweep` after the same `sys.path.insert` idiom, then use
`sweep.FIXTURE_SEED`, `sweep.CHART_DIM`, `sweep.EMBED_DIM`, `sweep.HIDDEN`, `sweep.BASE_CFG`,
`sweep.N_POINTS`, `sweep.N_POINTS_AMENDED`, `sweep.ROLL_FLOOR`. The analytic `H_true` vector block
(`swiss_roll_curvature_sweep_run.py` lines 252–268) is inline in `run_cell` and is therefore
copied verbatim, **including its `pin < 1e-12` assertion against the fixture's sealed `H_norm`** —
that assertion is the thing that makes the copy safe, because it fails loudly if the copy drifts
from the sealed derivation. State that reasoning in a comment where the copy sits.

Declare the ladder in module source, before any number exists, in the same idiom the phase already
uses (`swiss_roll_curvature_sweep_run.py`'s `ROLL_FLOOR` docstring):

- `LADDER_N_CHARTS = 8`, `LADDER_N_POINTS = sweep.N_POINTS_AMENDED` (12000).
- `LADDER_SEEDS = (0, 3)` — chosen from the recorded `n=12000` baseline: seed 0 is the weakest
  baseline of the five (`rho_chart = 0.2877`) and seed 3 is the median (`0.8234`). The docstring
  states this choice and its reason explicitly, so that spanning the range rather than picking the
  two ceiling seeds (1 and 2, at 0.9377 and 0.9743, where no improvement is measurable) is on the
  record as a deliberate choice made before the ladder ran.
- `LADDER_WEIGHTS = (0.0, 1e-3, 1e-2, 1e-1)`.
- `LADDER_MODE_DEFAULT = "isometry"` — the developer-specified primary. The docstring records the
  pre-declared prediction from measured fact 5: chart coordinates live in `(0,1)^d` because
  `ChartEncoder` ends in a sigmoid, so the strict isometry penalty contains a scale term fighting
  reconstruction, while `cond(g)` is scale-invariant and is what the conformal variant targets.
  If the isometry arm fails its mechanism check, the conformal arm is the **pre-declared** branch,
  not a post-hoc axis switch.
- `LADDER_MAX_EPOCHS_CANDIDATES = (150, 100, 75, 50)` and `LADDER_BUDGET_S = 3000`.
- Early stopping is **disabled** for every cell: `early_stop_patience = max_epochs + 1`. State
  why in the docstring — `train_cae` early-stops on **total** loss, the prior is added to total,
  so leaving early stopping on would let the prior weight change the training length and confound
  every column of the table. This is `03-08-DEFECTS-01.md` defect 2 in a new costume, and it is
  headed off by construction rather than caught afterwards.

`run_cell(weight, seed, mode, max_epochs)` follows `sweep.run_cell`'s sequence exactly with three
differences, each of which is commented in place: (a) the CAE fit is wrapped in
`decoder_priors.decoder_prior_active(model, weight, mode)`; (b) **no plain autoencoder is fitted**
— Part B retires that comparison and the `weight = 0` arm is the matched control (measured fact 6
records why this leaves `rho_chart` bit-identical); (c) `cfg` carries the disabled early stopping
and the chosen `max_epochs`. Record per cell: `weight`, `mode`, `seed`, `n_charts_configured`,
`n_charts_used`, `n_points`, `max_epochs`, `epochs_run`, `early_stopped`, `rho_chart`,
`mre_chart`, `median_cosine_similarity`, `median_magnitude_ratio`, `magnitude_ratio_cv`,
`calibration_slope`, `calibration_intercept`, `calibration_r2`, `cond_median`, `cond_max`,
`cae_mse_per_dim`, `final_penalty_value`, `train_wall_s`, `curv_wall_s`, `device`,
`torch_version`. Append-only JSONL at `notebooks/.cache/quick_isometry_prior_sweep.jsonl`, keyed
`(mode, weight, seed)`, with `--resume`, copying `sweep.load_completed` / `sweep.append_record`'s
idiom.

Four CLI entry points beyond the plain run:

`--dry-run` — print the planned grid, the declared ladder and the read-out rules, run nothing.

`--anchor-check` — the faithfulness gate. Runs one cell at `weight=0.0`, `n_charts=8`, `seed=0`,
`n_points=sweep.N_POINTS` (3000), `sweep.BASE_CFG` **verbatim** (early stopping as configured
there, not disabled — this cell reproduces the sealed protocol, not the ladder protocol), and
asserts `rho_chart == -0.06041003026778113` **exactly** (`==`, not a tolerance). On mismatch,
raise naming both values and saying that the runner's fixture, protocol or metric call sequence
has drifted from `swiss_roll_curvature_sweep_run.py` — this is a faithfulness failure, not a
tolerance question. Writes nothing to the record file.

`--probe` — the budget probe, in the `timing_probe` idiom this repo already uses. Runs two short
cells at `max_epochs=10`, one at `weight=0.0` and one at `weight=LADDER_WEIGHTS[-1]`, at the
ladder's `n_points` and `n_charts`. From them derive per-epoch seconds `t0` and `t1` and the
per-cell curvature cost `c` (which does not scale with epochs). Project each candidate
`E` in `LADDER_MAX_EPOCHS_CANDIDATES` as `8*c + E*(2*t0 + 6*t1)` — two `weight=0` cells and six
`weight>0` cells — and print the largest `E` whose projection is at or under `LADDER_BUDGET_S`.
If even the smallest candidate exceeds the budget, print `BUDGET NOT MET` with the projections
and exit non-zero rather than silently degrading the design further. Writes nothing to the record.

`--summary` — read the record back and print, with no composite score anywhere:
  1. The full per-cell table, one row per `(mode, weight, seed)`.
  2. A per-weight row: seed-median of each of `rho_chart`, `median_magnitude_ratio`,
     `calibration_slope`, `cond_median`, `cond_max`, `cae_mse_per_dim`, with the per-seed values
     listed inline. With two seeds the median is the midpoint of the pair; say so in the header
     rather than letting a reader assume a robust statistic.
  3. An `epochs_run` uniformity assertion: every cell must have `epochs_run == max_epochs` and
     `early_stopped == False`. If any cell differs, print `TRAINING-LENGTH CONFOUND` naming the
     cells and refuse to print the read-outs below.
  4. **MECHANISM CHECK**, printed before anything else is interpreted: `cond_median` must be
     strictly lower at the largest weight than at `weight=0`, and non-increasing across the
     ladder. Print `MECHANISM DEMONSTRATED` or `MECHANISM NOT DEMONSTRATED` with the four values.
     On `NOT DEMONSTRATED`, print that the penalty is not doing what it claims and that no other
     column below is evidence about anything.
  5. **BIAS CHECK**, the point of the experiment. Print exactly one of:
     - `FLATTENING SUSPECTED` — `rho_chart` non-decreasing across the ladder AND
       `median_magnitude_ratio` non-increasing across the ladder AND
       `median_magnitude_ratio` at the largest weight below 0.90. Print that rank correlation
       improved while scale collapsed, that this is the signature of a prior that is flattening
       the surface it is meant to leave alone, and that adoption is NOT recommended.
     - `CLEAN IMPROVEMENT` — some weight `w*` has `rho_chart` above the `weight=0` value AND
       `|median_magnitude_ratio - 1| <= 0.10` AND `|calibration_slope - 1| <= 0.20` at `w*`.
     - `NO EFFECT` — otherwise. Print the numbers regardless.
     Whichever fires, print the full `rho_chart` / `median_magnitude_ratio` / `calibration_slope`
     ladder underneath it. A null and a negative result are printed exactly as prominently as a
     positive one.
  6. An unconditional caveat naming the two limitations: two seeds cannot separate a small effect
     from seed noise, and the roll's `cond(g)` is already benign (1.4–8.3), so the roll can
     demonstrate the mechanism and the absence of bias but **cannot** demonstrate a fix at the PU
     pathology of 4.9e7.

Then run `--dry-run` and `--anchor-check` and capture both outputs for the SUMMARY.
  </action>
  <verify>
    <automated>.venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --dry-run && .venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --anchor-check && git diff --quiet notebooks/diagnostics/swiss_roll_curvature_sweep_run.py && echo ANCHOR_AND_GUARD_OK</automated>
  </verify>
  <done>`--dry-run` prints the declared grid and read-out rules and writes nothing; `--anchor-check` reproduces `rho_chart = -0.06041003026778113` exactly and exits 0; `swiss_roll_curvature_sweep_run.py` is byte-for-byte unchanged; no record file was written by either invocation.</done>
</task>

<task type="auto">
  <name>Task 3: run the ladder and write the spike note</name>
  <files>.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md</files>
  <action>
Run `--probe` first and record its projections and the chosen `LADDER_MAX_EPOCHS`. If it prints
`BUDGET NOT MET`, stop here, write the note with the probe numbers and a `BUDGET NOT MET`
outcome, and take the halt to Task 5's checkpoint — that is a real terminal branch, not an error
to work around.

Otherwise run the isometry ladder at the chosen epoch count (8 cells, resumable), then run
`--summary`.

**Branch, pre-declared in Task 2's source and executed here:** if the isometry arm prints
`MECHANISM NOT DEMONSTRATED`, run the conformal arm at the same protocol, seeds and weights
(`--mode conformal`, 6 further cells since `weight=0` is mode-independent and already recorded),
and summarize both. If the isometry arm prints `MECHANISM DEMONSTRATED`, do **not** run the
conformal arm in this task — record it as the untaken branch with its reasoning intact.

Write `.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md`, following the
shape of `03-08-DEFECTS-01.md` (a dated header, a summary table, then sections). It must contain:

- Header: date, `Status: spike — gates nothing, selects nothing, adopts nothing`, and
  `Raised by: the developer` with the C0/C2 motivation in two sentences.
- The motivation: reconstruction loss is C0, curvature is C2, and small C0 error does not bound
  C2 error; and the disjoint-regularizer finding with its file and line references.
- What was built, in one short section, including the scoped-shim decision and the fact that
  `cae.py` was not edited, plus the four-line `extra_loss` seam **proposed** for the developer.
- **The full per-cell table inline.** `notebooks/.cache/` is gitignored, so the JSONL record is not
  durable in git — if the table is not in this note, the spike's data does not survive. Include
  every recorded column.
- The per-weight seed-median table.
- The mechanism check verdict with its four `cond_median` values.
- The bias check verdict, stated as the criterion and then the outcome. State the criterion
  in the note's own words: a correct result is `rho_chart` up AND `median_magnitude_ratio`
  staying near 1; `rho_chart` up while `median_magnitude_ratio` drifts below 1 is the surface
  being flattened, and rank correlation improving while scale collapses is not a success.
- Whether the conformal branch was taken, and if not, why not.
- Limitations, unhedged: two seeds; the roll's `cond(g)` is already benign so the roll cannot
  demonstrate a fix at the PU pathology; the ladder's `weight=0` arm is a fresh fixed-length
  measurement and is not comparable cell-for-cell to the sealed `n=12000` grid.
- **The notebook obligation**, in its own short section: this spike is a script, not the CLAUDE.md
  Swiss roll notebook, with the reasoning from `<design_decisions_resolved_here>` restated; and if
  the prior is adopted, `notebooks/03_swiss_roll_isometry_prior_check.ipynb` must exist and pass
  before any PU fit runs with a non-zero prior weight.
- A closing "what this does NOT claim" section: no adoption, no PU number, no `n_charts`
  selection, and nothing about whether the prior helps at `d = 20`, `D = 768`.
  </action>
  <verify>
    <automated>.venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --summary && test -f .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md && grep -qE 'MECHANISM (DEMONSTRATED|NOT DEMONSTRATED)' .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md && grep -qE 'FLATTENING SUSPECTED|CLEAN IMPROVEMENT|NO EFFECT|BUDGET NOT MET' .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md && grep -q 'median_magnitude_ratio' .planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md && echo NOTE_OK</automated>
  </verify>
  <done>The ladder ran (or halted on budget with the halt recorded); `--summary` prints the full table, the mechanism check and the bias check; the note carries the full per-cell table inline, both verdicts, the limitations and the notebook obligation.</done>
  <reversibility rating="reversible">A cache-only JSONL record plus one additive planning note; no sealed artifact is touched.</reversibility>
</task>

<task type="auto">
  <name>Task 4: retire D-12's escalation trigger on the record</name>
  <files>.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md, .planning/STATE.md</files>
  <action>
Write `.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md`. Additive only —
do not edit `03-CONTEXT.md`, `03-08-PLAN.md`, `03-08-DEFECTS-01.md`, `03-08-SUPPLEMENT-02.md`,
any SUMMARY, or any sealed verdict. Do not edit
`notebooks/diagnostics/curvature_field_pu_run.py` in this task: the trigger stays computed and
printed, and this note is what governs how it is read.

Sections, in this order:

**1. The decision.** Stop comparing the CAE against a plain autoencoder. The plain AE was only
ever an instrument for detecting a broken or undertrained CAE; if the CAE succeeds at
reconstruction at both the C0 and the C2 level, a relative comparison against a model there is no
intention of shipping adds nothing. A direct absolute bar is strictly better than a proxy.
Developer decision, dated.

**2. D-12 FIRED BEFORE IT WAS RETIRED — the section that must be unmissable.** Put it this high
and give it its own heading. Quote measured fact 3's `--select-only` block verbatim: on the
corrected grid (defects 1 and 2 fixed, defect 3's normalizer replaced), against the **matched**
`latent_dim=20` control, `n_charts=4` selected, `loses_reconstruction=True`,
`loses_ph_agreement=True`, `TRIGGER FIRES = True`. State in one sentence, unhedged, that the
retirement is **not** a criterion being dropped because it returned an unfavourable answer: the
answer it returned is recorded here in full, and the reason for retirement is that the comparison
was the wrong instrument for the question. Name the distinction explicitly. Note that this is the
same phase that already carries `02.6-FINDINGS.md` §4's record of a criterion changed after an
unfavourable result, which is exactly why this one is recorded rather than quietly replaced.

**3. The C0/C2 argument.** Reconstruction loss is a C0 quantity; curvature is a C2 one; small C0
error does not bound C2 error. Cite `chart_curvature.py`'s own worked example, which already
states this from the other side: a decoder learning `y = 0.7 a x^2` where the truth is
`y = a x^2` has tiny reconstruction error wherever the sampled `x` sit near zero, while its
second derivative is `1.4a` instead of `2a` — 30% curvature attenuation with no reconstruction
signal at all. A comparison conducted entirely in C0 therefore cannot rank two models on a C2
question, whichever way it comes out.

**4. The disjoint-regularizer finding.** `cae.train_cae` regularizes `model.chart_encoders`
through `cae.lipschitz_penalty`; curvature differentiates `chart_curvature.chart_decoder_map`,
which is `model.chart_decoders[i]` composed with `model.embedding_decoder`. The two sets share no
parameter, so nothing in the training objective constrains the decoder's derivatives at any
order. Give the measured consequence: `cond(g)` up to `4.886e7` on the PU grid against the roll's
1.4–122, destroying roughly seven digits of float64 in the `g^-1` contraction inside
`H = sum_jk g^jk II_jk`. Link to `03-NOTE-isometry-prior-spike.md` as the first attempt at a fix
and state its outcome in one line.

**5. The replacement criterion, direct and two-part.**
  - **C0 leg** — held-out reconstruction below an **absolute** threshold on `mse_per_dim`, no
    control model involved. Print the measured distribution as the ground for choosing it: the
    nine corrected-grid CAE cells span 8.899e-05 … 2.438e-04 with mean residual L2 norm
    (`mean_norm`) 0.2468 … 0.4112. Propose one number with the derivation shown, and mark it
    **PROPOSED, awaiting developer ratification at this task's checkpoint** — do not present a
    planner-chosen threshold as settled. State plainly that a threshold chosen after seeing this
    distribution is weaker than one pre-registered, and that this is a known cost of replacing
    the criterion mid-phase rather than something to paper over.
  - **C2 leg** — curvature fidelity against analytic `H` on the Swiss roll clearing the existing
    `ROLL_FLOOR = 0.65` on median `rho_chart`. This bar is unchanged, already declared in
    `swiss_roll_curvature_sweep_run.py`'s source before any Phase 3 number existed, and is simply
    pointed at rather than restated with new values.
  - Both legs must clear. Neither leg references a control model.

**6. Consequences, named precisely.** D-12's escalation trigger no longer drives any decision.
`notebooks/diagnostics/curvature_field_pu_run.py`'s `print_d12_trigger` stays in place and keeps
printing — its output is now context, not a trigger — and this note is what a reader must consult
to interpret it. `03-08-PLAN.md` Task 3 is the D-12 escalation checkpoint; it is **not** rewritten,
and this note records that its decision is answered here: no `d` sweep is escalated to on the
strength of the retired trigger. Anyone executing `03-08` must read this note first. List every
file that mentions D-12 so no reference is missed: `03-CONTEXT.md` (D-12 itself),
`03-07-PLAN.md`, `03-07-SUMMARY.md`, `03-08-PLAN.md`, `03-08-DEFECTS-01.md`,
`03-08-SUPPLEMENT-02.md`, `03-VALIDATION.md`, `03-GPU-RUNBOOK.md`,
`03-07-SUPPLEMENT-01.md`, `03-08-DECLARATION-01.md`. State that none of them is edited.

**7. What this note does not do.** It does not reopen, soften or reinterpret any sealed verdict.
It does not change any recorded number. It does not select `n_charts`. It does not claim the CAE
is good — it claims the plain-AE comparison was the wrong way to ask.

Then update `.planning/STATE.md` with an **`Edit`**, never a `Write` — the file is 558 lines and a
whole-file write would destroy phase records outside the edit window. Insert one short paragraph
immediately after the `03-08 first grid run INVALIDATED …` block, recording: the corrected grid
has now been re-run (15 records); D-12's trigger fired on both legs and is **retired** by
`03-NOTE-d12-retirement.md`; the replacement C0/C2 criterion; and the isometry prior spike with a
one-line outcome and a pointer to `03-NOTE-isometry-prior-spike.md`. Do not delete or reword any
existing STATE.md text.
  </action>
  <verify>
    <automated>test -f .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md && grep -q 'TRIGGER FIRES = True' .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md && grep -qi 'wrong instrument' .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md && grep -q 'ROLL_FLOOR' .planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md && grep -q '03-NOTE-d12-retirement.md' .planning/STATE.md && git diff HEAD --numstat -- .planning/STATE.md | awk 'BEGIN{a=0;d=0} {a=$1; d=$2} END{exit (a>0 && d==0)?0:1}' && echo D12_NOTE_OK</automated>
  </verify>
  <done>The note exists and carries the fired-then-retired record with the verbatim trigger output, the C0/C2 argument, the disjoint-regularizer finding, the two-part replacement criterion with its C0 number marked as proposed, and the full list of D-12 references left unedited; STATE.md gained a paragraph and lost no lines; no sealed artifact was modified.</done>
  <reversibility rating="reversible">Two additive documents; the STATE.md edit is purely additive and revertible line-for-line.</reversibility>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>
Part A: `notebooks/pu_manifold/decoder_priors.py` (isometry and conformal priors on the chart
decoder, default off, `cae.py` untouched), its test file, and
`notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py` — with the Swiss roll anchor
`rho_chart = -0.06041003026778113` reproduced exactly through the new path. The weight ladder ran
on the roll and `.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md`
carries the full table, the mechanism check and the bias check.

Part B: `.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md` retires D-12's
escalation trigger, records that it fired first on both legs of the corrected grid, and replaces
it with a direct C0/C2 criterion.
  </what-built>
  <how-to-verify>
1. Read `03-NOTE-isometry-prior-spike.md`. The mechanism verdict and the bias verdict are the two
   lines that matter. **Four decisions are yours:**
   - Adopt the prior, or not. A `FLATTENING SUSPECTED` or `MECHANISM NOT DEMONSTRATED` read-out
     means do not adopt.
   - If the isometry arm demonstrated its mechanism, whether to run the conformal arm anyway —
     the note records why it was left untaken.
   - Whether to apply the permanent seam in `cae.py`: a four-line optional `extra_loss=None`
     parameter on `train_cae`, replacing the scoped `cae.chart_loss` shim. It was deliberately
     **not** applied. The shim works and is tested; the parameter is cleaner and touches the
     sealed 02.2 file.
   - Confirm the notebook judgement: spike script now,
     `notebooks/03_swiss_roll_isometry_prior_check.ipynb` mandatory before any PU fit runs with a
     non-zero prior weight. The reasoning is in the note's own section.
2. Read `03-NOTE-d12-retirement.md`. **Ratify or replace the proposed absolute C0 threshold** —
   it is marked PROPOSED and is the one number in that note a planner should not have chosen
   alone. The measured distribution it was derived from is printed beside it.
3. Re-run the anchor if you want it in front of you:
   `.venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --anchor-check`
4. Confirm the working tree: `git status --short` should show the six files this plan touched plus
   the five pre-existing unrelated paths (`M CLAUDE.md`, `.planning/config.json`,
   `02-NOTE-phase-2-stage-on-hold.md`, `02.2-UAT.md`, `02.5-…/.gitkeep`) — those five are **not
   ours and were never staged**.
  </how-to-verify>
  <resume-signal>Type "approved", or give the four Part-A decisions and the C0 threshold.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| planning docs → future executors | A note read later is the only thing standing between a retired criterion and a reader who re-applies it |
| new prior → the measured estimand | A regularizer that constrains the wrong object can make the curvature field a function of the regularizer weight |
| new runner → sealed anchors | A re-derived fixture or metric sequence produces plausible numbers that are not comparable to any sealed one |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-Q-01 | Tampering | the estimand itself | critical | mitigate | The prior is first-order only. A curvature or Hessian-norm penalty is forbidden in the module docstring with its reasoning; the bias check (`median_magnitude_ratio`, `calibration_slope`) is a gating read-out, not a reported column |
| T-Q-02 | Tampering | `cae.py`'s RNG-order-dependent anchors | high | mitigate | `cae.py` is never edited; `git diff --quiet notebooks/pu_manifold/cae.py` is an acceptance criterion on Task 1; the anchor `-0.06041003026778113` is asserted with `==` in Task 2 |
| T-Q-03 | Tampering | the ladder's comparability | high | mitigate | Early stopping disabled for every cell and `epochs_run == max_epochs` asserted in `--summary`; a violation refuses to print the read-outs (`03-08-DEFECTS-01.md` defect 2 by construction) |
| T-Q-04 | Repudiation | D-12's retirement | high | mitigate | The verbatim `TRIGGER FIRES = True` output is quoted in the note under its own heading, above the retirement, so the record cannot be read as a criterion dropped for returning an unfavourable answer |
| T-Q-05 | Information disclosure | the spike's data | medium | mitigate | `notebooks/.cache/` is gitignored, so the full per-cell table is written inline into the committed note |
| T-Q-06 | Elevation of privilege | the scoped `cae.chart_loss` shim | medium | mitigate | `weight == 0` installs nothing; restore-on-exit including on exception is a tested behaviour; the shim never escapes the context manager |
| T-Q-07 | Tampering | the conformal branch | medium | accept | Pre-declared in the runner's source before any number exists, with its architectural reasoning (measured fact 5), so it cannot be a post-hoc axis switch |
| T-Q-08 | Tampering | package supply chain | low | accept | No package is installed by this plan — torch, numpy and pytest are already present. No `pip install` step exists, so no legitimacy gate applies |
| T-Q-09 | Tampering | unrelated working-tree files | medium | mitigate | Commits stage explicitly named paths only; `git add -A` is forbidden; the five pre-existing unrelated paths are named in the checkpoint |
</threat_model>

<verification>
```
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
.venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --anchor-check
.venv/bin/python notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py --summary
git diff --quiet notebooks/pu_manifold/cae.py
git diff --quiet notebooks/pu_manifold/chart_curvature.py
git diff --quiet notebooks/pu_manifold/__init__.py
git diff --quiet notebooks/diagnostics/swiss_roll_curvature_sweep_run.py
git status --short
```
The suite reports 289 + N passed with the pre-existing 1 skipped. The anchor check exits 0.
`git status --short` shows only this plan's six files plus the five pre-existing unrelated paths.
</verification>

<commits>
Stage explicitly named paths only. Never `git add -A`. The five pre-existing unrelated paths
(`CLAUDE.md`, `.planning/config.json`, `02-NOTE-phase-2-stage-on-hold.md`, `02.2-UAT.md`,
`02.5-…/.gitkeep`) are not ours and are never staged.

1. `feat(decoder-priors): isometry and conformal priors on the chart decoder, opt-in and default off`
   — `notebooks/pu_manifold/decoder_priors.py`, `notebooks/pu_manifold/tests/test_decoder_priors.py`
2. `feat(diagnostics): Swiss roll isometry-prior weight-ladder runner`
   — `notebooks/diagnostics/swiss_roll_isometry_prior_sweep_run.py`
3. `docs(03): record the isometry prior spike result`
   — `.planning/phases/03-decoder-curvature-field/03-NOTE-isometry-prior-spike.md`
4. `docs(03): retire D-12's escalation trigger, record the C0/C2 argument`
   — `.planning/phases/03-decoder-curvature-field/03-NOTE-d12-retirement.md`, `.planning/STATE.md`
</commits>

<success_criteria>
1. `notebooks/pu_manifold/cae.py` is byte-for-byte unchanged, and the prior is reachable on a real
   Swiss roll fit without it.
2. The anchor `rho_chart = -0.06041003026778113` at `n_charts=8, seed=0, n_points=3000` reproduces
   **exactly** through the new runner.
3. `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` is green, 289 + N passed, 1 skipped.
4. The ladder reports, per weight and per seed, `cond(g)` median and max, `rho_chart`,
   `mse_per_dim`, `median_magnitude_ratio` and `calibration_slope`, uncollapsed.
5. Every cell trained for the same number of epochs, asserted, and no cell early-stopped.
6. **The bias-check acceptance criterion.** A correct result is `rho_chart` up AND
   `median_magnitude_ratio` staying near 1. `rho_chart` rising while `median_magnitude_ratio`
   drifts systematically below 1 is the prior flattening the surface, is reported as
   `FLATTENING SUSPECTED`, and blocks any adoption recommendation. Exactly one of
   `FLATTENING SUSPECTED` / `CLEAN IMPROVEMENT` / `NO EFFECT` is printed and written into the
   note — including when it is not the favourable one.
7. The mechanism check gates interpretation: if `cond(g)` does not fall as the weight rises,
   `MECHANISM NOT DEMONSTRATED` is printed and no other column is read as evidence.
8. `03-NOTE-d12-retirement.md` records D-12 as having **fired on both legs of the corrected grid
   before** being retired, with the verbatim trigger output, and states unambiguously that the
   reason for retirement is that the comparison was the wrong instrument.
9. The replacement C0/C2 criterion is written down, its C0 number marked PROPOSED and ratified at
   the checkpoint, its C2 leg pointing at the existing `ROLL_FLOOR = 0.65`.
10. No sealed finding, summary, verdict, notebook or runner is edited; every change is additive.
</success_criteria>

<output>
Create `.planning/quick/260815-e1t-isometry-prior-spike-and-retire-cae-vs-p/SUMMARY.md` when done,
recording: the mechanism and bias verdicts with their numbers, whether the conformal branch was
taken, the chosen `LADDER_MAX_EPOCHS` and the probe's projection, the anchor reproduction, the
final test count, the four checkpoint decisions and the ratified C0 threshold.
</output>
