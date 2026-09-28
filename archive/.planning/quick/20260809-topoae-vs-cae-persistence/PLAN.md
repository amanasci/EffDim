---
phase: quick-20260809-topoae-vs-cae-persistence
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: true
requirements: [QUICK-TC-01, QUICK-TC-02, QUICK-TC-03]
files_modified:
  - notebooks/quick_topoae_vs_cae_persistence.ipynb
  - .planning/STATE.md

must_haves:
  truths:
    - "QUICK-TC-01 (does the instrument behave?) is answered by a printed line comparing each TopoAE rung against a plain-AE baseline at the SAME latent dimension — if TopoAE does not beat its dimension-matched baseline, the notebook says the measurement is suspect rather than reporting a model result"
    - "QUICK-TC-02 (how bad is the CAE, in units that mean something?) is answered against three external references, never against TopoAE alone: the dimension-matched plain-AE baseline, a chance floor from a random latent, and an ambient-perturbation ladder that expresses each model's H0 agreement as an equivalent per-point displacement in nearest-neighbour spacings"
    - "QUICK-TC-03 (invents or destroys?) is answered by loss_x_to_z and loss_z_to_x reported separately and never summed, alongside the two scale-free directional edge rates (ambient-MST edges retained in the latent MST; latent-MST edges absent from the ambient MST)"
    - "The notebook states in its own opening text, before any result, that TopoAE's training objective IS the metric being scored, so a TopoAE win over CAE is close to tautological and the informative content lies in the calibrated magnitudes and the directional split"
    - "The notebook states in its own text that every persistence result here is 0-dimensional only — connected-component merge structure via an MST edge set — and cannot see loops or voids, because no persistence library is installed and none is installed by this work"
    - "No raw topological_fidelity value is ever compared across models of differing latent dimension: every fidelity number is reported as a ratio against a plain-AE baseline at the same latent dimension, and the notebook demonstrates the dimension artifact rather than asserting it"
    - "The PU comparison's primary evaluation set is the intersection of the CAE holdout and the TopoAE holdout — the only rows held out by both — with its size computed in the notebook, and any secondary evaluation on the fuller TopoAE holdout carries a printed count of how many of those rows were CAE training rows"
    - "Every reported difference is accompanied by a resampling spread from repeated disjoint half-splits of the evaluation rows, so a gap smaller than sampling noise is reported as unresolved rather than as a result"
    - "The Swiss roll half reports across at least three training seeds with the spread shown, never a single-seed number, and reports surviving chart counts per seed"
    - "The notebook produces no verdict artifact, writes nothing into notebooks/.cache/, retrains no sealed fit, and reopens no sealed verdict (CAE_VERDICT, the 02.4 TopoAE verdict, CURVATURE_VERDICT)"
    - "Nothing under .planning/phases/02.5-local-curvature-feasibility-cae-re-gate/ is created, edited or deleted, and no 02.5 checkpoint state changes"
    - "Every pre-existing tracked file is unmodified except .planning/STATE.md's Quick Tasks Completed table; the only new tracked file is one notebook"
  artifacts:
    - "notebooks/quick_topoae_vs_cae_persistence.ipynb — one executed Jupyter notebook committed with outputs, ~14-18 cells"
    - ".planning/quick/20260809-topoae-vs-cae-persistence/SUMMARY.md"
  key_links:
    - "the 383-row CAE-holdout/TopoAE-holdout intersection <-> every PU number in the notebook (scoring the two models on different point sets, or on rows one of them trained on, makes the comparison meaningless; the intersection is computed in-notebook and its size asserted, never transcribed)"
    - "topological_fidelity's latent_unit_scale normalization <-> the ratio-against-a-dimension-matched-baseline rule (fidelity grows with latent dimension as a pure artifact — measured 277/893/1929 at d=8/20/40 on the same 383 rows and the same model family — so a raw cross-dimension comparison reports dimension, not topology)"
    - "the ambient perturbation ladder's sigma = f * median_nn / sqrt(D) <-> the printed realized median displacement norm (perturbing each of 768 coordinates by f * median_nn displaces a point by sqrt(768) * f * median_nn, ~28x the intended amount; without the sqrt(D) division and the realized-displacement printout the floor is silently meaningless)"
    - "cae_fit npz z_all (the 40-d initial-encoder embedding) <-> the CAE's only globally-comparable coordinate (chart coordinates are chart-local; pairwise distances between points in different charts carry no geometric meaning — cae.embedding_distortion raises ValueError on exactly this misuse)"
    - "cae.PlainAutoEncoder(768, d, hidden=(250,250,250), activation='silu') + cae.arrays_to_state_dict <-> the cached plain-AE baseline npz weights (a mismatched constructor loads a shape-mismatched state dict and every downstream number silently means something else)"
---

<objective>
Run one quick experiment, as a new notebook, asking whether a Topological Auto-Encoder or a Chart Auto-Encoder better preserves 0-dimensional persistent homology, on (1) the Swiss roll and (2) the PU embeddings.

Purpose: 02.5-09 found the CAE fragments the Swiss roll into chart-sized pieces joined by near-straight chords. If that fragmentation also shows up in the PU representation's connectivity skeleton, it is a concrete, measurable statement about what the CAE's global embedding does to the data — the kind of statement the 02.5 checkpoint decision needs. This experiment gates nothing, produces no verdict artifact, and reopens no sealed verdict.

The plan tracks three requirement IDs, one per question, which exist only for this quick task:
- **QUICK-TC-01** — *Does the instrument behave?* TopoAE should beat a plain-AE baseline at its own latent dimension. If it does not, the measurement is broken and no model conclusion follows.
- **QUICK-TC-02** — *How bad is the CAE, in units that mean something?* Not "worse than TopoAE" — by how much, relative to a dimension-matched baseline, a chance floor, and an external perturbation ladder.
- **QUICK-TC-03** — *Does the CAE invent structure or destroy it?* `loss_x_to_z` versus `loss_z_to_x`, reported separately and never summed, plus the two scale-free directional edge rates.

Output: one executed notebook, `notebooks/quick_topoae_vs_cae_persistence.ipynb`, committed with outputs, plus this task's SUMMARY and one row in STATE.md's Quick Tasks Completed table. Nothing else.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@notebooks/pu_manifold/topoae.py
@notebooks/02.4_swiss_roll_topoae_check.ipynb
@notebooks/02.2_swiss_roll_cae_check.ipynb
@notebooks/diagnostics/topoae_evaluate_run.py
</context>

<the_confound_state_it_do_not_bury_it>
**TopoAE's training objective is `topological_loss`. `topological_fidelity` is that same quantity evaluated on held-out data.** So "TopoAE preserves persistent homology better than a CAE" is close to tautological — it is the model that was trained to optimize the metric, being scored on the metric.

This does not make the experiment worthless. It changes what it can conclude, and the notebook must say so in its opening markdown cell, before any number, in its own words. A notebook that prints "TopoAE wins" and stops has answered nothing.

What survives the confound, and what this notebook is actually for, is QUICK-TC-02 and QUICK-TC-03: the calibrated magnitude of the CAE's distortion against references external to both models, and the direction of that distortion. Given 02.5-09, the standing prior is that the CAE **invents** structure (spurious components and gaps from chart fragmentation) rather than destroying it. Confirming or refuting that is the single most informative result available here.
</the_confound_state_it_do_not_bury_it>

<critical_constraint>
**Additive only. Nothing existing may be deleted or rewritten.**

- The only new file is `notebooks/quick_topoae_vs_cae_persistence.ipynb`. The only edit to a pre-existing tracked file is one appended row in `.planning/STATE.md`'s Quick Tasks Completed table.
- **READ-ONLY, never edited:** `notebooks/pu_manifold/{cae,topoae,curvature,curvature_probe,chart_curvature,cache,mknn,geometry_probes,subsample}.py`, every file under `notebooks/diagnostics/`, and every existing notebook. The `Edit` and `Write` tools are prohibited on all of them.
- **`src/effdim/` is not modified.** CLAUDE.md bars it for the whole v1.1 milestone.
- **No package is installed and `pyproject.toml` is not modified.** CLAUDE.md bars it for the whole milestone. If this work concludes that H1 persistence is essential, that is a *finding to write down and stop on*, not a licence to install anything.
- **`notebooks/.cache/` is read-only.** Load with `np.load` / `Path.read_text()` / `cache.cache_path(...)` for path construction only. Never call `cache.npz_cache`, `cache.json_cache`, `cache.joblib_cache`, `cae.write_cae_verdict`, `cae.write_cae_handoff`, `topoae.write_topoae_verdict`, `topoae.write_topoae_handoff`, or `topoae.clear_stale_handoff` — every one of them writes into the cache, and the last would delete a sealed artifact.
- **Sealed fits are never retrained, overwritten or re-keyed.** The three 02.2 CAE fits, the sixteen 02.4 `amend01` fits and baselines are loaded read-only.
- **Sealed verdicts are not reopened, softened or recomputed:** `CAE_VERDICT = FAIL` (02.2), the 02.4 TopoAE verdict, `CURVATURE_VERDICT = FAIL` (02.5 stage 1). This experiment produces no verdict of any kind and no `*_VERDICT` key in any artifact.
- **Phase 02.5 is untouched.** `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/` gets no new file, no edit, no deletion. 02.5-09's blocking human-verify checkpoint stays open; plans 02.5-10..13 stay blocked. This quick task neither resolves nor advances them.
- **Notebook name is `notebooks/quick_topoae_vs_cae_persistence.ipynb`.** Do NOT name it `*_swiss_roll_*_check.ipynb` — that pattern is reserved for CLAUDE.md's mandated sanity-check notebooks, which forbid all `.cache` access, and the PU half of this experiment necessarily reads cached fits.
- **Import, do not reimplement.** Every model class, training loop and persistence quantity comes from `pu_manifold.cae` and `pu_manifold.topoae`: `ChartAutoEncoder`, `PlainAutoEncoder`, `train_cae`, `train_plain_ae`, `train_topoae`, `arrays_to_state_dict`, `reconstruction_stats`, `chart_survival`, `persistence_pairs`, `pairwise_distances_f64`, `topological_fidelity`, `topological_loss`, `latent_unit_scale`, `t1_gate_value`, `rank_structure`. Set arithmetic over `persistence_pairs`' returned index array (intersection sizes, Jaccard) is notebook-level arithmetic over a tested output and is allowed inline. **If you find yourself wanting to add a new function to `pu_manifold/` with a test, stop: that is the signal this has outgrown quick mode.** Record it in the SUMMARY as a finding and do not add it.
- **Committed executed, with outputs**, end to end, no error outputs.
</critical_constraint>

<reference_facts>
Established during planning against the live repo and venv. Do not re-derive; do verify the assertions the tasks name.

**Environment.** `/home/akagi/Documents/Projects/EffDim/.venv/bin/python`, python 3.14, torch 2.13.0+cpu, numpy 2.5.1. `.venv/bin/jupyter` and `jupyter-nbconvert` present. Full suite baseline `286 passed`. `ripser`, `gudhi`, `persim`, `giotto-tda` all **ABSENT** — verified by import.

**Module entry points (all tested; `test_topoae.py` has 39 tests including `test_persistence_pairs_is_a_spanning_tree`, `test_topological_fidelity_gates_on_worse_direction`, `test_topological_fidelity_is_scale_invariant_in_the_latent`).**
- `topoae.persistence_pairs(D) -> (n-1, 2) int64` — MST edge set of a square symmetric distance array. **0-dimensional only**: connected-component merge structure. It cannot see loops or voids.
- `topoae.topological_fidelity(x, z) -> {"loss_x_to_z", "loss_z_to_x", "worse"}` — whole-set H0 agreement between an ambient array and a row-aligned latent array. `d_x` unnormalized; `d_z` computed after `z * latent_unit_scale(z)`. Deliberately never returns a sum: `loss_x_to_z` catches DESTROYED structure, `loss_z_to_x` catches INVENTED structure.
- `topoae.t1_gate_value(fid_model, fid_baseline)` — the `worse`/`worse` ratio, with a zero/non-finite denominator guard.
- `topoae.pairwise_distances_f64`, `topoae.latent_unit_scale`, `topoae.rank_structure(x, z, k)`, `topoae.train_topoae(model, x_train, cfg)`.
- `cae.ChartAutoEncoder`, `cae.PlainAutoEncoder`, `cae.train_cae`, `cae.train_plain_ae`, `cae.reconstruction_stats`, `cae.chart_survival`, `cae.arrays_to_state_dict`.

**The dimension artifact — measured during planning, on the same 383 rows, same model family (cached plain-AE baselines):**

| baseline | `loss_x_to_z` | `loss_z_to_x` |
|---|---|---|
| plain AE d=8  | 277.4  | 225.5  |
| plain AE d=20 | 892.9  | 797.0  |
| plain AE d=40 | 1928.9 | 1746.4 |

`latent_unit_scale` fixes *mean per-dimension* variance at 1, so typical latent distances grow like `sqrt(d)` and the summed squared discrepancy against a fixed ambient scale grows with `d`. **A raw fidelity value is a report about latent dimension, not about topology.** Only ratios against a plain-AE baseline at the same `d` are meaningful. This is the 02.4 evaluation runner's own stated Erratum 1 / Limitation 3.

**PU artifacts, all under `notebooks/.cache/`, all read-only, key `43cf438bc944c509`.**
- Ambient data: `subsample_20260729_a79b3460b838fd0a.npz`, array key `legacysurvey`, shape `(10000, 768)`, float64, L2-normalized onto the unit sphere.
- CAE (02.2, sealed): `cae_fit_43cf438bc944c509_seed{20260803,20260804,20260805}.npz`. Architecture `in_dim=768, embed_dim=40, chart_dim=20, n_charts=16, hidden=[250,250,250], activation="silu"`. Each npz carries `train_idx (8000,)`, `holdout_idx (2000,)`, **`z_all (10000, 40)`** — the initial encoder's global embedding for all rows, already computed — plus `p_all (10000, 16)`, `chart_argmax_all`, `y_holdout`, and the flattened state dict. Split seed `20260803`.
- TopoAE (02.4): `topoae_fit_43cf438bc944c509_amend01_seed{seed}_d{d}.npz` for `d in (8,16,20,24,32,40)` at seed `20260806`, plus seeds `20260807`/`20260808` at `d=20` only. Each carries **`z_all (10000, d)`** and `y_holdout`. Primary rung `d=20`.
- Plain-AE baselines: `topoae_baseline_43cf438bc944c509_amend01_seed{seed}_d{d}.npz`, same `(d, seed)` grid. These carry **weights and `y_holdout` only — no `z_all`** — so they must be rebuilt and re-encoded.
- Shared split: `topoae_split_43cf438bc944c509.npz` with `train_idx (8000,)`, `holdout_idx (2000,)`. Split seed `20260806`.
- The **pre-amendment** (untagged, `epochs_run=15`) topoae stems also exist on disk. They are the preserved record of a fixed stopping-rule defect. **Never read them.** Only `amend01`-tagged stems.

**The split problem, measured during planning.** The CAE and TopoAE were trained under *different* holdout splits (seeds `20260803` vs `20260806`). Of the TopoAE's 2000 holdout rows, **1617 are CAE training rows**, and the intersection held out by both models is **383 rows**. Scoring the CAE on the full TopoAE holdout hands it 81% train-seen data — a bias in the CAE's favour. The primary evaluation set is therefore the 383-row intersection. (Recompute in-notebook with `np.intersect1d`; assert the size rather than transcribing it.)

**Loading idiom, copied from `notebooks/diagnostics/topoae_evaluate_run.py` STEP 1 (verified working during planning):**
```
npz   = dict(np.load(cache.cache_path(stem, "npz")))
meta  = json.loads(cache.cache_path(meta_stem, "json").read_text())
model = cae.PlainAutoEncoder(768, d, hidden=(250,250,250), activation="silu")
model.load_state_dict(cae.arrays_to_state_dict(npz, model.state_dict()))
model.eval()
```
`torch.tensor(X, dtype=torch.float32)` for the ambient array; encode under `torch.no_grad()`.

**Timing, measured during planning.** `topological_fidelity` costs ~0.21 s at n=383 and ~7.6 s at n=2000 (the `n^2 log n` sort dominates). Budget the resampling null at the small n.

**Pilot measurements — for sizing and sanity only. The notebook recomputes everything; if its numbers differ materially from these, the notebook is right and the discrepancy is reported in the SUMMARY. Do not transcribe these into the notebook as constants or expected values.** On the 383-row intersection, fraction of ambient-MST edges also present in the latent MST: CAE `z_all` (embed 40) 0.183; plain-AE d40 0.628; TopoAE d40 0.668; plain-AE d20 0.644; TopoAE d20 0.673; plain-AE d8 0.547; TopoAE d8 0.581. Ambient perturbation ladder at correctly-scaled displacement: 0.05x median-NN -> 0.992 retained, 0.10x -> 0.961, 0.25x -> 0.940, 0.50x -> 0.859, 1.00x -> 0.660. Median ambient nearest-neighbour distance on those rows ~0.251. The instrument is not saturated and it discriminates.

**The perturbation-scaling trap.** To displace each point by a Euclidean norm of `f * median_nn` in `D = 768` dimensions, the per-coordinate Gaussian sigma is `f * median_nn / sqrt(D)`. Perturbing each coordinate by `f * median_nn` directly displaces a point by `sqrt(768) * f * median_nn` — about 28x too far, and it collapsed the retained fraction to 0.199 at a nominal "0.1x" during planning. The notebook must use the `sqrt(D)` division **and print the realized median displacement norm** as proof.

**Swiss roll configs from the two existing check notebooks (import unchanged, do not reinvent).**
- Data (both notebooks, identical): `X_raw, t = make_swiss_roll(n_samples=3000, noise=0.0, random_state=SEED)`; `X = (X_raw - X_raw.mean(axis=0)) / X_raw.std()` — one global scalar std, shape preserved. 80/20 split by `np.random.default_rng(SEED).permutation(3000)`. `VIEW = dict(elev=12, azim=-78)` and the x-z scatter are what make the spiral legible.
- CAE (`02.2_swiss_roll_cae_check.ipynb`): `ChartAutoEncoder(in_dim=3, embed_dim=8, chart_dim=2, n_charts=8, hidden=[64,64], activation="silu")`; cfg `lr=1e-3, weight_decay=1e-4, batch=64, max_epochs=300, early_stop_patience=25, early_stop_min_delta=1e-4, n_charts=8, fps_pretrain_epochs=20, lip_weight=1e-3, lip_every_n_steps=1`.
- TopoAE (`02.4_swiss_roll_topoae_check.ipynb`): `PlainAutoEncoder(3, 2, hidden=(64,64,64), activation="silu")`; cfg `lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=150, lambda_topo=0.1, warmup_frac=0.25, ramp_frac=0.25`. Plain-AE baseline: same constructor and the same cfg minus the three topo keys.
- **The two notebooks disagree on width, depth, learning rate and epoch budget.** A comparison at mismatched capacity is not a comparison; Task 3 resolves this explicitly.

**02.5-09's seed finding, which this notebook must not repeat the mistake of ignoring.** Across torch seeds 0/1/2/3 the Swiss roll CAE used 8/8/3/5 charts and its chart-decoder curvature Spearman ran -0.0604/-0.1444/0.8665/0.4250. A single-seed Swiss roll number here would be misleading.

**House style** — `notebooks/02.2_chart_autoencoder.ipynb` and `notebooks/02_k_sensitivity_refit.ipynb`: `## §N. Title` markdown headers each followed by one code cell; a §1 provenance cell printing versions, git short SHA and cwd; constants named at module level with explanatory `assert`s; `=== §N: title ===` print banners; fixed-width tables built with f-string column widths and `"=" * N` rules.
</reference_facts>

<measurement_design>
Two instruments, both computed from `persistence_pairs` / `topological_fidelity`, applied identically in both halves. Every model is reported **beside a plain-AE baseline at its own latent dimension** — never on its own, never against a baseline at a different dimension.

**Instrument A — fidelity ratio (comparable with the 02.4 gate).** `topological_fidelity(x_eval, z_model)` and the same for the dimension-matched baseline, combined by `t1_gate_value` (ratio of `worse`). Below 1.0 means better than the baseline. The two directional terms are additionally reported as their own ratios, separately, never summed.

**Instrument B — scale-free MST edge agreement (the primary instrument for QUICK-TC-03).** From the ambient MST edge set `E_x = persistence_pairs(d_x)` and the latent MST edge set `E_z = persistence_pairs(d_z)` on the same rows (`d_z` computed after `latent_unit_scale`, matching `topological_fidelity`'s own convention), report:
- `retained  = |E_x ∩ E_z| / |E_x|` — ambient merge structure the latent kept. Low means DESTROYED.
- `spurious  = |E_z \ E_x| / |E_z|` — latent merges with no ambient counterpart. High means INVENTED.
- `jaccard   = |E_x ∩ E_z| / |E_x ∪ E_z|`.
Pure set arithmetic on a tested function's output, in [0, 1], with no scale and no dimension normalization.

**Three calibrations, all required. A number without them is uninterpretable — this phase already paid for that lesson at stage 1.**

1. **Identity self-test.** Passing the ambient array itself as the latent must give `retained == 1.0` exactly and `jaccard == 1.0`. If it does not, the instrument is wired wrong and the notebook halts there.
2. **Chance floor.** A Gaussian random latent of the same dimension, same rows. Establishes where "no relationship" sits (expected near zero). Every reported value then lies on a scale with both ends pinned.
3. **Ambient perturbation ladder — the external floor.** Displace every ambient point by a Euclidean norm of `f * median_nn` for `f in (0.25, 0.5, 1.0, 2.0)` using `sigma = f * median_nn / sqrt(D)`, at several noise seeds, and recompute `retained` against the unperturbed ambient MST. This is what "same manifold, resolution-scale jitter" costs. Each model's `retained` is then bracketed against this ladder and reported as an **equivalent displacement in nearest-neighbour spacings** — the sentence a reader can actually act on.

**The resampling null (spread).** The literal construction "compute fidelity between two independent subsamples" does **not** type-check: `topological_fidelity` and `topological_loss` are row-paired — they index the same rows in both matrices — so two disjoint point sets cannot be fed to them, and doing so would silently return arithmetic over unrelated rows. State this in the notebook in one sentence and use the construction that does hold: **repeated disjoint half-splits.** For `R = 20` draws, partition the evaluation rows into two disjoint halves and recompute every statistic for every model on each half, giving 40 half-samples. Report each model's median and 5th/95th percentiles, and — the tighter and correct comparison — the **paired** per-half difference distribution (TopoAE minus CAE on the same half). A gap whose paired distribution straddles zero is reported as unresolved at this sample size, not as a result. All comparisons are made at fixed `n`: the fidelity loss is a sum over `n-1` edges and is not comparable across differing point counts.
</measurement_design>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end slice — the two instruments, all three calibrations, and one like-for-like PU trio</name>
  <precondition>All of `notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz`, `topoae_split_43cf438bc944c509.npz`, `cae_fit_43cf438bc944c509_seed20260803.npz`, `topoae_fit_43cf438bc944c509_amend01_seed20260806_d40.npz` and `topoae_baseline_43cf438bc944c509_amend01_seed20260806_d40.npz` exist and are non-empty, and `/home/akagi/Documents/Projects/EffDim/.venv/bin/jupyter` is executable. Assert this first and halt on any absence — there is no regeneration path and nothing here may be retrained.</precondition>
  <files>notebooks/quick_topoae_vs_cae_persistence.ipynb</files>
  <read_first>notebooks/pu_manifold/topoae.py lines 44-170 and 455-500 (persistence_pairs, topological_loss, topological_fidelity, latent_unit_scale — the exact normalization conventions this notebook must match), notebooks/diagnostics/topoae_evaluate_run.py lines 195-240 (the reload/rebuild/encode idiom), notebooks/02.2_chart_autoencoder.ipynb §1-§2 (the provenance and constants cell idiom)</read_first>
  <action>
Create `notebooks/quick_topoae_vs_cae_persistence.ipynb` as nbformat 4.5 with the same `kernelspec` as `notebooks/02.2_chart_autoencoder.ipynb`. This task lays one complete vertical path — framing, environment, instruments, all three calibrations, one like-for-like PU comparison, one read-out — so the whole measurement stack is proven before it is expanded. It needs no training at all, which is why it goes first.

Use the final section numbering from the outset so nothing renumbers later. This task creates cells 0 and §1, §2, §3, §4 and a partial §8. Sections §5, §6, §7 and the full §8-§9 are simply absent and get inserted at their numbered positions by Tasks 2 and 3.

**Cell 0 (markdown) — framing.** Title `# Quick experiment — TopoAE vs CAE: which preserves persistent homology better?`. In your own words, and before any number:
- The three questions, labelled QUICK-TC-01/02/03 as in this plan's objective.
- The confound, prominently: TopoAE's training objective is the topological loss, and the fidelity statistic here is that same quantity on held-out data, so a TopoAE win over the CAE is close to tautological. Say what therefore survives the confound — the calibrated magnitude and the direction of the CAE's distortion — and say that "TopoAE wins" on its own answers nothing.
- The H0-only limitation: everything measured here is 0-dimensional, an MST edge set over pairwise distances, i.e. connected-component merge structure. It cannot see loops or voids. No persistence library is installed and none is installed by this work; if H1 turns out to be essential, that is a finding for the SUMMARY, not a licence to install.
- The prohibitions binding this notebook: no training of any sealed fit, nothing written into `notebooks/.cache/`, no verdict of any kind produced, no sealed verdict reopened or recomputed.
- One sentence on where this sits: run before resolving 02.5-09's open checkpoint, gating nothing.

**§1 markdown + code — environment and provenance.** `import gc, json, subprocess, sys` and `from pathlib import Path`; `NOTEBOOK_DIR = Path.cwd()`; `sys.path.insert(0, str(NOTEBOOK_DIR))` guarded by a membership check — never import from `src/effdim/`. Assert `NOTEBOOK_DIR.name == "notebooks"` with a message naming the kernel working directory as the cause. Then `import numpy as np, torch, matplotlib.pyplot as plt`, `from pu_manifold import cache`, `from pu_manifold import cae`, `from pu_manifold import topoae`. Print a `=== Reproducibility header ===` block with python/numpy/torch versions, `git rev-parse --short HEAD`, cwd. Then probe for `ripser`, `gudhi`, `persim`, `gtda` in a try/except loop and print each as present or absent, so the H0-only limitation is **evidenced in the output**, not merely claimed in prose.

**§2 markdown + code — the two instruments, and why raw fidelity numbers do not compare across latent dimension.** Markdown: define Instrument A and Instrument B exactly as `<measurement_design>` specifies, including that `loss_x_to_z` catches destroyed structure and `loss_z_to_x` catches invented structure and that they are never summed here. Code: define three small helpers at module level in this cell — `mst_edge_set(D) -> set[tuple[int,int]]` wrapping `topoae.persistence_pairs`; `edge_agreement(x, z) -> dict` returning `retained`, `spurious`, `jaccard`, `n_edges_x`, `n_edges_z` and applying `topoae.latent_unit_scale` to `z` before distances so it matches `topological_fidelity`'s own convention; and `measure(x, z) -> dict` merging `topoae.topological_fidelity(x, z)` with `edge_agreement(x, z)`. Keep them under about 25 lines total; they are glue over tested functions, not new science. Then the dimension-artifact demonstration: load the cached plain-AE baselines at `d in (8, 20, 40)`, rebuild and encode the evaluation rows from §3, and print a three-row table of raw `loss_x_to_z`/`loss_z_to_x` showing the values climbing with `d`. Close the cell with a printed statement of the rule that follows and binds the rest of the notebook: **every fidelity number is reported only as a ratio against a plain-AE baseline at the same latent dimension.** (Order note: this cell needs §3's evaluation rows — either place §2's code cell after §3's, or split §2 into a markdown definition here and put the demonstration table in the §3 cell. Choose one and keep the section headers in numeric order.)

**§3 markdown + code — the PU evaluation set, and why it is only a few hundred rows.** Code: load `X = np.load(...)["legacysurvey"]` and build `x_all_t = torch.tensor(X, dtype=torch.float32)`. Load `topoae_split_43cf438bc944c509.npz` and `cae_fit_43cf438bc944c509_seed20260803.npz`. Compute `eval_idx = np.intersect1d(topoae_holdout, cae_holdout)` and `leaked = np.intersect1d(topoae_holdout, cae_train)`. Print: the two split sizes, `len(eval_idx)`, `len(leaked)`, and a one-line explanation that the two models were trained under different split seeds so only the intersection is genuinely held out by both, and that scoring the CAE on the full TopoAE holdout would hand it mostly train-seen rows. Assert `len(eval_idx) >= 300` with a message explaining that below that the MST is too small for the half-split null; assert the intersection is disjoint from both training sets. Bind `x_eval = x_all_t[torch.from_numpy(eval_idx)]`.

**§4 markdown + code — calibration: identity, chance, and the ambient perturbation ladder.** Code, in this order:
1. Identity self-test: `edge_agreement(x_eval, x_eval)` and assert `retained == 1.0` and `jaccard == 1.0` exactly, with a message saying the instrument is mis-wired if this fails. Print it.
2. Chance floor: a `np.random.default_rng`-drawn Gaussian latent at `d=40`, same rows, over 5 draws; print median `retained`/`jaccard`.
3. Perturbation ladder: compute `median_nn` as the median over rows of the smallest off-diagonal ambient distance. For `f in (0.25, 0.5, 1.0, 2.0)` and 5 noise seeds each, draw `sigma = f * median_nn / np.sqrt(D)` Gaussian noise, recompute the perturbed MST, and report median `retained` against the unperturbed ambient MST — **and print the realized median per-point displacement norm alongside its ratio to `median_nn`**, which must come out at approximately `f`. Assert that realized ratio is within 20% of `f` for every rung, with a message naming the `sqrt(D)` scaling as the thing that broke if it fails. Assert `retained` is non-increasing in `f`. Print a fixed-width ladder table.
4. Define and print a helper `equivalent_displacement(retained_value)` that reports which two rungs of the ladder a given `retained` falls between, in nearest-neighbour spacings, so every later model row can be read in that unit.

**§8 markdown + code (partial, this task) — the like-for-like PU trio.** Markdown: state the pairing and why it is the like-for-like one. The CAE's only globally-comparable coordinate is the 40-dimensional initial-encoder embedding `z_all`; its `chart_dim=20` coordinates are chart-local, so a pairwise distance between two points in different charts carries no geometric meaning — `cae.embedding_distortion` raises `ValueError` on exactly that misuse. So the CAE's comparable representation is 40-d, and the like-for-like TopoAE rung is `d=40`, with `topoae_baseline ... _d40` as the shared denominator for both. Note explicitly that TopoAE's own primary rung is `d=20` and that comparing the CAE against it would not be like-for-like. Code: measure three models on `x_eval` — CAE `z_all[eval_idx]` from seed 20260803, TopoAE `z_all[eval_idx]` from `_amend01_seed20260806_d40`, and the rebuilt-and-encoded plain-AE `d40` baseline. Print a fixed-width table with, per model: `loss_x_to_z`, `loss_z_to_x`, `worse`, the `t1_gate_value` ratio against the d40 baseline, `retained`, `spurious`, `jaccard`, and the `equivalent_displacement` reading. Then print the machine-readable sentinel on one line at full precision:

`TRIO n_eval=<int> cae_retained=<f> topoae_retained=<f> plain_retained=<f> cae_ratio=<f> topoae_ratio=<f> chance_retained=<f>`

Task 2's and Task 3's external verifiers parse this line out of the committed outputs, so the format is load-bearing. Also print, on their own lines, `FLOOR f=<f> retained=<f> realized_disp_ratio=<f>` for each ladder rung.

Then execute the notebook end to end and leave its outputs stored in the file.

Do not draw a conclusion in this task beyond what the numbers show; the interpretive read-out is §9, written in Task 3.
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/quick_topoae_vs_cae_persistence.ipynb; CB=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB" && CA=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CB" = "$CA" ] && [ -z "$(git -C $REPO diff --name-only --diff-filter=MDR HEAD -- . ':!.planning')" ] && $REPO/.venv/bin/python - "$NB" <<'PY'
import json, pathlib, re, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert code, "notebook has no code cells"
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [tk for tk in ("npz_cache(", "json_cache(", "joblib_cache(", "write_cae_verdict",
                        "write_cae_handoff", "write_topoae_verdict", "write_topoae_handoff",
                        "clear_stale_handoff", "train_topoae(", "train_cae(", "train_plain_ae(",
                        "pip install", "src.effdim", "from effdim") if tk in src]
assert not banned, f"forbidden call in executable source: {banned}"
assert "_amend01_" in src, "must read only amend01-tagged topoae stems"
assert not re.search(r"topoae_(fit|baseline)_43cf438bc944c509_seed", src), "pre-amendment stem referenced"
for fn in ("persistence_pairs", "topological_fidelity", "latent_unit_scale", "t1_gate_value",
           "arrays_to_state_dict", "PlainAutoEncoder"):
    assert fn in src, f"library entry point not called: {fn}"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", []) if o.get("output_type") == "stream")
m = re.search(r"TRIO n_eval=(\d+) cae_retained=(\S+) topoae_retained=(\S+) plain_retained=(\S+) cae_ratio=(\S+) topoae_ratio=(\S+) chance_retained=(\S+)", text)
assert m, "TRIO sentinel absent from stored outputs"
n_eval = int(m.group(1)); vals = [float(m.group(i)) for i in range(2, 8)]
assert n_eval >= 300, f"evaluation set too small: {n_eval}"
for v in vals[:3] + [vals[5]]:
    assert 0.0 <= v <= 1.0, f"retained fraction outside [0,1]: {v}"
floors = re.findall(r"FLOOR f=(\S+) retained=(\S+) realized_disp_ratio=(\S+)", text)
assert len(floors) >= 4, f"perturbation ladder has too few rungs: {len(floors)}"
fs = [float(a) for a, _, _ in floors]; rs = [float(b) for _, b, _ in floors]; ds = [float(c) for _, _, c in floors]
assert fs == sorted(fs), "ladder rungs not in increasing f"
assert all(rs[i] >= rs[i+1] - 1e-12 for i in range(len(rs)-1)), f"retained not non-increasing in f: {rs}"
for f, d in zip(fs, ds):
    assert abs(d - f) <= 0.2 * f, f"realized displacement ratio {d} != nominal {f} — sqrt(D) scaling wrong"
for probe in ("ripser", "gudhi"):
    assert probe in text, f"H0-only limitation not evidenced in output: {probe} probe missing"
print("TASK1_OK", len(code), "code cells,", "n_eval =", n_eval)
PY</automated>
  </verify>
  <done>The notebook exists, executes clean end to end, and its stored outputs contain: the absence of every persistence library printed as evidence; an identity self-test at exactly 1.0; a chance floor; a four-rung perturbation ladder whose realized displacements match their nominal multiples within 20% and whose retained fractions are non-increasing; and one like-for-like PU trio (CAE embed-40, TopoAE d40, plain-AE d40) reported on both instruments against the shared d40 baseline. `notebooks/.cache/` is byte-identical before and after, no pre-existing tracked file is modified, no training or cache-write call appears in executable source, no pre-amendment topoae stem is referenced, and `TASK1_OK` prints.</done>
</task>

<task type="auto">
  <name>Task 2: Expand the PU half — full TopoAE ladder, all three CAE seeds, the directional split, and the resampling null</name>
  <files>notebooks/quick_topoae_vs_cae_persistence.ipynb</files>
  <read_first>notebooks/diagnostics/topoae_evaluate_run.py lines 279-330 and 388-425 (how the sealed run assembles per-rung tables and baseline-relative ratios — copy the shape, not the gating), notebooks/pu_manifold/topoae.py lines 535-575 (the ratio helpers and their zero-denominator guard)</read_first>
  <action>
Insert §5, §6 and §7 at their reserved positions and expand §8. Do not renumber existing headers. Do not alter cells 0 or §1-§4 except where the §8 sentinel requires it.

**§5 markdown + code, between §4 and §6 — the TopoAE ladder against dimension-matched baselines (QUICK-TC-01).** Markdown: this is the known-answer check on the instrument. TopoAE was trained to minimise this quantity, so it should beat a plain AE at its own latent dimension; if it does not, the measurement is broken and no model conclusion follows from anything else in the notebook. Code: for every `d in (8, 16, 20, 24, 32, 40)` at seed 20260806, load the TopoAE `z_all` and rebuild-and-encode the matching plain-AE baseline, measure both on `x_eval`, and print a fixed-width row per rung with: `worse` for each, the `t1_gate_value` ratio, `retained` for each and their difference, `spurious` for each, and the `equivalent_displacement` reading for both. Emit one sentinel line per rung:

`LADDER d=<int> topoae_worse=<f> plain_worse=<f> ratio=<f> topoae_retained=<f> plain_retained=<f>`

Then print a single explicit verdict-free statement of whether TopoAE beats its dimension-matched baseline at every rung, at some rungs, or none, and emit `Q1 rungs_won=<int> rungs_total=<int>`. Do not soften a negative: if the ladder shows TopoAE failing to beat its own baseline, say plainly that the instrument's behaviour is in question and that the rest of the notebook's comparisons inherit that doubt.

**§6 markdown + code, between §5 and §7 — the CAE across its three sealed seeds, and the directional split (QUICK-TC-02, QUICK-TC-03).** Code: for each of the three sealed CAE fits (seeds 20260803/04/05), measure `z_all[eval_idx]` on `x_eval` and print `loss_x_to_z`, `loss_z_to_x`, their ratio to each other, the `t1_gate_value` against the shared d40 baseline, `retained`, `spurious`, `jaccard`, and the equivalent-displacement reading. Print the across-seed spread (min/median/max) for each column, so no single-seed CAE number stands alone.

Then the directional analysis, which is this notebook's most informative single result. Report, side by side for the CAE, the TopoAE d40 and the plain-AE d40:
- `loss_x_to_z / baseline_loss_x_to_z` and `loss_z_to_x / baseline_loss_z_to_x` — the two directions as **separate** baseline-relative ratios. Never sum them. Guard against a zero or non-finite denominator and raise rather than emitting an `inf`.
- `retained` (ambient merge structure kept) and `spurious` (latent merges with no ambient counterpart), which are scale-free and carry none of the fidelity statistic's dimension sensitivity.
Print an explicit reading of which failure mode dominates for the CAE, in the notebook's own words, from the numbers actually measured — INVENTS if the invented-structure side dominates, DESTROYS if the destroyed side does, BOTH or NEITHER as the numbers warrant. State the standing prior from 02.5-09 (chart fragmentation into pieces joined by near-straight chords predicts INVENTS) and say plainly whether these numbers confirm or refute it. **Refuting the prior is a perfectly good outcome and must be reported as readily as confirming it.** Emit `Q3 verdict=<INVENTS|DESTROYS|BOTH|NEITHER> cae_xz_ratio=<f> cae_zx_ratio=<f> cae_retained=<f> cae_spurious=<f>`.

**§7 markdown + code, between §6 and §8 — the resampling null.** Markdown: state in one or two sentences why the literal "fidelity between two independent subsamples" construction is not used — `topological_fidelity` and `topological_loss` are row-paired, indexing the same rows in both distance matrices, so two disjoint point sets cannot be passed to them and doing so would return arithmetic over unrelated rows — and that repeated disjoint half-splits give the spread instead. Note that all comparisons hold `n` fixed because the fidelity loss is a sum over `n-1` edges. Code: for `R = 20` draws with a fixed seed, partition `eval_idx` into two disjoint halves and recompute, on each of the 40 half-samples, `retained` and the fidelity ratio for the CAE (seed 20260803), TopoAE d40 and plain-AE d40. Print per model the median and 5th/95th percentiles. Then the paired comparison: per half-sample, `topoae_retained - cae_retained`, and report its median and 5th/95th percentiles. Print an explicit statement of whether the interval excludes zero, and therefore whether the gap is resolved at this sample size or not. Emit `NULL stat=retained model=<name> p05=<f> med=<f> p95=<f>` per model and `NULL_PAIRED diff=topoae_minus_cae p05=<f> med=<f> p95=<f> resolved=<true|false>`. Budget note: `topological_fidelity` costs ~0.2 s at n≈380 and ~7.6 s at n=2000, so run the null at the half-split size, not on the full holdout.

**§8 expansion — the secondary evaluation set, with its bias declared.** Append to §8: repeat the like-for-like trio on the full 2000-row TopoAE holdout, printing first the count of those rows that were CAE **training** rows and one sentence stating that this set is biased **in the CAE's favour**, so a CAE result that is bad here is bad despite the advantage. Emit `SECONDARY n_eval=<int> n_cae_train_leak=<int> cae_retained=<f> topoae_retained=<f> plain_retained=<f>`. Keep the §8 `TRIO` line from Task 1 unchanged; this is an additional line, not a replacement.

Re-execute the notebook end to end.
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/quick_topoae_vs_cae_persistence.ipynb; CB=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB" && CA=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CB" = "$CA" ] && [ -z "$(git -C $REPO diff --name-only --diff-filter=MDR HEAD -- . ':!.planning')" ] && $REPO/.venv/bin/python - "$NB" <<'PY'
import json, pathlib, re, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [tk for tk in ("npz_cache(", "json_cache(", "joblib_cache(", "write_cae_verdict",
                        "write_cae_handoff", "write_topoae_verdict", "write_topoae_handoff",
                        "clear_stale_handoff", "train_topoae(", "train_cae(", "train_plain_ae(",
                        "pip install") if tk in src]
assert not banned, f"forbidden call in executable source: {banned}"
assert not re.search(r"topoae_(fit|baseline)_43cf438bc944c509_seed", src), "pre-amendment stem referenced"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", []) if o.get("output_type") == "stream")
rungs = re.findall(r"LADDER d=(\d+) topoae_worse=(\S+) plain_worse=(\S+) ratio=(\S+) topoae_retained=(\S+) plain_retained=(\S+)", text)
assert len(rungs) == 6, f"expected 6 ladder rungs, got {len(rungs)}"
assert sorted(int(r[0]) for r in rungs) == [8, 16, 20, 24, 32, 40], [r[0] for r in rungs]
for d, tw, pw, ratio, tr, pr in rungs:
    tw, pw, ratio = float(tw), float(pw), float(ratio)
    assert pw > 0 and abs(ratio - tw / pw) <= 1e-6 * max(1.0, abs(ratio)), f"d={d}: ratio inconsistent with its own operands"
    assert 0.0 <= float(tr) <= 1.0 and 0.0 <= float(pr) <= 1.0, f"d={d}: retained outside [0,1]"
q1 = re.search(r"Q1 rungs_won=(\d+) rungs_total=(\d+)", text); assert q1, "Q1 sentinel absent"
assert int(q1.group(2)) == 6, "Q1 must be reported over all six rungs"
q3 = re.search(r"Q3 verdict=(INVENTS|DESTROYS|BOTH|NEITHER) cae_xz_ratio=(\S+) cae_zx_ratio=(\S+) cae_retained=(\S+) cae_spurious=(\S+)", text)
assert q3, "Q3 sentinel absent or verdict outside the allowed set"
for g in (2, 3, 4, 5):
    v = float(q3.group(g)); assert v == v and abs(v) != float("inf"), "non-finite value in Q3 sentinel"
nulls = re.findall(r"NULL stat=retained model=(\S+) p05=(\S+) med=(\S+) p95=(\S+)", text)
assert len(nulls) >= 3, f"resampling null missing models: {len(nulls)}"
for name, p05, med, p95 in nulls:
    a, b, c = float(p05), float(med), float(p95)
    assert a <= b <= c, f"{name}: percentiles out of order"
    assert 0.0 <= a and c <= 1.0, f"{name}: retained percentiles outside [0,1]"
pair = re.search(r"NULL_PAIRED diff=topoae_minus_cae p05=(\S+) med=(\S+) p95=(\S+) resolved=(true|false)", text)
assert pair, "paired null sentinel absent"
p05, p95, resolved = float(pair.group(1)), float(pair.group(3)), pair.group(4) == "true"
assert resolved == (p05 > 0 or p95 < 0), "resolved flag disagrees with its own interval straddling zero"
sec = re.search(r"SECONDARY n_eval=(\d+) n_cae_train_leak=(\d+) ", text)
assert sec and int(sec.group(2)) > 0, "secondary set must declare its CAE-training-row count"
assert re.search(r"TRIO n_eval=(\d+)", text), "primary TRIO sentinel lost"
print("TASK2_OK", len(code), "code cells; Q1", q1.group(1) + "/" + q1.group(2), "Q3", q3.group(1))
PY</automated>
  </verify>
  <done>The notebook reports all six TopoAE rungs against dimension-matched plain-AE baselines with each ratio internally consistent with its own operands; all three sealed CAE seeds with their spread; the two fidelity directions as separate baseline-relative ratios that are never summed, alongside the scale-free retained/spurious rates and an explicit INVENTS/DESTROYS/BOTH/NEITHER reading measured rather than assumed; a 20-draw disjoint-half-split null whose paired interval determines the printed `resolved` flag; and a secondary 2000-row evaluation declaring how many of its rows the CAE trained on. The cache is unchanged, nothing pre-existing is modified, and `TASK2_OK` prints.</done>
</task>

<task type="auto">
  <name>Task 3: Swiss roll half at matched capacity across seeds, the closing read-out, and commit</name>
  <files>notebooks/quick_topoae_vs_cae_persistence.ipynb, .planning/STATE.md</files>
  <read_first>notebooks/02.2_swiss_roll_cae_check.ipynb code cells 4, 6, 8, 10, 12 (data generation, CAE cfg, the two-panel 3-D + x-z plot idiom, the matched-baseline comparison, the chart-assignment plot), notebooks/02.4_swiss_roll_topoae_check.ipynb code cells 4, 6, 12 (the TopoAE cfg and the shared-ambient-pairing edge-agreement idiom)</read_first>
  <action>
Insert §9 (Swiss roll) between §8 and the close, then add §10 (read-out) and §11 (limitations), execute clean, and commit.

**§9 markdown + code — the Swiss roll half.**

Markdown, first: state the capacity problem and how it is resolved. The two existing check notebooks train at different widths, depths, learning rates and epoch budgets (CAE: `hidden=[64,64]`, `lr=1e-3`, 300 epochs; TopoAE: `hidden=(64,64,64)`, `lr=3e-4`, 150 epochs), and a comparison at mismatched capacity is not a comparison. This notebook therefore runs **one common protocol** across every Swiss roll model — `hidden=(64,64,64)`, `lr=3e-4`, `weight_decay=1e-4`, `batch=64`, `max_epochs=300`, `early_stop_patience=25`, `early_stop_min_delta=1e-4` — keeping only each model's own intrinsic terms (the CAE keeps `fps_pretrain_epochs=20` and `lip_weight=1e-3`, which are part of the model, not a capacity advantage; the TopoAE keeps `lambda_topo=0.1`, `warmup_frac=0.25`, `ramp_frac=0.25`). Record the deviation from each source notebook explicitly in the markdown, and note that the epoch budget is the larger of the two so neither model is truncated relative to its own reference.

Data: `make_swiss_roll(n_samples=3000, noise=0.0, random_state=DATA_SEED)` with `DATA_SEED` fixed, centred and divided by one global scalar std. One fixed 80/20 split shared by every model and every seed — only the **training** seed varies, never the data or the split. Plot the input as a 3-D scatter and an x-z scatter side by side, coloured by the arc-length parameter `t`, with `VIEW = dict(elev=12, azim=-78)`.

Models per training seed, exactly parallel to the PU half so the two halves answer the same question the same way:
- `cae.ChartAutoEncoder(in_dim=3, embed_dim=8, chart_dim=2, n_charts=8, hidden=[64,64,64], activation="silu")` — scored on its **global 8-d embedding** `model.encode(x)`, for the same reason as the PU half: chart coordinates are chart-local and not globally comparable.
- `cae.PlainAutoEncoder(3, 8, hidden=(64,64,64))` — the CAE's dimension-matched denominator.
- `cae.PlainAutoEncoder(3, 2, ...)` trained by `topoae.train_topoae` — the TopoAE at the roll's true intrinsic dimension.
- `cae.PlainAutoEncoder(3, 2, ...)` trained by `cae.train_plain_ae` — the TopoAE's dimension-matched denominator, and the CLAUDE.md-required matched baseline.

Seeds: `SEEDS = (0, 1, 2, 3)` — the same torch seeds 02.5-09 used, so the chart-count spread here is directly comparable to that finding. **Budget guard:** train seed 0's four models first, print the measured wallclock and a projection for the full seed set; if the projection exceeds 20 minutes, drop to `SEEDS = (0, 1, 2)` and print that the reduction happened and why. Never drop below three seeds — the spread is the point, and a single-seed Swiss roll number would be misleading given 02.5-09.

Per seed, measure every model on the held-out rows with the same `measure()` helper from §2 and report: `retained`, `spurious`, `jaccard`, the fidelity ratio against the dimension-matched plain AE, `cae.reconstruction_stats` `mse_per_dim`, mean relative reconstruction error, and — for the CAE — `cae.chart_survival(model, prune_tol=1e-2)`'s surviving count. Print a per-seed table and then the across-seed min/median/max for each column. Emit one sentinel line per seed and model:

`SWISS seed=<int> model=<name> latent_d=<int> retained=<f> ratio=<f> rel_err=<f> charts=<int>/<int>`

(use `charts=0/0` for the models that have none).

Plots, per CLAUDE.md's Swiss roll requirements: original and reconstruction side by side, both as 3-D scatter and as x-z scatter, coloured by `t`, for the CAE and the TopoAE at one representative seed. Plus the 2-D TopoAE latent scatter coloured by `t`, so a reader can see whether the roll unrolled with its colour ordering intact. Note in the markdown that the CAE's global embedding is 8-d and so has no 2-D latent scatter, which is itself part of the finding.

Close §9 with the run's total training wallclock and the `early_stopped` / `wallclock_truncated` flags for every fit.

**§10 markdown + code — the read-out.** Answer the three questions explicitly, each in its own short paragraph, using only numbers printed above and naming them:
- **QUICK-TC-01** — did TopoAE beat its dimension-matched baseline, on the PU ladder and on the Swiss roll? If not, say the instrument's behaviour is in question and that every comparison below inherits that doubt.
- **QUICK-TC-02** — how far is the CAE from the plain-AE baseline at its own dimension, where does it sit between the chance floor and 1.0, and what per-point displacement in nearest-neighbour spacings is it equivalent to? Explicitly distinguish the two cases the calibration exists to separate: a CAE marginally worse than TopoAE but far better than the baseline is a very different finding from one at or below the baseline.
- **QUICK-TC-03** — invents or destroys, from the directional numbers, with the 02.5-09 prior confirmed or refuted.
Then one paragraph on what the resampling null says about which of these gaps are resolved at this sample size and which are not. Emit `ANSWERS q1=<PASS|FAIL|MIXED> q2=<f> q3=<INVENTS|DESTROYS|BOTH|NEITHER> resolved=<true|false>` where `q2` is the CAE's fidelity ratio against its dimension-matched baseline on the primary PU evaluation set.

**§11 markdown — limitations, no code cell.** One bullet each, factual:
- H0 only. This is an MST edge set: connected-component merge structure. It says nothing about loops or voids. No persistence library is installed and none was installed. If the question that matters is H1, this notebook cannot answer it — say so as the finding.
- The confound, restated: TopoAE optimised this objective and the CAE did not.
- The split mismatch: the two models were trained under different holdout seeds, the primary evaluation set is only the few hundred rows held out by both, and the larger secondary set is biased in the CAE's favour.
- The CAE's `chart_dim=20` coordinates were deliberately not scored, and why. The CAE's globally-comparable representation is 40-d, not 20-d.
- Cross-dimension comparisons rest on a ratio to a dimension-matched baseline, which controls the scale artifact but not the fact that a higher-dimensional bottleneck faces an easier task.
- Baseline seed coverage: the cached plain-AE baselines have three seeds only at `d=20`, one seed elsewhere.
- No dimension-matched CAE variant was trained on the Swiss roll (an `embed_dim=2` CAE), so the CAE row there is read through the same ratio rule as the PU half rather than a direct 2-d match.
- This experiment gates nothing, produces no verdict artifact, and changes no sealed verdict or open checkpoint.

**Finally:** re-execute the notebook end to end one last time so committed outputs correspond exactly to committed source. Append one row to `.planning/STATE.md`'s **Quick Tasks Completed** table only — `| 20260809-topoae-vs-cae-persistence | TopoAE vs CAE: which preserves persistent homology better (Swiss roll + PU) | 2026-08-09 | <short-sha> | [20260809-topoae-vs-cae-persistence](./quick/20260809-topoae-vs-cae-persistence/) |` — using `Edit` with a scoped replacement, never `Write`, and touching no other line of STATE.md. Then commit the notebook, the plan directory and the STATE.md row. Do not `git add` anything else, and add nothing under `.planning/phases/02.5-*/`.
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/quick_topoae_vs_cae_persistence.ipynb; CB=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 --ExecutePreprocessor.kernel_name=python3 "$NB" && CA=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CB" = "$CA" ] && [ -z "$(git -C $REPO diff --name-only --diff-filter=MDR HEAD -- . ':!.planning')" ] && [ -z "$(git -C $REPO status --porcelain -- .planning/phases/02.5-local-curvature-feasibility-cae-re-gate/)" ] && [ "$(git -C $REPO diff --numstat HEAD -- .planning/STATE.md | awk '{print $2}')" = "0" ] && $REPO/.venv/bin/python - "$NB" <<'PY'
import json, pathlib, re, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    assert c.get("outputs"), "a code cell stored no output"
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
for n in range(1, 12):
    assert re.search(rf"##\s*§{n}\.", md), f"section header §{n} missing"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [tk for tk in ("npz_cache(", "json_cache(", "joblib_cache(", "write_cae_verdict",
                        "write_cae_handoff", "write_topoae_verdict", "write_topoae_handoff",
                        "clear_stale_handoff", "pip install", "subprocess.run([\"pip") if tk in src]
assert not banned, f"forbidden call in executable source: {banned}"
assert ".cache" not in src.replace("cache.cache_path", "").replace("from pu_manifold import cache", ""), "direct .cache path manipulation outside cache.cache_path"
assert not re.search(r"topoae_(fit|baseline)_43cf438bc944c509_seed", src), "pre-amendment stem referenced"
assert "make_swiss_roll" in src and "train_topoae(" in src and "train_cae(" in src and "train_plain_ae(" in src, "Swiss roll half did not train all model families"
assert "chart_survival" in src, "CAE chart survival not reported"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", []) if o.get("output_type") == "stream")
sw = re.findall(r"SWISS seed=(\d+) model=(\S+) latent_d=(\d+) retained=(\S+) ratio=(\S+) rel_err=(\S+) charts=(\d+)/(\d+)", text)
seeds = sorted({int(r[0]) for r in sw}); models = {r[1] for r in sw}
assert len(seeds) >= 3, f"Swiss roll ran too few seeds: {seeds}"
assert len(models) >= 4, f"Swiss roll model set incomplete: {models}"
for r in sw:
    assert 0.0 <= float(r[3]) <= 1.0, f"retained outside [0,1]: {r}"
    assert float(r[4]) > 0.0, f"non-positive ratio: {r}"
per_seed = {s: {r[1] for r in sw if int(r[0]) == s} for s in seeds}
assert all(v == models for v in per_seed.values()), f"model set not identical across seeds: {per_seed}"
a = re.search(r"ANSWERS q1=(PASS|FAIL|MIXED) q2=(\S+) q3=(INVENTS|DESTROYS|BOTH|NEITHER) resolved=(true|false)", text)
assert a, "ANSWERS sentinel absent or a field outside its allowed set"
assert float(a.group(2)) == float(a.group(2)), "q2 is not a finite number"
for prior in ("TRIO n_eval=", "LADDER d=", "Q1 rungs_won=", "Q3 verdict=", "NULL_PAIRED", "SECONDARY n_eval=", "FLOOR f="):
    assert prior in text, f"sentinel lost from an earlier task: {prior}"
for term in ("H0", "loop"):
    assert term in md, f"H0-only limitation not stated in the notebook's own text: {term}"
print("TASK3_OK", len(code), "code cells;", len(seeds), "seeds;", a.group(1), a.group(3), "resolved=" + a.group(4))
PY</automated>
  </verify>
  <done>The notebook carries §1 through §11; the Swiss roll half trained all four model families under one common protocol across at least three seeds with identical model coverage per seed, reported chart survival, and produced the CLAUDE.md-required side-by-side 3-D and x-z plots coloured by `t` plus the 2-D latent scatter; §10 answers all three questions with an `ANSWERS` sentinel whose fields are drawn from the allowed sets; §11 states the H0-only limitation and the rest in the notebook's own text; every earlier sentinel survives; `notebooks/.cache/` is byte-identical before and after; `git status` under `.planning/phases/02.5-*/` is empty; `.planning/STATE.md` has exactly one added line and zero deletions; no other pre-existing tracked file is modified; and `TASK3_OK` prints. The notebook, the plan directory and the STATE.md row are committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| cached artifact -> notebook | `notebooks/.cache/*.npz` deserialized by `np.load`; archives produced locally by this repo's own runners |
| notebook -> cached artifact | any write path back into `.cache/` would mutate or delete a sealed 02.2 / 02.4 artifact |
| notebook -> repository working tree | `nbconvert --inplace` writes into `notebooks/`; a wrong path or whole-file write destroys existing tracked work |
| quick task -> phase 02.5 state | 02.5-09 sits at an open blocking checkpoint with 02.5-10..13 behind it; any write into that phase directory corrupts a blocked workflow |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-TC-01 | Tampering | notebook writing into or deleting from `notebooks/.cache/` (`clear_stale_handoff` unlinks a sealed handoff) | critical | mitigate | `<critical_constraint>` forbids every cache-write and handoff-delete entry point by name; every task's `<verify>` greps comment-stripped executable source for them and hashes the whole cache tree (path, size, mtime) before and after execution, requiring equality |
| T-TC-02 | Tampering | accidental edit or deletion of an existing notebook, runner or module | high | mitigate | `files_modified` is one new notebook plus one appended STATE.md row; every task's `<verify>` runs `git diff --diff-filter=MDR HEAD` outside `.planning/` and requires it empty; Task 3 additionally requires STATE.md's deleted-line count to be zero |
| T-TC-03 | Tampering | writing into the blocked 02.5 phase directory | high | mitigate | Task 3's `<verify>` requires `git status --porcelain -- .planning/phases/02.5-*/` to be empty; the constraint block names the prohibition explicitly |
| T-TC-04 | Repudiation | reading the pre-amendment (buggy, `epochs_run=15`) topoae stems and reporting them as the 02.4 result | high | mitigate | Tasks 1 and 2 `<verify>` assert `_amend01_` appears in source and that no bare `topoae_{fit,baseline}_43cf438bc944c509_seed` stem does |
| T-TC-05 | Repudiation | reporting a raw fidelity value across differing latent dimensions, i.e. reporting dimension as if it were topology | high | mitigate | the ratio-to-dimension-matched-baseline rule is stated in `<measurement_design>`, demonstrated in §2 from measured data, and enforced by Task 2's verifier recomputing each ladder ratio from its own printed operands |
| T-TC-06 | Repudiation | a perturbation floor silently mis-scaled by `sqrt(D)`, making the external calibration meaningless | high | mitigate | §4 prints the realized median displacement norm; Task 1's verifier asserts it matches the nominal multiple within 20% and that retained is non-increasing in `f` |
| T-TC-07 | Repudiation | notebook outputs not corresponding to committed source | medium | mitigate | Task 3 re-executes immediately before commit; verifiers assert contiguous `execution_count` from 1, that every code cell stored output, and that no cell holds an error output |
| T-TC-08 | Repudiation | a single-seed Swiss roll number read as a stable result, contradicting 02.5-09's measured seed instability | medium | mitigate | Task 3 requires at least three training seeds with identical model coverage per seed, asserted by the verifier over the `SWISS` sentinel lines, and requires the across-seed spread to be printed |
| T-TC-09 | Elevation of privilege | scope creep into new tested module functions, turning a quick task into an unplanned phase | medium | mitigate | `<critical_constraint>` bars adding functions to `pu_manifold/` and directs the executor to record the pressure in the SUMMARY instead; `files_modified` would have to change for it to happen, which the `git diff` gate catches |
| T-TC-10 | Denial of service | Swiss roll training overrunning (16-20 fits, per-batch MST during TopoAE training) | medium | mitigate | Task 3 times seed 0 first and projects; above 20 minutes the seed set drops to three and the reduction is printed; nbconvert runs under an explicit 5400 s timeout |
| T-TC-11 | Information disclosure | committed outputs leaking absolute paths / environment detail | low | accept | the §1 provenance cell prints cwd and versions by design, matching the repo's other committed notebooks |
| T-TC-12 | Tampering | `np.load` on an `.npz` from an untrusted source | low | accept | every archive is produced by this repository's own runners under `notebooks/.cache/`; `allow_pickle` stays at its safe default and no object arrays are stored |
| T-TC-SC | Tampering | npm/pip/cargo installs | high | mitigate | **No package is installed.** CLAUDE.md bars `pyproject.toml` edits for the whole v1.1 milestone; every dependency is already present in `.venv`. Every task's `<verify>` greps executable source for an install call. The four absent persistence libraries are probed and printed as absent, never installed — no `[ASSUMED]`/`[SUS]` package is introduced, so no legitimacy checkpoint applies |
</threat_model>

<verification>
1. All three tasks print their `TASK{N}_OK` sentinel.
2. `git status --porcelain -- . ':!.planning'` shows exactly one entry: the new `notebooks/quick_topoae_vs_cae_persistence.ipynb`. Under `.planning/`, only this quick-task directory and one added STATE.md line.
3. `find notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum` is identical before and after a full execution — the cache is provably read-only to this work, including the sealed `cae_verdict_*`, `topoae_verdict_*` and `curvature_verdict_stage1_*` artifacts.
4. `git status --porcelain -- .planning/phases/02.5-local-curvature-feasibility-cae-re-gate/` is empty; 02.5-09's checkpoint is still open and 02.5-10..13 still blocked.
5. `.venv/bin/python -m pytest` still reports `286 passed` — no module under `notebooks/pu_manifold/` was touched, so this must be unchanged.
6. Read the notebook top to bottom: the confound and the H0-only limitation appear before any result; every fidelity number carries a dimension-matched baseline beside it; every headline difference carries a resampling interval; the three questions are answered by name in §10; §11 states what the experiment cannot support.
7. No file anywhere contains a new `*_VERDICT` key, and no verdict artifact is created.
8. Grep the notebook for the strings `CAE_VERDICT`, `TOPOAE_VERDICT`, `CURVATURE_VERDICT`: they may appear only inside markdown prose describing sealed prior results, never as a value this notebook computes, revises or writes.
</verification>

<success_criteria>
- One new executed notebook, `notebooks/quick_topoae_vs_cae_persistence.ipynb`, committed with outputs, answering QUICK-TC-01/02/03 by name
- The confound (TopoAE optimised the metric it is scored on) and the H0-only limitation stated in the notebook's own text before any result, with the four absent persistence libraries printed as evidence
- No raw fidelity value compared across latent dimensions; the dimension artifact demonstrated from measured data and every fidelity number reported as a ratio against a plain-AE baseline at the same dimension
- Three calibrations present and behaving: identity self-test at exactly 1.0, a chance floor, and a correctly `sqrt(D)`-scaled ambient perturbation ladder whose realized displacements are printed and verified
- Every headline difference carries a disjoint-half-split resampling interval, and gaps that straddle zero are reported as unresolved rather than as results
- The CAE's invents-versus-destroys direction reported from separate, never-summed directional quantities, with 02.5-09's prior explicitly confirmed or refuted
- The Swiss roll half run at matched capacity under one common protocol, across at least three seeds with the spread and surviving chart counts shown, with the required 3-D and x-z plots coloured by the arc-length parameter
- Zero training of any sealed fit, zero cache writes, zero verdict artifacts, zero changes to any sealed verdict, and nothing written into phase 02.5
- If the work ran into pressure to add a tested module function, that pressure is recorded in the SUMMARY as the signal it outgrew quick mode — not acted on
</success_criteria>

<output>
Create `.planning/quick/20260809-topoae-vs-cae-persistence/SUMMARY.md` when done.
</output>
