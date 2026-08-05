---
phase: quick-260805-brr
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: true
requirements: [CAE-03, CAE-04, CAE-05, CAE-06, CAE-07]
files_modified:
  - notebooks/02.2_chart_autoencoder.ipynb

must_haves:
  truths:
    - "Reading notebooks/02.2_chart_autoencoder.ipynb top to bottom tells you what the Chart Auto-Encoder experiment tested, against which three pre-registered thresholds, what each gate measured, and why CAE_VERDICT is FAIL — without opening cae_train_run.py, cae_evaluate_run.py, or cae.py"
    - "The notebook's recomputed distortion, rcycle_ratio and recon_margin reproduce the published values in cae_verdict_43cf438bc944c509.json, and the notebook halts loudly inside the executing cell if any of them does not"
    - "Every code cell in the committed notebook carries a real execution_count and real stored outputs, contiguous from 1, with no error output anywhere"
    - "No model is trained by the notebook: every number derives from an npz or json already present in notebooks/.cache/, reloaded through pu_manifold.cae's own constructors and metric functions"
    - "The notebook writes nothing into notebooks/.cache/ — the sealed verdict artifact it regression-checks against is strictly read-only to it, and the cache directory is byte-for-byte unchanged by a full execution"
    - "A reader can see from the closing section alone exactly which parts of the two runner scripts the notebook does not reproduce, and whether any of them changes a gate value"
    - "Every pre-existing file in the repository is unmodified: the only tracked-file change this plan produces is the addition of one new notebook"
  artifacts:
    - "notebooks/02.2_chart_autoencoder.ipynb — a single executed Jupyter notebook, committed with outputs, mirroring the section rhythm of notebooks/02_k_sensitivity_refit.ipynb"
  key_links:
    - "notebook §2 pre-registered constants <-> the thresholds/seeds/holdout block recorded inside cae_verdict_43cf438bc944c509.json (a constant that drifts silently changes what is being tested, and the §2 asserts are the only thing that catches it)"
    - "notebook §10 recomputed gate metrics <-> cae_verdict_43cf438bc944c509.json metrics (this comparison is the entire evidence that the distillation is faithful; without it the notebook is just a plausible-looking retelling)"
    - "sys.path.insert(0, str(Path.cwd())) <-> notebooks/pu_manifold (a kernel whose working directory is not notebooks/ either fails to import or resolves a different package; the §1 cwd assert is what turns that into a loud failure)"
    - "arrays_to_state_dict(npz, model.state_dict()) <-> build_cae's architecture constants D_CHART/L_EMBED/N_CHARTS_INIT/HIDDEN_WIDTH/ACTIVATION (a mismatched constructor loads a shape-mismatched state dict and every downstream metric silently means something else)"
---

<objective>
Distill the Phase 02.2 Chart Auto-Encoder experiment into one digestible, executed Jupyter notebook at `notebooks/02.2_chart_autoencoder.ipynb`, mirroring the section rhythm of `notebooks/02_k_sensitivity_refit.ipynb`.

Purpose: the experiment currently exists as two 640-line runner scripts plus a 1187-line library, and its result is a sealed FAIL that no one can inspect without reading all three. The notebook makes the actual science — three pre-registered gates, three measured numbers, one verdict — readable in one pass, and makes the surrounding scaffolding visible enough that a reader can judge for themselves whether the implementing agent over-engineered the process.

Output: one new notebook, executed end-to-end against the already-cached fits and committed with its outputs. Nothing else. No source file, no runner script, no library module, and no cache artifact is deleted, edited, or regenerated.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/02.2-chart-autoencoder-validity-test-inserted/02.2-PREREGISTRATION.md
@.planning/phases/02.2-chart-autoencoder-validity-test-inserted/02.2-FINDINGS.md
@notebooks/02_k_sensitivity_refit.ipynb
@notebooks/diagnostics/cae_evaluate_run.py
@notebooks/pu_manifold/cae.py
</context>

<critical_constraint>
**Additive only. Nothing existing may be deleted or edited.**

- The only file this plan creates or touches is `notebooks/02.2_chart_autoencoder.ipynb`.
- `notebooks/diagnostics/cae_train_run.py`, `notebooks/diagnostics/cae_evaluate_run.py`, `notebooks/pu_manifold/cae.py`, `notebooks/02_k_sensitivity_refit.ipynb` and every other tracked file stay exactly as they are. The `Edit` and `Write` tools are prohibited on all of them.
- `notebooks/.cache/` is read-only to this plan. Load artifacts with `np.load` / `Path.read_text()`, never through `npz_cache` / `json_cache` / `joblib_cache` / `write_cae_verdict` / `write_cae_handoff`, all of which write into the cache and would touch the sealed verdict artifact the notebook regression-checks against.

**No training. Ever.** All eight fits (three CAE seeds, one ReLU control, two plain-AE controls, two MDS-decoder baselines) are already trained and cached. Retraining is a multi-hour protocol and is out of scope in every task here. The notebook reloads weights via `cae.arrays_to_state_dict` and recomputes only the metrics. Training is *described* in a markdown cell and *evidenced* by the cached `cae_fit_meta_*.json` fields (`epochs_run`, `wallclock_s`, `early_stopped`, `wallclock_truncated`) — never re-executed.

**Import, do not reimplement.** Every model class and every metric comes from `pu_manifold.cae`: `ChartAutoEncoder`, `arrays_to_state_dict`, `chart_survival`, `select_overlap_pairs`, `r_cycle`, `reconstruction_stats`, `embedding_distortion`, `unfaithfulness_coverage`, `verdict_from_metrics`, `GATING_METRICS`, `VERDICT_RULE`. Plus `pu_manifold.cache.cache_path` and `pu_manifold.geometry_probes.distortion_stats`. If a computation already exists in that library, the notebook calls it. A notebook that re-derives a metric inline is a second implementation that can silently disagree with the sealed one.
</critical_constraint>

<reference_facts>
Established before planning; do not re-derive.

**Frozen identifiers.** `FIT_KEY = "43cf438bc944c509"`; subsample stem `subsample_20260729_a79b3460b838fd0a` (array key `legacysurvey`, shape (10000, 768)).

**Pre-registered constants** (`02.2-PREREGISTRATION.md` Section 4, mirrored at the top of `cae_train_run.py`):
`D_CHART=20`, `L_EMBED=40`, `N_CHARTS_INIT=16`, `HIDDEN_WIDTH=250`, `ACTIVATION="silu"`, `LIP_WEIGHT=1e-2`, `LIP_EVERY_N_STEPS=1`, `WEIGHT_DECAY=1e-4`, `ADAM_LR=3e-4`, `BATCH=64`, `MAX_EPOCHS=40`, `EARLY_STOP_PATIENCE=5`, `EARLY_STOP_MIN_DELTA=1e-4`, `WALLCLOCK_CEILING_S=7200`, `FPS_PRETRAIN_EPOCHS=5`, `PRUNE_TOL=1e-2`, `OVERLAP_P_MIN=0.05`, `OVERLAP_MIN_POINTS=200`, `HOLDOUT_FRACTION=0.2`, `SPLIT_SEED=20260803`, `SEEDS=(20260803, 20260804, 20260805)`, `MAIN_SEED=20260803`, `UNFAITHFUL_SAMPLES=1000`, `UNFAITHFUL_SEED=20260806`, `PAIR_SEED=20260731`, `PAIR_COUNT=200000`, `MDS_BASELINE_P=(8, 20)`, `PLAIN_AE_LATENTS=(20, 40)`, `THRESH_DISTORTION_MAX=0.15`, `THRESH_RCYCLE_RATIO_MAX=2.0`, `THRESH_RECON_MARGIN=0.10`.

**Published values** in `notebooks/.cache/cae_verdict_43cf438bc944c509.json` (the regression target):
`CAE_VERDICT="FAIL"`; `metrics.distortion=0.29698133226319146`, `metrics.rcycle_ratio=1.0893662590388085`, `metrics.recon_margin=3.5863496159842887`; `thresholds={distortion:0.15, rcycle_ratio:2.0, recon_margin:0.9}`; `metrics.t1.global_scale_factor=0.3467897685183669`, `t1.median_signed_rel=0.0003670455905122144`, `t1.p95_abs_rel=0.8160568763962887`, `t1.n_pairs=200000`; `t2.mean_r_cycle=0.6454455852508545`, `t2.mean_base=0.5924963986128569`, `t2.n_qualify=2000`; `t3.mse_cae=1.2544764129476053e-4`, `mse_plain_ae20=3.497920022510993e-5`, `mse_plain_ae40=3.408036315105611e-5`, `mse_mds_dec8=5.6707493270503915e-5`, `mse_mds_dec20=5.153384247242297e-5`; `unfaithfulness=0.04233336076140404`, `coverage=0.024`; `chart_count_surviving.n_charts_surviving=16` of `n_charts_initial=16`; `holdout={n:2000, n_train:8000}`; `versions={numpy:"2.5.1", scipy:"1.18.0", torch:"2.13.0+cpu", python:"3.14.6"}`.

**Cached npz array keys** in each `cae_fit_*`/`cae_ctrl_*` npz: 286 keys total — the flattened `state_dict` weights plus `train_idx`, `holdout_idx`, `z_all` (10000, 40), `p_all` (10000, 16), `chart_argmax_all`, `y_holdout` (2000, 768). `cae_fit_meta_*.json` keys: `activation, cfg, early_stopped, epochs_run, history, numpy_version, seed, timestamp, torch_version, wallclock_s, wallclock_truncated`. The MDS-decoder metas additionally carry `mse_linear_floor_holdout`.

**Model constructors** (copy the shapes exactly; a mismatch loads a shape-mismatched state dict):
- `ChartAutoEncoder(in_dim=768, embed_dim=L_EMBED, chart_dim=D_CHART, n_charts=N_CHARTS_INIT, hidden=[HIDDEN_WIDTH]*3, activation=ACTIVATION)`

**Metric call shapes** (lifted from `cae_evaluate_run.py`, which is the sealed computation):
- T1: `embedding_distortion(z=z_main, geo_pairs=geo_pairs_r2, rows=pair_rows, cols=pair_cols, train_mask=train_pair_mask, chart_dim=D_CHART, embed_dim=L_EMBED)` where `geo_pairs_r2` comes from `mds_eigenspectrum_{FIT_KEY}.npz`, `pair_rows/pair_cols` from `cae_pairs_{FIT_KEY}.npz["rows"/"cols"]`, and `train_pair_mask = np.isin(rows, train_idx) & np.isin(cols, train_idx)`. Gate value is `["median_abs_rel"]`.
- T2: `select_overlap_pairs(p_all[holdout_idx], OVERLAP_P_MIN, OVERLAP_MIN_POINTS, surviving_indices)` returns `{rows, alpha, beta}`; group rows by `(alpha, beta)`, and per group call `r_cycle(model_main, x_batch, alpha, beta)` and compute the matched base `2.0 * ||x - y_argmax||` from `model_main(x_batch)`'s `p`/`y_charts`. Gate value is `mean(r_cycle_vals) / mean(base_vals)`.
- T3: `reconstruction_stats(x_holdout.double(), torch.tensor(y_holdout, dtype=torch.float64))` per fit; gate value is `max(mse_cae/mse_plain_ae20, mse_cae/mse_mds_dec8)` against threshold `1.0 - THRESH_RECON_MARGIN`.
- Verdict: `verdict_from_metrics(metrics, thresholds)` with `metrics` carrying the three `GATING_METRICS` keys, returning `(verdict, gate_detail)`.

**Style target** — `notebooks/02_k_sensitivity_refit.ipynb`, 27 cells: a framing markdown cell 0 stating binding prohibitions; `## §N. Title` markdown headers each followed by one code cell; a §1 provenance cell printing `=== Reproducibility header ===` with python/numpy/scipy/torch versions, git short SHA and cwd; pre-registered constants as named module-level values in their own cell with inline `assert`s carrying explanatory messages; a shared-machinery cell; per-result cells; a closing fixed-width comparison table built with f-string column widths and `"=" * N` rules; loud inline assertions throughout; `=== §N: title ===` print banners.

**Runner scaffolding the notebook deliberately does not reproduce** (material for §11; all factual, verified against the two runners):
1. `cae_train_run.py` STEP 0b and `cae_evaluate_run.py` STEP 0b both shell out to `git log --diff-filter=A` plus `git merge-base --is-ancestor` to prove the pre-registration commit precedes HEAD (CAE-01). The block is duplicated verbatim across the two files.
2. `cae_train_run.py` STEP 0c redraws the 200,000-pair geodesic sample, then — unable to compare it against the un-loaded 1.55 GiB Isomap pickle — asserts only that a second, differently-seeded redraw differs from the first.
3. `_protocol_cfg` reflects roughly 18 constants into every cache key, so any protocol edit invalidates every cached fit.
4. `run_and_cache` threads a `box` dict closure so the npz cache and the json meta cache share a single train call.
5. A 50-step `timing_probe` exists solely to branch `lip_every_n_steps` from 1 to 4 when a projected wallclock exceeds 7200 s.
6. `cae_evaluate_run.py` imports `cae_train_run`, which re-runs that runner's preconditions and its entire fit registry as a cache-hit read, purely to obtain roughly 25 module-level constants.
7. `cae_evaluate_run.py` rebuilds five models (`relu_model`, both `plainae_models`, both `mdsdec_models`) that no metric ever consumes — every T3 number comes from the stored `y_holdout` arrays, not from a forward pass.
8. The verdict artifact carries prose `assumptions` and `remediation` lists, a `sign_convention` paragraph and a `recon_margin_gate_note` paragraph.
9. Training itself: eight fits, multi-hour.
</reference_facts>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end slice — reload the cached fits and reproduce gate T1</name>
  <precondition>All of `notebooks/.cache/cae_fit_43cf438bc944c509_seed{20260803,20260804,20260805}.npz`, the matching `cae_fit_meta_*.json`, `cae_ctrl_relu_*`, `cae_ctrl_plainae{20,40}_*`, `cae_ctrl_mdsdec{8,20}_*` (npz + meta json each), `cae_pairs_43cf438bc944c509.npz`, `mds_eigenspectrum_43cf438bc944c509.npz`, `subsample_20260729_a79b3460b838fd0a.npz`, `geometry_probes_43cf438bc944c509.json` and `cae_verdict_43cf438bc944c509.json` exist and are non-empty, and `/home/akagi/Documents/Projects/EffDim/.venv/bin/jupyter` is executable. Assert this first and halt on any absence — there is no regeneration path.</precondition>
  <files>notebooks/02.2_chart_autoencoder.ipynb</files>
  <read_first>notebooks/02_k_sensitivity_refit.ipynb (cells 0, 2, 4, 8, 12, 22 — the framing, provenance, constants, machinery, ordering-assert and table idioms this notebook copies), notebooks/diagnostics/cae_evaluate_run.py (STEP 0a, STEP 1, STEP 3 — the reload and T1 computation being distilled), notebooks/pu_manifold/cae.py lines 149-223 and 956-1030 (ChartAutoEncoder constructor/forward, fit_global_scale, embedding_distortion)</read_first>
  <action>
Create the new notebook `notebooks/02.2_chart_autoencoder.ipynb` as nbformat 4.5 with the same `kernelspec` as `02_k_sensitivity_refit.ipynb` (`name: python3`, display name `Python 3 (effdim .venv)`). This task lays down one complete vertical path — environment, constants, cached-fit inventory, model reload, one gate, one regression check against the sealed artifact — so the reload-and-recompute architecture is proven end-to-end before the remaining gates are expanded onto it.

Use the final section numbering from the outset so nothing renumbers later. This task creates cells for §1, §2, §3, §4, §6 and a partial §10; §5, §7, §8, §9 and §11 are simply absent and get inserted at their numbered positions by Tasks 2 and 3.

Cell 0 (markdown) — framing. Title `# 02.2 — Chart Auto-Encoder Validity Test`. State: this is the execution of a pre-registered analysis (`02.2-PREREGISTRATION.md`, ratified before any fit ran, CAE-01); the three gates and their thresholds; the strict-less-than conjunction rule with no MARGINAL tier; and the binding prohibitions for this notebook specifically — it trains nothing, writes nothing into `notebooks/.cache/`, revises no threshold, and reproduces a recorded FAIL rather than searching for a PASS.

§1 markdown + code — environment and provenance. In the code cell: `import gc, json, resource, subprocess, sys` and `from pathlib import Path`; `NOTEBOOK_DIR = Path.cwd()` and `sys.path.insert(0, str(NOTEBOOK_DIR))` guarded by a membership check, exactly as notebook 02 does — never import from `src/effdim/`. Assert `NOTEBOOK_DIR.name == "notebooks"` with a message naming the kernel working directory as the cause, so a wrong cwd fails here instead of resolving a different package. Then `import numpy as np, scipy, torch`, `from pu_manifold import cache`, `from pu_manifold import cae as cae_mod`, `from pu_manifold import geometry_probes as gp`. Define `FIT_KEY = "43cf438bc944c509"`. Load the sealed artifact read-only: `PUBLISHED = json.loads(cache.cache_path(f"cae_verdict_{FIT_KEY}", "json").read_text())`. Print a `=== Reproducibility header (02.2) ===` block with python/numpy/scipy/torch versions, `git rev-parse --short HEAD`, cwd, and `FIT_KEY`. Assert the four live versions equal `PUBLISHED["versions"]`, with a message explaining that the §10 regression check compares floating-point results against a run made under those exact versions and cannot be trusted across a drift.

§2 markdown + code — pre-registered constants. Declare every constant listed in `<reference_facts>` as a named module-level value, grouped and commented by role (architecture / optimisation / gating thresholds / splits and seeds / control ladders), each group headed by its `02.2-PREREGISTRATION.md` Section 4 provenance. Then assert the subset the sealed artifact independently records, so drift is caught rather than assumed: `THRESH_DISTORTION_MAX == PUBLISHED["thresholds"]["distortion"]`, `THRESH_RCYCLE_RATIO_MAX == PUBLISHED["thresholds"]["rcycle_ratio"]`, `1.0 - THRESH_RECON_MARGIN == PUBLISHED["thresholds"]["recon_margin"]`, `list(SEEDS) == PUBLISHED["seeds"]["all_seeds"]`, `MAIN_SEED == PUBLISHED["seeds"]["main_seed"]`, `HOLDOUT_FRACTION == PUBLISHED["holdout"]["holdout_fraction"]`, `SPLIT_SEED == PUBLISHED["holdout"]["split_seed"]`, `OVERLAP_MIN_POINTS == PUBLISHED["holdout"]["t2_min_points_required"]`, `PRUNE_TOL == PUBLISHED["chart_count_surviving"]["prune_tol"]`. Each assert carries a message naming the constant and both values. Print the three gate thresholds and the verdict rule (`cae_mod.VERDICT_RULE`).

§3 markdown + code — the eight cached fits, and what training them cost. Markdown: describe the training protocol in prose — initial encoder to R^40, sixteen over-specified chart encoders into (0,1)^20, per-chart decoders, one shared embedding decoder, a partition-of-unity chart predictor; the paper's loss plus a Lipschitz penalty on chart-encoder spectral norms; five epochs of FPS-seeded per-chart pre-training (eq. 5), without which the second chart never activates; Adam at 3e-4, batch 64, up to 40 epochs, early stop at patience 5, wallclock ceiling 7200 s. State plainly that this notebook does not run any of it: the eight fits are cached and this notebook reloads them. Code: assert every required cache path exists and is non-empty (the ten stems listed in the precondition), then load the eight `*_meta_*.json` files and print a fixed-width table of `run_id | epochs_run | wallclock_s | early_stopped | wallclock_truncated | stopping_reason`, where `stopping_reason` is derived as wallclock ceiling / early-stop plateau / epoch cap. Print the total wallclock across all eight runs. Assert no run was wallclock-truncated.

§4 markdown + code — reload. Load the three CAE seed npz files and the five control npz files with `np.load`. Assert `train_idx` and `holdout_idx` are bit-identical across all eight (`np.array_equal`) — a metric computed across mismatched splits would silently mean nothing — and assert their shapes are 8000 and 2000. Load `X = np.load(subsample)["legacysurvey"]`, build `x_all_t = torch.tensor(X, dtype=torch.float32)` and slice `x_train_t` / `x_holdout_t` by the index arrays. Rebuild the three CAE seed models only: construct with the §2 architecture constants and `model.load_state_dict(cae_mod.arrays_to_state_dict(npz, model.state_dict()))`, then `model.eval()`. Bind `model_main = seed_models[MAIN_SEED]`. Do not construct models for the ReLU, plain-AE or MDS-decoder fits — every T3 number comes from their stored `y_holdout` arrays; note this in a comment as a deliberate deviation and record it for §11. Memory discipline, mirroring notebook 02: after the state dicts are loaded, keep only the small arrays needed downstream (`train_idx`, `holdout_idx`, main-seed `z_all` and `p_all`, and each fit's `y_holdout`), drop the full npz dicts and `X`, then `del` and `gc.collect()`. Print the number of models rebuilt, the retained-array inventory, and peak RSS via `resource.getrusage`.

§6 markdown + code — gate T1, geodesic distortion. Markdown: T1 measures how far the initial encoder's global embedding `z_all` distorts the cached 200,000-pair geodesic sample after a single global scale factor is fit on train-only pairs; threshold 0.15, strict. Code: load `rows`/`cols` from `cae_pairs_{FIT_KEY}.npz` and `geo_pairs_r2` from `mds_eigenspectrum_{FIT_KEY}.npz` (read only that one array, not the whole archive), build `train_pair_mask`, call `cae_mod.embedding_distortion` with the exact kwargs in `<reference_facts>`, and bind `T1 = result["median_abs_rel"]`. Print a `=== §6: gate T1 ===` banner with `median_abs_rel`, `median_signed_rel`, `p95_abs_rel`, `global_scale_factor`, `n_pairs`, the threshold and the pass boolean. Also compute the non-gating holdout-only companion the way the runner does — reuse the already-fit `global_scale_factor` rather than refitting on the holdout subset — via `gp.distortion_stats` over the holdout pair mask, and print it labelled as non-gating.

§10 markdown + code (partial, this task) — regression check against the sealed artifact. Assert `abs(T1 - PUBLISHED["metrics"]["distortion"]) <= 1e-9 * abs(PUBLISHED["metrics"]["distortion"])` with a message printing both values, and likewise assert `global_scale_factor`, `median_signed_rel`, `p95_abs_rel` and `n_pairs` against `PUBLISHED["metrics"]["t1"]`. Then print, on a single line at 17 significant digits, the machine-readable sentinel `REGRESSION_OK distortion=<value> rcycle_ratio=nan recon_margin=nan verdict=pending` — Task 2 fills the remaining three fields. Task 2's and Task 3's external verifiers parse this line out of the committed outputs, so the format is load-bearing.

Then execute the notebook end-to-end and leave its outputs stored in the file (see `<verify>`).
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/02.2_chart_autoencoder.ipynb; CACHE_BEFORE=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB" && CACHE_AFTER=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CACHE_BEFORE" = "$CACHE_AFTER" ] && git -C $REPO diff --quiet HEAD -- notebooks/ src/ tests/ pyproject.toml && $REPO/.venv/bin/python - "$NB" <<'PY'
import json, pathlib, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
assert code, "notebook has no code cells"
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [t for t in ("cae_train_run", "train_cae(", "train_plain_ae", "train_mlp_decoder",
                      "write_cae_verdict", "write_cae_handoff", "npz_cache(", "json_cache(",
                      "joblib_cache(", "timing_probe") if t in src]
assert not banned, f"forbidden training/cache-write call in executable source: {banned}"
assert "PUBLISHED" in src and "embedding_distortion" in src, "T1 regression path absent"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", [])
               if o.get("output_type") == "stream")
assert "REGRESSION_OK distortion=" in text, "sentinel line absent from stored outputs"
print("TASK1_OK", len(code), "code cells executed")
PY</automated>
  </verify>
  <done>`notebooks/02.2_chart_autoencoder.ipynb` exists, executes clean end-to-end under the venv kernel, and its stored outputs show T1 reproducing `PUBLISHED["metrics"]["distortion"]` to 1e-9 relative. The cache directory hash is unchanged by the run, no tracked file is modified, no training or cache-write call appears in executable source, and `TASK1_OK` prints.</done>
</task>

<task type="auto">
  <name>Task 2: Expand onto the proven slice — gates T2 and T3, chart survival, non-gating evidence, full verdict</name>
  <files>notebooks/02.2_chart_autoencoder.ipynb</files>
  <read_first>notebooks/diagnostics/cae_evaluate_run.py STEP 2 and STEP 4-7 (chart survival, T2, T3, non-gating evidence, verdict assembly), notebooks/pu_manifold/cae.py lines 347-373, 787-955, 1054-1099 (reconstruction_stats, chart_survival, r_cycle, select_overlap_pairs, unfaithfulness_coverage, GATING_METRICS, verdict_from_metrics)</read_first>
  <action>
Insert the remaining sections at their already-reserved numbered positions and rewrite §10 into the full verdict section. Do not renumber any existing header. Do not alter the §1-§4 or §6 cells except where §10's sentinel line requires it.

§5 markdown + code, inserted between §4 and §6 — chart survival (CAE-05). Markdown: sixteen charts were over-specified on purpose; weight decay was expected to prune the surplus, and the surviving count is read off a posteriori at decoder-weight-norm tolerance `PRUNE_TOL`. Code: call `cae_mod.chart_survival(model, PRUNE_TOL)` for each of the three seeds, print a per-seed line of `surviving/initial` plus the surviving and pruned index lists, and print the per-chart mass ratios for the main seed. Bind `surviving_indices_main`. Assert the main seed's `n_charts_surviving` and `n_charts_initial` equal `PUBLISHED["chart_count_surviving"]`'s values, and assert stability across seeds by comparing each seed's surviving count to `PUBLISHED["chart_count_by_seed"]`. Note in the markdown that all sixteen surviving is itself the finding: pruning removed nothing.

§7 markdown + code, inserted between §6 and §8 — gate T2, chart-transition cycle residual (eq. 8). Markdown: T2 measures whether a point encoded in chart alpha, decoded, re-encoded in chart beta and decoded back returns to itself; the residual is normalised by a matched base quantity — twice the single-pass argmax-chart reconstruction norm, matching eq. 8's two-term unsquared form — so the ratio is scale-free; threshold 2.0, strict. Code: slice `p_holdout = p_all[holdout_idx]`, call `cae_mod.select_overlap_pairs(p_holdout, OVERLAP_P_MIN, OVERLAP_MIN_POINTS, surviving_indices_main)`, group the returned row positions by their `(alpha, beta)` chart pair, and under `torch.no_grad()` compute per group both `cae_mod.r_cycle(model_main, x_batch, alpha, beta)` and the matched base `2.0 * torch.linalg.vector_norm(x_batch - y_argmax, dim=-1)` where `y_argmax` is selected from `model_main(x_batch)["y_charts"]` at `p.argmax(dim=1)`. Bind `T2 = mean(r_cycle_vals) / mean(base_vals)`. Print a `=== §7: gate T2 ===` banner with the qualifying row count, the number of distinct chart pairs, both means, the ratio, the threshold and the pass boolean. Assert the qualifying count equals `PUBLISHED["metrics"]["t2"]["n_qualify"]`.

§8 markdown + code, inserted between §7 and §9 — gate T3, held-out reconstruction against matched-capacity controls. Markdown: state the capacity match (same hidden width 250, same three layers per side, identical training protocol) and the algebra — the pre-registration's rule is `mse_cae < (1 - margin) * mse_control` for both gating controls, and dividing through by `mse_control` puts it in the same strict-less-than value/threshold form as T1 and T2, with `max` over the two ratios encoding the AND. Code: for the main CAE seed and each of the five controls, call `cae_mod.reconstruction_stats(x_holdout_t.double(), torch.tensor(y_holdout, dtype=torch.float64))` on the stored `y_holdout` arrays retained in §4. Bind `T3 = max(mse_cae/mse_plain_ae20, mse_cae/mse_mds_dec8)` against `1.0 - THRESH_RECON_MARGIN`. Print a fixed-width table of all six fits — `mse_per_dim`, `dim_mse_mean`, `dim_mse_median`, `dim_mse_p95`, `dim_mse_max` — then the two gating ratios, the max, the threshold, and the pass boolean. Add the two MDS-decoder linear floors from their meta json as context rows. Assert each of the six `mse_per_dim` values against its counterpart under `PUBLISHED["metrics"]["t3"]` to 1e-9 relative.

§9 markdown + code, inserted between §8 and §10 — non-gating evidence. Code: `cae_mod.unfaithfulness_coverage(model_main, x_train_t, surviving_indices_main, UNFAITHFUL_SAMPLES, UNFAITHFUL_SEED)` — note in a comment that its internal pairwise distance allocates roughly 1000 x 8000 and free the result's intermediates afterward with `del` plus `gc.collect()`. Print unfaithfulness, coverage, `n_samples`, `n_distinct`. Then the CAE-06 activation substitution: `mse_relu_control - mse_cae`, printed with the sign convention spelled out in one sentence (positive means the pre-registered SiLU reconstructs better than the ReLU control). Then read `geometry_probes_{FIT_KEY}.json` and print the two classical q=0 Krein rows (p=40 and p=8, `median_abs_rel`) plus the recorded `working_dimension`, labelled explicitly as reported-not-gating context from Phase 02.1. Assert unfaithfulness and coverage against `PUBLISHED` to 1e-6 relative.

§10 markdown + code — rewrite into the full verdict. Markdown: the rule is a conjunction of three strict-less-than comparisons with no MARGINAL tier, applied mechanically. Code: assemble `metrics = {"distortion": T1, "rcycle_ratio": T2, "recon_margin": T3}` and `thresholds = {"distortion": THRESH_DISTORTION_MAX, "rcycle_ratio": THRESH_RCYCLE_RATIO_MAX, "recon_margin": 1.0 - THRESH_RECON_MARGIN}`, assert `set(metrics) == set(cae_mod.GATING_METRICS)`, and call `verdict, gate_detail = cae_mod.verdict_from_metrics(metrics, thresholds)`. Print a fixed-width three-row table — gate | what it measures | value | threshold | passed — bracketed by `"=" * N` rules in the style of notebook 02's §9, followed by `CAE_VERDICT = {verdict}`. Then the full regression block: assert all three gate values against `PUBLISHED["metrics"]` (distortion and recon_margin at 1e-9 relative, rcycle_ratio at 1e-6 relative because it routes through model forward passes), assert `verdict == PUBLISHED["CAE_VERDICT"]`, and assert `gate_detail`'s per-gate `passed` booleans match the sign of each published value against its published threshold. Finally print the sentinel at 17 significant digits with every field populated: `REGRESSION_OK distortion=<v> rcycle_ratio=<v> recon_margin=<v> verdict=FAIL`. Close the cell with a short printed statement that the FAIL is a measured outcome reproduced from cached weights, not a result being retried until it passes.

Re-execute the notebook end-to-end (see `<verify>`).
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/02.2_chart_autoencoder.ipynb; CACHE_BEFORE=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB" && CACHE_AFTER=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CACHE_BEFORE" = "$CACHE_AFTER" ] && git -C $REPO diff --quiet HEAD -- notebooks/ src/ tests/ pyproject.toml && $REPO/.venv/bin/python - "$NB" "$REPO/notebooks/.cache/cae_verdict_43cf438bc944c509.json" <<'PY'
import json, pathlib, re, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
pub = json.loads(pathlib.Path(sys.argv[2]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [t for t in ("cae_train_run", "train_cae(", "train_plain_ae", "train_mlp_decoder",
                      "write_cae_verdict", "write_cae_handoff", "npz_cache(", "json_cache(",
                      "joblib_cache(", "timing_probe") if t in src]
assert not banned, f"forbidden training/cache-write call in executable source: {banned}"
for fn in ("embedding_distortion", "r_cycle", "select_overlap_pairs", "reconstruction_stats",
           "chart_survival", "unfaithfulness_coverage", "verdict_from_metrics"):
    assert fn in src, f"library entry point not called: {fn}"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", [])
               if o.get("output_type") == "stream")
m = re.search(r"REGRESSION_OK distortion=(\S+) rcycle_ratio=(\S+) recon_margin=(\S+) verdict=(\w+)", text)
assert m, "populated sentinel line absent from stored outputs"
got = {"distortion": float(m.group(1)), "rcycle_ratio": float(m.group(2)), "recon_margin": float(m.group(3))}
for key, rtol in (("distortion", 1e-9), ("rcycle_ratio", 1e-6), ("recon_margin", 1e-9)):
    exp = pub["metrics"][key]
    assert abs(got[key] - exp) <= rtol * abs(exp), f"{key}: notebook {got[key]!r} vs sealed {exp!r}"
assert m.group(4) == pub["CAE_VERDICT"], f"verdict {m.group(4)} vs sealed {pub['CAE_VERDICT']}"
print("TASK2_OK", got, m.group(4))
PY</automated>
  </verify>
  <done>All three gate values recomputed inside the notebook reproduce `cae_verdict_43cf438bc944c509.json` within the stated tolerances, the recomputed verdict string equals the sealed `FAIL`, chart survival / unfaithfulness / coverage / per-fit MSEs all match their published counterparts, every named library entry point is actually called rather than reimplemented, the cache directory is unchanged and no tracked file is modified. `TASK2_OK` prints.</done>
</task>

<task type="auto">
  <name>Task 3: Add the scaffolding-vs-science closing section, execute clean, commit with outputs</name>
  <files>notebooks/02.2_chart_autoencoder.ipynb</files>
  <read_first>notebooks/diagnostics/cae_train_run.py STEP 0b, STEP 0c, `_protocol_cfg`, `run_and_cache`, the timing-probe block; notebooks/diagnostics/cae_evaluate_run.py lines 1-60 and 490-580 (the cross-runner import and the prose fields of the verdict artifact)</read_first>
  <action>
Append §11 as markdown only — no new code cell — then re-execute and commit.

§11 markdown, `## §11. What this notebook does not reproduce, and why`. Two lists, factual, no editorialising beyond what is verifiable in the two runners.

First list, "reproduced here": the three gate metrics, the verdict, chart survival across all three seeds, unfaithfulness and coverage, the per-fit reconstruction table, and the training cost of all eight fits — every one of them regression-checked against the sealed artifact in §10.

Second list, "not reproduced here, with the reason and whether it moves a gate value". Cover each of the nine items enumerated in this plan's `<reference_facts>` scaffolding list, one bullet apiece, each naming the runner and the construct concretely (the git ancestry block duplicated across both runners; the geodesic redraw whose self-check compares one redraw against a second redraw rather than against the cached pairs; `_protocol_cfg`'s roughly eighteen reflected constants; `run_and_cache`'s `box` closure; the fifty-step timing probe whose sole effect is a `lip_every_n_steps` 1-to-4 branch above a projected 7200 s; the evaluation runner importing the training runner to obtain roughly twenty-five module-level constants, re-running its preconditions and full fit registry as a cache-hit read; the five models rebuilt in the evaluation runner that no metric consumes; the verdict artifact's prose `assumptions`/`remediation`/`sign_convention`/`recon_margin_gate_note` fields; and training itself). For each, state in one clause whether omitting it changes any of the three gate numbers. State explicitly and separately that CAE-01's pre-registration ordering is a property of the repository's commit history that was established before the sealed run and cannot be re-established by re-running a check now — which is precisely why this notebook cites it rather than re-executing it.

Close §11 with a single sentence stating what the reader is being invited to judge: the numbers in §6-§10 came out of the cached weights unchanged, so everything in the second list is scaffolding the result did not depend on, and the reader can weigh that against the cost of maintaining it.

Then re-execute the notebook end-to-end one final time so the committed file's outputs correspond exactly to the committed source, and commit it as a new file. Do not `git add` anything else.
  </action>
  <verify>
    <automated>REPO=/home/akagi/Documents/Projects/EffDim; NB=$REPO/notebooks/02.2_chart_autoencoder.ipynb; CACHE_BEFORE=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum); $REPO/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB" && CACHE_AFTER=$(find $REPO/notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum) && [ "$CACHE_BEFORE" = "$CACHE_AFTER" ] && [ -z "$(git -C $REPO diff --name-only --diff-filter=MDR HEAD -- . ':!.planning')" ] && $REPO/.venv/bin/python - "$NB" "$REPO/notebooks/.cache/cae_verdict_43cf438bc944c509.json" <<'PY'
import json, pathlib, re, sys
nb = json.loads(pathlib.Path(sys.argv[1]).read_text())
pub = json.loads(pathlib.Path(sys.argv[2]).read_text())
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
counts = [c.get("execution_count") for c in code]
assert counts == list(range(1, len(code) + 1)), f"execution_count not contiguous from 1: {counts}"
for c in code:
    for o in c.get("outputs", []):
        assert o.get("output_type") != "error", f"error output: {o.get('ename')}: {o.get('evalue')}"
    assert c.get("outputs"), "a code cell stored no output"
for n in range(1, 12):
    assert f"## §{n}." in md, f"section header §{n} missing"
src = "".join(ln.split("#", 1)[0] for c in code for ln in c["source"])
banned = [t for t in ("cae_train_run", "train_cae(", "train_plain_ae", "train_mlp_decoder",
                      "write_cae_verdict", "write_cae_handoff", "npz_cache(", "json_cache(",
                      "joblib_cache(", "timing_probe") if t in src]
assert not banned, f"forbidden training/cache-write call in executable source: {banned}"
text = "".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", [])
               if o.get("output_type") == "stream")
m = re.search(r"REGRESSION_OK distortion=(\S+) rcycle_ratio=(\S+) recon_margin=(\S+) verdict=(\w+)", text)
assert m, "populated sentinel line absent from stored outputs"
got = {"distortion": float(m.group(1)), "rcycle_ratio": float(m.group(2)), "recon_margin": float(m.group(3))}
for key, rtol in (("distortion", 1e-9), ("rcycle_ratio", 1e-6), ("recon_margin", 1e-9)):
    exp = pub["metrics"][key]
    assert abs(got[key] - exp) <= rtol * abs(exp), f"{key}: notebook {got[key]!r} vs sealed {exp!r}"
assert m.group(4) == pub["CAE_VERDICT"]
print("TASK3_OK", len(code), "code cells,", m.group(4))
PY</automated>
  </verify>
  <done>The notebook carries §1 through §11, every code cell stores real outputs from the final execution, all three gate values and the verdict still reproduce the sealed artifact, `git diff --diff-filter=MDR HEAD` outside `.planning/` is empty (nothing existing was modified or deleted), and the notebook is committed with its outputs. `TASK3_OK` prints.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| cached artifact -> notebook | `notebooks/.cache/*.npz` is deserialized by `np.load`; the `.npz` archives were produced locally by this repo's own runners |
| notebook -> cached artifact | any write path from the notebook back into `.cache/` would mutate the sealed verdict the notebook is supposed to be judged against |
| notebook -> repository working tree | `nbconvert --inplace` writes into `notebooks/`; a wrong path or a whole-file write elsewhere destroys existing tracked work |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-BRR-01 | Tampering | notebook writing into `notebooks/.cache/` | high | mitigate | `<critical_constraint>` forbids `npz_cache`/`json_cache`/`joblib_cache`/`write_cae_verdict`/`write_cae_handoff`; every task's `<verify>` greps comment-stripped executable source for those calls and hashes the whole cache tree (path, size, mtime) before and after execution, requiring equality |
| T-BRR-02 | Tampering | accidental edit or deletion of existing tracked files | high | mitigate | `files_modified` is a single new file; every task's `<verify>` runs `git diff --diff-filter=MDR HEAD` outside `.planning/` and requires it empty, so any modification or deletion of a pre-existing file fails the task |
| T-BRR-03 | Repudiation | notebook outputs that do not correspond to its committed source | medium | mitigate | Task 3 re-executes immediately before commit; verifiers assert `execution_count` is contiguous from 1, that every code cell stored output, and that no cell holds an error output |
| T-BRR-04 | Information disclosure | committed outputs leaking absolute paths / environment detail | low | accept | The §1 provenance cell prints cwd and versions by design, matching `02_k_sensitivity_refit.ipynb`; the repository is the user's own and already commits that notebook with the same header |
| T-BRR-05 | Tampering | `np.load` on a `.npz` from an untrusted source | low | accept | Every archive read is produced by this repository's own runners under `notebooks/.cache/`; `allow_pickle` stays at its safe default and no object arrays are stored |
| T-BRR-06 | Denial of service | a runaway cell exhausting memory (eight npz reloads plus a 1000x8000 pairwise allocation) | medium | mitigate | §4 retains only the small downstream arrays and drops the full npz dicts with `del` + `gc.collect()`; §9 frees the `unfaithfulness_coverage` intermediates; §4 prints peak RSS; nbconvert runs with an explicit per-cell timeout |
| T-BRR-SC | Tampering | package installs | low | accept | No package is installed by this plan; every dependency (torch, numpy, scipy, jupyter) is already pinned in `notebooks/requirements-notebooks.txt` and present in `.venv`. No `[ASSUMED]`/`[SUS]` package is introduced, so no legitimacy checkpoint applies |
</threat_model>

<verification>
1. All three tasks print their `TASK{N}_OK` sentinel.
2. `git status --porcelain -- . ':!.planning'` shows exactly one entry: the new `notebooks/02.2_chart_autoencoder.ipynb`. No other file is added, modified, renamed or deleted.
3. `find notebooks/.cache -type f -printf '%p %s %T@\n' | sort | sha256sum` is identical before and after a full notebook execution — the cache is provably read-only to this work.
4. Open the notebook and read it top to bottom: §1-§4 establish provenance, constants, training cost and the reload; §5-§9 produce the evidence; §10 states the verdict and proves it reproduces the sealed artifact; §11 states what was left out. No section requires opening a runner script to follow.
5. The three published gate values (0.29698133226319146 / 1.0893662590388085 / 3.5863496159842887) and `CAE_VERDICT = FAIL` appear in the notebook's stored outputs as recomputed values, not as transcribed constants — verified by the external parser in each task's `<verify>`, which compares the notebook's sentinel line against `cae_verdict_43cf438bc944c509.json` independently of the notebook's own asserts.
6. `notebooks/diagnostics/cae_train_run.py`, `notebooks/diagnostics/cae_evaluate_run.py`, `notebooks/pu_manifold/cae.py` and `notebooks/02_k_sensitivity_refit.ipynb` are byte-identical to `HEAD`.
</verification>

<success_criteria>
- One new executed notebook, `notebooks/02.2_chart_autoencoder.ipynb`, committed with outputs, in the section-numbered style of `notebooks/02_k_sensitivity_refit.ipynb`
- Zero training runs: every number reloaded from `notebooks/.cache/` through `pu_manifold.cae`'s own constructors and metric functions, with nothing reimplemented inline
- The three gate metrics and the verdict reproduce `cae_verdict_43cf438bc944c509.json`, asserted inside the notebook and re-checked from outside it against the committed outputs
- Nothing existing is deleted or modified — not the two runner scripts, not `cae.py`, not the Isomap notebook, not a single cache artifact
- §11 lets a reader see exactly which parts of the 1,280 lines of runner script the notebook skipped and whether any of them changed a gate number, so the over-engineering question can be judged on evidence rather than on the notebook's opinion
</success_criteria>

<output>
Create `.planning/quick/260805-brr-distill-the-cae-experiment-into-a-notebo/260805-brr-SUMMARY.md` when done.
</output>
