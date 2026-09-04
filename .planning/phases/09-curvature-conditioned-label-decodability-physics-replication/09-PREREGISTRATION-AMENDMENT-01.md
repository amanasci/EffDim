# Phase 9 Pre-Registration Amendment 01 — Sphere-Projected Decoder Image

**Date:** 2026-09-04. **Status:** applied. **Raised by:** the executing agent, on validating the
curvature instrument against analytic decoders with known answers after the Wave A `H_rad`
backstop failed at every `d` (`09-WAVE-A-RESULTS.md` § 3).

**This amendment SUPERSEDES the `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` freeze
(`09-PREREGISTRATION.md`) in full.** Every constant that freeze filled remains ratified except as
stated below; the one sealed-module edit adds two constants and extends one rule string. The
freeze commit that carries this amendment is the commit immediately following this document's
first commit in git history — see that commit's message and `git show --stat` for its SHA. No
SHA is quoted in this document's first commit because, by the commit-ordering discipline it
follows, the document is written before that commit exists; a follow-up docs commit records it
in the header line below.

**Amendment 01 freeze commit SHA:** _(filled by the follow-up docs commit)_

## Why this amendment exists, and why it is legitimate

Wave A (`09-WAVE-A-RESULTS.md`, run 2026-09-04 under the `5f7fbe2` freeze) recorded the `H_rad`
backstop truth failing at every `d`: the median radial component of the decoder mean-curvature
vector sat 15-27% away from the exact `-d` the pre-registration expected. The plan reported that
plainly as a poor-fit-vs-real-geometry ambiguity it could not resolve. This amendment resolves
it, by validating the instrument on decoders whose curvature is known in closed form, with the
sealed code unchanged.

**Measured 2026-09-04, sealed code unchanged, analytic decoders, `d=16`, `D=768`:**

- `decoder_curvature.plain_decoder_curvature` is an exact implementation of `H = tr_g(II)`:
  on a sphere, a latitude-sphere and a cylinder the known mean-curvature vectors are recovered to
  ~1e-15, direction included. The differentiator is not the problem.
- `physics_curvature_probe.decompose_radial_tangential` is exact ONLY when the decoder image
  lies in the unit sphere. Then `H_rad = -d` exactly (trace convention) and `H_tan` is the
  sphere-intrinsic mean curvature — the quantity `CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"`
  and D9-11 always said the verdict measures.
- The production instrument, `cae.PlainAutoEncoder.decode`, has `out_activation=None`. Its image
  is not constrained to the sphere the L2-normalised data occupy (`EMBEDDING_NORMALIZATION`).
  The production Wave A anchor tables show it: at `d=16`, `H_rad` p05 / median / p95 =
  `-23.5 / -20.4 / -17.3` against the exact `-16`, with **59% of anchors off by more than
  `0.25 d`** (re-measured from the returned `09_anchor_table_d16_mag_r.npz`: `-23.51 / -20.39 /
  -17.34`, 58.6%). The same tables give `-27.4 / -24.2 / -20.9` (33.8% off by `> 0.25 d`) at
  `d=20`, `-32.8 / -29.6 / -25.2` (20.3%) at `d=25`, `-40.1 / -36.8 / -31.1` (6.4%) at `d=32`.
  The decoder-image norms themselves sit at p05 / p50 / p95 ≈ `0.96 / 0.98 / 1.00` at every `d`:
  a few percent off the sphere in norm, which the second derivative amplifies into the `H_rad`
  spread above.
- A calibrated synthetic contamination with that measured `d=16` spread drops the rank agreement
  of `H_tan` with the true sphere-intrinsic curvature to **0.53-0.06**. Under the old freeze the
  gating field was, at the anchor level, largely noise about the quantity it was declared to
  measure.
- Taking the curvature of the RENORMALISED decoder `F/||F||` instead restores exactness:
  `H_rad = -d` identically and `H_tan` exact to 1e-13, verified under the harshest of the
  contaminations above. That is the fix, and it is the whole fix.

**This is an instrument-correctness finding established on decoders with known answers,
independent of every Physics number.** No threshold, verdict rule, control, null, anchor set,
probe or seed moves. Unlike `08-PREREGISTRATION-AMENDMENT-01.md` (where no Phase 8 number
existed), Phase 9 numbers DO exist — Wave A's four per-`d` verdicts (`DOES NOT CLEAR` at every
`d`) and Wave B (`WAVE_B_NOT_TRIGGERED`). The direction in which the projected field will move
the controlled partial is not known in advance and was not estimated before this amendment was
raised; the amendment is motivated by the closed-form tests above, not by any outcome. Every
number produced under the old freeze is retained and will be reported beside the new one, never
replaced (see "Why this is an amendment and not a post-hoc adjustment").

## The developer's decision, verbatim

The developer was shown the instrument validation above (the exactness on analytic decoders, the
`H_rad` spread in the production anchor tables, the contamination rank-agreement figures, and
the renormalised-decoder result) and replied, **2026-09-04 UTC**:

> implement the fix and rerun, ensuring to adhere to the ssh server guidelines

This is a real human response to a presented finding, given directly — not a standing
authorization and not an auto-approval under `AUTO_CFG`/`AUTO_CHAIN`. It authorizes (a) the
sealed-module and runner changes below, (b) a fresh freeze commit, and (c) a re-run on the
execution host under `CLAUDE.md`'s remote-compute rules and
`docs/remote-compute/eleutherai-pod-user-guide.md` (read in full before any remote command;
everything under `/mnt/ssd-cluster`, never `/root`; long jobs in `tmux`; only available compute).

## Exactly these changes, and nothing else

### 1. Sealed module `notebooks/pu_manifold/physics_curvature_probe.py` — the only sealed-module edit

- **Added** `DECODER_IMAGE_PROJECTION = "sphere"`, literal-guarded by `assert_preregistered()`
  the way `CURVATURE_CONVENTION == "trace"` is: any other value raises.
- **Added** `DECODER_IMAGE_PROJECTION_RULE`, exact-equality-guarded against a module-owned
  `_REQUIRED_DECODER_IMAGE_PROJECTION_RULE` the way the five existing rule strings are. Its
  text: curvature is evaluated on the sphere-projected decoder map `F/||F||` so the decoder
  image lies in the unit sphere the L2-normalised data occupy; under this the radial component
  `H_rad` equals `-d` identically (trace convention) and `H_tan_norm` is the mean curvature
  within the sphere; `H_rad` is recorded as a check, never as a result.
- Both names **added to `_REQUIRED_CONSTANTS`** (75 guarded constants, up from 73), so an UNSET
  value fails fast exactly as every other constant does.
- **`RADIAL_DECOMPOSITION_RULE`'s text extended** by one sentence stating that the image handed
  to `decompose_radial_tangential` (and the `H_vec` differentiated) is now the projected one;
  the formula it names is unchanged.
- `assert_preregistered()`'s docstring updated to count six equality checks instead of five.

No other constant changes value and no function body changes. `git diff` of the freeze commit
shows exactly these hunks.

### 2. Runner `notebooks/diagnostics/09_physics_curvature_run.py` (wiring commit, not the freeze commit)

- **Added** `SphereProjectedDecoder(torch.nn.Module)`: holds the trained model, exposes
  `.decoder = model.decoder` (so `decoder_curvature.assert_c2_decoder` still inspects the real
  activation modules) and `.decode(z) = F / ||F||` with `F = model.decode(z)`. Its parameters
  are the wrapped model's own, so `chart_curvature._assert_float64`'s guard runs against the
  real float64 weights.
- In `fit_and_field_at_anchors`, when `pcp.DECODER_IMAGE_PROJECTION == "sphere"`, both `image`
  and `plain_decoder_curvature` are computed through the wrapper, constructed after
  `model.eval().double()`. The autoencoder fit itself, the encoder, `var_explained` and the
  anchor codes are untouched.
- The `row_kind="fit"` record row gains `H_rad_max_abs_dev` = max over anchors of `|H_rad + d|`
  (expected ~1e-12 under the projection) and `decoder_image_projection`, beside the existing
  `H_rad_expected` / `H_rad_median`; no existing key is removed. The per-`d` banner prints one
  line naming the projection and that check.
- Every mode, flag and record shape is otherwise unchanged; `--mode smoke` still ends
  `SMOKE PASS`.

### 3. `FREEZE_COMMIT_SHA` re-wired (wiring commit)

The amendment freeze commit's full SHA replaces `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` in
`notebooks/diagnostics/09_physics_curvature_run.py`,
`notebooks/diagnostics/09_row_alignment_proof_run.py`,
`notebooks/pu_manifold/tests/test_physics_curvature_probe.py` and
`notebooks/pu_manifold/tests/test_physics_labels.py`, following `09-05-SUMMARY.md`'s two-commit
discipline exactly. The old SHA is now the REJECTED one: both test suites carry a test asserting
the superseded SHA is not the accepted freeze, and the runner gate's exact-equality check
(`CR-01`) refuses `--freeze-commit 5f7fbe27…` outright.

## What this amendment does not change

- **No verdict rule, threshold or gating logic.** `VERDICT_RULE`, `VERDICT_VALUES`,
  `PER_D_VERDICT_VALUES`, `VERDICT_SENTENCE_RULE`, `VERDICT_STATISTIC`, `FWER_ALPHA`,
  `P_VALUE_FLOOR_RULE`, `CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm"`, `H_NORM_IS_NON_GATING`,
  `RAW_RHO_IS_NON_GATING`, `STRATIFIED_NULL_IS_NON_GATING`, `REPORT_BOTH_NULLS_UNCONDITIONALLY`,
  `REPORTING_BLOCK_ROWS` / `REPORTING_BLOCK_RULE`.
- **No anchor, neighbourhood, probe or fit constant.** `N_ANCHORS`, `ANCHOR_DRAW_SEED`,
  `ANCHOR_POOL`, `K_NEIGHBOURS`, `SPLIT_SEED`, `HOLDOUT_FRACTION`, `AE_IN_DIM`, `AE_HIDDEN`,
  `AE_ACTIVATION`, `MAX_EPOCHS`, `TORCH_INIT_SEED`, `TRAIN_CFG`, `D_SWEEP`, `ALPHA_RIDGE`,
  `ALPHA_GRID`, `N_OOF_FOLDS`, `OOF_FOLD_SEED`, `LOCAL_R2_RULE`, `MIN_FINITE_NEIGHBOURS`,
  `MIN_IMAGE_NORM`, `CURVATURE_SOURCE_FUNCTION`, `CURVATURE_CONVENTION`.
- **No control, null or calibration constant.** `CONTROLS`, `STRATIFICATION_FIELD`,
  `STRATA_GRID`, `STRATIFIED_NULL_DRAWS`, `N_PERMUTATIONS`, `PERMUTATION_SEED`,
  `NULL_CONSTRUCTION_RULE`, `N_BOOTSTRAP`, `BOOTSTRAP_RULE`, `POSITIVE_CONTROL_*`,
  `SHUFFLED_LABEL_*`, `SEED_HANDLING_RULE`, `TORCH_INIT_SEEDS_WAVE_B`,
  `SEED_VERDICT_COMBINATION_RULE`, `WAVE_B_TRIGGER_RULE`.
- **Nothing in `physics_labels.py`** — the row alignment, label map, sentinels and margin ratified
  in `09-DATA-MANIFEST.md` § 7 and proved in `09-ALIGNMENT-PROOF.md` stand; the alignment proof
  is not re-run (it never touches the decoder).
- **No other sealed module** (`decoder_curvature.py`, `cae.py`, `chart_curvature.py`,
  `crossmodal_curvature.py`, …) and nothing in `src/effdim/`.
- **The verdict itself is not touched by this document.** The re-run produces it under the new
  freeze; this document does not anticipate it.

## Why this is an amendment and not a post-hoc adjustment

- It changes what the gating field measures to **what the pre-registration always said it
  measured**: D9-11 and `RADIAL_DECOMPOSITION_RULE` define `H_tan_norm` as the curvature
  tangential to the sphere, with `H_rad` as the sphere's own `-d` backstop. The old freeze's
  instrument did not deliver that quantity; the closed-form tests above prove the projected one
  does. Nothing about the hypothesis, the statistic, the controls or the decision rule moves.
- It was raised **from a failed pre-registered check** (`H_rad` within 10% of `-d`,
  `09-08-PLAN.md` must-haves) and diagnosed on analytic decoders, not from any Physics number
  being convenient or inconvenient. The old verdicts are `DOES NOT CLEAR` at every `d`; the
  direction of the projected result is not known in advance.
- **Every number produced under the old freeze is retained, never replaced.** The re-run writes
  to a SEPARATE output root; the original Wave A record and anchor tables are preserved; the
  final reporting shows old beside new, labelled by freeze SHA.
- It is recorded in its own numbered document, carried by its own freeze commit, and gated by
  the same strict-ancestor proof as the original — `09-PREREGISTRATION.md`'s own closing rule for
  the "only remedy" after a number exists.

## The re-run: separate output root, originals preserved

- **On the execution host:** `EFFDIM_09_OUTPUT_ROOT=/mnt/ssd-cluster/effdim/phase9-out-amend01`
  (a fresh directory; the Wave A / Wave B root `/mnt/ssd-cluster/effdim/phase9-out` is not
  written to again). `HF_HOME` stays at the host's persistent cache. Every mode is gated on the
  amendment freeze SHA, run inside `tmux`, on available compute only, per `CLAUDE.md`'s
  remote-compute section and the pod user guide read in full first.
- **Locally:** the returned bundle is extracted under a separate root (set
  `EFFDIM_09_OUTPUT_ROOT` to a fresh directory such as `notebooks/.cache/09-amend01/` before
  `--mode verdict` / `--mode bundle`), so `notebooks/.cache/09_physics_curvature.jsonl`, the
  `09_anchor_table_d{16,20,25,32}_*.npz` files and the archived
  `09-artifacts-pod128-20260904T181406Z.tar.gz` from Wave A, and Wave B's own bundle, stay
  byte-identical.
- Modes re-run: `dsweep`, `positive-control` (x4), `shuffled-label` (x4), `verdict`, `seeds`
  (if triggered), `bundle`. `proof` / `search` / `manifest` are not re-run — `physics_labels.py`
  is untouched and the alignment proof does not depend on the decoder.
- Every record row under the new root carries the amendment freeze SHA in `freeze_commit` and the
  literal `decoder_image_projection = "sphere"` in its fit rows, so old and new rows can never
  be confused.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Amendment 01 — supersedes `09-PREREGISTRATION.md` (freeze `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`) in full*
