# Phase 5: Curvature-Conditioned Linear Decodability - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Measure whether **linear crossmodal decodability degrades as manifold curvature increases**.

Concretely: fit one linear map `W : hsc -> legacysurvey` on frozen PU embeddings, score
held-out per-point residuals, bucket those residuals by the decoder-side mean-curvature
magnitude `||H||`, and test — under a rule frozen before any PU number exists — whether
decodability is worse where `||H||` is larger.

**This is NOT Phase 4 repeated.** Phase 4 split on curvature *direction* (sign of the
projection onto the leading eigenvector of the unit-`H` covariance) and measured *MKNN
neighbourhood overlap*. Phase 5 splits on curvature *magnitude* and measures *linear
predictability*. Different axis, different statistic, different failure modes.

**In scope:** the probe, the decoder-side `||H||` field, the bucketing, the pre-registration,
the verdict, and the phase record.

**Out of scope:** validating the curvature field (see D-01), any new estimator, any CAE
retraining, any architecture change, any external astrophysical label.

</domain>

<decisions>
## Implementation Decisions

### Probe target and protocol

- **D5-01:** The probe predicts **the other modality**: a linear map `W : hsc[i] (768) ->
  legacysurvey[i] (768)` on frozen embeddings. The cached PU subsample carries no label
  column (`hsc`, `legacysurvey`, `hsc_norms`, `ls_norms`, `row_indices` only), so there is
  no external response variable available without sourcing and row-aligning new data. The
  other-modality target needs no new data and is the linear analogue of exactly what Phase 4
  measured with MKNN, which makes the two phases' results directly comparable.

- **D5-02:** **Fit globally, evaluate per-region.** One `W` is fit on a held-out training
  split drawn from the whole manifold; per-point residuals are computed on the test split and
  then bucketed by `||H||`. One model everywhere, so any bucket-to-bucket difference is a
  property of the data's local decodability rather than of fitting different models to
  different amounts of data. — **Reversibility:** costly — the alternative (per-region fits)
  changes what the verdict means, and switching after numbers exist would be a post-hoc
  re-specification of the pre-registered statistic.

  **Explicitly rejected:** per-region independent fits as the headline. Smaller buckets would
  get noisier `W` purely from having fewer training points, reintroducing the sample-size
  artifact that undercut Phase 4's verdict (see D5-08).

### The curvature field

- **D5-03:** The split field is **decoder-side** mean curvature — autodiff through the CAE
  chart decoder (`notebooks/pu_manifold/decoder_curvature.py`), not Phase 4's point-cloud
  `centroid_mean_curvature`. This is the quantity the user asked for and it is what makes
  Phase 5 a test of the *decoder's* geometry rather than of the raw point cloud's.

- **D5-04:** **Pool the three cached CAE seeds into one averaged `||H||` field**
  (`03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt`); the pooled field is the verdict field.
  — **Reversibility:** one-way — the pooled field is what the pre-registered bucket edges are
  cut on; changing the pooling rule after freeze invalidates the pre-registration.

- **D5-05:** **Inter-seed agreement is measured and reported as a diagnostic, and does not
  change the verdict.** Report pairwise `spearman` between the three seeds' `||H||` fields and
  between each seed and the pooled field. Phase 03.1 found the curvature ordering was *not*
  seed-consistent, so pooling may be averaging fields that share no signal. If the seeds
  disagree, the record must say the pooled field is an artifact of averaging — while still
  reporting the verdict the frozen rule produces. This makes the pooling assumption falsifiable
  without making the verdict contingent on it.

- **D5-06:** `CURVATURE_CONVENTION = "trace"` — `H = tr_g(II)` unnormalized, a unit `d`-sphere
  giving `||H|| = d`. Non-negotiable per the spike-findings requirements; this codebase has
  already shipped and fixed one factor-of-`d` bug.

### Split axis and bucketing

- **D5-07:** Split on **`||H||` magnitude**, not Phase 4's direction sign. Bucket edges are
  pre-registered (D5-09) before any PU probe number exists. A continuous
  `spearman(||H||_i, residual_i)` over all test points is reported alongside the bucketed
  comparison as the binning-free version of the same question.

- **D5-08:** **Bucket sizes must be reported, and the bucketed comparison must be checked
  against a size-matched version.** Phase 4's HOLDS verdict was substantially explained by a
  2x region-size imbalance acting through MKNN's `k/n` chance floor. A residual-based statistic
  does not have that specific chance floor, but the general lesson stands: report `n` per
  bucket and confirm the effect survives subsampling the larger bucket to equal `n`.

### Scientific-conduct guarantees

- **D5-09:** **Full pre-registration freeze, Phase 4 discipline.** Bucket edges, the probe
  protocol, the train/test split rule, the seed-pooling rule, the scoring metric, the seed,
  and the full `VERDICT_RULE` text are frozen as named constants in committed source, plus a
  committed `05-PREREGISTRATION.md`, **before any PU probe number exists**. Git ancestry must
  be provable: the freeze commit must be an ancestor of the first commit carrying a probe
  number, and `git diff <freeze> HEAD -- <constants file>` must be empty at verification.
  This is the one guarantee Phase 4 kept completely intact and it is being deliberately
  repeated. — **Reversibility:** one-way — the entire evidential value of the verdict depends
  on the freeze preceding the numbers; it cannot be reconstructed after the fact.

- **D5-10:** The runner must **refuse to compute a bucketed probe number** unless the
  pre-registration constants and the frozen `||H||` field artifact both already exist —
  a hard guard that raises rather than computing, mirroring Phase 4's `--mode regional` guard.

### Accepted gaps (deliberate, stated up front)

- **D5-11:** **Phase 5 runs with no known-answer anchor, and this is a deliberate choice.**
  The sealed `d=20` decoder row is `rank_spearman_rho = -0.015106571347065712` — decoder-side
  curvature at `d=20` has essentially zero rank correlation with analytic curvature on the only
  control that tests it, and direction is near a coin flip (52-75% of points anti-aligned).
  **A Swiss roll / low-`d` anchor stage was offered and declined.** The consequence, which must
  be stated in `05-FINDINGS.md` in the phase's own words rather than by reference: any
  relationship this phase measures between `||H||` and probe residual rests on a field with no
  demonstrated relationship to true curvature, so a detected effect cannot be attributed to
  curvature by anything in this phase.

  Unresolved mitigating context, to be reported but **not** used to upgrade the result: the
  sealed saddle control sets a constant analytic Hessian, so `||H||` varies there only through
  the pullback metric. It may be structurally unable to show ordering, which would make
  `-0.015` a fact about the fixture rather than about the decoder. That question is open and is
  explicitly not for autonomous action.

- **D5-12:** The CAE underlying the decoder **failed its own validity gate**
  (`CAE_VERDICT = FAIL`, Phase 02.2), Phase 3 runs on a deliberate override of that gate, and
  Phase 03.1 found the metric repaired by the `scale` prior but the ordering only partially and
  non-seed-consistently moved. Every Phase 5 number inherits that chain and the record must say so.

- **D5-13:** **The density confound is expected to be weaker here than in Phase 4, and this
  must be verified rather than assumed.** Phase 4 measured `spearman(density, ||H||) = -0.0273`
  (essentially nil) against `spearman(density, signed_projection) = +0.8208`. The confound that
  wrecked Phase 4's attribution attached to curvature *direction*; Phase 5 splits on
  *magnitude*. Re-measure `spearman(density, ||H||)` on the **decoder-side pooled field** — the
  Phase 4 number was measured on the point-cloud field and does not automatically transfer.
  Report it either way.

### Claude's Discretion

- Train/test split fraction and cross-validation scheme (subject to being frozen at D5-09).
- Residual metric details: per-point squared error vs cosine vs normalized residual — planner
  chooses, then freezes. Whichever is chosen, `R^2` and a per-point residual must both be
  derivable so the bucketed and continuous versions share one underlying quantity.
- Number and placement of `||H||` buckets (tertiles vs quartiles), frozen at D5-09.
- Whether the probe is fit on raw or re-normalized embeddings, given both modalities are
  already L2-normalized upstream.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scientific-conduct precedent (the pattern being repeated)
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-PREREGISTRATION.md` — the
  freeze format, the ratification note convention, and the constants-in-source pattern
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-FINDINGS.md` — the
  accepted-gaps standard Phase 5's record must match, incl. the region-size artifact write-up
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-VERIFICATION.md` — how
  the ordering guarantee was proved mechanically from git ancestry

### Curvature at high `d` — read before touching the field
- `.claude/skills/spike-findings-effdim/SKILL.md` — requirements: trace convention, never report
  a rank statistic without the direction axis, `r/R` disclosure, no sealed-module edits from a spike
- `.claude/skills/spike-findings-effdim/references/high-d-curvature-feasibility.md` — the sealed
  `d=20` decoder row `rho = -0.015106571347065712`; the four measured dead ends; the direction
  coin-flip; the `r/R` wall
- `.claude/skills/spike-findings-effdim/references/curvature-estimator-validation.md` — the
  anchor-at-low-`d` protocol Phase 5 is deliberately declining (D5-11)

### The decoder chain Phase 5 inherits
- `.planning/phases/02.2-chart-autoencoder-validity-test-inserted/02.2-FINDINGS.md` — `CAE_VERDICT = FAIL`
- `.planning/phases/03.1-decoder-metric-regularization-inserted/03.1-FINDINGS.md` — metric repaired
  by `scale`, ordering only partially and non-seed-consistently moved
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-NOTE-phase-2-stage-on-hold.md` §3 —
  the deliberate override under which Phase 3 and everything downstream runs

### Project rules
- `CLAUDE.md` — additive only; never modify `src/effdim/` during v1.1; notebooks committed with
  outputs, executed end to end; the Swiss roll rule and when it applies

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `notebooks/pu_manifold/decoder_curvature.py` — `plain_decoder_curvature(model, z)`; the
  decoder-side autodiff path. Exists and is tested; Phase 5 writes no new estimator.
- `notebooks/pu_manifold/curvature.py` — `mean_curvature_vector(jacobian, hessian)`, the
  trace-convention core.
- `notebooks/pu_manifold/curvature_probe.py` — `mean_curvature_norm(H_vec)` for the `||H||`
  magnitudes; `centroid_mean_curvature` is Phase 4's point-cloud path, **not** Phase 5's field.
- `notebooks/.cache/03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt` — the three sealed CAE fits.
- `notebooks/pu_manifold/cache.py` — cache-path and manifest conventions used by every runner.
- Phase 4's `notebooks/diagnostics/region_partition_mknn_run.py` — the two-column PU loader
  (`load_pu_pair`), the JSONL append convention, the `--selfcheck` / `--smoke` / real-mode
  structure, and the pre-registration guard pattern. **Read for pattern; extend nothing in it.**

### Established Patterns
- Constants + `VERDICT_RULE` + `assert_preregistered()` live in a sealed
  `notebooks/pu_manifold/` module; the diagnostics runner imports them and refuses to run
  without them (Phase 4's `region_partition.py` is the reference implementation).
- Every runner writes one JSONL row per cell with full provenance (seed, subsample path, all
  frozen constants echoed) into `notebooks/.cache/`.
- Known-answer self-check as a `--selfcheck` mode, run before any real number.

### Integration Points
- New module `notebooks/pu_manifold/linear_probe.py` (probe fit/score + frozen constants) and
  new runner `notebooks/diagnostics/curvature_probe_decodability_run.py` — both additive.
- Reads the same resolved subsample npz Phase 4 used; the resolution rule (lexicographically
  first on a row-count tie) must match so both phases describe the same 10,000 points.

</code_context>

<specifics>
## Specific Ideas

The user's original framing, recorded verbatim as the shape to build:

1. fit CAE onto PU embedding
2. using decoder side, get measures of mean curvature across the manifold M, which we say the
   decoder approximates
3. fit a linear probe on frozen layer, predicting response of PU embeddings. Then seeing probe
   performance across various spots of mean curvature
4. determine if linear probe performance is worse at high curvature and better at low curvature

Steps 1 and 2 are already-built machinery (three cached fits; `decoder_curvature.py`). Step 3's
"response" was underspecified — there is no label in the data — and was resolved to the other
modality at D5-01. Step 4 is the verdict, frozen at D5-09.

The hypothesis as stated: **probe `R^2` decreases as `||H||` increases.** The pre-registered rule
must make "no detectable relationship" a complete and valid outcome, not a near-miss.

</specifics>

<deferred>
## Deferred Ideas

- **Swiss roll / low-`d` anchor for the probe methodology.** Offered and declined for Phase 5
  (D5-11). Would establish that the probe-vs-curvature method detects a known effect where
  curvature is genuinely recoverable, before asking it about PU. Still the single highest-value
  follow-up if Phase 5's result is ambiguous.
- **Resolving the saddle-control fixture question** — build a `d=20` fixture with a
  non-constant analytic Hessian and re-score the decoder, to settle whether `rho = -0.015` is
  the decoder's failure or the fixture's. Its own phase; blocks nothing here.
- **An external astrophysical label** (redshift, magnitude, morphology) as the probe target.
  Scientifically the most interesting framing and the one closest to a representation-quality
  claim, but requires sourcing labels and proving row-alignment against `row_indices` first.
- **Per-region independent probe fits at matched `n`**, as a sensitivity analysis alongside the
  global fit. Rejected as the headline at D5-02; could be added later without disturbing the
  pre-registration if declared as sensitivity-only.

</deferred>

---

*Phase: 5-curvature-conditioned-linear-decodability*
*Context gathered: 2026-08-24*
