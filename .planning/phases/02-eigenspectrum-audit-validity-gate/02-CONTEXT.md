# Phase 2: Eigenspectrum Audit & Validity Gate - Context

**Gathered:** 2026-07-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 audits the classical-MDS eigenspectrum of the Phase 1 Isomap fit, freezes the
embedding dimension `d`, and emits a machine-readable PASS/MARGINAL/FAIL gate verdict that
Phase 3 checks before spending any compute. It covers SPEC-01..07 (7 requirements).

Concretely: load `isomap.dist_matrix_` from `notebooks/.cache/isomap_43cf438bc944c509.joblib`,
double-centre it by hand to get the **full** 10,000-eigenvalue spectrum (never
`kernel_pca_.eigenvalues_`, which is truncated to `n_components` by construction and cannot
show a negative tail), report the negative-eigenvalue statistics against pre-registered
thresholds, locate the residual-variance elbow by a stated criterion, freeze `d`, and write
`gate_verdict.json`.

**Explicitly NOT in this phase:** the decoder, the fundamental forms, the mean curvature
field, the synthetic-control falsification test (all Phase 3, DEC-*/CURV-*), and region
partitioning or MKNN (Phase 4, REGN-*/MKNN-*).

Milestone-wide constraints that bound this phase: notebook-only work — `src/effdim/` and
`pyproject.toml` are not modified. Phase 2 appends as **§6 onward** in the existing
`notebooks/01_manifold_and_gate.ipynb` per Phase 1's D-01, and must never renumber §0-§5.

**Hard gate:** this phase's terminal artifact is the verdict. A FAIL halts the milestone
here, and that documented failure is itself a complete, reportable outcome — not an error
state to work around.

</domain>

<decisions>
## Implementation Decisions

### Gate verdict rule (SPEC-02, SPEC-03, SPEC-06)

- **D-01:** The verdict is a function of **two** statistics computed from the full spectrum:
  - `r = |λ_min_neg| / λ_max_pos` — the standard non-Euclideanity diagnostic named in
    `PITFALLS.md` Pitfall 3; catches a single large negative eigenvalue.
  - `m = Σ|λ_neg| / Σ|λ|` — negative mass fraction; catches a long diffuse negative tail
    that `r` alone reads as clean (500 negatives each at `0.01 · λ_max` pass on `r` while
    carrying real non-Euclidean mass).

  The verdict is the **worse of the two**. Rejected: `r` alone (the blind spot above), and a
  three-statistic panel adding top-`d` variance capture (partly redundant with the SPEC-04
  elbow, and a third threshold to pre-register).

- **D-02:** Thresholds are **fixed pre-registered literals**, not derived:

  | verdict | condition |
  |---|---|
  | PASS | `r < 0.10` **and** `m < 0.05` |
  | MARGINAL | `r < 0.25` **and** `m < 0.15` |
  | FAIL | otherwise |

  `r` below ~0.1 is the conventional reading of negligible non-Euclideanity. Rejected:
  stricter cutoffs (`r < 0.05`/`m < 0.02`) — foundation-model geodesic matrices routinely
  carry a visible negative tail, user chose not to raise FAIL odds on a convention; and
  empirical calibration against known-Euclidean/curved controls — most defensible, but
  costs 2+ extra Isomap fits and needs `r`/`m` comparable-across-`n` argument.
  Reversibility: one-way in practice — pre-registered *before* the spectrum is seen so they
  cannot be revised after. Changing them once known is the garden-of-forking-paths failure
  this design prevents; a later change is a documented amendment, not a quiet edit.

- **D-03:** Pre-registration **mirrors the Phase 1 §4.0 mechanism** exactly: a `§6.0`
  constants cell holding the `r`/`m` cutoffs, guarded by a **cell-index assertion** that it
  executes before the double-centring cell, with the literal threshold values copied
  **verbatim into `gate_verdict.json`** so the artifact carries its own decision rule. One
  pattern across the whole notebook. Rejected: a git-committed constants module (stronger
  timestamp proof, but a reader of the notebook alone cannot see the guarantee), and both
  together (belt-and-braces, costs a module plus a separate commit step).

- **D-04:** The verdict is **spectral only**. The Phase 1 provenance fields —
  `short_circuit_risk=False`, `k_auto_extended=False`, `n_components_no_headroom=True`,
  `k_star=15`, `fit_key` — are copied into `gate_verdict.json` as provenance so a reader
  sees exactly which fit was audited, but they **never move the verdict**. Rejected: a
  flag-downgrade rule capping a forced-connectivity fit at MARGINAL (honest about D-11's
  tension, but inert here — both flags are False and a re-fit keeps `k*=15`), and a full
  composite verdict.

### Elbow, frozen `d`, and the re-fit branch (SPEC-04, SPEC-05)

- **D-05:** The elbow is located by **kneedle / maximum curvature** on the
  residual-variance-vs-dimension curve, via an explicit deterministic implementation
  (normalize both axes, take the point of maximum distance from the chord). The **d-axis
  sweep range must be pre-registered alongside the thresholds in §6.0** — kneedle's answer
  depends on how far out the curve runs. Rejected: a residual-below-constant cutoff (answers
  a related but different question; the requirement's word is "elbow"), and largest eigengap
  (fragile when the spectrum decays smoothly).

- **D-06:** **Both** residual curves are computed and plotted together:
  - **Tenenbaum residual variance**, `1 − R²(D_geodesic, D_embedded)` — the canonical Isomap
    definition and **the curve the elbow is read from**.
  - **Eigenvalue-based residual**, `1 − cumsum(λ_pos)/Σλ_pos` — free once the spectrum
    exists, carried as a cross-check.

  Where the two curves **diverge** is itself a non-Euclideanity signal that reinforces the
  `r`/`m` gate — report it, don't just plot it.
  **Consequence the planner must carry:** this forces eigenvectors out of the §6 spectrum
  step (see D-09), and the R² computation needs point-pair subsampling to stay tractable at
  n=10,000.

- **D-07:** If the elbow lands **at or below 18**, freeze `d = elbow` and take columns
  `0..d-1` of the cached 18-d embedding. Classical-MDS eigenvectors are **nested**, so that
  slice *is* the exact d-dimensional solution — no re-fit, no new cache entry. **State the
  nesting argument explicitly in the notebook** so the slice is not mistaken for an
  approximation. Downstream win: smaller decoder input means fewer parameters, a
  better-conditioned first fundamental form (CURV-04), and tighter coordinate support for
  CURV-08. Rejected: `d = 18` always (carries near-noise directions into the decoder and the
  curvature field, and the frozen `d` would then not follow from the SPEC-04 curve as SPEC-05
  implies), and an elbow-with-floor rule (another pre-registered constant, and exactly the
  kind that invites post-hoc adjustment).
  — **Reversibility:** one-way — `d` is the decoder's input dimension, the coordinate space
  the curvature field lives on, and the basis for Phase 4's region partition. Changing it
  after Phase 3 invalidates the decoder, the curvature field, and every MKNN number.

- **D-08:** If the elbow **exceeds 18**, the notebook **halts and a human decides**. Halt
  message states: elbow value, required `n_components`, cost (fresh Isomap fit, minutes of
  compute, another ~1.55 GiB cache entry under a new `fit_key`), and the exact constant to
  change. A human edits `ANALYSIS_CFG` and re-runs — same documented-halt-with-remediation
  posture as Phase 1's D-11 k-ceiling branch and SPEC-07's FAIL path. Rejected: auto-re-fit
  (a Run All would silently spend minutes and ~1.55 GiB, execution-order-dependent record),
  and a pre-emptive wide fit at `n_components=40` (removes the branch but pays the re-fit
  unconditionally, quietly reversing D-12's no-headroom choice).
  *Note:* this branch is live, not hypothetical — `n_components_no_headroom=True` is set in
  the Phase 1 handoff precisely to flag it.

### Spectrum computation (SPEC-01)

- **D-09:** Double-centring uses the **in-place mean form** — subtract row means, subtract
  column means, add the grand mean, operating on the squared-distance array — which is
  algebraically identical to `B = -0.5 · J D² J` but avoids two dense 10,000³ GEMMs and holds
  roughly one extra 800 MB array instead of three. **Guard it:** assert the mean form and the
  literal `J` form agree to floating tolerance on a small random matrix, so the optimisation
  is *verified*, not assumed. Rejected: the literal `J` matmul from `PITFALLS.md` (several GB
  peak and minutes of BLAS for a result the mean form gets for free), and float32 — `r` and
  `m` are statistics about the smallest, near-zero end of the spectrum, which is exactly where
  float32's ~1e-7 relative error lives; it can manufacture or erase the very negative tail
  the gate measures.

- **D-10:** **Split eigensolve**, two calls on the same `B`:
  1. `scipy.linalg.eigvalsh(B)` — all 10,000 eigenvalues. `r` and `m` need the whole negative
     tail, so nothing less will do. Values-only avoids carrying a 10k×10k eigenvector array.
  2. `scipy.linalg.eigh(B, subset_by_index=[n-K, n-1])` — the top-`K` eigenvectors the R²
     residual curve needs past `d=18`. **`K` is pre-registered as the kneedle sweep ceiling.**
     LAPACK `syevr`: deterministic, no random start vector, consistent with D-15's
     `eigen_solver="dense"` rationale.

  Rejected: one full `eigh` (an extra 800 MB resident and more wall clock to produce ~9,950
  eigenvectors nothing reads), and values-only + reusing the cached `embedding_` columns
  (zero extra eigen-work, but `embedding_` has only 18 columns so the residual curve could
  not extend past `d=18` and kneedle might be reading a curve that has not visibly flattened).

- **D-11:** The spectrum is **persisted as an npz cache** through the existing
  `pu_manifold.cache.npz_cache` helper, keyed on a cfg dict carrying `fit_key` **plus the `K`
  ceiling**: all 10,000 eigenvalues (~80 KB), the top-`K` eigenvectors (10,000×K, a few MB),
  and both residual curves. Under 10 MB, follows the existing D-13/D-14 artifact contract, and
  makes every later Restart-and-Run-All of §6 near-instant instead of minutes of LAPACK.

- **D-12:** `§6` loads the fit with `joblib.load(..., mmap_mode="r")` so `dist_matrix_` is
  memory-mapped rather than resident, extracts the array, **drops the `Isomap` object before
  centring**, and prints **peak RSS** alongside the timing. Cold runs stay within a few GB;
  warm runs skip the load entirely because the spectrum npz already exists. Rejected: a plain
  `joblib.load` (peak sits at 800 MB of pickle-resident matrix plus every centring
  intermediate), and adding a hard pre-flight RAM assertion (platform-dependent probe and a
  number that will need revisiting).

### Verdict artifact and enforcement (SPEC-06, SPEC-07)

- **D-13:** The artifact is `notebooks/.cache/gate_verdict_{fit_key}.json`, written through
  the existing `json_cache` contract. **Keying it by `fit_key` binds the verdict
  inseparably to the fit it audited** — a re-fit at a larger dimension produces a new
  `fit_key` and therefore a new verdict file, so there is no way to read a stale PASS against
  a changed fit. Accepted cost: `.cache/` is gitignored, so the verdict is not visible in the
  repo and must be regenerated by whoever runs the notebook. Rejected: a committed unkeyed
  `notebooks/gate_verdict.json` (repo-visible, but one file a re-fit overwrites and a
  `fit_key` mismatch only caught if a consumer checks), and both copies (sync step, and the
  two can disagree).

- **D-14:** **Enforcement is inline in notebook 02's first cell** — it opens the JSON for the
  current `fit_key` and asserts the verdict itself. **No new `pu_manifold` module.** D-02's
  four-module package (`subsample`, `cache`, `curvature`, `mknn`) stands unchanged; the
  verdict computation and the artifact write also live in the §6 notebook cells.
  *Tension recorded deliberately, user decision:* the rejected alternative was a
  `pu_manifold/gate.py` carrying `require_gate()` — one import, unit-testable alongside the
  existing 14 tests, and a single implementation for every consumer. The user chose inline,
  where the check is visible exactly where it matters rather than behind an import. **Accepted
  cost the planner must carry: the gate rule then lives in notebook prose, so Phase 4 (or any
  later consumer) re-implements it and can re-implement it differently.** Do not silently swap
  the module back in — but do make the inline check a copyable, self-contained block, and
  make D-15's self-contained schema carry enough that a re-implementation cannot drift on the
  thresholds.

- **D-15:** MARGINAL **proceeds** — the roadmap already makes Phase 3 depend on "a PASS or
  MARGINAL gate verdict" — but the verdict and its `r`/`m` values are **re-printed at the top
  of notebook 02 and carried into every downstream artifact**, so the curvature field and the
  Phase 4 MKNN numbers are never read without the caveat attached. Rejected: requiring an
  explicit `ACK_MARGINAL` constant (stronger against a Run-All sailing past a borderline
  spectrum, but a manual step on every fresh clone), and treating MARGINAL as FAIL (would
  need the roadmap amended, not just the notebook).

- **D-16:** `gate_verdict.json` is **self-contained** — a reader reconstructs the whole
  decision without opening the notebook. Fields: verdict; `r`; `m`; the three thresholds
  **verbatim**; elbow value and criterion name; frozen `d`; the d-axis sweep range and the `K`
  ceiling; `fit_key`; `k_star`; the Phase 1 flags; timestamp; library versions. **On FAIL**,
  the enumerated remediation list — re-fit at a different `k`, resample with a new seed,
  accept the documented FAIL as the milestone outcome — is written into **both** the artifact
  and the halt message. Rejected: a minimal verdict + pointer to §6 (a future reader with the
  JSON alone could not tell which thresholds produced the verdict), and adding a test that
  exercises the FAIL path on a synthetic verdict.
  *Planner note:* that rejected FAIL-path test is worth a second look during planning. SPEC-07's
  halt is a branch that will very likely never execute on real data, which is exactly the kind
  that rots untested — and D-14 put the check in notebook prose rather than in a testable
  module, which removes the other safety net.

### Claude's Discretion

The user selected a concrete option on every question — no "you decide" answers. The
following were named as open sub-decisions during the discussion and never locked:

- The **d-axis sweep range** kneedle runs over, and the **`K` eigenvector ceiling** (D-05,
  D-10) — these are the same number and must be pre-registered in §6.0 with the thresholds.
- The **point-pair subsampling scheme** for the Tenenbaum R² residual at n=10,000 (D-06):
  how many pairs, drawn from which seed, and the argument that the estimate is stable.
- The floating **tolerance** for the D-09 mean-form vs `J`-form equivalence assertion, and
  the size of the small random matrix it runs on.
- How **SPEC-02's steep-dropoff location** is defined and plotted, distinct from the SPEC-04
  elbow. Raised as a remaining gray area and not discussed — the planner picks.
- The **§6 sub-section layout and figure set** (§6.0 pre-registration, spectrum, negativity
  statistics, residual curves, `d` freeze, verdict). Raised as a remaining gray area and not
  discussed.
- Cell-output hygiene for §6's figures, inheriting §0.4's policy — plots and small tables,
  never a bulk 10,000×10,000 repr in a cell output.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

ROADMAP.md carries no `Canonical refs:` line for this phase; the list below was accumulated
from REQUIREMENTS.md, PROJECT.md, the Phase 1 CONTEXT/handoff, and the `.planning/research/`
set.

### Requirements & scope (read first)
- `.planning/REQUIREMENTS.md` §Spectral Validity Gate (SPEC) — SPEC-01..07 verbatim, the
  seven requirements this phase must satisfy
- `.planning/REQUIREMENTS.md` §Out of Scope — the exclusion table bounding this phase
- `.planning/ROADMAP.md` §Phase 2 — goal, the five success criteria, and the **Hard gate**
  note stating a FAIL halts the milestone as a legitimate complete outcome
- `.planning/PROJECT.md` §Key Decisions — locked milestone-level decisions (notebook-only,
  single model, `k*=15` frozen by the pre-registered plateau rule)

### Phase 1 inheritance (binding)
- `.planning/phases/01-data-loading-manifold-reconstruction/01-CONTEXT.md` — **D-01**
  (Phase 2 appends as §6+ in the same notebook, never renumbering §0-§5), **D-02** (the
  four-module `pu_manifold` package, unchanged by D-14), **D-05** (unconditional L2
  normalization, so the ambient space is S⁷⁶⁷), **D-12** (no-headroom `n_components`, the
  source of D-08's branch), **D-13/D-14** (cache artifact and key contract that D-11 and
  D-13 extend), **D-15** (`eigen_solver="dense"` determinism rationale)
- `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` — the 14-key Phase 1→2 interface.
  `notes_for_phase2` carries four constraints this phase is built around; `config` carries
  the fit parameters; `flags` carries the provenance D-04 copies into the verdict.
- `.planning/phases/01-data-loading-manifold-reconstruction/01-04-SUMMARY.md` — how `k*=15`
  was frozen and what was sealed
- `.planning/WINDOWS.md` window #1 — `STAGE2_K` uneven spacing; the plateau is maximal in
  index space, not k space. Open, disclosed, not acted on. Context for how much weight the
  gate should give `k*=15`.

### Implementation guidance
- `.planning/research/PITFALLS.md` **Pitfall 3** — the core reference for this phase: why
  `kernel_pca_.eigenvalues_` is a false negative by construction, the double-centring recipe
  D-09 optimises, and the `|λ_min_neg|/λ_max_pos` ratio D-01 adopts as `r`
- `.planning/research/PITFALLS.md` Pitfall 2 — short-circuit signatures; context for the
  provenance flags D-04 records
- `.planning/research/PITFALLS.md` Pitfall 4 — why normalized-then-Euclidean is the
  defensible choice *for the classical-MDS eigenspectrum audit specifically*
- `.planning/research/ARCHITECTURE.md` §Caching Strategy and §Architectural Patterns
  Pattern 1 — the config-hash checkpointing contract D-11 and D-13 write into
- `.planning/research/ARCHITECTURE.md` §Determinism and Reproducibility — the
  `eigen_solver="dense"` rationale D-10's solver choice stays consistent with

### Code
- `notebooks/01_manifold_and_gate.ipynb` §6 — the reserved Phase 2 section, currently a
  single markdown stub. §0.4 (output hygiene) and §0.5 (section numbering) are the
  conventions §6 inherits; §4.0 is the pre-registration pattern D-03 mirrors.
- `notebooks/pu_manifold/cache.py` — `config_key`, `npz_cache`, `json_cache`, `cache_path`.
  D-11 and D-13 use these directly; do not write a parallel caching path.

### External
- arXiv:2509.19453 — Duraphe, Smith, Sourav & Wu, *The Platonic Universe: Do Foundation
  Models See the Same Sky?* Origin paper. Not needed to implement Phase 2.

</canonical_refs>

<code_context>
## Existing Code Insights

No `.planning/codebase/` maps exist; this section comes from a direct scan.

### Reusable Assets
- `notebooks/pu_manifold/cache.py` — `config_key(cfg)` (sha256 over `json.dumps(sort_keys=True)`),
  `npz_cache(stem, cfg, compute_fn)`, `json_cache(stem, cfg, compute_fn)`, `joblib_cache(...)`,
  `cache_path(stem, ext)`, plus a `_manifest_matches` guard and an `_assert_inside_cache` path
  check. **D-11's spectrum npz and D-13's verdict json both go through these** — the manifest
  mechanism is what makes `fit_key` binding actually hold.
- `notebooks/.cache/isomap_43cf438bc944c509.joblib` (~1.55 GiB) — the single `k*=15` fit.
  Carries `dist_matrix_` (the 10,000×10,000 geodesic matrix SPEC-01 needs), `embedding_`
  (10,000×18), `nbrs_`, `kernel_pca_`.
- `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` — read for `fit_key`, `k_star`,
  `n_components`, `d_provisional`, and the three `flags` D-04 copies into the verdict.
- `src/effdim/` — **not called by this phase.** Phase 1 used `compute_dim` for the ISO-03
  pre-audit; Phase 2's spectrum is hand-rolled from `dist_matrix_`.

### Established Patterns
- **Pre-registration with a cell-index assertion** (§4.0) — constants declared in a cell that
  provably executes before the cell that consumes them. D-03 reuses this verbatim.
- **Documented halt with remediation enumerated** — Phase 1's D-11 k-ceiling branch reads as
  an explicit `for/else` with the remediation options spelled out in the assertion message.
  D-08 and SPEC-07's FAIL path both follow this shape.
- **Config-hash cache keys, gitignored artifacts** — `notebooks/.cache/` is already in
  `.gitignore`; no gitignore change needed.
- **Notebook committed with outputs intact** (§0.4) — no `nbconvert --clear-output`; plots and
  small tables only, never a bulk array repr.
- Notebook deps are pinned in `notebooks/requirements-notebooks.txt` and installed by the
  first cell. Phase 2 adds no new dependency — `scipy.linalg` is already core.

### Integration Points
- `§6` of `notebooks/01_manifold_and_gate.ipynb` → `notebooks/.cache/isomap_{fit_key}.joblib`
  (mmap read of `dist_matrix_` per D-12)
- `§6` → `notebooks/.cache/spectrum_{key}.npz` (new, D-11) and
  `notebooks/.cache/gate_verdict_{fit_key}.json` (new, D-13)
- `notebooks/02_*.ipynb` first cell → `gate_verdict_{fit_key}.json`, inline check per D-14
- `pyproject.toml`, `src/effdim/`, and `notebooks/pu_manifold/*.py` are **not modified** by
  this phase (D-14 declines the `gate.py` module)

</code_context>

<specifics>
## Specific Ideas

- The §6.0 pre-registration cell should hold, in one place: `R_MAX_PASS=0.10`, `M_MAX_PASS=0.05`,
  `R_MAX_MARGINAL=0.25`, `M_MAX_MARGINAL=0.15`, the d-axis sweep ceiling (which is also the
  `K` eigenvector ceiling), and the R² pair-subsample size — followed by the same cell-index
  assertion §4.0 uses.
- The equivalence guard for D-09 should read as a small explicit test inside the notebook:
  build a random 50×50 distance matrix, compute `B` both ways, `np.testing.assert_allclose`.
  Cheap, visible, and it converts "the mean form is the same thing" from a claim into a
  demonstration.
- Plot both residual curves on one axis with the kneedle point marked, so the reader sees the
  Tenenbaum curve, the eigenvalue curve, their divergence, and the chosen `d` in a single
  figure.
- The negative tail is worth plotting on its own — the sorted spectrum with zero marked, so
  the shape of the negative end (one large outlier vs a long diffuse tail) is visible rather
  than compressed into `r` and `m`.
- The FAIL halt should read like Phase 1's k-ceiling `else` branch: an assertion whose message
  spells out all three remediation options, so a reader who hits it knows what the choices are
  without going back to the roadmap.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. No scope creep was raised.

Three items surfaced as **downstream consequences or accepted costs** rather than deferred
ideas; they are recorded inline above and must not be lost:

- **D-14's accepted cost:** the gate rule lives in notebook prose, not a module, so Phase 4
  or any later consumer re-implements it and can drift. D-16's self-contained schema is the
  mitigation.
- **D-16's planner note:** the rejected FAIL-path test deserves a second look during planning —
  SPEC-07's halt is a branch that will never fire on real data, and D-14 removed the module
  that would have made it testable.
- **Carried from Phase 1 (still live for Phase 3):** because of Phase 1's D-05, the ambient
  space is the unit sphere S⁷⁶⁷, so Phase 3's CURV-06 synthetic controls must be matched
  **on the sphere**, not against a flat plane in ℝ⁷⁶⁸.

</deferred>

---

*Phase: 2-Eigenspectrum Audit & Validity Gate*
*Context gathered: 2026-07-31*
