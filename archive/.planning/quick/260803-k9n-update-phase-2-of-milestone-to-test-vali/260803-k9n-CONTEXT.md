# Quick Task 260803-k9n: Update milestone docs — Chart Autoencoder validity test phase - Context

**Gathered:** 2026-08-03
**Status:** Ready for planning

<domain>
## Task Boundary

Documentation-only update to `.planning/ROADMAP.md` (and STATE.md tracking). Insert a new decimal phase **02.2 "Chart Autoencoder Validity Test" (INSERTED)** after Phase 02.1, defining an empirical validity test of the Chart Auto-Encoder (CAE) method (Schonsheck, Chen, Lai — arXiv:1912.10094) on the PU embedding data, with an explicit PASS/FAIL gate:

- **PASS** → milestone proceeds to Phase 3 (decoder & curvature field, decoding from CAE representation)
- **FAIL** → findings documented, milestone stays in the phase-2 stage (Phase 3 remains blocked)

No implementation in this task. The user will prompt separately to plan the implementation of Phase 02.2 after this doc update lands.
</domain>

<decisions>
## Implementation Decisions

### Where the CAE test lands (user-ratified via AskUserQuestion)
- **Insert Phase 02.2** as a new INSERTED decimal phase after 02.1. Chosen over amending Phase 2 (closed FAIL-verdict audit phase, 1 plan remaining) and over folding into 02.1 (literature-only, no-package-installs constraints conflict with empirical training).
- Phase 02.1 survey stays intact. Phase 2's remaining plan 02-03 (gate verdict artifact) unaffected.
- Phase 3's dependency must be amended: depends on Phase 02.2 PASS (in addition to 02.1's representation decision context).

### Gate semantics (from user directive)
- Success = move onto Phase 3. Fail = document and stay on phase-2 stage.
- Gate should mirror the existing machine-readable-verdict pattern (Phase 2's `gate_verdict.json` precedent).

### Claude's Discretion
- Exact success-criteria wording, requirement IDs (suggest CAE-01.. series), and which validity metrics from the paper to pre-register (reconstruction error, unfaithfulness/coverage, chart-transition cycle residual R_cycle, a-posteriori chart count, topology/proximity preservation).
- How to phrase the ReLU conflict: paper's reference implementation uses ReLU; Phase 3's CURV requirements demand C2-smooth activations (no ReLU family) — the roadmap text must flag that the CAE decoders need smooth activations for Phase 3 to consume them.
</decisions>

<specifics>
## Specific Ideas — CAE paper facts the roadmap text can cite

Paper: "Chart Auto-Encoders for Manifold Structured Data", Schonsheck, Chen, Lai, arXiv:1912.10094v2 (2020).

- **Architecture:** initial encoder E: R^m → R^l (l ≈ 2d minimal near-isometric embedding, Nash–Kuiper motivation); N chart encoders E_α: R^l → chart spaces Z_α = (0,1)^d; per-chart decoders D_α: Z_α → R^l; one shared embedding decoder D: R^l → R^m; chart predictor P outputs partition-of-unity probabilities p_α.
- **Assumption relief:** only *local* charts are Euclidean — no global Euclidean embeddability of the manifold/geodesic metric is assumed. This is precisely the flat-target assumption Phase 2's gate falsified (m = 0.412071, 41% negative eigenvalue mass). CAE is a coordinate-producing candidate under the 02.1 fork.
- **Loss (eq. 3):** L(x,W) = min_α e_α − Σ_β ℓ_β log(p_β), where e_α = ‖x − y_α‖², ℓ = softmax(−e_α). Plus Lipschitz regularization R_Lip (eq. 4) on chart-encoder spectral norms; FPS-seeded per-chart pre-training (eq. 5); optional PCA-based chart orientation init (eq. 6).
- **Chart count:** over-specify N; unused charts decay under weight decay + regularization and are pruned by decoder-weight-norm tolerance — chart count obtained *a posteriori*.
- **Transition consistency check (eq. 8):** cycle residual R_cycle(x) re-encodes decoded data through a second chart — measures chart-transition + reconstruction error. Natural validity metric.
- **Theory:** universal manifold approximation theorem (Thm 2) for compact d-dimensional data manifolds with reach τ: ε-faithful representation with L > d charts; latent space homeomorphic to the manifold (Thm 1: plain single-chart AE *cannot* ε-faithfully represent non-contractible topology).
- **Empirical validity metrics used in paper:** reconstruction error, *unfaithfulness* (distance of latent-sampled generations from training set), *coverage* (fraction of training modes hit), topology preservation (latent-space proximity of consecutive frames / no periodic breaks), chart overlap smoothness.
- **Known friction with Phase 3:** reference implementation is TensorFlow + ReLU activations; CURV-01..03 require C2-smooth activations throughout the decoder — activation swap (e.g., tanh/softplus/SiLU) required and must be stated in the phase text. Also: theorem covers *compact* manifolds; PU point cloud compactness is an assumption to note, not verify.
- **Caveat to record:** CAE validates topology/geometry preservation *of a reconstruction*; it does not by itself adjudicate the 41% negative-mass geometric reading (02.1-04's terminal artifact still owns that judgment).
</specifics>

<canonical_refs>
## Canonical References

- arXiv:1912.10094 (Chart Auto-Encoders) — method under test
- `.planning/ROADMAP.md` — Phase 2 / 02.1 / 3 text and the INSERTED-phase precedent (02.1)
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md` — FAIL evidence the new phase text should reference
- Phase 2 `gate_verdict.json` pattern — precedent for the machine-readable PASS/FAIL artifact
</canonical_refs>
