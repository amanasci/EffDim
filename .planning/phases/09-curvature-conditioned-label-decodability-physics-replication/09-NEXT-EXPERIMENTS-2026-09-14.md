# Next experiments (designed 2026-09-14, not yet run): known-surface alignment demo and cross-encoder probe-facing test

Purpose: turn "curvature helps a probe where the manifold bends the way the label bends" from a
theorem plus one-encoder association into (K) a demonstration with known geometry and (X) a
cross-encoder test. Both reuse existing runners; pod guide sha `5e94dd31…` verified 2026-09-14.

## K. Known-surface alignment demonstration (local or pod, ~30 min)
Runner to extend: `notebooks/diagnostics/09_fixture_probe_facing_split_run.py` (imports the fixture
probe-facing runner unchanged). Generator `adj.InSphereGenerator`: surface is normalize([stereo(z); 0.8 h(z); 0…])
with h(z) a scalar sum of four Gaussian bumps, so the in-sphere bending direction is the h-coordinate and
II^S ∝ 0.8·Hess h(z) (after Christoffel correction).
Add two labels: `bump_plus`: y = a·z + β h(z); `bump_minus`: y = a·z − β h(z), β chosen so that
‖Hess_M y‖ is comparable to ‖⟨w_N, II^S⟩‖ (start β = 0.8; check medians of `hess_label` vs `pf_tan` and
adjust). Label gradients/Hessians: central finite differences of h in the chart (fd step `fx.FD_STEP`),
then covariant correction `Hess_M y = ∂²y − Γ·∂y` as `pf.label_derivatives` does for the others.
Predictions (sealed three-control partial vs local R², γ ∈ {−1, 0.6}):
  - `align_cos_tan` partial: positive for bump_plus, negative for bump_minus;
  - `pf_tan` (shape) partial: positive for bump_plus, negative for bump_minus;
  - `hess_mismatch` partial: negative for both;
  - `pf_rad` (sphere term) unaffected in sign.
A pass = all four hold at both γ. Record `notebooks/.cache/09_fixture_alignment_demo.jsonl`; write
Supplement 10; one sentence in Section 5 ("on a known surface the shape-term sign flips with the
sign of the label's bending") and a row in Figure 1a or Appendix.

## X. Cross-encoder probe-facing test (pod, ~1 h per encoder in parallel)
Runner: `09_physics_probe_facing_split_run.py --fit-seed 0 --hessian-xfit --label-table <cached parquet>`
with two new options to add: `--parquet-path` and `--embedding-column` that monkeypatch
`physics_labels.PHYSICS_PARQUET_PATH` / `PHYSICS_COLUMN` before `load_physics`, and `in_dim` taken from
`X.shape[1]` (pcp.AE_IN_DIM is 768; CLIP-B is 512, ConvNeXt-B/ViT-L 1024 — verify per file).
Embeddings: `hf://datasets/UniverseTBD/pu-embeddings/physics/{dinov3_vitb16,clip_base,convnext_base,vit_large}_test.parquet`
(listed on HF 2026-09-14). Download each with `huggingface_hub.hf_hub_download` into
`/mnt/ssd-cluster/effdim/hf-cache` first (the hf:// streaming read stalled twice on 2026-09-12) and
point `--parquet-path` at the local file. Row order must match the ViT-B file (same test split; verify
n = 86,471 and re-run the label-join check `pl.shifted_pairing` logic, or compare the row_index column if
present) — this is the one correctness gate; do not proceed if the row count differs.
d = 16, 512 anchors, k = 2,048, four labels, 2,000 permutations; one tmux session per encoder
(`bash ablate.sh`-style script; 16 threads each; four in parallel is fine on 128 cores).
Predictions: `hess_mismatch_emp` negative and `align_cos_tan` positive on mag_r/photo_z/smooth for every
encoder; `pf_tan` sign free to vary by encoder and label; `pf_rad` negative under α=100 (also run
`--alpha 1` for mag_r if time). Cross-fit rows give reliability per encoder.
Outputs: `ablate-out/09_physics_probe_facing_split_<encoder>.jsonl`; extend `appendix_gen.py` with an
Appendix D table (encoder × label: mismatch, alignment, shape, sphere); Supplement 11; replace the
"Across encoders" paragraph's last clause with the result.

## Order and gates
Run K first (cheap, decides whether the alignment mechanism is real with exact geometry). If K fails,
X is still worth running but the paper's alignment sentence must be softened further. Then X.
Everything is additive: no sealed module edits, new records only.

## Status 2026-09-15 (session end)

**K, first attempt (run, negative for the fixture, not for the mechanism).** Labels `bumpalt_{0.3,1.0}` =
a·z + β h_alt(z) added to `09_fixture_probe_facing_split_run.py` (h_alt = generator bumps with amplitude
signs [1,−1,−1,1]). Record `notebooks/.cache/09_fixture_alignment_demo.jsonl` (γ = −1, 0.6; 1,000 draws).
- d=4 smoke fixture (n=4000, D=64, k=128): alignment partial **+0.43 / +0.45** (p=0.005), mismatch
  −0.64 / −0.60, shape term n.s. — the mechanism shows where bending is resolvable
  (median pf_tan 0.31–0.37 vs label Hessian 2.4–2.5, ratio ≈ 1:7).
- d=16 production fixture, γ=−1: alignment **−0.03 (n.s.)** at both β; mismatch −0.71; shape term −0.41
  (sampling-coupled). Medians: pf_tan 0.04–0.05 vs hess_label 24.6, ratio ≈ 1:500. The stereographic chart's
  Christoffel term makes the covariant Hessian of any z-dependent label ≈ 25, so the in-sphere bending is
  invisible to the alignment cosine (median 0.09). This is a fixture-scale limitation, already stated in the
  paper ("cannot resolve the shape term").
- **Redesign to try (cheap):** add runner options to override `adj.FIXTURE`: `scale_choices` ×0.25 (latents
  nearer the origin, near-flat chart, small Γ) and bump amplitude `a` ×5–10 (larger in-sphere II), keep
  n, k, anchors. Target ratio pf_tan : hess_label ≳ 1:5 as in the smoke. Then re-run `bumpalt_{0.3,1.0}`
  at γ ∈ {−1, 0.6}. Pass = alignment partial positive, mismatch negative, at both γ.

**X, blocked at download.** `dl_enc.py` (hf_hub_download of dinov3_vitb16 / clip_base / convnext_base /
vit_large / vit_base physics parquets into `/mnt/ssd-cluster/effdim/hf-cache`) sat in Ceph metadata wait
(`ceph_mdsc_wait_r`, D state) for >5 min with no bytes written; tmux `effdim-dlenc`. Do not stack
filesystem commands on `/mnt/ssd-cluster` while it is in that state (guide). If it never completes, kill it
and read each parquet by streaming `pq.read_table("hf://…", columns=[col])` inside the runner (the ViT-B
embeddings read that way in ≈10 s on 2026-09-12). Runner is ready: `--parquet-path`, `--embedding-column`
(e.g. `clip_base_galaxies`; confirm column names from the parquet schema), and a loader shim that accepts
widths ≠ 768 (sealed loader hard-codes 768). Row-count gate 86,471 is enforced by the shim; row *order*
must still be checked against the ViT-B file (same `test` split; compare an id column if present).
