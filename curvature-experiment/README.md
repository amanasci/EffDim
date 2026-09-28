# curvature-experiment/

The code that produces the records read by [`../paper/`](../paper/README.md). No results are
stated here; see the paper for those.

The instrument: a plain auto-encoder (three hidden layers, SiLU) is fit on embeddings and its
decoder is projected onto the unit sphere (`F/||F||`, since embeddings are unit-normalized).
Automatic differentiation of the projected decoder at a latent code gives the Jacobian `J`, the
metric `g = J^T J` and the second fundamental form `II = P_N D^2F` by the normal projector
`P_N = I - J g^-1 J^T`. The probe-facing curvature is `K = <w_N, II>`, the second fundamental
form contracted with the normal component of a fitted probe's weight vector; a label's own
intrinsic Hessian `Hess_M y` is estimated separately (from exact geometry on a known surface, or
from a local quadratic fit to the data). The code below computes these quantities and their
associations with a linear probe's local accuracy; it does not compute the paper's numbers.

## Layout

`runners/` — one script per experiment, each a `09_*` CLI (original names kept for provenance).
One line per runner, what it computes and (where distinct) the family of record it writes:

- `09_instrument_adjudication_run.py` — known-answer validation of the decoder curvature
  instrument on an explicit in-sphere generator with closed-form curvature. Writes
  `09_instrument_adjudication.jsonl`.
- `09_fixture_probe_decodability_run.py` — the sealed probe-pipeline statistic (curvature vs.
  local R^2) on the same fixture, with the density-curvature coupling set by construction.
  Writes `09_fixture_probe_decodability*.jsonl`.
- `09_fixture_probe_facing_run.py` — per-anchor exact geometry on the fixture (probe-facing
  curvature, Hessian mismatch, gradient mismatch, bias, and the residual expansion's predicted
  local R^2), term by term. Writes `09_fixture_probe_facing.jsonl`.
- `09_fixture_probe_facing_split_run.py` — splits the fixture's probe-facing curvature into its
  in-sphere (shape) and radial (sphere) parts and adds the alignment with the label's Hessian.
  Writes `09_fixture_probe_facing_split.jsonl`.
- `09_physics_curvature_run.py` — not a CLI; library helpers (`fit_and_field_at_anchors`,
  `SphereProjectedDecoder`, `_oof_predictions_for_label`, `_THREADS`) loaded by the runners
  below as the module `runner`.
- `09_physics_probe_facing_run.py` — probe-facing curvature (`<w_N, II>`) from the decoder
  instrument on the real Physics anchors. Writes `09_physics_probe_facing.jsonl`.
- `09_physics_probe_facing_split_run.py` — the shape/sphere split and the label's intrinsic
  Hessian estimated from data, on the Physics anchors; also the decoder-ablation, cross-fit,
  weak-ridge and cross-encoder variants (`--fit-seed`, `--hidden`, `--hessian-xfit`, `--alpha`,
  `--parquet-path`, `--embedding-column`, `--label-table`). Writes
  `09_physics_probe_facing_split*.jsonl`.
- `09_physics_normal_scaling_run.py` — the counterfactual intervention: scales the readout's
  in-sphere shape quadratic by `t` at a fixed manifold and scores the change in local R^2.
  Writes `09_physics_normal_scaling_*.{jsonl,npz}`.
- `09_physics_normal_scaling_thin_run.py` — recomputes the counterfactual's 512-anchor panel and
  saves the pairwise neighbourhood-overlap matrix between them, for a sign test with independent
  units. Writes `09_physics_normal_scaling_*_thin.npz`; `paper/generate/appendix_gen.py` computes
  the actual pairwise-disjoint-neighbourhood (independent) set from that matrix at generation
  time, not this runner.
- `09_row_alignment_proof_run.py` — the row-alignment proof between the embeddings and label
  tables (out-of-fold R^2 curve over a shift set). Kept as provenance for
  `ALIGNMENT_ASSUMED_OFFSET` and loaded by `test_physics_labels`; not itself a source of a paper
  number.

`pu_manifold/` — the shared, notebook-scoped helper package (never installed, imported as a
plain relative package). One line per module:

- `cache.py` — `CACHE_DIR` resolution (`EFFDIM_CACHE_DIR`, else `curvature-experiment/.cache`)
  and the config-hash-keyed `npz_cache` with sidecar-manifest verification.
- `cae.py` — the plain auto-encoder (`PlainAutoEncoder`) used as the decoder instrument.
- `chart_curvature.py` — exact mean curvature through a decoder, by `torch.func` autodiff.
- `curvature_probe.py` — a local mean-curvature estimator (arrays in, arrays/dicts out).
- `decoder_curvature.py` — exact mean curvature through a decoder with no chart routing
  (`chart_curvature.py` with the chart-index composition removed); `plain_decoder_curvature`.
- `crossmodal_curvature.py` — `split_indices`, the seeded train/holdout split the auto-encoder
  fit uses (the rest of the Phase 7 module is archived).
- `cross_split_curvature.py` — split-half cross statistics for a mean-curvature field.
- `physics_curvature_probe.py` — the Phase 9 statistics module: pre-registration constants and
  guard, the out-of-fold ridge wrapper, the anchor draw, the radial/tangential decomposition,
  the three-control partial Spearman, the Freedman-Lane null and the verdict rules.
- `physics_labels.py` — loads the `UniverseTBD/pu-embeddings` physics embeddings and the
  `Smith42/galaxies@v2.0` catalogue labels, and proves their row alignment.
- `linear_probe.py` — `fit_probe`/`predict_probe`, the `RidgeCV` wrapper that
  `physics_curvature_probe.oof_ridge_predictions` calls (the rest of the Phase 5 module is
  archived).
- `subsample.py` — the seeded row draw `draw_row_indices` and `l2_normalize` (the PU loader and
  its alignment check are archived); `draw_row_indices` is also how
  `physics_curvature_probe.anchor_indices` draws the 512 holdout anchors, not just how the
  row-alignment proof draws its rows.
- `__init__.py` — package docstring and re-exports of the `cache` and `subsample` names above.

Definitions cut from these modules during the closure are kept verbatim under
`archive/pu_manifold_trimmed/`.

`tests/` — unit tests for the modules above and the runners' CLI/monkeypatch surface.

`notebooks/` — Swiss roll sanity checks (per the root `CLAUDE.md` rule): `02.6_swiss_roll_plainae_curvature_check.ipynb`
tests the plain auto-encoder's curvature estimate; `09.1_swiss_roll_probe_decodability_check.ipynb`
and `09.2_swiss_roll_density_decoupling_check.ipynb` are the low-dimensional first pass of the
probe-decodability design that `09_fixture_probe_decodability_run.py` later ran at production
scale (`09-SUPPLEMENT-03` in the archived planning docs).

## Checks

```bash
pytest curvature-experiment/tests
python curvature-experiment/runners/<runner>.py --mode smoke ...   # every runner but the thin one supports --mode smoke
```

`09_physics_normal_scaling_thin_run.py` has no smoke mode; `--help` is expected to parse and its
imports to resolve.

Maintainer-only provenance for the restructure: `docs/superpowers/harness/gate.sh <tree> <label>`
(for example `gate.sh . mycheck`) is the equivalence gate that compared smoke output, both test
suites and the generators' output against the pre-closure baseline. It is not runnable from a
fresh clone: it needs a local baseline under `$CLOSURE_WORK` (default `~/.cache/effdim-closure`,
not in the repository) and hard-codes the author's venv
(`/home/akagi/Documents/Projects/EffDim/.venv`) and record directory
(`/home/akagi/Documents/Projects/EffDim/notebooks/.cache`).

See [`REPRODUCE.md`](REPRODUCE.md) for the exact invocations that produced the records the paper
reads, the pod-only inputs, record hashes and caveats.
