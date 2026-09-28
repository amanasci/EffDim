# paper/

This is the manuscript for the ML4PS 2026 submission "Linear Probes on Curved Latent Spaces",
plus the scripts that turn experiment records into its tables and figures. The paper claims
that a linear probe restricted to a curved embedding manifold has a nonzero intrinsic Hessian
(`Hess_M(w.x) = <w_N, II>`, the "probe-facing curvature"), validates a decoder-based estimate
of it on a synthetic surface of known curvature, and shows on 86,471 galaxy embeddings (five
encoders) that the mismatch between probe-facing curvature and a label's own intrinsic Hessian
is associated with local readout accuracy, with a counterfactual intervention supporting the
direction of the effect. The code that produces the records this folder reads is in
[`../curvature-experiment/`](../curvature-experiment/README.md); this folder never imports it
except through [`records.py`](records.py), which resolves where the records live.

## Paper element -> code map

| Paper element | Generator (`paper/`) | Records | Runner (`curvature-experiment/runners/`) | Modules (`curvature-experiment/pu_manifold/`) |
|---|---|---|---|---|
| `tab:real` (Appendix C main table; compiled numbering differs from the LaTeX label) | `generate/table_main_gen.py`, spliced by `generate/appendix_gen.py` | `09_physics_probe_facing_split.jsonl` | `09_physics_probe_facing_split_run.py`, from the ViT-B per-anchor geometry written by `09_physics_probe_facing_run.py`[^chain] | `physics_curvature_probe`, `physics_labels`, `cae` |
| `tab:meancurv` (Appendix E; compiled numbering differs from the LaTeX label) | hand-written in `latex/main.tex`; numbers verified with `generate/table_main_gen.py`'s check-line output | `09_physics_probe_facing_split.jsonl` | `09_physics_probe_facing_split_run.py`, from the ViT-B per-anchor geometry written by `09_physics_probe_facing_run.py`[^chain] | `physics_curvature_probe`, `physics_labels`, `cae` |
| Main-text checks (Section 5 and Appendix E prose numbers) | `generate/table_main_gen.py`'s `check_lines` function, run by hand for review; its output is not pasted anywhere and produces no committed file | `09_physics_probe_facing_split.jsonl` | `09_physics_probe_facing_split_run.py` | `physics_curvature_probe`, `physics_labels`, `cae` |
| `tab:ablate` (Appendix A, decoder ablations) | `generate/appendix_gen.py` | `09_physics_probe_facing_split.jsonl`, `_seed1.jsonl`, `_seed2.jsonl`, `_w400.jsonl` | `09_physics_probe_facing_split_run.py` (`--fit-seed`, `--hidden`) | `physics_curvature_probe`, `physics_labels`, `cae` |
| `tab:sens` (Appendix B, cross-fit and weak-ridge sensitivity) | `generate/appendix_gen.py` | `09_physics_probe_facing_split_xfit.jsonl`, `_alpha1.jsonl`, and `09_physics_probe_facing_split.jsonl` (the `alpha=100` parenthetical column) | `09_physics_probe_facing_split_run.py` (`--hessian-xfit`, `--alpha`) | `physics_curvature_probe`, `physics_labels`, `cae` |
| `tab:xenc` / `tab:xencx` (Appendix C, cross-encoder) | `generate/appendix_gen.py` | `09_physics_probe_facing_split_{dinov3_vitb16,clip_base,convnext_base,vit_large}.jsonl` | `09_physics_probe_facing_split_run.py` (`--parquet-path`, `--embedding-column`, `--label-table`); the ViT-B row from the geometry written by `09_physics_probe_facing_run.py`[^chain] | `physics_curvature_probe`, `physics_labels`, `cae` |
| `tab:cf` and the intervention numbers (Appendix D) | `generate/appendix_gen.py` | `09_physics_normal_scaling_{vit_base_d16,vit_base_d20,dinov3_vitb16_d16,clip_base_d16,convnext_base_d16,vit_large_d16}.{jsonl,npz}` | `09_physics_normal_scaling_run.py`; the two ViT-B runs read the geometry written by `09_physics_probe_facing_run.py`[^chain] | `physics_curvature_probe`, `physics_labels`, `cae` |
| Sign tests on thinned anchors (Appendix D) | `generate/appendix_gen.py` (min-degree greedy independent set, computed in-script) | `09_physics_normal_scaling_*_thin.npz` | `09_physics_normal_scaling_thin_run.py` | `physics_curvature_probe` |
| Figure 1 (`fig1_intervention.pdf`, main text) | `latex/figures/make_fig_intervention.py` | same six `09_physics_normal_scaling_*.{jsonl,npz}` runs as `tab:cf` | `09_physics_normal_scaling_run.py` | `physics_curvature_probe`, `physics_labels`, `cae` |
| `figF` panel (a): known-surface partials vs. sampling | `latex/figures/make_fig1.py` | `09_fixture_probe_facing.jsonl`, `09_fixture_probe_decodability.jsonl`, `09_fixture_probe_decodability_gamma2.jsonl` | `09_fixture_probe_facing_run.py` (imports `09_fixture_probe_decodability_run.py`) | `physics_curvature_probe` |
| `figF` panel (b) / `fig1_probe_facing.pdf` (generated, not included in the build): galaxies, decoder | `latex/figures/make_fig1.py` | `09_physics_probe_facing_split.jsonl` | `09_physics_probe_facing_split_run.py` | `physics_curvature_probe`, `physics_labels`, `cae` |
| Validation numbers 0.999 / 1.00 / 0.94 (Section 2 "Validation" paragraph, Appendix E) | hand-written; verified against the record | `09_instrument_adjudication.jsonl` | `09_instrument_adjudication_run.py` | `physics_curvature_probe`, `chart_curvature`, `curvature_probe`, `decoder_curvature` (+ `09_physics_curvature_run.py`, imported as `runner`) |
| Appendix E known-surface numbers (partial range −0.29 to +0.26 as sampling `gamma` varies; sphere/shape term values) | hand-written; verified against the record | `09_fixture_probe_facing_split.jsonl`, `09_fixture_probe_facing.jsonl` | `09_fixture_probe_facing_split_run.py` (imports `09_fixture_probe_facing_run.py`) | `physics_curvature_probe` |
| "86,471" (galaxy count, Introduction and Data paragraph) | hand-written constant | any physics record's `environment` row; row count is a pipeline convention justified by the alignment proof | `09_row_alignment_proof_run.py` (borderline runner, kept as provenance for the row-count/alignment convention; not itself a source of a paper number) | `physics_labels` |

Every runner and module named above exists under `curvature-experiment/` (verify with
`ls curvature-experiment/runners curvature-experiment/pu_manifold`). The Appendix E label in
this table is the manuscript's own section title ("Appendix E: the shape and sphere terms, and
mean curvature"); it is hand-written, not spliced by `appendix_gen.py`.

[^chain]: `09_physics_probe_facing_split_run.py` loads `09_physics_probe_facing_run.py` by file
    path, which loads `09_instrument_adjudication_run.py`, which loads
    `09_physics_curvature_run.py` (the `runner` helpers). All four files are needed for any row
    whose runner is the split runner or `09_physics_normal_scaling_run.py` (which loads the split
    runner the same way).

## Regenerating

```bash
python paper/generate/appendix_gen.py        # splices Appendix A/B/C/D into latex/main.tex between the AUTOGEN markers
python paper/latex/figures/make_fig_intervention.py   # writes figures/fig1_intervention.{pdf,png}
python paper/latex/figures/make_fig1.py               # writes figures/figF_mean_curvature.{pdf,png} and figures/fig1_probe_facing.{pdf,png} (generated but not included in the build)
bash paper/latex/check.sh                              # builds the PDF and checks the page count
pytest paper/tests                                     # appendix_gen.py reproduces main.tex byte-for-byte, given the record cache
```

All three scripts read records through [`records.py`](records.py), which resolves the cache
directory from `EFFDIM_CACHE_DIR` if set, otherwise `../curvature-experiment/.cache/`. None of
them depend on the working directory they are run from. See
[`../curvature-experiment/REPRODUCE.md`](../curvature-experiment/REPRODUCE.md) for how each
record file was produced.
