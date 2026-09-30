# Tensor fidelity: validating the full probe-facing tensors on a known manifold

Date: 2026-09-30
Branch: `tensor-fidelity` (from `encoder-scaling@75d721f`)

## Goal

Answer the reviewer's first concern: the synthetic validation so far
(`runners/09_instrument_adjudication_run.py`) checks only the **trace** of the second fundamental
form (`H_tan`), while the paper's diagnostic uses its full contraction with the probe. This
sub-project checks the full tensors the split runner estimates — the probe-facing tensor
`pf_full = <w_N, II>`, its in-sphere part `pf_tan`, the label Hessian `hess_y` and the mismatch
`hess_y - pf_full` — against exact truth on a generator whose geometry is known.

This is sub-project A of three, agreed in brainstorming on 2026-09-30:
A (this spec) tensor fidelity; B runner robustness (validation-tuned alpha, extra controls,
held-out test, thinned-anchor partials, surrogate fidelity); C scale-out (reduced galaxy sweep and
QM9). A and B are independent; C depends on B.

Success criteria:

1. The runner reports, per configuration, the tensor cosines and relative errors of the four
   tensors and the across-anchor Spearman of the mismatch norm and alignment cosine against truth.
2. Pass lines, fixed in source before any run: on the noise-free small fixture, median `pf_full`
   cosine >= 0.8 and mismatch-norm Spearman >= 0.7. Paper-scale runs are reported against these,
   with no separate line.
3. All new tests pass; the CPU equivalence gate (`docs/superpowers/harness/gate.sh`) still passes.
4. `paper/latex/main.tex` is unchanged.

## Decisions (agreed in brainstorming)

| Topic | Decision |
|---|---|
| Fixture | Both scales: small (d=4, D=64) for recovery curves; the adjudication's paper-scale fixture (d=16, D=768, n=86,471) as a does-it-hold check |
| Device | Small grid on local CPU; paper-scale on the pod GPU with `--device cuda --deterministic` (also validates the GPU path sub-project C relies on) |
| Truth | float64 autodiff of the generator on CPU, at each anchor's own latent, independent of the device under test |
| Comparison | Every d x d tensor lifted to the ambient tangent plane (chart-invariant), then compared there |
| Probe | One fitted ridge probe `w` shared by estimate and truth; only `w_N` is recomputed from the true tangent plane |
| Pass lines | median `pf_full` cosine >= 0.8, mismatch Spearman >= 0.7, noise-free small fixture |

## Fixture

Reused unchanged from `runners/09_instrument_adjudication_run.py`: `InSphereGenerator`
(`G(z) = normalize([stereo(z); a * bumps(z); 0...]) @ Q^T`, mapping into the unit sphere),
`draw_latents` (scale mixture, ~3x variation in k-NN radius), `generate_points`, the
`FIXTURE` (d=16, D=768, n=86,471, a=0.8, k=2048, 512 anchors) and `SMOKE` (d=4, D=64) dictionaries,
and the patch-noise protocol (`X = normalize(G(z) + eps)`, eps isotropic, scaled to a fraction of
the median k-NN patch radius of the noiseless cloud). Labels are functions of the true latent z and
carry no label noise.

## Labels

- The smoke fixture's two: `lin` (`z . a1`) and `nonlin` (`sin(2 z . a1) + (z . a2)^2`).
- A family `y_lambda = <w0, G(z)> + lambda * h(z)`, lambda in {0, 0.5, 1, 2}. `w0` is a seeded
  random unit vector in R^D; `h` is a sum of two seeded Gaussian bumps in z (width 0.8). Both
  terms are scaled to unit standard deviation over the sample, so lambda is a ratio.

At lambda = 0 the covariant Hessian of `y` on the manifold is exactly `<w0_N, II>` (the normal part
of a linear ambient function contracted with II — the paper's identity), so a probe that recovers
`w ~ w0` has true mismatch near zero. Raising lambda adds mismatch in a controlled way. This avoids
hand-building "aligned" labels, which would be circular because the probe `w` depends on the label.

## Truth (per anchor, float64, CPU)

At each anchor's own latent z (known, since we generated it):

- `J`, `Hess`, `II`, `g`, `ginv` of `G` by `ppf.decoder_geometry(G, z)` (the same function the
  runner applies to the fitted decoder, here applied to the exact generator; `G` has `.decode`).
  The adjudication's exactness check carries over: max |H_rad + d| < 1e-8.
- The covariant label Hessian `nabla^2 f = d^2 f - Gamma^k d_k f`, with `f(z)` the label as a
  function of the generator latent and `Gamma` from `g = J^T J`, by autodiff.
- `w_N` from the shared probe `w` and the true `J`; then true `pf_full`, `pf_tan`, mismatch.

## Comparison

The decoder's chart differs from the generator's, so raw d x d matrices are not comparable. Each
(0,2) tensor `T` in a chart with Jacobian `J` is lifted to the ambient tangent plane as
`T_amb = J^+T T J^+` (D x D), which does not depend on the chart. The runner's quadratic fit in
tangent-projected coordinates estimates the covariant Hessian at the anchor, so `hess_y` lifts
the same way.

Metrics per anchor, then medians and curves over the grid:

- tensor cosine `<T_est, T_true>_F / (|T_est|_F |T_true|_F)` for `pf_full`, `pf_tan`, `hess_y`,
  mismatch;
- relative error `|T_est - T_true|_F / |T_true|_F` for the same four;
- across anchors: Spearman rho of estimated vs true mismatch norm and alignment cosine
  (`align_cos_full`).

Tangent-plane misalignment between decoder and generator counts toward the error; that is part of
what the estimator gets wrong.

## Grid

- Small, local CPU: `SMOKE` geometry (d=4, D=64); n in {4000, 16000, 64000}; noise in
  {0, 0.125, 0.25, 0.5} x median patch radius; 3 seeds (latents and decoder init). 36 runs x 6
  labels. k and anchors scale as in `SMOKE` (k=128, 64 anchors) at n=4000; k = n/32 above that,
  anchors 64.
- Paper scale, pod GPU: `FIXTURE`, noise in {0, 0.25}, seed 0. 2 runs x 6 labels.
- Decoder protocol per run: `ppf.fit_decoder` with `pcp.MAX_EPOCHS`, `pcp.AE_HIDDEN`, probe
  `Ridge(alpha=pcp.ALPHA_RIDGE)`, sphere-projected decoder — the paper's protocol unchanged.

## Code

Additive only; no existing runner or `pu_manifold` module changes.

- `curvature-experiment/runners/10_tensor_fidelity_run.py`
  - imports unchanged: the adjudication runner (fixture, generator, noise), the probe-facing
    runner (`fit_decoder`, `decoder_geometry`, `metric_norms`), the split runner
    (`local_quadratics`, `split_columns`);
  - adds: labels, truth (covariant Hessian, true `w_N` tensors), ambient lift, scoring, pass-line
    constants `TENSOR_COS_PASS = 0.8`, `MISMATCH_RHO_PASS = 0.7`;
  - flags: `--mode small|paper`, `--n`, `--noise`, `--seed`, `--threads`, `--device`,
    `--deterministic`, `--max-epochs` (test override only), `--record-path`;
  - writes one JSONL row per (configuration, label) to `EFFDIM_CACHE_DIR/10_tensor_fidelity.jsonl`
    (environment row first, device stated, as the split runner does).
- `curvature-experiment/runners/10_tensor_fidelity_report.py`: reads the records, writes
  `curvature-experiment/results/tensor-fidelity/REPORT.md` (pass lines, tables) and
  `fig_tensor_fidelity.png` (cosine vs noise per tensor, one panel per n).
- `curvature-experiment/tests/test_tensor_fidelity.py`.

## Tests (TDD)

1. Chart invariance: reparametrise the generator's latent `z -> A z` (A invertible); lifted tensors
   agree to 1e-10.
2. Identity: at lambda = 0 the truth's covariant Hessian equals `<w0_N, II>` to 1e-10.
3. Exactness: max |H_rad + d| < 1e-8 at the anchors.
4. Scoring sanity: a tensor scored against itself gives cosine 1, relative error 0; against its
   negative, cosine -1.
5. End-to-end smoke: the runner at n=4000, noise 0, 3 epochs writes rows with finite metrics for
   every label.
6. CPU gate: `gate.sh` passes (no existing code changes).

## Out of scope

Everything in sub-projects B and C; editing `paper/latex/main.tex`; any Physics (real-data)
record.
