# Curvature programme handoff

This folder is the colleague- and agent-facing package of the curvature / probe
research line. It does not replace frozen output trees. Start here, then
[`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) and [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md).

## 1. One-paragraph scientific overview

The programme asks whether local geometry of normalized image-encoder embeddings
is associated with how well photometric labels can be read by linear probes.
Two instruments were used: a **finite-patch quadratic chart (Q)** on \(k\)-neighbours,
and a **pointwise decoder second fundamental form (D)** obtained by differentiating
an autoencoder. Both can be contracted against a probe weight to give
**probe-aligned** curvature, and compared to a local **label Hessian**.
The objects are related but not interchangeable. Several headline associations
are real at a frozen ViT-B / \(d=16\) / \(k=2048\) scale; several attempted
unifications failed their own gates.

## 2. Current defensible conclusion

On frozen ViT-B charts, higher split-half residual **trace** curvature
\(K_H^{\mathrm{cross}}\) is associated with worse global-probe performance and a
larger *relative* benefit from local adaptation, and that pattern survives
geometry resampling. It is rank- and \(k\)-conditioned, not a universal law,
not intrinsic curvature, and not reproduced as a joint global-plus-adaptation
effect on other encoders. Pointwise D-residual does not tell the same story.
Leakage-safe task-aligned Q energy predicts held-out global MSE in aggregate,
with a `mag_r_desi` sign flip; the matching decoder statistic is null. The
label-Hessian mismatch experiment is **`label_hessian_unreliable`**.

## 3. What Q measures

A local quadratic chart \(f(u)=x_0+Ju+\tfrac12 Q(u,u)\) on a finite \(k\)-patch.
After removing manifold-tangential acceleration and the sphere-radial term forced
by unit normalization, \(B^S\) is sphere-normal residual bending.
\(K_H^{\mathrm{cross}}\) is the signed, unclamped, split-half inner product of
mean-curvature vectors. It is trace-only, patch- and \(k\)-dependent, and not
full \(B^S\) energy. See `inspect-curvature-definition KHcross`.

## 4. What D measures

Pointwise jets of a decoder \(F\) (raw or normalized). D-full is the Euclidean
second fundamental form of the raw image. D-residual is \(II^S\) after
differentiating through \(F/\|F\|\). Under the unnormalized trace convention,
\(H^E=H^S-dx\). D is pointwise; Q is a finite-patch fit. See
`inspect-curvature-definition D_residual`.

## 5. Why D and Q can disagree

They estimate different tensors at different spatial scales, with different
trace conventions and different noise biases. On ViT-B, `KHcross` tracks global
error and relative adaptation; `C_H` from D-residual is globally null and
associated with *worse* patch \(R^2\). Controlled \(\rho(C_H,K_H)\) even flips
sign relative to the raw correlation. Do not collapse them into “the curvature.”

## 6. What probe-aligned curvature measures

\(B_w=\langle w_N,II\rangle=\operatorname{Hess}_{\mathcal M}(w^\top x)\).
On the sphere this splits into a shape term and a sphere term whose \(g\)-norm
is \(\sqrt{d}\,|\hat y-b_0|\). The sphere term is an algebraic diagnostic of the
probe prediction, not representation-specific bending. For Q the energy must stay
split-half and signed.

## 7. What the label-Hessian mismatch experiment attempted

It asked whether probe error tracks \(\|\Delta\|_g=\|H_y-B_w\|_g\) and whether
\(H_y\) aligns with \(B_w\). Estimating \(H_y\) at \(d=16\) needs 136 quadratic
coefficients. The frozen reliability gate failed. Decision:
**`label_hessian_unreliable`**. A positive mismatch-size aggregate is not a
confirmed mechanism.

## 8. Confirmed findings

- ViT-B `KHcross` ↔ global MSE / \(R^2\) at \(d=16\), \(k=2048\).
- Same statistic ↔ relative \(\Delta_{\mathrm{adapt}}\); mean \(\Delta_{\mathrm{adapt}}<0\).
- Geometry-resampling stability of those three associations.
- Scale dependence of the original error association.
- Cross-model Q: representation-specific, not a joint law.
- \(K_{\mathrm{dir}}\) ≠ \(K_H\); most residual energy is trace-free.
- Task-aligned Q aggregate on ViT-B; decoder task-aligned null.
- D-residual useful on matched fixtures; globally null vs \(R_G^2\) on ViT-B.
- Cross-model D-residual: patch degradation only.

Exact numbers sit in [`CLAIM_MATRIX.csv`](CLAIM_MATRIX.csv) and the registry.

## 9. Failed or unresolved claims

- Universal curvature–probe law.
- Intrinsic manifold curvature recovered by Q.
- Local probes better on average.
- D-residual reproduces the Q global/adaptation story.
- QLCA as a low-dimensional chart constraint (audit: rank 136).
- Label-Hessian mechanism (`label_hessian_unreliable`).
- Historical leakage-unsafe raw correlations \(+0.347\) / \(+0.328\) as confirmatory.
- AE 600-epoch scale-match tree: incomplete, different protocol.

## 10. Frozen experiment map

See [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md). LPA, QLCA, CMCLA, and FCR
were pulled from the host; `COMPLETE.json` / `decision.json` hashes match the
frozen synthesis manifest.

## 11. Code map

| Piece | Path |
|---|---|
| Reusable package | `experiments/geometry/curvature/` |
| Handoff docs | `experiments/curvature_program/` |
| Historical runners | `experiments/geometry/physics_*/`, `run_*.py` |
| Tests | `tests/test_curvature_handoff.py` |
| Example configs | `configs/curvature_handoff/` |
| CLI | `PYTHONPATH=experiments python -m geometry.curvature <command>` |
| Scoped agent rules | `experiments/geometry/AGENTS.md` |

Historical scripts were **not** rewritten to import the new package.

## 12. Data and checkpoint requirements

Not in the source-only bundle:

- encoder embeddings and \(k\)NN graphs
- frozen Q charts (`physics_multimodel_graph_prior_quadratic` on host)
- photometric catalogue columns
- AE checkpoints (`reproduction_plain_ae_400`)
- full `outputs/` trees

Set `PLATONIC_ROOT`, `PLATONIC_OUTPUTS`, `PLATONIC_EMBEDDINGS`,
`PLATONIC_CHECKPOINTS`. Do not hard-code `$HOME`.

Still missing locally (not pulled; large data sources): encoder embeddings,
neighbour graphs, catalogue tables, AE checkpoints, and
`physics_multimodel_graph_prior_quadratic` (~276 MB data source, no COMPLETE).

## 13. Environment setup

Prefer the repository’s existing `pyproject.toml` / pip extras.

```bash
python -m pip install -e ".[dev,curvature]"
# decoder autodiff only, if needed later:
python -m pip install -e ".[curvature-torch]"
export PYTHONPATH="$PLATONIC_ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"
```

| Item | Typical historical value |
|---|---|
| Python | 3.12.3 (host) |
| NumPy / SciPy / scikit-learn | NumPy 2.3.2 recorded on host |
| PyTorch | 2.8.0+cu128 on the Ubuntu RTX 6000 host |
| CUDA | 12.8 (cu128 wheel) |
| pandas | required for joins / tables |
| Hardware | CPU enough for unit tests and smoke; full Q/D jobs used the host GPU |
| Disk | source bundle is small; full geometry trees are tens of GB |
| Historical runtimes | minutes (fixtures) to ~13 h (CMCLA) |

Optional torch is isolated in the `curvature-torch` extra. CLI `--help` must
not import torch.

## 14. Verification commands

```bash
export PYTHONPATH=experiments
python -m geometry.curvature verify-curvature-artifacts --help
python -m geometry.curvature reproduce-curvature-headlines --help
python -m geometry.curvature inspect-curvature-definition --help
python -m geometry.curvature curvature-handoff-smoke --help
python -m geometry.curvature inspect-curvature-definition KHcross
python -m geometry.curvature curvature-handoff-smoke
python -m pytest tests/test_curvature_handoff.py -q
```

These read frozen tables or run analytic identities. They do not fit geometry.

## 15. How to add a new experiment safely

1. Read this README and `AGENT_CONTEXT.md`.
2. Check the registry: do not repeat a completed tree.
3. Create `experiments/geometry/<new_name>/` and `outputs/geometry/<new_name>/`.
4. Keep `sample_id` joins, leakage-safe \(w\), train-only \(H_y\), signed cross estimates.
5. Name the estimator explicitly.
6. Write `METHODS.md` and machine-readable tables as you go.
7. Write `decision.json` from pre-registered gates, then `COMPLETE.json` last.
8. Do not edit old output trees or manuscripts.

## 16. Recommended next experiment (proposed, not run)

**Low-variance residual-direction test.** Instead of 136 \(H_y\) entries, test
the probe-induced quadratic direction directly. For a held-out-safe global probe
set \(q_w(u)=\tfrac12 u^\top B_w u\). On training-labelled neighbours fit

\[
r_L(u)=a_0+a_1^\top u, \qquad
r_B(u)=a_0+a_1^\top u+c\,q_w(u)
\]

to residuals \(r=y-\hat y\). On untouched neighbours

\[
\Delta_B=\mathrm{MSE}(r_L)-\mathrm{MSE}(r_B).
\]

This asks whether probe residuals contain structure along the probe-induced
curvature direction. It is **proposed**, not completed.

## 17. Claims that must not appear in a paper

- Q or D measures intrinsic manifold curvature.
- There is a universal curvature–performance law.
- Local probes outperform the global probe on average.
- D-residual replicated the ViT-B Q story.
- All four photometric labels behave the same.
- Cross-model joint replication of global penalty plus adaptation.
- A causal effect of curvature on decoding.
- The Hessian-mismatch P1 number as a confirmed mechanism.

## 18. Contact / handoff checklist

- [ ] Colleague can import `geometry.curvature` with `PYTHONPATH=experiments`
- [ ] `curvature-handoff-smoke` exits 0
- [ ] Registry labels match local/host `decision.json`
- [ ] Host trees (LPA, QLCA, CMCLA, FCR) located or marked missing
- [ ] No manuscript or `outputs/` file was edited in this packaging pass
- [ ] Next science, if any, is the 1-D residual-direction test — not another 136-parameter \(H_y\) sweep
