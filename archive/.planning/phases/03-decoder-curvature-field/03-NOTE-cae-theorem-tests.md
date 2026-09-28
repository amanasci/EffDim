# Testing the CAE paper's theorems against PU and the Swiss roll

**Written:** 2026-08-15
**Status:** findings note — no gate, no verdict, nothing sealed
**Prompted by:** the developer's objection that a CAE is proven to beat a plain auto-encoder on
reconstruction, so our measured 3.5x deficit had to be an experimental fault. That objection was
correct and found three defects (`03-08-DEFECTS-01.md`). This note records what happened when we
then went to the paper's theorems directly rather than redesigning the model.

---

## 1. Implementation audit — faithful

Audited `notebooks/pu_manifold/cae.py` against the paper's architecture section, from the
un-mangled source text.

| Paper | Our code | Verdict |
|---|---|---|
| `E: R^m -> R^l` | `InitialEncoder(in_dim, embed_dim)`, no output activation | faithful |
| `E_a: R^l -> Z_a = (0,1)^d` | `ChartEncoder`, **Sigmoid** output | faithful — exactly `(0,1)^d` |
| `D_a: Z_a -> R^l` | `ChartDecoder(chart_dim, embed_dim)` | faithful — to embedding space, not ambient |
| `D: R^l -> R^m` | `EmbeddingDecoder(embed_dim, out_dim)` | faithful |
| `P` on `x`, `z` and/or `z_a` | `ChartPredictor(embed_dim, n_charts)`, softmax | faithful — paper permits `z` |
| `y_a = D.D_a.E_a.E(x)`, output `argmax p_a` | `_decode_from_chart_coords` / `reconstruct` | faithful |
| eq. 3 `min_a e_a - sum_b l_b log p_b`, `l = softmax(-e)` | `chart_loss` | faithful |
| eq. 4 `R_Lip` on the **chart encoders** | `lipschitz_penalty(model.chart_encoders)` | faithful |
| eq. 5 pre-train, FPS seeds | `fps_pretrain_loss` | see deviation below |
| charts die by decoder weight norm < tol | `_chart_decoder_logmass` / `chart_survival` | faithful |
| `l = 2d` (Nash-Kuiper) | `PU_EMBED_DIM = 40 = 2 x 20` | faithful |

**Deviations, both deliberate:**

1. **ReLU -> SiLU.** Required, documented (DEC-02/CAE-06). ReLU's second derivative is identically
   zero, so the second fundamental form would be identically zero and the whole curvature programme
   would return zeros.
2. **eq. 5 third term sign.** The paper writes `+ sum_b delta_ab log(p_b)` = `+log(p_a)` inside a
   loss being *minimised*, which would drive the seed point's own chart probability toward zero. Our
   code uses `-log(p_a)` (standard cross-entropy). We believe the paper has a sign typo and our
   reading is the sensible one, but it is a literal deviation and is recorded as one.

**Not implemented:** eq. 6 `R_cords` (explicitly optional in the paper) and eq. 8 `R_cycle`, the
chart-transition residual. `R_cycle` is a directly relevant diagnostic we lack — atlas quality is
precisely what is in question — and is worth adding.

**Correction to an earlier claim in this phase.** Session notes previously described "Lipschitz
regularisation on the encoders while curvature differentiates the decoders" as a gap in our setup.
Per eq. 4 that IS the paper's design. It is not an implementation defect. It remains a real
limitation *for our purpose*: nothing in the paper's objective constrains decoder derivatives at any
order, because the paper never intended the decoder to be differentiated twice.

## 2. Theorem 2 — the sample-complexity test, and why it does NOT settle anything

`n > beta_1 (log beta_2 + log(1/nu))`, `beta_1 = C (eps/4)^-d (1-(eps/8tau)^2)^-d/2`,
`C = vol(M)/vol(B_1^d)`, and the guarantee is in **sup-norm**: `sup_x ||x - D.E(x)|| <= eps`.

The reach factor is `>= 1` whenever `eps < tau/2`, so dropping it gives a valid and nearly tight
**lower bound** on the required `n`. The reach therefore does not need to be known to run the test in
the direction that matters.

### PU, at `d = 20`

With the measured best cell (`mse_per_dim = 8.899e-05`, RMS `||x-y|| ~ 0.26`) and the maximally
generous `C = 1`: `n > 8.78e24` against our `n = 10,000`. **Short by ~21 orders of magnitude.** Even
at an `eps = 1.0` — an error comparable to the data's own scale — still short by 10 orders.

### Swiss roll, at `d = 2` — the validation that overturns the inference

Measured directly rather than assumed. Analytic surface area of sklearn's roll, rescaled by the
mandated `1/global_std`; `tau` estimated as `min(1/kappa_max, half the inter-sheet gap)`:

```
C = vol(M)/vol(B_1^2) = 9.40        (NOT 1 — the earlier C=1 assumption was ~10x optimistic)
tau (est, scaled)     = 0.394       -> eps must be < tau/2 = 0.197

     n  nc  L>d?  mean err   SUP err (=eps)  eps<tau/2?  n req by Thm 2  satisfied?
  3000   2 False    0.0254      0.2099          False           n/a         n/a
  3000   3  True    0.2145      0.7048          False           n/a         n/a
 12000   2 False    0.0371      0.1647           True      7.23e+04       False
 12000   3  True    0.0391      0.3314          False           n/a         n/a
```

**Theorem 2's precondition fails on the Swiss roll too** — the best config needs `n ~ 72,000` and we
used 12,000 — **yet the CAE demonstrably works there** (`rho_chart = 0.8302`, clears the 0.65 floor
on all five seeds, beats the raw-point baseline).

**Consequence, and it cuts against the earlier analysis in this phase.** Theorem 2 is a *sufficient*
condition. The roll is a direct counterexample to the inference "precondition fails -> method fails".
Failing it by 21 orders on PU therefore does NOT prove the CAE cannot work on PU. An earlier framing
in this session that the theorem "explains" the PU deficit was overclaimed and is retracted here.

What survives is weaker: the shortfall is not a constant factor but grows exponentially in `d`
(6x at `d=2`, 1e21 at `d=20`), so the two regimes are not comparable in kind. That is a reason to
keep investigating, not a conclusion.

## 3. Theorem 2's `L > d` precondition — tested, and it cannot be met by configuration

Theorem 2 asserts the existence of a CAE **with `L > d` charts**. At `d = 20` that requires at least
21. Every PU configuration run before this test used `n_charts` in `{4, 8, 16}` — all below `d`. On
the roll, `n_charts = 2` violates `L > d = 2`, and `nc=2` is the config that won the Step-1 gate.

Tested at `n_charts = 21`, three seeds, PU, full protocol:

```
 nc      seed  ep  occ  mse_per_dim   SUP err  mean err     sec
 21  20260813  40   20   1.2212e-04    0.7494    0.2926    4017
 21  20260814  39   11   1.0549e-04    0.9846    0.2680    4549
 21  20260815  36    9   1.2441e-04    0.7780    0.2951    3806
```

Median `mse_per_dim = 1.221e-04` versus `nc=16`'s `1.203e-04` and `nc=8`'s `1.227e-04` —
**indistinguishable. The chart-count precondition was not the binding constraint.**

**But the precondition is still not met in effect.** Occupancy is `20, 11, 9` of 21 configured. The
theorem's `L` is the chart count of the constructed CAE; our trained model prunes below it, and all
three seeds end with effective `L <= d = 20`. The a-posteriori chart-count mechanism the paper relies
on (weight decay plus eq. 4) drives the atlas below the threshold the theorem requires. Satisfying
`L > d` in effect would require over-specifying much higher AND weakening the pruning pressure — two
requirements in direct tension.

## 4. The sup-norm instrumentation gap

All three theorems are stated in sup-norm. **Every diagnostic in this milestone reports means.**
`reconstruction_stats` gives `mse_per_dim` and `dim_mse_max` — the latter is a max over output
*dimensions*, not over points, so it is not the theorem's quantity either.

First sup-norm measurements taken here:

| setting | mean err | sup err | ratio |
|---|---|---|---|
| roll, n=12000, nc=2 | 0.0371 | 0.1647 | 4.4x |
| PU, nc=21 (median seed) | 0.2926 | 0.7494 | 2.6x |

Nothing can be tested against an `eps` until sup-norm reconstruction is a recorded diagnostic. This
is cheap to add and blocks every remaining theorem test.

## 5. Theorem 3 — never tested

Theorem 3 needs an `eps/2`-dense sample on a geodesic neighbourhood `M_r(p)` and a per-chart sup-norm
error. We record neither per-chart density nor per-chart sup error, only global means. No chart's
local approximation quality has ever been measured against an `eps` in this milestone.

## 6. Theorem 1 — untestable on PU as things stand

Theorem 1 needs `eps < tau(M)`. PU's reach is unknown and hard to estimate. The roll shows `tau` IS
estimable when the geometry is known, which is what the synthetic controls
(`notebooks/pu_manifold/synthetic_controls.py`, plan 03-04) are for — known-curvature manifolds
zero-padded into high ambient dimension, where padding keeps the added directions totally geodesic so
the true answer survives exactly. Those give a non-degenerate reference at `D = 768`, which is the
only construction in the repo that could test Theorem 1 at production dimensionality.

There is one available inference if sup-norm and `tau` are ever both in hand: our plain AE
reconstructs *better* than the CAE, so if it achieves an `eps`-faithful representation with
`eps < tau` through a simply-connected `R^d` latent, Theorem 1 says PU's manifold is topologically
trivial — a real finding about the data, not the model.

## 7. What is and is not established

**Established:**
- The implementation is faithful to the paper, with two deliberate documented deviations.
- Theorem 2's sample-complexity precondition fails on PU by ~21 orders, and on the Swiss roll by ~6x.
- The bound is therefore measurably loose and cannot be used to conclude the CAE must fail on PU.
- `L > d` makes no measurable difference to PU reconstruction, and cannot be satisfied in effect
  because the atlas prunes below `d` regardless of configuration.
- Sup-norm error runs 2.6-4.4x the mean and has never been recorded.

**Not established:**
- Whether the CAE can succeed on PU. Nothing here proves it cannot.
- Whether PU's intrinsic dimension is really ~20. Every calculation above is exponentially sensitive
  to `d`, and `d = 20` comes from TwoNN/local-PCA estimates clustering 18-25, not from a proof.
- Whether the plain AE's advantage survives sup-norm comparison — it has only ever been compared on
  means, and the developer has retired that comparison (`03-NOTE-d12-retirement.md`) on the grounds
  that a direct C0+C2 bar is a better instrument than a relative one.
