# METHODS

Bounded real-data test of fixture-validated **pointwise sphere-residual decoder curvature**
(`D-residual`) against frozen ViT-B global and patch probe outcomes.

## Estimand

Raw decoder `F: R^{16} → R^{768}`. Differentiate through `F̃ = F/||F||`.

`H^S = (1/d) g^{ab} B^S_{ab}` with `B^S = P_{N,S} D²F̃` and `P_{N,S} = I - xx^T - P_T`.
Primary scalar `C_H = ||H^S||`. This is **not** intrinsic curvature. Intrinsic scalar
departure is `ΔScal_D = ||tr_g B^S||² - ||B^S||_g²` on a hash-selected 128-anchor subset.

Historical full curvature `H^E = (1/d) tr_g (I-P_T)D²F` on raw `decode` is a **control only**.

The local quadratic statistic `K_H^cross` is an empirical finite-patch quantity, not geometric
ground truth.

## Decoder

`PlainAutoEncoder` hidden (250,250,250) SiLU, AdamW lr=1e-3, weight decay 1e-4, batch 128,
400 epochs, no early stopping. Seeds `{0,1,2}` change initialization and minibatch order.
The 512 evaluation anchors are excluded from training; neighbouring objects may remain.
Label-blind: no labels, probe risks, or curvature correlations enter training or checkpointing.

## Inference

Frozen outcomes: local OOF `R_G^2`, `R_P^2`, `Δ_adapt = MSE_G - MSE_P` on the same valid objects.
Controls: log kNN radius, local label variance, evaluation count.
Rank-space Freedman–Lane, B_perm=10000, B_boot=2000, Holm over P1–P3.
P3 permutes the curvature residual once per replicate (joint in `R_G^2`,`R_P^2`).
