# AE field averaged on the frozen k=2048 patch

Instrument: Phase 9 `PlainAutoEncoder` + sphere-projected `H_tan` (Amendment 01).
Cloud: frozen ViT-B 16,384-row subset, same 512 anchors and k=2048 neighbours as `K_H`.
Not a replay of the 86,471-row Phase 9 fit.

Holdout variance explained: 0.9471. Median `H_rad`=-16.0000 (want −16).

## Do the instruments agree after scale matching?

Controlled Spearman (frozen 3 controls):

| | vs `K_H` | vs `K_H` | radius only | vs local `R_G^2` |
|---|---:|---:|---:|---:|
| pointwise `‖H_tan‖` | 0.202 | 0.189 | -0.452 | -0.019 |
| patch-mean `‖H_tan‖` | 0.134 | 0.130 | -0.523 | 0.004 |
| split-cross `⟨H̄_A, H̄_B⟩` | -0.049 | -0.047 |  | 0.082 |
| frozen `K_H` | 1 |  | 0.765 | -0.240 |

Pointwise vs patch-mean `‖H_tan‖`: raw ρ = 0.610.
Runtime 4.3 min.
