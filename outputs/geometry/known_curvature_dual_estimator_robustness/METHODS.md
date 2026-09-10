# METHODS

Bounded known-answer validation of D (Austin decoder autodiff) and Q (frozen local quadratic).
d=16, D=28, n≤5000, 64 hash-stable clean anchors, k=1024, ≤12 AEs, 60-minute wall.

Fixtures F0/F1/F2/F4 from the known-curvature audit generators, padded/rotated into R^{28}.
Sampling: Haar uniform (S0 n=5000, S1 n=1500); non-uniform w=exp(β s(u)) with β=log(10)/2 (S2/S3).
Noise scaled to s_x = median ||x-mean x||. Anchors and curvature truth stay clean.

D trained with the reproduction protocol (PlainAutoEncoder 250³ SiLU, AdamW, 400 epochs, seed 0/20260816).
Q is the unmodified production path. T2/T3 labels follow this brief (matched vs uniform), not the prior audit names.
