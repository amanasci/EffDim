# Cross-model pointwise sphere-residual decoder curvature

Bounded replication of the fixture-validated ViT-B D-residual analysis
(`C_H = ||H^S||`) on DINOv3, CLIP, ConvNeXt-B and ViT-L.

ViT-B is the frozen reference and is excluded from the H1–H3 aggregate.
Optional Q resampling is resource-contingent and must not displace the
primary D-residual run.

Hard limits: 90 minutes, 12 new decoder fits, seeds `{0,1,2}`, `d=16`,
512 shared anchors, no probe refits, no manuscript edits.
