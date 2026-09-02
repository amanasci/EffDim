# Methods

Model `vit_base`, k=2048, d=16, frozen nested charts.
Global decoding: five-fold ridge OOF predictions already stored per target.
Local R² / MSE: geographic score of those fixed predictions in the k=2048 neighbourhood.
Quadratic models: same nested-CV L / UQ / BS path as QLCA (`_process_anchor`).
Controls: log kNN radius (neighbourhood geometry), local label variance (this target), evaluation count.
Inference: rank-space Freedman–Lane permutations and bootstrap medians.
