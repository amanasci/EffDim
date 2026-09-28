No external API integration: Phase 7 is an offline, single-user statistics pipeline over
already-cached local `.npz` arrays, composing in-repo modules (`mknn.py`, `cae.py`,
`decoder_curvature.py`, `curvature_probe.py`, `cross_split_curvature.py`) and already-installed
libraries (`torch`, `scipy` 1.18.0, `scikit-learn`, `numpy`). There is no network I/O, no SDK, no
service, and no package-manager install in this phase — `07-RESEARCH.md` § Package Legitimacy
Audit records the same finding independently.
