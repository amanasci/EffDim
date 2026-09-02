No external API integration: Phase 9 is an offline, single-user statistics pipeline that reads
two public HuggingFace dataset repositories as static, column-projected parquet
(`UniverseTBD/pu-embeddings` config `physics_vit_base_test` and `Smith42/galaxies` at revision
`v2.0`), then composes in-repo modules (`physics_labels.py`, `physics_curvature_probe.py`,
already-sealed `cae.py`/`decoder_curvature.py`/`cross_split_curvature.py`) over already-installed
libraries (`datasets`, `huggingface_hub`, `torch`, `scikit-learn`, `scipy`, `pyarrow`, `pandas`).
There is no request/response surface, no authentication, no write path back to either repository,
and no versioned API contract to enumerate capabilities against — `09-RESEARCH.md`'s
Architectural Responsibility Map states directly that this milestone has no browser, server, or
CDN tier: everything is a single-process, CPU-bound analysis reading from and writing to
`notebooks/.cache/`.

Both HuggingFace reads are anonymous, read-only, column-projected parquet fetches over an
already-vendored loader (`datasets`/`pyarrow`'s `hf://` path), not calls against a request/
response API with its own versioning or auth surface — there is nothing here for a capability
matrix to enumerate. The revision pin `Smith42/galaxies@v2.0` is a data-provenance constant frozen
in `physics_labels.py` (the labels do not exist on the repo's default `main` revision, only on the
`v2.0` branch), not an API version being negotiated at call time; it identifies which snapshot of
a static dataset this phase reads, exactly as a `git` ref pins a commit, and carries none of the
compatibility or deprecation semantics an API version number would.
