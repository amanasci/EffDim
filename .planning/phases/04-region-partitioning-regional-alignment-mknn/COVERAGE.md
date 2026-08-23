# API Coverage — Phase 4 (Region Partitioning & Regional Alignment / MKNN)

No external API integration: the phase reads a locally generated `notebooks/.cache/subsample_*.npz`,
computes with numpy, scipy and scikit-learn (all already pinned in `requirements-notebooks.txt` and
already imported by sealed modules), and writes JSONL/npz into the gitignored `notebooks/.cache/`
through `cache.py`'s `_assert_inside_cache` containment guard — there is no network call, no SDK,
no service and no credential anywhere in its scope.

> The only external *reference* in scope is arXiv:2509.19453, a paper whose published Legacy-vs-HSC
> range (0.34%–2.25%) MKNN-02 compares a locally computed number against. A cited paper is not an
> integrated API surface and has no capability matrix.

*Declared at plan time, 2026-08-23. Accepted by the seal-time `api-coverage.verify-pre` gate in place
of a capability matrix.*
