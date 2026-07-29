# Stack Research

**Domain:** notebook-only manifold-curvature analysis (HF dataset streaming -> Isomap -> torch decoder -> `torch.func` curvature -> MKNN)
**Researched:** 2026-07-29
**Confidence:** HIGH — every version below was read directly from the PyPI JSON API (`pypi.org/pypi/<pkg>/json`) or from official docs (`scikit-learn.org`, `huggingface.co/docs`, `download.pytorch.org`) on the research date, and the `Isomap`/`KernelPCA` claims were cross-checked against the scikit-learn `main` branch source (`sklearn/manifold/_isomap.py`).

This file covers **only new, notebook-only additions**. `numpy`, `scipy`, `scikit-learn` (1.9.0, confirmed current), and `faiss-cpu` (1.14.3, confirmed current) are already core `effdim` dependencies and are reused, not re-added. Nothing here goes in `pyproject.toml` — every package below installs from inside the notebook via `%pip install`.

## Recommended Stack

### Core Technologies (new, notebook-only)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `datasets` | 5.0.1 | Pull `UniverseTBD/pu-embeddings`, config `legacysurvey_dinov3_vitb16` | Official HF loader. Its `configs:` YAML in the dataset's `README.md` maps each of the 163 config names to one explicit parquet path (confirmed via `GET /api/datasets/UniverseTBD/pu-embeddings`), so `load_dataset(..., name="legacysurvey_dinov3_vitb16")` reads that small YAML and then fetches **only** `legacysurvey/legacysurvey_dinov3_vitb16.parquet` — verified by HTTP HEAD at 580,362,951 bytes (~553 MiB). The other 162 configs are never touched, streaming or not. |
| `torch` | 2.13.0 (CPU build) | MLP decoder + `torch.func` curvature (jacrev/jacfwd/hessian/vmap) | Released 2026-07-08 (3 weeks before this research date) — current stable. CPU-only wheels are published under `download.pytorch.org/whl/cpu` for cp310–cp312 (confirmed present for `torch-2.13.0+cpu`), so it installs and runs with zero GPU/CUDA requirement, satisfying the "no GPU or deep-learning stack in core" constraint since it lives only in the notebook. |
| `matplotlib` | 3.11.1 | Curvature-field scatter plots over 2-D/3-D Isomap coordinates | Already the de-facto standard; ships `mpl_toolkits.mplot3d` in the same package (no separate install) for the 3-D Isomap-coordinate case, and its sequential/diverging colormaps (`viridis`, `coolwarm`) are exactly what a scalar `|H|` field needs. No project currently depends on it, so it is a genuinely new addition, but it is the smallest, most standard choice available. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `huggingface_hub` | 1.25.1 (pulled in transitively by `datasets>=5.0,<6`, which pins `huggingface-hub>=0.25,<2.0`) | Low-level single-file download | Only needed as an **explicit** notebook install if you bypass `datasets` and call `hf_hub_download(repo_id="UniverseTBD/pu-embeddings", repo_type="dataset", filename="legacysurvey/legacysurvey_dinov3_vitb16.parquet")` directly (see Alternatives below). Otherwise `datasets` already brings a compatible version in. |
| `hf_xet` | 1.5.2 (auto-installed by `huggingface_hub>=0.32.0`, so implicitly by 1.25.1) | Fast chunk-deduplicated transfer for this specific file | Nothing to do — it installs itself. Relevant only because this dataset repo is Xet-backed (its `resolve/main/...` URL redirects through `us.aws.cdn.hf.co/xet-bridge-us/...`), not classic Git-LFS, so the download path differs from older HF datasets you may have used before. |

## Installation

```python
# Cell 1 — CPU-only torch (the plain `pip install torch` wheel on Linux bundles
# several GB of nvidia-cublas/cudnn/nccl runtime deps even on a CPU-only machine;
# the dedicated CPU index avoids that entirely)
%pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Cell 2 — HF dataset loader + plotting
%pip install -q "datasets==5.0.1" "matplotlib==3.11.1"
```

Two separate `%pip install` invocations are intentional: mixing `--index-url` (which *replaces* the default index) with packages that only exist on PyPI in the same command risks pip resolving `torch` from the wrong index. This two-cell pattern matches PyTorch's own documented CPU-install instructions.

`torch.func` needs no separate install or import shim:

```python
from torch.func import jacrev, jacfwd, hessian, vmap  # stable since torch 2.0, no functorch needed
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Dataset fetch | `datasets.load_dataset(..., name="legacysurvey_dinov3_vitb16", split="train")` (non-streaming) | `streaming=True` | Per-config `data_files` routing already isolates the download to one ~553 MiB file *regardless* of the streaming flag — streaming buys no reduction in bytes transferred here. Streaming's `IterableDataset.shuffle(buffer_size=...)` is a **windowed reservoir shuffle**, not a uniform sample, unless `buffer_size` approaches the full 101,725-row config (at which point you've paid the download cost anyway). A plain `load_dataset(...).shuffle(seed=...).select(range(10000))` gives a true uniform 10k sample with one line and no buffer-size tuning. |
| Dataset fetch | `datasets.load_dataset` | `hf_hub_download(...)` + `pyarrow.parquet.read_table(...).to_pandas().sample(n=10000)` | Equally correct and arguably more transparent (one explicit file path, no Arrow-cache side effects) — a legitimate lower-level alternative. Not recommended as the primary path only because `datasets` already handles the `Sequence(Value("float32"))` embedding columns and gives `.with_format("numpy")` for free, avoiding hand-rolled list-column unpacking. Use `hf_hub_download` if you want to avoid the `datasets` dependency's Arrow re-caching of the parquet file to disk. |
| k-NN for MKNN | `faiss.IndexFlatL2` (faiss-cpu, already a core dep) | `sklearn.neighbors.NearestNeighbors` | Both are fast enough in isolation at n=10,000, d=768 (sub-second to a few seconds for exact brute-force k-NN). faiss wins for this pipeline specifically because MKNN's bootstrap-CI step re-runs k-NN over resampled point sets many times (hundreds to low-thousands of bootstrap iterations); faiss's BLAS-backed matrix distance computation stays consistently fast across repeated calls, and reusing faiss avoids a second k-NN code path when `effdim.geometry` already depends on it internally. |
| Plotting | `matplotlib` (+ built-in `mpl_toolkits.mplot3d`) | `plotly` / `pyvista` | This is a small number of static research-figure scatter plots (2-D/3-D points colored by `|H|`) for a notebook, not an interactive dashboard. `matplotlib` already ships 3-D scatter support; adding a browser-based plotting stack for this scope is unjustified weight. If interactive rotation of the 3-D Isomap embedding is genuinely wanted later, `%pip install ipympl` + `%matplotlib widget` is a one-line addition to the *existing* matplotlib stack rather than a new plotting library. |
| MDS eigenspectrum audit | `scipy.linalg.eigh` / `scipy.sparse.linalg.eigsh` on the manually double-centered `-0.5 * J @ (isomap.dist_matrix_**2) @ J` | `Isomap.kernel_pca_.eigenvalues_` | Confirmed by reading `sklearn/manifold/_isomap.py`: `Isomap.fit` constructs its internal `KernelPCA(n_components=self.n_components, ...)`, so `kernel_pca_.eigenvalues_`/`eigenvectors_` are truncated to just the requested `n_components` (2–10) and are the **top only** — the negative eigenvalue tail that flags manifold non-Euclidean-ness is never retained. `scipy` is already a core dependency, so no new package is needed; use `scipy.sparse.linalg.eigsh(B, k=..., which="LA")` for the top few and `which="SA"` for the most negative few, avoiding a full O(n³) dense `eigh` on a 10,000×10,000 matrix unless you specifically want the entire spectrum. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| `umap-learn` | The described pipeline (steps 1–7) uses Isomap exclusively for manifold reconstruction; UMAP is not called anywhere in the plan even though it was flagged as a candidate notebook dep in the project's `Constraints` section. Adding it now would be speculative. | `sklearn.manifold.Isomap` (already the plan) — add UMAP only if a future milestone explicitly compares embeddings. |
| `functorch` (standalone package) | Deprecated since torch 2.0; `functorch.jacrev`/`jacfwd` etc. were slated for deletion in torch >=2.3 and torch is now at 2.13. The PyPI `functorch` package (latest 2.0.0) is a compatibility shim only. | `torch.func.jacrev` / `jacfwd` / `hessian` / `vmap` — already part of `torch`, no separate install. |
| `torchvision`, `torchaudio` | Nothing in the pipeline loads raw images or audio — the dataset ships pre-computed 768-d embedding vectors. These pull in extra CV/audio codec dependencies for zero benefit here. | Plain `torch` tensors only. |
| `pytorch-lightning`, `accelerate`, `deepspeed` | Massive overkill for training a small MLP decoder on a 10,000×768 CPU array. A manual `for epoch in range(...)` loop with `torch.optim.Adam` is suffient and keeps the notebook's dependency surface minimal, matching the "no deep-learning stack" spirit of the constraint even though `torch` itself is unavoidable. | Hand-written training loop. |
| `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER=1` | Deprecated — HF's docs confirm all repos (including this one) now transfer through the `hf_xet` binary, and `hf_transfer` support has been removed from `huggingface_hub` for Xet-backed repos. Setting this env var is now a no-op at best. | Nothing — `hf_xet` (auto-installed by `huggingface_hub`) already handles fast transfer for this dataset. |
| `gudhi`, `ripser`, `giotto-tda` (persistent-homology / TDA libraries) | Tempting given the "manifold curvature" framing, but the pipeline computes mean curvature **analytically** from the decoder's Jacobian/Hessian via `torch.func`, not via topological data analysis. TDA libraries solve a different problem (Betti numbers / persistence diagrams) than a differential-geometric curvature field. | `torch.func.jacrev`/`hessian` on the decoder, as planned. |
| `megaman`, `pydiffmap` (large-scale/approximate manifold-learning libraries) | Built for landmark/Nyström-approximated geodesic computation at n >> 10,000. At n=10,000 the dense, exact `sklearn.manifold.Isomap` already fits in ~1 GB and completes in low minutes (see Version Compatibility below) — no approximation is needed at this scale, and the project's own decision log explicitly chose 10k to keep Isomap exact. | `sklearn.manifold.Isomap` (already the plan). |
| `pandas` as a hard requirement | Not load-bearing anywhere in the plan — `datasets.Dataset.with_format("numpy")` and plain `numpy` arrays cover every step (Isomap input, decoder training data, MKNN neighbor sets). | `numpy` (already core). Fine to use `pandas` opportunistically for quick inspection cells if convenient, but don't design the pipeline around it. |
| `plotly`, `pyvista`, `mayavi` | Interactive/GL-based 3-D plotting stacks; unnecessary weight for static research figures. | `matplotlib` + `mpl_toolkits.mplot3d` (see Alternatives). |

## Stack Patterns by Variant

**If running in a constrained/offline environment where a 553 MiB download is a problem:**
- Use `hf_hub_download` with `local_files_only=False` once, then `local_files_only=True` on reruns to avoid re-fetching — same effect as `datasets`' own local caching, but explicit.
- Because this dataset repo is Xet-backed, there is no benefit to `streaming=True` for saving bandwidth on repeat runs; the file downloads once and is cached under `~/.cache/huggingface/` either way.

**If the notebook environment's Python is older than 3.10:**
- `torch==2.13.0` publishes wheels only for cp310–cp312; `datasets==5.0.1` requires Python `>=3.10.0` (confirmed via PyPI metadata). Both **hard-block** on Python 3.8/3.9, even though `effdim`'s own `pyproject.toml` declares `requires-python = ">=3.8"` for the core package.
- Practical implication: this notebook needs a Python 3.10+ (3.11+ recommended — see below) kernel. If the dev environment is still on 3.8/3.9, create a separate notebook-only virtualenv/kernel rather than trying to pin older `torch`/`datasets` releases, since older `datasets` releases lack the native per-config parquet routing behavior relied on above.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.13.0` (CPU wheel) | Python 3.10, 3.11, 3.12 | No cp313/cp38/cp39 CPU wheels observed on `download.pytorch.org/whl/cpu` at research time. |
| `datasets==5.0.1` | Python `>=3.10.0`; `huggingface-hub>=0.25,<2.0`; `pyarrow>=21.0.0`; `numpy>=1.17` | `numpy>=1.17` is far below whatever `effdim` already pins, so no conflict with core numpy. |
| `scikit-learn` (already core, currently 1.9.0 on PyPI) | Python `>=3.11` | Worth noting even though out of scope for this milestone's *new* deps: scikit-learn's own Python floor has already drifted to 3.11, and `faiss-cpu` (currently 1.14.3) now requires `>=3.10`. Combined with `torch`/`datasets` above, **Python 3.11 is the practical floor for this notebook**, one full minor version above the package's declared `>=3.8` — a pre-existing drift in the core deps' own version floors, not something this milestone introduces, but it now determines the notebook's minimum interpreter. |
| `matplotlib==3.11.1` | Standard support matrix; no unusual constraint interacting with the rest of this stack | — |

**`Isomap.fit` at n=10,000, d=768 — concrete memory/time profile:**
- Input array: 10,000 × 768 × 4 bytes (float32) ≈ 31 MB — negligible.
- `dist_matrix_` (dense n×n geodesic distances, float64): 10,000² × 8 bytes ≈ 763 MiB. This is the dominant **persisted** attribute on the fitted `Isomap` object (confirmed by scikit-learn source: it is a plain `(n_samples, n_samples)` dense array), matching the ~1 GB pickled-size figure already established for this milestone.
- **Peak transient memory during `.fit()`** is higher than the persisted 763 MiB: `KernelPCA` internally builds a second `(n,n)` centered-kernel array (~763 MiB) from `dist_matrix_`, plus dense-eigensolver workspace. Budget roughly **2–3 GB of RAM** for the `.fit()` call itself, settling back down to ~1 GB once only `dist_matrix_` + the small `(n, n_components)` embedding remain.
- **Wall-clock time** is dominated by the shortest-path (geodesic) stage, not the eigendecomposition: scikit-learn's documented complexity is `O[N²(k + log N)]` for Dijkstra vs `O[N³]` for Floyd–Warshall, and `path_method="auto"` will choose Dijkstra at N=10,000. The eigendecomposition stage is only `O[d·N²]` for a small `d` (2–10 requested components) and, with `eigen_solver="auto"`, will typically resolve to the iterative ARPACK path rather than a full dense solve, which is comparatively fast for a handful of eigenvectors. No published third-party benchmark for this exact n/d combination was found — treat "a few minutes, not hours, on a modern multi-core machine" as an estimate derived from the documented complexity bounds, not a measured number, and pass `n_jobs=-1` to `Isomap(...)` to parallelize the k-NN graph construction stage.

## Sources

- `pypi.org/pypi/<package>/json` for `torch`, `datasets`, `scikit-learn`, `faiss-cpu`, `huggingface_hub`, `hf_xet`, `matplotlib`, `pyarrow`, `functorch` — official PyPI registry, queried directly 2026-07-29. HIGH confidence (primary source, current release + `requires_python` read directly from registry metadata).
- `download.pytorch.org/whl/cpu/torch/` — directory listing confirming `torch-2.13.0+cpu` wheels exist for cp310–cp312/Linux/Win/aarch64/s390x. HIGH confidence.
- `scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html` and `.../sklearn.decomposition.KernelPCA.html` — official docs, fetched 2026-07-29 (scikit-learn 1.9.0). HIGH confidence.
- `raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/manifold/_isomap.py` — read directly to confirm `KernelPCA(n_components=self.n_components, ...)` truncation behavior. HIGH confidence (primary source code).
- `huggingface.co/docs/datasets/en/package_reference/loading_methods` — official `datasets` docs (v4.8.4 doc build, behavior consistent with current 5.0.1), confirms `name=`/`streaming=` semantics and parquet row-group streaming behavior. HIGH confidence.
- `huggingface.co/api/datasets/UniverseTBD/pu-embeddings` and `.../parquet` — HF Hub API, queried directly 2026-07-29, confirms the explicit per-config `data_files` YAML mapping and the exact parquet path for `legacysurvey_dinov3_vitb16`. HIGH confidence (primary source, live repo metadata).
- HTTP `HEAD` against `huggingface.co/datasets/UniverseTBD/pu-embeddings/resolve/main/legacysurvey/legacysurvey_dinov3_vitb16.parquet` (followed redirect) — confirms 580,362,951 bytes (~553 MiB) and the Xet-backed (`us.aws.cdn.hf.co/xet-bridge-us/...`) storage path. HIGH confidence (direct measurement).
- `huggingface.co/docs/hub/en/xet/using-xet-storage` and related HF docs (via web search) — confirms `hf_xet` auto-install behavior since `huggingface_hub>=0.32.0` and deprecation of `hf_transfer`/`HF_HUB_ENABLE_HF_TRANSFER`. MEDIUM-HIGH confidence (web-search-surfaced but corroborated by official doc titles/URLs).
- `docs.pytorch.org` (`torch.func` tutorial and `functorch.jacrev`/`jacfwd` reference pages) plus the GitHub `pytorch/pytorch` issue tracker — confirms `functorch.jacrev`/`jacfwd` deprecated since torch 2.0 and slated for deletion in torch >=2.3, superseded by `torch.func`. MEDIUM confidence (web-search synthesis of official doc pages; not independently re-fetched page-by-page, but consistent across multiple independent official-domain hits).

---
*Stack research for: EffDim v1.1 "PU Manifold Curvature" notebook*
*Researched: 2026-07-29*
