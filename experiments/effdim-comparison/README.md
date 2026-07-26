# Effective-dimension comparison experiments

This directory contains the complete experiment harnesses, raw outputs, charts,
Markdown reports, and PDFs used to compare spectral, local-neighbour, and
Landmark-Isomap dimensionality estimates.

## Directory layout

- `scripts/` — benchmark and report-generation programs.
- `results/` — raw JSON/CSV outputs, generated SVG charts, Markdown reports,
  and PDFs.

## Reports

- [`results/effdim-stability-universe.pdf`](results/effdim-stability-universe.pdf)
  — bootstrap, sparse-sampling, and contiguous-region stability on JWST, DESI,
  and Legacy Survey embeddings.
- [`results/effdim-known-dimension.pdf`](results/effdim-known-dimension.pdf)
  — recovery on isotropic linear subspaces with known dimensions.
- [`results/effdim-spectrum-shape.pdf`](results/effdim-spectrum-shape.pdf)
  — effects of eigenvalue decay and ambient noise on PCA-95, Shannon,
  participation ratio, and Rényi dimensions.
- [`results/effdim-manifold-recovery.pdf`](results/effdim-manifold-recovery.pdf)
  — recovery on linear spaces, spheres, tori, a nonlinear chain, and a Swiss
  roll, including CAGRA Landmark Isomap and real UniverseTBD embeddings.

The corresponding `.md` files are editable report sources. All linked SVGs are
stored beside them so the reports remain self-contained after relocation.
`results/effdim-stability-universe-jittered-25.json` preserves the superseded
25-replicate jittered-bootstrap run for provenance; the published stability
report uses the later 100-replicate deduplicated bootstrap.

## Main conclusions

1. PCA-95 is a global variance-retention dimension, not a general intrinsic
   manifold dimension.
2. Spectral effective ranks agree on flat spectra but separate strongly under
   eigenvalue decay and weak noise floors.
3. MiND-MLk has the lowest median error on clean synthetic manifolds, while
   participation ratio is comparatively robust under moderate ambient noise.
4. Landmark Isomap works well on connected manifolds that admit a useful
   geodesic unfolding, but closed topology and disconnected neighbour graphs
   are important failure modes.
5. On the real embeddings, Landmark Isomap is stable for `k=10`, `k=15`, and
   `k=20`:

   | Dataset | k=10 | k=15 | k=20 |
   |:---|---:|---:|---:|
   | JWST | 9.6 ± 0.9 | 9.4 ± 0.5 | 9.0 ± 0.0 |
   | DESI | 6.6 ± 0.9 | 6.2 ± 0.4 | 6.2 ± 0.4 |
   | Legacy Survey | 7.8 ± 0.8 | 8.4 ± 1.1 | 8.2 ± 1.1 |

   These values are close to participation-ratio and higher-order Rényi
   estimates, but the real datasets have no known latent dimension.

## Research questions

The experiments were designed to separate several questions that are often
collapsed into the phrase “effective dimension”:

1. **Repeatability:** does an estimator return a similar value after bootstrap,
   sparse sampling, or partitioning the dataset?
2. **Known-rank accuracy:** does it recover the rank of an isotropic linear
   subspace?
3. **Spectrum sensitivity:** how does eigenvalue decay change PCA-95, Shannon,
   participation-ratio, and Rényi dimensions?
4. **Manifold recovery:** can local estimators recover the latent dimension of
   nonlinear shapes embedded into a much larger ambient space?
5. **Noise robustness:** when ambient noise makes the observed distribution
   full-dimensional, which methods remain informative about the clean latent
   manifold?
6. **Neighbourhood sensitivity:** how much does Landmark Isomap change between
   `k=10`, `k=15`, and `k=20`?

These are distinct statistical targets. Agreement between methods is expected
only in special cases, such as an isotropic linear subspace with a flat
nonzero covariance spectrum.

## Estimator glossary

### Spectral and global estimators

| Estimator | Quantity measured | Important interpretation |
|:---|:---|:---|
| PCA-95 | Number of principal components required to retain 95% variance | Compression/global variance dimension; sensitive to a broad weak noise floor |
| Participation ratio | \(1/\sum_i p_i^2\), where \(p_i\) are normalized eigenvalues | Effective number of strongly occupied covariance directions |
| Shannon effective rank | \(\exp(-\sum_i p_i\log p_i)\) | Entropy-weighted spectral dimension |
| Rényi effective rank | Generalized entropy effective rank for α=2–5 | Increasing α places progressively more weight on dominant directions |
| Geometric-mean ED | Geometric aggregation of normalized eigenvalues | Numerically unsuitable for rank-deficient covariance without explicit null-space handling |

### Local-neighbour and manifold estimators

| Estimator | Basis | Observed limitations in these experiments |
|:---|:---|:---|
| MLE | Local neighbour-distance ratios | Accurate at low/moderate clean dimensions; underestimates high dimensions and degrades under noise |
| Two-NN | Ratio of first and second neighbour distances | Fast, but sensitive to ties, duplicate points, and some nonlinear sampling patterns |
| DANCo | Distance and angle concentration | Good on several low-dimensional clean manifolds; saturates on higher-dimensional linear spaces |
| MiND-MLi | MiND maximum-likelihood variant | Strong downward bias in the current implementation |
| MiND-MLk | k-neighbour MiND estimate | Best median clean-manifold recovery in this suite |
| ESS | Simplified directional-centroid statistic in this codebase | Repeatable but not a calibrated Expected Simplex Skewness dimension estimator |
| TLE | Current implementation follows the same formula as MLE | Results are consequently identical to MLE in these runs |
| Landmark Isomap | Residual-variance elbow after geodesic embedding | Requires a connected graph and a globally useful unfolding; closed topology can produce embedding-span rather than latent dimension |

GMST was excluded from these experiments because its computational cost does
not fit the repeated 10k–100k sample benchmark design.

## Experimental design

### 1. UniverseTBD sampling stability

Three real embedding matrices were evaluated:

| Dataset | Rows | Features |
|:---|---:|---:|
| JWST | 1,496 | 1,024 |
| DESI | 20,465 | 768 |
| Legacy Survey | 100,000 | 1,024 |

Each matrix was tested using:

- The complete dataset.
- 100 bootstrap draws. Duplicate row selections were merged, producing a
  random unique subset containing approximately 63% of rows.
- 25 independent uniform samples containing 10% of rows.
- 12 contiguous stored-order chunks.

All non-GMST methods used streaming covariance and exact cuVS brute-force
`k=10` neighbours. The contiguous chunks test stored-order sensitivity; they
are not physical sky regions because the exported matrices do not include sky
coordinates.

Selected whole-dataset estimates:

| Method | JWST | DESI | Legacy Survey |
|:---|---:|---:|---:|
| PCA-95 | 145.000 | 64.000 | 110.000 |
| Shannon effective rank | 36.134 | 22.157 | 25.981 |
| Participation ratio | 11.752 | 8.659 | 9.241 |
| Rényi α=5 | 6.388 | 5.027 | 5.633 |
| MLE/TLE | 12.560 | 18.172 | 22.560 |
| DANCo | 4.533 | 5.591 | 5.620 |
| MiND-MLk | 11.464 | 16.539 | 20.712 |

The large PCA-95/effective-rank disagreement indicates highly anisotropic
covariance spectra: many weak directions collectively account for nontrivial
variance, while a much smaller set dominates spectral concentration.

### 2. Isotropic known-rank subspaces

Five independent Gaussian datasets were generated at each true dimension
`d ∈ {5, 10, 20, 50, 100}`. Every dataset contained 10,000 points embedded
into a random 256-dimensional linear subspace.

| Method | d=5 | d=10 | d=20 | d=50 | d=100 |
|:---|---:|---:|---:|---:|---:|
| PCA-95 | 5.000 | 10.000 | 19.000 | 48.000 | 94.200 |
| Participation ratio | 4.998 | 9.990 | 19.960 | 49.736 | 99.007 |
| Shannon effective rank | 4.999 | 9.995 | 19.980 | 49.867 | 99.499 |
| Rényi α=5 | 4.994 | 9.976 | 19.902 | 49.351 | 97.600 |
| MLE/TLE | 5.716 | 10.862 | 19.012 | 35.808 | 55.022 |
| Two-NN | 5.045 | 9.967 | 17.932 | 35.263 | 55.199 |
| MiND-MLk | 5.248 | 10.023 | 17.613 | 33.147 | 51.017 |

The spectral methods agree here because the synthetic nonzero eigenvalues are
approximately equal. This is an intentionally favourable case and does not
represent the spectra of the real embeddings.

### 3. Eigenvalue decay and ambient noise

Rank-100 covariance spectra in 256 ambient dimensions used six profiles:

- Flat.
- Power-law decay with exponents 0.5, 1, and 2.
- Exponential decay with scales 10 and 25.

Each profile was evaluated without noise and at 30, 20, and 10 dB SNR. Five
10,000-sample trials were compared with exact population-spectrum values.

Exact noiseless population dimensions:

| Spectrum | Algebraic rank | PCA-95 | Shannon | Participation ratio | Rényi α=5 |
|:---|---:|---:|---:|---:|---:|
| Flat | 100 | 95.000 | 100.000 | 100.000 | 100.000 |
| Power 0.5 | 100 | 91.000 | 84.412 | 66.618 | 35.871 |
| Power 1.0 | 100 | 78.000 | 39.677 | 16.458 | 7.758 |
| Power 2.0 | 100 | 11.000 | 4.808 | 2.470 | 1.848 |
| Exponential 10 | 100 | 30.000 | 27.181 | 20.015 | 14.984 |
| Exponential 25 | 100 | 68.000 | 61.919 | 48.208 | 36.542 |

For the steep power-2 spectrum, adding a 10 dB ambient noise floor changed
PCA-95 from 11 to 116 while participation ratio moved only from 2.47 to 2.98
and Rényi α=5 from 1.85 to 2.08. This reproduces the qualitative real-data
pattern: weak variance spread over many ambient directions strongly affects a
cumulative threshold without substantially changing dominant-direction
effective ranks.

### 4. Nonlinear manifold recovery

Every clean manifold contained 10,000 points and was randomly orthogonally
embedded into 256 dimensions. Five trials were run at no noise and at 30, 20,
and 10 dB SNR.

The manifold suite contained:

- Linear Gaussian subspaces with `d=2, 5, 10, 20`.
- Unit spheres \(S^d\) with intrinsic `d=2, 5, 10, 20`.
- Product tori \(T^d\) with intrinsic `d=2, 5, 10, 20`.
- A one-dimensional open nonlinear chain represented by polynomial and Fourier
  coordinates.
- A two-dimensional Swiss roll.

Median absolute relative error across all 14 shapes:

| Method | No noise | 30 dB | 20 dB | 10 dB |
|:---|---:|---:|---:|---:|
| PCA-95 | 35.0% | 35.0% | 35.0% | 1,890.0% |
| Shannon effective rank | 19.2% | 20.6% | 31.2% | 134.3% |
| Participation ratio | 15.0% | 15.1% | 17.3% | 38.6% |
| Rényi α=5 | 14.8% | 14.9% | 16.2% | 28.7% |
| MLE/TLE | 11.8% | 27.8% | 40.7% | 191.2% |
| Two-NN | 11.6% | 12.2% | 43.9% | 329.7% |
| DANCo | 7.3% | 41.7% | 47.3% | 55.7% |
| MiND-MLk | 4.6% | 19.5% | 29.7% | 172.7% |
| Landmark Isomap | 15.0% | 21.5% | 22.5% | 29.5% |

Spectral metrics recover global linear span rather than nonlinear latent
dimension. A sphere of intrinsic dimension `d` spans approximately `d+1`
linear axes, while the standard cosine/sine embedding of a `d`-torus spans
`2d`. Landmark Isomap shows a similar global-topology limitation: it exactly
recovers the linear spaces and Swiss roll but returns approximately `d+1` for
spheres and `2d` for low/moderate-dimensional tori.

### 5. Landmark Isomap implementation

The scalable Isomap implementation uses:

1. cuVS CAGRA to construct an approximate nearest-neighbour graph over all
   points.
2. Symmetrization of the weighted graph using Euclidean edge lengths.
3. Selection of 512 random landmarks from the largest connected component.
4. Sparse Dijkstra distances from landmarks rather than a full all-pairs
   geodesic matrix.
5. Classical MDS on landmark geodesic distances.
6. An automated dimension choice from the maximum normalized
   residual-variance elbow over candidate dimensions.

The synthetic manifold suite used `k=10` and searched candidate dimensions
1–50. The real embeddings searched dimensions 1–150 and repeated the landmark
selection five times.

The nonlinear chain demonstrates why connectivity must be reported: its clean
`k=10` graph placed only about 46% of points in the largest component, making
the resulting Isomap estimate invalid as a whole-dataset dimension.

On the real embeddings, all graphs were fully connected for `k=10`, `k=15`,
and `k=20`. Median per-landmark-selection runtime, excluding the shared CAGRA
graph build, increased with dataset size:

| Dataset | Rows | k=10 | k=15 | k=20 |
|:---|---:|---:|---:|---:|
| JWST | 1,496 | 0.26 s | 0.28 s | 0.30 s |
| DESI | 20,465 | 2.17 s | 2.55 s | 2.81 s |
| Legacy Survey | 100,000 | 11.50 s | 13.51 s | 15.67 s |

## How to interpret the results

There is no single estimator that is “best” without first defining the target:

- Use **PCA-95** when the question is how many linear components are required
  to preserve 95% of variance.
- Use **participation ratio or Shannon/Rényi effective rank** when the question
  concerns concentration of the covariance spectrum.
- Use **local estimators such as MiND-MLk, MLE, or Two-NN** when the target is a
  clean local manifold dimension and neighbour geometry is reliable.
- Use **Landmark Isomap** when geodesic unfolding is scientifically meaningful,
  the graph is connected, and topology does not prevent a useful Euclidean
  representation.
- Report multiple families on unknown real data. Agreement is informative, but
  disagreement usually reflects different estimands rather than an
  implementation error.

For the UniverseTBD embeddings, a practical summary should report at least
PCA-95, participation ratio, Shannon effective rank, a higher-order Rényi
dimension, and Landmark Isomap with graph-connectivity diagnostics.

## Limitations

- The real embedding datasets do not have known intrinsic dimensions.
- Stored-order chunks are not spatial sky regions.
- Ambient isotropic Gaussian noise is a controlled stress test, not a complete
  model of neural-embedding noise.
- Isomap dimension selection depends on graph neighbourhood, landmarks,
  residual-variance sampling, and elbow definition.
- Five synthetic replicates quantify gross stability but not high-precision
  confidence intervals.
- The Isomap comparison uses CAGRA approximate neighbours, while the core local
  estimators use exact cuVS brute-force neighbours.
- The current ESS implementation is a proxy statistic and should not be
  interpreted as standard calibrated ESS.
- Geometric-mean ED requires better treatment of numerical null eigenvalues
  before it is suitable for rank-deficient embeddings.

## Artifact guide

| Prefix | Raw data | Report sources | Purpose |
|:---|:---|:---|:---|
| `effdim-stability-universe` | JSON, raw CSV, summary CSV | Markdown, PDF, nine SVG histogram grids | Real-data sampling stability |
| `effdim-known-dimension` | JSON, CSV | Markdown, PDF, two SVG charts | Isotropic known-rank recovery |
| `effdim-spectrum-shape` | JSON, CSV | Markdown, PDF, two SVG charts | Eigenvalue decay and noise floors |
| `effdim-manifold-recovery` | JSON, CSV | Markdown, PDF, three SVG charts | Nonlinear manifolds, noise, and real Isomap comparison |
| `effdim-isomap-recovery` | JSON | Incorporated into manifold report | Synthetic Landmark-Isomap trials |
| `effdim-isomap-real*` | JSON | Incorporated into manifold report | Real Isomap `k=10/15/20` sensitivity |

## Scripts

- `bench_effdim_stability.py` — full, bootstrap, sparse, and region tests on
  real matrices.
- `render_effdim_stability_report.py` — stability CSV, SVG, and Markdown
  generation.
- `bench_known_dimension.py` — isotropic known-rank subspaces.
- `render_known_dimension_report.py` — known-rank charts and report.
- `bench_spectrum_shape.py` — analytic spectrum profiles and SNR sweep.
- `render_spectrum_shape_report.py` — spectrum/noise charts and report.
- `bench_manifold_recovery.py` — nonlinear shape and noise benchmark.
- `render_manifold_recovery_report.py` — combined manifold, Isomap, and
  real-data comparison report.
- `bench_isomap_recovery.py` — synthetic CAGRA Landmark-Isomap benchmark.
- `bench_isomap_real.py` — full real-dataset Landmark-Isomap benchmark.

The benchmark environment requires the compiled `effdim._native` extension,
NumPy, SciPy, CuPy, and cuVS. GPU nearest-neighbour work was run on an NVIDIA
RTX 6000 Blackwell. PDF generation additionally requires Pandoc and XeLaTeX.

## Re-running benchmarks

The commands below assume the compiled extension and GPU dependencies are
available in the active Python environment. Dataset paths are illustrative and
should be adjusted to the local cache.

### Stability experiment

```bash
python experiments/effdim-comparison/scripts/bench_effdim_stability.py \
  /path/to/jwst_dinov3_vitl16.npy \
  /path/to/desi_dinov3_small_vitl16.npy \
  /path/to/legacysurvey_dinov3_vitl16.npy \
  --bootstrap-iterations 100 \
  --sparse-iterations 25 \
  --sparse-fraction 0.10 \
  --regions 12 \
  --output experiments/effdim-comparison/results/effdim-stability-universe.json
```

### Known-rank experiment

```bash
python experiments/effdim-comparison/scripts/bench_known_dimension.py \
  --dimensions 5 10 20 50 100 \
  --samples 10000 \
  --ambient-dimension 256 \
  --repeats 5 \
  --output experiments/effdim-comparison/results/effdim-known-dimension.json
```

### Spectrum-shape experiment

```bash
python experiments/effdim-comparison/scripts/bench_spectrum_shape.py \
  --profiles flat power_0.5 power_1.0 power_2.0 exp_10 exp_25 \
  --snr-db none 30 20 10 \
  --rank 100 \
  --ambient-dimension 256 \
  --samples 10000 \
  --repeats 5 \
  --output experiments/effdim-comparison/results/effdim-spectrum-shape.json
```

### Nonlinear manifold experiment

```bash
python experiments/effdim-comparison/scripts/bench_manifold_recovery.py \
  --samples 10000 \
  --ambient-dimension 256 \
  --snr-db none 30 20 10 \
  --repeats 5 \
  --k 10 \
  --output experiments/effdim-comparison/results/effdim-manifold-recovery.json
```

### Synthetic Landmark Isomap

```bash
python experiments/effdim-comparison/scripts/bench_isomap_recovery.py \
  --samples 10000 \
  --ambient-dimension 256 \
  --snr-db none 30 20 10 \
  --repeats 5 \
  --k 10 \
  --landmarks 512 \
  --max-dimension 50 \
  --output experiments/effdim-comparison/results/effdim-isomap-recovery.json
```

### Real Landmark Isomap

Run once for each desired graph neighbourhood (`10`, `15`, and `20` were used
in the published comparison):

```bash
python experiments/effdim-comparison/scripts/bench_isomap_real.py \
  /path/to/jwst_dinov3_vitl16.npy \
  /path/to/desi_dinov3_small_vitl16.npy \
  /path/to/legacysurvey_dinov3_vitl16.npy \
  --k 20 \
  --landmarks 512 \
  --max-dimension 150 \
  --pair-count 100000 \
  --repeats 5 \
  --output experiments/effdim-comparison/results/effdim-isomap-real-k20.json
```

## Regenerating reports

From the repository root:

```bash
python experiments/effdim-comparison/scripts/render_effdim_stability_report.py \
  --results experiments/effdim-comparison/results/effdim-stability-universe.json \
  --output-prefix experiments/effdim-comparison/results/effdim-stability-universe

python experiments/effdim-comparison/scripts/render_known_dimension_report.py \
  --results experiments/effdim-comparison/results/effdim-known-dimension.json \
  --output-prefix experiments/effdim-comparison/results/effdim-known-dimension

python experiments/effdim-comparison/scripts/render_spectrum_shape_report.py \
  --results experiments/effdim-comparison/results/effdim-spectrum-shape.json \
  --output-prefix experiments/effdim-comparison/results/effdim-spectrum-shape

python experiments/effdim-comparison/scripts/render_manifold_recovery_report.py \
  --results experiments/effdim-comparison/results/effdim-manifold-recovery.json \
  --isomap-results experiments/effdim-comparison/results/effdim-isomap-recovery.json \
  --real-stability-results experiments/effdim-comparison/results/effdim-stability-universe.json \
  --real-isomap-results experiments/effdim-comparison/results/effdim-isomap-real.json \
  --real-isomap-k15-results experiments/effdim-comparison/results/effdim-isomap-real-k15.json \
  --real-isomap-k20-results experiments/effdim-comparison/results/effdim-isomap-real-k20.json \
  --output-prefix experiments/effdim-comparison/results/effdim-manifold-recovery
```

To regenerate a PDF:

```bash
pandoc experiments/effdim-comparison/results/effdim-manifold-recovery.md \
  --from gfm \
  --resource-path=experiments/effdim-comparison/results \
  --pdf-engine=xelatex \
  -V geometry:margin=0.5in \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -o experiments/effdim-comparison/results/effdim-manifold-recovery.pdf
```
