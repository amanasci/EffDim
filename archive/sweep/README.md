# Cross-Model Validity-Gate Sweep

Self-contained. Copy this directory anywhere, install the requirements, run one command.
Nothing outside this folder is needed.

## What you're running, and why

We found that Isomap's final step — flattening geodesic distances into Euclidean coordinates
via classical MDS — is **invalid** for DINOv3 ViT-B/16 embeddings of Legacy Survey galaxies.
About half the eigenvalues of the double-centred geodesic matrix are negative, carrying ~41%
of total absolute eigenvalue mass. A distance matrix that does this cannot be embedded in flat
space at any dimension.

We ruled out the obvious explanations: numerical error, implementation bugs, a single
short-circuit graph edge, kNN hop inflation (tested across a 6× range of `k`), L2
normalization, absence of manifold structure, the specific survey column, and the specific
10,000 galaxies drawn (a ~90% disjoint resample reproduces the number to 0.03%).

**One variable was never varied: the model.** This sweep varies it, across all 35 vision
foundation models in `UniverseTBD/pu-embeddings` (the embedding set from *The Platonic
Universe*, arXiv:2509.19453).

The question: **is Isomap unsuitable for all of them, or is DINOv3 special?**

## Run it

```bash
pip install -r requirements.txt
python run_sweep.py --workers 8
```

That's it. It is resumable — re-run the same command after any interruption and completed
models are skipped.

To sanity-check your environment first (one model, ~2 min, ~0.6 GB download):

```bash
python run_sweep.py --only dinov3_vitb16
```

### Picking `--workers`

Each model needs **~3.5 GB RAM** and one core-group at peak. The script pins BLAS threads per
worker (`--threads`, default 4) so parallel workers don't oversubscribe.

```
--workers  <=  min(cores // 4,  RAM_GB // 4)
```

On a 64-core / 256 GB box, `--workers 16 --threads 4` is comfortable and finishes in roughly
10 minutes. Serial (`--workers 1`) takes about 90 minutes.

## Resources

- **~20 GB download.** Only the `legacysurvey` column of each parquet is read; the paired
  `hsc` column is skipped, halving the 39.9 GB total. Files stream from HuggingFace and are
  not kept.
- **~2 minutes CPU per model** (~70 min total serial): one Isomap fit plus one dense
  10,000×10,000 float64 eigendecomposition.
- **No GPU.** The bottleneck is a dense symmetric eigensolve. GPU would not help meaningfully
  and float32 would corrupt the measurement — `m` sums ~5,000 small negative eigenvalues, and
  single precision raises the noise floor by ~9 orders of magnitude.

## What to send back

```
model_sweep_results.jsonl
```

A few hundred KB. One JSON object per model: the gate statistics, ambient dimension, intrinsic
dimension estimates, timings, and graph connectivity. **No embeddings, no galaxy data** — only
measurements.

The script also prints the full table and the verdict at the end. A copy of that stdout is
useful but not required.

## Please don't change these

The thresholds, `k=15`, `n_components=18`, and the 10,000 row indices are **pre-registered** —
fixed and committed before any model was run, so that a 35-way comparison can't be narrated
after the fact. They are the reason this result will be believable.

- `row_indices_20260729.npy` pins the exact galaxies every model is compared on. The script
  refuses to start if its hash doesn't match.
- The script runs a **control check first**: it re-measures `dinov3_vitb16` and halts unless it
  reproduces the published `r=0.052419`, `m=0.412071` to 1e-5. If that fails, your environment
  differs from the reference run and nothing else would be comparable.
- Every model is run and reported regardless of outcome. Failures are recorded as failures with
  their traceback, never silently dropped.

If something genuinely needs to change, tell us rather than editing — an unrecorded change to a
pre-registered constant is the one thing that would sink this.

## Reading the output

```
r = |lambda_min_neg| / lambda_max_pos     is there one dominant negative outlier?
m = sum|lambda_neg| / sum|lambda|         how much total mass is negative?

PASS      r < 0.10  AND  m < 0.05
MARGINAL  r < 0.25  AND  m < 0.15
FAIL      otherwise
```

Strict less-than at every boundary; the verdict is the worse of the two.

DINOv3 ViT-B/16 gives `r = 0.052419` (passes) and `m = 0.412071` (fails by ~3×). That shape
matters: `r` passing while `m` fails means the negativity is a long diffuse tail, not one bad
edge — which is why reporting `r` alone would have missed it entirely.

**Either outcome is a real result.** If every model FAILs, the finding is stronger and more
general than the single-model version — it says something about deep vision embeddings of
galaxies, not about DINOv3. If some model passes, that's a candidate encoder for a
flat-embedding pipeline and equally worth knowing.
