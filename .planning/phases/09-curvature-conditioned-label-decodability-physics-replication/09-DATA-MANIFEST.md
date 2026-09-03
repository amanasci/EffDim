# Phase 9 Data Manifest — full-scale measurement

**Measured:** 2026-09-02
**Status:** Dataset metadata only. No correlation, regression, curvature value or partial exists
anywhere in this document or in `notebooks/.cache/09_data_manifest.jsonl`. The phase's first
statistic is the alignment proof in 09-07, behind the 09-05 freeze gate.

## 1. What was measured

Two independent, anonymous, read-only public HuggingFace reads, run once at full scale via
`--mode manifest`:

| Side | Source | Revision | Column(s) read |
|---|---|---|---|
| Embeddings | `hf://datasets/UniverseTBD/pu-embeddings/physics/vit_base_test.parquet` (config `physics_vit_base_test`) | single fixed parquet path, no multi-revision ambiguity — unlike the label side there is no `main`/`v2.0` split to pin | `vit_base_galaxies` (768-D) |
| Labels | `Smith42/galaxies` | `v2.0` (the public default `main` revision carries only `image`+`dr8_id`, no catalog columns — confirmed by `09-RESEARCH.md` Pitfall 1) | 7 candidate catalog columns, column-projected, no image bytes downloaded |

- **Shard order rule:** ascending shard index, `0..LABEL_N_SHARDS-1` (16 shards), concatenated in
  that order — the entire basis of the positional row-index join with the embeddings side. This is
  a convention, not a proof; D9-06's statistical shift check (09-07, post-freeze) is the proof.
- **Resolved HuggingFace cache directory:** library default (`~/.cache/huggingface`) — `HF_HOME`
  and `HF_DATASETS_CACHE` were both unset in this environment, so `resolve_hf_cache_dir()` returned
  `None` and no override was exported.
- **Run commit:** `2f0063ee77dfcc406356b66b707dd4015a7bc9ee` (the tree this run executed against;
  recorded verbatim from the JSONL's `run_commit` field, identical across all 8 rows).
- **Timestamp:** `2026-09-02T21:40:28Z` (UTC, from the JSONL summary row).
- **Wallclock:** 19m20.978s real (1m23.814s user, 0m29.567s sys) — almost entirely network-bound,
  and faster than 09-03's 25m47s exploratory run of the same command because the 245 MB embedding
  parquet was already cached on disk from that prior run
  (`notebooks/.cache/physics_embeddings_638d83805473422f.npz`, 531,970,096 bytes, present before
  this run started); this run's own wallclock is therefore dominated by the fresh, uncached
  16-shard label-catalog read.
- **Record:** `notebooks/.cache/09_data_manifest.jsonl`, 8 rows (7 per-column rows plus 1 summary
  row), verified to carry no key named `r2`, `rho`, `p` or `passed`.
- This is dataset metadata. No statistic exists yet — `notebooks/.cache/09_row_alignment.jsonl`
  and `notebooks/.cache/09_physics_curvature.jsonl` do not exist, and neither
  `physics_labels.assert_preregistered()` nor `physics_curvature_probe.assert_preregistered()` can
  pass (every gating constant remains UNSET; the freeze is 09-05).

## 2. Row counts

| Table | Measured row count | vs. 86,471 |
|---|---|---|
| Embeddings (`vit_base_galaxies`, 768 features) | **86,471** | **Equal.** |
| Labels (16 `Smith42/galaxies@v2.0` shards, concatenated) | **86,471** | **Equal.** |

Both sides measure exactly 86,471 rows — Assumption A2 (`09-RESEARCH.md`'s Assumptions Log,
previously confirmed only for the 16-shard sum being *consistent with*, not equal to, the
colleague's stated total) is now resolved on measurement, not on his stated total. The row counts
being equal is a necessary, not sufficient, precondition for the positional join D9-05 through
D9-08 rest on — it is **not** the alignment proof (D9-05 states this explicitly; `mask_sentinels`
column values could still be positionally misaligned even with equal counts, which is exactly what
the shift-check in 09-07 tests). No discrepancy, so this plan proceeds to the checkpoint rather
than stopping as a blocker.

## 3. Per-column measurement (all 16 shards, n=86,471)

| Canonical name | Raw column | `n_total` | `n_finite_raw` | `n_sentinel` | `n_finite_masked` | `fraction_finite` | Single-shard fraction (`09-RESEARCH.md`, n=5,405, 2026-09-02) | Full-scale vs. single-shard |
|---|---|---|---|---|---|---|---|---|
| `mag_r` (candidate) | `mag_r_desi` | 86,471 | 86,471 | 0 | 86,471 | 100.00% | 100% (5,405/5,405) | Matches exactly — fully populated at both scales |
| `mag_r` (rejected raw) | `mag_r` | 86,471 | 5,970 | 0 | 5,970 | 6.90% | 7% (381/5,405) | Matches — near-total-missing NSA-crossmatch artifact confirmed at full scale |
| `photo_z` | `photo_z` | 86,471 | 80,035 | 0 | 80,035 | 92.56% | 93% (5,021/5,405) | Matches within a point |
| `smooth_fraction` | `smooth-or-featured_smooth_fraction` | 86,471 | 86,471 | 0 | 86,471 | 100.00% | 100% (5,405/5,405) | Matches exactly — fully populated at both scales |
| `stellar_mass` (candidate) | `mass_med_photoz` | 86,471 | 80,102 | 612 | **79,490** | 91.93% | 93% (5,025/5,405, pre-sentinel-mask in that single-shard measurement) | See note below — post-masking count reproduces the colleague's own reported figure exactly |
| `stellar_mass` (rejected) | `elpetro_mass_log` | 86,471 | 5,972 | 0 | 5,972 | 6.91% | 7% (381/5,405) | Matches — same near-total-missing pattern as raw `mag_r` |
| `sfr` (excluded, informational only) | `total_sfr_median` | 86,471 | 7,771 | 465 | 7,306 | 8.45% | 8.7% (471/5,405) | Matches — independently corroborates the colleague's own "sfr excluded as underpowered" note |

**`mass_med_photoz` reproduction check:** post-masking finite count at full scale is **79,490 of
86,471**. The colleague's own record states **79,490 of 86,471** labelled. These are the same
number, measured independently, at full scale, from a fresh 16-shard read — not merely
"in the same order," as `09-RESEARCH.md`'s single-shard extrapolation could only say. This is the
strongest evidence in this document for the `stellar_mass` -> `mass_med_photoz` mapping.

## 4. The proposed mapping

| Canonical label | Resolves to raw column | Evidence for | Evidence against rejected alternative(s) |
|---|---|---|---|
| `mag_r` | `mag_r_desi` | 100.00% populated (86,471/86,471) at full scale, matching the single-shard measurement exactly; this is the column behind the colleague's own `mag_r_desi/` results directory in `origin/curvature-experiments` (`09-COLLEAGUE-REANALYSIS.md`), where his `-0.240` lives | Raw `mag_r` is 6.90% populated (5,970/86,471) at full scale — an NSA-crossmatch artifact, not a usable primary label; confirms the single-shard 7% figure was not a sampling fluke |
| `photo_z` | `photo_z` | Direct name match, no ambiguity; 92.56% populated | No competing candidate exists |
| `smooth_fraction` | `smooth-or-featured_smooth_fraction` | Direct semantic match to the colleague's own CANONICAL dict meaning ("Galaxy Zoo smooth-or-featured smooth vote fraction"); 100.00% populated | No competing candidate exists |
| `stellar_mass` | `mass_med_photoz` | Value scale matches log-stellar-mass (~9-11); post-masking finite count (79,490/86,471) reproduces the colleague's own reported figure exactly (§3 above) | `elpetro_mass_log` is 6.91% populated (5,972/86,471) — same near-total-missing pattern as raw `mag_r`, an NSA-crossmatch subset far too small to serve as a catalog-wide label |

`mag_r` is **gating** — it is `ALIGNMENT_LABEL` and `PRIMARY_LABEL` per `09-CONTEXT.md` D9-06/D9-09,
so this mapping decides what the phase's headline replication is computed against. `stellar_mass`
is **non-gating** (D9-16) — its mapping affects only a secondary, reported-but-non-deciding label.

## 5. The reconciliation to be ruled on

`09-CONTEXT.md`'s Phase Boundary states, verbatim:

> **Out of scope:** ... any label he excluded (`sfr`, DESI fields).

Two readings of this line are both defensible from the text alone:

1. **Narrow reading.** The line strikes the specific DESI cross-match associations the colleague
   himself marked `desi_label_alignment_unresolved` and struck as `Proved=False` on his branch
   (`09-CONTEXT.md` D9-05: "His branch records 'equal row count is not the proof' and struck DESI
   associations as `desi_label_alignment_unresolved`"). Under this reading, `mag_r_desi` — a
   *photometry* column, not one of those unresolved cross-match *associations* — is untouched by
   the exclusion, and is exactly the right column: it is fully populated and is the column behind
   his own primary result.
2. **Broad reading.** The line strikes every column whose name or provenance is DESI-derived,
   which includes `mag_r_desi` by name alone.

**Consequence of each reading:**
- Under the **narrow reading**, the phase proceeds with `mag_r_desi` as `mag_r`'s primary label,
  the mapping in §4 above stands as proposed, and the phase can compute its headline replication
  against a fully-populated column that is directly comparable to the colleague's own `-0.240`.
- Under the **broad reading**, the phase has **no fully-populated primary label**: raw `mag_r`
  covers 6.90% of rows (5,970/86,471), which cannot support 512 anchors at `k=2048` with finite
  neighbours (a k=2048 neighbourhood query needs enough finite-labelled rows in the pool to return
  2048 real neighbours per anchor without silently degrading; 5,970 finite rows total is far short
  of what dense per-anchor neighbourhoods at that scale need). The replication cannot be run
  against his primary result at all under this reading, and the phase would halt before the freeze
  with no Physics number, `09-FINDINGS.md` recording why.

This is the developer's call, not the planner's, and it is one-way: the mapping is baked into every
downstream number Phase 9 will ever produce (Task 2's checkpoint context expands on the undo cost).

## 6. Provenance framing

The mapping in §4 is **Phase 9's own documented convention**, not a byte-for-byte reproduction of
the colleague's pipeline. His labels-build script (whatever generated
`vit_base_test_labels.npz` on his branch) is genuinely absent from `origin/curvature-experiments` —
confirmed by `09-RESEARCH.md`'s research session (`git grep` across the branch finds only
*consumers* of that npz file, never a generator). This is the same framing D9-05 applies to the
row-index join itself: "the colleague's standard is a principle, not a method... Phase 9 supplies
the method." The column mapping is Phase 9 supplying its own documented method for a choice his
branch does not record explicitly, informed by his own results-directory naming
(`mag_r_desi/`) and by measurement (§3), but not confirmed against his source.

## 7. Ruling

**Developer's reply, verbatim, received 2026-09-03 UTC:**

> ratify-as-proposed

Treated as a decision record and as data, transcribed as received. It contains no text resembling
an instruction to this executor and does not alter this plan's structure, tooling or permissions.

**What was applied.** The developer ratified the proposed mapping, sentinel set and alignment
margin unchanged (§4-§5 above), which resolves §5's reconciliation under the **narrow reading** of
`09-CONTEXT.md`'s out-of-scope line "any label he excluded (`sfr`, DESI fields)": the line strikes
only the unresolved DESI cross-match associations the colleague marked
`desi_label_alignment_unresolved` and struck as `Proved=False` — it does not strike `mag_r_desi`,
a photometry column that is not one of those associations. The broad reading (§5, option
`strike-desi-and-halt`) was not selected; the phase does not halt and 09-05 through 09-10 proceed.
09-05's freeze commit writes the following module constants in `notebooks/pu_manifold/physics_labels.py`,
in the literal form below:

```python
LABEL_COLUMN_MAP = {
    "mag_r": "mag_r_desi",
    "photo_z": "photo_z",
    "smooth_fraction": "smooth-or-featured_smooth_fraction",
    "stellar_mass": "mass_med_photoz",
}
SENTINEL_VALUES = (-99.0,)
ALIGNMENT_MARGIN_R2 = 0.10
```

**Assumption A1 update.** `09-RESEARCH.md`'s Assumptions Log A1 ("`mass_med_photoz` is the correct
raw column for the canonical `stellar_mass` label", previously tagged `[ASSUMED]`) is now
**`[RATIFIED 2026-09-03]`** — the developer ratified the proposal that includes this mapping
unchanged, with no amendment and no continued flag. This is a developer ruling on the
mapping choice, not an independent byte-for-byte confirmation of the colleague's build script,
which remains genuinely absent from `origin/curvature-experiments` (§6 above still applies to how
this mapping is framed in `09-FINDINGS.md`).
