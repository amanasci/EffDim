# Expanding Physics Probes via External Catalogue Cross-Matching

## The Key: What Identifiers Do We Have?

`Smith42/galaxies` v2.0 exposes **two join handles** per row in `METADATA_COLUMNS`:

| Field | Type | Value example | Notes |
|---|---|---|---|
| `dr8_id` | string | `"8543-4321"` | DESI DR8 unique ID (`brickid-objid`); primary key for Legacy Survey Tractor cross-match |
| `iauname` | string | `"J123456.7+654321"` | IAU coordinate-based name; used in NSA and Galaxy Zoo DESI catalogs |

Both are **retained by the streaming loader** (they appear in `remove_columns` only for the image-extraction pipeline — we can skip that call when loading for labels only).

---

## Catalogue 1 — Galaxy Zoo DESI Advanced Parquet (Walmsley et al. 2023)

### What it provides

The GZ DESI catalog covers **~8.7 million DESI DR8 galaxies** — the same population as `Smith42/galaxies`. It uses a deeper GZ decision tree than the 8 questions currently in `METADATA_COLUMNS`. The `_advanced` variant exposes all vote fractions.

**Additional questions not in `METADATA_COLUMNS` today:**

| Question prefix | Answers | New probes |
|---|---|---|
| `how-rounded` | `round`, `in-between`, `cigar-shaped` | 3 |
| `edge-on-bulge` | `boxy`, `none`, `rounded` | 3 |
| `spiral-winding` | `tight`, `medium`, `loose` | 3 |
| `spiral-arm-count` | `1`, `2`, `3`, `4`, `more-than-4`, `cant-tell` | 6 |
| `odd-feature` | `ring`, `lens-or-arc`, `disturbed`, `irregular`, `other`, `merger`, `dust-lane` | 7 |
| `clumps` | `yes`, `no` | 2 |

**~24 additional vote fractions → ~16 independent (after removing sum-to-1 duplicates)**

### Join key

`dr8_id` maps directly to the GZ DESI catalog's identifier. The GZ DESI parquet also contains `iauname` as a redundant join key.

```
Smith42/galaxies.dr8_id  ←→  gz_desi_advanced.dr8_id   (exact string match)
```

### Data access

```python
import polars as pl

# Download once from Zenodo (DOI: 10.5281/zenodo.7786416)
# ~4.5 GB for _advanced, ~1.5 GB for _friendly
GZ_DESI_URL = "https://zenodo.org/records/8364746/files/gz_desi_deep_learning_catalog_advanced.parquet"

gz = pl.read_parquet("gz_desi_advanced.parquet")
# Select only fraction columns + identifier
gz_cols = ["dr8_id"] + [c for c in gz.columns if c.endswith("_fraction")]
gz = gz.select(gz_cols)
```

### Match rate

Expected: **~95–99%** — same DR8 footprint, same parent catalog.

---

## Catalogue 2 — DESI DR8 Tractor (Legacy Survey)

### What it provides

The Tractor catalog adds **multi-band photometry** not included in `METADATA_COLUMNS`:

| Column | Band | New probe |
|---|---|---|
| `flux_g`, `flux_r`, `flux_z` | Optical grz | `mag_g`, `mag_r`, `mag_z` (already in v2.0) |
| `flux_w1`, `flux_w2` | WISE mid-IR | `mag_W1`, `mag_W2` ← **new** |
| `flux_w3`, `flux_w4` | WISE thermal | `mag_W3`, `mag_W4` ← **new** |
| `flux_ivar_*` | Uncertainties | Signal-to-noise ratios ← **new** |
| `shape_r` | Effective radius | Galaxy size (arcsec) ← **new** |
| `shape_e1`, `shape_e2` | Ellipticity components | Orientation/ellipticity ← **new** |

Derived from Tractor fluxes:
- `g_minus_W1` = optical-IR colour (star-forming vs quiescent separator)
- `r_minus_W2` = another stellar mass proxy
- `W1_minus_W2` = AGN/star formation indicator
- `snr_g`, `snr_r`, `snr_z`, `snr_w1`, `snr_w2` = detection significance

**~10 new photometric + ~5 derived colour probes**

### Join key

`dr8_id` encodes `brickid-objid` directly:

```python
def parse_dr8_id(dr8_id: str) -> tuple[int, int]:
    """Parse 'brickid-objid' string → (brickid, objid) integers."""
    parts = dr8_id.split("-")
    return int(parts[0]), int(parts[1])
```

The Tractor catalog is partitioned by brick. Access options:

**Option A — Legacy Survey viewer API (easy, no download)**
```python
import requests

def query_tractor_by_radec(ra, dec, radius_arcsec=1.0):
    """Query Legacy Survey viewer for Tractor catalog within radius."""
    url = "https://www.legacysurvey.org/viewer/ls-dr10/api/objsearch/"
    params = {"ra": ra, "dec": dec, "radius": radius_arcsec / 3600, "layer": "ls-dr8"}
    r = requests.get(url, params=params)
    return r.json()
```

**Option B — NOIRLab Astro Data Lab SQL (recommended for bulk)**
```python
# pip install dl  (Astro Data Lab client)
from dl import queryClient as qc

# Join on release+brickid+objid
query = """
SELECT ls.flux_w1, ls.flux_w2, ls.flux_w3, ls.flux_w4,
       ls.shape_r, ls.shape_e1, ls.shape_e2,
       ls.flux_ivar_g, ls.flux_ivar_r, ls.flux_ivar_z
FROM ls_dr8.tractor AS ls
WHERE ls.brickid = {brickid} AND ls.objid = {objid}
"""
result = qc.query(sql=query)
```

**Option C — Pre-built cross-match table on Astro Data Lab**

Data Lab hosts a pre-joined `ls_dr8 × galaxy_zoo_desi` table — can pull WISE photometry and GZ morphology together in one query using `dr8_id`.

### Match rate

Expected: **~100%** — the Smith42/galaxies sample is drawn directly from the DR8 Tractor catalog.

---

## Catalogue 3 — NSA / MPA-JHU (Spectroscopic subset only)

### What it provides

For the ~30–40% of DESI DR8 galaxies with SDSS spectroscopic coverage:

| Property | Source | New probe |
|---|---|---|
| Velocity dispersion $\sigma_v$ | MPA-JHU | Dynamical mass proxy, morphology correlate |
| H$\alpha$ EW | MPA-JHU | Star formation activity |
| D4000 break | MPA-JHU | Stellar age indicator |
| [O III]/H$\beta$ (BPT x-axis) | MPA-JHU | AGN vs. star-forming |
| [N II]/H$\alpha$ (BPT y-axis) | MPA-JHU | Gas metallicity |
| Effective radius $R_e$ | NSA | Size (independent of Sersic model) |
| Sérsic index (NSA) | NSA | Morphology (independent measurement) |

### Join key

`iauname` → NSA `IAUNAME` (exact match, or RA/Dec within 1 arcsec as fallback).

### Caveat

Only ~30% coverage → introduces selection bias (SDSS spectroscopic targets are brighter/closer). Treat spectroscopic probes as a **separate probe set** with a flag (`has_spec`), or train probes only on the spectroscopic subsample.

---

## Recommended Implementation Plan

### Phase 0 — Already available (33 v2.0 columns + derived)

Use `METADATA_COLUMNS` + ~11 derived features = **~44 probes** immediately.

### Phase 1 — GZ DESI Advanced Parquet (highest priority, cleanest)

**Effort:** ~1 day

1. Download `gz_desi_advanced.parquet` from Zenodo once (4.5 GB).
2. Write `build_crossmatch_cache.py` in `experiments/physics-probe-subspace/`:
   - Stream `Smith42/galaxies` test split, collect `dr8_id` + row index.
   - Left-join with GZ DESI on `dr8_id` using Polars.
   - Cache result as `crossmatch_gz_desi.parquet` under `$PLATONIC_ROOT/data_hf/crossmatch/`.
3. Update `_common.py` → `load_physics_labels()` to optionally merge GZ DESI columns.
4. New probes: **+~16 independent** → total **~60 probes** → meets 50–100 target.

```python
# Skeleton of build_crossmatch_cache.py
import polars as pl
from datasets import load_dataset

def build_gz_desi_crossmatch(out_path, n_rows=None):
    ds = load_dataset("Smith42/galaxies", revision="v2.0", split="test", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if n_rows and i >= n_rows:
            break
        rows.append({"row_idx": i, "dr8_id": row["dr8_id"]})
    
    galaxies_df = pl.DataFrame(rows)
    gz = pl.read_parquet("gz_desi_advanced.parquet").select(
        ["dr8_id"] + [c for c in ... if c.endswith("_fraction")]
    )
    
    merged = galaxies_df.join(gz, on="dr8_id", how="left")
    merged.write_parquet(out_path)
    print(f"Matched {merged['dr8_id'].is_not_null().sum()} / {len(merged)}")
```

### Phase 2 — DESI Tractor WISE Photometry (medium effort)

**Effort:** ~1–2 days

1. Use Astro Data Lab Python client (`pip install noaodatalab`).
2. Write `fetch_tractor_wise.py` — bulk SQL query batched by `brickid`.
3. Cache `tractor_wise_photometry.parquet` with `dr8_id`, `flux_w1`, `flux_w2`, `shape_r`, `shape_e1`, `shape_e2`.
4. New probes: **+~10** → total **~70 probes**.

### Phase 3 — NSA/MPA-JHU Spectroscopic (optional, low coverage)

**Effort:** ~2–3 days (mostly handling the partial coverage)

1. Download MPA-JHU VAC from SDSS DR8 server.
2. Cross-match on `iauname` or RA/Dec (astropy `SkyCoord.match_to_catalog_sky()`).
3. Train probes only on spectroscopic subset; flag the restriction in results.
4. New probes: **+~7 spectroscopic** → total **~77 probes**.

---

## Summary: Probe Counts by Phase

| Phase | Source | Join key | New probes | Running total | Coverage |
|---|---|---|---|---|---|
| Phase 0 (current) | `METADATA_COLUMNS` + derived | — | ~44 | **~44** | 100% |
| Phase 1 | GZ DESI Advanced (Zenodo) | `dr8_id` | ~16 | **~60** | ~95–99% |
| Phase 2 | DESI DR8 Tractor (Astro Data Lab) | `dr8_id` | ~10 | **~70** | ~100% |
| Phase 3 | MPA-JHU / NSA spectroscopic | `iauname` / RA–Dec | ~7 | **~77** | ~30–40% |

> [!NOTE]
> **Phase 1 alone** (GZ DESI join via `dr8_id`) gets us to ~60 probes — safely within the 50–100 target from the CONTEXT.md. Phases 2–3 are optional enhancements.

---

## Key Technical Considerations

### Preserving row order
The embeddings in the parquets are in fixed row order. The cross-match must produce arrays aligned to that order. Use `row_idx` as the anchor key, not galaxy ID, to ensure alignment even for non-matches (fill with NaN).

### Handling non-matches
All joins should be **left joins** (keep all Smith42/galaxies rows). Non-matched rows get `NaN` → the probe's `_clean_inputs()` filter silently drops them. Probe diagnostics should report `n_valid` per property.

### GZ DESI `proportion_asked` filter
For the advanced catalog, filter each question's fractions by `proportion_asked > 0.5` to ensure we only train probes on galaxies for whom the question was relevant.

### One-time vs. online access
- GZ DESI parquet: download once (~4.5 GB), store at `$PLATONIC_ROOT/data_hf/crossmatch/`.
- Tractor photometry: query Astro Data Lab once, cache result parquet.
- Both are **offline after caching** — the experiment itself has no network dependency.
