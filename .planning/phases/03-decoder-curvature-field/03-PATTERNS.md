# Phase 3: Decoder & Curvature Field - Pattern Map

**Mapped:** 2026-08-13
**Files analyzed:** 13
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `notebooks/pu_manifold/chart_curvature.py` (EDIT: D-08 mode toggle) | service (differentiable math) | transform | itself (existing `chart_mean_curvature`/`_chunked_jacobian`) | exact — extend in place |
| `notebooks/pu_manifold/derivative_bridge.py` (EDIT: WR-01/02/03) | service (verification) | batch/transform | itself (existing `finite_difference_jacobian`/`_hessian`, `calibrate_fd_step`) | exact — fix in place |
| `notebooks/pu_manifold/synthetic_controls.py` (NEW) | utility / fixture-generator | transform | `notebooks/pu_manifold/decoder_curvature.py` (duplicate-and-pin pattern) + `test_curvature_probe.py`'s `_flat_plane_fixture`/`_sample_sphere` | role-match, structural |
| `notebooks/pu_manifold/tests/test_curvature_probe.py` (EDIT: add D-09 forward/reverse equivalence tests) | test | request-response (pure function assertions) | itself — `test_chart_curvature_dxd_solve_matches_explicit_projector` (line 1503) | exact |
| `notebooks/pu_manifold/tests/test_derivative_bridge.py` (EDIT: WR-01/02/03 regressions) | test | request-response | itself (existing derivative_bridge tests) | exact |
| `notebooks/pu_manifold/tests/test_synthetic_controls.py` (NEW) | test | request-response | `notebooks/pu_manifold/tests/test_decoder_curvature.py` (`_SphereDecoder`/`_LinearDecoder` known-answer fixtures) | role-match, structural |
| `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` (NEW) | runner script | batch | `notebooks/diagnostics/template_benchmark_run.py` (resumable/smoke/dry-run skeleton) + `notebooks/diagnostics/curvature_feasibility_sweep_run.py` (curvature-sweep cell structure) | role-match, exact domain |
| `notebooks/diagnostics/curvature_field_pu_run.py` (NEW) | runner script | batch | `notebooks/diagnostics/cae_train_run.py` (single-fit-plus-diagnostics runner) + `template_benchmark_run.py` (resumability) | role-match |
| `notebooks/diagnostics/synthetic_control_run.py` (NEW) | runner script | batch | `notebooks/diagnostics/cae_train_run.py` | role-match |
| `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` (NEW) | notebook (presentation/sanity) | request-response | `notebooks/02.5_swiss_roll_chart_curvature_check.ipynb` (named reference) + `notebooks/02.2_swiss_roll_cae_check.ipynb` (CLAUDE.md's named reference) | exact |
| `notebooks/03_pu_curvature_field.ipynb` (NEW) | notebook (presentation) | request-response | `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` / prior presentation notebooks reading runner JSONL output | role-match |
| `notebooks/03_synthetic_control.ipynb` (NEW) | notebook (presentation) | request-response | same as above | role-match |
| `.planning/REQUIREMENTS.md` (EDIT: re-mint DEC/CURV IDs) | config/spec | — | itself (existing table) | exact |

## Pattern Assignments

### `notebooks/pu_manifold/chart_curvature.py` (service, transform) — D-08/D-09

**Analog:** itself — `chart_mean_curvature` (lines 254–398), `_chunked_jacobian` (240–248).

**Core pattern to copy for the `mode` toggle** (from `chart_mean_curvature`, lines 340–367 — the Jacobian/Hessian construction to be dispatched on `mode`):
```python
# notebooks/pu_manifold/chart_curvature.py:350-367
J = vmap(jacrev(decode_one))(chunk)
if tuple(J.shape) != (VMAP_CHUNK, out_dim, chart_dim):
    raise ValueError(...)  # RESEARCH Pitfall 5 shape guard -- keep for BOTH modes

Hess = vmap(hessian(decode_one))(chunk)
if tuple(Hess.shape) != (VMAP_CHUNK, out_dim, chart_dim, chart_dim):
    raise ValueError(...)  # keep for BOTH modes -- wrong composition still "runs"
```
**Recommended shape (from RESEARCH Pattern 2, itself grounded in this file's own precedent):**
```python
def _jacobian_hessian(decode_one, chunk, mode: str):
    if mode == "reverse":
        J = vmap(jacrev(decode_one))(chunk)
        Hess = vmap(hessian(decode_one))(chunk)          # hessian == jacfwd(jacrev(f))
    elif mode == "forward":
        J = vmap(jacfwd(decode_one))(chunk)
        Hess = vmap(jacfwd(jacfwd(decode_one)))(chunk)    # spike this composition FIRST
    else:
        raise ValueError(f"chart_mean_curvature: unknown mode {mode!r}")
    return J, Hess
```
Everything downstream of `J, Hess` (lines 369–397: the `g`-trace-first, `d×d`-solve, normal-project pattern) is **Pattern 1** and must not be touched — see next block. Only the Jacobian/Hessian construction branches on `mode`; the shape assertions (lines 351, 360) must fire identically for both branches, per D-09's own warning that a wrong composition "still runs."

**Downstream math to preserve byte-for-byte** (lines 369–387 — the trace-first-then-project identity; do not re-derive):
```python
g = torch.einsum("boi,boj->bij", J, J)
eye_d = _batched_eye(VMAP_CHUNK, chart_dim, g.dtype, g.device)
g_inv = torch.linalg.solve(g, eye_d)

raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)
alpha = torch.linalg.solve(g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)).squeeze(-1)
H_parts.append((raw - torch.einsum("boi,bi->bo", J, alpha))[:n_real].detach())
```

**Guard/error-handling pattern to copy** (`_assert_float64`, lines 202–223 — refuse-and-name-the-fix, never silently downgrade):
```python
def _assert_float64(model: Any, z_chart: torch.Tensor) -> None:
    if z_chart.dtype != torch.float64:
        raise ValueError(f"... got z_chart.dtype={z_chart.dtype}. Pass z_chart.double().")
    ...
```

**Do not touch:** `decoder_curvature.py`, `derivative_bridge.py` — D-08 scopes the toggle to `chart_curvature.py` only; `chart_curvature_field` (line 415) also needs the `mode` param threaded through per D-08.

---

### `notebooks/pu_manifold/tests/test_curvature_probe.py` (test) — D-09 equivalence

**Analog:** `test_chart_curvature_dxd_solve_matches_explicit_projector` (lines 1503–1538), extracted verbatim as the exact structural template for the new forward/reverse equivalence test:
```python
# notebooks/pu_manifold/tests/test_curvature_probe.py:1503-1538
def test_chart_curvature_dxd_solve_matches_explicit_projector():
    model = _small_cae("silu", seed=3)
    chart_idx = 1
    z_chart = torch.rand(6, model.chart_dim, dtype=torch.float64)

    decode_one = cc.chart_decoder_map(model, chart_idx)
    J = vmap(jacrev(decode_one))(z_chart)
    Hess = vmap(hessian(decode_one))(z_chart)

    # --- RESEARCH Pattern 4, transcribed verbatim, as the independent reference ---
    g = torch.einsum("boi,boj->bij", J, J)
    g_inv = torch.linalg.inv(g)
    proj = torch.eye(J.shape[1], dtype=J.dtype)[None] - torch.einsum("boi,bij,bpj->bop", J, g_inv, J)
    II = torch.einsum("bop,bpjk->bojk", proj, Hess)
    H_reference = torch.einsum("bij,boij->bo", g_inv, II)

    H_actual = cc.chart_mean_curvature(model, z_chart, chart_idx)["H_vec"]

    assert H_reference.shape == H_actual.shape
    torch.testing.assert_close(H_actual, H_reference, rtol=1e-9, atol=1e-12)
```
**For D-09:** replace the "independent reference" with a call to `chart_mean_curvature(model, z_chart, chart_idx, mode="forward")` (or the module's forward-path helper) and compare it to the `mode="reverse"` result at float64 round-off tolerance (the same `rtol=1e-9, atol=1e-12` band this test already uses). Reuse `_small_cae(...)` (defined earlier in this file) as the fixture-construction helper — do not build a new toy CAE. Keep the existing sphere known-answer test and shape assertions unmodified per D-09's explicit requirement.

**Recommended pre-step (per RESEARCH's "empirical spike" recommendation):** before writing the full equivalence-test suite, run a 2-line smoke test — `vmap(jacfwd(jacfwd(decode_one)))(small_batch)` against a real `cae.ChartAutoEncoder`-shaped decoder — to confirm no "batching rule not implemented" `RuntimeError` fires. Fallback if it does: `jacfwd(jacrev(f))` for the Hessian (cheap forward Jacobian, reverse-composed Hessian).

---

### `notebooks/pu_manifold/synthetic_controls.py` (NEW module)

**Analog for the module-level "duplicate, never edit sealed code" pattern:** `notebooks/pu_manifold/decoder_curvature.py` lines 1–75 (module docstring + convention-agreement guard at import time):
```python
# notebooks/pu_manifold/decoder_curvature.py:55-74
CURVATURE_CONVENTION = "trace"
"""... Declared here (rather than merely imported) so that a drift in either sealed
module's own convention constant breaks this module's import instead of silently
propagating a factor-of-d error ..."""

if CURVATURE_CONVENTION != _CHART_CURVATURE_CONVENTION:
    raise ValueError(
        f"decoder_curvature.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"chart_curvature.CURVATURE_CONVENTION={_CHART_CURVATURE_CONVENTION!r}. Two modules "
        f"computing the same mathematics must never silently diverge on which convention "
        f"they report under."
    )
```
Copy this exact "declare + assert-agree-at-import" idiom in `synthetic_controls.py` against `curvature_probe.CURVATURE_CONVENTION`.

**Analog for the actual fixture geometry (flat plane / sphere):** `notebooks/pu_manifold/tests/test_decoder_curvature.py` — `_LinearDecoder` (lines 96–113, exact-zero-curvature linear map) and `_SphereDecoder` (lines 115–136, inverse-stereographic sphere with `‖H‖ = d/R` known answer):
```python
# notebooks/pu_manifold/tests/test_decoder_curvature.py:115-136
class _SphereDecoder(nn.Module):
    def __init__(self, R: float):
        super().__init__()
        self.R = float(R)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        s = z[:, 0] ** 2 + z[:, 1] ** 2
        denom = 1.0 + s
        x = 2.0 * z[:, 0] / denom
        y = 2.0 * z[:, 1] / denom
        w = (s - 1.0) / denom
        return self.R * torch.stack([x, y, w], dim=1)
```
Generalize to `d=20` per RESEARCH's construction ("sample uniformly on the unit d-sphere in R^{d+1}, zero-pad to R^768, rotate by fixed Q" — analytic `‖H‖ = d/R` exactly, `d`-parametric, no re-derivation needed).

**Analog for computing curvature on the saddle without inventing new math:** call `curvature_probe.graph_mean_curvature(grad, hess)` directly (already tested, trace-convention-pinned) — do not write a new projector/trace formula. `grad = x @ Q`, `hess = Q` broadcast, `Q = diag(s_1..s_20)` mixed-sign per RESEARCH.

**Do not edit:** `curvature_probe.py` is sealed 02.5 — this phase has no edit authorization on it (only `chart_curvature.py` per D-08, `derivative_bridge.py` per D-14).

---

### `notebooks/pu_manifold/tests/test_synthetic_controls.py` (NEW)

**Analog:** `notebooks/pu_manifold/tests/test_decoder_curvature.py`, full file structure (291 lines) — same "known-answer or independent reimplementation, never merely plausible" discipline stated in its own module docstring (lines 1–14):
```python
# notebooks/pu_manifold/tests/test_decoder_curvature.py:9-13
"""Every test here pins a function against an input whose answer is known independently
(a sphere, a flat linear map, a ReLU decoder that must raise, the sealed
curvature_probe.swiss_roll_analytic_H_scaled module) or against an equivalent
reimplementation, never merely "plausible" -- same discipline as test_curvature_probe.py."""
```
Copy the shape of `test_plain_decoder_curvature_flat_linear_decoder_is_exactly_zero` (line 144, exact-zero assertion, `== 0.0` no tolerance) and `test_plain_decoder_curvature_sphere_known_answer` (line 152, `max_dev < 1e-12` machine-precision band since this is exact autodiff of a closed-form map, not a point cloud). For the saddle fixture, RESEARCH explicitly flags it as `[ASSUMED]` with no existing exact analog — add a `test_saddle_fixture_matches_graph_mean_curvature`-shaped finite-difference cross-check, following `curvature_probe.py`'s own precedent of pinning `graph_mean_curvature` against an independent finite-difference computation (do not trust the saddle construction by inspection alone).

---

### `notebooks/pu_manifold/derivative_bridge.py` (EDIT: WR-01/02/03)

**WR-01 location and fix (bound-method-vs-model bug):** `_assert_float64` calls at lines 156, 205, 308 currently pass `decode_batch` (a bound method) where `chart_curvature._assert_float64` expects the *model* — confirmed by reading `_assert_float64`'s own definition (`chart_curvature.py:202`, which calls `getattr(model, "parameters", None)`; a bound method has no `.parameters`, so the per-parameter float64 guard silently no-ops). Fix: pass the actual model object at each of the three call sites (156, 205, 308), not the decode closure.

**WR-03 location:** `calibrate_fd_step` (line 296) computes its autodiff Hessian unchunked — apply the same chunking discipline already used elsewhere in this file (`_chunked_eval`, lines 124–141, and the chunked autodiff block at lines 438–446) rather than inventing a new chunk loop.

**WR-02:** relative-error columns (`max_abs_relative`, line 382) exceed 100% against near-zero references — this is a reporting/interpretation fix (e.g. reporting `max_abs` alongside `max_abs_relative` rather than relying on the relative column alone near zero), not a computation bug; no single code line to copy, but the existing `_chunked_eval`/`finite_difference_hessian` return-dict shape (lines 184–260) is the analog for where the additional column belongs.

**Analog for the chunking pattern to reuse for WR-03** (lines 124–141):
```python
# notebooks/pu_manifold/derivative_bridge.py:124-141 (paraphrased structure)
def _chunked_eval(decode_batch, points):
    """Invoke decode_batch on chunks of at most MAX_FD_ROWS rows, concatenating."""
    ...
```

---

### `notebooks/diagnostics/swiss_roll_curvature_sweep_run.py` (NEW runner)

**Analog for CLI skeleton (argparse `--smoke`/`--dry-run`/`--resume`/`--max-combos`):** `notebooks/diagnostics/template_benchmark_run.py`, `main()` (lines 700–937). Extracted skeleton:
```python
# notebooks/diagnostics/template_benchmark_run.py:701-812 (structure to copy)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke", action="store_true", help="...")
parser.add_argument("--dry-run", action="store_true", help="...")
parser.add_argument("--resume", action="store_true", help="...")
parser.add_argument("--record-path", type=str, default=None, help="...")
parser.add_argument("--max-combos", type=int, default=None, help="...")
args = parser.parse_args()

if args.dry_run:
    # print planned grid, exit without running or writing anything
    return

completed_by_key = load_completed(record_path) if args.resume else {}
for combo_index, (config, ...) in enumerate(combos):
    key = (...)
    if args.resume and key in completed_by_key:
        print(f"  [skip, resumed] {key}")
        continue
    if args.max_combos is not None and n_run_this_invocation >= args.max_combos:
        break
    t_combo_start = time.monotonic()
    ...
    append_record(record_path, record)   # append-only JSONL, resumability index
```

**Resumability/persistence pattern** (`load_completed`/`append_record`, lines 673–692):
```python
def load_completed(record_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    completed = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed[(rec["config_id"], rec["template"])] = rec
    return completed

def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(_to_jsonable(record)) + "\n")
```
Adapt the key to `(n_charts, seed)` for the Swiss-roll sweep. Use `notebooks/.cache/` for `record_path` per the milestone convention (never touched by the mandatory sanity notebook).

**Analog for the domain-specific cell/sweep structure (curvature-specific, same package):** `notebooks/diagnostics/curvature_feasibility_sweep_run.py` — `_cell_from_base`/`_cell_key`/`_register`/`_measure_one`/`_compute_sweep` (lines 377–583) is the closest same-domain precedent for iterating an `n_charts`-like axis and recording per-cell curvature diagnostics; adapt its cell-registration idiom rather than the generic benchmark runner's combination-of-template logic.

**Gate logic to implement (D-01/D-02/D-04):** median `rho_chart` over ≥5 seeds per `n_charts`, full spread and full sweep table printed, floor `0.65` applied to the best config — write this directly in the runner's `main()`/summary step; per D-15, no separate gate-machinery module, verdict JSON, or threshold table file.

---

### `notebooks/diagnostics/curvature_field_pu_run.py` (NEW runner)

**Analog:** `notebooks/diagnostics/cae_train_run.py` — single-fit-plus-diagnostics runner shape (fits one `ChartAutoEncoder`, records `reconstruction_stats`, timing). Reuse its training-and-record pattern for each of the 9 `(n_charts, seed)` PU fits, and reuse `template_benchmark_run.py`'s `load_completed`/`append_record` resumability idiom (same as above) since D-13's budget (~3–5h) demands resumability. Reuse `cae.timing_probe` (already in `cae_train_run.py`'s call graph per RESEARCH) for the pre-sweep timing probe RESEARCH recommends.

**Diagnostics to compute per fit (D-07), calling only existing, unedited modules:**
- `max cond(g)` — from `chart_curvature.chart_curvature_field`'s `metric_condition_number` return.
- argmax chart occupancy — `model.chart_probs(z).argmax(dim=1)` value counts (**not** `cae.chart_survival`, which is known-broken per RESEARCH/CONTEXT — cite this explicitly in the runner's docstring).
- held-out reconstruction — `cae.reconstruction_stats` (existing, unedited).
- PH H0/H1 agreement — `persistence_probe.py`'s agreement function (existing, unedited; H2 explicitly excluded).

---

### `notebooks/diagnostics/synthetic_control_run.py` (NEW runner)

**Analog:** same skeleton as `curvature_field_pu_run.py` (`cae_train_run.py` + `template_benchmark_run.py` resumability), fitting `cae.ChartAutoEncoder` to each of the three `synthetic_controls.py` fixtures (flat/sphere/saddle) at PU's selected `n_charts`/`chart_dim=20`/`D=768`, "same architecture and protocol" per CURV-06.

---

### `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` (NEW, CLAUDE.md-mandated)

**Analog:** `notebooks/02.5_swiss_roll_chart_curvature_check.ipynb` (named reference to copy the shape of) and `notebooks/02.2_swiss_roll_cae_check.ipynb` (CLAUDE.md's canonical reference implementation). **Additive only — neither is rewritten.**

Structure to copy (per CLAUDE.md's own required shape, verbatim from the project instructions):
1. Import model code unchanged from `notebooks/pu_manifold/` (here: `chart_curvature`, `cae`).
2. Generate Swiss roll in-notebook via `sklearn.datasets.make_swiss_roll` (~3,000 points, `noise=0.0`, fixed seed), centred and divided by one global std.
3. `chart_dim = 2`.
4. Train from scratch, <2 min CPU, never touch `notebooks/.cache/`.
5. Side-by-side 3-D and x-z scatter plots, coloured by `t`.
6. Compare against `cae.PlainAutoEncoder` matched baseline.
7. End with 3–4 printed pass/fail lines + one-sentence read-out.

**Distinct from the gate:** per RESEARCH's explicit anti-pattern warning, this notebook is single-seed, `≤15` cells, and must NOT contain the 5-seed × `n_charts` sweep (that lives in `swiss_roll_curvature_sweep_run.py`). No gate machinery per D-15 — this is a sanity check, not the Step-1 gate.

---

### `notebooks/03_pu_curvature_field.ipynb` / `notebooks/03_synthetic_control.ipynb` (NEW, presentation)

**Analog:** general pattern across prior presentation notebooks (`notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` etc.) — read the runner's JSONL record via `load_completed`-shaped loader, render tables/plots, no training or expensive computation inline. No specific code excerpt required beyond the loader pattern already shown above.

---

## Shared Patterns

### Float64 discipline and refuse-don't-degrade guards
**Source:** `chart_curvature._assert_float64` (lines 202–223), reused unchanged by `derivative_bridge.py` and `decoder_curvature.py`.
**Apply to:** every new function in `synthetic_controls.py` that differentiates a decoder — import and call `chart_curvature._assert_float64`, do not write a new float64 guard.

### C2-activation guard
**Source:** `chart_curvature.assert_c2_activation` (lines 131–167); `decoder_curvature.assert_c2_decoder` (lines 80–100) shows the fallback pattern for a model with no `.activation` attribute.
**Apply to:** any new decoder fit (synthetic controls) before differentiating it — raise, never warn.

### Trace-first-then-project identity (Pattern 1, RESEARCH)
**Source:** `chart_curvature.chart_mean_curvature` lines 369–387.
**Apply to:** all curvature code this phase writes or extends (forward-mode toggle, any new `chart_curvature`-style computation). Never rewrite toward the textbook projector-then-trace form — it is a memory-blowing anti-pattern at `D=768` (151 MB projector per 32-row chunk).

### Curvature convention regression guard
**Source:** `decoder_curvature.py` lines 55–74 (declare-and-assert-agree-at-import-time idiom).
**Apply to:** `synthetic_controls.py` — declare its own `CURVATURE_CONVENTION = "trace"` and assert agreement with `chart_curvature.CURVATURE_CONVENTION` and `curvature_probe.CURVATURE_CONVENTION` at import time, exactly as `decoder_curvature.py` does.

### Resumable-runner JSONL record + CLI flags
**Source:** `template_benchmark_run.py` `load_completed`/`append_record`/`main()` argparse block (lines 673–812).
**Apply to:** all three new runner scripts (`swiss_roll_curvature_sweep_run.py`, `curvature_field_pu_run.py`, `synthetic_control_run.py`).

### Duplicate-and-pin-by-test, never edit sealed code
**Source:** `decoder_curvature.py`'s module docstring (lines 1–36) stating it is "a strict simplification of already-reviewed code, not a new derivation," pinned by dedicated equivalence tests rather than trusted by inspection.
**Apply to:** `synthetic_controls.py` relative to `curvature_probe.py` — reuse `graph_mean_curvature` unmodified; write only new fixture-construction code.

## No Analog Found

None — every file in scope has at least a role-match analog within the same `notebooks/pu_manifold/` and `notebooks/diagnostics/` tree. The one genuinely novel piece of math (the saddle fixture in `synthetic_controls.py`) has no existing *exact* analog but does have a clear compositional analog (`curvature_probe.graph_mean_curvature` + the existing rotate-and-pad pattern from `_flat_plane_fixture`/`_sample_sphere`), so it is listed above as role-match rather than in this section.

## Metadata

**Analog search scope:** `notebooks/pu_manifold/`, `notebooks/pu_manifold/tests/`, `notebooks/diagnostics/`, `notebooks/*.ipynb` (Swiss roll checks only). `src/effdim/` and `pyproject.toml` excluded per CLAUDE.md (frozen this milestone).
**Files scanned:** `chart_curvature.py` (787 lines, read in full), `decoder_curvature.py` (partial, pattern-bearing sections), `curvature_probe.py` (referenced, not fully read — sealed, only its public API consumed), `derivative_bridge.py` (partial, WR-0x sections), `test_curvature_probe.py` (1860 lines, targeted section at 1503–1560), `test_decoder_curvature.py` (291 lines, read in full), `template_benchmark_run.py` (937 lines, targeted sections), `curvature_feasibility_sweep_run.py` (function signatures only), `cae_train_run.py` (not opened — referenced by name/role from RESEARCH), notebook filenames enumerated via `ls`.
**Pattern extraction date:** 2026-08-13
