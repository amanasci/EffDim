"""Phase 9 SUPPLEMENTARY runner: the colleague's nested-chart split-half curvature estimator
``K_H^cross`` dropped into OUR Phase 9 pipeline with nothing else changed.

STATUS. This is a post-hoc supplementary experiment. It is NOT pre-registered, it does NOT feed
the frozen Phase 9 verdict (`--mode verdict` of `09_physics_curvature_run.py` never reads the
record this script writes), and it swaps the curvature INSTRUMENT only. Every other component --
the sealed data loaders, the 512 Wave A anchors, the k=2048 neighbourhood panel, the frozen
out-of-fold ridge probe and local R2, the three controls, the rank-partial Spearman, the
Freedman-Lane and density-stratified nulls, the paired anchor bootstrap and every frozen constant
-- is the production pipeline's own, reached by importing the sealed primitives from
``pu_manifold.physics_curvature_probe`` / ``pu_manifold.physics_labels`` and the production
runner's own helpers from ``09_physics_curvature_run.py``. Nothing is re-implemented here.

THE QUESTION. His frozen headline is controlled rho(K_H^cross, local OOF R2 of the ``mag_r``
probe) = -0.240 (raw -0.412) at chart rank d=16, k=2048, 512 anchors; +0.143 at d=12; -0.233 at
d=20. Our plain-autoencoder decoder-curvature instrument gave +0.347 (raw +0.425) at d=16 in the
frozen Phase 9 record. Does HIS instrument reproduce HIS sign inside OUR pipeline?

HIS CODE IS IMPORTED UNCHANGED from a read-only checkout at ``COLLEAGUE_COMMIT`` (``--colleague-
root``, its ``experiments/`` directory placed on ``sys.path``): ``nested_pca_frame`` and
``_fit_rank`` from ``geometry.physics_activation_atlas.nested_dimension_curvature``, ``RIDGES``
from ``full_curvature_audit``, and ``_rows_from_fits`` (the nanmean-over-splits aggregation) from
``geometry.physics_adaptive_dataset_curvature_probe.curvature_stage``. The call pattern is
``fit_kh_panel``'s exactly: ``Xloc = X[neigh[ai, :k]].astype(np.float64)``;
``x0, J, ev, _ = nested_pca_frame(Xloc, max(d_values), device)``;
``fits = _fit_rank(Xloc, x0, J, d, k, n_splits=3, seed=0, ai)``; ``K_H_cross`` and ``R_H`` are
the nanmean over the splits that fit. ``ai`` is the anchor's 0-based position in the anchor
array, as in his loop.

DEPENDENCY SHIM (reported, not hidden). His branch is not self-contained: it does not ship
the sibling package ``topology.physics_activation_density_ph`` that four of his modules import
at module level -- ``paths.py`` (``load_col``, ``platonic_root``, ``resolve_path``), ``data.py``
(``PreparedActivations``, ``effective_rank_from_cov``, ``l2_normalize``, ``prepare_activations``,
``summarize_population``), ``coordinates.py`` (``effective_rank_from_cov``) and ``charts.py``
(``density_stratified_landmarks``, ``farthest_point_landmarks``). Those ten names are stubbed in
``notebooks/diagnostics/colleague_shims/topology/physics_activation_density_ph/`` and are NEVER
called on the path we use: every stub raises ``NotImplementedError`` if called, and the
``nested_pca_frame`` + ``_fit_rank`` path runs to completion against them (``--mode smoke``
proves this on every run). The shim directory is put on ``sys.path`` ahead of his
``experiments/`` root; his checkout is never written to.

NEIGHBOURHOOD CONVENTION (measured, see ``colleague_neighbourhoods``). His neighbourhoods are
built by ``knn_torch_ip`` (top-k inner product on row-L2-normalised embeddings -- the same
ordering as Euclidean distance on the unit sphere, which is what our sealed ``knn_panel``'s
``NearestNeighbors`` uses on the same row-L2-normalised ``X``), then the anchor itself is
REMOVED (``row = row[row != a]``) so his stored ``neigh[ai, :k]`` holds k=2048 neighbours that
EXCLUDE the anchor. Our sealed ``knn_panel`` queries ``X_all[anchor_idx]`` against ``X_all`` and
therefore returns the anchor as its own first neighbour at distance 0, so our 2048 columns
INCLUDE the anchor. The statistics side (local R2, controls) keeps the sealed panel untouched.
For the rows handed to HIS estimator this script queries the sealed ``knn_panel`` once more at
k+1, drops the anchor's own index exactly as his ``build_extended_knn_gpu`` does, and keeps the
first k -- his convention, k=2048 non-self neighbours, for the curvature rows only.

Usage:
    python notebooks/diagnostics/09_colleague_estimator_run.py --mode smoke \\
        --colleague-root <checkout> --record-path notebooks/.cache/09_scratch_colleague_smoke.jsonl
    python notebooks/diagnostics/09_colleague_estimator_run.py --mode run \\
        --colleague-root <checkout> --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 \\
        --threads 8 [--d 12,16,20] [--device cpu] [--output-root <dir>]
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
_RUNNER_PATH = DIAGNOSTICS_ROOT / "09_physics_curvature_run.py"

# Load the production runner FIRST, before numpy/torch are imported anywhere in this process:
# its module-level code applies the `--threads` cap (OMP/MKL/NUMEXPR env vars, then
# `torch.set_num_threads`) from `sys.argv` -- the same flag name this script uses -- so the cap
# mechanism is the production one, called rather than copied. It also puts `notebooks/` on
# `sys.path` for the `pu_manifold` imports below.
_spec = importlib.util.spec_from_file_location("physics_curvature_run", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402
from pu_manifold import subsample  # noqa: E402

EXPERIMENT = "supplementary-colleague-estimator"
RECORD_STEM_SUPPLEMENTARY = "09_colleague_estimator"
COLLEAGUE_COMMIT = "97efb2eb6cd7dec7f2c568f53c534752ff3c32c8"
COLLEAGUE_N_SPLITS = 3  # fit_kh_panel's n_splits default
COLLEAGUE_SEED = 0  # his curvature stage seed
COLLEAGUE_D_VALUES = (12, 16, 20)  # his parity set; d=16 is his primary
CURVATURE_FIELD = "K_H_cross"

# For the reader only -- printed beside the d=16 mag_r row of the final table. Never gates.
COLLEAGUE_REFERENCE = {12: {"controlled": 0.143}, 16: {"raw": -0.412, "controlled": -0.240}, 20: {"controlled": -0.233}}
OUR_AE_REFERENCE_D16 = {"raw": 0.425, "controlled": 0.347}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SHIM_ROOT = DIAGNOSTICS_ROOT / "colleague_shims"
"""On-disk stubs for his absent `topology.physics_activation_density_ph` dependency (module
docstring, DEPENDENCY SHIM). Placed on `sys.path` ahead of his `experiments/` root."""


def load_colleague_estimator(colleague_root: str) -> Dict[str, Any]:
    """Puts the shim directory and then `<colleague_root>/experiments` on `sys.path` (shim
    first), and imports his functions UNCHANGED. Returns them plus the checkout's resolved HEAD
    and whether the `topology` package that resolved is our shim."""
    root = Path(colleague_root).resolve()
    experiments = root / "experiments"
    if not experiments.is_dir():
        raise FileNotFoundError(f"--colleague-root {root} has no experiments/ directory.")
    for entry in (str(experiments), str(SHIM_ROOT)):  # shim ends up first
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True)
    colleague_head = head.stdout.strip() if head.returncode == 0 else None

    from geometry.physics_activation_atlas.full_curvature_audit import RIDGES  # noqa: E402
    from geometry.physics_activation_atlas.nested_dimension_curvature import (  # noqa: E402
        _fit_rank, nested_pca_frame,
    )
    from geometry.physics_adaptive_dataset_curvature_probe.curvature_stage import (  # noqa: E402
        METRIC_COLS, _rows_from_fits,
    )
    import topology  # noqa: E402  -- resolved by his import chain above

    topology_file = getattr(topology, "__file__", None) or ""
    return {
        "root": root,
        "colleague_head": colleague_head,
        "topology_is_shim": topology_file.startswith(str(SHIM_ROOT)),
        "topology_file": topology_file,
        "nested_pca_frame": nested_pca_frame,
        "_fit_rank": _fit_rank,
        "_rows_from_fits": _rows_from_fits,
        "metric_cols": list(METRIC_COLS),
        "ridges": list(RIDGES),
    }


def colleague_neighbourhoods(X: np.ndarray, anchor_idx: np.ndarray, k: int) -> Dict[str, Any]:
    """His convention for the curvature rows (module docstring, NEIGHBOURHOOD CONVENTION): the
    sealed `pcp.knn_panel` queried at k+1, the anchor's own index dropped exactly as his
    `build_extended_knn_gpu` does (`row = row[row != a]`), first k kept. Reports how many
    anchors were their own first neighbour in the k+1 query."""
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64)
    panel_plus = pcp.knn_panel(X, anchor_idx, k + 1)
    idx_plus = np.asarray(panel_plus["indices"], dtype=np.int64)
    neigh = np.zeros((anchor_idx.shape[0], k), dtype=np.int64)
    n_self_first = 0
    n_self_absent = 0
    for i, a in enumerate(anchor_idx):
        row = idx_plus[i]
        if row[0] == a:
            n_self_first += 1
        if not np.any(row == a):
            n_self_absent += 1
        row = row[row != a]
        neigh[i] = row[:k]
    return {"neigh": neigh, "n_self_first": n_self_first, "n_self_absent": n_self_absent}


def colleague_curvature_at_anchors(
    X: np.ndarray,
    neigh: np.ndarray,
    d_values: Sequence[int],
    est: Dict[str, Any],
    device: torch.device,
    n_splits: int,
    seed: int,
    progress_every: int = 32,
) -> Dict[int, Dict[str, np.ndarray]]:
    """`fit_kh_panel`'s loop body, per anchor: one nested PCA frame at `max(d_values)`, then
    `_fit_rank` at each d, aggregated by his own `_rows_from_fits`. Returns, per d, arrays of
    his metric columns plus `n_splits_ok`."""
    nested_pca_frame = est["nested_pca_frame"]
    _fit_rank = est["_fit_rank"]
    _rows_from_fits = est["_rows_from_fits"]
    metric_cols = est["metric_cols"]
    n_anchors, k = neigh.shape
    d_max = int(max(d_values))
    out: Dict[int, Dict[str, np.ndarray]] = {
        int(d): {**{c: np.full(n_anchors, np.nan, dtype=np.float64) for c in metric_cols},
                 "n_splits_ok": np.zeros(n_anchors, dtype=np.int64)}
        for d in d_values
    }
    t0 = time.monotonic()
    for ai in range(n_anchors):
        Xloc = X[neigh[ai, :k]].astype(np.float64)
        x0, J, _ev, _diag = nested_pca_frame(Xloc, d_max, device)
        for d in d_values:
            d = int(d)
            if J.shape[1] < d:
                continue
            fits = _fit_rank(Xloc, x0, J, d, k, n_splits, seed, ai)
            rec = _rows_from_fits(ai, d, k, fits)
            for c in metric_cols:
                out[d][c][ai] = float(rec[c])
            out[d]["n_splits_ok"][ai] = int(rec.get("n_splits_ok", 0))
        if progress_every and ((ai + 1) % progress_every == 0 or ai + 1 == n_anchors):
            elapsed = time.monotonic() - t0
            print(
                f"[colleague-curvature] {ai + 1}/{n_anchors} anchors, elapsed {elapsed:.1f}s, "
                f"~{elapsed / (ai + 1):.2f}s/anchor",
                flush=True,
            )
    return out


def _partial_block(
    x_full: np.ndarray, r2: np.ndarray, controls: np.ndarray, log_knn_radius: np.ndarray,
    strata_grid: Sequence[int], n_strat_draws: int, strat_seed: int, n_boot: int, boot_seed: int,
) -> Dict[str, Any]:
    """The per-(d, label) statistics `run_dsweep` computes inside its d loop, on the finite
    anchor set: raw Spearman, the 3-control partial, the density-stratified null at each entry
    of the strata grid and the paired anchor bootstrap -- the same sealed calls, in the same
    order. The Freedman-Lane family-wise null is NOT here: as in `run_dsweep` it is computed
    once per label across every d after the loop, on one common surrogate."""
    finite = np.isfinite(r2) & np.isfinite(x_full)
    x_f, y_f, z_f = x_full[finite], r2[finite], controls[finite]
    raw_rho = float(spearmanr(x_f, y_f).statistic) if x_f.size > 1 else float("nan")
    controlled = float(pcp.controlled_partial(x_f, y_f, z_f))
    strat = {
        int(s): pcp.stratified_partial_null_3control(
            x_f, y_f, z_f, log_knn_radius[finite], int(s), n_strat_draws, strat_seed,
        )
        for s in strata_grid
    }
    boot = pcp.paired_anchor_bootstrap(x_f, y_f, z_f, n_boot, boot_seed)
    return {
        "finite": finite, "n_finite": int(finite.sum()), "raw_rho": raw_rho,
        "controlled": controlled, "stratified": strat, "bootstrap": boot,
    }


def _parse_d_values(text: Optional[str]) -> Tuple[int, ...]:
    if not text:
        return COLLEAGUE_D_VALUES
    return tuple(int(tok) for tok in text.split(",") if tok.strip())


def _refuse_production_record(record_path: Path) -> None:
    if record_path.name == f"{pcp.RECORD_STEM}.jsonl":
        print(
            f"ERROR: this supplementary script refuses to write to the frozen production record "
            f"{record_path}; use the default {RECORD_STEM_SUPPLEMENTARY}.jsonl or another path.",
            file=sys.stderr,
        )
        sys.exit(2)


def run_smoke(args: argparse.Namespace) -> bool:
    """Synthetic end-to-end tracer on CPU: a curved d=4 surface in R^64, one synthetic label,
    his estimator at small d on his neighbourhood convention, then our sealed statistics with
    shrunken draw counts. Calls neither `assert_preregistered()`; writes stage rows to an
    explicit `--record-path` only."""
    print(
        "\n" + "=" * 78 + "\nSUPPLEMENTARY SMOKE ON SYNTHETIC ARRAYS -- NOT A DELIVERABLE, "
        "PRODUCES NO PHYSICS NUMBER.\n" + "=" * 78 + "\n"
    )
    runner._describe_environment()
    record_path = runner.resolve_record_path(args.record_path)
    _refuse_production_record(record_path)

    n, ambient, d_true = args.smoke_rows, 64, 4
    d_values = (4, 6)
    n_anchors, k, alpha, n_folds, seed = 64, 256, 100.0, 5, 20260902
    device = torch.device(args.device)

    stages: List[Tuple[str, bool]] = []

    def _stage(name: str, measured: Any, expected: Any, passed: bool) -> None:
        status = "PASS" if passed else "FAIL"
        print(f"stage={name} measured={measured} expected={expected} {status}")
        runner.append_record_row(
            {"stage": name, "measured": measured, "expected": expected, "passed": bool(passed),
             "experiment": EXPERIMENT},
            record_path,
        )
        stages.append((name, passed))

    t_start = time.monotonic()

    # 1. his code imports unchanged
    est = load_colleague_estimator(args.colleague_root)
    _stage("colleague_import", est["colleague_head"], COLLEAGUE_COMMIT, est["colleague_head"] == COLLEAGUE_COMMIT)
    print(f"topology resolved to shim: {est['topology_is_shim']} ({est['topology_file']})")

    # 2. synthetic curved surface (quadratic embedding of a d=4 latent), row-L2-normalised
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_true))
    quad = np.column_stack([Z[:, a] * Z[:, b] for a in range(d_true) for b in range(a, d_true)])
    A1 = rng.normal(size=(d_true, ambient)) / np.sqrt(d_true)
    A2 = rng.normal(size=(quad.shape[1], ambient)) / np.sqrt(quad.shape[1])
    X_raw = Z @ A1 + 0.5 * quad @ A2 + rng.normal(size=ambient) * 3.0
    X, _ = subsample.l2_normalize(X_raw)
    w = rng.normal(size=ambient)
    y = X @ w + 0.01 * rng.normal(size=n)

    idx = pcp.anchor_indices(n_rows=n, split_seed=seed, holdout_fraction=0.2, n_anchors=n_anchors, anchor_seed=seed)
    anchor_idx = idx["anchor_idx"]

    # 3. neighbourhood conventions: ours includes self at column 0, his excludes it
    knn = pcp.knn_panel(X, anchor_idx, k)
    ours_self_first = int(np.sum(knn["indices"][:, 0] == anchor_idx))
    _stage("sealed_knn_self_first", ours_self_first, n_anchors, ours_self_first == n_anchors)
    his = colleague_neighbourhoods(X, anchor_idx, k)
    no_self = int(np.sum(np.any(his["neigh"] == anchor_idx[:, None], axis=1)))
    _stage("colleague_neigh_excludes_self", no_self, 0, no_self == 0 and his["neigh"].shape == (n_anchors, k))

    # 4. OOF probe, local R2, controls (sealed)
    y_hat = runner._oof_predictions_for_label(X, y, alpha, n_folds, seed)
    panel = pcp.local_r2_panel(y, y_hat, knn["indices"], min_finite=10)
    n_valid = int(np.sum(np.isfinite(panel["r2"])))
    _stage("local_r2", n_valid, ">0", n_valid > 0)
    controls = np.column_stack([knn["log_knn_radius"], panel["local_label_variance"], panel["local_evaluation_count"]])

    # 5. his estimator
    t0 = time.monotonic()
    curv = colleague_curvature_at_anchors(
        X, his["neigh"], d_values, est, device, COLLEAGUE_N_SPLITS, COLLEAGUE_SEED, progress_every=0,
    )
    print(f"colleague estimator wallclock: {time.monotonic() - t0:.1f}s for {n_anchors} anchors x {d_values}")
    kh = curv[d_true][CURVATURE_FIELD]
    n_finite = int(np.sum(np.isfinite(kh)))
    _stage("colleague_curvature_finite", n_finite, f">= {int(0.9 * n_anchors)}", n_finite >= int(0.9 * n_anchors))
    n_all_splits = int(np.sum(curv[d_true]["n_splits_ok"] == COLLEAGUE_N_SPLITS))
    _stage("colleague_n_splits_ok", n_all_splits, f">= {int(0.9 * n_anchors)}", n_all_splits >= int(0.9 * n_anchors))
    r_h_median = float(np.nanmedian(curv[d_true]["R_H"]))
    _stage("colleague_R_H_median", round(r_h_median, 4), "> 0 (split halves agree on H)", r_h_median > 0.0)

    # 6. our sealed statistics on his field
    block = _partial_block(
        kh, panel["r2"], controls, knn["log_knn_radius"], strata_grid=(5,), n_strat_draws=args.smoke_permutations,
        strat_seed=seed, n_boot=args.smoke_permutations, boot_seed=seed,
    )
    _stage("controlled_partial", round(block["controlled"], 4), "finite", bool(np.isfinite(block["controlled"])))
    _stage("stratified_null", block["stratified"][5]["p_display"], "p in (0, 1]", 0.0 < block["stratified"][5]["p"] <= 1.0)
    finite_all = np.isfinite(panel["r2"])
    for d in d_values:
        finite_all &= np.isfinite(curv[d][CURVATURE_FIELD])
    fwer = pcp.permutation_fwer(
        {d: curv[d][CURVATURE_FIELD][finite_all] for d in d_values}, panel["r2"][finite_all],
        controls[finite_all], n_permutations=args.smoke_permutations, seed=seed,
    )
    _stage("permutation_fwer", fwer["global"]["p_display"], "p in (0, 1]", 0.0 < fwer["global"]["p"] <= 1.0)
    boot = block["bootstrap"]
    _stage("bootstrap", (round(boot["ci_low"], 4), round(boot["ci_high"], 4)), "ci_low <= ci_high", boot["ci_low"] <= boot["ci_high"])

    all_passed = all(passed for _, passed in stages)
    print(f"\ntotal wallclock: {time.monotonic() - t_start:.1f}s")
    print(f"record written to: {record_path}")
    print("\nSMOKE PASS" if all_passed else "\nSMOKE FAIL")
    return all_passed


def run_real(args: argparse.Namespace) -> bool:
    """`--mode run`: the supplementary experiment on the Physics data. Gate, then in order:
    loaders, the Wave A anchors (never redrawn), sealed k-NN panel, OOF/local R2/controls per
    label, his estimator per anchor per d, our statistics per d/label on the single field
    `K_H_cross`, the family-wise Freedman-Lane null per label across all d, the summary table."""
    env = runner._gate_and_environment(args)
    freeze_commit = runner._git_rev_parse(args.freeze_commit)
    run_commit = runner._git_rev_parse("HEAD")
    record_path = runner.resolve_record_path(args.record_path, default_stem=RECORD_STEM_SUPPLEMENTARY)
    _refuse_production_record(record_path)
    output_root = pcp.resolve_output_root()
    d_values = _parse_d_values(args.d)
    device = torch.device(args.device)

    print(f"\n{'=' * 78}\n[step 0] colleague estimator import from {args.colleague_root}\n{'=' * 78}")
    est = load_colleague_estimator(args.colleague_root)
    if est["colleague_head"] != COLLEAGUE_COMMIT:
        print(
            f"ERROR: colleague checkout HEAD is {est['colleague_head']!r}, expected "
            f"{COLLEAGUE_COMMIT}. Refusing to run against a different version of his code.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"colleague commit {est['colleague_head']} ok; RIDGES={est['ridges']}; "
        f"metric columns={est['metric_cols']}; topology resolved to shim={est['topology_is_shim']}"
    )

    common = {
        "freeze_commit": freeze_commit, "run_commit": run_commit, "colleague_commit": COLLEAGUE_COMMIT,
        "experiment": EXPERIMENT,
    }
    runner.append_record_row(
        {**env, **common, "colleague_root": str(est["root"]), "topology_is_shim": est["topology_is_shim"], "topology_file": est["topology_file"],
         "d_values": list(d_values), "n_splits": COLLEAGUE_N_SPLITS, "estimator_seed": COLLEAGUE_SEED,
         "device": str(device), "timestamp_utc": _utc_now()},
        record_path,
    )

    all_labels = (pl.PRIMARY_LABEL,) + pl.SECONDARY_LABELS
    k = pcp.K_NEIGHBOURS

    print(f"\n{'=' * 78}\n[step 1] embeddings + labels through the sealed loaders\n{'=' * 78}")
    t0 = time.monotonic()
    emb = pl.load_physics_embeddings()
    X, n_rows = emb["X"], emb["n_rows"]
    print(f"[load] physics embeddings: n_rows={n_rows} wallclock={time.monotonic() - t0:.1f}s")
    t0 = time.monotonic()
    table = pl.load_label_table(columns=list(pl.LABEL_COLUMN_MAP.values()))
    print(f"[load] label table: wallclock={time.monotonic() - t0:.1f}s")
    offset_perm = pl.shifted_pairing(n_rows, pl.ALIGNMENT_ASSUMED_OFFSET)
    y_by_label = {
        name: pl.canonical_label(table, name, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)[offset_perm]
        for name in all_labels
    }

    print(f"\n{'=' * 78}\n[step 2] the 512 Wave A anchors, read from the anchor table (never redrawn)\n{'=' * 78}")
    wave_a_path = runner._anchor_table_path(output_root, 16, pl.PRIMARY_LABEL)
    wave_a = runner.load_anchor_table(wave_a_path)
    anchor_idx = np.asarray(wave_a["anchor_idx"], dtype=np.int64)
    if anchor_idx.shape[0] != pcp.N_ANCHORS:
        print(f"ERROR: {wave_a_path} carries {anchor_idx.shape[0]} anchors, expected {pcp.N_ANCHORS}.", file=sys.stderr)
        sys.exit(1)
    redraw = pcp.anchor_indices(
        n_rows=n_rows, split_seed=pcp.SPLIT_SEED, holdout_fraction=pcp.HOLDOUT_FRACTION,
        n_anchors=pcp.N_ANCHORS, anchor_seed=pcp.ANCHOR_DRAW_SEED,
    )["anchor_idx"]
    if not np.array_equal(np.asarray(redraw, dtype=np.int64), anchor_idx):
        print(f"ERROR: anchors in {wave_a_path} differ from the frozen deterministic draw; refusing to proceed.", file=sys.stderr)
        sys.exit(1)
    print(f"anchors: {anchor_idx.shape[0]} from {wave_a_path.name}, equal to the frozen draw")

    print(f"\n{'=' * 78}\n[step 3] sealed k-NN panel at K_NEIGHBOURS={k}, plus his self-excluded rows\n{'=' * 78}")
    t0 = time.monotonic()
    knn = pcp.knn_panel(X, anchor_idx, k)
    print(f"[knn] sealed panel wallclock={time.monotonic() - t0:.1f}s")
    t0 = time.monotonic()
    his = colleague_neighbourhoods(X, anchor_idx, k)
    print(
        f"[knn] colleague rows wallclock={time.monotonic() - t0:.1f}s; anchors self-first in the k+1 query: "
        f"{his['n_self_first']}/{anchor_idx.shape[0]}; self absent: {his['n_self_absent']}"
    )

    print(f"\n{'=' * 78}\n[step 4] OOF probe + local R2 + mse/sst per label (sealed)\n{'=' * 78}")
    panel_by_label: Dict[str, Dict[str, Any]] = {}
    for name in all_labels:
        t0 = time.monotonic()
        y_hat = runner._oof_predictions_for_label(X, y_by_label[name], pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
        panel = pcp.local_r2_panel(y_by_label[name], y_hat, knn["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        mse, sst = runner.local_mse_sst_panel(y_by_label[name], y_hat, knn["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        const_eval = bool(np.all(panel["local_evaluation_count"] == panel["local_evaluation_count"][0]))
        panel_by_label[name] = {"panel": panel, "mse": mse, "sst": sst, "const_eval": const_eval}
        print(
            f"[oof/local_r2] label={name} n_masked_anchors={panel['n_masked_anchors']} "
            f"local_evaluation_count_constant={const_eval} wallclock={time.monotonic() - t0:.1f}s"
        )

    print(f"\n{'=' * 78}\n[step 5] the three controls (sealed construction)\n{'=' * 78}")
    controls_by_label = {
        name: np.column_stack([
            knn["log_knn_radius"],
            panel_by_label[name]["panel"]["local_label_variance"],
            panel_by_label[name]["panel"]["local_evaluation_count"],
        ])
        for name in all_labels
    }
    print(f"controls: {pcp.CONTROLS}")

    print(f"\n{'=' * 78}\n[step 6] his K_H^cross at every anchor, d={d_values}, n_splits={COLLEAGUE_N_SPLITS}, seed={COLLEAGUE_SEED}\n{'=' * 78}")
    t0 = time.monotonic()
    curv = colleague_curvature_at_anchors(X, his["neigh"], d_values, est, device, COLLEAGUE_N_SPLITS, COLLEAGUE_SEED)
    wallclock_curvature = time.monotonic() - t0
    for d in d_values:
        kh = curv[d][CURVATURE_FIELD]
        n_nonfinite = int(np.sum(~np.isfinite(kh)))
        splits_ok = curv[d]["n_splits_ok"]
        summary = {
            "row_kind": "curvature_summary", "d": int(d), "field": CURVATURE_FIELD, "k": int(k),
            "n_anchors": int(anchor_idx.shape[0]), "n_nonfinite_anchors": n_nonfinite,
            "K_H_cross_median": float(np.nanmedian(kh)), "K_H_cross_p05": float(np.nanpercentile(kh, 5)),
            "K_H_cross_p95": float(np.nanpercentile(kh, 95)), "R_H_median": float(np.nanmedian(curv[d]["R_H"])),
            "dS_median": float(np.nanmedian(curv[d]["dS"])),
            "n_splits_ok_histogram": {str(v): int(c) for v, c in zip(*np.unique(splits_ok, return_counts=True))},
            "wallclock_curvature_all_d_s": wallclock_curvature, **common, "timestamp_utc": _utc_now(),
        }
        runner.append_record_row(summary, record_path)
        print(
            f"[d={d}] K_H_cross median={summary['K_H_cross_median']:.4e} R_H median={summary['R_H_median']:.3f} "
            f"non-finite anchors={n_nonfinite} n_splits_ok={summary['n_splits_ok_histogram']}"
        )

    print(f"\n{'=' * 78}\n[step 7] anchor tables + partial / stratified null / bootstrap per d and label\n{'=' * 78}")
    results: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for d in d_values:
        table_out: Dict[str, np.ndarray] = {
            "anchor_idx": anchor_idx, "log_knn_radius": np.asarray(knn["log_knn_radius"], dtype=np.float64),
            "n_splits_ok": curv[d]["n_splits_ok"],
        }
        for c in est["metric_cols"]:
            table_out[c] = np.asarray(curv[d][c], dtype=np.float64)
        for name in all_labels:
            panel = panel_by_label[name]["panel"]
            table_out[f"r2_{name}"] = np.asarray(panel["r2"], dtype=np.float64)
            table_out[f"mse_{name}"] = np.asarray(panel_by_label[name]["mse"], dtype=np.float64)
            table_out[f"sst_{name}"] = np.asarray(panel_by_label[name]["sst"], dtype=np.float64)
            table_out[f"local_label_variance_{name}"] = np.asarray(panel["local_label_variance"], dtype=np.float64)
            table_out[f"local_evaluation_count_{name}"] = np.asarray(panel["local_evaluation_count"], dtype=np.int64)
        table_path = output_root / f"09_colleague_anchor_table_d{d}.npz"
        runner.write_anchor_table(table_out, table_path)

        for name in all_labels:
            panel = panel_by_label[name]["panel"]
            block = _partial_block(
                curv[d][CURVATURE_FIELD], panel["r2"], controls_by_label[name], knn["log_knn_radius"],
                pcp.STRATA_GRID, pcp.STRATIFIED_NULL_DRAWS, pcp.STRATIFIED_NULL_SEED, pcp.N_BOOTSTRAP, pcp.BOOTSTRAP_SEED,
            )
            results[(d, name)] = block
            runner.append_record_row(
                {"row_kind": "anchor_summary", "d": int(d), "label": name, "field": CURVATURE_FIELD,
                 "n_anchors": int(anchor_idx.shape[0]), "n_masked_anchors": panel["n_masked_anchors"],
                 "n_nonfinite_curvature": int(np.sum(~np.isfinite(curv[d][CURVATURE_FIELD]))),
                 "n_finite_anchors": block["n_finite"],
                 "local_evaluation_count_constant": panel_by_label[name]["const_eval"],
                 "anchor_table_path": str(table_path), **common, "timestamp_utc": _utc_now()},
                record_path,
            )
            runner.append_record_row(
                {"row_kind": "partial", "d": int(d), "label": name, "field": CURVATURE_FIELD,
                 "n_finite_anchors": block["n_finite"], "raw_rho": block["raw_rho"],
                 "controlled_partial": block["controlled"], **common, "timestamp_utc": _utc_now()},
                record_path,
            )
            for n_strata, strat in block["stratified"].items():
                runner.append_record_row(
                    {"row_kind": "null", "null_type": "stratified", "d": int(d), "label": name, "field": CURVATURE_FIELD,
                     "n_strata": int(n_strata), "observed": strat["observed"], "p": strat["p"],
                     "p_display": strat["p_display"], "floor_reached": strat["floor_reached"],
                     **common, "timestamp_utc": _utc_now()},
                    record_path,
                )
            boot = block["bootstrap"]
            runner.append_record_row(
                {"row_kind": "bootstrap", "d": int(d), "label": name, "field": CURVATURE_FIELD,
                 "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "n_boot": boot["n_boot"],
                 **common, "timestamp_utc": _utc_now()},
                record_path,
            )
            print(
                f"[d={d}] label={name} n_finite={block['n_finite']} raw_rho={block['raw_rho']:+.4f} "
                f"controlled={block['controlled']:+.4f} strat_p={{" +
                ", ".join(f"{s}: {v['p_display']}" for s, v in block["stratified"].items()) +
                f"}} boot95=({boot['ci_low']:+.4f}, {boot['ci_high']:+.4f})"
            )

    print(f"\n{'=' * 78}\n[step 8] Freedman-Lane family-wise null per label across all d (one common surrogate)\n{'=' * 78}")
    fwer_by: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for name in all_labels:
        panel = panel_by_label[name]["panel"]
        finite = np.isfinite(panel["r2"])
        for d in d_values:
            finite &= np.isfinite(curv[d][CURVATURE_FIELD])
        y_f, z_f = panel["r2"][finite], controls_by_label[name][finite]
        t0 = time.monotonic()
        fwer = pcp.permutation_fwer(
            {d: curv[d][CURVATURE_FIELD][finite] for d in d_values}, y_f, z_f, pcp.N_PERMUTATIONS, pcp.PERMUTATION_SEED,
        )
        for d in d_values:
            per_d = fwer["per_d"][d]
            fwer_by[(d, name)] = per_d
            runner.append_record_row(
                {"row_kind": "null", "null_type": "fwer", "d": int(d), "label": name, "field": CURVATURE_FIELD,
                 "n_finite_anchors": int(finite.sum()), "observed_rho": per_d["observed_rho"], "p": per_d["p"],
                 "p_display": per_d["p_display"], "floor_reached": per_d["floor_reached"],
                 **common, "timestamp_utc": _utc_now()},
                record_path,
            )
        runner.append_record_row(
            {"row_kind": "null", "null_type": "fwer_global", "label": name, "field": CURVATURE_FIELD,
             "d_values": list(d_values), "n_finite_anchors": int(finite.sum()), "p": fwer["global"]["p"],
             "p_display": fwer["global"]["p_display"], "floor_reached": fwer["global"]["floor_reached"],
             **common, "timestamp_utc": _utc_now()},
            record_path,
        )
        print(f"[fwer] label={name} global p_display={fwer['global']['p_display']} wallclock={time.monotonic() - t0:.1f}s")

    print(f"\n{'=' * 78}\nSUMMARY -- colleague K_H^cross inside the Phase 9 pipeline (no verdict is computed)\n{'=' * 78}")
    print(f"{'d':>4} {'label':<16}{'raw rho':>10}{'controlled':>12}{'FWER p':>14}{'strat p (S=' + ','.join(map(str, pcp.STRATA_GRID)) + ')':>28}{'n finite':>10}")
    for d in d_values:
        for name in all_labels:
            block, per_d = results[(d, name)], fwer_by[(d, name)]
            strat_txt = " / ".join(block["stratified"][s]["p_display"] for s in pcp.STRATA_GRID)
            print(
                f"{d:>4} {name:<16}{block['raw_rho']:>+10.4f}{block['controlled']:>+12.4f}{per_d['p_display']:>14}"
                f"{strat_txt:>28}{block['n_finite']:>10}"
            )
            if name == pl.PRIMARY_LABEL and d in COLLEAGUE_REFERENCE:
                ref = COLLEAGUE_REFERENCE[d]
                ref_txt = ", ".join(f"{key} {val:+.3f}" for key, val in ref.items())
                print(f"{'':>4} {'  his frozen ref:':<20}{ref_txt}")
                if d == 16:
                    print(
                        f"{'':>4} {'  our AE d=16:':<20}raw {OUR_AE_REFERENCE_D16['raw']:+.3f}, "
                        f"controlled {OUR_AE_REFERENCE_D16['controlled']:+.3f} (plain-autoencoder decoder curvature, frozen record)"
                    )
    print(f"\nSUPPLEMENTARY RUN done. Record: {record_path}. This record never feeds --mode verdict.")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "run"], default="smoke")
    p.add_argument("--colleague-root", type=str, required=True, help="read-only checkout at COLLEAGUE_COMMIT")
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--freeze-commit", type=str, default=None)
    p.add_argument("--d", type=str, default=None, help="comma-separated chart ranks; default 12,16,20")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--smoke-rows", type=int, default=3000)
    p.add_argument("--smoke-permutations", type=int, default=200)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_root and pcp.OUTPUT_ROOT_ENV_VAR:
        os.environ[pcp.OUTPUT_ROOT_ENV_VAR] = args.output_root
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    if args.mode == "smoke":
        sys.exit(0 if run_smoke(args) else 1)
    sys.exit(0 if run_real(args) else 1)


if __name__ == "__main__":
    main()
