"""Phase 02.6 decoder-substrate screening: the persistent-homology read-out matrix across
all three in-repo candidates (a plain autoencoder, a TopoAE-trained ``cae.PlainAutoEncoder``,
and the Chart Auto-Encoder as the measured negative control), at four seeds each.

**This script measures and prints; it applies no bar of any kind, emits no verdict token,
seals nothing and writes no artifact.** The criterion, its bars, and the exact sixteen-cell
read-out matrix are ratified in
``.planning/phases/02.6-decoder-substrate-screening/02.6-SCREENING-RULE-02.md`` -- every
number this script prints is measurement only; the ranking that reads these numbers is
assembled later, in ``02.6-FINDINGS-02.md`` (plan 02.6-15), never here. This is the same
shape ``notebooks/diagnostics/decoder_substrate_screen_run.py`` (plan 02.6-05) established
for the halted curvature axis -- that sibling script is not edited, moved, or replaced by
this one.

The candidate set is exactly the three in-repo substrates (D-20): the CAE enters
unconditionally this time, screened alongside the two free candidates with no tier gating
-- persistence diagrams are built from point coordinates alone and need no C2 smoothness
guard, so the chart-fragmented decoder that excluded the CAE from the halted curvature axis
needs no special casing on this axis.

The read-out matrix (D-03 x D-04 x D-05 x D-06): 2 references (the exact intrinsic
2-manifold; the ambient 3-D point cloud) x 2 degrees (H0, H1) x 2 distances (bottleneck,
Wasserstein, each normalised by the reference diagram's own max persistence) x 2 spaces
(encoder latent, decoder image) = 16 numbers per seed per candidate, 64 across four seeds
(D-11), 192 across the three candidates -- Swiss-roll side only. All sixteen are reported
separately and never weighted, summed, averaged, or collapsed into any single figure.
``persistence_probe.READOUT_CELLS`` is the one place the sixteen canonical labels are
written down; every table this script emits keys off that tuple rather than re-spelling
them.

Invoke:
    .venv/bin/python notebooks/diagnostics/decoder_substrate_ph_screen_run.py --smoke
    .venv/bin/python notebooks/diagnostics/decoder_substrate_ph_screen_run.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import analytic_param, cae, curvature_probe, persistence_probe, topoae  # noqa: E402

# --- Swiss roll fixture / shared plain-AE + TopoAE model constants, carried forward
# VERBATIM from decoder_substrate_screen_run.py (plan 02.6-05) -- unchanged by the axis
# switch, so a runner number on the old axis and a runner number here at the same seed
# still share the identical fixture and the identical plain-AE/TopoAE fit protocol.
FIXTURE_SEED = 20260807
N_POINTS = 3000
LATENT_DIM = 2
HIDDEN = (64, 64, 64)
LAMBDA_TOPO = 0.1
HOLDOUT_FRACTION = 0.2

# early_stop_patience is DELIBERATELY ABSENT from CFG_COMMON below, matching plan 02.6-05's
# own choice: both cae.train_plain_ae and topoae.train_topoae default it to max_epochs + 1
# when the key is unset, so the plain-AE and TopoAE arms run the identical full-budget
# schedule with no early-stopping asymmetry between them.
CFG_COMMON = dict(lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=150)

DEFAULT_SEEDS = (0, 1, 2, 3)  # ratified seed protocol, 02.6-SCREENING-RULE-02.md

# --- new constants for this axis (D-20) ----------------------------------------------------

CANDIDATES = ("plainae", "topoae", "cae")

# The CAE's own architecture and configuration, reused VERBATIM from
# notebooks/02.2_swiss_roll_cae_check.ipynb (cells 2 and 6) -- not re-derived, not tuned for
# this screen. Chart dimension equals LATENT_DIM, matching the roll's true intrinsic
# dimension the same way the plain-AE/TopoAE bottleneck does.
CAE_EMBED_DIM = 8
CAE_N_CHARTS = 8
CAE_HIDDEN = (64, 64)
CAE_PRUNE_TOL = 1e-2  # chart-survival tolerance, also carried verbatim from the 02.2 notebook
CFG_CAE = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch=64,
    max_epochs=300,
    early_stop_patience=25,
    early_stop_min_delta=1e-4,
    fps_pretrain_epochs=20,
    lip_weight=1e-3,
    lip_every_n_steps=1,
)

# Ratified Q1 (02.6-SCREENING-RULE-02.md): per-cloud pre-scaling via
# topoae.latent_unit_scale, applied SYMMETRICALLY to every cloud -- model and reference
# alike -- before diagram construction, with the applied scalar printed per cloud.
# persistence_probe.cloud_distance_matrix and .readout_matrix both take `prescale` as a
# required positional argument with no default, so this module constant is the one place
# this convention is written down; every call site below threads it through explicitly.
PRESCALE_CLOUDS = True

# The sixteen canonical labels, decomposed once at import time into their four axes --
# never re-spelled, always read off persistence_probe.READOUT_CELLS itself, so this
# runner's tables cannot silently drift from the ratified document's own list and order.
_CELLS = [tuple(cell.split("|")) for cell in persistence_probe.READOUT_CELLS]

_SPACE_REFERENCE_PAIRS = []
for _space, _reference, _degree, _distance_norm in _CELLS:
    _pair = (_space, _reference)
    if _pair not in _SPACE_REFERENCE_PAIRS:
        _SPACE_REFERENCE_PAIRS.append(_pair)

_DEGREES = []
for _space, _reference, _degree, _distance_norm in _CELLS:
    if _degree not in _DEGREES:
        _DEGREES.append(_degree)

_DISTANCES = []
for _space, _reference, _degree, _distance_norm in _CELLS:
    _distance = _distance_norm[: -len("_norm")]
    if _distance not in _DISTANCES:
        _DISTANCES.append(_distance)

_fmt = lambda v: f"{float(v):.4f}" if np.isfinite(v) else "nan"  # noqa: E731


def build_fixture(n: int, fixture_seed: int) -> dict:
    """The Swiss roll fixture, built once per invocation and shared across every seed and
    every candidate -- only the torch/split seed varies per screened row (D-11). Returns a
    dict carrying the ambient array ``X`` (centred, divided by one global scalar), the
    spiral parameter ``t``, ``global_std``, the float32 training tensor ``x_all``, and the
    exact intrinsic-plane reference (D-03's reference (a)) via
    ``analytic_param.swiss_roll_intrinsic_plane``.

    ``X``'s ruling column (index 1) is already scaled by the same single global scalar the
    rest of ``X`` went through, so it is passed straight through to
    ``swiss_roll_intrinsic_plane`` and never re-divided -- re-dividing would double-scale
    it.
    """
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=fixture_seed)
    intrinsic_ref = analytic_param.swiss_roll_intrinsic_plane(fx["t"], fx["X"][:, 1], fx["global_std"])
    x_all = torch.tensor(fx["X"], dtype=torch.float32)
    return {
        "X": fx["X"],
        "t": fx["t"],
        "global_std": fx["global_std"],
        "x_all": x_all,
        "intrinsic_ref": intrinsic_ref,
    }


def train_candidate(candidate: str, x_train: torch.Tensor, seed: int):
    """Trains one of the three in-repo candidates (D-20) from scratch and returns
    ``(model.eval(), fit)``.

    ``"plainae"`` and ``"topoae"`` build the identical
    ``cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")`` after
    ``torch.manual_seed(seed)`` -- the TopoAE arm is a plain autoencoder trained under a
    different loss, never a different architecture, exactly as
    ``decoder_substrate_screen_run.py`` states. ``"plainae"`` trains under
    ``cae.train_plain_ae``; ``"topoae"`` trains under ``topoae.train_topoae`` with
    ``lambda_topo=LAMBDA_TOPO`` and a quarter-warmup/quarter-ramp schedule.

    ``"cae"`` builds ``cae.ChartAutoEncoder`` at the constants above and trains under
    ``cae.train_cae`` -- the measured negative control, screened unconditionally, with no
    special casing (persistent homology needs no C2 smoothness guard).

    Raises ``ValueError`` naming the received value for any other candidate string.
    """
    torch.manual_seed(seed)
    if candidate == "plainae":
        model = cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")
        fit = cae.train_plain_ae(model, x_train, dict(seed=seed, **CFG_COMMON))
    elif candidate == "topoae":
        model = cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")
        fit = topoae.train_topoae(
            model,
            x_train,
            dict(seed=seed, lambda_topo=LAMBDA_TOPO, warmup_frac=0.25, ramp_frac=0.25, **CFG_COMMON),
        )
    elif candidate == "cae":
        model = cae.ChartAutoEncoder(
            in_dim=3,
            embed_dim=CAE_EMBED_DIM,
            chart_dim=LATENT_DIM,
            n_charts=CAE_N_CHARTS,
            hidden=CAE_HIDDEN,
            activation="silu",
        )
        fit = cae.train_cae(model, x_train, dict(seed=seed, n_charts=CAE_N_CHARTS, **CFG_CAE))
    else:
        raise ValueError(
            f"train_candidate: unknown candidate {candidate!r}; expected one of {CANDIDATES}."
        )
    return model.eval(), fit


def screen_seed(candidate: str, fx: dict, seed: int, prescale: bool) -> dict:
    """One (candidate, seed) measurement: trains from scratch on an 80/20 split of the
    shared fixture, then measures the full sixteen-cell read-out matrix on the HELD-OUT
    rows only, with the SAME held-out row indices used for both references -- a mismatched
    index set would silently compare different subsets of the manifold (T-02.6R-17).

    Split via ``np.random.default_rng(seed).permutation``, matching
    ``decoder_substrate_screen_run.py``'s own protocol: the fixture is held fixed and only
    the torch/split seed varies.

    Decoder-image extraction is candidate-dependent and load-bearing (a correction to
    ``02.6-RESEARCH.md`` Pattern 3 and ``02.6-PATTERNS.md``, verified against source):
    ``model.forward(x)["y"]`` for the two plain-autoencoder arms, ``model.reconstruct(x)``
    for the CAE, since ``cae.ChartAutoEncoder.forward`` returns no ``"y"`` key. Latent
    extraction is ``model.encode(x)`` for every candidate; per the ratified disposition of
    option group R (R1), the CAE's latent is its global ``embed_dim`` embedding, never
    chart-local coordinates.

    Calls ``persistence_probe.readout_matrix`` exactly once -- four diagrams
    (``latent``, ``decoder_image``, ``intrinsic``, ``ambient``), decomposed into the
    sixteen canonical cells by that function, never sixteen separate calls.

    Returns one flat dict carrying every key ``readout_matrix`` returns, plus ``candidate``,
    ``seed``, ``latent_dim``, ``wallclock_s``, ``epochs_run``, ``early_stopped``,
    ``recon_rel_err`` (the held-out mean of ``||x - y|| / ||x||``, the established
    per-point-ratio convention), and, for the ``"cae"`` arm only,
    ``n_charts_surviving``/``n_charts_initial`` from ``cae.chart_survival``.
    """
    x_all = fx["x_all"]
    n = x_all.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * (1 - HOLDOUT_FRACTION))
    train_idx = perm[:n_train]
    holdout_idx = perm[n_train:]
    x_train = x_all[torch.from_numpy(train_idx)]
    x_holdout = x_all[torch.from_numpy(holdout_idx)]

    t0 = time.monotonic()
    model, fit = train_candidate(candidate, x_train, seed)
    wallclock_s = time.monotonic() - t0

    with torch.no_grad():
        if candidate == "cae":
            y_holdout = model.reconstruct(x_holdout)
        else:
            y_holdout = model(x_holdout)["y"]
        z_holdout = model.encode(x_holdout)

    recon_rel_err = float(
        (
            torch.linalg.vector_norm(x_holdout - y_holdout, dim=-1)
            / torch.linalg.vector_norm(x_holdout, dim=-1)
        ).mean()
    )

    latent_np = z_holdout.detach().cpu().numpy().astype(np.float64)
    decoder_image_np = y_holdout.detach().cpu().numpy().astype(np.float64)
    intrinsic_ref_holdout = fx["intrinsic_ref"][holdout_idx]
    ambient_ref_holdout = fx["X"][holdout_idx]

    spaces = {"latent": latent_np, "decoder_image": decoder_image_np}
    references = {"intrinsic": intrinsic_ref_holdout, "ambient": ambient_ref_holdout}
    readout = persistence_probe.readout_matrix(spaces, references, prescale)

    latent_dim = CAE_EMBED_DIM if candidate == "cae" else LATENT_DIM

    row = dict(readout)
    row.update(
        {
            "candidate": candidate,
            "seed": int(seed),
            "latent_dim": int(latent_dim),
            "wallclock_s": float(wallclock_s),
            "epochs_run": int(fit["epochs_run"]),
            "early_stopped": bool(fit["early_stopped"]),
            "recon_rel_err": recon_rel_err,
        }
    )
    if candidate == "cae":
        survival = cae.chart_survival(model, prune_tol=CAE_PRUNE_TOL)
        row["n_charts_surviving"] = int(survival["n_charts_surviving"])
        row["n_charts_initial"] = int(survival["n_charts_initial"])
    return row


def markdown_table(rows):
    """Two GitHub-flavoured markdown tables, keyed entirely off ``_SPACE_REFERENCE_PAIRS``/
    ``_DEGREES``/``_DISTANCES`` (themselves derived from ``persistence_probe.READOUT_CELLS``
    at import time, never re-spelled here).

    The first table has one row per (candidate, seed, space, reference) -- 48 rows at the
    full four-seed, three-candidate sweep -- carrying the candidate, seed, space,
    reference, latent dimension, each cloud's own applied pre-scaling factor, and for every
    (degree, distance) pair the normalised value, the raw value, the reference's own max
    persistence, and (bottleneck only) the saturation flag. Four normalised numbers per row
    times 48 rows is exactly the 192-number read-out matrix.

    The second table has one row per (candidate, space, reference, degree, distance) -- 48
    rows -- carrying mean, median and ``SPREAD = max - min`` of the normalised value over
    however many of the four seeds produced a finite reading for that cell.

    Neither table ever emits a column or a row that weights, sums, averages, or ranks cells
    against each other, and neither table emits a summary row across candidates.
    """
    header1 = ["candidate", "seed", "space", "reference", "latent_dim", "space_scale", "ref_scale"]
    for degree in _DEGREES:
        for distance in _DISTANCES:
            header1.append(f"{degree}_{distance}_norm")
            header1.append(f"{degree}_{distance}_raw")
            if distance == "bottleneck":
                header1.append(f"{degree}_{distance}_saturated")
        header1.append(f"{degree}_ref_max_persistence")

    lines1 = ["| " + " | ".join(header1) + " |", "| " + " | ".join("---" for _ in header1) + " |"]
    for row in rows:
        for space, reference in _SPACE_REFERENCE_PAIRS:
            cells = [
                row["candidate"],
                str(row["seed"]),
                space,
                reference,
                str(row["latent_dim"]),
                _fmt(row[f"{space}|applied_scale"]),
                _fmt(row[f"{reference}|applied_scale"]),
            ]
            for degree in _DEGREES:
                for distance in _DISTANCES:
                    norm_key = f"{space}|{reference}|{degree}|{distance}_norm"
                    raw_key = f"{space}|{reference}|{degree}|{distance}_raw"
                    cells.append(_fmt(row[norm_key]))
                    cells.append(_fmt(row[raw_key]))
                    if distance == "bottleneck":
                        sat_key = f"{space}|{reference}|{degree}|saturated"
                        cells.append(str(bool(row[sat_key])))
                cells.append(_fmt(row[f"{reference}|{degree}|ref_max_persistence"]))
            lines1.append("| " + " | ".join(cells) + " |")
    measurement_table = "\n".join(lines1)

    header2 = ["candidate", "space", "reference", "degree", "distance", "mean", "median", "SPREAD"]
    lines2 = ["| " + " | ".join(header2) + " |", "| " + " | ".join("---" for _ in header2) + " |"]

    seen_candidates = []
    for row in rows:
        if row["candidate"] not in seen_candidates:
            seen_candidates.append(row["candidate"])

    for candidate in seen_candidates:
        cand_rows = [r for r in rows if r["candidate"] == candidate]
        for space, reference in _SPACE_REFERENCE_PAIRS:
            for degree in _DEGREES:
                for distance in _DISTANCES:
                    key = f"{space}|{reference}|{degree}|{distance}_norm"
                    values = [r[key] for r in cand_rows]
                    finite = [v for v in values if np.isfinite(v)]
                    if finite:
                        mean_s = _fmt(float(np.mean(finite)))
                        median_s = _fmt(float(np.median(finite)))
                        spread_s = _fmt(float(max(finite) - min(finite)))
                    else:
                        mean_s = median_s = spread_s = "nan"
                    lines2.append(
                        "| "
                        + " | ".join(
                            [candidate, space, reference, degree, distance, mean_s, median_s, spread_s]
                        )
                        + " |"
                    )
    spread_table = "\n".join(lines2)

    return measurement_table, spread_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--n", type=int, default=N_POINTS)
    parser.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    parser.add_argument("--max-epochs", type=int, default=CFG_COMMON["max_epochs"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: --n 300 --max-epochs 3, a single seed, all three candidates. "
            "Exercises every code path -- fixture, all three training loops, the shared "
            "read-out-matrix path, both tables -- in well under two minutes. This is the "
            "runner's automated <verify>."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.n = 300
        args.max_epochs = 3
        args.seeds = args.seeds[:1]

    CFG_COMMON["max_epochs"] = args.max_epochs
    CFG_CAE["max_epochs"] = args.max_epochs

    print("=" * 78)
    print("Phase 02.6 decoder-substrate persistent-homology screening runner")
    print(
        "Measures and prints only. Applies no bar of any kind and writes no artifact -- "
        "the phase's ranking is assembled from this table by a later plan, never here."
    )
    print(
        f"candidates={args.candidates}  seeds={args.seeds}  n={args.n}  "
        f"fixture_seed={args.fixture_seed}  max_epochs={args.max_epochs}  "
        f"prescale_clouds={PRESCALE_CLOUDS}"
    )
    print("=" * 78)

    fx = build_fixture(args.n, args.fixture_seed)
    n = fx["x_all"].shape[0]

    # --- reference provenance, printed once before any candidate row -----------------------
    # Reproduces 02.6-SCREENING-RULE-02.md's own illustrative recipe exactly when defaults
    # are used (n=3000, fixture_seed=20260807): a 600-point held-out subsample drawn by
    # permuting on the FIXTURE seed, never a torch/split seed, and with NO pre-scaling
    # (matching notebooks/pu_manifold/tests/test_persistence_probe.py's own pinned
    # `standard_references` fixture) -- so these printed numbers are directly comparable to
    # the ratified document's own recorded figures.
    n_holdout = int(n * HOLDOUT_FRACTION)
    provenance_idx = np.random.default_rng(args.fixture_seed).permutation(n)[:n_holdout]
    intrinsic_ref = fx["intrinsic_ref"][provenance_idx]
    ambient_ref = fx["X"][provenance_idx]

    print()
    print(
        f"Reference provenance (fixture-seed subsample, n_holdout={n_holdout}, "
        "prescale=False -- matches the ratified document's own illustrative recipe):"
    )
    ref_diagrams = {}
    for name, cloud in (("intrinsic", intrinsic_ref), ("ambient", ambient_ref)):
        D, scale = persistence_probe.cloud_distance_matrix(cloud, prescale=False)
        dgms = persistence_probe.persistence_diagram(D)
        ref_diagrams[name] = dgms
        diameter = float(D.max())
        for k, degree in enumerate(_DEGREES):
            dgm = dgms[k]
            top_life = persistence_probe.max_persistence(dgm)
            print(
                f"  {name} {degree}: n_features={dgm.shape[0]}  top_life={top_life:.4f}  "
                f"own_diameter={diameter:.4f}  ref_max_persistence(denominator)={top_life:.4f}  "
                f"applied_scale={scale:.4f}"
            )

    h1_agreement = persistence_probe.ph_agreement(ref_diagrams["intrinsic"][1], ref_diagrams["ambient"][1])
    ambient_h1_saturation = persistence_probe.saturation_value(ref_diagrams["ambient"][1])
    print(
        f"  between-references H1 bottleneck: {h1_agreement['bottleneck']:.5f}  "
        f"ambient-reference own saturation value: {ambient_h1_saturation:.5f}  "
        f"saturated={h1_agreement['saturated']}"
    )
    print(
        "  Citing the 'Bottleneck saturation' travelling caveat by name "
        "(02.6-SCREENING-RULE-02.md): the (H1, ambient) bottleneck cell is expected to sit "
        "at this same value for any candidate whose latent has correctly unrolled the roll "
        "-- a saturated cell is reported with its flag and is never read as a measurement."
    )

    all_rows = []
    for candidate in args.candidates:
        print()
        print("=" * 78)
        print(f"CANDIDATE: {candidate}")

        def _provenance_line(candidate: str) -> str:
            """The architectural asymmetry, printed rather than hidden: the plain-AE and
            TopoAE arms share one architecture and one config; the CAE is a different
            architecture family with its own."""
            if candidate == "cae":
                return (
                    f"  architecture: cae.ChartAutoEncoder(in_dim=3, embed_dim={CAE_EMBED_DIM}, "
                    f"chart_dim={LATENT_DIM}, n_charts={CAE_N_CHARTS}, hidden={CAE_HIDDEN}, "
                    f"activation='silu')  latent_dim(global embedding)={CAE_EMBED_DIM}  "
                    f"config={CFG_CAE}"
                )
            extra = (
                f"  lambda_topo={LAMBDA_TOPO} warmup_frac=0.25 ramp_frac=0.25"
                if candidate == "topoae"
                else ""
            )
            return (
                f"  architecture: cae.PlainAutoEncoder(in_dim=3, latent_dim={LATENT_DIM}, "
                f"hidden={HIDDEN}, activation='silu')  latent_dim={LATENT_DIM}  "
                f"config={CFG_COMMON}{extra}"
            )

        print(_provenance_line(candidate))
        print("=" * 78)
        for seed in args.seeds:
            print(f"  [seed={seed}] training...")
            row = screen_seed(candidate, fx, seed, PRESCALE_CLOUDS)
            all_rows.append(row)
            extra = (
                f"  n_charts_surviving={row['n_charts_surviving']}/{row['n_charts_initial']}"
                if candidate == "cae"
                else ""
            )
            print(
                f"  [seed={seed}] wallclock_s={row['wallclock_s']:.1f}  "
                f"epochs_run={row['epochs_run']}  early_stopped={row['early_stopped']}  "
                f"recon_rel_err={row['recon_rel_err']:.4f}{extra}"
            )

    measurement_table, spread_table = markdown_table(all_rows)

    print()
    print("=" * 78)
    print("Measurement table -- one row per (candidate, seed, space, reference):")
    print("=" * 78)
    print(measurement_table)

    print()
    print("=" * 78)
    print(
        "Spread table -- mean / median / SPREAD (max - min) over the seeds actually run, "
        "one row per (candidate, space, reference, degree, distance):"
    )
    print("=" * 78)
    print(spread_table)

    print()
    print("=" * 78)
    print(
        "Measurements only. This runner applies no bar of any kind and writes no artifact "
        "-- the phase's ranking is assembled from this table by a later plan, never here."
    )
    print("=" * 78)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
