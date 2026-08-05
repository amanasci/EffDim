"""The expensive check named in 02.1-AMENDMENT-02.md SS6.4: does the pseudo-Euclidean
advantage survive a Euclidean-latent decoder?

signature_transfer measured distance distortion with the signature discarded -- a cheap
proxy. This trains actual decoders and measures held-out reconstruction, which is what
Phase 3 would care about. Every decision rule is fixed in
02.1-DECODER-PREREGISTRATION.md (e0861a1), committed before this script ran.

Four arms, one protocol, only the input coordinates differ. Both raw and standardized
preprocessing are run; if they disagree on a rule, NO rule fires (the pre-registration
says that disagreement is itself the finding).

This is NOT Phase 3 work and satisfies none of DEC-01..05. It trains decoders to decide a
representation question and nothing more.

Precondition: the cached eigenpairs and subsample must exist (gitignored, irreproducible
here) -- halts rather than regenerating, which would change provenance.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/decoder_transfer_check.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pu_manifold import geometry_probes as gp
from pu_manifold.cache import cache_path, json_cache

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"
SUBSAMPLE = "subsample_20260729_a79b3460b838fd0a"

# --- pre-registered constants (02.1-DECODER-PREREGISTRATION.md, "Protocol") ----------------
HIDDEN = (256, 256)
EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3
SPLIT_FRACTION = 0.8
SPLIT_SEED = 20260805
TORCH_SEED = 20260805
# "## Decision rules": the reduction_fraction the distance ladder measured for (40,25)
PROMISED = 0.183742
TIE_BAND = 0.01
# 02.1-PREREGISTRATION.md pair sample, reused verbatim
PAIR_COUNT, PAIR_SEED = 200_000, 20260731

# --- pre-registered arms ("## Arms"); D is control-only, not a candidate -------------------
ARMS = {
    "A_krein_40_25": {"p": 40, "q": 25, "role": "contested winner; the literal SS6.4 check"},
    "B_classical_18_0": {"p": 18, "q": 0, "role": "q=0 best -- the flat floor Phase 2 gated FAIL"},
    "C_classical_40_0": {"p": 40, "q": 0, "role": "width-matched partner for D"},
    "D_krein_20_20": {"p": 20, "q": 20, "role": "CONTROL ONLY -- not on the ladder, not a candidate"},
}

print("=" * 78)
print(f"Decoder-side check -- fit_key = {FIT_KEY}")
print("rules fixed in 02.1-DECODER-PREREGISTRATION.md before this ran")
print("=" * 78)
t_start = time.time()

# --- inputs: cached eigenpairs only. No new fit, no new eigensolve. ------------------------
_spec = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")
_bot = np.load(f"{CACHE}/krein_bottom_{FIT_KEY}.npz")
_sub = np.load(f"{CACHE}/{SUBSAMPLE}.npz")

eigvals_top, eigvecs_top = _spec["eigvals_top"], _spec["eigvecs_top"]       # descending
eigvals_bottom, eigvecs_bottom = _bot["eigvals_bottom"], _bot["eigvecs_bottom"]  # ascending
# geo_pairs_r2 holds geodesic DISTANCES (dist_matrix_[rows, cols]); the statistic is on
# squared distances, so square them exactly as geometry_probes_run.py does.
d2_geo = _spec["geo_pairs_r2"].astype(np.float64) ** 2
Y = np.asarray(_sub["legacysurvey"], dtype=np.float64)                      # the fit's target
n = Y.shape[0]

assert np.all(np.diff(eigvals_top) <= 0), "eigvals_top must be descending"
assert np.all(np.diff(eigvals_bottom) >= 0), "eigvals_bottom must be ascending (most negative first)"
assert eigvecs_top.shape == (n, 40) and eigvecs_bottom.shape == (n, 40)
print(f"  target {Y.shape}  norms {np.linalg.norm(Y, axis=1).mean():.4f} "
      f"+/- {np.linalg.norm(Y, axis=1).std():.2e}")


def build_coords(p: int, q: int):
    """X = eigvecs * sqrt(|eigvals|), positive block then negative block. The decoder reads
    these as ordinary real coordinates under the quadratic form -- which is the point."""
    blocks, vals = [], []
    if p:
        blocks.append(eigvecs_top[:, :p] * np.sqrt(np.abs(eigvals_top[:p]))[None, :])
        vals.append(eigvals_top[:p])
    if q:
        blocks.append(eigvecs_bottom[:, :q] * np.sqrt(np.abs(eigvals_bottom[:q]))[None, :])
        vals.append(eigvals_bottom[:q])
    return np.concatenate(blocks, axis=1), np.concatenate(vals)


# --- distance distortion per arm, same statistic and same fixed pair sample ----------------
rng = np.random.default_rng(PAIR_SEED)
rows, cols = gp.draw_geo_pairs(rng, n, PAIR_COUNT)
assert d2_geo.shape == rows.shape, "cached geo_pairs_r2 must align with the regenerated pairs"

# Alignment is not provable here without the 1.55 GiB joblib, so prove it by reproducing the
# three published ladder rungs instead. A misaligned pair sample cannot hit all three.
_PUBLISHED_RUNGS = {(40, 25): 0.065190, (18, 0): 0.079864, (40, 0): 0.179641}
for (_p, _q), _expected in _PUBLISHED_RUNGS.items():
    _vals = np.concatenate([eigvals_top[:_p]] + ([eigvals_bottom[:_q]] if _q else []))
    _vecs = np.concatenate([eigvecs_top[:, :_p]] + ([eigvecs_bottom[:, :_q]] if _q else []), axis=1)
    _got = gp.distortion_stats(
        gp.pseudo_euclidean_sq_distances(_vals, _vecs, rows, cols), d2_geo
    )["median_abs_rel"]
    assert abs(_got - _expected) < 1e-5, (
        f"rung (p={_p}, q={_q}) gives {_got:.6f}, published {_expected:.6f} -- the pair sample "
        f"or the coordinate construction does not match 02.1-03's. Halting: every distortion "
        f"number below would be meaningless."
    )
print(f"  pair sample + coordinate construction verified against all "
      f"{len(_PUBLISHED_RUNGS)} published ladder rungs")

# --- split, fixed seed, identical for every arm --------------------------------------------
perm = np.random.default_rng(SPLIT_SEED).permutation(n)
n_train = int(SPLIT_FRACTION * n)
train_idx, hold_idx = perm[:n_train], perm[n_train:]
print(f"  split {len(train_idx)} train / {len(hold_idx)} held out  (seed {SPLIT_SEED})")


class Decoder(nn.Module):
    """input_dim -> 256 -> 256 -> 768, tanh. C2-smooth: DEC-02 forbids ReLU-family, whose
    second derivative is identically zero."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        h1, h2 = HIDDEN
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_and_score(X: np.ndarray, tag: str) -> float:
    """One arm, one preprocessing. Every hyperparameter is pre-registered; nothing is tuned."""
    torch.manual_seed(TORCH_SEED)  # identical initialisation draw for every arm
    x_all = torch.tensor(X, dtype=torch.float32)
    y_all = torch.tensor(Y, dtype=torch.float32)
    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_ho, y_ho = x_all[hold_idx], y_all[hold_idx]

    model = Decoder(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n_tr = len(x_tr)

    for epoch in range(EPOCHS):
        order = torch.randperm(n_tr)
        for start in range(0, n_tr, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        mse_per_dim = float(((model(x_ho) - y_ho) ** 2).mean())
    print(f"    {tag:34s} mse_per_dim = {mse_per_dim:.8f}   ({time.time() - t_start:.0f}s)")
    return mse_per_dim


results = {}
for name, spec in ARMS.items():
    p, q = spec["p"], spec["q"]
    X, vals_sel = build_coords(p, q)

    # distance distortion under the arm's own signed form, for the record
    if q:
        vecs_sel = np.concatenate(
            [eigvecs_top[:, :p], eigvecs_bottom[:, :q]], axis=1
        )
    else:
        vecs_sel = eigvecs_top[:, :p]
    d2_rep = gp.pseudo_euclidean_sq_distances(vals_sel, vecs_sel, rows, cols)
    dist = gp.distortion_stats(d2_rep, d2_geo)

    print(f"\n  {name}  (p={p}, q={q}, {X.shape[1]} dims) -- {spec['role']}")
    print(f"    distance distortion median_abs_rel = {dist['median_abs_rel']:.6f}")

    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    results[name] = {
        "p": p, "q": q, "input_dims": int(X.shape[1]), "role": spec["role"],
        "distance_distortion_median_abs_rel": dist["median_abs_rel"],
        "distance_distortion_median_signed_rel": dist["median_signed_rel"],
        "mse_per_dim_standardized": train_and_score(X_std, "standardized (primary)"),
        "mse_per_dim_raw": train_and_score(X, "raw (robustness)"),
    }


def rel_improvement(better: str, worse: str, key: str) -> float:
    """Positive when `better` reconstructs better than `worse`."""
    b, w = results[better][key], results[worse][key]
    return (w - b) / w


# =============================================================================================
print("\n" + "=" * 78)
print("RESULTS")
print("=" * 78)
print(f"{'arm':20s}{'dims':>6s}{'dist_distortion':>18s}{'mse (std)':>14s}{'mse (raw)':>14s}")
for name, r in results.items():
    print(f"{name:20s}{r['input_dims']:>6d}{r['distance_distortion_median_abs_rel']:>18.6f}"
          f"{r['mse_per_dim_standardized']:>14.8f}{r['mse_per_dim_raw']:>14.8f}")

RI_std = rel_improvement("A_krein_40_25", "B_classical_18_0", "mse_per_dim_standardized")
RI_raw = rel_improvement("A_krein_40_25", "B_classical_18_0", "mse_per_dim_raw")
CTRL_std = rel_improvement("D_krein_20_20", "C_classical_40_0", "mse_per_dim_standardized")
CTRL_raw = rel_improvement("D_krein_20_20", "C_classical_40_0", "mse_per_dim_raw")

print(f"\nRI  = relative improvement of A (Krein 40,25) over B (classical 18,0)")
print(f"  standardized (primary) : {RI_std:+.6f}")
print(f"  raw (robustness)       : {RI_raw:+.6f}")
print(f"  PROMISED by the distance ladder: {PROMISED:+.6f}")
print(f"\nCTRL = relative improvement of D (Krein 20,20) over C (classical 40,0), matched width")
print(f"  standardized : {CTRL_std:+.6f}")
print(f"  raw          : {CTRL_raw:+.6f}")


def classify(ri: float) -> str:
    if ri >= PROMISED - TIE_BAND:
        return "RULE_1_RETRACTION"
    if ri > TIE_BAND:
        return "RULE_2_PARTIAL"
    return "RULE_3_CONFIRMATION"


verdict_std, verdict_raw = classify(RI_std), classify(RI_raw)
agree = verdict_std == verdict_raw
VERDICT = verdict_std if agree else "INCONCLUSIVE_PREPROCESSING_DISAGREEMENT"

print(f"\n  standardized -> {verdict_std}")
print(f"  raw          -> {verdict_raw}")
print(f"  agree        -> {agree}")
print(f"\nVERDICT: {VERDICT}")

CTRL_POSITIVE = bool(CTRL_std > TIE_BAND and CTRL_raw > TIE_BAND)
print(f"RULE_4 signature control positive (both preprocessings): {CTRL_POSITIVE}")
print("  (must be reported in 02.1-RECOMMENDATION.md even if Rule 3 fires)")

_cfg = {
    "fit_key": FIT_KEY, "epochs": EPOCHS, "hidden": list(HIDDEN), "lr": LR,
    "batch_size": BATCH_SIZE, "split_seed": SPLIT_SEED, "torch_seed": TORCH_SEED,
    "promised": PROMISED, "preregistration": "02.1-DECODER-PREREGISTRATION.md",
}
RECORD = json_cache(
    f"decoder_transfer_{FIT_KEY}", _cfg,
    lambda: {
        "fit_key": FIT_KEY,
        "preregistration": "02.1-DECODER-PREREGISTRATION.md",
        "arms": results,
        "RI_standardized": RI_std,
        "RI_raw": RI_raw,
        "promised": PROMISED,
        "control_standardized": CTRL_std,
        "control_raw": CTRL_raw,
        "verdict_standardized": verdict_std,
        "verdict_raw": verdict_raw,
        "preprocessings_agree": agree,
        "verdict": VERDICT,
        "rule_4_control_positive": CTRL_POSITIVE,
        "protocol": {
            "architecture": f"input -> {HIDDEN[0]} -> {HIDDEN[1]} -> 768",
            "activation": "tanh (C2-smooth; DEC-02 forbids ReLU-family)",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "split": f"{SPLIT_FRACTION:.0%} train / {1 - SPLIT_FRACTION:.0%} held out",
            "split_seed": SPLIT_SEED, "torch_seed": TORCH_SEED,
        },
        "scope": (
            "NOT Phase 3 work; satisfies none of DEC-01..05. Trains decoders to decide a "
            "representation question only. Arm D is control-only and no adoption decision "
            "may be made from it."
        ),
    },
)
print(f"\nwrote {cache_path(f'decoder_transfer_{FIT_KEY}', 'json').name}  "
      f"({time.time() - t_start:.0f}s total)")
