"""Phase 9 production-runner helpers, loaded by the kept `09_*` runners as the module `runner`.

It provides `fit_and_field_at_anchors` (autoencoder fit, then the sphere-projected decoder's
curvature field at the anchor latent codes), `SphereProjectedDecoder` (Amendment 01's decoder map
`F(z) / ||F(z)||`), `_oof_predictions_for_label` (out-of-fold ridge predictions on a label's finite
rows) and `_THREADS`. It has no CLI of its own. Importing it has side effects: it reads
`--threads` from `sys.argv` (default 8), sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `torch.set_num_threads` before any numerical work, and puts `notebooks/`
and `notebooks/diagnostics/` on `sys.path`.
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Returns the string value passed for `flag` in `argv`, accepting BOTH argparse-standard
    forms -- `--flag value` and `--flag=value` -- or `None` if `flag` was not passed in either
    form. Kept dependency-free so it can run here, above the torch import."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap MUST be set before any import pulling in torch/numpy (07_crossmodal_curvature_run.py
# precedent: concurrent torch jobs measured driving load up ~10x).
_THREADS = 8
_threads_arg = _flag_value_from_argv("--threads", sys.argv)
if _threads_arg is not None:
    try:
        _THREADS = int(_threads_arg)
    except ValueError:
        pass
os.environ["OMP_NUM_THREADS"] = str(_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_THREADS)

import time
from pathlib import Path
from typing import Any, Dict

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_THREADS)

from pu_manifold import cae  # noqa: E402
from pu_manifold import crossmodal_curvature  # noqa: E402
from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402


def _oof_predictions_for_label(
    X: np.ndarray, y_full: np.ndarray, alpha: float, n_folds: int, fold_seed: int
) -> np.ndarray:
    """`pcp.oof_ridge_predictions` requires every row of `y` to be finite (its own structural
    out-of-fold proof guard); a Physics label may carry sentinel-masked `NaN` rows (`photo_z`
    and `stellar_mass` are not 100% populated -- 09-DATA-MANIFEST.md). Fits and predicts OOF
    only on the finite subset of rows, scattering the result back into a full-length array with
    `NaN` at every non-finite row -- never widening the fold structure to hold out a row with no
    real label value. Returns an all-`NaN` array, rather than raising, when no row is finite."""
    y_full = np.asarray(y_full, dtype=np.float64).ravel()
    finite = np.isfinite(y_full)
    y_hat_full = np.full(y_full.shape[0], np.nan, dtype=np.float64)
    if not np.any(finite):
        return y_hat_full
    y_hat_full[finite] = pcp.oof_ridge_predictions(
        X[finite], y_full[finite], alpha=alpha, n_folds=n_folds, fold_seed=fold_seed
    )
    return y_hat_full


class SphereProjectedDecoder(torch.nn.Module):
    """Amendment 01 (`pcp.DECODER_IMAGE_PROJECTION == "sphere"`): the decoder map whose
    curvature is differentiated is `F(z) / ||F(z)||`, so the decoder image lies in the unit
    sphere the L2-normalised data occupy and `H_rad == -d` identically. `.decoder` is the wrapped
    model's own decoder so `decoder_curvature.assert_c2_decoder` inspects the real activation
    modules; the parameters are the wrapped model's own, so `_assert_float64` sees the real dtype
    (construct this AFTER `model.eval().double()`)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.decoder = model.decoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        F = self.model.decode(z)
        return F / torch.linalg.norm(F, dim=-1, keepdim=True)


def fit_and_field_at_anchors(
    X: np.ndarray,
    d: int,
    anchor_idx: np.ndarray,
    in_dim: int,
    hidden: Any,
    activation: str,
    train_cfg: Dict[str, Any],
    max_epochs: int,
    torch_init_seed: int,
    split_seed: int,
    holdout_fraction: float,
) -> Dict[str, Any]:
    """`08_radial_curvature_decomposition_run.py`'s `fit_and_decompose` path, except
    `plain_decoder_curvature` and `model.decode` are evaluated at the ANCHOR latent codes only,
    never at all rows -- D9-04's deliberate departure from Phase 7's `FIELD_EVALUATED_ON`
    convention. Passes the MODEL to `plain_decoder_curvature`, never a bound method, so its
    float64 guard actually runs. Returns `H_vec`, `image`, `metric_condition_number`,
    `var_explained` and the two wallclocks."""
    torch.manual_seed(torch_init_seed)
    model = cae.PlainAutoEncoder(in_dim=in_dim, latent_dim=d, hidden=hidden, activation=activation)

    train_idx, holdout_idx = crossmodal_curvature.split_indices(X.shape[0], split_seed, holdout_fraction)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]

    cfg = dict(train_cfg)
    cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x_train32, cfg)
    wallclock_fit_s = time.monotonic() - t0

    model.eval().double()
    # Amendment 01: curvature (and the image handed to decompose_radial_tangential) is taken on
    # the sphere-projected decoder F/||F||. The fit, encoder, var_explained and anchor codes are
    # untouched. Built after .double() so the wrapper shares the float64 weights.
    curvature_model: torch.nn.Module = model
    if pcp.DECODER_IMAGE_PROJECTION == "sphere":
        curvature_model = SphereProjectedDecoder(model).eval()
    anchor_idx_t = torch.as_tensor(np.asarray(anchor_idx), dtype=torch.long)
    x_anchor64 = x64[anchor_idx_t]
    with torch.no_grad():
        z_anchor = model.encode(x_anchor64)
        y_holdout = model(x_holdout64)["y"]
        image = curvature_model.decode(z_anchor).detach().cpu().numpy()

    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    var_explained = 1.0 - recon["mse_total"] / sig

    t0 = time.monotonic()
    field = decoder_curvature.plain_decoder_curvature(curvature_model, z_anchor)
    wallclock_field_s = time.monotonic() - t0

    H_vec = field["H_vec"].detach().cpu().numpy()
    metric_condition_number = field["metric_condition_number"].detach().cpu().numpy()

    return {
        "H_vec": H_vec,
        "image": image,
        "metric_condition_number": metric_condition_number,
        "var_explained": float(var_explained),
        "wallclock_fit_s": wallclock_fit_s,
        "wallclock_field_s": wallclock_field_s,
    }
