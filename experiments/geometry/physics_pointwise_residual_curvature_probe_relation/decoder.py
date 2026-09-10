"""Train or reuse PlainAutoEncoder under the reproduction protocol. Label-blind."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_NB = Path(__file__).resolve().parents[3] / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold import cae

from .config import AE_ACTIVATION, AE_HIDDEN, D_LAT, DECODER_SEEDS, MAX_EPOCHS, TRAIN_CFG_TEMPLATE
from .io_util import file_meta, write_json


def train_cfg_for_seed(seed: int) -> dict[str, Any]:
    cfg = dict(TRAIN_CFG_TEMPLATE)
    cfg["seed"] = int(seed)
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["early_stop_patience"] = MAX_EPOCHS + 1
    return cfg


def compatible_checkpoint(path: Path, *, in_dim: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        meta_p = path.with_suffix(".json")
        if not meta_p.exists():
            return False
        meta = json.loads(meta_p.read_text())
        return (
            int(meta.get("seed", -1)) == int(seed)
            and int(meta.get("latent_dim", -1)) == D_LAT
            and int(meta.get("in_dim", -1)) == int(in_dim)
            and tuple(meta.get("hidden", ())) == tuple(AE_HIDDEN)
            and str(meta.get("activation", "")) == AE_ACTIVATION
            and int(meta.get("max_epochs", -1)) == MAX_EPOCHS
            and bool(meta.get("anchors_excluded", False))
            and str(meta.get("protocol", "")) == "reproduction_plain_ae_400"
        )
    except Exception:
        return False


def build_model(in_dim: int, seed: int) -> cae.PlainAutoEncoder:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return cae.PlainAutoEncoder(
        in_dim=in_dim, latent_dim=D_LAT, hidden=AE_HIDDEN, activation=AE_ACTIVATION
    )


def train_one(
    X_train: np.ndarray,
    *,
    seed: int,
    device: str,
    ckpt_dir: Path,
) -> dict[str, Any]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"vit_base_plain_ae_seed{int(seed)}.npz"
    meta_p = ckpt.with_suffix(".json")
    in_dim = int(X_train.shape[1])
    if compatible_checkpoint(ckpt, in_dim=in_dim, seed=seed):
        model = build_model(in_dim, seed)
        with np.load(ckpt) as z:
            arrays = {k: z[k] for k in z.files}
        model.load_state_dict(cae.arrays_to_state_dict(arrays, model.state_dict()))
        model.eval().double()
        meta = json.loads(meta_p.read_text())
        meta["reused"] = True
        return {"model": model, "meta": meta, "ckpt": str(ckpt), "reused": True}

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model(in_dim, seed)
    model.train().float().to(dev)
    x32 = torch.tensor(np.asarray(X_train, dtype=np.float32), device=dev)
    cfg = train_cfg_for_seed(seed)
    t0 = time.time()
    info = cae.train_plain_ae(model, x32, cfg)
    t_train = time.time() - t0
    model.cpu().eval().double()
    arrays = cae.state_dict_to_arrays(model.state_dict())
    np.savez(ckpt, **arrays)
    meta = {
        "seed": int(seed),
        "in_dim": in_dim,
        "latent_dim": D_LAT,
        "hidden": list(AE_HIDDEN),
        "activation": AE_ACTIVATION,
        "max_epochs": MAX_EPOCHS,
        "cfg": cfg,
        "protocol": "reproduction_plain_ae_400",
        "anchors_excluded": True,
        "n_train": int(X_train.shape[0]),
        "device": str(dev),
        "wallclock_s": float(t_train),
        "epochs_run": int(info.get("epochs_run", MAX_EPOCHS)),
        "early_stopped": bool(info.get("early_stopped", False)),
        "final_recon": float(info["history"][-1]["recon"]) if info.get("history") else float("nan"),
        "reused": False,
        "label_blind": True,
        "checkpoint": str(ckpt),
        "weights": file_meta(ckpt),
    }
    write_json(meta_p, meta, force=True)
    return {"model": model, "meta": meta, "ckpt": str(ckpt), "reused": False, "info": info}


def encode_decode(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(np.asarray(X, dtype=np.float64))
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)
    return z.cpu().numpy(), y.cpu().numpy()


def recon_row(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum((X - Y) ** 2, axis=1)
