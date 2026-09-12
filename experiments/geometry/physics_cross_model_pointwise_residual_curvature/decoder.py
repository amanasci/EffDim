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

from .config import AE_ACTIVATION, AE_HIDDEN, D_LAT, MAX_EPOCHS, PROTOCOL, SOURCE_PRCR, TRAIN_CFG_TEMPLATE
from .io_util import file_meta, platonic_root, resolve_path, write_json


def train_cfg_for_seed(seed: int) -> dict[str, Any]:
    cfg = dict(TRAIN_CFG_TEMPLATE)
    cfg["seed"] = int(seed)
    cfg["max_epochs"] = MAX_EPOCHS
    cfg["early_stop_patience"] = MAX_EPOCHS + 1
    return cfg


def ckpt_name(model: str, seed: int) -> str:
    return f"{model}_plain_ae_seed{int(seed)}.npz"


def compatible_checkpoint(path: Path, *, in_dim: int, seed: int, model: str) -> bool:
    if not path.exists():
        return False
    try:
        meta = json.loads(path.with_suffix(".json").read_text())
        return (
            int(meta.get("seed", -1)) == int(seed)
            and int(meta.get("latent_dim", -1)) == D_LAT
            and int(meta.get("in_dim", -1)) == int(in_dim)
            and tuple(meta.get("hidden", ())) == tuple(AE_HIDDEN)
            and str(meta.get("activation", "")) == AE_ACTIVATION
            and int(meta.get("max_epochs", -1)) == MAX_EPOCHS
            and bool(meta.get("anchors_excluded", False))
            and str(meta.get("protocol", "")) == PROTOCOL
            and str(meta.get("model", model)) == model
        )
    except Exception:
        return False


def candidate_ckpts(model: str, seed: int, ckpt_dir: Path, in_dim: int) -> list[Path]:
    names = [ckpt_dir / ckpt_name(model, seed)]
    if model == "vit_base":
        root = platonic_root()
        names.append(resolve_path(root, SOURCE_PRCR) / "checkpoints" / f"vit_base_plain_ae_seed{int(seed)}.npz")
    return names


def build_model(in_dim: int, seed: int) -> cae.PlainAutoEncoder:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return cae.PlainAutoEncoder(in_dim=in_dim, latent_dim=D_LAT, hidden=AE_HIDDEN, activation=AE_ACTIVATION)


def load_model(path: Path, *, in_dim: int, seed: int):
    model = build_model(in_dim, seed)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    model.load_state_dict(cae.arrays_to_state_dict(arrays, model.state_dict()))
    model.eval().double()
    return model


def train_one(
    X_train: np.ndarray,
    *,
    model_id: str,
    seed: int,
    device: str,
    ckpt_dir: Path,
) -> dict[str, Any]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    in_dim = int(X_train.shape[1])
    dest = ckpt_dir / ckpt_name(model_id, seed)
    for cand in candidate_ckpts(model_id, seed, ckpt_dir, in_dim):
        if compatible_checkpoint(cand, in_dim=in_dim, seed=seed, model=model_id) or (
            model_id == "vit_base" and compatible_vitb_legacy(cand, in_dim=in_dim, seed=seed)
        ):
            model = load_model(cand, in_dim=in_dim, seed=seed)
            meta_p = cand.with_suffix(".json")
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            trained_in_this_tree = bool(
                cand.resolve() == dest.resolve() and not bool(meta.get("reused", True))
            )
            meta["reused"] = True
            meta["reuse_path"] = str(cand)
            meta["model"] = model_id
            meta["trained_in_this_tree"] = trained_in_this_tree
            if cand.resolve() != dest.resolve():
                # copy weights into this tree without rewriting the preserved source
                arrays = cae.state_dict_to_arrays(model.state_dict())
                if not dest.exists():
                    np.savez(dest, **arrays)
                write_json(dest.with_suffix(".json"), {**meta, "checkpoint": str(dest)}, force=True)
            return {
                "model": model,
                "meta": meta,
                "ckpt": str(cand),
                "reused": True,
                "trained_in_this_tree": trained_in_this_tree,
            }

    requested = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    dev = torch.device(requested)
    model = build_model(in_dim, seed)
    cfg = train_cfg_for_seed(seed)
    t0 = time.time()
    try:
        model.train().float().to(dev)
        x32 = torch.tensor(np.asarray(X_train, dtype=np.float32), device=dev)
        info = cae.train_plain_ae(model, x32, cfg)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or requested == "cpu":
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dev = torch.device("cpu")
        model = build_model(in_dim, seed)
        model.train().float().to(dev)
        x32 = torch.tensor(np.asarray(X_train, dtype=np.float32), device=dev)
        info = cae.train_plain_ae(model, x32, cfg)
        cfg = {**cfg, "oom_fallback": "cpu"}
    t_train = time.time() - t0
    model.cpu().eval().double()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    arrays = cae.state_dict_to_arrays(model.state_dict())
    np.savez(dest, **arrays)
    meta = {
        "seed": int(seed),
        "model": model_id,
        "in_dim": in_dim,
        "latent_dim": D_LAT,
        "hidden": list(AE_HIDDEN),
        "activation": AE_ACTIVATION,
        "max_epochs": MAX_EPOCHS,
        "cfg": cfg,
        "protocol": PROTOCOL,
        "anchors_excluded": True,
        "n_train": int(X_train.shape[0]),
        "device": str(dev),
        "wallclock_s": float(t_train),
        "epochs_run": int(info.get("epochs_run", MAX_EPOCHS)),
        "early_stopped": bool(info.get("early_stopped", False)),
        "final_recon": float(info["history"][-1]["recon"]) if info.get("history") else float("nan"),
        "reused": False,
        "label_blind": True,
        "checkpoint": str(dest),
        "weights": file_meta(dest),
        "same_split_across_seeds": True,
    }
    write_json(dest.with_suffix(".json"), meta, force=True)
    return {"model": model, "meta": meta, "ckpt": str(dest), "reused": False, "info": info}


def compatible_vitb_legacy(path: Path, *, in_dim: int, seed: int) -> bool:
    """PRCR checkpoints omit model= in meta but are the reference protocol."""
    if not path.exists():
        return False
    try:
        meta = json.loads(path.with_suffix(".json").read_text())
        return (
            int(meta.get("seed", -1)) == int(seed)
            and int(meta.get("latent_dim", -1)) == D_LAT
            and int(meta.get("in_dim", -1)) == int(in_dim)
            and tuple(meta.get("hidden", ())) == tuple(AE_HIDDEN)
            and str(meta.get("activation", "")) == AE_ACTIVATION
            and int(meta.get("max_epochs", -1)) == MAX_EPOCHS
            and bool(meta.get("anchors_excluded", False))
            and str(meta.get("protocol", "")) == PROTOCOL
        )
    except Exception:
        return False


def encode_decode(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(np.asarray(X, dtype=np.float64))
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)
    return z.cpu().numpy(), y.cpu().numpy()


def recon_row(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum((X - Y) ** 2, axis=1)
