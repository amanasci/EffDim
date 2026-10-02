"""Embed the QM9 molecule table with one SMILES encoder (pod GPU, outside the job queue).

float32; masked mean of the last hidden state over the attention mask (special tokens included, as the tokenizer adds
them); right padding; pad = eos where the tokenizer has none; pinned model revisions; deterministic algorithms.
Gates: N finite non-zero rows; the first 2,048 molecules embedded twice byte-identical; one molecule embedded alone and
beside a longer one within 1e-5 x its norm (else batch size 1, recorded, and the check repeated).

Usage:
    python -m sweep.embed --encoder chemberta_5m_mtr --manifest molecules.yaml --out-dir <root>/hf \\
        --cache-dir /tmp/effdim-hf --batch-size 256 --device cuda
    python -m sweep.embed --pin <dir of sidecar json> --manifest molecules.yaml
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch initialises CUDA

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sweep.manifest import MOLECULES_PATH, Encoder, Manifest, load_manifest, update_manifest  # noqa: E402

POOLING = "masked mean of last_hidden_state over the attention mask (special tokens included)"
DETERMINISM_ROWS = 2048
INVARIANCE_TOL = 1e-5
REMOTE_CODE = {"molformer_xl": {"deterministic_eval": True}}     # name -> extra from_pretrained kwargs (trust_remote_code)
# name -> BOS id prepended to every input. Empty: ChemFM pretrains on SMILES + eos with no BOS (TheLuoFengLab/ChemFM
# @ee35b23, pretraining/lit_gpt/tokenizer.py encode(bos=False) as called by phrase_datasets/pretrain/*/tokenize_data*.py).
PREPEND_BOS: Dict[str, int] = {}
PARAMS_FROM_WEIGHTS = ("chemberta_5m_mtr", "chemberta_10m_mtr", "chemberta_77m_mtr", "chemberta_10m_mlm", "chemberta_77m_mlm")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smiles_sha256(smiles: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(smiles).encode()).hexdigest()


def prepare_tokenizer(tok) -> bool:
    """Right padding; pad = eos when the tokenizer has no pad token. Returns whether pad was set to eos."""
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        return True
    return False


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def embed_smiles(model, tok, smiles: Sequence[str], batch_size: int, device: str, bos_id: Optional[int] = None) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, len(smiles), batch_size):
        batch = tok(list(smiles[i:i + batch_size]), padding=True, return_tensors="pt")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if bos_id is not None:   # the tokenizer adds no BOS but the model was trained with one (ChemFM, Step 7)
            ids = torch.cat([torch.full((ids.shape[0], 1), bos_id, dtype=ids.dtype), ids], dim=1)
            mask = torch.cat([torch.ones((mask.shape[0], 1), dtype=mask.dtype), mask], dim=1)
        ids, mask = ids.to(device), mask.to(device)
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        out.append(masked_mean(hidden.float(), mask).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def validate_embeddings(E: np.ndarray, n_rows: int, dim: int) -> None:
    if E.shape != (n_rows, dim):
        raise ValueError(f"embeddings shape {E.shape}, expected ({n_rows}, {dim})")
    bad = ~np.isfinite(E).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows with a non-finite value (first {np.flatnonzero(bad)[:5].tolist()})")
    zero = np.flatnonzero(~E.any(axis=1))
    if zero.size:
        raise ValueError(f"{zero.size} all-zero rows (first {zero[:5].tolist()})")


def batch_invariance(model, tok, short: str, long: str, batch_size: int, device: str, bos_id: Optional[int] = None) -> float:
    """max |alone - beside a longer molecule| / |alone| for one molecule."""
    alone = embed_smiles(model, tok, [short], 1, device, bos_id)[0]
    together = embed_smiles(model, tok, [long, short], batch_size, device, bos_id)[1]
    return float(np.max(np.abs(alone - together)) / max(float(np.linalg.norm(alone)), 1e-30))


def choose_batch_size(model, tok, short: str, long: str, batch_size: int, device: str,
                      bos_id: Optional[int] = None) -> Tuple[int, float, bool]:
    """(batch size, max relative difference, fell back). Batch size 1 when the check fails, and the check is repeated."""
    inv = batch_invariance(model, tok, short, long, batch_size, device, bos_id)
    if inv <= INVARIANCE_TOL:
        return batch_size, inv, False
    # At batch size 1 no molecule is ever padded beside another, so the repeat is satisfied by construction: it re-runs
    # the embedding path (and would catch a nondeterministic model), it is not a second test of invariance.
    inv1 = batch_invariance(model, tok, short, long, 1, device, bos_id)
    if inv1 > INVARIANCE_TOL:
        raise SystemExit(f"batch invariance fails even at batch size 1: {inv1:.3g}")
    return 1, inv1, True


PARAMS_CONVENTION = "unique tensors in the checkpoint weight files (tied weights once), *position_ids buffers excluded"


def checkpoint_param_count(snapshot: Path) -> int:
    """Parameters in the checkpoint's weight files under PARAMS_CONVENTION: *.safetensors when present (safetensors
    stores no shared tensors), else pytorch_model*.bin with tied tensors counted once by data_ptr()."""
    files = sorted(Path(snapshot).glob("*.safetensors"))
    if files:
        from safetensors import safe_open
        n = 0
        for f in files:
            with safe_open(str(f), framework="pt") as s:
                for k in s.keys():
                    if not k.endswith("position_ids"):
                        n += int(np.prod(s.get_slice(k).get_shape()))
        return n
    files = sorted(Path(snapshot).glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"no weight files in {snapshot}")
    seen, n = set(), 0
    for f in files:
        for k, t in torch.load(f, map_location="cpu", weights_only=True).items():
            if k.endswith("position_ids") or t.data_ptr() in seen:
                continue
            seen.add(t.data_ptr()); n += int(t.numel())
    return n


def load_model(enc: Encoder, cache_dir: Path, device: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    snap = Path(snapshot_download(enc.hf_id, revision=enc.revision, cache_dir=str(cache_dir)))
    extra = REMOTE_CODE.get(enc.name)
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=extra is not None)
    import transformers
    dt = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"     # 4.x under Task 0 ruling B
    model = AutoModel.from_pretrained(str(snap), trust_remote_code=extra is not None, **{dt: torch.float32}, **(extra or {}))
    model = model.float().eval().to(device)
    assert next(model.parameters()).dtype == torch.float32
    return snap, tok, model


def embed_encoder(enc: Encoder, m: Manifest, out_dir: Path, cache_dir: Path, batch_size: int, device: str) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import transformers
    if _sha256_file(m.label_table) != m.label_table_sha256:
        raise SystemExit(f"molecule table sha256 differs from the manifest: {m.label_table}")
    torch.use_deterministic_algorithms(True)
    smiles = pd.read_parquet(m.label_table, columns=["smiles_canonical"])["smiles_canonical"].tolist()
    bos = PREPEND_BOS.get(enc.name)
    snap, tok, model = load_model(enc, Path(cache_dir), device)
    pad_is_eos = prepare_tokenizer(tok)
    params = checkpoint_param_count(snap)
    short, long = min(smiles, key=len), max(smiles, key=len)
    batch_size, inv, fallback = choose_batch_size(model, tok, short, long, batch_size, device, bos)
    print(f"[embed] {enc.name}: batch invariance {inv:.3g} at batch size {batch_size}" + (" (fallback)" if fallback else ""), flush=True)
    a = embed_smiles(model, tok, smiles[:DETERMINISM_ROWS], batch_size, device, bos)
    b = embed_smiles(model, tok, smiles[:DETERMINISM_ROWS], batch_size, device, bos)
    if a.tobytes() != b.tobytes():
        raise SystemExit(f"determinism check FAILED for {enc.name}: max |diff| {float(np.max(np.abs(a - b))):.3g}")
    E = embed_smiles(model, tok, smiles, batch_size, device, bos)
    validate_embeddings(E, m.n_rows, enc.dim)
    # the full run batches its first rows exactly as the check did when the batch size divides DETERMINISM_ROWS
    # (256, 64 and 1 all do), so those rows must match the checked ones byte for byte
    if DETERMINISM_ROWS % batch_size == 0 and E[:DETERMINISM_ROWS].tobytes() != a.tobytes():
        raise SystemExit(f"determinism check FAILED for {enc.name}: the full run's first {DETERMINISM_ROWS} rows differ from the checked rows")
    dest = Path(out_dir) / enc.parquet_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    n, D = E.shape
    col = pa.ListArray.from_arrays(pa.array(np.arange(0, n * D + 1, D, dtype=np.int32)), pa.array(E.ravel()))
    pq.write_table(pa.table({enc.column: col}), dest)
    side = {"encoder": enc.name, "hf_id": enc.hf_id, "revision": enc.revision, "transformers": transformers.__version__,
            "torch": torch.__version__, "dtype": "float32", "pooling": POOLING, "padding_side": "right",
            "pad_is_eos": pad_is_eos, "prepend_bos_id": bos, "batch_size": batch_size,
            "batch_invariance_fallback": fallback, "batch_invariance_max_rel_diff": inv,
            "determinism_rows": DETERMINISM_ROWS, "smiles_sha256": smiles_sha256(smiles),
            "label_table_sha256": m.label_table_sha256, "n_rows": int(n), "dim": int(D), "checkpoint_params": int(params),
            "device": device, "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "parquet_sha256": _sha256_file(dest)}
    dest.with_suffix(".json").write_text(json.dumps(side, indent=1, sort_keys=True) + "\n")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # the whole cache dir: huggingface_hub keeps shared blobs at <cache_dir>/blobs, outside models--<id>
    shutil.rmtree(Path(cache_dir), ignore_errors=True)
    return side


def pin(manifest_path, sidecar_dir) -> None:
    """Copy parquet_sha256 (all encoders) and checkpoint_params (ChemBERTa-2) from the sidecars into the manifest."""
    m = load_manifest(manifest_path)
    updates: Dict[str, dict] = {}
    for e in m.encoders:
        side = json.loads((Path(sidecar_dir) / f"{e.name}.json").read_text())
        kv = {"parquet_sha256": side["parquet_sha256"]}
        if e.name in PARAMS_FROM_WEIGHTS:
            kv.update(params=int(side["checkpoint_params"]),
                      params_source=f"computed from the checkpoint weights of {e.hf_id}@{e.revision}: {PARAMS_CONVENTION} (sweep/embed.py)")
        updates[e.name] = kv
    update_manifest(manifest_path, encoders=updates)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=MOLECULES_PATH)
    ap.add_argument("--encoder")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/effdim-hf"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pin", type=Path, default=None, help="copy the sidecars' sha256s (and ChemBERTa-2 params) into the manifest")
    a = ap.parse_args()
    if a.pin is not None:
        pin(a.manifest, a.pin)
        print(f"pinned {a.manifest}")
        return
    m = load_manifest(a.manifest)
    enc = next((e for e in m.encoders if e.name == a.encoder), None)
    if enc is None:
        raise SystemExit(f"unknown encoder {a.encoder!r}")
    side = embed_encoder(enc, m, a.out_dir, a.cache_dir, a.batch_size, a.device)
    print(f"EMBED_DONE {enc.name} rows {side['n_rows']} dim {side['dim']} batch_size {side['batch_size']} "
          f"fallback {side['batch_invariance_fallback']} sha256 {side['parquet_sha256']}", flush=True)


if __name__ == "__main__":
    main()
