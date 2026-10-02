"""Embedding helpers: masked mean, tokenizer settings, validation, batch invariance, determinism, pinning."""
import json
import types

import numpy as np
import pytest
import torch

from sweep import embed
from sweep.manifest import MOLECULES_PATH, load_manifest


class StubTok:
    """Char-level stand-in for an HF tokenizer: <s> + chars + </s>, no pad token until prepare_tokenizer sets one."""
    def __init__(self, pad_token=None):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
        self.pad_token, self.eos_token, self.padding_side = pad_token, "</s>", "left"

    def _ids(self, s):
        return [1] + [3 + (ord(c) % 29) for c in s] + [2]

    def __call__(self, texts, padding=True, return_tensors="pt"):
        seqs = [self._ids(t) for t in texts]; L = max(map(len, seqs)); pad = self.vocab[self.pad_token]
        right = self.padding_side == "right"
        ids = [s + [pad] * (L - len(s)) if right else [pad] * (L - len(s)) + s for s in seqs]
        mask = [[1] * len(s) + [0] * (L - len(s)) if right else [0] * (L - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def _tiny_llama():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaModel
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64)
    return LlamaModel(cfg).eval()


class Leaky(torch.nn.Module):
    """Pools over the padding too: a molecule's output depends on what it is batched with."""
    def __init__(self):
        super().__init__(); torch.manual_seed(0); self.emb = torch.nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        return types.SimpleNamespace(last_hidden_state=h + h.mean(dim=1, keepdim=True))


class Recorder(torch.nn.Module):
    """Records the ids and mask it is called with."""
    def __init__(self):
        super().__init__(); self.emb = torch.nn.Embedding(32, 4); self.seen = []

    def forward(self, input_ids, attention_mask):
        self.seen.append((input_ids.clone(), attention_mask.clone()))
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))


def test_bos_is_prepended():
    model, tok = Recorder(), StubTok()
    embed.prepare_tokenizer(tok)
    embed.embed_smiles(model, tok, ["C", "CCO"], 2, "cpu", bos_id=7)
    ids, mask = model.seen[0]
    plain = tok(["C", "CCO"])
    assert ids[:, 0].tolist() == [7, 7] and mask[:, 0].tolist() == [1, 1]
    assert torch.equal(ids[:, 1:], plain["input_ids"]) and torch.equal(mask[:, 1:], plain["attention_mask"])


def test_checkpoint_param_count_dedupes_and_skips_buffers(tmp_path):
    w = torch.zeros(4, 5)
    torch.save({"emb.weight": w, "lm_head.weight": w, "emb.position_ids": torch.arange(10), "bias": torch.zeros(3)},
               tmp_path / "pytorch_model.bin")
    assert embed.checkpoint_param_count(tmp_path) == 23          # tied weight once, position_ids excluded


def test_masked_mean_ignores_padding():
    h = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    m = torch.tensor([[1, 1, 0], [1, 0, 0]])
    torch.testing.assert_close(embed.masked_mean(h, m), torch.tensor([[1.0, 2.0], [6.0, 7.0]]))


def test_prepare_tokenizer_right_padding_and_eos_pad():
    tok = StubTok()
    assert embed.prepare_tokenizer(tok) is True and tok.padding_side == "right" and tok.pad_token == "</s>"
    tok2 = StubTok(pad_token="<pad>")
    assert embed.prepare_tokenizer(tok2) is False and tok2.pad_token == "<pad>"


def test_validate_embeddings_rejects_rows_nan_and_zero():
    E = np.ones((4, 3), np.float32)
    embed.validate_embeddings(E, 4, 3)
    with pytest.raises(ValueError, match="shape"):
        embed.validate_embeddings(E, 5, 3)
    bad = E.copy(); bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        embed.validate_embeddings(bad, 4, 3)
    bad = E.copy(); bad[2] = 0.0
    with pytest.raises(ValueError, match="all-zero"):
        embed.validate_embeddings(bad, 4, 3)


def test_tiny_llama_is_batch_invariant_and_deterministic():
    model, tok = _tiny_llama(), StubTok()
    embed.prepare_tokenizer(tok)
    E = embed.embed_smiles(model, tok, ["C", "CCO", "c1ccccc1O"], 2, "cpu")
    assert E.shape == (3, 16) and E.dtype == np.float32
    assert E.tobytes() == embed.embed_smiles(model, tok, ["C", "CCO", "c1ccccc1O"], 2, "cpu").tobytes()
    assert embed.batch_invariance(model, tok, "CO", "CC(=O)OC1CCCCC1N", 8, "cpu") <= embed.INVARIANCE_TOL
    assert embed.choose_batch_size(model, tok, "CO", "CC(=O)OC1CCCCC1N", 8, "cpu")[0::2] == (8, False)


def test_choose_batch_size_falls_back_on_padding_leak():
    model, tok = Leaky().eval(), StubTok()
    embed.prepare_tokenizer(tok)
    assert embed.batch_invariance(model, tok, "CO", "CC(=O)OC1CCCCC1N", 8, "cpu") > embed.INVARIANCE_TOL
    assert embed.choose_batch_size(model, tok, "CO", "CC(=O)OC1CCCCC1N", 8, "cpu") == (1, 0.0, True)


def test_pin_copies_shas_and_chemberta_params(tmp_path):
    p = tmp_path / "m.yaml"; p.write_text(MOLECULES_PATH.read_text())
    side = tmp_path / "sidecars"; side.mkdir()
    for i, e in enumerate(load_manifest(p).encoders):
        (side / f"{e.name}.json").write_text(json.dumps({"parquet_sha256": f"{i:02d}" * 32, "checkpoint_params": 1000 + i}))
    embed.pin(p, side)
    by = {e.name: e for e in load_manifest(p).encoders}
    assert by["chemberta_5m_mtr"].params == 1000 and "position_ids" in by["chemberta_5m_mtr"].params_source
    assert by["molformer_xl"].params == 46805760 and by["molformer_xl"].parquet_sha256 == "05" * 32
    assert all(e.parquet_sha256 for e in by.values())


def test_embed_encoder_writes_parquet_sidecar_and_removes_cache(tmp_path, monkeypatch):
    import dataclasses
    import hashlib
    import pandas as pd
    from sweep.intrinsic_dim import load_embeddings
    smiles = ["CCO", "C", "c1ccccc1O", "CC(=O)O", "N#N"]
    table = tmp_path / "table.parquet"
    pd.DataFrame({"smiles_canonical": smiles}).to_parquet(table)
    m = load_manifest(MOLECULES_PATH)
    m = dataclasses.replace(m, label_table=str(table), label_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                            n_rows=len(smiles))
    enc = dataclasses.replace(next(e for e in m.encoders if e.name == "chemberta_5m_mtr"), dim=4)
    cache = tmp_path / "cache"
    snap = cache / "models--DeepChem--ChemBERTa-5M-MTR" / "snapshots" / "rev"; snap.mkdir(parents=True)
    torch.save({"w": torch.zeros(3, 4)}, snap / "pytorch_model.bin")
    (cache / "blobs").mkdir(); (cache / "blobs" / "stray").write_bytes(b"x" * 10)     # hub 1.25 keeps shared blobs here
    torch.manual_seed(0)
    model = Recorder().eval()
    monkeypatch.setattr(embed, "load_model", lambda e, c, d: (snap, StubTok(), model))
    side = embed.embed_encoder(enc, m, tmp_path / "hf", cache, 2, "cpu")
    dest = tmp_path / "hf" / enc.parquet_file
    E = load_embeddings(dest, enc.column)
    tok = StubTok(); embed.prepare_tokenizer(tok)
    alone = np.concatenate([embed.embed_smiles(model, tok, [s], 1, "cpu") for s in smiles])
    np.testing.assert_allclose(E, alone, rtol=0, atol=1e-6)                        # rows in table order
    assert json.loads(dest.with_suffix(".json").read_text()) == side
    assert side["n_rows"] == 5 and side["checkpoint_params"] == 12
    assert side["parquet_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert not cache.exists()
