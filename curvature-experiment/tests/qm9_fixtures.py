"""Shared fixtures for the QM9 (molecule) sweep tests."""
import json
from pathlib import Path

MOLECULES = ("chemberta_5m_mtr", "chemberta_10m_mtr", "chemberta_77m_mtr", "chemberta_10m_mlm", "chemberta_77m_mlm",
             "molformer_xl", "chemfm_1b", "chemfm_3b")
MOL_MANIFEST = Path(__file__).resolve().parents[1] / "molecules.yaml"
MOL_LABELS = ("gap", "mu", "alpha", "cv")


def write_d_file(tmp_path, d_ids) -> Path:
    """A molecules_d.json: d_run = min(d_ID, 20), every estimate equal to d_ID."""
    src = {n: {"d_ID": int(d), "d_run": min(int(d), 20),
               "estimates": {k: float(d) for k in ("mle", "two_nn", "tle", "mind_mlk")}} for n, d in d_ids.items()}
    p = Path(tmp_path) / "molecules_d.json"
    p.write_text(json.dumps(src, indent=1, sort_keys=True) + "\n")
    return p
