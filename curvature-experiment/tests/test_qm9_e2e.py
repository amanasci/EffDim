"""End-to-end CPU smoke of the molecule battery: split -> cf -> thin -> robust on a synthetic 4,000-row table,
through the argv the job builder writes for molecules (about 3 minutes)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sweep.jobs import Layout, build_jobs
from sweep.manifest import load_manifest

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
LABS = ("gap", "mu", "alpha", "cv")
N, D = 4000, 64


def _toy(tmp_path):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((N, 4))
    X = np.tanh(z @ rng.standard_normal((4, D))) + 0.05 * rng.standard_normal((N, D))
    table = pd.DataFrame({"qm9_index": np.arange(1, N + 1), "gap": z[:, 0] + 0.3 * z[:, 1] ** 2,
                          "mu": np.sin(z[:, 1]) + 0.1 * rng.standard_normal(N), "alpha": z[:, 2] * z[:, 3],
                          "cv": z[:, 0] - z[:, 3] ** 2})
    lt = tmp_path / "toy_molecules.parquet"; table.to_parquet(lt, index=False)
    root = tmp_path / "out"
    emb = root / "hf" / "molecules" / "toy.parquet"; emb.parent.mkdir(parents=True)
    col = pa.ListArray.from_arrays(pa.array(np.arange(0, N * D + 1, D, dtype=np.int32)), pa.array(X.astype(np.float32).ravel()))
    pq.write_table(pa.table({"toy_molecules": col}), emb)
    src = {"repo": "toy", "snapshot": "toy", "n_rows": N, "label_table": str(lt),
           "label_table_sha256": hashlib.sha256(lt.read_bytes()).hexdigest(), "labels": list(LABS),
           "label_map": "identity", "expected_rows": N,
           "encoders": [{"name": "toy", "family": "Toy", "dim": D, "params": 1, "params_source": "test", "in_paper": False,
                         "parquet_file": "molecules/toy.parquet", "column": "toy_molecules"}]}
    mp = tmp_path / "toy.yaml"; mp.write_text(yaml.safe_dump(src, sort_keys=False))
    dp = tmp_path / "toy_d.json"
    dp.write_text(json.dumps({"toy": {"d_ID": 4, "d_run": 4, "estimates": {k: 4.0 for k in ("mle", "two_nn", "tle", "mind_mlk")}}}))
    return load_manifest(mp), Layout(root), dp


def _cpu(argv, suffix):
    a = list(argv)
    if "--device" in a:
        a[a.index("--device") + 1] = "cpu"
    if suffix == "main_xfit":
        a += ["--n-permutations", "50"]
    if suffix == "robust":
        a[a.index("--guard") + 1] = "refit"
        a += ["--n-perm", "50", "--n-boot", "20"]
    return a


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_molecule_battery_end_to_end_on_cpu(tmp_path):
    m, lay, dp = _toy(tmp_path)
    jobs = {j.suffix: j for j in build_jobs(m, lay, sys.executable, RUNNERS, threads=2, d_file=dp)}
    assert set(jobs) == {"main_xfit", "seed1", "seed2", "main_d16", "cf", "thin", "robust"}
    for s in ("main_xfit", "cf", "thin", "robust"):
        for o in jobs[s].outputs:
            Path(o).parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(_cpu(jobs[s].argv, s), capture_output=True, text=True, cwd=str(RUNNERS.parent))
        assert r.returncode == 0, f"{s}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    sha = hashlib.sha256(dp.read_bytes()).hexdigest()
    for s in ("main_xfit", "cf", "robust"):
        env = next(r for r in _rows(jobs[s].outputs[0]) if r["row"] == "environment")
        assert (env["d_file_sha256"], env["label_map"], env["expected_rows"]) == (sha, "identity", N), s
    res = [r for r in _rows(jobs["main_xfit"].outputs[0]) if r["row"] == "result"]
    assert {r["label"] for r in res} == set(LABS) and all(r["d"] == 4 for r in res)
    assert all(np.isfinite(r["columns"]["hess_mismatch_emp"]["multiscale"]["partial"]) for r in res)
    with np.load(jobs["cf"].outputs[1]) as z:
        assert all(f"{l}:S_model:r2_curve" in z.files for l in LABS)
    with np.load(jobs["thin"].outputs[0]) as z:
        assert "overlap" in z.files
    rb = _rows(jobs["robust"].outputs[0])
    guard = next(r for r in rb if r["row"] == "guard")
    assert guard["passed"] is True and guard["n_split"] == 2 * len(LABS) and guard["max_abs_diff_cf"] <= 1e-12
    assert {r["label"] for r in rb if r["row"] == "result"} == set(LABS)
