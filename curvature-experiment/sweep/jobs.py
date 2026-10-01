"""Expand an encoder manifest x the per-encoder battery into job specs for run_queue.

Galaxies: the six-job battery at d = 16. Molecules (with a d file): the same battery at each encoder's d_run from
the d file, the molecule flags on split/cf/robust, plus main_d16 when d_run != 16. Without a d file the job argv are
byte-identical to the galaxy sweep's."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sweep.manifest import DEFAULT_PATH, Encoder, Manifest, load_manifest

JOB_SUFFIXES = ("main_xfit", "seed1", "seed2", "cf", "thin", "robust")
SPLIT = "09_physics_probe_facing_split_run.py"
CF = "09_physics_normal_scaling_run.py"
THIN = "09_physics_normal_scaling_thin_run.py"
ROBUST = "11_review_robustness_run.py"
D16 = "main_d16"
# per split job: (fit seed, extra flags, d)
SPLIT_JOBS = {
    "main_xfit": (0, ("--hessian-xfit",), 16),
    "seed1": (1, (), 16),
    "seed2": (2, (), 16),
}


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def records(self) -> Path: return Path(self.root) / "records"
    @property
    def geometry(self) -> Path: return Path(self.root) / "geometry"
    @property
    def arrays(self) -> Path: return Path(self.root) / "arrays"
    @property
    def done(self) -> Path: return Path(self.root) / "done"
    @property
    def logs(self) -> Path: return Path(self.root) / "logs"
    @property
    def timing(self) -> Path: return Path(self.root) / "timing"

    def hf_parquet(self, enc: Encoder) -> Path:
        return Path(self.root) / "hf" / enc.parquet_file


@dataclass(frozen=True)
class JobSpec:
    id: str
    encoder: str
    suffix: str
    argv: Tuple[str, ...]
    outputs: Tuple[str, ...]
    deps: Tuple[str, ...]
    kind: str


def _common(enc: Encoder, m: Manifest, lay: Layout, threads: int) -> Tuple[str, ...]:
    return ("--parquet-path", str(lay.hf_parquet(enc)), "--embedding-column", enc.column,
            "--label-table", m.label_table, "--threads", str(threads))


def load_d_file(path) -> Tuple[Dict[str, int], str]:
    """{encoder: d_run} from a d file ({name: {d_ID, d_run, estimates}}), and the file's sha256."""
    raw = Path(path).read_bytes()
    return {name: int(v["d_run"]) for name, v in json.loads(raw).items()}, hashlib.sha256(raw).hexdigest()


def _molecule_flags(m: Manifest, d_sha: str) -> Tuple[str, ...]:
    missing = [n for n in ("labels", "label_map", "expected_rows") if getattr(m, n) is None]
    if missing:
        raise ValueError(f"a d file needs a molecule manifest; the manifest has no {', '.join(missing)}")
    return ("--labels", ",".join(m.labels), "--label-map", m.label_map, "--expected-rows", str(m.expected_rows),
            "--d-file-sha256", d_sha)


def build_jobs(m: Manifest, lay: Layout, python: str, runners_dir: Path, threads: int,
               encoders: Optional[Sequence[str]] = None, d_file=None) -> List[JobSpec]:
    selected = list(m.encoders)
    if encoders is not None:
        known = {e.name for e in m.encoders}
        unknown = [n for n in encoders if n not in known]
        if unknown:
            raise ValueError(f"unknown encoders: {', '.join(unknown)}")
        selected = [e for e in m.encoders if e.name in set(encoders)]
    d_runs: Dict[str, int] = {}
    mol: Tuple[str, ...] = ()
    if d_file is not None:
        d_runs, d_sha = load_d_file(d_file)
        mol = _molecule_flags(m, d_sha)
        missing = [e.name for e in selected if e.name not in d_runs]
        if missing:
            raise ValueError(f"{d_file} has no d for encoders: {', '.join(missing)}")
    jobs: List[JobSpec] = []
    for enc in selected:
        rec = lambda s: str(lay.records / f"scaling__{enc.name}__{s}.jsonl")
        geo_dir = lay.geometry / enc.name
        d_enc = d_runs.get(enc.name)
        d_geo = 16 if d_enc is None else d_enc
        geo_npz = str(geo_dir / f"09_probe_facing_geometry_d{d_geo}_seed0.npz")
        for s, (seed, extra, d) in SPLIT_JOBS.items():
            dd = d if d_enc is None else d_enc
            argv = (python, str(runners_dir / SPLIT), "--mode", "physics", "--d-values", str(dd),
                    "--fit-seed", str(seed), *extra, *_common(enc, m, lay, threads),
                    "--device", "cuda", "--deterministic", "--record-path", rec(s), *mol)
            outs: Tuple[str, ...] = (rec(s),)
            if s == "main_xfit":
                argv = argv + ("--geometry-out", str(geo_dir))
                outs = outs + (geo_npz,)
            jobs.append(JobSpec(f"{enc.name}__{s}", enc.name, s, argv, outs, (), "split"))
        if d_enc is not None and d_enc != 16:
            argv = (python, str(runners_dir / SPLIT), "--mode", "physics", "--d-values", "16", "--fit-seed", "0",
                    *_common(enc, m, lay, threads), "--device", "cuda", "--deterministic", "--record-path", rec(D16), *mol)
            jobs.append(JobSpec(f"{enc.name}__{D16}", enc.name, D16, argv, (rec(D16),), (), "split"))
        arrays = str(lay.arrays / f"scaling__{enc.name}__cf.npz")
        cf_argv = (python, str(runners_dir / CF), "--mode", "physics", "--d", str(d_geo),
                   "--geometry-npz", geo_npz,
                   *_common(enc, m, lay, threads), "--arrays-out", arrays, "--record-path", rec("cf"), *mol)
        jobs.append(JobSpec(f"{enc.name}__cf", enc.name, "cf", cf_argv, (rec("cf"), arrays), (f"{enc.name}__main_xfit",), "cf"))
        thin_out = str(lay.arrays / f"scaling__{enc.name}__thin.npz")
        th_argv = (python, str(runners_dir / THIN), "--parquet-path", str(lay.hf_parquet(enc)),
                   "--embedding-column", enc.column, "--arrays-npz", arrays, "--out", thin_out, "--threads", str(threads))
        jobs.append(JobSpec(f"{enc.name}__thin", enc.name, "thin", th_argv, (thin_out,), (f"{enc.name}__cf",), "thin"))
        rb_argv = (python, str(runners_dir / ROBUST), "--encoder", enc.name, "--geometry-npz", geo_npz,
                   "--parquet-path", str(lay.hf_parquet(enc)), "--embedding-column", enc.column,
                   "--label-table", m.label_table, "--label-table-sha256", str(m.label_table_sha256),
                   "--published-split", rec("main_xfit"), "--published-cf", arrays, "--published-cf-record", rec("cf"),
                   "--guard", "exact", "--threads", str(threads), "--record-path", rec("robust"),
                   *mol, *(() if d_enc is None else ("--d", str(d_enc))))
        jobs.append(JobSpec(f"{enc.name}__robust", enc.name, "robust", rb_argv, (rec("robust"),),
                            (f"{enc.name}__main_xfit", f"{enc.name}__cf"), "robust"))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runners", default=str(Path(__file__).resolve().parents[1] / "runners"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--encoders", default=None)
    ap.add_argument("--manifest", default=str(DEFAULT_PATH))
    ap.add_argument("--d-file", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for j in build_jobs(load_manifest(a.manifest), Layout(Path(a.root)), a.python, Path(a.runners), a.threads,
                       encoders=a.encoders.split(",") if a.encoders else None, d_file=a.d_file):
        print(f"{j.id} {','.join(j.deps) or '-'} -> {j.outputs[0]}")


if __name__ == "__main__":
    main()
