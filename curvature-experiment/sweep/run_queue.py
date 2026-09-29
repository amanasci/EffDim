"""Run sweep jobs on free GPUs, one job per GPU, with done-markers so a pod restart only loses
the jobs that were in flight. Idempotent: rerunning skips jobs whose done-marker still matches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from sweep.jobs import JobSpec, Layout, build_jobs
from sweep.manifest import load_manifest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_outputs(job: JobSpec) -> Tuple[bool, str]:
    primary = Path(job.outputs[0])
    if not primary.exists() or primary.stat().st_size == 0:
        return False, f"missing or empty {primary}"
    if job.kind == "thin":
        with np.load(primary) as z:
            if "overlap" not in z.files:
                return False, f"{primary} has no 'overlap' array"
    else:
        try:
            rows = [json.loads(l) for l in primary.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError as e:
            return False, f"{primary} is not valid jsonl: {e}"
        kinds = [r.get("row") for r in rows]
        if "environment" not in kinds:
            return False, f"{primary} has no environment row"
        if not any(k not in ("environment", "fit") for k in kinds):
            return False, f"{primary} has no result rows"
    for extra in job.outputs[1:]:
        p = Path(extra)
        if not p.exists() or p.stat().st_size == 0:
            return False, f"missing or empty {p}"
    return True, "ok"


def _marker(job: JobSpec, lay: Layout) -> Path:
    return lay.done / f"{job.id}.done"


def is_done(job: JobSpec, lay: Layout) -> bool:
    m = _marker(job, lay)
    if not m.exists():
        return False
    rec = json.loads(m.read_text())
    for path, sha in rec["outputs"].items():
        p = Path(path)
        if not p.exists() or _sha256(p) != sha:
            return False
    return True


def mark_done(job: JobSpec, lay: Layout) -> None:
    lay.done.mkdir(parents=True, exist_ok=True)
    rec = {"id": job.id, "outputs": {o: _sha256(Path(o)) for o in job.outputs},
           "finished": datetime.now(timezone.utc).isoformat()}
    _marker(job, lay).write_text(json.dumps(rec, indent=1))


def quarantine(job: JobSpec, lay: Layout) -> None:
    stamp = int(time.time())
    for o in job.outputs:
        p = Path(o)
        if p.exists():
            p.rename(p.with_name(f"{p.name}.partial.{stamp}"))


def run_queue(jobs: List[JobSpec], lay: Layout, gpus: List[str], poll_s: float = 5.0,
              launcher: Callable = subprocess.Popen) -> Dict[str, List[str]]:
    lay.logs.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}
    out: Dict[str, List[str]] = {"done": [], "failed": [], "skipped": []}
    for j in jobs:
        if is_done(j, lay):
            status[j.id] = "done"; out["skipped"].append(j.id)
    pending = [j for j in jobs if j.id not in status]
    running: Dict[str, Tuple[JobSpec, subprocess.Popen, object]] = {}   # gpu -> (job, proc, logfile)
    while pending or running:
        for gpu, (j, proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close(); del running[gpu]
            ok, why = (validate_outputs(j) if proc.returncode == 0 else (False, f"exit {proc.returncode}"))
            if ok:
                mark_done(j, lay); status[j.id] = "done"; out["done"].append(j.id)
            else:
                status[j.id] = "failed"; out["failed"].append(j.id)
                print(f"[queue] FAILED {j.id}: {why}", flush=True)
        blocked = [j for j in pending if any(status.get(d) == "failed" for d in j.deps)]
        for j in blocked:
            pending.remove(j); status[j.id] = "blocked"
            print(f"[queue] blocked {j.id} (dependency failed)", flush=True)
        free = [g for g in gpus if g not in running]
        started_any = False
        for j in [j for j in pending if all(status.get(d) == "done" for d in j.deps)]:
            if not free:
                break
            gpu = free.pop(0)
            quarantine(j, lay)
            for o in j.outputs:
                Path(o).parent.mkdir(parents=True, exist_ok=True)
            log = open(lay.logs / f"{j.id}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            running[gpu] = (j, launcher(list(j.argv), env=env, stdout=log, stderr=subprocess.STDOUT), log)
            pending.remove(j)
            started_any = True
            print(f"[queue] start {j.id} on GPU {gpu}", flush=True)
        if running:
            time.sleep(poll_s)
        elif pending and not started_any:
            # nothing running and nothing newly started: remaining jobs wait on blocked/failed
            # or otherwise-unsatisfiable dependencies. They can never become ready, so stop
            # rather than busy-loop forever.
            for j in pending:
                status[j.id] = "blocked"
            pending = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices that are free, e.g. 0,1,2")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runners", default=str(Path(__file__).resolve().parents[1] / "runners"))
    ap.add_argument("--only", default=None, help="run only jobs whose id contains this substring")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lay = Layout(Path(a.root))
    jobs = build_jobs(load_manifest(), lay, a.python, Path(a.runners), a.threads)
    if a.only:
        wanted = {j.id for j in jobs if a.only in j.id}
        needed = set(wanted)
        by = {j.id: j for j in jobs}
        for jid in list(wanted):
            stack = list(by[jid].deps)
            while stack:
                d = stack.pop(); needed.add(d); stack.extend(by[d].deps)
        jobs = [j for j in jobs if j.id in needed]
    if a.dry_run:
        for j in jobs:
            print(("DONE " if is_done(j, lay) else "TODO ") + j.id)
        return
    res = run_queue(jobs, lay, [g.strip() for g in a.gpus.split(",") if g.strip()])
    print(json.dumps({k: len(v) for k, v in res.items()}), flush=True)
    if res["failed"]:
        print("failed: " + ", ".join(res["failed"]), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
