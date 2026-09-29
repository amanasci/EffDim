"""Run sweep jobs on free GPUs, one job per GPU, with done-markers so a pod restart only loses
the jobs that were in flight. Idempotent: rerunning skips jobs whose done-marker still matches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
    try:
        rec = json.loads(m.read_text())
        outputs = rec["outputs"]
    except (json.JSONDecodeError, KeyError, TypeError):
        # a pod restart mid-write can leave a truncated/invalid marker; treat it as
        # not-done rather than crashing the queue, and let the job rerun.
        return False
    pruned = set(rec.get("pruned", []))
    for path, sha in outputs.items():
        if path in pruned:
            # deliberately deleted after use (see prune_geometry); a missing file here
            # is expected, not a failure, so skip the existence/sha check entirely.
            continue
        p = Path(path)
        if not p.exists() or _sha256(p) != sha:
            return False
    return True


def _write_marker(rec: dict, marker: Path) -> None:
    tmp = marker.with_name(marker.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(rec, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


def mark_done(job: JobSpec, lay: Layout) -> None:
    lay.done.mkdir(parents=True, exist_ok=True)
    rec = {"id": job.id, "outputs": {o: _sha256(Path(o)) for o in job.outputs},
           "finished": datetime.now(timezone.utc).isoformat()}
    _write_marker(rec, _marker(job, lay))


def prune_geometry(main_xfit_job: JobSpec, lay: Layout) -> None:
    """Delete a completed main_xfit job's geometry npz (outputs[1]) once the encoder's
    downstream cf/thin jobs no longer need it, and record the path under the "pruned" key
    in main_xfit's done-marker so is_done() treats its absence as expected, not a failure.

    Ordering matters for crash-safety: the marker is rewritten (atomically, via
    os.replace) to record the path as pruned *before* the file is deleted. If the process
    dies between those two steps, the marker already says the file is allowed to be
    missing, so a subsequent is_done() check sees a consistent (if not yet deleted)
    state instead of a done job with a "missing" required output.
    """
    if len(main_xfit_job.outputs) < 2:
        return
    geo = Path(main_xfit_job.outputs[1])
    marker = _marker(main_xfit_job, lay)
    try:
        rec = json.loads(marker.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    pruned = list(rec.get("pruned", []))
    if str(geo) not in pruned:
        pruned.append(str(geo))
        rec["pruned"] = pruned
        _write_marker(rec, marker)
    if geo.exists():
        geo.unlink()


def quarantine(job: JobSpec, lay: Layout) -> None:
    stamp = int(time.time())
    for o in job.outputs:
        p = Path(o)
        if p.exists():
            p.rename(p.with_name(f"{p.name}.partial.{stamp}"))


def run_queue(jobs: List[JobSpec], lay: Layout, gpus: List[str], poll_s: float = 5.0,
              launcher: Callable = subprocess.Popen, keep_geometry: bool = False,
              min_free_gb: float = 10.0) -> Dict[str, List[str]]:
    lay.logs.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}
    out: Dict[str, List[str]] = {"done": [], "failed": [], "skipped": [], "blocked": [], "deferred": []}
    by_id: Dict[str, JobSpec] = {j.id: j for j in jobs}
    for j in jobs:
        if is_done(j, lay):
            status[j.id] = "done"; out["skipped"].append(j.id)
            # a thin job can already be done at startup (an earlier invocation ran it, or
            # this process crashed between mark_done(thin) and the prune_geometry() call
            # that used to only happen on the live done-transition below) -- prune here
            # too so a geometry npz doesn't survive just because run_queue was restarted
            # after its thin job finished.
            if j.kind == "thin" and not keep_geometry:
                mx = by_id.get(f"{j.encoder}__main_xfit")
                if mx is not None:
                    prune_geometry(mx, lay)
    pending = [j for j in jobs if j.id not in status]
    running: Dict[str, Tuple[JobSpec, subprocess.Popen, object]] = {}   # gpu -> (job, proc, logfile)
    min_free_bytes = min_free_gb * (1024 ** 3)
    disk_low = False
    while pending or running:
        for gpu, (j, proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close(); del running[gpu]
            ok, why = (validate_outputs(j) if proc.returncode == 0 else (False, f"exit {proc.returncode}"))
            if ok:
                mark_done(j, lay); status[j.id] = "done"; out["done"].append(j.id)
                if j.kind == "thin" and not keep_geometry:
                    mx = by_id.get(f"{j.encoder}__main_xfit")
                    if mx is not None:
                        prune_geometry(mx, lay)
            else:
                status[j.id] = "failed"; out["failed"].append(j.id)
                print(f"[queue] FAILED {j.id}: {why}", flush=True)
        # Propagate to a fixpoint, not just one level: a job whose dependency was itself
        # just blocked (rather than directly failed) is equally unsatisfiable and must be
        # labeled "blocked" too -- otherwise, if nothing is running by the time we reach
        # the disk-guard catch-all below, it would be mislabeled "deferred" (implying it's
        # merely waiting on disk space, when it can in fact never run this session).
        unsatisfiable = True
        while unsatisfiable:
            newly_blocked = [j for j in pending if any(status.get(d) in ("failed", "blocked") for d in j.deps)]
            for j in newly_blocked:
                pending.remove(j); status[j.id] = "blocked"; out["blocked"].append(j.id)
                print(f"[queue] blocked {j.id} (dependency failed or blocked)", flush=True)
            unsatisfiable = bool(newly_blocked)
        free = [] if disk_low else [g for g in gpus if g not in running]
        started_any = False
        for j in [j for j in pending if all(status.get(d) == "done" for d in j.deps)]:
            if not free:
                break
            if not disk_low:
                free_bytes = shutil.disk_usage(lay.root).free
                if free_bytes < min_free_bytes:
                    disk_low = True
                    print(f"[queue] free disk on {lay.root} is {free_bytes / (1024**3):.1f} GB, "
                          f"below --min-free-gb {min_free_gb}; launching nothing further, "
                          "letting running jobs finish", flush=True)
            if disk_low:
                break
            gpu = free.pop(0)
            quarantine(j, lay)
            for o in j.outputs:
                Path(o).parent.mkdir(parents=True, exist_ok=True)
            log = open(lay.logs / f"{j.id}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            pending.remove(j)
            try:
                proc = launcher(list(j.argv), env=env, stdout=log, stderr=subprocess.STDOUT)
            except Exception as e:
                log.close()
                status[j.id] = "failed"; out["failed"].append(j.id)
                print(f"[queue] FAILED {j.id}: launcher raised {e!r}", flush=True)
                continue
            running[gpu] = (j, proc, log)
            started_any = True
            print(f"[queue] start {j.id} on GPU {gpu}", flush=True)
        if running:
            time.sleep(poll_s)
        elif pending and not started_any:
            if disk_low:
                # remaining jobs are healthy but we refuse to launch more while disk is low;
                # report them as deferred (distinct from "blocked", which means unsatisfiable
                # deps) so the caller can retry later once space is freed.
                for j in pending:
                    status[j.id] = "deferred"; out["deferred"].append(j.id)
                pending = []
            else:
                # nothing running and nothing newly started: remaining jobs wait on blocked/
                # failed or otherwise-unsatisfiable dependencies. They can never become ready,
                # so stop rather than busy-loop forever.
                for j in pending:
                    status[j.id] = "blocked"; out["blocked"].append(j.id)
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
    ap.add_argument("--keep-geometry", action="store_true",
                     help="don't delete each encoder's main_xfit geometry npz after its thin job finishes")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                     help="stop launching new jobs (letting running ones finish) once free disk "
                          "under --root drops below this many GB")
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
    res = run_queue(jobs, lay, [g.strip() for g in a.gpus.split(",") if g.strip()],
                     keep_geometry=a.keep_geometry, min_free_gb=a.min_free_gb)
    print(json.dumps({k: len(v) for k, v in res.items()}), flush=True)
    if res["failed"]:
        print("failed: " + ", ".join(res["failed"]), flush=True)
    if res["blocked"]:
        print("blocked: " + ", ".join(res["blocked"]), flush=True)
    if res["deferred"]:
        print("deferred (low disk, rerun once space is freed): " + ", ".join(res["deferred"]), flush=True)
    if res["failed"] or res["deferred"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
