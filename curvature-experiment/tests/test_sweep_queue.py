import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sweep.jobs import JobSpec, Layout
from sweep.run_queue import is_done, mark_done, prune_geometry, quarantine, run_queue, validate_outputs


def _split_job(lay, jid, body_rows, deps=(), exit_code=0):
    rec = lay.records / f"{jid}.jsonl"
    rows = [{"row": "environment"}] + [{"row": r} for r in body_rows]
    code = ("import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
            f"p.write_text(''.join(json.dumps(r)+'\\n' for r in {rows!r})); sys.exit({exit_code})")
    return JobSpec(jid, "enc", jid, (sys.executable, "-c", code, str(rec)), (str(rec),), tuple(deps), "split")


def test_success_marks_done_and_second_run_skips(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    r1 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r1["done"] == ["a__main"] and is_done(j, lay)
    r2 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r2["skipped"] == ["a__main"]


def test_exit_zero_without_result_rows_is_not_done(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", [])
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r["failed"] == ["a__main"] and not is_done(j, lay)


def test_failed_dependency_blocks_dependents(tmp_path):
    lay = Layout(tmp_path)
    a = _split_job(lay, "a__main_xfit", ["result"], exit_code=1)
    b = _split_job(lay, "a__cf", ["result"], deps=["a__main_xfit"])
    r = run_queue([a, b], lay, gpus=["0", "1"], poll_s=0.05)
    assert r["failed"] == ["a__main_xfit"] and "a__cf" not in r["done"]
    assert r["blocked"] == ["a__cf"]


def test_partial_record_is_moved_aside_and_rerun(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    rec = Path(j.outputs[0]); rec.parent.mkdir(parents=True)
    rec.write_text('{"row": "environment"}\n')           # killed mid-run: no done marker
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r["done"] == ["a__main"]
    assert list(rec.parent.glob("a__main.jsonl.partial.*"))


def test_done_marker_invalid_if_output_changed(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    run_queue([j], lay, gpus=["0"], poll_s=0.05)
    Path(j.outputs[0]).write_text('{"row": "environment"}\n')
    assert not is_done(j, lay)


def test_one_job_per_gpu(tmp_path):
    lay = Layout(tmp_path)
    seen = []

    def launcher(argv, env=None, stdout=None, stderr=None):
        seen.append(env["CUDA_VISIBLE_DEVICES"])
        return subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)

    jobs = [_split_job(lay, f"e{i}__main", ["result"]) for i in range(4)]
    run_queue(jobs, lay, gpus=["3", "5"], poll_s=0.05, launcher=launcher)
    assert set(seen) == {"3", "5"} and len(seen) == 4


def test_thin_validation_requires_overlap(tmp_path):
    lay = Layout(tmp_path)
    out = tmp_path / "t.npz"
    np.savez(out, other=np.zeros(2))
    j = JobSpec("a__thin", "a", "thin", (sys.executable, "-c", "pass"), (str(out),), (), "thin")
    ok, why = validate_outputs(j)
    assert not ok and "overlap" in why


def test_corrupt_done_marker_is_not_done_and_job_reruns(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    r1 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r1["done"] == ["a__main"] and is_done(j, lay)
    marker = lay.done / "a__main.done"
    marker.write_text("{")           # truncated by a pod restart mid-write
    assert not is_done(j, lay)       # must not raise, must report not-done
    r2 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r2["done"] == ["a__main"] and is_done(j, lay)


def test_mark_done_leaves_no_tmp_file(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert not list(lay.done.glob("*.tmp"))


def _npz_job(jid, kind, outputs, deps=(), exit_code=0, thin=False):
    """A fake job that writes a .jsonl (with environment+result rows) or a .npz for each
    path in `outputs`, mimicking the shape run_queue's validate_outputs() expects for
    "split"/"cf" (jsonl) vs "thin" (npz with an 'overlap' array) jobs."""
    array_kw = "overlap=np.zeros(2)" if thin else "x=np.zeros(2)"
    code = (
        "import sys, json, pathlib, numpy as np\n"
        "for p in sys.argv[1:]:\n"
        "    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if p.suffix == '.npz':\n"
        f"        np.savez(p, {array_kw})\n"
        "    else:\n"
        "        p.write_text(json.dumps({'row': 'environment'}) + chr(10) + "
        "json.dumps({'row': 'result'}) + chr(10))\n"
        f"sys.exit({exit_code})\n"
    )
    enc, suffix = jid.split("__", 1)
    argv = (sys.executable, "-c", code, *outputs)
    return JobSpec(jid, enc, suffix, argv, tuple(outputs), tuple(deps), kind)


def _geometry_chain(lay, enc="a"):
    geo = str(lay.geometry / enc / "09_probe_facing_geometry_d16_seed0.npz")
    rec_mx = str(lay.records / f"{enc}__main_xfit.jsonl")
    rec_cf = str(lay.records / f"{enc}__cf.jsonl")
    arrays = str(lay.arrays / f"{enc}__cf.npz")
    thin_out = str(lay.arrays / f"{enc}__thin.npz")
    mx = _npz_job(f"{enc}__main_xfit", "split", (rec_mx, geo))
    cf = _npz_job(f"{enc}__cf", "cf", (rec_cf, arrays), deps=(f"{enc}__main_xfit",))
    thin = _npz_job(f"{enc}__thin", "thin", (thin_out,), deps=(f"{enc}__cf",), thin=True)
    return mx, cf, thin, Path(geo)


def test_geometry_pruned_after_thin_job_done(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    r = run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert set(r["done"]) == {"a__main_xfit", "a__cf", "a__thin"}
    assert not geo.exists()
    rec = json.loads((lay.done / "a__main_xfit.done").read_text())
    assert rec["pruned"] == [str(geo)]


def test_main_xfit_stays_done_after_pruning_second_run_skips_all(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert not geo.exists()
    assert is_done(mx, lay) and is_done(cf, lay) and is_done(thin, lay)
    r2 = run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert set(r2["skipped"]) == {"a__main_xfit", "a__cf", "a__thin"}
    assert r2["done"] == []


def test_keep_geometry_flag_keeps_the_file(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05, keep_geometry=True)
    assert geo.exists()
    rec = json.loads((lay.done / "a__main_xfit.done").read_text())
    assert "pruned" not in rec


def test_prune_marks_marker_before_deleting_file(tmp_path, monkeypatch):
    # Simulate a crash between the marker rewrite and the unlink: force Path.unlink to
    # raise right after prune_geometry has (atomically) rewritten the marker. The marker
    # must already list the path as pruned, and is_done() must not crash either way.
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    run_queue([mx], lay, gpus=["0"], poll_s=0.05)
    assert is_done(mx, lay)

    real_unlink = Path.unlink

    def boom(self, *a, **k):
        raise OSError("simulated crash after marker rewrite")

    monkeypatch.setattr(Path, "unlink", boom)
    with pytest.raises(OSError):
        prune_geometry(mx, lay)
    monkeypatch.setattr(Path, "unlink", real_unlink)

    rec = json.loads((lay.done / "a__main_xfit.done").read_text())
    assert rec["pruned"] == [str(geo)]
    assert geo.exists()          # delete never happened
    assert is_done(mx, lay)      # still done: pruned path is exempt from the sha check

    geo.unlink()                 # now actually gone (as a later successful prune would do)
    assert is_done(mx, lay)      # missing pruned file is fine too


def test_disk_guard_defers_remaining_jobs_when_free_space_low(tmp_path, monkeypatch):
    lay = Layout(tmp_path)
    jobs = [_split_job(lay, f"e{i}__main", ["result"]) for i in range(4)]

    real_disk_usage = shutil.disk_usage
    calls = {"n": 0}

    def fake_disk_usage(path):
        calls["n"] += 1
        real = real_disk_usage(path)
        # first call reports plenty of headroom (independent of the test machine's actual
        # free space), every call after that reports below the 10 GB default threshold so
        # nothing new gets launched once the guard has tripped once.
        free = 100 * (1024 ** 3) if calls["n"] == 1 else 5 * (1024 ** 3)
        return type(real)(total=real.total, used=real.used, free=free)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
    r = run_queue(jobs, lay, gpus=["0"], poll_s=0.05, min_free_gb=10.0)
    assert len(r["done"]) + len(r["deferred"]) == 4
    assert r["deferred"], "expected some jobs to be deferred once disk usage dropped"
    assert calls["n"] > 1


def test_launcher_exception_marks_job_failed_and_closes_log(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])

    def bad_launcher(argv, env=None, stdout=None, stderr=None):
        raise OSError("boom")

    r = run_queue([j], lay, gpus=["0"], poll_s=0.05, launcher=bad_launcher)
    assert r["failed"] == ["a__main"]
    log = lay.logs / "a__main.log"
    assert log.exists()
    log.unlink()   # would raise on POSIX if still open and locked elsewhere; mainly checks no leak
