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


def test_prune_runs_for_already_done_thin_job_at_startup(tmp_path):
    # Simulate a prior invocation that got as far as marking "thin" done but crashed
    # (or was an older build) before pruning ran: the geometry file is back, and the
    # main_xfit marker has no "pruned" key, even though thin (and everything upstream of
    # it) is already done. A fresh run_queue() call, on startup alone (nothing needs to
    # launch), must still prune it and continue to report all three jobs as skipped.
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert not geo.exists()

    geo.parent.mkdir(parents=True, exist_ok=True)
    np.savez(geo, x=np.zeros(2))
    marker_path = lay.done / "a__main_xfit.done"
    rec = json.loads(marker_path.read_text())
    rec.pop("pruned", None)
    marker_path.write_text(json.dumps(rec))
    assert geo.exists()
    assert is_done(mx, lay) and is_done(thin, lay)

    r2 = run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert set(r2["skipped"]) == {"a__main_xfit", "a__cf", "a__thin"}
    assert r2["done"] == []
    assert not geo.exists()
    rec2 = json.loads(marker_path.read_text())
    assert rec2["pruned"] == [str(geo)]


def test_keep_geometry_skips_startup_prune_too(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05, keep_geometry=True)
    assert geo.exists()
    # already done from the run above; a second call with keep_geometry must not prune.
    run_queue([mx, cf, thin], lay, gpus=["0", "1", "2"], poll_s=0.05, keep_geometry=True)
    assert geo.exists()


def test_disk_low_does_not_mislabel_transitively_blocked_jobs_as_deferred(tmp_path, monkeypatch):
    # a fails; b depends on a (directly blocked); c depends on b (unsatisfiable two
    # levels removed from the failure -- only caught by propagating "blocked" to a
    # fixpoint, not by a single dependency-status check). z is independent of all three
    # and is only held back by the disk guard. Once a fails and the disk guard trips
    # (while trying to launch z, the only thing left that's actually launch-ready), b and
    # c must both be reported as "blocked" -- not "deferred", which would wrongly imply
    # they're merely waiting for space to free up -- while z (which really is just
    # waiting on space) is "deferred".
    lay = Layout(tmp_path)
    a = _split_job(lay, "a__main_xfit", ["result"], exit_code=1)
    b = _split_job(lay, "a__cf", ["result"], deps=["a__main_xfit"])
    c = _split_job(lay, "a__thin", ["result"], deps=["a__cf"])
    z = _split_job(lay, "z__main", ["result"])

    real_disk_usage = shutil.disk_usage
    calls = {"n": 0}

    def fake_disk_usage(path):
        calls["n"] += 1
        real = real_disk_usage(path)
        # first call (before "a" launches) reports plenty of headroom; every call after
        # that (the attempt to launch z, the only job left that's actually ready) reports
        # low free space, so the guard trips there instead of blocking "a" itself.
        free = 100 * (1024 ** 3) if calls["n"] == 1 else 5 * (1024 ** 3)
        return type(real)(total=real.total, used=real.used, free=free)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
    r = run_queue([a, b, c, z], lay, gpus=["0"], poll_s=0.05, min_free_gb=10.0)
    assert r["failed"] == ["a__main_xfit"]
    assert set(r["blocked"]) == {"a__cf", "a__thin"}
    assert r["deferred"] == ["z__main"]


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


def test_prune_waits_for_robust(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    rb_rec = str(lay.records / "a__robust.jsonl")
    # robust fails: thin is done but the geometry must stay
    robust = _npz_job("a__robust", "robust", (rb_rec,), deps=("a__main_xfit", "a__cf"), exit_code=1)
    run_queue([mx, cf, thin, robust], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert geo.exists()


def test_prune_after_thin_and_robust_both_done(tmp_path):
    lay = Layout(tmp_path)
    mx, cf, thin, geo = _geometry_chain(lay)
    rb_rec = str(lay.records / "a__robust.jsonl")
    robust = _npz_job("a__robust", "robust", (rb_rec,), deps=("a__main_xfit", "a__cf"))
    run_queue([mx, cf, thin, robust], lay, gpus=["0", "1", "2"], poll_s=0.05)
    assert not geo.exists()


def test_order_largest_first_is_stable():
    from sweep.run_queue import order_largest_first
    js = [JobSpec(f"{e}__{s}", e, s, (), ("x",), (), "split") for e in ("small", "big", "mid") for s in ("a", "b")]
    got = [j.id for j in order_largest_first(js, {"small": 384, "big": 4096, "mid": 1024})]
    assert got == ["big__a", "big__b", "mid__a", "mid__b", "small__a", "small__b"]


from qm9_fixtures import MOL_MANIFEST, MOLECULES, write_d_file

needs_timeout = pytest.mark.skipif(shutil.which("timeout") is None, reason="needs GNU timeout")


def _capture():
    seen = []

    def launcher(argv, env=None, stdout=None, stderr=None):
        seen.append(argv)
        return subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
    return seen, launcher


def _timing(lay, jid):
    return json.loads((lay.timing / f"{jid}.json").read_text())


def test_no_timeout_leaves_argv_alone_and_still_times(tmp_path):
    from sweep.run_queue import job_argv
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    seen, launcher = _capture()
    run_queue([j], lay, gpus=["0"], poll_s=0.05, launcher=launcher)
    assert seen[0] == list(j.argv) == job_argv(j, lay)
    t = _timing(lay, "a__main")
    assert set(t) == {"exit", "max_rss_kb", "wall_s"} and t["exit"] == 0 and t["max_rss_kb"] > 0 and t["wall_s"] >= 0


@needs_timeout
def test_timeout_prefixes_argv(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    seen, launcher = _capture()
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05, launcher=launcher, timeout_h=8)
    assert seen[0][:2] == ["timeout", "8h"] and seen[0][2:] == list(j.argv)
    assert r["done"] == ["a__main"] and _timing(lay, "a__main")["exit"] == 0


@needs_timeout
def test_timed_out_job_gets_timing_file(tmp_path):
    lay = Layout(tmp_path)
    rec = str(lay.records / "a__main.jsonl")
    j = JobSpec("a__main", "a", "main", (sys.executable, "-c", "import time; time.sleep(60)"), (rec,), (), "split")
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05, timeout_h=0.0005)          # about 2 s
    assert r["failed"] == ["a__main"] and _timing(lay, "a__main")["exit"] == 124


def test_main_reads_manifest_and_d_file(tmp_path, monkeypatch, capsys):
    from sweep import run_queue as rq
    d = write_d_file(tmp_path, {n: 23 for n in MOLECULES})
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(tmp_path / "root"), "--gpus", "0", "--threads", "3",
                                      "--manifest", str(MOL_MANIFEST), "--d-file", str(d), "--encoders", "chemfm_3b", "--dry-run"])
    rq.main()
    ids = [w for w in capsys.readouterr().out.split() if w.startswith("chemfm_3b__")]
    assert "chemfm_3b__main_d16" in ids and len(ids) == 7
