import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sweep.jobs import JobSpec, Layout
from sweep.run_queue import is_done, mark_done, quarantine, run_queue, validate_outputs


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
