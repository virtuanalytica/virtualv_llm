import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmarks"))
import gpu_queue as gq  # noqa: E402


@pytest.fixture
def q(tmp_path, monkeypatch):
    monkeypatch.setattr(gq, "LOCK_DIR", tmp_path / "locks")
    (tmp_path / "locks").mkdir()
    monkeypatch.setattr(gq, "ALARM", tmp_path / "alarm.md")
    monkeypatch.setattr(gq, "gpu_memory_used", lambda: {})
    monkeypatch.setenv("GPUQ_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("GPUQ_ALARM", str(tmp_path / "alarm.md"))
    return tmp_path / "queue"


def wait_for(cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def job(id, gpus, cmd="true", **kw):
    return {"id": id, "lane": kw.pop("lane", "B"), "gpus": gpus, "cmd": cmd, "max_hours": kw.pop("max_hours", 0.01), **kw}


def test_one_job_per_gpu_and_dependencies_wait(q):
    gq.submit(q, job("a", [4], "sleep 1"))
    gq.submit(q, job("b", [4]))
    gq.submit(q, job("c", [5], after=["a"]))
    gq.tick(q)
    assert sorted(p.stem for p in (q / "running").glob("*.json")) == ["a"]
    assert wait_for(lambda: (q / "done/a.json").exists())
    gq.tick(q)
    assert wait_for(lambda: (q / "done/b.json").exists() and (q / "done/c.json").exists() or gq.tick(q))
    assert json.loads((q / "done/c.json").read_text())["exit_code"] == 0


def test_failing_and_overdue_jobs_land_in_failed(q):
    gq.submit(q, job("bad", [1], "exit 3"))
    gq.submit(q, job("slow", [2], "sleep 60", max_hours=2 / 3600))
    gq.tick(q)
    assert wait_for(lambda: (q / "failed/bad.json").exists() and (q / "failed/slow.json").exists())
    slow = json.loads((q / "failed/slow.json").read_text())
    assert slow["exit_code"] == 124 and slow["reason"] == "max_hours reached"
    assert json.loads((q / "failed/bad.json").read_text())["exit_code"] == 3


def test_gpu_used_outside_the_queue_is_not_taken(q, monkeypatch):
    monkeypatch.setattr(gq, "gpu_memory_used", lambda: {3: 25000})
    gq.submit(q, job("x", [3]))
    gq.tick(q)
    assert (q / "pending/x.json").exists()


def test_server_is_started_before_and_stopped_after_the_command(q, tmp_path):
    marker = tmp_path / "server_alive"
    srv = ("python3 -c \"import http.server,pathlib;pathlib.Path('%s').write_text('1');"
           "http.server.HTTPServer(('127.0.0.1',18765),http.server.SimpleHTTPRequestHandler).serve_forever()\"" % marker)
    gq.submit(q, job("s", [5], "curl -sf http://127.0.0.1:18765/ >/dev/null", server=srv,
                     server_url="http://127.0.0.1:18765/", max_hours=0.05))
    gq.tick(q)
    assert wait_for(lambda: (q / "done/s.json").exists(), timeout=40)
    with pytest.raises(Exception):
        from urllib.request import urlopen
        urlopen("http://127.0.0.1:18765/", timeout=2)


def test_validation_and_duplicate_ids(q):
    with pytest.raises(ValueError):
        gq.submit(q, job("long", [1], max_hours=5))
    with pytest.raises(ValueError):
        gq.submit(q, job("srv", [1], server="x"))
    gq.submit(q, job("dup", [1]))
    with pytest.raises(ValueError):
        gq.submit(q, job("dup", [1]))


def test_idle_lane_raises_an_alarm_with_a_reason(q, monkeypatch):
    gq.dirs(q)
    (q / "lanes.json").write_text(json.dumps({"C": [1, 2]}))
    (q / "lane_state.json").write_text(json.dumps({"C": time.time() - 3600}))
    gq.tick(q)
    text = gq.ALARM.read_text()
    assert "lane C" in text and "no pending work" in text
    assert "C:idle" in (q / "status.log").read_text()
