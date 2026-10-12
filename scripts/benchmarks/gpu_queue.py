#!/usr/bin/env python3
"""File-based GPU job queue: keeps measurement lanes busy without an interactive session.

Jobs are JSON files in <root>/pending. A cron tick (`gpu_queue.py tick`, every two minutes under
flock) reaps finished or overdue jobs and starts every pending job whose GPUs are free and whose
dependencies are done. Each job runs in a detached wrapper (`gpu_queue.py run <file>`) that holds a
per-GPU flock for its whole life, optionally starts a model server first, runs the command with a
hard time limit, and moves the job file to done/ or failed/ with the exit code.

Job fields:
  id          unique name (also the file name)
  lane        label used for the idle report (e.g. "B")
  gpus        list of physical GPU indices (PCI bus order); [] for a CPU job
  cmd         shell command for the measurement
  cwd         working directory (default: root)
  max_hours   hard limit, at most 3 (whole job including server start)
  after       list of job ids that must be in done/ first (optional)
  server      shell command that serves a model for cmd (optional; stopped when cmd ends)
  server_url  URL that answers 200 once the server is ready (required with server)
  priority    lower runs first (default 50)
  resource    name of a shared server the job uses (e.g. "mom-live"); at most one running job per resource,
              so batch jobs do not crowd out each other or interactive use of that server (optional)

Idle report: every tick appends to <root>/status.log at most once an hour. A lane whose GPUs have
been idle for more than 30 minutes gets a line in the alarm file, with the reason (no pending work,
blocked dependency, or GPUs held by something outside the queue).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(os.environ.get("GPUQ_ROOT", "/media/knight2/EDS2/gpuqueue"))
ALARM = Path(os.environ.get("GPUQ_ALARM", str(Path.home() / "backlog/GPU_QUEUE_ALARM.md")))
LOCK_DIR = Path(os.environ.get("GPUQ_LOCK_DIR", "/tmp"))
MAX_HOURS = 3.0
IDLE_ALARM_S = 30 * 60
FREE_MIB = 1000  # a GPU with more memory in use than this belongs to someone else


def now() -> float:
    return time.time()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dirs(root: Path) -> dict[str, Path]:
    d = {k: root / k for k in ("pending", "running", "done", "failed", "logs")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.replace(path)


def validate(job: dict) -> None:
    for key in ("id", "lane", "gpus", "cmd"):
        if key not in job:
            raise ValueError(f"job without {key!r}")
    if float(job.get("max_hours", MAX_HOURS)) > MAX_HOURS:
        raise ValueError(f"{job['id']}: max_hours above {MAX_HOURS}")
    if job.get("server") and not job.get("server_url"):
        raise ValueError(f"{job['id']}: server without server_url")


def gpu_memory_used() -> dict[int, int]:
    """MiB in use per GPU; empty when nvidia-smi is unavailable (tests override this)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    used = {}
    for line in out.strip().splitlines():
        idx, mib = (x.strip() for x in line.split(","))
        used[int(idx)] = int(mib)
    return used


def lock_free(gpu: int) -> bool:
    with open(LOCK_DIR / f"gpuq_gpu{gpu}.lock", "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap(d: dict[str, Path]) -> None:
    """Move running jobs whose wrapper died without finishing to failed/ (the wrapper moves the rest)."""
    for path in d["running"].glob("*.json"):
        job = load(path)
        pid = job.get("wrapper_pid")
        if pid and not alive(pid) and path.exists():
            job.update(state="failed", finished=stamp(), reason="wrapper vanished")
            save(d["failed"] / path.name, job)
            path.unlink(missing_ok=True)


def startable(d: dict[str, Path], memory: dict[int, int]) -> list[Path]:
    done = {p.stem for p in d["done"].glob("*.json")}
    running = [load(p) for p in d["running"].glob("*.json")]
    busy = {g for j in running for g in j.get("gpus", [])}
    in_use = {j["resource"] for j in running if j.get("resource")}
    picked, claimed = [], set()
    jobs = sorted(d["pending"].glob("*.json"), key=lambda p: (load(p).get("priority", 50), p.stat().st_mtime))
    for path in jobs:
        job = load(path)
        gpus = set(job["gpus"])
        if not set(job.get("after", [])) <= done:
            continue
        if gpus & (busy | claimed):
            continue
        if job.get("resource") and job["resource"] in in_use:
            continue
        if any(memory.get(g, 0) > FREE_MIB or not lock_free(g) for g in gpus):
            continue
        picked.append(path)
        claimed |= gpus
        if job.get("resource"):
            in_use.add(job["resource"])
    return picked


def launch(d: dict[str, Path], path: Path) -> None:
    job = load(path)
    validate(job)
    target = d["running"] / path.name
    path.replace(target)
    log = d["logs"] / f"{job['id']}.log"
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", str(target)],
                            stdout=open(log, "ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env={**os.environ, "GPUQ_ROOT": str(d["pending"].parent)})
    job.update(state="running", started=stamp(), wrapper_pid=proc.pid, log=str(log))
    save(target, job)


def lane_report(d: dict[str, Path], lanes: dict[str, list[int]], memory: dict[int, int]) -> None:
    state_file = d["pending"].parent / "lane_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    running = [load(p) for p in d["running"].glob("*.json")]
    pending = [load(p) for p in d["pending"].glob("*.json")]
    done = {p.stem for p in d["done"].glob("*.json")}
    lines, alarms = [], []
    for lane, gpus in lanes.items():
        mine = [j["id"] for j in running if j.get("lane") == lane]
        if mine:
            state[lane] = now()
            lines.append(f"{lane}:busy({','.join(mine)})")
            continue
        idle_s = now() - state.setdefault(lane, now())
        waiting = [j for j in pending if j.get("lane") == lane]
        if not waiting:
            reason = "no pending work"
        elif all(not set(j.get("after", [])) <= done for j in waiting):
            reason = "waiting on dependencies"
        elif any(memory.get(g, 0) > FREE_MIB for g in gpus):
            reason = "GPUs held outside the queue"
        else:
            reason = "startable, next tick"
        lines.append(f"{lane}:idle {idle_s / 60:.0f}min ({reason})")
        if idle_s > IDLE_ALARM_S:
            alarms.append(f"- {stamp()} lane {lane} (GPU {gpus}) idle {idle_s / 60:.0f} min: {reason}")
    state_file.write_text(json.dumps(state))
    status = d["pending"].parent / "status.log"
    last = state.get("_last_status", 0)
    if now() - last >= 3600 or alarms:
        with open(status, "a") as fh:
            fh.write(f"{stamp()} " + " | ".join(lines) + "\n")
        state["_last_status"] = now()
        state_file.write_text(json.dumps(state))
    if alarms and now() - state.get("_last_alarm", 0) >= 3600:
        ALARM.parent.mkdir(parents=True, exist_ok=True)
        new = not ALARM.exists()
        with open(ALARM, "a") as fh:
            if new:
                fh.write("# GPU-wachtrij: leegloop-alarm\n\nAutomatisch geschreven door gpu_queue.py (hoogstens één blok per uur).\n\n")
            fh.write("\n".join(alarms) + "\n")
        state["_last_alarm"] = now()
        state_file.write_text(json.dumps(state))


def tick(root: Path) -> None:
    d = dirs(root)
    reap(d)
    memory = gpu_memory_used()
    for path in startable(d, memory):
        launch(d, path)
    lanes_file = root / "lanes.json"
    if lanes_file.exists():
        lane_report(d, json.loads(lanes_file.read_text()), memory)


def wait_ready(url: str, deadline: float, server: subprocess.Popen) -> bool:
    while now() < deadline:
        if server.poll() is not None:
            return False
        try:
            with urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(path: Path) -> int:
    """Wrapper: hold the GPU locks, start the server, run cmd under the deadline, file the result."""
    # A SIGTERM to the wrapper must still stop the server and the command (the finally block below);
    # without this, killing a wrapper left its model server holding the GPUs.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    d = dirs(path.parent.parent)
    job = load(path)
    handles = []
    for g in job["gpus"]:
        fh = open(LOCK_DIR / f"gpuq_gpu{g}.lock", "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # lost a race: back to pending, nothing ran
            job.update(state="pending")
            job.pop("wrapper_pid", None)
            save(d["pending"] / path.name, job)
            path.unlink(missing_ok=True)
            return 75
        handles.append(fh)
    deadline = now() + float(job.get("max_hours", MAX_HOURS)) * 3600
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in job["gpus"]) if job["gpus"] else ""}
    cwd = job.get("cwd") or str(path.parent.parent)
    server = None
    code, reason = 1, ""
    try:
        if job.get("server"):
            server = subprocess.Popen(job["server"], shell=True, cwd=cwd, env=env, start_new_session=True,
                                      stdout=open(d["logs"] / f"{job['id']}.server.log", "ab"), stderr=subprocess.STDOUT)
            if not wait_ready(job["server_url"], min(deadline, now() + 20 * 60), server):
                reason = "server did not become ready"
                return 1
        proc = subprocess.Popen(job["cmd"], shell=True, cwd=cwd, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=max(1.0, deadline - now()))
        except subprocess.TimeoutExpired:
            stop(proc)
            code, reason = 124, "max_hours reached"
        except SystemExit:
            stop(proc)
            code, reason = 143, "wrapper terminated"
        return code
    finally:
        stop(server)
        job.update(finished=stamp(), exit_code=code, state="done" if code == 0 else "failed")
        if reason:
            job["reason"] = reason
        save(d["done" if code == 0 else "failed"] / path.name, job)
        path.unlink(missing_ok=True)


def submit(root: Path, job: dict) -> Path:
    validate(job)
    d = dirs(root)
    for k in ("pending", "running", "done"):
        if (d[k] / f"{job['id']}.json").exists():
            raise ValueError(f"job {job['id']} already in {k}/")
    path = d["pending"] / f"{job['id']}.json"
    save(path, {**job, "state": "pending", "submitted": stamp()})
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick")
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    s = sub.add_parser("submit")
    s.add_argument("job_json", type=Path, help="file with one job object or a list of jobs")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "tick":
        tick(ROOT)
    elif a.cmd == "run":
        return run(a.job)
    elif a.cmd == "submit":
        jobs = json.loads(a.job_json.read_text())
        for job in jobs if isinstance(jobs, list) else [jobs]:
            print("submitted", submit(ROOT, job).name)
    else:
        d = dirs(ROOT)
        for k in ("running", "pending", "failed", "done"):
            names = sorted(p.stem for p in d[k].glob("*.json"))
            print(f"{k:8} {len(names):3}  {' '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
