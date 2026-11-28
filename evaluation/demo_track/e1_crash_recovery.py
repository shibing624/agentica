#!/usr/bin/env python3
"""E1 — unattended work when the gateway is SIGKILLed and comes back.

Everything runs against a throwaway gateway: its own data root, its own port,
no IM channel logged in, one profile copied from the real config. Nothing here
touches the gateway the desktop app is running, or the user's own sessions.

One fresh session per trial:

  A   orphan     the job is still running when the gateway dies, and ends while
                the gateway is down
  B0  control    the gateway dies after the job's report is in the log and the
                model has answered it
  B1  settle+Δ   the gateway dies Δ seconds after the job's record settles
  B2  entry      the gateway dies the moment the report entry appears in the log
  C0  open first the gateway dies mid-turn; after the restart the session is
                opened (which builds its agent) before anyone writes to it
  C1  message    the gateway dies mid-turn; after the restart the first thing
                that happens is a new user message

    python3 evaluation/demo_track/e1_crash_recovery.py --smoke
    python3 evaluation/demo_track/e1_crash_recovery.py --trials A=12,B0=3,B1=8,B2=6,C0=6,C1=6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[2]
REAL_CONFIG = Path.home() / ".agentica" / "config.yaml"
APP_PYTHON = Path("/Applications/Agentica.app/Contents/Resources/runtime/python/bin/python3")

# A report is an automated entry whose text carries the job id and one of these.
# The tool result of the execute call also names the job id, which is why the id
# alone is not enough.
REPORT_MARKERS = (
    "Automatic report:",
    "background terminals finished",
    "ended while nobody was watching",
)
SCENARIOS = ("A", "B0", "B1", "B2", "C0", "C1")

START = (
    "Use the execute tool exactly once, with background=true, to run this command:\n"
    "sleep {seconds}; echo {marker}\n"
    "Do not wait for it and do not call any other tool. When the tool call returns, "
    "reply with the single word OK."
)
POKE = "Reply with the single word OK."
LIVE = (
    "In the current working directory, create three files named step1.txt, step2.txt "
    "and step3.txt. Use the write_file tool once per file, one call at a time (wait for "
    "each result before the next), and put only the file's own name inside each file. "
    "Then reply with the single word DONE."
)


class Pace:
    def __init__(self, smoke: bool) -> None:
        self.job_a = 8 if smoke else 20
        # Long enough that the job is still running when the turn that started
        # it is over, so "Δ after the record settles" counts from the job's own
        # end and not from the end of that turn.
        self.job_b = 6 if smoke else 10
        self.untouched = 20 if smoke else 45
        self.after_open = 30 if smoke else 60
        self.after_poke = 10 if smoke else 20


def wait_for(pred: Callable[[], Any], timeout: float, interval: float = 0.5) -> float | None:
    """Seconds until ``pred()`` was truthy, or None if it never was."""
    t0 = time.monotonic()
    while True:
        if pred():
            return round(time.monotonic() - t0, 2)
        if time.monotonic() - t0 >= timeout:
            return None
        time.sleep(interval)


def http(method: str, url: str, body: Any = None, timeout: float = 30.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


class Turn(threading.Thread):
    """One blocking POST /api/chat, so the caller can act while the turn runs."""

    def __init__(self, url: str, body: dict, timeout: float) -> None:
        super().__init__(daemon=True)
        self.url, self.body, self.timeout = url, body, timeout
        self.done = False
        self.ok = False
        self.content: str | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.content = http("POST", self.url, self.body, timeout=self.timeout)["content"]
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - the error is the measurement
            self.error = f"{type(exc).__name__}: {exc}"
        self.done = True


class Gateway:
    def __init__(self, python: str, src: Path, home: Path, env_extra: dict[str, str]) -> None:
        self.python, self.src, self.home, self.env_extra = python, src, home, env_extra
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.starts = 0

    def _env(self) -> dict[str, str]:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            "AGENTICA_HOME": str(self.home),
            "GATEWAY_AUTH": "false",
            # Both packages from the checkout, ahead of whatever the interpreter's
            # site-packages holds, so the run tests one commit.
            "PYTHONPATH": f"{REPO}{os.pathsep}{self.src}",
            "PYTHONUNBUFFERED": "1",
        }
        env.update(self.env_extra)
        return env

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def start(self) -> float:
        """Start (or restart) the gateway; returns seconds until it answered."""
        if self.proc is not None and self.proc.poll() is None:
            self.kill()
        self.starts += 1
        log_path = self.home / f"gateway-{self.starts}.log"
        log = open(log_path, "ab")
        self.proc = subprocess.Popen(
            [self.python, "-c", "from agentica_gateway.main import main; main()",
             "--port", "0", "--parent-pid", str(os.getpid())],
            cwd=self.home, env=self._env(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        runtime = self.home / "cache" / "gateway" / "runtime.json"

        def up() -> bool:
            if self.proc.poll() is not None:
                raise RuntimeError(f"gateway exited with {self.proc.returncode}; see {log_path}")
            try:
                rec = json.loads(runtime.read_text())
            except (OSError, ValueError):
                return False
            if rec["pid"] != self.proc.pid:
                return False
            self.port = int(rec["port"])
            try:
                return http("GET", self.url("/api/health"), timeout=5) is not None
            except (urllib.error.URLError, OSError, ValueError):
                return False

        secs = wait_for(up, 240, 0.5)
        if secs is None:
            raise RuntimeError(f"gateway did not come up; see {log_path}")
        log.flush()
        # The personal WeChat channel is always registered; with no cached token
        # under this data root it must stay logged out, and nothing else may
        # register at all. Stated as the fact that must hold.
        text = log_path.read_text(errors="replace")
        registered = {
            line.rsplit("Channel registered: ", 1)[1].strip()
            for line in text.splitlines() if "Channel registered: " in line
        }
        if registered - {"wechat"} or "WeChat: Connected" in text or "WeChat: no cached token" not in text:
            raise RuntimeError(
                f"the test gateway reached IM state it must not have (registered={sorted(registered)}); "
                f"see {log_path}"
            )
        return round(secs, 1)

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=15)

    def sweep(self) -> None:
        """Also take whatever the gateway spawned into its process group."""
        self.kill()
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def resolve_python(explicit: str | None) -> str:
    python = explicit or (str(APP_PYTHON) if APP_PYTHON.exists() else sys.executable)
    probe = subprocess.run(
        [python, "-c", "import agentica, agentica_gateway; print(agentica.__file__)"],
        capture_output=True, text=True, timeout=60,
    )
    if probe.returncode != 0:
        sys.exit(f"{python} cannot import agentica and agentica_gateway:\n{probe.stderr}")
    return python


def make_home(profile: str) -> Path:
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    prof = config["profiles"][profile]
    home = Path(tempfile.mkdtemp(prefix="agentica-e1-"))
    home.chmod(0o700)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"active_profile": profile, "profiles": {profile: prof}}, allow_unicode=True),
        encoding="utf-8",
    )
    return home


def mkwork(home: Path, session_id: str) -> Path:
    work = home / "work" / session_id
    work.mkdir(parents=True)
    # The real path: the gateway keys a session's project directory by the
    # resolved work_dir, the CLI by the string it was given, and a symlinked
    # temp dir (macOS /var -> /private/var) would hand the two surfaces
    # different projects for the same session.
    return work.resolve()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Session:
    """What is on disk for one session: its transcript and its job records."""

    def __init__(self, home: Path, session_id: str) -> None:
        self.home, self.session_id = home, session_id
        self._log_path: Path | None = None

    @property
    def log_path(self) -> Path:
        """Resolved on first use: the file exists only after the session's first turn."""
        if self._log_path is None:
            self._log_path = next((self.home / "projects").rglob(f"{self.session_id}.jsonl"))
        return self._log_path

    def entries(self) -> list[dict]:
        entries = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue  # the line being written right now
        return entries

    def records(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for path in sorted((self.log_path.parent / self.session_id / "background").glob("term_*.json")):
            rec = json.loads(path.read_text())
            out[rec["id"]] = rec
        return out


def count_reports(entries: list[dict], job_id: str) -> int:
    return sum(
        1 for e in entries
        if e.get("automated") is True
        and job_id in str(e.get("content"))
        and any(marker in str(e["content"]) for marker in REPORT_MARKERS)
    )


def chat(gw: Gateway, session_id: str, work: Path, message: str, timeout: float) -> tuple[bool, str | None]:
    turn = Turn(gw.url("/api/chat"), {
        "message": message, "session_id": session_id, "work_dir": str(work),
        "approval_mode": "allow-all",
    }, timeout)
    turn.start()
    turn.join(timeout + 30)
    if not turn.done:
        return False, "turn thread did not finish"
    return turn.ok, (turn.content if turn.ok else turn.error)


def list_jobs(gw: Gateway, session_id: str) -> list[dict]:
    """The job list only: reads the records, builds nothing."""
    return http("GET", gw.url(f"/api/sessions/{session_id}/background"))["jobs"]


def open_session(gw: Gateway, session_id: str) -> None:
    """What opening a session does in the web UI: builds its agent, runs no turn."""
    http("GET", gw.url(f"/api/sessions/{session_id}/usage"), timeout=120)


def start_job(gw: Gateway, sess: Session, work: Path, seconds: int, marker: str) -> dict:
    ok, reply = chat(gw, sess.session_id, work, START.format(seconds=seconds, marker=marker), 300.0)

    def has_job() -> bool:
        return bool(sess.records())

    if wait_for(has_job, 60, 0.5) is None:
        files = sorted(str(p.name) for p in sess.log_path.parent.rglob("*") if p.is_file())[:40]
        return {"found": False, "start_turn_ok": ok, "start_reply": reply, "session_files": files}
    job_id, rec = next(iter(sess.records().items()))
    return {"found": True, "job_id": job_id, "pid": int(rec["pid"]), "log_path": rec["log_path"],
            "start_turn_ok": ok, "start_reply": reply}


def files_written(entries: list[dict]) -> dict[str, bool]:
    """file name -> its write_file call has a successful result on disk."""
    results = {e["tool_call_id"]: not e["is_error"] for e in entries if e.get("type") == "tool"}
    out: dict[str, bool] = {}
    for e in entries:
        for call in e.get("tool_calls") or []:
            if call["function"]["name"] != "write_file":
                continue
            args = json.loads(call["function"]["arguments"])
            out[Path(args["file_path"]).name] = results.get(call["id"], False)
    return out


def scenario_a(gw: Gateway, home: Path, run: str, i: int, p: Pace) -> dict:
    sid = f"s_e1_a_{run}_{i:02d}"
    work = mkwork(home, sid)
    marker = f"E1-A-{i}-DONE"
    gw.start()
    sess = Session(home, sid)
    r: dict[str, Any] = {"scenario": "A", "trial": i, "session_id": sid}
    job = start_job(gw, sess, work, p.job_a, marker)
    if not job["found"]:
        r["invalid"] = "the job was not started"
        r.update(job)
        return r
    job_id = job["job_id"]
    r.update(job_id=job_id, job_pid=job["pid"], start_turn_ok=job["start_turn_ok"], start_reply=job["start_reply"])
    gw.kill()
    r["job_alive_after_kill"] = pid_alive(job["pid"])
    r["reports_at_kill"] = count_reports(sess.entries(), job_id)

    def ended() -> bool:
        return not pid_alive(job["pid"]) and marker in Path(job["log_path"]).read_text(errors="replace")

    r["job_ended_while_down"] = wait_for(ended, p.job_a + 60, 0.5) is not None
    r["reports_while_down"] = count_reports(sess.entries(), job_id)
    r["restart_s"] = gw.start()

    def reported() -> bool:
        return count_reports(sess.entries(), job_id) > 0

    r["t_report_untouched"] = wait_for(reported, p.untouched, 1.0)
    mine = [j for j in list_jobs(gw, sid) if j["id"] == job_id]
    r["status_after_list"] = mine[0]["status"]
    r["exit_code_after_list"] = mine[0]["exit_code"]
    if r["t_report_untouched"] is None:
        open_session(gw, sid)
        r["t_report_after_open"] = wait_for(reported, p.after_open, 1.0)
    if r["t_report_untouched"] is None and r.get("t_report_after_open") is None:
        chat(gw, sid, work, POKE, 120.0)
        time.sleep(p.after_poke)
    # The record is marked a moment after the report is logged; read it after
    # that moment, not in the same instant the entry appeared.
    time.sleep(2)
    r["reports_total"] = count_reports(sess.entries(), job_id)
    rec = sess.records()[job_id]
    r["record_delivered_at_set"] = rec["delivered_at"] is not None
    r["record_returncode"] = rec.get("returncode")
    return r


def scenario_b(gw: Gateway, home: Path, run: str, i: int, p: Pace, rng: random.Random, arm: str) -> dict:
    sid = f"s_e1_{arm.lower()}_{run}_{i:02d}"
    work = mkwork(home, sid)
    marker = f"E1-{arm}-{i}-DONE"
    gw.start()
    sess = Session(home, sid)
    r: dict[str, Any] = {"scenario": arm, "trial": i, "session_id": sid}
    job = start_job(gw, sess, work, p.job_b, marker)
    if not job["found"]:
        r["invalid"] = "the job was not started"
        r.update(job)
        return r
    job_id = job["job_id"]
    r.update(job_id=job_id, job_pid=job["pid"], start_turn_ok=job["start_turn_ok"], start_reply=job["start_reply"])

    def settled() -> bool:
        return sess.records()[job_id].get("finished_at") is not None

    def reported() -> bool:
        return count_reports(sess.entries(), job_id) > 0

    def turn_over() -> bool:
        return http("GET", gw.url(f"/api/chat/runs/active?session_id={sid}"), timeout=10)["run"] is None

    r["t_settle"] = wait_for(settled, 90, 0.1)
    if r["t_settle"] is None:
        r["invalid"] = "the record never settled"
        return r
    settled_at = time.monotonic()
    if arm == "B0":
        r["t_entry"] = wait_for(reported, 240, 0.2)
        if r["t_entry"] is None:
            r["invalid"] = "the report entry never appeared"
            return r
        r["t_turn_done"] = wait_for(turn_over, 240, 0.2)
        if r["t_turn_done"] is None:
            r["invalid"] = "the report turn never finished"
            return r
        time.sleep(3)
    elif arm == "B1":
        r["delta_s"] = rng.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
        time.sleep(r["delta_s"])
    else:
        r["t_entry"] = wait_for(reported, 240, 0.05)
        if r["t_entry"] is None:
            r["invalid"] = "the report entry never appeared"
            return r
    # When the SIGKILL landed, counted from the record settling: the poll
    # interval and the sleep are both part of the requested Δ.
    r["kill_after_settle_s"] = round(time.monotonic() - settled_at, 3)
    gw.kill()
    r["reports_at_kill"] = count_reports(sess.entries(), job_id)
    r["marker_set_at_kill"] = sess.records()[job_id]["delivered_at"] is not None
    r["restart_s"] = gw.start()
    open_session(gw, sid)
    if r["reports_at_kill"] == 0:
        r["t_report_after_restart"] = wait_for(reported, p.after_open, 0.5)
    chat(gw, sid, work, POKE, 120.0)
    time.sleep(p.after_poke)
    r["reports_total"] = count_reports(sess.entries(), job_id)
    r["record_delivered_at_set"] = sess.records()[job_id]["delivered_at"] is not None
    r["claim_file_left"] = (sess.log_path.parent / sid / "background" / f"{job_id}.claim").exists()
    return r


def scenario_c(gw: Gateway, home: Path, run: str, i: int, arm: str) -> dict:
    sid = f"s_e1_{arm.lower()}_{run}_{i:02d}"
    work = mkwork(home, sid)
    gw.start()
    r: dict[str, Any] = {"scenario": arm, "trial": i, "session_id": sid}
    turn = Turn(gw.url("/api/chat"), {
        "message": LIVE, "session_id": sid, "work_dir": str(work), "approval_mode": "allow-all",
    }, 300.0)
    turn.start()
    r["t_first_file"] = wait_for(lambda: (work / "step1.txt").exists(), 240, 0.1)
    if r["t_first_file"] is None:
        r["invalid"] = "step1.txt never appeared"
        return r
    gw.kill()
    r["files_at_kill"] = sorted(f.name for f in work.glob("step*.txt"))
    turn.join(timeout=15)
    entries = Session(home, sid).entries()
    r["tool_entries_at_kill"] = sum(1 for e in entries if e.get("type") == "tool")
    r["restart_s"] = gw.start()
    r["active_run_after_restart"] = (
        http("GET", gw.url(f"/api/chat/runs/active?session_id={sid}"))["run"] is not None
    )
    if arm == "C0":
        open_session(gw, sid)
    sess = Session(home, sid)
    r["sealed_before_message"] = any(e.get("finish_reason") == "interrupted" for e in sess.entries())
    ok, reply = chat(gw, sid, work, POKE, 180.0)
    r["usable_after_restart"] = ok
    r["poke_reply"] = reply
    entries = sess.entries()
    r["sealed_count"] = sum(1 for e in entries if e.get("finish_reason") == "interrupted")
    on_disk = sorted(f.name for f in work.glob("step*.txt"))
    written = files_written(entries)
    r["files_on_disk"] = on_disk
    r["files_created_after_restart"] = len(on_disk) - len(r["files_at_kill"])
    r["files_without_recorded_result"] = [f for f in on_disk if not written[f]]
    return r


def self_test() -> None:
    """The report detector must return 0 for look-alikes and N for N reports."""
    report = {
        "type": "user", "automated": True,
        "content": "[Background terminal #7 (term_7) finished: exit 0 after 3s]\n\nAutomatic report: a "
                   "command you started in the background has finished.",
    }
    orphan = {"type": "user", "automated": True,
              "content": "[Background terminal #7 (term_7) ended while nobody was watching]\nCommand: x"}
    batch = {"type": "user", "automated": True,
             "content": "[2 background terminals finished]\n- #7 (term_7) finished: exit 0\n\n"
                        "Automatic report: 2 command(s) you started in the background have finished"}
    echo = {"type": "assistant", "content": report["content"]}
    tool = {"type": "tool", "content": "Started in background as term_7 (pid 1)."}
    cases = [
        ([tool], 0), ([echo], 0),
        ([report], 1), ([orphan], 1), ([batch], 1),
        ([tool, report], 1), ([report, report], 2), ([batch, orphan], 2),
        ([dict(report, content=report["content"].replace("term_7", "term_8"))], 0),
    ]
    for entries, want in cases:
        got = count_reports(entries, "term_7")
        if got != want:
            sys.exit(f"self-test failed: count_reports -> {got}, expected {want}, for {entries}")
    # files_written: a call is "recorded" only with a successful result row.
    call = {"type": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "write_file", "arguments": '{"file_path": "a.txt", "content": "x"}'}},
        {"id": "c2", "function": {"name": "write_file", "arguments": '{"file_path": "b.txt", "content": "x"}'}},
    ]}
    rows = [{"type": "tool", "tool_call_id": "c1", "is_error": False},
            {"type": "tool", "tool_call_id": "c2", "is_error": True}]
    got = files_written([call] + rows)
    if got != {"a.txt": True, "b.txt": False}:
        sys.exit(f"self-test failed: files_written -> {got}")
    print("self-test ok: report detector, file recorder")


def parse_trials(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, _, n = part.partition("=")
        name = name.strip()
        if name:
            if name not in SCENARIOS or int(n) <= 0:
                sys.exit(f"bad trial spec {part!r}; scenarios are {SCENARIOS}")
            out[name] = int(n)
    return out


def med(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def tally(totals: list[int]) -> str:
    return f"lost={sum(t == 0 for t in totals)} once={sum(t == 1 for t in totals)} duplicate={sum(t >= 2 for t in totals)}"


def summarize(results: list[dict]) -> None:
    print("\n=== E1 summary ===")
    for name in SCENARIOS:
        rs = [r for r in results if r["scenario"] == name]
        if not rs:
            continue
        valid = [r for r in rs if "invalid" not in r and "harness_error" not in r]
        print(f"\n[{name}] trials={len(rs)} valid={len(valid)} not valid={len(rs) - len(valid)}")
        n = len(valid)
        if name == "A" and valid:
            print(f"  job alive right after SIGKILL:        {sum(r['job_alive_after_kill'] for r in valid)}/{n}")
            print(f"  job ended while the gateway was down: {sum(r['job_ended_while_down'] for r in valid)}/{n}")
            print(f"  reports logged while down (control):  {sum(r['reports_while_down'] for r in valid)} (must be 0)")
            print(f"  reported with nobody touching it:     {sum(r['t_report_untouched'] is not None for r in valid)}/{n}")
            print(f"  reported after the session was opened: {sum(r.get('t_report_after_open') is not None for r in valid)}/{n}"
                  f"  median {med([r['t_report_after_open'] for r in valid if r.get('t_report_after_open') is not None])} s")
            print(f"  final report count:                   {tally([r['reports_total'] for r in valid])}")
            statuses: dict[str, int] = {}
            for r in valid:
                statuses[r["status_after_list"]] = statuses.get(r["status_after_list"], 0) + 1
            print(f"  status shown after restart: {statuses}; exit code shown as null: "
                  f"{sum(r['exit_code_after_list'] is None for r in valid)}/{n}; "
                  f"record marked delivered: {sum(r['record_delivered_at_set'] for r in valid)}/{n}")
        if name in ("B0", "B1", "B2") and valid:
            print(f"  final report count: {tally([r['reports_total'] for r in valid])}")
            print(f"  reports at kill 0/1: {sum(r['reports_at_kill'] == 0 for r in valid)}/{sum(r['reports_at_kill'] == 1 for r in valid)}; "
                  f"marker set at kill: {sum(r['marker_set_at_kill'] for r in valid)}/{n}")
            if name == "B0":
                print(f"  median settle -> report entry {med([r['t_entry'] for r in valid])} s; "
                      f"entry -> report turn finished {med([r['t_turn_done'] - r['t_entry'] for r in valid])} s")
            if name == "B1":
                print("  (SIGKILL after settle, reports in log at kill, marker set at kill) -> final report count")
                for r in valid:
                    print(f"    {r['kill_after_settle_s']:.2f}s, {r['reports_at_kill']}, {int(r['marker_set_at_kill'])}"
                          f" -> {r['reports_total']}")
            if name == "B2":
                print(f"  median settle -> report entry: {med([r['t_entry'] for r in valid])} s; "
                      f"entry -> report turn finished {med([r['kill_after_settle_s'] - r['t_entry'] for r in valid])} s at the latest kill")
        if name in ("C0", "C1") and valid:
            print(f"  files on disk at kill: {dict((k, sum(1 for r in valid if len(r['files_at_kill']) == k)) for k in (1, 2, 3))}")
            print(f"  run still listed as active after restart: {sum(r['active_run_after_restart'] for r in valid)}/{n}")
            print(f"  interrupted turn sealed before the new message: {sum(r['sealed_before_message'] for r in valid)}/{n}; "
                  f"sealed exactly once in total: {sum(r['sealed_count'] == 1 for r in valid)}/{n}")
            print(f"  session answered the new message: {sum(r['usable_after_restart'] for r in valid)}/{n}")
            print(f"  leftover task resumed inside the new turn (files created after restart): "
                  f"{sum(r['files_created_after_restart'] > 0 for r in valid)}/{n}")
            print(f"  files on disk with no successful result row: "
                  f"{sum(len(r['files_without_recorded_result']) for r in valid)} across {n} trials")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="one trial per scenario with short waits")
    ap.add_argument("--trials", help="e.g. A=12,B0=3,B1=8,B2=6,C0=6,C1=6")
    ap.add_argument("--profile", default="hy4", help="profile copied from ~/.agentica/config.yaml")
    ap.add_argument("--python", help="interpreter that can import agentica and agentica_gateway")
    ap.add_argument("--gateway-src", default=str(REPO / "gateway" / "src"))
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra env var for the test gateway, e.g. an API key the profile refers to")
    ap.add_argument("--out", help="results file (jsonl)")
    ap.add_argument("--seed", type=int, default=20261128)
    ap.add_argument("--keep-home", action="store_true")
    args = ap.parse_args(argv)

    self_test()
    default = "A=12,B0=3,B1=8,B2=6,C0=6,C1=6"
    if args.smoke:
        default = ",".join(f"{name}=1" for name in SCENARIOS)
    trials = parse_trials(args.trials or default)
    rng = random.Random(args.seed)
    python = resolve_python(args.python)
    home = make_home(args.profile)
    run = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else REPO / "evaluation" / "demo_track" / "outputs" / f"e1-{run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    gw = Gateway(python, Path(args.gateway_src), home, dict(item.split("=", 1) for item in args.env))
    p = Pace(args.smoke)
    results: list[dict] = []
    print(f"home={home}\nout={out}\nprofile={args.profile} python={python}\n"
          f"gateway_src={args.gateway_src} commit={head}\ntrials={trials}", flush=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": True, "run": run, "commit": head, "profile": args.profile,
                             "trials": trials, "smoke": args.smoke, "python": python,
                             "gateway_src": args.gateway_src}) + "\n")
        try:
            for name, count in trials.items():
                for i in range(1, count + 1):
                    t0 = time.monotonic()
                    try:
                        if name == "A":
                            r = scenario_a(gw, home, run, i, p)
                        elif name in ("B0", "B1", "B2"):
                            r = scenario_b(gw, home, run, i, p, rng, name)
                        else:
                            r = scenario_c(gw, home, run, i, name)
                    except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                        r = {"scenario": name, "trial": i, "harness_error": f"{type(exc).__name__}: {exc}"}
                    r["wall_s"] = round(time.monotonic() - t0, 1)
                    results.append(r)
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    brief = {k: v for k, v in r.items() if k not in (
                        "scenario", "trial", "session_id", "job_id", "job_pid", "start_turn_ok", "start_reply",
                        "files_on_disk", "session_files",
                    )}
                    print(name, i, json.dumps(brief, ensure_ascii=False), flush=True)
        finally:
            gw.sweep()
            if not args.keep_home:
                shutil.rmtree(home, ignore_errors=True)
    summarize(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
