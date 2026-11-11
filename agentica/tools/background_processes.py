# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Background OS process registry for long-running execute commands.
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SAFE_JS_STAMP_MOD = 4_000_000_000_000_000


def _projects_root() -> Path:
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    projects_dir = os.path.expanduser(os.getenv("AGENTICA_PROJECTS_DIR", os.path.join(home, "projects")))
    return Path(projects_dir)


def _activity_stamp_file(user_id: Optional[str]) -> Path:
    from agentica.compression.tool_result_storage import safe_user_segment

    return _projects_root() / safe_user_segment(user_id) / ".pulse"


def touch_activity_stamp(user_id: Optional[str] = None) -> None:
    """Bump the per-account cross-process activity stamp (1 stat on read)."""
    path = _activity_stamp_file(user_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def read_activity_stamp_us(user_id: Optional[str] = None) -> int:
    """Microsecond mtime of the per-account activity stamp (JS safe integer)."""
    try:
        ns = _activity_stamp_file(user_id).stat().st_mtime_ns
    except OSError:
        return 0
    return (ns // 1000) % _SAFE_JS_STAMP_MOD


def is_job_pid_alive(pid: int) -> bool:
    """True when detached background job ``pid`` is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    if os.name != "nt":
        try:
            # Every background job is spawned with start_new_session=True
            # (setsid), so its process group id equals its own pid.
            return os.getpgid(pid) == pid
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return True


def _background_root(cwd: Optional[str], user_id: Optional[str]) -> Path:
    """Return the project-scoped directory for background command logs."""
    from agentica.compression.tool_result_storage import safe_user_segment, sanitize_path

    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    projects_dir = os.path.expanduser(os.getenv("AGENTICA_PROJECTS_DIR", os.path.join(home, "projects")))
    real_cwd = os.path.realpath(os.path.expanduser(cwd or os.getcwd()))
    return Path(projects_dir) / safe_user_segment(user_id) / sanitize_path(real_cwd) / "background"


def read_log_tail(log_path: str, max_lines: int = 5, max_chars: int = 2000) -> str:
    """Return the tail of a background command log, without the ``$ cmd`` header."""
    path = Path(log_path)
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_chars * 4:
                f.seek(max(0, size - max_chars * 4))
            data = f.read()
    except OSError:
        return ""

    text = data.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("$ "):
        lines = lines[1:]
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def read_last_log_line(log_path: str, max_chars: int = 200) -> str:
    """Latest non-empty output line (ANSI stripped) for one-line live progress."""
    raw = read_log_tail(log_path, max_lines=3, max_chars=max(max_chars * 3, 600))
    if not raw:
        return ""
    clean = _ANSI_ESCAPE_RE.sub("", raw)
    lines = [" ".join(ln.split()) for ln in clean.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("$ ")]
    if not lines:
        return ""
    last = lines[-1]
    return (last[: max_chars - 1] + "…") if len(last) > max_chars else last


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _read_json_record(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def _format_elapsed(seconds: float) -> str:
    """Duration, not a clock: ``5m13s``, ``5m3s``, ``3m18s``. No leading zeros."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        text = f"{hours}h"
        if minutes:
            text += f"{minutes}m"
        if secs:
            text += f"{secs}s"
        return text
    if minutes:
        return f"{minutes}m{secs}s" if secs else f"{minutes}m"
    return f"{secs}s"


@dataclass
class BackgroundProcess:
    """One shell command running outside the agent turn.

    ``kind`` says what started it, which decides how it is counted and how it is
    reported when it ends: ``"command"`` is execute(background=True), ``"delegate"``
    is a whole agentica session working on ``label`` (see the delegate tool).
    """

    id: str
    num: int
    process: subprocess.Popen
    command: str
    cwd: Optional[str]
    log_path: str
    started_at: float
    kind: str = "command"
    label: str = ""
    stop_requested: bool = False
    finished_at: Optional[float] = None
    record_path: Optional[str] = None
    # Set by the watcher thread once the process is reaped. Callers that need to
    # block until the command is done wait on this instead of racing the watcher
    # for the same waitpid.
    finished: threading.Event = field(default_factory=threading.Event)

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.returncode is None

    @property
    def elapsed(self) -> str:
        return _format_elapsed((self.finished_at or time.time()) - self.started_at)


@dataclass(frozen=True)
class BackgroundProcessCompleted:
    """Completion event emitted exactly once for a detached shell command."""

    id: str
    num: int
    pid: int
    command: str
    cwd: Optional[str]
    log_path: str
    started_at: float
    completed_at: float
    returncode: int
    kind: str = "command"
    label: str = ""
    stop_requested: bool = False

    @property
    def elapsed(self) -> str:
        return _format_elapsed(self.completed_at - self.started_at)


class BackgroundProcessRegistry:
    """Thread-safe registry shared by execute(), /ps, /stop and the status bar."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        delivers_results: bool = False,
        *,
        session_id: Optional[str] = None,
        records_dir: Optional[Path] = None,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._counter = 0
        self._items: dict[str, BackgroundProcess] = {}
        self._completed: queue.Queue[BackgroundProcessCompleted] = queue.Queue()
        self._user_id = user_id
        self._session_id: Optional[str] = None
        self._records_dir: Optional[Path] = None
        self._on_change = on_change
        # Whether the owner drains completions back to the agent as a new
        # message (the interactive CLI does, per `deliver_background_results`).
        # Off by default: a registry nobody drains delivers nothing. The execute
        # tool reads it to tell the model where a finished command's result
        # goes, so it must be set before that tool is built.
        self.delivers_results = delivers_results
        if session_id and records_dir is not None:
            self.bind_session(session_id, records_dir)

    def bind_session(self, session_id: str, records_dir: Path | str) -> None:
        """Bind durable per-session job records so Desktop/Web/IM/CLI share state."""
        rdir = Path(records_dir)
        max_existing = 0
        if rdir.is_dir():
            for p in rdir.glob("term_*.json"):
                stem = p.stem
                suffix = stem.removeprefix("term_")
                if suffix.isdigit():
                    max_existing = max(max_existing, int(suffix))
        with self._lock:
            self._session_id = session_id
            self._records_dir = rdir
            if max_existing > self._counter:
                self._counter = max_existing

    def set_on_change(self, on_change: Optional[Callable[[], None]]) -> None:
        with self._lock:
            self._on_change = on_change

    def _notify_change(self) -> None:
        with self._lock:
            cb = self._on_change
            uid = self._user_id
            has_records = self._records_dir is not None
        if has_records:
            touch_activity_stamp(uid)
        if cb is not None:
            cb()

    @staticmethod
    def _record_payload(item: BackgroundProcess, session_id: Optional[str]) -> Dict[str, Any]:
        rc = item.returncode if item.finished.is_set() else None
        return {
            "id": item.id,
            "num": item.num,
            "pid": item.pid,
            "owner_pid": os.getpid(),
            "session_id": session_id or "",
            "kind": item.kind,
            "label": item.label,
            "command": item.command,
            "cwd": item.cwd or "",
            "log_path": item.log_path,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
            "returncode": rc,
            "stop_requested": item.stop_requested,
        }

    def set_user_id(self, user_id: Optional[str]) -> None:
        """Set the user segment used for future background command logs."""
        with self._lock:
            self._user_id = user_id

    @property
    def user_id(self) -> Optional[str]:
        with self._lock:
            return self._user_id

    def start(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        kind: str = "command",
        label: str = "",
        argv: Optional[List[str]] = None,
    ) -> BackgroundProcess:
        """Spawn ``command`` detached from the turn.

        ``env`` is merged over the current environment (it does not replace it),
        so a caller only names what it wants to change.

        ``argv``, when given, is what actually runs (no shell) while ``command``
        stays what the log header and ``/ps`` show — the OS sandbox wrapper is
        an implementation detail of how the command ran, not the command.
        """
        with self._lock:
            if self._records_dir is not None and self._records_dir.is_dir():
                for p in self._records_dir.glob("term_*.json"):
                    suffix = p.stem.removeprefix("term_")
                    if suffix.isdigit():
                        self._counter = max(self._counter, int(suffix))
            self._counter += 1
            num = self._counter
            proc_id = f"term_{num}"
            user_id = self._user_id
            session_id = self._session_id
            record_path = str(self._records_dir / f"{proc_id}.json") if self._records_dir is not None else None

        root = _background_root(cwd, user_id)
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = root / f"{stamp}-{proc_id}.log"
        log_fh = open(log_path, "ab", buffering=0)
        try:
            # One physical line, always: readers strip the header by dropping a
            # leading `$ ` line, and a command that spans lines (a delegated
            # worker's whole task does) would otherwise leak into the output.
            header = f"$ {command}".replace("\n", "\\n") + "\n\n"
            header = header.encode("utf-8", errors="replace")
            log_fh.write(header)
            process = subprocess.Popen(
                argv if argv is not None else command,
                shell=argv is None,
                cwd=cwd,
                env={**os.environ, **env} if env else None,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name != "nt",
            )
        except Exception:
            log_fh.close()
            log_path.unlink(missing_ok=True)
            raise
        finally:
            if not log_fh.closed:
                log_fh.close()

        item = BackgroundProcess(
            id=proc_id,
            num=num,
            process=process,
            command=command,
            cwd=cwd,
            log_path=str(log_path),
            started_at=time.time(),
            kind=kind,
            label=label,
            record_path=record_path,
        )
        with self._lock:
            self._items[proc_id] = item
        if record_path is not None:
            _write_json_atomic(Path(record_path), self._record_payload(item, session_id))
        self._notify_change()
        threading.Thread(
            target=self._watch_process,
            args=(item, session_id),
            daemon=True,
            name=f"{proc_id}_watcher",
        ).start()
        return item

    def _watch_process(self, item: BackgroundProcess, session_id: Optional[str] = None) -> None:
        returncode = item.process.wait()
        completed_at = time.time()
        with self._lock:
            item.finished_at = completed_at
            item.finished.set()
            event = BackgroundProcessCompleted(
                id=item.id,
                num=item.num,
                pid=item.pid,
                command=item.command,
                cwd=item.cwd,
                log_path=item.log_path,
                started_at=item.started_at,
                completed_at=completed_at,
                returncode=int(returncode),
                kind=item.kind,
                label=item.label,
                stop_requested=item.stop_requested,
            )
            payload = self._record_payload(item, session_id) if item.record_path else None
        if item.record_path and payload is not None:
            payload["returncode"] = int(returncode)
            payload["finished_at"] = completed_at
            _write_json_atomic(Path(item.record_path), payload)
        self._completed.put(event)
        self._notify_change()

    def wait_completed(self, timeout: Optional[float] = None) -> BackgroundProcessCompleted:
        """Wait for the next completed command emitted by a watcher."""
        return self._completed.get(timeout=timeout)

    def list(self, *, include_finished: bool = False, kind: Optional[str] = None) -> List[BackgroundProcess]:
        with self._lock:
            items = list(self._items.values())
        if kind is not None:
            items = [item for item in items if item.kind == kind]
        if include_finished:
            return items
        return [item for item in items if item.running]

    def running_count(self, *, kind: Optional[str] = None) -> int:
        return len(self.list(include_finished=False, kind=kind))

    def _matches(self, item: BackgroundProcess, target: str) -> bool:
        target = target.strip()
        if not target:
            return False
        return target in {item.id, str(item.num), f"#{item.num}", str(item.pid)}

    def get(self, target: str) -> Optional[BackgroundProcess]:
        """Look up one command by id, number or pid. Finished ones stay findable."""
        with self._lock:
            items = list(self._items.values())
        for item in items:
            if self._matches(item, target):
                return item
        return None

    def stop(self, target: Optional[str] = None) -> List[BackgroundProcess]:
        with self._lock:
            items = list(self._items.values())
        if target and target.lower() not in {"all", "*"}:
            items = [item for item in items if self._matches(item, target)]
        stopped: List[BackgroundProcess] = []
        for item in items:
            with self._lock:
                if not item.running:
                    continue
                # Mark the stop intent while holding the same lock the watcher
                # uses to snapshot its completion event. If the process exits
                # after this point, the event is reliably classified as an
                # explicit stop rather than a command failure.
                item.stop_requested = True
                session_id = self._session_id
            if item.record_path:
                _write_json_atomic(Path(item.record_path), self._record_payload(item, session_id))
            if os.name != "nt":
                try:
                    os.killpg(item.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                item.process.terminate()
            try:
                item.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(item.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    item.process.kill()
                try:
                    item.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            stopped.append(item)
        return stopped


# Signals that ask a process to stop (`kill`, `pkill`, Ctrl+C). A command
# ended by one of these was stopped, it did not fail.
# SIGKILL is deliberately absent: it is also how the OOM killer ends a job.
_STOP_SIGNALS = {signal.SIGTERM, signal.SIGINT}


def exit_status(returncode: int) -> str:
    """``finished`` / ``stopped`` / ``failed`` from a Popen return code.

    Popen reports death-by-signal as ``-signum``.
    """
    if returncode == 0:
        return "finished"
    if returncode < 0 and -returncode in _STOP_SIGNALS:
        return "stopped"
    return "failed"


def exit_detail(returncode: int) -> str:
    """``exit 1`` for a normal exit, ``SIGTERM`` for death by signal."""
    if returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            pass
    return f"exit {returncode}"


def background_result_for_agent(event: BackgroundProcessCompleted) -> str:
    """Render a finished background command as a report for the agent."""
    status = exit_status(event.returncode)
    if event.kind == "delegate":
        return _delegate_result_for_agent(event, status)
    lines = [
        f"[Background terminal #{event.num} ({event.id}) {status}: "
        f"{exit_detail(event.returncode)} after {event.elapsed}]",
        f"Command: {event.command}",
        f"Log: {event.log_path}",
    ]
    tail = read_log_tail(event.log_path, max_lines=20, max_chars=4000)
    if tail:
        lines.extend(["", "Output tail:", tail])
    lines.extend(
        [
            "",
            "This is an automatic report of a command you started in the background. "
            "Pick the work back up where that command left off, or report the outcome "
            "to the user if nothing is left to do.",
        ]
    )
    return "\n".join(lines)


def _delegate_result_for_agent(event: BackgroundProcessCompleted, status: str) -> str:
    """Render a finished delegated session as a report for the agent that sent it.

    The worker ran with ``--print``, so its stdout is its final answer and not a
    command log — it gets a much larger slice than a background command's tail,
    because that answer is the entire deliverable.
    """
    answer = read_log_tail(event.log_path, max_lines=120, max_chars=8000)
    lines = [f'[Delegated task "{event.label}" {status} after {event.elapsed} ({event.id})]']
    if event.returncode == 0:
        lines.extend(["", "Its report:", answer or "(the worker produced no output)"])
        lines.extend(
            [
                "",
                "This is the whole of what that session hands back. Fold it into your "
                "own work, or report it to the user if nothing is left to do.",
            ]
        )
    else:
        lines.extend(
            [
                f"Exit code: {event.returncode}",
                f"Log: {event.log_path}",
                "",
                "Output tail:",
                answer or "(no output)",
            ]
        )
        lines.extend(
            [
                "",
                "The delegated session did not finish its task. Decide whether to do "
                "the work yourself or tell the user why it failed — do not simply "
                "delegate the same task again.",
            ]
        )
    return "\n".join(lines)


def background_job_view_from_process(item: BackgroundProcess) -> Dict[str, Any]:
    """Client-facing projection of an in-process ``BackgroundProcess``."""
    if not item.finished.is_set():
        status = "running"
    elif item.stop_requested:
        status = "stopped"
    else:
        rc = item.returncode
        if rc is None:
            status = "stopped"
        elif rc == 0:
            status = "completed"
        elif exit_status(rc) == "stopped":
            status = "stopped"
        else:
            status = "failed"
    display_label = item.label.strip() if item.label else ""
    if not display_label:
        display_label = item.command
    return {
        "id": item.id,
        "num": item.num,
        "pid": item.pid,
        "kind": item.kind,
        "label": display_label,
        "command": "" if item.kind == "delegate" else item.command,
        "cwd": item.cwd or "",
        "log_path": item.log_path,
        "status": status,
        "exit_code": item.returncode if item.finished.is_set() else None,
        "progress": read_last_log_line(item.log_path) if status == "running" else "",
        "started_at_ms": int(item.started_at * 1000),
        "finished_at_ms": int(item.finished_at * 1000) if item.finished_at else None,
    }


def _infer_settled_ts(log_path: str, record_path: Path, started_at: float) -> float:
    for candidate in (Path(log_path) if log_path else None, record_path):
        if candidate is None:
            continue
        try:
            mt = candidate.stat().st_mtime
            if mt >= started_at:
                return mt
        except OSError:
            continue
    return max(started_at, time.time())


def _view_from_disk_record(record_path: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
    started_at = float(rec.get("started_at") or 0.0)
    finished_at = rec.get("finished_at")
    pid = int(rec.get("pid") or 0)
    log_path = str(rec.get("log_path") or "")
    stop_requested = bool(rec.get("stop_requested"))
    returncode = rec.get("returncode")

    if finished_at is None:
        if is_job_pid_alive(pid):
            status = "running"
        else:
            fresh = _read_json_record(record_path)
            if fresh is not None and fresh.get("finished_at") is not None:
                rec = fresh
                finished_at = float(rec["finished_at"])
                stop_requested = bool(rec.get("stop_requested"))
                returncode = rec.get("returncode")
            else:
                finished_at = _infer_settled_ts(log_path, record_path, started_at)
                rec = dict(rec, finished_at=finished_at)
                _write_json_atomic(record_path, rec)

    if finished_at is not None:
        finished_at = float(finished_at)
        if stop_requested or returncode is None:
            status = "stopped"
        else:
            rc_int = int(returncode)
            returncode = rc_int
            if rc_int == 0:
                status = "completed"
            elif exit_status(rc_int) == "stopped":
                status = "stopped"
            else:
                status = "failed"

    kind = str(rec.get("kind") or "command")
    raw_cmd = str(rec.get("command") or "")
    raw_label = str(rec.get("label") or "").strip()
    display_label = raw_label or raw_cmd
    return {
        "id": str(rec.get("id") or record_path.stem),
        "num": int(rec.get("num") or 0),
        "pid": pid,
        "kind": kind,
        "label": display_label,
        "command": "" if kind == "delegate" else raw_cmd,
        "cwd": str(rec.get("cwd") or ""),
        "log_path": log_path,
        "status": status,
        "exit_code": int(returncode) if (finished_at is not None and returncode is not None) else None,
        "progress": read_last_log_line(log_path) if status == "running" else "",
        "started_at_ms": int(started_at * 1000),
        "finished_at_ms": int(finished_at * 1000) if finished_at is not None else None,
    }


def load_session_job_views(
    records_dir: Optional[Path],
    registry: Optional[BackgroundProcessRegistry] = None,
) -> List[Dict[str, Any]]:
    """Merge live in-memory jobs with durable per-session job records on disk."""
    views: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    if registry is not None:
        for item in registry.list(include_finished=True):
            views.append(background_job_view_from_process(item))
            seen_ids.add(item.id)
    if records_dir is not None and records_dir.is_dir():
        for path in sorted(records_dir.glob("*.json")):
            rec = _read_json_record(path)
            if rec is None:
                continue
            jid = str(rec.get("id") or path.stem)
            if jid in seen_ids:
                continue
            views.append(_view_from_disk_record(path, rec))
            seen_ids.add(jid)
    views.sort(key=lambda v: (int(v.get("started_at_ms") or 0), int(v.get("num") or 0)))
    return views


def count_running_session_jobs(
    records_dir: Optional[Path],
    registry: Optional[BackgroundProcessRegistry] = None,
) -> int:
    """Count running background jobs for a session (0-cost when no records exist)."""
    has_dir = records_dir is not None and records_dir.is_dir()
    if registry is None and not has_dir:
        return 0
    return sum(1 for v in load_session_job_views(records_dir, registry) if v["status"] == "running")


def _matches_record_target(rec: Dict[str, Any], target: str) -> bool:
    t = target.strip()
    if not t:
        return False
    num_s = str(rec.get("num") or "")
    pid_s = str(rec.get("pid") or "")
    return t in {str(rec.get("id") or ""), num_s, f"#{num_s}", pid_s}


def _kill_detached_pid(pid: int) -> None:
    if pid <= 0 or not is_job_pid_alive(pid):
        return
    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not is_job_pid_alive(pid):
            return
        time.sleep(0.05)
    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and is_job_pid_alive(pid):
        time.sleep(0.05)


def stop_session_job(
    records_dir: Optional[Path],
    target: str,
    registry: Optional[BackgroundProcessRegistry] = None,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Stop one background job by id / #num / pid across processes."""
    clean = target.strip()
    if not clean or clean.lower() in {"all", "*"}:
        return None
    if registry is not None:
        item = registry.get(clean)
        if item is not None:
            registry.stop(item.id)
            touch_activity_stamp(user_id)
            return background_job_view_from_process(item)
    if records_dir is None or not records_dir.is_dir():
        return None
    for path in sorted(records_dir.glob("*.json")):
        rec = _read_json_record(path)
        if rec is None or not _matches_record_target(rec, clean):
            continue
        pid = int(rec.get("pid") or 0)
        if rec.get("finished_at") is None:
            rec["stop_requested"] = True
            _write_json_atomic(path, rec)
            _kill_detached_pid(pid)
            fresh = _read_json_record(path) or rec
            if fresh.get("finished_at") is None and not is_job_pid_alive(pid):
                fresh["stop_requested"] = True
                fresh["finished_at"] = time.time()
                _write_json_atomic(path, fresh)
            rec = fresh
        touch_activity_stamp(user_id)
        return _view_from_disk_record(path, rec)
    return None


def stop_all_session_jobs(
    records_dir: Optional[Path],
    registry: Optional[BackgroundProcessRegistry] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stop every running background job belonging to this session."""
    stopped_views: List[Dict[str, Any]] = []
    seen: set[str] = set()
    if registry is not None:
        for item in registry.stop("all"):
            stopped_views.append(background_job_view_from_process(item))
            seen.add(item.id)
    if records_dir is not None and records_dir.is_dir():
        for path in sorted(records_dir.glob("*.json")):
            rec = _read_json_record(path)
            if rec is None:
                continue
            jid = str(rec.get("id") or path.stem)
            if jid in seen or rec.get("finished_at") is not None:
                continue
            pid = int(rec.get("pid") or 0)
            if not is_job_pid_alive(pid):
                continue
            rec["stop_requested"] = True
            _write_json_atomic(path, rec)
            _kill_detached_pid(pid)
            fresh = _read_json_record(path) or rec
            if fresh.get("finished_at") is None and not is_job_pid_alive(pid):
                fresh["stop_requested"] = True
                fresh["finished_at"] = time.time()
                _write_json_atomic(path, fresh)
            stopped_views.append(_view_from_disk_record(path, fresh))
    if stopped_views:
        touch_activity_stamp(user_id)
    return stopped_views

