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
from uuid import uuid4

from agentica.ticket import current_ticket_id
from agentica.utils.log import logger

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
    # Dispatch this job belongs to (``agentica.ticket``), stamped at ``start()``
    # because the watcher thread that writes the completion record does not
    # inherit the turn's context. Empty means unmanaged — a job started outside
    # any turn, or before tickets existed.
    ticket_id: str = ""
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
    # The durable record this event is about, when the job had one. Carried on
    # the event so a consumer can look up *that* record instead of guessing the
    # path from whatever session it happens to be attached to now — a session
    # switch would otherwise make the event path and the disk sweep disagree
    # and deliver the same report twice.
    record_path: str = ""

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
            # Which dispatch this job belongs to. Empty = unmanaged; readers must
            # treat a missing value as "not tracked", never mint a replacement.
            "ticket_id": item.ticket_id,
            # Delivery is a fact about the *record*, not about this process: the
            # job's report has to be delivered exactly once even when the process
            # that started it is gone by the time it finishes. ``None`` here means
            # "not delivered yet" — see ``deliver_undelivered_jobs``.
            "delivered_at": None,
            "delivery_owner": None,
            "delivery_attempts": 0,
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

    @property
    def records_dir(self) -> Optional[Path]:
        """Where this registry's durable job records live, if it has any.

        Public because reading a session's jobs is not the registry's private
        business: the CLI, the gateway and the IM bridge must all ask the *same*
        question (memory ∪ disk) or one of them shows a subset. Callers pass
        this to ``load_session_job_views`` / ``stop_session_job`` instead of
        reaching into ``_records_dir``.
        """
        with self._lock:
            return self._records_dir

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
            # Stamped here, not in the watcher: this call happens inside the
            # turn, so the ambient ticket is readable; the watcher thread that
            # writes the completion record has no context of its own.
            ticket_id=current_ticket_id() or "",
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
                record_path=item.record_path or "",
            )
            payload = self._record_payload(item, session_id) if item.record_path else None
        if item.record_path and payload is not None:
            payload["returncode"] = int(returncode)
            payload["finished_at"] = completed_at
            # Never clobber a delivery marker. The completion write is the last
            # writer before anything can be delivered, so this can only trigger
            # on a re-watch of an already-settled record — but losing the marker
            # would mean re-delivering a report the user has already seen.
            prior = _read_json_record(Path(item.record_path)) or {}
            for key in ("delivered_at", "delivery_owner", "delivery_attempts"):
                if prior.get(key):
                    payload[key] = prior[key]
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
    tail = read_log_tail(event.log_path, max_lines=20, max_chars=4000)
    # ``agentica watch`` exits 75 when its checks lose contact, and 124 when the
    # job has ended but the result check timed out. Both are about the watch,
    # not an exit code the job itself produced. The log line is the marker;
    # the process status alone says "failed" for either.
    watch_note = ""
    header_status = status
    if "[lost contact:" in tail:
        header_status = "lost contact"
        watch_note = (
            "The checks lost contact (exit 75). The job's state is unknown and it "
            "may still be running; do not report the job as failed. Fix the checks "
            "and watch it again."
        )
    elif "result probe timed out" in tail:
        header_status = "result unknown"
        watch_note = (
            "The watched job ended, but the result check timed out, so its exit "
            "code is unknown. Read the output before deciding whether it succeeded."
        )
    lines = [
        f"[Background terminal #{event.num} ({event.id}) {header_status}: "
        f"{exit_detail(event.returncode)} after {event.elapsed}]",
        f"Command: {event.command}",
        f"Log: {event.log_path}",
    ]
    if watch_note:
        lines.extend(["", watch_note])
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
        "elapsed": item.elapsed,
        "ticket_id": item.ticket_id or "",
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
        "elapsed": _format_elapsed((finished_at if finished_at is not None else time.time()) - started_at),
        "ticket_id": str(rec.get("ticket_id") or ""),
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


# ============================================================================
# Durable delivery
# ============================================================================
#
# A finished background job's report used to be an event in the memory of the
# process that started it (``_completed`` queue -> the owner's poll loop). That
# is fine while the owner is alive and loses the report outright when it is not:
# a job that outlives the CLI that started it simply ends, silently. The record
# on disk always knew the job had finished — nothing ever asked it.
#
# The four ownership rules this section enforces, in the order they bite:
#
# 1. **One writer per fact.** ``delivered_at`` / ``delivery_owner`` are written
#    only by the delivery helpers below; everything else (start, completion,
#    stop, view-synthesis) reads the record and puts the fields back untouched.
#    A fact with two writers is the bug; a fact with one writer and three
#    readers is a cache.
# 2. **"Not yet" is not "no".** ``delivered_at`` is ``None`` until delivery.
#    A surface that renders "no report" from a ``None`` is inventing a terminal
#    value out of a pending one — the same mistake as a session titled "Chat"
#    because its title job hasn't finished.
# 3. **Delivery is idempotent and recorded.** Any process bound to the same
#    ``records_dir`` may adopt an undelivered record, so two of them can race;
#    the ``.claim`` file makes one win, and the marker makes a second delivery
#    visible instead of silent. Exactly-once across a crash inside the delivery
#    window is not achievable with files; at-least-once plus a marker is, and
#    the marker is what lets a reader notice the duplicate.
# 4. **Nothing retries forever.** ``delivery_attempts`` caps a report that
#    raises, so one poisoned record cannot become a 5-second loop in every
#    process that opens the session.
#
# A job the user stopped, or that ended without a return code, has no report to
# deliver. Those records are marked delivered (with no text) rather than left
# permanently "pending" — pending means "someone still owes this", and nobody
# does.

DELIVERY_CLAIM_TTL_SECONDS = 300.0
DELIVERY_MAX_ATTEMPTS = 3
DELIVERY_MAX_AGE_SECONDS = 24 * 60 * 60.0


def delivery_owner(prefix: str = "") -> str:
    """Identify this process in a record's ``delivery_owner`` field."""
    label = f"{prefix}:" if prefix else ""
    return f"{label}pid:{os.getpid()}"


def _claim_path(record_path: Path) -> Path:
    # ``.claim``, not ``.json``: ``load_session_job_views`` globs ``*.json`` and
    # a claim file must never be mistaken for a job record.
    return record_path.with_suffix(".claim")


def _read_claim(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    """Plain "does this pid exist", for claim liveness.

    Deliberately NOT ``is_job_pid_alive``: that one also asserts the pid leads
    its own process group, because every *job* is started with
    ``start_new_session=True``. A delivery claimer is an ordinary process (the
    gateway, the CLI), so reusing the job predicate here reports every live
    claim as stale and quietly turns the lock off.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _claim_is_stale(claim: Dict[str, Any], ttl: float, *, age: float) -> bool:
    """A claim is abandoned when its *owner* is gone, not when it got old.

    Liveness of the owner pid is the primary signal. An age-based TTL alone was
    wrong in the one case that matters: a delivery legitimately waits on a
    terminal that is mid-turn (the attach call's timeout is minutes), so a TTL
    shorter than that turned the claim into a second delivery of the same
    report — the duplicate this module exists to prevent, produced by its own
    expiry. The TTL therefore only bounds a claim whose owner cannot be read at
    all (an empty or half-written file).
    """
    try:
        pid = int(claim.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid > 0:
        return not _pid_is_alive(pid)
    return age > ttl


def claim_job_delivery(
    record_path: Path | str,
    owner: str,
    *,
    ttl: float = DELIVERY_CLAIM_TTL_SECONDS,
) -> bool:
    """Take the exclusive right to deliver one job's report.

    ``O_CREAT | O_EXCL`` on ``<job>.claim`` is the whole lock: it is atomic on
    every filesystem this project supports, needs no daemon, and leaves the
    evidence behind when a process dies mid-delivery (the TTL then lets the next
    process adopt the record instead of losing it).
    """
    path = Path(record_path)
    claim_file = _claim_path(path)
    payload = {"owner": owner, "pid": os.getpid(), "at": time.time()}
    for _ in range(3):
        try:
            fd = os.open(claim_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = _read_claim(claim_file)
            try:
                age = time.time() - claim_file.stat().st_mtime
            except OSError:
                age = 0.0
            if existing is None and age <= ttl:
                # Created but not written yet (the O_CREAT .. os.write window),
                # or written short. Not abandoned — wait for the TTL.
                return False
            if existing is not None and not _claim_is_stale(existing, ttl, age=age):
                return False
            # Steal by rename: ``os.replace`` is atomic, so of N processes that
            # all decided this claim is dead exactly one wins. Unlinking first
            # let two stealers delete each other's fresh claim and both return
            # True, which is a second delivery of the same report.
            stolen = claim_file.with_name(
                f"{claim_file.name}.stale.{os.getpid()}.{uuid4().hex[:6]}"
            )
            try:
                os.replace(claim_file, stolen)
            except OSError:
                continue
            try:
                stolen.unlink()
            except OSError:
                pass
            continue
        except OSError:
            return False
        try:
            os.write(fd, json.dumps(payload).encode("utf-8"))
        finally:
            os.close(fd)
        return True
    return False


def release_job_claim(record_path: Path | str) -> None:
    """Give up a claim without marking the job delivered (a retry will re-claim)."""
    try:
        _claim_path(Path(record_path)).unlink()
    except OSError:
        pass


def mark_job_delivered(record_path: Path | str, owner: str) -> bool:
    """Record that this job's report reached its destination. Idempotent.

    The write is verified by reading it back. ``_write_json_atomic`` swallows
    ``OSError`` by design, and an unverified marker means the very next sweep
    offers the same report again — a duplicate produced by the mechanism that
    exists to prevent duplicates. So when the marker cannot be persisted the
    claim is deliberately kept: the report *was* delivered, and a live owner's
    claim never expires (staleness is pid liveness, not age), which is what
    stops this process from sending it twice. A process that starts after this
    one exits will find the claim stale and deliver once more — at-least-once,
    with ``delivery_owner`` left on the record to show who already sent it.
    """
    path = Path(record_path)
    rec = _read_json_record(path)
    if rec is None:
        return False
    if rec.get("delivered_at"):
        release_job_claim(path)
        return True
    rec["delivered_at"] = time.time()
    rec["delivery_owner"] = owner
    _write_json_atomic(path, rec)
    fresh = _read_json_record(path)
    if not (fresh and fresh.get("delivered_at")):
        logger.warning(
            f"could not record delivery of {path.name}; keeping the claim so this "
            f"process does not deliver it again"
        )
        return False
    release_job_claim(path)
    return True


def note_job_delivery_failure(record_path: Path | str) -> None:
    path = Path(record_path)
    rec = _read_json_record(path)
    if rec is None:
        return
    try:
        attempts = int(rec.get("delivery_attempts") or 0)
    except (TypeError, ValueError):
        attempts = 0
    rec["delivery_attempts"] = attempts + 1
    _write_json_atomic(path, rec)


def completed_from_record(rec: Dict[str, Any]) -> Optional[BackgroundProcessCompleted]:
    """Rebuild the completion event a record describes, or ``None`` if it can't.

    Lets a report be rendered identically by the process that ran the job and by
    one that merely found the record later — one formatter, two callers.
    """
    finished_at = rec.get("finished_at")
    returncode = rec.get("returncode")
    if finished_at is None or returncode is None:
        return None
    try:
        return BackgroundProcessCompleted(
            id=str(rec.get("id") or ""),
            num=int(rec.get("num") or 0),
            pid=int(rec.get("pid") or 0),
            command=str(rec.get("command") or ""),
            cwd=str(rec.get("cwd") or "") or None,
            log_path=str(rec.get("log_path") or ""),
            started_at=float(rec.get("started_at") or 0.0),
            completed_at=float(finished_at),
            returncode=int(returncode),
            kind=str(rec.get("kind") or "command"),
            label=str(rec.get("label") or ""),
            stop_requested=bool(rec.get("stop_requested")),
        )
    except (TypeError, ValueError):
        return None


def _orphan_report(rec: Dict[str, Any]) -> str:
    """Report a job that ended while the process that started it was gone.

    Nobody was left to record its exit code. Saying that plainly, with the tail
    of the log, is the report: the alternative is what this module was written
    to stop — a job that ends with nobody told.
    """
    tail = read_log_tail(str(rec.get("log_path") or ""), max_lines=20, max_chars=4000)
    lines = [
        f"[Background terminal #{rec.get('num')} ({rec.get('id')}) ended while nobody was watching]",
        f"Command: {rec.get('command') or ''}",
        f"Log: {rec.get('log_path') or ''}",
        "",
        "The process that started this job is gone, so its exit code was never "
        "recorded and it is no longer running. Read the output before deciding "
        "whether it succeeded.",
    ]
    if tail:
        lines.extend(["", "Output tail:", tail])
    return "\n".join(lines)


def background_report_from_record(rec: Dict[str, Any]) -> Optional[str]:
    """Render a durable record as the agent-facing report.

    ``None`` means the job genuinely has nothing to say: the user stopped it, or
    it has not settled. A job that settled without an exit code (its owner died
    first) still owes a report, and gets :func:`_orphan_report`.
    """
    if rec.get("stop_requested"):
        return None
    if rec.get("finished_at") is None:
        return None
    event = completed_from_record(rec)
    if event is None:
        return _orphan_report(rec)
    return background_result_for_agent(event)


def record_is_settled(records_dir: Optional[Path], job_id: str) -> bool:
    """True when the durable record for ``job_id`` knows the job finished.

    The completion write is best-effort (``_write_json_atomic`` swallows disk
    errors), so "a record file exists" is not the same as "the record is
    usable": a start-time record whose completion write failed has no
    ``finished_at`` and can never be adopted by the sweep. A caller holding the
    in-memory completion event is the only one that can still deliver it.
    """
    if records_dir is None or not job_id:
        return False
    return record_file_is_settled(records_dir / f"{job_id}.json")


def record_file_is_settled(record_path: Optional[Path | str]) -> bool:
    """Same question as :func:`record_is_settled`, for a known record path.

    A completion event carries the path of *its own* record, which is not
    necessarily under the records dir the consumer is bound to now (the user may
    have switched sessions). Asking about the real file is what keeps the event
    path and the sweep disjoint.
    """
    if not record_path:
        return False
    rec = _read_json_record(Path(record_path))
    return bool(rec and rec.get("finished_at") is not None)


def undelivered_job_records(
    records_dir: Optional[Path],
    *,
    max_age_seconds: float = DELIVERY_MAX_AGE_SECONDS,
) -> List[tuple[Path, Dict[str, Any]]]:
    """Completed records whose report nobody has delivered yet.

    ``max_age_seconds`` is a horizon, not a policy decision: a report from a job
    that finished a week ago is noise, and re-opening a session must not replay
    a backlog of them. Within the horizon an undelivered report is owed.
    """
    if records_dir is None or not records_dir.is_dir():
        return []
    now = time.time()
    pending: List[tuple[Path, Dict[str, Any]]] = []
    for path in sorted(records_dir.glob("*.json")):
        rec = _read_json_record(path)
        if rec is None:
            continue
        # ``delivered_at`` is written by every record this module creates. A
        # record that does not have the key at all was written before delivery
        # state existed: its report was delivered (or lost) by the old in-memory
        # path and there is no way to tell which. Treat it as settled — the same
        # rule as "a missing ticket means unmanaged": a legacy record must not
        # be replayed as newly owed, and must not be given a made-up state.
        if "delivered_at" not in rec or rec.get("delivered_at"):
            continue
        finished_at = rec.get("finished_at")
        if finished_at is None:
            continue
        try:
            settled = float(finished_at)
            attempts = int(rec.get("delivery_attempts") or 0)
        except (TypeError, ValueError):
            continue
        if now - settled > max_age_seconds:
            continue
        if attempts >= DELIVERY_MAX_ATTEMPTS:
            continue
        pending.append((path, rec))
    return pending


def deliver_undelivered_jobs(
    records_dir: Optional[Path],
    deliver: Callable[[Dict[str, Any], str], None],
    *,
    owner: str = "",
    max_age_seconds: float = DELIVERY_MAX_AGE_SECONDS,
    claim_ttl: float = DELIVERY_CLAIM_TTL_SECONDS,
    max_reports: int = 5,
) -> int:
    """Deliver every finished-but-unreported job under ``records_dir`` once.

    Returns how many reports were handed to ``deliver``. Safe to call from any
    process that shares the directory, on bind and on a poll: claiming makes the
    passes disjoint, and the marker makes a second pass a no-op.
    """
    who = owner or delivery_owner()
    delivered = 0
    for path, rec in undelivered_job_records(records_dir, max_age_seconds=max_age_seconds):
        # A session opened after some downtime can owe several reports at once.
        # Delivering them in one burst would queue that many agent turns; the
        # caller's next pass picks up the rest.
        if delivered >= max_reports:
            break
        if not claim_job_delivery(path, who, ttl=claim_ttl):
            continue
        try:
            report = background_report_from_record(rec)
            if report is None:
                # Settled with nothing to say (stopped, or no exit code): mark it
                # so it stops reading as "someone still owes this".
                mark_job_delivered(path, who)
                continue
            deliver(rec, report)
        except Exception as exc:  # noqa: BLE001 - one poisoned record must not stop the sweep
            logger.debug(f"background report delivery failed for {path.name}: {exc}")
            note_job_delivery_failure(path)
            release_job_claim(path)
            continue
        mark_job_delivered(path, who)
        delivered += 1
    return delivered
