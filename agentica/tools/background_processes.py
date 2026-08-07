# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Background OS process registry for long-running execute commands.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _background_root(cwd: Optional[str], user_id: Optional[str]) -> Path:
    """Return the project-scoped directory for background command logs."""
    from agentica.compression.tool_result_storage import safe_user_segment, sanitize_path

    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    projects_dir = os.path.expanduser(os.getenv("AGENTICA_PROJECTS_DIR", os.path.join(home, "projects")))
    real_cwd = os.path.realpath(os.path.expanduser(cwd or os.getcwd()))
    return Path(projects_dir) / safe_user_segment(user_id) / sanitize_path(real_cwd) / "background"


def _shorten_command(command: str, limit: int = 90) -> str:
    text = " ".join(command.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


@dataclass
class BackgroundProcess:
    """One shell command running outside the agent turn."""

    id: str
    num: int
    process: subprocess.Popen
    command: str
    cwd: Optional[str]
    log_path: str
    started_at: float
    stop_requested: bool = False
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
        return _format_elapsed(time.time() - self.started_at)

    @property
    def preview(self) -> str:
        return _shorten_command(self.command)


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
    stop_requested: bool = False

    @property
    def elapsed(self) -> str:
        return _format_elapsed(self.completed_at - self.started_at)

    @property
    def preview(self) -> str:
        return _shorten_command(self.command)


class BackgroundProcessRegistry:
    """Thread-safe registry shared by execute(), /ps, /stop and the status bar."""

    def __init__(self, user_id: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._counter = 0
        self._items: dict[str, BackgroundProcess] = {}
        self._completed: queue.Queue[BackgroundProcessCompleted] = queue.Queue()
        self._user_id = user_id

    def set_user_id(self, user_id: Optional[str]) -> None:
        """Set the user segment used for future background command logs."""
        with self._lock:
            self._user_id = user_id

    @property
    def user_id(self) -> Optional[str]:
        with self._lock:
            return self._user_id

    def start(self, command: str, *, cwd: Optional[str] = None) -> BackgroundProcess:
        with self._lock:
            self._counter += 1
            num = self._counter
            proc_id = f"term_{num}"
            user_id = self._user_id

        root = _background_root(cwd, user_id)
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = root / f"{stamp}-{proc_id}.log"
        log_fh = open(log_path, "ab", buffering=0)
        try:
            header = f"$ {command}\n\n".encode("utf-8", errors="replace")
            log_fh.write(header)
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
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
        )
        with self._lock:
            self._items[proc_id] = item
        threading.Thread(
            target=self._watch_process,
            args=(item,),
            daemon=True,
            name=f"{proc_id}_watcher",
        ).start()
        return item

    def _watch_process(self, item: BackgroundProcess) -> None:
        returncode = item.process.wait()
        completed_at = time.time()
        with self._lock:
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
                stop_requested=item.stop_requested,
            )
        item.finished.set()
        self._completed.put(event)

    def wait_completed(self, timeout: Optional[float] = None) -> BackgroundProcessCompleted:
        """Wait for the next completed command emitted by a watcher."""
        return self._completed.get(timeout=timeout)

    def list(self, *, include_finished: bool = False) -> List[BackgroundProcess]:
        with self._lock:
            items = list(self._items.values())
        if include_finished:
            return items
        return [item for item in items if item.running]

    def running_count(self) -> int:
        return len(self.list(include_finished=False))

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
