# -*- coding: utf-8 -*-
"""Interactive shells for the right-hand panel (dsh's sidebar terminal).

A terminal is a login shell on a pseudo-terminal, owned by the gateway rather
than by the socket watching it: collapsing the panel or reloading the page
detaches, and the next attach replays what the shell printed meanwhile. Output
never reaches an agent's context.

POSIX only — the pty module has no Windows backend.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import signal
import struct
import subprocess
import termios
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from agentica.utils.log import logger

MAX_TERMINALS = 8
#: What one attach replays. A full-screen program redraws itself on the next
#: resize, so the tail is enough to put the screen back.
REPLAY_BYTES = 256 * 1024
#: A shell nobody has watched for this long is reaped.
UNATTENDED_SECONDS = 2 * 60 * 60
READ_CHUNK = 64 * 1024

Listener = Callable[[dict], None]


class TerminalError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _login_shell() -> str:
    shell = os.environ.get("SHELL", "")
    if shell and os.access(shell, os.X_OK):
        return shell
    for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
        if os.access(candidate, os.X_OK):
            return candidate
    raise TerminalError(500, "No shell found on this machine")


def _set_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


@dataclass
class Terminal:
    id: str
    owner: str
    session_id: str
    cwd: str
    shell: str
    process: subprocess.Popen
    fd: int
    cols: int
    rows: int
    created_at: float = field(default_factory=time.time)
    buffer: bytearray = field(default_factory=bytearray)
    listeners: Set[Listener] = field(default_factory=set)
    exit_code: Optional[int] = None
    #: Last moment somebody was attached; the reaper measures from here.
    watched_at: float = field(default_factory=time.time)

    def info(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "shell": os.path.basename(self.shell),
            "cols": self.cols,
            "rows": self.rows,
            "created_at": self.created_at,
            "exited": self.exit_code is not None,
            "exit_code": self.exit_code,
        }


class TerminalManager:
    def __init__(self) -> None:
        self._terms: Dict[str, Terminal] = {}

    def list(self, owner: str) -> List[dict]:
        return [t.info() for t in self._terms.values() if t.owner == owner]

    def get(self, owner: str, term_id: str) -> Terminal:
        term = self._terms.get(term_id)
        if term is None or term.owner != owner:
            raise TerminalError(404, "No such terminal")
        return term

    def create(self, owner: str, session_id: str, cwd: str, cols: int, rows: int) -> Terminal:
        if os.name != "posix":
            raise TerminalError(501, "Terminals need a POSIX host")
        self._reap_unattended()
        if len(self._terms) >= MAX_TERMINALS:
            raise TerminalError(429, f"At most {MAX_TERMINALS} terminals at once")
        workdir = Path(cwd).expanduser()
        if not workdir.is_dir():
            raise TerminalError(400, f"Not a directory: {cwd}")
        shell = _login_shell()
        cols, rows = max(2, min(cols, 1000)), max(1, min(rows, 1000))
        master, slave = pty.openpty()
        _set_size(slave, cols, rows)
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        try:
            process = subprocess.Popen(
                [shell, "-l"],
                stdin=slave, stdout=slave, stderr=slave,
                cwd=str(workdir), env=env,
                start_new_session=True, close_fds=True,
            )
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        term = Terminal(
            id=uuid.uuid4().hex[:12], owner=owner, session_id=session_id,
            cwd=str(workdir), shell=shell, process=process, fd=master, cols=cols, rows=rows,
        )
        self._terms[term.id] = term
        asyncio.get_running_loop().add_reader(master, self._on_readable, term)
        logger.debug(f"terminal {term.id} started: {shell} in {workdir}")
        return term

    def attach(self, term: Terminal, listener: Listener) -> bytes:
        term.listeners.add(listener)
        term.watched_at = time.time()
        return bytes(term.buffer)

    def detach(self, term: Terminal, listener: Listener) -> None:
        term.listeners.discard(listener)
        term.watched_at = time.time()

    def write(self, term: Terminal, data: str) -> None:
        if term.exit_code is not None:
            return
        payload = data.encode("utf-8")
        while payload:
            try:
                n = os.write(term.fd, payload)
            except BlockingIOError:
                # The shell is not reading; dropping keystrokes beats blocking
                # the event loop every other request runs on.
                return
            payload = payload[n:]

    def resize(self, term: Terminal, cols: int, rows: int) -> None:
        if term.exit_code is not None:
            return
        term.cols, term.rows = max(2, min(cols, 1000)), max(1, min(rows, 1000))
        _set_size(term.fd, term.cols, term.rows)
        # Tell the foreground program, not just the shell, so a full-screen
        # app redraws at the new size.
        try:
            os.killpg(os.getpgid(term.process.pid), signal.SIGWINCH)
        except ProcessLookupError:
            pass

    def close(self, owner: str, term_id: str) -> None:
        term = self.get(owner, term_id)
        self._kill(term)
        self._terms.pop(term.id, None)

    def close_all(self) -> None:
        for term in list(self._terms.values()):
            self._kill(term)
        self._terms.clear()

    def _on_readable(self, term: Terminal) -> None:
        try:
            chunk = os.read(term.fd, READ_CHUNK)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._on_exit(term)
            return
        term.buffer += chunk
        if len(term.buffer) > REPLAY_BYTES:
            del term.buffer[: len(term.buffer) - REPLAY_BYTES]
        text = chunk.decode("utf-8", errors="replace")
        for listener in list(term.listeners):
            listener({"type": "output", "data": text})

    def _on_exit(self, term: Terminal) -> None:
        loop = asyncio.get_running_loop()
        loop.remove_reader(term.fd)
        try:
            term.exit_code = term.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            term.process.kill()
            term.exit_code = term.process.wait()
        try:
            os.close(term.fd)
        except OSError:
            pass
        for listener in list(term.listeners):
            listener({"type": "exit", "code": term.exit_code})

    def _kill(self, term: Terminal) -> None:
        if term.exit_code is None:
            try:
                asyncio.get_running_loop().remove_reader(term.fd)
            except RuntimeError:
                pass
            try:
                os.killpg(os.getpgid(term.process.pid), signal.SIGHUP)
            except ProcessLookupError:
                pass
            try:
                term.exit_code = term.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(term.process.pid), signal.SIGKILL)
                term.exit_code = term.process.wait()
            try:
                os.close(term.fd)
            except OSError:
                pass
        for listener in list(term.listeners):
            listener({"type": "closed"})
        term.listeners.clear()

    def _reap_unattended(self) -> None:
        now = time.time()
        for term in list(self._terms.values()):
            if not term.listeners and now - term.watched_at > UNATTENDED_SECONDS:
                logger.info(f"terminal {term.id}: unattended for {UNATTENDED_SECONDS}s, closing")
                self._kill(term)
                self._terms.pop(term.id, None)


manager = TerminalManager()
