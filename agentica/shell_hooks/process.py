# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: One hook invocation: spawn the user's command, hand it JSON on
stdin, read one JSON document back on stdout.

``stdout`` is read on its own thread for every invocation, including the
fire-and-forget notices. That is not symmetry for its own sake: an unread pipe
fills and the child blocks writing to it, so a consumer that prints anything
would otherwise wedge a run. Nothing here interprets the reply — ``protocol``
does that — so this module's only opinion is "is this a JSON document yet".

Process group: ``start_new_session=True`` plus ``os.killpg``. A consumer that
spawns its own children would otherwise survive the kill and keep our pipe open.
The package already relies on this pattern (``execute_tool.py``,
``utils/async_utils.py``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from typing import Dict, List, Optional, Sequence

from agentica.utils.log import logger

#: An upper bound on how much stdout we will accumulate. A consumer prints one
#: small document; anything past this is a consumer that is not speaking the
#: protocol, and reading it forever would only leak.
MAX_OUTPUT_BYTES = 64 * 1024

_READ_CHUNK = 4096


def kill_process_group(proc: Optional[subprocess.Popen]) -> None:
    """SIGKILL the whole group. Safe to call twice, and safe after exit."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:  # already gone
            pass


class HookProcess:
    """One in-flight hook invocation.

    Lifecycle: ``start()`` → poll ``finished`` / ``stdout`` → ``kill()`` when the
    answer is no longer wanted (the user answered in the terminal, the turn was
    cancelled, the CLI is exiting). ``kill()`` is idempotent and is what closes
    stdout, so a consumer that is still talking cannot block the reader forever.
    """

    def __init__(
        self,
        command: Sequence[str],
        payload: Dict[str, object],
        *,
        env: Optional[Dict[str, str]] = None,
    ):
        self._command: List[str] = list(command)
        self._payload = payload
        self._env = env
        self._proc: Optional[subprocess.Popen] = None
        self._stdout = ""
        self._done = threading.Event()
        self._killed = False

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def started(self) -> bool:
        return self._proc is not None

    @property
    def stdout(self) -> Optional[str]:
        """Whatever the consumer printed, once the reader has finished."""
        return self._stdout or None

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def returncode(self) -> Optional[int]:
        """The child's exit status, once it has been reaped; None while running.

        ``poll()`` is what reaps a killed child. Asking it here rather than
        calling ``os.kill(pid, 0)`` is deliberate: signal 0 succeeds for a
        zombie, so it cannot tell "dead" from "dead and not yet collected".
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> bool:
        """Spawn and feed stdin. False means the hook could not be run at all.

        Never raises: "the hook is not installed / not executable" is a normal
        configuration state and must degrade to the terminal prompt, not to a
        failed run.
        """
        if not self._command or not self._command[0]:
            return False
        env = None
        if self._env:
            env = dict(os.environ)
            env.update({k: str(v) for k, v in self._env.items()})
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except (OSError, ValueError) as exc:
            logger.debug(f"shell hooks: could not spawn {self._command!r}: {exc}")
            self._proc = None
            return False

        stdin = self._proc.stdin
        if stdin is None:  # impossible with stdin=PIPE, but keep the type honest
            self.kill()
            return False
        try:
            data = json.dumps(self._payload, ensure_ascii=False).encode("utf-8")
            stdin.write(data)
            stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            # The consumer exited without reading its stdin. That is "no reply".
            logger.debug(f"shell hooks: could not write the payload: {exc}")
            self.kill()
            return False

        threading.Thread(
            target=self._read_stdout, name="agentica-hook-reader", daemon=True
        ).start()
        return True

    def _read_stdout(self) -> None:
        """Read until one JSON document has arrived, the pipe closes, or the cap."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._done.set()
            return
        buf = b""
        try:
            while len(buf) < MAX_OUTPUT_BYTES:
                chunk = proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                if _is_json_document(buf):
                    break
        except (OSError, ValueError) as exc:
            logger.debug(f"shell hooks: stdout read failed: {exc}")
        finally:
            self._stdout = buf.decode("utf-8", errors="replace")
            self._done.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the reader. True when it finished within ``timeout``."""
        return self._done.wait(timeout)

    def kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        kill_process_group(self._proc)
        try:
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()
        except Exception:  # closing an already-closed pipe
            pass


def _is_json_document(buf: bytes) -> bool:
    """Has a complete JSON document arrived?

    A consumer may print its document and then keep the pipe open (a wrapper that
    waits for its own children, say). Waiting for EOF there would hold a thread
    until someone killed the process, so the document itself ends the read.
    """
    text = buf.decode("utf-8", errors="replace").strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
