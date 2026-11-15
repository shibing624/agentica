# -*- coding: utf-8 -*-
"""In-process stand-ins for the shell-hook process boundary.

The hook wire is a decision layer: whose reply wins, what counts as a usable
reply, what happens when nobody ever answers. Those are the decisions the
regression suite pins, and none of them needs a real child process — a spawned
interpreter only adds wall time and machine-dependent flake. Real subprocess /
process-group acceptance belongs to development-time validation.

:class:`FakeHookProcess` implements exactly the interface production code uses
(``agentica.shell_hooks.process.HookProcess``): ``start`` / ``started`` /
``finished`` / ``completed_at`` / ``stdout`` / ``pid`` / ``returncode`` /
``exit_code`` / ``read_stderr`` / ``stderr`` / ``wait`` / ``kill``. A test can
hand one to ``HookRequest`` directly, or let production construct it through a
namespace monkeypatch.

Every instance is scripted: when stdout becomes readable (``after``, a
monotonic offset, never a sleep in the test), what the exit status is, what
stderr says, and whether ``start()`` succeeds at all. :class:`FakeHooks` is the
factory to monkeypatch into a module namespace; it records every process the
production code would have spawned and scripts each one by the first word of
its command, so the command is a label rather than an executable.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

#: Longest single blocking slice inside ``wait``. Completion is time-based and
#: there is deliberately no timer thread, so ``wait`` wakes itself up to check
#: whether the scripted reply is due. Small enough to honour a 2 ms offset.
_SLICE = 0.002

#: A scripted reply: a JSON-able body (serialized as the consumer's stdout), raw
#: stdout text (a plain-text ``UserPromptSubmit`` reply), or a callable that
#: receives the payload — so a test can echo the payload's ``request_id`` back.
ScriptedReply = Union[Dict[str, Any], str, Callable[[Dict[str, Any]], Any], None]


class FakeHookProcess:
    """One in-process hook invocation, scripted instead of spawned.

    ``kill`` is idempotent and is what ends a still-running consumer, exactly as
    in production: a kill before the scripted reply is due discards that reply.
    """

    def __init__(
        self,
        command: Sequence[str],
        payload: Dict[str, Any],
        *,
        env: Optional[Dict[str, str]] = None,
        capture_stderr: bool = False,
        reply: ScriptedReply = None,
        after: float = 0.0,
        exit_code: int = 0,
        stderr: str = "",
        open_stderr: bool = False,
        hang: bool = False,
        starts: bool = True,
        pid: Optional[int] = None,
    ):
        self.command = list(command or [])
        self.payload = payload
        self.env = env
        #: Recorded so a test can check the caller asked for the refusal reason.
        self.capture_stderr = capture_stderr
        self._reply = reply
        self._after = float(after)
        self._exit_code = int(exit_code)
        self._stderr = stderr
        #: Model a grandchild that inherited stderr: the pipe stays open after
        #: the hook itself exited, so a blocking read would wait for the child.
        self._open_stderr = bool(open_stderr)
        self._stderr_closed = threading.Event()
        self._hang = bool(hang)
        self._starts = bool(starts)
        self._pid = pid
        self._started = False
        self._killed = False
        self._kill_count = 0
        self._done = threading.Event()
        self._stdout: Optional[str] = None
        self._returncode: Optional[int] = None
        self._completed_at: Optional[float] = None
        self._finish_at: Optional[float] = None

    # ------------------------------------------------------------------ state

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    @property
    def started(self) -> bool:
        """True once ``start()`` succeeded; a killed process stays "started"."""
        return self._started

    @property
    def stdout(self) -> Optional[str]:
        self._tick()
        return self._stdout or None

    @property
    def stderr(self) -> str:
        return self._stderr

    @property
    def finished(self) -> bool:
        self._tick()
        return self._done.is_set()

    @property
    def completed_at(self) -> float:
        self._tick()
        assert self._completed_at is not None
        return self._completed_at

    @property
    def returncode(self) -> Optional[int]:
        self._tick()
        return self._returncode

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def kill_count(self) -> int:
        return self._kill_count

    # ------------------------------------------------------------------- turn

    def start(self) -> bool:
        if not self._starts:
            return False
        self._started = True
        self._pid = self._pid if self._pid is not None else id(self) % 100000 + 1000
        if not self._hang:
            self._finish_at = time.monotonic() + self._after
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the scripted reply; True when it is in within ``timeout``."""
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        while True:
            self._tick()
            if self._done.is_set():
                return True
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            if self._finish_at is not None:
                # Wake up when the scripted reply is due, not only on timeout.
                slice_ = max(min(self._finish_at - now, 0.05), _SLICE)
            elif deadline is not None:
                # Nothing scripted (a hanging consumer): block until killed or
                # until the caller's budget is spent. ``kill`` sets the event.
                slice_ = max(deadline - now, _SLICE)
            else:
                slice_ = 0.05  # bounded, so a hang without a budget cannot wedge
            if deadline is not None:
                slice_ = min(slice_, max(deadline - now, 0.0))
            if slice_ <= 0:
                continue
            self._done.wait(slice_)

    def exit_code(self, timeout: Optional[float] = None) -> Optional[int]:
        if not self._done.is_set() and timeout is not None:
            self.wait(timeout)
        return self._returncode

    def read_stderr(self, timeout: Optional[float] = None) -> str:
        if not self._done.is_set() and timeout is not None:
            self.wait(timeout)
        if self._open_stderr and not self._stderr_closed.is_set():
            # The handle is held by somebody else, exactly like a grandchild's
            # inherited pipe: block no longer than the caller's bound, then
            # report what had arrived — nothing, in this script.
            self._stderr_closed.wait(1.0 if timeout is None else max(timeout, 0.0))
            if not self._stderr_closed.is_set():
                return ""
        return self._stderr

    def kill(self, *, reap: bool = True) -> None:
        """Kill the (imaginary) process group; idempotent, like production."""
        if self._killed:
            return
        self._killed = True
        self._kill_count += 1
        self._stderr_closed.set()  # kill closes the pipes, as in production
        if not self._done.is_set():
            self._returncode = -9
            self._completed_at = time.monotonic()
            self._stdout = None
            self._done.set()

    # --------------------------------------------------------------- internal

    def _tick(self) -> None:
        """Complete the scripted reply once its time has come."""
        if self._done.is_set() or self._finish_at is None:
            return
        if time.monotonic() >= self._finish_at:
            self._complete()

    def _complete(self) -> None:
        self._stdout = self._render()
        self._returncode = self._exit_code
        self._completed_at = self._finish_at
        self._done.set()

    def _render(self) -> Optional[str]:
        reply = self._reply
        if callable(reply):
            reply = reply(self.payload)
        if reply is None or reply is False:
            return None
        if isinstance(reply, str):
            return reply
        return json.dumps(reply, ensure_ascii=False) + "\n"


class FakeHooks:
    """Factory that scripts fakes by command word and records them.

    Monkeypatch ``module.HookProcess`` with :attr:`factory`; production then
    "spawns" an in-process fake instead of a child. ``behaviours`` maps the
    first word of a consumer's command to ``FakeHookProcess`` keyword arguments
    (or a callable taking the payload and returning them).
    """

    def __init__(self, **behaviours: Any):
        self.made: List[FakeHookProcess] = []
        self._behaviours = behaviours

    def factory(
        self,
        command: Sequence[str],
        payload: Dict[str, Any],
        *,
        env: Optional[Dict[str, str]] = None,
        capture_stderr: bool = False,
    ) -> FakeHookProcess:
        process = FakeHookProcess(
            command,
            payload,
            env=env,
            capture_stderr=capture_stderr,
            **self._script(command, payload),
        )
        self.made.append(process)
        return process

    def install(self, monkeypatch: Any, module: Any) -> "FakeHooks":
        monkeypatch.setattr(module, "HookProcess", self.factory)
        return self

    @property
    def started(self) -> List[FakeHookProcess]:
        return [process for process in self.made if process.started]

    @property
    def payloads(self) -> List[Dict[str, Any]]:
        return [process.payload for process in self.made]

    def _script(self, command: Sequence[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        key = str(command[0]) if command else ""
        script = self._behaviours.get(key, {})
        if callable(script):
            script = script(payload)
        return dict(script or {})


def install_fakes(monkeypatch: Any, module: Any, **behaviours: Any) -> FakeHooks:
    """Monkeypatch ``module.HookProcess`` with a recording fake factory."""
    return FakeHooks(**behaviours).install(monkeypatch, module)


def started_fake(
    *,
    command: Sequence[str] = ("fake-hook",),
    payload: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> FakeHookProcess:
    """A fake already ``start()``-ed, for tests that build ``HookRequest`` directly."""
    process = FakeHookProcess(list(command), dict(payload or {}), **kwargs)
    assert process.start(), "the fake was scripted not to start"
    return process
