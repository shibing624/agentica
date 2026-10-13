# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP client — drive an external coding agent (Codex, Claude Code, ...).

``agentica/acp/`` is the other direction (agentica as the agent an editor
drives) and its package import requires the ``[acp]`` extra; nothing here
does, so this lives outside it.

Wire format: JSON-RPC 2.0, one JSON object per line on the child's
stdin/stdout (what ``codex-acp`` / ``claude-agent-acp`` / ``opencode acp``
speak). Lines can be far larger than asyncio's 64 KiB stream limit — Codex
sends its whole skills catalogue in one ``session/update``.

Threads, not asyncio streams, own the child: the interactive CLI runs every
turn in its own ``asyncio.run()``, and a session has to outlive the turn that
opened it. An asyncio transport is bound to its loop, so the next turn could
not read from it. The reader thread hands results to whichever loop is
waiting through ``_await_threadsafe``.
"""
import asyncio
import concurrent.futures
import json
import os
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

PROTOCOL_VERSION = 1
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603
_CANCEL_GRACE_SECONDS = 15
_TERMINATE_GRACE_SECONDS = 3
_STDERR_TAIL_BYTES = 2000

# (permission request params) -> agentica decision: allow / allow_prefix / deny / deny_prefix
PermissionDecider = Callable[[Dict[str, Any]], Coroutine[Any, Any, str]]


class ACPClientError(RuntimeError):
    """The runner failed, exited, or answered a request with an error."""


async def _await_threadsafe(future: concurrent.futures.Future) -> Any:
    """Await a reader-thread future from the running loop without binding it.

    ``asyncio.wrap_future`` would schedule onto this loop when the result
    lands; if this turn's loop is gone by then, that callback raises inside
    the reader thread. Here the late result is simply dropped, and cancelling
    the await never cancels the request itself.
    """
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    def _apply(done: concurrent.futures.Future) -> None:
        if waiter.done():
            return
        error = done.exception()
        if error is not None:
            waiter.set_exception(error)
        else:
            waiter.set_result(done.result())

    def _on_done(done: concurrent.futures.Future) -> None:
        try:
            loop.call_soon_threadsafe(_apply, done)
        except RuntimeError:
            pass  # loop closed: the turn that asked has ended and wants nothing

    future.add_done_callback(_on_done)
    return await waiter


def select_permission_outcome(options: List[Dict[str, Any]], decision: str) -> Dict[str, Any]:
    """Map an agentica approval decision onto the options the agent offered."""
    if decision in ("allow", "allow_prefix"):
        kinds = ("allow_once", "allow_always")
    else:
        kinds = ("reject_once", "reject_always")
    for kind in kinds:
        for option in options:
            if option["kind"] == kind:
                return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
    return {"outcome": {"outcome": "cancelled"}}


@dataclass
class PromptResult:
    """What one ``session/prompt`` produced."""

    stop_reason: str
    text: str
    tool_calls: List[Dict[str, str]]
    plan: List[Dict[str, str]]
    permissions_allowed: int
    permissions_denied: int
    timed_out: bool = False


@dataclass
class _Turn:
    loop: asyncio.AbstractEventLoop
    chunks: List[str] = field(default_factory=list)
    tool_calls: Dict[str, Dict[str, str]] = field(default_factory=dict)
    plan: List[Dict[str, str]] = field(default_factory=list)
    allowed: int = 0
    denied: int = 0
    approvals: List[concurrent.futures.Future] = field(default_factory=list)


class ACPSession:
    """One runner process holding one ACP session."""

    def __init__(
        self,
        argv: List[str],
        *,
        cwd: str,
        env: Optional[Dict[str, str]] = None,
        decide_permission: Optional[PermissionDecider] = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = cwd
        self.session_id = ""
        self._env = {**os.environ, **env} if env else None
        self._decide_permission = decide_permission
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 0
        self._pending: Dict[int, concurrent.futures.Future] = {}
        self._exit_error: Optional[ACPClientError] = None
        self._turn: Optional[_Turn] = None
        self._stderr = tempfile.NamedTemporaryFile(
            prefix="agentica-acp-", suffix=".stderr.log", delete=False
        )

    # ---- lifecycle ----------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def busy(self) -> bool:
        return self._turn is not None

    async def open(self) -> None:
        """Spawn the runner, ``initialize``, and create the session."""
        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd=self.cwd,
                env=self._env,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise ACPClientError(f"runner command not found: {self.argv[0]}") from e
        self._reader = threading.Thread(target=self._read_loop, name="acp-reader", daemon=True)
        self._reader.start()
        init = await _await_threadsafe(self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            # The runner keeps its own file and terminal tools; we only relay.
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
        }))
        if init["protocolVersion"] != PROTOCOL_VERSION:
            raise ACPClientError(
                f"runner speaks ACP protocol {init['protocolVersion']}, this client {PROTOCOL_VERSION}"
            )
        created = await _await_threadsafe(
            self._request("session/new", {"cwd": self.cwd, "mcpServers": []})
        )
        self.session_id = created["sessionId"]

    def close(self) -> None:
        """Stop the runner and its whole process group. Safe to call twice."""
        proc = self._proc
        # Died on its own before we stopped it: keep its stderr for the error.
        crashed = self._exit_error is not None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                proc.wait()
        if proc is not None:
            with self._write_lock:
                proc.stdin.close()
        if self._reader is not None:
            self._reader.join(timeout=_TERMINATE_GRACE_SECONDS)
        # Closing stdout under a reader still blocked in read() waits on the
        # buffer lock forever. The reader is only still there if a process
        # that escaped the group kill holds the write end; leave the fd to it.
        if proc is not None and (self._reader is None or not self._reader.is_alive()):
            proc.stdout.close()
        self._stderr.close()
        if not crashed:
            try:
                os.unlink(self._stderr.name)
            except FileNotFoundError:
                pass

    # ---- prompt -------------------------------------------------------

    async def prompt(self, text: str, *, timeout: float) -> PromptResult:
        """Send one user message and collect everything until it stops.

        On timeout ``session/cancel`` goes out and the partial turn is
        returned; on cancellation of this await it goes out and the
        cancellation propagates. The session stays usable either way.
        """
        if self._turn is not None:
            raise ACPClientError("this session is still answering the previous message")
        turn = _Turn(loop=asyncio.get_running_loop())
        self._turn = turn
        pending = self._request("session/prompt", {
            "sessionId": self.session_id,
            "prompt": [{"type": "text", "text": text}],
        })
        timed_out = False
        try:
            try:
                response = await asyncio.wait_for(_await_threadsafe(pending), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                self._cancel(turn)
                try:
                    response = await asyncio.wait_for(
                        _await_threadsafe(pending), timeout=_CANCEL_GRACE_SECONDS
                    )
                except asyncio.TimeoutError:
                    response = {"stopReason": "cancelled"}
        except asyncio.CancelledError:
            self._cancel(turn)
            raise
        finally:
            self._turn = None
        return PromptResult(
            stop_reason=response["stopReason"],
            text="".join(turn.chunks),
            tool_calls=list(turn.tool_calls.values()),
            plan=turn.plan,
            permissions_allowed=turn.allowed,
            permissions_denied=turn.denied,
            timed_out=timed_out,
        )

    def _cancel(self, turn: _Turn) -> None:
        for approval in turn.approvals:
            approval.cancel()
        if not self.alive:
            return
        try:
            self._send({"jsonrpc": "2.0", "method": "session/cancel",
                        "params": {"sessionId": self.session_id}})
        except ACPClientError:
            pass  # exited between the check and the write; nothing left to cancel

    # ---- wire ---------------------------------------------------------

    def _send(self, message: Dict[str, Any]) -> None:
        line = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise self._exit_error or ACPClientError("runner closed its input") from e

    def _request(self, method: str, params: Dict[str, Any]) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._state_lock:
            if self._exit_error is not None:
                future.set_exception(self._exit_error)
                return future
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = future
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return future

    def _read_loop(self) -> None:
        for raw in self._proc.stdout:
            try:
                message = json.loads(raw)
            except ValueError:
                continue  # a runner's stray log line on stdout is not protocol
            if "method" in message and "id" in message:
                threading.Thread(
                    target=self._answer_request, args=(message,), name="acp-request", daemon=True
                ).start()
            elif "method" in message:
                if message["method"] == "session/update":
                    self._on_update(message["params"]["update"])
            else:
                with self._state_lock:
                    future = self._pending.pop(message["id"], None)
                if future is None:
                    continue
                if "error" in message:
                    future.set_exception(ACPClientError(
                        f"runner error {message['error']['code']}: {message['error']['message']}"
                    ))
                else:
                    future.set_result(message["result"])
        code = self._proc.wait()
        error = ACPClientError(f"runner exited with code {code}{self._stderr_tail()}")
        with self._state_lock:
            self._exit_error = error
            pending, self._pending = self._pending, {}
        for future in pending.values():
            future.set_exception(error)

    def _stderr_tail(self) -> str:
        try:
            with open(self._stderr.name, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - _STDERR_TAIL_BYTES))
                tail = fh.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""
        return f"; stderr ({self._stderr.name}):\n{tail}" if tail else ""

    def _on_update(self, update: Dict[str, Any]) -> None:
        turn = self._turn
        if turn is None:
            return
        kind = update["sessionUpdate"]
        if kind == "agent_message_chunk":
            content = update["content"]
            if content["type"] == "text":
                turn.chunks.append(content["text"])
        elif kind == "tool_call":
            turn.tool_calls[update["toolCallId"]] = {
                "title": update["title"],
                "kind": update.get("kind", "other"),
                "status": update.get("status", "pending"),
            }
        elif kind == "tool_call_update":
            call = turn.tool_calls.setdefault(
                update["toolCallId"], {"title": "", "kind": "other", "status": "pending"}
            )
            for key in ("title", "kind", "status"):
                if update.get(key) is not None:
                    call[key] = update[key]
        elif kind == "plan":
            turn.plan = [
                {"content": entry["content"], "status": entry["status"]}
                for entry in update["entries"]
            ]

    def _answer_request(self, message: Dict[str, Any]) -> None:
        try:
            if message["method"] == "session/request_permission":
                result = self._permission(message["params"])
                reply = {"jsonrpc": "2.0", "id": message["id"], "result": result}
            else:
                reply = {"jsonrpc": "2.0", "id": message["id"], "error": {
                    "code": _METHOD_NOT_FOUND,
                    "message": f"client does not implement {message['method']}",
                }}
        except Exception as e:  # the agent blocks until it hears back, whatever happened
            reply = {"jsonrpc": "2.0", "id": message["id"],
                     "error": {"code": _INTERNAL_ERROR, "message": str(e)}}
        try:
            self._send(reply)
        except ACPClientError:
            pass  # runner already gone; nobody is waiting for this reply

    def _permission(self, params: Dict[str, Any]) -> Dict[str, Any]:
        turn = self._turn
        options = params["options"]
        if turn is None or self._decide_permission is None:
            return select_permission_outcome(options, "deny")
        try:
            approval = asyncio.run_coroutine_threadsafe(self._decide_permission(params), turn.loop)
        except RuntimeError:
            return select_permission_outcome(options, "deny")  # the turn's loop is gone
        turn.approvals.append(approval)
        try:
            decision = approval.result()
        except concurrent.futures.CancelledError:
            return {"outcome": {"outcome": "cancelled"}}
        if decision in ("allow", "allow_prefix"):
            turn.allowed += 1
        else:
            turn.denied += 1
        return select_permission_outcome(options, decision)
