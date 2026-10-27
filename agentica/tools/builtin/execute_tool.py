# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in execute/wait tools
"""
import asyncio
import contextlib
import os
import re
import signal
import tempfile
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4


from agentica.tools.base import Tool
from agentica.tools.background_processes import BackgroundProcessRegistry, read_log_tail
from agentica.tools.os_sandbox import (
    SandboxPolicy,
    describe_policy,
    resolve_policy_paths,
    unsupported_reason,
    wrap_command,
)
from agentica.tools.safety import (
    check_command_safety,
    command_matches_blocked,
    is_catastrophic,
    redact_sensitive_text,
)
from agentica.security.redact import redact_tool_outputs_enabled
from agentica.utils.async_utils import close_subprocess_transport, terminate_subprocess
from agentica.utils.log import logger
from agentica.compression.tool_result_storage import (
    _build_persisted_message,
    _build_truncated_message,
    can_recover_spill,
    get_tool_result_path,
)

# Unix conventions where a non-zero exit is the answer, not a crashed
# command. Used only to decide whether to raise; the exit code is still
# printed. No notes are appended to the result.
_EXPECTED_NONZERO = {
    "grep": {1}, "egrep": {1}, "fgrep": {1}, "rg": {1}, "ag": {1}, "ack": {1},
    "diff": {1}, "colordiff": {1},
    "find": {1},
    "test": {1}, "[": {1},
    "curl": {6, 7, 22, 28},
    "git": {1},
    "pytest": {1, 5},
    "ruff": {1}, "mypy": {1}, "pyright": {1}, "basedpyright": {1},
    "flake8": {1}, "pylint": {1}, "eslint": {1}, "tsc": {1},
    "python": {1},
}


def _first_command_name(segment: str) -> str:
    """Bare executable of one ``;`` / ``&&`` / ``|`` piece (env assignments skipped)."""
    words = segment.split()
    base_cmd = ""
    cmd_index = -1
    for i, w in enumerate(words):
        if "=" in w and not w.startswith("-"):
            continue
        if w == "env":
            continue
        base_cmd = w.split("/")[-1]
        cmd_index = i
        break
    if base_cmd in {"python", "python3"} and cmd_index >= 0:
        if len(words) > cmd_index + 2 and words[cmd_index + 1] == "-m":
            base_cmd = words[cmd_index + 2].split(".")[0].split("/")[-1]
    return base_cmd


def _expected_nonzero_exit(command: str, exit_code: int) -> bool:
    """True when this command's non-zero exit is a normal result."""
    if exit_code == 0:
        return False
    segments = re.split(r"\s*(?:\|\||&&|[|;])\s*", command)
    last_segment = (segments[-1] if segments else command).strip()
    return exit_code in _EXPECTED_NONZERO.get(_first_command_name(last_segment), ())


# ─── File safety guards ──────────────────────────────────────────────────────
# Ported from hermes-agent tools/file_tools.py

# Paths that would hang the process (infinite output or blocking input)

_SELF_DETACHING_COMMAND = re.compile(r"(?<!&)&\s*$")
_LEADING_SLEEP = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)\b")
_MAX_FOREGROUND_SLEEP_SECONDS = 120
# Drain at most this much per stream. A 600k-line source file is tens of MB and
# fits; ``yes`` / ``/dev/urandom`` must not fill the disk. Hitting the cap kills
# the process so the drain cannot pin the turn.
_EXECUTE_SPILL_HARD_CAP_BYTES = 64 * 1024 * 1024
_PIPE_CHUNK = 65536
# After a cap-triggered SIGKILL the pipes hit EOF on their own; this bounds
# the wait for a grandchild that outlived the group and still holds one.
_POST_KILL_DRAIN_SECONDS = 5
# The command reaches /bin/sh through its environment, not its argv. As
# ``sh -c <command>`` the command text is the shell's own cmdline, so a
# ``pkill -f <pattern>`` inside it matches the shell running it and SIGTERMs
# it: whatever follows the pkill never runs and the call reports exit -15.
# Every exit -15 in two long production sessions (18 of 18) was this.
_COMMAND_ENV = "AGENTICA_COMMAND"
_RUN_COMMAND_FROM_ENV = f'eval "unset {_COMMAND_ENV}; ${_COMMAND_ENV}"'


def _spool_size_and_lines(spool) -> tuple[int, int]:
    """Bytes and newlines already in ``spool`` (a drain cut off mid-way)."""
    spool.seek(0)
    size = newlines = 0
    for chunk in iter(lambda: spool.read(_PIPE_CHUNK), b""):
        size += len(chunk)
        newlines += chunk.count(b"\n")
    return size, newlines


async def _drain_stream(stream, spool, hard_cap: int) -> tuple[int, int, bool]:
    """Copy ``stream`` into ``spool`` up to ``hard_cap`` bytes.

    Returns ``(bytes_written, newline_count, hit_hard_cap)``. Hitting the cap
    stops writing; the caller must then kill the process so a generator like
    ``yes`` cannot pin the drain.
    """
    if stream is None:
        return 0, 0, False
    total = 0
    newlines = 0
    while True:
        data = await stream.read(_PIPE_CHUNK)
        if not data:
            break
        if total >= hard_cap:
            return total, newlines, True
        room = hard_cap - total
        chunk = data if len(data) <= room else data[:room]
        spool.write(chunk)
        total += len(chunk)
        newlines += chunk.count(b"\n")
        if len(data) > room:
            return total, newlines, True
    return total, newlines, False


def _kill_process_group(proc) -> None:
    """SIGKILL the child's whole process group.

    Deliberately not ``terminate_subprocess``: that calls
    ``process.communicate()``, which starts its own readers on pipes our
    drain coroutines are already reading — asyncio raises ``read() called
    while another coroutine is already waiting for incoming data``. Here the
    drain owns the pipes, so the kill has to be the raw syscall.

    Not gated on ``proc.returncode``. ``cmd &`` reaps the shell while a
    grandchild still holds our pipes; skipping the signal there leaves the
    writer alive and the other drain waiting for EOF. ``pid == pgid``
    because we spawn with ``start_new_session`` — ``getpgid`` of an already
    reaped leader raises and would miss the group.
    """
    if proc is None:
        return
    pid = proc.pid
    if pid is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGKILL)
            return
        proc.kill()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        proc.kill()


async def _cancel_drain_tasks(tasks) -> None:
    """Drop leftover pipe readers before anyone else touches the pipes.

    A cancelled or timed-out turn falls through to ``terminate_subprocess``,
    which ``communicate()``s. Those readers must be finished first.
    """
    leftover = [task for task in tasks if not task.done()]
    for task in leftover:
        task.cancel()
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)


def _decode_spool_head_tail(spool, max_chars: int) -> str:
    """UTF-8 preview of a spool without loading the middle into memory."""
    spool.seek(0, os.SEEK_END)
    size = spool.tell()
    if size == 0:
        return ""
    head_n = min(size, max(1, int(max_chars * 0.4)))
    tail_n = min(size, max_chars - head_n) if size > head_n else 0
    spool.seek(0)
    head = spool.read(head_n)
    if size <= head_n + tail_n:
        rest = spool.read()
        return (head + rest).decode("utf-8", errors="replace")
    spool.seek(size - tail_n)
    tail = spool.read(tail_n)
    omitted = size - len(head) - len(tail)
    return (
        head.decode("utf-8", errors="replace")
        + f"\n\n... [{omitted} bytes omitted] ...\n\n"
        + tail.decode("utf-8", errors="replace")
    )


def _copy_spools_to_path(path: str, stdout_spool, stderr_spool) -> int:
    """Write stdout then optional stderr to ``path``. Returns bytes written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "wb") as fh:
        stdout_spool.seek(0)
        while True:
            chunk = stdout_spool.read(_PIPE_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
        stderr_spool.seek(0, os.SEEK_END)
        if stderr_spool.tell():
            marker = b"\n[stderr]\n"
            fh.write(marker)
            written += len(marker)
            stderr_spool.seek(0)
            while True:
                chunk = stderr_spool.read(_PIPE_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    return written


def _copy_spools_to_path_redacted(path: str, stdout_spool, stderr_spool) -> int:
    """``_copy_spools_to_path`` with secrets masked, for redacting deployments.

    The preview handed to the model is redacted, so the file the model is
    told to ``read_file`` must be too — otherwise the toggle promises
    redaction while the full copy on disk stays plaintext, and one
    ``read_file`` recovers every secret the toggle was meant to hide.

    Whole-stream rather than chunked on purpose: redaction rewrites spans
    that cross any boundary (a PEM block matches across newlines), so a
    chunked pass needs carry-over state that is easy to get subtly wrong.
    The input is already bounded by ``_EXECUTE_SPILL_HARD_CAP_BYTES``.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "wb") as fh:
        for spool in (stdout_spool, stderr_spool):
            spool.seek(0, os.SEEK_END)
            if not spool.tell():
                continue
            if spool is stderr_spool:
                marker = b"\n[stderr]\n"
                fh.write(marker)
                written += len(marker)
            spool.seek(0)
            text = spool.read().decode("utf-8", errors="replace")
            text = redact_sensitive_text(text) or ""
            data = text.encode("utf-8", errors="replace")
            fh.write(data)
            written += len(data)
    return written


# Default for a single `wait` call when the model omits timeout. Same contract
# as execute(timeout=...): the caller decides, no silent upper clamp.
_DEFAULT_WAIT_SECONDS = 300
_WAIT_POLL_INTERVAL = 0.2

# Appended to the `execute` description only where a shared registry exists to
# own the process (an interactive CLI session). Kept out of the docstring so
# the parameter and the prose describing it appear and disappear together: a
# session with nowhere to register a detached process has no `wait`, no
# listing, and no way to stop it, so offering `background` there would buy an
# orphan that outlives the agent and nobody can see.
_BACKGROUND_GUIDANCE = """
- ``background=True`` decides how long the command lives, not how many run at
  once: use it when the output is not needed this turn or the command may
  outlive one call (a killed foreground command loses its output; a background
  one keeps its log). For speed, use ``parallel_safe``.
  {delivery}
  Never wait inside a turn with ``sleep`` or polling (a 120s+ leading ``sleep``
  is refused), and never detach with a trailing ``&``.
- A background command is stopped when the session exits. For a job that must
  outlive the session, load the ``long-running-jobs`` skill first.
"""

_DELIVERED = (
    "When it exits, its exit code and log tail come back to you as a new message,\n"
    "  so you can end the turn and pick up from there; ``wait`` with the returned\n"
    "  id is only for needing the result within this turn."
)
_NOT_DELIVERED = (
    "A background command's exit is reported to the user, not to you; ``wait`` with\n"
    "  the returned id is what brings it back to you."
)

# Appended, together with the ``escalate`` parameter, only when the OS sandbox
# is on — same rule as background: the parameter and its prose come as a pair.
_SANDBOX_GUIDANCE = """
- Commands run inside an OS sandbox: they may write only inside the work
  directory (and its git worktrees), temp dirs and package caches; the network
  is off (localhost works); credential paths such as ``~/.ssh`` are unreadable.
  When the sandbox is what stops a command — it failed with "Operation not
  permitted" / a connection error, or it plainly needs the network or a write
  elsewhere (``git push``, an install from the network) — call execute with
  ``escalate=True``: it runs outside the sandbox once the user approves. Do not
  route around the sandbox any other way.
"""


class BuiltinExecuteTool(Tool):
    """
    Built-in command execution tool using async subprocess.
    Exposed as execute function for consistent naming in Agent.
    """

    def __init__(self, work_dir: Optional[str] = None, timeout: int = 120,
                 max_output_length: int = 20000, sandbox_config=None,
                 background_process_registry: Optional[BackgroundProcessRegistry] = None,
                 max_foreground_timeout: int = 600):
        """
        Initialize BuiltinExecuteTool.

        Args:
            work_dir: Work directory for command execution
            timeout: Default command execution timeout in seconds. With
                ``SandboxConfig.enabled`` the default is
                ``sandbox_config.max_execution_time`` instead.
            max_output_length: Maximum length of output to return
            sandbox_config: SandboxConfig instance for command restriction enforcement
                (``os_sandbox`` puts every command inside the OS sandbox and
                adds the ``escalate`` parameter)
            background_process_registry: shared registry owning detached
                commands. Supplied by surfaces that can also list, report and
                stop them (the interactive CLI). Without one, ``background``
                and ``wait`` are not offered at all rather than being backed by
                a private registry nobody can reach: a detached process would
                outlive the agent with no way to see it or kill it.
            max_foreground_timeout: Largest per-call ``timeout`` accepted when
                ``background`` exists. A longer foreground call is refused (not
                clamped) with a pointer to ``background=True``, which keeps the
                log when the turn ends. Without a background registry there is
                no alternative to offer, so no cap applies.
        """
        super().__init__(name="builtin_execute_tool")
        self._work_dir: Optional[Path] = Path(work_dir) if work_dir else None
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._sandbox_config = sandbox_config
        self._background_process_registry = background_process_registry
        self._os_sandbox = bool(sandbox_config is not None and sandbox_config.os_sandbox)
        if self._os_sandbox:
            reason = unsupported_reason()
            if reason is not None:
                raise RuntimeError(
                    f"SandboxConfig.os_sandbox is on, but this machine cannot run it: {reason}. "
                    "Install it or turn the OS sandbox off."
                )
        # Override timeout from sandbox config if set
        if sandbox_config and sandbox_config.enabled and sandbox_config.max_execution_time:
            self._timeout = sandbox_config.max_execution_time
        self._max_foreground_timeout = (
            max(self._timeout, max_foreground_timeout) if background_process_registry is not None else None
        )
        self.register(self.execute, is_destructive=True)

        # Large bash outputs are persisted to disk (context gets preview only).
        # Threshold matches _max_output_length so persist actually fires —
        # a higher max_result_size_chars used to sit above the in-tool truncate
        # and never ran. read_file keeps max_result_size_chars=None.
        self.functions["execute"].max_result_size_chars = self._max_output_length
        # Execute tool manages its own timeout internally via asyncio.wait_for
        # on the subprocess. Skip the outer timeout wrapper in Model.run_function_calls.
        self.functions["execute"].manages_own_timeout = True
        # Shell has no tool-level answer to "may this run in parallel": the same
        # tool issues `pytest tests/a` and `git commit`. The caller declares it
        # per call; absent the flag the batch stays serial and ordered.
        self.functions["execute"].parallel_arg = "parallel_safe"

        # Whether `background` exists at all is decided per instance, and the
        # schema is normally derived per class from the signature — so prepare
        # this one now and mark it done, or the agent's own pass would put the
        # parameter and its prose straight back.
        execute_fn = self.functions["execute"]
        execute_fn.process_entrypoint()
        cap_note = (
            f" Max {self._max_foreground_timeout}s; longer work goes in background=True."
            if self._max_foreground_timeout is not None else ""
        )
        execute_fn.description = execute_fn.description.replace(
            "{default_timeout}", str(self._timeout)
        ).replace("{timeout_cap}", cap_note)
        execute_fn.parameters["properties"]["timeout"]["description"] = (
            f"seconds; default {self._timeout}.{cap_note}"
        )
        if self._background_process_registry is not None:
            execute_fn.description += _BACKGROUND_GUIDANCE.format(
                delivery=_DELIVERED if self._background_process_registry.delivers_results
                else _NOT_DELIVERED,
            )
            self.register(self.wait, concurrency_safe=True, is_read_only=True, is_destructive=False)
            # `wait` may block past the outer 120s executor wrapper; it owns its
            # timeout so the caller-supplied value is not cut short mid-wait.
            self.functions["wait"].manages_own_timeout = True
        else:
            execute_fn.parameters["properties"].pop("background", None)
        if self._os_sandbox:
            execute_fn.description += _SANDBOX_GUIDANCE
            execute_fn.parameters["properties"]["escalate"]["description"] = (
                "run outside the OS sandbox; the user is asked first. Only when "
                "the sandbox is what stops the command"
            )
        else:
            execute_fn.parameters["properties"].pop("escalate", None)
        execute_fn.skip_entrypoint_processing = True

    def set_work_dir(self, work_dir: str) -> None:
        """Point the session default cwd at another directory.

        Per-call ``work_dir`` on ``execute`` is the way to run in another
        checkout without moving the session. This setter stays for
        construction-time wiring (``--worktree``).
        """
        self._work_dir = Path(work_dir)

    def _call_cwd(self, work_dir: str = "") -> Optional[str]:
        """Session cwd, or the directory this call asked to use."""
        raw = (work_dir or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                base = self._work_dir or Path.cwd()
                path = base / path
            return str(Path(os.path.abspath(path)))
        return str(self._work_dir) if self._work_dir else None

    async def _sandbox_policy(self) -> SandboxPolicy:
        """The OS sandbox policy for one command, roots resolved now.

        Resolved per call rather than once: a worktree created mid-session
        must be writable without restarting, and that is a ``git`` question.
        """
        config = self._sandbox_config
        session_dir = str(self._work_dir) if self._work_dir else os.getcwd()
        roots, read_only = await resolve_policy_paths([*config.writable_dirs, session_dir])
        return SandboxPolicy(
            writable_roots=roots,
            read_only_paths=read_only,
            blocked_components=tuple(config.blocked_paths),
            network_access=config.network_access,
        )

    def _spill_target(
        self, cwd: Optional[str] = None
    ) -> tuple[str, Optional[str], Optional[str]]:
        """``(session_id, user_id, cwd)`` for a captured overflow file."""
        fn = self.functions.get("execute")
        agent = fn._agent if fn is not None else None
        session_id = "default"
        user_id = None
        spill_cwd = cwd if cwd is not None else (
            str(self._work_dir) if self._work_dir else None
        )
        if agent is not None:
            session_id = agent.session_id or "default"
            if agent.workspace is not None:
                user_id = agent.workspace.user_id
        return session_id, user_id, spill_cwd

    async def _drain_both(self, proc, stdout_spool, stderr_spool):
        """Drain both pipes, killing the child the moment a stream hits the cap.

        ``asyncio.gather`` (ALL_COMPLETED) deadlocks here. ``_drain_stream``
        returns as soon as *its* stream crosses the hard cap, but the child is
        still alive and still writing; the other drain then blocks forever
        waiting for an EOF that cannot arrive while this pipe stays full. The
        command hangs until ``effective_timeout`` and the cap is never
        enforced — a 70 MB ``cat`` pinned the turn for the full 120 s default.

        So wait on FIRST_COMPLETED, then on the rest if the first stream
        was under the cap (stderr often EOFs first). Kill if *any* stream
        tripped the cap — not only the one that finished first. The pipes
        reaching EOF is what lets a remaining drain return promptly.
        """
        tasks = [
            asyncio.ensure_future(
                _drain_stream(proc.stdout, stdout_spool, _EXECUTE_SPILL_HARD_CAP_BYTES)
            ),
            asyncio.ensure_future(
                _drain_stream(proc.stderr, stderr_spool, _EXECUTE_SPILL_HARD_CAP_BYTES)
            ),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
            hit_cap = any(task.result()[2] for task in done)
            # The first stream to finish is often the quiet one (stderr
            # closed, nothing written). The flood may still be on the
            # other pipe — wait for it, then kill if *that* trip hit cap.
            if not hit_cap and pending:
                await asyncio.wait(pending)
                hit_cap = any(task.result()[2] for task in tasks)
            if hit_cap:
                _kill_process_group(proc)
                still = [task for task in tasks if not task.done()]
                if still:
                    await asyncio.wait(still, timeout=_POST_KILL_DRAIN_SECONDS)
            await _cancel_drain_tasks(tasks)
        except BaseException:
            await _cancel_drain_tasks(tasks)
            raise
        return tuple(
            task.result() if task.done() and not task.cancelled() else (0, 0, False)
            for task in tasks
        )

    def _finish_captured_output(
        self,
        stdout_spool,
        stderr_spool,
        *,
        out_bytes: int,
        err_bytes: int,
        out_lines: int,
        err_lines: int,
        hit_hard_cap: bool,
        cwd: Optional[str] = None,
    ) -> str:
        """Turn drained pipes into a result that never exceeds ``max_output_length``.

        Under the cap this is the decoded stdout/stderr, same as ``communicate``.
        Over it, the full bytes stay on disk (or are discarded) and the returned
        string is a preview — Layer 1 cannot evict the live round, so this is
        the only bound that can save the next model call.
        """
        marker = len(b"\n[stderr]\n") if err_bytes else 0
        combined = out_bytes + err_bytes + marker
        under_cap = combined <= self._max_output_length and not hit_hard_cap
        if under_cap:
            stdout_spool.seek(0)
            stderr_spool.seek(0)
            parts = []
            if out_bytes:
                parts.append(stdout_spool.read().decode("utf-8", errors="replace"))
            if err_bytes:
                parts.append(
                    "[stderr]\n" + stderr_spool.read().decode("utf-8", errors="replace")
                )
            return "\n".join(parts).strip()

        preview_parts = []
        if out_bytes:
            preview_parts.append(_decode_spool_head_tail(stdout_spool, self._max_output_length))
        if err_bytes:
            preview_parts.append(
                "[stderr]\n" + _decode_spool_head_tail(stderr_spool, self._max_output_length)
            )
        preview = "\n".join(preview_parts).strip()
        n_lines = out_lines + err_lines + (1 if err_bytes else 0)
        # Header, not preview: `_build_persisted_message` used to say
        # "Full output" / "read_file for the rest" while this note sat
        # inside `_preview()` where the model can miss it.
        truncated_note = (
            f"\n\n[stopped after {_EXECUTE_SPILL_HARD_CAP_BYTES} bytes. The command was "
            f"killed, so any saved copy is INCOMPLETE — the first "
            f"{_EXECUTE_SPILL_HARD_CAP_BYTES} bytes only. Re-run with a "
            f"bound (| head / | tail) or on a narrower input.]"
            if hit_hard_cap
            else ""
        )

        session_id, user_id, cwd = self._spill_target(cwd)
        names = set(self.functions) if self.functions else {"execute"}
        if can_recover_spill(names):
            file_path = get_tool_result_path(
                f"execute-{uuid4().hex[:12]}",
                cwd=cwd, session_id=session_id, user_id=user_id,
            )
            try:
                copier = (
                    _copy_spools_to_path_redacted
                    if redact_tool_outputs_enabled()
                    else _copy_spools_to_path
                )
                size = copier(file_path, stdout_spool, stderr_spool)
            except OSError as e:
                logger.warning(f"Failed to persist execute overflow to {file_path}: {e}")
                return _build_truncated_message(preview + truncated_note)
            return _build_persisted_message(
                file_path, preview, size_bytes=size, n_lines=n_lines,
                incomplete=hit_hard_cap,
            )
        return _build_truncated_message(preview + truncated_note)

    async def execute(
            self,
            command: str,
            timeout: Optional[int] = None,
            background: bool = False,
            parallel_safe: bool = False,
            work_dir: str = "",
            escalate: bool = False,
    ) -> str:
        """Executes a shell command, capturing both stdout and stderr.

        Any shell command goes here: explore, analyze, verify, build, git, and
        pipelines that shape stdout. Split or combine as needed:
        do not force every probe into one script. Dependent steps share one command (``&&``,
        pipes); a miss that must not stop the rest uses ``;`` and
        ``2>/dev/null``. Search with ``rg -n PAT`` or
        ``rg -g '*.py' -n TODO src``.
        Never ``rg -rn``: ``-r`` is ``--replace``.
        Search from ``.`` or a known directory; do not assemble
        a long absolute path from package layout. Bound noisy output with
        ``| head`` / ``| tail``; do not ``cat`` whole source files. Quote paths
        with spaces. Heredocs (``python3 - <<'EOF'``) work.

        - The command is passed unchanged to the shell; safety checks may
          refuse it but never rewrite it.
        - Foreground timeout: {default_timeout}s by default.{timeout_cap}
        - ``parallel_safe=True`` runs this call together with the adjacent
          calls in the same message. Default: calls run one at a time in the
          order issued, and a failing command cancels the ones after it. Only
          for commands independent of every sibling (two test suites, two
          reads); never ``git add`` + ``git commit`` or anything touching one
          file twice.
        - ``work_dir`` runs this command in that directory (e.g. a path
          ``worktree`` returned); omit for the session directory.
        - Output over the limit is saved to a file; you get a head/tail
          preview and its path.
        - Git: prefer new commits over amending; ask before ``reset --hard`` /
          ``push --force``; never ``--no-verify`` unless asked.

        Args:
            command: Exact shell command. Not rewritten.
            timeout: seconds
            parallel_safe: run concurrently with adjacent calls in this message

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        # `parallel_safe` is read by the executor (via Function.parallel_arg) to
        # decide the schedule before this ever runs; it means nothing here.
        del parallel_safe

        if (
            timeout is not None
            and self._max_foreground_timeout is not None
            and timeout > self._max_foreground_timeout
            and not background
        ):
            raise ValueError(
                f"timeout={timeout}s exceeds the {self._max_foreground_timeout}s foreground limit. "
                "Run it with background=True: it keeps its log when the turn ends, and its exit "
                "comes back to you."
            )
        effective_timeout = self._timeout if timeout is None else max(1, timeout)

        # Sandbox: check blocked commands (best-effort, not a true security sandbox)
        from agentica.agent.approvals import approved_by_user

        if self._os_sandbox and escalate and not approved_by_user.get():
            # The approval layer routes every escalation to a human in
            # ask/auto; reaching here unapproved means nothing asked one (an
            # SDK agent without ``approve``), so the sandbox stays.
            raise PermissionError(
                "escalate=true runs the command outside the sandbox, which needs the "
                "user's approval, and nothing in this session can ask for it. Run it "
                "inside the sandbox, or tell the user the exact command to run."
            )

        sandbox_on = bool(self._sandbox_config and self._sandbox_config.enabled)
        skip_hard_block = approved_by_user.get() or not sandbox_on
        if sandbox_on and not skip_hard_block:
            cmd_lower = command.lower().strip()
            for blocked in self._sandbox_config.blocked_commands:
                if command_matches_blocked(command, blocked):
                    logger.warning(f"Sandbox: blocked command: {command[:100]}")
                    raise PermissionError(
                        "Sandbox blocked this command for security reasons."
                    )

            # Sandbox: check allowed_commands whitelist (prefix match on first token)
            # Only enforced when allowed_commands is explicitly set (non-None).
            allowed = self._sandbox_config.allowed_commands
            if allowed is not None:
                # Extract the first token (bare executable name, strip path prefix)
                first_token = cmd_lower.split()[0] if cmd_lower.split() else ""
                # Normalize: strip leading path (e.g. "/usr/bin/python3" → "python3")
                first_token_base = os.path.basename(first_token)
                if not any(
                    first_token_base == a.lower() or first_token_base.startswith(a.lower())
                    for a in allowed
                ):
                    logger.warning(
                        f"Sandbox: command '{first_token_base}' not in allowed_commands "
                        f"{allowed}: {command[:100]}"
                    )
                    raise PermissionError(
                        f"Sandbox blocked this command — '{first_token_base}' is not "
                        f"in the allowed_commands list: {allowed}"
                    )

        # Safety: dangerous command patterns. A catastrophic one (root delete,
        # mkfs, fork bomb, writing a disk device) is refused in every mode
        # unless a human approved this exact call; the other blocks apply only
        # with the sandbox on, so allow-all keeps running them (logged).
        safety = check_command_safety(command)
        if safety["action"] == "block":
            logger.warning(f"Safety blocked command: {safety['reason']} — {command[:100]}")
            if is_catastrophic(safety) and not approved_by_user.get():
                raise PermissionError(
                    f"{safety['reason']}. Refused in every permission mode; if this is really "
                    "intended, the user must run it themselves."
                )
            if not skip_hard_block:
                raise PermissionError(f"{safety['reason']}. Use a safer alternative.")
        if safety["action"] == "warn":
            logger.info(f"Safety warning: {safety['reason']} — {command[:100]}")

        logger.debug(f"Executing command: {command}")
        cwd = self._call_cwd(work_dir)
        self_detaching = bool(_SELF_DETACHING_COMMAND.search(command))
        policy: Optional[SandboxPolicy] = None
        argv = None
        if self._os_sandbox and not escalate:
            policy = await self._sandbox_policy()
            argv = wrap_command(_RUN_COMMAND_FROM_ENV, policy)
        elif os.name != "nt":
            argv = ["/bin/sh", "-c", _RUN_COMMAND_FROM_ENV]
        env = {_COMMAND_ENV: command} if argv is not None else None

        if background:
            if self._background_process_registry is None:
                raise ValueError(
                    "background is not available in this session — there is nothing "
                    "to track a detached command, so it could neither be waited on "
                    "nor stopped. Run it in the foreground, raising timeout if needed."
                )
            if self_detaching:
                raise ValueError(
                    "Remove the trailing '&': with background=True the shell would "
                    "fork the work away and exit at once, reporting a completion "
                    "while the command is still running. Pass the plain command."
                )
            item = self._background_process_registry.start(
                command, cwd=cwd, argv=argv, env=env,
            )
            return (
                f"Started background command #{item.num} "
                f"(PID {item.pid}, id: {item.id}).\n"
                f"Log: {item.log_path}"
            )

        slept = _LEADING_SLEEP.match(command)
        if slept and float(slept.group(1)) >= _MAX_FOREGROUND_SLEEP_SECONDS:
            raise ValueError(
                f"Refusing to hold this turn for {slept.group(1)}s. To wait for a "
                "background command, call wait(id=...): it returns the moment the "
                "command exits and reports its exit code, so it never overshoots. To "
                "wait on an external condition that has no completion event, retry "
                "until it succeeds instead of sleeping through the whole wait, e.g. "
                "`until curl -sf http://host/health; do sleep 5; done`."
            )

        proc = None
        timed_out = False
        drained = False
        stdout_spool = tempfile.SpooledTemporaryFile(max_size=self._max_output_length)
        stderr_spool = tempfile.SpooledTemporaryFile(max_size=self._max_output_length)
        hit_hard_cap = False
        out_bytes = err_bytes = 0
        out_lines = err_lines = 0
        try:
            try:
                try:
                    if argv is not None:
                        proc = await asyncio.create_subprocess_exec(
                            *argv,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=cwd,
                            env={**os.environ, **(env or {})},
                            start_new_session=True,
                        )
                    else:
                        proc = await asyncio.create_subprocess_shell(
                            command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=cwd,
                            start_new_session=os.name != "nt",
                        )
                except FileNotFoundError:
                    # The shell is on PATH; what is missing is the directory we
                    # asked to run in. Raised bare, this surfaces as
                    # ``[Errno 2] No such file or directory: '<dir>'`` — a path
                    # the model never mentioned, so it reads the error as a typo
                    # in its own command and retries variants instead. This
                    # happens for real when another session merges away the git
                    # worktree this one works in, and it takes every shell
                    # command down with it, so name the cause and the way out.
                    if cwd is not None and not os.path.isdir(cwd):
                        raise FileNotFoundError(
                            f"work_dir is not a directory: {cwd}\n"
                            "Nothing ran — the command is fine, the directory it "
                            "would run in is gone."
                        ) from None
                    raise
                out, err = await asyncio.wait_for(
                    self._drain_both(proc, stdout_spool, stderr_spool),
                    timeout=effective_timeout,
                )
                (out_bytes, out_lines, out_cap) = out
                (err_bytes, err_lines, err_cap) = err
                hit_hard_cap = out_cap or err_cap
                if not hit_hard_cap and proc.returncode is None:
                    await proc.wait()
                drained = True
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(f"Command timed out after {effective_timeout}s: {command}")
            finally:
                # A timeout or a cancelled turn leaves the group running with our
                # pipes open. `not drained` rather than `returncode is None`: the
                # shell in `cmd & ...` exits immediately while the grandchild it
                # backgrounded keeps the write end, which is the state where the
                # unclosed transport later dumps "Event loop is closed" into the
                # next turn's TUI.
                if proc is not None and not drained:
                    await terminate_subprocess(
                        proc,
                        process_group=True,
                        grace_period=5 if timed_out else 0,
                    )
                close_subprocess_transport(proc)

            if timed_out:
                # The drain was cut off, so its counts never came back; what it
                # wrote is still in the spools. Returning it is the difference
                # between "hung" and "hung after listing two of three dirs" or
                # "hung in test_foo" — the line that says what to do next.
                out_bytes, out_lines = _spool_size_and_lines(stdout_spool)
                err_bytes, err_lines = _spool_size_and_lines(stderr_spool)
            output = self._finish_captured_output(
                stdout_spool, stderr_spool,
                out_bytes=out_bytes, err_bytes=err_bytes,
                out_lines=out_lines, err_lines=err_lines,
                hit_hard_cap=hit_hard_cap,
                cwd=cwd,
            )
            if timed_out:
                if redact_tool_outputs_enabled():
                    output = redact_sensitive_text(output)
                raise TimeoutError(
                    f"Command timed out after {effective_timeout} seconds and was "
                    "killed. "
                    + (f"Output before the timeout:\n{output}" if output
                       else "It printed nothing before that.")
                )

            # -9 is the SIGKILL *we* sent for crossing the cap, not the
            # command failing. Reporting it as "[Exit code: -9]" (and then
            # raising on it below) would turn a deliberate truncation into a
            # tool error the model has to retry.
            killed_by_cap = hit_hard_cap and proc.returncode in (-9, -signal.SIGKILL)
            if proc.returncode and proc.returncode != 0 and not killed_by_cap:
                output = f"{output}\n\n[Exit code: {proc.returncode}]"
                # Every non-zero exit, "expected" ones included: curl's 7 and
                # git's 1 are exactly how a denied connection or write shows up.
                if policy is not None:
                    output = f"{output}\n\n{describe_policy(policy, cwd)}"

            logger.debug(f"Command exit code: {proc.returncode}")
            if not output:
                output = f"Command executed successfully (exit code: {proc.returncode})"

            if redact_tool_outputs_enabled():
                output = redact_sensitive_text(output)

            if self_detaching:
                output = (
                    f"{output}\n\n[Note: trailing '&' detached this work; "
                    "it is untracked.]"
                )

            if (
                proc.returncode
                and proc.returncode != 0
                and not killed_by_cap
                and not _expected_nonzero_exit(command, proc.returncode)
            ):
                raise RuntimeError(
                    f"Command exited with code {proc.returncode}.\n{output}"
                )

            return output
        finally:
            stdout_spool.close()
            stderr_spool.close()

    async def wait(self, id: str, timeout: int = _DEFAULT_WAIT_SECONDS) -> str:
        """Blocks until a background command finishes, then reports how it went.

        Returns the instant the command exits, so a generous timeout costs no
        more than the command itself. Use it when a later step of this turn needs
        the result of something already backgrounded. Whether a finished command
        also comes back to you on its own is stated under ``execute``.

        It is not the default. Something you need the result of, that fits in one
        tool call, should just run in the foreground with a raised ``timeout``.
        And do not loop on it: if the command is still running after a wait or
        two, it is long enough that holding the turn is worse than ending it —
        say what is running and stop; its completion drives the next step.

        Never guess a duration with ``sleep``: a fixed wait keeps waiting after
        the command has already failed, and cannot tell you its exit code.

        Args:
            id: Background command id from execute(background=True), e.g. "term_4"
            timeout: Seconds this one call may wait (default 300, no upper cap —
                same as ``execute(timeout=...)``). Reaching it does not stop the
                command: the result reports the progress so far and you may wait
                again.

        Returns:
            str: Exit code and log tail once it finishes, or its output so far
        """
        item = self._background_process_registry.get(id)
        if item is None:
            started = self._background_process_registry.list(include_finished=True)
            known = [p.id for p in started]
            # Listing every id ever started (92 at the top of a long session)
            # said nothing about which one is meant; the recent few and the
            # ones still running are what the model can act on.
            running = [p.id for p in started if p.running]
            recent = known[-5:]
            hint = (
                f"Still running: {', '.join(running)}. "
                if running else "Nothing is running now. "
            ) + (
                f"Most recent ids: {', '.join(recent)} ({len(known)} started this session)."
                if known else "None have been started in this session."
            )
            raise ValueError(
                f"No background command {id!r}. {hint}"
            )

        effective_timeout = max(1, int(timeout))
        deadline = time.monotonic() + effective_timeout
        while not item.finished.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(_WAIT_POLL_INTERVAL)

        # A delegated session's stdout is its final report, not a command log,
        # so it is named for the task and quoted at length.
        delegated = item.kind == "delegate"
        what = f'Delegated task "{item.label}"' if delegated else f"Background command #{item.num}"
        tail = read_log_tail(
            item.log_path,
            max_lines=120 if delegated else 40,
            max_chars=8000 if delegated else 4000,
        )
        if item.finished.is_set():
            header = (
                f"{what} ({item.id}) exited with code "
                f"{item.returncode} after {item.elapsed}."
            )
        else:
            header = (
                f"{what} ({item.id}) is still running after "
                f"{item.elapsed}; this wait timed out but it was not stopped. "
                f"If it has already outlasted a wait or two, stop waiting: end your "
                f"turn and let the completion notice the user gets drive the next step."
            )
        lines = [header, f"Log: {item.log_path}"]
        if not delegated:
            lines.insert(1, f"Command: {item.command}")
        if tail:
            lines.append("")
            lines.append(tail)
        return "\n".join(lines)

