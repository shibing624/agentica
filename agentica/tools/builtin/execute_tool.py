# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in execute/wait tools
"""
import asyncio
import os
import re
import time
from pathlib import Path
from typing import Optional


from agentica.tools.base import Tool
from agentica.tools.background_processes import BackgroundProcessRegistry, read_log_tail
from agentica.tools.safety import (
    check_command_safety,
    command_matches_blocked,
    redact_sensitive_text,
)
from agentica.security.redact import redact_tool_outputs_enabled
from agentica.utils.async_utils import close_subprocess_transport, terminate_subprocess
from agentica.utils.log import logger

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


def _expected_nonzero_exit(command: str, exit_code: int) -> bool:
    """True when this command's non-zero exit is a normal result."""
    if exit_code == 0:
        return False
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()
    words = last_segment.split()
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
    return exit_code in _EXPECTED_NONZERO.get(base_cmd, ())


# ─── File safety guards ──────────────────────────────────────────────────────
# Ported from hermes-agent tools/file_tools.py

# Paths that would hang the process (infinite output or blocking input)

_SELF_DETACHING_COMMAND = re.compile(r"(?<!&)&\s*$")
_LEADING_SLEEP = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)\b")
_MAX_FOREGROUND_SLEEP_SECONDS = 120

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
  once. Use it when the output is not needed to continue this turn, or when the
  command may outlive one tool call: a foreground command killed by ``timeout``
  or a cancelled turn loses everything it printed, while a background one keeps
  its log. Something you need right now that fits in one call belongs in the
  foreground with a raised ``timeout`` — reach for ``parallel_safe`` rather than
  ``background`` when the goal is speed. A background command's exit is reported
  to the user, not to you; ``wait`` with the returned id is what brings it back
  to you. Never improvise a wait with ``sleep``, polling, or a blocking ``tail``;
  a 120s+ leading ``sleep`` is refused. A trailing ``&`` detaches into an
  untracked orphan that nothing can wait on or stop, so prefer
  ``background=True``.
  Example: execute(command="make release", background=True)
"""


class BuiltinExecuteTool(Tool):
    """
    Built-in command execution tool using async subprocess.
    Exposed as execute function for consistent naming in Agent.
    """

    def __init__(self, work_dir: Optional[str] = None, timeout: int = 120,
                 max_output_length: int = 20000, sandbox_config=None,
                 background_process_registry: Optional[BackgroundProcessRegistry] = None):
        """
        Initialize BuiltinExecuteTool.

        Args:
            work_dir: Work directory for command execution
            timeout: Default command execution timeout in seconds. Callers may
                override per-invocation via ``execute(timeout=...)``; overrides
                are applied as-is with no upper cap — the caller decides.
            max_output_length: Maximum length of output to return
            sandbox_config: SandboxConfig instance for command restriction enforcement
            background_process_registry: shared registry owning detached
                commands. Supplied by surfaces that can also list, report and
                stop them (the interactive CLI). Without one, ``background``
                and ``wait`` are not offered at all rather than being backed by
                a private registry nobody can reach: a detached process would
                outlive the agent with no way to see it or kill it.
        """
        super().__init__(name="builtin_execute_tool")
        self._work_dir: Optional[Path] = Path(work_dir) if work_dir else None
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._sandbox_config = sandbox_config
        self._background_process_registry = background_process_registry
        # Override timeout from sandbox config if set
        if sandbox_config and sandbox_config.enabled and sandbox_config.max_execution_time:
            self._timeout = sandbox_config.max_execution_time
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
        if self._background_process_registry is not None:
            execute_fn.description += _BACKGROUND_GUIDANCE
            self.register(self.wait, concurrency_safe=True, is_read_only=True, is_destructive=False)
            # `wait` may block past the outer 120s executor wrapper; it owns its
            # timeout so the caller-supplied value is not cut short mid-wait.
            self.functions["wait"].manages_own_timeout = True
        else:
            execute_fn.parameters["properties"].pop("background", None)
        execute_fn.skip_entrypoint_processing = True

    def set_work_dir(self, work_dir: str) -> None:
        """Run subsequent commands in another directory, mid-session.

        Counterpart of ``BuiltinFileTool.set_work_dir``: a session that moves
        into a git worktree must run its tests and its git commands there, not
        in the checkout it happened to start in. Deliberately not a registered
        tool function — only ``Agent.rebind_work_dir`` moves a session, and it
        moves everything at once.
        """
        self._work_dir = Path(work_dir)

    async def execute(
            self,
            command: str,
            timeout: Optional[int] = None,
            background: bool = False,
            parallel_safe: bool = False,
    ) -> str:
        """Executes a shell command, capturing both stdout and stderr.

        Any shell command goes here: programs (git, python, pytest, pip, npm,
        make, docker, curl) and pipelines that shape command output — filter,
        sort, unique, count, head, tail.

        Prefer one long ``execute`` over many short ones when the steps share a
        directory. Each extra call is another model round-trip. Pack a verify /
        build / launch sequence with pipes, ``&&``, and a heredoc
        (``python3 - <<'EOF'`` … ``EOF``). Newlines in the command string are
        required for a heredoc and are passed through unchanged.

        You own what comes back. Bound each noisy program with ``| head`` /
        ``| tail``; oversized output is persisted to a session file and the
        context keeps only a head/tail preview plus the path. Chain dependent
        commands with ``&&``, not ``;``. Check state read-only before a write.
        Surgical, context-sensitive edits still belong in ``apply_patch``; a
        one-shot script is for the same substitution across several files, or
        work that is already a program.

        Before executing:
        1. Verify target directory exists (use glob first if unsure)
        2. Always quote file paths with spaces: cd "/path with spaces/"
        3. For a multi-step in one tree, start with ``cd /abs/path &&``

        Usage notes:
        - The command string is passed unchanged to the system shell after
          safety validation. Safety policies may block a command, but never
          rewrite it. Quotes, escapes, newlines, and source code remain exact.
        - Foreground commands timeout after 120 seconds by default
        - You may specify a custom ``timeout`` (in seconds) for long-running
          commands; there is no upper cap — the caller decides.
        - ``parallel_safe=True`` lets this call run at the same time as the
          other calls in the same message. Default is False: calls run one at a
          time, in the order you issued them, and a failing command cancels the
          ones after it. Set it only when every command in the batch is
          independent of the others — nothing one writes is read or written by
          another. Two test suites, two `git log` reads, two builds in separate
          directories: yes. `git add` then `git commit`, `pip install` then the
          command that imports it, anything touching one file twice: no, and
          `&&` in a single call is the right way to say that anyway. When
          unsure, leave it off; the cost is waiting, and the cost of getting it
          wrong is a corrupted working tree.
        - stdout and stderr are decoded as UTF-8; invalid bytes are replaced.
          Oversized output is persisted to a session file with a head/tail
          preview and path. When output redaction is enabled, detected secrets
          are replaced before the result reaches you. Unterminated PEM
          private-key blocks are always redacted.

        Git safety:
        - Prefer creating new commits over amending existing ones
        - Before destructive operations (git reset --hard, git push --force),
          consider safer alternatives and check with the user first
        - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign)
          unless the user explicitly requests it

        Examples:
            - One verify-then-next-step call (preferred over N short executes)::

                  cd /abs/project && pytest -q --tb=no | rg '^FAILED' | sort && python -m build 2>&1 | tail -8

            - Same substitution across several files, then a peek (newlines are the command)::

                  cd /abs/project && python3 - <<'EOF'
                  import pathlib
                  edits = {
                      'src/app/store.py': ('OldName', 'NewName'),
                      'docs/USAGE.md': ('OldName is', 'NewName is'),
                  }
                  for path, (old, new) in edits.items():
                      p = pathlib.Path(path)
                      p.write_text(p.read_text().replace(old, new))
                      print('updated', path)
                  EOF
                  echo '=== pyproject.toml ==='
                  python3 -c "from pathlib import Path; print(chr(10).join(Path('pyproject.toml').read_text().splitlines()[:25]))"

            - execute(command="API_ENV=dev python3 scripts/smoke.py && sleep 2 && curl -sI http://127.0.0.1:8000 | head -8")
            - execute(command="pytest tests/gateway -q --tb=no | rg '^FAILED' | sort")
            - execute(command="rg -n '^## ' CHANGELOG.md | head -20")
            - execute(command="git diff --stat | tail -5")
            - execute(command="npm install && npm test", timeout=300)
            - execute(command="pytest tests/unit -q", parallel_safe=True)
              alongside execute(command="pytest tests/e2e -q", parallel_safe=True)

        Args:
            command: Exact shell command to execute without normalization or repair
            timeout: optional timeout in seconds (default 120, no upper cap)
            parallel_safe: run concurrently with the other calls in this
                message. Only for commands independent of every sibling call;
                the default runs the batch serially in order

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        # `parallel_safe` is read by the executor (via Function.parallel_arg) to
        # decide the schedule before this ever runs; it means nothing here.
        del parallel_safe

        # Apply timeout: use per-call override if provided, else the tool default.
        # No upper cap — the caller decides.
        effective_timeout = self._timeout if timeout is None else max(1, timeout)

        # Sandbox: check blocked commands (best-effort, not a true security sandbox)
        from agentica.agent.approvals import approved_by_user

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

        # Safety: check dangerous command patterns (always active unless a
        # human already allowed this call — otherwise allow-all / the card
        # would still raise PermissionError).
        safety = check_command_safety(command)
        if safety["action"] == "block":
            logger.warning(f"Safety blocked command: {safety['reason']} — {command[:100]}")
            if not skip_hard_block:
                raise PermissionError(f"{safety['reason']}. Use a safer alternative.")
        if safety["action"] == "warn":
            logger.info(f"Safety warning: {safety['reason']} — {command[:100]}")

        logger.debug(f"Executing command: {command}")
        cwd = str(self._work_dir) if self._work_dir else None
        self_detaching = bool(_SELF_DETACHING_COMMAND.search(command))

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
            item = self._background_process_registry.start(command, cwd=cwd)
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
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=os.name != "nt",
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout,
            )
            drained = True
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning(f"Command timed out after {effective_timeout}s: {command}")
            raise TimeoutError(
                f"Command timed out after {effective_timeout} seconds"
            ) from None
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

        # Combine stdout and stderr
        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")

        output = "\n".join(output_parts).strip()

        if proc.returncode and proc.returncode != 0:
            output = f"{output}\n\n[Exit code: {proc.returncode}]"

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
            and not _expected_nonzero_exit(command, proc.returncode)
        ):
            raise RuntimeError(
                f"Command exited with code {proc.returncode}.\n{output}"
            )

        return output

    async def wait(self, id: str, timeout: int = _DEFAULT_WAIT_SECONDS) -> str:
        """Blocks until a background command finishes, then reports how it went.

        Returns the instant the command exits, so a generous timeout costs no
        more than the command itself. Use it when a later step of your plan needs
        the result of something already backgrounded: a background command's exit
        is reported to the user and never to you, so this is the only way to
        reach that later step.

        It is not the default. Something you need the result of, that fits in one
        tool call, should just run in the foreground with a raised ``timeout``.
        And do not loop on it: if the command is still running after a wait or
        two, it is long enough that holding the turn is worse than ending it —
        say what is running and stop, and the completion notice the user gets
        drives the next step.

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
            known = [p.id for p in self._background_process_registry.list(include_finished=True)]
            raise ValueError(
                f"No background command {id!r}. "
                + (f"Started so far: {', '.join(known)}." if known
                   else "None have been started in this session.")
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

