# -*- coding: utf-8 -*-
"""Tests for BuiltinExecuteTool."""
import asyncio
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentica.tools.background_processes import (
    BackgroundProcess,
    BackgroundProcessRegistry,
    read_log_tail,
)
from agentica.tools.builtin import BuiltinExecuteTool


class _BlockingPipe:
    def __init__(self, started: asyncio.Event):
        self._started = started

    async def read(self, n: int = -1) -> bytes:
        self._started.set()
        await asyncio.Future()
        return b""


class BlockingSubprocess:
    """Minimal subprocess double whose first pipe read blocks."""

    def __init__(self):
        self.started = asyncio.Event()
        self.returncode = None
        self._transport = None
        self.stdout = _BlockingPipe(self.started)
        self.stderr = _BlockingPipe(self.started)


class TestBuiltinExecuteTool:
    def test_execute_registered_as_raw_string_tool(self, execute_tool):
        function = execute_tool.functions["execute"]
        function.process_entrypoint(strict=False)

        assert "passed unchanged" in function.description

    def test_background_command_management_tools_not_registered(self, execute_tool):
        assert "list_background_commands" not in execute_tool.functions
        assert "stop_background_command" not in execute_tool.functions

    def test_execute_simple_command(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo hello"))
        assert "hello" in result

    def test_execute_docstring_allows_pipeline_rg(self):
        doc = BuiltinExecuteTool.execute.__doc__ or ""
        assert "NOT grep, rg, or ag" not in doc
        assert "NOT cat, head, tail" not in doc
        assert "not cat" not in doc
        assert "not find" not in doc
        assert "sed -i" not in doc
        assert "Bad examples" not in doc
        assert "| head" in doc or "| rg" in doc
        assert "rg -n TODO src || grep -n TODO src" in doc
        assert "2>/dev/null" in doc
        assert "apply_patch" in doc
        assert "write_file" in doc
        assert "package layout" in doc
        assert "multi-file" in doc
        assert "not `;`" not in doc
        assert "write_text" not in doc
        assert "find . -type f" not in doc
        assert "xargs ls" not in doc
        assert "(find, ls, cat, awk)" not in doc
        assert "Prefer one long" in doc
        assert "<<'EOF'" in doc
        assert "read_file" in doc
        assert "DO NOT use newlines" not in doc
        assert "avoid cd when possible" not in doc
        assert "swift" not in doc
        assert "App.app" not in doc

    def test_execute_runs_heredoc_and_chained_echo(self, execute_tool, tmp_dir):
        """A multi-line command is the product: newlines stay, && / heredoc run."""
        command = (
            f"cd {tmp_dir} && {sys.executable} - <<'EOF'\n"
            "print('from-heredoc')\n"
            "EOF\n"
            "echo after"
        )
        result = asyncio.run(execute_tool.execute(command))
        assert "from-heredoc" in result
        assert "after" in result

    def test_execute_oversized_output_is_persisted_not_returned_whole(self, tmp_dir, monkeypatch):
        """A cat of a huge file must not enter the result string. Layer 1
        cannot evict the live round, so the capture itself has to bound it."""
        projects = Path(tmp_dir) / "projects"
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(projects))
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=400)
        payload = "H" * 80 + "M" * 400 + "T" * 80
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(' + repr(payload) + ')')}"
        result = asyncio.run(tool.execute(command))
        assert "<persisted-output>" in result
        assert "M" * 400 not in result
        assert "...[truncated]" not in result
        path_line = next(
            ln.strip() for ln in result.splitlines()
            if ln.strip().endswith(".txt") and "tool-results" in ln.replace("\\", "/")
        )
        assert payload in Path(path_line).read_text(encoding="utf-8")
        assert tool.functions["execute"].max_result_size_chars == 400

    def test_execute_error_path_bounds_output_before_raise(self, tmp_dir, monkeypatch):
        """Failing commands still must not leak the full dump through error."""
        projects = Path(tmp_dir) / "projects"
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(projects))
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=400)
        payload = "M" * 2000
        command = (
            f"{shlex.quote(sys.executable)} -c "
            f"{shlex.quote('print(' + repr(payload) + '); import sys; sys.exit(1)')}"
        )
        with pytest.raises(RuntimeError, match="exited with code 1") as excinfo:
            asyncio.run(tool.execute(command))
        err = str(excinfo.value)
        assert "<persisted-output>" in err
        assert "M" * 2000 not in err

    def test_failed_execute_persists_via_layer0(self, tmp_dir, monkeypatch):
        """A failing command's oversized output reaches the model as a
        persisted preview + path, not a truncation — the Layer 0 hook runs on
        the assembled result content (function_call.error) as well."""
        from agentica.model.base import Model
        from agentica.tools.base import FunctionCall
        from agentica.tools.builtin import BuiltinFileTool

        projects = Path(tmp_dir) / "projects"
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(projects))

        class _M(Model):
            @property
            def request_kwargs(self):
                return {}

            async def invoke(self, messages):
                raise NotImplementedError

            async def invoke_stream(self, messages):
                if False:
                    yield None

            async def response(self, messages):
                raise NotImplementedError

            async def response_stream(self, messages):
                if False:
                    yield None

        model = _M(id="fake-test")
        execute_tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=400)
        model.functions = dict(execute_tool.functions)
        model.functions.update(BuiltinFileTool(work_dir=tmp_dir).functions)

        payload = "M" * 2000
        command = (
            f"{shlex.quote(sys.executable)} -c "
            f"{shlex.quote('print(' + repr(payload) + '); import sys; sys.exit(1)')}"
        )
        fc = FunctionCall(
            function=model.functions["execute"],
            arguments={"command": command},
            call_id="call_err_1",
        )

        async def _run():
            results = []
            async for _ in model.run_function_calls(
                function_calls=[fc], function_call_results=results,
            ):
                pass
            return results

        results = asyncio.run(_run())
        content = results[0].content or ""
        assert results[0].tool_call_error is True
        assert "<persisted-output>" in content
        assert "M" * 2000 not in content  # context keeps only the preview
        path_line = next(
            ln.strip() for ln in content.splitlines()
            if ln.strip().endswith(".txt") and "tool-results" in ln.replace("\\", "/")
        )
        assert "M" * 100 in Path(path_line).read_text(encoding="utf-8")

    def test_execute_background_registers_process(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry(user_id="alice@example.com")
        tool = BuiltinExecuteTool(
            work_dir=tmp_dir,
            background_process_registry=registry,
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"

        result = asyncio.run(tool.execute(command, background=True))

        try:
            running = registry.list()
            assert len(running) == 1
            item = running[0]
            assert f"PID {item.pid}" in result
            assert f"id: {item.id}" in result
            assert "Log:" in result
            assert Path(item.log_path).exists()
            assert str(Path(tmp_dir) / ".agentica") not in item.log_path
            projects_dir = Path(
                os.environ.get(
                    "AGENTICA_PROJECTS_DIR",
                    str(agentica_home / "projects"),
                )
            )
            assert str(projects_dir / "alice@example.com") in item.log_path
            assert str(projects_dir / "default") not in item.log_path
        finally:
            registry.stop()
        assert registry.running_count() == 0

    def test_execute_background_result_is_id_pid_and_log(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"

        try:
            result = asyncio.run(tool.execute(command, background=True))
        finally:
            registry.stop()

        assert "Started background command #1" in result
        assert "id: term_1" in result
        assert "Log:" in result
        assert "reported to the user" not in result
        assert "wait(id=" not in result

    def test_execute_runs_self_detaching_command_but_flags_it(self, execute_tool):
        result = asyncio.run(execute_tool.execute("nohup echo started > /dev/null 2>&1 &"))

        assert "untracked" in result
        assert "background=True" not in result

    def test_self_detaching_command_stays_a_short_note(self, bg_execute_tool):
        result = asyncio.run(bg_execute_tool.execute("nohup echo started > /dev/null 2>&1 &"))

        assert "untracked" in result
        assert "It is untracked" not in result

    def test_execute_refuses_self_detaching_command_in_background_mode(self, bg_execute_tool):
        """Here the '&' is not merely untracked but wrong: the registry would
        watch a shell that exits at once and announce a completion while the
        command is still running."""
        with pytest.raises(ValueError, match="Remove the trailing '&'"):
            asyncio.run(bg_execute_tool.execute("python3 run.py &", background=True))

    def test_background_is_not_offered_without_a_registry(self, execute_tool):
        """An SDK agent has nowhere to register a detached command: no listing,
        no completion report, no way to stop it. Offering the knob there buys
        an orphan that outlives the agent, so the schema must not carry it."""
        properties = execute_tool.functions["execute"].parameters["properties"]

        assert "background" not in properties
        assert "wait" not in execute_tool.functions
        assert "background=True" not in execute_tool.functions["execute"].description

    def test_a_forced_background_call_starts_no_process(self, execute_tool):
        """Belt and braces for a model that passes the argument anyway."""
        with pytest.raises(ValueError, match="background is not available"):
            asyncio.run(execute_tool.execute("sleep 30", background=True))

    def test_background_is_offered_with_a_registry(self, bg_execute_tool):
        properties = bg_execute_tool.functions["execute"].parameters["properties"]

        assert "background" in properties
        assert "wait" in bg_execute_tool.functions
        assert "background=True" in bg_execute_tool.functions["execute"].description

    def test_execute_leaves_plain_commands_unflagged(self, execute_tool):
        """`2>&1` contains an ampersand without detaching anything."""
        result = asyncio.run(execute_tool.execute("echo ok 2>&1"))

        assert "ok" in result
        assert "untracked" not in result

    def test_execute_refuses_long_foreground_sleep(self, execute_tool):
        """The observed poll: background the job, then `sleep 330 && tail log`,
        which re-blocks the turn backgrounding had just freed."""
        with pytest.raises(ValueError, match="Refusing to hold this turn") as excinfo:
            asyncio.run(execute_tool.execute("sleep 330 && tail -2 /tmp/run.log"))

        # A caller waiting on something Agentica does not track needs the correct
        # form, not just a refusal.
        assert "until curl -sf" in str(excinfo.value)
        # The refusal must name the primitive that replaces the blind sleep.
        assert "wait(id=...)" in str(excinfo.value)

    def test_execute_allows_retry_loop_waiting_on_external_condition(self, execute_tool):
        """The recommended form exits on success, so it must not be refused."""
        result = asyncio.run(
            execute_tool.execute("until echo ready; do sleep 5; done")
        )
        assert "ready" in result

    def test_execute_allows_short_sleep_for_service_startup(self, execute_tool):
        result = asyncio.run(execute_tool.execute("sleep 0.05 && echo up"))
        assert "up" in result

    def test_wait_returns_as_soon_as_the_command_exits(self, tmp_dir):
        """The point of `wait` over `sleep N`: a generous timeout costs only as
        much wall time as the command itself."""
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'import time; time.sleep(0.05); print("summary ready")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            started = await tool.execute(command, background=True)
            item_id = registry.list(include_finished=True)[0].id
            assert item_id in started
            return await tool.wait(item_id, timeout=120)

        began = time.monotonic()
        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()
        elapsed = time.monotonic() - began

        assert "exited with code 0" in result
        assert "summary ready" in result
        assert elapsed < 30

    def test_wait_reports_progress_without_stopping_the_command(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'import time; print("phase 1", flush=True); time.sleep(30)'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            await tool.execute(command, background=True)
            item_id = registry.list()[0].id
            clock = {"t": 0.0}

            def fake_monotonic():
                return clock["t"]

            async def fake_sleep(seconds):
                clock["t"] += seconds

            with (
                patch("agentica.tools.builtin.execute_tool.time.monotonic", fake_monotonic),
                patch("agentica.tools.builtin.execute_tool.asyncio.sleep", fake_sleep),
            ):
                return await tool.wait(item_id, timeout=1)

        try:
            result = asyncio.run(scenario())
            assert "still running" in result
            assert "phase 1" in result
            assert registry.running_count() == 1
            # A job on the scale of hours must not be waited on in a loop; the
            # user's completion notice is what should drive the next step.
            assert "stop waiting: end your turn" in result
        finally:
            registry.stop()

    def test_wait_on_finished_command_returns_immediately(self, tmp_dir):
        """A command that finished while the agent did something else must still
        be reportable — otherwise the result is lost to the conversation."""
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'print("early")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            await tool.execute(command, background=True)
            item = registry.list(include_finished=True)[0]
            assert item.finished.wait(timeout=30)
            return await tool.wait(item.id, timeout=300)

        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "exited with code 0" in result
        assert "early" in result

    def test_wait_reports_a_failing_command_exit_code(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('raise SystemExit(3)')}"

        async def scenario():
            await tool.execute(command, background=True)
            return await tool.wait(registry.list(include_finished=True)[0].id, timeout=60)

        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "exited with code 3" in result

    def test_wait_on_unknown_id_lists_known_ids(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(1)')}"

        async def scenario():
            await tool.execute(command, background=True)
            await tool.wait("term_99", timeout=1)

        try:
            with pytest.raises(ValueError, match="No background command 'term_99'") as excinfo:
                asyncio.run(scenario())
        finally:
            registry.stop()

        assert "term_1" in str(excinfo.value)

    def test_wait_honors_timeout_above_default(self, bg_execute_tool):
        """Caller-supplied timeout is applied as-is — same as execute(timeout=...).

        A silent 300s clamp made ``wait(timeout=600)`` behave like 300, so the
        model could not actually ask for a longer single wait.
        """
        execute_tool = bg_execute_tool
        assert execute_tool.functions["wait"].manages_own_timeout is True

        registry = execute_tool._background_process_registry
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        async def fake_sleep(seconds):
            clock["t"] += seconds

        async def scenario():
            await execute_tool.execute(command, background=True)
            item_id = registry.list()[0].id
            with (
                patch("agentica.tools.builtin.execute_tool.time.monotonic", fake_monotonic),
                patch("agentica.tools.builtin.execute_tool.asyncio.sleep", fake_sleep),
            ):
                result = await execute_tool.wait(item_id, timeout=400)
            return result, clock["t"]

        try:
            result, waited = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "still running" in result
        assert waited >= 400, f"timeout=400 was clamped; only waited {waited:.1f}s"

    def test_execute_background_emits_completion_event(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry(user_id="alice@example.com")
        tool = BuiltinExecuteTool(
            work_dir=tmp_dir,
            background_process_registry=registry,
        )
        script = 'print("done")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        result = asyncio.run(tool.execute(command, background=True))
        event = registry.wait_completed(timeout=5)

        assert "Started background command #1" in result
        assert event.id == "term_1"
        assert event.num == 1
        assert event.returncode == 0
        assert event.stop_requested is False
        assert "done" in Path(event.log_path).read_text(encoding="utf-8")
        assert registry.running_count() == 0

    def test_background_stop_marks_completion_event_as_stop_requested(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
        item = registry.start(command, cwd=tmp_dir)

        stopped = registry.stop(item.id)
        event = registry.wait_completed(timeout=5)

        assert stopped == [item]
        assert event.id == item.id
        assert event.stop_requested is True
        with pytest.raises(queue.Empty):
            registry.wait_completed(timeout=0.01)

    def test_registry_passes_extra_env_to_the_child_without_replacing_it(self, tmp_dir):
        """The delegate tool marks a worker's depth this way; the child still
        needs the rest of the environment (PATH, API keys) to run at all."""
        registry = BackgroundProcessRegistry()
        script = 'import os; print(os.environ["DEPTH_MARKER"], "PATH" in os.environ)'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        item = registry.start(command, cwd=tmp_dir, env={"DEPTH_MARKER": "1"})
        assert item.finished.wait(timeout=30)

        assert "1 True" in Path(item.log_path).read_text(encoding="utf-8")

    def test_a_multi_line_command_does_not_leak_into_the_output(self, tmp_dir):
        """A delegated worker's command carries its whole task, newlines and all.
        The log header has to stay one line or readers, which strip the header by
        dropping a leading `$ ` line, hand the rest of it back as output."""
        registry = BackgroundProcessRegistry()
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(1)')} # first line\\\n# second line"

        item = registry.start(command, cwd=tmp_dir)
        assert item.finished.wait(timeout=30)

        assert read_log_tail(item.log_path) == "1"

    def test_a_delegated_process_is_counted_and_reported_as_one(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(1)')}"

        registry.start(command, cwd=tmp_dir)
        registry.start(command, cwd=tmp_dir, kind="delegate", label="parser port")
        event_kinds = {
            registry.wait_completed(timeout=10).kind,
            registry.wait_completed(timeout=10).kind,
        }

        assert registry.running_count(kind="delegate") == 0
        assert [p.label for p in registry.list(include_finished=True, kind="delegate")] == ["parser port"]
        assert event_kinds == {"command", "delegate"}

    def test_background_stop_tolerates_wait_after_sigkill_timeout(self):
        registry = BackgroundProcessRegistry()
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("command", 2),
            subprocess.TimeoutExpired("command", 2),
        ]
        item = BackgroundProcess(
            id="term_1",
            num=1,
            process=process,
            command="command",
            cwd=None,
            log_path="/tmp/term_1.log",
            started_at=0,
        )
        registry._items[item.id] = item

        with patch("agentica.tools.background_processes.os.killpg") as killpg:
            stopped = registry.stop(item.id)

        assert stopped == [item]
        assert killpg.call_count == 2

    def test_background_start_removes_log_when_popen_fails(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry()

        with patch(
            "agentica.tools.background_processes.subprocess.Popen",
            side_effect=OSError("cannot start"),
        ), pytest.raises(OSError, match="cannot start"):
            registry.start("command", cwd=str(tmp_dir))

        assert list(agentica_home.rglob("*.log")) == []

    def test_execute_returns_exit_code_on_failure(self, execute_tool):
        with pytest.raises(RuntimeError, match="exit(ed)? (with )?code 42"):
            asyncio.run(execute_tool.execute("exit 42"))

    def test_execute_treats_python_module_linter_exit_one_as_diagnostics(self, execute_tool, tmp_dir):
        Path(tmp_dir, "ruff.py").write_text(
            "import sys\n"
            "print('UP009 UTF-8 encoding declaration is unnecessary')\n"
            "sys.exit(1)\n"
        )

        result = asyncio.run(
            execute_tool.execute(
                f"PYTHONPATH={shlex.quote(tmp_dir)} python3 -m ruff check sample.py"
            )
        )

        assert "UP009" in result
        assert "[Exit code: 1]" in result
        assert "Diagnostics found" not in result

    def test_execute_still_raises_for_plain_python3_exit_one(self, execute_tool):
        with pytest.raises(RuntimeError, match="Command exited with code 1"):
            asyncio.run(execute_tool.execute("python3 -c 'import sys; sys.exit(1)'"))

    def test_execute_captures_stderr(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo error_msg >&2"))
        assert "error_msg" in result

    def test_execute_timeout(self):
        tool = BuiltinExecuteTool(timeout=0.05)
        with pytest.raises(TimeoutError, match="timed out"):
            asyncio.run(tool.execute("sleep 30"))

    def test_execute_cancellation_reaps_from_finally(self, tmp_dir):
        """Ctrl+C must reap from `finally`, which also covers the timeout path.

        `except CancelledError` reached only one of the two ways a command is
        abandoned, and left the other holding a subprocess and its pipes.
        """
        async def cancel_running_command():
            process = BlockingSubprocess()
            cleanup = AsyncMock()
            with patch(
                "agentica.tools.builtin.execute_tool.asyncio.create_subprocess_shell",
                new=AsyncMock(return_value=process),
            ), patch(
                "agentica.tools.builtin.execute_tool.terminate_subprocess",
                cleanup,
            ):
                tool = BuiltinExecuteTool(work_dir=tmp_dir)
                task = asyncio.create_task(tool.execute("sleep 60"))
                await process.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            cleanup.assert_awaited_once()
            assert cleanup.await_args.args == (process,)
            assert cleanup.await_args.kwargs["process_group"] is True

        asyncio.run(cancel_running_command())

    @pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup")
    def test_execute_cancellation_reaps_subprocess_group(self, tmp_dir):
        pid_file = Path(tmp_dir, "child.pid")
        script = (
            "import os, time; "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "time.sleep(60)"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
        tool = BuiltinExecuteTool(work_dir=tmp_dir)
        child_pid = None

        async def cancel_running_command():
            task = asyncio.create_task(tool.execute(command))
            for _ in range(200):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists(), "child process did not start"

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            pid = int(pid_file.read_text())
            for _ in range(200):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return pid
                await asyncio.sleep(0.01)
            return pid

        try:
            child_pid = asyncio.run(cancel_running_command())
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if child_pid is not None:
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass

    def test_execute_python_code(self, execute_tool):
        result = asyncio.run(execute_tool.execute("python3 -c 'print(2+3)'"))
        assert "5" in result

    def test_execute_preserves_python_single_quotes(self, execute_tool):
        python = shlex.quote(sys.executable)
        command = f'{python} -c "from pathlib import Path; print(Path(\'.\').resolve().name)"'

        result = asyncio.run(execute_tool.execute(command))

        assert result == Path(execute_tool._work_dir).name

    def test_execute_preserves_literal_backslash_n(self, execute_tool):
        python = shlex.quote(sys.executable)
        command = f'''{python} -c 'print("\\n".join(["a", "b"]))' '''.strip()

        result = asyncio.run(execute_tool.execute(command))

        assert result == "a\nb"

    def test_execute_call_arguments_remain_exact(self, execute_tool):
        from agentica.tools.base import get_function_call

        execute_tool.functions["execute"].process_entrypoint(strict=False)
        commands = [
            "true",
            "  printf preserved  ",
            'python3 -c "print(True, False, None)"',
        ]
        for command in commands:
            function_call = get_function_call(
                "execute",
                json.dumps({"command": command, "timeout": "5"}),
                functions=execute_tool.functions,
            )
            assert function_call.error is None
            assert function_call.arguments["command"] == command
            assert function_call.arguments["timeout"] == 5

    def test_execute_multiline_python(self, execute_tool):
        cmd = '''python3 -c "def f(n):
    return n * 2
print(f(21))"'''
        result = asyncio.run(execute_tool.execute(cmd))
        assert "42" in result

    def test_execute_cwd(self, tmp_dir):
        tool = BuiltinExecuteTool(work_dir=tmp_dir)
        result = asyncio.run(tool.execute("pwd"))
        assert tmp_dir in result


class TestExecuteOutputCap:
    """The capture hard cap: it must bound the turn, not pin it.

    ``_drain_stream`` returns as soon as *its* stream crosses the cap while
    the child is still writing. Waiting for both drains (``gather``) then
    blocks on an EOF that cannot arrive through a pipe nobody is reading,
    so the command hangs until ``timeout`` and the cap never fires.
    """

    @staticmethod
    def _run(cmd, **kwargs):
        tool = BuiltinExecuteTool(work_dir="/tmp", **kwargs)
        return asyncio.run(tool.execute(cmd, timeout=20))

    def test_generator_over_the_cap_returns_promptly(self, tmp_dir, monkeypatch):
        """A command that never stops writing must be killed, not waited out."""
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(Path(tmp_dir) / "projects"))
        # Emit slightly more than the cap, then keep going forever.
        overflow = (
            "import sys\n"
            "buf = b'x' * (1 << 20)\n"
            "w = sys.stdout.buffer.write\n"
            "for _ in range(70):\n"
            "    w(buf)\n"
            "while True:\n"
            "    w(buf)\n"
        )
        script = Path(tmp_dir) / "flood.py"
        script.write_text(overflow)

        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        started = time.monotonic()
        result = asyncio.run(tool.execute(f"{shlex.quote(sys.executable)} {script}", timeout=20))
        elapsed = time.monotonic() - started

        # The point: bounded well under the 20s timeout, not pinned to it.
        assert elapsed < 15, f"cap did not fire; hung for {elapsed:.1f}s"
        assert "<persisted-output>" in result
        assert "INCOMPLETE" in result
        assert "Full output saved" not in result
        assert "Use read_file" not in result
        assert "Do not read_file" in result

    def test_cap_kill_is_not_reported_as_a_command_failure(self, tmp_dir, monkeypatch):
        """SIGKILL we sent is our truncation, not the command exiting badly."""
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(Path(tmp_dir) / "projects"))
        script = Path(tmp_dir) / "flood2.py"
        script.write_text(
            "import sys\nbuf = b'x' * (1 << 20)\nw = sys.stdout.buffer.write\n"
            "for _ in range(70):\n    w(buf)\nwhile True:\n    w(buf)\n"
        )
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        result = asyncio.run(tool.execute(f"{shlex.quote(sys.executable)} {script}", timeout=20))
        assert "Exit code: -9" not in result

    def test_truncated_spill_says_it_is_incomplete(self, tmp_dir, monkeypatch):
        """The header must not invite read_file of a killed-at-cap copy."""
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(Path(tmp_dir) / "projects"))
        script = Path(tmp_dir) / "flood3.py"
        script.write_text(
            "import sys\nbuf = b'x' * (1 << 20)\nw = sys.stdout.buffer.write\n"
            "for _ in range(70):\n    w(buf)\nwhile True:\n    w(buf)\n"
        )
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        result = asyncio.run(tool.execute(f"{shlex.quote(sys.executable)} {script}", timeout=20))
        assert "INCOMPLETE" in result
        assert "killed" in result
        assert "Full output saved" not in result
        assert "Use read_file" not in result
        assert "Do not read_file" in result

    def test_normal_oversized_output_is_not_marked_incomplete(self, tmp_dir, monkeypatch):
        """Over max_output_length but under the cap: the copy is complete."""
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(Path(tmp_dir) / "projects"))
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        payload = "H" * 400
        cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(' + repr(payload) + ')')}"
        result = asyncio.run(tool.execute(cmd))
        assert "<persisted-output>" in result
        assert "INCOMPLETE" not in result
        assert "Full output saved" in result
        assert "Use read_file" in result

    def test_stderr_closed_first_still_kills_on_stdout_cap(self, tmp_dir, monkeypatch):
        """Cap on the second-finishing stream must still SIGKILL, not wait it out."""
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(Path(tmp_dir) / "projects"))
        script = Path(tmp_dir) / "flood_stderr_closed.py"
        script.write_text(
            "import sys\n"
            "sys.stderr.close()\n"
            "buf = b'x' * (1 << 20)\n"
            "w = sys.stdout.buffer.write\n"
            "for _ in range(70):\n"
            "    w(buf)\n"
            "while True:\n"
            "    w(buf)\n"
        )
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        started = time.monotonic()
        result = asyncio.run(tool.execute(f"{shlex.quote(sys.executable)} {script}", timeout=20))
        elapsed = time.monotonic() - started
        assert elapsed < 15, f"second-stream cap did not kill; hung for {elapsed:.1f}s"
        assert "INCOMPLETE" in result
        assert "Full output saved" not in result

    @pytest.mark.skipif(os.name == "nt", reason="POSIX killpg")
    def test_kill_process_group_is_not_gated_on_returncode(self, monkeypatch):
        """A reaped shell (`cmd &`) still has a grandchild in the group."""
        from agentica.tools.builtin.execute_tool import _kill_process_group

        killed = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        proc = MagicMock()
        proc.returncode = 0
        proc.pid = 4242
        _kill_process_group(proc)
        assert killed == [(4242, signal.SIGKILL)]


class TestExecuteRedactionReachesTheSpillFile:
    """With redaction on, the on-disk copy must be masked too.

    The model is handed a redacted preview plus a path and told to
    ``read_file`` it. A plaintext copy behind that path makes the toggle a
    promise it does not keep.
    """

    def test_spilled_file_is_redacted_when_the_toggle_is_on(self, tmp_dir, monkeypatch):
        projects = Path(tmp_dir) / "projects"
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(projects))
        monkeypatch.setenv("AGENTICA_REDACT_TOOL_OUTPUTS", "1")

        secret = "sk-" + "A" * 40
        payload = ("noise\n" * 300) + f"token={secret}\n" + ("noise\n" * 300)
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(' + repr(payload) + ')')}"
        result = asyncio.run(tool.execute(cmd))

        path_line = next(
            ln.strip() for ln in result.splitlines()
            if ln.strip().endswith(".txt") and "tool-results" in ln.replace("\\", "/")
        )
        on_disk = Path(path_line).read_text(encoding="utf-8")
        assert secret not in on_disk
        assert "noise" in on_disk, "redaction must not blank ordinary output"

    def test_spilled_file_is_untouched_when_the_toggle_is_off(self, tmp_dir, monkeypatch):
        """Default off: byte-exact round-trips (read_file -> apply_patch) must hold."""
        projects = Path(tmp_dir) / "projects"
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(projects))
        monkeypatch.delenv("AGENTICA_REDACT_TOOL_OUTPUTS", raising=False)

        payload = "marker_" + ("M" * 600)
        tool = BuiltinExecuteTool(work_dir=tmp_dir, max_output_length=200)
        cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(' + repr(payload) + ')')}"
        result = asyncio.run(tool.execute(cmd))

        path_line = next(
            ln.strip() for ln in result.splitlines()
            if ln.strip().endswith(".txt") and "tool-results" in ln.replace("\\", "/")
        )
        assert payload in Path(path_line).read_text(encoding="utf-8")
