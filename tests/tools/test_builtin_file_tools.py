# -*- coding: utf-8 -*-
"""Tests for BuiltinFileTool (read / write / glob / grep)."""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

from agentica.tools.builtin import BuiltinFileTool


class BlockingSubprocess:
    """Minimal subprocess double whose first communicate call blocks."""

    def __init__(self):
        self.started = asyncio.Event()
        self.returncode = None
        self._transport = None

    async def communicate(self):
        self.started.set()
        await asyncio.Future()



class TestBuiltinFileToolReadFile:
    def test_path_tool_descriptions_require_grounded_paths(self):
        for fn in (
            BuiltinFileTool.read_file,
            BuiltinFileTool.grep,
            BuiltinFileTool.glob,
            BuiltinFileTool.apply_patch,
        ):
            description = " ".join((fn.__doc__ or "").split()).lower()
            assert "ground" in description
            assert "speculative absolute path" in description

    def test_empty_file_returns_reminder(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "empty.txt")
        Path(fp).touch()
        result = asyncio.run(file_tool.read_file(fp))
        assert "<system-reminder>" in result
        assert "0 bytes" in result

    def test_read_simple_file(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "test.txt")
        p.write_text("line1\nline2\nline3\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "[File metadata:" not in result

    def test_read_file_with_offset_limit(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "lines.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        # Read lines 5-9 (offset=4, limit=5)
        result = asyncio.run(file_tool.read_file(str(p), offset=4, limit=5))
        assert "line5" in result
        assert "line9" in result
        # line1 should not be present since offset skips it
        assert "line1\t" not in result  # Use tab to avoid matching "line10"

    def test_read_nonexistent_file_returns_path_state(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        candidate = existing / "usage.py"
        candidate.write_text("x = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.read_file(str(existing / "usage_tracking.py")))

        msg = str(exc.value)
        assert "File not found:" in msg
        assert "Resolved path:" in msg
        assert f"Nearest existing parent: {existing}" in msg
        assert "Candidate paths:" not in msg
        assert str(candidate) not in msg
        assert "Next step:" not in msg

    def test_read_long_lines_truncated(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "long.txt")
        p.write_text("A" * 5000 + "\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Default max_line_length is 2000, line should be truncated with "..."
        assert "..." in result

    def test_read_file_shows_line_numbers(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "numbered.txt")
        p.write_text("alpha\nbeta\ngamma\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Should contain line numbers (cat -n format)
        assert "\t" in result  # tab separator between line number and content


class TestMissingPathErrors:
    """Missing-path errors expose path state without guessing candidates."""

    def test_grep_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        (existing / "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.grep("class .*Config", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Path not found:" in msg
        assert "Resolved path:" in msg
        assert "Nearest existing parent:" in msg
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg
        assert "Next step:" not in msg

    def test_glob_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.glob("*.py", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Directory not found:" in msg
        assert "Did you mean:" not in msg
        assert "run glob '**/config.py'" not in msg

    def test_apply_patch_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.apply_patch(
                "*** Begin Patch\n"
                "*** Update File: missing-dir/config.py\n"
                "@@\n"
                "-a\n"
                "+b\n"
                "*** End Patch"
            ))

        msg = str(exc.value)
        assert "File not found:" in msg or "not found" in msg.lower()
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg


class TestBuiltinFileToolReadCorrectness:
    """read_file always reflects current disk content (no memoization)."""

    def test_second_read_returns_same_content(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "cached.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        r1 = asyncio.run(file_tool.read_file(str(p)))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert r1 == r2  # no cache: each call re-reads, content identical

    def test_external_size_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_size.txt")
        p.write_text("v1")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("v2 with more bytes")
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "v2 with more bytes" in r2

    def test_external_mtime_only_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_mtime.txt")
        p.write_text("same-size-a")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("same-size-b")  # identical size, different content
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "same-size-b" in r2

    def test_apply_patch_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "edit.txt")
        p.write_text("hello world\n")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.apply_patch(
            "*** Begin Patch\n"
            "*** Update File: edit.txt\n"
            "@@\n"
            "-hello world\n"
            "+hello agentica\n"
            "*** End Patch"
        ))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "hello agentica" in r
        assert "hello world" not in r

    def test_write_file_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "w.txt")
        p.write_text("old content")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.write_file(str(p), "new content"))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "new content" in r
        assert "old content" not in r

    def test_offset_and_limit_select_distinct_windows(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "pages.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        r1 = asyncio.run(file_tool.read_file(str(p), offset=0, limit=5))
        r2 = asyncio.run(file_tool.read_file(str(p), offset=5, limit=5))
        assert r1 != r2
        assert "line6" in r2 and "line6" not in r1


class TestBuiltinFileToolWriteFile:
    def test_write_result_carries_actual_file_change(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "existing.txt")
        Path(fp).write_text("before\n")

        result = asyncio.run(file_tool.write_file(fp, "after\n"))

        assert result.display_meta == {
            "files": [{
                "path": str(Path(fp).resolve()),
                "action": "update",
                "before": "before\n",
                "after": "after\n",
            }]
        }

    def test_write_new_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "new.txt")
        result = asyncio.run(file_tool.write_file(fp, "hello world"))
        assert "Created" in result
        assert Path(fp).read_text() == "hello world"

    def test_write_overwrite_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "existing.txt")
        Path(fp).write_text("old")
        result = asyncio.run(file_tool.write_file(fp, "new content"))
        assert "Updated" in result
        assert Path(fp).read_text() == "new content"

    def test_write_creates_parent_dirs(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "a", "b", "c.txt")
        result = asyncio.run(file_tool.write_file(fp, "nested"))
        assert "Created" in result
        assert Path(fp).read_text() == "nested"

    def test_write_returns_absolute_path(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "abs.txt")
        result = asyncio.run(file_tool.write_file(fp, "test"))
        # Result should contain the absolute path
        assert tmp_dir in result


class TestBuiltinFileToolGlob:
    def test_glob_py_files(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.py").write_text("")
        Path(tmp_dir, "b.py").write_text("")
        Path(tmp_dir, "c.txt").write_text("")
        result = asyncio.run(file_tool.glob("*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_glob_manages_own_timeout(self, file_tool):
        from agentica.tools.builtin.file_tool import _GLOB_TIMEOUT
        fn = file_tool.functions["glob"]
        assert fn.manages_own_timeout is True
        assert _GLOB_TIMEOUT == 20

    def test_glob_default_timeout_still_bounds(self, tmp_dir):
        """When the walk hangs, the hardcoded default still bounds it."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_glob(self, pattern):
            _time.sleep(0.2)
            return iter(())

        with patch("agentica.tools.builtin.file_tool._GLOB_TIMEOUT", 0.05), \
             patch.object(Path, "glob", slow_glob):
            with pytest.raises(TimeoutError, match=r"Narrow `path`"):
                asyncio.run(tool.glob("**/*", str(tmp_dir)))

    def test_glob_recursive(self, file_tool, tmp_dir):
        sub = Path(tmp_dir, "sub")
        sub.mkdir()
        Path(sub, "deep.py").write_text("")
        Path(tmp_dir, "top.py").write_text("")
        result = asyncio.run(file_tool.glob("**/*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2

    def test_glob_no_matches(self, file_tool, tmp_dir):
        result = asyncio.run(file_tool.glob("*.xyz", tmp_dir))
        files = json.loads(result)
        assert files == []

    def test_glob_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.glob("*", "/nonexistent_dir_xyz"))


class TestBuiltinFileToolGrep:
    def test_grep_schema_omits_timeout_and_multiline(self):
        import inspect
        from agentica.tools.builtin.execute_tool import BuiltinExecuteTool

        grep_params = inspect.signature(BuiltinFileTool.grep).parameters
        glob_params = inspect.signature(BuiltinFileTool.glob).parameters
        wait_params = inspect.signature(BuiltinExecuteTool.wait).parameters
        execute_params = inspect.signature(BuiltinExecuteTool.execute).parameters
        assert "timeout" not in grep_params
        assert "multiline" not in grep_params
        assert "timeout" not in glob_params
        assert "timeout" in wait_params
        assert "timeout" in execute_params

    def test_grep_docstring_states_context_precedence(self):
        doc = BuiltinFileTool.grep.__doc__ or ""
        assert "ignored when" in doc
        assert "before_context" in doc
        assert "after_context" in doc

    def test_grep_context_lines_wins_over_before_after(self, file_tool, tmp_dir):
        import shutil
        if shutil.which("rg") is None:
            pytest.skip("rg not installed")
        Path(tmp_dir, "f.py").write_text("LINE_A\nLINE_B\nMATCH\nLINE_C\nLINE_D\n")
        result = asyncio.run(
            file_tool.grep(
                "MATCH",
                str(tmp_dir),
                context_lines=1,
                before_context=10,
                after_context=10,
            )
        )
        assert "LINE_B" in result
        assert "LINE_C" in result
        assert "LINE_A" not in result
        assert "LINE_D" not in result

    def test_grep_fallback_context_lines(self, file_tool, tmp_dir):
        Path(tmp_dir, "f.py").write_text("LINE_A\nLINE_B\nMATCH\nLINE_C\nLINE_D\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep("MATCH", str(tmp_dir), context_lines=1))
        assert "LINE_B" in result
        assert "LINE_C" in result
        assert "LINE_A" not in result
        assert "LINE_D" not in result
        assert "MATCH" in result

    def test_grep_fallback_asymmetric_before_after(self, file_tool, tmp_dir):
        Path(tmp_dir, "f.py").write_text("LINE_A\nLINE_B\nMATCH\nLINE_C\nLINE_D\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            before_only = asyncio.run(
                file_tool.grep("MATCH", str(tmp_dir), before_context=1)
            )
            after_only = asyncio.run(
                file_tool.grep("MATCH", str(tmp_dir), after_context=1)
            )
        assert "LINE_B" in before_only and "LINE_C" not in before_only
        assert "LINE_C" in after_only and "LINE_B" not in after_only

    def test_grep_fallback_context_lines_wins_over_before_after(self, file_tool, tmp_dir):
        Path(tmp_dir, "f.py").write_text("LINE_A\nLINE_B\nMATCH\nLINE_C\nLINE_D\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(
                file_tool.grep(
                    "MATCH",
                    str(tmp_dir),
                    context_lines=1,
                    before_context=10,
                    after_context=10,
                )
            )
        assert "LINE_B" in result
        assert "LINE_C" in result
        assert "LINE_A" not in result
        assert "LINE_D" not in result

    def test_grep_fallback_separated_matches_get_group_separator(self, file_tool, tmp_dir):
        Path(tmp_dir, "f.py").write_text("M1\nX\nY\nZ\nM2\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            grouped = asyncio.run(file_tool.grep("M[12]", str(tmp_dir), context_lines=1))
        assert "M1" in grouped and "M2" in grouped
        assert "X" in grouped and "Z" in grouped
        assert "Y" not in grouped
        assert "--" in grouped

    def test_grep_fallback_overlapping_context_merges(self, file_tool, tmp_dir):
        Path(tmp_dir, "f.py").write_text("A\nM1\nB\nM2\nC\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep("M[12]", str(tmp_dir), context_lines=1))
        assert "A" in result and "B" in result and "C" in result
        assert result.count("B") == 1
        assert "--" not in result

    def test_grep_default_returns_content_with_line_numbers(self, file_tool, tmp_dir):
        # Default output_mode is "content" — must return matching lines with
        # line numbers, not just a path list. A bare path-only response was the
        # root cause of dumb-model retry loops where the model couldn't tell
        # whether it had actually seen the code yet.
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "b.txt").write_text("goodbye world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir))
        assert "a.txt" in result
        assert "hello world" in result, "default mode must include matched line content"
        assert "c.txt" not in result

    def test_grep_files_with_matches_mode_returns_paths_only(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(
            file_tool.grep("hello", tmp_dir, output_mode="files_with_matches")
        )
        assert "a.txt" in result
        assert "hello world" not in result
        assert "c.txt" not in result

    def test_grep_content_mode(self, file_tool, tmp_dir):
        Path(tmp_dir, "code.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        result = asyncio.run(file_tool.grep("def", tmp_dir, output_mode="content"))
        assert "def foo" in result
        assert "def bar" in result

    def test_grep_no_matches(self, file_tool, tmp_dir):
        Path(tmp_dir, "empty.txt").write_text("nothing\n")
        result = asyncio.run(file_tool.grep("zzzzz", tmp_dir))
        assert "No matches" in result

    def test_grep_nonexistent_path(self, file_tool):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            asyncio.run(file_tool.grep("test", "/nonexistent_xyz"))

    def test_grep_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("gate_passed = True\n")
        result = asyncio.run(file_tool.grep("gate_passed", str(fp), include="*.py"))
        assert "gate_passed = True" in result

    def test_grep_fallback_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("commit_pass = True\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep("commit_pass", str(fp), include="*.py"))
        assert "commit_pass = True" in result

    def test_grep_case_insensitive(self, file_tool, tmp_dir):
        Path(tmp_dir, "mixed.txt").write_text("Hello WORLD\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir, case_insensitive=True, output_mode="content"))
        assert "Hello" in result

    def test_grep_fixed_strings(self, file_tool, tmp_dir):
        Path(tmp_dir, "regex.txt").write_text("price is $10.00\n")
        # $ and . are special in regex; fixed_strings should match literally
        result = asyncio.run(file_tool.grep("$10.00", tmp_dir, fixed_strings=True, output_mode="content"))
        assert "$10.00" in result

    def test_grep_manages_own_timeout(self, file_tool):
        """grep must self-limit so the outer 120s executor wrapper is skipped."""
        from agentica.tools.builtin.file_tool import _GREP_TIMEOUT
        fn = file_tool.functions["grep"]
        assert fn.manages_own_timeout is True
        assert _GREP_TIMEOUT == 20

    def test_grep_fallback_times_out(self, tmp_dir):
        """When rg is unavailable, the pure-Python fallback still hard-times-out
        instead of running up to the outer 120s executor limit."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        # Slow sync fallback worker; the real _run_grep_fallback wraps it with
        # asyncio.wait_for, so the timeout fires well before this returns.
        def slow_worker(*args, **kwargs):
            _time.sleep(0.2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None), \
             patch("agentica.tools.builtin.file_tool._GREP_TIMEOUT", 0.05):
            with pytest.raises(TimeoutError, match=r"Narrow `path`"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_default_timeout_still_bounds(self, tmp_dir):
        """When no timeout arg is passed, the module default still bounds the
        search (a timeout must always be set — bad disk / regex hang)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(0.2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.builtin.file_tool._GREP_TIMEOUT", 0.05), \
             patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 0.05 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_cancellation_cleans_up_subprocess(self, tmp_dir):
        async def cancel_running_grep():
            process = BlockingSubprocess()
            cleanup = AsyncMock()
            with patch(
                "agentica.tools.builtin.file_tool.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ), patch(
                "agentica.tools.builtin.file_tool.terminate_subprocess",
                cleanup,
            ), patch(
                "agentica.tools.builtin.file_tool.shutil.which",
                return_value="/usr/bin/rg",
            ):
                tool = BuiltinFileTool(work_dir=tmp_dir)
                task = asyncio.create_task(tool.grep("needle", str(tmp_dir)))
                await process.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            cleanup.assert_awaited_once_with(process)

        asyncio.run(cancel_running_grep())



class TestFileToolRegistrationGuard:
    """End-to-end guard: every BuiltinFileTool function must be visible in
    the final tool schema sent to the LLM.

    This test exists because a past bug placed self.register() calls outside
    __init__(), causing read_file / glob to silently disappear from the
    model's tool list while execute remained available — the model then fell
    back to shell commands.
    """

    EXPECTED_FUNCTIONS = {"read_file", "write_file",
                          "apply_patch", "glob", "grep"}

    def test_file_tool_functions_in_tool_dict(self):
        """Tool.functions dict must contain all file operations after init."""
        tool = BuiltinFileTool(work_dir="/tmp")
        registered = set(tool.functions.keys())
        missing = self.EXPECTED_FUNCTIONS - registered
        assert not missing, f"Functions missing from Tool.functions: {missing}"

    def test_file_tool_functions_reach_model_api_schema(self):
        """After Agent.update_model(), every file function must appear in
        Model.get_tools_for_api() — the payload actually sent to the LLM."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        file_tool = BuiltinFileTool(work_dir="/tmp")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[file_tool],
        )
        agent.update_model()

        api_tools = agent.model.get_tools_for_api()
        api_names = {t["function"]["name"] for t in api_tools}
        missing = self.EXPECTED_FUNCTIONS - api_names
        assert not missing, (
            f"Functions missing from Model.get_tools_for_api(): {missing}. "
            f"Likely cause: self.register() not called in __init__."
        )
        by_name = {
            t["function"]["name"]: t["function"]["parameters"]["properties"]
            for t in api_tools
        }
        assert "timeout" not in by_name["grep"]
        assert "multiline" not in by_name["grep"]
        assert "timeout" not in by_name["glob"]

    def test_auto_mode_schema_includes_request_path_access(self):
        from agentica.agent import Agent
        from agentica.agent.config import ToolConfig
        from agentica.model.openai import OpenAIChat

        # The tool is built with the default tier; the agent's tier is what
        # decides, so an "auto" agent must never end up sandboxed with no way
        # to ask for an exception.
        file_tool = BuiltinFileTool(work_dir="/tmp")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[file_tool],
            tool_config=ToolConfig(permission_mode="auto"),
        )
        agent.update_model()
        api_names = {t["function"]["name"] for t in agent.model.get_tools_for_api()}
        assert "request_path_access" in api_names
        assert "ls" not in api_names
        assert "edit_file" not in api_names

        agent.set_permission_mode("allow-all")
        agent.update_model()
        api_names = {t["function"]["name"] for t in agent.model.get_tools_for_api()}
        assert "request_path_access" not in api_names

