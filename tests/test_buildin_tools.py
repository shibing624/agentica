# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for buildin_tools.py built-in tools (async-first).

All tools in buildin_tools.py are async. Tests use asyncio.run() to drive them.
LLM-dependent tools (BuiltinTaskTool) are tested with mocked Agent/Model.
"""

import pytest
import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# bs4 / lxml / markdownify / requests are in agentica core deps (since v1.3.6),
# so no skip needed for builtin web search / fetch url tools.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agentica.tools.buildin_tools import (
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinWebSearchTool,
    BuiltinFetchUrlTool,
    BuiltinTodoTool,
    BuiltinTaskTool,
    get_builtin_tools,
)
from agentica.tools.builtin.web_tools import (
    BuiltinFetchUrlTool as CanonicalBuiltinFetchUrlTool,
    BuiltinWebSearchTool as CanonicalBuiltinWebSearchTool,
)
from agentica.tools.builtin.task_state_tools import (
    BuiltinTodoTool as CanonicalBuiltinTodoTool,
)
from agentica.model.message import Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for file operation tests."""
    d = tempfile.mkdtemp(prefix="test_buildin_tools_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def file_tool(tmp_dir):
    """BuiltinFileTool scoped to a temp directory."""
    return BuiltinFileTool(work_dir=tmp_dir)


@pytest.fixture
def execute_tool(tmp_dir):
    """BuiltinExecuteTool scoped to a temp directory."""
    return BuiltinExecuteTool(work_dir=tmp_dir)


@pytest.fixture
def todo_tool():
    return BuiltinTodoTool()


# ===========================================================================
# BuiltinFileTool tests
# ===========================================================================

class TestBuiltinFileToolLs:
    def test_ls_empty_dir(self, file_tool, tmp_dir):
        result = asyncio.run(file_tool.ls(tmp_dir))
        items = json.loads(result)
        assert isinstance(items, list)
        assert len(items) == 0

    def test_ls_with_files(self, file_tool, tmp_dir):
        # Create a file and a subdirectory
        Path(tmp_dir, "hello.txt").write_text("hello")
        Path(tmp_dir, "subdir").mkdir()

        result = asyncio.run(file_tool.ls(tmp_dir))
        items = json.loads(result)
        names = {i["name"] for i in items}
        assert "hello.txt" in names
        assert "subdir" in names
        # Check types
        types = {i["name"]: i["type"] for i in items}
        assert types["hello.txt"] == "file"
        assert types["subdir"] == "dir"

    def test_ls_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.ls("/nonexistent_dir_abc123"))

    def test_ls_file_not_dir(self, file_tool, tmp_dir):
        f = Path(tmp_dir, "afile.txt")
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            asyncio.run(file_tool.ls(str(f)))


class TestBuiltinFileToolReadFile:
    def test_read_simple_file(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "test.txt")
        p.write_text("line1\nline2\nline3\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_file_with_offset_limit(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "lines.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        # Read lines 5-9 (offset=4, limit=5)
        result = asyncio.run(file_tool.read_file(str(p), offset=4, limit=5))
        assert "line5" in result
        assert "line9" in result
        # line1 should not be present since offset skips it
        assert "line1\t" not in result  # Use tab to avoid matching "line10"

    def test_read_nonexistent_file(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.read_file("/nonexistent/file.txt"))

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


class TestBuiltinFileToolWriteFile:
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


class TestBuiltinFileToolEditFile:
    def test_single_edit(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "edit.txt")
        Path(fp).write_text("hello world")
        result = asyncio.run(file_tool.edit_file(fp, "world", "python"))
        assert "Successfully" in result
        assert Path(fp).read_text() == "hello python"

    def test_multiple_edits_via_separate_calls(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "multi.txt")
        Path(fp).write_text("aaa bbb ccc")
        result1 = asyncio.run(file_tool.edit_file(fp, "aaa", "111"))
        assert "Successfully" in result1
        result2 = asyncio.run(file_tool.edit_file(fp, "ccc", "333"))
        assert "Successfully" in result2
        assert Path(fp).read_text() == "111 bbb 333"

    def test_multi_edit_best_effort_partial_success(self, file_tool, tmp_dir):
        """Default best-effort: successful edits land, failing ones are reported."""
        fp = os.path.join(tmp_dir, "be.txt")
        Path(fp).write_text("alpha bravo charlie")
        # First edit succeeds, second fails (string absent), third succeeds.
        result = asyncio.run(file_tool.multi_edit_file(fp, [
            {"old_string": "alpha", "new_string": "ALPHA"},
            {"old_string": "DOES_NOT_EXIST", "new_string": "x"},
            {"old_string": "charlie", "new_string": "CHARLIE"},
        ]))
        # Successful edits MUST be persisted.
        assert Path(fp).read_text() == "ALPHA bravo CHARLIE"
        # Output names which edits failed so the LLM can retry only those.
        assert "Partially applied" in result
        assert "Edit 2: FAILED" in result
        assert "2/3 succeeded" in result

    def test_multi_edit_best_effort_all_fail_no_write(self, file_tool, tmp_dir):
        """If every edit fails in best-effort mode the file is left untouched."""
        fp = os.path.join(tmp_dir, "be_all.txt")
        Path(fp).write_text("untouched")
        result = asyncio.run(file_tool.multi_edit_file(fp, [
            {"old_string": "nope1", "new_string": "x"},
            {"old_string": "nope2", "new_string": "y"},
        ]))
        assert Path(fp).read_text() == "untouched"
        assert "No edits applied" in result
        assert "2/2 failed" in result

    def test_python_error_hint_null_literal(self):
        """NameError on JSON `null`/`true`/`false` gets a structured hint
        pointing the LLM at the source rather than retrying blindly."""
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "Traceback ...\n  File ...\nNameError: name 'null' is not defined"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "JSON literal" in hint
        assert "None" in hint

    def test_python_error_hint_syntax_error(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "  File 'x.py', line 5\n    if x = 1\nSyntaxError: invalid syntax"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "SyntaxError" in hint

    def test_python_error_hint_module_not_found(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "ModuleNotFoundError: No module named 'foobar_widget'"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "dependency" in hint or "pip install" in hint

    def test_python_error_hint_returns_none_for_normal_output(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        assert _detect_python_error_hint("") is None
        assert _detect_python_error_hint("Hello world") is None
        assert _detect_python_error_hint("AssertionError: 1 != 2") is None  # genuine logic bug

    def test_multi_edit_atomic_mode_rolls_back_on_failure(self, file_tool, tmp_dir):
        """continue_on_error=False keeps the historical atomic semantics."""
        fp = os.path.join(tmp_dir, "atomic.txt")
        Path(fp).write_text("alpha bravo")
        with pytest.raises(ValueError):
            asyncio.run(file_tool.multi_edit_file(fp, [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "MISSING", "new_string": "x"},
            ], continue_on_error=False))
        # Atomic mode: alpha edit MUST be rolled back too.
        assert Path(fp).read_text() == "alpha bravo"

    def test_edit_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "replall.txt")
        Path(fp).write_text("x x x")
        result = asyncio.run(file_tool.edit_file(fp, "x", "y", replace_all=True))
        assert "Successfully" in result
        assert Path(fp).read_text() == "y y y"

    def test_edit_string_not_found(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "nf.txt")
        Path(fp).write_text("hello")
        with pytest.raises(ValueError):
            asyncio.run(file_tool.edit_file(fp, "zzz", "yyy"))
        # File should be unchanged
        assert Path(fp).read_text() == "hello"

    def test_edit_nonexistent_file(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.edit_file("/no/such/file.txt", "a", "b"))

    def test_edit_multiple_matches_no_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "dup.txt")
        Path(fp).write_text("foo bar foo")
        with pytest.raises(ValueError, match="Found 2 occurrences"):
            asyncio.run(file_tool.edit_file(fp, "foo", "baz"))
        # File unchanged
        assert Path(fp).read_text() == "foo bar foo"

    def test_edit_no_side_effect_on_failure(self, file_tool, tmp_dir):
        """A failed edit should not modify the file."""
        fp = os.path.join(tmp_dir, "atomic.txt")
        Path(fp).write_text("aaa bbb")
        with pytest.raises(ValueError):
            asyncio.run(file_tool.edit_file(fp, "zzz", "999"))
        # File should be unchanged
        assert Path(fp).read_text() == "aaa bbb"


class TestBuiltinFileToolGlob:
    def test_glob_py_files(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.py").write_text("")
        Path(tmp_dir, "b.py").write_text("")
        Path(tmp_dir, "c.txt").write_text("")
        result = asyncio.run(file_tool.glob("*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

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

    def test_grep_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.grep("test", "/nonexistent_xyz"))

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
        fn = file_tool.functions["grep"]
        assert fn.manages_own_timeout is True

    def test_grep_fallback_times_out(self, tmp_dir):
        """When rg is unavailable, the pure-Python fallback still hard-times-out
        instead of running up to the outer 120s executor limit."""
        import time as _time
        from agentica.tools import buildin_tools
        tool = BuiltinFileTool(work_dir=tmp_dir)

        # Slow sync fallback worker; the real _run_grep_fallback wraps it with
        # asyncio.wait_for, so the timeout fires well before this returns.
        def slow_worker(*args, **kwargs):
            _time.sleep(0.5)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.buildin_tools.shutil.which", return_value=None), \
             patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 0.1):
            with pytest.raises(TimeoutError, match=r"grep timed out"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_timeout_arg_respected(self, tmp_dir):
        """The LLM-passed `timeout` arg is used as-is, overriding the default
        (no clamping, no upper cap)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        # Patch the default _GREP_TIMEOUT up to 100s; passing timeout=1 must
        # still fire at 1s, proving the caller's value wins and is not clamped
        # back toward the default.
        with patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 100), \
             patch("agentica.tools.buildin_tools.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 1 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir), timeout=1))

    def test_grep_default_timeout_still_bounds(self, tmp_dir):
        """When no timeout arg is passed, the module default still bounds the
        search (a timeout must always be set — bad disk / regex hang)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 1), \
             patch("agentica.tools.buildin_tools.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 1 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir)))


# ===========================================================================
# BuiltinExecuteTool tests
# ===========================================================================

class TestBuiltinExecuteTool:
    def test_execute_simple_command(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo hello"))
        assert "hello" in result

    def test_execute_returns_exit_code_on_failure(self, execute_tool):
        with pytest.raises(RuntimeError, match="exit(ed)? (with )?code 42"):
            asyncio.run(execute_tool.execute("exit 42"))

    def test_execute_captures_stderr(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo error_msg >&2"))
        assert "error_msg" in result

    def test_execute_timeout(self):
        tool = BuiltinExecuteTool(timeout=1)
        with pytest.raises(TimeoutError, match="timed out"):
            asyncio.run(tool.execute("sleep 30"))

    def test_execute_python_code(self, execute_tool):
        result = asyncio.run(execute_tool.execute("python3 -c 'print(2+3)'"))
        assert "5" in result

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


# ===========================================================================
# BuiltinWebSearchTool tests
# ===========================================================================


def test_web_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinWebSearchTool is CanonicalBuiltinWebSearchTool
    assert BuiltinFetchUrlTool is CanonicalBuiltinFetchUrlTool


def test_task_state_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinTodoTool is CanonicalBuiltinTodoTool


def test_get_builtin_tools_still_returns_expected_tool_types():
    tools = get_builtin_tools(work_dir="/tmp")
    tool_names = {tool.__class__.__name__ for tool in tools}
    assert "BuiltinFileTool" in tool_names
    assert "BuiltinExecuteTool" in tool_names
    assert "BuiltinWebSearchTool" in tool_names
    assert "BuiltinFetchUrlTool" in tool_names
    assert "BuiltinTodoTool" in tool_names
    assert "BuiltinTaskTool" in tool_names

class TestBuiltinWebSearchTool:
    def test_web_search_delegates_to_baidu(self):
        """Verify web_search calls BaiduSearchTool.baidu_search under the hood."""
        tool = BuiltinWebSearchTool()

        mock_result = json.dumps([{"title": "test", "url": "http://example.com", "content": "result"}])
        tool._search.baidu_search = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search("test query"))
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "test"
        tool._search.baidu_search.assert_awaited_once_with("test query", max_results=5)

    def test_web_search_multiple_queries(self):
        tool = BuiltinWebSearchTool()
        mock_result = json.dumps({"q1": [], "q2": []})
        tool._search.baidu_search = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search(["q1", "q2"], max_results=3))
        tool._search.baidu_search.assert_awaited_once_with(["q1", "q2"], max_results=3)

    def test_web_search_error_handling(self):
        tool = BuiltinWebSearchTool()
        tool._search.baidu_search = AsyncMock(side_effect=Exception("network error"))

        # After方案A: search failures propagate as exceptions instead of Error strings.
        # Runner/FunctionCall.invoke captures them into function_call.error.
        with pytest.raises(Exception, match="network error"):
            asyncio.run(tool.web_search("fail"))


# ===========================================================================
# BuiltinFetchUrlTool tests
# ===========================================================================

class TestBuiltinFetchUrlTool:
    def test_fetch_url_delegates_to_crawler(self):
        """Verify fetch_url calls UrlCrawlerTool.url_crawl under the hood."""
        tool = BuiltinFetchUrlTool()

        mock_result = json.dumps({"url": "http://example.com", "content": "page content", "save_path": "/tmp/x"})
        tool._crawler.url_crawl = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.fetch_url("http://example.com"))
        parsed = json.loads(result)
        assert parsed["url"] == "http://example.com"
        assert parsed["content"] == "page content"
        tool._crawler.url_crawl.assert_awaited_once_with("http://example.com")


# ===========================================================================
# BuiltinTodoTool tests
# ===========================================================================

class TestBuiltinTodoTool:
    def test_write_todos_basic(self, todo_tool):
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
        ])
        parsed = json.loads(result)
        assert "2 items" in parsed["message"]
        assert len(parsed["todos"]) == 2
        assert parsed["all_completed"] is False
        assert parsed["verification_nudge"] is False

    def test_write_todos_invalid_status(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"content": "Bad", "status": "unknown"}])

    def test_write_todos_missing_content(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"status": "pending"}])

    def test_write_todos_none(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos(None)

    def test_write_todos_empty_list(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([])

    def test_write_todos_overwrites(self, todo_tool):
        """Writing new todos replaces old ones entirely."""
        todo_tool.write_todos([{"content": "Old", "status": "pending"}])
        todo_tool.write_todos([{"content": "New1", "status": "pending"}, {"content": "New2", "status": "pending"}])
        assert len(todo_tool.todos) == 2
        contents = [t["content"] for t in todo_tool.todos]
        assert "Old" not in contents
        assert "New1" in contents

    def test_set_agent_stores_on_agent(self):
        """When set_agent is called, todos are stored on agent.todos."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = []
        tool.set_agent(mock_agent)

        tool.write_todos([
            {"content": "Task X", "status": "pending"},
            {"content": "Task Y", "status": "in_progress"},
        ])
        # Todos should be stored on mock_agent.todos
        assert len(mock_agent.todos) == 2
        assert mock_agent.todos[0]["content"] == "Task X"
        assert mock_agent.todos[1]["content"] == "Task Y"

    def test_standalone_mode_uses_local_todos(self):
        """Without set_agent, todos are stored locally on the tool."""
        tool = BuiltinTodoTool()
        tool.write_todos([{"content": "Local task", "status": "pending"}])
        assert len(tool.todos) == 1
        assert tool.todos[0]["content"] == "Local task"
        # _agent should be None
        assert tool._agent is None

    def test_todos_property_reads_from_agent(self):
        """The todos property should read from agent when agent is set."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = [{"id": "1", "content": "Agent task", "status": "completed"}]
        tool.set_agent(mock_agent)
        assert tool.todos == mock_agent.todos

    # ---- Auto-clear tests (mirrors CC allDone logic) ----

    def test_auto_clear_when_all_completed(self, todo_tool):
        """All-completed todos should auto-clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["all_completed"] is True
        # Internal list should be cleared
        assert len(todo_tool.todos) == 0
        # But the returned todos should still show what was submitted
        assert len(parsed["todos"]) == 2

    def test_no_auto_clear_when_not_all_completed(self, todo_tool):
        """Partial completion should NOT clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "in_progress"},
        ])
        parsed = json.loads(result)
        assert parsed["all_completed"] is False
        assert len(todo_tool.todos) == 2

    # ---- Verification nudge tests (mirrors CC structural nudge) ----

    def test_verification_nudge_3plus_all_completed_no_verify(self, todo_tool):
        """3+ all-completed tasks with no verification keyword -> nudge fires."""
        result = todo_tool.write_todos([
            {"content": "Implement feature A", "status": "completed"},
            {"content": "Implement feature B", "status": "completed"},
            {"content": "Implement feature C", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is True
        assert "NOTE:" in parsed["message"]

    def test_no_nudge_when_less_than_3_tasks(self, todo_tool):
        """< 3 tasks all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is False

    def test_no_nudge_when_not_all_completed(self, todo_tool):
        """3+ tasks but not all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
            {"content": "Task C", "status": "in_progress"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is False

    def test_no_nudge_when_verification_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'verify' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Verify implementation", "status": "completed"},
            {"content": "Deploy to staging", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is False

    def test_no_nudge_when_test_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'test' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Write unit tests", "status": "completed"},
            {"content": "Update docs", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is False

    def test_no_nudge_when_lint_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'lint' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Refactor module", "status": "completed"},
            {"content": "Run linting", "status": "completed"},
            {"content": "Deploy", "status": "completed"},
        ])
        parsed = json.loads(result)
        assert parsed["verification_nudge"] is False

    # ---- _needs_verification_nudge static method tests ----

    def test_needs_verification_nudge_static(self):
        """Direct test of the static nudge detection method."""
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is True

        # Has 'check' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Check results", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'review' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Code review", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'validate' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Validate output", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

    # ---- Tool result message format tests ----

    def test_tool_result_contains_guidance_text(self, todo_tool):
        """Tool result should contain guidance text for the LLM."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
        ])
        parsed = json.loads(result)
        assert "Ensure that you continue" in parsed["message"]
        assert "proceed with the current tasks" in parsed["message"]


# ===========================================================================
# BuiltinTaskTool tests (requires mocking LLM / Agent)
# ===========================================================================

class TestBuiltinTaskTool:
    """``BuiltinTaskTool`` is a thin LLM-facing adapter around
    ``SubagentRegistry.spawn``. Tests focus on the adapter contract; the
    runtime behavior of ``spawn`` itself is covered by ``test_subagent.py``.
    """

    def test_task_without_parent_returns_error(self):
        """Unbound tool (no parent agent) cannot spawn anything."""
        tool = BuiltinTaskTool()
        result = asyncio.run(tool.task("do something"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "not bound" in parsed["error"]

    def test_task_forwards_to_spawn_and_serializes_completed(self):
        """Adapter calls ``SubagentRegistry().spawn`` and JSON-serializes the result."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        spawn_result = {
            "status": "completed",
            "agent_type": "code",
            "subagent_name": "Code Agent",
            "content": "answer is 42",
            "tool_calls_summary": [{"name": "read_file", "info": "x.py"}],
            "tool_count": 1,
            "execution_time": 0.123,
            "run_id": "abc",
        }

        async def fake_spawn(self, **kwargs):
            assert kwargs["task"] == "compute 6 * 7"
            assert kwargs["agent_type"] == "code"
            return spawn_result

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("compute 6 * 7", subagent_type="code"))

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["subagent_type"] == "code"
        assert parsed["subagent_name"] == "Code Agent"
        assert parsed["result"] == "answer is 42"
        assert parsed["tool_count"] == 1
        assert parsed["execution_time"] == 0.123

    def test_task_serializes_error_result(self):
        """Adapter surfaces spawn errors through the LLM-facing JSON envelope."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        async def fake_spawn(self, **kwargs):
            return {
                "status": "error",
                "error": "Subagent timed out after 5 seconds",
                "agent_type": "code",
                "content": "",
            }

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("slow"))

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "timed out" in parsed["error"]
        assert parsed["subagent_type"] == "code"

    def test_format_tool_brief_read_file(self):
        brief = BuiltinTaskTool._format_tool_brief("read_file", {"file_path": "/a/b/c.py"})
        assert "c.py" in brief

    def test_format_tool_brief_grep(self):
        brief = BuiltinTaskTool._format_tool_brief("grep", {"pattern": "hello"}, "found 3 matches")
        assert "hello" in brief

    def test_format_tool_brief_execute(self):
        brief = BuiltinTaskTool._format_tool_brief("execute", {"command": "ls -la /tmp"})
        assert "ls -la" in brief

    def test_format_tool_brief_default(self):
        brief = BuiltinTaskTool._format_tool_brief("unknown_tool", {"key": "value"})
        assert "key=" in brief

    def test_set_parent_agent(self):
        tool = BuiltinTaskTool()
        mock_agent = MagicMock()
        tool.set_parent_agent(mock_agent)
        assert tool._parent_agent is mock_agent

    def test_task_declares_own_timeout_management(self):
        tool = BuiltinTaskTool()
        assert tool.functions["task"].manages_own_timeout is True
        assert tool.functions["task"].interrupt_behavior == "block"

    def test_task_passes_model_override_to_spawn(self):
        """When ``model_override`` is set, the adapter forwards it to spawn."""
        custom_model = MagicMock()
        tool = BuiltinTaskTool(model_override=custom_model)
        tool.set_parent_agent(MagicMock())

        captured: Dict[str, Any] = {}

        async def fake_spawn(self, **kwargs):
            captured.update(kwargs)
            return {"status": "completed", "agent_type": "code", "content": "ok",
                    "tool_calls_summary": [], "tool_count": 0, "execution_time": 0}

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            asyncio.run(tool.task("test", subagent_type="code"))

        assert captured["model_override"] is custom_model


# ===========================================================================
# Agent auto-wire tests (Agent.__init__ wires TodoTool / TaskTool)
# ===========================================================================

class TestAgentAutoWire:
    """Agent.__init__ clones stateful tools per-agent (so the user's original
    instance is never overwritten when the same logical tool is reused across
    multiple agents) and wires the per-agent clone to ``self``."""

    def test_agent_wires_todo_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTodoTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        todo_tool = BuiltinTodoTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[todo_tool],
        )
        # User's original tool is left untouched (isolation contract)
        assert todo_tool._agent is None
        # Agent owns its own clone, wired to itself
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        assert wired is not todo_tool
        assert wired._agent is agent

    def test_agent_wires_task_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTaskTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        task_tool = BuiltinTaskTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[task_tool],
        )
        assert task_tool._parent_agent is None
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTaskTool))
        assert wired is not task_tool
        assert wired._parent_agent is agent

    def test_todo_tool_stores_on_agent(self):
        """After wiring, write_todos on the agent's clone stores todos on agent.todos."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[BuiltinTodoTool()],
        )
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        wired.write_todos([
            {"content": "Test task", "status": "pending"},
        ])
        assert len(agent.todos) == 1
        assert agent.todos[0]["content"] == "Test task"


# ===========================================================================
# OpenAI stream_finish_reason capture tests
# ===========================================================================

class TestOpenAIStreamFinishReason:
    """Test that OpenAIChat.response_stream correctly captures finish_reason."""

    def _make_openai_chat(self):
        from agentica.model.openai import OpenAIChat
        return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

    def test_finish_reason_captured_from_last_chunk(self):
        """stream_finish_reason should be captured from the chunk where finish_reason is not None."""
        model = self._make_openai_chat()

        # Build mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.audio = None
        chunk1.choices[0].delta.tool_calls = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "stop"
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.audio = None
        chunk2.choices[0].delta.tool_calls = None
        chunk2.usage = None

        async def mock_invoke_stream(messages):
            yield chunk1
            yield chunk2

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]
        collected = []

        async def run():
            async for resp in model.response_stream(messages=messages):
                collected.append(resp)

        asyncio.run(run())
        assert model.last_finish_reason == "stop"

    def test_finish_reason_length_captured(self):
        """When output is truncated, finish_reason should be 'length'."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = "length"
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "partial output..."
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.audio = None
        chunk.choices[0].delta.tool_calls = None
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason == "length"

    def test_finish_reason_none_when_no_choices(self):
        """When stream has no choices, finish_reason should remain None."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason is None


# ===========================================================================
# Guard: BuiltinFileTool functions MUST reach Model.get_tools_for_api()
# ===========================================================================

class TestFileToolRegistrationGuard:
    """End-to-end guard: every BuiltinFileTool function must be visible in
    the final tool schema sent to the LLM.

    This test exists because a past bug placed self.register() calls outside
    __init__(), causing read_file / ls / glob to silently disappear from the
    model's tool list while execute remained available — the model then fell
    back to shell commands.
    """

    EXPECTED_FUNCTIONS = {"ls", "read_file", "write_file", "edit_file",
                          "multi_edit_file", "glob", "grep"}

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


# ===========================================================================
# AskUserQuestionTool tests
# ===========================================================================

class TestAskUserQuestionTool:
    def test_user_input_manages_own_timeout(self):
        """user_input/confirm must wait indefinitely for the user (CC/Cursor
        semantics), not be auto-passed by the outer ~120s tool-executor timeout."""
        from agentica.tools.user_input_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "ok")
        assert tool.functions["user_input"].manages_own_timeout is True
        assert tool.functions["confirm"].manages_own_timeout is True

    def test_user_input_uses_callback(self):
        from agentica.tools.user_input_tool import AskUserQuestionTool

        captured = {}

        def cb(prompt, options=None):
            captured["prompt"] = prompt
            captured["options"] = options
            return "my answer"

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(tool.user_input(prompt="What now?", mode="text"))
        assert result["response"] == "my answer"
        assert "What now?" in captured["prompt"]
