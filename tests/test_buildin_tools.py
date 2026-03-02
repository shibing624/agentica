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
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agentica.tools.buildin_tools import (
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinWebSearchTool,
    BuiltinFetchUrlTool,
    BuiltinTodoTool,
    BuiltinTaskTool,
    BuiltinMemoryTool,
    get_builtin_tools,
)


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
        result = asyncio.run(file_tool.ls("/nonexistent_dir_abc123"))
        assert "Error" in result

    def test_ls_file_not_dir(self, file_tool, tmp_dir):
        f = Path(tmp_dir, "afile.txt")
        f.write_text("x")
        result = asyncio.run(file_tool.ls(str(f)))
        assert "Error" in result


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
        result = asyncio.run(file_tool.read_file("/nonexistent/file.txt"))
        assert "Error" in result

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

    def test_edit_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "replall.txt")
        Path(fp).write_text("x x x")
        result = asyncio.run(file_tool.edit_file(fp, "x", "y", replace_all=True))
        assert "Successfully" in result
        assert Path(fp).read_text() == "y y y"

    def test_edit_string_not_found(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "nf.txt")
        Path(fp).write_text("hello")
        result = asyncio.run(file_tool.edit_file(fp, "zzz", "yyy"))
        assert "Error" in result
        # File should be unchanged
        assert Path(fp).read_text() == "hello"

    def test_edit_nonexistent_file(self, file_tool):
        result = asyncio.run(file_tool.edit_file("/no/such/file.txt", "a", "b"))
        assert "Error" in result

    def test_edit_multiple_matches_no_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "dup.txt")
        Path(fp).write_text("foo bar foo")
        result = asyncio.run(file_tool.edit_file(fp, "foo", "baz"))
        # Should fail because there are multiple matches and replace_all is not set
        assert "Error" in result or "Found 2 occurrences" in result
        # File unchanged
        assert Path(fp).read_text() == "foo bar foo"

    def test_edit_no_side_effect_on_failure(self, file_tool, tmp_dir):
        """A failed edit should not modify the file."""
        fp = os.path.join(tmp_dir, "atomic.txt")
        Path(fp).write_text("aaa bbb")
        result = asyncio.run(file_tool.edit_file(fp, "zzz", "999"))
        assert "Error" in result
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
        result = asyncio.run(file_tool.glob("*", "/nonexistent_dir_xyz"))
        assert "Error" in result


class TestBuiltinFileToolGrep:
    def test_grep_files_with_matches(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "b.txt").write_text("goodbye world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir))
        assert "a.txt" in result
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
        result = asyncio.run(file_tool.grep("test", "/nonexistent_xyz"))
        assert "Error" in result

    def test_grep_case_insensitive(self, file_tool, tmp_dir):
        Path(tmp_dir, "mixed.txt").write_text("Hello WORLD\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir, case_insensitive=True, output_mode="content"))
        assert "Hello" in result

    def test_grep_fixed_strings(self, file_tool, tmp_dir):
        Path(tmp_dir, "regex.txt").write_text("price is $10.00\n")
        # $ and . are special in regex; fixed_strings should match literally
        result = asyncio.run(file_tool.grep("$10.00", tmp_dir, fixed_strings=True, output_mode="content"))
        assert "$10.00" in result


# ===========================================================================
# BuiltinExecuteTool tests
# ===========================================================================

class TestBuiltinExecuteTool:
    def test_execute_simple_command(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo hello"))
        assert "hello" in result

    def test_execute_returns_exit_code_on_failure(self, execute_tool):
        result = asyncio.run(execute_tool.execute("exit 42"))
        assert "Exit code: 42" in result

    def test_execute_captures_stderr(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo error_msg >&2"))
        assert "error_msg" in result

    def test_execute_timeout(self):
        tool = BuiltinExecuteTool(timeout=1)
        result = asyncio.run(tool.execute("sleep 30"))
        assert "timed out" in result

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

        result = asyncio.run(tool.web_search("fail"))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "network error" in parsed["error"]


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

    def test_read_todos_empty(self, todo_tool):
        result = todo_tool.read_todos()
        parsed = json.loads(result)
        assert parsed["summary"]["total"] == 0

    def test_read_todos_after_write(self, todo_tool):
        todo_tool.write_todos([
            {"content": "Do X", "status": "completed"},
            {"content": "Do Y", "status": "pending"},
            {"content": "Do Z", "status": "in_progress"},
        ])
        result = todo_tool.read_todos()
        parsed = json.loads(result)
        assert parsed["summary"]["total"] == 3
        assert parsed["summary"]["completed"] == 1
        assert parsed["summary"]["pending"] == 1
        assert parsed["summary"]["in_progress"] == 1

    def test_write_todos_invalid_status(self, todo_tool):
        result = todo_tool.write_todos([{"content": "Bad", "status": "unknown"}])
        assert "Error" in result

    def test_write_todos_missing_content(self, todo_tool):
        result = todo_tool.write_todos([{"status": "pending"}])
        assert "Error" in result

    def test_write_todos_none(self, todo_tool):
        result = todo_tool.write_todos(None)
        assert "Error" in result

    def test_write_todos_empty_list(self, todo_tool):
        result = todo_tool.write_todos([])
        assert "Error" in result

    def test_write_todos_overwrites(self, todo_tool):
        """Writing new todos replaces old ones entirely."""
        todo_tool.write_todos([{"content": "Old", "status": "pending"}])
        todo_tool.write_todos([{"content": "New1", "status": "pending"}, {"content": "New2", "status": "pending"}])
        result = todo_tool.read_todos()
        parsed = json.loads(result)
        assert parsed["summary"]["total"] == 2
        contents = [t["content"] for t in parsed["todos"]]
        assert "Old" not in contents
        assert "New1" in contents


# ===========================================================================
# BuiltinTaskTool tests (requires mocking LLM / Agent)
# ===========================================================================

class TestBuiltinTaskTool:
    def test_task_no_model_returns_error(self):
        """If no model is available, task should return an error."""
        tool = BuiltinTaskTool(model=None)
        # No parent agent set either → model is None
        result = asyncio.run(tool.task("do something"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "No model" in parsed["error"]

    def test_task_prevents_nested_subagent(self):
        """Subagent nesting should be blocked."""
        tool = BuiltinTaskTool(model=MagicMock())

        # Create a mock parent agent with an agent_id
        mock_parent = MagicMock()
        mock_parent.agent_id = "test-agent-123"
        tool.set_parent_agent(mock_parent)

        # Mock SubagentRegistry.is_subagent to return True (agent is already a subagent)
        async def run():
            with patch("agentica.subagent.SubagentRegistry.is_subagent", return_value=True):
                return await tool.task("nested task")

        result = asyncio.run(run())
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "Nested" in parsed["error"]

    def test_task_with_mocked_agent(self):
        """Full task flow with a mocked Agent that returns a stream."""
        mock_model = MagicMock()

        tool = BuiltinTaskTool(model=mock_model)

        # Mock the Agent constructor and its run_stream
        mock_chunk = MagicMock()
        mock_chunk.event = "RunResponse"
        mock_chunk.content = "Task result: 42"
        mock_chunk.tools = None

        async def mock_run_stream(*args, **kwargs):
            yield mock_chunk

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_stream = mock_run_stream

        # Agent is imported locally inside task() as `from agentica.agent import Agent`
        with patch("agentica.agent.Agent", return_value=mock_agent_instance):
            result = asyncio.run(tool.task("compute 6 * 7", subagent_type="code"))

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "42" in parsed["result"]
        assert parsed["subagent_type"] == "code"

    def test_task_exception_handling(self):
        """If the subagent raises an exception, task should return error JSON."""
        mock_model = MagicMock()
        tool = BuiltinTaskTool(model=mock_model)

        async def mock_run_stream(*args, **kwargs):
            raise RuntimeError("LLM crashed")
            yield  # noqa: E501 — unreachable, but makes this an async generator

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_stream = mock_run_stream

        # Agent is imported locally inside task() as `from agentica.agent import Agent`
        with patch("agentica.agent.Agent", return_value=mock_agent_instance):
            result = asyncio.run(tool.task("fail task"))

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "error" in parsed

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
        mock_agent.work_dir = "/some/path"
        tool.set_parent_agent(mock_agent)
        assert tool._parent_agent is mock_agent
        assert tool._work_dir == "/some/path"


# ===========================================================================
# BuiltinMemoryTool tests
# ===========================================================================

class TestBuiltinMemoryTool:
    def test_save_memory_no_workspace(self):
        tool = BuiltinMemoryTool(workspace=None)
        result = asyncio.run(tool.save_memory("remember this"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "No workspace" in parsed["error"]

    def test_save_memory_empty_content(self):
        tool = BuiltinMemoryTool()
        result = asyncio.run(tool.save_memory(""))
        assert "Error" in result

    def test_save_memory_with_mock_workspace(self):
        mock_ws = MagicMock()
        mock_ws.save_memory = AsyncMock(return_value=None)

        tool = BuiltinMemoryTool(workspace=mock_ws)
        result = asyncio.run(tool.save_memory("user likes Python", long_term=True))
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["memory_type"] == "long-term"
        mock_ws.save_memory.assert_awaited_once_with("user likes Python", long_term=True)

    def test_save_memory_daily(self):
        mock_ws = MagicMock()
        mock_ws.save_memory = AsyncMock(return_value=None)

        tool = BuiltinMemoryTool(workspace=mock_ws)
        result = asyncio.run(tool.save_memory("check docs tomorrow", long_term=False))
        parsed = json.loads(result)
        assert parsed["memory_type"] == "daily"

    def test_save_memory_workspace_error(self):
        mock_ws = MagicMock()
        mock_ws.save_memory = AsyncMock(side_effect=IOError("disk full"))

        tool = BuiltinMemoryTool(workspace=mock_ws)
        result = asyncio.run(tool.save_memory("important"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "disk full" in parsed["error"]

    def test_set_workspace(self):
        tool = BuiltinMemoryTool()
        assert tool._workspace is None
        mock_ws = MagicMock()
        tool.set_workspace(mock_ws)
        assert tool._workspace is mock_ws


# ===========================================================================
# get_builtin_tools factory function
# ===========================================================================

class TestGetBuiltinTools:
    def test_default_returns_all_tools(self):
        tools = get_builtin_tools(include_skills=False)
        # Should have: file, execute, web_search, fetch_url, todo, task, memory
        assert len(tools) >= 7
        tool_names = [t.name for t in tools]
        assert "builtin_file_tool" in tool_names
        assert "builtin_execute_tool" in tool_names
        assert "builtin_web_search_tool" in tool_names
        assert "builtin_fetch_url_tool" in tool_names
        assert "builtin_todo_tool" in tool_names
        assert "builtin_task_tool" in tool_names
        assert "builtin_memory_tool" in tool_names

    def test_exclude_all(self):
        tools = get_builtin_tools(
            include_file_tools=False,
            include_execute=False,
            include_web_search=False,
            include_fetch_url=False,
            include_todos=False,
            include_task=False,
            include_skills=False,
            include_memory=False,
        )
        assert len(tools) == 0

    def test_only_file_tools(self):
        tools = get_builtin_tools(
            include_file_tools=True,
            include_execute=False,
            include_web_search=False,
            include_fetch_url=False,
            include_todos=False,
            include_task=False,
            include_skills=False,
            include_memory=False,
        )
        assert len(tools) == 1
        assert tools[0].name == "builtin_file_tool"


# ===========================================================================
# Async nature verification
# ===========================================================================

class TestAsyncNature:
    """Verify that all registered tool methods are async (coroutine functions)."""

    def test_file_tool_methods_are_async(self):
        import inspect
        tool = BuiltinFileTool()
        for name in ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]:
            fn = getattr(tool, name)
            assert inspect.iscoroutinefunction(fn), f"{name} should be async"

    def test_execute_tool_is_async(self):
        import inspect
        tool = BuiltinExecuteTool()
        assert inspect.iscoroutinefunction(tool.execute)

    def test_web_search_is_async(self):
        import inspect
        tool = BuiltinWebSearchTool()
        assert inspect.iscoroutinefunction(tool.web_search)

    def test_fetch_url_is_async(self):
        import inspect
        tool = BuiltinFetchUrlTool()
        assert inspect.iscoroutinefunction(tool.fetch_url)

    def test_task_is_async(self):
        import inspect
        tool = BuiltinTaskTool()
        assert inspect.iscoroutinefunction(tool.task)

    def test_save_memory_is_async(self):
        import inspect
        tool = BuiltinMemoryTool()
        assert inspect.iscoroutinefunction(tool.save_memory)

    def test_todo_methods_are_sync(self):
        """write_todos and read_todos are pure in-memory, should remain sync."""
        import inspect
        tool = BuiltinTodoTool()
        assert not inspect.iscoroutinefunction(tool.write_todos)
        assert not inspect.iscoroutinefunction(tool.read_todos)
