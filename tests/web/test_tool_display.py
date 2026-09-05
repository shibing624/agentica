# -*- coding: utf-8 -*-
"""Web chat tool-call display + steer/think freeze.

The formatter lives in ``web/src/lib/toolDisplay.ts`` (CLI
``format_tool_display`` port). These cases pin the Python source of truth so
the two cannot silently drift on the tools the chat row shows.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

from agentica.cli.display.tool_format import format_tool_display


def test_read_file_glob_grep():
    assert "foo.py" in format_tool_display("read_file", {"file_path": "/tmp/proj/foo.py", "offset": 0, "limit": 80})
    assert "L1-" in format_tool_display("read_file", {"file_path": "a.py"})
    assert "*.py in" in format_tool_display("glob", {"pattern": "*.py", "path": "."})
    grep = format_tool_display("grep", {"pattern": "hello", "path": "src"})
    assert "'hello'" in grep
    assert "src" in grep


def test_writes_execute_todos_search():
    assert format_tool_display("write_file", {"file_path": "agentica/cli/commands/session.py"}) == (
        "agentica/cli/commands/session.py"
    )
    cwd_abs = os.path.join(os.getcwd(), "agentica", "cli", "commands", "session.py")
    assert format_tool_display("write_file", {"file_path": cwd_abs}) == (
        "agentica/cli/commands/session.py"
    )
    assert format_tool_display("write_file", {"file_path": "/x/y/z.py"}) == "/x/y/z.py"
    assert format_tool_display("apply_patch", {
        "patch": "*** Update File: agentica/cli/commands/session.py\n@@\n-a\n+b\n",
    }) == "agentica/cli/commands/session.py"
    assert format_tool_display("apply_patch", {
        "patch": f"*** Update File: {cwd_abs}\n@@\n-a\n+b\n",
    }) == "agentica/cli/commands/session.py"
    assert format_tool_display("apply_patch", {
        "patch": "*** Add File: a.py\n*** Update File: b.py\n",
    }) == "a.py, b.py"
    assert "pytest" in format_tool_display("execute", {"command": "pytest tests/a.py"})
    todos = format_tool_display("write_todos", {"todos": [
        {"content": "done", "status": "completed"},
        {"content": "next", "status": "pending"},
    ]})
    assert "✓ done" in todos and "○ next" in todos
    assert "alpha" in format_tool_display("web_search", {"queries": ["alpha", "beta"]})
    assert format_tool_display("fetch_url", {"url": "https://example.com/x"}).startswith("https://")


def test_task_peers_and_default():
    task = format_tool_display("task", {
        "subagent_type": "explore",
        "description": "find the bug\nin two files",
    })
    assert "subagent_type=" in task
    assert "find the bug" in task
    assert format_tool_display("list_agents", {}) == ""
    send = format_tool_display("send_message", {"target": "web-af", "message": "please commit"})
    assert "→ web-af" in send
    assert "please commit" in send
    mem = format_tool_display("save_memory", {"title": "note", "content": "keep this", "memory_type": "project"})
    assert "title=" in mem and "note" in mem
    search = format_tool_display("search_memory", {"query": "worktree lock"})
    assert "query=" in search
    wt = format_tool_display("worktree", {"action": "use", "name": "gateway-peers"})
    assert "action=" in wt and "use" in wt
    assert "gateway-peers" in wt
