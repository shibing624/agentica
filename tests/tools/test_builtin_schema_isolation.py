"""A tool schema must not name a tool that lives on another class.

SDK callers load builtins a la carte (files without execute, memory
without file tools). A description that says "use execute" then 404s.
"""
import re

from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.builtin import (
    BuiltinExecuteTool,
    BuiltinFetchUrlTool,
    BuiltinFileTool,
    BuiltinMemoryTool,
    BuiltinTodoTool,
    BuiltinWebSearchTool,
)
from agentica.tools.builtin.delegate_tool import BuiltinDelegateTool
from agentica.tools.builtin_task_tool import BuiltinTaskTool

# Names that are separately installable. A function on one Tool must not
# mention a name that is only provided by a different Tool.
_FOREIGN = {
    "read_file": frozenset({"execute", "wait", "task", "delegate", "web_search", "fetch_url"}),
    "write_file": frozenset({"execute", "wait", "task", "delegate", "web_search", "fetch_url"}),
    "apply_patch": frozenset({"execute", "wait", "task", "delegate", "web_search", "fetch_url"}),
    "glob": frozenset({"execute", "wait", "task", "delegate", "web_search", "fetch_url"}),
    "grep": frozenset({"execute", "wait", "task", "delegate", "web_search", "fetch_url"}),
    "execute": frozenset({
        "read_file", "write_file", "apply_patch", "glob",
        "task", "delegate", "web_search", "fetch_url",
    }),
    "wait": frozenset({
        "read_file", "write_file", "apply_patch", "glob", "grep",
        "task", "web_search", "fetch_url",
    }),
    "delegate": frozenset({
        "task", "execute", "read_file", "write_file",
        "apply_patch", "glob", "grep", "web_search", "fetch_url",
    }),
    "task": frozenset({
        "execute", "wait", "delegate", "read_file", "write_file",
        "apply_patch", "glob", "grep", "web_search", "fetch_url",
    }),
    "web_search": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep", "glob",
        "task", "delegate", "fetch_url",
    }),
    "fetch_url": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep", "glob",
        "task", "delegate", "web_search",
    }),
    "write_todos": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep",
        "task", "delegate",
    }),
    "save_memory": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep", "glob",
        "task", "delegate",
    }),
    "search_memory": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep", "glob",
        "task", "delegate",
    }),
}


def _mentions(text: str, name: str) -> bool:
    """True when *name* is used as a tool, not as English or an rg flag."""
    return re.search(rf"(?:`{re.escape(name)}`|{re.escape(name)}\()", text) is not None


def _texts_for(tool, name: str) -> str:
    fn = tool.functions[name]
    parts = [fn.description or "", str(fn.parameters or "")]
    prompt = tool.get_system_prompt()
    if prompt:
        parts.append(prompt)
    return "\n".join(parts)


async def _noop_search(queries, max_results=5):
    return ""


def _tools():
    registry = BackgroundProcessRegistry()
    return [
        BuiltinFileTool(work_dir="/tmp"),
        BuiltinExecuteTool(work_dir="/tmp", background_process_registry=registry),
        BuiltinTaskTool(),
        BuiltinDelegateTool(
            background_process_registry=registry,
            permission_mode=lambda: "allow-all",
        ),
        BuiltinWebSearchTool(search_fn=_noop_search),
        BuiltinFetchUrlTool(),
        BuiltinTodoTool(),
        BuiltinMemoryTool(),
    ]


def test_builtin_schemas_do_not_name_foreign_tools():
    hits = []
    for tool in _tools():
        for name in tool.functions:
            banned = _FOREIGN.get(name)
            if not banned:
                continue
            text = _texts_for(tool, name)
            for other in sorted(banned):
                if _mentions(text, other):
                    hits.append(f"{name} names {other}")
    assert hits == []


def test_delegate_schema_names_wait():
    registry = BackgroundProcessRegistry()
    tool = BuiltinDelegateTool(
        background_process_registry=registry,
        permission_mode=lambda: "allow-all",
    )
    doc = tool.functions["delegate"].description or ""
    assert "wait(id=" in doc


def test_tools_md_does_not_name_execute():
    from agentica.prompts.base.tools import get_tools_prompt
    content = get_tools_prompt()
    assert not _mentions(content, "execute")
    assert not _mentions(content, "task")
    assert not _mentions(content, "delegate")
