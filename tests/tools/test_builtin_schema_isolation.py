"""A tool schema must not name a tool that lives on another class.

SDK callers load builtins a la carte (files without execute, memory
without file tools). A description that says "use execute" then 404s.
"""
import re

from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.builtin import (
    BuiltinContextTool,
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
    "search_session": frozenset({
        "execute", "read_file", "write_file", "apply_patch", "grep", "glob",
        "task", "delegate", "web_search", "fetch_url",
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
        BuiltinContextTool(),
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


def test_every_builtin_tool_property_has_description_and_clean_top_doc():
    """Every builtin tool schema must document all parameters on ``properties``
    without leaving raw Google-docstring ``Args:`` / ``Returns:`` boilerplate
    in the top-level tool description."""
    from unittest.mock import MagicMock
    from agentica.tools.ask_user_question_tool import AskUserQuestionTool
    from agentica.tools.builtin.vision_tool import BuiltinVisionTool
    from agentica.tools.cron_tool import CronTool
    from agentica.tools.goal_tool import GoalTool
    from agentica.tools.peer_tool import PeerMessagingTool
    from agentica.tools.self_manage_tool import SelfManageTool
    from agentica.tools.skill_tool import SkillTool
    from agentica.tools.worktree_tool import WorktreeTool

    all_tools = [
        *_tools(),
        BuiltinVisionTool(),
        SkillTool(),
        AskUserQuestionTool(),
        WorktreeTool(MagicMock()),
        PeerMessagingTool(MagicMock()),
        CronTool(),
        SelfManageTool(),
        GoalTool(MagicMock()),
    ]
    missing_prop_docs = []
    leaked_sections = []
    for tool_inst in all_tools:
        for fn_name, fn in tool_inst.functions.items():
            fn.process_entrypoint()
            desc = fn.description or ""
            if "\nArgs:" in desc or "\nReturns:" in desc:
                leaked_sections.append(fn_name)
            for prop_name, spec in (fn.parameters.get("properties") or {}).items():
                if not isinstance(spec, dict) or not str(spec.get("description") or "").strip():
                    missing_prop_docs.append(f"{fn_name}.{prop_name}")

    assert leaked_sections == []
    assert missing_prop_docs == []


def test_key_builtin_tool_schemas_match_structured_contracts():
    """Pin structural schema improvements on apply_patch, task, delegate,
    write_todos, and web_search."""
    registry = BackgroundProcessRegistry()
    file_tool = BuiltinFileTool(work_dir="/tmp")
    patch_fn = file_tool.functions["apply_patch"]
    patch_fn.process_entrypoint()
    assert "*** Add File:" in (patch_fn.description or "")
    assert "*** Delete File:" in (patch_fn.description or "")
    assert "*** End of File" in (patch_fn.description or "")
    assert "*** Add File:" in patch_fn.parameters["properties"]["patch"]["description"]

    task_fn = BuiltinTaskTool().functions["task"]
    task_fn.process_entrypoint()
    assert task_fn.parameters["properties"]["model_tier"]["enum"] == ["auxiliary", "main"]
    assert "cannot see" in task_fn.parameters["properties"]["description"]["description"] or (
        "does not share" in task_fn.parameters["properties"]["description"]["description"]
    )

    delegate_fn = BuiltinDelegateTool(
        background_process_registry=registry,
        permission_mode=lambda: "allow-all",
    ).functions["delegate"]
    delegate_fn.process_entrypoint()
    assert list(delegate_fn.parameters["properties"])[:2] == ["label", "task"]
    assert delegate_fn.parameters["required"] == ["task"]

    todos_fn = BuiltinTodoTool().functions["write_todos"]
    todos_fn.process_entrypoint()
    assert todos_fn.parameters["required"] == ["todos"]
    item_props = todos_fn.parameters["properties"]["todos"]["items"]["properties"]
    assert item_props["status"]["enum"] == ["pending", "in_progress", "completed"]

    ws_fn = BuiltinWebSearchTool(search_fn=_noop_search).functions["web_search"]
    ws_fn.process_entrypoint()
    assert ws_fn.parameters["properties"]["queries"]["type"] == "array"


def test_docstring_args_auto_populate_custom_tool_properties():
    from agentica.tools.base import Function
    from agentica.tools.decorators import tool

    @tool
    def lookup_ticket(ticket_id: str, limit: int = 5) -> str:
        """Fetch recent events for a ticket.

        Args:
            ticket_id: Canonical ticket identifier (e.g. "t_123").
            limit: Maximum number of events to return,
                ordered newest first.

        Returns:
            Formatted event log lines.
        """
        return f"{ticket_id}:{limit}"

    fn = Function.from_callable(lookup_ticket)
    assert "Args:" not in (fn.description or "")
    assert "Returns:" not in (fn.description or "")
    props = fn.parameters["properties"]
    assert props["ticket_id"]["description"] == 'Canonical ticket identifier (e.g. "t_123").'
    assert props["limit"]["description"] == "Maximum number of events to return, ordered newest first."

