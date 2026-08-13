# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical built-in tools package — factory + tool classes
"""

from typing import List, Optional, TYPE_CHECKING

from agentica.tools.base import Tool
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.builtin.file_tool import BuiltinFileTool
from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
from agentica.tools.builtin.task_state_tools import BuiltinMemoryTool, BuiltinTodoTool
from agentica.tools.builtin.web_tools import (
    BuiltinFetchUrlTool,
    BuiltinWebSearchTool,
    list_web_search_providers,
    register_web_search_backend,
)
from agentica.tools.builtin_task_tool import BuiltinTaskTool

if TYPE_CHECKING:
    from agentica.model.base import Model

__all__ = [
    "BuiltinFileTool",
    "BuiltinExecuteTool",
    "BuiltinFetchUrlTool",
    "BuiltinWebSearchTool",
    "BuiltinTodoTool",
    "BuiltinMemoryTool",
    "BuiltinTaskTool",
    "get_builtin_tools",
    "register_web_search_backend",
    "list_web_search_providers",
]


def get_builtin_tools(
        work_dir: Optional[str] = None,
        include_file_tools: bool = True,
        include_execute: bool = True,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        include_todos: bool = True,
        include_task: bool = True,
        include_skills: bool = False,
        include_ask_user_question: bool = False,
        task_model: Optional["Model"] = None,
        custom_skill_dirs: Optional[List[str]] = None,
        ask_user_question_callback=None,
        sandbox_config=None,
        background_process_registry: Optional[BackgroundProcessRegistry] = None,
        enable_diagnostics: bool = False,
        diagnostics_servers: Optional[List[str]] = None,
        diagnostics_errors_only: bool = True,
        web_search_provider: Optional[str] = None,
        peer_conflict_checker=None,
) -> List[Tool]:
    """
    Get the list of built-in tools for Agent.

    Args:
        work_dir: Work directory for file operations
        include_file_tools: Whether to include file tools (ls, read_file, write_file,
            edit_file, apply_patch, glob, grep)
        include_execute: Whether to include code execution tool
        include_web_search: Whether to include web search tool
        include_fetch_url: Whether to include URL fetching tool
        include_todos: Whether to include task management tools
        include_task: Whether to include subagent task tool
        include_skills: Whether to include skill tool for executing skills (default: False)
        include_ask_user_question: Whether to include ask_user_question tool for human-in-the-loop (default: False)
        task_model: Optional model for the cheap (``auxiliary``) tier of subagents
            spawned by the ``task`` tool. When ``None`` the parent agent's
            ``resolve_auxiliary_model("task")`` decides. Definitions that opt
            into ``model_tier: main`` run on the parent's own model.
        custom_skill_dirs: Custom skill directories to load (optional)
        ask_user_question_callback: Custom callback for ask_user_question tool (optional)
        sandbox_config: SandboxConfig instance for security isolation (optional)
        background_process_registry: Shared registry for execute(background=True) (optional)
        enable_diagnostics: When True, attach an LSP diagnostics checker to the
            file tool so write/edit results report newly-introduced type/import/
            syntax errors. Language servers start lazily on the first edit
            (not during Agent construction). Requires a language server
            (e.g. ``pip install 'pyright[nodejs]'``); degrades to a no-op if
            initialize fails. Default False.
        diagnostics_servers: LSP server names to use (default ["pyright"]).
        diagnostics_errors_only: When True (default), only severity "error"
            diagnostics are surfaced to the model.
        web_search_provider: Engine behind the ``web_search`` tool, e.g.
            "baidu", "bocha", "serper", "exa", "duckduckgo", "zhipu", "mcp".
            Defaults to the ``AGENTICA_WEB_SEARCH`` env var, then to
            ``DEFAULT_WEB_SEARCH_PROVIDER``. The tool name stays ``web_search``
            whichever engine is used.

    Returns:
        List of tools
    """
    tools = []

    if include_file_tools:
        diagnostics_checker = None
        if enable_diagnostics:
            from agentica.lsp_diagnostics import LspDiagnosticsChecker
            # Attach eagerly; LSP servers start lazily on first file edit.
            diagnostics_checker = LspDiagnosticsChecker(
                work_dir=work_dir,
                servers=diagnostics_servers,
                errors_only=diagnostics_errors_only,
            )
        tools.append(BuiltinFileTool(
            work_dir=work_dir,
            sandbox_config=sandbox_config,
            diagnostics_checker=diagnostics_checker,
            consent_callback=ask_user_question_callback,
            peer_conflict_checker=peer_conflict_checker,
        ))

    if include_execute:
        tools.append(BuiltinExecuteTool(
            work_dir=work_dir,
            sandbox_config=sandbox_config,
            background_process_registry=background_process_registry,
        ))

    if include_web_search:
        tools.append(BuiltinWebSearchTool(provider=web_search_provider))

    if include_fetch_url:
        tools.append(BuiltinFetchUrlTool())

    if include_todos:
        tools.append(BuiltinTodoTool())

    if include_task:
        tools.append(BuiltinTaskTool(auxiliary_model=task_model))

    if include_skills:
        from agentica.tools.skill_tool import SkillTool
        tools.append(SkillTool(custom_skill_dirs=custom_skill_dirs, auto_load=True))

    if include_ask_user_question:
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool
        tools.append(AskUserQuestionTool(input_callback=ask_user_question_callback))

    return tools
