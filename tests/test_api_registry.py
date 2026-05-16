# -*- coding: utf-8 -*-
"""Regression tests for the centralized top-level API registry."""

from agentica.api_registry import (
    LAZY_IMPORTS,
    PROVIDER_ALIAS_EXPORTS,
    PUBLIC_API_ALL,
)


def test_api_registry_contains_core_lazy_exports():
    assert LAZY_IMPORTS["SqliteDb"] == "agentica.db.sqlite"
    assert LAZY_IMPORTS["AskUserQuestionTool"] == "agentica.tools.user_input_tool"
    assert LAZY_IMPORTS["AskUserQuestionRequired"] == "agentica.tools.user_input_tool"
    assert "UserInputTool" not in LAZY_IMPORTS
    assert "UserInputRequired" not in LAZY_IMPORTS
    # OpenAIChat + builtin tools are eager (openai is a hard dep, builtin tools have no extra deps)
    assert "OpenAIChat" not in LAZY_IMPORTS
    assert "BuiltinTodoTool" not in LAZY_IMPORTS
    assert "BuiltinFileTool" not in LAZY_IMPORTS


def test_eager_top_level_imports_are_directly_accessible():
    """OpenAIChat and 7 builtin tools must be importable via `from agentica import X`."""
    from agentica import (
        OpenAIChat,
        BuiltinFileTool, BuiltinExecuteTool, BuiltinFetchUrlTool,
        BuiltinWebSearchTool, BuiltinTodoTool, BuiltinTaskTool,
        BuiltinMemoryTool,
    )
    assert OpenAIChat is not None
    assert BuiltinFileTool is not None
    assert BuiltinExecuteTool is not None
    assert BuiltinFetchUrlTool is not None
    assert BuiltinWebSearchTool is not None
    assert BuiltinTodoTool is not None
    assert BuiltinTaskTool is not None
    assert BuiltinMemoryTool is not None


def test_api_registry_contains_provider_aliases():
    assert "DeepSeekChat" in PROVIDER_ALIAS_EXPORTS
    assert "MoonshotChat" in PROVIDER_ALIAS_EXPORTS
    assert "ZhipuAIChat" in PROVIDER_ALIAS_EXPORTS
    assert "NvidiaChat" in PROVIDER_ALIAS_EXPORTS
    assert "PROVIDER_FACTORIES" in PROVIDER_ALIAS_EXPORTS


def test_agentica_public_api_uses_registry_names():
    import agentica

    assert agentica.__all__ == PUBLIC_API_ALL
    assert "DeepSeekChat" in agentica.__all__
    assert "SqliteDb" in agentica.__all__
    assert "AskUserQuestionTool" in agentica.__all__
    assert "AskUserQuestionRequired" in agentica.__all__
    assert "UserInputTool" not in agentica.__all__
    assert "UserInputRequired" not in agentica.__all__


def test_ask_user_question_tool_lazy_import():
    import agentica
    from agentica.tools.user_input_tool import AskUserQuestionRequired, AskUserQuestionTool

    assert agentica.AskUserQuestionTool is AskUserQuestionTool
    assert agentica.AskUserQuestionRequired is AskUserQuestionRequired


def test_agentica_dir_does_not_expose_registry_internals():
    import agentica

    visible = dir(agentica)
    assert "api_registry" not in visible
    assert "LAZY_IMPORTS" not in visible
    assert "PUBLIC_API_ALL" not in visible


def test_top_level_lazy_access_returns_symbol():
    import agentica
    from agentica.subagent import SubagentType

    assert agentica.SubagentType is SubagentType
