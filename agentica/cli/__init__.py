# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica
"""
from agentica.cli.config import (
    TOOL_ICONS,
    TOOL_REGISTRY,
    MODEL_REGISTRY,
    EXAMPLE_MODELS,
    BUILTIN_TOOLS,
    history_file,
    console,
    get_console,
    set_active_console,
    parse_args,
    configure_tools,
    get_model,
    create_agent,
)
from agentica.cli.display import (
    COLORS,
    StreamDisplayManager,
    format_tool_display,
    display_tool_call,
    display_user_message,
    display_token_stats,
    display_diff,
    render_markdown_response,
    show_help,
    print_header,
)
from agentica.cli.interactive import run_interactive
from agentica.cli.setup import (
    resolve_model_config,
    run_onboarding,
    PROVIDER_PRESETS,
)
from agentica.cli.main import main

__all__ = [
    "TOOL_ICONS",
    "TOOL_REGISTRY",
    "MODEL_REGISTRY",
    "EXAMPLE_MODELS",
    "history_file",
    "console",
    "parse_args",
    "configure_tools",
    "get_model",
    "create_agent",
    "COLORS",
    "StreamDisplayManager",
    "format_tool_display",
    "display_tool_call",
    "display_user_message",
    "display_token_stats",
    "display_diff",
    "render_markdown_response",
    "show_help",
    "print_header",
    "run_interactive",
    "resolve_model_config",
    "run_onboarding",
    "PROVIDER_PRESETS",
    "main",
]
