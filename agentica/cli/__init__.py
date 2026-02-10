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
    history_file,
    console,
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
    show_help,
    print_header,
)
from agentica.cli.interactive import run_interactive
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
    "show_help",
    "print_header",
    "run_interactive",
    "main",
]
