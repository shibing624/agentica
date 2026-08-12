# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica
"""

import importlib


_EXPORTS = {
    "TOOL_ICONS": ("agentica.cli.runtime", "TOOL_ICONS"),
    "TOOL_REGISTRY": ("agentica.cli.runtime", "TOOL_REGISTRY"),
    "MODEL_REGISTRY": ("agentica.cli.runtime", "MODEL_REGISTRY"),
    "EXAMPLE_MODELS": ("agentica.cli.runtime", "EXAMPLE_MODELS"),
    "BUILTIN_TOOLS": ("agentica.cli.runtime", "BUILTIN_TOOLS"),
    "history_file": ("agentica.cli.runtime", "history_file"),
    "console": ("agentica.cli.runtime", "console"),
    "get_console": ("agentica.cli.runtime", "get_console"),
    "set_active_console": ("agentica.cli.runtime", "set_active_console"),
    "parse_args": ("agentica.cli.runtime", "parse_args"),
    "configure_tools": ("agentica.cli.runtime", "configure_tools"),
    "get_model": ("agentica.cli.runtime", "get_model"),
    "create_agent": ("agentica.cli.runtime", "create_agent"),
    "COLORS": ("agentica.cli.display", "COLORS"),
    "StreamDisplayManager": ("agentica.cli.display", "StreamDisplayManager"),
    "format_tool_display": ("agentica.cli.display", "format_tool_display"),
    "display_tool_call": ("agentica.cli.display", "display_tool_call"),
    "display_user_message": ("agentica.cli.display", "display_user_message"),
    "display_token_stats": ("agentica.cli.display", "display_token_stats"),
    "display_diff": ("agentica.cli.display", "display_diff"),
    "render_markdown_response": ("agentica.cli.display", "render_markdown_response"),
    "show_help": ("agentica.cli.display", "show_help"),
    "print_header": ("agentica.cli.display", "print_header"),
    "run_interactive": ("agentica.cli.interactive", "run_interactive"),
    "resolve_model_config": ("agentica.cli.setup", "resolve_model_config"),
    "run_onboarding": ("agentica.cli.setup", "run_onboarding"),
    "PROVIDER_PRESETS": ("agentica.cli.setup", "PROVIDER_PRESETS"),
}

_SUBMODULES = {
    "commands": "agentica.cli.commands",
    "setup": "agentica.cli.setup",
    "self_manage": "agentica.cli.self_manage",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr_name)
    if name in _SUBMODULES:
        return importlib.import_module(_SUBMODULES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _main_entrypoint(*args, **kwargs):
    """Console-script entrypoint wrapper.

    Installed scripts import ``agentica.cli:main``. Importing the real
    ``agentica.cli.main`` submodule temporarily sets ``agentica.cli.main`` to
    that module object on the package, so expose a real wrapper here instead of
    relying on lazy ``__getattr__`` for this one name.
    """
    from agentica.cli.main import main as _main

    globals()["main"] = _main_entrypoint
    return _main(*args, **kwargs)


main = _main_entrypoint


def __dir__():
    eager_names = [name for name in globals() if not name.startswith("_")]
    return sorted(set(eager_names) | set(_EXPORTS) | set(_SUBMODULES) | {"main"})


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
