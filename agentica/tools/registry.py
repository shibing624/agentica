# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Global tool registry for managing named tools.

Usage:
    from agentica.tools.registry import register_tool, get_tool, list_tools

    register_tool("calculator", calculate_func)
    tool = get_tool("calculator")
    all_tools = list_tools()
"""
from typing import Dict, Union, Callable, List


_TOOL_REGISTRY: Dict[str, Union[Callable, "Tool"]] = {}


def register_tool(name: str, tool_or_func: Union[Callable, "Tool"]) -> None:
    """Register a tool to the global registry.

    Args:
        name: Unique name for the tool.
        tool_or_func: A callable function or Tool instance.
    """
    _TOOL_REGISTRY[name] = tool_or_func


def get_tool(name: str) -> Union[Callable, "Tool"]:
    """Get a registered tool by name.

    Args:
        name: The tool name.

    Returns:
        The registered tool or function.

    Raises:
        KeyError: If the tool is not found.
    """
    if name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{name}' not found. Available: {sorted(_TOOL_REGISTRY.keys())}"
        )
    return _TOOL_REGISTRY[name]


def list_tools() -> List[str]:
    """List all registered tool names.

    Returns:
        Sorted list of tool names.
    """
    return sorted(_TOOL_REGISTRY.keys())


def unregister_tool(name: str) -> None:
    """Remove a tool from the registry.

    Args:
        name: The tool name to remove.

    Raises:
        KeyError: If the tool is not found.
    """
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found.")
    del _TOOL_REGISTRY[name]


def clear_registry() -> None:
    """Remove all tools from the registry."""
    _TOOL_REGISTRY.clear()
