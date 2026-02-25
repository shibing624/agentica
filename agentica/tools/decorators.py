# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: @tool decorator for attaching metadata to tool functions.

Usage:
    from agentica.tools.decorators import tool

    @tool(name="web_search", description="Search the web")
    def search(query: str, max_results: int = 5) -> str:
        ...

    agent = Agent(tools=[search])
"""
from functools import wraps
from typing import Optional


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    show_result: bool = False,
    sanitize_arguments: bool = True,
    stop_after_tool_call: bool = False,
):
    """Decorator: attach tool metadata to a function for Agent auto-detection.

    When a decorated function is passed to Agent(tools=[...]), Function.from_callable()
    will detect the _tool_metadata attribute and use it instead of parsing docstring/type hints.

    Args:
        name: Tool name (defaults to function __name__).
        description: Tool description (defaults to function docstring).
        show_result: Whether to show tool result to the user.
        sanitize_arguments: Whether to sanitize tool arguments.
        stop_after_tool_call: Whether to stop agent execution after this tool call.

    Returns:
        Decorated function with _tool_metadata attribute.
    """
    def decorator(func):
        func._tool_metadata = {
            "name": name or func.__name__,
            "description": description or (func.__doc__ or "").strip(),
            "show_result": show_result,
            "sanitize_arguments": sanitize_arguments,
            "stop_after_tool_call": stop_after_tool_call,
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_metadata = func._tool_metadata
        return wrapper

    return decorator
