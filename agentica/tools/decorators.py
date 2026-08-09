# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: @tool decorator for attaching metadata to tool functions.

Supports BOTH bare and parameterized usage (since v1.3.6):

    # Bare decorator (NEW in v1.3.6)::
    from agentica import tool

    @tool
    def search(query: str) -> str:
        '''Search the web for a query.'''
        ...

    # Parameterized::
    @tool(name="web_search", description="Search the web")
    def search(query: str, max_results: int = 5) -> str:
        ...

    agent = Agent(tools=[search])
"""
from functools import wraps
from typing import Callable, Optional, Union


def _apply_metadata(
    func: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    show_result: bool = False,
    stop_after_tool_call: bool = False,
    concurrency_safe: bool = False,
    is_read_only: bool = False,
    is_destructive: bool = False,
    deferred: bool = False,
    interrupt_behavior: str = "cancel",
    available_when: Optional[Callable[[], bool]] = None,
) -> Callable:
    """Attach _tool_metadata attribute to func and return a wrapper.

    Factored out so we can call it from both bare and parameterized paths.
    """
    metadata = {
        "name": name or func.__name__,
        "description": description or (func.__doc__ or "").strip(),
        "show_result": show_result,
        "stop_after_tool_call": stop_after_tool_call,
        "concurrency_safe": concurrency_safe,
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "deferred": deferred,
        "interrupt_behavior": interrupt_behavior,
        "available_when": available_when,
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._tool_metadata = metadata
    # Also attach to the original func for introspection
    func._tool_metadata = metadata
    return wrapper


def tool(
    func_or_name: Union[Callable, Optional[str]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    show_result: bool = False,
    stop_after_tool_call: bool = False,
    concurrency_safe: bool = False,
    is_read_only: bool = False,
    is_destructive: bool = False,
    deferred: bool = False,
    interrupt_behavior: str = "cancel",
    available_when: Optional[Callable[[], bool]] = None,
):
    """Decorator: attach tool metadata to a function for Agent auto-detection.

    Supports BOTH bare and parameterized usage:

        @tool
        def my_func(x: str) -> str: ...

        @tool(name="my_func", description="...")
        def my_func(x: str) -> str: ...

    When a decorated function is passed to Agent(tools=[...]),
    Function.from_callable() will detect the _tool_metadata attribute.

    Args:
        func_or_name: Either the decorated function (bare usage) or a name string.
            For backward compat with `@tool(name="...")`, the first positional
            parameter is also treated as `name` when it is a string.
        name: Tool name (defaults to function __name__). Keyword-only.
        description: Tool description (defaults to function docstring).
        show_result: Whether to show tool result to the user.
        stop_after_tool_call: Whether to stop agent execution after this tool call.
        concurrency_safe: If True the tool may run in parallel with other
            concurrency_safe tools (e.g. read_file, glob, grep).
            Write/shell tools should keep this False (default).
        is_read_only: If True, the tool only reads data and never modifies state.
        is_destructive: If True, the tool performs irreversible operations
            (delete, overwrite, send, execute).
        deferred: If True, tool description is not sent to LLM by default.
            The tool can be discovered via tool_search and loaded on demand.
        interrupt_behavior: "cancel" (tool can be terminated mid-execution)
            or "block" (tool must complete before honoring cancellation).
        available_when: Optional callback that returns True when the tool
            should be exposed to the LLM. False hides the tool schema.

    Returns:
        Decorated function with _tool_metadata attribute.
    """
    # Case 1: Bare decorator usage -> @tool
    # func_or_name is the decorated function itself.
    if callable(func_or_name):
        return _apply_metadata(
            func_or_name,
            name=name,
            description=description,
            show_result=show_result,
            stop_after_tool_call=stop_after_tool_call,
            concurrency_safe=concurrency_safe,
            is_read_only=is_read_only,
            is_destructive=is_destructive,
            deferred=deferred,
            interrupt_behavior=interrupt_behavior,
            available_when=available_when,
        )

    # Case 2: Parameterized usage -> @tool(name="...") or @tool("...")
    # func_or_name is None or a string (legacy first-positional name).
    resolved_name = name if name is not None else func_or_name

    def decorator(func: Callable) -> Callable:
        return _apply_metadata(
            func,
            name=resolved_name,
            description=description,
            show_result=show_result,
            stop_after_tool_call=stop_after_tool_call,
            concurrency_safe=concurrency_safe,
            is_read_only=is_read_only,
            is_destructive=is_destructive,
            deferred=deferred,
            interrupt_behavior=interrupt_behavior,
            available_when=available_when,
        )

    return decorator
