# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Print response methods for Agent (Async-First)

All core print methods are async. Synchronous wrappers use run_sync().
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from agentica.utils.log import logger
from agentica.utils.async_utils import run_sync
from agentica.model.message import Message
from agentica.run_display import RunDisplayEventKind, classify_run_response
from agentica.run_response import ToolCallInfo
from agentica.utils.message import get_text_from_message

# Same set as the web row: the call line is enough. CLI still prints a
# one-liner count at completion; SDK stdout matches the web body policy.
_HIDE_RESULT_TOOLS = frozenset({"read_file", "glob", "grep"})
_RESULT_PREVIEW_CHARS = 200


def _tool_call_line(tool_info, work_dir: Optional[str] = None) -> str:
    """CLI ``format_tool_display`` copy on one stdout line."""
    from pathlib import Path

    from agentica.cli.display.tool_format import format_tool_display

    name = tool_info.tool_name or "unknown"
    # Without work_dir the formatter falls back to os.getcwd(), which is the
    # wrong root whenever the agent runs against another directory
    # (delegate(work_dir=...), SDK embedding).
    display = format_tool_display(
        name,
        tool_info.tool_args or {},
        work_dir=Path(work_dir) if work_dir else None,
    )
    if display:
        return f"  🔧 {name} {display}"
    return f"  🔧 {name}"


def _tool_result_line(tool_info) -> Optional[str]:
    name = tool_info.tool_name or "unknown"
    if name in _HIDE_RESULT_TOOLS and not tool_info.is_error:
        return None
    result = str(tool_info.content or "")
    if not result:
        return None
    preview = (
        result[:_RESULT_PREVIEW_CHARS] + "..."
        if len(result) > _RESULT_PREVIEW_CHARS
        else result
    )
    return f"     📤 {name}: {preview}"


class PrinterMixin:
    """Mixin class containing print response methods for Agent."""

    async def print_response(
        self,
        message: Optional[Union[List, Dict, str, Message]] = None,
        *,
        messages: Optional[List[Union[Dict, Message]]] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_tool_calls: bool = True,
        **kwargs: Any,
    ) -> None:
        """Print the response from the Agent (non-streaming, async).

        For streaming output, use print_response_stream().
        For synchronous usage, use print_response_sync() / print_response_stream_sync().
        """
        if show_message and message is not None:
            message_content = get_text_from_message(message)
            print("=" * 80)
            print("📝 MESSAGE")
            print("=" * 80)
            print(message_content)
            print()

        run_response = await self.run(message=message, messages=messages, **kwargs)

        print("=" * 80)
        print("🤖 RESPONSE")
        print("=" * 80)

        has_reasoning = (
            show_reasoning and hasattr(run_response, "reasoning_content") and run_response.reasoning_content
        )
        if has_reasoning:
            print("💭 THINKING")
            print("-" * 40)
            print(run_response.reasoning_content)

        tools_shown = False
        if show_tool_calls and run_response.tools:
            print()
            for tool in run_response.tools:
                info = tool if isinstance(tool, ToolCallInfo) else ToolCallInfo.from_dict(tool)
                print(_tool_call_line(info, getattr(self, "work_dir", None)))
                result_line = _tool_result_line(info)
                if result_line:
                    print(result_line)
                tools_shown = True

        if has_reasoning or tools_shown:
            print()
            print("-" * 40)
            print("💬 ANSWER")
            print("-" * 40)

        if run_response.content:
            print(run_response.content)

    async def print_response_stream(
        self,
        message: Optional[Union[List, Dict, str, Message]] = None,
        *,
        messages: Optional[List[Union[Dict, Message]]] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_tool_calls: bool = True,
        stream_intermediate_steps: bool = False,
        **kwargs: Any,
    ) -> None:
        """Print the streaming response from the Agent (async).

        Usage:
            await agent.print_response_stream("...")
        """
        from agentica.run_config import RunConfig

        if self.response_model is not None:
            logger.warning("Structured output does not support streaming. Falling back to non-streaming.")
            return await self.print_response(
                message=message,
                messages=messages,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_tool_calls=show_tool_calls,
                **kwargs,
            )

        if show_message and message is not None:
            message_content = get_text_from_message(message)
            print("=" * 80)
            print("📝 MESSAGE")
            print("=" * 80)
            print(message_content)
            print()

        print("=" * 80)
        print("🤖 RESPONSE")
        print("=" * 80)

        _last_content = ""
        _last_reasoning = ""
        _in_thinking = False
        _need_answer_header = False

        _save = kwargs.pop("save_response_to_file", None)
        _cfg = kwargs.pop("config", None) or RunConfig(
            stream_intermediate_steps=stream_intermediate_steps or show_tool_calls,
            save_response_to_file=_save,
        )
        if not _cfg.stream_intermediate_steps and (stream_intermediate_steps or show_tool_calls):
            _cfg.stream_intermediate_steps = True

        async for run_response in self.run_stream(
            message=message,
            messages=messages,
            config=_cfg,
            **kwargs,
        ):
            display_event = classify_run_response(run_response)
            kind = display_event.kind

            if kind in (
                RunDisplayEventKind.METADATA_SKIP,
                RunDisplayEventKind.TELEMETRY_ONLY,
            ):
                continue

            if kind in (
                RunDisplayEventKind.TOOL_STARTED,
                RunDisplayEventKind.TOOL_COMPLETED,
            ):
                _in_thinking = False
                if not show_tool_calls:
                    continue
                tool_info = run_response.tool_call
                if tool_info is None:
                    continue
                if kind == RunDisplayEventKind.TOOL_STARTED:
                    print(
                        f"\n{_tool_call_line(tool_info, getattr(self, 'work_dir', None))}",
                        flush=True,
                    )
                    _need_answer_header = True
                else:
                    result_line = _tool_result_line(tool_info)
                    if result_line:
                        print(result_line, flush=True)
                    _need_answer_header = True
                continue

            if show_reasoning and run_response.reasoning_content:
                if run_response.reasoning_content != _last_reasoning:
                    if not _in_thinking:
                        print("💭 THINKING")
                        print("-" * 40)
                        _in_thinking = True
                    print(run_response.reasoning_content, end="", flush=True)
                    _last_reasoning = run_response.reasoning_content
                    _need_answer_header = True

            if run_response.content and run_response.content != _last_content:
                if _need_answer_header:
                    print()
                    print("-" * 40)
                    print("💬 ANSWER")
                    print("-" * 40)
                    _need_answer_header = False
                    _in_thinking = False
                print(run_response.content, end="", flush=True)
                _last_content = run_response.content

        print()  # final newline

    def print_response_sync(
        self,
        message: Optional[Union[List, Dict, str, Message]] = None,
        *,
        messages: Optional[List[Union[Dict, Message]]] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_tool_calls: bool = True,
        **kwargs: Any,
    ) -> None:
        """Synchronous wrapper for print_response() (non-streaming)."""
        run_sync(
            self.print_response(
                message=message,
                messages=messages,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_tool_calls=show_tool_calls,
                **kwargs,
            )
        )

    def print_response_stream_sync(
        self,
        message: Optional[Union[List, Dict, str, Message]] = None,
        *,
        messages: Optional[List[Union[Dict, Message]]] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_tool_calls: bool = True,
        stream_intermediate_steps: bool = False,
        **kwargs: Any,
    ) -> None:
        """Synchronous wrapper for print_response_stream()."""
        run_sync(
            self.print_response_stream(
                message=message,
                messages=messages,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_tool_calls=show_tool_calls,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )
        )

    def cli_app(
        self,
        message: Optional[str] = None,
        user: str = "User",
        emoji: str = "😎",
        stream: bool = True,
        show_message: bool = False,
        exit_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Command line interface for the Agent.

        A simple REPL for quick agent testing. For full-featured CLI with tools,
        skills, and session management, use `from agentica.cli import main`.

        Args:
            message: Initial message to send (optional)
            user: User name display
            emoji: Emoji for user prompt
            stream: Whether to use streaming output (default: True for better interactive experience)
            exit_on: List of commands to exit (default: ["exit", "quit", "bye"])
            **kwargs: Additional arguments passed to print methods
        """
        if message:
            if stream:
                self.print_response_stream_sync(message=message, show_message=show_message, **kwargs)
            else:
                self.print_response_sync(message=message, show_message=show_message, **kwargs)

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            try:
                user_input = input(f"{emoji} {user}: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if user_input.strip() in _exit_on:
                print("Goodbye!")
                break

            if stream:
                self.print_response_stream_sync(message=user_input, show_message=show_message, **kwargs)
            else:
                self.print_response_sync(message=user_input, show_message=show_message, **kwargs)
