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
    TYPE_CHECKING,
)

from agentica.utils.log import logger
from agentica.utils.async_utils import run_sync
from agentica.model.message import Message
from agentica.run_response import RunEvent
from agentica.utils.message import get_text_from_message

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class PrinterMixin:
    """Mixin class containing print response methods for Agent."""

    async def print_response(
            self: "Agent",
            message: Optional[Union[List, Dict, str, Message]] = None,
            *,
            messages: Optional[List[Union[Dict, Message]]] = None,
            stream: bool = False,
            show_message: bool = True,
            show_reasoning: bool = True,
            show_tool_calls: bool = False,
            show_intermediate_steps: bool = True,
            **kwargs: Any,
    ) -> None:
        """Print the response from the Agent (async).

        For synchronous usage, use print_response_sync().
        """
        if self.response_model is not None:
            stream = False

        if show_message and message is not None:
            message_content = get_text_from_message(message)
            print("=" * 80)
            print("📝 MESSAGE")
            print("=" * 80)
            print(message_content)
            print()

        if stream and self.is_streamable:
            print("=" * 80)
            print("🤖 RESPONSE")
            print("=" * 80)

            _response_content = ""
            _reasoning_content = ""
            _reasoning_displayed = False
            _final_content_printed = False

            run_generator = self.run_stream(
                message=message,
                messages=messages,
                stream_intermediate_steps=show_tool_calls,
                **kwargs,
            )

            async for run_response in run_generator:
                event = getattr(run_response, 'event', '')

                if show_tool_calls and event == RunEvent.tool_call_started.value:
                    tool_info = run_response.tools[-1] if run_response.tools else None
                    if tool_info:
                        tool_name = tool_info.get("tool_name", "unknown")
                        tool_args = tool_info.get("tool_args", {})
                        display_args = {}
                        for k, v in tool_args.items():
                            if isinstance(v, str) and len(v) > 100:
                                display_args[k] = v[:100] + "..."
                            else:
                                display_args[k] = v
                        print(f"\n  🔧 {tool_name}({display_args})", flush=True)
                    continue

                if show_tool_calls and event == RunEvent.tool_call_completed.value:
                    tool_info = run_response.tools[-1] if run_response.tools else None
                    if tool_info:
                        tool_result = tool_info.get("content", "")
                        result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                        print(f"     📤 {result_preview}", flush=True)
                    continue


                if event in (RunEvent.run_started.value, RunEvent.run_completed.value,
                            RunEvent.updating_memory.value):
                    continue

                if (show_reasoning and hasattr(run_response, 'reasoning_content') and
                        run_response.reasoning_content):
                    if not _reasoning_displayed:
                        print("💭 THINKING")
                        print("-" * 40)
                        _reasoning_displayed = True
                    if run_response.reasoning_content != _reasoning_content:
                        print(run_response.reasoning_content, end='', flush=True)
                        _reasoning_content = run_response.reasoning_content

                if run_response.content and run_response.content != _response_content:
                    if _reasoning_displayed and _response_content == "":
                        print()
                        print("-" * 40)
                        print("💬 ANSWER")
                        print("-" * 40)
                    print(run_response.content, end='', flush=True)
                    _response_content = run_response.content
        else:
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

            if show_tool_calls and run_response.tools:
                print()
                for tool in run_response.tools:
                    tool_name = tool.get("tool_name", "unknown")
                    tool_args = tool.get("tool_args", {})
                    display_args = {}
                    for k, v in tool_args.items():
                        if isinstance(v, str) and len(v) > 100:
                            display_args[k] = v[:100] + "..."
                        else:
                            display_args[k] = v
                    print(f"  🔧 {tool_name}({display_args})")
                    tool_result = tool.get("content", "")
                    result_preview = (
                        str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                    )
                    print(f"     📤 {result_preview}")

            if has_reasoning or (show_tool_calls and run_response.tools):
                print()
                print("-" * 40)
                print("💬 ANSWER")
                print("-" * 40)

            if run_response.content:
                print(run_response.content)

    def print_response_sync(
            self: "Agent",
            message: Optional[Union[List, Dict, str, Message]] = None,
            *,
            messages: Optional[List[Union[Dict, Message]]] = None,
            stream: bool = False,
            show_message: bool = True,
            show_reasoning: bool = True,
            show_tool_calls: bool = False,
            show_intermediate_steps: bool = True,
            **kwargs: Any,
    ) -> None:
        """Synchronous wrapper for print_response()."""
        run_sync(self.print_response(
            message=message,
            messages=messages,
            stream=stream,
            show_message=show_message,
            show_reasoning=show_reasoning,
            show_tool_calls=show_tool_calls,
            show_intermediate_steps=show_intermediate_steps,
            **kwargs,
        ))

    def cli_app(
            self: "Agent",
            message: Optional[str] = None,
            user: str = "User",
            emoji: str = "😎",
            stream: bool = False,
            exit_on: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        """Command line interface for the Agent."""
        if message:
            self.print_response_sync(message=message, stream=stream, **kwargs)

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            try:
                message = input(f"{emoji} {user}: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if message.strip() in _exit_on:
                print("Goodbye!")
                break

            self.print_response_sync(message=message, stream=stream, **kwargs)
