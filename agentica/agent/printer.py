# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Print response methods for Agent

This module contains methods for printing agent responses to console,
supporting both streaming and non-streaming modes.
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

from agentica.utils.log import logger
from agentica.model.message import Message
from agentica.run_response import RunEvent
from agentica.utils.message import get_text_from_message

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class PrinterMixin:
    """Mixin class containing print response methods for Agent."""

    def print_response(
            self: "Agent",
            message: Optional[Union[List, Dict, str, Message]] = None,
            *,
            messages: Optional[List[Union[Dict, Message]]] = None,
            stream: bool = False,
            show_message: bool = True,
            show_reasoning: bool = True,
            show_intermediate_steps: bool = True,
            **kwargs: Any,
    ) -> None:
        """Print the response from the Agent.
        
        Args:
            message: The message to send to the agent.
            messages: List of messages to send.
            stream: Whether to stream the response.
            show_message: Whether to show the input message.
            show_reasoning: Whether to show reasoning content (for thinking models).
            show_intermediate_steps: Whether to show intermediate steps for multi-round.
        """
        if self.response_model is not None:
            stream = False

        # Show message
        if show_message and message is not None:
            message_content = get_text_from_message(message)
            print("=" * 80)
            print("📝 MESSAGE")
            print("=" * 80)
            print(message_content)
            print()

        # Handle streaming response
        if stream and self.is_streamable:
            print("=" * 80)
            print("🤖 RESPONSE")
            print("=" * 80)

            _response_content = ""
            _reasoning_content = ""
            _reasoning_displayed = False
            _final_content_printed = False

            run_generator = self._run(message=message, messages=messages, stream=True, **kwargs)

            for run_response in run_generator:
                event = getattr(run_response, 'event', '')

                # Skip multi-round intermediate events in streaming mode
                if event in (RunEvent.multi_round_tool_call.value, 
                            RunEvent.multi_round_tool_result.value,
                            RunEvent.multi_round_completed.value):
                    continue

                # For multi-round, only print content from multi_round_turn event (final turn)
                if self.enable_multi_round:
                    if event == RunEvent.multi_round_turn.value:
                        # Stream reasoning content
                        if (show_reasoning and hasattr(run_response, 'reasoning_content') and
                                run_response.reasoning_content):
                            if not _reasoning_displayed:
                                print("💭 THINKING")
                                print("-" * 40)
                                _reasoning_displayed = True
                            if run_response.reasoning_content != _reasoning_content:
                                print(run_response.reasoning_content, end='', flush=True)
                                _reasoning_content = run_response.reasoning_content

                        # Stream content
                        if run_response.content and run_response.content != _response_content:
                            if _reasoning_displayed and _response_content == "":
                                print()
                                print("-" * 40)
                                print("💬 ANSWER")
                                print("-" * 40)
                            print(run_response.content, end='', flush=True)
                            _response_content = run_response.content
                            _final_content_printed = True
                    # Skip final run_response if we already printed content
                    elif not event and _final_content_printed:
                        continue
                else:
                    # Regular streaming (non-multi-round)
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
            # Non-streaming response (includes multi-round)
            if self.enable_multi_round:
                print("=" * 80)
                print("🤖 MULTI-ROUND RESPONSE")
                print("=" * 80)

                final_response = None
                run_generator = self._run(message=message, messages=messages, stream=False, **kwargs)

                for run_response in run_generator:
                    event = getattr(run_response, 'event', '')

                    # Handle multi-round turn
                    if event == RunEvent.multi_round_turn.value and show_intermediate_steps:
                        # Get turn info from extra_data
                        turn_info = ""
                        if run_response.extra_data and run_response.extra_data.add_messages:
                            for msg in run_response.extra_data.add_messages:
                                if msg.role == "info":
                                    turn_info = msg.content or ""
                        print(f"\n{'─'*20} {turn_info} {'─'*20}")

                        # Show reasoning content for thinking models
                        if show_reasoning and run_response.reasoning_content:
                            reasoning_preview = run_response.reasoning_content
                            if len(reasoning_preview) > 500:
                                reasoning_preview = reasoning_preview[:500] + "..."
                            print(f"💭 Thinking: {reasoning_preview}")

                        # Show content if available
                        if run_response.content:
                            content_preview = run_response.content
                            if len(content_preview) > 300:
                                content_preview = content_preview[:300] + "..."
                            print(f"💬 Content: {content_preview}")

                    # Handle tool call
                    elif event == RunEvent.multi_round_tool_call.value and show_intermediate_steps:
                        print(f"  🔧 Tool: {run_response.content}")

                    # Handle tool result
                    elif event == RunEvent.multi_round_tool_result.value and show_intermediate_steps:
                        print(f"     📤 Result: {run_response.content}")

                    # Handle completion
                    elif event == RunEvent.multi_round_completed.value:
                        if show_intermediate_steps:
                            print(f"\n{'='*20} ✅ {run_response.content} {'='*20}")

                    # Store final response
                    final_response = run_response

                # Print final answer
                if final_response and final_response.content:
                    print("\n" + "=" * 80)
                    print("📋 FINAL ANSWER")
                    print("=" * 80)
                    print(final_response.content)
            else:
                # Regular non-streaming response
                run_response = self.run(message=message, messages=messages, stream=False, **kwargs)

                print("=" * 80)
                print("🤖 RESPONSE")
                print("=" * 80)

                if (show_reasoning and hasattr(run_response, 'reasoning_content') and
                        run_response.reasoning_content):
                    print("💭 THINKING")
                    print("-" * 40)
                    print(run_response.reasoning_content)
                    print()
                    print("-" * 40)
                    print("💬 ANSWER")
                    print("-" * 40)

                if run_response.content:
                    print(run_response.content)

    async def aprint_response(
            self: "Agent",
            message: Optional[Union[List, Dict, str, Message]] = None,
            *,
            messages: Optional[List[Union[Dict, Message]]] = None,
            stream: bool = False,
            show_message: bool = True,
            show_reasoning: bool = True,
            show_intermediate_steps: bool = True,
            **kwargs: Any,
    ) -> None:
        """Async print the response from the Agent.
        
        Args:
            message: The message to send to the agent.
            messages: List of messages to send.
            stream: Whether to stream the response.
            show_message: Whether to show the input message.
            show_reasoning: Whether to show reasoning content (for thinking models).
            show_intermediate_steps: Whether to show intermediate steps for multi-round.
        """
        if self.response_model is not None:
            stream = False

        # Show message
        if show_message and message is not None:
            message_content = get_text_from_message(message)
            print("=" * 80)
            print("📝 MESSAGE")
            print("=" * 80)
            print(message_content)
            print()

        # Handle streaming response
        if stream and self.is_streamable:
            print("=" * 80)
            print("🤖 RESPONSE")
            print("=" * 80)

            _response_content = ""
            _reasoning_content = ""
            _reasoning_displayed = False
            _final_content_printed = False

            arun_generator = self._arun(message=message, messages=messages, stream=True, **kwargs)

            async for run_response in arun_generator:
                event = getattr(run_response, 'event', '')

                # Skip multi-round intermediate events in streaming mode
                if event in (RunEvent.multi_round_tool_call.value, 
                            RunEvent.multi_round_tool_result.value,
                            RunEvent.multi_round_completed.value):
                    continue

                # For multi-round, only print content from multi_round_turn event
                if self.enable_multi_round:
                    if event == RunEvent.multi_round_turn.value:
                        # Stream reasoning content
                        if (show_reasoning and hasattr(run_response, 'reasoning_content') and
                                run_response.reasoning_content):
                            if not _reasoning_displayed:
                                print("💭 THINKING")
                                print("-" * 40)
                                _reasoning_displayed = True
                            if run_response.reasoning_content != _reasoning_content:
                                print(run_response.reasoning_content, end='', flush=True)
                                _reasoning_content = run_response.reasoning_content

                        # Stream content
                        if run_response.content and run_response.content != _response_content:
                            if _reasoning_displayed and _response_content == "":
                                print()
                                print("-" * 40)
                                print("💬 ANSWER")
                                print("-" * 40)
                            print(run_response.content, end='', flush=True)
                            _response_content = run_response.content
                            _final_content_printed = True
                    # Skip final run_response if we already printed content
                    elif not event and _final_content_printed:
                        continue
                else:
                    # Regular streaming (non-multi-round)
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
            # Non-streaming response (includes multi-round)
            if self.enable_multi_round:
                print("=" * 80)
                print("🤖 MULTI-ROUND RESPONSE")
                print("=" * 80)

                final_response = None
                arun_generator = self._arun(message=message, messages=messages, stream=False, **kwargs)

                async for run_response in arun_generator:
                    event = getattr(run_response, 'event', '')

                    # Handle multi-round turn
                    if event == RunEvent.multi_round_turn.value and show_intermediate_steps:
                        # Get turn info from extra_data
                        turn_info = ""
                        if run_response.extra_data and run_response.extra_data.add_messages:
                            for msg in run_response.extra_data.add_messages:
                                if msg.role == "info":
                                    turn_info = msg.content or ""
                        print(f"\n{'─'*20} {turn_info} {'─'*20}")

                        # Show reasoning content for thinking models
                        if show_reasoning and run_response.reasoning_content:
                            reasoning_preview = run_response.reasoning_content
                            if len(reasoning_preview) > 500:
                                reasoning_preview = reasoning_preview[:500] + "..."
                            print(f"💭 Thinking: {reasoning_preview}")

                        # Show content if available
                        if run_response.content:
                            content_preview = run_response.content
                            if len(content_preview) > 300:
                                content_preview = content_preview[:300] + "..."
                            print(f"💬 Content: {content_preview}")

                    # Handle tool call
                    elif event == RunEvent.multi_round_tool_call.value and show_intermediate_steps:
                        print(f"  🔧 Tool: {run_response.content}")

                    # Handle tool result
                    elif event == RunEvent.multi_round_tool_result.value and show_intermediate_steps:
                        print(f"     📤 Result: {run_response.content}")

                    # Handle completion
                    elif event == RunEvent.multi_round_completed.value:
                        if show_intermediate_steps:
                            print(f"\n{'='*20} ✅ {run_response.content} {'='*20}")

                    # Store final response
                    final_response = run_response

                # Print final answer
                if final_response and final_response.content:
                    print("\n" + "=" * 80)
                    print("📋 FINAL ANSWER")
                    print("=" * 80)
                    print(final_response.content)
            else:
                # Regular non-streaming response
                run_response = await self.arun(message=message, messages=messages, stream=False, **kwargs)

                print("=" * 80)
                print("🤖 RESPONSE")
                print("=" * 80)

                if (show_reasoning and hasattr(run_response, 'reasoning_content') and
                        run_response.reasoning_content):
                    print("💭 THINKING")
                    print("-" * 40)
                    print(run_response.reasoning_content)
                    print()
                    print("-" * 40)
                    print("💬 ANSWER")
                    print("-" * 40)

                if run_response.content:
                    print(run_response.content)

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
            self.print_response(message=message, stream=stream, **kwargs)

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

            self.print_response(message=message, stream=stream, **kwargs)
