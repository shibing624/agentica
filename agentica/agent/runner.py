# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Run execution methods for Agent

This module contains the core run/arun methods and multi-round execution logic.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    cast,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    overload,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from uuid import uuid4
from pydantic import BaseModel, ValidationError

from agentica.utils.log import logger
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunEvent, RunResponse, RunResponseExtraData
from agentica.memory import AgentRun
from agentica.tools.base import get_function_call_for_tool_call
from agentica.utils.string import parse_structured_output
from agentica.utils.langfuse_integration import langfuse_trace_context

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class RunnerMixin:
    """Mixin class containing run execution methods for Agent."""

    def save_run_response_to_file(self: "Agent", message: Optional[Union[str, List, Dict, Message]] = None) -> None:
        if self.save_response_to_file is not None and self.run_response is not None:
            message_str = None
            if message is not None:
                if isinstance(message, str):
                    message_str = message
                else:
                    logger.warning("Did not use message in output file name: message is not a string")
            try:
                fn = self.save_response_to_file.format(
                    name=self.name, session_id=self.session_id, user_id=self.user_id, message=message_str
                )
                fn_path = Path(fn)
                if not fn_path.parent.exists():
                    fn_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(self.run_response.content, str):
                    fn_path.write_text(self.run_response.content)
                else:
                    fn_path.write_text(json.dumps(self.run_response.content, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"Failed to save output to file: {e}")

    def _aggregate_metrics_from_run_messages(self: "Agent", messages: List[Message]) -> Dict[str, Any]:
        aggregated_metrics: Dict[str, Any] = defaultdict(list)

        # Use a defaultdict(list) to collect all values for each assistant message
        for m in messages:
            if m.role == "assistant" and m.metrics is not None:
                for k, v in m.metrics.items():
                    aggregated_metrics[k].append(v)
        return aggregated_metrics

    def generic_run_response(
            self: "Agent", content: Optional[str] = None, event: RunEvent = RunEvent.run_response
    ) -> RunResponse:
        return RunResponse(
            run_id=self.run_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            content=content,
            tools=self.run_response.tools,
            images=self.run_response.images,
            videos=self.run_response.videos,
            model=self.run_response.model,
            messages=self.run_response.messages,
            reasoning_content=self.run_response.reasoning_content,
            extra_data=self.run_response.extra_data,
            event=event.value,
        )

    def _run(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> Iterator[RunResponse]:
        """Run the Agent with optional multi-round strategy.

        All LLM calls within this run are grouped under a single Langfuse trace
        when Langfuse is configured. This enables proper tracking of multi-turn
        conversations and tool-calling sequences.
        """
        # Prepare input for Langfuse trace
        trace_input = message if isinstance(message, str) else str(message) if message else None

        # Get trace name: agent name or default
        trace_name = self.name or "agent-run"

        # Get tags from model if available
        langfuse_tags = None
        if self.model and hasattr(self.model, 'langfuse_tags'):
            langfuse_tags = self.model.langfuse_tags

        # Wrap the entire run in a Langfuse trace context
        # This groups all LLM calls (including tool-calling iterations) under one trace
        with langfuse_trace_context(
                name=trace_name,
                session_id=self.session_id,
                user_id=self.user_id,
                tags=langfuse_tags,
                input_data=trace_input,
        ) as trace:
            final_response = None

            if self.enable_multi_round:
                for response in self._run_multi_round(
                        message=message,
                        stream=stream,
                        audio=audio,
                        images=images,
                        videos=videos,
                        messages=messages,
                        stream_intermediate_steps=stream_intermediate_steps,
                        **kwargs
                ):
                    final_response = response
                    yield response
            else:
                for response in self._run_single_round(
                        message=message,
                        stream=stream,
                        audio=audio,
                        images=images,
                        videos=videos,
                        messages=messages,
                        stream_intermediate_steps=stream_intermediate_steps,
                        **kwargs
                ):
                    final_response = response
                    yield response

            # Set output on trace before context exits
            if final_response:
                output_content = final_response.content
                if isinstance(output_content, BaseModel):
                    output_content = output_content.model_dump()
                trace.set_output(output_content)
                trace.set_metadata("run_id", final_response.run_id)
                trace.set_metadata("model", final_response.model)

    def _run_single_round(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> Iterator[RunResponse]:
        """Run the Agent with a message and return the response.

        Steps:
        1. Setup: Update the model class and resolve context
        2. Read existing session from storage
        3. Prepare messages for this run
        4. Reason about the task if reasoning is enabled
        5. Generate a response from the Model (includes running function calls)
        6. Update Memory
        7. Save session to storage
        8. Save output to file if save_response_to_file is set
        9. Set the run_input
        """
        # Check if streaming is enabled
        self.stream = stream and self.is_streamable
        # Check if streaming intermediate steps is enabled
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        # Create the run_response object
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Setup: Update the model class and resolve context
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # 2. Read existing session from storage
        self.read_from_storage()

        # 3. Prepare messages for this run
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )

        # Get the number of messages in messages_for_model that form the input for this run
        # We track these to skip when updating memory
        num_input_messages = len(messages_for_model)

        # Yield a RunStarted event
        if self.stream_intermediate_steps:
            yield self.generic_run_response("Run started", RunEvent.run_started)

        # 5. Generate a response from the Model (includes running function calls)
        model_response: ModelResponse
        self.model = cast(Model, self.model)
        if self.stream:
            model_response = ModelResponse(content="", reasoning_content="")
            for model_response_chunk in self.model.response_stream(messages=messages_for_model):
                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                    if model_response_chunk.reasoning_content is not None:
                        # Accumulate reasoning content instead of overwriting
                        if model_response.reasoning_content is None:
                            model_response.reasoning_content = ""
                        model_response.reasoning_content += model_response_chunk.reasoning_content
                        # For streaming, yield only the new chunk, not the accumulated content
                        # Clear content to avoid mixing with reasoning_content in the same yield
                        self.run_response.content = None
                        self.run_response.reasoning_content = model_response_chunk.reasoning_content
                        self.run_response.created_at = model_response_chunk.created_at
                        yield self.run_response
                    if model_response_chunk.content and model_response.content is not None:
                        model_response.content += model_response_chunk.content
                        # Clear reasoning_content to avoid mixing with content in the same yield
                        self.run_response.reasoning_content = None
                        self.run_response.content = model_response_chunk.content
                        self.run_response.created_at = model_response_chunk.created_at
                        yield self.run_response
                elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                    # Add tool call to the run_response
                    tool_call_dict = model_response_chunk.tool_call
                    if tool_call_dict is not None:
                        if self.run_response.tools is None:
                            self.run_response.tools = []
                        self.run_response.tools.append(tool_call_dict)
                    if self.stream_intermediate_steps:
                        yield self.generic_run_response(
                            content=model_response_chunk.content,
                            event=RunEvent.tool_call_started,
                        )
                elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                    # Update the existing tool call in the run_response
                    tool_call_dict = model_response_chunk.tool_call
                    if tool_call_dict is not None and self.run_response.tools:
                        tool_call_id_to_update = tool_call_dict["tool_call_id"]
                        # Use a dictionary comprehension to create a mapping of tool_call_id to index
                        tool_call_index_map = {tc["tool_call_id"]: i for i, tc in enumerate(self.run_response.tools)}
                        # Update the tool call if it exists
                        if tool_call_id_to_update in tool_call_index_map:
                            self.run_response.tools[tool_call_index_map[tool_call_id_to_update]] = tool_call_dict
                    if self.stream_intermediate_steps:
                        yield self.generic_run_response(
                            content=model_response_chunk.content,
                            event=RunEvent.tool_call_completed,
                        )
        else:
            model_response = self.model.response(messages=messages_for_model)
            # Handle structured outputs
            if self.response_model is not None and self.structured_outputs and model_response.parsed is not None:
                self.run_response.content = model_response.parsed
                self.run_response.content_type = self.response_model.__name__
            else:
                self.run_response.content = model_response.content
            if model_response.audio is not None:
                self.run_response.audio = model_response.audio
            if model_response.reasoning_content is not None:
                self.run_response.reasoning_content = model_response.reasoning_content
            self.run_response.messages = messages_for_model
            self.run_response.created_at = model_response.created_at

            # Extract tool call info from messages for non-streaming mode
            tool_calls_data = []
            for msg in messages_for_model:
                m = msg if isinstance(msg, Message) else None
                if m is None:
                    continue
                if m.role == "tool" and m.tool_name:
                    tool_calls_data.append({
                        "tool_call_id": m.tool_call_id,
                        "tool_name": m.tool_name,
                        "tool_args": m.tool_args,
                        "content": m.content,
                        "tool_call_error": getattr(m, 'tool_call_error', False),
                        "metrics": m.metrics if hasattr(m, 'metrics') else {},
                    })
            if tool_calls_data:
                self.run_response.tools = tool_calls_data

        # Build a list of messages that belong to this particular run
        run_messages = user_messages + messages_for_model[num_input_messages:]
        if system_message is not None:
            run_messages.insert(0, system_message)
        # Update the run_response
        self.run_response.messages = run_messages
        self.run_response.metrics = self._aggregate_metrics_from_run_messages(run_messages)
        # Update the run_response content if streaming as run_response will only contain the last chunk
        if self.stream:
            self.run_response.content = model_response.content
            if model_response.reasoning_content:
                self.run_response.reasoning_content = model_response.reasoning_content
            # Also update the reasoning_content with the complete accumulated content
            if hasattr(model_response, 'reasoning_content') and model_response.reasoning_content:
                self.run_response.reasoning_content = model_response.reasoning_content

        # 6. Update Memory
        if self.stream_intermediate_steps:
            yield self.generic_run_response(
                content="Updating memory",
                event=RunEvent.updating_memory,
            )

        # Add the system message to the memory
        if system_message is not None:
            self.memory.add_system_message(system_message, system_message_role=self.system_message_role)
        # Add the user messages and model response messages to memory
        self.memory.add_messages(messages=(user_messages + messages_for_model[num_input_messages:]))

        # Create an AgentRun object to add to memory
        agent_run = AgentRun(response=self.run_response)
        if message is not None:
            user_message_for_memory: Optional[Message] = None
            if isinstance(message, str):
                user_message_for_memory = Message(role=self.user_message_role, content=message)
            elif isinstance(message, Message):
                user_message_for_memory = message
            if user_message_for_memory is not None:
                agent_run.message = user_message_for_memory
                # Update the memories with the user message if needed
                if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                    self.memory.update_memory(input=user_message_for_memory.get_content_string())
        elif messages is not None and len(messages) > 0:
            for _m in messages:
                _um = None
                if isinstance(_m, Message):
                    _um = _m
                elif isinstance(_m, dict):
                    try:
                        _um = Message.model_validate(_m)
                    except Exception as e:
                        logger.warning(f"Failed to validate message: {e}")
                else:
                    logger.warning(f"Unsupported message type: {type(_m)}")
                    continue
                if _um:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(_um)
                    if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        self.memory.update_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        # Add AgentRun to memory
        self.memory.add_run(agent_run)

        # Update the session summary if needed
        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            self.memory.update_summary()

        # 7. Save session to storage
        self.write_to_storage()

        # 8. Save output to file if save_response_to_file is set
        self.save_run_response_to_file(message=message)

        # 9. Set the run_input
        if message is not None:
            if isinstance(message, str):
                self.run_input = message
            elif isinstance(message, Message):
                self.run_input = message.to_dict()
            else:
                self.run_input = message
        elif messages is not None:
            self.run_input = [m.to_dict() if isinstance(m, Message) else m for m in messages]

        if self.stream_intermediate_steps:
            yield self.generic_run_response(
                content=self.run_response.content,
                event=RunEvent.run_completed,
            )

        # -*- Yield final response if not streaming so that run() can get the response
        if not self.stream:
            yield self.run_response

    # =============================================================================
    # Multi-round Hook Methods (can be overridden by subclasses like DeepAgent)
    # =============================================================================

    def _on_pre_step(
            self: "Agent",
            step: int,
            messages: List[Message]
    ) -> Tuple[bool, Optional[str]]:
        """
        Hook called before each step in multi-round execution.
        
        Override this method in subclasses to implement custom pre-step logic
        such as context overflow handling.
        
        Args:
            step: Current step number
            messages: Current message list
            
        Returns:
            (should_force_answer, optional_warning_message)
            - should_force_answer: If True, stop execution and force final answer
            - optional_warning_message: Message to inject into context (e.g., warnings)
        """
        return False, None

    def _on_tool_call(self: "Agent", tool_name: str, step: int) -> Optional[str]:
        """
        Hook called when a tool is about to be executed.
        
        Override this method in subclasses to implement custom tool call logic
        such as repetitive behavior detection.
        
        Args:
            tool_name: Name of the tool being called
            step: Current step number
            
        Returns:
            Optional warning message to inject into context
        """
        return None

    def _on_post_step(self: "Agent", step: int, messages: List[Message]) -> None:
        """
        Hook called after each step in multi-round execution.
        
        Override this method in subclasses to implement custom post-step logic
        such as reflection prompts.
        
        Args:
            step: Current step number
            messages: Current message list (can be modified in place)
        """
        pass

    def _run_multi_round(
            self: "Agent",
            message,
            stream,
            audio,
            images,
            videos,
            messages,
            stream_intermediate_steps,
            **kwargs
    ):
        """Run the Agent with a multi-round strategy for better search accuracy.

        This method implements a multi-round conversation strategy similar to DeepSeekAgent,
        where the agent loops until no more tool calls are needed.
        Key changes from previous implementation:
        - Uses tool_calls presence as loop condition (not <answer> tags)
        - Supports multiple tool calls per round
        - Preserves reasoning_content for context continuity
        - No artificial guidance prompts
        """
        # Initialize basic settings
        self.stream = stream and self.is_streamable
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Setup: Update model and resolve context
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # Disable model's internal tool execution - we handle it manually
        self.model = cast(Model, self.model)
        original_run_tools = self.model.run_tools
        self.model.run_tools = False

        # 2. Read existing session from storage
        self.read_from_storage()

        # Add introduction if provided
        if self.introduction is not None:
            self.add_introduction(self.introduction)

        # 3. Prepare initial messages for this run
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )

        num_input_messages = len(messages_for_model)

        # Start multi-round execution event
        if self.stream_intermediate_steps:
            yield self.generic_run_response(event=RunEvent.run_started)

        # 4. Multi-round execution loop
        all_run_messages = []
        if system_message is not None:
            all_run_messages.append(system_message)
        all_run_messages.extend(user_messages)

        model_response = ModelResponse(content='')
        current_round = 0

        try:
            for current_round in range(1, self.max_rounds + 1):
                logger.debug(f"Turn {current_round}/{self.max_rounds}")

                # Hook: Pre-step processing (for DeepAgent context management)
                should_force_answer, warning_msg = self._on_pre_step(current_round, messages_for_model)
                if warning_msg:
                    messages_for_model.append(Message(role="system", content=warning_msg))
                if should_force_answer:
                    logger.warning(f"Force answer triggered at turn {current_round}")
                    break

                # Token limit check (legacy, kept for backward compatibility)
                total_content = " ".join([str(msg.content or "") for msg in messages_for_model])
                if len(total_content) > self.max_tokens * 3:
                    logger.warning(f"Token limit approaching, stopping at turn {current_round}")
                    break

                if self.stream_intermediate_steps:
                    yield self.generic_run_response(f"Turn {current_round}", RunEvent.run_response)

                # Call model
                model_response = self.model.response(messages=messages_for_model)

                # Get assistant message from messages_for_model (model.response appends it)
                assistant_message = messages_for_model[-1] if messages_for_model else None
                if assistant_message and assistant_message.role == "assistant":
                    # Ensure reasoning_content is preserved
                    if model_response.reasoning_content and not assistant_message.reasoning_content:
                        assistant_message.reasoning_content = model_response.reasoning_content
                    all_run_messages.append(assistant_message)

                # Update run_response
                if model_response.content:
                    self.run_response.content = model_response.content
                if model_response.reasoning_content:
                    self.run_response.reasoning_content = model_response.reasoning_content

                # Yield intermediate response for multi-round turn
                yield RunResponse(
                    content=model_response.content,
                    reasoning_content=model_response.reasoning_content,
                    event=RunEvent.multi_round_turn.value,
                    extra_data=RunResponseExtraData(
                        add_messages=[Message(role="info", content=f"Turn {current_round}/{self.max_rounds}")]
                    )
                )

                # Log response preview
                content_preview = (model_response.content or model_response.reasoning_content or "")[:500]
                if content_preview:
                    logger.debug(f"Turn {current_round} response: {content_preview}...")

                # Check for tool calls - handle multiple tool calls
                has_tool_calls = (
                    assistant_message and
                    assistant_message.tool_calls and
                    len(assistant_message.tool_calls) > 0
                )

                if has_tool_calls:
                    tool_results = []
                    for tool_call in assistant_message.tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        func_info = tool_call.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args_str = func_info.get("arguments", "{}")

                        logger.debug(f"Tool call: {func_name}({func_args_str})")

                        # Hook: Check for repetitive behavior (for DeepAgent)
                        repetition_warning = self._on_tool_call(func_name, current_round)
                        if repetition_warning:
                            messages_for_model.append(Message(role="system", content=repetition_warning))

                        # Yield tool call event
                        yield RunResponse(
                            content=f"{func_name}({func_args_str[:500]}{'...' if len(func_args_str) > 500 else ''})",
                            event=RunEvent.multi_round_tool_call.value
                        )

                        if self.stream_intermediate_steps:
                            yield self.generic_run_response(
                                f"Calling tool: {func_name}",
                                RunEvent.tool_call_started
                            )

                        # Execute tool
                        try:
                            function_call = get_function_call_for_tool_call(tool_call, self.model.functions)
                            if function_call is not None:
                                # Execute and get result (supports async tools in sync context)
                                function_call.execute()
                                result_str = str(function_call.result) if function_call.result is not None else ""

                                # Yield tool result event
                                result_preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                                yield RunResponse(
                                    content=f"{func_name}: {result_preview}",
                                    event=RunEvent.multi_round_tool_result.value
                                )

                                # Create tool message
                                tool_message = Message(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=result_str
                                )
                                tool_results.append(tool_message)

                                if self.stream_intermediate_steps:
                                    yield self.generic_run_response(
                                        f"Tool {func_name} completed",
                                        RunEvent.tool_call_completed
                                    )
                            else:
                                # Tool not found
                                error_msg = f"Tool {func_name} not found"
                                logger.warning(error_msg)
                                yield RunResponse(
                                    content=f"Error: {error_msg}",
                                    event=RunEvent.multi_round_tool_result.value
                                )
                                tool_message = Message(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=error_msg
                                )
                                tool_results.append(tool_message)
                        except Exception as e:
                            error_msg = f"Error executing tool {func_name}: {str(e)}"
                            logger.error(error_msg)
                            yield RunResponse(
                                content=f"Error: {error_msg}",
                                event=RunEvent.multi_round_tool_result.value
                            )
                            tool_message = Message(
                                role="tool",
                                tool_call_id=tool_call_id,
                                content=error_msg
                            )
                            tool_results.append(tool_message)

                    # Add all tool results to messages
                    for tool_msg in tool_results:
                        messages_for_model.append(tool_msg)
                        all_run_messages.append(tool_msg)

                    # Check if compression is needed
                    if self.compression_manager is not None:
                        if self.compression_manager.should_compress(
                            messages_for_model,
                            tools=self.model.functions if hasattr(self.model, 'functions') else None,
                            model=self.model,
                        ):
                            self.compression_manager.compress(messages_for_model)
                            logger.debug(f"Compressed tool results, stats: {self.compression_manager.get_stats()}")

                    # Hook: Post-step processing (for DeepAgent reflection)
                    self._on_post_step(current_round, messages_for_model)
                else:
                    # No tool calls - task completed
                    logger.debug("No tool calls, task completed")
                    yield RunResponse(
                        content=f"Task completed in {current_round} turns",
                        event=RunEvent.multi_round_completed.value
                    )
                    break

        finally:
            # Restore original run_tools setting
            self.model.run_tools = original_run_tools

        # 5. Finalize response
        if model_response.content:
            self.run_response.content = model_response.content
        if model_response.audio is not None:
            self.run_response.audio = model_response.audio
        if model_response.reasoning_content:
            self.run_response.reasoning_content = model_response.reasoning_content

        self.run_response.messages = all_run_messages
        self.run_response.metrics = self._aggregate_metrics_from_run_messages(all_run_messages)
        self.run_response.created_at = getattr(model_response, 'created_at', None)

        # 6. Update Memory
        if self.stream_intermediate_steps:
            yield self.generic_run_response(content="Updating memory", event=RunEvent.updating_memory)

        if system_message is not None:
            self.memory.add_system_message(system_message, system_message_role=self.system_message_role)
        self.memory.add_messages(messages=(user_messages + messages_for_model[num_input_messages:]))

        # Create AgentRun
        agent_run = AgentRun(response=self.run_response)
        if message is not None:
            user_message_for_memory: Optional[Message] = None
            if isinstance(message, str):
                user_message_for_memory = Message(role=self.user_message_role, content=message)
            elif isinstance(message, Message):
                user_message_for_memory = message
            if user_message_for_memory is not None:
                agent_run.message = user_message_for_memory
                if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                    self.memory.update_memory(input=user_message_for_memory.get_content_string())
        elif messages is not None and len(messages) > 0:
            for _m in messages:
                _um = None
                if isinstance(_m, Message):
                    _um = _m
                elif isinstance(_m, dict):
                    try:
                        _um = Message.model_validate(_m)
                    except Exception as e:
                        logger.warning(f"Failed to validate message: {e}")
                else:
                    logger.warning(f"Unsupported message type: {type(_m)}")
                    continue
                if _um:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(_um)
                    if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        self.memory.update_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        self.memory.add_run(agent_run)

        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            self.memory.update_summary()

        # 7. Save session to storage
        self.write_to_storage()

        # 8. Save output to file if configured
        self.save_run_response_to_file(message)

        # 9. Set run input
        if message is not None:
            self.run_input = message
        elif messages is not None:
            self.run_input = messages

        # Final completion event
        if self.stream_intermediate_steps:
            yield self.generic_run_response(
                f"Multi-round completed in {current_round} turns",
                RunEvent.run_completed
            )
        logger.debug(f"Multi-round completed in {current_round} turns")

        yield self.run_response

    def _wrap_stream_with_timeout(
            self: "Agent",
            stream_iter: Iterator[RunResponse],
    ) -> Iterator[RunResponse]:
        """Wrap a streaming iterator with timeout control.
        
        Implements two types of timeout:
        - first_token_timeout: Maximum time to wait for the first token
        - run_timeout: Maximum total time for the entire stream
        
        Uses a background thread to fetch items from the iterator with timeout.
        """
        import threading
        import queue
        import time
        
        result_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        error_holder = [None]  # Use list to allow modification in thread
        
        def producer():
            """Background thread that reads from the iterator."""
            try:
                for item in stream_iter:
                    if stop_event.is_set():
                        break
                    result_queue.put(("item", item))
                result_queue.put(("done", None))
            except Exception as e:
                error_holder[0] = e
                result_queue.put(("error", e))
        
        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        start_time = time.time()
        first_token_received = False
        
        try:
            while True:
                # Calculate timeout for this iteration
                if not first_token_received and self.first_token_timeout is not None:
                    # Waiting for first token
                    timeout = self.first_token_timeout
                elif self.run_timeout is not None:
                    # Check total elapsed time
                    elapsed = time.time() - start_time
                    remaining = self.run_timeout - elapsed
                    if remaining <= 0:
                        logger.warning(f"Stream run timed out after {self.run_timeout} seconds")
                        yield RunResponse(
                            run_id=str(uuid4()),
                            content=f"Stream run timed out after {self.run_timeout} seconds",
                            event="RunTimeout",
                        )
                        return
                    timeout = remaining
                else:
                    timeout = None  # No timeout
                
                try:
                    msg_type, data = result_queue.get(timeout=timeout)
                    
                    if msg_type == "done":
                        return
                    elif msg_type == "error":
                        raise data
                    elif msg_type == "item":
                        if not first_token_received:
                            first_token_received = True
                        yield data
                        
                except queue.Empty:
                    # Timeout occurred
                    if not first_token_received:
                        logger.warning(f"First token timed out after {self.first_token_timeout} seconds")
                        yield RunResponse(
                            run_id=str(uuid4()),
                            content=f"First token timed out after {self.first_token_timeout} seconds",
                            event="FirstTokenTimeout",
                        )
                    else:
                        logger.warning(f"Stream run timed out after {self.run_timeout} seconds")
                        yield RunResponse(
                            run_id=str(uuid4()),
                            content=f"Stream run timed out after {self.run_timeout} seconds",
                            event="RunTimeout",
                        )
                    return
        finally:
            stop_event.set()

    def _run_with_timeout(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent with timeout control (non-streaming only).
        
        Uses ThreadPoolExecutor to implement timeout for the entire run.
        If timeout is reached, returns a RunResponse indicating timeout.
        """
        import concurrent.futures
        
        def run_inner():
            # Run without timeout flag to avoid recursion
            if self.response_model is not None and self.parse_response:
                resp = self._run(
                    message=message,
                    stream=False,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                run_response = None
                for response in resp:
                    run_response = response
                
                if self.structured_outputs:
                    if isinstance(run_response.content, self.response_model):
                        return run_response
                
                if isinstance(run_response.content, str):
                    try:
                        structured_output = parse_structured_output(run_response.content, self.response_model)
                        if structured_output is not None:
                            run_response.content = structured_output
                            run_response.content_type = self.response_model.__name__
                    except Exception as e:
                        logger.warning(f"Failed to convert response to output model: {e}")
                return run_response
            else:
                resp = self._run(
                    message=message,
                    stream=False,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                final_response = None
                for response in resp:
                    final_response = response
                return final_response
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_inner)
            try:
                result = future.result(timeout=self.run_timeout)
                return result
            except concurrent.futures.TimeoutError:
                logger.warning(f"Agent run timed out after {self.run_timeout} seconds")
                # Return a timeout response
                timeout_response = RunResponse(
                    run_id=str(uuid4()),
                    content=f"Agent run timed out after {self.run_timeout} seconds",
                    event="RunTimeout",
                )
                return timeout_response

    @overload
    def run(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: Literal[False] = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            **kwargs: Any,
    ) -> RunResponse:
        ...

    @overload
    def run(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: Literal[True] = True,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> Iterator[RunResponse]:
        ...

    def run(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> Union[RunResponse, Iterator[RunResponse]]:
        """Run the Agent with a message and return the response.
        
        Timeout settings:
        - run_timeout: Maximum total execution time (in seconds). For non-streaming, aborts the run.
                       For streaming, wraps the iterator with timeout control.
        - first_token_timeout: Maximum time to wait for the first token (in seconds, streaming only).
        """
        # Handle timeout for non-streaming mode
        if self.run_timeout is not None and not stream:
            return self._run_with_timeout(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )

        # If a response_model is set, return the response as a structured output
        if self.response_model is not None and self.parse_response:
            # Set stream=False and run the agent
            logger.debug("Setting stream=False as response_model is set")
            # Consume entire generator to ensure Langfuse trace context cleanup runs
            resp = self._run(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )
            run_response: RunResponse = None
            for response in resp:
                run_response = response

            # If the model natively supports structured outputs, the content is already in the structured format
            if self.structured_outputs:
                # Do a final check confirming the content is in the response_model format
                if isinstance(run_response.content, self.response_model):
                    return run_response

            # Otherwise convert the response to the structured format
            if isinstance(run_response.content, str):
                try:
                    structured_output = parse_structured_output(run_response.content, self.response_model)

                    # Update RunResponse
                    if structured_output is not None:
                        run_response.content = structured_output
                        run_response.content_type = self.response_model.__name__
                        if self.run_response is not None:
                            self.run_response.content = structured_output
                            self.run_response.content_type = self.response_model.__name__
                    else:
                        logger.warning("Failed to convert response to response_model")
                except Exception as e:
                    logger.warning(f"Failed to convert response to output model: {e}")
            else:
                logger.warning("Something went wrong. Run response content is not a string")
            return run_response
        else:
            if stream and self.is_streamable:
                resp = self._run(
                    message=message,
                    stream=True,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                # Wrap with timeout if configured
                if self.run_timeout is not None or self.first_token_timeout is not None:
                    return self._wrap_stream_with_timeout(resp)
                return resp
            else:
                resp = self._run(
                    message=message,
                    stream=False,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                # Consume entire generator to ensure Langfuse trace context cleanup runs
                final_response = None
                for response in resp:
                    final_response = response
                return final_response

    async def _arun(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Async Run the Agent with optional multi-round strategy."""

        if self.enable_multi_round:
            async for response in self._arun_multi_round(
                    message,
                    stream=stream,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs
            ):
                yield response
        else:
            async for response in self._arun_single_round(
                    message,
                    stream=stream,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs
            ):
                yield response

    async def _arun_single_round(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Async Run the Agent with a message and return the response (single round)."""

        # Check if streaming is enabled
        self.stream = stream and self.is_streamable
        # Check if streaming intermediate steps is enabled
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        # Create the run_response object
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Update the Model (set defaults, add tools, etc.)
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # 2. Read existing session from storage
        self.read_from_storage()

        # Add introduction if provided
        if self.introduction is not None:
            self.add_introduction(self.introduction)

        # 3. Prepare messages for this run
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )

        # Get the number of messages in messages_for_model that form the input for this run
        # We track these to skip when updating memory
        num_input_messages = len(messages_for_model)

        # Yield a RunStarted event
        if self.stream_intermediate_steps:
            yield self.generic_run_response("Run started", RunEvent.run_started)

        # 5. Generate a response from the Model (includes running function calls)
        # Start memory classification in parallel for optimization
        memory_classification_tasks = []
        if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
            if message is not None:
                user_message_for_memory: Optional[Message] = None
                if isinstance(message, str):
                    user_message_for_memory = Message(role=self.user_message_role, content=message)
                elif isinstance(message, Message):
                    user_message_for_memory = message
                if user_message_for_memory is not None:
                    # Start memory classification in parallel with LLM response generation
                    memory_task = asyncio.create_task(
                        self.memory.ashould_update_memory(input=user_message_for_memory.get_content_string())
                    )
                    memory_classification_tasks.append((user_message_for_memory, memory_task))
            elif messages is not None and len(messages) > 0:
                for _m in messages:
                    _um = None
                    if isinstance(_m, Message):
                        _um = _m
                    elif isinstance(_m, dict):
                        try:
                            _um = Message(**_m)
                        except Exception as e:
                            logger.error(f"Error converting message to Message: {e}")
                    if _um:
                        memory_task = asyncio.create_task(
                            self.memory.ashould_update_memory(input=_um.get_content_string())
                        )
                        memory_classification_tasks.append((_um, memory_task))

        model_response: ModelResponse
        self.model = cast(Model, self.model)
        if stream and self.is_streamable:
            model_response = ModelResponse(content="", reasoning_content="")
            model_response_stream = self.model.aresponse_stream(messages=messages_for_model)
            async for model_response_chunk in model_response_stream:  # type: ignore
                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                    # Handle reasoning_content (for thinking models like DeepSeek-R1)
                    if model_response_chunk.reasoning_content is not None:
                        if model_response.reasoning_content is None:
                            model_response.reasoning_content = ""
                        model_response.reasoning_content += model_response_chunk.reasoning_content
                        yield RunResponse(
                            event=RunEvent.run_response,
                            reasoning_content=model_response_chunk.reasoning_content,
                            run_id=self.run_id,
                            session_id=self.session_id,
                            agent_id=self.agent_id
                        )
                    # Handle content
                    if model_response_chunk.content is not None and model_response.content is not None:
                        model_response.content += model_response_chunk.content
                        yield RunResponse(
                            event=RunEvent.run_response,
                            content=model_response_chunk.content,
                            run_id=self.run_id,
                            session_id=self.session_id,
                            agent_id=self.agent_id
                        )
                elif model_response_chunk.event == ModelResponseEvent.tool_call_started.value:
                    # Add tool call to the run_response
                    tool_call_dict = model_response_chunk.tool_call
                    if tool_call_dict is not None:
                        if self.run_response.tools is None:
                            self.run_response.tools = []
                        self.run_response.tools.append(tool_call_dict)
                    if self.stream_intermediate_steps:
                        yield self.generic_run_response(
                            f"Running tool: {tool_call_dict.get('name') if tool_call_dict else 'Unknown'}",
                            RunEvent.tool_call_started,
                        )
                elif model_response_chunk.event == ModelResponseEvent.tool_call_completed.value:
                    # Update the existing tool call in the run_response
                    tool_call_dict = model_response_chunk.tool_call
                    if tool_call_dict is not None and self.run_response.tools:
                        for tool_call in self.run_response.tools:
                            if tool_call.get("id") == tool_call_dict.get("id"):
                                tool_call.update(tool_call_dict)
                                break
                    if self.stream_intermediate_steps:
                        yield self.generic_run_response(
                            f"Tool completed: {tool_call_dict.get('name') if tool_call_dict else 'Unknown'}",
                            RunEvent.tool_call_completed,
                        )
        else:
            model_response = await self.model.aresponse(messages=messages_for_model)
            # Handle structured outputs
            if self.response_model is not None and self.structured_outputs and model_response.parsed is not None:
                self.run_response.content = model_response.parsed
                self.run_response.content_type = self.response_model.__name__
            else:
                self.run_response.content = model_response.content
            self.run_response.messages = messages_for_model
            self.run_response.created_at = model_response.created_at

            # Extract tool call info from messages for non-streaming mode
            tool_calls_data = []
            for msg in messages_for_model:
                m = msg if isinstance(msg, Message) else None
                if m is None:
                    continue
                if m.role == "tool" and m.tool_name:
                    tool_calls_data.append({
                        "tool_call_id": m.tool_call_id,
                        "tool_name": m.tool_name,
                        "tool_args": m.tool_args,
                        "content": m.content,
                        "tool_call_error": getattr(m, 'tool_call_error', False),
                        "metrics": m.metrics if hasattr(m, 'metrics') else {},
                    })
            if tool_calls_data:
                self.run_response.tools = tool_calls_data

        # Build a list of messages that belong to this particular run
        run_messages = user_messages + messages_for_model[num_input_messages:]
        if system_message is not None:
            run_messages.insert(0, system_message)
        # Update the run_response
        self.run_response.messages = run_messages
        self.run_response.metrics = self._aggregate_metrics_from_run_messages(run_messages)
        # Update the run_response content if streaming as run_response will only contain the last chunk
        if self.stream:
            self.run_response.content = model_response.content
            if model_response.reasoning_content:
                self.run_response.reasoning_content = model_response.reasoning_content

        # 6. Update Memory
        if self.stream_intermediate_steps:
            yield self.generic_run_response("Updating memory", RunEvent.updating_memory)

        # Add the system message to the memory
        if system_message is not None:
            self.memory.add_system_message(system_message, system_message_role=self.system_message_role)
        # Add the user messages and model response messages to memory
        self.memory.add_messages(messages=(user_messages + messages_for_model[num_input_messages:]))

        # Create an AgentRun object to add to memory
        agent_run = AgentRun(response=self.run_response)

        # Process memory classification results that were started in parallel
        if memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
            for user_message, memory_task in memory_classification_tasks:
                try:
                    # Wait for the memory classification result
                    should_update_memory = await memory_task
                    if should_update_memory:
                        await self.memory.aupdate_memory(input=user_message.get_content_string())
                except Exception as e:
                    logger.warning(f"Error in memory processing: {e}")
                    # Fallback to original method
                    await self.memory.aupdate_memory(input=user_message.get_content_string())

        # Handle agent_run message assignment for non-parallel case or fallback
        if message is not None:
            user_message_for_memory: Optional[Message] = None
            if isinstance(message, str):
                user_message_for_memory = Message(role=self.user_message_role, content=message)
            elif isinstance(message, Message):
                user_message_for_memory = message
            if user_message_for_memory is not None:
                agent_run.message = user_message_for_memory
                # If no parallel processing was done, use original method
                if not memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                    await self.memory.aupdate_memory(input=user_message_for_memory.get_content_string())
        elif messages is not None and len(messages) > 0:
            for _m in messages:
                _um = None
                if isinstance(_m, Message):
                    _um = _m
                elif isinstance(_m, dict):
                    try:
                        _um = Message(**_m)
                    except Exception as e:
                        logger.error(f"Error converting message to Message: {e}")
                else:
                    logger.warning(f"Unsupported message type: {type(_m)}")
                    continue
                if _um:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(_um)
                    # If no parallel processing was done, use original method
                    if not memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        await self.memory.aupdate_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        # Add AgentRun to memory
        self.memory.add_run(agent_run)

        # Update the session summary if needed
        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            await self.memory.aupdate_summary()

        # 7. Save session to storage
        self.write_to_storage()

        # 8. Save output to file if save_response_to_file is set
        self.save_run_response_to_file(message=message)

        # 9. Set the run_input
        if message is not None:
            if isinstance(message, str):
                self.run_input = message
            elif isinstance(message, Message):
                self.run_input = message.to_dict()
            else:
                self.run_input = message
        elif messages is not None:
            self.run_input = [m.to_dict() if isinstance(m, Message) else m for m in messages]

        if self.stream_intermediate_steps:
            yield self.generic_run_response(self.run_response.content, RunEvent.run_completed)

        # -*- Yield final response if not streaming so that run() can get the response
        if not self.stream:
            yield self.run_response

    async def _arun_multi_round(
            self: "Agent",
            message,
            stream,
            audio,
            images,
            videos,
            messages,
            stream_intermediate_steps,
            **kwargs
    ):
        """Async Run the Agent with a multi-round strategy for better search accuracy.

        This method implements a multi-round conversation strategy similar to DeepSeekAgent,
        where the agent loops until no more tool calls are needed.
        """
        # Initialize basic settings
        self.stream = stream and self.is_streamable
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Setup: Update model and resolve context
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # Disable model's internal tool execution - we handle it manually
        self.model = cast(Model, self.model)
        original_run_tools = self.model.run_tools
        self.model.run_tools = False

        # 2. Read existing session from storage
        self.read_from_storage()

        # Add introduction if provided
        if self.introduction is not None:
            self.add_introduction(self.introduction)

        # 3. Prepare initial messages for this run
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )

        num_input_messages = len(messages_for_model)

        # Start multi-round execution event
        if self.stream_intermediate_steps:
            yield self.generic_run_response(event=RunEvent.run_started)

        # 4. Multi-round execution loop
        all_run_messages = []
        if system_message is not None:
            all_run_messages.append(system_message)
        all_run_messages.extend(user_messages)

        model_response = ModelResponse(content='')
        current_round = 0

        try:
            for current_round in range(1, self.max_rounds + 1):
                logger.debug(f"Turn {current_round}/{self.max_rounds}")

                # Token limit check
                total_content = " ".join([str(msg.content or "") for msg in messages_for_model])
                if len(total_content) > self.max_tokens * 3:
                    logger.warning(f"Token limit approaching, stopping at turn {current_round}")
                    break

                if self.stream_intermediate_steps:
                    yield self.generic_run_response(f"Turn {current_round}", RunEvent.run_response)

                # Call model
                model_response = await self.model.aresponse(messages=messages_for_model)

                # Get assistant message from messages_for_model (model.aresponse appends it)
                assistant_message = messages_for_model[-1] if messages_for_model else None
                if assistant_message and assistant_message.role == "assistant":
                    # Ensure reasoning_content is preserved
                    if model_response.reasoning_content and not assistant_message.reasoning_content:
                        assistant_message.reasoning_content = model_response.reasoning_content
                    all_run_messages.append(assistant_message)

                # Update run_response
                if model_response.content:
                    self.run_response.content = model_response.content
                if model_response.reasoning_content:
                    self.run_response.reasoning_content = model_response.reasoning_content

                # Yield intermediate response for multi-round turn
                yield RunResponse(
                    content=model_response.content,
                    reasoning_content=model_response.reasoning_content,
                    event=RunEvent.multi_round_turn.value,
                    extra_data=RunResponseExtraData(
                        add_messages=[Message(role="info", content=f"Turn {current_round}/{self.max_rounds}")]
                    )
                )

                # Log response preview
                content_preview = (model_response.content or "")[:200]
                logger.debug(f"Turn {current_round} response: {content_preview}...")

                # Check for tool calls - handle multiple tool calls
                has_tool_calls = (
                    assistant_message and
                    assistant_message.tool_calls and
                    len(assistant_message.tool_calls) > 0
                )

                if has_tool_calls:
                    tool_results = []
                    for tool_call in assistant_message.tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        func_info = tool_call.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args_str = func_info.get("arguments", "{}")

                        logger.debug(f"Tool call: {func_name}({func_args_str})")

                        # Hook: Check for repetitive behavior (for DeepAgent)
                        repetition_warning = self._on_tool_call(func_name, current_round)
                        if repetition_warning:
                            messages_for_model.append(Message(role="system", content=repetition_warning))

                        # Yield tool call event
                        yield RunResponse(
                            content=f"{func_name}({func_args_str[:100]}{'...' if len(func_args_str) > 100 else ''})",
                            event=RunEvent.multi_round_tool_call.value
                        )

                        if self.stream_intermediate_steps:
                            yield self.generic_run_response(
                                f"Calling tool: {func_name}",
                                RunEvent.tool_call_started
                            )

                        # Execute tool
                        try:
                            function_call = get_function_call_for_tool_call(tool_call, self.model.functions)
                            if function_call is not None:
                                # Execute and get result (async if available)
                                if hasattr(function_call, 'aexecute'):
                                    await function_call.aexecute()
                                else:
                                    function_call.execute()
                                result_str = str(function_call.result) if function_call.result is not None else ""

                                # Yield tool result event
                                result_preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                                yield RunResponse(
                                    content=f"{func_name}: {result_preview}",
                                    event=RunEvent.multi_round_tool_result.value
                                )

                                # Create tool message
                                tool_message = Message(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=result_str
                                )
                                tool_results.append(tool_message)

                                if self.stream_intermediate_steps:
                                    yield self.generic_run_response(
                                        f"Tool {func_name} completed",
                                        RunEvent.tool_call_completed
                                    )
                            else:
                                # Tool not found
                                error_msg = f"Tool {func_name} not found"
                                logger.warning(error_msg)
                                yield RunResponse(
                                    content=f"Error: {error_msg}",
                                    event=RunEvent.multi_round_tool_result.value
                                )
                                tool_message = Message(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=error_msg
                                )
                                tool_results.append(tool_message)
                        except Exception as e:
                            error_msg = f"Error executing tool {func_name}: {str(e)}"
                            logger.error(error_msg)
                            yield RunResponse(
                                content=f"Error: {error_msg}",
                                event=RunEvent.multi_round_tool_result.value
                            )
                            tool_message = Message(
                                role="tool",
                                tool_call_id=tool_call_id,
                                content=error_msg
                            )
                            tool_results.append(tool_message)

                    # Add all tool results to messages
                    for tool_msg in tool_results:
                        messages_for_model.append(tool_msg)
                        all_run_messages.append(tool_msg)

                    # Check if compression is needed
                    if self.compression_manager is not None:
                        if self.compression_manager.should_compress(
                            messages_for_model,
                            tools=self.model.functions if hasattr(self.model, 'functions') else None,
                            model=self.model,
                        ):
                            self.compression_manager.compress(messages_for_model)
                            logger.debug(f"Compressed tool results, stats: {self.compression_manager.get_stats()}")

                    # Hook: Post-step processing (for DeepAgent reflection)
                    self._on_post_step(current_round, messages_for_model)
                else:
                    # No tool calls - task completed
                    logger.debug("No tool calls, task completed")
                    yield RunResponse(
                        content=f"Task completed in {current_round} turns",
                        event=RunEvent.multi_round_completed.value
                    )
                    break

        finally:
            # Restore original run_tools setting
            self.model.run_tools = original_run_tools

        # 5. Finalize response
        if model_response.content:
            self.run_response.content = model_response.content
        if model_response.audio is not None:
            self.run_response.audio = model_response.audio
        if model_response.reasoning_content:
            self.run_response.reasoning_content = model_response.reasoning_content

        self.run_response.messages = all_run_messages
        self.run_response.metrics = self._aggregate_metrics_from_run_messages(all_run_messages)
        self.run_response.created_at = getattr(model_response, 'created_at', None)

        # 6. Update Memory
        if self.stream_intermediate_steps:
            yield self.generic_run_response(content="Updating memory", event=RunEvent.updating_memory)

        if system_message is not None:
            self.memory.add_system_message(system_message, system_message_role=self.system_message_role)
        self.memory.add_messages(messages=(user_messages + messages_for_model[num_input_messages:]))

        # Create AgentRun
        agent_run = AgentRun(response=self.run_response)
        if message is not None:
            user_message_for_memory: Optional[Message] = None
            if isinstance(message, str):
                user_message_for_memory = Message(role=self.user_message_role, content=message)
            elif isinstance(message, Message):
                user_message_for_memory = message
            if user_message_for_memory is not None:
                agent_run.message = user_message_for_memory
                if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                    await self.memory.aupdate_memory(input=user_message_for_memory.get_content_string())
        elif messages is not None and len(messages) > 0:
            for _m in messages:
                _um = None
                if isinstance(_m, Message):
                    _um = _m
                elif isinstance(_m, dict):
                    try:
                        _um = Message.model_validate(_m)
                    except Exception as e:
                        logger.warning(f"Failed to validate message: {e}")
                else:
                    logger.warning(f"Unsupported message type: {type(_m)}")
                    continue
                if _um:
                    if agent_run.messages is None:
                        agent_run.messages = []
                    agent_run.messages.append(_um)
                    if self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        await self.memory.aupdate_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        self.memory.add_run(agent_run)

        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            await self.memory.aupdate_summary()

        # 7. Save session to storage
        self.write_to_storage()

        # 8. Save output to file if configured
        self.save_run_response_to_file(message)

        # 9. Set run input
        if message is not None:
            self.run_input = message
        elif messages is not None:
            self.run_input = messages

        # Final completion event
        if self.stream_intermediate_steps:
            yield self.generic_run_response(
                f"Multi-round completed in {current_round} turns",
                RunEvent.run_completed
            )
        logger.debug(f"Multi-round completed in {current_round} turns")

        yield self.run_response

    @overload
    async def arun(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: Literal[False] = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            **kwargs: Any,
    ) -> RunResponse:
        ...

    @overload
    async def arun(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: Literal[True] = True,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        ...

    async def arun(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            stream: bool = False,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> Union[RunResponse, AsyncIterator[RunResponse]]:
        """Async Run the Agent with a message and return the response."""

        # If a response_model is set, return the response as a structured output
        if self.response_model is not None and self.parse_response:
            # Set stream=False and run the agent
            logger.debug("Setting stream=False as response_model is set")
            run_response = await self._arun(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            ).__anext__()

            # If the model natively supports structured outputs, the content is already in the structured format
            if self.structured_outputs:
                # Do a final check confirming the content is in the response_model format
                if isinstance(run_response.content, self.response_model):
                    return run_response

            # Otherwise convert the response to the structured format
            if isinstance(run_response.content, str):
                try:
                    structured_output = None
                    try:
                        if hasattr(self.response_model, 'model_validate_json'):
                            structured_output = self.response_model.model_validate_json(run_response.content)
                        elif hasattr(self.response_model, 'parse_raw'):  # Pydantic v1 兼容
                            structured_output = self.response_model.parse_raw(run_response.content)
                        elif issubclass(self.response_model, BaseModel):
                            data = json.loads(run_response.content)
                            structured_output = self.response_model(**data)
                        else:
                            data = json.loads(run_response.content)
                            structured_output = self.response_model(**data) if isinstance(data,
                                                                                          dict) else self.response_model(
                                data)
                    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
                        logger.warning(f"Failed to convert response to response_model: {exc}")
                        if run_response.content.startswith("```json"):
                            cleaned_content = run_response.content.replace("```json", "").replace("```", "").strip()
                            try:
                                if hasattr(self.response_model, 'model_validate_json'):
                                    structured_output = self.response_model.model_validate_json(cleaned_content)
                                else:
                                    data = json.loads(cleaned_content)
                                    structured_output = self.response_model(**data) if isinstance(data,
                                                                                                  dict) else self.response_model(
                                        data)
                            except Exception as e:
                                logger.error(f"Failed to parse cleaned JSON response: {e}")

                    # -*- Update Agent response
                    if structured_output is not None:
                        run_response.content = structured_output
                        run_response.content_type = self.response_model.__name__
                        if self.run_response is not None:
                            self.run_response.content = structured_output
                            self.run_response.content_type = self.response_model.__name__
                except Exception as e:
                    logger.warning(f"Failed to convert response to output model: {e}")
            else:
                logger.warning("Something went wrong. Run response content is not a string")
            return run_response
        else:
            if stream and self.is_streamable:
                resp = self._arun(
                    message=message,
                    stream=True,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                return resp
            else:
                resp = self._arun(
                    message=message,
                    stream=False,
                    audio=audio,
                    images=images,
                    videos=videos,
                    messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps,
                    **kwargs,
                )
                # For multi-round mode, consume entire generator to get final response
                if self.enable_multi_round:
                    final_response = None
                    async for response in resp:
                        final_response = response
                    return final_response
                else:
                    return await resp.__anext__()

    async def arun_stream(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            *,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Async streaming run - yields response chunks as they arrive."""
        async for chunk in self._arun(
            message=message,
            stream=True,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            stream_intermediate_steps=stream_intermediate_steps,
            **kwargs,
        ):
            yield chunk
