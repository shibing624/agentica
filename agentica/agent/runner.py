# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Run execution methods for Agent (Async-First)

All core logic is implemented as async. Synchronous access is provided via
run_sync() / iter_over_async() thin wrappers from agentica.utils.async_utils.
"""

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
from agentica.utils.async_utils import run_sync, iter_over_async
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
    """Mixin class containing run execution methods for Agent.

    All core methods are async. Synchronous wrappers (run_sync, print_response)
    delegate to the async implementations via run_sync() / iter_over_async().
    """

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

    # =========================================================================
    # Core async _run: wraps single/multi-round under a Langfuse trace
    # =========================================================================

    async def _run(
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
        """Run the Agent with optional multi-round strategy.

        All LLM calls within this run are grouped under a single Langfuse trace
        when Langfuse is configured.
        """
        trace_input = message if isinstance(message, str) else str(message) if message else None
        trace_name = self.name or "agent-run"

        langfuse_tags = None
        if self.model and hasattr(self.model, 'langfuse_tags'):
            langfuse_tags = self.model.langfuse_tags

        with langfuse_trace_context(
                name=trace_name,
                session_id=self.session_id,
                user_id=self.user_id,
                tags=langfuse_tags,
                input_data=trace_input,
        ) as trace:
            final_response = None

            if self.enable_multi_round:
                async for response in self._run_multi_round(
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
                async for response in self._run_single_round(
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

            if final_response:
                output_content = final_response.content
                if isinstance(output_content, BaseModel):
                    output_content = output_content.model_dump()
                trace.set_output(output_content)
                trace.set_metadata("run_id", final_response.run_id)
                trace.set_metadata("model", final_response.model)

    # =========================================================================
    # Single-round execution (async)
    # =========================================================================

    async def _run_single_round(
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
        """Run the Agent with a message and return the response (single round).

        Steps:
        1. Setup: Update the model class and resolve context
        2. Read existing session from storage
        3. Prepare messages for this run
        4. Generate a response from the Model (includes running function calls)
        5. Update Memory
        6. Save session to storage
        7. Save output to file if save_response_to_file is set
        8. Set the run_input
        """
        self.stream = stream and self.is_streamable
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Setup
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # 2. Read existing session
        self.read_from_storage()

        # Add introduction if provided
        if self.introduction is not None:
            self.add_introduction(self.introduction)

        # 3. Prepare messages
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )
        num_input_messages = len(messages_for_model)

        if self.stream_intermediate_steps:
            yield self.generic_run_response("Run started", RunEvent.run_started)

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
                    memory_task = asyncio.create_task(
                        self.memory.should_update_memory(input=user_message_for_memory.get_content_string())
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
                            self.memory.should_update_memory(input=_um.get_content_string())
                        )
                        memory_classification_tasks.append((_um, memory_task))

        # 4. Generate response from the Model
        model_response: ModelResponse
        self.model = cast(Model, self.model)
        if stream and self.is_streamable:
            model_response = ModelResponse(content="", reasoning_content="")
            model_response_stream = self.model.response_stream(messages=messages_for_model)
            self._cancelled = False
            async for model_response_chunk in model_response_stream:
                self._check_cancelled()
                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
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
            self._cancelled = False
            self._check_cancelled()
            model_response = await self.model.response(messages=messages_for_model)
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

        # Build run messages
        run_messages = user_messages + messages_for_model[num_input_messages:]
        if system_message is not None:
            run_messages.insert(0, system_message)
        self.run_response.messages = run_messages
        self.run_response.metrics = self._aggregate_metrics_from_run_messages(run_messages)
        if self.stream:
            self.run_response.content = model_response.content
            if model_response.reasoning_content:
                self.run_response.reasoning_content = model_response.reasoning_content

        # 5. Update Memory
        if self.stream_intermediate_steps:
            yield self.generic_run_response("Updating memory", RunEvent.updating_memory)

        if system_message is not None:
            self.memory.add_system_message(system_message, system_message_role=self.system_message_role)
        self.memory.add_messages(messages=(user_messages + messages_for_model[num_input_messages:]))

        agent_run = AgentRun(response=self.run_response)

        # Process memory classification results that were started in parallel
        if memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
            for user_message, memory_task in memory_classification_tasks:
                try:
                    should_update_memory = await memory_task
                    if should_update_memory:
                        await self.memory.update_memory(input=user_message.get_content_string())
                except Exception as e:
                    logger.warning(f"Error in memory processing: {e}")
                    await self.memory.update_memory(input=user_message.get_content_string())

        # Handle agent_run message assignment
        if message is not None:
            user_message_for_memory: Optional[Message] = None
            if isinstance(message, str):
                user_message_for_memory = Message(role=self.user_message_role, content=message)
            elif isinstance(message, Message):
                user_message_for_memory = message
            if user_message_for_memory is not None:
                agent_run.message = user_message_for_memory
                if not memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                    await self.memory.update_memory(input=user_message_for_memory.get_content_string())
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
                    if not memory_classification_tasks and self.memory.create_user_memories and self.memory.update_user_memories_after_run:
                        await self.memory.update_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        self.memory.add_run(agent_run)

        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            await self.memory.update_summary()

        # 6. Save session to storage
        self.write_to_storage()

        # 7. Save output to file
        self.save_run_response_to_file(message=message)

        # 8. Set run_input
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

        if not self.stream:
            yield self.run_response

    # =========================================================================
    # Multi-round Hook Methods (can be overridden by subclasses like DeepAgent)
    # =========================================================================

    def _on_pre_step(
            self: "Agent",
            step: int,
            messages: List[Message]
    ) -> Tuple[bool, Optional[str]]:
        """Hook called before each step in multi-round execution.

        Override this method in subclasses to implement custom pre-step logic
        such as context overflow handling.

        Returns:
            (should_force_answer, optional_warning_message)
        """
        return False, None

    def _on_tool_call(self: "Agent", tool_name: str, step: int) -> Optional[str]:
        """Hook called when a tool is about to be executed.

        Override this method in subclasses to implement custom tool call logic
        such as repetitive behavior detection.
        """
        return None

    def _on_post_step(self: "Agent", step: int, messages: List[Message]) -> None:
        """Hook called after each step in multi-round execution.

        Override this method in subclasses to implement custom post-step logic
        such as reflection prompts.
        """
        pass

    # =========================================================================
    # Multi-round execution (async)
    # =========================================================================

    async def _run_multi_round(
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
        """Run the Agent with a multi-round strategy.

        Loops until no more tool calls are needed or max_rounds is reached.
        """
        self.stream = stream and self.is_streamable
        self.stream_intermediate_steps = stream_intermediate_steps and self.stream
        self.run_id = str(uuid4())
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, agent_id=self.agent_id)

        # 1. Setup
        self.update_model()
        self.run_response.model = self.model.id if self.model is not None else None
        if self.context is not None and self.resolve_context:
            self._resolve_context()

        # Disable model's internal tool execution - we handle it manually
        self.model = cast(Model, self.model)
        original_run_tools = self.model.run_tools
        self.model.run_tools = False

        # 2. Read existing session
        self.read_from_storage()

        if self.introduction is not None:
            self.add_introduction(self.introduction)

        # 3. Prepare initial messages
        system_message, user_messages, messages_for_model = self.get_messages_for_run(
            message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
        )
        num_input_messages = len(messages_for_model)

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
            self._cancelled = False
            for current_round in range(1, self.max_rounds + 1):
                self._check_cancelled()
                logger.debug(f"Turn {current_round}/{self.max_rounds}")

                # Hook: Pre-step processing (for DeepAgent context management)
                should_force_answer, warning_msg = self._on_pre_step(current_round, messages_for_model)
                if warning_msg:
                    messages_for_model.append(Message(role="system", content=warning_msg))
                if should_force_answer:
                    logger.warning(f"Force answer triggered at turn {current_round}")
                    break

                # Token limit check (character-based approximation)
                total_content = " ".join([str(msg.content or "") for msg in messages_for_model])
                if len(total_content) > self.model.context_window * 3:
                    logger.warning(f"Token limit approaching, stopping at turn {current_round}")
                    break

                if self.stream_intermediate_steps:
                    yield self.generic_run_response(f"Turn {current_round}", RunEvent.run_response)

                # Call model (async)
                model_response = await self.model.response(messages=messages_for_model)

                # Get assistant message
                assistant_message = messages_for_model[-1] if messages_for_model else None
                if assistant_message and assistant_message.role == "assistant":
                    if model_response.reasoning_content and not assistant_message.reasoning_content:
                        assistant_message.reasoning_content = model_response.reasoning_content
                    all_run_messages.append(assistant_message)

                # Update run_response
                if model_response.content:
                    self.run_response.content = model_response.content
                if model_response.reasoning_content:
                    self.run_response.reasoning_content = model_response.reasoning_content

                # Yield intermediate response
                yield RunResponse(
                    content=model_response.content,
                    reasoning_content=model_response.reasoning_content,
                    event=RunEvent.multi_round_turn.value,
                    extra_data=RunResponseExtraData(
                        add_messages=[Message(role="info", content=f"Turn {current_round}/{self.max_rounds}")]
                    )
                )

                content_preview = (model_response.content or model_response.reasoning_content or "")[:500]
                if content_preview:
                    logger.debug(f"Turn {current_round} response: {content_preview}...")

                # Check for tool calls
                has_tool_calls = (
                    assistant_message and
                    assistant_message.tool_calls and
                    len(assistant_message.tool_calls) > 0
                )

                if has_tool_calls:
                    tool_results = []
                    for tool_call in assistant_message.tool_calls:
                        self._check_cancelled()
                        tool_call_id = tool_call.get("id", "")
                        func_info = tool_call.get("function", {})
                        func_name = func_info.get("name", "")
                        func_args_str = func_info.get("arguments", "{}")

                        logger.debug(f"Tool call: {func_name}({func_args_str})")

                        # Hook: Check for repetitive behavior
                        repetition_warning = self._on_tool_call(func_name, current_round)
                        if repetition_warning:
                            messages_for_model.append(Message(role="system", content=repetition_warning))

                        yield RunResponse(
                            content=f"{func_name}({func_args_str[:500]}{'...' if len(func_args_str) > 500 else ''})",
                            event=RunEvent.multi_round_tool_call.value
                        )

                        if self.stream_intermediate_steps:
                            yield self.generic_run_response(
                                f"Calling tool: {func_name}",
                                RunEvent.tool_call_started
                            )

                        # Execute tool (async)
                        try:
                            function_call = get_function_call_for_tool_call(tool_call, self.model.functions)
                            if function_call is not None:
                                await function_call.execute()
                                result_str = str(function_call.result) if function_call.result is not None else ""

                                result_preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                                yield RunResponse(
                                    content=f"{func_name}: {result_preview}",
                                    event=RunEvent.multi_round_tool_result.value
                                )

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
                            await self.compression_manager.compress(messages_for_model)
                            logger.debug(f"Compressed tool results, stats: {self.compression_manager.get_stats()}")

                    # Hook: Post-step processing
                    self._on_post_step(current_round, messages_for_model)
                else:
                    logger.debug("No tool calls, task completed")
                    yield RunResponse(
                        content=f"Task completed in {current_round} turns",
                        event=RunEvent.multi_round_completed.value
                    )
                    break

        finally:
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
                    await self.memory.update_memory(input=user_message_for_memory.get_content_string())
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
                        await self.memory.update_memory(input=_um.get_content_string())
                else:
                    logger.warning("Unable to add message to memory")
        self.memory.add_run(agent_run)

        if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
            await self.memory.update_summary()

        # 7. Save session to storage
        self.write_to_storage()

        # 8. Save output to file
        self.save_run_response_to_file(message)

        # 9. Set run input
        if message is not None:
            self.run_input = message
        elif messages is not None:
            self.run_input = messages

        if self.stream_intermediate_steps:
            yield self.generic_run_response(
                f"Multi-round completed in {current_round} turns",
                RunEvent.run_completed
            )
        logger.debug(f"Multi-round completed in {current_round} turns")

        yield self.run_response

    # =========================================================================
    # Timeout wrappers
    # =========================================================================

    async def _wrap_stream_with_timeout(
            self: "Agent",
            stream_iter: AsyncIterator[RunResponse],
    ) -> AsyncIterator[RunResponse]:
        """Wrap an async streaming iterator with timeout control."""
        import time

        start_time = time.time()
        first_token_received = False

        async for item in stream_iter:
            if not first_token_received:
                elapsed = time.time() - start_time
                if self.first_token_timeout is not None and elapsed > self.first_token_timeout:
                    logger.warning(f"First token timed out after {self.first_token_timeout} seconds")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"First token timed out after {self.first_token_timeout} seconds",
                        event="FirstTokenTimeout",
                    )
                    return
                first_token_received = True

            if self.run_timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > self.run_timeout:
                    logger.warning(f"Stream run timed out after {self.run_timeout} seconds")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"Stream run timed out after {self.run_timeout} seconds",
                        event="RunTimeout",
                    )
                    return

            yield item

    async def _run_with_timeout(
            self: "Agent",
            message: Optional[Union[str, List, Dict, Message]] = None,
            audio: Optional[Any] = None,
            images: Optional[Sequence[Any]] = None,
            videos: Optional[Sequence[Any]] = None,
            messages: Optional[Sequence[Union[Dict, Message]]] = None,
            stream_intermediate_steps: bool = False,
            **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent with timeout control (non-streaming only)."""
        try:
            coro = self._consume_run(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )
            result = await asyncio.wait_for(coro, timeout=self.run_timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Agent run timed out after {self.run_timeout} seconds")
            return RunResponse(
                run_id=str(uuid4()),
                content=f"Agent run timed out after {self.run_timeout} seconds",
                event="RunTimeout",
            )

    async def _consume_run(
            self: "Agent",
            message=None,
            *,
            stream=False,
            audio=None,
            images=None,
            videos=None,
            messages=None,
            stream_intermediate_steps=False,
            **kwargs,
    ) -> RunResponse:
        """Consume the _run async generator and return the final response.

        Handles both structured output parsing and regular responses.
        """
        if self.response_model is not None and self.parse_response:
            run_response = None
            async for response in self._run(
                    message=message, stream=False, audio=audio, images=images,
                    videos=videos, messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps, **kwargs):
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
                        if self.run_response is not None:
                            self.run_response.content = structured_output
                            self.run_response.content_type = self.response_model.__name__
                except Exception as e:
                    logger.warning(f"Failed to convert response to output model: {e}")
            return run_response
        else:
            final_response = None
            async for response in self._run(
                    message=message, stream=False, audio=audio, images=images,
                    videos=videos, messages=messages,
                    stream_intermediate_steps=stream_intermediate_steps, **kwargs):
                final_response = response
            return final_response

    # =========================================================================
    # Public API: async run() + run_sync()
    # =========================================================================

    @overload
    async def run(
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
    async def run(
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

    async def run(
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
        """Run the Agent with a message and return the response.

        This is the primary async API. For synchronous usage, use run_sync().

        Timeout settings:
        - run_timeout: Maximum total execution time (in seconds).
        - first_token_timeout: Maximum time to wait for the first token (streaming only).
        """
        # Handle timeout for non-streaming mode
        if self.run_timeout is not None and not stream:
            return await self._run_with_timeout(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )

        # Structured output path
        if self.response_model is not None and self.parse_response:
            return await self._consume_run(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )

        # Streaming path
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
            if self.run_timeout is not None or self.first_token_timeout is not None:
                return self._wrap_stream_with_timeout(resp)
            return resp

        # Non-streaming path
        final_response = None
        async for response in self._run(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
        ):
            final_response = response
        return final_response

    def run_sync(
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
        """Synchronous wrapper for run().

        For non-streaming: returns RunResponse directly.
        For streaming: returns a synchronous Iterator[RunResponse].
        """
        if stream and self.is_streamable:
            async_iter = self._run(
                message=message,
                stream=True,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            )
            return iter_over_async(async_iter)
        else:
            return run_sync(self.run(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                stream_intermediate_steps=stream_intermediate_steps,
                **kwargs,
            ))
