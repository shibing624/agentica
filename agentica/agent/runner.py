# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Run execution methods for Agent (Async-First)

Async-first public API:
- async: run(), run_stream()
- sync adapters: run_sync(), run_stream_sync()

Notes:
- `run(stream=True)` is intentionally not supported. Streaming is an explicit method.
- Multi-round execution is NOT part of the base Agent runtime. If needed, implement it in
  specialized subclasses (e.g. DeepAgent).
"""

import asyncio
import json
import queue
import threading
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel

from agentica.utils.log import logger
from agentica.utils.async_utils import run_sync
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunEvent, RunResponse
from agentica.memory import AgentRun
from agentica.utils.string import parse_structured_output
from agentica.utils.langfuse_integration import langfuse_trace_context


class RunnerMixin:
    """Mixin class containing run execution methods for Agent.

    All core methods are async. Synchronous wrappers (run_sync, run_stream_sync, print_response)
    delegate to the async implementations via `run_sync()`.
    """

    def save_run_response_to_file(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        save_response_to_file: Optional[str] = None,
    ) -> None:
        _save_path = save_response_to_file
        if _save_path is None or self.run_response is None:
            return
        message_str = None
        if message is not None:
            if isinstance(message, str):
                message_str = message
            else:
                logger.warning("Did not use message in output file name: message is not a string")
        try:
            fn = _save_path.format(
                name=self.name, message=message_str
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

    def _aggregate_metrics_from_run_messages(self, messages: List[Message]) -> Dict[str, Any]:
        aggregated_metrics: Dict[str, Any] = defaultdict(list)
        for m in messages:
            if m.role == "assistant" and m.metrics is not None:
                for k, v in m.metrics.items():
                    aggregated_metrics[k].append(v)
        return aggregated_metrics

    def generic_run_response(self, content: Optional[str] = None, event: RunEvent = RunEvent.run_response) -> RunResponse:
        return RunResponse(
            run_id=self.run_id,
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
    # Core execution engine (async-only, single-round)
    # =========================================================================

    async def _run_impl(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: bool = False,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        stream_intermediate_steps: bool = False,
        save_response_to_file: Optional[str] = None,
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Unified execution engine.

        This is the ONLY core runtime for base `Agent`.

        - Non-streaming users should call `run()` (which consumes this generator).
        - Streaming users should call `run_stream()` (which returns this generator).

        All LLM calls within this run are grouped under a single Langfuse trace (if enabled).
        """

        async def _run_core() -> AsyncIterator[RunResponse]:
            # NOTE: this is the previous single-round implementation.
            self.stream = stream and self.is_streamable
            self.stream_intermediate_steps = stream_intermediate_steps and self.stream
            self.run_id = str(uuid4())
            self.run_response = RunResponse(run_id=self.run_id, agent_id=self.agent_id)

            # 1. Setup
            self.update_model()
            self.run_response.model = self.model.id if self.model is not None else None
            if self.context is not None:
                self._resolve_context()

            # Add introduction if provided
            if self.prompt_config.introduction is not None:
                self.add_introduction(self.prompt_config.introduction)

            # 3. Prepare messages
            system_message, user_messages, messages_for_model = await self.get_messages_for_run(
                message=message, audio=audio, images=images, videos=videos,
                messages=messages, add_messages=add_messages, **kwargs
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
                        user_message_for_memory = Message(role=self.prompt_config.user_message_role, content=message)
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
                                agent_id=self.agent_id,
                            )
                        if model_response_chunk.content is not None and model_response.content is not None:
                            model_response.content += model_response_chunk.content
                            yield RunResponse(
                                event=RunEvent.run_response,
                                content=model_response_chunk.content,
                                run_id=self.run_id,
                                agent_id=self.agent_id,
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
                        tool_calls_data.append(
                            {
                                "tool_call_id": m.tool_call_id,
                                "tool_name": m.tool_name,
                                "tool_args": m.tool_args,
                                "content": m.content,
                                "tool_call_error": getattr(m, "tool_call_error", False),
                                "metrics": m.metrics if hasattr(m, "metrics") else {},
                            }
                        )
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
                self.memory.add_system_message(system_message, system_message_role=self.prompt_config.system_message_role)
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
                    user_message_for_memory = Message(role=self.prompt_config.user_message_role, content=message)
                elif isinstance(message, Message):
                    user_message_for_memory = message
                if user_message_for_memory is not None:
                    agent_run.message = user_message_for_memory
                    if (
                        not memory_classification_tasks
                        and self.memory.create_user_memories
                        and self.memory.update_user_memories_after_run
                    ):
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
                        if (
                            not memory_classification_tasks
                            and self.memory.create_user_memories
                            and self.memory.update_user_memories_after_run
                        ):
                            await self.memory.update_memory(input=_um.get_content_string())
                    else:
                        logger.warning("Unable to add message to memory")
            self.memory.add_run(agent_run)

            if self.memory.create_session_summary and self.memory.update_session_summary_after_run:
                await self.memory.update_summary()

            # 7. Save output to file
            self.save_run_response_to_file(message=message, save_response_to_file=save_response_to_file)

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

        trace_input = message if isinstance(message, str) else str(message) if message else None
        trace_name = self.name or "agent-run"

        langfuse_tags = None
        if self.model and hasattr(self.model, "langfuse_tags"):
            langfuse_tags = self.model.langfuse_tags

        with langfuse_trace_context(
            name=trace_name,
            tags=langfuse_tags,
            input_data=trace_input,
        ) as trace:
            final_response: Optional[RunResponse] = None
            async for response in _run_core():
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
    # Timeout wrappers
    # =========================================================================

    async def _wrap_stream_with_timeout(
        self,
        stream_iter: AsyncIterator[RunResponse],
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
    ) -> AsyncIterator[RunResponse]:
        """Wrap an async streaming iterator with timeout control."""
        import time

        start_time = time.time()
        first_token_received = False

        async for item in stream_iter:
            if not first_token_received:
                elapsed = time.time() - start_time
                if first_token_timeout is not None and elapsed > first_token_timeout:
                    logger.warning(f"First token timed out after {first_token_timeout} seconds")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"First token timed out after {first_token_timeout} seconds",
                        event="FirstTokenTimeout",
                    )
                    return
                first_token_received = True

            if run_timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > run_timeout:
                    logger.warning(f"Stream run timed out after {run_timeout} seconds")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"Stream run timed out after {run_timeout} seconds",
                        event="RunTimeout",
                    )
                    return

            yield item

    async def _run_with_timeout(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        run_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent with timeout control (non-streaming only)."""
        try:
            coro = self._consume_run(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                **kwargs,
            )
            result = await asyncio.wait_for(coro, timeout=run_timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Agent run timed out after {run_timeout} seconds")
            return RunResponse(
                run_id=str(uuid4()),
                content=f"Agent run timed out after {run_timeout} seconds",
                event="RunTimeout",
            )

    async def _consume_run(
        self,
        message=None,
        *,
        audio=None,
        images=None,
        videos=None,
        messages=None,
        **kwargs,
    ) -> RunResponse:
        """Consume the _run_impl async generator and return the final response."""
        run_response = None
        async for response in self._run_impl(
            message=message,
            stream=False,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            **kwargs,
        ):
            run_response = response

        if self.response_model is not None:
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

    # =========================================================================
    # Public API: async run()/run_stream() + sync adapters
    # =========================================================================

    async def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        save_response_to_file: Optional[str] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent and return the final response (non-streaming).

        This is the primary async API.

        Args:
            run_timeout: Maximum total execution time (in seconds). From RunConfig.
            first_token_timeout: Maximum time to wait for first token (in seconds). From RunConfig.
            save_response_to_file: File path pattern to save response. From RunConfig.
            add_messages: Whether to add messages to memory.
        """
        if run_timeout is not None:
            return await self._run_with_timeout(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                run_timeout=run_timeout,
                add_messages=add_messages,
                save_response_to_file=save_response_to_file,
                **kwargs,
            )

        # Structured output path
        if self.response_model is not None:
            return await self._consume_run(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                add_messages=add_messages,
                save_response_to_file=save_response_to_file,
                **kwargs,
            )

        final_response = None
        async for response in self._run_impl(
            message=message,
            stream=False,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            save_response_to_file=save_response_to_file,
            **kwargs,
        ):
            final_response = response
        return final_response

    async def run_stream(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        stream_intermediate_steps: bool = False,
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        save_response_to_file: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Run the Agent and stream incremental responses.

        Usage:
            async for chunk in agent.run_stream("..."):
                ...
        """
        if self.response_model is not None:
            raise ValueError("Structured output does not support streaming. Use run() instead.")

        resp: AsyncIterator[RunResponse] = self._run_impl(
            message=message,
            stream=True,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            stream_intermediate_steps=stream_intermediate_steps,
            save_response_to_file=save_response_to_file,
            **kwargs,
        )
        if run_timeout is not None or first_token_timeout is not None:
            resp = self._wrap_stream_with_timeout(
                resp, run_timeout=run_timeout, first_token_timeout=first_token_timeout
            )

        async for item in resp:
            yield item

    def run_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        save_response_to_file: Optional[str] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Synchronous wrapper for `run()` (non-streaming only)."""
        return run_sync(
            self.run(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                add_messages=add_messages,
                run_timeout=run_timeout,
                first_token_timeout=first_token_timeout,
                save_response_to_file=save_response_to_file,
                **kwargs,
            )
        )

    def run_stream_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        stream_intermediate_steps: bool = False,
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        save_response_to_file: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[RunResponse]:
        """Synchronous wrapper for `run_stream()`.

        Internally runs the async iterator in a dedicated background thread.
        """

        def _iter_from_async(ait: AsyncIterator[RunResponse]) -> Iterator[RunResponse]:
            sentinel = object()
            q: "queue.Queue[object]" = queue.Queue()

            def _producer() -> None:
                async def _consume() -> None:
                    try:
                        async for item in ait:
                            q.put(item)
                    except BaseException as e:
                        q.put(e)
                    finally:
                        try:
                            if hasattr(ait, "aclose"):
                                await ait.aclose()  # type: ignore[attr-defined]
                        finally:
                            q.put(sentinel)

                asyncio.run(_consume())

            thread = threading.Thread(target=_producer, daemon=True)
            thread.start()

            try:
                while True:
                    item = q.get()
                    if item is sentinel:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield cast(RunResponse, item)
            finally:
                # Best-effort: let the producer finish; it will be daemon anyway.
                pass

        return _iter_from_async(
            self.run_stream(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                add_messages=add_messages,
                stream_intermediate_steps=stream_intermediate_steps,
                run_timeout=run_timeout,
                first_token_timeout=first_token_timeout,
                save_response_to_file=save_response_to_file,
                **kwargs,
            )
        )
