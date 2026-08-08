# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner streaming adapters and idle/run timeout wrappers
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)
from uuid import uuid4


from agentica.utils.log import logger
from agentica.utils.async_utils import run_sync
from agentica.hooks import RunHooks
from agentica.model.message import Message
from agentica.run_input import merge_run_config, reject_unknown_run_kwargs
from agentica.run_response import AgentCancelledError, RunResponse
from agentica.run_config import RunConfig
from agentica.utils.string import parse_structured_output

if TYPE_CHECKING:
    pass



class StreamMixin:
    """Extracted Runner methods."""

    agent: Any
    _run_impl: Any
    _resolve_max_api_retry: Any
    _isolate_fallback_models: Any

    async def _wrap_stream_with_timeout(
        self,
        stream_iter: AsyncIterator[RunResponse],
        run_timeout: Optional[float] = None,
        first_token_timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> AsyncIterator[RunResponse]:
        """Wrap an async streaming iterator with timeout control.

        Three independent timeouts (any one can fire):
        - first_token_timeout: max seconds to wait for the first token.
        - idle_timeout:        max seconds between consecutive tokens.
                               Detects "silent hang" where the connection stays
                               open but no data flows (mirrors CC's stream idle
                               watchdog in claude.ts).
        - run_timeout:         max total wall-clock seconds for the entire stream.
        """
        start_time = time.time()
        first_token_received = False
        last_token_time = start_time

        async for item in stream_iter:
            now = time.time()

            if not first_token_received:
                elapsed = now - start_time
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
                elapsed = now - start_time
                if elapsed > run_timeout:
                    logger.warning(f"Stream run timed out after {run_timeout} seconds")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"Stream run timed out after {run_timeout} seconds",
                        event="RunTimeout",
                    )
                    return

            # Idle watchdog: detect "silent hang" between tokens
            if idle_timeout is not None and first_token_received:
                idle_elapsed = now - last_token_time
                if idle_elapsed > idle_timeout:
                    logger.warning(f"Stream idle timeout: no token for {idle_elapsed:.1f}s (limit {idle_timeout}s)")
                    yield RunResponse(
                        run_id=str(uuid4()),
                        content=f"Stream idle timeout: no new token for {idle_timeout} seconds",
                        event="StreamIdleTimeout",
                    )
                    return

            last_token_time = now
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
        agent = self.agent
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

        if run_response is None:
            raise RuntimeError("Agent run completed without producing a response")

        if agent.response_model is not None:
            if agent.use_structured_outputs:
                if isinstance(run_response.content, agent.response_model):
                    return run_response

            if isinstance(run_response.content, str):
                try:
                    structured_output = parse_structured_output(run_response.content, agent.response_model)
                    if structured_output is not None:
                        run_response.content = structured_output
                        run_response.content_type = agent.response_model.__name__
                        if agent.run_response is not None:
                            agent.run_response.content = structured_output
                            agent.run_response.content_type = agent.response_model.__name__
                except Exception as e:
                    logger.warning(
                        f"Failed to convert response to output model "
                        f"'{agent.response_model.__name__ if agent.response_model else None}': {e} "
                        f"[agent={agent.identifier}, run_id={agent.run_id}]"
                    )

        return run_response

    async def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
        hooks: Optional[RunHooks] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent and return the final response (non-streaming)."""
        reject_unknown_run_kwargs(kwargs)
        config = merge_run_config(config, timeout=timeout, hooks=hooks)
        run_timeout = config.run_timeout
        save_response_to_file = config.save_response_to_file
        effective_hooks = config.hooks
        enabled_tools = config.enabled_tools
        enabled_skills = config.enabled_skills
        source = config.source

        self.agent._run_max_cost_usd = config.max_cost_usd
        self.agent._run_max_api_retry = self._resolve_max_api_retry(
            self.agent,
            config,
        )
        self.agent._run_fallback_models = self._isolate_fallback_models(
            list(config.fallback_models or self.agent.fallback_models or [])
        )
        self.agent._run_fallback_on_break = bool(config.fallback_on_break or self.agent.fallback_on_break)

        if run_timeout is not None:
            return await self._run_with_timeout(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                run_timeout=run_timeout,
                save_response_to_file=save_response_to_file,
                hooks=effective_hooks,
                enabled_tools=enabled_tools,
                enabled_skills=enabled_skills,
                source=source,
                **kwargs,
            )

        if self.agent.response_model is not None:
            return await self._consume_run(
                message=message,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                save_response_to_file=save_response_to_file,
                hooks=effective_hooks,
                enabled_tools=enabled_tools,
                enabled_skills=enabled_skills,
                source=source,
                **kwargs,
            )

        final_response = None
        try:
            async for response in self._run_impl(
                message=message,
                stream=False,
                audio=audio,
                images=images,
                videos=videos,
                messages=messages,
                save_response_to_file=save_response_to_file,
                hooks=effective_hooks,
                enabled_tools=enabled_tools,
                enabled_skills=enabled_skills,
                source=source,
                **kwargs,
            ):
                final_response = response
        except asyncio.CancelledError:
            self.agent._cancelled = False
            raise AgentCancelledError("Agent run cancelled by user") from None
        if final_response is None:
            raise RuntimeError("Agent run completed without producing a response")
        return final_response

    async def run_stream(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
        hooks: Optional[RunHooks] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Run the Agent and stream incremental responses."""
        reject_unknown_run_kwargs(kwargs)
        config = merge_run_config(config, timeout=timeout, hooks=hooks)
        stream_intermediate_steps = config.stream_intermediate_steps
        run_timeout = config.run_timeout
        first_token_timeout = config.first_token_timeout
        idle_timeout = config.idle_timeout
        save_response_to_file = config.save_response_to_file
        effective_hooks = config.hooks
        enabled_tools = config.enabled_tools
        enabled_skills = config.enabled_skills
        source = config.source

        self.agent._run_max_cost_usd = config.max_cost_usd
        self.agent._run_max_api_retry = self._resolve_max_api_retry(
            self.agent,
            config,
        )
        self.agent._run_fallback_models = self._isolate_fallback_models(
            list(config.fallback_models or self.agent.fallback_models or [])
        )
        self.agent._run_fallback_on_break = bool(config.fallback_on_break or self.agent.fallback_on_break)

        if self.agent.response_model is not None:
            raise ValueError("Structured output does not support streaming. Use run() instead.")

        resp: AsyncIterator[RunResponse] = self._run_impl(
            message=message,
            stream=True,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            stream_intermediate_steps=stream_intermediate_steps,
            save_response_to_file=save_response_to_file,
            hooks=effective_hooks,
            enabled_tools=enabled_tools,
            enabled_skills=enabled_skills,
            source=source,
            **kwargs,
        )
        if run_timeout is not None or first_token_timeout is not None or idle_timeout is not None:
            resp = self._wrap_stream_with_timeout(
                resp,
                run_timeout=run_timeout,
                first_token_timeout=first_token_timeout,
                idle_timeout=idle_timeout,
            )

        try:
            async for item in resp:
                yield item
        except asyncio.CancelledError:
            self.agent._cancelled = False
            raise AgentCancelledError("Agent run cancelled by user") from None

    def run_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
        hooks: Optional[RunHooks] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Synchronous wrapper for `run()` (non-streaming only)."""
        return run_sync(
            self.run(
                message=message,
                messages=messages,
                audio=audio,
                images=images,
                videos=videos,
                timeout=timeout,
                hooks=hooks,
                config=config,
                **kwargs,
            )
        )

    def run_stream_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        timeout: Optional[float] = None,
        hooks: Optional[RunHooks] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> Iterator[RunResponse]:
        """Synchronous wrapper for `run_stream()`."""

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

            completed = False
            try:
                while True:
                    # Use timeout so KeyboardInterrupt can be delivered promptly
                    try:
                        item = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is sentinel:
                        completed = True
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield cast(RunResponse, item)
            finally:
                # If the caller stopped consuming early (break / exception /
                # GeneratorExit) the producer thread is still driving the agent
                # to completion in the background — burning tokens silently.
                # Cancel it (thread-safe) so an abandoned stream doesn't keep
                # calling tools/LLMs with nobody listening.
                if not completed:
                    logger.info("run_stream_sync consumer exited early; cancelling background agent run")
                    self.agent.cancel()

        return _iter_from_async(
            self.run_stream(
                message=message,
                messages=messages,
                audio=audio,
                images=images,
                videos=videos,
                timeout=timeout,
                hooks=hooks,
                config=config,
                **kwargs,
            )
        )

