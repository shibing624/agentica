# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner class composing mixins; thin orchestration surface
"""

import json
from typing import (
    Any,
    Dict,
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

from pydantic import BaseModel

from agentica.utils.log import logger
from agentica.hooks import AgentHooks, RunHooks, _CompositeAgentHooks, _CompositeRunHooks
from agentica.model.message import Message
from agentica.run_events import RunEventRecord, RunEventType

if TYPE_CHECKING:
    from agentica.agent import Agent


from agentica.runner.compress import CompressMixin
from agentica.runner.loop import LoopMixin
from agentica.runner.persist import PersistMixin
from agentica.runner.retry_fallback import RetryMixin
from agentica.runner.steer import SteerMixin
from agentica.runner.stream import StreamMixin


class Runner(CompressMixin, RetryMixin, PersistMixin, SteerMixin, StreamMixin, LoopMixin):
    """Independent execution engine for Agent.

    All core methods are async. Synchronous wrappers (run_sync, run_stream_sync)
    delegate to the async implementations via `run_sync()`.

    The agentic loop (tool call → LLM → tool call → ...) is driven here,
    NOT in the Model layer. Model.response()/response_stream() do a single
    LLM call + tool execution; Runner loops until no more tool calls remain.
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent

    def _emit_event(
        self,
        event_type: RunEventType,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured RunEventRecord through the agent's event callback.

        Always safe to call: silently no-ops when the agent has no callback,
        no run_id yet, or the callback raises (event bus must never break a run).
        """
        agent = self.agent
        cb = agent._event_callback
        run_ctx = agent.run_context
        if run_ctx is None:
            return
        record = RunEventRecord(
            run_id=run_ctx.run_id,
            event_type=event_type,
            agent_id=run_ctx.agent_id,
            parent_run_id=run_ctx.parent_run_id,
            payload=payload or {},
        )
        if cb is not None:
            try:
                cb(record.to_dict())
            except Exception as e:
                # Event bus is the single telemetry entry point. Failures must
                # be visible (warning, not debug) and carry a traceback so a
                # broken display callback or langfuse exporter is diagnosable.
                # We still swallow the exception: a misbehaving event consumer
                # must never abort the agent run itself.
                logger.warning(
                    f"event callback failed for {event_type.value}: {e}",
                    exc_info=True,
                )

    @staticmethod
    def _serialize_langfuse_data(value: Any) -> Any:
        """Convert Agentica objects to Langfuse-friendly JSON-like data.

        Used for the trace input field (built from incoming user messages),
        not for hook records — HookRecorder owns its own serializer with
        depth/cycle/length caps.
        """
        if value is None:
            return None
        if isinstance(value, Message):
            return value.to_model_dict()
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): Runner._serialize_langfuse_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Runner._serialize_langfuse_data(item) for item in value]
        return str(value)

    @staticmethod
    def _extract_langfuse_message_text(value: Union[Dict, Message]) -> Optional[str]:
        """Extract display text from a Message-like object for trace input."""
        if isinstance(value, Message):
            content = value.content
        else:
            content = value.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return None
        return json.dumps(content, ensure_ascii=False)

    @classmethod
    def _build_langfuse_trace_input(
        cls,
        message: Optional[Union[str, List, Dict, Message]],
        messages: Optional[Sequence[Union[Dict, Message]]],
    ) -> Any:
        """Build the root trace input for both message and messages modes."""
        if messages is not None:
            for candidate in reversed(messages):
                role = candidate.role if isinstance(candidate, Message) else candidate.get("role")
                if role == "user":
                    text = cls._extract_langfuse_message_text(candidate)
                    if text is not None:
                        return text
            return cls._serialize_langfuse_data(list(messages))
        if isinstance(message, Message):
            text = cls._extract_langfuse_message_text(message)
            if text is not None:
                return text
        if isinstance(message, dict):
            text = cls._extract_langfuse_message_text(message)
            if text is not None:
                return text
        return cls._serialize_langfuse_data(message)

    async def _dispatch_agent_hook(
        self,
        method_name: str,
        call_factory: Callable[[AgentHooks], Awaitable[Any]],
    ) -> None:
        agent_hooks = self.agent.hooks
        if agent_hooks is None:
            return
        if isinstance(agent_hooks, list):
            agent_hooks = _CompositeAgentHooks(agent_hooks)
        if isinstance(agent_hooks, _CompositeAgentHooks):
            await call_factory(agent_hooks)
        else:
            await self.agent._hook_recorder.run(
                agent_hooks,
                "agent",
                method_name,
                call_factory(agent_hooks),
                base_class=AgentHooks,
            )

    async def _dispatch_run_hook(
        self,
        method_name: str,
        call_factory: Callable[[RunHooks], Awaitable[Any]],
    ) -> None:
        run_hooks = self.agent._run_hooks
        if run_hooks is None:
            return
        if isinstance(run_hooks, list):
            run_hooks = _CompositeRunHooks(run_hooks)
        if isinstance(run_hooks, _CompositeRunHooks):
            await call_factory(run_hooks)
        else:
            await self.agent._hook_recorder.run(
                run_hooks,
                "run",
                method_name,
                call_factory(run_hooks),
                base_class=RunHooks,
            )

    async def _dispatch_user_prompt_hook(self, message: str) -> Optional[str]:
        run_hooks = self.agent._run_hooks
        if run_hooks is None:
            return None
        if isinstance(run_hooks, list):
            run_hooks = _CompositeRunHooks(run_hooks)
        if isinstance(run_hooks, _CompositeRunHooks):
            return await run_hooks.on_user_prompt(agent=self.agent, message=message)
        return await self.agent._hook_recorder.run(
            run_hooks,
            "run",
            "on_user_prompt",
            run_hooks.on_user_prompt(agent=self.agent, message=message),
            base_class=RunHooks,
        )

