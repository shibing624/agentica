# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner - Independent execution engine for Agent (Async-First)

Runner is decoupled from Agent:
- Agent defines identity and capabilities ("who I am, what I can do")
- Runner handles execution (LLM calls, tool calls, streaming, memory updates)

Async-first public API:
- async: run(), run_stream()
- sync adapters: run_sync(), run_stream_sync()

Notes:
- `run(stream=True)` is intentionally not supported. Streaming is an explicit method.
- Multi-round execution is NOT part of the base Runner. If needed, implement it in
  specialized subclasses (e.g. DeepRunner).
"""

import asyncio
import copy
import json
import queue
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    cast,
    Dict,
    Awaitable,
    Callable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel

from agentica.utils.log import logger, _run_id_var, _parent_run_id_var, _short as _short_run_id
from agentica.utils.async_utils import run_sync
from agentica.compression.micro import micro_compact
from agentica.compression.tool_result_storage import enforce_tool_result_budget
from agentica.cost_tracker import CostTracker
from agentica.hooks import AgentHooks, RunHooks, _CompositeAgentHooks, _CompositeRunHooks
from agentica.model.base import Model
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.model.usage import Usage
from agentica.run_input import merge_run_config, reject_unknown_run_kwargs
from agentica.run_response import AgentCancelledError, RunBreakReason, RunEvent, RunResponse
from agentica.run_config import RunConfig
from agentica.run_context import RunContext, RunSource, RunStatus, TaskAnchor
from agentica.run_events import RunEventRecord, RunEventType
from agentica.memory import AgentRun
from agentica.utils.string import parse_structured_output
from agentica.utils.tokens import count_tokens
from agentica.utils.langfuse_integration import langfuse_trace_context
from agentica.tools.base import FunctionCall
from agentica.guardrails.agent import (
    normalize_input_for_guardrails,
    run_input_guardrails,
    run_output_guardrails,
)

if TYPE_CHECKING:
    from agentica.agent import Agent


class LoopBreak(NamedTuple):
    """Structured result of a Runner safety check that aborts the agentic loop.

    ``reason`` is a stable machine code (RunBreakReason value); ``message`` is a
    human-readable detail. Both are surfaced on ``RunResponse`` so downstream
    never has to parse internal error text out of ``content``.
    """

    reason: str
    message: str


@dataclass
class ModelCallResult:
    """A model response plus the concrete provider that produced it."""

    response: Any
    used_model: Model
    used_fallback: bool = False

    def __getattr__(self, name: str) -> Any:
        # Transparent delegation to the wrapped response (ModelResponse or the
        # stream iterator) for ergonomic access at call sites/tests. Guard the
        # backing field so a half-built instance (deepcopy/pickle) can't recurse
        # forever via getattr(self.response, ...) before `response` is set.
        if name == "response":
            raise AttributeError(name)
        return getattr(self.response, name)


@dataclass
class ToolHandlingResult:
    """Runner-owned tool execution summary for the current LLM turn."""

    had_tool_calls: bool
    tool_results: List[Dict[str, Any]]


class Runner:
    """Independent execution engine for Agent.

    All core methods are async. Synchronous wrappers (run_sync, run_stream_sync)
    delegate to the async implementations via `run_sync()`.

    The agentic loop (tool call → LLM → tool call → ...) is driven here,
    NOT in the Model layer. Model.response()/response_stream() do a single
    LLM call + tool execution; Runner loops until no more tool calls remain.
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent

    # =========================================================================
    # Internal event emission (Phase 3 of arch_v5.md)
    # =========================================================================

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

    # =========================================================================
    # Hook dispatching (Issue #1/#2 fix)
    #
    # Composite hooks own chain/parallel semantics. Runner just hands them the
    # call site and lets HookRecorder (installed as observer) capture each
    # leaf invocation. Single-hook (non-composite) callers go through the
    # recorder directly so audit trail stays uniform.
    # =========================================================================

    async def _dispatch_agent_hook(
        self,
        method_name: str,
        call_factory: Callable[[AgentHooks], Awaitable[Any]],
    ) -> None:
        hooks = self.agent.hooks
        if hooks is None:
            return
        if isinstance(hooks, _CompositeAgentHooks):
            await call_factory(hooks)
        else:
            await self.agent._hook_recorder.run(
                hooks,
                "agent",
                method_name,
                call_factory(hooks),
                base_class=AgentHooks,
            )

    async def _dispatch_run_hook(
        self,
        method_name: str,
        call_factory: Callable[[RunHooks], Awaitable[Any]],
    ) -> None:
        hooks = self.agent._run_hooks
        if hooks is None:
            return
        if isinstance(hooks, _CompositeRunHooks):
            await call_factory(hooks)
        else:
            await self.agent._hook_recorder.run(
                hooks,
                "run",
                method_name,
                call_factory(hooks),
                base_class=RunHooks,
            )

    async def _dispatch_user_prompt_hook(self, message: str) -> Optional[str]:
        hooks = self.agent._run_hooks
        if hooks is None:
            return None
        if isinstance(hooks, _CompositeRunHooks):
            return await hooks.on_user_prompt(agent=self.agent, message=message)
        return await self.agent._hook_recorder.run(
            hooks,
            "run",
            "on_user_prompt",
            hooks.on_user_prompt(agent=self.agent, message=message),
            base_class=RunHooks,
        )

    # =========================================================================
    # Agentic loop helpers (safety checks + state)
    # =========================================================================

    @staticmethod
    def _response_has_tool_calls(messages: List[Message]) -> bool:
        """Check if the latest messages include tool call results.

        After Model.response()/response_stream() runs, any tool calls are
        executed and tool result messages are appended. We detect this by
        looking for tool-role messages after the last assistant message.
        """
        if not messages:
            return False
        # Walk backwards: if we find a tool message before an assistant message,
        # there were tool calls in this turn.
        for m in reversed(messages):
            if m.role == "tool":
                return True
            if m.role == "assistant":
                # Check if this assistant message has stop_after_tool_call
                if m.stop_after_tool_call:
                    return False
                # If assistant has tool_calls, tool results should follow
                if m.tool_calls:
                    return True
                return False
        return False

    @staticmethod
    def _check_death_spiral(messages: List[Message], state: "LoopState") -> bool:
        """Detect consecutive all-error tool turns (death spiral).

        Returns True if the death spiral threshold is reached.
        """
        # Find the latest batch of tool messages (after last assistant)
        tool_messages = []
        for m in reversed(messages):
            if m.role == "tool":
                tool_messages.append(m)
            elif m.role == "assistant":
                break

        if not tool_messages:
            return False

        all_errors = all(m.tool_call_error for m in tool_messages)
        if all_errors:
            state.consecutive_all_error_turns += 1
        else:
            state.consecutive_all_error_turns = 0

        return state.consecutive_all_error_turns >= state.death_spiral_threshold

    @staticmethod
    def _check_cost_budget(cost_tracker, max_cost_usd: Optional[float]) -> Optional[str]:
        """Check if the cost budget has been exceeded.

        Returns an error message if exceeded, None otherwise.
        """
        if max_cost_usd is None or cost_tracker is None:
            return None
        if cost_tracker.total_cost_usd >= max_cost_usd:
            return (
                f"Cost budget exceeded: ${cost_tracker.total_cost_usd:.4f} >= ${max_cost_usd:.4f}. Stopping execution."
            )
        return None

    @staticmethod
    def _check_stop_after_tool_call(messages: List[Message]) -> bool:
        """Check if any recent message has stop_after_tool_call flag."""
        for m in reversed(messages):
            if m.stop_after_tool_call:
                return True
            if m.role == "assistant" and not m.tool_calls:
                break
        return False

    @staticmethod
    def _get_last_assistant_message(messages: List[Message]) -> Optional[Message]:
        """Get the last assistant message from the message list."""
        for m in reversed(messages):
            if m.role == "assistant":
                return m
        return None

    def _loop_safety_checks(
        self,
        messages: List[Message],
        loop_state: "LoopState",
        agent: "Agent",
    ) -> Optional["LoopBreak"]:
        """Run all per-turn safety checks.

        Returns a ``LoopBreak`` (structured reason + human message) when the loop
        must abort, or ``None`` to continue. The message is intentionally NOT
        appended to the assistant content — the Runner records it on
        ``RunResponse.break_reason`` / ``break_message`` so it never leaks into
        the user-facing reply.
        """
        if self._check_death_spiral(messages, loop_state):
            return LoopBreak(
                RunBreakReason.DEATH_SPIRAL.value,
                "All tool calls have failed repeatedly. Stopping to prevent infinite loop.",
            )

        if loop_state.max_turns is not None and loop_state.turn_count >= loop_state.max_turns:
            return LoopBreak(
                RunBreakReason.MAX_TURNS.value,
                f"Reached max_turns={loop_state.max_turns} limit. Returning results collected so far.",
            )

        _cost_msg = self._check_cost_budget(agent.model._cost_tracker, agent._run_max_cost_usd)
        if _cost_msg:
            return LoopBreak(RunBreakReason.COST_BUDGET.value, _cost_msg)

        return None

    async def _recover_with_fallback(
        self,
        messages: List[Message],
        loop_state: "LoopState",
        agent: "Agent",
        break_reason: str,
    ) -> Optional[ModelResponse]:
        """After a loop break, do ONE tool-free inference with the fallback chain.

        Opt-in via ``Agent.fallback_on_break`` / ``RunConfig.fallback_on_break``.
        The full message history (including the failed tool calls) is replayed so
        the fallback model can see what went wrong — no synthetic hint is injected.
        Fallback models carry no bound tools (``update_model`` only binds tools to
        the primary), so the reply is naturally tool-free.

        Returns the recovery ModelResponse (its assistant message is appended to
        ``messages``), or None when break-recovery is disabled, no fallback is
        configured, or every fallback failed.
        """
        if not agent._run_fallback_on_break or not agent._run_fallback_models:
            return None

        fb_chain = list(agent._run_fallback_models)
        primary_id = agent.model.id if agent.model else "?"

        # Reuse the retry/fallback machinery: treat the first fallback as the
        # "primary" of this recovery call and the rest as its own fallbacks.
        saved_chain = agent._run_fallback_models
        agent._run_fallback_models = fb_chain[1:]
        try:
            recovery_call = await self._call_with_retry(
                fb_chain[0],
                messages,
                loop_state,
                agent,
                stream=False,
            )
            recovery = recovery_call.response
        except Exception as exc:
            logger.error(
                f"[fallback.on_break] recovery inference failed (reason={break_reason}, primary={primary_id}): {exc}"
            )
            return None
        finally:
            agent._run_fallback_models = saved_chain

        used_id = loop_state.last_used_model_id or fb_chain[0].id
        logger.warning(f"[fallback.on_break] recovered: primary={primary_id} -> used={used_id} (reason={break_reason})")
        cb = agent._event_callback
        if cb is not None:
            try:
                cb(
                    {
                        "type": "fallback.on_break",
                        "agent_name": agent.name or "Agent",
                        "primary_model": primary_id,
                        "used_model": used_id,
                        "break_reason": break_reason,
                    }
                )
            except Exception as e:
                logger.warning(f"event callback failed for fallback.on_break: {e}")
        return recovery

    @staticmethod
    def _inject_steering(messages: List[Message], agent: "Agent") -> None:
        """Flush pending user steering into the message list before an inference.

        Guidance buffered via ``agent.steer()`` (possibly from another thread)
        must reach the model on its very next call. We PREFER to fold it into the
        most recent tool result rather than appending a standalone user message:

        - It keeps a single trailing tool/user turn, so providers that enforce
          strict role alternation (Anthropic treats tool results as user-role)
          never see two consecutive user turns — both on this call and on later
          history replays, since ``messages`` is the persisted run transcript.
        - When there is no trailing tool result to fold into (e.g. the very first
          inference of a run), we fall back to appending a user message.

        Draining here — right before each inference — guarantees delivery: if a
        run ends before the buffer is flushed, the leftover guidance survives on
        the agent and is delivered at the start of the next run.
        """
        drained = agent._drain_steer()
        if not drained:
            return
        marker = "\n\n".join(f"[User guidance received while you were working]\n{guidance}" for guidance in drained)
        last = messages[-1] if messages else None
        if last is not None and last.role == "tool":
            existing = (last.content or "").rstrip()
            last.content = f"{existing}\n\n{marker}" if existing else marker
            logger.debug("Folded steering guidance into the latest tool result")
        else:
            messages.append(Message(role="user", content=marker))
            logger.debug("Injected steering guidance as a user message")

    @staticmethod
    def _tool_records_from_messages(
        messages: List[Message],
        *,
        fallback_compacted: bool = False,
        fallback_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role != "tool" or not msg.tool_name:
                continue
            record = {
                "tool_call_id": msg.tool_call_id,
                "tool_name": msg.tool_name,
                "tool_args": msg.tool_args,
                "content": msg.content,
                "tool_call_error": msg.tool_call_error or False,
                "metrics": msg.metrics if msg.metrics else {},
            }
            if fallback_compacted:
                record["fallback_compacted"] = True
                record["replay"] = False
                if fallback_model:
                    record["fallback_model"] = fallback_model
            records.append(record)
        return records

    @staticmethod
    def _compact_fallback_transaction(
        messages: List[Message],
        marker: Message,
        model_response: ModelResponse,
        used_model: "Model",
    ) -> List[Dict[str, Any]]:
        """Hide provider-specific fallback tool transcript from future model replay.

        ``marker`` is the first message the fallback produced. We resolve its
        live index by object identity so a compression pass that dropped/moved
        messages between transaction start and now can't make us fold the wrong
        segment: if the marker is gone we skip compaction rather than corrupt
        the transcript.
        """
        start = next((i for i, m in enumerate(messages) if m is marker), -1)
        if start < 0:
            return []
        segment = messages[start:]
        audit_tools = Runner._tool_records_from_messages(
            segment,
            fallback_compacted=True,
            fallback_model=used_model.id,
        )
        final_content = model_response.content
        if final_content is None:
            for msg in reversed(segment):
                if msg.role == "assistant" and not msg.tool_calls and msg.content is not None:
                    final_content = msg.content
                    break
        compacted = Message(
            role="assistant",
            content=final_content or "",
            reasoning_content=model_response.reasoning_content,
            finish_reason=model_response.finish_reason,
            provider_data={
                "fallback_compacted": True,
                "fallback_model": used_model.id,
            },
        )
        messages[start:] = [compacted]
        return audit_tools

    @staticmethod
    def _loop_post_response(
        messages: List[Message],
        model: "Model",
        loop_state: "LoopState",
        had_tool_calls: bool,
    ) -> bool:
        """Check if the agentic loop should continue after a response.

        Returns True to continue looping, False to break.
        """
        if not had_tool_calls:
            # Max-tokens recovery: if output was truncated, inject "Continue"
            _finish = model.last_finish_reason
            if _finish == "length" and loop_state.max_tokens_recovery_count < loop_state.max_tokens_recovery_limit:
                loop_state.max_tokens_recovery_count += 1
                messages.append(Message(role="user", content="Continue from where you left off."))
                logger.debug(
                    f"[loop] max_tokens recovery #{loop_state.max_tokens_recovery_count}: "
                    "injecting 'Continue' and looping"
                )
                return True  # continue
            logger.debug(f"[loop] exit: no tool_calls, finish_reason={_finish!r}, turn={loop_state.turn_count}")
            return False  # break

        # Check stop_after_tool_call
        for m in reversed(messages):
            if m.stop_after_tool_call:
                logger.debug(f"[loop] exit: stop_after_tool_call on {m.tool_name!r}, turn={loop_state.turn_count}")
                return False  # break
            if m.role == "assistant" and not m.tool_calls:
                break
        return True  # continue (tool calls processed, loop again)

    async def _execute_tool_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        agent: "Agent",
        model: "Model",
        tool_role: str = "tool",
        stream: bool = False,
    ) -> AsyncIterator[ModelResponse]:
        """Execute parsed tool calls with hooks, yielding ModelResponse events.

        This is the Runner-owned tool execution method. It wraps Model.run_function_calls()
        with proper hook dispatch using the Agent reference directly (no _agent_ref needed).
        """
        async for tool_response in model.run_function_calls(
            function_calls=function_calls,
            function_call_results=function_call_results,
            tool_role=tool_role,
        ):
            yield tool_response

    async def _handle_tool_calls_in_runner(
        self,
        messages: List[Message],
        agent: "Agent",
        model: "Model",
        stream: bool = False,
    ) -> ToolHandlingResult:
        """Check for tool calls in the last assistant message, execute them, format results."""
        assistant_msg = self._get_last_assistant_message(messages)
        if assistant_msg is None or not assistant_msg.tool_calls:
            return ToolHandlingResult(False, [])

        # Parse tool calls (provider-specific)
        function_calls, provider_metadata = model.parse_tool_calls(assistant_msg, messages, tool_role="tool")
        if not function_calls:
            # All tool calls had errors (already appended to messages by parse_tool_calls)
            return ToolHandlingResult(True, [])

        # Log what the LLM asked for this turn — primary signal when diagnosing
        # "tool loop too many iterations" or "model keeps retrying the same tool".
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            _names = [fc.function.name for fc in function_calls]
            _first_args = {}
            if function_calls:
                _args = function_calls[0].arguments or {}
                if isinstance(_args, dict):
                    _first_args = {
                        k: (str(v)[:80] + "..." if len(str(v)) > 80 else v) for k, v in list(_args.items())[:3]
                    }
            logger.debug(f"[tool-calls] LLM requested {len(function_calls)} tool(s): {_names} first_args={_first_args}")

        # Execute tool calls
        function_call_results: List[Message] = []
        tool_role = provider_metadata.get("tool_role", "tool")
        async for _tool_resp in self._execute_tool_calls(
            function_calls=function_calls,
            function_call_results=function_call_results,
            agent=agent,
            model=model,
            tool_role=tool_role,
            stream=stream,
        ):
            pass  # Events consumed by streaming loop if needed

        tool_records = self._tool_records_from_messages(function_call_results)

        # Format and append results (provider-specific)
        model.format_tool_results(function_call_results, messages, provider_metadata)
        return ToolHandlingResult(True, tool_records)

    async def _handle_tool_calls_in_runner_stream(
        self,
        messages: List[Message],
        agent: "Agent",
        model: "Model",
    ) -> AsyncIterator[ModelResponse]:
        """Streaming version: execute tool calls and yield ModelResponse events.

        Yields tool_call_started / tool_call_completed events for streaming consumers.
        Returns after all tool calls are done.
        """
        assistant_msg = self._get_last_assistant_message(messages)
        if assistant_msg is None or not assistant_msg.tool_calls:
            return

        function_calls, provider_metadata = model.parse_tool_calls(assistant_msg, messages, tool_role="tool")
        if not function_calls:
            return

        function_call_results: List[Message] = []
        tool_role = provider_metadata.get("tool_role", "tool")
        async for tool_resp in self._execute_tool_calls(
            function_calls=function_calls,
            function_call_results=function_call_results,
            agent=agent,
            model=model,
            tool_role=tool_role,
            stream=True,
        ):
            yield tool_resp

        model.format_tool_results(function_call_results, messages, provider_metadata)

    @staticmethod
    def _prepare_model_for_runner_call(current: "Model", primary: "Model") -> None:
        """Bind runner-owned tool context to the concrete model used for this call."""
        current.run_tools = False
        if current is primary:
            return
        current.tools = primary.tools
        current.functions = primary.functions
        current.tool_choice = primary.tool_choice
        current.tool_call_limit = primary.tool_call_limit
        current.max_concurrent_tools = primary.max_concurrent_tools
        current._cost_tracker = primary._cost_tracker

    @staticmethod
    def _isolate_fallback_models(models: List["Model"]) -> List["Model"]:
        """Give this run its own fallback model instances.

        A fallback model is commonly shared across agents/runs (one backup model
        handed to many agents). Since ``_prepare_model_for_runner_call`` rebinds
        tools/cost_tracker onto the concrete model used for a call, two concurrent
        runs sharing the same fallback object would stomp on each other. Shallow-
        clone per run (the canonical Model clone), reset runtime state, and drop
        client refs so each run owns an isolated instance bound to its own loop.
        """
        isolated: List["Model"] = []
        for source in models:
            # Only real Model instances are clonable. Anything else (e.g. test
            # doubles) is passed through untouched so it keeps its identity.
            if not isinstance(source, Model):
                isolated.append(source)
                continue
            cloned = copy.copy(source)
            cloned.metrics = {}
            cloned.usage = Usage()
            for attr in ("client", "http_client", "async_client"):
                if hasattr(cloned, attr):
                    setattr(cloned, attr, None)
            isolated.append(cloned)
        return isolated

    @staticmethod
    def _resolve_max_api_retry(agent: "Agent", config: RunConfig) -> int:
        """Resolve and validate Runner-level API call attempts for one run."""
        value = config.max_api_retry if config.max_api_retry is not None else agent.max_api_retry
        if value < 1:
            raise ValueError("max_api_retry must be >= 1; use 1 to disable Runner-level same-model retry")
        return value

    @staticmethod
    async def _maybe_compress_messages(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
    ) -> None:
        """Run the multi-stage compression pipeline before each LLM call.

        Stages ordered cheapest-first (mirrors CC's queryLoop pre-processing):
          Stage 1 - Tool result budget (free, O(n))
          Stage 2 - Micro-compact (free, O(n))
          Stage 3 - Rule-based compress (free, O(n))
          Stage 4 - Auto-compact (costly, LLM summarisation)
          Stage 5 (reactive compact) is handled in _call_with_retry on API error.
        """
        cb = agent._event_callback
        agent_name = agent.name or "Agent"

        # Stage 1: tool result budget (persist oversized results to disk)
        _sid = agent.run_id or "default"
        _uid = agent.workspace.user_id if agent.workspace is not None else None
        _recent_tools = [m for m in messages if m.role == "tool" and not m.compressed_content]
        if _recent_tools:
            enforce_tool_result_budget(
                tool_results=_recent_tools,
                session_id=_sid,
                user_id=_uid,
            )

        # Stage 2: micro-compact (clear old tool results, free)
        n = micro_compact(messages)
        if n:
            logger.debug(f"Stage 2 (micro-compact): cleared {n} old tool result(s)")
            if cb is not None:
                cb(
                    {
                        "type": "compact.micro",
                        "agent_name": agent_name,
                        "cleared": n,
                    }
                )

        # Stage 3 & 4 require CompressionManager
        if not agent.tool_config.compress_tool_results:
            return
        cm = agent.tool_config.compression_manager
        if cm is None:
            return

        async def _fire_compact_hooks(event: str) -> None:
            if agent._run_hooks is not None:
                fn = getattr(agent._run_hooks, event, None)
                if fn is not None:
                    await fn(agent=agent, messages=messages)

        # Stage 3: rule-based compress (truncate + drop old rounds, free)
        if cm.should_compress(messages, tools=model.tools, model=model):
            await _fire_compact_hooks("on_pre_compact")
            logger.debug("Stage 3 (rule-based compress): truncating + dropping old messages")
            before = len(messages)
            t0 = time.monotonic()
            await cm.compress(
                messages,
                tools=model.tools,
                model=model,
                trigger="threshold",
                task_anchor=agent.task_anchor,
                user_id=_uid,
            )
            compression_report = cm.get_stats().get("last_report")
            if compression_report and agent.run_response is not None:
                agent.run_response.metrics = agent.run_response.metrics or {}
                agent.run_response.metrics["compression"] = {"last_report": compression_report}
            await _fire_compact_hooks("on_post_compact")
            if cb is not None:
                cb(
                    {
                        "type": "compact.rule_based",
                        "agent_name": agent_name,
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                        "report": compression_report,
                    }
                )

        # Stage 4: auto-compact via LLM summarisation.
        # auto_compact() returns False fast when threshold not met; only fire
        # events when it actually compresses (avoids per-turn spam).
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model)
        if compacted:
            logger.debug("Stage 4 (auto-compact): conversation summarised by LLM")
            await _fire_compact_hooks("on_post_compact")
            if cb is not None:
                cb(
                    {
                        "type": "compact.auto",
                        "agent_name": agent_name,
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                    }
                )

    @staticmethod
    async def _try_reactive_compact(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
    ) -> bool:
        """Attempt emergency compression on prompt_too_long. Returns True if compacted.

        Stage 5 (reactive compact) is handled in _call_with_retry on API error.
        """
        cm = agent.tool_config.compression_manager if agent is not None else None
        if cm is None:
            return False
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model, force=True)
        if compacted:
            logger.info("Reactive compact triggered (prompt_too_long) -- retrying")
            cb = agent._event_callback
            if cb is not None:
                cb(
                    {
                        "type": "compact.reactive",
                        "agent_name": agent.name or "Agent",
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                    }
                )
            return True
        return False

    @staticmethod
    def _persist_assistant_tool_calls(agent: "Agent") -> None:
        """Persist the turn's assistant tool-call messages to the session log.

        The session log otherwise records only ``user`` -> ``tool`` results ->
        final ``assistant`` text. The intermediate assistant messages that
        CARRY the ``tool_calls`` are never written, so on resume the tool
        results become orphaned (a ``tool`` message with no preceding assistant
        holding ``tool_calls``) and the provider rejects the replay with
        "messages with role 'tool' must be a response to a preceding message
        with 'tool_calls'".

        A single agentic turn may issue tool calls across several assistant
        rounds (e.g. ``read_file`` then ``grep``). For each round we must log
        ``assistant(tool_calls)`` immediately followed by its ``tool`` result,
        preserving the exact interleaving — OpenAI-compatible providers require
        every ``tool`` message to immediately follow the assistant message that
        requested it. Grouping all assistants before all tools (the previous
        implementation) re-introduced the same 400 on resume for any multi-round
        tool turn.

        We walk ``run_response.messages`` (this turn only, see the
        ``num_input_messages`` slice in ``_run_impl``) in order: for each
        assistant-with-tool_calls we log the assistant, and for each
        ``role="tool"`` message we log the matching tool result (rich metadata
        taken from ``run_response.tools`` by id). ``_build_messages`` already
        lists ``tool_calls`` / ``tool_call_id`` in its replay fields, so the
        replay reconstructs a valid, interleaved sequence.

        Called once per turn (before the final assistant text is logged) so the
        JSONL order is
        ``user -> assistant(tool_calls) -> tool -> assistant(tool_calls) -> tool -> ... -> assistant(text)``.
        """
        if agent._session_log is None:
            return
        tool_by_id = {
            tc.get("tool_call_id"): tc
            for tc in (agent.run_response.tools or [])
        }
        _functions = (agent.model.functions or {}) if agent.model else {}
        for msg in agent.run_response.messages or []:
            if not isinstance(msg, Message):
                continue
            if msg.role == "assistant" and msg.tool_calls:
                _text = msg.content if isinstance(msg.content, str) else ""
                agent._session_log.append(
                    "assistant",
                    _text,
                    tool_calls=msg.tool_calls,
                )
            elif msg.role == "tool":
                _tc = tool_by_id.get(msg.tool_call_id)
                if _tc is not None:
                    _tool_content = _tc.get("content", "") or ""
                    if len(_tool_content) > 2000:
                        _tool_content = _tool_content[:2000] + "\n... [truncated]"
                    _origin_meta: Dict[str, Any] = {}
                    _fn = _functions.get(_tc.get("tool_name", ""))
                    if _fn is not None and _fn.origin is not None:
                        _origin_meta["origin_type"] = _fn.origin.type
                        if _fn.origin.provider_name:
                            _origin_meta["origin_provider_name"] = _fn.origin.provider_name
                        if _fn.origin.agent_name:
                            _origin_meta["origin_agent_name"] = _fn.origin.agent_name
                        if _fn.origin.source_tool_name:
                            _origin_meta["origin_source_tool_name"] = _fn.origin.source_tool_name
                    agent._session_log.append(
                        "tool_audit" if _tc.get("replay") is False else "tool",
                        _tool_content,
                        tool_name=_tc.get("tool_name", ""),
                        tool_call_id=_tc.get("tool_call_id", ""),
                        is_error=_tc.get("tool_call_error", False),
                        fallback_compacted=_tc.get("fallback_compacted", False),
                        fallback_model=_tc.get("fallback_model"),
                        replay=_tc.get("replay", True),
                        **_origin_meta,
                    )
                else:
                    # Tool message without matching FunctionCall metadata: log a
                    # minimal entry so resume still has a valid assistant->tool pair.
                    agent._session_log.append(
                        "tool",
                        msg.content if isinstance(msg.content, str) else "",
                        tool_call_id=msg.tool_call_id or "",
                    )

    @staticmethod
    def _strip_tool_artifacts(msgs: List[Message], *, drop_system: bool = False) -> List[Message]:
        """Drop tool-call/tool-result artifacts (OpenAI *and* Anthropic wire formats).

        Keeps only plain user/assistant text so history recorded under one
        provider can be replayed on another. Handles both OpenAI-style
        (role="tool" + assistant.tool_calls) and Anthropic-style (list content
        blocks with tool_use/tool_result) encodings. Used to recover from
        cross-provider tool-call format mismatches — see
        _sanitize_tool_history_after_error.
        """
        from agentica.agent.history_filter import strip_all_tool_artifacts

        return strip_all_tool_artifacts(msgs, drop_system=drop_system)

    @staticmethod
    def _sanitize_tool_history_after_error(agent: "Agent", messages: List[Message]) -> None:
        """Strip tool-call artifacts from history after a tool-history API error.

        Recovery path for resuming a session whose history was recorded under
        a different model provider (e.g. Claude) on a provider that rejects
        the resulting 'tool' role messages (e.g. "Messages with role 'tool'
        must be a response to a preceding message with 'tool_calls'"). Only
        user/assistant text is needed going forward, so tool_calls/tool
        results are dropped entirely — both from working-memory-backed
        history (so future turns stay clean) and the in-flight ``messages``
        list (so the immediate retry succeeds).
        """
        for run in agent.working_memory.runs:
            if run.response and run.response.messages:
                run.response.messages = Runner._strip_tool_artifacts(run.response.messages, drop_system=True)
        messages[:] = Runner._strip_tool_artifacts(messages)

    @staticmethod
    async def _call_with_retry(
        model: "Model",
        messages: List[Message],
        state: "LoopState",
        agent: "Agent",
        *,
        stream: bool = False,
    ):
        """Call model.response()/response_stream() with retry, reactive compact,
        and cross-provider fallback chain.

        Returns:
            ModelCallResult(response=ModelResponse, used_model=...) for non-stream.
            ModelCallResult(response=async iterator, used_model=...) for stream.

        Per-call fallback (not per-run): each invocation starts from the primary
        ``model`` argument; ``agent._run_fallback_models`` are tried in order
        when triggered by:

          1. ``finish_reason in CONTENT_FILTER_FINISH_REASONS`` (non-stream only;
             stream cannot detect this until consumed).
          2. Exception whose text matches ``CONTENT_FILTER_HINTS`` (any provider
             that raises instead of flagging a finish reason).
          3. Fallback-only API error (``FALLBACK_ONLY_SUBSTRINGS``), such as
             connection failure / 502 / 503 / bad gateway.
          4. Retryable API error (``RETRYABLE_SUBSTRINGS``) that exhausted local
             exponential backoff on the current model.

        Reactive compact (``prompt_too_long``) is NOT a fallback trigger —
        switching providers does not solve a too-long context.
        """
        # Dedup by object identity, not ``model.id``: two distinct instances may
        # share an id but differ in api_key/base_url (a legitimate second
        # fallback). Identity only drops the same object listed twice (e.g. the
        # primary also present in the fallback chain).
        candidates: List["Model"] = []
        seen_candidate_objs: set[int] = set()
        for candidate in [model, *(agent._run_fallback_models or [])]:
            if id(candidate) in seen_candidate_objs:
                continue
            candidates.append(candidate)
            seen_candidate_objs.add(id(candidate))
        last_exc: Optional[BaseException] = None
        # Reset per-call bookkeeping. last_used_* reflects the model that
        # actually produced the response returned by THIS call invocation.
        state.last_used_model_id = None
        state.last_used_model_idx = -1
        primary_id = model.id
        trigger: Optional[str] = None  # "content_filter" | "fallback_only" | "exhausted_retry"

        def _emit_fallback_recovery(used_model_id: str, used_idx: int) -> None:
            """Audit-log + event-bus a successful fallback recovery.

            Only fires when the answer came from a fallback (idx > 0). Gives
            ops a single grep-able marker for "this run was rescued by a
            fallback model" without scraping retry/switch warnings.
            """
            logger.warning(
                f"[fallback.recovered] primary={primary_id} -> used={used_model_id} (idx={used_idx}, trigger={trigger})"
            )
            cb = agent._event_callback
            if cb is not None:
                try:
                    cb(
                        {
                            "type": "fallback.recovered",
                            "agent_name": agent.name or "Agent",
                            "primary_model": primary_id,
                            "used_model": used_model_id,
                            "fallback_index": used_idx,
                            "trigger": trigger,
                        }
                    )
                except Exception as e:
                    logger.warning(f"event callback failed for fallback.recovered: {e}")

        for model_idx, current in enumerate(candidates):
            is_fallback = model_idx > 0
            if is_fallback:
                logger.warning(
                    f"[fallback] switching to {current.id} ({model_idx}/{len(candidates) - 1}) trigger={trigger}"
                )

            for attempt in range(state.max_api_retry):
                message_checkpoint = len(messages)
                Runner._prepare_model_for_runner_call(current, model)
                try:
                    if stream:
                        # Stream: defer content_filter detection to the consumer.
                        # Exception-based fallbacks (timeout/5xx/content_filter
                        # raised at connect time) are still handled below.
                        # Record optimistically; consumer may flip later if
                        # finish_reason turns out to be content_filter.
                        state.last_used_model_id = current.id
                        state.last_used_model_idx = model_idx
                        if is_fallback:
                            _emit_fallback_recovery(current.id, model_idx)
                        return ModelCallResult(
                            response=current.response_stream(messages=messages),
                            used_model=current,
                            used_fallback=is_fallback,
                        )

                    resp = await current.response(messages=messages)

                    # Non-stream: content_filter is a normal-return finish_reason.
                    _fr = (resp.finish_reason or "").lower()
                    if _fr in state.CONTENT_FILTER_FINISH_REASONS:
                        logger.warning(
                            f"[content_filter] {current.id} returned "
                            f"finish_reason={resp.finish_reason!r}; "
                            f"trying next fallback"
                        )
                        # Model.response() may already have appended the blocked
                        # assistant message. Fallback must retry the same clean
                        # prompt, not continue after the primary model's refusal.
                        del messages[message_checkpoint:]
                        trigger = "content_filter"
                        last_exc = RuntimeError(f"content_filter on {current.id} (finish_reason={resp.finish_reason})")
                        break  # exit retry loop, go to next model

                    # Success: stamp who actually answered.
                    state.last_used_model_id = current.id
                    state.last_used_model_idx = model_idx
                    if is_fallback:
                        _emit_fallback_recovery(current.id, model_idx)
                    return ModelCallResult(
                        response=resp,
                        used_model=current,
                        used_fallback=is_fallback,
                    )

                except Exception as exc:
                    del messages[message_checkpoint:]
                    last_exc = exc
                    err = str(exc).lower()

                    # One-shot recovery: cross-provider tool-call/tool-result
                    # format mismatch (e.g. resuming a session recorded under
                    # a different model provider). Strip tool artifacts and
                    # retry immediately — independent of max_api_retry /
                    # fallback-candidate bookkeeping, since this is a
                    # guaranteed-safe structural fix, not a flaky transient.
                    is_tool_history_error = any(h in err for h in state.TOOL_HISTORY_HINTS)
                    if is_tool_history_error and not state.tool_history_sanitized_done:
                        state.tool_history_sanitized_done = True
                        logger.warning(
                            f"[tool_history] {current.id} rejected tool-call "
                            f"history (likely cross-model resume); stripping "
                            f"tool messages and retrying once: {exc}"
                        )
                        Runner._sanitize_tool_history_after_error(agent, messages)
                        try:
                            if stream:
                                state.last_used_model_id = current.id
                                state.last_used_model_idx = model_idx
                                if is_fallback:
                                    _emit_fallback_recovery(current.id, model_idx)
                                return ModelCallResult(
                                    response=current.response_stream(messages=messages),
                                    used_model=current,
                                    used_fallback=is_fallback,
                                )
                            resp = await current.response(messages=messages)
                            state.last_used_model_id = current.id
                            state.last_used_model_idx = model_idx
                            if is_fallback:
                                _emit_fallback_recovery(current.id, model_idx)
                            return ModelCallResult(response=resp, used_model=current, used_fallback=is_fallback)
                        except Exception as exc2:
                            del messages[message_checkpoint:]
                            last_exc = exc2
                            err = str(exc2).lower()
                            # Fall through to the normal classification below
                            # with the post-sanitize exception.

                    # Reactive compact: prompt_too_long -> emergency compress.
                    # Only attempted on the primary model; fallbacks inherit the
                    # already-compacted message list.
                    is_too_long = any(h in err for h in state.PROMPT_TOO_LONG_HINTS)
                    if is_too_long and not state.reactive_compact_done and not is_fallback:
                        state.reactive_compact_done = True
                        if await Runner._try_reactive_compact(messages, agent, current):
                            continue

                    # Content filter raised as exception (some providers do this
                    # instead of setting finish_reason). No point retrying same
                    # model — moderation is deterministic.
                    is_content_filter = any(h in err for h in state.CONTENT_FILTER_HINTS)
                    if is_content_filter:
                        logger.warning(f"[content_filter] {current.id} raised content_filter exception: {exc}")
                        trigger = "content_filter"
                        break  # next model

                    # Hard outage errors: do not retry the same model. Switch
                    # directly to the next fallback model when configured.
                    is_fallback_only = any(r in err for r in state.FALLBACK_ONLY_SUBSTRINGS)
                    if is_fallback_only:
                        logger.warning(f"[fallback] {current.id} hit non-retry outage: {exc}; trying next fallback")
                        trigger = "fallback_only"
                        break

                    # Retryable transient errors: backoff within current model.
                    # Merge SDK defaults with model-level + env-level user
                    # extensions, so deployment-specific proxy markers
                    # (e.g. a private "venus_error") become retryable
                    # without touching SDK source.
                    _retryable = current.get_retryable_substrings(state.RETRYABLE_SUBSTRINGS)
                    is_retryable = any(r in err for r in _retryable)
                    if is_retryable and attempt < state.max_api_retry - 1:
                        wait = (2**attempt) + random.uniform(0.0, 1.0)
                        logger.warning(
                            f"[APIRetry] {current.id} attempt "
                            f"{attempt + 1}/{state.max_api_retry}, "
                            f"retrying same model in {wait:.1f}s: {exc}"
                        )
                        await asyncio.sleep(wait)
                        continue

                    if is_retryable:
                        # Exhausted local attempts → fall through to next model.
                        logger.warning(
                            f"[APIRetry] {current.id} exhausted {state.max_api_retry} attempts; trying next fallback"
                        )
                        trigger = "exhausted_retry"
                        break

                    # Truly non-retryable (auth, malformed request, etc.).
                    # Fallback would not help — propagate immediately.
                    raise

        # All models in the chain failed.
        logger.error(f"[fallback] All {len(candidates)} models exhausted. Last error: {last_exc}")
        raise RuntimeError(
            f"LLM call failed across {len(candidates)} model(s) "
            f"(primary + {len(candidates) - 1} fallback). Last error: {last_exc}"
        ) from last_exc

    def save_run_response_to_file(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        save_response_to_file: Optional[str] = None,
    ) -> None:
        _save_path = save_response_to_file
        if _save_path is None or self.agent.run_response is None:
            return
        message_str = None
        if message is not None:
            if isinstance(message, str):
                message_str = message
            else:
                logger.warning("Did not use message in output file name: message is not a string")
        try:
            fn = _save_path.format(name=self.agent.name, message=message_str)
            fn_path = Path(fn)
            if not fn_path.parent.exists():
                fn_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(self.agent.run_response.content, str):
                fn_path.write_text(self.agent.run_response.content)
            else:
                fn_path.write_text(json.dumps(self.agent.run_response.content, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(
                f"Failed to save output to file '{save_response_to_file}': {e} "
                f"[agent={self.agent.identifier}, run_id={self.agent.run_id}]"
            )

    def _aggregate_metrics_from_run_messages(self, messages: List[Message]) -> Dict[str, Any]:
        aggregated_metrics: Dict[str, Any] = defaultdict(list)
        for m in messages:
            if m.role == "assistant" and m.metrics is not None:
                for k, v in m.metrics.items():
                    aggregated_metrics[k].append(v)
        return aggregated_metrics

    def generic_run_response(
        self, content: Optional[str] = None, event: RunEvent = RunEvent.run_response
    ) -> RunResponse:
        return RunResponse(
            run_id=self.agent.run_id,
            agent_id=self.agent.agent_id,
            content=content,
            tools=self.agent.run_response.tools,
            images=self.agent.run_response.images,
            videos=self.agent.run_response.videos,
            model=self.agent.run_response.model,
            messages=self.agent.run_response.messages,
            reasoning_content=self.agent.run_response.reasoning_content,
            extra_data=self.agent.run_response.extra_data,
            event=event.value,
        )

    def _persist_interrupted_turn(
        self,
        agent,
        message: Any,
        messages: Any,
        user_messages: List[Message],
        system_message: Optional[Message],
        messages_for_model: List[Message],
        num_input_messages: int,
        model_response: Any,
    ) -> None:
        """On user cancel, preserve the turn instead of discarding it.

        Mirrors the success-path memory + session-log persistence: the user
        question and the partial assistant answer (up to the interruption
        point) are kept as a completed Q&A turn, with an ``[用户中断了回答]``
        marker appended. Only applies to message-based runs (``messages`` is
        None) — the CLI path; pre-built ``messages`` runs manage their own
        history and are left untouched.
        """
        if messages is not None:
            return
        # Capture the partial streamed content as the authoritative answer.
        if model_response.content:
            agent.run_response.content = model_response.content
        partial = agent.run_response.content or ""
        marker = "[用户中断了回答]"
        persisted = (f"{partial}\n\n{marker}") if partial else marker
        agent.run_response.content = persisted

        # The model layer appends the assistant message only AFTER the stream
        # completes, so on a mid-stream cancel it's absent from
        # messages_for_model. Patch the turn's last assistant message if one
        # exists (cancel during tool exec / between turns), else synthesize one
        # (cancel mid-stream) so /history surfaces the partial + marker.
        turn_msgs = list(messages_for_model[num_input_messages:])
        last_asst = None
        for _m in reversed(turn_msgs):
            if isinstance(_m, Message) and _m.role == "assistant":
                last_asst = _m
                break
        if last_asst is not None:
            last_asst.content = persisted
        else:
            turn_msgs.append(Message(role="assistant", content=persisted))

        # working_memory so /history shows this exchange.
        if system_message is not None:
            agent.working_memory.add_system_message(
                system_message,
                system_message_role=agent.prompt_config.system_message_role,
            )
        agent.working_memory.add_messages(messages=(user_messages + turn_msgs))
        agent_run = AgentRun(response=agent.run_response)
        if user_messages:
            agent_run.message = user_messages[0]
            agent_run.messages = list(user_messages)
        agent.working_memory.add_run(agent_run)

        # session log so /resume restores this turn.
        if agent._session_log is not None:
            _user_text = None
            if isinstance(message, str):
                _user_text = message
            elif isinstance(message, Message):
                _user_text = message.content if isinstance(message.content, str) else str(message.content)
            if _user_text:
                agent._session_log.append("user", _user_text)
            # Log assistant tool-call messages AND their tool results in the
            # exact interleaved order so /resume rebuilds a valid
            # assistant(tool_calls)->tool sequence instead of orphaned tools.
            self._persist_assistant_tool_calls(agent)
            agent._session_log.append("assistant", persisted, finish_reason="cancelled")

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
        stream_intermediate_steps: bool = False,
        save_response_to_file: Optional[str] = None,
        hooks: Optional[RunHooks] = None,
        enabled_tools: Optional[List[str]] = None,
        enabled_skills: Optional[List[str]] = None,
        source: RunSource = RunSource.sdk,
        **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Unified execution engine.

        This is the ONLY core runtime for base `Agent`.

        - Non-streaming users should call `run()` (which consumes this generator).
        - Streaming users should call `run_stream()` (which returns this generator).

        All LLM calls within this run are grouped under a single Langfuse trace (if enabled).

        Args:
            message: Primary user input for this turn. In message mode the
                Runner sends ``[system] + [working_memory.history] + [message]``.
            messages: Full transcript for this run. Mutually exclusive with
                ``message``. In messages mode the Runner sends
                ``[system] + [messages]`` and does not append working memory
                history or persist the transcript back into working memory.

        Final message order sent to the model:
            * message mode:  [system_message] + [history] + [message]
            * messages mode: [system_message] + [messages]
        """
        agent = self.agent

        # Pre-generate this run's id and bind it to the log ContextVar *before*
        # any log records are emitted. Every log line for this run (including
        # the ``[user] -> ...`` chat line, tool calls, and any child asyncio
        # Tasks that inherit our Context) will carry a ``run=<8hex>`` prefix,
        # letting concurrent runs be reconstructed from a shared log file.
        #
        # We deliberately do NOT reset the ContextVar on exit: wrapping the
        # entire async-generator body in try/finally is more surgery than the
        # benefit warrants, and the three public entry points (``run``,
        # ``arun``, ``run_stream``) already give clean isolation for typical
        # usage — ``run`` starts a fresh event loop, and separate ``arun``
        # calls typically run on their own asyncio Tasks. Users who chain
        # multiple ``arun`` calls on the *same* Task and need strict per-run
        # isolation can wrap them in ``agentica.utils.log.bind_run_context``.
        _pregen_run_id = str(uuid4())
        _run_id_var.set(_short_run_id(_pregen_run_id))
        if agent._parent_run_id:
            _parent_run_id_var.set(_short_run_id(agent._parent_run_id))

        async def _run_core() -> AsyncIterator[RunResponse]:
            nonlocal message  # on_user_prompt hook may reassign message
            # Guard: warn if this agent instance is already running concurrently.
            # Agent is not thread-safe — concurrent runs share mutable state
            # (run_id, run_response, _run_hooks, _enabled_tools, model.functions).
            # Swarm autonomous mode avoids this by cloning agents before parallel dispatch.
            if agent._running:
                logger.warning(
                    f"Agent '{agent.identifier}' is already running. "
                    "Concurrent reuse of the same Agent instance is not safe — "
                    "run_id, run_response, and model state will be overwritten. "
                    "Create a separate Agent instance for concurrent execution."
                )

            # Guard: early return if no input provided
            if message is not None and messages is not None:
                raise ValueError("message and messages are mutually exclusive")
            if messages is not None and (audio is not None or images is not None or videos is not None):
                raise ValueError("audio/images/videos can only be used with message, not messages")

            if message is None and (messages is None or len(messages) == 0):
                logger.warning(
                    f"Agent '{agent.identifier}' called with no message and no messages. Returning empty response."
                )
                yield RunResponse(
                    run_id=str(uuid4()),
                    agent_id=agent.agent_id,
                    content="",
                    event=RunEvent.run_response.value,
                )
                return

            # Open the steering window under _steer_lock (flips _running True
            # and clears any stale guidance from a prior run). Keeps steer()'s
            # accept/reject decision atomic w.r.t. run start/end.
            agent._begin_steer_window()
            # CHAT-level lines are the conversation flow record. Print the
            # full user message — turn boundaries are *the* thing to keep at
            # CHAT level, so truncating them defeats the purpose of the level.
            logger.chat(f"[user] -> {agent.identifier}: {str(message) if message else '<no message>'}")
            # Capture asyncio handles so cancel() can hard-cancel from another thread
            try:
                agent._run_loop = asyncio.get_running_loop()
                agent._run_task = asyncio.current_task()
            except RuntimeError:
                pass
            # SDK-first run lifecycle (arch_v5.md Phase 0/1/3):
            # build RunContext + TaskAnchor BEFORE any try/except so the
            # original goal is anchored from the very first event we emit.
            #
            # TaskAnchor is *session-scoped*, not run-scoped. The first run of
            # a session pins the user's original goal; subsequent runs in the
            # same session reuse it so retrieval and the prompt's "Original
            # Task" block stay stable across multi-turn conversations.
            # When session_id changes, the anchor resets so a brand-new
            # conversation can establish its own original task.
            if agent.task_anchor is None or agent._anchor_session_id != agent.session_id:
                # Standing-goal awareness (P1 S1): if the session has a
                # persisted ACTIVE goal, bind the anchor to it instead of
                # the (possibly transient) current message. This makes goal
                # semantics work for any SDK entry point — gateway, ACP,
                # cron, scripts — not just the CLI.
                _persisted_goal: Optional[Dict[str, Any]] = None
                if agent._session_log is not None:
                    try:
                        _persisted_goal = agent._session_log.load_goal()
                    except Exception as exc:  # pragma: no cover - I/O edge
                        logger.debug("load_goal failed at run start: %s", exc)
                        _persisted_goal = None

                if (
                    _persisted_goal is not None
                    and _persisted_goal.get("status") == "active"
                    and _persisted_goal.get("objective")
                ):
                    objective = str(_persisted_goal["objective"])
                    agent.task_anchor = TaskAnchor(
                        goal=objective,
                        source_query=objective,
                        source="goal",
                    )
                else:
                    # source defaults to "message" — not rendered into the
                    # system prompt, only used as a retrieval query. See
                    # TaskAnchor.to_prompt_block for the gate.
                    anchor_input = message
                    if anchor_input is None and messages:
                        for candidate in reversed(messages):
                            if isinstance(candidate, Message) and candidate.role == "user":
                                anchor_input = candidate
                                break
                            if isinstance(candidate, dict) and candidate.get("role") == "user":
                                anchor_input = candidate
                                break
                    agent.task_anchor = TaskAnchor.from_message(anchor_input)
                agent._anchor_session_id = agent.session_id
            _anchor = agent.task_anchor

            _run_source = RunSource.subagent if agent._parent_run_id else source
            _run_ctx = RunContext(
                run_id=_pregen_run_id,
                session_id=agent.session_id,
                parent_run_id=agent._parent_run_id,
                agent_id=agent.agent_id,
                source=_run_source,
                task_anchor=_anchor,
            )
            agent.run_context = _run_ctx
            agent.run_id = _run_ctx.run_id
            try:  # R-01 fix: ensure _running is reset on any exception
                agent.stream = stream and agent.is_streamable
                agent.stream_intermediate_steps = stream_intermediate_steps and agent.stream
                agent.run_response = RunResponse(run_id=agent.run_id, agent_id=agent.agent_id)
                agent._hook_recorder.reset()
                # Guards the cancel handler: once the success path has persisted
                # this turn to working_memory, a Ctrl+C during the post-completion
                # hooks must NOT re-persist (which would double-add the exchange
                # and wrongly stamp an interruption marker on a finished answer).
                _memory_persisted = False
                _run_ctx.mark_running()
                self._emit_event(
                    RunEventType.run_started,
                    {
                        "agent_name": agent.name or "Agent",
                        "source_query": _anchor.source_query,
                        "session_id": agent.session_id,
                    },
                )

                # --- Session resume (CC-style JSONL) ---
                # On first run, if a session log exists AND working_memory has
                # not already been hydrated (e.g. by `/resume` in the CLI, which
                # eagerly loads history), replay messages from the last compact
                # boundary into working_memory as reconstructed AgentRuns so the
                # prompt builder (get_messages_from_last_n_runs) actually sees
                # them. Just appending to working_memory.messages is not enough
                # — that field is for archive/trim, not prompt assembly.
                if (
                    agent._session_log is not None
                    and agent._session_log.exists()
                    and len(agent.working_memory.runs) == 0
                    and len(agent.working_memory.messages) == 0
                ):
                    resumed_messages = agent._session_log.load()
                    if resumed_messages:
                        runs_built = agent.working_memory.hydrate_runs_from_history(resumed_messages)
                        logger.debug(
                            f"Session resumed from JSONL: {len(resumed_messages)} messages, "
                            f"{runs_built} runs reconstructed"
                        )

                # --- Initialise CostTracker for this run ---
                _cost_tracker = CostTracker()
                agent.run_response.cost_tracker = _cost_tracker

                # --- Freeze workspace snapshots on first run (prompt cache stability) ---
                # Hermes-style: freeze context + memory at session start so the
                # system prompt prefix stays identical across turns.
                if (
                    agent.workspace is not None
                    and agent.workspace.exists()
                    and agent.workspace.get_frozen_context() is None
                ):
                    # Use the run's TaskAnchor as the freeze query so memory
                    # retrieval is bound to the *original* goal, not whatever
                    # `message` happens to be on subsequent runs.
                    await agent.workspace.freeze_snapshots(query=_anchor.source_query)

                # Set query-level tool/skill filtering (cleared after run)
                agent._enabled_tools = enabled_tools
                agent._enabled_skills = enabled_skills

                # Merge default run hooks (e.g. auto-archive) with user-provided hooks
                effective_hooks = None
                if hooks is not None and agent._default_run_hooks is not None:
                    effective_hooks = _CompositeRunHooks([agent._default_run_hooks, hooks])
                elif hooks is not None:
                    effective_hooks = hooks
                elif agent._default_run_hooks is not None:
                    effective_hooks = agent._default_run_hooks
                if effective_hooks is not None:
                    agent._run_hooks = effective_hooks

                # Wire HookRecorder as observer so composites record each leaf
                # invocation without duplicating dispatch logic in Runner/Model.
                if isinstance(agent._run_hooks, _CompositeRunHooks):
                    agent._run_hooks.set_observer(agent._hook_recorder.run)
                if isinstance(agent.hooks, _CompositeAgentHooks):
                    agent.hooks.set_observer(agent._hook_recorder.run)

                # 1. Setup
                agent.update_model()
                agent.run_response.model = agent.model.id if agent.model is not None else None
                if agent.context is not None:
                    agent._resolve_context()

                # v3: Initialise a fresh CostTracker for this run and attach it to the model.
                # The tracker accumulates USD cost across all LLM invoke() calls via
                # Model.update_usage_metrics() / update_stream_metrics().
                # Attach the same CostTracker to the model for accumulating USD cost
                # across all LLM invoke() calls via update_usage_metrics().
                if agent.model is not None:
                    agent.model._cost_tracker = _cost_tracker

                # Reset compression circuit breaker for this run
                if agent.tool_config.compression_manager is not None:
                    agent.tool_config.compression_manager.reset_run_state()

                # Add introduction if provided
                if agent.prompt_config.introduction is not None:
                    agent.add_introduction(agent.prompt_config.introduction)

                # --- Lifecycle: agent start ---
                await self._dispatch_agent_hook(
                    "on_start",
                    lambda hook: hook.on_start(agent=agent),
                )
                await self._dispatch_run_hook(
                    "on_agent_start",
                    lambda hook: hook.on_agent_start(agent=agent),
                )

                # --- Lifecycle: on_user_prompt hook ---
                # Allows hooks to inspect/modify user input before message assembly.
                if isinstance(message, str) and agent._run_hooks is not None:
                    try:
                        modified = await self._dispatch_user_prompt_hook(message)
                        if modified is not None:
                            message = modified
                    except Exception as e:
                        # Fail-open by design: hook errors never block the user's
                        # request. Early-stage product -- minimize user disruption.
                        logger.warning(f"on_user_prompt hook error: {e}")

                # --- Agent-level input guardrails ---
                # Run BEFORE message assembly / LLM. A blocked guardrail raises
                # InputGuardrailTripwireTriggered and aborts the run with no
                # model call, no token cost.
                #
                # Inspect the COMPLETE inbound surface — not just `message`.
                # Callers may provide a full transcript via `messages=[...]`
                # and attach multimodal payloads via `audio` / `images` /
                # `videos`. All of these reach the model, so the guardrail must
                # see them before any model call.
                if agent.input_guardrails:
                    _guard_input = normalize_input_for_guardrails(
                        message=message,
                        audio=audio,
                        images=images,
                        videos=videos,
                        messages=messages,
                    )
                    await run_input_guardrails(
                        agent=agent,
                        input_data=_guard_input,
                        guardrails=agent.input_guardrails,
                        context=agent.context,
                    )

                # 3. Prepare messages
                system_message, user_messages, messages_for_model = await agent.get_messages_for_run(
                    message=message, audio=audio, images=images, videos=videos, messages=messages, **kwargs
                )
                num_input_messages = len(messages_for_model)

                if agent.stream_intermediate_steps:
                    yield self.generic_run_response("Run started", RunEvent.run_started)

                # 4. Generate response from the Model
                # The agentic loop (tool call → LLM → ...) is driven here.
                loop_state = LoopState(
                    max_turns=agent._max_turns,
                    max_api_retry=agent._run_max_api_retry,
                )

                model_response: ModelResponse
                agent.model = cast(Model, agent.model)

                # Disable tool execution in Model layer — Runner owns tool execution now.
                # Model.response() / response_stream() will still parse tool_calls into
                # the assistant message, but won't execute them.
                agent.model.run_tools = False

                # Build hooks from Agent (they live on Agent, not Model).
                # De-dup state (_overflow_warning_emitted) is intentionally
                # persistent across runs for the same Agent instance, so the CLI
                # does not repeatedly emit the same overflow notice every user
                # turn.
                _pre_tool_hook = agent._build_pre_tool_hook()
                _post_tool_hook = agent._build_post_tool_hook()

                if stream and agent.is_streamable:
                    # ============================================================
                    # STREAMING agentic loop
                    # ============================================================
                    model_response = ModelResponse(content="", reasoning_content="")
                    agent._cancelled = False
                    fallback_transaction_marker: Optional[Message] = None
                    fallback_transaction_model: Optional[Model] = None

                    while True:
                        loop_state.turn_count += 1
                        logger.debug(
                            f"[stream] Turn {loop_state.turn_count}: "
                            f"agent={agent.identifier}, messages={len(messages_for_model)}"
                        )

                        # Safety checks (death spiral + cost budget). The break
                        # reason is recorded as structured metadata on the
                        # RunResponse, NOT streamed as content, so downstream
                        # never has to strip internal error text from the reply.
                        _loop_break = self._loop_safety_checks(
                            messages_for_model,
                            loop_state,
                            agent,
                        )
                        if _loop_break:
                            agent.run_response.break_reason = _loop_break.reason
                            agent.run_response.break_message = _loop_break.message
                            logger.warning(
                                f"Agent '{agent.identifier}': loop aborted "
                                f"({_loop_break.reason}) — {_loop_break.message}"
                            )
                            _recovery = await self._recover_with_fallback(
                                messages_for_model,
                                loop_state,
                                agent,
                                _loop_break.reason,
                            )
                            if _recovery is not None:
                                agent.run_response.fallback_used = True
                                if loop_state.last_used_model_id is not None:
                                    agent.run_response.model = loop_state.last_used_model_id
                                if _recovery.content:
                                    # Recovery replaces the (usually empty)
                                    # partial. We yield it as a chunk AND make it
                                    # the authoritative final content; if earlier
                                    # turns streamed visible text, consumers should
                                    # treat run_response.content as the source of
                                    # truth (see docs: fallback_on_break).
                                    model_response.content = _recovery.content
                                    yield RunResponse(
                                        event=RunEvent.run_response,
                                        content=_recovery.content,
                                        run_id=agent.run_id,
                                        agent_id=agent.agent_id,
                                    )
                            break

                        # Safety: cancellation
                        agent._check_cancelled()

                        # Pre-tool hook (context overflow handling)
                        if _pre_tool_hook is not None and loop_state.turn_count > 1:
                            _skip = await _pre_tool_hook(messages_for_model, [])
                            if _skip:
                                # Hook says skip this batch — let model reconsider
                                continue

                        # Mid-run steering: flush any guidance pushed since the last
                        # inference so the model sees it on THIS call.
                        self._inject_steering(messages_for_model, agent)

                        active_model = fallback_transaction_model or agent.model

                        # Compression pipeline (cheapest-first, before LLM call)
                        await self._maybe_compress_messages(messages_for_model, agent, active_model)

                        # --- Lifecycle: LLM start (stream) ---
                        await self._dispatch_run_hook(
                            "on_llm_start",
                            lambda hook: hook.on_llm_start(agent=agent, messages=messages_for_model),
                        )

                        call_start = len(messages_for_model)
                        model_call = await self._call_with_retry(
                            active_model, messages_for_model, loop_state, agent, stream=True
                        )
                        model_response_stream = model_call.response
                        active_model = model_call.used_model
                        if model_call.used_fallback:
                            agent.run_response.fallback_used = True
                        if model_call.used_fallback or fallback_transaction_model is not None:
                            fallback_transaction_model = active_model
                        # Stamp truthful model id onto RunResponse: reflects the
                        # model that actually answered, including any per-call
                        # fallback. Optimistic for streaming (final answer may
                        # still hit content_filter at end-of-stream, but this
                        # call did at least connect to `last_used_model_id`).
                        if loop_state.last_used_model_id is not None:
                            agent.run_response.model = loop_state.last_used_model_id
                        try:
                            async for model_response_chunk in model_response_stream:
                                agent._check_cancelled()
                                if model_response_chunk.event == ModelResponseEvent.assistant_response.value:
                                    if model_response_chunk.reasoning_content is not None:
                                        if model_response.reasoning_content is None:
                                            model_response.reasoning_content = ""
                                        model_response.reasoning_content += model_response_chunk.reasoning_content
                                        yield RunResponse(
                                            event=RunEvent.run_response,
                                            reasoning_content=model_response_chunk.reasoning_content,
                                            run_id=agent.run_id,
                                            agent_id=agent.agent_id,
                                        )
                                    if model_response_chunk.content is not None and model_response.content is not None:
                                        model_response.content += model_response_chunk.content
                                        yield RunResponse(
                                            event=RunEvent.run_response,
                                            content=model_response_chunk.content,
                                            run_id=agent.run_id,
                                            agent_id=agent.agent_id,
                                        )
                        except Exception as exc:
                            # _call_with_retry's tool-history sanitize-retry never
                            # fires for streaming: it only wraps generator
                            # *creation* (lazy, no HTTP call yet), not consumption
                            # — the actual request/error happens here, on the
                            # first chunk. Mirror that one-shot recovery at the
                            # point the error actually surfaces.
                            err = str(exc).lower()
                            is_tool_history_error = any(h in err for h in loop_state.TOOL_HISTORY_HINTS)
                            if is_tool_history_error and not loop_state.tool_history_sanitized_done:
                                loop_state.tool_history_sanitized_done = True
                                del messages_for_model[call_start:]
                                logger.warning(
                                    f"[tool_history] {active_model.id} rejected tool-call "
                                    f"history mid-stream (likely cross-model resume); "
                                    f"stripping tool messages and retrying once: {exc}"
                                )
                                Runner._sanitize_tool_history_after_error(agent, messages_for_model)
                                continue
                            raise

                        # Streaming appends the turn's assistant message during
                        # consumption, so capture the transaction-start marker now
                        # (once) — the first message produced at ``call_start``.
                        if (
                            fallback_transaction_model is not None
                            and fallback_transaction_marker is None
                            and call_start < len(messages_for_model)
                        ):
                            fallback_transaction_marker = messages_for_model[call_start]

                        # --- Lifecycle: LLM end (stream) ---
                        await self._dispatch_run_hook(
                            "on_llm_end",
                            lambda hook: hook.on_llm_end(agent=agent, response=model_response),
                        )

                        # --- Runner-owned tool execution (streaming) ---
                        # Model.response_stream() only parsed tool_calls (run_tools=False).
                        # Runner now executes them, yielding tool events.
                        _had_tool_calls = False
                        assistant_msg = self._get_last_assistant_message(messages_for_model)
                        if assistant_msg is not None and assistant_msg.tool_calls:
                            _had_tool_calls = True
                            async for tool_resp in self._handle_tool_calls_in_runner_stream(
                                messages_for_model,
                                agent,
                                active_model,
                            ):
                                if tool_resp.event == ModelResponseEvent.tool_call_started.value:
                                    tool_call_dict = tool_resp.tool_call
                                    if tool_call_dict is not None:
                                        if agent.run_response.tools is None:
                                            agent.run_response.tools = []
                                        if fallback_transaction_model is not None:
                                            tool_call_dict["fallback_compacted"] = True
                                            tool_call_dict["replay"] = False
                                            tool_call_dict["fallback_model"] = fallback_transaction_model.id
                                        agent.run_response.tools.append(tool_call_dict)
                                    if agent.stream_intermediate_steps:
                                        yield self.generic_run_response(
                                            f"Running tool: {tool_call_dict.get('tool_name') if tool_call_dict else 'Unknown'}",
                                            RunEvent.tool_call_started,
                                        )
                                elif tool_resp.event == ModelResponseEvent.tool_call_completed.value:
                                    tool_call_dict = tool_resp.tool_call
                                    if tool_call_dict is not None and agent.run_response.tools:
                                        target_id = tool_call_dict.get("tool_call_id")
                                        for tool_call in agent.run_response.tools:
                                            if target_id is not None and tool_call.get("tool_call_id") == target_id:
                                                tool_call.update(tool_call_dict)
                                                break
                                    if agent.stream_intermediate_steps:
                                        yield self.generic_run_response(
                                            f"Tool completed: {tool_call_dict.get('tool_name') if tool_call_dict else 'Unknown'}",
                                            RunEvent.tool_call_completed,
                                        )
                                elif tool_resp.event == ModelResponseEvent.assistant_response.value:
                                    if tool_resp.content is not None:
                                        yield RunResponse(
                                            event=RunEvent.run_response,
                                            content=tool_resp.content,
                                            run_id=agent.run_id,
                                            agent_id=agent.agent_id,
                                        )
                                    if tool_resp.reasoning_content is not None:
                                        yield RunResponse(
                                            event=RunEvent.run_response,
                                            reasoning_content=tool_resp.reasoning_content,
                                            run_id=agent.run_id,
                                            agent_id=agent.agent_id,
                                        )

                        # Post-tool hook (todo reminder injection)
                        if _post_tool_hook is not None:
                            await _post_tool_hook(messages_for_model, [])

                        # Check if loop should continue
                        should_continue = self._loop_post_response(
                            messages_for_model,
                            active_model,
                            loop_state,
                            _had_tool_calls,
                        )
                        if not should_continue:
                            if fallback_transaction_model is not None and fallback_transaction_marker is not None:
                                self._compact_fallback_transaction(
                                    messages_for_model,
                                    fallback_transaction_marker,
                                    model_response,
                                    fallback_transaction_model,
                                )
                                fallback_transaction_model = None
                                fallback_transaction_marker = None
                            break

                else:
                    # ============================================================
                    # NON-STREAMING agentic loop
                    # ============================================================
                    agent._cancelled = False
                    model_response = ModelResponse()
                    fallback_transaction_marker: Optional[Message] = None
                    fallback_transaction_model: Optional[Model] = None
                    fallback_compacted_tools: List[Dict[str, Any]] = []

                    while True:
                        loop_state.turn_count += 1
                        logger.debug(
                            f"[non-stream] Turn {loop_state.turn_count}: "
                            f"agent={agent.identifier}, messages={len(messages_for_model)}"
                        )
                        agent._check_cancelled()

                        # Safety checks (death spiral + cost budget). The break
                        # reason is recorded as structured metadata on the
                        # RunResponse, NOT appended to content, so downstream
                        # never has to strip internal error text from the reply.
                        _loop_break = self._loop_safety_checks(
                            messages_for_model,
                            loop_state,
                            agent,
                        )
                        if _loop_break:
                            agent.run_response.break_reason = _loop_break.reason
                            agent.run_response.break_message = _loop_break.message
                            logger.warning(
                                f"Agent '{agent.identifier}': loop aborted "
                                f"({_loop_break.reason}) — {_loop_break.message}"
                            )
                            _recovery = await self._recover_with_fallback(
                                messages_for_model,
                                loop_state,
                                agent,
                                _loop_break.reason,
                            )
                            if _recovery is not None:
                                model_response = _recovery
                                agent.run_response.fallback_used = True
                                if loop_state.last_used_model_id is not None:
                                    agent.run_response.model = loop_state.last_used_model_id
                            break

                        # Pre-tool hook (context overflow handling)
                        if _pre_tool_hook is not None and loop_state.turn_count > 1:
                            _skip = await _pre_tool_hook(messages_for_model, [])
                            if _skip:
                                continue  # Let model reconsider

                        # Mid-run steering: flush any guidance pushed since the last
                        # inference so the model sees it on THIS call.
                        self._inject_steering(messages_for_model, agent)

                        active_model = fallback_transaction_model or agent.model

                        # Compression pipeline (cheapest-first, before LLM call)
                        await self._maybe_compress_messages(messages_for_model, agent, active_model)

                        # --- Lifecycle: LLM start (non-stream) ---
                        await self._dispatch_run_hook(
                            "on_llm_start",
                            lambda hook: hook.on_llm_start(agent=agent, messages=messages_for_model),
                        )

                        call_start = len(messages_for_model)
                        model_call = await self._call_with_retry(
                            active_model, messages_for_model, loop_state, agent, stream=False
                        )
                        model_response = model_call.response
                        active_model = model_call.used_model
                        if model_call.used_fallback:
                            agent.run_response.fallback_used = True
                        if model_call.used_fallback or fallback_transaction_model is not None:
                            if fallback_transaction_marker is None and call_start < len(messages_for_model):
                                fallback_transaction_marker = messages_for_model[call_start]
                            fallback_transaction_model = active_model
                        # Stamp truthful model id onto RunResponse: reflects the
                        # model that actually answered, including any per-call
                        # fallback. Final turn naturally wins because each turn
                        # overwrites the previous value.
                        if loop_state.last_used_model_id is not None:
                            agent.run_response.model = loop_state.last_used_model_id

                        # --- Lifecycle: LLM end (non-stream) ---
                        await self._dispatch_run_hook(
                            "on_llm_end",
                            lambda hook: hook.on_llm_end(agent=agent, response=model_response),
                        )

                        # --- Runner-owned tool execution ---
                        tool_result = await self._handle_tool_calls_in_runner(
                            messages_for_model,
                            agent,
                            active_model,
                            stream=False,
                        )
                        _had_tool_calls = tool_result.had_tool_calls
                        if fallback_transaction_model is not None and tool_result.tool_results:
                            for record in tool_result.tool_results:
                                record["fallback_compacted"] = True
                                record["replay"] = False
                                record["fallback_model"] = fallback_transaction_model.id
                            fallback_compacted_tools.extend(tool_result.tool_results)

                        # Post-tool hook (todo reminder injection)
                        if _post_tool_hook is not None:
                            await _post_tool_hook(messages_for_model, [])

                        # Check if loop should continue
                        should_continue = self._loop_post_response(
                            messages_for_model,
                            active_model,
                            loop_state,
                            _had_tool_calls,
                        )
                        if not should_continue:
                            if fallback_transaction_model is not None and fallback_transaction_marker is not None:
                                fallback_compacted_tools.extend(
                                    self._compact_fallback_transaction(
                                        messages_for_model,
                                        fallback_transaction_marker,
                                        model_response,
                                        fallback_transaction_model,
                                    )
                                )
                                fallback_transaction_model = None
                                fallback_transaction_marker = None
                            break

                    # --- Context window usage warning ---
                    _window = agent.model.context_window
                    if _window:
                        _ctx_tokens = count_tokens(messages_for_model, None, agent.model.id, None)
                        _pct = _ctx_tokens / _window
                        agent.run_response.metrics = agent.run_response.metrics or {}
                        agent.run_response.metrics["context_window_pct"] = round(_pct, 3)
                        if _pct >= 0.8:
                            logger.warning(
                                f"Agent '{agent.identifier}': context usage "
                                f"{_ctx_tokens:,}/{_window:,} tokens ({_pct:.0%})"
                            )
                    if (
                        agent.response_model is not None
                        and agent.use_structured_outputs
                        and model_response.parsed is not None
                    ):
                        agent.run_response.content = model_response.parsed
                        agent.run_response.content_type = agent.response_model.__name__
                    else:
                        agent.run_response.content = model_response.content
                    if model_response.audio is not None:
                        agent.run_response.audio = model_response.audio
                    if model_response.reasoning_content is not None:
                        agent.run_response.reasoning_content = model_response.reasoning_content
                    agent.run_response.messages = messages_for_model
                    agent.run_response.created_at = model_response.created_at

                    # Extract tool call info from messages for non-streaming mode
                    tool_calls_data = list(fallback_compacted_tools)
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
                                    "tool_call_error": m.tool_call_error or False,
                                    "metrics": m.metrics if m.metrics else {},
                                }
                            )
                    if tool_calls_data:
                        deduped_tool_calls = []
                        seen_tool_keys = set()
                        for tool_call in tool_calls_data:
                            key = (
                                tool_call.get("tool_call_id"),
                                tool_call.get("tool_name"),
                                tool_call.get("content"),
                            )
                            if key in seen_tool_keys:
                                continue
                            seen_tool_keys.add(key)
                            deduped_tool_calls.append(tool_call)
                        agent.run_response.tools = deduped_tool_calls

                # Build run messages
                run_messages = user_messages + messages_for_model[num_input_messages:]
                if system_message is not None:
                    run_messages.insert(0, system_message)
                agent.run_response.messages = run_messages
                existing_metrics = agent.run_response.metrics or {}
                aggregated_metrics = self._aggregate_metrics_from_run_messages(run_messages)
                if existing_metrics:
                    aggregated_metrics.update(existing_metrics)
                agent.run_response.metrics = aggregated_metrics
                agent.run_response.usage = agent.model.usage if agent.model else None

                # v3: attach CostTracker to RunResponse
                if agent.model is not None and agent.model._cost_tracker is not None:
                    agent.run_response.cost_tracker = agent.model._cost_tracker

                if agent.stream:
                    agent.run_response.content = model_response.content
                    if model_response.reasoning_content:
                        agent.run_response.reasoning_content = model_response.reasoning_content

                # --- Agent-level output guardrails ---
                # MUST run BEFORE memory persistence, summary update, file save,
                # the run_completed yield, and on_end hooks. A blocked guardrail
                # raises OutputGuardrailTripwireTriggered; if we ran this after
                # working_memory.add_run() / update_summary() / save_run_response_to_file()
                # the rejected content would still leak into persisted state and
                # poison subsequent turns. Persistence happens only when the
                # output is allowed.
                _output = agent.run_response.content
                if agent.output_guardrails:
                    await run_output_guardrails(
                        agent=agent,
                        agent_output=_output,
                        guardrails=agent.output_guardrails,
                        context=agent.context,
                    )

                # 5. Update Memory
                if agent.stream_intermediate_steps:
                    yield self.generic_run_response("Updating memory", RunEvent.updating_memory)

                if messages is None:
                    if system_message is not None:
                        agent.working_memory.add_system_message(
                            system_message,
                            system_message_role=agent.prompt_config.system_message_role,
                        )
                    agent.working_memory.add_messages(
                        messages=(user_messages + messages_for_model[num_input_messages:])
                    )

                    agent_run = AgentRun(response=agent.run_response)
                    if user_messages:
                        agent_run.message = user_messages[0]
                        agent_run.messages = list(user_messages)
                    agent.working_memory.add_run(agent_run)
                    _memory_persisted = True

                    if (
                        agent.working_memory.create_session_summary
                        and agent.working_memory.update_session_summary_after_run
                    ):
                        await agent.working_memory.update_summary()

                # 6. Save output to file
                self.save_run_response_to_file(message=message, save_response_to_file=save_response_to_file)

                # 7. Set run_input
                if message is not None:
                    if isinstance(message, str):
                        agent.run_input = message
                    elif isinstance(message, Message):
                        agent.run_input = message.to_model_dict()
                    else:
                        agent.run_input = message
                elif messages is not None:
                    agent.run_input = [m.to_model_dict() if isinstance(m, Message) else m for m in messages]

                if agent.stream_intermediate_steps:
                    yield self.generic_run_response(agent.run_response.content, RunEvent.run_completed)

                # --- Lifecycle: agent end ---
                await self._dispatch_agent_hook(
                    "on_end",
                    lambda hook: hook.on_end(agent=agent, output=_output),
                )
                await self._dispatch_run_hook(
                    "on_agent_end",
                    lambda hook: hook.on_agent_end(agent=agent, output=_output),
                )

                if not agent.stream:
                    yield agent.run_response

                # Clear query-level tool/skill filtering and per-run hooks after run
                agent._enabled_tools = None
                agent._enabled_skills = None
                agent._run_hooks = None

                # --- Session persist (CC-style JSONL append) ---
                # Log the complete turn: user input + tool results + assistant output
                if agent._session_log is not None and messages is None:
                    # 1. Log user input
                    _user_text = None
                    if isinstance(message, str):
                        _user_text = message
                    elif isinstance(message, Message):
                        _user_text = message.content if isinstance(message.content, str) else str(message.content)
                    if _user_text:
                        agent._session_log.append("user", _user_text)

                    # 2. Log assistant tool-call messages AND their tool results in
                    #    the exact interleaved order, so /resume rebuilds a valid
                    #    assistant(tool_calls)->tool sequence instead of orphaned
                    #    (or mis-ordered) tool messages.
                    self._persist_assistant_tool_calls(agent)

                    # 3. Log assistant output (with model info + usage, mirrors CC)
                    _assistant_text = agent.run_response.content
                    if _assistant_text and isinstance(_assistant_text, str):
                        _model_meta = {}
                        if agent.run_response.model:
                            _model_meta["model"] = agent.run_response.model
                        elif agent.model:
                            _model_meta["model"] = agent.model.id
                        if model_response.finish_reason:
                            _model_meta["finish_reason"] = model_response.finish_reason
                        if agent.run_response.reasoning_content:
                            _model_meta["reasoning_content"] = agent.run_response.reasoning_content
                        if agent.run_response.metrics:
                            _model_meta["metrics"] = agent.run_response.metrics
                        if agent.model and agent.model.usage and agent.model.usage.request_usage_entries:
                            _last_usage = agent.model.usage.request_usage_entries[-1]
                            _model_meta["usage"] = {
                                "input_tokens": _last_usage.input_tokens,
                                "output_tokens": _last_usage.output_tokens,
                            }
                        agent._session_log.append("assistant", _assistant_text, **_model_meta)

                # Run reached natural completion -- mark + emit terminal event.
                _run_ctx.mark_completed()
                self._emit_event(
                    RunEventType.run_completed,
                    {
                        "duration_seconds": _run_ctx.duration_seconds,
                        "had_response": agent.run_response.content is not None,
                    },
                )
            except (AgentCancelledError, asyncio.CancelledError) as _cancel_exc:
                _run_ctx.mark_cancelled(reason=str(_cancel_exc) or "cancelled")
                self._emit_event(
                    RunEventType.run_cancelled,
                    {"reason": _run_ctx.error},
                )
                # Preserve the interrupted turn (question + partial answer) so
                # history doesn't lose the whole exchange on Ctrl+C. Skip if the
                # success path already persisted this turn (cancel struck during
                # post-completion hooks) — otherwise we'd double-add and stamp an
                # interruption marker on a fully-finished answer.
                if not _memory_persisted:
                    try:
                        self._persist_interrupted_turn(
                            agent,
                            message,
                            messages,
                            user_messages,
                            system_message,
                            messages_for_model,
                            num_input_messages,
                            model_response,
                        )
                    except Exception:
                        logger.warning("interrupted-turn persistence failed", exc_info=True)
                raise
            except Exception as _run_exc:
                _run_ctx.mark_failed(error=f"{type(_run_exc).__name__}: {_run_exc}")
                self._emit_event(
                    RunEventType.run_failed,
                    {
                        "error": _run_ctx.error,
                        "exception_type": type(_run_exc).__name__,
                    },
                )
                raise
            finally:
                # Close the steering window under _steer_lock: flips _running
                # False and drops late-arriving guidance so it can't leak into
                # the next run. After this, steer() returns False and the CLI
                # falls back to queuing a fresh turn.
                agent._end_steer_window()
                agent._run_loop = None
                agent._run_task = None
                agent._run_max_api_retry = agent.max_api_retry

        trace_input = self._build_langfuse_trace_input(message, messages)
        trace_name = agent.name or "agent-run"

        langfuse_tags = None
        if agent.model and hasattr(agent.model, "langfuse_tags"):
            langfuse_tags = agent.model.langfuse_tags

        run_state_token = Model.begin_run_state()
        try:
            with langfuse_trace_context(
                name=trace_name,
                session_id=agent.session_id,
                user_id=agent.user_id,
                tags=langfuse_tags,
                input_data=trace_input,
            ) as trace:
                final_response: Optional[RunResponse] = None
                try:
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
                        # Symmetric counterpart of the "[user] -> agent" chat log
                        # at run entry: record the agent's final reply so each
                        # turn has both inbound and outbound lines at CHAT level.
                        # Print the full reply for the same reason the inbound
                        # line is full — CHAT level is the authoritative
                        # conversation transcript.
                        reply_preview = str(output_content) if output_content else "<no reply>"
                        logger.chat(f"[{agent.identifier}] -> user: {reply_preview}")
                finally:
                    # Issue #3 fix: dump hook_calls even on exception, so
                    # failure traces still carry the hook timeline that's
                    # most valuable for diagnostics.
                    if agent._hook_recorder:
                        trace.set_metadata("hook_calls", agent._hook_recorder.export())
        finally:
            Model.reset_run_state(run_state_token)
            # Release async HTTP clients on the (still-alive) run loop so they
            # are not garbage-collected later on a closed loop — httpx raises
            # "Event loop is closed" during aclose() in that case. The next turn
            # rebuilds a fresh client bound to its own loop. The CLI runs each
            # turn on a fresh asyncio.run loop, so this is what makes /model
            # switches and per-turn churn of Anthropic/OpenAI clients clean.
            for _m in (agent.model, agent.auxiliary_model):
                if _m is not None and hasattr(_m, "close_client"):
                    try:
                        await _m.close_client()
                    except Exception:
                        pass

    # =========================================================================
    # Timeout wrappers
    # =========================================================================

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

    # =========================================================================
    # Public API: async run()/run_stream() + sync adapters
    # =========================================================================

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
