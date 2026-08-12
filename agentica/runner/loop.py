# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner agentic loop (_run_impl) and turn orchestration
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncIterator,
    cast,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel

from agentica.utils.log import logger, _run_id_var, _parent_run_id_var, _short
from agentica.agent.history_filter import strip_tool_artifacts_from_memory
from agentica.cost_tracker import CostTracker
from agentica.hooks import RunHooks, _CompositeAgentHooks, _CompositeRunHooks
from agentica.model.base import Model
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import AgentCancelledError, RunBreakReason, RunEvent, RunResponse
from agentica.run_context import RunContext, RunSource, TaskAnchor
from agentica.run_events import RunEventType
from agentica.memory import AgentRun
from agentica.utils.tokens import count_tokens
from agentica.utils.langfuse_integration import langfuse_trace_context
from agentica.tools.base import FunctionCall
from agentica.guardrails.agent import (
    normalize_input_for_guardrails,
    run_input_guardrails,
    run_output_guardrails,
)
from agentica.guardrails.core import GuardrailTriggered

if TYPE_CHECKING:
    from agentica.agent import Agent

from agentica.runner.types import LoopBreak, ToolHandlingResult


class LoopMixin:
    """Extracted Runner methods."""

    agent: Any
    _build_langfuse_trace_input: Any
    _tool_records_from_messages: Any

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

        if agent.model is not None:
            _cost_msg = self._check_cost_budget(agent.model._cost_tracker, agent._run_max_cost_usd)
            if _cost_msg:
                return LoopBreak(RunBreakReason.COST_BUDGET.value, _cost_msg)

        return None

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
        _run_id_var.set(_short(_pregen_run_id))
        if agent._parent_run_id:
            _parent_run_id_var.set(_short(agent._parent_run_id))

        async def _run_core() -> AsyncIterator[RunResponse]:  # pyright: ignore[reportGeneralTypeIssues] - main agent loop is intentionally centralized.
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
            #
            # messages mode note: when the caller supplies a pre-built transcript
            # via ``messages=[...]`` (message is None), we surface the last
            # user-role entry so CHAT logs still show what the user just said
            # this turn — otherwise every messages-mode run would log
            # ``<no message>`` and mask the real inbound content. dict/Message
            # element shapes are both handled; content may be a plain string or
            # a multimodal list (image_url / audio / video parts) — we render
            # multimodal lists as a compact tag so the transcript stays scannable.
            if message is not None:
                _chat_preview = str(message)
            elif messages:
                _last_user_content = None
                for _m in reversed(messages):
                    _role = _m.role if isinstance(_m, Message) else _m.get("role")
                    if _role != "user":
                        continue
                    _last_user_content = (
                        _m.content if isinstance(_m, Message) else _m.get("content")
                    )
                    break
                if isinstance(_last_user_content, list):
                    _parts = []
                    for _p in _last_user_content:
                        _ptype = _p.get("type") if isinstance(_p, dict) else None
                        if _ptype == "text":
                            _parts.append(_p.get("text", ""))
                        elif _ptype:
                            _parts.append(f"[{_ptype}]")
                        else:
                            _parts.append(str(_p))
                    _chat_preview = " ".join(p for p in _parts if p)
                elif _last_user_content is None:
                    _chat_preview = f"<messages len={len(messages)}, no user role>"
                else:
                    _chat_preview = str(_last_user_content)
            else:
                _chat_preview = "<no message>"
            logger.chat(f"[user] -> {agent.identifier}: {_chat_preview}")
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
            _restore_guidance_after_run = False
            _saved_tool_policy_prompts: List[str] = []
            _saved_session_guidance_prompts: List[str] = []
            _saved_session_guidance_snapshot: Optional[str] = None
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
                # Bound before anything can raise so the cancel / failure
                # handlers below can always read them, however early the run
                # dies. An empty `user_messages` is the signal that the turn
                # never reached message assembly and holds nothing to preserve.
                system_message: Optional[Message] = None
                user_messages: List[Message] = []
                messages_for_model: List[Message] = []
                num_input_messages = 0
                input_message_ids: set = set()
                model_response = ModelResponse(content="")
                loop_state = LoopState(
                    max_turns=agent._max_turns,
                    max_api_retry=agent._run_max_api_retry,
                )
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
                    # model= lets the log distrust a compact boundary written
                    # under another lineage (model/branch change): it replays
                    # the canonical transcript instead of the stale summary.
                    _m = getattr(agent, "model", None)
                    resumed_messages = agent._session_log.load(
                        model=getattr(_m, "id", None) if _m is not None else None
                    )
                    if resumed_messages:
                        runs_built = agent.working_memory.hydrate_runs_from_history(resumed_messages)
                        # The transcript stores tool rounds in the OpenAI wire
                        # shape; a provider that speaks another one would 400
                        # on replay, so it continues from the text turns only.
                        if (
                            runs_built
                            and agent.model is not None
                            and not agent.model.supports_replayed_tool_history
                        ):
                            strip_tool_artifacts_from_memory(agent.working_memory)
                        logger.debug(
                            f"Session resumed from JSONL: {len(resumed_messages)} messages, "
                            f"{runs_built} runs reconstructed"
                        )

                # --- Initialise CostTracker for this run ---
                _cost_tracker = CostTracker()
                agent.run_response.cost_tracker = _cost_tracker

                # Set query-level tool/skill filtering before prompt assembly.
                # A per-run enabled_skills whitelist is a semantic contract, so
                # the advertised skill catalogue must match the execution gate.
                # Re-render only for that explicit override; ordinary skill usage
                # ranking remains session-frozen for cache stability.
                agent._enabled_tools = enabled_tools
                agent._enabled_skills = enabled_skills
                if enabled_skills is not None:
                    _restore_guidance_after_run = True
                    _saved_tool_policy_prompts = list(agent._tool_policy_prompts)
                    _saved_session_guidance_prompts = list(agent._session_guidance_prompts)
                    _saved_session_guidance_snapshot = agent._session_guidance_snapshot
                    agent.refresh_tool_system_prompts()
                    agent._session_guidance_snapshot = None

                # --- Freeze prompt snapshots on first run (prompt cache stability) ---
                # Hermes-style: freeze everything the system prompt reads from
                # live state at session start so its bytes stay identical across
                # turns. The skills block is agent-side and has no workspace to
                # depend on, so it is frozen unconditionally.
                agent.freeze_session_guidance()
                if (
                    agent.workspace is not None
                    and agent.workspace.exists()
                    and agent.workspace.get_frozen_context() is None
                ):
                    # Use the run's TaskAnchor as the freeze query so memory
                    # retrieval is bound to the *original* goal, not whatever
                    # `message` happens to be on subsequent runs.
                    await agent.workspace.freeze_snapshots(query=_anchor.source_query)

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
                input_message_ids = {id(m) for m in messages_for_model}

                if agent.stream_intermediate_steps:
                    yield self.generic_run_response("Run started", RunEvent.run_started)

                # 4. Generate response from the Model
                # The agentic loop (tool call → LLM → ...) is driven here.
                agent.model = cast(Model, agent.model)

                # Disable tool execution in Model layer — Runner owns tool execution now.
                # Model.response() / response_stream() will still parse tool_calls into
                # the assistant message, but won't execute them.
                agent.model.run_tools = False

                # Build hooks from Agent (they live on Agent, not Model).
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

                        # Mid-run steering: flush any guidance pushed since the last
                        # inference so the model sees it on THIS call.
                        self._inject_steering(messages_for_model, agent)
                        self._inject_peer_messages(messages_for_model, agent)

                        active_model = fallback_transaction_model or agent.model

                        # Compression pipeline (cheapest-first, before LLM call)
                        await self._maybe_compress_messages(
                            messages_for_model, agent, active_model, loop_state
                        )

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
                                self._sanitize_tool_history_after_error(agent, messages_for_model)
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
                                            tool_call=tool_call_dict,
                                        )
                                elif tool_resp.event == ModelResponseEvent.tool_call_completed.value:
                                    tool_call_dict = tool_resp.tool_call
                                    if tool_call_dict is not None and agent.run_response.tools:
                                        target_id = tool_call_dict.get("tool_call_id")
                                        persisted_tool_call = {
                                            key: value for key, value in tool_call_dict.items()
                                            if key != "tool_display_meta"
                                        }
                                        for tool_call in agent.run_response.tools:
                                            if target_id is not None and tool_call.get("tool_call_id") == target_id:
                                                tool_call.update(persisted_tool_call)
                                                break
                                    if agent.stream_intermediate_steps:
                                        yield self.generic_run_response(
                                            f"Tool completed: {tool_call_dict.get('tool_name') if tool_call_dict else 'Unknown'}",
                                            RunEvent.tool_call_completed,
                                            tool_call=tool_call_dict,
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

                        # Mid-run steering: flush any guidance pushed since the last
                        # inference so the model sees it on THIS call.
                        self._inject_steering(messages_for_model, agent)
                        self._inject_peer_messages(messages_for_model, agent)

                        active_model = fallback_transaction_model or agent.model

                        # Compression pipeline (cheapest-first, before LLM call)
                        await self._maybe_compress_messages(
                            messages_for_model, agent, active_model, loop_state
                        )

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

                # Build run messages. When a compression stage dropped messages
                # the num_input_messages prefix boundary no longer exists, and
                # slicing it would silently discard this turn's own reply. What
                # survived compaction *is* the whole conversation state, so
                # store that; the superseded runs are dropped below.
                if loop_state.context_collapsed:
                    run_messages = [m for m in messages_for_model if m.role != "system"]
                else:
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
                    # The flat archive (/history, /export) only ever wants this
                    # turn's additions. Identity is the reliable marker once
                    # compaction has reshuffled the list; the summary messages it
                    # injected are new, and belong in the transcript.
                    if loop_state.context_collapsed:
                        turn_messages = [
                            m for m in messages_for_model
                            if id(m) not in input_message_ids and m.role != "system"
                        ]
                    else:
                        turn_messages = messages_for_model[num_input_messages:]
                    agent.working_memory.add_messages(messages=(user_messages + turn_messages))

                    agent_run = AgentRun(response=agent.run_response)
                    if user_messages:
                        agent_run.message = user_messages[0]
                        agent_run.messages = list(user_messages)
                    if loop_state.context_collapsed:
                        # This run carries the post-compaction conversation in
                        # full, so the runs it summarises must go — otherwise the
                        # next turn rebuilds the pre-compaction history from them
                        # and the compaction saves nothing beyond this run.
                        agent.working_memory.runs.clear()
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
                        _user_meta = self._provider_replay_meta(user_messages[-1]) if user_messages else {}
                        agent._session_log.append("user", _user_text, **_user_meta)

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
                        _final_assistant = next(
                            (
                                item for item in reversed(agent.run_response.messages or [])
                                if isinstance(item, Message) and item.role == "assistant" and not item.tool_calls
                            ),
                            None,
                        )
                        if _final_assistant is not None:
                            _model_meta.update(self._provider_replay_meta(_final_assistant))
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
                    self._try_persist_incomplete_turn(
                        agent,
                        message,
                        messages,
                        user_messages,
                        system_message,
                        messages_for_model,
                        num_input_messages,
                        model_response,
                        loop_state,
                        input_message_ids,
                        marker="[User interrupted the response]",
                        finish_reason="cancelled",
                    )
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
                # A crashed turn used to be discarded whole, so the instruction
                # it carried vanished from history and the next "continue" was
                # answered from the previous turn's context. Keep it — except
                # for a guardrail block, where dropping the rejected content IS
                # the point.
                if not _memory_persisted and not isinstance(_run_exc, GuardrailTriggered):
                    _err = _run_ctx.error or type(_run_exc).__name__
                    if len(_err) > 200:
                        _err = _err[:197] + "..."
                    self._try_persist_incomplete_turn(
                        agent,
                        message,
                        messages,
                        user_messages,
                        system_message,
                        messages_for_model,
                        num_input_messages,
                        model_response,
                        loop_state,
                        input_message_ids,
                        marker=f"[Run failed: {_err}]",
                        finish_reason="error",
                    )
                raise
            finally:
                # Close the steering window under _steer_lock: flips _running
                # False and parks late-arriving guidance on the agent (see
                # pop_undelivered_steer) so it can't leak into the next run
                # and is never dropped. After this, steer() returns False and
                # the CLI falls back to queuing a fresh turn.
                agent._end_steer_window()
                if _restore_guidance_after_run:
                    agent._tool_policy_prompts = _saved_tool_policy_prompts
                    agent._session_guidance_prompts = _saved_session_guidance_prompts
                    agent._session_guidance_snapshot = _saved_session_guidance_snapshot
                agent._enabled_tools = None
                agent._enabled_skills = None
                agent._run_loop = None
                agent._run_task = None
                agent._run_max_api_retry = agent.max_api_retry

        trace_input = self._build_langfuse_trace_input(message, messages)
        trace_name = agent.name or "agent-run"

        langfuse_tags = None
        model_for_tags = cast(Any, agent.model)
        if model_for_tags is not None and hasattr(model_for_tags, "langfuse_tags"):
            langfuse_tags = model_for_tags.langfuse_tags

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
                        cast(Any, logger).chat(f"[{agent.identifier}] -> user: {reply_preview}")
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
