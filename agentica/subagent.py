# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Subagent system for managing ephemeral agent tasks

This module implements a subagent system that allows main agents to:
- Spawn isolated subagents for complex tasks
- Track subagent lifecycle and results
- Support different subagent types with varying tool permissions
- Enable parallel execution of multiple subagents

Based on the subagent design pattern from modern AI coding assistants.
"""
import asyncio
import copy
import json
import time
import uuid

import dataclasses
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Union,
)
from datetime import datetime

from agentica.handoff import default_handoff_mapper
from agentica.run_response import AgentCancelledError
from agentica.tools.base import Function, ModelTool, Tool
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model


class SubagentType(str, Enum):
    """Types of subagents with different capabilities."""

    # Explore agent: read-only, specialized for codebase exploration
    EXPLORE = "explore"

    # Research agent: web search and document analysis
    RESEARCH = "research"

    # Code agent: code generation and execution
    CODE = "code"

    # Custom agent: user-defined subagent type
    CUSTOM = "custom"


@dataclass
class SubagentConfig:
    """Configuration for a subagent type."""

    # Subagent type identifier
    type: SubagentType

    # Human-readable name
    name: str

    # Description of the subagent's capabilities
    description: str

    # System prompt for this subagent type
    system_prompt: str

    # Allowed tools (None means all tools from parent, empty list means no tools)
    allowed_tools: Optional[List[str]] = None

    # Denied tools (takes precedence over allowed_tools)
    denied_tools: Optional[List[str]] = None

    # Maximum number of tool calls allowed for this subagent
    tool_call_limit: int = 100

    # Maximum LLM loop turns (safety net for runaway subagents)
    max_turns: int = 100

    # Whether this subagent can spawn its own subagents
    can_spawn_subagents: bool = False

    # --- Permission isolation ---
    # Whether the subagent inherits the parent agent's workspace memory
    inherit_workspace: bool = False
    # Whether the subagent inherits the parent agent's knowledge base
    inherit_knowledge: bool = False
    # Whether the parent agent's current context summary is prepended to the task
    inherit_context: bool = False
    # Timeout for subagent execution in seconds (0 = no timeout).
    # Real-world code / research subagents routinely need 5–15 min. The
    # previous default of 300s was the single biggest cause of "task tool
    # keeps failing" reports — raise it to 30 min and let long tasks finish.
    timeout: int = 1800


@dataclass
class SubagentRun:
    """Represents a single subagent execution."""

    # Unique run identifier
    run_id: str

    # Subagent type
    subagent_type: SubagentType

    # Parent agent_id (who spawned this subagent)
    parent_agent_id: str

    # Task label/description
    task_label: str

    # Full task description
    task_description: str
    
    # Timestamp when started
    started_at: datetime
    
    # Current status
    status: Literal[
        "pending", "running", "completed", "error", "cancelled",
        "timeout", "max_turns", "tool_call_limit", "truncated",
    ] = "pending"
    
    # Timestamp when ended (if finished)
    ended_at: Optional[datetime] = None
    
    # Result from the subagent
    result: Optional[str] = None
    
    # Error message if failed
    error: Optional[str] = None
    
    # Token usage statistics
    token_usage: Optional[Dict[str, int]] = None


class SubagentRegistry:
    """
    Registry for tracking and managing subagent runs.

    This is a singleton-like class that tracks all subagent executions
    across the application lifetime.
    """

    # Tools blocked for ALL subagent types (prevent recursion + privilege escalation).
    # Pattern borrowed from hermes-agent delegate_tool.py.
    BLOCKED_TOOLS = {"delegate_task", "spawn_subagent", "save_memory", "shell", "task"}

    # Max subagent nesting depth (parent -> child -> grandchild = depth 2)
    MAX_DEPTH = 2

    # Max concurrent subagents in a single spawn_batch call
    MAX_CONCURRENT = 3
    
    _instance: Optional["SubagentRegistry"] = None
    
    def __new__(cls) -> "SubagentRegistry":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runs: Dict[str, SubagentRun] = {}
            cls._instance._listeners: List[Callable[[SubagentRun], None]] = []
        return cls._instance
    
    def register(self, run: SubagentRun) -> None:
        """Register a new subagent run. Auto-cleans old completed runs when exceeding threshold."""
        # Auto-cleanup when accumulated runs exceed threshold
        if len(self._runs) > 100:
            self.cleanup_completed(max_age_seconds=600)
        self._runs[run.run_id] = run
        logger.debug(f"Registered subagent run: {run.run_id} ({run.subagent_type.value})")
    
    def get(self, run_id: str) -> Optional[SubagentRun]:
        """Get a subagent run by ID."""
        return self._runs.get(run_id)
    
    def get_for_parent(self, parent_agent_id: str) -> List[SubagentRun]:
        """Get all subagent runs spawned by a parent agent."""
        return [
            run for run in self._runs.values()
            if run.parent_agent_id == parent_agent_id
        ]

    def get_active(self) -> List[SubagentRun]:
        """Get all currently running subagents."""
        return [
            run for run in self._runs.values()
            if run.status in ("pending", "running")
        ]
    
    def update_status(
        self,
        run_id: str,
        status: Literal[
            "running", "completed", "error", "cancelled",
            "timeout", "max_turns", "tool_call_limit", "truncated",
        ],
        result: Optional[str] = None,
        error: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update the status of a subagent run."""
        run = self._runs.get(run_id)
        if run is None:
            logger.warning(f"Cannot update status: subagent run {run_id} not found")
            return
        
        run.status = status
        if status in ("completed", "error", "cancelled"):
            run.ended_at = datetime.now()
        if result is not None:
            run.result = result
        if error is not None:
            run.error = error
        if token_usage is not None:
            run.token_usage = token_usage
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(run)
            except Exception as e:
                logger.error(f"Subagent listener error: {e}")
    
    def on_complete(self, callback: Callable[[SubagentRun], None]) -> None:
        """Register a callback for when a subagent completes."""
        self._listeners.append(callback)
    
    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed/cancelled runs older than max_age_seconds."""
        now = datetime.now()
        to_remove = []
        
        for run_id, run in self._runs.items():
            if run.status in ("completed", "error", "cancelled") and run.ended_at:
                age = (now - run.ended_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(run_id)
        
        for run_id in to_remove:
            del self._runs[run_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old subagent runs")

        return len(to_remove)

    # ================================================================
    # Execution methods (spawn isolated subagents and run to completion)
    # ================================================================

    @staticmethod
    def _build_inherited_context(parent_agent: "Agent") -> str:
        """Build a concise context summary from the parent agent."""
        parts: List[str] = []

        working_memory = parent_agent.working_memory
        if working_memory is not None and working_memory.summary is not None:
            summary_text = working_memory.summary.summary
            if isinstance(summary_text, str) and summary_text.strip():
                parts.append(summary_text.strip())

        if not parts:
            run_response = parent_agent.run_response
            if run_response is not None and isinstance(run_response.content, str) and run_response.content.strip():
                parts.append(run_response.content.strip())

        if not parts:
            parent_context = parent_agent.context
            if isinstance(parent_context, str) and parent_context.strip():
                parts.append(parent_context.strip())
            elif isinstance(parent_context, dict) and parent_context:
                # ``Agent._resolve_context`` allows context values to be
                # callables or arbitrary resolved objects (datetime, etc.),
                # so naive ``json.dumps`` would crash on the inheritance
                # path. ``default=str`` keeps a best-effort string snapshot.
                try:
                    serialized = json.dumps(
                        parent_context,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                except TypeError:
                    serialized = repr(parent_context)
                parts.append(serialized)

        return "\n\n".join(parts)

    @staticmethod
    def _tool_names(tool: Any) -> List[str]:
        """Extract callable function names represented by a parent tool entry."""
        if isinstance(tool, Function):
            return [tool.name]
        if isinstance(tool, Tool):
            return list(tool.functions.keys())
        if isinstance(tool, ModelTool):
            if tool.type == "function" and isinstance(tool.function, dict):
                function_name = tool.function.get("name")
                if isinstance(function_name, str) and function_name:
                    return [function_name]
            return []
        if callable(tool):
            return [tool.__name__]
        return []

    def _select_child_tools(self, parent_tools: List[Any], config: SubagentConfig) -> List[Any]:
        """Filter parent tools according to subagent policy.

        Keeps explicit contracts readable:
        - Parent agent is expected to be a real Agent
        - Tool list may still contain mixed tool representations, so normalize
          names in one helper instead of scattering dynamic lookups.
        """
        child_tools: List[Any] = []

        for tool in parent_tools:
            candidate_names = self._tool_names(tool)
            allowed_names = [
                name
                for name in candidate_names
                if name not in self.BLOCKED_TOOLS
                and (config.allowed_tools is None or name in config.allowed_tools)
                and (config.denied_tools is None or name not in config.denied_tools)
            ]

            if not allowed_names:
                continue

            if isinstance(tool, Tool):
                # Critical isolation step:
                # ``tool.functions[name].entrypoint`` may be a bound method on
                # the parent's tool instance (e.g. ``parent_todo_tool.write_todos``).
                # Reusing those Function objects on a freshly-constructed Tool
                # would still mutate the parent agent's state.
                #
                # ``tool.clone()`` is the canonical way to obtain an instance
                # whose ``Function.entrypoint`` callables are rebound to the
                # cloned tool. For stateless tools that return ``self``, we
                # copy the Tool wrapper so filtering does not alter the
                # parent's ``functions`` dict.
                cloned = tool.clone()
                if cloned is tool:
                    cloned = Tool(name=tool.name, description=tool.description)
                    cloned.functions = OrderedDict(tool.functions)
                cloned.functions = OrderedDict(
                    (name, cloned.functions[name])
                    for name in allowed_names
                    if name in cloned.functions
                )
                child_tools.append(cloned)
                continue

            child_tools.append(tool)

        return child_tools

    @staticmethod
    def _clone_parent_model(source_model: "Model") -> "Model":
        """Shallow-clone a parent Model so the subagent owns isolated runtime state.

        Resets runtime fields (tools/functions/tool_choice/metrics/usage) and
        clears any HTTP client references — the parent's client belongs to the
        parent's event loop and would race with concurrent subagents.

        ``Model`` is a ``@dataclass`` (not Pydantic) so ``copy.copy`` is the
        canonical clone. ``model_copy`` exists only on Pydantic ``BaseModel``
        subclasses; we still try it first because some user-supplied model
        wrappers may be Pydantic-based.
        """
        from agentica.model.usage import Usage

        if hasattr(source_model, "model_copy"):
            cloned = source_model.model_copy()
        else:
            cloned = copy.copy(source_model)
        cloned.tools = None
        cloned.functions = None
        cloned.function_call_stack = None
        cloned.tool_choice = None
        cloned.metrics = {}
        cloned.usage = Usage()
        for attr in ("client", "http_client", "async_client"):
            if hasattr(cloned, attr):
                setattr(cloned, attr, None)
        return cloned

    async def _run_child_streaming(
        self,
        parent_agent: "Agent",
        child: "Agent",
        task: str,
        run_id: Optional[str] = None,
        output_sink: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Drive the subagent through ``run_stream`` and bubble events to parent CLI.

        Always streams (zero-cost when no ``_event_callback`` is wired) so
        ``BuiltinTaskTool`` and any external caller share the same execution
        loop. ``run_id`` is propagated into every emitted event so the CLI can
        correlate concurrent subagents (e.g. for ``[1] [2]`` batch prefixes).

        If ``output_sink`` is provided, ``content`` and ``tool_calls_summary``
        are mirrored to it live on every chunk. This lets ``spawn()`` recover
        partial output when the coroutine is cancelled by ``asyncio.wait_for``
        (timeout) or by the user (Ctrl+C). Without this, all work done before
        the interrupt is silently discarded.

        Returns ``{content, tool_calls_summary}``.
        """
        from agentica.run_config import RunConfig
        from agentica.tools.builtin_task_tool import BuiltinTaskTool

        cb: Optional[Callable[[Dict[str, Any]], None]] = parent_agent._event_callback
        if cb is not None:
            child._event_callback = cb
            cb({
                "type": "subagent.start",
                "run_id": run_id,
                "agent_name": child.name,
                "task": task,
            })

        # ``chunk.tools`` mirrors ``agent.run_response.tools`` — the cumulative
        # list of every tool call in the run so far. Iterating it on each chunk
        # without dedupe re-emits events for every previously-seen tool, which
        # both spams the parent CLI and inflates ``tool_count``. Track which
        # tool_call_ids we have already announced to keep events one-per-call.
        seen_started: set = set()
        seen_completed: set = set()
        tool_calls_log: List[Dict[str, Any]] = []
        log_index_by_id: Dict[str, int] = {}
        final_content = ""

        def _mirror_to_sink() -> None:
            """Copy current progress into the caller-provided sink so partial
            output survives cancellation / timeout."""
            if output_sink is None:
                return
            output_sink["content"] = final_content
            output_sink["tool_calls_summary"] = [
                {"name": tc["name"], "info": tc.get("info", "")}
                for tc in tool_calls_log
            ]

        _mirror_to_sink()

        async for chunk in child.run_stream(task, config=RunConfig(stream_intermediate_steps=True)):
            if chunk is None:
                continue
            if chunk.event in ("ToolCallStarted", "ToolCallCompleted") and chunk.tools:
                for tool_info in chunk.tools:
                    tool_name = tool_info.get("tool_name") or tool_info.get("name", "")
                    if not tool_name:
                        continue
                    call_id = tool_info.get("tool_call_id") or tool_info.get("id")
                    tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                    content = tool_info.get("content")
                    brief = BuiltinTaskTool._format_tool_brief(tool_name, tool_args, content)
                    if chunk.event == "ToolCallStarted":
                        if call_id and call_id in seen_started:
                            continue
                        if call_id:
                            seen_started.add(call_id)
                            log_index_by_id[call_id] = len(tool_calls_log)
                        tool_calls_log.append({"name": tool_name, "info": brief})
                        _mirror_to_sink()
                        if cb is not None:
                            cb({
                                "type": "subagent.tool_started",
                                "run_id": run_id,
                                "agent_name": child.name,
                                "tool_name": tool_name,
                                "info": brief,
                                "args": tool_args if isinstance(tool_args, dict) else {},
                            })
                    else:  # ToolCallCompleted
                        # Ignore entries that have not actually completed in
                        # this chunk (no content / metrics yet).
                        if content is None and not (tool_info.get("metrics") or {}).get("time"):
                            continue
                        if call_id and call_id in seen_completed:
                            continue
                        if call_id:
                            seen_completed.add(call_id)
                            idx = log_index_by_id.get(call_id)
                            if idx is not None:
                                tool_calls_log[idx]["info"] = brief
                                _mirror_to_sink()
                        if cb is not None:
                            elapsed_t = (tool_info.get("metrics") or {}).get("time")
                            cb({
                                "type": "subagent.tool_completed",
                                "run_id": run_id,
                                "agent_name": child.name,
                                "tool_name": tool_name,
                                "info": brief,
                                "elapsed": elapsed_t,
                                "is_error": tool_info.get("tool_call_error", False),
                            })
            if chunk.event == "RunResponse" and chunk.content:
                final_content += str(chunk.content)
                _mirror_to_sink()

        if cb is not None:
            cb({
                "type": "subagent.end",
                "run_id": run_id,
                "agent_name": child.name,
                "response": final_content,
                "tool_count": len(tool_calls_log),
            })

        return {
            "content": final_content,
            "tool_calls_summary": [
                {"name": tc["name"], "info": tc.get("info", "")}
                for tc in tool_calls_log
            ],
        }

    async def spawn(
        self,
        parent_agent: "Agent",
        task: str,
        agent_type: Union[str, SubagentType] = SubagentType.CODE,
        context: str = "",
        depth: int = 1,
        model_override: Optional["Model"] = None,
        timeout_override: Optional[int] = None,
        max_turns_override: Optional[int] = None,
        tool_call_limit_override: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
        resume_from_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Spawn an isolated subagent and run it to completion.

        Single source of truth for subagent execution. Handles:
          - depth check (``MAX_DEPTH=2``) and nested-spawn permission
          - registry registration / status updates
          - parent model cloning (isolated tools/functions/usage/HTTP client)
          - tool inheritance (parent's tools, filtered by ``BLOCKED_TOOLS`` +
            config ``allowed_tools`` / ``denied_tools``; ``Agent._post_init``
            clones each ``Tool`` again to keep agent-bound state isolated)
          - streamed event bubbling to parent's ``_event_callback``
          - usage merge back into parent model
          - timeout enforcement and graceful error reporting

        Args:
            parent_agent: The parent Agent instance.
            task: Task description for the subagent.
            agent_type: Type of subagent (determines tool permissions).
            context: Optional context to inject into the subagent prompt.
            depth: Current nesting depth (1 = direct child of user agent).
            model_override: Optional model to use for the subagent. When ``None``
                the parent's model is cloned. Useful when the caller wants the
                subagent to use a cheaper/faster model than the parent.

        Returns:
            Dict with keys ``status``, ``content``, ``agent_type``, ``run_id``,
            and on completion ``tool_calls_summary``, ``execution_time``,
            ``tool_count``. On error ``error`` is populated and ``content`` is
            empty.
        """
        # ``Agent.context`` is typed as ``Optional[Dict[str, Any]]`` but other
        # parts of the runtime accept loose shapes (string summaries are even
        # rendered by ``_build_inherited_context`` below). Coerce to a dict so
        # ``.get`` is always safe — non-dict context simply contributes no
        # depth/permission hints.
        raw_parent_context = parent_agent.context
        parent_context: Dict[str, Any] = (
            raw_parent_context if isinstance(raw_parent_context, dict) else {}
        )
        inherited_depth = int(parent_context.get("_subagent_depth", 0)) + 1
        depth = max(depth, inherited_depth)

        if depth > self.MAX_DEPTH:
            return {
                "status": "error",
                "error": f"Max subagent depth exceeded ({self.MAX_DEPTH})",
                "agent_type": str(agent_type),
                "content": "",
            }

        if depth > 1 and not bool(parent_context.get("_can_spawn_subagents", True)):
            return {
                "status": "error",
                "error": "Nested subagent spawning is not allowed for this subagent type.",
                "agent_type": str(agent_type),
                "content": "",
            }

        config = get_subagent_config(agent_type)
        if config is None:
            return {
                "status": "error",
                "error": f"Unknown subagent type: {agent_type}",
                "agent_type": str(agent_type),
                "content": "",
            }

        # Apply per-call overrides so the parent Agent's ReAct loop can retry
        # a failed / truncated task with a larger budget or a tweaked prompt
        # without having to register a whole new subagent type. We copy the
        # config to avoid mutating the shared registry entry.
        if any(v is not None for v in (
            timeout_override, max_turns_override,
            tool_call_limit_override, system_prompt_override,
        )):
            config = dataclasses.replace(
                config,
                timeout=timeout_override if timeout_override is not None else config.timeout,
                max_turns=max_turns_override if max_turns_override is not None else config.max_turns,
                tool_call_limit=(
                    tool_call_limit_override if tool_call_limit_override is not None
                    else config.tool_call_limit
                ),
                system_prompt=(
                    system_prompt_override if system_prompt_override is not None
                    else config.system_prompt
                ),
            )

        # Resume support: pull the previous run's partial output and stitch it
        # into the new task so the subagent continues from where it left off.
        # This is the missing piece that lets the parent Agent iterate on a
        # timed-out / truncated task instead of having to restart from zero.
        if resume_from_run_id:
            prev = self.get(resume_from_run_id)
            if prev is None:
                return {
                    "status": "error",
                    "error": f"resume_from_run_id={resume_from_run_id!r} not found in registry",
                    "agent_type": config.type.value,
                    "content": "",
                }
            prev_partial = (prev.result or "").strip()
            prev_status = prev.status
            resume_prefix = (
                f"[RESUME] You are continuing a previous {config.type.value} subagent run "
                f"(run_id={resume_from_run_id}, previous status={prev_status}). "
                f"Below is the partial output from that run — read it, then finish the task. "
                f"Do NOT redo work already done; pick up where it stopped.\n\n"
                f"--- PREVIOUS PARTIAL OUTPUT ---\n{prev_partial or '(empty)'}\n"
                f"--- END OF PREVIOUS OUTPUT ---\n\n"
                f"[CONTINUATION TASK]\n"
            )
            task = resume_prefix + task

        source_model = model_override or parent_agent.model
        if source_model is None:
            return {
                "status": "error",
                "error": "No model available for subagent. Configure a model on the parent agent.",
                "agent_type": config.type.value,
                "content": "",
            }

        run_id = str(uuid.uuid4())
        run = SubagentRun(
            run_id=run_id,
            subagent_type=config.type,
            parent_agent_id=parent_agent.agent_id,
            task_label=task[:50] + "..." if len(task) > 50 else task,
            task_description=task,
            started_at=datetime.now(),
            status="running",
        )
        self.register(run)

        _parent_label = parent_agent.name or parent_agent.agent_id
        logger.chat(
            f"[spawn] {_parent_label} -> {config.type.value} subagent: "
            f"{task[:120]}{'...' if len(task) > 120 else ''}"
        )

        parent_tools = parent_agent.tools or []
        child_tools = self._select_child_tools(parent_tools, config)

        instructions_parts: List[str] = [config.system_prompt]

        # Structured handoff: default_handoff_mapper bundles parent identity,
        # condensed instructions, workspace summary, recent history and the
        # caller-supplied context string into one Markdown block. When
        # config.inherit_context is False we still emit the user-supplied
        # `context` arg via the mapper's extra_context slot, but skip the
        # parent-derived sections (instructions/workspace/history) so a
        # non-inheriting subagent stays isolated.
        handoff_ctx = default_handoff_mapper(
            parent_agent=parent_agent,
            task=task,
            extra_context=context or None,
        )
        if not config.inherit_context:
            handoff_ctx.parent_instructions = None
            handoff_ctx.parent_workspace_summary = None
            handoff_ctx.parent_history_excerpt = None
        instructions_parts.append("\n" + handoff_ctx.render())
        instructions_parts.append("\nBe direct and efficient. Complete the task and stop.")

        from agentica.agent import Agent
        from agentica.agent.config import PromptConfig, ToolConfig

        child_kwargs: Dict[str, Any] = dict(
            name=f"{parent_agent.name or 'agent'}_sub_{config.name}",
            model=self._clone_parent_model(source_model),
            instructions="\n".join(instructions_parts),
            tools=child_tools,
            prompt_config=PromptConfig(markdown=True),
            tool_config=ToolConfig(tool_call_limit=config.tool_call_limit),
            context={
                "_subagent_depth": depth,
                "_can_spawn_subagents": config.can_spawn_subagents,
            },
        )
        if config.inherit_workspace and parent_agent.workspace is not None:
            child_kwargs["workspace"] = parent_agent.workspace
        if config.inherit_knowledge and parent_agent.knowledge is not None:
            child_kwargs["knowledge"] = parent_agent.knowledge

        child = Agent(**child_kwargs)
        child._max_turns = config.max_turns

        # Wire arch_v5.md Phase 0 lineage so the child's RunContext records
        # who spawned it (used by Runner._run_impl to set parent_run_id +
        # RunSource.subagent on the child run). Read once by the Runner and
        # cleared after the run. Both attributes are declared on Agent
        # (run_context defaults to None when no run is active).
        if parent_agent.run_context is not None:
            child._parent_run_id = parent_agent.run_context.run_id
        elif parent_agent.run_id:
            child._parent_run_id = parent_agent.run_id

        start_time = time.time()
        # Shared mutable sink so we can recover partial output when the child
        # run is aborted by timeout / user cancel / unexpected exception.
        # Without this, ``final_content`` lives only inside ``_run_child_streaming``
        # and is silently discarded on interrupt — which is exactly why users
        # were seeing "task failed with no content" for long-running subagents.
        partial_sink: Dict[str, Any] = {"content": "", "tool_calls_summary": []}
        try:
            run_coro = self._run_child_streaming(
                parent_agent, child, task, run_id=run_id, output_sink=partial_sink
            )
            if config.timeout > 0:
                stream_result = await asyncio.wait_for(run_coro, timeout=config.timeout)
            else:
                stream_result = await run_coro
        except asyncio.TimeoutError:
            elapsed = round(time.time() - start_time, 3)
            partial_content = partial_sink.get("content") or ""
            partial_tools = partial_sink.get("tool_calls_summary") or []
            logger.warning(
                "Subagent timed out after %ss (partial: %d chars, %d tool calls)",
                config.timeout, len(partial_content), len(partial_tools),
            )
            timeout_note = (
                f"[Subagent timed out after {config.timeout}s. "
                f"Partial output below reflects {len(partial_tools)} completed tool call(s).]"
            )
            content_out = (
                f"{timeout_note}\n\n{partial_content}" if partial_content else timeout_note
            )
            self.update_status(run_id=run_id, status="timeout", result=content_out)
            return {
                "status": "timeout",
                "error": f"Subagent timed out after {config.timeout} seconds",
                "agent_type": config.type.value,
                "subagent_name": config.name,
                "run_id": run_id,
                "content": content_out,
                "tool_calls_summary": partial_tools,
                "tool_count": len(partial_tools),
                "elapsed_seconds": elapsed,
                "partial": True,
                "next_action": (
                    f"Partial output above reflects the work completed before the "
                    f"{config.timeout}s timeout. Prefer synthesizing it directly. Only if the "
                    "task genuinely needs to run to completion may you resume it once via the "
                    f"task tool with resume_from_run_id={run_id!r} and a larger timeout "
                    f"(e.g. timeout={config.timeout * 2}); do not resume repeatedly."
                ),
            }
        except (asyncio.CancelledError, AgentCancelledError):
            # Distinguish two paths that both surface as CancelledError:
            #
            # 1. ``asyncio.wait_for`` fires its own cancel on timeout, and if
            #    the inner ``run_stream`` catches ``CancelledError`` and
            #    re-raises as ``AgentCancelledError`` (which it does), the
            #    ``TimeoutError`` branch above never sees it. We reroute to
            #    the timeout payload by checking elapsed time.
            #
            # 2. Real user Ctrl+C: propagate so the whole parent run tears
            #    down cleanly (but still persist partial output to registry).
            elapsed = round(time.time() - start_time, 3)
            partial_content = partial_sink.get("content") or ""
            partial_tools = partial_sink.get("tool_calls_summary") or []

            if config.timeout > 0 and elapsed >= config.timeout - 0.5:
                logger.warning(
                    "Subagent timed out after %ss via cancel path "
                    "(partial: %d chars, %d tool calls)",
                    config.timeout, len(partial_content), len(partial_tools),
                )
                timeout_note = (
                    f"[Subagent timed out after {config.timeout}s. "
                    f"Partial output below reflects {len(partial_tools)} completed tool call(s).]"
                )
                content_out = (
                    f"{timeout_note}\n\n{partial_content}" if partial_content else timeout_note
                )
                self.update_status(run_id=run_id, status="timeout", result=content_out)
                return {
                    "status": "timeout",
                    "error": f"Subagent timed out after {config.timeout} seconds",
                    "agent_type": config.type.value,
                    "subagent_name": config.name,
                    "run_id": run_id,
                    "content": content_out,
                    "tool_calls_summary": partial_tools,
                    "tool_count": len(partial_tools),
                    "elapsed_seconds": elapsed,
                    "partial": True,
                    "next_action": (
                        f"Partial output above reflects the work completed before the "
                        f"{config.timeout}s timeout. Prefer synthesizing it directly. Only if "
                        "the task genuinely needs to run to completion may you resume it once "
                        f"via the task tool with resume_from_run_id={run_id!r} and a larger "
                        f"timeout (e.g. timeout={config.timeout * 2}); do not resume repeatedly."
                    ),
                }

            # Genuine cancel: persist partial + propagate.
            cancel_note = (
                f"[Subagent cancelled after {elapsed}s. "
                f"Partial output reflects {len(partial_tools)} completed tool call(s).]"
            )
            content_out = (
                f"{cancel_note}\n\n{partial_content}" if partial_content else cancel_note
            )
            self.update_status(
                run_id=run_id, status="cancelled", result=content_out, error="cancelled by user"
            )
            raise
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            partial_content = partial_sink.get("content") or ""
            partial_tools = partial_sink.get("tool_calls_summary") or []
            logger.error(f"Subagent execution failed: {e}")
            self.update_status(run_id=run_id, status="error", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "agent_type": config.type.value,
                "subagent_name": config.name,
                "run_id": run_id,
                "content": partial_content,
                "tool_calls_summary": partial_tools,
                "tool_count": len(partial_tools),
                "elapsed_seconds": elapsed,
                "partial": bool(partial_content or partial_tools),
                "next_action": (
                    (
                        "Partial output above was recovered before the error. Prefer using it "
                        "directly. If the task must be completed you may resume it once via the "
                        f"task tool with resume_from_run_id={run_id!r}; do not retry repeatedly."
                    )
                    if partial_content or partial_tools else
                    (
                        "The subagent failed before producing any output. If this task is "
                        "essential you may retry once with a simpler description or a different "
                        "subagent_type; otherwise proceed without it."
                    )
                ),
            }

        elapsed = round(time.time() - start_time, 3)
        raw_content = stream_result["content"]
        tool_calls_summary = stream_result["tool_calls_summary"]

        # Detect ``Runner`` graceful break (max_turns / tool_call_limit). When
        # this happens the run returns normally but ``break_reason`` is set on
        # the last RunResponse. Surface it as a first-class status so callers
        # know the answer was truncated by budget rather than by the model
        # declaring itself done.
        break_reason = None
        run_response = getattr(child, "run_response", None)
        if run_response is not None:
            break_reason = getattr(run_response, "break_reason", None)

        if parent_agent.model is not None and child.model is not None:
            parent_agent.model.usage.merge(child.model.usage)

        if break_reason:
            # ``break_reason`` from runner.py is a plain string like
            # "MAX_TURNS" / "TOOL_CALL_LIMIT". Normalise to lowercase status.
            reason_str = str(break_reason).lower()
            if "turn" in reason_str:
                status_out = "max_turns"
            elif "tool" in reason_str:
                status_out = "tool_call_limit"
            else:
                status_out = "truncated"
            note = (
                f"[Subagent stopped at {reason_str} limit after {len(tool_calls_summary)} "
                f"tool call(s). Output below is what was produced before the limit.]"
            )
            final_content = f"{note}\n\n{raw_content}" if raw_content else note
            self.update_status(run_id=run_id, status=status_out, result=final_content)
            return {
                "status": status_out,
                "error": f"stopped at {reason_str} limit",
                "content": final_content,
                "agent_type": config.type.value,
                "subagent_name": config.name,
                "run_id": run_id,
                "tool_calls_summary": tool_calls_summary,
                "tool_count": len(tool_calls_summary),
                "elapsed_seconds": elapsed,
                "partial": True,
                "next_action": (
                    f"Partial output above was produced before the {reason_str} limit. Prefer "
                    "synthesizing it directly. Only if the task genuinely needs to finish may "
                    f"you resume it once via the task tool with resume_from_run_id={run_id!r} "
                    f"and a larger budget (e.g. max_turns={(config.max_turns or 100) * 2}, "
                    f"tool_call_limit={(config.tool_call_limit or 100) * 2}); do not resume "
                    "repeatedly."
                ),
            }

        final_content = raw_content or "Subagent completed but returned no content."
        self.update_status(run_id=run_id, status="completed", result=final_content)

        logger.chat(
            f"[return] {config.type.value} subagent -> {parent_agent.name or parent_agent.agent_id}: "
            f"{(final_content or '')[:120]}{'...' if len(final_content or '') > 120 else ''}"
        )

        return {
            "status": "completed",
            "content": final_content,
            "agent_type": config.type.value,
            "subagent_name": config.name,
            "run_id": run_id,
            "tool_calls_summary": stream_result["tool_calls_summary"],
            "tool_count": len(stream_result["tool_calls_summary"]),
            "execution_time": elapsed,
        }

    async def spawn_batch(
        self,
        parent_agent: "Agent",
        tasks: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Spawn multiple subagents concurrently with bounded parallelism.

        Args:
            parent_agent: The parent Agent instance.
            tasks: List of task specs, each with keys:
                - "task" (str, required): Task description
                - "type" (str/SubagentType, optional): Agent type (default: CODE)
                - "context" (str, optional): Additional context
            max_concurrent: Max parallel subagents (default: MAX_CONCURRENT).

        Returns:
            List of result dicts in same order as input tasks.
        """
        sem = asyncio.Semaphore(max_concurrent or self.MAX_CONCURRENT)

        async def _run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                task = spec.get("task")
                if not isinstance(task, str) or not task.strip():
                    return {
                        "status": "error",
                        "error": "Task spec missing required 'task' field",
                        "agent_type": str(spec.get("type", SubagentType.CODE)),
                        "content": "",
                    }
                # No broad ``except Exception`` here on purpose: ``spawn()``
                # already converts every operational failure (timeout, model
                # error, tool failure, depth limit, …) into a ``status=error``
                # dict. Anything that still propagates is a programmer error
                # (e.g. wrong agent_type / context shape, AttributeError on
                # parent_agent), and we WANT it to crash loudly instead of
                # silently degrading the whole batch into "task failed".
                # Cancellation / timeouts inherit BaseException (3.8+) and
                # propagate through ``gather`` automatically.
                return await self.spawn(
                    parent_agent=parent_agent,
                    task=task,
                    agent_type=spec.get("type", SubagentType.CODE),
                    context=spec.get("context", ""),
                )

        return list(await asyncio.gather(*[_run_one(t) for t in tasks]))


# ============== Default Subagent Configurations ==============

# Explore agent: specialized for codebase exploration (read-only)
EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.EXPLORE,
    name="Explore Agent",
    description="""Fast agent specialized for exploring codebases and searching for information.
Use this agent when you need to:
- Search for files using glob patterns
- Search file contents with regex
- Read and analyze source code
- Understand project structure""",
    system_prompt="""You are a file search specialist. You excel at thoroughly navigating and exploring codebases.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use glob for broad file pattern matching
- Use grep for searching file contents with regex
- Use read_file when you know the specific file path you need to read
- Use ls to list directory contents and understand project structure
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Do NOT create or modify any files - you are read-only
- Do NOT run commands that modify the user's system state

Complete the user's search request efficiently and report your findings clearly.""",
    allowed_tools=["ls", "read_file", "glob", "grep"],  # Read-only tools
    denied_tools=["write_file", "edit_file", "execute", "task"],  # No write/execute/spawn
    tool_call_limit=150,
    max_turns=150,
    timeout=1800,
    can_spawn_subagents=False,
)


# Research agent: web search and analysis
RESEARCH_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.RESEARCH,
    name="Research Agent",
    description="""Research agent specialized for web search and document analysis.
Use this agent for:
- Searching the web for information
- Fetching and analyzing web pages
- Synthesizing research findings""",
    system_prompt="""You are a research specialist that excels at finding and analyzing information.

Guidelines:
1. Use web_search to find relevant information on the web
2. Use fetch_url to read web page contents
3. Synthesize your findings into a clear, well-organized summary
4. Cite your sources when providing information
5. Be objective and fact-based in your analysis

Complete your research task and provide a comprehensive summary of your findings.""",
    allowed_tools=["web_search", "fetch_url", "read_file", "ls", "glob", "grep"],
    denied_tools=["write_file", "edit_file", "execute", "task"],
    tool_call_limit=150,
    max_turns=150,
    timeout=1800,
    can_spawn_subagents=False,
)


# Code agent: code generation and execution
CODE_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.CODE,
    name="Code Agent",
    description="""Code agent specialized for code generation and execution.
Use this agent for:
- Writing and executing code
- Running tests and commands
- Code analysis and debugging""",
    system_prompt="""You are a code specialist that excels at writing and executing code.

Guidelines:
1. Write clean, well-documented code
2. Test your code before reporting results
3. Handle errors gracefully and report them clearly
4. Follow best practices for the programming language being used
5. Provide clear explanations of what your code does

Complete your coding task and provide a summary of the results.""",
    allowed_tools=["read_file", "write_file", "edit_file", "execute", "ls", "glob", "grep"],
    denied_tools=["task"],  # Cannot spawn nested subagents
    tool_call_limit=200,
    max_turns=200,
    timeout=1800,
    can_spawn_subagents=False,
    inherit_context=True,  # Code tasks benefit from parent context
)


# Registry of all default subagent configurations
DEFAULT_SUBAGENT_CONFIGS: Dict[SubagentType, SubagentConfig] = {
    SubagentType.EXPLORE: EXPLORE_SUBAGENT_CONFIG,
    SubagentType.RESEARCH: RESEARCH_SUBAGENT_CONFIG,
    SubagentType.CODE: CODE_SUBAGENT_CONFIG,
}

# Custom subagent configurations (user-defined, keyed by string name)
_CUSTOM_SUBAGENT_CONFIGS: Dict[str, SubagentConfig] = {}


def register_custom_subagent(
    name: str,
    description: str,
    system_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    denied_tools: Optional[List[str]] = None,
    tool_call_limit: int = 100,
) -> SubagentConfig:
    """
    Register a custom subagent type.
    
    This allows users to define their own subagent types without modifying code.
    Custom subagents are accessible by their name string (case-insensitive).
    
    Args:
        name: Unique name for the subagent (e.g., "code-reviewer", "data-analyst")
        description: Description of what this subagent does
        system_prompt: System prompt for the subagent
        allowed_tools: List of allowed tool names (None = inherit from parent)
        denied_tools: List of denied tool names
        tool_call_limit: Maximum number of tool calls allowed for this subagent
        
    Returns:
        The created SubagentConfig
        
    Example:
        >>> register_custom_subagent(
        ...     name="code-reviewer",
        ...     description="Reviews code for quality and bugs",
        ...     system_prompt="You are a code review expert...",
        ...     allowed_tools=["read_file", "ls", "glob", "grep"],
        ...     tool_call_limit=10,
        ... )
    """
    config = SubagentConfig(
        type=SubagentType.CUSTOM,  # Custom subagents have their own type
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        denied_tools=denied_tools or ["task"],  # Prevent nesting by default
        tool_call_limit=tool_call_limit,
        can_spawn_subagents=False,
    )
    _CUSTOM_SUBAGENT_CONFIGS[name.lower()] = config
    logger.info(f"Registered custom subagent: {name}")
    return config


def unregister_custom_subagent(name: str) -> bool:
    """
    Unregister a custom subagent type.
    
    Args:
        name: Name of the subagent to unregister
        
    Returns:
        True if found and removed, False otherwise
    """
    key = name.lower()
    if key in _CUSTOM_SUBAGENT_CONFIGS:
        del _CUSTOM_SUBAGENT_CONFIGS[key]
        logger.info(f"Unregistered custom subagent: {name}")
        return True
    return False


def get_subagent_config(subagent_type: Union[str, SubagentType]) -> Optional[SubagentConfig]:
    """
    Get the configuration for a subagent type.

    Lookup order:
    1. Custom subagent configs (by name string)
    2. Default subagent configs (by SubagentType enum)
    3. Aliases (e.g., "explorer" -> EXPLORE)
    """
    if isinstance(subagent_type, str):
        # First check custom configs (case-insensitive)
        custom_config = _CUSTOM_SUBAGENT_CONFIGS.get(subagent_type.lower())
        if custom_config is not None:
            return custom_config

        # Then try to parse as SubagentType enum
        try:
            subagent_type = SubagentType(subagent_type)
        except ValueError:
            # Try mapping common aliases
            aliases = {
                "explorer": SubagentType.EXPLORE,
                "researcher": SubagentType.RESEARCH,
                "coder": SubagentType.CODE,
            }
            subagent_type = aliases.get(subagent_type.lower())
            if subagent_type is None:
                return None

    return DEFAULT_SUBAGENT_CONFIGS.get(subagent_type)


def get_available_subagent_types() -> List[Dict[str, str]]:
    """
    Get a list of available subagent types with their descriptions.
    
    Returns both default and custom subagent types.
    """
    result = []
    
    # Add default configs
    for config in DEFAULT_SUBAGENT_CONFIGS.values():
        result.append({
            "type": config.type.value,
            "name": config.name,
            "description": config.description,
            "is_custom": False,
        })
    
    # Add custom configs
    for name, config in _CUSTOM_SUBAGENT_CONFIGS.items():
        result.append({
            "type": name,  # Custom types use their name as type
            "name": config.name,
            "description": config.description,
            "is_custom": True,
        })
    
    return result


def get_custom_subagent_configs() -> Dict[str, SubagentConfig]:
    """Get all registered custom subagent configurations."""
    return _CUSTOM_SUBAGENT_CONFIGS.copy()
