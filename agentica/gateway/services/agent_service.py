# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Agent service - wraps the agentica SDK.

Key design decisions:
- LRU cache for Agent instances (bounded by settings.agent_max_sessions)
- Per-session work_dir stored separately from global settings
- Fail fast on initialization errors (no silent mock mode)
- cancel_session(session_id) for precise stream cancellation
- Agent build timeout to guard against SDK hangs
- Uses DeepAgent (batteries-included) instead of manual Agent + builtin tools
- Per-session stream lock prevents concurrent streams on the same session
"""
import asyncio
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Any, Dict

from agentica.utils.log import logger
from agentica import DeepAgent
from agentica.run_display import RunDisplayEventKind, classify_run_response
from agentica.run_response import AgentCancelledError
from agentica.run_config import RunConfig
from agentica.run_context import RunSource
from agentica.workspace import Workspace
from agentica.global_config import (
    apply_global_config,
    set_active_profile,
    provider_api_key_env,
)
from agentica.memory.session_log import SessionLog

from ..config import settings
from .model_factory import (
    create_model, get_cron_tools, get_cron_instructions,
    get_self_manage_tools, get_self_manage_instructions,
)
from .response_formatter import extract_metrics, format_tool_call_args, format_tool_result

# Timeout in seconds for building a new Agent instance (guards against SDK hangs)
_AGENT_BUILD_TIMEOUT_S = 30

# Session-id prefix reserved for scheduled (cron) job runs. These sessions are
# never surfaced by list_sessions() (the chat sidebar) — cron execution
# history lives in the dedicated TaskRun store (agentica.cron.jobs), not the
# chat session log. See AgentService.run_cron().
CRON_SESSION_PREFIX = "scheduled_"

# Web sessions default to "ask" approval mode, which strips write tools
# (write_file/edit_file/execute) from the schema sent to the model (see
# _run_config_for_session below) — but the agent's static tool instructions
# still describe the full toolset for prompt-cache reasons. Without this, a
# model that tries a stripped tool anyway gets back an opaque "Function not
# found" instead of understanding why. Baked in once as a standing agent
# instruction (not per-message) so it doesn't bloat session history and
# stays part of the cache-friendly static prompt zone.
_APPROVAL_MODE_INSTRUCTION = (
    "This session's approval mode can restrict tool access at runtime: in "
    "\"ask\" mode, only read-only tools are enabled "
    "(ls/read_file/glob/grep/web_search/fetch_url/task) — write_file, "
    "edit_file, and execute are disabled. In \"auto\" mode, writes are "
    "restricted to the session's work_dir. If a tool call unexpectedly "
    "fails with \"Function ... not found\", it almost certainly means the "
    "current approval mode disabled it — do not retry the call. Instead, "
    "tell the user the current mode is read-only and that they need to "
    "switch to \"auto\" or \"allow-all\" mode (in the approval selector "
    "next to the send button) to edit files or run commands."
)


@dataclass
class ChatResult:
    """Chat response from the agent."""
    content: str
    tool_calls: int = 0
    session_id: str = ""
    user_id: str = ""
    tools_used: List[str] = field(default_factory=list)
    reasoning: str = ""
    metrics: Optional[Dict[str, Any]] = None


class LRUAgentCache:
    """Thread-unsafe but asyncio-safe LRU cache for DeepAgent instances."""

    def __init__(self, max_size: int = 50):
        self._cache: OrderedDict[str, DeepAgent] = OrderedDict()
        self.max_size = max_size

    def get(self, session_id: str) -> Optional[DeepAgent]:
        if session_id not in self._cache:
            return None
        self._cache.move_to_end(session_id)
        return self._cache[session_id]

    def put(self, session_id: str, agent: DeepAgent) -> None:
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            self._cache[session_id] = agent
            return
        self._cache[session_id] = agent
        if len(self._cache) > self.max_size:
            evicted_id, _ = self._cache.popitem(last=False)
            logger.debug(f"LRU evicted agent for session: {evicted_id}")

    def delete(self, session_id: str) -> bool:
        if session_id in self._cache:
            del self._cache[session_id]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)


class AgentService:
    """Agent service wrapping the agentica SDK.

    Features:
    - Workspace config layer (AGENT.md, PERSONA.md, MEMORY.md, etc.)
    - Session history management (per session_id)
    - LRU-bounded Agent instance cache (evicts on overflow)
    - Per-session working directory
    - Per-session run lock (prevents concurrent chat/stream on same session)
    - Scheduler tool integration
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        extra_tools: Optional[List[Any]] = None,
        extra_instructions: Optional[List[str]] = None,
    ):
        self.workspace_path = Path(workspace_path or settings.workspace_path).expanduser()
        self.extra_tools = extra_tools or []
        self.extra_instructions = extra_instructions or []

        self._cache = LRUAgentCache(max_size=settings.agent_max_sessions)
        # Per-session work_dir overrides; falls back to settings.base_dir
        self._session_work_dirs: Dict[str, str] = {}
        self._session_approval_modes: Dict[str, str] = {}
        # Per-session run locks: prevents concurrent runs (chat or stream) on the same session.
        # The underlying Agent instance is NOT thread-safe for concurrent reuse.
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._workspace: Optional[Workspace] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ============== Model config (single source of truth: `settings`) ==============
    # These proxy directly to the gateway's global `settings` singleton instead
    # of keeping a duplicate copy, so routes only ever need to write to one
    # place (settings.xxx) and every reader (here, routes/config.py, etc.)
    # sees the same value with no risk of drift.

    @property
    def model_provider(self) -> str:
        return settings.model_provider

    @model_provider.setter
    def model_provider(self, value: str) -> None:
        settings.model_provider = value

    @property
    def model_name(self) -> str:
        return settings.model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        settings.model_name = value

    @property
    def model_base_url(self) -> str:
        return settings.model_base_url

    @property
    def model_api_key(self) -> str:
        return settings.model_api_key

    @property
    def model_reasoning_effort(self) -> str:
        return settings.model_reasoning_effort

    @property
    def max_tokens(self) -> int:
        return settings.max_tokens

    @property
    def temperature(self) -> float:
        return settings.temperature

    @property
    def top_p(self) -> float:
        return settings.top_p

    @property
    def context_window(self) -> int:
        return settings.context_window

    @property
    def auxiliary_model_provider(self) -> str:
        return settings.auxiliary_model_provider

    @property
    def auxiliary_model_name(self) -> str:
        return settings.auxiliary_model_name

    @property
    def auxiliary_base_url(self) -> str:
        return settings.auxiliary_base_url

    @property
    def auxiliary_api_key(self) -> str:
        return settings.auxiliary_api_key

    def _build_sibling_model(self, prefix: str) -> Optional[Any]:
        """Build a sibling (auxiliary) model if a model name is configured.

        Returns None when no sibling name is set — DeepAgent will reuse
        the main model. The auxiliary model also serves as the task subagent
        model (CLI unified them).
        """
        sibling_name = getattr(self, f"{prefix}_model_name")
        if not sibling_name:
            return None
        provider = getattr(self, f"{prefix}_model_provider") or self.model_provider
        base_url = getattr(self, f"{prefix}_base_url") or None
        api_key = getattr(self, f"{prefix}_api_key") or None
        return create_model(
            provider,
            sibling_name,
            base_url=base_url,
            api_key=api_key,
            thinking=settings.model_thinking,
        )

    # ============== Initialization ==============

    async def _ensure_initialized(self) -> None:
        """Ensure the workspace is initialized (idempotent, Lock-protected)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._do_initialize)

    def _do_initialize(self) -> None:
        """Initialize the shared Workspace (sync, runs in thread).

        Raises RuntimeError on failure — callers must handle this explicitly.
        No silent mock mode.
        """
        try:
            self._workspace = Workspace(self.workspace_path)
            if not self._workspace.exists():
                self._workspace.initialize()
                logger.info(f"Workspace initialized at {self.workspace_path}")

            self._initialized = True
            logger.info("AgentService initialized successfully")
            logger.info(f"Model: {self.model_provider}/{self.model_name}")
            logger.info(f"Workspace: {self.workspace_path}")

        except Exception as e:
            logger.error(
                f"AgentService initialization failed: {e}\n"
                f"Check your API key, model provider, and agentica version."
            )
            raise RuntimeError(f"AgentService init failed: {e}") from e

    def _build_agent(self, session_id: str) -> DeepAgent:
        """Build a new DeepAgent instance (sync, runs in thread).

        DeepAgent auto-includes: builtin tools, skills, agentic prompt,
        compression (compress_tool_results + context_overflow_threshold=0.8
        with compress-before-evict), workspace memory (auto_archive +
        auto_extract_memory + relevance recall), experience capture (tool
        errors / user corrections / success patterns), memory tools.

        auxiliary_model / task_model default to the main model (DeepAgent
        default). Pass AGENTICA_AUXILIARY_* / AGENTICA_TASK_* env vars (or build
        AgentService with the matching kwargs) to route them to a
        different provider / model / base_url / api_key.

        Scheduled (cron) jobs are unattended, don't need flagship-model
        quality, and can run frequently — so when an auxiliary model is
        configured, cron sessions use it as their *main* model to cut cost.
        Interactive chat sessions are unaffected.
        """
        if session_id.startswith(CRON_SESSION_PREFIX) and self.auxiliary_model_name:
            model = self._build_sibling_model("auxiliary")
            auxiliary_model = None
            task_model = None
        else:
            model = create_model(
                self.model_provider,
                self.model_name,
                base_url=self.model_base_url or None,
                api_key=self.model_api_key or None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                context_window=self.context_window,
                reasoning_effort=self.model_reasoning_effort,
                thinking=settings.model_thinking,
            )
            auxiliary_model = self._build_sibling_model("auxiliary")
            # The auxiliary model also serves as the task subagent model (CLI
            # unified them — no separate task_model config anymore).
            task_model = auxiliary_model
        # Per-session project dir (set via routes/chat.py::_apply_session_work_dir
        # from the frontend's session.dir) takes precedence — falls back to the
        # global settings.base_dir only when the session has none set.
        work_dir = self.get_session_work_dir(session_id)

        # Extra tools: user-provided + cron + self-management (self-awareness)
        extra = list(self.extra_tools)
        cron_tools = get_cron_tools()
        extra.extend(cron_tools)
        self_manage_tools = get_self_manage_tools()
        extra.extend(self_manage_tools)

        instructions = list(self.extra_instructions) if self.extra_instructions else None
        if cron_tools:
            if instructions is None:
                instructions = []
            instructions.append(get_cron_instructions())
        if self_manage_tools:
            if instructions is None:
                instructions = []
            instructions.append(get_self_manage_instructions())
        if instructions is None:
            instructions = []
        instructions.append(_APPROVAL_MODE_INSTRUCTION)

        agent = DeepAgent(
            session_id=session_id,
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            tools=extra if extra else None,
            workspace=self._workspace,
            work_dir=work_dir,
            num_history_turns=settings.num_history_turns,
            instructions=instructions,
            debug=settings.debug,
            # memory, skills, user input, experience capture, workspace memory
            # all on by DeepAgent default — no explicit overrides needed.
            include_ask_user_question=True,
            permission_mode=self.get_session_approval_mode(session_id),
        )

        tool_count = len(agent.tools) if agent.tools else 0
        logger.info(
            f"DeepAgent built: {tool_count} tools "
            f"(extra={len(extra)}, cron={len(cron_tools)}, self_manage={len(self_manage_tools)})"
        )
        return agent

    async def _get_agent(self, session_id: str) -> DeepAgent:
        """Return the cached Agent for a session, creating one if absent.

        Raises RuntimeError if the agent cannot be built (e.g. SDK error).
        Times out after _AGENT_BUILD_TIMEOUT_S seconds.
        """
        agent = self._cache.get(session_id)
        if agent is not None:
            return agent

        try:
            agent = await asyncio.wait_for(
                asyncio.to_thread(self._build_agent, session_id),
                timeout=_AGENT_BUILD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Agent build timed out after {_AGENT_BUILD_TIMEOUT_S}s "
                f"for session {session_id}. Check MCP server connectivity."
            )

        self._cache.put(session_id, agent)
        logger.info(f"Agent created for session: {session_id} (cache size: {len(self._cache)})")
        return agent

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return (or create) the per-session run lock.

        Both chat() and chat_stream() acquire this lock to prevent concurrent
        runs on the same Agent instance, which is not thread-safe.
        """
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    # Default approval mode for web sessions: "auto" — file edits and commands
    # are allowed (writes restricted to work_dir). "ask" is opt-in via the
    # approval selector next to the send button. See agentica.agent.permissions
    # for the exact semantics of each of the 3 tiers ("ask"/"auto"/"allow-all").
    _DEFAULT_APPROVAL_MODE = "auto"

    def set_session_approval_mode(self, session_id: str, mode: str) -> None:
        """Persist the selected approval mode for a session.

        If an Agent is already cached for this session, its permission mode
        is switched in place (no rebuild) via ``Agent.set_permission_mode``.
        """
        from agentica.agent.permissions import PERMISSION_MODES

        normalized = (mode or self._DEFAULT_APPROVAL_MODE).strip().lower()
        if normalized not in PERMISSION_MODES:
            normalized = self._DEFAULT_APPROVAL_MODE
        self._session_approval_modes[session_id] = normalized

        cached_agent = self._cache.get(session_id)
        if cached_agent is not None:
            cached_agent.set_permission_mode(normalized)

    def get_session_approval_mode(self, session_id: str) -> str:
        return self._session_approval_modes.get(session_id, self._DEFAULT_APPROVAL_MODE)

    def _run_config_for_session(
        self,
        session_id: str,
        source: RunSource,
        *,
        stream_intermediate_steps: bool = False,
    ) -> RunConfig:
        # Tool permission enforcement lives on the Agent itself now
        # (tool_config.permission_mode + sandbox_config, kept in sync with
        # the session's approval mode by set_session_approval_mode above).
        return RunConfig(stream_intermediate_steps=stream_intermediate_steps, source=source)


    # ============== Public API ==============

    def get_context_window(self, session_id: Optional[str] = None) -> int:
        """Return the context window size for the model used by a session.

        When ``session_id`` is omitted, returns the context window of an
        arbitrary cached agent (useful for a general status check before any
        specific session is known). Falls back to 128000 if no matching
        agent is cached yet (``context_window`` is a declared field on every
        Model, defaulting to 128000, so no per-call fallback is needed once
        an agent exists).
        """
        if session_id is None:
            session_id = next(iter(self._cache.keys()), None)
        agent = self._cache.get(session_id) if session_id else None
        if agent and agent.model:
            return agent.model.context_window
        return 128000

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        source: RunSource = RunSource.gateway,
    ) -> ChatResult:
        """Send a message and return the full response (non-streaming).

        Acquires a per-session lock to prevent concurrent runs on the same
        Agent instance (which is not thread-safe).

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier (for workspace memory isolation)

        Returns:
            ChatResult with content, tool_calls, metrics

        Raises:
            RuntimeError: If another run is already active on this session.
        """
        await self._ensure_initialized()

        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            agent = await self._get_agent(session_id)

            try:
                if self._workspace:
                    await asyncio.to_thread(self._workspace.set_user, user_id)

                response = await agent.run(
                    message,
                    config=self._run_config_for_session(session_id, source),
                )

                content = (response.content or "").strip()
                tools_used: List[str] = []
                tool_calls = 0

                if response.tools:
                    tool_calls = len(response.tools)
                    for tool in response.tools:
                        if isinstance(tool, dict):
                            tools_used.append(tool.get("tool_name", tool.get("name", "unknown")))
                        else:
                            tools_used.append(str(tool))

                return ChatResult(
                    content=content,
                    tool_calls=tool_calls,
                    session_id=session_id,
                    user_id=user_id,
                    tools_used=tools_used,
                    metrics=extract_metrics(agent),
                )

            except Exception as e:
                logger.error(f"AgentService.chat error (session={session_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                )

    async def run_cron(self, message: str, job_id: str, user_id: str = "default") -> ChatResult:
        """Run a scheduled job's prompt on a brand-new, independent Agent.

        Unlike chat()/chat_stream(), this never reads from or writes to the
        interactive-session Agent cache (self._cache): every cron run builds
        and discards its own Agent, so job run N+1 never inherits any
        conversation state from run N or from any interactive chat session.
        The per-job session id keeps run-lock / work_dir / approval-mode
        bookkeeping scoped to the job (one entry per job, not one per run),
        but is CRON_SESSION_PREFIX-tagged so it's excluded from
        list_sessions() (the chat sidebar).
        """
        await self._ensure_initialized()
        session_id = f"{CRON_SESSION_PREFIX}{job_id}"

        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(f"Job '{job_id}' already has an active run.")

        async with lock:
            agent = await asyncio.wait_for(
                asyncio.to_thread(self._build_agent, session_id),
                timeout=_AGENT_BUILD_TIMEOUT_S,
            )
            try:
                if self._workspace:
                    await asyncio.to_thread(self._workspace.set_user, user_id)

                response = await agent.run(
                    message,
                    config=self._run_config_for_session(session_id, RunSource.cron),
                )
                return ChatResult(
                    content=(response.content or "").strip(),
                    tool_calls=len(response.tools) if response.tools else 0,
                    session_id=session_id,
                    user_id=user_id,
                    metrics=extract_metrics(agent),
                )
            except Exception as e:
                logger.error(f"AgentService.run_cron error (job={job_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                )

    async def run_goal(
        self,
        objective: str,
        session_id: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """Drive a bounded standing-goal loop (Agent.run_goal) for the web UI's
        "/goal <objective>" command.

        Budgets are capped conservatively since this runs synchronously behind
        a single HTTP request in a local, single-user gateway — there is no
        UI for cancelling a runaway loop mid-flight yet.
        """
        await self._ensure_initialized()

        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            agent = await self._get_agent(session_id)
            if self._workspace:
                await asyncio.to_thread(self._workspace.set_user, user_id)
            # Carry the prior conversation into the goal loop so the model sees
            # the context the user has been building up. Web previously lost it
            # because run_goal() clones a fresh agent with empty working memory.
            seed_messages = agent.working_memory.get_messages()
            result = await agent.run_goal(
                objective,
                turn_budget=15,
                token_budget=80_000,
                wall_clock_budget_sec=300,
                seed_messages=seed_messages,
            )
            return {
                "status": result.status,
                "reason": result.reason,
                "content": result.response_content,
                "turns_used": result.turns_used,
            }

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        source: RunSource = RunSource.gateway,
        on_content: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
        on_tool_result: Optional[Callable[[str, str], Any]] = None,
        on_thinking: Optional[Callable[[str], Any]] = None,
    ) -> ChatResult:
        """Send a message and stream the response via callbacks.

        Acquires the per-session run lock to prevent concurrent runs
        (both chat and chat_stream) on the same Agent instance.

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier
            on_content: Called with each content delta
            on_tool_call: Called when a tool call starts (name, args)
            on_tool_result: Called when a tool call completes (name, result)
            on_thinking: Called with each reasoning delta

        Returns:
            ChatResult with accumulated content + metrics

        Raises:
            RuntimeError: If another run is already active on this session.
        """
        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            return await self._chat_stream_impl(
                message, session_id, user_id,
                source, on_content, on_tool_call, on_tool_result, on_thinking,
            )

    async def _chat_stream_impl(
        self,
        message: str,
        session_id: str,
        user_id: str,
        source: RunSource,
        on_content: Optional[Callable[[str], Any]],
        on_tool_call: Optional[Callable[[str, dict], Any]],
        on_tool_result: Optional[Callable[[str, str], Any]],
        on_thinking: Optional[Callable[[str], Any]],
    ) -> ChatResult:
        """Internal stream implementation (called under per-session lock)."""
        await self._ensure_initialized()
        agent = await self._get_agent(session_id)

        try:
            if self._workspace:
                await asyncio.to_thread(self._workspace.set_user, user_id)

            full_content = ""
            reasoning_content = ""
            tools_used: List[str] = []
            tool_calls = 0

            async for chunk in agent.run_stream(
                message,
                config=self._run_config_for_session(
                    session_id,
                    source,
                    stream_intermediate_steps=True,
                ),
            ):
                if chunk is None:
                    continue

                display_event = classify_run_response(chunk)

                if display_event.kind == RunDisplayEventKind.TOOL_STARTED:
                    tool_info = chunk.tools[-1] if chunk.tools else None
                    if tool_info:
                        tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                        tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                        display_args = format_tool_call_args(tool_name, tool_args)
                        tools_used.append(tool_name)
                        tool_calls += 1
                        if on_tool_call:
                            await on_tool_call(tool_name, display_args)
                    continue

                if display_event.kind == RunDisplayEventKind.TOOL_COMPLETED:
                    if chunk.tools and on_tool_result:
                        for ti in reversed(chunk.tools):
                            if "content" in ti:
                                t_name, result_str, _ = format_tool_result(ti)
                                await on_tool_result(t_name, result_str)
                                break
                    continue

                if display_event.kind == RunDisplayEventKind.METADATA_SKIP:
                    continue
                if display_event.kind == RunDisplayEventKind.TELEMETRY_ONLY:
                    continue

                if display_event.kind == RunDisplayEventKind.CONTENT_DELTA:
                    if chunk.reasoning_content:
                        reasoning_content += chunk.reasoning_content
                        if on_thinking:
                            await on_thinking(chunk.reasoning_content)

                    if chunk.content:
                        full_content += chunk.content
                        if on_content:
                            await on_content(chunk.content)

            return ChatResult(
                content=full_content.strip(),
                tool_calls=tool_calls,
                session_id=session_id,
                user_id=user_id,
                tools_used=tools_used,
                reasoning=reasoning_content,
                metrics=extract_metrics(agent),
            )

        except (asyncio.CancelledError, AgentCancelledError, KeyboardInterrupt):
            logger.info(f"AgentService stream cancelled (session={session_id})")
            raise

        except Exception as e:
            logger.error(f"AgentService.chat_stream error (session={session_id}): {e}")
            return ChatResult(
                content=f"Error: {e}",
                tool_calls=0,
                session_id=session_id,
                user_id=user_id,
            )

    # ============== Session management ==============

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List sessions from the persistent SessionLog (single source of truth).

        Returns rich metadata (name/preview/timestamps) so the UI can show
        sessions across restarts, not just the in-memory LRU cache.

        Scheduled (cron) job runs are excluded — they're not interactive chat
        sessions and shouldn't clutter the sidebar; their execution history
        is tracked separately via the cron TaskRun store.
        """
        out: List[Dict[str, Any]] = []
        for s in SessionLog.list_sessions():
            sid = s["session_id"]
            if sid.startswith(CRON_SESSION_PREFIX):
                continue
            preview = SessionLog.session_preview(s["path"])
            first_user = (preview or {}).get("first_user", "")
            name = s.get("name") or (first_user[:40] if first_user else "Chat")
            out.append({
                "session_id": sid,
                "name": name,
                "preview": first_user,
                "user_count": (preview or {}).get("user_count", 0),
                "last_timestamp": s.get("last_timestamp"),
                "size_bytes": s.get("size_bytes", 0),
                "archived": bool(s.get("archived")),
            })
        return out

    def has_active_runs(self) -> bool:
        """Return True if any session currently has an in-flight run.

        Used by profile switch to reject switching mid-run (the agent cache
        clear would evict an agent whose run is still streaming).
        """
        return any(lock.locked() for lock in self._session_locks.values())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session: cached Agent + persistent SessionLog JSONL + meta.

        Removes the on-disk JSONL and sidecar meta so the session does not
        reappear after restart (SessionLog is the single source of truth).
        Returns True if either the cache or the on-disk log existed.
        """
        removed = self._cache.delete(session_id)
        self._session_work_dirs.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        log_existed = False
        try:
            log = SessionLog(session_id=session_id)
            if log.path.exists():
                log.path.unlink()
                log_existed = True
            meta = log.base_dir / f"{session_id}.meta.json"
            if meta.exists():
                meta.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove SessionLog for {session_id}: {e}")
        logger.debug(f"Session deleted: {session_id}")
        return removed or log_existed

    def rename_session(self, session_id: str, name: str) -> None:
        """Rename a session by writing the sidecar .meta.json (SessionLog)."""
        SessionLog.rename_session(session_id, name)

    def archive_session(self, session_id: str, archived: bool = True) -> None:
        """Archive/unarchive a session by writing SessionLog sidecar metadata."""
        SessionLog.archive_session(session_id, archived=archived)

    def clear_session(self, session_id: str) -> bool:
        """Alias for delete_session (for compatibility)."""
        return self.delete_session(session_id)

    def cancel_session(self, session_id: str) -> bool:
        """Cancel the in-flight run for a specific session.

        Returns True if the session has an agent to cancel, False otherwise.
        """
        agent = self._cache.get(session_id)
        if agent is None:
            return False
        try:
            agent.cancel()
            logger.debug(f"Cancelled agent for session: {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel session {session_id}: {e}")
            return False

    # ============== Work directory ==============

    def set_session_work_dir(self, session_id: str, work_dir: str) -> None:
        """Set the working directory for a specific session.

        Per-session work_dirs override the global settings.base_dir.
        Does NOT clear other sessions' agents.
        """
        self._session_work_dirs[session_id] = work_dir

    def get_session_work_dir(self, session_id: str) -> str:
        """Get the working directory for a session (falls back to global base_dir)."""
        return self._session_work_dirs.get(session_id, str(settings.base_dir))

    def update_work_dir(self, new_dir: str) -> None:
        """Update the global work_dir and clear ALL cached agents.

        Called when the user changes the global working directory via the UI.
        All agents must be rebuilt to pick up the new directory.
        """
        self._cache.clear()
        self._session_work_dirs.clear()
        logger.info(f"Global work_dir updated to: {new_dir}, all agent instances cleared")

    # ============== Memory ==============

    async def save_memory(self, content: str, user_id: str = "default", long_term: bool = False) -> None:
        """Persist content to Workspace memory."""
        await self._ensure_initialized()
        if self._workspace and self._workspace.exists():
            await asyncio.to_thread(self._workspace.set_user, user_id)
            await self._workspace.write_memory(content)
            logger.debug(f"Memory saved for user {user_id}: {content[:50]}...")

    async def get_memory(self, user_id: str = "default", query: str = "", limit: int = 5) -> str:
        """Retrieve memory for a user via search_memory (keyword/bigram matching).

        Args:
            user_id: User identifier
            query: Search query (empty returns recent entries)
            limit: Maximum number of entries
        """
        await self._ensure_initialized()
        if self._workspace and self._workspace.exists():
            await asyncio.to_thread(self._workspace.set_user, user_id)
            results = self._workspace.search_memory(query=query, limit=limit)
            if results:
                return "\n\n".join(
                    f"**{r.get('title', 'Memory')}**: {r.get('content', '')}"
                    for r in results
                )
        return ""

    async def get_workspace_context(self, user_id: str = "default") -> str:
        """Retrieve workspace context prompt for a user."""
        await self._ensure_initialized()
        if self._workspace and self._workspace.exists():
            await asyncio.to_thread(self._workspace.set_user, user_id)
            return await self._workspace.get_context_prompt() or ""
        return ""

    async def list_users(self) -> List[str]:
        """List all known users from Workspace."""
        await self._ensure_initialized()
        if self._workspace:
            return await asyncio.to_thread(self._workspace.list_users)
        return []

    async def get_user_info(self, user_id: str) -> dict:
        """Get workspace user info."""
        await self._ensure_initialized()
        if self._workspace:
            return await asyncio.to_thread(self._workspace.get_user_info, user_id=user_id)
        return {"user_id": user_id}

    # ============== Hot reload ==============

    async def reload_profile(self, profile_name: Optional[str] = None) -> None:
        """Switch to a different config.yaml profile at runtime.

        Reloads main + auxiliary model config from the active profile (or
        ``profile_name`` if given), projects the profile's api_key/env into
        os.environ, then clears the agent cache so agents rebuild on next
        request with the new model.
        """
        if profile_name:
            set_active_profile(profile_name)
        profile = apply_global_config() or {}
        aux_profile = profile.get("auxiliary_model") or {}
        if not isinstance(aux_profile, dict):
            aux_profile = {}

        # apply_global_config uses setdefault semantics, so switching between
        # two profiles on the SAME provider would leave the old api_key in
        # place. Force-overwrite the provider env vars so the new profile's
        # key actually takes effect for SDK code paths that read the env.
        new_provider = profile.get("model_provider")
        new_api_key = profile.get("api_key")
        if new_provider and new_api_key:
            os.environ[provider_api_key_env(new_provider)] = new_api_key
        if aux_profile:
            aux_provider = aux_profile.get("model_provider")
            aux_key = aux_profile.get("api_key")
            if aux_provider and aux_key:
                os.environ[provider_api_key_env(aux_provider)] = aux_key

        async with self._init_lock:
            if profile.get("model_provider"):
                settings.model_provider = profile["model_provider"]
            if profile.get("model_name"):
                settings.model_name = profile["model_name"]
            settings.model_base_url = profile.get("base_url") or settings.model_base_url
            settings.model_api_key = profile.get("api_key") or settings.model_api_key
            settings.model_reasoning_effort = profile.get("reasoning_effort") or settings.model_reasoning_effort
            settings.max_tokens = int(profile.get("max_tokens") or 0)
            settings.temperature = float(profile.get("temperature") or 0)
            settings.top_p = float(profile.get("top_p") or 0)
            settings.context_window = int(profile.get("context_window") or 0)
            settings.auxiliary_model_provider = aux_profile.get("model_provider") or ""
            settings.auxiliary_model_name = aux_profile.get("model_name") or ""
            settings.auxiliary_base_url = aux_profile.get("base_url") or ""
            settings.auxiliary_api_key = aux_profile.get("api_key") or ""
            self._initialized = False
            self._cache.clear()
            logger.info(
                f"Profile reloaded: {profile_name or 'active'} -> "
                f"{self.model_provider}/{self.model_name}"
            )

    async def _invalidate_cache(self) -> None:
        """Clear the agent cache so agents rebuild on next request.

        Used when a runtime-only setting changes (e.g. thinking toggle) that
        does not require re-reading the profile.
        """
        async with self._init_lock:
            self._initialized = False
            self._cache.clear()
            logger.info("Agent cache invalidated (will rebuild on next request)")

    async def add_tool(self, tool: Any) -> None:
        """Dynamically add a tool; clears agent cache to force rebuild."""
        async with self._init_lock:
            self.extra_tools.append(tool)
            self._initialized = False
            self._cache.clear()

    def add_instruction(self, instruction: str) -> None:
        """Append an instruction to all existing agents."""
        self.extra_instructions.append(instruction)
        for session_id in self._cache.keys():
            agent = self._cache.get(session_id)
            if agent:
                agent.add_instruction(instruction)

    # ============== Properties ==============

    @property
    def workspace(self) -> Optional[Workspace]:
        """Shared Workspace instance (synchronous; call after initialization)."""
        return self._workspace

    @property
    def agent(self) -> Optional[DeepAgent]:
        """Deprecated: returns an arbitrary cached DeepAgent.

        Prefer cancel_session(session_id) for targeted cancellation.
        """
        sessions = self._cache.keys()
        if sessions:
            return self._cache.get(sessions[0])
        return None
