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
from typing import Optional, Callable, List, Any, Dict, TYPE_CHECKING

from agentica.utils.log import logger
from agentica import DeepAgent
from agentica.agent.config import ToolConfig
from agentica.run_display import RunDisplayEventKind, classify_run_response
from agentica.run_response import AgentCancelledError
from agentica.run_config import RunConfig
from agentica.run_context import RunSource
from agentica.workspace import Workspace
from agentica.global_config import (
    apply_global_config,
    get_setting,
    set_active_profile,
    provider_api_key_env,
)
from agentica.memory.session_log import SessionLog
from agentica.skills import get_skill_registry, load_system_skills

from ..config import settings
from .media_understanding import media_understanding
from .model_factory import (
    create_model, get_cron_tools, get_cron_instructions,
    get_self_manage_tools, get_self_manage_instructions,
)
from .response_formatter import extract_metrics, format_tool_call_args, format_tool_result
from .session_usage import turn_usage_payload, usage_payload

if TYPE_CHECKING:
    from ..channels.base import InboundMedia

# Timeout in seconds for building a new Agent instance (guards against SDK hangs)
_AGENT_BUILD_TIMEOUT_S = 30

# Session-id prefix reserved for scheduled (cron) job runs. These sessions are
# never surfaced by list_sessions() (the chat sidebar) — cron execution
# history lives in the dedicated TaskRun store (agentica.cron.jobs), not the
# chat session log. See AgentService.run_cron().
CRON_SESSION_PREFIX = "scheduled_"


def goal_event_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact status dict for the web goal bar (CLI ``status_line`` / budget)."""
    tokens_used = int(payload.get("tokens_used") or 0)
    token_budget = payload.get("token_budget")
    turns_used = int(payload.get("turns_used") or 0)
    turn_budget = payload.get("turn_budget")
    wall_used = float(payload.get("wall_clock_used_sec") or 0)
    wall_budget = payload.get("wall_clock_budget_sec")
    parts: List[str] = []
    if token_budget is not None:
        parts.append(f"tokens {tokens_used:,}/{int(token_budget):,}")
    else:
        parts.append(f"tokens {tokens_used:,}")
    if turn_budget is not None:
        parts.append(f"turns {turns_used}/{int(turn_budget)}")
    else:
        parts.append(f"turns {turns_used}")
    if wall_budget is not None:
        parts.append(f"wall {wall_used:.0f}s/{float(wall_budget):.0f}s")
    return {
        "status": payload.get("status") or "active",
        "objective": payload.get("objective") or "",
        "progress": " · ".join(parts),
        "turns_used": turns_used,
        "tokens_used": tokens_used,
        "token_budget": token_budget,
        "turn_budget": turn_budget,
        "wall_clock_used_sec": wall_used,
        "wall_clock_budget_sec": wall_budget,
        "message": payload.get("message") or "",
    }

# Web sessions default to "ask" approval mode, which strips write tools
# (write_file/edit_file/apply_patch/execute) from the schema sent to the model (see
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
    "edit_file, apply_patch, and execute are disabled. In \"auto\" mode, writes are "
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
    # User-facing one-liners about how inbound media was handled (non-base
    # model used, media skipped, …); the channel layer prefixes these to the
    # reply so IM users see them.
    media_notes: List[str] = field(default_factory=list)
    # Session-level /usage snapshot (context breakdown + billed totals).
    usage: Optional[Dict[str, Any]] = None
    # This run's CostTracker split (input / cache read / hit / output / cost).
    turn_usage: Optional[Dict[str, Any]] = None


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

    def contains(self, session_id: str) -> bool:
        """Membership without the LRU touch ``get`` performs.

        Liveness probes (``AgentService.has_cached_session``) must not reorder
        the cache: a polling caller would otherwise keep whatever it asks about
        alive and evict the session the user is actually talking to.
        """
        return session_id in self._cache

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)


class AgentService:
    """Agent service wrapping the agentica SDK.

    Features:
    - Workspace config layer (AGENTS.md, MEMORY.md, etc.)
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
        # Peer-channel identity for this gateway's own agent sessions
        # (gateway/services/agent_peers.py), injected at startup by main.py's
        # lifespan once the channels exist. None means "no peer channel": the
        # SDK-embedded AgentService and every test build agents without one,
        # and they simply get no list_agents / send_message.
        self.agent_peers: Optional[Any] = None

    # ============== Model config (single source of truth: `settings`) ==============
    # These proxy directly to the gateway's global `settings` singleton instead
    # of keeping a duplicate copy, so routes only ever need to write to one
    # place (settings.xxx) and every reader (here, routes/settings.py, etc.)
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
    def model_wire_api(self) -> str:
        return settings.model_wire_api

    @property
    def model_reasoning(self) -> str:
        return settings.model_reasoning

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

    @property
    def auxiliary_wire_api(self) -> str:
        return settings.auxiliary_wire_api

    @property
    def auxiliary_reasoning(self) -> str:
        return settings.auxiliary_reasoning

    @property
    def auxiliary_reasoning_effort(self) -> str:
        return settings.auxiliary_reasoning_effort

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
            wire_api=getattr(self, f"{prefix}_wire_api"),
            reasoning=getattr(self, f"{prefix}_reasoning"),
            reasoning_effort=getattr(self, f"{prefix}_reasoning_effort"),
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

    def _build_agent(self, session_id: str, owner: Optional[str] = None) -> DeepAgent:
        """Build a new DeepAgent instance (sync, runs in thread).

        DeepAgent auto-includes: builtin tools, skills, agentic prompt,
        two-layer compression (eviction then LLM summarisation),
        workspace memory (auto_archive +
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
                wire_api=self.model_wire_api,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                context_window=self.context_window,
                reasoning=self.model_reasoning,
                reasoning_effort=self.model_reasoning_effort,
                thinking="enabled",
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
        cron_tools = get_cron_tools(owner=self._owner(owner))
        extra.extend(cron_tools)
        self_manage_tools = get_self_manage_tools()
        extra.extend(self_manage_tools)

        # Peer messaging: the same list_agents / send_message a CLI session has,
        # so a line typed in IM ("让三个会话都把改动提交了") reaches this
        # machine's live CLI sessions instead of an agent that cannot see them.
        # A cron run is excluded: it builds a throwaway agent that is never
        # cached, so publishing presence for it would advertise a mailbox that
        # stops being read the moment the job ends.
        peer_session = None
        if self.agent_peers is not None and not session_id.startswith(CRON_SESSION_PREFIX):
            from agentica.tools.peer_tool import PeerMessagingTool

            peer_session = self.agent_peers.session_for(session_id, cwd=work_dir)
            extra.insert(0, PeerMessagingTool(peer_session))

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

        permission_mode = self.get_session_approval_mode(session_id)
        enable_evict = get_setting("enable_evict", True)
        enable_auto_compact = get_setting("enable_auto_compact", True)
        load_system_skills()
        agent = DeepAgent(
            session_id=session_id,
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            description=(
                "You are DeepAgent, an AI coding agent served over the Agentica "
                "gateway (web UI and chat channels). You help users with software "
                "engineering and general tasks through a conversational interface."
            ),
            tools=extra if extra else None,
            workspace=self._workspace,
            work_dir=work_dir,
            user_id=self._owner(owner),
            num_history_turns=settings.num_history_turns,
            instructions=instructions,
            debug=settings.debug,
            # memory, skills, experience capture, workspace memory all on by
            # DeepAgent default — no explicit overrides needed. ask_user_question
            # is intentionally OFF: the gateway is headless (web/chat channels)
            # with no stdin, so the tool would block on a bare input() call.
            include_ask_user_question=False,
            permission_mode=permission_mode,
            tool_config=ToolConfig(
                auto_load_mcp=True,
                permission_mode=permission_mode,
                enable_evict=bool(enable_evict),
                enable_auto_compact=bool(enable_auto_compact),
            ),
        )

        # Lets the Runner drain replies from other sessions between tool
        # batches (agent.peer_session), exactly as it does in the CLI.
        agent.peer_session = peer_session

        tool_count = len(agent.tools) if agent.tools else 0
        logger.info(
            f"DeepAgent built: {tool_count} tools "
            f"(extra={len(extra)}, cron={len(cron_tools)}, self_manage={len(self_manage_tools)})"
        )
        return agent

    async def _get_agent(self, session_id: str, owner: Optional[str] = None) -> DeepAgent:
        """Return the cached Agent for a session, creating one if absent.

        Raises RuntimeError if the agent cannot be built (e.g. SDK error).
        Times out after _AGENT_BUILD_TIMEOUT_S seconds.

        A cached agent is only reused for the owner it was built for. The cache
        is keyed by session id alone, and the id comes from the browser — so
        without this check one account presenting another's id would run against
        the agent already built for that partition, which is the one thing the
        partition exists to prevent.
        """
        agent = self._cache.get(session_id)
        if agent is not None and agent.user_id == self._owner(owner):
            return agent

        try:
            agent = await asyncio.wait_for(
                asyncio.to_thread(self._build_agent, session_id, owner),
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

    def has_cached_session(self, session_id: str) -> bool:
        """Whether a built Agent is still cached for this session."""
        return self._cache.contains(session_id)

    def is_session_active(self, session_id: str) -> bool:
        """Whether a run is in flight on this session (its lock is held)."""
        lock = self._session_locks.get(session_id)
        return lock is not None and lock.locked()

    def _note_peer_turn(self, session_id: str, message: str) -> None:
        """Publish "what this session is working on" for other sessions to read.

        A CLI peer publishes the line the user typed; this is the same thing for
        a chat session, and it is what makes a listing on the other side
        ("working on: 把 gateway 的 peer 工具接上") worth reading. Called after
        the agent exists, so the session is already registered.
        """
        if self.agent_peers is not None:
            self.agent_peers.note_turn(session_id, message)

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

    async def session_usage(
        self,
        session_id: str,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """CLI ``/usage`` for one web session: occupancy + session billing.

        Builds the Agent if it is not cached yet, because the window
        breakdown is measured from the live prompt (system / skills / tools
        / history), not from the last SSE ``done`` event.
        """
        await self._ensure_initialized()
        agent = await self._get_agent(session_id, owner)
        return await usage_payload(agent, model_provider=self.model_provider)

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

    @staticmethod
    def _expand_skill_invocation(message: str) -> str:
        """Turn a ``/skill-trigger [arguments]`` message into the skill prompt.

        The web input box inserts skill references as plain ``/trigger`` text,
        so without this the model only ever saw the raw slash line. Uses the
        same renderer as the CLI so both surfaces frame arguments identically.
        """
        expanded = get_skill_registry().expand_invocation(message)
        return expanded if expanded is not None else message

    @staticmethod
    async def _prepare_run_media(agent: Any, message: str, media: Optional[List["InboundMedia"]]):
        """Route inbound images/audio/video the same way IM channels do.

        Capable base models get the payload on ``agent.run(images=)`` /
        ``audio=``. Everything else is described by ``settings.media_model``
        (Gemini) and appended to the user text.
        """
        if not media:
            return message, None, None, []
        plan = await media_understanding.prepare(
            media,
            base_model_id=agent.model.id or "",
            base_supports_images=bool(agent.model.supports_images),
        )
        if plan.text_parts:
            message = message + "\n\n" + "\n\n".join(plan.text_parts)
        return message, (plan.images or None), plan.audio, plan.notes

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        source: RunSource = RunSource.gateway,
        media: Optional[List["InboundMedia"]] = None,
        owner: Optional[str] = None,
    ) -> ChatResult:
        """Send a message and return the full response (non-streaming).

        Acquires a per-session lock to prevent concurrent runs on the same
        Agent instance (which is not thread-safe).

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier (for workspace memory isolation)
            owner: Which ``users/<id>/`` partition this session is stored in.
                Web routes pass the signed-in account; unset means the machine's
                own partition (cron, IM channels, the CLI).
            media: Downloaded inbound media (image/voice/video payloads) from
                the channel. Images attach to the run when the base model can
                see them; audio/video attach when the base is Gemini. Anything
                else is described/transcribed by ``settings.media_model``.

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
            agent = await self._get_agent(session_id, owner)
            self._note_peer_turn(session_id, message)
            message, run_images, run_audio, media_notes = await self._prepare_run_media(
                agent, message, media,
            )

            try:
                if self._workspace:
                    await asyncio.to_thread(self._workspace.set_user, user_id)

                response = await agent.run(
                    self._expand_skill_invocation(message),
                    config=self._run_config_for_session(session_id, source),
                    images=run_images,
                    audio=run_audio,
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
                    media_notes=media_notes,
                    usage=await usage_payload(agent, model_provider=self.model_provider),
                    turn_usage=turn_usage_payload(agent),
                )

            except Exception as e:
                logger.error(f"AgentService.chat error (session={session_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                    media_notes=media_notes,
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
                asyncio.to_thread(self._build_agent, session_id, user_id),
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
        owner: Optional[str] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Drive a bounded standing-goal loop (Agent.run_goal) for the web UI's
        "/goal <objective>" command.

        Budgets are capped conservatively since this runs behind a single
        HTTP request in a local gateway. ``on_event`` receives a compact
        status dict after each ``goal.*`` event so the web bar can tick.
        """
        await self._ensure_initialized()

        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        def _cb(event, payload: Dict[str, Any]) -> None:
            if on_event is None:
                return
            data = goal_event_payload(payload)
            data["event"] = getattr(event, "value", str(event))
            on_event(data)

        async with lock:
            agent = await self._get_agent(session_id, owner)
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
                event_callback=_cb if on_event else None,
            )
            return {
                "status": result.status,
                "reason": result.reason,
                "content": result.response_content,
                "turns_used": result.turns_used,
            }

    async def compact_session(
        self,
        session_id: str,
        owner: Optional[str] = None,
        instructions: str = "",
    ) -> Dict[str, Any]:
        """Summarise this session's history — the web counterpart of CLI ``/compact``.

        Same two-step as the CLI: native provider compact first, then
        ``CompressionManager.auto_compact(force=True)``. A failed compact
        leaves the transcript unchanged.
        """
        await self._ensure_initialized()
        lock = self._get_session_lock(session_id)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            agent = await self._get_agent(session_id, owner)
            wm = agent.working_memory
            messages = wm.messages
            msg_count = len(messages)
            if msg_count == 0:
                return {"ok": False, "error": "No messages to compact."}

            custom = instructions.strip() or None
            model = agent.model
            hooks = agent._run_hooks
            if hooks is not None:
                await hooks.on_pre_compact(agent=agent, messages=messages)

            native_compacted = False
            if model.supports_native_compaction:
                try:
                    result = await model.compact_context(messages, instructions=custom)
                    if result is None:
                        raise RuntimeError("model advertised native compaction but returned no checkpoint")
                except Exception as error:
                    logger.warning(
                        "Native compaction failed (%s); falling back to local compaction", error
                    )
                else:
                    messages[-1].provider_checkpoint = result.checkpoint
                    wm.collapse_runs(messages)
                    if agent._session_log is not None:
                        agent._session_log.append_provider_checkpoint(result.checkpoint)
                    native_compacted = True

            if not native_compacted:
                cm = agent.tool_config.compression_manager if agent.tool_config else None
                if cm is None:
                    return {"ok": False, "error": "No compression manager on this agent; nothing to compact with."}
                compacted = await cm.auto_compact(
                    messages,
                    model=model,
                    force=True,
                    working_memory=wm,
                    custom_instructions=custom,
                )
                if not compacted:
                    return {"ok": False, "error": "Compaction failed; conversation left unchanged."}
                wm.collapse_runs(messages)

            if hooks is not None:
                await hooks.on_post_compact(agent=agent, messages=messages)

            return {
                "ok": True,
                "native": native_compacted,
                "messages_before": msg_count,
                "messages_after": len(messages),
                "usage": await usage_payload(agent, model_provider=self.model_provider),
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
        owner: Optional[str] = None,
        media: Optional[List["InboundMedia"]] = None,
    ) -> ChatResult:
        """Send a message and stream the response via callbacks.

        Acquires the per-session run lock to prevent concurrent runs
        (both chat and chat_stream) on the same Agent instance.

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier
            owner: Which ``users/<id>/`` partition this session is stored in.
                Web routes pass the signed-in account; unset means the machine's
                own partition (cron, IM channels, the CLI).
            on_content: Called with each content delta
            on_tool_call: Called when a tool call starts (name, args)
            on_tool_result: Called when a tool call completes (name, result)
            on_thinking: Called with each reasoning delta
            media: Pasted/attached images from the web UI (same routing as IM).

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
                owner, media,
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
        owner: Optional[str],
        media: Optional[List["InboundMedia"]],
    ) -> ChatResult:
        """Internal stream implementation (called under per-session lock)."""
        await self._ensure_initialized()
        agent = await self._get_agent(session_id, owner)
        self._note_peer_turn(session_id, message)
        message, run_images, run_audio, media_notes = await self._prepare_run_media(
            agent, message, media,
        )

        try:
            if self._workspace:
                await asyncio.to_thread(self._workspace.set_user, user_id)

            full_content = ""
            reasoning_content = ""
            tools_used: List[str] = []
            tool_calls = 0

            async for chunk in agent.run_stream(
                self._expand_skill_invocation(message),
                config=self._run_config_for_session(
                    session_id,
                    source,
                    stream_intermediate_steps=True,
                ),
                images=run_images,
                audio=run_audio,
            ):
                if chunk is None:
                    continue

                display_event = classify_run_response(chunk)

                if display_event.kind == RunDisplayEventKind.TOOL_STARTED:
                    tool_info = chunk.tool_call
                    if tool_info:
                        tool_name = tool_info.tool_name or "unknown"
                        display_args = format_tool_call_args(tool_name, tool_info.tool_args)
                        tools_used.append(tool_name)
                        tool_calls += 1
                        if on_tool_call:
                            await on_tool_call(tool_name, display_args)
                    continue

                if display_event.kind == RunDisplayEventKind.TOOL_COMPLETED:
                    if chunk.tool_call and on_tool_result:
                        t_name, result_str, _ = format_tool_result(chunk.tool_call)
                        await on_tool_result(t_name, result_str)
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
                media_notes=media_notes,
                usage=await usage_payload(agent, model_provider=self.model_provider),
                turn_usage=turn_usage_payload(agent),
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

    def list_sessions(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions from the persistent SessionLog (single source of truth).

        Returns rich metadata (name/preview/timestamps/work_dir) so the UI can
        show sessions across restarts, not just the in-memory LRU cache.

        The web sidebar is grouped by project, so this lists **every** project
        under the owner's ``users/<id>/`` tree — not only ``settings.base_dir``.
        Listing just the default work dir made a finished chat vanish after
        opening its trace: navigating remounted the shell, refetched this
        list, and overwrote the sidebar with a set that did not include it.

        Scheduled (cron) job runs are excluded — they're not interactive chat
        sessions and shouldn't clutter the sidebar; their execution history
        is tracked separately via the cron TaskRun store.
        """
        uid = self._owner(owner)
        projects = SessionLog.list_projects(user_id=uid)
        if not projects:
            projects = [{
                "base_dir": self._session_base_dir(str(settings.base_dir), owner),
                "work_dir": str(settings.base_dir),
            }]
        out: List[Dict[str, Any]] = []
        for project in projects:
            base_dir = project["base_dir"]
            work_dir = project.get("work_dir") or ""
            for s in SessionLog.list_sessions(base_dir=base_dir):
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
                    "work_dir": s.get("work_dir") or work_dir,
                })
        out.sort(key=lambda row: str(row.get("last_timestamp") or ""), reverse=True)
        return out

    def session_log_for(self, session_id: str, owner: Optional[str] = None) -> SessionLog:
        """Open the on-disk SessionLog for a session (may not exist yet)."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        return SessionLog(session_id=session_id, base_dir=base_dir)

    def _locate_session(self, session_id: str, owner: Optional[str] = None) -> tuple[str, str]:
        """Find the project directory that actually holds this session's jsonl.

        ``_session_work_dirs`` is process memory. After a gateway restart it is
        empty, and guessing ``settings.base_dir`` is how View trace 404'd a log
        that had been written under another project. Search the owner's tree.
        """
        jsonl = f"{session_id}.jsonl"
        remembered = self._session_work_dirs.get(session_id)
        if remembered:
            base = self._session_base_dir(remembered, owner)
            if (Path(base) / jsonl).is_file():
                return base, remembered
        fallback_work = str(settings.base_dir)
        fallback = self._session_base_dir(fallback_work, owner)
        if (Path(fallback) / jsonl).is_file():
            return fallback, fallback_work
        hits = [
            h for h in SessionLog.find_sessions(session_id, user_id=self._owner(owner))
            if h.get("session_id") == session_id
        ]
        if hits:
            hit = hits[0]
            work_dir = hit.get("work_dir") or fallback_work
            if work_dir:
                self._session_work_dirs[session_id] = work_dir
            return str(hit["base_dir"]), work_dir
        return fallback, remembered or fallback_work

    def has_active_runs(self) -> bool:
        """Return True if any session currently has an in-flight run.

        Used by profile switch to reject switching mid-run (the agent cache
        clear would evict an agent whose run is still streaming).
        """
        return any(lock.locked() for lock in self._session_locks.values())

    def delete_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Delete a session: cached Agent + persistent SessionLog JSONL + meta.

        Removes the on-disk JSONL and sidecar meta so the session does not
        reappear after restart (SessionLog is the single source of truth).
        Returns True if either the cache or the on-disk log existed.
        """
        base_dir, _work_dir = self._locate_session(session_id, owner)
        removed = self._cache.delete(session_id)
        self._session_work_dirs.pop(session_id, None)
        if self.agent_peers is not None:
            # A deleted session must stop being addressable at once, not on the
            # peer loop's next liveness sweep: until it is unpublished its name
            # is still in every other session's list_agents.
            self.agent_peers.forget(session_id)
        self._session_locks.pop(session_id, None)
        log_existed = False
        try:
            log = SessionLog(session_id=session_id, base_dir=base_dir)
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

    def rename_session(self, session_id: str, name: str, owner: Optional[str] = None) -> None:
        """Rename a session by writing the sidecar .meta.json (SessionLog)."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.rename_session(session_id, name, base_dir=base_dir)

    def archive_session(
        self, session_id: str, archived: bool = True, owner: Optional[str] = None,
    ) -> None:
        """Archive/unarchive a session by writing SessionLog sidecar metadata."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.archive_session(session_id, archived=archived, base_dir=base_dir)

    def clear_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Alias for delete_session (for compatibility)."""
        return self.delete_session(session_id, owner)

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

    # ============== Session storage scoping ==============

    def _session_base_dir(self, work_dir: str, owner: Optional[str] = None) -> str:
        """Resolve the SessionLog storage dir for a given project work_dir.

        Sessions are scoped by project (work_dir) + owner, mirroring exactly how
        the Agent writes them (see ``SessionLog`` construction in the agent).
        This is what makes the Web sidebar and the CLI ``/resume`` list a
        consistent set of sessions for the same project + user.
        """
        from agentica.project_store import project_base_dir
        return project_base_dir(work_dir, user_id=self._owner(owner))

    @staticmethod
    def _owner(owner: Optional[str]) -> str:
        """Which ``users/<id>/`` partition a call reads and writes.

        Unset means the machine's own partition (``settings.default_user_id``),
        which is what an IM channel and the CLI share. Cron jobs carry their
        own ``user_id`` and pass it here so a scheduled run writes into the
        account that created the job. Only the web surface has accounts, so
        only its routes pass an owner — and they
        take it from the session cookie, never from the request body: a body
        field naming somebody else's partition is not a parameter, it is a way
        in.
        """
        return owner or settings.default_user_id

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
            settings.model_wire_api = profile.get("wire_api") or ""
            settings.model_reasoning = profile.get("reasoning") or ""
            settings.model_reasoning_effort = profile.get("reasoning_effort") or ""
            settings.max_tokens = int(profile.get("max_tokens") or 0)
            settings.temperature = float(profile.get("temperature") or 0)
            settings.top_p = float(profile.get("top_p") or 0)
            settings.context_window = int(profile.get("context_window") or 0)
            settings.auxiliary_model_provider = aux_profile.get("model_provider") or ""
            settings.auxiliary_model_name = aux_profile.get("model_name") or ""
            settings.auxiliary_base_url = aux_profile.get("base_url") or ""
            settings.auxiliary_api_key = aux_profile.get("api_key") or ""
            settings.auxiliary_wire_api = aux_profile.get("wire_api") or ""
            settings.auxiliary_reasoning = aux_profile.get("reasoning") or ""
            settings.auxiliary_reasoning_effort = aux_profile.get("reasoning_effort") or ""
            self._initialized = False
            self._cache.clear()
            logger.info(
                f"Profile reloaded: {profile_name or 'active'} -> "
                f"{self.model_provider}/{self.model_name}"
            )

    async def _invalidate_cache(self) -> None:
        """Clear the agent cache so agents rebuild on next request.

        Used when a runtime-only setting changes (model switch, work dir)
        that does not require re-reading the profile.
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
