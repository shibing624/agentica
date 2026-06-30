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

from ..config import settings
from .model_factory import create_model, get_cron_tools, get_cron_instructions
from .response_formatter import extract_metrics, format_tool_call_args, format_tool_result

# Timeout in seconds for building a new Agent instance (guards against SDK hangs)
_AGENT_BUILD_TIMEOUT_S = 30


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
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        extra_tools: Optional[List[Any]] = None,
        extra_instructions: Optional[List[str]] = None,
        # Optional sibling overrides. Each can differ in any of
        # provider/model/base_url/api_key; unspecified fields fall back to the
        # main-model values so users can override just what differs. When
        # `{prefix}_model_name` is empty, the sibling is not built and DeepAgent
        # reuses the main model.
        auxiliary_model_provider: Optional[str] = None,
        auxiliary_model_name: Optional[str] = None,
        auxiliary_base_url: Optional[str] = None,
        auxiliary_api_key: Optional[str] = None,
        task_model_provider: Optional[str] = None,
        task_model_name: Optional[str] = None,
        task_base_url: Optional[str] = None,
        task_api_key: Optional[str] = None,
    ):
        self.workspace_path = Path(workspace_path or settings.workspace_path).expanduser()
        self.model_name = model_name or settings.model_name
        self.model_provider = model_provider or settings.model_provider
        self.extra_tools = extra_tools or []
        self.extra_instructions = extra_instructions or []

        # Sibling model configs — empty string treated as "not set".
        self.auxiliary_model_provider = auxiliary_model_provider or settings.auxiliary_model_provider
        self.auxiliary_model_name = auxiliary_model_name or settings.auxiliary_model_name
        self.auxiliary_base_url = auxiliary_base_url or settings.auxiliary_base_url
        self.auxiliary_api_key = auxiliary_api_key or settings.auxiliary_api_key
        self.task_model_provider = task_model_provider or settings.task_model_provider
        self.task_model_name = task_model_name or settings.task_model_name
        self.task_base_url = task_base_url or settings.task_base_url
        self.task_api_key = task_api_key or settings.task_api_key

        self._cache = LRUAgentCache(max_size=settings.agent_max_sessions)
        # Per-session work_dir overrides; falls back to settings.base_dir
        self._session_work_dirs: Dict[str, str] = {}
        # Per-session run locks: prevents concurrent runs (chat or stream) on the same session.
        # The underlying Agent instance is NOT thread-safe for concurrent reuse.
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._workspace: Optional[Workspace] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    def _build_sibling_model(self, prefix: str) -> Optional[Any]:
        """Build a sibling (auxiliary/task) model if a model name is configured.

        Fields fall through to main-model values so callers can override
        only what differs (e.g. a different model_name on the same provider,
        or a different base_url+api_key on the same provider/model).
        Returns None when no sibling name is set — DeepAgent will reuse
        the main model.
        """
        sibling_name = getattr(self, f"{prefix}_model_name")
        if not sibling_name:
            return None
        provider = getattr(self, f"{prefix}_model_provider") or self.model_provider
        base_url = getattr(self, f"{prefix}_base_url") or None
        api_key = getattr(self, f"{prefix}_api_key") or None
        return create_model(provider, sibling_name, base_url=base_url, api_key=api_key)

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

    def _build_agent(self) -> DeepAgent:
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
        """
        model = create_model(self.model_provider, self.model_name)
        auxiliary_model = self._build_sibling_model("auxiliary")
        task_model = self._build_sibling_model("task")
        work_dir = str(settings.base_dir)

        # Extra tools: user-provided + cron
        extra = list(self.extra_tools)
        cron_tools = get_cron_tools()
        extra.extend(cron_tools)

        instructions = list(self.extra_instructions) if self.extra_instructions else None
        if cron_tools:
            if instructions is None:
                instructions = []
            instructions.append(get_cron_instructions())

        agent = DeepAgent(
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            tools=extra if extra else None,
            workspace=self._workspace,
            work_dir=work_dir,
            num_history_turns=6,
            instructions=instructions,
            debug=settings.debug,
            # memory, skills, user input, experience capture, workspace memory
            # all on by DeepAgent default — no explicit overrides needed.
            include_user_input=True,
        )

        tool_count = len(agent.tools) if agent.tools else 0
        logger.info(f"DeepAgent built: {tool_count} tools (extra={len(extra)}, cron={len(cron_tools)})")
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
                asyncio.to_thread(self._build_agent),
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

    # ============== Public API ==============

    def get_context_window(self, session_id: str) -> int:
        """Return the context window size for the model used by a session.

        Falls back to 128000 if the session has no cached agent or
        the model doesn't expose a context_window attribute.
        """
        agent = self._cache.get(session_id)
        if agent and agent.model:
            return getattr(agent.model, "context_window", 128000)
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

                response = await agent.run(message, config=RunConfig(source=source))

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
                config=RunConfig(stream_intermediate_steps=True, source=source),
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

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return self._cache.keys()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its cached Agent instance.

        Returns True if the session existed, False otherwise.
        """
        removed = self._cache.delete(session_id)
        self._session_work_dirs.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        if removed:
            logger.debug(f"Session deleted: {session_id}")
        return removed

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

    async def reload_model(self, model_provider: str, model_name: str) -> None:
        """Switch model at runtime; clears all cached agents."""
        async with self._init_lock:
            self.model_provider = model_provider
            self.model_name = model_name
            self._initialized = False
            self._cache.clear()
            logger.info(f"Model reloaded: {model_provider}/{model_name}")

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
