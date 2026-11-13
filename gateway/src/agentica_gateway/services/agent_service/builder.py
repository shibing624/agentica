# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent build, cache lookup, per-session locks, approvals, and ask_user_question bridge.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional

from agentica import DeepAgent
from agentica.agent.approvals import (
    SessionGrants,
    apply_path_grant_on_agent,
    make_approve,
    sync_grants_from_project,
)
from agentica.agent.config import ExperienceConfig, ToolConfig, WorkspaceMemoryConfig
from agentica.compression.manager import parse_compact_token_limit
from agentica.config.profiles import get_setting
from agentica.run.config import RunConfig
from agentica.run.context import RunSource
from agentica.skills import load_system_skills
from agentica.tools.ask_user_question_tool import AskUserQuestionTimeout, AskUserQuestionTool
from agentica.utils.log import logger
from agentica.workspace import Workspace

from ...config import settings
from .types import _AGENT_BUILD_TIMEOUT_S, _APPROVAL_MODE_INSTRUCTION


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


class AgentBuilderMixin:
    """Workspace init, DeepAgent construction, session locks, approvals, and ask callback."""

    _agent_builds: Dict[str, "asyncio.Future[DeepAgent]"]

    async def _ensure_initialized(self) -> None:
        """Ensure the workspace is initialized (idempotent, Lock-protected)."""
        self._loop = asyncio.get_running_loop()
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
        """Build a new DeepAgent instance (sync, runs in thread)."""
        pkg = _pkg()
        cfg = self.session_model_config(session_id, owner)
        model = pkg.create_model(
            cfg["model_provider"],
            cfg["model_name"],
            base_url=cfg["base_url"] or None,
            api_key=cfg["api_key"] or None,
            wire_api=cfg["wire_api"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            context_window=cfg["context_window"],
            reasoning=cfg["reasoning"],
            reasoning_effort=cfg["reasoning_effort"],
            thinking="enabled",
            enable_cache_control=cfg.get("enable_cache_control"),
            cache_control_messages=cfg.get("cache_control_messages"),
            cache_control_session_header=cfg.get("cache_control_session_header") or None,
            cache_keepalive=cfg.get("cache_keepalive"),
            extra_body=cfg.get("extra_body"),
            extra_headers=cfg.get("extra_headers"),
            default_headers=cfg.get("default_headers"),
        )
        auxiliary_model = self._sibling_model(cfg, "auxiliary")
        # The auxiliary model also serves as the task subagent model (CLI
        # unified them — no separate task_model config anymore).
        task_model = auxiliary_model
        fallback_models = self._fallback_models(cfg)
        max_api_retry = int(cfg.get("max_api_retry") or 2)
        work_dir = self.get_session_work_dir(session_id, owner)

        extra = list(self.extra_tools)
        cron_tools = pkg.get_cron_tools(owner=self._owner(owner))
        extra.extend(cron_tools)
        self_manage_tools = pkg.get_self_manage_tools()
        extra.extend(self_manage_tools)

        extra.append(AskUserQuestionTool(input_callback=self._make_ask_callback(session_id, owner)))

        instructions = list(self.extra_instructions) if self.extra_instructions else None
        if cron_tools:
            if instructions is None:
                instructions = []
            instructions.append(pkg.get_cron_instructions())
        if self_manage_tools:
            if instructions is None:
                instructions = []
            instructions.append(pkg.get_self_manage_instructions())
        if instructions is None:
            instructions = []
        instructions.append(_APPROVAL_MODE_INSTRUCTION)

        permission_mode = self.get_session_approval_mode(session_id, owner)
        enable_evict = get_setting("enable_evict", True)
        enable_auto_compact = get_setting("enable_auto_compact", True)
        profile = pkg.get_profile(self.session_profile_name(session_id, owner) or pkg.get_active_profile_name()) or {}
        compact_token_limit = parse_compact_token_limit(profile.get("compact_token_limit"))
        if compact_token_limit is None:
            compact_token_limit = parse_compact_token_limit(
                get_setting("compact_token_limit", None)
            )
        auto_extract = self._auto_extract_for(owner)
        load_system_skills()
        agent = pkg.DeepAgent(
            session_id=session_id,
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            session_profile=cfg.get("profile") or None,
            fallback_models=fallback_models,
            max_api_retry=max_api_retry,
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
            # DeepAgent default — no explicit overrides needed. The builtin
            # ask_user_question stays off: it would read stdin. The gateway
            # mounts its own (above), answered through the session's live turn.
            include_ask_user_question=False,
            background_process_registry=self._background_registry(session_id, owner),
            permission_mode=permission_mode,
            tool_config=ToolConfig(
                auto_load_mcp=True,
                permission_mode=permission_mode,
                enable_evict=bool(enable_evict),
                enable_auto_compact=bool(enable_auto_compact),
                compact_token_limit=compact_token_limit,
            ),
            long_term_memory_config=WorkspaceMemoryConfig(
                auto_archive=True,
                auto_extract_memory=auto_extract,
                extract_every_n_turns=10,
                extract_min_seconds_between=60,
                load_workspace_context=True,
                load_workspace_memory=True,
                recall_memories=True,
                recall_sessions=True,
                track_intents=True,
                max_memory_entries=5,
            ),
            experience_config=ExperienceConfig(
                capture_tool_errors=True,
                capture_user_corrections=True,
                capture_success_patterns=False,
                judge_every_n_turns=10,
                judge_min_seconds_between=60,
            ),
        )

        agent.approve = self._make_session_approve(session_id, owner)
        self._restore_path_grants(agent, session_id, owner)
        if (
            agent._session_log is not None
            and agent._session_log.exists()
            and len(agent.working_memory.runs) == 0
            and len(agent.working_memory.messages) == 0
        ):
            if not self.is_session_active(session_id, owner):
                agent._session_log.seal_incomplete_turn()
            resumed_messages = agent._session_log.load(
                model=model.id if model is not None else None
            )
            if resumed_messages:
                agent.working_memory.hydrate_runs_from_history(resumed_messages)

        tool_count = len(agent.tools) if agent.tools else 0
        logger.info(
            f"DeepAgent built: {tool_count} tools "
            f"(extra={len(extra)}, cron={len(cron_tools)}, self_manage={len(self_manage_tools)})"
        )
        return agent

    async def _get_agent(self, session_id: str, owner: Optional[str] = None) -> DeepAgent:
        """Return the cached Agent for a session, creating one if absent.

        Concurrent callers share one build. Otherwise a usage poll racing a
        chat start builds a second Agent that replaces the running one in the
        cache, and steer / cancel reach the idle copy.
        """
        key = self._sk(session_id, owner)
        agent = self._cache.get(key)
        if agent is not None and agent.user_id == self._owner(owner):
            if key not in self._stale_agents or self.is_session_active(session_id, owner):
                return agent
        build = self._agent_builds.get(key)
        if build is None:
            build = asyncio.ensure_future(self._build_and_cache_agent(key, session_id, owner))
            self._agent_builds[key] = build

            def _forget(done: asyncio.Future) -> None:
                if self._agent_builds.get(key) is done:
                    del self._agent_builds[key]

            build.add_done_callback(_forget)
        return await asyncio.shield(build)

    async def _build_and_cache_agent(self, key: str, session_id: str, owner: Optional[str]) -> DeepAgent:
        self._stale_agents.discard(key)

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

        self._cache.put(key, agent)
        logger.info(f"Agent created for session: {session_id} (cache size: {len(self._cache)})")
        return agent

    def _get_session_lock(self, session_id: str, owner: Optional[str] = None) -> asyncio.Lock:
        """Return (or create) the per-session run lock."""
        key = self._sk(session_id, owner)
        if key not in self._session_locks:
            self._session_locks[key] = asyncio.Lock()
        return self._session_locks[key]

    def has_cached_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Whether a built Agent is still cached for this session."""
        return any(self._cache.contains(k) for k in self._map_keys(session_id, owner))

    def is_session_active(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Whether a run is in flight on this session (its lock is held)."""
        for key in self._map_keys(session_id, owner):
            lock = self._session_locks.get(key)
            if lock is not None and lock.locked():
                return True
        return False

    def _make_ask_callback(self, session_id: str, owner: Optional[str]) -> Callable[[str, Optional[List[str]]], str]:
        """``ask_user_question`` answered through the session's live turn."""
        owner_id = self._owner(owner)

        def ask(prompt: str, options: Optional[List[str]] = None) -> str:
            pkg = _pkg()
            wait_s = pkg.QUESTION_WAIT_S
            loop = self._loop
            turn = pkg.live_turn.active(session_id, owner_id)
            if loop is None or turn is None:
                raise AskUserQuestionTimeout(0)
            opts = [str(o) for o in (options or [])]

            async def open_and_wait() -> Optional[str]:
                if turn.done:
                    return None
                question, future = turn.open_question(prompt, opts)
                if turn.question_notifier is not None:
                    try:
                        await turn.question_notifier(question)
                    except Exception:
                        # The question is already on the turn. A failed IM
                        # send must not abandon the Web/Desktop card or fail
                        # the tool before anyone can answer.
                        logger.exception("ask_user_question notifier failed")
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout=wait_s)
                except asyncio.TimeoutError:
                    turn.close_question(question.question_id)
                    return None

            answer = asyncio.run_coroutine_threadsafe(open_and_wait(), loop).result()
            if answer is None:
                raise AskUserQuestionTimeout(wait_s)
            return answer

        return ask

    _DEFAULT_APPROVAL_MODE = "auto"

    def set_session_approval_mode(
        self, session_id: str, mode: str, owner: Optional[str] = None,
    ) -> None:
        """Persist the selected approval mode for a session."""
        from agentica.agent.permissions import PERMISSION_MODES

        normalized = (mode or self._DEFAULT_APPROVAL_MODE).strip().lower()
        if normalized not in PERMISSION_MODES:
            normalized = self._DEFAULT_APPROVAL_MODE
        self._session_approval_modes[self._sk(session_id, owner)] = normalized

        cached_agent = self._cache.get(self._sk(session_id, owner))
        if cached_agent is not None:
            cached_agent.set_permission_mode(normalized)

    def get_session_approval_mode(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_approval_modes.get(
            self._sk(session_id, owner), self._DEFAULT_APPROVAL_MODE,
        )

    def _grants_for(self, session_id: str, owner: Optional[str] = None) -> SessionGrants:
        key = self._sk(session_id, owner)
        grants = self._session_grants.get(key)
        if grants is None:
            grants = SessionGrants()
            self._session_grants[key] = grants
            sync_grants_from_project(
                grants,
                work_dir=self.get_session_work_dir(session_id, owner),
                user_id=self._owner(owner),
            )
        return grants

    def _restore_path_grants(
        self, agent: DeepAgent, session_id: str, owner: Optional[str] = None,
    ) -> None:
        """Re-apply in-session path grants after an Agent rebuild."""
        grants = self._grants_for(session_id, owner)
        for path in grants.path_exact:
            apply_path_grant_on_agent(agent, path, prefix=False)
        for path in grants.path_prefixes:
            apply_path_grant_on_agent(agent, path, prefix=True)

    def _make_session_approve(self, session_id: str, owner: Optional[str] = None):
        """``Agent.approve``: look up the current LiveTurn on every wait."""
        owner_id = self._owner(owner)

        def get_registry():
            turn = _pkg().live_turn.active(session_id, owner_id)
            return None if turn is None else turn.approvals

        def publish(pending):
            lt = _pkg().live_turn
            turn = lt.active(session_id, owner_id)
            if turn is None:
                return
            turn.publish(lt.approval_event(pending))

        def apply_path_grant(path: str, prefix: bool) -> None:
            agent = self._cache.get(self._sk(session_id, owner))
            if agent is not None:
                apply_path_grant_on_agent(agent, path, prefix=prefix)

        return make_approve(
            get_mode=lambda: self.get_session_approval_mode(session_id, owner),
            get_grants=lambda: self._grants_for(session_id, owner),
            get_registry=get_registry,
            get_work_dir=lambda: self.get_session_work_dir(session_id, owner),
            publish=publish,
            apply_path_grant=apply_path_grant,
            get_user_id=lambda: owner_id,
        )

    def _run_config_for_session(
        self,
        session_id: str,
        source: RunSource,
        *,
        channel: Optional[str] = None,
        stream_intermediate_steps: bool = False,
        automated: bool = False,
        input_meta: Optional[Dict[str, Any]] = None,
    ) -> RunConfig:
        return RunConfig(
            stream_intermediate_steps=stream_intermediate_steps, source=source, channel=channel,
            automated_input=automated, input_meta=input_meta,
        )
