# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent build, cache lookup, per-session locks, approvals, and ask_user_question bridge.
"""
from __future__ import annotations

import asyncio
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agentica import DeepAgent
from agentica.agent.approvals import (
    SessionGrants,
    apply_path_grant_on_agent,
    classify_pending,
    make_approve,
    sync_grants_from_project,
)
from agentica.agent.config import ExperienceConfig, ToolConfig, WorkspaceMemoryConfig
from agentica.agent.permissions import (
    validate_permission_mode,
    validate_sandbox_mode,
)
from agentica.config.prefs import normalize_cli_prefs
from agentica.compression.manager import parse_compact_token_limit
from agentica.config.profiles import get_setting, human_wait_timeout
from agentica.run.config import RunConfig
from agentica.run.context import RunSource
from agentica.skills import load_system_skills
from agentica.tools.ask_user_question_tool import AskUserQuestionTimeout
from agentica.tools.browser_tool import AgentBrowserDriver, BrowserTool
from agentica.utils.log import logger
from agentica.workspace import Workspace

from ...config import settings
from .types import _AGENT_BUILD_TIMEOUT_S


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


class AgentBuilderMixin:
    """Workspace init, DeepAgent construction, session locks, approvals, and ask callback."""

    _agent_builds: Dict[str, "asyncio.Future[DeepAgent]"]

    @staticmethod
    async def _close_browser_tools(tools: List[BrowserTool]) -> None:
        """End these browser daemons; shutdown must still try the rest on failure."""
        results = await asyncio.gather(*(tool.close() for tool in tools), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Browser daemon cleanup failed: %s", result)

    async def _drain_released_agents(self) -> None:
        """Release what sessions that left the cache (and were not rebuilt) hold.

        Browser daemons are closed. The per-session lookups that are rebuilt
        from disk on the next request (work dir, log dir, approval tier) are
        dropped, and so is an idle background registry: its report thread exits
        once the registry is no longer the session's, and a later build adopts
        any finished report from the durable records. A session with a run, a
        running job or a delivery in flight keeps everything.
        """
        keys = self._released_agent_keys[:]
        self._released_agent_keys.clear()
        orphaned = []
        for key in dict.fromkeys(keys):
            if self._cache.contains(key):
                continue
            if key in self._browser_tools:
                orphaned.append(self._browser_tools.pop(key))
            lock = self._session_locks.get(key)
            background = self._session_background.get(key)
            if (
                (lock is not None and lock.locked())
                or key in self._bg_delivery_inflight
                or (background is not None and background.running_count() > 0)
            ):
                continue
            self._session_work_dirs.pop(key, None)
            self._session_log_dirs.pop(key, None)
            self._session_approval_modes.pop(key, None)
            self._session_locks.pop(key, None)
            self._stale_agents.discard(key)
            self._session_background.pop(key, None)
        await self._close_browser_tools(orphaned)

    async def close_browser_sessions(self) -> None:
        """Close all per-session browser daemons before the gateway exits."""
        await self._drain_released_agents()
        tools = list(self._browser_tools.values())
        self._browser_tools.clear()
        await self._close_browser_tools(tools)

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
            extra_body=cfg.get("extra_body"),
            extra_headers=cfg.get("extra_headers"),
            default_headers=cfg.get("default_headers"),
        )
        auxiliary_model = self._sibling_model(cfg, "auxiliary")
        # The auxiliary model also serves as the task subagent model (CLI
        # unified them — no separate task_model config anymore).
        task_model = auxiliary_model
        # Auto-tier review: profile auto_review_model block, else the auxiliary
        # model, else the main model (Agent.resolve_auxiliary_model order).
        review_model = self._sibling_model(cfg, "auto_review")
        auxiliary_task_models = {"auto_review": review_model} if review_model is not None else None
        fallback_models = self._fallback_models(cfg)
        max_api_retry = int(cfg.get("max_api_retry") or 2)
        log_dir, work_dir = self._locate_session(session_id, owner)
        if not Path(work_dir).is_dir():
            raise FileNotFoundError(f"Session working directory no longer exists: {work_dir}")

        from agentica.tools.product import PRODUCT_DESCRIPTION, PRODUCT_INSTRUCTIONS, build_product_tools
        from ..model_factory import _gateway_cron_job_runner

        extra = build_product_tools(
            surface="gateway",
            work_dir=work_dir,
            owner=self._owner(owner),
            session_id=session_id,
            service=self,
            cron_job_runner=_gateway_cron_job_runner,
            extra_tools=list(self.extra_tools),
        )
        if settings.browser_tool_enabled:
            # Keep AF_UNIX socket paths short, including the daemon's directory.
            browser_session = sha256(self._sk(session_id, owner).encode()).hexdigest()[:24]
            driver = AgentBrowserDriver(session=f"ag-gw-{browser_session}", restore=False, isolated=True)
            browser_tool = BrowserTool(driver=driver)
            self._browser_tools[self._sk(session_id, owner)] = browser_tool
            extra.append(browser_tool)

        # Same identity and instructions as the CLI (agentica.tools.product):
        # a session that moves between terminal, web and IM keeps its prefix.
        instructions = [PRODUCT_INSTRUCTIONS, *(self.extra_instructions or [])]

        permission_mode = self.get_session_approval_mode(session_id, owner)
        sandbox_mode, sandbox_network_access = self._sandbox_settings()
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
            session_base_dir=log_dir,
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            auxiliary_task_models=auxiliary_task_models,
            session_profile=cfg.get("profile") or None,
            fallback_models=fallback_models,
            max_api_retry=max_api_retry,
            description=PRODUCT_DESCRIPTION,
            tools=extra if extra else None,
            workspace=self._workspace,
            work_dir=work_dir,
            user_id=self._owner(owner),
            num_history_turns=settings.num_history_turns,
            instructions=instructions,
            debug=settings.debug,
            # memory, skills, experience capture, workspace memory all on by
            # DeepAgent default. ask_user_question is the builtin one, mounted
            # where the CLI mounts it (same tool order); its callback answers
            # through the session's live turn instead of stdin.
            include_ask_user_question=True,
            ask_user_question_callback=self._make_ask_callback(session_id, owner),
            ask_user_question_batch_callback=self._make_ask_batch_callback(session_id, owner),
            background_process_registry=self._background_registry(session_id, owner),
            permission_mode=permission_mode,
            sandbox=sandbox_mode,
            sandbox_network_access=sandbox_network_access,
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
            f"(extra={len(extra)})"
        )
        return agent

    async def _get_agent(self, session_id: str, owner: Optional[str] = None, *, for_run: bool = False) -> DeepAgent:
        """Return the cached Agent for a session, creating one if absent.

        Concurrent callers share one build. Otherwise a usage poll racing a
        chat start builds a second Agent that replaces the running one in the
        cache, and steer / cancel reach the idle copy.
        """
        key = self._sk(session_id, owner)

        def current(agent: DeepAgent) -> bool:
            return (
                key not in self._stale_agents
                and agent._session_model_selection == self.session_log_for(session_id, owner).model_selection()
            )

        # If the selection changes during construction, give the run one
        # rebuild. Further changes must not keep a request waiting indefinitely.
        for _ in range(2):
            agent = self._cache.get(key)
            if agent is not None and agent.user_id == self._owner(owner):
                if (not for_run and self.is_session_active(session_id, owner)) or current(agent):
                    return agent
            build = self._agent_builds.get(key)
            if build is None:
                build = asyncio.ensure_future(self._build_and_cache_agent(key, session_id, owner))
                self._agent_builds[key] = build

                def _forget(done: asyncio.Future) -> None:
                    if self._agent_builds.get(key) is done:
                        del self._agent_builds[key]

                build.add_done_callback(_forget)
            agent = await asyncio.shield(build)
            # A usage poll can have started building before a model update.
            if not for_run or current(agent):
                return agent
        raise RuntimeError("Session model changed repeatedly while it was being prepared. Please retry.")

    async def _build_and_cache_agent(self, key: str, session_id: str, owner: Optional[str]) -> DeepAgent:
        self._stale_agents.discard(key)
        selection = self.session_log_for(session_id, owner).model_selection()

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

        agent._session_model_selection = selection
        agent.set_permission_mode(self.get_session_approval_mode(session_id, owner))
        self._cache.put(key, agent)
        await self._drain_released_agents()
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

        def ask(
            prompt: str,
            options: Optional[List[str]] = None,
            multi_select: bool = False,
        ) -> str:
            pkg = _pkg()
            wait_s = human_wait_timeout()
            loop = self._loop
            turn = pkg.live_turn.active(session_id, owner_id)
            if loop is None or turn is None:
                raise AskUserQuestionTimeout(0)
            opts = [str(o) for o in (options or [])]

            async def open_and_wait() -> Optional[str]:
                if turn.done:
                    return None
                question, future = turn.open_question(prompt, opts, multi_select)
                if turn.question_notifier is not None:
                    try:
                        await turn.question_notifier(question)
                    except Exception:
                        # The question is already on the turn. A failed IM
                        # send must not abandon the Web/Desktop card or fail
                        # the tool before anyone can answer.
                        logger.exception("ask_user_question notifier failed")
                answer = await turn.wait_for_question(question, future, wait_s)
                assert answer is None or isinstance(answer, str)
                return answer

            answer = asyncio.run_coroutine_threadsafe(open_and_wait(), loop).result()
            if answer is None:
                raise AskUserQuestionTimeout(wait_s)
            return answer

        return ask

    def _make_ask_batch_callback(
        self, session_id: str, owner: Optional[str],
    ) -> Callable[[list[dict]], Awaitable[dict[str, str] | None]]:
        owner_id = self._owner(owner)

        async def ask(questions: list[dict]) -> dict[str, str] | None:
            turn = _pkg().live_turn.active(session_id, owner_id)
            if turn is None or turn.done:
                raise AskUserQuestionTimeout(0)
            if turn.question_notifier is not None:
                # IM capability cards still deliver one question at a time.
                return None
            wait_s = human_wait_timeout()
            question, future = turn.open_question(questions=questions)
            answer = await turn.wait_for_question(question, future, wait_s)
            if answer is None:
                raise AskUserQuestionTimeout(wait_s)
            assert isinstance(answer, dict)
            return answer

        return ask

    _DEFAULT_APPROVAL_MODE = "auto"

    @staticmethod
    def _sandbox_settings() -> tuple:
        """``(sandbox mode, network access)`` from config.yaml ``settings``.

        Default off. A deployment-level value degrades to off with a warning
        when this host cannot enforce it, instead of failing every request; an
        explicit ``agentica --sandbox`` argument still raises in the CLI path
        (argument = intent, config = deployment setting).
        """
        mode = get_setting("sandbox", "off")
        if mode in (None, "", "off"):
            return "off", False
        try:
            validate_sandbox_mode(mode)
        except ValueError as e:
            logger.warning("settings.sandbox is invalid (%s); using off", e)
            return "off", False
        from agentica.tools.os_sandbox import unsupported_reason

        reason = unsupported_reason()
        if reason:
            logger.warning(
                "settings.sandbox=%r is ignored: %s", mode, reason,
            )
            return "off", False
        return mode, bool(get_setting("sandbox_network_access", False))

    def set_session_approval_mode(
        self, session_id: str, mode: str, owner: Optional[str] = None,
    ) -> None:
        """Persist the selected approval mode for a session and apply it live."""
        normalized = (mode or self._DEFAULT_APPROVAL_MODE).strip().lower()
        validate_permission_mode(normalized)
        key = self._sk(session_id, owner)
        if self._saved_approval_mode(session_id, owner) != normalized:
            # One field for every client: the CLI's ``cli.permissions`` sidecar
            # key. ``--resume`` in the terminal and a fork read the same tier
            # the web set, and the session list reads it back after a restart.
            #
            # The live tier is what this turn runs with. A sidecar that cannot
            # be written (read-only home, full disk) must not turn a mode switch
            # into a 500 on a live control path — the in-memory value below is
            # still applied, and the next successful switch persists it.
            try:
                self.session_log_for(session_id, owner).set_cli_prefs({"permissions": normalized})
            except Exception as exc:  # noqa: BLE001 - persistence must not break the switch
                logger.warning(
                    "could not persist approval mode %r for %s: %s", normalized, session_id, exc,
                )
        self._session_approval_modes[key] = normalized

        cached_agent = self._cache.get(key)
        if cached_agent is not None:
            cached_agent.set_permission_mode(normalized)

        owner_id = self._owner(owner)
        turn = _pkg().live_turn.active(session_id, owner_id)
        if turn is not None and turn.approvals.size:
            grants = self._grants_for(session_id, owner)
            work_dir = self.get_session_work_dir(session_id, owner)
            sync_grants_from_project(grants, work_dir=work_dir, user_id=owner_id)
            for pending in turn.approvals.list():
                route = classify_pending(normalized, pending, grants, work_dir=work_dir)
                if route in ("allow", "deny"):
                    turn.resolve_approval(pending.tool_call_id, route, by="mode")

    def get_session_approval_mode(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._saved_approval_mode(session_id, owner) or self._DEFAULT_APPROVAL_MODE

    def seed_session_approval_mode(
        self, session_id: str, mode: Optional[str], owner: Optional[str] = None,
    ) -> None:
        """Give a chat that never had a tier its starting one (an IM chat
        takes the account default). A chat that has one keeps it: the tier
        picked on the web for this conversation is not undone by the next
        message arriving from WeChat."""
        if mode and self._saved_approval_mode(session_id, owner) is None:
            self.set_session_approval_mode(session_id, mode, owner)

    def _saved_approval_mode(self, session_id: str, owner: Optional[str] = None) -> Optional[str]:
        """The tier this chat was set to, or ``None`` if it never was.

        The default is not stored: a chat that never had a tier reports none,
        so the Web shows the tier its next run will actually be sent with.
        """
        key = self._sk(session_id, owner)
        mode = self._session_approval_modes.get(key)
        if mode is None:
            saved = normalize_cli_prefs(self.session_log_for(session_id, owner).get_cli_prefs()).get("permissions")
            if saved is None:
                return None
            mode = self._session_approval_modes[key] = saved
        return mode

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
            wait_s = human_wait_timeout()
            if wait_s > 0:
                # A no-op once someone decided: ``expire`` finds nothing parked.
                asyncio.get_running_loop().call_later(
                    wait_s, turn.expire_approval, pending.tool_call_id, wait_s,
                )

        def apply_path_grant(path: str, prefix: bool) -> None:
            agent = self._cache.get(self._sk(session_id, owner))
            if agent is not None:
                apply_path_grant_on_agent(agent, path, prefix=prefix)

        async def review(fc):
            agent = self._cache.get(self._sk(session_id, owner))
            review_fn = getattr(agent, "review_tool_call", None) if agent is not None else None
            return None if review_fn is None else await review_fn(fc)

        return make_approve(
            get_mode=lambda: self.get_session_approval_mode(session_id, owner),
            get_grants=lambda: self._grants_for(session_id, owner),
            get_registry=get_registry,
            get_work_dir=lambda: self.get_session_work_dir(session_id, owner),
            publish=publish,
            apply_path_grant=apply_path_grant,
            get_user_id=lambda: owner_id,
            review=review,
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
