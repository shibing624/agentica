# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: AgentService core composing model, builder, execution, and session mixins.
"""
from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agentica import DeepAgent
from agentica.agent.approvals import SessionGrants
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.utils.log import logger
from agentica.workspace import Workspace

from ...config import settings
from .builder import AgentBuilderMixin
from .execution import AgentExecutionMixin
from .models import AgentModelMixin
from .sessions import AgentSessionsMixin
from .types import LRUAgentCache, _OWNER_SEP


class AgentService(AgentModelMixin, AgentBuilderMixin, AgentExecutionMixin, AgentSessionsMixin):
    """Agent service wrapping the agentica SDK."""

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
        self._session_work_dirs: Dict[str, str] = {}
        self._session_approval_modes: Dict[str, str] = {}
        self._session_grants: Dict[str, SessionGrants] = {}
        self._session_preview_cache: OrderedDict[str, tuple[tuple[int, int], Dict[str, Any]]] = OrderedDict()
        self._session_preview_lock = threading.Lock()
        self._titled: set[tuple[str, str]] = set()
        self._title_tasks: set[asyncio.Task] = set()
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._stale_agents: set[str] = set()
        self._agent_builds = {}
        self._workspace: Optional[Workspace] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._session_background: Dict[str, BackgroundProcessRegistry] = {}
        self.background_report_handler: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @staticmethod
    def _owner(owner: Optional[str]) -> str:
        """Which ``users/<id>/`` partition a call reads and writes."""
        return owner or settings.default_user_id

    @classmethod
    def _sk(cls, session_id: str, owner: Optional[str] = None) -> str:
        return f"{cls._owner(owner)}{_OWNER_SEP}{session_id}"

    def _map_keys(self, session_id: str, owner: Optional[str] = None) -> List[str]:
        """Keys in the owner-partitioned maps that refer to ``session_id``."""
        if owner is not None:
            return [self._sk(session_id, owner)]
        suffix = f"{_OWNER_SEP}{session_id}"
        keys = {self._sk(session_id, None)}
        for k in self._cache.keys():
            if k == session_id or k.endswith(suffix):
                keys.add(k)
        for k in self._session_locks:
            if k == session_id or k.endswith(suffix):
                keys.add(k)
        return list(keys)

    def update_work_dir(self, new_dir: str) -> None:
        """Update the global work_dir and clear ALL cached agents."""
        self._cache.clear()
        self._session_work_dirs.clear()
        logger.info(f"Global work_dir updated to: {new_dir}, all agent instances cleared")

    def _auto_extract_for(self, owner: Optional[str]) -> bool:
        """Per-account toggle for conversation memory extraction. Default on."""
        from agentica_gateway.routes.settings import _clean_prefs, _read_prefs

        prefs = _clean_prefs(_read_prefs(self._owner(owner)))
        return bool(prefs.get("auto_extract_memory", True))

    def _switch_workspace_user(self, owner: str) -> Workspace:
        if self._workspace is None:
            raise RuntimeError("Workspace is not initialized")
        self._workspace.set_user(self._owner(owner))
        return self._workspace

    # ============== User AGENTS.md / MEMORY.md ==============

    async def read_user_agents_md(self, owner: str) -> dict:
        """Return this account's user-level AGENTS.md for the settings page."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        content = await asyncio.to_thread(ws.read_user_agents_md)
        return {
            "content": content,
            "path": str(ws.user_agent_md_path()),
            "empty_template": Workspace._is_empty_template(content),
            "user_id": self._owner(owner),
        }

    async def write_user_agents_md(self, owner: str, content: str) -> dict:
        """Replace this account's user-level AGENTS.md."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        try:
            await asyncio.to_thread(ws.write_user_agents_md, content)
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return await self.read_user_agents_md(owner)

    async def read_memory_index(self, owner: str) -> dict:
        """Return this account's MEMORY.md index entries for the settings page."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        entries = await ws.list_memory_index_entries()
        return {
            "auto_extract": self._auto_extract_for(owner),
            "index_path": str(ws.memory_index_path()),
            "entries": entries,
            "user_id": self._owner(owner),
        }

    async def read_memory_entry(self, owner: str, filename: str) -> dict | None:
        """One MEMORY.md topic file for the settings page."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        entries = await ws.list_memory_index_entries()
        meta = next((e for e in entries if e.get("filename") == filename), None)
        if not meta:
            return None
        body, origin = await ws.read_memory_entry(filename)
        return {
            "filename": filename,
            "title": meta.get("title", ""),
            "hook": meta.get("hook", ""),
            "content": body,
            "origin_work_dir": origin,
            "path": str(ws.get_user_memory_dir() / filename),
            "user_id": self._owner(owner),
        }

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

    async def _invalidate_cache(self) -> None:
        """Clear the agent cache so agents rebuild on next request."""
        async with self._init_lock:
            self._initialized = False
            self._drop_cached_agents()
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

    @property
    def workspace(self) -> Optional[Workspace]:
        """Shared Workspace instance (synchronous; call after initialization)."""
        return self._workspace

    @property
    def agent(self) -> Optional[DeepAgent]:
        """Deprecated: returns an arbitrary cached DeepAgent."""
        sessions = self._cache.keys()
        if sessions:
            return self._cache.get(sessions[0])
        return None
