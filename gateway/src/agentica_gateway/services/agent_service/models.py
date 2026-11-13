# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model config resolution, profile switching, and side-task calls for AgentService.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from agentica.config.profiles import (
    apply_global_config,
    provider_api_key_env,
    set_active_profile,
)
from agentica.utils.log import logger

from ...config import settings
from .types import _profile_model_config


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


class AgentModelMixin:
    """Model config properties, per-session profile resolution, and hot reload."""

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

    def _default_model_config(self) -> Dict[str, Any]:
        """The gateway default (``settings``): config.yaml's active profile, or
        the env vars when no profile is configured."""
        pkg = _pkg()
        active_name = pkg.get_active_profile_name()
        active = pkg.get_profile(active_name) if active_name else {}
        aux = active.get("auxiliary_model") if isinstance(active.get("auxiliary_model"), dict) else {}
        matches_active = (
            bool(active)
            and active.get("model_provider") == self.model_provider
            and active.get("model_name") == self.model_name
        )
        raw_fallbacks = active.get("fallback_models") if matches_active else None
        fallbacks: List[Dict[str, Any]] = []
        if isinstance(raw_fallbacks, list):
            for fb in raw_fallbacks:
                if isinstance(fb, dict) and fb.get("model_provider") and fb.get("model_name"):
                    fallbacks.append(dict(fb))
        max_api_retry = active.get("max_api_retry") if matches_active else None
        return {
            "profile": active_name,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "base_url": self.model_base_url,
            "api_key": self.model_api_key,
            "wire_api": self.model_wire_api,
            "reasoning": self.model_reasoning,
            "reasoning_effort": self.model_reasoning_effort,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "enable_cache_control": active.get("enable_cache_control") if matches_active else None,
            "cache_control_messages": active.get("cache_control_messages") if matches_active else None,
            "cache_control_session_header": (active.get("cache_control_session_header") or "") if matches_active else "",
            "cache_keepalive": active.get("cache_keepalive") if matches_active else None,
            "extra_body": (dict(active["extra_body"]) if isinstance(active.get("extra_body"), dict) else None) if matches_active else None,
            "extra_headers": (dict(active["extra_headers"]) if isinstance(active.get("extra_headers"), dict) else None) if matches_active else None,
            "default_headers": (dict(active["default_headers"]) if isinstance(active.get("default_headers"), dict) else None) if matches_active else None,
            "fallback_models": fallbacks,
            "max_api_retry": int(max_api_retry) if max_api_retry is not None else 2,
            "auxiliary_model_provider": self.auxiliary_model_provider,
            "auxiliary_model_name": self.auxiliary_model_name,
            "auxiliary_base_url": self.auxiliary_base_url,
            "auxiliary_api_key": self.auxiliary_api_key,
            "auxiliary_wire_api": self.auxiliary_wire_api,
            "auxiliary_reasoning": self.auxiliary_reasoning,
            "auxiliary_reasoning_effort": self.auxiliary_reasoning_effort,
            "auxiliary_extra_body": (dict(aux["extra_body"]) if isinstance(aux.get("extra_body"), dict) else None) if matches_active else None,
            "auxiliary_extra_headers": (dict(aux["extra_headers"]) if isinstance(aux.get("extra_headers"), dict) else None) if matches_active else None,
        }

    def _session_profile(self, session_id: str, owner: Optional[str] = None) -> tuple[str, str]:
        """``(profile, source)`` a session runs on; ``("", "default")`` is the
        gateway default.

        A session the user picked a model for keeps it (``<id>.meta.json``
        ``profile_name``, the same sidecar field the CLI resumes from). A
        session without a choice follows its project's default
        (``project.json`` ``active_profile``), then config.yaml's. A name that
        is no longer in config.yaml falls through to the next layer.
        """
        pkg = _pkg()
        profiles = pkg.get_profiles()
        picked = self.session_log_for(session_id, owner).get_profile_name()
        if picked and picked in profiles:
            return picked, "session"
        project = pkg.get_project_profile(self.get_session_work_dir(session_id, owner))
        if project and project in profiles:
            return project, "project"
        return "", "default"

    def session_profile_name(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_profile(session_id, owner)[0]

    def session_profile_source(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_profile(session_id, owner)[1]

    def session_model_config(self, session_id: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """The flat model config a session's agent is built from."""
        name = self.session_profile_name(session_id, owner)
        if name:
            profile = _pkg().get_profile(name)
            if profile and profile.get("model_provider") and profile.get("model_name"):
                return _profile_model_config(name, profile)
        return self._default_model_config()

    async def set_session_profile(self, session_id: str, name: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """Pin a session to a profile; other sessions keep theirs.

        A run in flight finishes on its current model; the session rebuilds on
        its next request.
        """
        self.session_log_for(session_id, owner).set_profile(name, "session")
        key = self._sk(session_id, owner)
        lock = self._session_locks.get(key)
        if lock is not None and lock.locked():
            self._stale_agents.add(key)
        else:
            self._cache.delete(key)
        return self.session_model_config(session_id, owner)

    def _session_provider(self, session_id: str, owner: Optional[str] = None) -> str:
        """Provider slug of the session's model, for the ``provider/model`` label."""
        return self.session_model_config(session_id, owner)["model_provider"]

    @staticmethod
    def _sibling_model(cfg: Dict[str, Any], prefix: str) -> Optional[Any]:
        """Build a sibling (auxiliary) model if a model name is configured.

        Returns None when no sibling name is set — DeepAgent will reuse
        the main model. The auxiliary model also serves as the task subagent
        model (CLI unified them).
        """
        sibling_name = cfg[f"{prefix}_model_name"]
        if not sibling_name:
            return None
        return _pkg().create_model(
            cfg[f"{prefix}_model_provider"] or cfg["model_provider"],
            sibling_name,
            base_url=cfg[f"{prefix}_base_url"] or None,
            api_key=cfg[f"{prefix}_api_key"] or None,
            wire_api=cfg[f"{prefix}_wire_api"],
            max_tokens=cfg.get("max_tokens") or 0,
            temperature=cfg.get("temperature") or 0.0,
            top_p=cfg.get("top_p") or 0.0,
            context_window=cfg.get("context_window") or 0,
            reasoning=cfg[f"{prefix}_reasoning"],
            reasoning_effort=cfg[f"{prefix}_reasoning_effort"],
            thinking=settings.model_thinking,
            enable_cache_control=cfg.get("enable_cache_control"),
            cache_control_messages=cfg.get("cache_control_messages"),
            cache_control_session_header=cfg.get("cache_control_session_header") or None,
            cache_keepalive=cfg.get("cache_keepalive"),
            extra_body=cfg.get(f"{prefix}_extra_body"),
            extra_headers=cfg.get(f"{prefix}_extra_headers"),
        )

    @staticmethod
    def _fallback_models(cfg: Dict[str, Any]) -> List[Any]:
        """Build fallback Model instances from resolved profile fallback entries."""
        create_model = _pkg().create_model
        models: List[Any] = []
        for fb in cfg.get("fallback_models") or []:
            if not isinstance(fb, dict) or not fb.get("model_provider") or not fb.get("model_name"):
                continue
            models.append(
                create_model(
                    fb["model_provider"],
                    fb["model_name"],
                    base_url=fb.get("base_url") or None,
                    api_key=fb.get("api_key") or None,
                    wire_api=fb.get("wire_api") or "",
                    reasoning=fb.get("reasoning") or "",
                    reasoning_effort="" if fb.get("wire_api") == "responses" else (fb.get("reasoning_effort") or ""),
                    extra_body=fb.get("extra_body") if isinstance(fb.get("extra_body"), dict) else None,
                    extra_headers=fb.get("extra_headers") if isinstance(fb.get("extra_headers"), dict) else None,
                    default_headers=fb.get("default_headers") if isinstance(fb.get("default_headers"), dict) else None,
                )
            )
        return models

    def _build_sibling_model(self, prefix: str) -> Optional[Any]:
        return self._sibling_model(self._default_model_config(), prefix)

    def side_task_model(self) -> Any:
        """The model for gateway side tasks (titles, commit messages, prompt
        polish): the auxiliary model when configured, else the main model."""
        aux = self._build_sibling_model("auxiliary")
        if aux is not None:
            return aux
        return _pkg().create_model(
            self.model_provider,
            self.model_name,
            base_url=self.model_base_url or None,
            api_key=self.model_api_key or None,
            wire_api=self.model_wire_api,
            reasoning=self.model_reasoning,
            reasoning_effort=self.model_reasoning_effort,
        )

    async def side_task(self, system: str, user: str) -> str:
        """One system+user call on :meth:`side_task_model`; the reply text."""
        from agentica.model.message import Message

        model = self.side_task_model()
        response = await model.response([
            Message(role="system", content=system),
            Message(role="user", content=user),
        ])
        return ((response.content if response else "") or "").strip()

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
            self._drop_cached_agents()
            logger.info(
                f"Profile reloaded: {profile_name or 'active'} -> "
                f"{self.model_provider}/{self.model_name}"
            )
