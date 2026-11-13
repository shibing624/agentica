# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: AgentService package wrapping the agentica SDK for Web, Desktop, IM, and Cron.
"""
from __future__ import annotations

from agentica import DeepAgent
from agentica.config.profiles import (
    get_active_profile_name,
    get_profile,
    get_profiles,
    get_project_profile,
)
from agentica.skills import get_skill_registry

from ...config import settings
from .. import chat_images, live_turn
from ..media_understanding import media_understanding
from ..model_factory import (
    create_model,
    get_cron_instructions,
    get_cron_tools,
    get_self_manage_instructions,
    get_self_manage_tools,
)
from ..response_formatter import extract_metrics, format_tool_call_args, format_tool_result
from ..session_usage import turn_usage_payload, usage_payload
from .core import AgentService
from .types import (
    CRON_SESSION_PREFIX,
    IM_SESSION_SOURCES,
    QUESTION_WAIT_S,
    ChatResult,
    LRUAgentCache,
    _AGENT_BUILD_TIMEOUT_S,
    _APPROVAL_MODE_INSTRUCTION,
    _COMPACT_NOTES_LIMIT,
    _OWNER_SEP,
    _TITLE_PROMPT,
    _cache_key_is_pinned,
    _infer_im_source_from_session_id,
    _log_report_failure,
    _profile_model_config,
    _read_compact_notes,
    dispatch_stream_chunk,
    goal_event_payload,
    is_system_temp_path,
)

__all__ = [
    "AgentService",
    "CRON_SESSION_PREFIX",
    "ChatResult",
    "DeepAgent",
    "IM_SESSION_SOURCES",
    "LRUAgentCache",
    "QUESTION_WAIT_S",
    "_AGENT_BUILD_TIMEOUT_S",
    "_APPROVAL_MODE_INSTRUCTION",
    "_COMPACT_NOTES_LIMIT",
    "_OWNER_SEP",
    "_TITLE_PROMPT",
    "_cache_key_is_pinned",
    "_infer_im_source_from_session_id",
    "_log_report_failure",
    "_profile_model_config",
    "_read_compact_notes",
    "chat_images",
    "create_model",
    "dispatch_stream_chunk",
    "extract_metrics",
    "format_tool_call_args",
    "format_tool_result",
    "get_active_profile_name",
    "get_cron_instructions",
    "get_cron_tools",
    "get_profile",
    "get_profiles",
    "get_project_profile",
    "get_self_manage_instructions",
    "get_self_manage_tools",
    "get_skill_registry",
    "goal_event_payload",
    "is_system_temp_path",
    "live_turn",
    "media_understanding",
    "settings",
    "turn_usage_payload",
    "usage_payload",
]
