# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Gateway services: agent orchestration, channel management, and message routing.
"""
from .agent_service import AgentService, ChatResult
from .channel_manager import ChannelManager
from .model_factory import create_model
from .response_formatter import extract_metrics, format_tool_call_args, format_tool_result
from .router import MessageRouter, RoutingRule

__all__ = [
    "AgentService",
    "ChatResult",
    "ChannelManager",
    "MessageRouter",
    "RoutingRule",
    "create_model",
    "extract_metrics",
    "format_tool_call_args",
    "format_tool_result",
]
