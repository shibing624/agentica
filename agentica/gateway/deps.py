# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Dependency injection helpers for FastAPI routes.

All routes access shared service instances through these Depends() functions.
Global instances are set during app lifespan startup.
"""
import asyncio
from typing import Any, Optional
from fastapi import HTTPException

from .services.agent_service import AgentService
from .services.channel_manager import ChannelManager
from .services.router import MessageRouter

# Global service instances — set in main.py lifespan
agent_service: Optional[AgentService] = None
channel_manager: Optional[ChannelManager] = None
message_router: Optional[MessageRouter] = None
# AgentRunner (agentica.cron.scheduler.AgentRunner protocol) used to execute
# cron jobs immediately (HTTP "run now" and the agent's own cronjob(action=
# "run") tool call) — same runner the background ticker uses, so an
# immediate run and a scheduled run behave identically.
cron_runner: Optional[Any] = None
# The gateway's single asyncio event loop, captured at startup. The cron
# tool's immediate-run path is invoked from a worker thread (sync tool
# entrypoint), so it schedules its coroutine back onto this loop with
# run_coroutine_threadsafe() instead of spinning up a second event loop —
# that keeps it on the same AgentService session locks/state as every other
# request instead of creating a cross-loop hazard.
main_loop: Optional[asyncio.AbstractEventLoop] = None


def get_agent_service() -> AgentService:
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return agent_service


def get_channel_manager() -> ChannelManager:
    if not channel_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    return channel_manager


def get_cron_runner() -> Any:
    if not cron_runner:
        raise HTTPException(status_code=503, detail="Service not ready")
    return cron_runner
