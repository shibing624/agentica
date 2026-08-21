# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Channel routes: /api/channels, /api/send, /webhook/*

Simplified to use the SDK cron module (agentica.cron) instead of the old
gateway-embedded SchedulerService.
"""
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import deps
from ..channels.base import ChannelType
from ..channels.catalog import channel_overview
from ..channels.wechat import WeChatChannel
from ..config import settings
from ..models import SendRequest
from ..services.channel_manager import ChannelManager

router = APIRouter()


@router.get("/api/channels")
async def list_channels(
    request: Request,
    cm: ChannelManager = Depends(deps.get_channel_manager),
):
    """Registered channels plus the full web/IM catalog the Personal Assistant page uses."""
    port = request.url.port if request.url.port is not None else settings.port
    return channel_overview(
        cm,
        host=settings.host,
        port=port,
        web_url=f"{str(request.base_url).rstrip('/')}/chat",
    )


def _wechat_channel(cm: ChannelManager) -> WeChatChannel:
    ch = cm.get_channel(ChannelType.WECHAT)
    if not isinstance(ch, WeChatChannel):
        raise HTTPException(status_code=503, detail="WeChat channel is not registered")
    return ch


@router.post("/api/channels/wechat/qr")
async def wechat_qr_start(
    request: Request,
    cm: ChannelManager = Depends(deps.get_channel_manager),
):
    """Mint a WeChat login QR. Scan it in the Personal Assistant page."""
    owner = request.state.principal.user_id
    try:
        return await _wechat_channel(cm).start_web_qr(owner=owner)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/channels/wechat/qr")
async def wechat_qr_poll(
    id: str,
    request: Request,
    cm: ChannelManager = Depends(deps.get_channel_manager),
):
    """Poll one WeChat QR until confirmed, expired, or still waiting."""
    if not id.strip():
        raise HTTPException(status_code=400, detail="Missing qr id")
    owner = request.state.principal.user_id
    try:
        return await _wechat_channel(cm).poll_web_qr(id.strip(), owner=owner)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/send")
async def send_message(
    request: SendRequest,
    cm: ChannelManager = Depends(deps.get_channel_manager),
):
    success = await cm.send(request.channel, request.channel_id, request.message)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to send message")
    return {"status": "sent"}


@router.post("/webhook/feishu")
async def feishu_webhook(request: dict):
    """Feishu webhook endpoint (URL verification + event delivery)."""
    if "challenge" in request:
        return {"challenge": request["challenge"]}
    return {"status": "ok"}
