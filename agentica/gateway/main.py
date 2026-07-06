# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: FastAPI application entry point.

Responsibilities:
- App creation and lifespan management
- Middleware registration (CORS, authentication, request ID)
- Route registration (delegates to src/routes/)
- Channel setup and channel message handler
- Serve static files and SPA HTML
"""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from agentica.run_context import RunSource
from agentica.utils.log import logger
from . import deps
from agentica.version import __version__
from .config import settings
from .services.agent_service import AgentService
from .services.channel_manager import ChannelManager
from .services.router import MessageRouter
from .routes import chat, config as config_routes, scheduler as scheduler_routes, channels, ws

# ContextVar holding the current request ID — async-safe, no threading issues
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Return the request ID for the current async context."""
    return _request_id_var.get()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup; clean up on shutdown."""
    logger.info("=" * 50)
    logger.info(f"  Agentica Gateway v{__version__}")
    logger.info(f"  Workspace: {settings.workspace_path}")
    logger.info(f"  Model:     {settings.model_provider}/{settings.model_name}")
    if settings.gateway_token:
        logger.info("  Auth:      token enabled")
    else:
        logger.warning("  Auth:      GATEWAY_TOKEN not set — API is open (local dev only)")
    logger.info("=" * 50)

    # Agent service
    agent_svc = AgentService(workspace_path=str(settings.workspace_path))
    # Eagerly initialize so startup errors surface immediately (fail fast)
    try:
        await agent_svc._ensure_initialized()
    except RuntimeError as e:
        logger.error(f"FATAL: {e}")
        raise

    deps.agent_service = agent_svc

    # Channel manager + message router
    deps.channel_manager = ChannelManager()
    deps.message_router = MessageRouter(default_agent="main")

    # Cron scheduler — tick every 60s using the new SDK cron module
    from agentica.cron.scheduler import tick as cron_tick

    cron_runner = _GatewayAgentRunner(agent_svc)

    async def _cron_ticker():
        while True:
            await asyncio.sleep(60)
            try:
                await cron_tick(agent_runner=cron_runner)
            except Exception as e:
                logger.error(f"Cron tick error: {e}")

    cron_task = asyncio.create_task(_cron_ticker())
    logger.info("Cron scheduler started (60s tick)")

    # Channels
    await _setup_channels()

    logger.info(f"Gateway started — http://{settings.host}:{settings.port}/chat")

    yield

    # Shutdown
    logger.info("Shutting down...")
    cron_task.cancel()
    try:
        await cron_task
    except (asyncio.CancelledError, Exception):
        pass
    if deps.channel_manager:
        await deps.channel_manager.disconnect_all()
    logger.info("Goodbye!")


# ============== App ==============

app = FastAPI(
    title="Agentica Gateway",
    description="Python AI Agent Gateway",
    version=__version__,
    lifespan=lifespan,
)

# CORS — allow all origins by default (personal daemon, locked down via gateway_token)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request ID middleware ==============

@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """Assign a unique request ID to every request.

    - Stores the ID in the async ContextVar so handlers can read it via
      ``get_request_id()`` and include it in log messages when relevant.
    - Echoes the ID back in the X-Request-ID response header for client tracing.
    """
    req_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    token = _request_id_var.set(req_id)
    try:
        response = await call_next(request)
    finally:
        _request_id_var.reset(token)
    response.headers["X-Request-ID"] = req_id
    return response


# ============== Authentication middleware ==============

# Endpoints that bypass token authentication
_AUTH_EXEMPT = {"/", "/health", "/api/health", "/chat", "/webhook/feishu"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next) -> Response:
    """Enforce gateway_token for all /api/* and /ws requests.

    Skips auth when:
    - gateway_token is not configured (local dev)
    - path is in _AUTH_EXEMPT
    - path is a static asset
    """
    if not settings.gateway_token:
        return await call_next(request)

    path = request.url.path
    if path in _AUTH_EXEMPT or path.startswith("/static"):
        return await call_next(request)

    # Check Authorization header (Bearer <token>) or ?token= query param
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
    if not token:
        token = request.query_params.get("token", "")

    if token != settings.gateway_token:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: invalid or missing gateway token"},
        )

    return await call_next(request)


# ============== Static files + SPA ==============

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/chat", response_class=HTMLResponse)
async def web_chat():
    """Serve the single-page web UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============== Route registration ==============

app.include_router(config_routes.router)
app.include_router(chat.router)
app.include_router(scheduler_routes.router)
app.include_router(channels.router)
app.include_router(ws.router)


# ============== Channel setup ==============

async def _setup_channels() -> None:
    """Instantiate and connect configured channels."""
    if not deps.channel_manager:
        return

    from .channels.feishu import FeishuChannel
    from .channels.telegram import TelegramChannel
    from .channels.discord import DiscordChannel
    from .channels.qq import QQChannel
    from .channels.wecom import WeComChannel
    from .channels.dingtalk import DingTalkChannel
    from .channels.wechat import WeChatChannel

    if settings.feishu_app_id and settings.feishu_app_secret:
        try:
            feishu = FeishuChannel(
                app_id=settings.feishu_app_id,
                app_secret=settings.feishu_app_secret,
                allowed_users=settings.feishu_allowed_users,
                allowed_groups=settings.feishu_allowed_groups,
            )
            deps.channel_manager.register(feishu)
        except Exception as e:
            logger.error(f"Failed to create Feishu channel: {e}")

    if settings.telegram_bot_token:
        try:
            telegram = TelegramChannel(
                bot_token=settings.telegram_bot_token,
                allowed_users=settings.telegram_allowed_users,
            )
            deps.channel_manager.register(telegram)
        except Exception as e:
            logger.error(f"Failed to create Telegram channel: {e}")

    if settings.discord_bot_token:
        try:
            discord = DiscordChannel(
                bot_token=settings.discord_bot_token,
                allowed_users=settings.discord_allowed_users,
                allowed_guilds=settings.discord_allowed_guilds,
            )
            deps.channel_manager.register(discord)
        except Exception as e:
            logger.error(f"Failed to create Discord channel: {e}")

    if settings.qq_app_id and settings.qq_app_secret:
        try:
            qq = QQChannel(
                app_id=settings.qq_app_id,
                app_secret=settings.qq_app_secret,
                allowed_users=settings.qq_allowed_users,
            )
            deps.channel_manager.register(qq)
        except Exception as e:
            logger.error(f"Failed to create QQ channel: {e}")

    if settings.wecom_bot_id and settings.wecom_secret:
        try:
            wecom = WeComChannel(
                bot_id=settings.wecom_bot_id,
                secret=settings.wecom_secret,
                allowed_users=settings.wecom_allowed_users,
            )
            deps.channel_manager.register(wecom)
        except Exception as e:
            logger.error(f"Failed to create WeCom channel: {e}")

    if settings.dingtalk_client_id and settings.dingtalk_client_secret:
        try:
            dingtalk = DingTalkChannel(
                client_id=settings.dingtalk_client_id,
                client_secret=settings.dingtalk_client_secret,
                allowed_users=settings.dingtalk_allowed_users,
            )
            deps.channel_manager.register(dingtalk)
        except Exception as e:
            logger.error(f"Failed to create DingTalk channel: {e}")

    # Personal WeChat: only enabled when an explicit token file or
    # allowlist is configured (avoids triggering interactive QR login on
    # every gateway startup by accident).
    if settings.wechat_token_file or settings.wechat_allowed_users:
        try:
            wechat = WeChatChannel(
                token_file=settings.wechat_token_file,
                allowed_users=settings.wechat_allowed_users,
            )
            deps.channel_manager.register(wechat)
        except Exception as e:
            logger.error(f"Failed to create WeChat channel: {e}")

    deps.channel_manager.set_handler(_handle_channel_message)
    await deps.channel_manager.connect_all()


async def _handle_channel_message(message) -> None:
    """Route an incoming channel message through the agent and reply."""
    logger.info(f"[{message.channel.value}] {message.sender_id}: {message.content[:500]}")

    if not deps.agent_service:
        logger.error("Agent service not ready")
        return

    agent_id = deps.message_router.route(message)
    session_id = deps.message_router.get_session_id(message, agent_id)
    user_id = message.sender_id or settings.default_user_id

    try:
        result = await deps.agent_service.chat(
            message=message.content,
            session_id=session_id,
            user_id=user_id,
        )

        if result.content:
            await deps.channel_manager.send(
                message.channel,
                message.channel_id,
                result.content,
            )

        await ws.ws_manager.broadcast("channel.message", {
            "channel": message.channel.value,
            "sender": message.sender_id,
            "userId": user_id,
            "content": message.content[:100],
            "response": result.content[:100] if result.content else "",
        })

    except Exception as e:
        logger.error(f"Handle channel message error: {e}")
        try:
            await deps.channel_manager.send(
                message.channel,
                message.channel_id,
                "error processing message",
            )
        except Exception:
            pass


# ============== Scheduler agent runner ==============

class _GatewayAgentRunner:
    """Adapts AgentService to the AgentRunner protocol expected by JobExecutor."""

    def __init__(self, agent_svc: AgentService):
        self._svc = agent_svc

    async def run(self, prompt: str, context: Optional[dict] = None) -> str:
        ctx = context or {}
        job_id = ctx.get("job_id", str(uuid4()))
        user_id = ctx.get("user_id", settings.default_user_id)
        session_id = f"scheduled_{job_id}"

        result = await self._svc.chat(
            message=prompt,
            session_id=session_id,
            user_id=user_id,
            source=RunSource.cron,
        )
        return result.content


# ============== Entry point ==============

def main() -> None:
    """Start the gateway server."""
    import uvicorn
    uvicorn.run(
        "agentica.gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
