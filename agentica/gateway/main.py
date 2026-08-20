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
import argparse
import asyncio
import os
import socket
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    import uvicorn

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from agentica.config import AGENTICA_LOG_LEVEL
from agentica.utils.log import enable_process_file_logging, logger
from . import accounts, auth, deps, runtime
from agentica.version import __version__
from .config import settings
from .services.agent_service import AgentService
from .services.channel_manager import ChannelManager
from .services.agent_peers import GatewayAgentPeers
from .services.peer_bridge import PeerBridge
from .services.router import MessageRouter
from .routes import auth as auth_routes, chat, settings as settings_routes, scheduler as scheduler_routes, channels, ws, plugins as plugins_routes, traces as traces_routes

# ContextVar holding the current request ID — async-safe, no threading issues
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Return the request ID for the current async context."""
    return _request_id_var.get()


# ============== Lifespan ==============

def _display_home_path(path: str) -> str:
    """Render ``/Users/me/foo`` as ``~/foo`` for startup banners."""
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _entry_url(record: "runtime.GatewayRuntime") -> str:
    """The address to hand a human.

    Carries the token on the first hop, unless a password is set — then the
    login page is the way in and printing a credential in the log is a leak
    with no upside.
    """
    if record.token and not accounts.store().has_password():
        return f"{record.url}/chat?token={record.token}"
    return f"{record.url}/chat"


# How often the parent-pid watchdog looks. Short enough that a killed shell
# does not leave a gateway holding the port for long, long enough to be free.
PARENT_POLL_SECONDS = 2.0


async def _exit_with_parent(parent_pid: int) -> None:
    """Stop this process once ``parent_pid`` is gone.

    ``os._exit`` on purpose: a graceful shutdown would try to reach the peers
    tree and the IM channels of a session whose owner has already vanished, and
    the shell that would have waited for it is what just died.
    """
    while True:
        await asyncio.sleep(PARENT_POLL_SECONDS)
        if not runtime.is_pid_alive(parent_pid):
            logger.info(f"Parent process {parent_pid} is gone — exiting")
            runtime.unpublish(os.getpid())
            os._exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup; clean up on shutdown."""
    log_file = enable_process_file_logging()
    logger.info("=" * 50)
    logger.info(f"  Agentica Gateway v{__version__}")
    logger.info(f"  Workspace: {settings.workspace_path}")
    logger.info(f"  Work dir:  {settings.base_dir}")
    logger.info(f"  Model:     {settings.model_provider}/{settings.model_name}")
    if log_file:
        logger.info(f"  Log File ({AGENTICA_LOG_LEVEL}): {_display_home_path(log_file)}")
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

    # Cron scheduler — uses the same SDK cron module (agentica.cron.*) and
    # jobs.json store as the CLI, never the OS crontab. Gated by the same
    # `cron.enabled` config.yaml setting the CLI's `/cron daemon on` toggles,
    # so both surfaces share one on/off switch.
    from agentica.cron.scheduler import tick as cron_tick
    from agentica.global_config import get_setting

    cron_runner = _GatewayAgentRunner(agent_svc)
    deps.cron_runner = cron_runner
    deps.main_loop = asyncio.get_running_loop()

    cron_task = None
    if get_setting("cron.enabled", False):
        interval = int(get_setting("cron.interval", 60) or 60)

        async def _cron_ticker():
            while True:
                await asyncio.sleep(interval)
                try:
                    await cron_tick(agent_runner=cron_runner)
                except Exception as e:
                    logger.error(f"Cron tick error: {e}")

        cron_task = asyncio.create_task(_cron_ticker())
        logger.info(f"Cron scheduler started ({interval}s tick)")
    else:
        logger.info("Cron scheduler disabled (set `cron.enabled: true` in ~/.agentica/config.yaml to enable)")

    # Channels (IM integrations: WeChat / WeCom / Feishu / QQ / ... )
    await _setup_channels()

    # Peer bridge: `@list` / `@<session> <text>` from a chat app reaches the
    # agentica CLI sessions running on this machine. Started after the channels
    # because it sends through them.
    if settings.peer_bridge_enabled:
        # The same reach for the gateway's own agent: an `@` command is the
        # user addressing a session themselves, this is the agent doing it on
        # their behalf ("让三个会话都把改动提交了"). Same switch, because it is
        # the same trust boundary, and same ordering — replies are pushed back
        # through the channels. Built before the bridge, which needs its peer
        # ids to keep the gateway agent out of `@list`.
        deps.agent_peers = GatewayAgentPeers(
            channel_manager=deps.channel_manager,
            is_live=deps.agent_service.has_cached_session,
            is_busy=deps.agent_service.is_session_active,
        )
        deps.agent_service.agent_peers = deps.agent_peers
        deps.agent_peers.start()

        deps.peer_bridge = PeerBridge(
            deps.channel_manager,
            gateway_peer_ids=deps.agent_peers.peer_ids,
        )
        deps.peer_bridge.start()
    else:
        logger.info(
            "Peer bridge disabled (PEER_BRIDGE=false) — the gateway's own agent "
            "also gets no list_agents / send_message"
        )

    # Publish how to reach this process (port + token) so a desktop shell or a
    # second terminal can find it, and print the one URL that works on a cold
    # browser — with the token, since without it /chat is a 401.
    record = runtime.GatewayRuntime(
        pid=os.getpid(),
        host=settings.host,
        port=settings.port,
        token=auth.get_token() if auth.auth_enabled() else "",
        version=__version__,
    )
    runtime_path = runtime.publish(record)
    logger.info(f"Web service started — {_entry_url(record)}")
    if not auth.auth_enabled():
        logger.warning(
            "  API auth is OFF (GATEWAY_AUTH=false) — anything that can reach "
            f"{record.url} can run tools as you"
        )
    elif accounts.store().has_password():
        logger.info("  Sign in with your web password")
        logger.info(f"  Runtime (port + token): {_display_home_path(str(runtime_path))}")
    else:
        logger.info(f"  Runtime (port + token): {_display_home_path(str(runtime_path))}")
        logger.info(
            "  No web password set — the URL above is the way in. "
            "`agentica-gateway --set-password` switches to a login page."
        )

    # A desktop shell owns this process; if the shell is killed outright there
    # is nobody left to stop us, and an orphan gateway keeps holding the port
    # and the session locks. Polling is the only check that survives SIGKILL of
    # the parent (an atexit handler over there would never run).
    parent_task = None
    if settings.parent_pid:
        parent_task = asyncio.create_task(_exit_with_parent(settings.parent_pid))

    if deps.channel_manager.channels:
        enabled = ", ".join(c.value for c in deps.channel_manager.channels)
        logger.info(f"IM channels started — {enabled}")
    else:
        logger.info("IM channels — none enabled (configure a channel to enable)")

    yield

    # Shutdown
    logger.info("Shutting down...")
    runtime.unpublish(os.getpid())
    if parent_task is not None:
        parent_task.cancel()
    if cron_task is not None:
        cron_task.cancel()
        try:
            await cron_task
        except (asyncio.CancelledError, Exception):
            pass
    if deps.peer_bridge:
        await deps.peer_bridge.stop()
    if deps.agent_peers:
        await deps.agent_peers.stop()
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

# CORS — loopback only, and that is not a tightening for its own sake.
# Starlette answers `allow_origins=["*"]` + `allow_credentials=True` by echoing
# the *request's* origin, so with the token in a cookie any page you happened to
# be visiting could read this API and run tools. The SPA is same-origin (Vite's
# dev server proxies, so it is same-origin too), so nothing legitimate needs a
# wider list; the regex covers whatever port either of them is on.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(auth.token_middleware)


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


# ============== Static files + SPA ==============
# Production: Vite writes into gateway/ui/. pip users get that dist; Node is
# not required at runtime. Dev: `cd web && npm run dev` proxies /api to here.

_UI_DIR = Path(__file__).parent / "ui"
_UI_ASSETS = _UI_DIR / "assets"


def _spa_index() -> HTMLResponse:
    index = _UI_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<!doctype html><title>Agentica</title><p>Web UI is not built. "
            "From the repo: <code>cd web && npm install && npm run build</code>.</p>",
            status_code=503,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        content=index.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if _UI_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_UI_ASSETS), name="ui-assets")


@app.get("/chat", response_class=HTMLResponse)
@app.get("/chat/{full_path:path}", response_class=HTMLResponse)
@app.get("/traces", response_class=HTMLResponse)
@app.get("/traces/{full_path:path}", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def web_spa(full_path: str = ""):
    """Serve the SPA shell. Client router owns /chat, /traces and /login.

    ``/login`` is outside the token gate (see ``auth._OPEN_PATHS``) — it is the
    one page a signed-out browser must be able to load.
    """
    return _spa_index()


# ============== Desktop shell control ==============
# Set by main() so the shutdown route can ask uvicorn to wind down. None under
# a TestClient (no server object), which is exactly when the route should
# decline rather than kill the test runner.
_server: Optional["uvicorn.Server"] = None


@app.post("/api/desktop/shutdown")
async def desktop_shutdown(request: Request):
    """Wind down gracefully at the supervising shell's request.

    POSIX has SIGTERM, so this exists for Windows, where killing a child is a
    hard ``TerminateProcess``: the channels would never disconnect and the
    peers record would never be unpublished. The shell asks here first and
    kills only if the process is still up afterwards.

    Restricted to the machine token, not any session: a browser tab must not be
    able to stop the process behind it, and the shell holds the token already.
    """
    presented, _ = auth.machine_token(request)
    if auth.auth_enabled() and not auth.token_is_valid(presented):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if _server is None:
        return JSONResponse(
            {"error": "unavailable", "detail": "Not running under uvicorn."},
            status_code=503,
        )
    _server.should_exit = True
    return {"status": "stopping"}


# ============== Route registration ==============

app.include_router(auth_routes.router)
app.include_router(settings_routes.router)
app.include_router(chat.router)
app.include_router(traces_routes.router)
app.include_router(scheduler_routes.router)
app.include_router(plugins_routes.router)
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
    from .channels.slack import SlackChannel

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

    # Slack: enabled when both bot token and app-level (Socket Mode) token
    # are provided.
    if settings.slack_bot_token and settings.slack_app_token:
        try:
            slack = SlackChannel(
                bot_token=settings.slack_bot_token,
                app_token=settings.slack_app_token,
                allowed_users=settings.slack_allowed_users,
                allowed_channels=settings.slack_allowed_channels,
            )
            deps.channel_manager.register(slack)
        except Exception as e:
            logger.error(f"Failed to create Slack channel: {e}")

    deps.channel_manager.set_handler(_handle_channel_message)
    await deps.channel_manager.connect_all()


# Per-session FIFO queues for inbound channel messages. IM users (WeChat,
# etc.) routinely fire several messages in quick succession before the agent
# has replied; the session run-lock would reject the second message
# ("already has an active run") and it would be lost. Each session gets its
# own queue drained by a single worker task, so messages are answered in
# order, one at a time, instead of colliding on the lock.
_channel_queues: dict[str, asyncio.Queue] = {}
_channel_workers: dict[str, asyncio.Task] = {}
_channel_queue_lock = asyncio.Lock()

# Cap pending messages per session so a spamming user can't grow the queue
# without bound; messages beyond this are dropped with a warning.
_MAX_CHANNEL_QUEUE = 20


async def _handle_channel_message(message) -> None:
    """Enqueue an inbound channel message for serialized, in-order handling.

    Returns immediately after queuing. A per-session worker task drains the
    queue and processes one message at a time via ``_process_channel_message``,
    so rapid-fire messages from the same user never hit the session run-lock.
    """
    logger.info(f"[{message.channel.value}] {message.sender_id}: {message.content[:500]}")

    # Relaying to a CLI session runs no agent here, so it skips the queue: a
    # `@session stop` typed while the gateway's own agent is mid-turn must not
    # wait behind that turn — being able to interrupt is the whole point of it.
    if deps.peer_bridge is not None and await deps.peer_bridge.handle(message):
        return

    if not deps.agent_service:
        logger.error("Agent service not ready")
        return

    agent_id = deps.message_router.route(message)
    session_id = deps.message_router.get_session_id(message, agent_id)

    # Where a CLI session's answer to this chat session should be pushed. Noted
    # here, on the one path that knows both the session id and the conversation
    # it belongs to, rather than recovered later by splitting the session id.
    if deps.agent_peers is not None:
        deps.agent_peers.note_route(session_id, message.channel, message.channel_id)

    async with _channel_queue_lock:
        queue = _channel_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            _channel_queues[session_id] = queue
        if queue.qsize() >= _MAX_CHANNEL_QUEUE:
            logger.warning(
                f"Channel queue full for session {session_id} "
                f"({queue.qsize()} pending); dropping message"
            )
            return
        queue.put_nowait(message)
        worker = _channel_workers.get(session_id)
        if worker is None or worker.done():
            _channel_workers[session_id] = asyncio.create_task(
                _channel_queue_worker(session_id, queue)
            )


async def _channel_queue_worker(session_id: str, queue: asyncio.Queue) -> None:
    """Drain one session's message queue, processing messages sequentially.

    Exits (and removes itself from the registries) once the queue is empty.
    The empty-check and teardown happen under ``_channel_queue_lock`` so they
    are atomic with respect to ``_handle_channel_message`` enqueuing — a
    message added just as the worker is about to exit is never stranded.
    """
    while True:
        message = await queue.get()
        try:
            await _process_channel_message(message, session_id)
        finally:
            queue.task_done()

        async with _channel_queue_lock:
            if queue.empty():
                _channel_queues.pop(session_id, None)
                _channel_workers.pop(session_id, None)
                return


async def _process_channel_message(message, session_id: str) -> None:
    """Route a single channel message through the agent and reply."""
    user_id = message.sender_id or settings.default_user_id

    try:
        # Materialise media references (image/voice/video) the channel put
        # into metadata — only channels with media support return anything.
        media = []
        wanted = message.metadata.get("media") or []
        if wanted:
            channel = deps.channel_manager.get_channel(message.channel)
            if channel is not None:
                media = await channel.fetch_media(message)

        text = (message.content or "").strip()
        if not text and not media:
            # A media-only inbound whose download/decrypt failed must not
            # become agent.chat(""): Claude and compatible proxies reject empty user content
            # (``messages.N: user messages must have non-empty content``)
            # and the empty turn poisons the session history.
            if wanted:
                await deps.channel_manager.send(
                    message.channel,
                    message.channel_id,
                    "没能下载这条图片/语音，请再发一次，或配一句文字。",
                )
            return
        if not text:
            kind = media[0].kind
            text = {
                "image": "请看这条图片。",
                "voice": "请听这条语音。",
                "video": "请看这条视频。",
            }.get(kind, "请查看这条媒体。")

        result = await deps.agent_service.chat(
            message=text,
            session_id=session_id,
            user_id=user_id,
            media=media,
        )

        # Media notes (non-base model used / media skipped) prefix the reply
        # so the IM user sees how their image/voice/video was handled.
        reply = result.content
        if result.media_notes:
            notes = "\n".join(result.media_notes)
            reply = f"{notes}\n\n{reply}" if reply else notes

        if reply:
            await deps.channel_manager.send(
                message.channel,
                message.channel_id,
                reply,
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
    """Adapts AgentService to the AgentRunner protocol expected by JobExecutor.

    Runs each job on its own independent Agent (see
    AgentService.run_cron()) — never the shared interactive-chat cache — so
    scheduled jobs neither leak context between runs nor show up in the chat
    sidebar.
    """

    def __init__(self, agent_svc: AgentService):
        self._svc = agent_svc

    async def run(self, prompt: str, context: Optional[dict] = None) -> str:
        ctx = context or {}
        job_id = ctx.get("job_id", str(uuid4()))
        user_id = ctx.get("user_id", settings.default_user_id)

        result = await self._svc.run_cron(message=prompt, job_id=job_id, user_id=user_id)
        return result.content


# ============== Entry point ==============

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentica-gateway",
        description="Serve the agentica Web UI and HTTP/WebSocket API.",
    )
    parser.add_argument(
        "--host", default=settings.host,
        help="Bind address (default %(default)s; use 0.0.0.0 to accept from the LAN)",
    )
    parser.add_argument(
        "--port", type=int, default=settings.port,
        help="Port to bind; 0 picks a free one and reports it (default %(default)s)",
    )
    parser.add_argument(
        "--parent-pid", type=int, default=settings.parent_pid,
        help="Exit when this pid is gone. A desktop shell passes its own pid so "
             "killing the shell cannot leave the gateway running.",
    )
    parser.add_argument(
        "--set-password", action="store_true",
        help="Prompt for a web password and exit. Required before binding "
             "anything other than loopback.",
    )
    return parser.parse_args(argv)


def _bind(host: str, port: int) -> socket.socket:
    """Bind the listening socket here, before uvicorn.

    ``--port 0`` has to be resolved by whoever holds the socket: asking the OS
    for a free port and then handing the *number* to uvicorn re-opens the race
    the zero was there to avoid — something else can take it in between. So we
    bind, read the port back, and give uvicorn the socket.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


def _refuse_open_bind(host: str) -> Optional[str]:
    """Why this host must not be served, or None.

    Binding past loopback puts ``execute`` on the network. The token alone is
    not enough there: it is printed in a terminal and mailed around in a
    ``runtime.json`` path, and there is no way to change it without a restart.
    A password is the credential a person can rotate, so it is the price of
    admission — and refusing at startup is the only moment anybody is watching.

    ``GATEWAY_AUTH=false`` is not caught here on purpose: turning the gate off
    is an explicit instruction, and an explicit argument is intent. It gets a
    loud warning instead (see ``lifespan``).
    """
    if host in LOOPBACK_HOSTS or not auth.auth_enabled():
        return None
    if accounts.store().has_password():
        return None
    return (
        f"refusing to serve {host}: that is reachable from the network and no "
        "web password is set. Run `agentica-gateway --set-password` first, or "
        "bind 127.0.0.1."
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Start the gateway server."""
    import uvicorn

    args = _parse_args(argv)
    if args.set_password:
        raise SystemExit(accounts.set_password_interactive(settings.default_user_id))

    settings.host = args.host
    settings.parent_pid = args.parent_pid

    refusal = _refuse_open_bind(args.host)
    if refusal:
        print(f"agentica-gateway: {refusal}", file=sys.stderr)
        raise SystemExit(2)

    try:
        sock = _bind(args.host, args.port)
    except OSError as e:
        # The common one is "port already in use", and the useful next line is
        # not a traceback — it is which gateway already has it.
        existing = runtime.read()
        hint = ""
        if existing is not None and existing.port == args.port and runtime.is_pid_alive(existing.pid):
            hint = f" — pid {existing.pid} is already serving {existing.url}"
        print(f"agentica-gateway: cannot bind {args.host}:{args.port}: {e}{hint}", file=sys.stderr)
        raise SystemExit(1)

    # The real port, which with --port 0 is only knowable now. lifespan reads
    # settings.port to publish the runtime record, so it must be set first.
    settings.port = sock.getsockname()[1]

    # The app *object*, not "agentica.gateway.main:app". Under `python -m` the
    # import string makes uvicorn load this file a second time, under its real
    # name — so `main()` would set `_server` on the `__main__` copy while the
    # shutdown route ran in the other one and answered "not under uvicorn".
    # Passing the object also means the module is only executed once.
    global _server
    config = uvicorn.Config(app, reload=False)
    _server = uvicorn.Server(config)
    _server.run(sockets=[sock])


if __name__ == "__main__":
    main()
