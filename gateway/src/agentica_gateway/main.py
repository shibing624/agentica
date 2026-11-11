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
import time
import unicodedata
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import parse_qs
from uuid import uuid4

if TYPE_CHECKING:
    import uvicorn

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from agentica.config import AGENTICA_LOG_LEVEL
from agentica.shell_hooks import install_hook_egress, load_shell_hooks_config
from agentica.utils.log import enable_process_file_logging, logger
from . import accounts, auth, deps, runtime
from .card_page import (
    LINK_LIFETIME_SECONDS,
    PAGE_HEADERS,
    CardRegistry,
    card_page,
    words_page,
)
from agentica.version import __version__
from .config import settings
from .services.agent_service import AgentService
from .services.scheduler_service import GatewayCronRunner, GatewayScheduler
from .services.channel_manager import ChannelManager
from .services.im_bot import ImBotController
from .services.router import MessageRouter
from .routes import auth as auth_routes, chat, settings as settings_routes, scheduler as scheduler_routes, channels, ws, plugins as plugins_routes, traces as traces_routes, workspace as workspace_routes, terminals as terminal_routes, git as git_routes
from .terminals import manager as terminal_manager

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


def _install_gateway_hooks() -> None:
    """Wire ``settings.hooks`` here only when ``settings.hooks.gateway`` is on.

    The gateway is the same personal agent as the CLI behind another surface
    (web UI, desktop app, IM). But IM channels carry other people's messages:
    with hooks on, a recall consumer answers *them* from your memory and a
    capture consumer files *their* chats as yours. So it is opt-in.
    """
    cfg = load_shell_hooks_config()
    if not cfg.gateway:
        logger.info("Shell hooks: off for the gateway (set `settings.hooks.gateway: true` to enable)")
        return
    installed = install_hook_egress(cfg)
    names = [c.name for c in installed.consumers if c.enabled] if installed else []
    logger.info(f"Shell hooks: {', '.join(names) or 'none active'}")


def _sign_in_notice(user_id: str, password: str) -> str:
    """The boxed "here is your password" block for the startup log.

    A box because this is the one line in a noisy startup log that the user has
    to act on, and it competes with model-loading and channel chatter. Reprinted
    on every boot while the password is still the generated one — the previous
    design printed a per-process token URL, so anybody who scrolled past it, or
    whose gateway was started by launchd or the desktop shell, had no way back
    in short of resetting the password.
    """
    lines = [
        f"Web sign-in:  {user_id} / {password}",
        "",
        "This is the generated password and has not been changed yet.",
        "Change it in the web UI (设置 › 常规 › 访问) or with",
        "`agentica-gateway --set-password`.",
    ]
    # Padded by display width, not character count: the box has a Chinese line
    # in it and a CJK glyph occupies two terminal columns, so ljust() would
    # leave the right edge of the frame ragged.
    def cols(text: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

    width = max(cols(line) for line in lines)
    bar = "+" + "-" * (width + 4) + "+"
    body = [f"|  {line}{' ' * (width - cols(line))}  |" for line in lines]
    return "\n".join([bar, *body, bar])


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

    _install_gateway_hooks()

    # Agent service
    agent_svc = AgentService(workspace_path=str(settings.workspace_path))
    # Eagerly initialize so startup errors surface immediately (fail fast)
    try:
        await agent_svc._ensure_initialized()
    except RuntimeError as e:
        logger.error(f"FATAL: {e}")
        raise

    deps.agent_service = agent_svc

    agent_svc.background_report_handler = _background_report_handler(agent_svc)

    # Channel manager + message router
    deps.channel_manager = ChannelManager()
    deps.message_router = MessageRouter(default_agent="main")

    # Scheduled tasks — the same SDK cron store (agentica.cron.*) as the CLI,
    # never the OS crontab. Each run is delivered into a conversation (see
    # services/scheduler_service.py). On/off is the same `cron.enabled`
    # setting the CLI's `/cron daemon on` toggles; the web page flips it live.
    from agentica.config.profiles import get_setting

    async def _deliver_im(owner, session_id, text, meta):
        if deps.im_bot is None:
            return None
        return await deps.im_bot.deliver_scheduled_run(owner, session_id, text, meta)

    deps.scheduler = GatewayScheduler(GatewayCronRunner(agent_svc, deliver_im=_deliver_im))
    deps.cron_runner = deps.scheduler.runner
    deps.main_loop = asyncio.get_running_loop()
    if get_setting("cron.enabled", False):
        deps.scheduler.start()
    else:
        logger.info("Cron scheduler disabled (turn it on from the Automations page, or set `cron.enabled: true`)")

    # Channels (IM integrations: WeChat / WeCom / Feishu / QQ / ... ). One
    # gateway per data root owns them; a standby takes over when it exits.
    im_task = None
    if not await _claim_and_start_im():
        im_task = asyncio.create_task(_take_over_im_when_free())

    deps.card_registry = CardRegistry(secret_provider=auth.get_token)
    deps.im_bot = ImBotController(
        channel_manager=deps.channel_manager,
        agent_service=deps.agent_service,
        card_registry=deps.card_registry,
        user_id_provider=_home_account,
    )
    logger.info("IM bot controller started (/menu, /resume, /switch, /new, /cancel, /c/<token>)")

    # Publish how to reach this process (port + token) so a desktop shell or a
    # second terminal can find it.
    record = runtime.GatewayRuntime(
        pid=os.getpid(),
        host=settings.host,
        port=settings.port,
        token=auth.get_token() if auth.auth_enabled() else "",
        version=__version__,
    )
    runtime_path = runtime.publish(record)
    logger.info(f"Web service started — {record.url}/chat")
    if not auth.auth_enabled():
        logger.warning(
            "  API auth is OFF (GATEWAY_AUTH=false) — anything that can reach "
            f"{record.url} can run tools as you"
        )
    else:
        # Seeding here, not in `main()`: the CLI path is not the only way this
        # app gets served (uvicorn/gunicorn point at `main:app` too), and an
        # account that exists only when started one particular way is an
        # account that is missing exactly when somebody is locked out.
        seeded = accounts.store().seed_default_account()
        if accounts.store().password_is_initial():
            password = seeded or accounts.store().read_initial_password()
            if password:
                logger.info("\n" + _sign_in_notice(accounts.default_account_id(), password))
            else:
                # The 0600 copy is gone but the flag says it was never changed:
                # nobody can tell the user what to type, so say that instead of
                # printing a box with a blank in it.
                logger.warning(
                    "  The generated password was removed from "
                    f"{_display_home_path(str(accounts.store().initial_password_path))} "
                    "before it was changed — run `agentica-gateway --set-password`"
                )
        else:
            logger.info(f"  Sign in as {accounts.default_account_id()} with your web password")
        logger.info(f"  Runtime (port + token): {_display_home_path(str(runtime_path))}")

    # A desktop shell owns this process; if the shell is killed outright there
    # is nobody left to stop us, and an orphan gateway keeps holding the port
    # and the session locks. Polling is the only check that survives SIGKILL of
    # the parent (an atexit handler over there would never run).
    parent_task = None
    if settings.parent_pid:
        parent_task = asyncio.create_task(_exit_with_parent(settings.parent_pid))

    yield

    # Shutdown
    logger.info("Shutting down...")
    runtime.unpublish(os.getpid())
    if parent_task is not None:
        parent_task.cancel()
    if im_task is not None:
        im_task.cancel()
    await deps.scheduler.shutdown()
    if deps.im_bot:
        await deps.im_bot.stop()
    if deps.channel_manager:
        await deps.channel_manager.disconnect_all()
    terminal_manager.close_all()
    await asyncio.to_thread(deps.agent_service.stop_all_background_jobs)
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
# `web/` builds into web/dist; the wheel carries a copy at agentica_gateway/ui
# (gateway/hatch_build.py), so pip users never need Node. See ui_dist for the
# lookup order. Dev: `cd web && npm run dev` proxies /api here.

from agentica_gateway.ui_dist import ENV_VAR as _UI_ENV, resolve_ui_dir  # noqa: E402

_UI_DIR = resolve_ui_dir() or Path(__file__).parent / "ui"
_UI_ASSETS = _UI_DIR / "assets"


def _spa_index() -> HTMLResponse:
    index = _UI_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<!doctype html><title>Agentica</title><p>Web UI is not built. "
            "From the repo: <code>cd web && npm install && npm run build</code> "
            f"(writes <code>web/dist</code>), or point <code>{_UI_ENV}</code> at a built dist.</p>",
            status_code=503,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return HTMLResponse(
        content=index.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if _UI_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_UI_ASSETS), name="ui-assets")


@app.get("/favicon.ico")
@app.get("/favicon.png")
async def web_favicon(request: Request):
    """Tab icon for /chat and /login. Lives at the UI root, not under /assets."""
    name = "favicon.ico" if request.url.path.endswith(".ico") else "favicon.png"
    path = _UI_DIR / name
    if not path.is_file():
        return Response(status_code=404)
    media = "image/x-icon" if name.endswith(".ico") else "image/png"
    return FileResponse(path, media_type=media)


@app.get("/c/{token}")
async def get_card_page(token: str) -> Response:
    """Serve the zero-JS capability card page (/c/<token>) for mobile webviews."""
    registry = deps.card_registry
    card = registry.get_by_token(token) if registry is not None else None
    if card is None:
        return Response(
            content=words_page("没有这张卡片"),
            status_code=404,
            headers=PAGE_HEADERS,
        )
    if time.time() - card.created_at > LINK_LIFETIME_SECONDS:
        return Response(
            content=words_page("这个链接已经过期了"),
            status_code=401,
            headers=PAGE_HEADERS,
        )
    return Response(content=card_page(card), status_code=200, headers=PAGE_HEADERS)


@app.post("/c/{token}")
async def post_card_page(token: str, request: Request) -> Response:
    """Accept a form submission on /c/<token> and redirect (303 PRG)."""
    registry = deps.card_registry
    card = registry.get_by_token(token) if registry is not None else None
    if card is None:
        return Response(
            content=words_page("没有这张卡片"),
            status_code=404,
            headers=PAGE_HEADERS,
        )
    if time.time() - card.created_at > LINK_LIFETIME_SECONDS:
        return Response(
            content=words_page("这个链接已经过期了"),
            status_code=401,
            headers=PAGE_HEADERS,
        )
    raw_body = (await request.body()).decode("utf-8", errors="replace")
    form = parse_qs(raw_body, keep_blank_values=True)
    choice_vals = form.get("choice")
    choice = (
        int(choice_vals[0])
        if choice_vals and choice_vals[0].strip().lstrip("-").isdigit()
        else None
    )
    text_vals = form.get("text")
    text = text_vals[0] if text_vals else None
    registry.answer(token, choice=choice, text=text, by="链接")
    return Response(
        status_code=303,
        headers={"Location": f"/c/{token}", "Cache-Control": "no-store"},
    )


@app.post("/api/cards/hook")
async def post_card_hook(request: Request) -> JSONResponse:
    """Receive a CLI shell_hook event (needs.approval / needs.input / request.resolved)."""
    if deps.im_bot is None:
        return JSONResponse({"ok": False, "error": "im_bot not initialized"}, status_code=503)
    payload = await request.json()
    wait_param = request.query_params.get("wait") or payload.get("wait") or 0
    try:
        wait_s = min(max(float(wait_param), 0.0), 300.0)
    except (TypeError, ValueError):
        wait_s = 0.0
    result = await deps.im_bot.handle_hook_event(payload, wait_seconds=wait_s)
    return JSONResponse(result)


@app.get("/chat", response_class=HTMLResponse)
@app.get("/chat/{full_path:path}", response_class=HTMLResponse)
@app.get("/traces", response_class=HTMLResponse)
@app.get("/traces/{full_path:path}", response_class=HTMLResponse)
@app.get("/automations", response_class=HTMLResponse)
@app.get("/automations/{full_path:path}", response_class=HTMLResponse)
@app.get("/users", response_class=HTMLResponse)
@app.get("/users/{full_path:path}", response_class=HTMLResponse)
@app.get("/assistant", response_class=HTMLResponse)
@app.get("/assistant/{full_path:path}", response_class=HTMLResponse)
@app.get("/plugins", response_class=HTMLResponse)
@app.get("/plugins/{full_path:path}", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
async def web_spa(full_path: str = ""):
    """Serve the SPA shell. Client router owns /chat, /traces, /automations, /users,
    /assistant, /plugins and /login.

    ``/login`` is outside the token gate (see ``auth._OPEN_PATHS``) — it is the
    one page a signed-out browser must be able to load.
    """
    return _spa_index()


# ============== Desktop shell control ==============
# Set by main() so the shutdown route can ask uvicorn to wind down. None under
# a TestClient (no server object), which is exactly when the route should
# decline rather than kill the test runner.
_server: Optional["uvicorn.Server"] = None
_restart_requested: bool = False


def request_restart() -> bool:
    """Schedule a clean uvicorn exit followed by an in-place re-exec on the same port."""
    global _restart_requested
    if _server is None:
        return False
    _restart_requested = True
    _server.should_exit = True
    return True


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
app.include_router(workspace_routes.router)
app.include_router(terminal_routes.router)
app.include_router(git_routes.router)
app.include_router(traces_routes.router)
app.include_router(scheduler_routes.router)
app.include_router(plugins_routes.router)
app.include_router(channels.router)
app.include_router(ws.router)


# ============== Channel setup ==============

IM_TAKEOVER_POLL_S = 5.0


def _background_report_handler(agent_svc: AgentService):
    """Where a finished background command's report runs: the IM chat on its
    session (the user is on the phone), else a live web run any open page
    attaches to."""

    async def deliver(session_id: str, owner: str, report: str) -> None:
        if deps.im_bot is not None and await deps.im_bot.deliver_background_report(owner, session_id, report):
            return
        chat.start_background_report_run(agent_svc, session_id, owner, report)

    return deliver


async def _claim_and_start_im() -> bool:
    """Connect IM if this gateway may; see runtime.claim_im_channels."""
    if not runtime.claim_im_channels():
        logger.warning(
            f"IM channels are served by another gateway on this data root "
            f"(pid {runtime.im_lock_holder()}); this one will take over if it exits"
        )
        return False
    await _setup_channels()
    if deps.channel_manager.channels:
        enabled = ", ".join(c.value for c in deps.channel_manager.channels)
        logger.info(f"IM channels started — {enabled}")
    return True


async def _take_over_im_when_free() -> None:
    """A standby gateway serves IM once the owner is gone, without a restart."""
    while not runtime.claim_im_channels():
        await asyncio.sleep(IM_TAKEOVER_POLL_S)
    logger.info("IM owner gateway exited; this gateway takes over IM")
    await _setup_channels()


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

    # Personal WeChat: always registered. A cached token reconnects
    # silently; QR login is started from the Personal Assistant page, not
    # at process start (that used to pop a scan window on every boot).
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


# Inbound IM is full duplex. While this session's worker has a run in flight,
# a follow-up text line is steered into that run (``agent.steer``: delivered at
# the next tool-batch boundary, same as typing mid-run in the CLI) instead of
# waiting for the whole turn to finish. What cannot be steered — media, lines
# that arrive before the run opens its steer window, lines behind already
# queued ones — goes on a per-session FIFO drained by a single worker task.
# The worker takes everything pending as one batch, so N lines sent while the
# agent was busy become one turn and one reply, not N turns answering stale
# context. Without the queue the session run-lock would reject the second
# message ("already has an active run") and it would be lost.
_channel_queues: dict[str, asyncio.Queue] = {}
_channel_workers: dict[str, asyncio.Task] = {}
_channel_queue_lock = asyncio.Lock()

# Cap pending messages per session so a spamming user can't grow the queue
# without bound; messages beyond this are dropped with a warning.
_MAX_CHANNEL_QUEUE = 20


async def _enqueue_session_message(session_id: str, message) -> None:
    """Put one inbound line on this session's FIFO; start a worker if needed."""
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


def _home_account() -> str:
    """The one person this personal-assistant gateway belongs to.

    Web, desktop and IM share this partition (sessions, memory, AGENTS.md).
    Whoever scanned the WeChat QR while signed in is that person; until then
    it is ``settings.default_user_id``.
    """
    from .channels.base import ChannelType
    from .channels.wechat import WeChatChannel

    cm = deps.channel_manager
    if cm is not None:
        ch = cm.get_channel(ChannelType.WECHAT)
        if isinstance(ch, WeChatChannel) and ch.gateway_user_id:
            return ch.gateway_user_id
    return settings.default_user_id


async def _handle_channel_message(message) -> None:
    """Route an inbound IM message through ImBotController, or the fallback queue."""
    logger.info(f"[{message.channel.value}] {message.sender_id}: {message.content[:500]}")

    if deps.im_bot is not None and await deps.im_bot.handle(message):
        return

    if not deps.agent_service:
        logger.error("Agent service not ready")
        return

    agent_id = deps.message_router.route(message)
    session_id = deps.message_router.get_session_id(message, agent_id)

    if await _steer_running_turn(session_id, message):
        return
    await _enqueue_session_message(session_id, message)


async def _steer_running_turn(session_id: str, message) -> bool:
    """Hand a follow-up line to this session's in-flight run. True if taken.

    Only a run this channel worker started is steered: its reply goes back to
    this conversation, and the worker pops whatever is accepted too late for
    the run (``take_undelivered_steer``) and runs it as the next turn. Lines
    already waiting in the queue go first, so a line never overtakes them.
    Media needs the per-turn media routing in ``chat()``, so it is queued.
    False also covers the run ending between this check and ``steer()``.
    """
    if message.metadata.get("media"):
        return False
    text = (message.content or "").strip()
    if not text:
        return False
    async with _channel_queue_lock:
        worker = _channel_workers.get(session_id)
        if worker is None or worker.done():
            return False
        queue = _channel_queues.get(session_id)
        if queue is not None and not queue.empty():
            return False
        accepted = deps.agent_service.steer_session(session_id, text, owner=_home_account())
    if accepted:
        logger.info(f"[{message.channel.value}] steered into running turn of {session_id}")
    return accepted


async def _channel_queue_worker(session_id: str, queue: asyncio.Queue) -> None:
    """Drain one session's message queue, one batch per turn.

    Each turn takes every message pending at that moment. After the turn,
    lines that were steered too late for it (accepted during the final
    inference) run next, ahead of anything queued since: they were sent
    first.

    Exits (and removes itself from the registries) once the queue is empty.
    The empty-check and teardown happen under ``_channel_queue_lock`` so they
    are atomic with respect to ``_handle_channel_message`` enqueuing — a
    message added just as the worker is about to exit is never stranded.
    """
    carried: list = []
    while True:
        taken = [] if carried else [await queue.get()]
        while not queue.empty():
            taken.append(queue.get_nowait())
        batch, carried = carried + taken, []
        try:
            await _process_channel_batch(batch, session_id)
        finally:
            for _ in taken:
                queue.task_done()

        late = deps.agent_service.take_undelivered_steer(session_id, owner=_home_account())
        if late:
            carried = [replace(batch[-1], content="\n\n".join(late), metadata={}, attachments=[])]
            continue

        async with _channel_queue_lock:
            if queue.empty():
                _channel_queues.pop(session_id, None)
                _channel_workers.pop(session_id, None)
                return


async def _process_channel_batch(messages: list, session_id: str) -> None:
    """Run one turn for these messages (same conversation, in order) and reply.

    Texts are joined, media from every message is attached. The reply goes to
    the last message's conversation.
    """
    user_id = _home_account()
    last = messages[-1]

    try:
        parts: list[str] = []
        media = []
        for message in messages:
            # Materialise media references (image/voice/video) the channel put
            # into metadata — only channels with media support return anything.
            fetched = []
            wanted = message.metadata.get("media") or []
            if wanted:
                channel = deps.channel_manager.get_channel(message.channel)
                if channel is not None:
                    fetched = await channel.fetch_media(message)
            line = (message.content or "").strip()
            if not line and not fetched:
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
                continue
            if line:
                # Per line: a ``/skill`` sent second in a batch is still a skill call.
                parts.append(AgentService.expand_skill_invocation(line))
            media.extend(fetched)

        if not parts and not media:
            return
        # IM has no approval card, so it follows the tier the account picked
        # in the web selector; without a pick it stays on the gateway default.
        mode = settings_routes.approval_mode_pref(user_id)
        if mode:
            deps.agent_service.set_session_approval_mode(session_id, mode, owner=user_id)
        text = "\n\n".join(parts)
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
            owner=user_id,
            media=media,
            channel=last.channel.value,
        )

        # Media notes (non-base model used / media skipped) prefix the reply
        # so the IM user sees how their image/voice/video was handled.
        reply = result.content
        if result.media_notes:
            notes = "\n".join(result.media_notes)
            reply = f"{notes}\n\n{reply}" if reply else notes

        if reply:
            await deps.channel_manager.send(
                last.channel,
                last.channel_id,
                reply,
            )

        await ws.ws_manager.broadcast("channel.message", {
            "channel": last.channel.value,
            "sender": last.sender_id,
            "userId": user_id,
            # The lines as sent: ``text`` may hold a whole expanded skill prompt.
            "content": "\n\n".join(m.content or "" for m in messages).strip()[:100],
            "response": result.content[:100] if result.content else "",
        })

    except Exception as e:
        logger.error(f"Handle channel message error: {e}")
        try:
            await deps.channel_manager.send(
                last.channel,
                last.channel_id,
                "error processing message",
            )
        except Exception:
            pass


# ============== Scheduler agent runner ==============

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

    Binding past loopback puts ``execute`` on the network, and the price of
    admission is a password *the user chose*. The generated one does not count:
    it is printed to a log and kept in plaintext next to ``auth.json`` so that a
    locked-out owner can still get in on loopback — both fine for a machine you
    are sitting at, and neither true of a credential facing a network. So this
    refuses a fresh install as well as a seeded-but-unchanged one, and startup
    is the only moment anybody is watching.

    ``GATEWAY_AUTH=false`` is not caught here on purpose: turning the gate off
    is an explicit instruction, and an explicit argument is intent. It gets a
    loud warning instead (see ``lifespan``).
    """
    if host in LOOPBACK_HOSTS or not auth.auth_enabled():
        return None
    store = accounts.store()
    if store.has_password() and not store.password_is_initial():
        return None
    return (
        f"refusing to serve {host}: that is reachable from the network and the "
        "web password is still the generated one. Run "
        "`agentica-gateway --set-password` first, or bind 127.0.0.1."
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Start the gateway server."""
    import uvicorn

    args = _parse_args(argv)
    if args.set_password:
        raise SystemExit(accounts.set_password_interactive())

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

    # The app *object*, not "agentica_gateway.main:app". Under `python -m` the
    # import string makes uvicorn load this file a second time, under its real
    # name — so `main()` would set `_server` on the `__main__` copy while the
    # shutdown route ran in the other one and answered "not under uvicorn".
    # Passing the object also means the module is only executed once.
    global _server, _restart_requested
    _restart_requested = False
    config = uvicorn.Config(app, reload=False)
    _server = uvicorn.Server(config)
    _server.run(sockets=[sock])
    try:
        sock.close()
    except OSError:
        pass
    if _restart_requested:
        restart_argv = [
            sys.executable, "-m", "agentica_gateway.main",
            "--host", str(settings.host),
            "--port", str(settings.port),
        ]
        if settings.parent_pid:
            restart_argv += ["--parent-pid", str(settings.parent_pid)]
        os.execv(sys.executable, restart_argv)


if __name__ == "__main__":
    main()
