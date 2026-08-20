# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Who gets to talk to the gateway's API.

Why this exists at all: ``/api`` can switch profiles, read any path the agent
can reach and run ``execute``. Before the gate the whole surface was open to
anything that could open a TCP connection to the port, which on the previous
``0.0.0.0`` default meant every device on the same Wi-Fi.

There are exactly two credentials, and keeping them apart is the design:

- **The machine token.** Per process and random (``AGENTICA_GATEWAY_TOKEN``
  pins it). It proves "I can read a 0600 file on this machine", which is the
  same thing as "I could have run the agent myself". A desktop shell reads it
  out of ``runtime.json`` and redeems it for a session before the first
  navigation, so it never prompts. Header forms (``Authorization: Bearer``,
  ``X-Agentica-Token``) are for scripts.
- **A session.** Issued by ``accounts.py`` when somebody redeems the machine
  token or types the password, written down, and carried in an HttpOnly cookie.

The cookie used to hold the machine token itself. That was wrong in both
directions: a leaked cookie *was* the master credential with no way to revoke
one browser, and because the token is per process, every gateway restart sent
the user back to the terminal to copy a new URL. A session survives restarts
and can be closed on its own.

A browser is never asked for the token. It used to be — the banner printed
``/chat?token=…`` and that was the only way in on a fresh install — which read
as a second, worse login: the credential changed on every restart, it was
useless to anyone whose gateway was started detached or by the desktop shell,
and it made the browser and the desktop app behave differently for no reason
the user could see. Now first start seeds a password for the ``default``
account (``accounts.seed_default_account``) and browsers go to ``/login``. The
query form still works, because a script that has the token should not need a
second credential.

What is deliberately *not* behind the gate, and why each one has to be open:

- ``/webhook/*`` — third-party callbacks (Feishu). They authenticate with their
  own signature and cannot be taught our credentials, so gating them just
  breaks IM.
- ``/health``, ``/api/health``, ``/`` — liveness probes. A desktop shell polls
  one of these *before* it knows the token, which is the whole point of them.
- ``/assets/*`` — the compiled SPA. No user data, and serving a login page that
  cannot load its own stylesheet is worse than serving the stylesheet.
- ``/favicon.ico``, ``/favicon.png`` — the tab icon, requested from the origin
  root (including on ``/login`` before a session exists).
- ``/login`` and ``/api/auth/{status,login,logout}`` — the way in. A gate over
  the door is a locked room.
"""
import os
import secrets
from dataclasses import dataclass
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.responses import Response

from . import accounts

SESSION_COOKIE = "agentica_session"
HEADER_NAME = "X-Agentica-Token"
QUERY_NAME = "token"

# The header that makes a cross-site form POST impossible. See
# `_csrf_ok` for why one custom header is enough.
CLIENT_HEADER = "X-Agentica-Client"

# Prefixes served without a credential. Matched as prefixes, so
# `/webhook/feishu` and any future channel callback are covered by the one
# entry.
_OPEN_PREFIXES = ("/webhook/", "/assets/")
_OPEN_PATHS = (
    "/", "/health", "/api/health", "/docs", "/openapi.json", "/redoc",
    "/login", "/api/auth/status", "/api/auth/login", "/api/auth/logout",
    "/favicon.ico", "/favicon.png",
)

# Paths that answer with the SPA shell, so an unauthorized request should get
# somewhere it can act on rather than a JSON envelope no browser will render.
_SHELL_PREFIXES = ("/chat", "/traces")

_token: Optional[str] = None


@dataclass
class Principal:
    """Who the current request is.

    ``user_id`` is both the login and the data partition it owns
    (``users/<id>/`` — see ``accounts.py``), so a route that reads it gets the
    right sessions for free rather than having to map one id to the other.
    """

    user_id: str
    # How they got in: "session" (a signed-in browser), "token" (the machine
    # token presented directly — a script, or the printed URL's first hop),
    # "open" (the gate is off). Only "session" is ambient, which is what the
    # CSRF check and the password rules turn on.
    kind: str
    # "admin" or "user". Gates account management and nothing else: the rest of
    # the gateway configures one machine, and this is that machine's owner's
    # machine. Resolved from the account record on every request, so demoting
    # somebody does not wait for their cookie to expire.
    role: str = accounts.ROLE_USER

    @property
    def is_admin(self) -> bool:
        return self.role == accounts.ROLE_ADMIN


def auth_enabled() -> bool:
    """Whether the gate is on. On unless ``GATEWAY_AUTH`` says otherwise."""
    return os.getenv("GATEWAY_AUTH", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def get_token() -> str:
    """This process's machine token, generated once on first use.

    Generated lazily rather than at import so that a test (or an embedder) can
    set ``AGENTICA_GATEWAY_TOKEN`` after importing the app and still have it
    honoured.
    """
    global _token
    env = os.getenv("AGENTICA_GATEWAY_TOKEN", "").strip()
    if env:
        return env
    if _token is None:
        _token = secrets.token_urlsafe(32)
    return _token


def reset_token_for_tests() -> None:
    """Forget the generated token. Only a test has a reason to call this."""
    global _token
    _token = None


def token_account() -> str:
    """The account a machine-token holder acts as.

    The seeded one, which is also the data partition it owns. A token holder
    can read a 0600 file on this machine, so they are that machine's owner —
    handing them a *different* account would show the desktop shell an empty
    conversation list next to a browser that has them all.
    """
    return accounts.default_account_id()


def _is_open(path: str) -> bool:
    return path in _OPEN_PATHS or path.startswith(_OPEN_PREFIXES)


def token_is_valid(presented: Optional[str]) -> bool:
    if not presented:
        return False
    return secrets.compare_digest(presented, get_token())


def machine_token(request: Request) -> tuple[Optional[str], bool]:
    """The machine token this request carries, and whether it came from the URL.

    Not the cookie: that holds a session now, and a request that presents both
    is a browser whose session is what matters. Where it came from decides
    whether the request gets a session in return — a script presenting a header
    on every poll must not mint one per call.
    """
    from_query = request.query_params.get(QUERY_NAME)
    if from_query:
        return from_query, True

    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip(), False

    direct = request.headers.get(HEADER_NAME)
    if direct:
        return direct.strip(), False
    return None, False


def _csrf_ok(request: Request) -> bool:
    """Whether a write request could have come from a cross-site form.

    ``SameSite=Lax`` already stops the cookie from riding along on a cross-site
    POST, so this is a second line rather than the only one. It works because
    the three Content-Types an HTML form can produce
    (``application/x-www-form-urlencoded``, ``multipart/form-data``,
    ``text/plain``) are exactly the ones a form is *limited* to: a JSON body, or
    any custom header, forces a CORS preflight, which the loopback-only origin
    policy in ``main.py`` refuses for a foreign page.

    ``/api/upload`` genuinely needs multipart, so it carries ``CLIENT_HEADER``
    instead — the header is the part a form cannot forge, not the body type.
    """
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    if request.headers.get(CLIENT_HEADER):
        return True
    content_type = (request.headers.get("content-type") or "").lower()
    if not content_type:
        # No body to smuggle. A form always announces its type.
        return True
    return content_type.startswith("application/json")


def resolve(request: Request) -> tuple[Optional[Principal], bool]:
    """Who this request is, and whether a session cookie should be minted.

    The machine token is checked first: opening the printed URL is a deliberate
    act and must win over a cookie left behind by an earlier gateway.
    """
    presented, from_query = machine_token(request)
    if presented is not None:
        if token_is_valid(presented):
            account = token_account()
            # An admin either way: on a machine whose accounts have not been
            # seeded yet there is no record to read a role from, and the token
            # holder is the one who would seed it.
            return Principal(account, "token", _role(account, accounts.ROLE_ADMIN)), from_query
        return None, False

    session = accounts.store().read_session(request.cookies.get(SESSION_COOKIE))
    if session is not None:
        return Principal(session.user_id, "session", _role(session.user_id)), False
    return None, False


def _role(user_id: str, fallback: str = accounts.ROLE_USER) -> str:
    account = accounts.store().get_account(user_id)
    return account.role if account is not None else fallback


def set_session_cookie(response: Response, token: str) -> None:
    """Hand a session to the browser.

    ``samesite="lax"`` is not cosmetic: the cookie is what makes a plain
    ``/chat`` bookmark work, and without it any page on the internet could have
    a form POST to ``/api/chat`` ride along on it.
    """
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=int(accounts.SESSION_TTL.total_seconds()),
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def _unauthorized(request: Request) -> Response:
    """Refuse, in the shape the caller can act on.

    A browser asking for a page always goes to ``/login`` — there is always an
    account to sign in as, because first start seeds one. The hand-written HTML
    page that used to explain the token here is gone with the flow it described:
    it existed for the window where a gateway had a token and no password, and
    that window no longer opens.
    """
    from .runtime import RUNTIME_FILE

    if request.url.path.startswith(_SHELL_PREFIXES):
        nxt = request.url.path
        if request.url.query:
            nxt = f"{nxt}?{request.url.query}"
        return RedirectResponse(
            f"/login?next={nxt}", status_code=302, headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {
            "error": "unauthorized",
            "detail": (
                "Sign in at /login, or present the local token as "
                f"`Authorization: Bearer <token>`, `{HEADER_NAME}: <token>` or "
                f"`?{QUERY_NAME}=<token>`. The running gateway's token is in "
                f"{RUNTIME_FILE}."
            ),
        },
        status_code=401,
        headers={"Cache-Control": "no-store"},
    )


async def token_middleware(request: Request, call_next):
    """Gate every request that is not on the open list.

    ``OPTIONS`` passes through because a CORS preflight carries no cookie and
    no ``Authorization`` header by specification — gating it would refuse the
    question "may I ask?" and never see the request that carries a credential.
    """
    if not auth_enabled():
        # Still name who the request is: a route behind the gate reads
        # `request.state.principal` unconditionally, and the alternative is
        # every such route guessing at a missing attribute.
        request.state.principal = Principal(token_account(), "open", accounts.ROLE_ADMIN)
        return await call_next(request)

    if request.method == "OPTIONS" or _is_open(request.url.path):
        return await call_next(request)

    principal, mint = resolve(request)
    if principal is None:
        return _unauthorized(request)
    # Only a cookie is ambient, so only a cookie can be ridden. A caller that
    # presented the token in a header proved intent by presenting it, and
    # holding it to a form's Content-Type rules would break `curl -d`.
    if principal.kind == "session" and not _csrf_ok(request):
        return JSONResponse(
            {
                "error": "unsupported_media_type",
                "detail": (
                    "A write must send application/json, or carry the "
                    f"{CLIENT_HEADER} header (which is what /api/upload does "
                    "for its multipart body)."
                ),
            },
            status_code=415,
        )

    request.state.principal = principal
    response = await call_next(request)
    if mint:
        # The first hop from the printed URL: swap the token for a session so
        # the token stops appearing in the address bar and the browser survives
        # the next restart.
        set_session_cookie(
            response, accounts.store().open_session(principal.user_id, "token")
        )
    return response


def websocket_account(websocket) -> str:
    """Which account a ``/ws`` connection acts as.

    A browser arrives with the session cookie; a script presents the machine
    token and is therefore the machine's owner. The connect frame carries no
    account of its own — a socket that could name one would be choosing whose
    conversations to write to.
    """
    session = accounts.store().read_session(websocket.cookies.get(SESSION_COOKIE))
    return session.user_id if session is not None else token_account()


def websocket_token_ok(websocket, handshake_params: Optional[dict] = None) -> bool:
    """Whether a ``/ws`` handshake is authorized.

    The HTTP middleware never sees a websocket scope, so the check lives at the
    endpoint. A browser arrives with the session cookie; a script presents the
    machine token on the URL, in a header, or as ``params.auth.token`` in the
    connect frame — which the protocol has always documented and never verified.
    """
    if not auth_enabled():
        return True

    if accounts.store().read_session(websocket.cookies.get(SESSION_COOKIE)) is not None:
        return True

    candidates = [
        websocket.query_params.get(QUERY_NAME),
        websocket.headers.get(HEADER_NAME),
    ]
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidates.append(auth_header[7:].strip())
    if handshake_params:
        auth = handshake_params.get("auth")
        if isinstance(auth, dict):
            candidates.append(auth.get("token"))

    return any(token_is_valid(c) for c in candidates)
