# -*- coding: utf-8 -*-
"""Signing in and out.

Four routes, three of them open (a gate over the door is a locked room):

- ``GET  /api/auth/status`` — what the login page needs before it can render:
  is the gate even on, is there a password to type, am I already in. It says
  whether a password *exists*, which on the deployment that matters (loopback)
  is not a secret and is the difference between showing a form and showing the
  token instructions.
- ``POST /api/auth/login`` — password in, session cookie out.
- ``POST /api/auth/logout`` — closes the session server-side, not just the
  cookie: a copied cookie must stop working too.
- ``POST /api/auth/password`` — set or change it. Behind the gate.
"""
from fastapi import APIRouter, HTTPException, Request, Response

from .. import accounts, auth
from ..config import settings

router = APIRouter(prefix="/api/auth")


@router.get("/status")
async def auth_status(request: Request):
    session = accounts.store().read_session(request.cookies.get(auth.SESSION_COOKIE))
    return {
        "auth_enabled": auth.auth_enabled(),
        "password_set": accounts.store().has_password(),
        "authenticated": (not auth.auth_enabled()) or session is not None,
        "user_id": session.user_id if session else None,
        "via": session.via if session else None,
        "min_password_length": accounts.MIN_PASSWORD_LENGTH,
    }


@router.post("/login")
async def login(request: Request, response: Response):
    body = await request.json()
    user_id = str(body.get("user_id") or settings.default_user_id)
    password = str(body.get("password") or "")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required")
    if not accounts.store().has_password():
        raise HTTPException(
            status_code=409,
            detail="No password is set on this gateway. Open the printed "
                   "/chat?token=… URL, or run `agentica-gateway --set-password`.",
        )
    try:
        ok = accounts.store().check_password(user_id, password)
    except accounts.LoginThrottled as e:
        # 429 with the wait, so the page can say how long rather than looking
        # broken. Same answer for a wrong id as for a wrong password.
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {e.retry_after:.0f}s.",
            headers={"Retry-After": str(int(e.retry_after) + 1)},
        )
    if not ok:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    auth.set_session_cookie(response, accounts.store().open_session(user_id, "password"))
    return {"status": "ok", "user_id": user_id}


@router.post("/logout")
async def logout(request: Request, response: Response):
    accounts.store().close_session(request.cookies.get(auth.SESSION_COOKIE))
    auth.clear_session_cookie(response)
    return {"status": "ok"}


@router.post("/password")
async def set_password(request: Request, response: Response):
    """Set or change the password.

    The old password is required — except for a session that got in with the
    machine token (the printed URL, or the desktop shell). That holder already
    proved they can read a 0600 file on this machine, and on a gateway that
    never had a password there is no old one to type. This is the same
    reasoning as the "desktop session may set a password" allowance, applied to
    the one credential that outranks the password.
    """
    body = await request.json()
    new = str(body.get("password") or "")
    old = str(body.get("old_password") or "")
    principal: auth.Principal = request.state.principal
    user_id = principal.user_id

    session = accounts.store().read_session(request.cookies.get(auth.SESSION_COOKIE))
    privileged = principal.kind != "session" or (
        session is not None and session.via in ("token", "desktop")
    )

    if accounts.store().has_password(user_id) and not privileged:
        if not old or not accounts.store().check_password(user_id, old):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

    try:
        accounts.store().set_password(user_id, new)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # set_password signs every browser out, including this one — so hand this
    # one a fresh session instead of logging the user out of the page they just
    # used to set it.
    auth.set_session_cookie(response, accounts.store().open_session(user_id, "password"))
    return {"status": "ok", "user_id": user_id}
