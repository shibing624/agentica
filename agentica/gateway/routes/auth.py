# -*- coding: utf-8 -*-
"""Signing in and out, and who else may sign in.

Three of the sign-in routes are open (a gate over the door is a locked room):

- ``GET  /api/auth/status`` — what the login page needs before it can render:
  is the gate even on, is there a password to type, am I already in. It says
  whether a password *exists*, which on the deployment that matters (loopback)
  is not a secret and is the difference between showing a form and showing the
  token instructions.
- ``POST /api/auth/login`` — username + password in, session cookie out.
- ``POST /api/auth/logout`` — closes the session server-side, not just the
  cookie: a copied cookie must stop working too.
- ``POST /api/auth/password`` — set or change your own. Behind the gate.

The account table (``/api/auth/users``) is admin-only, and it is the only
admin-only surface in the gateway. Everything else configures one machine
(models, skills, cron, the working directory) and every signed-in account may
change it; adding accounts is the one act that decides who those accounts are.
    A new account is created with an initial password the admin typed; it is
    always a user, never a second administrator — one admin is enough, and a
    second one makes password recovery a maze. Changing a user's password is
    the new password alone; changing your own still needs the current one.
"""
from fastapi import APIRouter, HTTPException, Request, Response

from .. import accounts, auth

router = APIRouter(prefix="/api/auth")


def _require_admin(request: Request) -> auth.Principal:
    principal: auth.Principal = request.state.principal
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return principal


@router.get("/status")
async def auth_status(request: Request):
    session = accounts.store().read_session(request.cookies.get(auth.SESSION_COOKIE))
    # Signed out, the page still needs to know whether a password exists at
    # all; signed in, everything else is about *this* account.
    user_id = session.user_id if session else None
    account = accounts.store().get_account(user_id) if user_id else None
    return {
        "auth_enabled": auth.auth_enabled(),
        "password_set": accounts.store().has_password(),
        "authenticated": (not auth.auth_enabled()) or session is not None,
        "user_id": user_id,
        "via": session.via if session else None,
        "role": account.role if account else None,
        "is_admin": bool(account and account.is_admin),
        "default_account_id": accounts.default_account_id(),
        # The login page says so, and the settings block nags. Not a secret: a
        # gateway that still has its generated password says as much to anybody
        # who can reach the login page, which is the population that should be
        # told to change it.
        "password_is_initial": account.password_is_initial if account
        else accounts.store().password_is_initial(),
        "min_password_length": accounts.MIN_PASSWORD_LENGTH,
    }


@router.post("/login")
async def login(request: Request, response: Response):
    body = await request.json()
    # An empty username means the seeded account, so a bookmarked login page
    # and the desktop shell keep working without one.
    raw = str(body.get("username") or "").strip()
    if not raw:
        user_id = accounts.default_account_id()
    else:
        user_id = accounts.normalize_account_id(raw) or raw.lower()
    password = str(body.get("password") or "")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required")
    if not accounts.store().has_password():
        raise HTTPException(
            status_code=409,
            detail="No password is set on this gateway. Run "
                   "`agentica-gateway --set-password` on the machine running it.",
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
    """Set or change your own password.

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

    # set_password signs this account's browsers out, including this one — so
    # hand this one a fresh session instead of logging the user out of the page
    # they just used to set it.
    auth.set_session_cookie(response, accounts.store().open_session(user_id, "password"))
    return {"status": "ok", "user_id": user_id}


@router.get("/users")
async def list_users(request: Request):
    principal = _require_admin(request)
    return {
        "users": [a.to_dict() for a in accounts.store().list_accounts()],
        "current": principal.user_id,
        "min_password_length": accounts.MIN_PASSWORD_LENGTH,
    }


@router.post("/users")
async def create_user(request: Request):
    """Add an account. The initial password is the one the admin typed.

    A generated password that is shown once and then forgotten is a reset
    flow in disguise — the admin still has to tell the new person how to
    get in, so they type it here. New accounts are users; the seeded
    account is the only administrator. A new account starts empty: it owns
    ``users/<name>/``, so it sees its own conversations and memory rather
    than the creator's.
    """
    _require_admin(request)
    body = await request.json()
    password = str(body.get("password") or "")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required")
    username = accounts.normalize_account_id(str(body.get("username") or ""))
    role = str(body.get("role") or accounts.ROLE_USER)
    if role != accounts.ROLE_USER:
        raise HTTPException(
            status_code=400,
            detail="New accounts are users. The seeded account is the only administrator.",
        )
    try:
        accounts.store().create_account(username, password, role=accounts.ROLE_USER)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _scaffold_user(username)
    return {"status": "ok", "user_id": username}


@router.post("/users/{user_id}/password")
async def change_user_password(user_id: str, request: Request, response: Response):
    """Change an account's password. Admin only.

    Changing someone else is just the new password — the admin is already
    signed in, and asking for a current password (theirs or the target's)
    is a reset in disguise. Changing *your own* still needs the current
    one, same as ``POST /api/auth/password``: that is the built-in
    administrator proving they still know it. A token/desktop session may
    skip that proof.
    """
    principal = _require_admin(request)
    body = await request.json()
    new = str(body.get("password") or "")
    if accounts.store().get_account(user_id) is None:
        raise HTTPException(status_code=404, detail=f"{user_id} does not exist")

    if user_id == principal.user_id:
        old = str(body.get("old_password") or "")
        session = accounts.store().read_session(request.cookies.get(auth.SESSION_COOKIE))
        privileged = principal.kind != "session" or (
            session is not None and session.via in ("token", "desktop")
        )
        if not privileged:
            if not old or not accounts.store().check_password(principal.user_id, old):
                raise HTTPException(status_code=400, detail="Current password is incorrect")

    try:
        accounts.store().set_password(user_id, new)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if user_id == principal.user_id:
        auth.set_session_cookie(response, accounts.store().open_session(user_id, "password"))
    return {"status": "ok", "user_id": user_id}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, request: Request):
    """Remove an account. Its ``users/<id>/`` data is left on disk.

    Refused for your own account: signing yourself out by deleting the login
    you are holding is never what was meant, and it can leave a machine with
    no administrator.
    """
    principal = _require_admin(request)
    if user_id == principal.user_id:
        raise HTTPException(status_code=400, detail="You cannot remove your own account")
    try:
        accounts.store().delete_account(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


def _scaffold_user(user_id: str) -> None:
    """Create ``users/<id>/`` and a default Project named after the account.

    Sessions group by working directory, not by a Project entity — the
    directory is the Project. The folder is the sanitised username itself
    (``workspace/<id>/``), not a ``-default_project`` suffix.
    """
    from agentica.workspace import Workspace

    ws = Workspace(user_id=user_id)
    ws.initialize()
    (ws.path / user_id).mkdir(parents=True, exist_ok=True)
