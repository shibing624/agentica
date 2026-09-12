# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The shared secret that keeps the notify socket from being a way
for any local process to answer as the user.

The socket is local, but "local" is not "trusted": the desktop reply to
``needs.approval`` / ``needs.input`` is applied as the *user's* answer, so
without a token any process on the machine could approve a command or answer a
question while claiming to be the person at the keyboard. The token lives in a
file both sides read, mode ``0600``.

Creation rule: **only create it when it is absent, never overwrite.** Either
side may be the first to run, and whoever gets there first defines the secret;
clobbering it would lock the other side out with a 401 on every request.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from agentica.notify.config import NotifyConfig
from agentica.utils.log import logger

TOKEN_BYTES = 32


def ensure_token(config: NotifyConfig) -> Optional[str]:
    """Return the token to use, creating the token file if it is missing.

    Returns None when a token cannot be established; the caller decides whether
    that is fatal (it is not — the channel may still be open, and a server that
    requires a token will simply answer 401, which already degrades safely).
    """
    if config.token:
        return config.token
    path = config.resolved_token_file
    existing = _read_token(path)
    if existing:
        return existing
    created = _create_token(path)
    if created:
        logger.debug(f"notify sink: wrote a new token to {path}")
    return created


def _read_token(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _create_token(path: str) -> Optional[str]:
    """Create the token file with mode 0600, atomically.

    ``O_EXCL`` does two jobs: it makes the create-and-write atomic, and it
    loses the race gracefully if the other side created the file between our
    read and our write — in which case we adopt theirs.
    """
    token = secrets.token_hex(TOKEN_BYTES)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except OSError as exc:
        _warn_unavailable(path, exc)
        return None
    try:
        # Mode is set at creation, so there is never a moment where the secret
        # exists with wider permissions than intended.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # The file appeared between our read and our create. If it holds a real
        # token, that is the other side's and we adopt it. If it is empty or
        # whitespace, it is unusable *and* create-exclusive will never let us
        # fix it — rewrite in place, keeping the mode.
        existing = _read_token(path)
        if existing:
            return existing
        return _overwrite_token(path, token)
    except OSError as exc:
        _warn_unavailable(path, exc)
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError as exc:
        _warn_unavailable(path, exc)
        return None
    return token


def _overwrite_token(path: str, token: str) -> Optional[str]:
    """Write ``token`` into an existing file that holds nothing usable."""
    try:
        # Keep 0600 explicit: the file may have been created by something else
        # with wider permissions, and this is a secret.
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC, 0o600)
    except OSError as exc:
        _warn_unavailable(path, exc)
        return None
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # best effort: some filesystems have no meaningful mode
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError as exc:
        _warn_unavailable(path, exc)
        return None
    return token


def _warn_unavailable(path: str, exc: OSError) -> None:
    """A missing token is worth a warning, not a failure.

    Warning rather than debug because it changes what the channel can do: a
    server that requires a token will reject every request, and the user would
    otherwise see a sink that is silently useless.
    """
    logger.warning(
        f"notify sink: could not prepare the token file at {path} ({exc}); "
        f"requests will go out unauthenticated and a server requiring a token "
        f"will reject them."
    )
