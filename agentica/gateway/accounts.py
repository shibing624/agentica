# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Who may use this gateway, and which browsers are currently signed in.

Two things live here, and the split between them is the whole point of the
module. The **machine token** (``auth.py``) proves "I am on this machine and can
read a 0600 file"; it is per process, so it changes on every restart. A
**session** proves "this browser signed in once"; it is written down, so a
restart does not sign anybody out. Before this split the cookie held the machine
token itself, which meant every gateway restart sent the user back to the
terminal to copy a new URL, and a leaked cookie was the master credential with
no way to revoke just that browser.

Storage is ``$AGENTICA_HOME/gateway/auth.json``, ``0600``. Not the cache dir:
losing a password hash to a cache wipe is not a recoverable inconvenience, and
surviving a restart is the reason sessions are on disk at all.

Accounts are keyed by an account id so that a second account is a row here
rather than a redesign. Exactly one exists today — ``admin``, seeded on first
start — because there is no registration route and no admin UI.

**An account id is not a data user id.** ``settings.default_user_id`` partitions
sessions, memory and skills under ``users/<id>/`` on disk; this file only
answers "may you in". They were briefly the same string, which reads as a
simplification and is a trap: renaming the login would have moved every existing
conversation out of view.

A gateway with no password is not a gateway with no credential. First start
seeds ``admin`` with a generated one and prints it, because the alternative was
a ``/chat?token=…`` URL from the log — per process, so it changed on every
restart, and unusable to anyone whose gateway was started detached or by the
desktop shell. The plaintext is kept at ``initial-password`` (``0600``) *only*
while it is still the generated one, so a user who scrolled past the banner can
still get in; changing the password deletes it.

Password hashing is ``hashlib.scrypt`` — stdlib, no new dependency, and a real
slow hash. The stored form is ``scrypt$n$r$p$<salt b64>$<hash b64>``: the
parameters travel with the hash, so raising them later does not invalidate
every existing password.
"""
import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from agentica.config import AGENTICA_HOME
from agentica.utils.log import logger

AUTH_FILE = Path(AGENTICA_HOME) / "gateway" / "auth.json"

#: The one web account. Not a data user id — see the module docstring.
ADMIN_USER_ID = "admin"

# Six, because this gate's job is to stop a passer-by on the LAN, and the two
# real defences are elsewhere: scrypt makes an offline guess expensive, and the
# throttle below makes an online one take days. A longer minimum only pushes
# people toward reusing a password they already have.
MIN_PASSWORD_LENGTH = 6

# Generated passwords are read off a terminal and typed into a browser once, so
# the alphabet drops the glyphs that get misread (0/O, 1/l/I) and the groups are
# short. 8 symbols from 30 is ~39 bits — with the throttle below, guessing it
# online takes years, and it is meant to be replaced anyway.
_INITIAL_ALPHABET = "23456789abcdefghjkmnpqrstuvwxy"
_INITIAL_GROUPS = 2
_INITIAL_GROUP_LEN = 4

# scrypt cost. n=2**14 with r=8 is ~16MB and ~60ms here — the usual
# "interactive login" point, slow enough to make offline guessing expensive and
# fast enough that a login does not feel broken.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 64

# A session lasts a week, and any request made in its last day extends it back
# to a full week. So a browser used daily is never signed out, and one left
# alone for a week is.
SESSION_TTL = timedelta(days=7)
SESSION_RENEW_WITHIN = timedelta(days=1)

# Login throttling. A password typed by a human is guessable at machine speed,
# and this endpoint is reachable from the LAN in exactly the deployment that
# made a password mandatory. Five free tries, then 1s, 2s, 4s … capped at a
# minute, measured from the last failure; a success clears the count. Kept in
# process memory: a restart clears it, and restarting the gateway is slower
# than waiting out the cap.
_FREE_ATTEMPTS = 5
_BACKOFF_START_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 60.0
_FAILURE_IDLE_SECONDS = 900.0


class LoginThrottled(Exception):
    """Refused because too many attempts failed recently."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Too many failed attempts; retry in {retry_after:.0f}s")


@dataclass
class Session:
    """One signed-in browser."""

    user_id: str
    # How it was established: "password" (typed it), "token" (redeemed the
    # machine token, i.e. opened the printed URL), "desktop" (the shell). It
    # decides one thing: whether setting a password may skip the old one — a
    # token holder has already proved machine ownership, and on a gateway that
    # never had a password there is no old one to type.
    via: str
    created_at: str
    expires_at: str

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "via": self.via,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _digest(token: str) -> str:
    """What gets written down. The token itself never touches the disk, so a
    readable ``auth.json`` cannot be replayed as a session."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_initial_password() -> str:
    """A password a human can read off a terminal and type once."""
    groups = [
        "".join(secrets.choice(_INITIAL_ALPHABET) for _ in range(_INITIAL_GROUP_LEN))
        for _ in range(_INITIAL_GROUPS)
    ]
    return "-".join(groups)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_KEY_BYTES, maxmem=64 * 1024 * 1024,
    )
    return "$".join([
        "scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P),
        base64.b64encode(salt).decode(), base64.b64encode(key).decode(),
    ])


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash. A malformed record is a failed
    login, never an exception: the caller has one honest answer to give either
    way, and a hand-edited file must not 500 the login route."""
    parts = (stored or "").split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = base64.b64decode(parts[4])
        expected = base64.b64decode(parts[5])
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=64 * 1024 * 1024,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(actual, expected)


class AccountStore:
    """The one reader and writer of ``auth.json``.

    Every mutation re-reads before it writes: the CLI's ``--set-password`` and a
    running gateway are two processes over one file, and the loser of a
    read-modify-write would silently drop the other's password.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else AUTH_FILE
        # Per user_id: (consecutive failures, monotonic time of the last one).
        self._failures: dict[str, tuple[int, float]] = {}

    # ---- file ----

    def _read(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "accounts": {}, "sessions": {}}
        if not isinstance(raw, dict):
            return {"version": 1, "accounts": {}, "sessions": {}}
        raw.setdefault("accounts", {})
        raw.setdefault("sessions", {})
        return raw

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        # 0600 from creation, not fixed up afterwards: for the width of that
        # window the password hash would be world-readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self.path)

    # ---- accounts ----

    def has_password(self, user_id: Optional[str] = None) -> bool:
        """Whether anybody can sign in with a password.

        Asked without a ``user_id`` it means "is password login available at
        all", which is what the login page and the non-loopback startup guard
        need to know.
        """
        accounts = self._read()["accounts"]
        if user_id is None:
            return any(a.get("password") for a in accounts.values() if isinstance(a, dict))
        entry = accounts.get(user_id)
        return bool(isinstance(entry, dict) and entry.get("password"))

    def set_password(self, user_id: str, password: str) -> None:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        data = self._read()
        entry = data["accounts"].get(user_id)
        if not isinstance(entry, dict):
            entry = {"created_at": _now().isoformat(timespec="seconds")}
        entry["password"] = hash_password(password)
        entry["password_set_at"] = _now().isoformat(timespec="seconds")
        entry["password_is_initial"] = False
        data["accounts"][user_id] = entry
        # Every other browser is signed out. A password change is what a user
        # does after "somebody may have my cookie", so keeping those sessions
        # alive would defeat the only reason they changed it.
        data["sessions"] = {}
        self._write(data)
        self._failures.pop(user_id, None)
        # The generated one is no longer the way in, so stop keeping a readable
        # copy of it. Written before the file existed too, hence missing_ok.
        self.initial_password_path.unlink(missing_ok=True)

    # ---- first run ----

    @property
    def initial_password_path(self) -> Path:
        """Where the generated password is readable while it is still in use."""
        return self.path.parent / "initial-password"

    def seed_admin(self) -> Optional[str]:
        """Create the ``admin`` account on first start; return its password.

        Returns None when any account already exists, so this is safe to call on
        every boot. The plaintext is both returned (to print) and written
        ``0600`` (so a gateway started detached, or by the desktop shell, is
        still reachable from a browser later).
        """
        if self._read()["accounts"]:
            return None
        password = generate_initial_password()
        self.set_password(ADMIN_USER_ID, password)
        data = self._read()
        data["accounts"][ADMIN_USER_ID]["password_is_initial"] = True
        self._write(data)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.initial_password_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(password + "\n")
        except OSError as e:
            # The account exists and the password was just printed; not being
            # able to cache it is worth a line, not a failed startup.
            logger.warning(f"Could not store the initial password: {e}")
        return password

    def password_is_initial(self, user_id: str = ADMIN_USER_ID) -> bool:
        """Whether this account still uses the password the gateway generated."""
        entry = self._read()["accounts"].get(user_id)
        return bool(isinstance(entry, dict) and entry.get("password_is_initial"))

    def read_initial_password(self) -> Optional[str]:
        """The generated password, for reprinting on a later boot."""
        try:
            return self.initial_password_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def clear_password(self, user_id: str) -> None:
        """Drop the password (back to token-only). Also drops the sessions it
        authorised, for the same reason a change does."""
        data = self._read()
        entry = data["accounts"].get(user_id)
        if isinstance(entry, dict):
            entry.pop("password", None)
            entry.pop("password_set_at", None)
        data["sessions"] = {}
        self._write(data)
        self.initial_password_path.unlink(missing_ok=True)

    def check_password(self, user_id: str, password: str) -> bool:
        """Verify a password, subject to throttling. Raises ``LoginThrottled``.

        A failure against an id that does not exist is counted like any other,
        so the throttle cannot be used to enumerate which accounts are real.
        """
        now = time.monotonic()
        for key, (_, last) in list(self._failures.items()):
            if now - last > _FAILURE_IDLE_SECONDS:
                self._failures.pop(key, None)

        failures, last = self._failures.get(user_id, (0, 0.0))
        wait = self._backoff(failures)
        if wait and now < last + wait:
            raise LoginThrottled(last + wait - now)

        entry = self._read()["accounts"].get(user_id)
        stored = entry.get("password", "") if isinstance(entry, dict) else ""
        if stored and verify_password(password, stored):
            self._failures.pop(user_id, None)
            return True
        self._failures[user_id] = (failures + 1, time.monotonic())
        return False

    @staticmethod
    def _backoff(failures: int) -> float:
        excess = failures - _FREE_ATTEMPTS + 1
        if excess <= 0:
            return 0.0
        return min(_BACKOFF_START_SECONDS * 2 ** (excess - 1), _BACKOFF_CAP_SECONDS)

    # ---- sessions ----

    def open_session(self, user_id: str, via: str) -> str:
        """Issue a session and return its token (the only time it exists)."""
        token = secrets.token_urlsafe(32)
        now = _now()
        data = self._read()
        data["sessions"] = {
            h: s for h, s in data["sessions"].items()
            if isinstance(s, dict) and self._alive(s.get("expires_at"), now)
        }
        data["sessions"][_digest(token)] = Session(
            user_id=user_id,
            via=via,
            created_at=now.isoformat(timespec="seconds"),
            expires_at=(now + SESSION_TTL).isoformat(timespec="seconds"),
        ).to_dict()
        self._write(data)
        return token

    def read_session(self, token: Optional[str]) -> Optional[Session]:
        """Resolve a session token, renewing it when it is close to expiring.

        Expired records are deleted on the way past rather than swept on a
        timer: this is the only code that looks at them, so the read *is* the
        sweep.
        """
        if not token:
            return None
        digest = _digest(token)
        data = self._read()
        raw = data["sessions"].get(digest)
        if not isinstance(raw, dict):
            return None
        now = _now()
        if not self._alive(raw.get("expires_at"), now):
            data["sessions"].pop(digest, None)
            self._write(data)
            return None

        expires = _parse(raw.get("expires_at", ""))
        if expires is not None and expires - now < SESSION_RENEW_WITHIN:
            raw["expires_at"] = (now + SESSION_TTL).isoformat(timespec="seconds")
            data["sessions"][digest] = raw
            self._write(data)
        return Session(
            user_id=str(raw.get("user_id") or "default"),
            via=str(raw.get("via") or "password"),
            created_at=str(raw.get("created_at") or ""),
            expires_at=str(raw.get("expires_at") or ""),
        )

    def close_session(self, token: Optional[str]) -> None:
        if not token:
            return
        data = self._read()
        if data["sessions"].pop(_digest(token), None) is not None:
            self._write(data)

    @staticmethod
    def _alive(expires_at, now: datetime) -> bool:
        parsed = _parse(expires_at if isinstance(expires_at, str) else "")
        return parsed is not None and parsed > now


_store: Optional[AccountStore] = None


def store() -> AccountStore:
    """The process-wide store, built on first use."""
    global _store
    if _store is None:
        _store = AccountStore()
    return _store


def use_store_for_tests(path) -> AccountStore:
    """Point the store at ``path``. Only a test has a reason to call this —
    ``AGENTICA_HOME`` is read at import, so this is the seam that keeps a test
    run from writing a password hash into the user's real home."""
    global _store
    _store = AccountStore(path)
    return _store


def set_password_interactive(user_id: str = ADMIN_USER_ID) -> int:
    """``agentica-gateway --set-password``: prompt twice, write, exit.

    Lives here rather than in ``main.py`` because it is the only way to set a
    password before the server can serve the page that sets one — the chicken
    and egg of "binding to the LAN requires a password".
    """
    import getpass

    first = getpass.getpass(f"New password for {user_id}: ")
    second = getpass.getpass("Repeat: ")
    if first != second:
        print("Passwords do not match.")
        return 1
    try:
        store().set_password(user_id, first)
    except ValueError as e:
        print(str(e))
        return 1
    logger.info(f"Password set for {user_id} in {store().path}")
    print("Password set. All existing web sessions were signed out.")
    return 0
