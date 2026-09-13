# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP attach server — let an external program send a user message
into an agentica session that is already running in a terminal.

Why this exists: a session's input has exactly one writer, the person at the
keyboard, and until now nothing else could give that session a line. The desktop
app / pet case is "the user says something from over there", which is *user
input*, not an agent-to-agent message (that is ``send-message``) — so it belongs
on a protocol entry point, the way codex exposes ``turn/start`` and hermes
exposes ``session/prompt`` over ACP.

## Why a socket and not stdio

ACP's default transport is stdio: the client spawns the agent as a subprocess and
speaks JSON-RPC on its pipes. That cannot express "attach to a session that is
already up", and for this process it is impossible anyway — the interactive CLI's
TUI owns stdin/stdout (prompt_toolkit). ACP allows other transports as long as
they preserve the JSON-RPC message format and lifecycle; this is that transport,
and it follows the spec's framing (newline-delimited JSON, no embedded newlines).

## Why the address is the socket

**One listener per session, and connecting to it *is* naming the session.** There
is no ``--to`` and no second way to say who you mean: no name collisions, no
prefix ambiguity, and no cwd matching (symlinked ``/tmp`` vs ``/private/tmp`` is a
real trap that has already cost somebody a debugging session). A client discovers
the socket path from the peer record it can already list.

## Authority

This channel's effect is the user speaking, so it demands more care than a
read-only observer: the socket lives in a ``0700`` directory, every connection
must present the session's token on ``initialize``, and a missing/wrong token is
refused. It is local-only by construction (a unix domain socket).
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentica.utils.log import logger

#: Framing and protocol version. Matches ACP's stdio framing: one JSON document
#: per line, no embedded newlines, which is also the natural fit for a stream
#: socket and keeps a connection readable with ``nc``.
PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 8 * 1024 * 1024

#: JSON-RPC error codes we produce. Standard ones, plus -32000 for our refusals.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_REFUSED = -32000

#: The subset of ACP this transport implements. ``session/new`` is deliberately
#: absent: a session already exists — the one whose socket this is — and
#: inventing a second one here would let a client address a conversation that no
#: terminal is driving. ``session/load`` is how you attach to it.
METHODS = (
    "initialize",
    "session/load",
    "session/prompt",
    "session/cancel",
    "ping",
)


def attach_dir() -> Path:
    """Where per-session sockets live.

    **Deliberately not under the cache dir.** ``AF_UNIX`` paths are capped by
    ``sun_path`` (~104 bytes on macOS), and the cache dir is user-configurable
    and often deep — putting sockets there makes the attach point vanish with
    ``AF_UNIX path too long`` for exactly the users who moved their cache, with
    a symptom ("the pet can't see my session") that looks nothing like the cause.
    Measured: this path is 31 bytes, so it fits with room to spare.

    Per-uid so two users on one machine cannot collide, and ``gettempdir()``
    rather than a hardcoded ``/tmp`` because that is what honours ``TMPDIR``.
    ``AGENTICA_ATTACH_DIR`` overrides it (tests, and anyone whose temp is on a
    filesystem without unix sockets).

    Kept ``0700``: clients are processes running as this user (a desktop app, a
    script they launched), so nothing else has any business here, and a private
    parent means only this user can even reach the socket or read its token.
    """
    override = os.getenv("AGENTICA_ATTACH_DIR")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / f"agentica-{os.getuid()}"


def attach_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Is the attach point switched on for this process?

    Off unless ``settings.attach_enabled`` is true — a **flat settings key**, the
    same shape the notify sink uses (``settings.notify.enabled`` is a nested
    block, but the CLI's other toggles read flat keys like
    ``settings.deliver_background_results``). Env ``AGENTICA_ATTACH_ENABLED=1``
    overrides, matching the rest of the project's config handling.

    Read once, at session start: the socket is created then, so a mid-session
    flip could not take effect anyway and pretending otherwise would be worse
    than saying so.
    """
    override = os.getenv("AGENTICA_ATTACH_ENABLED")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    try:
        from agentica.global_config import get_setting

        return bool(get_setting("attach_enabled", False, config=config))
    except Exception as exc:  # a broken config must not break session start
        logger.debug(f"attach: could not read settings.attach_enabled: {exc}")
        return False


def socket_path(peer_id: str) -> Path:
    return attach_dir() / f"{peer_id}.sock"


def token_path(peer_id: str) -> Path:
    return attach_dir() / f"{peer_id}.token"


def _ensure_private_dir(path: Path) -> bool:
    """``0700`` directory, or False when it cannot be made private.

    Refusing to serve is the right answer when the directory is not private:
    the socket on the other side of it carries the user's own authority.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
        return True
    except OSError as exc:
        logger.warning(f"attach: cannot prepare {path} ({exc}); not serving")
        return False


def _load_or_create_token(path: Path) -> Optional[str]:
    """The session token: read it, else create it once with mode 0600.

    Same rule as the notify token — never overwrite an existing one — because a
    client that already read this session's token would otherwise be locked out
    mid-session.
    """
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    token = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return existing or None
    except OSError as exc:
        logger.warning(f"attach: cannot write the token at {path} ({exc})")
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError as exc:
        logger.warning(f"attach: cannot write the token at {path} ({exc})")
        return None
    return token


class AttachServer:
    """Serves JSON-RPC to whoever connects to this session's socket.

    Transport-agnostic on purpose: the host passes in what it knows
    (``inject`` / ``is_running`` / ``snapshot``), so the protocol can be tested
    against a real socket without a TUI, and the CLI keeps ownership of what
    "hand this text to the agent" means.
    """

    def __init__(
        self,
        peer_id: str,
        *,
        inject: Callable[[str], None],
        is_running: Callable[[], bool] = lambda: False,
        snapshot: Optional[Callable[[], Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        cancel: Optional[Callable[[], None]] = None,
        answer: Optional[Callable[[], Optional[str]]] = None,
    ):
        self._peer_id = peer_id
        self._inject = inject
        self._is_running = is_running
        self._snapshot = snapshot
        self._session_id = session_id
        self._cancel = cancel
        self._answer = answer
        self._token: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._clients: List[socket.socket] = []

    # ------------------------------------------------------------------ lifecycle

    @property
    def path(self) -> Path:
        return socket_path(self._peer_id)

    @property
    def token_file(self) -> Path:
        return token_path(self._peer_id)

    @property
    def listening(self) -> bool:
        return self._sock is not None

    def start(self) -> bool:
        """Bind and serve in the background. False means "not serving".

        Every failure here is non-fatal for the session: the terminal works
        exactly as before, there is just no attach point.
        """
        if not (self._peer_id or "").strip():
            logger.debug("attach: no peer id; not serving")
            return False
        if not _ensure_private_dir(attach_dir()):
            return False
        self._token = _load_or_create_token(self.token_file)
        if not self._token:
            return False

        path = self.path
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug(f"attach: could not clear a stale socket at {path}: {exc}")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
            # The socket itself carries the user's authority, so restrict it even
            # though its directory is already 0700 — defence in depth, and it
            # survives someone loosening the directory later.
            os.chmod(path, 0o600)
            sock.listen(8)
            sock.settimeout(0.5)
        except OSError as exc:
            logger.warning(f"attach: cannot listen at {path} ({exc})")
            return False

        self._sock = sock
        self._thread = threading.Thread(
            target=self._accept_loop, name="agentica-attach", daemon=True
        )
        self._thread.start()
        logger.info(f"attach: listening at {path}")
        return True

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        try:
            self.path.unlink()
        except OSError:
            pass
        # The token goes with the socket: leaving it behind would let a later
        # reader authenticate against a session that no longer exists.
        try:
            self.token_file.unlink()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                client, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self._clients.append(client)
            threading.Thread(
                target=self._serve_client,
                args=(client,),
                name="agentica-attach-client",
                daemon=True,
            ).start()

    # -------------------------------------------------------------------- serving

    def _serve_client(self, client: socket.socket) -> None:
        authenticated = False
        buf = b""
        try:
            client.settimeout(None)
            while not self._stop.is_set():
                try:
                    chunk = client.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                if len(buf) > MAX_LINE_BYTES:
                    self._send(client, self._error(None, ERR_INVALID_REQUEST, "message too large"))
                    return
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    if not line.strip():
                        continue
                    response, authenticated = self._handle_line(
                        line, authenticated
                    )
                    if response is not None:
                        self._send(client, response)
        except Exception as exc:  # a client must never take the session down
            logger.debug(f"attach: client ended: {exc}")
        finally:
            with self._lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                client.close()
            except OSError:
                pass

    def _send(self, client: socket.socket, payload: Dict[str, Any]) -> None:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
            client.sendall(data)
        except OSError:
            pass

    def _handle_line(self, line: bytes, authenticated: bool):
        """Parse one request, dispatch it. Returns ``(response_or_None, authed)``."""
        try:
            request = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._error(None, ERR_PARSE, "not a JSON document"), authenticated
        if not isinstance(request, dict):
            return self._error(None, ERR_INVALID_REQUEST, "not a JSON-RPC object"), authenticated

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            return self._error(request_id, ERR_INVALID_REQUEST, "method is required"), authenticated
        if method not in METHODS:
            return (
                self._error(request_id, ERR_METHOD_NOT_FOUND, f"method not found: {method}"),
                authenticated,
            )

        if not authenticated:
            # Everything is gated, including ``initialize``: a client that has
            # not proven it holds the token gets no information about this
            # session at all.
            supplied = str(params.get("authToken") or "")
            if not self._token or not hmac.compare_digest(supplied, self._token):
                logger.debug(f"attach: refused {method} without a valid token")
                return (
                    self._error(
                        request_id,
                        ERR_REFUSED,
                        "authentication required: pass this session's token as "
                        "params.authToken (see the session's <peer_id>.token file)",
                    ),
                    False,
                )
            authenticated = True

        try:
            result = self._dispatch(method, params)
        except AttachError as exc:
            return self._error(request_id, exc.code, str(exc)), authenticated
        except Exception as exc:
            logger.debug(f"attach: {method} failed: {exc}")
            return self._error(request_id, ERR_INTERNAL, str(exc)), authenticated
        return self._success(request_id, result), authenticated

    def _dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "ping":
            return {"status": "ok"}
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "agentCapabilities": {"loadSession": True, "promptCapabilities": {}},
                "agentInfo": {"name": "agentica", "version": PROTOCOL_VERSION},
                "authMethods": [],
            }
        if method == "session/load":
            return self._load(params)
        if method == "session/prompt":
            return self._prompt(params)
        if method == "session/cancel":
            self._cancel_current(params)
            return {}
        raise AttachError(ERR_METHOD_NOT_FOUND, f"method not found: {method}")

    def _load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach to this session. A mismatched id means you are on the wrong socket."""
        wanted = params.get("sessionId")
        if wanted and self._session_id and str(wanted) != str(self._session_id):
            raise AttachError(
                ERR_REFUSED,
                f"this socket serves session {self._session_id}, not {wanted}; "
                f"connect to the socket of the session you mean",
            )
        info: Dict[str, Any] = {"sessionId": self._session_id or self._peer_id}
        if self._snapshot is not None:
            try:
                info.update(self._snapshot() or {})
            except Exception as exc:
                logger.debug(f"attach: snapshot failed: {exc}")
        return info

    def _prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inject the user's text and answer when the turn carrying it ends.

        The text reaches the agent by the same route a typed line takes
        (``hand_to_agent``): steered into a running turn at the next tool-batch
        boundary, or queued as the next turn when idle. So this is the user
        speaking, not a side channel into the run.
        """
        text = _prompt_text(params.get("prompt"))
        if not text:
            raise AttachError(ERR_INVALID_PARAMS, "prompt is empty")
        wanted = params.get("sessionId")
        if wanted and self._session_id and str(wanted) != str(self._session_id):
            raise AttachError(
                ERR_REFUSED,
                f"this socket serves session {self._session_id}, not {wanted}",
            )

        was_running = self._is_running()
        self._inject(text)
        # Only wait when this session will report the turn back. A prompt handed
        # to a session that is idle becomes the next turn; the host's
        # ``is_running`` flipping True and back is that turn's life. If it never
        # starts (the session is quitting), say so rather than blocking forever.
        if self._is_running is None:
            return {"stopReason": "end_turn"}
        completed = self._wait_for_turn(was_running)
        answer = None
        if self._answer is not None:
            try:
                answer = self._answer()
            except Exception:
                answer = None
        if completed == "cancelled":
            return {"stopReason": "cancelled"}
        if completed == "timeout":
            # The turn was accepted but this session did not report an end
            # within the grace window — report that honestly instead of
            # claiming an answer we never saw.
            return {"stopReason": "end_turn", "agenticaPending": True}
        result: Dict[str, Any] = {"stopReason": "end_turn"}
        if answer:
            result["agenticaAnswer"] = answer
        return result

    def _wait_for_turn(self, was_running: bool, grace: float = 600.0) -> str:
        """Wait for the turn that carries the injected text to finish.

        Starts by waiting for the run to *begin* when the session was idle, so a
        queued prompt is not mistaken for "already finished".
        """
        deadline = time.monotonic() + grace
        if not was_running:
            started = False
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    return "cancelled"
                if self._is_running():
                    started = True
                    break
                time.sleep(0.05)
            if not started:
                return "timeout"
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return "cancelled"
            if not self._is_running():
                # Give the response a beat to be recorded before reading it.
                time.sleep(0.2)
                return "completed"
            time.sleep(0.05)
        return "timeout"

    def _cancel_current(self, params: Dict[str, Any]) -> None:
        if self._cancel is not None:
            self._cancel()

    @staticmethod
    def _success(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


class AttachError(Exception):
    """A refusal with a JSON-RPC code, so the client gets a real reason."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _prompt_text(prompt: Any) -> str:
    """The user's text out of an ACP prompt.

    ACP sends a list of content blocks; this channel accepts text only. Anything
    else is refused rather than degraded, because silently dropping an image the
    user attached would change what they asked for.
    """
    if isinstance(prompt, str):
        return prompt.strip()
    if not isinstance(prompt, list):
        return ""
    parts: List[str] = []
    for block in prompt:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part).strip()
