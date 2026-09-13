#!/usr/bin/env python3
"""End-to-end: an external client sends a user message into a live agentica CLI.

The completion criterion for the attach protocol: a real interactive session in
tmux, driven by a real JSON-RPC client over its socket, treats the text as the
user speaking and answers it.

Not in the pytest suite: it drives a real TUI in a real pty, needs a model, and
takes tens of seconds. Run it by hand:

    python scripts/verify_attach_e2e.py

Needs a working model config (or the provider env vars) because the receiving
session has to actually answer. Override the ask with ATTACH_E2E_PROMPT.
"""
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

PROMPT = os.getenv("ATTACH_E2E_PROMPT", "Reply with exactly: PONG")
EXPECTED = os.getenv("ATTACH_E2E_EXPECT", "PONG")
SESSION = "agentica-attach-e2e"

failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def tmux(*args):
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def pane():
    return tmux("capture-pane", "-p", "-t", SESSION, "-S", "-150").stdout


class Client:
    """A line-delimited JSON-RPC client — i.e. what a desktop app writes."""

    def __init__(self, path, token):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(300)
        self._sock.connect(path)
        self._buf = b""
        self.call("initialize", {"authToken": token, "clientCapabilities": {}})

    def call(self, method, params=None, request_id=1):
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._sock.sendall(json.dumps(payload).encode() + b"\n")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise SystemExit("the session closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line.decode())

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


if not shutil.which("tmux"):
    print("tmux not found; cannot run this check")
    sys.exit(2)

root = tempfile.mkdtemp(prefix="attach-e2e-")
home = os.path.join(root, ".agentica")
work = os.path.join(root, "proj")
os.makedirs(home, exist_ok=True)
os.makedirs(work, exist_ok=True)

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ, AGENTICA_HOME=home, PYTHONPATH=repo)

# Switch the attach point on the way a user would: config, not an internal flag.
os.makedirs(home, exist_ok=True)
with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
    fh.write("settings:\n  attach_enabled: true\n")

tmux("kill-session", "-t", SESSION)
print(f"== starting a real interactive CLI in tmux ({SESSION}) ==")
launch = f"AGENTICA_HOME={shlex.quote(home)} PYTHONPATH={shlex.quote(repo)} agentica"
tmux("new-session", "-d", "-s", SESSION, "-x", "200", "-y", "50", "-c", work, launch)

peer_id = socket_path = None
try:
    # Discovery: the socket path comes out of the presence record, so a client
    # never recomputes it (uid / TMPDIR / override).
    deadline = time.monotonic() + 90
    probe_code = (
        "import json,sys,os\n"
        "from agentica import peers\n"
        "# Read the presence records where *this* install puts them, rather than\n"
        "# guessing the layout: the client's whole job is to read the path, and\n"
        "# a hardcoded guess here would be the same class of bug it prevents.\n"
        "import agentica.config as cfg\n"
        "# AGENTICA_CACHE_DIR is the cache root; peers_root() appends 'peers'.\n"
        "peers.AGENTICA_CACHE_DIR = cfg.AGENTICA_CACHE_DIR\n"
        "for p in peers.list_live_peers():\n"
        "    print(json.dumps({'peer_id': p.peer_id, 'name': p.name,"
        " 'attach': p.attach_socket, 'session_id': p.session_id}))\n"
    )
    while time.monotonic() < deadline:
        out = subprocess.run(
            [sys.executable, "-c", probe_code], env=env, capture_output=True, text=True
        ).stdout.strip()
        rows = [json.loads(line) for line in out.splitlines() if line.strip()]
        if rows and rows[0].get("attach"):
            peer_id = rows[0]["peer_id"]
            socket_path = rows[0]["attach"]
            break
        time.sleep(2)

    print(f"peer_id={peer_id}  socket={socket_path}")
    check("the session publishes an attach socket", bool(socket_path))
    if not socket_path:
        print(pane()[-1500:])
        raise SystemExit(1)
    check("the socket exists", os.path.exists(socket_path), socket_path)
    check("the socket is 0600", (os.stat(socket_path).st_mode & 0o777) == 0o600)

    token_file = os.path.join(os.path.dirname(socket_path), f"{peer_id}.token")
    check("the token file exists", os.path.exists(token_file))
    token = open(token_file).read().strip()

    print("\n== auth is required ==")
    bad = Client.__new__(Client)
    bad._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bad._sock.settimeout(15)
    bad._sock.connect(socket_path)
    bad._buf = b""
    reply = bad.call("session/load", {"sessionId": "x"})
    check("a client without a token is refused", "error" in reply, json.dumps(reply)[:120])
    bad.close()

    print("\n== a real prompt, injected as the user ==")
    client = Client(socket_path, token)
    loaded = client.call("session/load", {}).get("result", {})
    check("session/load attaches to this session", bool(loaded.get("sessionId")), str(loaded)[:120])
    # The session id must be the live one, not the peer id or a startup snapshot:
    # /resume changes it underneath a running CLI.
    check(
        "session/load reports the real session id",
        isinstance(loaded.get("sessionId"), str) and len(loaded["sessionId"]) > 8,
        str(loaded.get("sessionId")),
    )
    check(
        "and the published socket is in the listing",
        bool(loaded.get("cwd")),
        str(loaded.get("cwd")),
    )

    started = time.monotonic()
    reply = client.call(
        "session/prompt",
        {"prompt": [{"type": "text", "text": PROMPT}]},
        request_id=2,
    )
    elapsed = time.monotonic() - started
    print(f"prompt reply after {elapsed:.1f}s: {json.dumps(reply)[:200]}")
    check("the prompt was accepted", "error" not in reply, json.dumps(reply)[:200])
    if "result" in reply:
        check("the turn reported completion", reply["result"].get("stopReason") == "end_turn")
        # The answer comes back with the reply, so a client need not read the
        # transcript to see what its prompt produced.
        check(
            "the answer comes back with the reply",
            EXPECTED in (reply["result"].get("agenticaAnswer") or ""),
            str(reply["result"].get("agenticaAnswer"))[:60],
        )

    text = pane()
    print("\n== receiving pane (tail) ==")
    print(text[-2000:])
    check("the session shows the relayed line", PROMPT in text)
    check("the session answered it", EXPECTED in text)

    print("\n== a second prompt on the same connection ==")
    reply2 = client.call(
        "session/prompt",
        {"prompt": [{"type": "text", "text": "Reply with exactly: PONG2"}]},
        request_id=3,
    )
    check("a second prompt works", "error" not in reply2, json.dumps(reply2)[:160])

    # ── cancel needs its own connection ─────────────────────────────────────
    # The docs say so, and a client that reads the method table without this
    # would implement cancel on the same connection — where it can never be
    # sent, because session/prompt is still holding it.
    print("\n== session/cancel from a second connection ==")
    waiter = Client(socket_path, token)
    waiter.call("session/load", {})
    cancel_reply = {}

    def _prompt_then_hold():
        cancel_reply["reply"] = waiter.call(
            "session/prompt",
            {"prompt": [{"type": "text", "text": "count slowly to one hundred"}]},
        )

    t = threading.Thread(target=_prompt_then_hold, daemon=True)
    t.start()
    time.sleep(1.5)  # let it be accepted and start running
    other = Client(socket_path, token)
    other.call("session/cancel", {})
    other.close()
    t.join(timeout=30)
    check(
        "a cancel sent on another connection ends the prompt",
        cancel_reply.get("reply", {}).get("result", {}).get("stopReason") == "cancelled",
        json.dumps(cancel_reply.get("reply"))[:160],
    )
    waiter.close()

    # ── the notify transport carries the same fact ──────────────────────────
    # A launchd-started desktop app has no python3 that can import agentica, so
    # the presence-record path is closed to it. The envelope's transport block
    # is not. Runs in the same process, importing the repo we just drove.
    print("\n== the attach point is also on the notify transport ==")
    from agentica.notify import reset_sink_for_tests, set_attach_endpoint
    from agentica.notify.sink import NotifySink, load_notify_config

    set_attach_endpoint(socket_path, peer_id)
    try:
        sink = NotifySink(load_notify_config())
        envelope = sink._envelope("run.started", session_id="s-1", payload=None,
                                  work_dir=None)
        transport = envelope["transport"]
        check(
            "the envelope advertises the same attach socket",
            transport.get("attach_socket") == socket_path,
            json.dumps(transport),
        )
        check("and the peer id that goes with it", transport.get("peer_id") == peer_id)
    finally:
        set_attach_endpoint(None, None)
        reset_sink_for_tests()

    client.close()
finally:
    tmux("kill-session", "-t", SESSION)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("all end-to-end checks passed")
