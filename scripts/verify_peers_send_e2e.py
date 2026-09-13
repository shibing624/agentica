#!/usr/bin/env python3
"""End-to-end: a real interactive CLI in tmux answers a message sent from a shell.

The completion criterion for `agentica peers send`: a plain external process sends
one line, and the session running in another terminal treats it as the user
speaking and answers.

Not in the pytest suite: it drives a real TUI in a real pty, needs a model, and
takes tens of seconds. Run it by hand:

    python scripts/verify_peers_send_e2e.py

It needs a working model config (~/.agentica/config.yaml or the provider env vars)
because the receiving session has to actually generate an answer. Set
PEERS_E2E_PROMPT to change what is asked (must be answerable in one short word).
"""
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

PROMPT = os.getenv("PEERS_E2E_PROMPT", "Reply with exactly: PONG")
EXPECTED = os.getenv("PEERS_E2E_EXPECT", "PONG")
SESSION = "peers-e2e"

failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def tmux(*args, **kw):
    return subprocess.run(["tmux", *args], capture_output=True, text=True, **kw)


def pane():
    return tmux("capture-pane", "-p", "-t", SESSION, "-S", "-120").stdout


if not shutil.which("tmux"):
    print("tmux not found; cannot run this check")
    sys.exit(2)

root = tempfile.mkdtemp(prefix="peers-e2e-")
home = os.path.join(root, ".agentica")
work = os.path.join(root, "proj")
os.makedirs(home, exist_ok=True)
os.makedirs(work, exist_ok=True)

# Run the code under test, not whatever `agentica` is installed as. The console
# script on PATH may be an editable install of another checkout (the main one),
# so the checkout this script lives in goes first on the import path.
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A separate AGENTICA_HOME keeps this run's peers, mailboxes and sessions out of
# the developer's real cache, so the check cannot see or disturb live sessions.
env = dict(os.environ, AGENTICA_HOME=home, PYTHONPATH=repo)


def cli(*args):
    """Run this checkout's CLI, with the isolated home."""
    return subprocess.run(
        [sys.executable, "-m", "agentica.cli.main", *args],
        env=env, capture_output=True, text=True, cwd=repo,
    )

tmux("kill-session", "-t", SESSION)
print(f"== starting a real interactive CLI in tmux ({SESSION}) ==")
# The env goes inside the command: a tmux session inherits the *server's*
# environment, which was set up long before this script and would otherwise send
# the CLI at the developer's real ~/.agentica.
launch = (
    f"AGENTICA_HOME={shlex.quote(home)} PYTHONPATH={shlex.quote(repo)} agentica"
)
tmux("new-session", "-d", "-s", SESSION, "-x", "200", "-y", "50",
     "-c", work, launch)
started = time.monotonic()
try:
    # Wait for the peer record to appear: that is the session being live and
    # addressable, which is the precondition for sending to it at all.
    peer_id = None
    listing = subprocess.CompletedProcess([], 1, "", "no attempt made")
    while time.monotonic() - started < 90:
        listing = cli("peers", "list")
        # The id is short: new_peer_id() takes uuid4().hex[:8] (agentica/peers.py).
        match = re.search(r"peer=([0-9a-f]+)", listing.stdout)
        if match:
            peer_id = match.group(1)
            break
        time.sleep(2)

    print(listing.stdout.strip()[:400])
    check("the tmux session registers as a live peer", bool(peer_id), str(peer_id))
    if not peer_id:
        raise SystemExit(1)

    name = re.search(r"(\S+)\s+\[peer=", listing.stdout)
    name = name.group(1) if name else peer_id
    print(f"\n== sending as the user via 'agentica peers send' (target {name}) ==")

    sent = subprocess.run(
        [sys.executable, "-m", "agentica.cli.main", "peers", "send",
         "--to", name, "--text", PROMPT],
        env=env, capture_output=True, text=True, cwd=repo,
    )
    print(sent.stdout.strip() or sent.stderr.strip())
    check("peers send exits 0", sent.returncode == 0, str(sent.returncode))

    # The pane should show the message arriving as the user's own line, and then
    # the model answering it.
    deadline = time.monotonic() + 180
    seen_user_line = seen_answer = False
    while time.monotonic() < deadline:
        text = pane()
        if not seen_user_line:
            seen_user_line = PROMPT in text
        if EXPECTED in text:
            seen_answer = True
            break
        time.sleep(3)

    tail = pane()[-2500:]
    print("\n== receiving pane (tail) ==")
    print(tail)
    check("the session shows the relayed line", seen_user_line)
    check("the session answered it", seen_answer)

    # An unknown target must fail loudly rather than silently going nowhere.
    bad = subprocess.run(
        [sys.executable, "-m", "agentica.cli.main", "peers", "send",
         "--to", "definitely-not-a-session", "--text", "hi"],
        env=env, capture_output=True, text=True, cwd=repo,
    )
    combined = bad.stdout + bad.stderr
    check("an unknown target exits non-zero", bad.returncode != 0, str(bad.returncode))
    check("and says why", "no live session matches" in combined, combined.strip()[:120])

    # And nothing was written for it: a refusal must not leave a stray mailbox.
    boxes = os.path.join(home, "cache", "peers", "mailbox")
    stray = []
    if os.path.isdir(boxes):
        for entry in os.listdir(boxes):
            stray.extend(os.listdir(os.path.join(boxes, entry)))
    check("a refused send writes no mailbox file", not stray, str(stray))
finally:
    tmux("kill-session", "-t", SESSION)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("all end-to-end checks passed")
