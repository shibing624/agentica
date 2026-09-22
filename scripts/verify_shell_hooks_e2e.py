#!/usr/bin/env python3
"""End-to-end proof of the external hook egress, against the real code.

Not part of the pytest suite on purpose: it spawns real processes, waits on real
child exits, and sleeps long enough to be a poor fit for CI. Run it by hand
before claiming the hook contract holds, or after touching shell_hooks:

    python scripts/verify_shell_hooks_e2e.py

What it does, with no mocks:
  * writes a temporary AGENTICA_HOME whose settings.hooks points at a recording
    script, and drives the real config loader
  * emits a real run.started and asserts the document that reaches the process
  * parks a real approval in a real ApprovalRegistry and lets the hook answer it
  * races that against a terminal answer and asserts the hook is killed, leaves
    no zombie, and never reaches its own end
  * checks the degradation ladder: missing command, disabled
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

ROOT = tempfile.mkdtemp(prefix="agentica-hook-e2e-")
atexit.register(shutil.rmtree, ROOT, ignore_errors=True)
HOME = os.path.join(ROOT, ".agentica")
os.makedirs(HOME, exist_ok=True)
LOG = os.path.join(HOME, "hook-calls.jsonl")
HOOK = os.path.join(HOME, "hook.py")

# 1. the hook command: records every payload, answers approvals with "allow"
with open(HOOK, "w") as fh:
    fh.write(textwrap.dedent(f"""
        import json, sys, pathlib
        doc = json.load(sys.stdin)
        pathlib.Path({LOG!r}).open("a").write(json.dumps(doc) + "\\n")
        if doc.get("hook_event_name") == "needs.approval":
            print(json.dumps({{"request_id": doc["request_id"], "decision": "allow"}}))
        elif doc.get("hook_event_name") == "needs.input":
            print(json.dumps({{"request_id": doc["request_id"], "answer": "answered by the hook"}}))
        else:
            print(json.dumps({{"ok": True}}))
    """))
os.chmod(HOOK, 0o755)

with open(os.path.join(HOME, "config.yaml"), "w") as fh:
    fh.write(textwrap.dedent(f"""
        settings:
          hooks:
            enabled: true
            consumers:
              - name: e2e
                command: ["{sys.executable}", "{HOOK}"]
    """))

env = dict(os.environ, AGENTICA_HOME=HOME)
failures = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


print("== 1. config is read from a real config.yaml ==")
code = textwrap.dedent("""
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress, get_hook_egress
    cfg = load_shell_hooks_config()
    print("effective", cfg.effective)
    print("consumers", [(c.name, c.command) for c in cfg.consumers])
    install_hook_egress(cfg)
    print("installed", get_hook_egress() is not None)
""")
out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
check("config read from config.yaml", "installed True" in out.stdout)

print("\n== 2. a run.started reaches the hook process ==")
code = textwrap.dedent("""
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress
    from agentica.notify.sink import notify_sink_dispatch
    from agentica.run.events import RunEventRecord, RunEventType
    install_hook_egress(load_shell_hooks_config())
    notify_sink_dispatch(
        RunEventRecord(run_id="r1", event_type=RunEventType.run_started,
                       payload={"agent_name": "E2E", "prompt": "do the thing"}),
        session_id="sess-e2e", work_dir="/tmp",
    )
    # give the fire-and-forget process a moment
    import time as _t; _t.sleep(1.5)
""")
subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
calls = []
if os.path.exists(LOG):
    calls = [json.loads(line) for line in open(LOG) if line.strip()]
started = [c for c in calls if c.get("hook_event_name") == "run.started"]
check("run.started arrived", bool(started))
if started:
    doc = started[0]
    check("  carries hook_event_name", "hook_event_name" in doc)
    check("  carries session_id", doc.get("session_id") == "sess-e2e", str(doc.get("session_id")))
    check("  carries the anchor as prompt", doc.get("prompt") == "do the thing", str(doc.get("prompt")))
    check("  carries cwd", doc.get("cwd") == "/tmp", str(doc.get("cwd")))
    check("  metadata only (no arguments)", "arguments" not in doc)

print("\n== 3. a real approval, answered by the hook, resolves the registry ==")
code = textwrap.dedent("""
    import asyncio, threading, time
    from agentica.agent.approvals import ApprovalRegistry, PendingApproval
    from agentica.cli.approvals import _offer_approval_to_hook
    from agentica.cli.interactive.session_state import _InputRequest
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress

    install_hook_egress(load_shell_hooks_config())

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    class Agent:
        session_id = "sess-e2e"; work_dir = "/tmp"
        run_context = None; model = None; session_log = None
        class _T: permission_mode = "ask"
        tool_config = _T()
        class _A: source_query = "the user message"
        task_anchor = _A()

    class State:
        current_agent = Agent(); approval_registry = ApprovalRegistry()

    state = State()
    pending = PendingApproval(tool_call_id="call_e2e", name="execute",
                              arguments={"command": "rm -rf build"},
                              question="run it?", preview="rm -rf build",
                              options=("allow", "deny"))

    async def register(): return state.approval_registry.wait(pending)
    waiter = asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
    async def await_it(): return await waiter
    fut = asyncio.run_coroutine_threadsafe(await_it(), loop)

    req = _InputRequest(prompt="approve?", kind="approval", approval_id="call_e2e",
                        approval_pending=pending, hook_request_id="request-e2e")
    state.input_request = req
    _offer_approval_to_hook(pending, state, loop, req)
    print("decision:", fut.result(timeout=15))
""")
out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
check("the hook's 'allow' became the approval decision", "decision: allow" in out.stdout)

print("\n== 4. the terminal answering first kills the hook ==")
trickle = os.path.join(HOME, "slow_hook.py")
marker = os.path.join(HOME, "slow-hook-finished")
pidfile = os.path.join(HOME, "slow-hook.pid")
with open(trickle, "w") as fh:
    fh.write(textwrap.dedent(f"""
        import json, time, pathlib, os, sys
        doc = json.load(sys.stdin)
        pathlib.Path({pidfile!r}).write_text(str(os.getpid()))
        time.sleep(4)
        pathlib.Path({marker!r}).write_text("finished")
        print(json.dumps({{"request_id": doc["request_id"], "decision": "deny"}}))
    """))
os.chmod(trickle, 0o755)
with open(os.path.join(HOME, "config.yaml"), "w") as fh:
    fh.write(textwrap.dedent(f"""
        settings:
          hooks:
            enabled: true
            consumers:
              - name: slow
                command: ["{sys.executable}", "{trickle}"]
                events:
                  needs.resolved: false
    """))
code = textwrap.dedent("""
    import asyncio, threading, time
    from agentica.agent.approvals import ApprovalRegistry, PendingApproval
    from agentica.cli.approvals import _offer_approval_to_hook
    from agentica.cli.interactive.session_state import _InputRequest
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress

    install_hook_egress(load_shell_hooks_config())
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    class Agent:
        session_id = "s"; work_dir = "/tmp"; task_anchor = None; run_context = None
        model = None; session_log = None
        class _T: permission_mode = "ask"
        tool_config = _T()
    class State:
        current_agent = Agent(); approval_registry = ApprovalRegistry()
    state = State()
    pending = PendingApproval(tool_call_id="c1", name="execute", arguments={},
                              question="q", preview="p", options=("allow", "deny"))

    async def register(): return state.approval_registry.wait(pending)
    waiter = asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
    async def await_it(): return await waiter
    fut = asyncio.run_coroutine_threadsafe(await_it(), loop)

    req = _InputRequest(prompt="approve?", kind="approval", approval_id="c1",
                        approval_pending=pending, hook_request_id="request-slow")
    state.input_request = req
    _offer_approval_to_hook(pending, state, loop, req)

    # the user types y in the terminal, on the loop thread (as the TUI does)
    async def decide(): return state.approval_registry.decide("c1", "allow")
    print("terminal decision applied:", asyncio.run_coroutine_threadsafe(decide(), loop).result(timeout=5))
    print("resolved to:", fut.result(timeout=10))
    time.sleep(6)
""")
out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
check("terminal answer stands", "resolved to: allow" in out.stdout)
check("the killed hook never reached its end", not os.path.exists(marker))

# Probe the hook's own pid. Grepping "ps -eo command" for the script name would
# match this very checker (its -c argument contains that name), which is the
# classic ps self-match; the pid is unambiguous.
if os.path.exists(pidfile):
    hook_pid = int(open(pidfile).read().strip())
    ps = subprocess.run(["ps", "-eo", "pid=,stat=,command="],
                        capture_output=True, text=True).stdout
    mine = [line.strip() for line in ps.splitlines()
            if line.split()[:1] == [str(hook_pid)]]
    check("the hook process is gone (no zombie left)", not mine, str(mine))
else:
    check("the hook recorded its pid before sleeping", False, "pidfile missing")

print("\n== 5. a missing command degrades silently (no decision, no crash) ==")
with open(os.path.join(HOME, "config.yaml"), "w") as fh:
    fh.write(textwrap.dedent("""
        settings:
          hooks:
            enabled: true
            consumers:
              - name: missing
                command: ["/nonexistent/hook-binary-xyz"]
    """))
code = textwrap.dedent("""
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress, start_hook_request
    install_hook_egress(load_shell_hooks_config())
    print("request:", start_hook_request("needs.approval",
          {"hook_event_name": "needs.approval", "request_id": "missing"}))
""")
out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
check("missing command yields no request and no crash", "request: None" in out.stdout)

print("\n== 6. disabled means nothing is forked ==")
with open(os.path.join(HOME, "config.yaml"), "w") as fh:
    fh.write("settings:\n  hooks:\n    enabled: false\n    consumers:\n      - name: off\n        command: [\"/bin/true\"]\n")
before = set(os.listdir("/tmp"))
code = textwrap.dedent("""
    from agentica.shell_hooks import load_shell_hooks_config, install_hook_egress, get_hook_egress
    from agentica.notify.sink import notify_sink_dispatch
    from agentica.run.events import RunEventRecord, RunEventType
    install_hook_egress(load_shell_hooks_config())
    print("egress:", get_hook_egress())
    notify_sink_dispatch(RunEventRecord(run_id="r", event_type=RunEventType.run_started), session_id="s")
""")
out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
check("disabled installs nothing", "egress: None" in out.stdout)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("all end-to-end checks passed")
