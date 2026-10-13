# -*- coding: utf-8 -*-
"""A scripted ACP agent for tests: newline-delimited JSON-RPC on stdin/stdout.

The prompt text picks the behaviour. Every message it receives is appended to
the JSON-lines file named by ``FAKE_ACP_RECORD``.
"""
import json
import os
import sys
import threading

RECORD = os.environ["FAKE_ACP_RECORD"]
_lock = threading.Lock()
_next_id = [1000]
_waiting = {}
_cancelled = threading.Event()


def record(entry):
    with open(RECORD, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def send(message):
    with _lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def update(session_id, payload):
    send({"jsonrpc": "2.0", "method": "session/update",
          "params": {"sessionId": session_id, "update": payload}})


def chunk(session_id, text):
    update(session_id, {"sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text}})


def ask(method, params):
    """Agent -> client request; block until the client answers."""
    with _lock:
        _next_id[0] += 1
        request_id = _next_id[0]
        event = threading.Event()
        _waiting[request_id] = [event, None]
    send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
    event.wait()
    return _waiting.pop(request_id)[1]


def run_prompt(request_id, params):
    session_id = params["sessionId"]
    text = params["prompt"][0]["text"]
    stop = "end_turn"
    if text.startswith("echo:"):
        body = text[len("echo:"):]
        half = len(body) // 2
        chunk(session_id, body[:half])
        update(session_id, {"sessionUpdate": "tool_call", "toolCallId": "t1",
                            "title": "printf hi > a.txt", "kind": "execute",
                            "status": "in_progress"})
        update(session_id, {"sessionUpdate": "tool_call_update", "toolCallId": "t1",
                            "status": "completed"})
        update(session_id, {"sessionUpdate": "plan", "entries": [
            {"content": "write a.txt", "status": "completed", "priority": "high"}]})
        chunk(session_id, body[half:])
    elif text == "pid":
        chunk(session_id, f"{os.getpid()} {session_id}")
    elif text == "permission":
        reply = ask("session/request_permission", {
            "sessionId": session_id,
            "toolCall": {"toolCallId": "p1", "title": "rm -rf build", "kind": "delete",
                         "locations": [{"path": "/w/build"}]},
            "options": [
                {"optionId": "always", "name": "Always", "kind": "allow_always"},
                {"optionId": "once", "name": "Allow", "kind": "allow_once"},
                {"optionId": "no", "name": "Reject", "kind": "reject_once"},
            ],
        })
        record({"permission_reply": reply})
        chunk(session_id, json.dumps(reply))
    elif text == "unknown":
        reply = ask("fs/read_text_file", {"sessionId": session_id, "path": "/etc/hosts"})
        record({"unknown_reply": reply})
        chunk(session_id, "still here")
    elif text == "crash":
        chunk(session_id, "about to crash")
        sys.stderr.write("fake agent: boom\n")
        sys.stderr.flush()
        os._exit(3)
    elif text == "hang":
        chunk(session_id, "working")
        _cancelled.wait()
        _cancelled.clear()
        stop = "cancelled"
    elif text == "big":
        chunk(session_id, "x" * 300_000)
    send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": stop}})


def main():
    if os.environ.get("FAKE_ACP_GRANDCHILD"):
        # What npx does: the agent is a child of the process we started.
        import subprocess
        child = subprocess.Popen(["sleep", "300"])
        record({"grandchild": child.pid})
    for raw in sys.stdin:
        message = json.loads(raw)
        record({"method": message.get("method"), "id": message.get("id")})
        if "method" not in message:  # a reply to one of our requests
            entry = _waiting[message["id"]]
            entry[1] = message
            entry[0].set()
            continue
        method = message["method"]
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": 1, "agentCapabilities": {}, "authMethods": []}})
        elif method == "session/new":
            # Not JSON-RPC: a stray log line on stdout must not break the client.
            sys.stdout.write("fake agent log line\n")
            sys.stdout.flush()
            send({"jsonrpc": "2.0", "id": message["id"],
                  "result": {"sessionId": f"s-{os.getpid()}"}})
        elif method == "session/prompt":
            threading.Thread(target=run_prompt, args=(message["id"], message["params"]),
                             daemon=True).start()
        elif method == "session/cancel":
            _cancelled.set()


if __name__ == "__main__":
    main()
