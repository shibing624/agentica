# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The hook wire format, as pure functions: a payload document out,
a reply parsed back.

No I/O lives here, so the protocol can be tested without a process and a
consumer's command is never in the test's critical path.

Two rules are decisions rather than formatting:

* Optional fields are **omitted, not null**. A missing key and an empty string
  are different things, and a consumer that must test both will eventually test
  one.
* An unrecognized reply is **no decision**, never a default. In particular it is
  never ``allow``: approving by accident is the one failure this channel must not
  have.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from agentica.notify.wire import ALLOWED_DECISIONS, clip_text


def build_payload(
    event: str,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    prompt: Optional[str] = None,
    options: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The JSON document handed to the hook on stdin.

    ``prompt`` is the run's **anchor text** — the user's message on an ordinary
    turn, the goal objective in a goal-driven session. A consumer may display it
    but must not read it as "what the user just typed".
    """
    import os

    doc: Dict[str, Any] = {"hook_event_name": event}
    if session_id:
        doc["session_id"] = str(session_id)
    clipped = clip_text(prompt)
    if clipped:
        doc["prompt"] = clipped
    cwd = work_dir or os.getcwd()
    if cwd:
        doc["cwd"] = str(cwd)
    if run_id:
        doc["run_id"] = str(run_id)
    if options:
        doc["options"] = [str(o) for o in options]
    for key, value in (extra or {}).items():
        if value is None or value == "" or value == []:
            continue
        doc[key] = value
    return doc


def parse_reply(stdout: Optional[str], event: str) -> Optional[Dict[str, Any]]:
    """A usable reply from the hook's stdout, or None for "no decision".

    Exit code is not consulted: a hook that fails to reply and one that declines
    to reply mean the same thing to us, because in both cases the user has not
    answered. The caller reads the terminal prompt instead.
    """
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    body = _first_json_document(stdout)
    if not isinstance(body, dict):
        # Includes malformed output and JSON that is not an object.
        return None

    if event == "needs.approval":
        decision = body.get("decision")
        if isinstance(decision, str) and decision in ALLOWED_DECISIONS:
            return {"decision": decision}
        return None

    if event == "needs.input":
        answer = body.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"answer": answer}
        return None

    # A reply to a notice is meaningless; treat it as no reply.
    return None


def _first_json_document(text: str) -> Any:
    """Parse the leading JSON document, ignoring anything after it."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    # A consumer that printed a document and then kept talking: take the first
    # line that parses rather than failing the whole reply.
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None
