# External Hook Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a user-configured command at lifecycle points, hand it JSON on stdin and take the user's reply on stdout, so any external desktop app / pet can integrate by installing a binary instead of implementing agentica's private socket protocol.

**Architecture:** A new `agentica/shell_hooks/` package owns the transport (config, wire, process). It is a **second egress** fanned out from the same dispatch point as the existing notify sink (`notify/sink.py`), so the goal-deferral logic for `run.completed` is shared rather than reimplemented. The blocking `needs.*` replies use one rule kept from the sink: no wait cap invented here, and the terminal and the hook race, first answer wins. `/await` (the sink's reply half) is deleted at the end, once the hook path is green.

**Tech Stack:** Python 3, `subprocess` + `start_new_session` + `os.killpg`, `threading`, `select`, pytest.

**Plan status vs. the RFC.** This plan follows `docs/rfcs/external-hook-egress.md` in shape, and corrects four things the RFC asserts that the code does not do. Every correction was verified in this worktree:

| RFC says | Actually |
|---|---|
| `run_events.py` `run_started` is the emit source; `run_events.py` "states the discipline" | The file is `agentica/run_events.py` (no `runner/`). The emit site is `runner/loop.py:605`; `_emit_event` is `runner/core.py:53`. |
| `options` is "`PendingApproval.options` after the CLI narrowed it via `visible_approval_decisions` (`cli/approvals.py:46-51`)" | `visible_approval_decisions` is called at **render** time (`cli/approvals.py:311`), never in the payload path. `PendingApproval.options` is already the narrowed tuple from `_approval_options` (`agent/approvals.py:759`) — that is what the sink sends and what we send. No extra narrowing. |
| `needs.*` payload carries `prompt` | It does not. `_approval_payload` (`notify/approvals.py:34-63`) has no `prompt`; `ask_via_desktop` has none either. `prompt` reaches the wire only on run events, via `_run_event_payload` from `source_query`. The hook builder must add it. |
| "the approval machinery parks on the `ApprovalRegistry` future ... `notify/approvals.py:172` is reused as-is" | That branch is reused for approvals. For **questions** there is no future at all: `wrap_ask_callback` calls `ask_via_desktop` **synchronously before** `inner(...)` (`notify/questions.py:90-98`), so a non-answering desktop blocks the terminal prompt out entirely — sequential, not a race. Task 6 fixes it by arming the terminal first. |

Deleting `/await` in the same change is the user's instruction and deviates from the RFC's own phasing (RFC Phases says "not deleting the sink in this change"). Consequence to accept knowingly: until VPetMac lands its adapter, the real desktop can **observe** but not **answer**.

---

## File Structure

| File | Responsibility |
|---|---|
| `agentica/shell_hooks/__init__.py` | Package exports. |
| `agentica/shell_hooks/config.py` | `settings.hooks` + `AGENTICA_HOOKS_*` → `ShellHooksConfig`. Read once, at install time. |
| `agentica/notify/wire.py` | The payload discipline both egresses share: clipping + the four decision words. One home, so the two cannot drift. |
| `agentica/shell_hooks/protocol.py` | Pure: payload document → dict, reply stdout → decision/answer. No I/O. |
| `agentica/shell_hooks/process.py` | One hook invocation: spawn, stdin, stdout reader, `killpg`. No protocol knowledge beyond "is this JSON". |
| `agentica/shell_hooks/egress.py` | Install-time decision + `hook_egress_dispatch` (fire-and-forget events). |
| `agentica/shell_hooks/requests.py` | The blocking `needs.*` path: `HookRequest` — spawn, poll, kill, never impose a cap. |
| `agentica/notify/sink.py` | **Modify**: fan out events to both egresses; drop the `/await` half in Task 7. |
| `agentica/cli/approvals.py` | **Modify**: replace the sink's `publish_approval` with the hook request. |
| `agentica/cli/interactive/app.py` | **Modify**: install the egress; make the ask path arm the terminal *first*, then race the hook. |

---

### Task 1: Config

**Files:**
- Create: `agentica/shell_hooks/__init__.py`
- Create: `agentica/shell_hooks/config.py`
- Test: `tests/shell_hooks/test_shell_hooks_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shell_hooks/test_shell_hooks_config.py
# -*- coding: utf-8 -*-
"""Config resolution for the external hook egress."""

from __future__ import annotations

import pytest

from agentica.shell_hooks.config import (
    SHELL_HOOK_EVENTS,
    load_shell_hooks_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ENABLED", "COMMAND", "TIMEOUT"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{name}", raising=False)


def _cfg(block):
    return load_shell_hooks_config({"settings": {"hooks": block}})


class TestDisabledByDefault:
    def test_no_settings_block_means_disabled(self):
        cfg = load_shell_hooks_config({})
        assert cfg.enabled is False
        assert cfg.command == []

    def test_enabled_without_a_command_is_still_ineffective(self):
        """enabled is not enough: without an argv there is nothing to run."""
        cfg = _cfg({"enabled": True})
        assert cfg.command == []


class TestCommandIsArgv:
    def test_a_list_is_taken_verbatim(self):
        cfg = _cfg({"command": ["/abs/notifier", "--from-agentica"]})
        assert cfg.command == ["/abs/notifier", "--from-agentica"]

    def test_a_string_is_refused_not_shell_split(self):
        """A string is a config mistake. Splitting it would invent quoting rules
        and silently run a different argv than the user wrote."""
        cfg = _cfg({"command": "/abs/notifier --flag"})
        assert cfg.command == []

    def test_blank_entries_are_dropped(self):
        cfg = _cfg({"command": ["/abs/notifier", "", "  "]})
        assert cfg.command == ["/abs/notifier"]


class TestEvents:
    def test_all_six_default_on(self):
        cfg = _cfg({"enabled": True})
        assert set(cfg.events) == set(SHELL_HOOK_EVENTS)
        assert all(cfg.event_enabled(e) for e in SHELL_HOOK_EVENTS)

    def test_an_event_can_be_switched_off(self):
        cfg = _cfg({"events": {"run.started": False}})
        assert cfg.event_enabled("run.started") is False
        assert cfg.event_enabled("run.completed") is True

    def test_an_unknown_event_name_is_ignored_not_added(self):
        cfg = _cfg({"events": {"tool.before": True}})
        assert "tool.before" not in cfg.events


class TestEnvOverrides:
    def test_env_beats_config(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_ENABLED", "1")
        monkeypatch.setenv("AGENTICA_HOOKS_COMMAND", "/from/env")
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": False, "command": ["/from/config"]}}}
        )
        assert cfg.enabled is True
        assert cfg.command == ["/from/env"]

    def test_empty_env_is_unset(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_COMMAND", "   ")
        cfg = _cfg({"command": ["/from/config"]})
        assert cfg.command == ["/from/config"]


class TestTimeout:
    def test_default_is_none_meaning_no_cap_of_ours(self):
        assert _cfg({"enabled": True}).timeout is None

    def test_a_number_is_read(self):
        assert _cfg({"timeout": 12}).timeout == 12.0

    def test_junk_is_no_cap(self):
        assert _cfg({"timeout": "soon"}).timeout is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentica.shell_hooks'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/shell_hooks/config.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Configuration for the external hook egress.

Read once at install time, like the notify sink: an egress is either wired or it
is not, and a mid-run config flip would leave half a channel behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.global_config import get_setting
from agentica.utils.log import logger

#: The six events on this wire. Four are lifecycle notices; the two ``needs.*``
#: ones take the other path (a request with a reply) and do not go through the
#: fire-and-forget dispatch, but they are gated by the same ``events`` block so a
#: user has one place to switch things off.
SHELL_HOOK_EVENTS = (
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "needs.approval",
    "needs.input",
)


def _env(name: str) -> Optional[str]:
    """Read ``AGENTICA_HOOKS_<name>``, treating empty as unset."""
    value = os.getenv(f"AGENTICA_HOOKS_{name}")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str) -> Optional[bool]:
    value = _env(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on")


def _parse_command(raw: Any) -> List[str]:
    """The command as an argv list.

    A list is the only accepted shape. A string is refused rather than split:
    splitting would invent quoting rules the user did not write, and the wire
    format is a JSON document, so a shell is never needed to pass it. A user who
    wants a shell writes ``["/bin/sh", "-c", "..."]`` explicitly, which is also
    how they recover things we do not send (``$PPID``, the controlling tty).
    """
    if isinstance(raw, str):
        if raw.strip():
            logger.warning(
                "shell hooks: settings.hooks.command must be an argv list, not a "
                "string; ignoring it. Write [\"/abs/path/to/notifier\"] — a shell "
                "is not implied, and $HOME is not expanded here."
            )
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(part).strip() for part in raw if str(part).strip()]


def _parse_timeout(raw: Any) -> Optional[float]:
    """Seconds, or None for "no cap of ours".

    None is the honest default and the one that matches the ``needs.*`` rule: a
    number baked into this layer would mean a desktop answer was allowed less
    time than a typed one.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


@dataclass
class ShellHooksConfig:
    """Resolved hook egress configuration."""

    enabled: bool = False
    command: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    events: Dict[str, bool] = field(
        default_factory=lambda: {e: True for e in SHELL_HOOK_EVENTS}
    )

    def event_enabled(self, event: str) -> bool:
        return bool(self.events.get(event, False))

    @property
    def effective(self) -> bool:
        """Wired only when it is switched on *and* there is something to run."""
        return bool(self.enabled and self.command)


def load_shell_hooks_config(config: Optional[Dict[str, Any]] = None) -> ShellHooksConfig:
    """Resolve ``settings.hooks`` plus env overrides. Env beats config.yaml."""
    try:
        block = get_setting("hooks", {}, config=config)
    except Exception as exc:  # a broken config must not break startup
        logger.debug(f"shell hooks: could not read settings.hooks: {exc}")
        block = {}
    if not isinstance(block, dict):
        block = {}

    cfg = ShellHooksConfig()
    if "enabled" in block:
        cfg.enabled = bool(block["enabled"])
    cfg.command = _parse_command(block.get("command"))
    cfg.timeout = _parse_timeout(block.get("timeout"))
    events = block.get("events")
    if isinstance(events, dict):
        for name in SHELL_HOOK_EVENTS:
            if name in events:
                cfg.events[name] = bool(events[name])

    if _env_bool("ENABLED") is not None:
        cfg.enabled = bool(_env_bool("ENABLED"))
    command = _env("COMMAND")
    if command is not None:
        cfg.command = _parse_command(command.split())
    if _env("TIMEOUT") is not None:
        cfg.timeout = _parse_timeout(_env("TIMEOUT"))

    return cfg
```

```python
# agentica/shell_hooks/__init__.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External shell hooks — run the user's own command at lifecycle
points and hand it JSON on stdin, the way every other coding CLI does.

This is the *executable* kind of hook. It is not ``agentica/hooks.py``
(``AgentHooks`` / ``RunHooks``), which are in-process Python observers and stay
as they are. See ``docs/rfcs/external-hook-egress.md``.
"""

from agentica.shell_hooks.config import ShellHooksConfig, load_shell_hooks_config

__all__ = ["ShellHooksConfig", "load_shell_hooks_config"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_config.py -q`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add agentica/shell_hooks/ tests/shell_hooks/test_shell_hooks_config.py
git commit -m "shell hooks: settings.hooks config (argv only, env overrides)"
```

---

### Task 2: The shared wire discipline + protocol

**Files:**
- Create: `agentica/notify/wire.py`
- Create: `agentica/shell_hooks/protocol.py`
- Modify: `agentica/notify/sink.py:648-665` (move clipping out), `:80` (move the four words out)
- Test: `tests/shell_hooks/test_shell_hooks_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shell_hooks/test_shell_hooks_protocol.py
# -*- coding: utf-8 -*-
"""The hook wire: payload documents out, replies in.

Every case here is a "would silently do the wrong thing" case: a reply that
cannot be understood must be no decision, never an allow.
"""

from __future__ import annotations

import json

from agentica.shell_hooks.protocol import build_payload, parse_reply


class TestPayload:
    def test_the_event_name_is_in_the_json_not_argv(self):
        doc = build_payload("needs.approval", session_id="s1", work_dir="/w")
        assert doc["hook_event_name"] == "needs.approval"

    def test_optional_fields_are_omitted_not_null(self):
        doc = build_payload("run.started", session_id="s1")
        assert "question" not in doc
        assert "options" not in doc
        assert "tool_call_id" not in doc

    def test_an_approval_carries_its_correlation_id_and_options(self):
        doc = build_payload(
            "needs.approval",
            session_id="s1",
            work_dir="/w",
            extra={
                "tool_name": "execute",
                "tool_call_id": "call_1",
                "options": ["allow", "deny"],
                "question": "run it?",
                "preview": "rm -rf build",
            },
        )
        assert doc["tool_call_id"] == "call_1"
        assert doc["options"] == ["allow", "deny"]

    def test_a_question_carries_no_decision_vocabulary(self):
        """The reply to needs.input is free text; offering the four approval
        words would invite a consumer to answer with one."""
        doc = build_payload("needs.input", session_id="s1", extra={"question": "which?"},
                            options=("date-fns",))
        assert "decision" not in doc
        assert doc["options"] == ["date-fns"]

    def test_prompt_is_clipped(self):
        doc = build_payload("needs.approval", session_id="s1", prompt="x" * 900)
        assert len(doc["prompt"]) < 900

    def test_cwd_defaults_to_the_process_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        doc = build_payload("run.started", session_id="s1")
        assert doc["cwd"] == str(tmp_path)


class TestReply:
    def test_a_decision(self):
        assert parse_reply('{"decision": "allow"}', "needs.approval") == {"decision": "allow"}

    def test_an_answer(self):
        assert parse_reply('{"answer": "date-fns"}', "needs.input") == {"answer": "date-fns"}

    def test_all_four_decision_words(self):
        for word in ("allow", "allow_prefix", "deny", "deny_prefix"):
            assert parse_reply(json.dumps({"decision": word}), "needs.approval") == {
                "decision": word
            }

    def test_an_unknown_decision_word_is_no_decision(self):
        assert parse_reply('{"decision": "sure"}', "needs.approval") is None

    def test_an_approval_body_is_not_an_answer(self):
        """``{decision: allow}`` is not a reply to a question, and must not be
        read as the string "allow"."""
        assert parse_reply('{"decision": "allow"}', "needs.input") is None

    def test_a_non_approval_body_is_not_a_decision(self):
        assert parse_reply('{"answer": "yes"}', "needs.approval") is None

    def test_blank_or_missing_is_no_decision(self):
        for text in ("", "   ", None, "not json", "{}", "[]", '{"decision": null}'):
            assert parse_reply(text, "needs.approval") is None

    def test_an_empty_answer_is_not_an_answer(self):
        assert parse_reply('{"answer": "   "}', "needs.input") is None

    def test_trailing_noise_is_tolerated_when_a_document_is_present(self):
        assert parse_reply('{"decision": "deny"}\n', "needs.approval") == {"decision": "deny"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentica.shell_hooks.protocol'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/notify/wire.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The payload discipline both external egresses share.

The notify sink and the hook egress put the same user-visible strings on a wire
that leaves the process, so "how much text, and which words count as a decision"
lives in exactly one place: two copies would drift, and the drift would be
invisible until a consumer rendered a half-prompt or accepted a fifth word.
"""

from __future__ import annotations

from typing import Any, Optional

#: How much of a prompt / question / answer goes on the wire. A consumer renders
#: a bubble, not a reader. The marker makes the cut visible, so a consumer can
#: tell "it said this much" from "it said 500 chars and more".
TEXT_LIMIT = 500
ELLIPSIS = "…"

#: Decisions a reply may carry back. Anything else is "no decision" rather than
#: being coerced — guessing here would approve a command.
ALLOWED_DECISIONS = frozenset({"allow", "deny", "allow_prefix", "deny_prefix"})


def clip_text(value: Any) -> Optional[str]:
    """A short, wire-safe slice of user-visible text, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= TEXT_LIMIT:
        return text
    return text[:TEXT_LIMIT] + ELLIPSIS
```

```python
# agentica/shell_hooks/protocol.py
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
  never ``allow``: approving by accident is the one failure this channel must
  not have.
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

    decision = body.get("decision")
    if event == "needs.approval":
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
```

Then in `agentica/notify/sink.py`: replace the local definitions with imports from `wire`.

```python
# sink.py, replacing the local _TEXT_LIMIT/_ELLIPSIS/_clip_text block:
from agentica.notify.wire import ALLOWED_DECISIONS as _ALLOWED_DECISIONS, clip_text as _clip_text
```

and delete lines 648-665 (`_TEXT_LIMIT`, `_ELLIPSIS`, `_clip_text`). Keep the call
sites using `_clip_text(...)`, so the rename is one import plus a deletion.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/shell_hooks/ tests/notify/ -q`
Expected: PASS — the new protocol tests, and every existing notify test unchanged (proving the sink still clips the same way).

- [ ] **Step 5: Commit**

```bash
git add agentica/notify/wire.py agentica/notify/sink.py agentica/shell_hooks/protocol.py tests/shell_hooks/test_shell_hooks_protocol.py
git commit -m "shell hooks: wire protocol shared with the notify sink (one payload discipline)"
```

---

### Task 3: One hook invocation

**Files:**
- Create: `agentica/shell_hooks/process.py`
- Test: `tests/shell_hooks/test_shell_hooks_process.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shell_hooks/test_shell_hooks_process.py
# -*- coding: utf-8 -*-
"""Spawning the user's command: real processes, because the failure modes are
process failure modes (a non-zero exit, an empty stdout, a child that outlives
its parent)."""

from __future__ import annotations

import os
import sys
import time

from agentica.shell_hooks.process import HookProcess


def _py(body: str):
    return [sys.executable, "-c", body]


class TestTheDocumentArrivesOnStdin:
    def test_the_hook_sees_the_payload(self):
        reader = _py(
            "import json,sys;"
            "doc=json.load(sys.stdin);"
            "print(json.dumps({'answer': doc['hook_event_name']}))"
        )
        proc = HookProcess(reader, {"hook_event_name": "needs.input"})
        assert proc.start() is True
        assert proc.wait(timeout=10) is True
        assert proc.stdout == '{"answer": "needs.input"}\n'


class TestFailuresAreNotDecisions:
    def test_a_non_zero_exit_still_returns_whatever_was_printed(self):
        proc = HookProcess(_py("import sys; print(''); sys.exit(3)"), {})
        assert proc.start() is True
        assert proc.wait(timeout=10) is True
        assert proc.stdout is not None  # the caller decides; see parse_reply

    def test_a_missing_command_does_not_raise(self):
        proc = HookProcess(["/nonexistent/notifier-xyz"], {})
        assert proc.start() is False

    def test_an_empty_command_does_not_raise(self):
        assert HookProcess([], {}).start() is False

    def test_a_hook_that_never_reads_stdin_does_not_hang_the_spawn(self):
        """A consumer that exits without reading must produce a broken pipe, not
        a wedged run."""
        proc = HookProcess(_py("raise SystemExit(0)"), {"k": "v" * 40_000_000})
        assert proc.start() in (True, False)
        proc.kill()


class TestKilling:
    def test_kill_is_idempotent_and_leaves_nothing_running(self):
        proc = HookProcess(_py("import time; time.sleep(60)"), {})
        assert proc.start() is True
        pid = proc.pid
        proc.kill()
        proc.kill()
        time.sleep(0.3)
        assert not _alive(pid)

    def test_kill_takes_the_whole_group(self):
        """A hook that spawned its own child must not leave it behind."""
        child = _py("import time; time.sleep(60)")
        body = (
            "import subprocess,sys,time;"
            f"subprocess.Popen({child!r});"
            "time.sleep(60)"
        )
        proc = HookProcess(_py(body), {})
        assert proc.start() is True
        time.sleep(0.8)  # let the grandchild exist
        proc.kill()
        time.sleep(0.3)
        assert _group_empty(proc.pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _group_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except OSError:
        return True
    return False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_process.py -q`
Expected: FAIL — `No module named 'agentica.shell_hooks.process'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/shell_hooks/process.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: One hook invocation: spawn the user's command, hand it JSON on
stdin, read one JSON document back on stdout.

``stdout`` is read on its own thread for every invocation, including the
fire-and-forget notices. That is not symmetry for its own sake: an unread pipe
fills and the child blocks writing to it, so a consumer that prints anything
would otherwise wedge a run. Nothing here interprets the reply — ``protocol``
does that — so this module's only opinion is "is this a JSON document yet".

Process group: ``start_new_session=True`` plus ``os.killpg``. A consumer that
spawns its own children would otherwise survive the kill and keep our pipe open.
The package already relies on this pattern (``execute_tool.py``,
``utils/async_utils.py``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from typing import Dict, List, Optional, Sequence

from agentica.utils.log import logger

#: An upper bound on how much stdout we will accumulate. A consumer prints one
#: small document; anything past this is a consumer that is not speaking the
#: protocol, and reading it forever would only leak.
MAX_OUTPUT_BYTES = 64 * 1024

_READ_CHUNK = 4096


def kill_process_group(proc: Optional[subprocess.Popen]) -> None:
    """SIGKILL the whole group. Safe to call twice, and safe after exit."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:  # already gone
            pass


class HookProcess:
    """One in-flight hook invocation.

    Lifecycle: ``start()`` → poll ``finished`` / ``stdout`` → ``kill()`` when the
    answer is no longer wanted (the user answered in the terminal, the turn was
    cancelled, the CLI is exiting). ``kill()`` is idempotent and is what closes
    stdout, so a consumer that is still talking cannot block the reader forever.
    """

    def __init__(
        self,
        command: Sequence[str],
        payload: Dict[str, object],
        *,
        env: Optional[Dict[str, str]] = None,
    ):
        self._command: List[str] = list(command)
        self._payload = payload
        self._env = env
        self._proc: Optional[subprocess.Popen] = None
        self._stdout = ""
        self._done = threading.Event()
        self._killed = False

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def started(self) -> bool:
        return self._proc is not None

    @property
    def stdout(self) -> Optional[str]:
        """Whatever the consumer printed, once the reader has finished."""
        return self._stdout or None

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def start(self) -> bool:
        """Spawn and feed stdin. False means the hook could not be run at all.

        Never raises: "the hook is not installed / not executable" is a normal
        configuration state and must degrade to the terminal prompt, not to a
        failed run.
        """
        if not self._command or not self._command[0]:
            return False
        env = None
        if self._env:
            env = dict(os.environ)
            env.update({k: str(v) for k, v in self._env.items()})
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except (OSError, ValueError) as exc:
            logger.debug(f"shell hooks: could not spawn {self._command!r}: {exc}")
            self._proc = None
            return False

        try:
            data = json.dumps(self._payload, ensure_ascii=False).encode("utf-8")
            self._proc.stdin.write(data)
            self._proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            # The consumer exited without reading its stdin. That is "no reply".
            logger.debug(f"shell hooks: could not write the payload: {exc}")
            self.kill()
            return False

        threading.Thread(
            target=self._read_stdout, name="agentica-hook-reader", daemon=True
        ).start()
        return True

    def _read_stdout(self) -> None:
        """Read until one JSON document has arrived, the pipe closes, or the cap."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._done.set()
            return
        buf = b""
        try:
            while len(buf) < MAX_OUTPUT_BYTES:
                chunk = proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                if _is_json_document(buf):
                    break
        except (OSError, ValueError) as exc:
            logger.debug(f"shell hooks: stdout read failed: {exc}")
        finally:
            self._stdout = buf.decode("utf-8", errors="replace")
            self._done.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the reader. True when it finished within ``timeout``."""
        return self._done.wait(timeout)

    def kill(self) -> None:
        if self._killed:
            return
        self._killed = True
        kill_process_group(self._proc)
        try:
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()
        except Exception:  # closing an already-closed pipe
            pass


def _is_json_document(buf: bytes) -> bool:
    """Has a complete JSON document arrived?

    A consumer may print its document and then keep the pipe open (a wrapper that
    waits for its own children, say). Waiting for EOF there would hold a thread
    until someone killed the process, so the document itself ends the read.
    """
    text = buf.decode("utf-8", errors="replace").strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_process.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentica/shell_hooks/process.py tests/shell_hooks/test_shell_hooks_process.py
git commit -m "shell hooks: run one hook command in its own process group"
```

---

### Task 4: Fan out the four lifecycle events

**Files:**
- Create: `agentica/shell_hooks/egress.py`
- Modify: `agentica/notify/sink.py:455-547` (fan out, drop the `sink is None` early return)
- Modify: `agentica/cli/interactive/app.py` (install next to `install_notify_sink`)
- Test: `tests/shell_hooks/test_shell_hooks_egress.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shell_hooks/test_shell_hooks_egress.py
# -*- coding: utf-8 -*-
"""The lifecycle fan-out: one event reaches every installed egress, and a broken
consumer cannot affect the sink or the run."""

from __future__ import annotations

import json
import sys
import time

import pytest

from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import (
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)
from agentica.run_events import RunEventRecord, RunEventType


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


class _SinkSpy:
    def __init__(self):
        self.events = []

    def emit_event(self, event, *, session_id=None, payload=None, work_dir=None):
        self.events.append((event, payload))


def _recorder(tmp_path):
    """A hook command that appends each payload to a file."""
    out = tmp_path / "seen.jsonl"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json,sys\n"
        f"open({str(out)!r},'a').write(json.dumps(json.load(sys.stdin))+'\\n')\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)], out


def _wait_for(path, count, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) >= count:
                return [json.loads(l) for l in lines]
        time.sleep(0.05)
    raise AssertionError(f"{path} never reached {count} payloads")


class TestInstall:
    def test_disabled_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=False)) is None
        assert get_hook_egress() is None

    def test_enabled_without_a_command_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=True, command=[])) is None

    def test_enabled_with_a_command_is_wired(self):
        cfg = install_hook_egress(ShellHooksConfig(enabled=True, command=["/bin/true"]))
        assert cfg is not None
        assert get_hook_egress() is not None


class TestDispatch:
    def test_an_event_reaches_the_command(self, tmp_path):
        command, out = _recorder(tmp_path)
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
        hook_egress_dispatch(
            "run.started",
            {"agent_name": "Agent", "prompt": "do the thing"},
            session_id="sess-1",
            work_dir="/w",
        )
        doc = _wait_for(out, 1)[0]
        assert doc["hook_event_name"] == "run.started"
        assert doc["session_id"] == "sess-1"
        assert doc["prompt"] == "do the thing"
        assert doc["cwd"] == "/w"

    def test_a_switched_off_event_is_not_sent(self, tmp_path):
        command, out = _recorder(tmp_path)
        install_hook_egress(
            ShellHooksConfig(enabled=True, command=command, events={"run.started": False})
        )
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        docs = _wait_for(out, 1)
        assert [d["hook_event_name"] for d in docs] == ["run.completed"]

    def test_no_egress_is_a_no_op(self):
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_broken_command_is_swallowed(self):
        install_hook_egress(ShellHooksConfig(enabled=True, command=["/nonexistent/xyz"]))
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_hanging_command_does_not_block_the_caller(self, tmp_path):
        install_hook_egress(
            ShellHooksConfig(
                enabled=True,
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
            )
        )
        started = time.monotonic()
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert time.monotonic() - started < 1.0


class TestTheSinkStillWorks:
    def test_the_sink_and_the_hook_both_receive_one_event(self, tmp_path, monkeypatch):
        """Fan-out, not replacement: the sink half is untouched."""
        import agentica.notify.sink as sink_mod
        from agentica.notify import install_sink, reset_sink_for_tests
        from agentica.notify.config import NotifyConfig

        command, out = _recorder(tmp_path)
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
        spy = _SinkSpy()
        monkeypatch.setattr(sink_mod, "_sink", spy)
        try:
            record = RunEventRecord(run_id="r1", event_type=RunEventType.run_started,
                                    payload={"agent_name": "A"})
            sink_mod.notify_sink_dispatch(record, session_id="s1")
            assert [e for e, _ in spy.events] == ["run.started"]
            assert _wait_for(out, 1)[0]["hook_event_name"] == "run.started"
        finally:
            reset_sink_for_tests()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_egress.py -q`
Expected: FAIL — `No module named 'agentica.shell_hooks.egress'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/shell_hooks/egress.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The hook egress — fire the user's command at lifecycle points.

Fire-and-forget for the four ``run.*`` events: spawn, write the document, and
return. The blocking ``needs.*`` path is ``requests.py``, because it needs a
reply and a race rather than a notification.

Installed once per process, like the sink, so a mid-run config flip cannot leave
a half-wired channel behind. ``enabled: false`` — or an enabled block with no
command — wires nothing at all: no thread, no process.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentica.shell_hooks.config import ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload
from agentica.utils.log import logger

#: Install-time decision: None means "this process runs no hook command".
_shell_hooks: Optional[ShellHooksConfig] = None


def install_hook_egress(
    config: Optional[ShellHooksConfig] = None,
) -> Optional[ShellHooksConfig]:
    """Wire the egress, or return None when there is nothing to run."""
    global _shell_hooks
    cfg = config if config is not None else load_shell_hooks_config()
    _shell_hooks = cfg if cfg.effective else None
    return _shell_hooks


def get_hook_egress() -> Optional[ShellHooksConfig]:
    """The installed config, or None when this process runs no hook command."""
    return _shell_hooks


def reset_hook_egress_for_tests() -> None:
    """Drop the installed egress so a test starts from a clean process state."""
    global _shell_hooks
    _shell_hooks = None


def hook_egress_dispatch(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Send one lifecycle event to the user's command. Never raises, never blocks.

    Called alongside the notify sink rather than instead of it: a broken consumer
    and a broken sink must not be able to take each other down, and observation
    must never break a run.
    """
    cfg = _shell_hooks
    if cfg is None or not cfg.event_enabled(event):
        return
    try:
        body = dict(payload or {})
        doc = build_payload(
            event,
            session_id=session_id,
            work_dir=work_dir,
            run_id=_run_id(agent),
            prompt=body.get("prompt"),
            extra={k: v for k, v in body.items() if k != "prompt"},
        )
        if extra:
            doc.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        HookProcess(cfg.command, doc).start()
    except Exception as exc:
        logger.debug(f"shell hooks: could not send {event}: {exc}")


def _run_id(agent: Any) -> Optional[str]:
    try:
        return getattr(getattr(agent, "run_context", None), "run_id", None)
    except Exception:
        return None
```

Now patch `agentica/notify/sink.py`. Replace `notify_sink_dispatch`
(lines 455-506), `_emit_completion` (509-519) and `goal_finished` (522-547):

```python
def _fan_out_event(
    name: str,
    payload: Dict[str, Any],
    *,
    session_id: Optional[str],
    work_dir: Optional[str],
    agent: Any = None,
) -> None:
    """Deliver one event to every installed egress.

    Both are observation channels: neither may break the other, and neither may
    break the run. The hook egress is imported lazily because it imports this
    module's wire helpers — a module-level import would be a cycle.
    """
    sink = _sink
    if sink is not None:
        try:
            sink.emit_event(name, session_id=session_id, work_dir=work_dir, payload=payload)
        except Exception as exc:
            logger.debug(f"notify sink: could not emit {name}: {exc}")
    try:
        from agentica.shell_hooks.egress import hook_egress_dispatch

        hook_egress_dispatch(
            name, payload, session_id=session_id, work_dir=work_dir, agent=agent
        )
    except Exception as exc:
        logger.debug(f"shell hooks: could not send {name}: {exc}")


def notify_sink_dispatch(
    record: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
) -> None:
    """Hand one ``RunEventRecord`` to every installed external egress.

    Called from ``Runner._emit_event`` alongside (not instead of) the in-process
    callback, so a broken egress and a broken callback cannot take each other
    down. This is also the shared home of the ``run.completed`` deferral below:
    both egresses must agree on whether a run is really over, so the decision is
    taken once here rather than per transport.

    ``goal.*`` events deliberately do not come through here: they are emitted by
    ``GoalManager`` on its own callback, and an external consumer has no use for
    the goal loop — it is an agentica implementation detail, not part of the
    contract.
    """
    try:
        event = getattr(record, "event_type", None)
        name = getattr(event, "value", None) or str(event)
        if name == "run.completed":
            # Deferred while a goal is driving this agent: the run that just
            # ended is one lap of several, so "you can come back now" is not
            # true yet. See ``goal_finished`` for why the signal is two-sided.
            if _goal_is_driving(agent) or not _nothing_more_queued():
                _mark_deferred(agent, _completion_payload(agent, record))
                return
            _emit_completion(
                _completion_payload(agent, record),
                session_id=session_id,
                work_dir=work_dir,
                agent=agent,
            )
            return
        _fan_out_event(
            name,
            _run_event_payload(record),
            session_id=session_id,
            work_dir=work_dir,
            agent=agent,
        )
    except Exception as exc:
        logger.debug(f"notify sink: dispatch failed: {exc}")


def _emit_completion(
    source: Any,
    *,
    session_id: Optional[str],
    work_dir: Optional[str],
    agent: Any = None,
) -> None:
    """Report one completed run, once, to every egress."""
    payload = dict(source) if isinstance(source, dict) else _run_event_payload(source)
    payload.setdefault("title", "run completed")
    _fan_out_event(
        "run.completed", payload, session_id=session_id, work_dir=work_dir, agent=agent
    )


def goal_finished(agent: Any, *, session_id: Optional[str] = None,
                  work_dir: Optional[str] = None) -> None:
    """A goal drove this agent and has now stopped. Report the held completion.

    Called from every place that knows no further lap is coming: the CLI's goal
    hook (each of its exit paths), ``Agent.run_goal`` (the SDK / Gateway driver),
    and the interactive loop's failure path.

    Only fires when a completion was actually held back: a session with no goal,
    or one whose goal never ran a lap, reports nothing extra.
    """
    if agent is None:
        return
    try:
        if not getattr(agent, _DEFERRED_FLAG, False):
            return
        setattr(agent, _DEFERRED_FLAG, False)
        # The held payload, not an empty one: a completion that carries no
        # duration or agent name would be a different shape from every other
        # completion on this wire, and the consumer has no way to know why.
        held = getattr(agent, _DEFERRED_PAYLOAD, None) or {}
        setattr(agent, _DEFERRED_PAYLOAD, None)
        _emit_completion(held, session_id=session_id, work_dir=work_dir, agent=agent)
    except Exception as exc:
        logger.debug(f"notify sink: could not report the deferred completion: {exc}")
```

**Why the `sink is None` early return had to go:** with hooks configured and no
notify sink, `notify_sink_dispatch` returned before reaching the hook. Removing
it is what lets one event reach one, both, or neither egress.

Then install it in the CLI, next to the sink (`cli/interactive/app.py:53` already
imports `install_sink as install_notify_sink`; find its call site with
`rg -n "install_notify_sink" agentica/cli/interactive/app.py` and add):

```python
    from agentica.shell_hooks import install_hook_egress

    install_hook_egress()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/shell_hooks/ tests/notify/ -q`
Expected: PASS. The notify suite is the regression gate for this task: it must
need **no edits** while the fan-out is added.

- [ ] **Step 5: Commit**

```bash
git add agentica/shell_hooks/egress.py agentica/notify/sink.py agentica/cli/interactive/app.py tests/shell_hooks/test_shell_hooks_egress.py
git commit -m "shell hooks: fan out lifecycle events to the hook command beside the sink"
```

---

### Task 5: `needs.approval` — one hook request, racing the terminal

**Files:**
- Create: `agentica/shell_hooks/requests.py`
- Modify: `agentica/agent/approvals.py:128-140` (add `is_parked`)
- Modify: `agentica/cli/approvals.py:269-282` (hook instead of the sink)
- Test: `tests/shell_hooks/test_shell_hooks_requests.py`, `tests/cli/test_cli_approval_hook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/shell_hooks/test_shell_hooks_requests.py
# -*- coding: utf-8 -*-
"""The blocking needs.* path. The two failures worth testing are "the answer
never arrives" and "the answer arrives after the user already answered"."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Tuple

import pytest

from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.requests import HookRequest, start_hook_request


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


@dataclass
class _Pending:
    tool_call_id: str = "call_1"
    name: str = "execute"
    arguments: dict = field(default_factory=dict)
    question: str = "run it?"
    preview: str = "rm -rf build"
    similar_label: str = ""
    options: Tuple[str, ...] = ("allow", "deny")


def _script(tmp_path, body, name="hook.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _config(command, **kw):
    return ShellHooksConfig(enabled=True, command=command, **kw)


def _answered(prompt, options=None):
    return [{"decision": "deny"}]


def _payload_for(pending):
    from agentica.shell_hooks.requests import approval_payload

    return approval_payload(pending, session_id="s1", work_dir="/w", prompt="the task")


class TestTheReplyComesBack:
    def test_a_decision_is_the_reply(self, tmp_path):
        cmd = _script(tmp_path, "import json,sys;print(json.dumps({'decision':'allow'}))")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=10) == {"decision": "allow"}
        finally:
            req.kill()

    def test_the_terminal_wins_when_it_answers_first(self, tmp_path):
        """The hook is still thinking; the user typed y. The reply must be
        discarded, and the process killed — not awaited."""
        cmd = _script(tmp_path, "import time;time.sleep(30)")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=1.0) is None  # no reply yet
            assert req.still_waiting is True
            req.kill()
            assert req.still_waiting is False
        finally:
            req.kill()

    def test_no_decision_reply_is_not_a_decision(self, tmp_path):
        cmd = _script(tmp_path, "import json,sys;print(json.dumps({'nope':1}))")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=2.0) is None
            assert req.still_waiting is True  # the process is still up: keep waiting
        finally:
            req.kill()

    def test_an_empty_stdout_is_not_a_decision(self, tmp_path):
        cmd = _script(tmp_path, "pass")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=2.0) is None
        finally:
            req.kill()

    def test_a_non_zero_exit_is_not_a_decision(self, tmp_path):
        cmd = _script(tmp_path, "import sys;sys.exit(4)")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=2.0) is None
        finally:
            req.kill()


class TestNoEgress:
    def test_no_egress_means_no_request(self):
        assert start_hook_request("needs.approval", {}) is None

    def test_a_switched_off_event_means_no_request(self, tmp_path):
        cmd = _script(tmp_path, "pass")
        install_hook_egress(_config(cmd, events={"needs.approval": False}))
        assert start_hook_request("needs.approval", {}) is None

    def test_a_missing_command_means_no_request(self):
        install_hook_egress(_config(["/nonexistent/notifier"]))
        assert start_hook_request("needs.approval", {}) is None


class TestNoCapOfOurs:
    def test_a_slow_reply_still_lands(self, tmp_path):
        """The harness imposes no deadline. A consumer that takes 3 seconds is
        still waiting for the same user the terminal is waiting for."""
        cmd = _script(tmp_path, "import json,sys,time;time.sleep(3);print(json.dumps({'decision':'deny'}))")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=20) == {"decision": "deny"}
        finally:
            req.kill()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/shell_hooks/test_shell_hooks_requests.py -q`
Expected: FAIL — `No module named 'agentica.shell_hooks.requests'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/shell_hooks/requests.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The blocking ``needs.*`` path — ask the user's command, and let
the terminal race it.

Two semantics are kept from the notify sink, and neither is negotiable:

1. **No wait cap invented here.** A deadline baked into this layer would mean a
   desktop answer got less time than a typed one. A command may set its own
   internal limit; that is the command's business and it is visible in the user's
   own config.
2. **The terminal and the hook race; whoever answers first wins.** The tool call
   is never parked on the hook process — the caller decides the moment the hook
   replies, and kills the process when the terminal got there first. A reply that
   arrives second is a race, not an error.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentica.shell_hooks.egress import get_hook_egress
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload, parse_reply
from agentica.notify.wire import clip_text
from agentica.utils.log import logger


def approval_payload(
    pending: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """The metadata slice of a pending approval, on the hook wire.

    ``question`` / ``preview`` come from ``describe_approval`` and are already the
    human-facing summary, so they are the right thing to send. The raw
    ``arguments`` are deliberately NOT sent: a consumer does not need the full
    command, and this channel stays metadata-only.

    ``options`` is ``PendingApproval.options`` — the tuple the tool itself
    decided on (``_approval_options``). It is sent as-is rather than re-narrowed
    here: a consumer that renders exactly what it is given cannot show a button
    the terminal would reject.
    """
    extra: Dict[str, Any] = {"tool_name": getattr(pending, "name", "") or ""}
    tool_call_id = getattr(pending, "tool_call_id", "") or ""
    if tool_call_id:
        extra["tool_call_id"] = tool_call_id
    for field in ("question", "preview", "similar_label"):
        value = getattr(pending, field, None)
        if value:
            extra[field] = str(value)
    options = getattr(pending, "options", None)
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.approval",
        session_id=session_id,
        work_dir=work_dir,
        prompt=prompt,
        extra=extra,
    )


def question_payload(
    prompt: str,
    options: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """A pending question. No decision vocabulary: the reply is free text."""
    extra: Dict[str, Any] = {"question": clip_text(prompt) or ""}
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.input",
        session_id=session_id,
        work_dir=work_dir,
        prompt=clip_text(prompt),
        extra=extra,
    )


class HookRequest:
    """One in-flight ``needs.*`` request to the user's command.

    ``wait_for_reply`` is the whole interface a caller needs: it returns the
    reply when one arrives, and ``None`` while there is nothing usable — which
    means "keep waiting, or answer in the terminal", never "the hook decided no".
    ``kill`` is what a caller does once the answer is no longer wanted.
    """

    def __init__(self, proc: HookProcess, event: str):
        self._proc = proc
        self._event = event

    @property
    def still_waiting(self) -> bool:
        """True while a reply could still arrive (the process is up)."""
        return self._proc.started and not self._proc.finished

    def wait_for_reply(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait up to ``timeout`` for a usable reply from the hook.

        ``timeout`` is the *caller's*, for its own polling convenience; it is not
        a cap on the user. It returns None on timeout and on every unusable
        reply, and the caller's terminal prompt remains the answer path.
        """
        try:
            self._proc.wait(timeout=timeout)
        except Exception as exc:
            logger.debug(f"shell hooks: waiting for {self._event} failed: {exc}")
            return None
        return parse_reply(self._proc.stdout, self._event)

    def kill(self) -> None:
        self._proc.kill()


def start_hook_request(event: str, payload: Dict[str, Any]) -> Optional[HookRequest]:
    """Spawn the user's command for a ``needs.*`` request, or None if there is none.

    Returns immediately — the caller keeps its own parking (the registry future
    for an approval, the prompt queue for a question) and polls this while it
    waits. Nothing here blocks the run.
    """
    cfg = get_hook_egress()
    if cfg is None or not cfg.event_enabled(event):
        return None
    try:
        proc = HookProcess(cfg.command, payload)
        if not proc.start():
            return None
        return HookRequest(proc, event)
    except Exception as exc:
        logger.debug(f"shell hooks: could not start {event}: {exc}")
        return None
```

Then add `is_parked` to `ApprovalRegistry` (`agentica/agent/approvals.py`, after
`size`):

```python
    def is_parked(self, tool_call_id: str) -> bool:
        """Is this id still waiting for someone to decide it?

        The hook path polls this: the terminal's answer is applied through
        ``decide``, so a ``False`` here means the user already answered
        elsewhere and the hook's process should be killed.
        """
        entry = self._pending.get(tool_call_id)
        return entry is not None and not entry.future.done()
```

Finally, replace the sink offer in `agentica/cli/approvals.py:258-282` with the
hook offer:

```python
        # Side-mounted, so the terminal prompt above is unchanged and still wins
        # whenever the user answers first. The hook command is the same question
        # offered to a desktop app, where the user may answer instead — the same
        # answer, applied the same way. The app has no authority of its own, and
        # with no hook configured this is a no-op.
        #
        # No timeout is passed: the terminal prompt here waits as long as the user
        # takes (``registry.wait`` has no deadline), so the hook gets the same
        # patience rather than a shorter clock of our own invention.
        _offer_approval_to_hook(pending, state, loop)
```

with, in the same module:

```python
def _offer_approval_to_hook(pending: PendingApproval, state: Any, loop: Any) -> None:
    """Offer ``pending`` to the user's hook command and return immediately.

    The process is polled on its own daemon thread. When the user answers in the
    terminal first, ``is_parked`` goes False and the process is killed — the
    hook's answer would be second, which is a race rather than an error.
    """
    from agentica.shell_hooks.requests import approval_payload, start_hook_request

    agent = state.current_agent
    request = start_hook_request(
        "needs.approval",
        approval_payload(
            pending,
            session_id=getattr(agent, "session_id", None),
            work_dir=getattr(agent, "work_dir", None),
            prompt=_anchor_text(agent),
        ),
    )
    if request is None:
        return

    registry = state.approval_registry
    tool_call_id = pending.tool_call_id
    if not tool_call_id:
        request.kill()
        return

    def _poll() -> None:
        try:
            while request.still_waiting:
                reply = request.wait_for_reply(timeout=0.2)
                if reply is not None:
                    decision = reply.get("decision")
                    if isinstance(decision, str):
                        loop.call_soon_threadsafe(
                            _apply_hook_decision, registry, tool_call_id, decision
                        )
                    return
                if registry is None or not registry.is_parked(tool_call_id):
                    # The terminal (or a cancel) resolved it while the hook was
                    # still thinking. Drop the hook's answer: it is second.
                    logger.debug(
                        f"shell hooks: approval {tool_call_id} was already decided; "
                        f"the hook answer arrived second"
                    )
                    return
        finally:
            request.kill()

    threading.Thread(
        target=_poll, name="agentica-hook-approval", daemon=True
    ).start()


def _apply_hook_decision(registry: Any, tool_call_id: str, decision: str) -> None:
    applied = registry.decide(tool_call_id, decision)
    if not applied:
        logger.debug(
            f"shell hooks: approval {tool_call_id} was already decided; "
            f"the hook answer arrived second"
        )


def _anchor_text(agent: Any) -> Optional[str]:
    """The run's anchor text: the user's message, or the goal objective.

    Read from the anchor rather than from the last message, so a goal-driven
    session shows what started the work instead of whichever lap is running.
    """
    anchor = getattr(agent, "task_anchor", None)
    return getattr(anchor, "source_query", None)
```

`import threading` must exist at the top of `cli/approvals.py` (add it if the
`rg -n "^import threading" agentica/cli/approvals.py` check is empty).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/shell_hooks/ tests/cli/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentica/shell_hooks/requests.py agentica/agent/approvals.py agentica/cli/approvals.py tests/shell_hooks/test_shell_hooks_requests.py
git commit -m "shell hooks: needs.approval races the terminal, kills the loser"
```

---

### Task 6: `needs.input` — arm the terminal first, then race

**Files:**
- Modify: `agentica/cli/interactive/app.py:236-334` (the ask callback)
- Test: `tests/cli/test_cli_ask_hook.py`

The bug this task fixes, verified here: `wrap_ask_callback` calls
`ask_via_desktop(...)` **before** `inner(prompt, options)`
(`notify/questions.py:90-98`). A desktop that does not answer makes the
terminal prompt arrive only after that wait — so the user cannot answer in the
terminal while the desktop is thinking. Arming in the other order without
concurrency is worse: the terminal blocks first and the desktop is never
offered. The two must be **concurrent**, and the terminal must be armed first
because only it can be armed without blocking.

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_cli_ask_hook.py
# -*- coding: utf-8 -*-
"""The question path: the terminal is armed before anything can block, and the
hook's answer is delivered through the same slot the typed answer uses."""

from __future__ import annotations

import sys
import time

import pytest

from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.requests import question_payload


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def test_the_question_offered_to_the_hook_carries_no_decision_vocabulary():
    doc = question_payload("which package?", ["date-fns", "dayjs"], session_id="s")
    assert doc["hook_event_name"] == "needs.input"
    assert doc["question"] == "which package?"
    assert doc["options"] == ["date-fns", "dayjs"]
    assert "decision" not in doc


# The concurrency itself is covered in tests/cli/test_cli_ask_race.py, which
# drives the real CLI callback: a fake terminal slot that records when it was
# armed, and a hook that answers. The assertion that matters is ordering —
# the slot is armed before the hook is spawned — and that both answers are
# possible outcomes.
```

plus, in `tests/cli/test_cli_ask_race.py`, a direct test of the race helper this
task adds:

```python
# tests/cli/test_cli_ask_race.py
# -*- coding: utf-8 -*-
"""The ask side of the race: whoever answers first wins, and the terminal is
armed first so it is never locked out by a hook that is still thinking."""

from __future__ import annotations

import queue
import sys
import threading
import time

import pytest

from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def _script(tmp_path, body):
    path = tmp_path / "hook.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


class _Slot:
    """Stands in for the CLI's armed ask prompt."""

    def __init__(self):
        self.armed_at = None
        self.result = queue.Queue(maxsize=1)
        self.resolved = False
        self.answer = None

    def arm(self):
        self.armed_at = time.monotonic()

    def submit(self, text) -> bool:
        if self.resolved:
            return False
        try:
            self.result.put_nowait(text)
            self.resolved = True
            return True
        except queue.Full:
            self.resolved = True
            return False

    def typed(self, text):
        self.submit(text)


class _HookSlot:
    """The hook side, with the same contract the CLI helper expects."""

    def __init__(self, request):
        self._request = request

    def poll(self):
        return self._request.wait_for_reply(timeout=0.05)

    def stop(self):
        self._request.kill()


def test_the_terminal_is_armed_before_the_hook_is_spawned(tmp_path):
    from agentica.cli.interactive.ask_race import race_ask

    install_hook_egress(
        ShellHooksConfig(
            enabled=True,
            command=_script(tmp_path, "import json,sys;print(json.dumps({'answer':'from hook'}))"),
        )
    )
    slot = _Slot()
    seen = {}

    def arm():
        slot.arm()
        seen["armed"] = time.monotonic()

    def spawn():
        seen["spawned"] = time.monotonic()
        from agentica.shell_hooks.requests import start_hook_request

        return start_hook_request("needs.input", {"hook_event_name": "needs.input"})

    answer = race_ask(slot, arm=arm, spawn=spawn, poll=60.0)
    assert answer == "from hook"
    assert seen["armed"] <= seen["spawned"]


def test_the_user_typing_first_wins(tmp_path):
    from agentica.cli.interactive.ask_race import race_ask

    install_hook_egress(
        ShellHooksConfig(
            enabled=True,
            command=_script(tmp_path, "import time;time.sleep(30)"),
        )
    )
    slot = _Slot()

    def arm():
        slot.arm()

    def spawn():
        from agentica.shell_hooks.requests import start_hook_request

        return start_hook_request("needs.input", {"hook_event_name": "needs.input"})

    threading.Timer(0.5, lambda: slot.typed("typed by hand")).start()
    assert race_ask(slot, arm=arm, spawn=spawn, poll=30.0) == "typed by hand"


def test_no_hook_means_the_terminal_answer_is_the_answer(tmp_path):
    from agentica.cli.interactive.ask_race import race_ask

    slot = _Slot()

    def arm():
        slot.arm()

    threading.Timer(0.3, lambda: slot.typed("terminal only")).start()
    assert race_ask(slot, arm=arm, spawn=lambda: None, poll=30.0) == "terminal only"


def test_a_hook_that_answers_nothing_leaves_the_question_open(tmp_path):
    from agentica.cli.interactive.ask_race import race_ask

    install_hook_egress(
        ShellHooksConfig(enabled=True, command=_script(tmp_path, "pass"))
    )
    slot = _Slot()

    def arm():
        slot.arm()

    def spawn():
        from agentica.shell_hooks.requests import start_hook_request

        return start_hook_request("needs.input", {"hook_event_name": "needs.input"})

    threading.Timer(0.8, lambda: slot.typed("typed late")).start()
    assert race_ask(slot, arm=arm, spawn=spawn, poll=30.0) == "typed late"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cli/test_cli_ask_race.py -q`
Expected: FAIL — `No module named 'agentica.cli.interactive.ask_race'`

- [ ] **Step 3: Write the implementation**

```python
# agentica/cli/interactive/ask_race.py
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Race an in-terminal question against the user's hook command.

Why this is its own module: an approval has a registry future to park on, so the
hook can be side-mounted beside it. A question has no such object — the answer is
a string that only the terminal's prompt slot can deliver. So the race has to be
run here, by the host that owns the slot.

Ordering is the whole design:

1. **Arm the terminal first.** Arming does not block, and it is the only step
   that must happen before anything can wait: the prompt widget polls
   ``state.input_request``, so a hook spawned first is simply not offered.
2. **Then spawn the hook.** A hook that answers without ever reading stdin
   (or one that is not installed) must not delay the prompt — that is how you
   deadlock a terminal by putting a request in front of an object that is only
   created after the request returns.
3. **The terminal's slot decides.** The hook's answer is delivered *through the
   same slot* as a typed answer, so the queue's first-writer-wins rule settles
   the race and everything downstream — the transcript echo, Ctrl+C, the run
   watchdog — behaves exactly as it does for typing.

No deadline of ours: the loop below waits as long as the caller's prompt does.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from agentica.utils.log import logger

_POLL_SECONDS = 0.05
#: How often the terminal slot is re-checked while waiting on the hook. Short
#: enough to feel immediate, long enough to cost nothing.
_TICK_SECONDS = 0.05


def race_ask(
    slot: Any,
    *,
    arm: Callable[[], None],
    spawn: Callable[[], Optional[Any]],
    poll: Optional[float] = None,
) -> str:
    """Arm the terminal, offer the question to the hook, return the first answer.

    ``slot`` is the host's prompt slot: it must expose ``resolved`` (has anyone
    answered?) and ``submit(text) -> bool`` (deliver an answer exactly once).
    ``spawn`` returns the hook request, or None when there is no hook. ``poll``
    is only a bound on this helper's own loop, never a cap on the user; None
    means wait as long as the question is open.

    Returns the answer text. ``""`` is a legitimate answer (the user accepted a
    blank field); the caller decides what an unanswered question means.
    """
    arm()
    request = spawn()
    if request is None:
        return _await_slot(slot, poll=poll)

    deadline = None if poll is None else time.monotonic() + poll
    try:
        while True:
            if getattr(slot, "resolved", False):
                # The user typed. The hook's answer, whenever it lands, is second.
                return _slot_answer(slot)
            reply = request.wait_for_reply(timeout=_TICK_SECONDS)
            if reply is not None:
                answer = reply.get("answer")
                if isinstance(answer, str):
                    if slot.submit(answer):
                        logger.info("[ask] answered from the hook command")
                        return answer
                    # The user got there a moment earlier; their answer stands.
                    return _slot_answer(slot)
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        request.kill()
    return _await_slot(slot, poll=poll)


def _await_slot(slot: Any, *, poll: Optional[float]) -> str:
    """Wait for the terminal's own answer, with no deadline of ours."""
    if poll is not None:
        deadline = time.monotonic() + poll
        while not getattr(slot, "resolved", False):
            if time.monotonic() >= deadline:
                return ""
            time.sleep(_POLL_SECONDS)
    else:
        while not getattr(slot, "resolved", False):
            time.sleep(_POLL_SECONDS)
    return _slot_answer(slot)


def _slot_answer(slot: Any) -> str:
    answer = getattr(slot, "answer", None)
    return str(answer) if answer is not None else ""
```

**Wiring.** The CLI's ask callback (`cli/interactive/app.py:236-314`) keeps its
existing body, with two changes. First, the prompt slot is armed *before* the
wait — today the slot is armed at line 258-266 and the wait starts at 276, which
is already the right order, so this part is: **remove the
`wrap_ask_callback`/`ask_via_desktop` path** (lines 316-334) and call the race
from the wait loop instead. Second, the watchdog loop at 277-291 gains the hook
check:

```python
        # Arm the prompt slot (unchanged), then offer the question to the hook
        # command and let the two race. The slot above is armed first on purpose:
        # the widget only polls ``state.input_request``, so a hook spawned first
        # would simply not be offered the question.
        from agentica.cli.interactive.ask_race import race_ask
        from agentica.shell_hooks.requests import question_payload, start_hook_request

        agent = getattr(state_ref, "current_agent", None)
        answer = race_ask(
            _SlotAdapter(req),
            arm=lambda: None,  # already armed above
            spawn=lambda: start_hook_request(
                "needs.input",
                question_payload(
                    prompt,
                    options,
                    session_id=getattr(agent, "session_id", None),
                    work_dir=getattr(agent, "work_dir", None),
                ),
            ),
            poll=None,
        )
```

where `_SlotAdapter` adapts the existing `_InputRequest` to the interface above
(put it next to the callback in `app.py`):

```python
class _SlotAdapter:
    """The armed ``_InputRequest`` seen as the race helper's slot.

    Read-only view: the helper must not clear ``state.input_request`` or touch
    the queue directly, because the prompt widget owns both. ``submit`` is the
    same call the Enter key makes, so a hook answer and a typed answer are
    delivered by exactly one code path.
    """

    def __init__(self, req: Any):
        self._req = req

    @property
    def resolved(self) -> bool:
        return bool(self._req.resolved)

    @property
    def answer(self) -> Any:
        try:
            return self._req.result.get_nowait()
        except queue.Empty:
            return None

    def submit(self, text: str) -> bool:
        if self._req.submit(text):
            # The widget treats a resolved request as closed; mirror the Enter
            # handler's cleanup so the prompt does not stay on screen.
            state_ref = self._state_ref
            if state_ref is not None and state_ref.input_request is self._req:
                state_ref.input_request = None
                app_ref = self._app_ref
                if app_ref is not None:
                    app_ref.invalidate()
            return True
        return False
```

Simplify during implementation: pass `state_ref` and `app_ref` into the adapter
constructor rather than reading them off `self`. The existing 292-314
post-processing (CANCELLED handling, logging, transcript echo) stays exactly as
it is and operates on the returned string.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/cli/ tests/shell_hooks/ -q`
Expected: PASS, and the existing ask tests (`tests/cli/test_cli_input_bg.py`)
unchanged — the CANCELLED sentinel path is untouched.

- [ ] **Step 5: Commit**

```bash
git add agentica/cli/interactive/ask_race.py agentica/cli/interactive/app.py tests/cli/test_cli_ask_race.py tests/cli/test_cli_ask_hook.py
git commit -m "shell hooks: needs.input arms the terminal first, then races the hook"
```

---

### Task 7: Delete `/await`

**Files:**
- Delete: `agentica/notify/questions.py`, `agentica/notify/approvals.py`
- Delete: `tests/notify/test_notify_questions.py`, `tests/notify/test_notify_approvals.py`
- Modify: `agentica/notify/sink.py` (drop `await_decision`, `_parse_decision`, `_ALLOWED_DECISIONS`, the `/await` docstring)
- Modify: `agentica/notify/__init__.py` (docstring: the sink is observe-only now)
- Modify: `tests/notify/test_notify_sink.py` (drop the `/await` cases)
- Modify: `docs/getting-started/notify-sink.md`

- [ ] **Step 1: Find every `/await` consumer**

Run:
```bash
rg -n "await_decision|ask_via_desktop|wrap_ask_callback|publish_approval|/await|_parse_decision|_ALLOWED_DECISIONS" agentica/ tests/ docs/
```
Expected: the two modules being deleted, their two test files, `sink.py`
(`await_decision`, `_parse_decision`, `_ALLOWED_DECISIONS`), `notify/__init__.py`
docstring, and the doc. `cli/approvals.py` and `cli/interactive/app.py` were
already migrated in Tasks 5-6. **If anything else appears, migrate it before
deleting** — that is the point of running this first.

- [ ] **Step 2: Delete the reply half of the sink**

```bash
git rm agentica/notify/questions.py agentica/notify/approvals.py \
       tests/notify/test_notify_questions.py tests/notify/test_notify_approvals.py
```

In `agentica/notify/sink.py`:
- delete `await_decision` and `_parse_decision` (the whole "answer-from-the-app path" section),
- delete the `_ALLOWED_DECISIONS` import/alias (the hook protocol owns the vocabulary now),
- update the module docstring: the sink reports; it does not take replies. Replace
  the two-path table with the single `/event` path and say where replies went.

In `agentica/notify/__init__.py`, update the docstring to match: the sink tells a
local app what a run is doing, and the reply path is the hook egress
(`agentica/shell_hooks/`).

- [ ] **Step 3: Run the notify suite**

Run: `python -m pytest tests/notify/ -q`
Expected: PASS. If `test_notify_sink.py` fails on `/await` cases, delete those
cases — they are testing a removed feature, and each deletion must have a
counterpart in `tests/shell_hooks/` (the reply semantics live there now).

- [ ] **Step 4: Prove no reply path is left in the sink**

Run:
```bash
rg -n "await_decision|/await|_parse_decision|_ALLOWED_DECISIONS" agentica/ docs/getting-started/notify-sink.md
```
Expected: no hits.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "notify sink: drop /await — replies are the hook egress's job now"
```

---

### Task 8: Docs, changelog, and the full gate

**Files:**
- Modify: `docs/getting-started/notify-sink.md`
- Create: `docs/getting-started/shell-hooks.md`
- Modify: `CHANGELOG.md`
- Modify: `agentica/hooks.py:37` (docstring pointer, per the RFC)

- [ ] **Step 1: Write the user-facing doc**

`docs/getting-started/shell-hooks.md` must cover, with the real config block:
`settings.hooks.enabled / command / timeout / events`; the argv-only rule and the
`$HOME`-is-not-expanded trap (a shell wrapper is how you get it); the six events;
the stdin document (one field table); the stdout reply and the three exit-code
meanings; the two semantics (no cap of ours; the terminal races the hook); and the
failure ladder (disabled, missing command, spawn error, non-zero exit, empty
stdout, junk JSON, unknown word → all mean "no decision", never `allow`).

- [ ] **Step 2: Update the sink doc**

`docs/getting-started/notify-sink.md`: remove the `/await` section, state that the
sink is observe-only, and point at `shell-hooks.md` for the reply path.

- [ ] **Step 3: Changelog**

Add one entry: the external hook egress (`settings.hooks`), the six events, and
that `/await` was removed in favour of it.

- [ ] **Step 4: Point `agentica/hooks.py` at the RFC**

The module docstring for the in-process hooks gets one line: it is not the
executable kind, which is `agentica/shell_hooks/` (see
`docs/rfcs/external-hook-egress.md`). Same reason the RFC names it: two different
things called "hooks" must not be confused by someone reading either file.

- [ ] **Step 5: The full gate**

```bash
python scripts/check_bare_ci_imports.py
python -m pytest tests/ -q
```

Expected: the bare-CI check passes (no new top-level import of anything optional —
`shell_hooks` uses only the stdlib), and the full suite is green. Per this
project's AGENTS.md: after pytest, **stop** — do not run `check_ast` here.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: external hook egress; notify sink is observe-only"
```

---

## Self-review against the RFC

**Covered:** config block and env overrides (T1); six events with a real emit
source each (T4 for the four `run.*`, T5/T6 for the two `needs.*`); argv not shell
string (T1); read-once install (T4); stdin document and the omitted-not-null rule
(T2); reply and the three exit-code meanings (T2, where exit code is deliberately
not consulted — a hook that fails and one that declines are the same to us); no
cap of ours and the race (T5, T6); the failure ladder and "never synthesize allow"
(T2, T5); process isolation and `killpg` (T3); metadata-only payload (T2, T5);
`needs.input` carries no decision vocabulary (T2, T6); no token, because there is
no shared endpoint (nothing to implement); `PreToolUse`/`PostToolUse` out of scope
(nothing added to `RunEventType`); deleting `/await` (T7, by the user's
instruction, after the hook path is green).

**Deliberately not followed, with reasons:** the RFC's phasing ("not deleting the
sink in this change") — the user asked for the deletion, so it is the last task
rather than omitted; and its "add the hook next to `notify_sink_dispatch`" — that
would have reimplemented the goal deferral per transport, so T4 makes the existing
deferral the shared home and fans out from it instead.

**Not verified, and flagged rather than assumed:** whether any *other* package
embeds agentica and calls `sink.await_decision` directly. T7 Step 1's grep is the
check, and the rule is migrate-before-delete.
