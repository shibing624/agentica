# -*- coding: utf-8 -*-
"""``UserPromptSubmit``: Claude Code's prompt hook, run unchanged by agentica.

A script written for Claude Code / Codex / CodeBuddy must work here as is, and
whatever it adds sits beside agentica's own recall, never instead of it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent.config import WorkspaceMemoryConfig
from agentica.agent.hooks import RunHooks
from agentica.memory.recall import MEMORY_RECALL_OPEN
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run.response import RunResponse
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.prompt import (
    PROMPT_EVENT,
    PromptBlocked,
    collect_prompt_context,
    parse_prompt_reply,
)
from agentica.workspace import Workspace


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("ENABLED", "CONSUMERS", "COMMAND"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{name}", raising=False)
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def _script(tmp_path, body, name="hook.py"):
    path = tmp_path / name
    path.write_text("import json,sys\npayload=json.load(sys.stdin)\n" + body, encoding="utf-8")
    return [sys.executable, str(path)]


# What a Claude Code memory hook prints: hookSpecificOutput.additionalContext.
CLAUDE_STYLE = (
    "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',"
    " 'additionalContext': '<ext>' + payload['prompt'] + '</ext>'}}))\n"
)


def _config(*commands, subscribe=True):
    return ShellHooksConfig(
        enabled=True,
        consumers=[
            HookConsumer(
                name=f"c{i}",
                command=cmd,
                events={PROMPT_EVENT: True} if subscribe else {},
            )
            for i, cmd in enumerate(commands)
        ],
    )


class TestSubscription:
    def test_an_existing_notifier_is_not_waited_on_before_each_turn(self):
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": True, "consumers": [
                {"name": "desktop", "command": ["/hook"]},
            ]}}}
        )
        assert cfg.consumers_for(PROMPT_EVENT) == []
        assert cfg.consumers_for("run.completed") == cfg.consumers

    def test_the_claude_event_name_subscribes(self):
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": True, "consumers": [
                {"name": "mem", "command": ["/hook"], "events": {"UserPromptSubmit": True}},
            ]}}}
        )
        assert [c.name for c in cfg.consumers_for(PROMPT_EVENT)] == ["mem"]


class TestReply:
    def test_claude_json_context(self):
        out = json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit", "additionalContext": " <m>x</m> "}})
        assert parse_prompt_reply(out, returncode=0) == ("<m>x</m>", None)

    def test_plain_stdout_is_context(self):
        """Claude Code adds plain-text stdout of this event to the context."""
        assert parse_prompt_reply("branch: release-42\n", returncode=0) == (
            "branch: release-42", None)

    def test_exit_2_refuses_with_stderr_as_the_reason(self):
        assert parse_prompt_reply(
            "", returncode=2, stderr="contains a key\n") == (None, "contains a key")

    @pytest.mark.parametrize("body", [
        {"decision": "block", "reason": "no"},
        {"continue": False, "reason": "no"},
    ])
    def test_a_json_refusal_is_honoured(self, body):
        assert parse_prompt_reply(json.dumps(body), returncode=0) == (None, "no")

    @pytest.mark.parametrize("out", ["", "   ", "{}", '{"additionalContext": "top-level"}'])
    def test_nothing_usable_adds_nothing(self, out):
        """Top-level additionalContext is ignored, as Claude Code ignores it."""
        assert parse_prompt_reply(out, returncode=0) == (None, None)

    def test_any_other_failure_is_neither_context_nor_refusal(self):
        """A crash's stdout is not context: it would put a traceback in the prompt."""
        assert parse_prompt_reply("Traceback ...", returncode=1) == (None, None)
        assert parse_prompt_reply('{"decision": "block"}', returncode=1) == (None, None)


class TestCollect:
    def test_the_full_prompt_reaches_the_hook_in_claudes_shape(self, tmp_path):
        seen = tmp_path / "seen.json"
        cmd = _script(tmp_path, f"open({str(seen)!r}, 'w').write(json.dumps(payload))\n" + CLAUDE_STYLE)
        query = "q" * 900  # notices clip to 500; recall needs the whole message
        assert collect_prompt_context(
            query, session_id="s1", work_dir=str(tmp_path),
            transcript_path="/t/s1.jsonl", timeout=10, config=_config(cmd),
        ) == f"<ext>{query}</ext>"
        doc = json.loads(seen.read_text())
        assert doc["hook_event_name"] == "UserPromptSubmit"
        assert (doc["session_id"], doc["prompt"], doc["transcript_path"]) == (
            "s1", query, "/t/s1.jsonl")
        assert doc["cwd"] == str(tmp_path)

    def test_every_subscriber_contributes_in_config_order(self, tmp_path):
        first = _script(tmp_path, CLAUDE_STYLE, "a.py")
        second = _script(tmp_path, "print('second')\n", "b.py")
        assert collect_prompt_context(
            "hi", timeout=10, config=_config(first, second)) == "<ext>hi</ext>\n\nsecond"

    def test_subscribers_run_in_parallel_and_a_slow_one_is_cut(self, tmp_path):
        slow = _script(tmp_path, "import time\ntime.sleep(30)\n" + CLAUDE_STYLE, "slow.py")
        fast = _script(tmp_path, CLAUDE_STYLE, "fast.py")
        started = time.monotonic()
        assert collect_prompt_context(
            "hi", timeout=1.0, config=_config(slow, fast)) == "<ext>hi</ext>"
        assert time.monotonic() - started < 5

    def test_exit_2_blocks_the_prompt(self, tmp_path):
        cmd = _script(tmp_path, "sys.stderr.write('contains a key'); sys.exit(2)\n")
        with pytest.raises(PromptBlocked) as caught:
            collect_prompt_context("hi", timeout=10, config=_config(cmd))
        assert caught.value.reason == "contains a key"

    def test_a_crashing_hook_adds_nothing(self, tmp_path):
        cmd = _script(tmp_path, "sys.exit(1)\n")
        assert collect_prompt_context("hi", timeout=10, config=_config(cmd)) is None

    def test_nothing_is_spawned_without_a_subscriber(self, tmp_path):
        marker = tmp_path / "ran"
        cmd = _script(tmp_path, f"open({str(marker)!r}, 'w').close()\n")
        assert collect_prompt_context("hi", timeout=5, config=_config(cmd, subscribe=False)) is None
        time.sleep(0.3)
        assert not marker.exists()


# A hook that leaves a background job behind (``dual-mem`` spawns a sync after
# answering) and the job inherits stderr: the pipe stays open after the hook
# itself exits.
_GRANDCHILD = (
    "import subprocess\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
)


class TestStderrNeverHangsTheTurn:
    def test_a_grandchild_holding_stderr_does_not_stall_the_reply(self, tmp_path):
        cmd = _script(tmp_path, _GRANDCHILD + CLAUDE_STYLE)
        started = time.monotonic()
        assert collect_prompt_context("hi", timeout=20, config=_config(cmd)) == "<ext>hi</ext>"
        assert time.monotonic() - started < 5

    def test_a_refusal_with_a_grandchild_still_reports_its_reason(self, tmp_path):
        cmd = _script(
            tmp_path,
            _GRANDCHILD + "sys.stderr.write('contains a key'); sys.stderr.flush(); sys.exit(2)\n",
        )
        started = time.monotonic()
        with pytest.raises(PromptBlocked) as caught:
            collect_prompt_context("hi", timeout=20, config=_config(cmd))
        assert caught.value.reason == "contains a key"
        assert time.monotonic() - started < 5

    def test_more_stderr_than_a_pipe_buffer_does_not_block_the_hook(self, tmp_path):
        cmd = _script(tmp_path, "sys.stderr.write('x' * 512 * 1024); sys.stderr.flush()\n" + CLAUDE_STYLE)
        started = time.monotonic()
        assert collect_prompt_context("hi", timeout=10, config=_config(cmd)) == "<ext>hi</ext>"
        assert time.monotonic() - started < 5


def test_plain_text_opening_with_a_json_scalar_is_not_cut_short(tmp_path):
    cmd = _script(
        tmp_path,
        "import time\nprint('42', flush=True)\ntime.sleep(0.3)\nprint('open items: 3')\n",
    )
    assert collect_prompt_context("hi", timeout=10, config=_config(cmd)) == "42\nopen items: 3"


async def _seed(ws: Workspace) -> None:
    await ws.write_memory_entry(
        title="prefer uv", content="Use uv, never pip, for installs.",
        memory_type="user", description="package manager choice uv pip",
    )


def _agent(tmp, **kw):
    from agentica import Agent

    agent = Agent(
        model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
        workspace=Workspace(tmp),
        enable_long_term_memory=True,
        long_term_memory_config=WorkspaceMemoryConfig(recall_memories=True),
        **kw,
    )
    agent.run_response = RunResponse()
    return agent


SELECT_UV = ModelResponse(content='{"files": ["user_prefer_uv.md"]}')


class TestAgentCoexistence:
    def test_built_in_recall_and_the_shell_hook_both_reach_the_message(self, tmp_path):
        install_hook_egress(_config(_script(tmp_path, CLAUDE_STYLE)))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, users, sent = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        content = sent[-1].content
        assert users[0].content == "install deps please"
        assert content.startswith(MEMORY_RECALL_OPEN)
        assert "Use uv, never pip" in content
        assert "<ext>install deps please</ext>" in content
        assert content.index("Use uv, never pip") < content.index("<ext>")
        assert content.endswith("install deps please")

    def test_a_crashing_hook_leaves_built_in_recall_intact(self, tmp_path):
        install_hook_egress(_config(_script(tmp_path, "sys.exit(1)\n")))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, _, sent = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        assert "Use uv, never pip" in sent[-1].content
        assert "<ext>" not in sent[-1].content

    def test_the_in_process_hook_adds_context_and_never_rewrites_the_message(self, tmp_path):
        class Provider(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return f"<py>{message}</py>"

        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            agent._run_hooks = Provider()
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, users, sent = asyncio.run(agent.get_messages_for_run(message="install deps"))
        assert sent[-1].content == "<py>install deps</py>\n\ninstall deps"
        assert users[0].content == "install deps"

    def test_a_blocking_hook_ends_the_run_before_any_model_call(self, tmp_path):
        install_hook_egress(_config(_script(tmp_path, "sys.stderr.write('no'); sys.exit(2)\n")))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)) as mock:
                with pytest.raises(PromptBlocked):
                    asyncio.run(agent.run("install deps"))
        main_calls = [c for c in mock.await_args_list if "files" not in str(c)]
        assert main_calls == []


class TestOnlyHumanTurns:
    """Recall and UserPromptSubmit are for what a person asked."""

    def _users(self, tmp_path, agent, message):
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)) as mock:
            _, _, sent = asyncio.run(agent.get_messages_for_run(message=message))
        return sent[-1].content, mock

    @pytest.mark.parametrize("mark", ["subagent", "automated", "goal"])
    def test_machine_written_turns_skip_both(self, tmp_path, mark):
        from agentica.agent.goals import CONTINUATION_PROMPT_PREFIX

        seen = tmp_path / "seen"
        install_hook_egress(_config(_script(
            tmp_path, f"open({str(seen)!r}, 'w').close()\n" + CLAUDE_STYLE)))
        message = "install deps please"
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            if mark == "subagent":
                agent._parent_run_id = "parent-run"
            elif mark == "automated":
                agent._run_automated_input = True
            else:
                message = f"{CONTINUATION_PROMPT_PREFIX}\n{message}"
            content, mock = self._users(tmp_path, agent, message)
        assert content == message
        assert not seen.exists()
        mock.assert_not_awaited()

    def test_a_human_turn_still_gets_both(self, tmp_path):
        install_hook_egress(_config(_script(tmp_path, CLAUDE_STYLE)))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            content, _ = self._users(tmp_path, agent, "install deps please")
        assert "Use uv, never pip" in content and "<ext>" in content
