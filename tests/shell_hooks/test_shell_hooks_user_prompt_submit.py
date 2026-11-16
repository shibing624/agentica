# -*- coding: utf-8 -*-
"""``UserPromptSubmit``: Claude Code's prompt hook, run unchanged by agentica.

A reply written for Claude Code / Codex / CodeBuddy must be understood here as
is, and whatever it adds sits beside agentica's own recall, never instead of it.
The transport is scripted with in-process fakes (``_fakes.FakeHookProcess``), so
what the tests pin is the decision layer — subscription, reply parsing, parallel
collection and the refusal — not that a shell could run a script.
"""

from __future__ import annotations

import asyncio
import json
import os
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
from agentica.shell_hooks import prompt as prompt_mod
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.prompt import (
    PROMPT_EVENT,
    PromptBlocked,
    collect_prompt_context,
    parse_prompt_reply,
)
from agentica.workspace import Workspace
from tests.shell_hooks._fakes import install_fakes


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("ENABLED", "CONSUMERS", "COMMAND"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{name}", raising=False)
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def _claude_context(payload):
    """What a Claude Code memory hook replies: hookSpecificOutput.additionalContext."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"<ext>{payload['prompt']}</ext>",
        }
    }


def _config(*labels, subscribe=True):
    return ShellHooksConfig(
        enabled=True,
        consumers=[
            HookConsumer(
                name=label,
                command=[label],
                events={PROMPT_EVENT: True} if subscribe else {},
            )
            for label in labels
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

    def test_plain_text_opening_with_a_json_scalar_is_the_whole_text(self):
        """A line that happens to be a JSON scalar must not cut the reply short."""
        assert parse_prompt_reply("42\nopen items: 3", returncode=0) == (
            "42\nopen items: 3", None)

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


class TestTheReadEndsOnlyAtADocument:
    """The reader's stopping rule, as a pure decision.

    A consumer may print its document and keep the pipe open, so the document
    itself ends the read rather than a process exit. Only an object counts: a
    plain-text reply may legitimately open with a JSON scalar (``42``,
    ``true``), and stopping there would cut off the rest of the text.
    """

    @pytest.mark.parametrize("stdout, complete", [
        ('{"a": 1}\n', True),
        ('  {"a": 1}  ', True),
        ("{}", True),
        ("42", False),
        ("42\nopen items: 3", False),
        ('"note"', False),
        ("[]", False),
        ('{"a": 1', False),
        ('{"a": 1}\ntrailing noise', False),
        ("plain text", False),
        ("", False),
    ])
    def test_only_a_complete_object_ends_the_read(self, stdout, complete):
        from agentica.shell_hooks.process import _is_json_document

        assert _is_json_document(stdout.encode()) is complete


class TestCollect:
    def test_the_full_prompt_reaches_the_hook_in_claudes_shape(self, monkeypatch):
        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        query = "q" * 900  # notices clip to 500; recall needs the whole message
        assert collect_prompt_context(
            query, session_id="s1", work_dir="/w",
            transcript_path="/t/s1.jsonl", timeout=1.0, config=_config("mem"),
        ) == f"<ext>{query}</ext>"
        assert len(hooks.made) == 1
        doc = hooks.made[0].payload
        assert doc["hook_event_name"] == "UserPromptSubmit"
        assert (doc["session_id"], doc["prompt"], doc["transcript_path"]) == (
            "s1", query, "/t/s1.jsonl")
        assert doc["cwd"] == "/w"
        # Only a refusal needs stderr, and the caller must ask for it up front.
        assert hooks.made[0].capture_stderr is True

    def test_every_contributor_is_joined_in_config_order(self, monkeypatch):
        install_fakes(
            monkeypatch,
            prompt_mod,
            first={"reply": "<one>", "after": 0.02},
            second={"reply": "two\n"},
        )
        assert collect_prompt_context(
            "hi", timeout=1.0, config=_config("first", "second")
        ) == "<one>\n\ntwo"

    def test_subscribers_run_in_parallel_and_a_slow_one_is_cut(self, monkeypatch):
        hooks = install_fakes(
            monkeypatch,
            prompt_mod,
            slow={"hang": True},
            fast={"reply": _claude_context},
        )
        started = time.monotonic()
        assert collect_prompt_context(
            "hi", timeout=0.1, config=_config("slow", "fast")) == "<ext>hi</ext>"
        # Both were started before either was awaited: they run in parallel.
        assert len(hooks.started) == 2
        assert hooks.made[0].killed is True  # the straggler is cut, not awaited
        assert time.monotonic() - started < 1.0

    def test_a_reply_that_lands_after_the_deadline_is_not_used(self, monkeypatch):
        # The reply is scripted far past the deadline, so it can never win even
        # if a loaded machine stalls the test for a while before the kill.
        hooks = install_fakes(
            monkeypatch, prompt_mod, slow={"reply": "<late>", "after": 5.0}
        )
        started = time.monotonic()
        assert collect_prompt_context(
            "hi", timeout=0.05, config=_config("slow")) is None
        assert hooks.made[0].killed is True
        assert time.monotonic() - started < 1.0

    def test_exit_2_blocks_the_prompt_with_stderr_as_the_reason(self, monkeypatch):
        hooks = install_fakes(
            monkeypatch, prompt_mod, refuse={"exit_code": 2, "stderr": "contains a key"}
        )
        with pytest.raises(PromptBlocked) as caught:
            collect_prompt_context("hi", timeout=1.0, config=_config("refuse"))
        assert caught.value.reason == "contains a key"
        assert hooks.made[0].capture_stderr is True

    def test_a_refusal_whose_stderr_stays_open_is_still_a_refusal(self, monkeypatch):
        """A hook can exit leaving a grandchild that inherited stderr, so the
        pipe may never close. Reading the reason is bounded (never the pipe's
        lifetime), and a refusal whose reason did not arrive still refuses —
        with a default reason, rather than being silently dropped."""
        monkeypatch.setattr(prompt_mod, "_STDERR_GRACE", 0.02)
        install_fakes(
            monkeypatch, prompt_mod, refuse={"exit_code": 2, "open_stderr": True}
        )
        started = time.monotonic()
        with pytest.raises(PromptBlocked) as caught:
            collect_prompt_context("hi", timeout=1.0, config=_config("refuse"))
        assert caught.value.reason == "the prompt hook refused this message"
        assert time.monotonic() - started < 1.0

    def test_a_json_refusal_blocks_the_prompt(self, monkeypatch):
        install_fakes(
            monkeypatch,
            prompt_mod,
            block={"reply": {"decision": "block", "reason": "no"}},
        )
        with pytest.raises(PromptBlocked) as caught:
            collect_prompt_context("hi", timeout=1.0, config=_config("block"))
        assert caught.value.reason == "no"

    def test_a_crashing_hook_adds_nothing(self, monkeypatch):
        install_fakes(
            monkeypatch,
            prompt_mod,
            boom={"exit_code": 1, "reply": "Traceback ..."},
        )
        assert collect_prompt_context("hi", timeout=1.0, config=_config("boom")) is None

    def test_nothing_is_spawned_without_a_subscriber(self, monkeypatch):
        """``UserPromptSubmit`` is opt-in: a notifier configured before the event
        existed must not be waited on before every message."""
        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        assert collect_prompt_context(
            "hi", timeout=0.1, config=_config("mem", subscribe=False)) is None
        assert hooks.made == []

    def test_a_blank_prompt_does_not_run_consumers(self, monkeypatch):
        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        assert collect_prompt_context("   ", timeout=0.1, config=_config("mem")) is None
        assert hooks.made == []

    def test_no_installed_config_means_no_context(self, monkeypatch):
        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        assert collect_prompt_context("hi", timeout=0.1) is None
        assert hooks.made == []


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
    def test_built_in_recall_and_the_shell_hook_both_reach_the_message(
        self, monkeypatch, tmp_path
    ):
        install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        install_hook_egress(_config("mem"))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, users, sent = asyncio.run(
                    agent.get_messages_for_run(message="install deps please"))
        content = sent[-1].content
        # History keeps the bytes that were sent (next request repeats them for
        # the prefix cache); what the person typed is durable_content().
        assert users[0].content == content
        assert users[0].durable_content() == "install deps please"
        assert content.startswith(MEMORY_RECALL_OPEN)
        assert "Use uv, never pip" in content
        assert "<ext>install deps please</ext>" in content
        assert content.index("Use uv, never pip") < content.index("<ext>")
        assert content.endswith("install deps please")

    def test_a_crashing_hook_leaves_built_in_recall_intact(self, monkeypatch, tmp_path):
        install_fakes(
            monkeypatch, prompt_mod, mem={"exit_code": 1, "reply": "Traceback ..."}
        )
        install_hook_egress(_config("mem"))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, _, sent = asyncio.run(
                    agent.get_messages_for_run(message="install deps please"))
        assert "Use uv, never pip" in sent[-1].content
        assert "<ext>" not in sent[-1].content

    def test_the_in_process_hook_adds_context_and_never_rewrites_the_message(
        self, tmp_path
    ):
        class Provider(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return f"<py>{message}</py>"

        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            agent._run_hooks = Provider()
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)):
                _, users, sent = asyncio.run(
                    agent.get_messages_for_run(message="install deps"))
        assert sent[-1].content == "<py>install deps</py>\n\ninstall deps"
        assert users[0].content == sent[-1].content
        assert users[0].durable_content() == "install deps"

    def test_a_blocking_hook_ends_the_run_before_any_model_call(
        self, monkeypatch, tmp_path
    ):
        install_fakes(
            monkeypatch, prompt_mod, refuse={"exit_code": 2, "stderr": "no"}
        )
        install_hook_egress(_config("refuse"))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)) as mock:
                with pytest.raises(PromptBlocked):
                    asyncio.run(agent.run("install deps"))
        main_calls = [c for c in mock.await_args_list if "files" not in str(c)]
        assert main_calls == []


class TestOnlyHumanTurns:
    """Recall and UserPromptSubmit are for what a person asked."""

    def _users(self, agent, message):
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=SELECT_UV)) as mock:
            _, _, sent = asyncio.run(agent.get_messages_for_run(message=message))
        return sent[-1].content, mock

    @pytest.mark.parametrize("mark", ["subagent", "automated", "goal"])
    def test_machine_written_turns_skip_both(self, monkeypatch, mark):
        from agentica.agent.goals import CONTINUATION_PROMPT_PREFIX

        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        install_hook_egress(_config("mem"))
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
            content, mock = self._users(agent, message)
        assert content == message
        # The consumer is never even started: a machine-written turn must not
        # pay for a hook process, not merely ignore its output.
        assert hooks.made == []
        mock.assert_not_awaited()

    def test_a_human_turn_still_gets_both(self, monkeypatch, tmp_path):
        hooks = install_fakes(monkeypatch, prompt_mod, mem={"reply": _claude_context})
        install_hook_egress(_config("mem"))
        with tempfile.TemporaryDirectory() as tmp:
            agent = _agent(tmp)
            asyncio.run(_seed(agent.workspace))
            content, _ = self._users(agent, "install deps please")
        assert "Use uv, never pip" in content and "<ext>" in content
        assert len(hooks.made) == 1
