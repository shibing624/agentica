# -*- coding: utf-8 -*-
"""``prompt.submit``: an external store adds context to each turn's message,
next to the built-in recall, never instead of it."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent.config import WorkspaceMemoryConfig
from agentica.memory.recall import MEMORY_RECALL_OPEN
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run.response import RunResponse
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.protocol import parse_reply
from agentica.shell_hooks.requests import request_prompt_context
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


ECHO = (
    "print(json.dumps({'request_id': payload['request_id'],"
    " 'additional_context': '<ext>' + payload['query'] + '</ext>'}))\n"
)


def _install(*commands, events=None):
    install_hook_egress(
        ShellHooksConfig(
            enabled=True,
            consumers=[
                HookConsumer(
                    name=f"c{i}",
                    command=cmd,
                    events={"prompt.submit": True} if events is None else events,
                )
                for i, cmd in enumerate(commands)
            ],
        )
    )


class TestOptIn:
    def test_an_existing_consumer_is_not_waited_on_before_each_turn(self):
        """A notifier written before prompt.submit existed lists no events."""
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": True, "consumers": [
                {"name": "desktop", "command": ["/hook"]},
            ]}}}
        )
        assert cfg.consumers_for("prompt.submit") == []
        assert cfg.consumers_for("run.completed") == cfg.consumers

    def test_naming_it_subscribes(self):
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": True, "consumers": [
                {"name": "mem", "command": ["/hook"], "events": {"prompt.submit": True}},
            ]}}}
        )
        assert [c.name for c in cfg.consumers_for("prompt.submit")] == ["mem"]


class TestReply:
    def test_context_is_taken_from_a_matching_reply(self):
        reply = parse_reply(
            '{"request_id": "r1", "additional_context": " <m>x</m> "}',
            "prompt.submit",
            request_id="r1",
        )
        assert reply == {"additional_context": "<m>x</m>"}

    @pytest.mark.parametrize(
        "text",
        [
            '{"request_id": "other", "additional_context": "x"}',
            '{"request_id": "r1", "additional_context": "  "}',
            '{"request_id": "r1", "answer": "x"}',
            "not json",
            "",
        ],
    )
    def test_anything_else_adds_nothing(self, text):
        assert parse_reply(text, "prompt.submit", request_id="r1") is None


class TestRequest:
    def test_the_full_message_reaches_the_consumer(self, tmp_path):
        """Notices clip prompt to 500 chars; recall needs the whole message."""
        _install(_script(tmp_path, ECHO))
        query = "q" * 900
        assert request_prompt_context(query, timeout=10) == f"<ext>{query}</ext>"

    def test_every_subscriber_contributes_in_config_order(self, tmp_path):
        first = _script(tmp_path, ECHO, "a.py")
        second = _script(
            tmp_path,
            "print(json.dumps({'request_id': payload['request_id'],"
            " 'additional_context': 'second'}))\n",
            "b.py",
        )
        _install(first, second)
        assert request_prompt_context("hi", timeout=10) == "<ext>hi</ext>\n\nsecond"

    def test_a_slow_subscriber_is_cut_at_the_recall_budget(self, tmp_path):
        """The slow one is killed at the deadline; the fast one still counts."""
        slow = _script(tmp_path, "import time\ntime.sleep(30)\n" + ECHO, "slow.py")
        fast = _script(tmp_path, ECHO, "fast.py")
        _install(slow, fast)
        started = time.monotonic()
        assert request_prompt_context("hi", timeout=1.0) == "<ext>hi</ext>"
        assert time.monotonic() - started < 5

    def test_nothing_is_spawned_without_a_subscriber(self, tmp_path):
        marker = tmp_path / "ran"
        cmd = _script(tmp_path, f"open({str(marker)!r}, 'w').close()\n" + ECHO)
        _install(cmd, events={})
        assert request_prompt_context("hi", timeout=5) is None
        time.sleep(0.5)
        assert not marker.exists()


async def _seed(ws: Workspace) -> None:
    await ws.write_memory_entry(
        title="prefer uv", content="Use uv, never pip, for installs.",
        memory_type="user", description="package manager choice uv pip",
    )


class TestAgentCoexistence:
    def _agent(self, tmp):
        from agentica import Agent

        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
            workspace=Workspace(tmp),
            enable_long_term_memory=True,
            long_term_memory_config=WorkspaceMemoryConfig(recall_memories=True),
        )
        agent.run_response = RunResponse()
        return agent

    def test_built_in_recall_and_the_hook_both_reach_the_message(self, tmp_path):
        _install(_script(tmp_path, ECHO))
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ):
                _, users, _ = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        content = users[0].content
        assert content.startswith(MEMORY_RECALL_OPEN)
        assert "Use uv, never pip" in content
        assert "<ext>install deps please</ext>" in content
        assert content.index("Use uv, never pip") < content.index("<ext>")
        assert content.endswith("install deps please")

    def test_a_failing_hook_leaves_built_in_recall_intact(self, tmp_path):
        _install(_script(tmp_path, "sys.exit(3)\n"))
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            asyncio.run(_seed(agent.workspace))
            with patch.object(
                OpenAIChat, "response",
                new=AsyncMock(return_value=ModelResponse(content='{"files": ["user_prefer_uv.md"]}')),
            ):
                _, users, _ = asyncio.run(agent.get_messages_for_run(message="install deps please"))
        assert "Use uv, never pip" in users[0].content
        assert "<ext>" not in users[0].content
