# -*- coding: utf-8 -*-
"""One session, every client: the CLI and the gateway send the same prefix.

A session moves between the terminal, the web page, the desktop app and IM
chats. If the CLI and the gateway build different system prompts or tool
schemas, every move is a cold prefix-cache start. This builds both agents for
the same project and compares the cached part of the request byte for byte.
"""
import asyncio
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica.agent.prompts import VOLATILE_SYSTEM_MARKER
from agentica.cli.runtime import create_agent
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica_gateway.services.agent_service import AgentService


def _stable_prefix(agent) -> str:
    message = asyncio.run(agent.get_system_message())
    return message.content.split(VOLATILE_SYSTEM_MARKER)[0]


def _tool_schemas(agent) -> str:
    agent.update_model()
    return json.dumps(agent.model.tools, sort_keys=True, default=str)


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(tmp_path / "projects"))
    (tmp_path / "home").mkdir()
    work = tmp_path / "proj"
    work.mkdir()
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    svc.set_session_work_dir("s1", str(work))
    gateway = svc._build_agent("s1")
    model = gateway.model
    cli = create_agent(
        {
            "model_provider": model.provider.lower(),
            "model_name": model.id,
            "base_url": model.base_url,
            "api_key": "sk-test-not-real",
            "debug": False,
            "work_dir": str(work),
            "session_id": "s1",
            "permissions": "auto",
        },
        None,
        svc._workspace,
        None,
        include_ask_user_question=True,
        ask_user_question_callback=lambda *a, **k: "",
        background_process_registry=BackgroundProcessRegistry(),
    )
    return cli, gateway


def test_cli_and_gateway_send_the_same_system_prompt(pair):
    cli, gateway = pair
    cli_prompt, gateway_prompt = _stable_prefix(cli), _stable_prefix(gateway)
    # Guard against comparing two empty strings.
    assert "Self-Awareness" in cli_prompt
    assert cli_prompt == gateway_prompt


def test_cli_and_gateway_send_the_same_tool_schemas(pair):
    cli, gateway = pair
    names = [t["function"]["name"] for t in json.loads(_tool_schemas(cli))]
    assert {"ask_user_question", "wait", "self_manage", "cronjob", "worktree"} <= set(names)
    assert _tool_schemas(cli) == _tool_schemas(gateway)
