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
    # Without the marker ``split`` returns the whole prompt and the comparison
    # still passes, while the cache boundary is gone.
    assert VOLATILE_SYSTEM_MARKER in message.content
    return message.content.split(VOLATILE_SYSTEM_MARKER)[0]


def _tool_schemas(agent) -> str:
    # Key order and list order as sent: the provider caches serialized bytes.
    agent.update_model()
    return json.dumps(agent.model.tools, default=str)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Build both agents once for this module's byte-comparison tests.

    Function scope rebuilt two full agents per test (1.6s each); the env is only
    read while building, so a module-scoped patch is enough.
    """
    tmp_path = tmp_path_factory.mktemp("parity")
    mp = pytest.MonkeyPatch()
    mp.setenv("AGENTICA_HOME", str(tmp_path / "home"))
    mp.setenv("AGENTICA_PROJECTS_DIR", str(tmp_path / "projects"))
    (tmp_path / "home").mkdir()
    work = tmp_path / "proj"
    work.mkdir()
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    svc.set_session_work_dir("s1", str(work))
    gateway = svc._build_agent("s1")
    cli = _build_cli(svc, gateway, work)
    try:
        yield cli, gateway, svc, work
    finally:
        mp.undo()


def _build_cli(svc, gateway, work):
    """The CLI agent for this project, built exactly the way `create_agent`
    builds the interactive one (same model/profile, same tool options)."""
    model = gateway.model
    return create_agent(
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


def test_cli_and_gateway_send_the_same_system_prompt(pair):
    cli, gateway, _svc, _work = pair
    cli_prompt, gateway_prompt = _stable_prefix(cli), _stable_prefix(gateway)
    # Guard against comparing two empty strings.
    assert "Self-Awareness" in cli_prompt
    assert cli_prompt == gateway_prompt


def test_cli_and_gateway_send_the_same_tool_schemas(pair):
    cli, gateway, _svc, _work = pair
    names = [t["function"]["name"] for t in json.loads(_tool_schemas(cli))]
    assert {"ask_user_question", "wait", "self_manage", "cronjob", "worktree"} <= set(names)
    assert _tool_schemas(cli) == _tool_schemas(gateway)


def test_rebuilding_an_agent_keeps_the_cached_prefix(pair):
    """A rebuilt agent (``/model``, a gateway restart) must not change the
    cached prefix: that turn would be a cold cache start for no reason."""
    cli, gateway, svc, work = pair
    cli2 = _build_cli(svc, gateway, work)
    gw2 = svc._build_agent("s1")
    assert _stable_prefix(cli2) == _stable_prefix(cli)
    assert _stable_prefix(gw2) == _stable_prefix(gateway)
    assert _tool_schemas(cli2) == _tool_schemas(cli)
    assert _tool_schemas(gw2) == _tool_schemas(gateway)


def test_im_command_list_in_the_prompt_matches_the_im_help():
    """The prompt's IM list is static text (a generated one would change the
    cached prefix); this keeps it from drifting from what /help offers."""
    from agentica.tools.product import PRODUCT_INSTRUCTIONS
    from agentica_gateway.services.im_bot import IM_COMMANDS

    im_line = PRODUCT_INSTRUCTIONS.split("- IM channels:")[1].split("\n\n")[0]
    listed = {cmd.strip("`,. \n") for cmd in im_line.split() if cmd.startswith("`/")}
    assert listed == {cmd for cmd, *_ in IM_COMMANDS}
