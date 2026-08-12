"""Focused tests for shell resume and transcript replay."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.session import (
    HistoryRenderStats,
    _cmd_history,
    _cmd_resume,
    display_resumed_transcript,
    hydrate_resumed_session,
)
from agentica.cli.main import main
from agentica.cli.runtime import parse_args
from agentica.cli.session_resume import prepare_startup_resume
from agentica.memory.models import AgentRun
from agentica.memory.session_log import SessionLog
from agentica.memory.working import WorkingMemory
from agentica.model.message import Message
from agentica.model.usage import Usage
from agentica.run_response import RunResponse
from agentica import global_config as gc


def test_parse_shell_resume_command():
    with patch.object(sys, "argv", ["agentica", "resume", "session-123"]):
        args = parse_args()

    assert args.command == "resume"
    assert args.resume_session_id == "session-123"


def test_hydrate_resumed_session_builds_prompt_runs(tmp_path):
    log = SessionLog("session-123", base_dir=str(tmp_path))
    log.append("user", "inspect the file")
    log.append("assistant", "partial answer\n\n[User interrupted the response]", finish_reason="cancelled")
    agent = SimpleNamespace(_session_log=log, working_memory=WorkingMemory(), model=None)

    messages, runs_built = hydrate_resumed_session(agent)

    assert len(messages) == 2
    assert runs_built == 1
    history = agent.working_memory.get_messages_from_last_n_runs()
    assert [message.role for message in history] == ["user", "assistant"]
    assert "partial answer" in history[-1].content


def test_resume_restores_session_profile_before_creating_agent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("AGENTICA_HOME", str(home))

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    gc.upsert_profile(
        "current-prof",
        {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-current",
        },
    )
    gc.upsert_profile(
        "resume-prof",
        {
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "sk-resume",
        },
        make_active=False,
    )

    current_log = SessionLog("current", base_dir=str(tmp_path), work_dir=str(work_dir))
    current_log.append("user", "current")
    target_log = SessionLog("session-target", base_dir=str(tmp_path), work_dir=str(work_dir))
    target_log.append("user", "resume me")
    target_log.set_profile("resume-prof", "session")

    ctx = CommandContext(
        agent_config={
            "profile_name": "current-prof",
            "profile_source": "project",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-current",
            "debug": False,
            "work_dir": str(work_dir),
        },
        current_agent=SimpleNamespace(_session_log=current_log, user_id="default", session_id="current"),
        extra_tools=[],
        workspace=None,
        skills_registry=None,
        tui_state={},
    )
    resumed_agent = SimpleNamespace(
        _session_log=target_log,
        working_memory=WorkingMemory(),
        model=SimpleNamespace(supports_replayed_tool_history=True),
        auxiliary_model=None,
        session_id="session-target",
    )
    console = MagicMock()

    with (
        patch("agentica.cli.commands.session.get_console", return_value=console),
        patch("agentica.cli.commands.session.create_agent", return_value=resumed_agent) as create,
        patch(
            "agentica.cli.commands.session.display_resumed_transcript",
            return_value=HistoryRenderStats(),
        ),
    ):
        result = _cmd_resume(ctx, "session-target")

    assert result["current_agent"] is resumed_agent
    agent_config = create.call_args.args[0]
    assert agent_config["profile_name"] == "resume-prof"
    assert agent_config["profile_source"] == "session"
    assert agent_config["model_provider"] == "deepseek"
    assert agent_config["model_name"] == "deepseek-v4-flash"
    assert gc.get_project_profile(str(work_dir)) == "resume-prof"


def test_prepare_startup_resume_carries_session_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("AGENTICA_HOME", str(home))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    log = SessionLog("startup-session", work_dir=str(work_dir))
    log.append("user", "hello")
    log.set_profile("venus", "session")

    agent_config = {"session_id": "startup-session", "work_dir": str(work_dir)}

    assert prepare_startup_resume(agent_config, printer=lambda _msg: None) is True
    assert agent_config["_resume_session_profile_name"] == "venus"
    assert agent_config["_resume_session_profile_source"] == "session"


def _tool_round_log(tmp_path, session_id):
    log = SessionLog(session_id, base_dir=str(tmp_path))
    log.append("user", "read config.py")
    log.append(
        "assistant",
        "",
        tool_calls=[
            {
                "id": "call-read",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"file_path": "config.py"})},
            }
        ],
    )
    log.append("tool", "PORT = 8080", tool_name="read_file", tool_call_id="call-read")
    log.append("assistant", "The port is 8080.")
    return log


def test_resume_keeps_tool_history_for_a_provider_that_can_replay_it(tmp_path):
    agent = SimpleNamespace(
        _session_log=_tool_round_log(tmp_path, "session-openai"),
        working_memory=WorkingMemory(),
        model=SimpleNamespace(supports_replayed_tool_history=True),
    )

    hydrate_resumed_session(agent)

    history = agent.working_memory.get_messages_from_last_n_runs()
    assert [message.role for message in history] == ["user", "assistant", "tool", "assistant"]


def test_resume_drops_tool_history_for_a_provider_that_cannot(tmp_path):
    """Anthropic cannot replay assistant.tool_calls + role="tool"; it gets the text."""
    agent = SimpleNamespace(
        _session_log=_tool_round_log(tmp_path, "session-claude"),
        working_memory=WorkingMemory(),
        model=SimpleNamespace(supports_replayed_tool_history=False),
    )

    hydrate_resumed_session(agent)

    history = agent.working_memory.get_messages_from_last_n_runs()
    assert [(m.role, m.content) for m in history] == [
        ("user", "read config.py"),
        ("assistant", "The port is 8080."),
    ]


def _history_run(messages):
    return AgentRun(response=RunResponse(messages=messages))


def test_display_resumed_transcript_collapses_tool_results():
    console = MagicMock()
    run = _history_run(
        [
            Message(role="user", content="read it"),
            Message(
                role="assistant",
                content="I will inspect it.",
                tool_calls=[
                    {
                        "id": "call-read",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"file_path": "README.md"}),
                        },
                    },
                    {
                        "id": "call-execute",
                        "function": {
                            "name": "execute",
                            "arguments": json.dumps({"command": "false"}),
                        },
                    },
                ],
            ),
            Message(
                role="tool",
                tool_name="read_file",
                tool_call_id="call-read",
                content="success output must stay hidden",
                tool_call_error=False,
            ),
            Message(
                role="tool",
                tool_name="execute",
                tool_call_id="call-execute",
                content="command failed with exit code 1",
                tool_call_error=True,
            ),
            Message(role="assistant", content="Done."),
        ]
    )

    with patch("agentica.cli.commands.session.get_console", return_value=console):
        stats = display_resumed_transcript([run], "session-123")

    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "You - run 1" in rendered
    assert "I will inspect it." in rendered
    assert "Agent - run 1" in rendered
    assert "Done." in rendered
    assert "read_filex1, executex1" in rendered
    assert "2 results hidden" in rendered
    assert "execute: command failed with exit code 1" in rendered
    assert "success output must stay hidden" not in rendered
    assert "Tool result:" not in rendered
    assert stats.run_count == 1
    assert stats.tool_call_count == 2
    assert stats.tool_result_count == 2
    assert stats.tool_error_count == 1


def test_history_reads_canonical_runs_and_opens_full_tools_in_pager():
    run = _history_run(
        [
            Message(role="user", content="inspect"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"file_path": "README.md"}),
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                tool_name="read_file",
                tool_call_id="call-1",
                content="the complete persisted tool result",
                tool_call_error=False,
            ),
            Message(role="assistant", content="Done."),
        ]
    )
    agent = SimpleNamespace(working_memory=WorkingMemory(runs=[run], messages=[]))
    pager = MagicMock()
    ctx = CommandContext(
        agent_config={},
        current_agent=agent,
        open_pager_callback=pager,
    )
    console = MagicMock()

    with patch("agentica.cli.commands.session.get_console", return_value=console):
        _cmd_history(ctx, "")
        compact = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "inspect" in compact
        assert "Done." in compact
        assert "the complete persisted tool result" not in compact

        _cmd_history(ctx, "tools 1")

    pager.assert_called_once()
    title, full_content = pager.call_args.args
    assert title == "Tool history - run 1"
    assert "Tool call: read_file" in full_content
    assert '"file_path": "README.md"' in full_content
    assert "Tool result: read_file [ok]" in full_content
    assert "the complete persisted tool result" in full_content


def test_resumed_transcript_limits_error_previews_per_run():
    messages = [Message(role="user", content="run checks")]
    for index in range(4):
        messages.append(
            Message(
                role="tool",
                tool_name=f"tool_{index}",
                content=f"error {index}",
                tool_call_error=True,
            )
        )
    console = MagicMock()

    with patch("agentica.cli.commands.session.get_console", return_value=console):
        display_resumed_transcript([_history_run(messages)], "session-123")

    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "tool_0: error 0" in rendered
    assert "tool_1: error 1" in rendered
    assert "tool_2: error 2" in rendered
    assert "tool_3: error 3" not in rendered
    assert "1 more errors hidden" in rendered


def test_noninteractive_interrupt_prints_resume_summary():
    with patch.object(
        sys,
        "argv",
        [
            "agentica",
            "--query",
            "long task",
            "--model_provider",
            "openai",
            "--model_name",
            "gpt-4o-mini",
            "--no-workspace",
            "--no-experience",
        ],
    ):
        args = parse_args()

    def interrupted_stream(_query):
        raise KeyboardInterrupt
        yield

    agent = SimpleNamespace(
        run_stream_sync=interrupted_stream,
        model=SimpleNamespace(usage=Usage()),
        session_id="session-interrupted",
        _session_log=SimpleNamespace(exists=lambda: True),
    )
    console = MagicMock()
    resolved = {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "base_url": None,
    }
    with (
        patch("agentica.cli.main.parse_args", return_value=args),
        patch("agentica.cli.main._enable_cli_file_logging"),
        patch("agentica.cli.main.resolve_model_config", return_value=resolved),
        patch("agentica.cli.main.create_agent", return_value=agent),
        patch("agentica.cli.main.get_console", return_value=console),
        # An interrupted one-shot run reports 130 in its exit status, the way a
        # shell expects; the summary is printed first.
        pytest.raises(SystemExit),
    ):
        main()

    rendered = "\n".join(
        value.plain if hasattr(value, "plain") else str(value)
        for call in console.print.call_args_list
        for value in call.args
    )
    assert "Interrupted." in rendered
    assert "Token usage: total=0 input=0 output=0" in rendered
    assert "agentica resume session-interrupted" in rendered
