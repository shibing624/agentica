"""Focused tests for shell resume and transcript replay."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentica.cli.commands import display_resumed_transcript, hydrate_resumed_session
from agentica.cli.main import main
from agentica.cli.runtime import parse_args
from agentica.memory.session_log import SessionLog
from agentica.memory.working import WorkingMemory
from agentica.model.usage import Usage


def test_parse_shell_resume_command():
    with patch.object(sys, "argv", ["agentica", "resume", "session-123"]):
        args = parse_args()

    assert args.command == "resume"
    assert args.resume_session_id == "session-123"


def test_hydrate_resumed_session_builds_prompt_runs(tmp_path):
    log = SessionLog("session-123", base_dir=str(tmp_path))
    log.append("user", "inspect the file")
    log.append("assistant", "partial answer\n\n[User interrupted the response]", finish_reason="cancelled")
    agent = SimpleNamespace(_session_log=log, working_memory=WorkingMemory())

    messages, runs_built = hydrate_resumed_session(agent)

    assert len(messages) == 2
    assert runs_built == 1
    history = agent.working_memory.get_messages_from_last_n_runs()
    assert [message.role for message in history] == ["user", "assistant"]
    assert "partial answer" in history[-1].content


def test_display_resumed_transcript_includes_full_tool_result():
    console = MagicMock()
    tool_output = "line 1\nline 2\nline 3"
    messages = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"file_path": "README.md"}),
                    }
                }
            ],
        },
        {"role": "tool", "tool_name": "read_file", "content": tool_output},
        {"role": "assistant", "content": "Done."},
    ]

    with patch("agentica.cli.commands.get_console", return_value=console):
        display_resumed_transcript(messages, "session-123")

    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "Tool call: read_file" in rendered
    assert "Tool result: read_file" in rendered
    assert tool_output in rendered


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
