"""Tests for the Codex-style summary shown by ``/new``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentica.cli.commands import CommandContext, _cmd_newchat
from agentica.cli.display import format_session_summary
from agentica.cli.interactive import SessionState, _print_interactive_exit_summary
from agentica.model.usage import RequestUsage, TokenDetails, Usage


def test_format_session_summary_shows_usage_and_resume_hint():
    usage = Usage()
    usage.add(
        RequestUsage(
            input_tokens=4_000,
            output_tokens=300,
            total_tokens=4_300,
            input_tokens_details=TokenDetails(cached_tokens=800),
            output_tokens_details=TokenDetails(reasoning_tokens=120),
        )
    )

    rendered = format_session_summary(
        elapsed_seconds=905,
        usage=usage,
        session_id="session-123",
    ).plain

    assert "Worked for 15m 05s" in rendered
    assert "Token usage: total=4,300 input=4,000 (+ 800 cached) output=300 (reasoning 120)" in rendered
    assert "To continue this session, run agentica resume session-123" in rendered


def test_format_session_summary_keeps_zero_usage_visible():
    rendered = format_session_summary(
        elapsed_seconds=1,
        usage=Usage(),
        session_id="session-zero",
    ).plain

    assert "Token usage: total=0 input=0 output=0" in rendered


def test_newchat_prints_summary_then_header_and_resets_session_state(monkeypatch):
    usage = Usage(input_tokens=10, output_tokens=2, total_tokens=12)
    model = SimpleNamespace(usage=usage)
    agent = SimpleNamespace(model=model, memory=SimpleNamespace(messages=[]), session_id="old-session")
    new_agent = SimpleNamespace(model=SimpleNamespace(usage=Usage()), session_id="new-session")
    console = MagicMock()
    print_header = MagicMock()
    tui_state = {"session_started_at": 100.0, "context_tokens": 42, "context_window": 128_000}
    ctx = CommandContext(
        agent_config={"model_provider": "openai", "model_name": "gpt-5.6-sol"},
        current_agent=agent,
        tui_state=tui_state,
    )
    monkeypatch.setattr("agentica.cli.commands.time.monotonic", lambda: 1_005.0)
    monkeypatch.setattr("agentica.cli.commands.create_agent", lambda *args, **kwargs: new_agent)
    monkeypatch.setattr("agentica.cli.commands.print_header", print_header)
    monkeypatch.setattr("agentica.cli.commands.get_console", lambda: console)

    result = _cmd_newchat(ctx)

    summary = console.print.call_args_list[0].args[0].plain
    assert "Worked for 15m 05s" in summary
    assert "agentica resume old-session" in summary
    print_header.assert_called_once()
    assert result["current_agent"] is new_agent
    assert result["session_started_at"] == 1_005.0


def test_interactive_exit_prints_resume_summary(monkeypatch):
    usage = Usage(input_tokens=10, output_tokens=2, total_tokens=12)
    agent = SimpleNamespace(model=SimpleNamespace(usage=usage), session_id="exit-session")
    console = MagicMock()

    monkeypatch.setattr("agentica.cli.interactive.time.monotonic", lambda: 130.0)
    monkeypatch.setattr("agentica.cli.interactive.get_console", lambda: console)

    _print_interactive_exit_summary(
        SessionState(current_agent=agent),
        {"session_started_at": 100.0},
    )

    rendered = console.print.call_args.args[0].plain
    assert "Worked for 0m 30s" in rendered
    assert "Token usage: total=12 input=10 output=2" in rendered
    assert "agentica resume exit-session" in rendered
