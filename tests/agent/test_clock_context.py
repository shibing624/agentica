"""Clock readings append without rewriting cached history or the human's ask."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agentica import Agent
from agentica.agent.config import PromptConfig, ToolConfig
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.session.log import SessionLog


class Clock(datetime):
    current = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current.astimezone(tz) if tz else cls.current


def agent(**config):
    return Agent(
        model=OpenAIChat(id="test-clock", api_key="test-key"),
        prompt_config=PromptConfig(add_datetime_to_instructions=True, **config),
        tool_config=ToolConfig(auto_load_mcp=False),
        add_history_to_context=True,
    )


def test_clock_crosses_midnight_without_rewriting_prior_messages():
    a = agent(time_zone="Asia/Shanghai")
    question = Message(role="user", content="用户原文")
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 15, 55, tzinfo=timezone.utc)
        system, first, _ = asyncio.run(a.get_messages_for_run(messages=[question]))
        assert first[-1].content == "<session_date>2026-11-26 (Thursday), 23:55:00 +0800 [Asia/Shanghai]</session_date>\n\n用户原文"
        assert first[-1].durable_content() == question.content == "用户原文"
        reply = Message(role="assistant", content="答复", created_at=int(Clock.current.timestamp()))
        prior = [*first, reply]
        Clock.current += timedelta(minutes=10)
        _, second, sent = asyncio.run(a.get_messages_for_run(messages=[*prior, Message(role="user", content="现在呢")]))
        assert sent[0].content == system.content
        assert sent[1:-1] == prior
        assert second[-1].content == "<session_date>2026-11-27 (Friday), 00:05:00 +0800 [Asia/Shanghai]; elapsed since preceding message: 600s</session_date>\n\n现在呢"
        assert second[-1].durable_content() == "现在呢"
        _, third, _ = asyncio.run(a.get_messages_for_run(messages=[*second, Message(role="user", content="继续")]))
        assert third[-1].content == "继续"


def test_clock_refresh_threshold_and_window_reset():
    a = agent(time_zone="UTC")
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        asyncio.run(a.get_messages_for_run(message="start"))
        Clock.current += timedelta(seconds=1799)
        _, users, _ = asyncio.run(a.get_messages_for_run(message="not stale"))
        assert users[-1].content == "not stale"
        Clock.current += timedelta(seconds=1)
        _, users, _ = asyncio.run(a.get_messages_for_run(message="stale"))
        assert "12:30:00" in users[-1].content
        assert users[-1].durable_content() == "stale"
        a.reset_window_pins()
        _, users, _ = asyncio.run(a.get_messages_for_run(message="fresh window"))
        assert "<session_date>" in users[-1].content


def test_process_zone_is_labeled_and_elapsed_uses_the_last_tool_message():
    a = agent()
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        previous = Message(role="tool", content="done", created_at=int(Clock.current.timestamp()) - 2400)
        _, users, _ = asyncio.run(a.get_messages_for_run(messages=[previous, Message(role="user", content="next")]))
    local = Clock.current.astimezone()
    assert f"{local.strftime('%H:%M:%S %z')} [process timezone {local.tzname()}]" in users[-1].content
    assert "elapsed since preceding message: 2400s" in users[-1].content


def test_zero_interval_refreshes_and_a_backward_clock_does_not_make_negative_elapsed():
    a = agent(time_zone="UTC", clock_refresh_seconds=0)
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        _, first, _ = asyncio.run(a.get_messages_for_run(message="start"))
        previous = Message(role="assistant", content="done", created_at=int(Clock.current.timestamp()))
        Clock.current -= timedelta(seconds=1)
        _, users, _ = asyncio.run(a.get_messages_for_run(messages=[*first, previous, Message(role="user", content="next")]))
    assert "11:59:59" in users[-1].content
    assert "elapsed since preceding message: 0s" in users[-1].content


def test_resumed_clock_keeps_wire_bytes_and_uses_logged_message_time(tmp_path):
    log = SessionLog("clock-resume", base_dir=str(tmp_path))
    log.append("user", "original", wire_content="<session_date>old reading</session_date>\n\noriginal", request_only_blocks=["<session_date>old reading</session_date>"])
    log.append("assistant", "done", timestamp="2026-11-22T12:00:00Z")
    history = log.load()
    a = agent(time_zone="UTC")
    a.working_memory.hydrate_runs_from_history(history)
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        _, users, sent = asyncio.run(a.get_messages_for_run(message="resumed"))
    assert sent[1].content == history[0]["content"]
    assert users[-1].content.endswith("elapsed since preceding message: 345600s</session_date>\n\nresumed")


@pytest.mark.parametrize("config", [{"clock_refresh_seconds": -1}, {"clock_refresh_seconds": float("inf")}, {"time_zone": "Invalid/Zone"}])
def test_invalid_clock_settings_fail_at_configuration(config):
    with pytest.raises(ValueError):
        PromptConfig(**config)


def test_plain_sdk_agent_remains_opt_in():
    a = Agent(model=OpenAIChat(id="test", api_key="test-key"), tool_config=ToolConfig(auto_load_mcp=False))
    _, users, _ = asyncio.run(a.get_messages_for_run(message="original"))
    assert users[-1].content == "original"


def test_clock_wire_prefix_and_original_text_survive_run_and_resume(tmp_path):
    class RecordingModel(OpenAIChat):
        async def response(self, messages, **kwargs):
            self.captured.append([m.to_model_dict() for m in messages])
            messages.append(Message(role="assistant", content="ack"))
            return ModelResponse(content="ack")

    a = agent(time_zone="UTC")
    a.model = RecordingModel(id="test-clock", api_key="test-key")
    a.model.captured = []
    a._session_log = SessionLog("clock-wire", base_dir=str(tmp_path))
    a._get_recall_blocks = lambda *args: asyncio.sleep(0, result=(None, None))
    a._get_prompt_hook_block = lambda *args: asyncio.sleep(0, result="<hook>notice</hook>")
    with patch("agentica.agent.prompts.datetime", Clock):
        Clock.current = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        asyncio.run(a.run("original question"))
        first = a.model.captured[0]
        Clock.current += timedelta(minutes=31)
        asyncio.run(a.run("later question"))
        second = a.model.captured[1]
    assert second[:len(first)] == first
    rows = list(map(json.loads, a._session_log.path.read_text().splitlines()))
    assert [row["content"] for row in rows if row["type"] == "user"] == ["original question", "later question"]
    replay = a._session_log.load()
    restored = agent(time_zone="UTC")
    restored.working_memory.hydrate_runs_from_history(replay)
    with patch("agentica.agent.prompts.datetime", Clock):
        _, users, sent = asyncio.run(restored.get_messages_for_run(message="resume question"))
    # Resume repeats every earlier wire message, including runtime-authored blocks.
    expected = a.working_memory.get_messages_from_last_n_runs(skip_role="system")
    assert [msg.to_model_dict() for msg in sent[1:-1]] == [msg.to_model_dict() for msg in expected]
    assert "12:31:00 +0000 [UTC]" in users[-1].content
    assert users[-1].durable_content() == "resume question"


def test_clock_is_available_to_automated_turns_and_subagents():
    a = agent(time_zone="UTC")
    a._run_automated_input = True
    a._parent_run_id = "parent"
    with patch("agentica.agent.prompts.datetime", Clock):
        _, users, _ = asyncio.run(a.get_messages_for_run(message="report finished"))
    assert users[-1].content.startswith("<session_date>")
    assert users[-1].durable_content() == "report finished"
