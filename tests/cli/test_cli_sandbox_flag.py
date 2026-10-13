# -*- coding: utf-8 -*-
"""``--sandbox`` / ``settings.sandbox`` -> what the CLI hands to ``DeepAgent``."""
from unittest.mock import MagicMock, patch

from agentica.cli import runtime


def _settings(monkeypatch, values):
    monkeypatch.setattr(runtime, "get_setting", lambda key, default=None: values.get(key, default))


def test_off_by_default(monkeypatch):
    _settings(monkeypatch, {})
    assert runtime._resolve_sandbox({}) == ("off", False)


def test_flag_beats_setting(monkeypatch):
    _settings(monkeypatch, {"sandbox": "off"})
    assert runtime._resolve_sandbox({"sandbox": "workspace-write"}) == ("workspace-write", False)


def test_setting_turns_it_on_with_network(monkeypatch):
    _settings(monkeypatch, {"sandbox": "workspace-write", "sandbox_network_access": True})
    assert runtime._resolve_sandbox({}) == ("workspace-write", True)


def test_parse_args_accepts_the_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["agentica", "--sandbox", "workspace-write"])
    assert runtime.parse_args().sandbox == "workspace-write"


def test_create_agent_forwards_the_mode_to_deep_agent(monkeypatch):
    _settings(monkeypatch, {"sandbox_network_access": True})
    captured = {}

    class FakeDeepAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tools = []

    with (
        patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
        patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
    ):
        runtime.create_agent(
            {"model_provider": "openai", "model_name": "m", "debug": False, "work_dir": None,
             "sandbox": "workspace-write"},
            extra_tools=[], workspace=None, skills_registry=None,
            include_ask_user_question=True,
        )
    assert captured["sandbox"] == "workspace-write"
    assert captured["sandbox_network_access"] is True
    assert "sandbox_config" not in captured
