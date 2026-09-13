# -*- coding: utf-8 -*-
"""Config resolution for the external hook egress."""

from __future__ import annotations

import pytest

from agentica.shell_hooks.config import (
    SHELL_HOOK_EVENTS,
    load_shell_hooks_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ENABLED", "COMMAND"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{name}", raising=False)


def _cfg(block):
    return load_shell_hooks_config({"settings": {"hooks": block}})


class TestDisabledByDefault:
    def test_no_settings_block_means_disabled(self):
        cfg = load_shell_hooks_config({})
        assert cfg.enabled is False
        assert cfg.command == []

    def test_enabled_without_a_command_is_still_ineffective(self):
        """enabled is not enough: without an argv there is nothing to run."""
        cfg = _cfg({"enabled": True})
        assert cfg.command == []


class TestCommandIsArgv:
    def test_a_list_is_taken_verbatim(self):
        cfg = _cfg({"command": ["/abs/notifier", "--from-agentica"]})
        assert cfg.command == ["/abs/notifier", "--from-agentica"]

    def test_a_string_is_refused_not_shell_split(self):
        """A string is a config mistake. Splitting it would invent quoting rules
        and silently run a different argv than the user wrote."""
        cfg = _cfg({"command": "/abs/notifier --flag"})
        assert cfg.command == []

    def test_blank_entries_are_dropped(self):
        cfg = _cfg({"command": ["/abs/notifier", "", "  "]})
        assert cfg.command == ["/abs/notifier"]


class TestEvents:
    def test_all_six_default_on(self):
        cfg = _cfg({"enabled": True})
        assert set(cfg.events) == set(SHELL_HOOK_EVENTS)
        assert all(cfg.event_enabled(e) for e in SHELL_HOOK_EVENTS)

    def test_an_event_can_be_switched_off(self):
        cfg = _cfg({"events": {"run.started": False}})
        assert cfg.event_enabled("run.started") is False
        assert cfg.event_enabled("run.completed") is True

    def test_an_unknown_event_name_is_ignored_not_added(self):
        cfg = _cfg({"events": {"tool.before": True}})
        assert "tool.before" not in cfg.events


class TestEnvOverrides:
    def test_env_beats_config(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_ENABLED", "1")
        monkeypatch.setenv("AGENTICA_HOOKS_COMMAND", "/from/env")
        cfg = load_shell_hooks_config(
            {"settings": {"hooks": {"enabled": False, "command": ["/from/config"]}}}
        )
        assert cfg.enabled is True
        assert cfg.command == ["/from/env"]

    def test_empty_env_is_unset(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_COMMAND", "   ")
        cfg = _cfg({"command": ["/from/config"]})
        assert cfg.command == ["/from/config"]


class TestRemovedTimeoutIsIgnored:
    def test_a_leftover_timeout_key_does_not_become_a_field(self):
        cfg = _cfg({"enabled": True, "timeout": 12})
        assert not hasattr(cfg, "timeout")
