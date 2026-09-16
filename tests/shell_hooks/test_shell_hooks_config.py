# -*- coding: utf-8 -*-
"""Config resolution for the external hook egress."""

from __future__ import annotations

import pytest

from agentica.shell_hooks.config import (
    SHELL_HOOK_EVENTS,
    HookConsumer,
    load_shell_hooks_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ENABLED", "CONSUMERS", "COMMAND"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{name}", raising=False)


def _cfg(block):
    return load_shell_hooks_config({"settings": {"hooks": block}})


class TestDisabledByDefault:
    def test_no_settings_block_means_disabled(self):
        cfg = load_shell_hooks_config({})
        assert cfg.enabled is False
        assert cfg.consumers == []

    def test_enabled_without_a_command_is_still_ineffective(self):
        """enabled is not enough: without an argv there is nothing to run."""
        cfg = _cfg({"enabled": True})
        assert cfg.consumers == []


class TestCommandIsArgv:
    def test_a_list_is_taken_verbatim(self):
        cfg = _cfg(
            {"consumers": [{"name": "desktop", "command": ["/abs/notifier", "--from-agentica"]}]}
        )
        assert cfg.consumers[0].command == ["/abs/notifier", "--from-agentica"]

    def test_a_string_is_refused_not_shell_split(self):
        """A string is a config mistake. Splitting it would invent quoting rules
        and silently run a different argv than the user wrote."""
        cfg = _cfg(
            {"consumers": [{"name": "desktop", "command": "/abs/notifier --flag"}]}
        )
        assert cfg.consumers[0].command == []

    def test_blank_entries_are_dropped(self):
        cfg = _cfg(
            {"consumers": [{"name": "desktop", "command": ["/abs/notifier", "", "  "]}]}
        )
        assert cfg.consumers[0].command == ["/abs/notifier"]

    def test_each_consumer_requires_a_unique_name(self):
        cfg = _cfg(
            {
                "consumers": [
                    {"name": "desktop", "command": ["/one"]},
                    {"name": "desktop", "command": ["/two"]},
                    {"command": ["/unnamed"]},
                ]
            }
        )
        assert [consumer.command for consumer in cfg.consumers] == [["/one"]]


class TestEvents:
    def test_all_events_default_on(self):
        consumer = HookConsumer(name="desktop", command=["/hook"])
        assert set(consumer.events) == set(SHELL_HOOK_EVENTS)
        assert all(consumer.event_enabled(e) for e in SHELL_HOOK_EVENTS)

    def test_an_event_can_be_switched_off(self):
        cfg = _cfg(
            {
                "enabled": True,
                "consumers": [
                    {
                        "name": "desktop",
                        "command": ["/hook"],
                        "events": {"run.started": False},
                    }
                ],
            }
        )
        consumer = cfg.consumers[0]
        assert consumer.event_enabled("run.started") is False
        assert consumer.event_enabled("run.completed") is True

    def test_an_unknown_event_name_is_ignored_not_added(self):
        consumer = HookConsumer(
            name="desktop", command=["/hook"], events={"tool.before": True}
        )
        assert "tool.before" not in consumer.events


class TestEnvOverrides:
    def test_env_beats_config(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_ENABLED", "1")
        monkeypatch.setenv(
            "AGENTICA_HOOKS_CONSUMERS",
            '[{"name":"env","command":["/path with spaces/hook","--flag"]}]',
        )
        cfg = load_shell_hooks_config(
            {
                "settings": {
                    "hooks": {
                        "enabled": False,
                        "consumers": [{"name": "config", "command": ["/from/config"]}],
                    }
                }
            }
        )
        assert cfg.enabled is True
        assert cfg.consumers[0].command == ["/path with spaces/hook", "--flag"]

    def test_empty_env_is_unset(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_CONSUMERS", "   ")
        cfg = _cfg({"consumers": [{"name": "config", "command": ["/from/config"]}]})
        assert cfg.consumers[0].command == ["/from/config"]

    def test_invalid_json_disables_configured_consumers(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOOKS_CONSUMERS", "not-json")
        cfg = _cfg({"consumers": [{"name": "config", "command": ["/from/config"]}]})
        assert cfg.consumers == []


class TestRemovedTimeoutIsIgnored:
    def test_a_leftover_timeout_key_does_not_become_a_field(self):
        cfg = _cfg({"enabled": True, "timeout": 12})
        assert not hasattr(cfg, "timeout")
