# -*- coding: utf-8 -*-
"""``--sandbox`` / ``settings.sandbox`` -> the SandboxConfig the CLI agent gets."""
import pytest

from agentica.cli import runtime


def _settings(monkeypatch, values):
    monkeypatch.setattr(runtime, "get_setting", lambda key, default=None: values.get(key, default))


def test_off_by_default_keeps_the_sdk_default(monkeypatch):
    _settings(monkeypatch, {})
    assert runtime._resolve_sandbox_config({}, "ask") is None


def test_flag_beats_setting(monkeypatch, tmp_path):
    _settings(monkeypatch, {"sandbox": "off"})
    config = runtime._resolve_sandbox_config(
        {"sandbox": "workspace-write", "work_dir": str(tmp_path)}, "ask"
    )
    assert config.os_sandbox is True
    assert config.network_access is False
    assert config.writable_dirs == [str(tmp_path)]
    # The path-check layer still follows the permission tier.
    assert config.enabled is True
    assert runtime._resolve_sandbox_config(
        {"sandbox": "workspace-write", "work_dir": str(tmp_path)}, "allow-all"
    ).enabled is False


def test_setting_turns_it_on_with_network(monkeypatch, tmp_path):
    _settings(monkeypatch, {"sandbox": "workspace-write", "sandbox_network_access": True})
    config = runtime._resolve_sandbox_config({"work_dir": str(tmp_path)}, "auto")
    assert config.os_sandbox is True
    assert config.network_access is True


def test_unknown_mode_fails_loudly(monkeypatch):
    _settings(monkeypatch, {"sandbox": "read-only"})
    with pytest.raises(ValueError, match="read-only"):
        runtime._resolve_sandbox_config({}, "ask")


def test_parse_args_accepts_the_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", ["agentica", "--sandbox", "workspace-write"])
    assert runtime.parse_args().sandbox == "workspace-write"
