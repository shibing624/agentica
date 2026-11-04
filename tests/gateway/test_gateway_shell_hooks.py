"""The gateway wires ``settings.hooks`` only when ``settings.hooks.gateway`` is on."""
import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica.shell_hooks import get_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig


@pytest.fixture(autouse=True)
def _clean_egress():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def _cfg(*, gateway):
    return ShellHooksConfig(
        enabled=True,
        consumers=[HookConsumer(name="memory", command=["/bin/true"])],
        gateway=gateway,
    )


def test_hooks_stay_off_without_the_gateway_switch(monkeypatch):
    import agentica_gateway.main as gateway_main

    monkeypatch.setattr(gateway_main, "load_shell_hooks_config", lambda: _cfg(gateway=False))
    gateway_main._install_gateway_hooks()
    assert get_hook_egress() is None


def test_gateway_switch_installs_the_configured_consumers(monkeypatch):
    import agentica_gateway.main as gateway_main

    monkeypatch.setattr(gateway_main, "load_shell_hooks_config", lambda: _cfg(gateway=True))
    gateway_main._install_gateway_hooks()
    installed = get_hook_egress()
    assert installed is not None
    assert [c.name for c in installed.consumers] == ["memory"]
