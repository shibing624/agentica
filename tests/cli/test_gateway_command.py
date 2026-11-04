# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: ``agentica-gateway`` as declared by the core distribution: it
forwards to agentica_gateway, or explains what to install. Runs on bare CI.
"""
import sys

import pytest

from agentica.cli import gateway_command


def test_missing_gateway_prints_install_hint(monkeypatch, capsys):
    # A None entry makes `import agentica_gateway.main` raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "agentica_gateway", None)
    monkeypatch.setitem(sys.modules, "agentica_gateway.main", None)
    assert gateway_command.main() == 1
    err = capsys.readouterr().err
    assert 'uv tool install --force "agentica[gateway]"' in err
    assert "pip install agentica-gateway" in err


def test_broken_installed_gateway_is_not_masked(monkeypatch):
    class _Broken:
        def __getattr__(self, name):
            raise ImportError("cannot import name 'x' from 'somewhere'", name="somewhere")

    monkeypatch.setitem(sys.modules, "agentica_gateway.main", _Broken())
    with pytest.raises(ImportError, match="somewhere"):
        gateway_command.main()


def test_forwards_to_the_real_entry_point(monkeypatch):
    calls = []

    class _Main:
        @staticmethod
        def main():
            calls.append(True)

    monkeypatch.setitem(sys.modules, "agentica_gateway.main", _Main())
    assert gateway_command.main() is None
    assert calls == [True]
