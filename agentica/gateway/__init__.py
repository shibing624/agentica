# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Deprecated alias for the ``agentica_gateway`` package.

The gateway (FastAPI server, Web UI, IM channels) moved out of the ``agentica``
distribution into its own, ``agentica-gateway``, importable as
``agentica_gateway``. This shim keeps ``import agentica.gateway...`` working for
a deprecation window: every ``agentica.gateway.X`` resolves to the *same module
object* as ``agentica_gateway.X`` (not a second copy), so module state such as
``deps`` singletons and ``monkeypatch.setattr`` targets stay shared.

Install the gateway with ``pip install agentica-gateway`` (or the unchanged
``pip install "agentica[gateway]"``). Will be removed in agentica 1.7.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

_OLD = __name__  # "agentica.gateway"
_NEW = "agentica_gateway"

try:
    _target = importlib.import_module(_NEW)
except ImportError as e:  # not installed, or its deps (fastapi) are missing
    raise ImportError(
        "agentica.gateway moved to the separate `agentica-gateway` package. Install with:\n\n"
        "    pip install agentica-gateway              # FastAPI + Web UI + cron\n"
        "    pip install \"agentica-gateway[feishu]\"    # + Feishu/Lark bot\n\n"
        "and import it as `agentica_gateway`."
    ) from e

warnings.warn(
    "`agentica.gateway` is deprecated; import `agentica_gateway` instead "
    "(pip install agentica-gateway). The alias will be removed in agentica 1.7.",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    """Hand back the already-imported ``agentica_gateway.X`` module."""

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self._real_name)

    def exec_module(self, module) -> None:
        # Already executed under its real name; running it again would build a
        # second FastAPI app and second set of singletons.
        return None

    def get_code(self, fullname):
        # `python -m agentica.gateway.main` (older Desktop builds spawn this):
        # runpy runs the real module's code as __main__.
        spec = importlib.util.find_spec(self._real_name)
        return spec.loader.get_code(self._real_name)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real = _NEW + fullname[len(_OLD):]
        real_spec = importlib.util.find_spec(real)
        if real_spec is None:
            return None
        # Carry the real file as origin: runpy (`python -m agentica.gateway.main`)
        # sets __file__ from it, and main.py resolves its UI dir off __file__.
        spec = importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(real), origin=real_spec.origin, is_package=False,
        )
        spec.has_location = real_spec.has_location
        return spec


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

# `agentica.gateway` itself behaves as the real package for attribute access.
sys.modules[_OLD] = _target
