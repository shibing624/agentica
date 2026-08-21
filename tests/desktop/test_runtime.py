# -*- coding: utf-8 -*-
"""Run the desktop runtime resolver tests (plain node, no Electron)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is required")

DESKTOP = Path(__file__).resolve().parents[2] / "desktop"


def test_runtime_resolver():
    subprocess.check_call([NODE, "--test", "runtime.test.js"], cwd=DESKTOP)
