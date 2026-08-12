# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli import commands as cli_commands
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestCLIImports(unittest.TestCase):
    """Test cases for CLI module imports."""

    def test_can_import_cli_module(self):
        """Test CLI module can be imported."""
        try:
            import agentica.cli

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import cli module: {e}")

    def test_can_import_agent(self):
        """Test Agent can be imported from CLI."""
        try:
            from agentica import Agent

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Agent: {e}")

    def test_console_entrypoint_main_is_callable(self):
        """The backward-compatible package-root main export remains callable."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agentica.cli import main; raise SystemExit(0 if callable(main) else 1)",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_main_stays_callable_after_submodule_import(self):
        """Import order must not turn the public package function into a module."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import agentica.cli.main; "
                    "from agentica.cli import main; "
                    "raise SystemExit(0 if callable(main) else 1)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_model_registry_keeps_callable_values(self):
        """Lazy startup must not change the exported registry value contract."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from agentica.cli import MODEL_REGISTRY; "
                    "raise SystemExit(0 if all(callable(v) for v in MODEL_REGISTRY.values()) else 1)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_anthropic_model_does_not_load_openai_sdk(self):
        """Selecting Anthropic should load only its provider SDK."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from agentica.cli.runtime import get_model; "
                    "get_model('anthropic', 'claude-sonnet-4-5', api_key='fake'); "
                    "raise SystemExit(1 if 'openai' in sys.modules else 0)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
