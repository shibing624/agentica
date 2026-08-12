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
        """Installed `agentica` script imports `agentica.cli:main`."""
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

    def test_console_entrypoint_main_stays_callable_after_call(self):
        """Importing agentica.cli.main inside the wrapper must not leave a module export behind."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, agentica.cli as c; "
                    "sys.argv=['agentica','--version']; "
                    "\ntry:\n    c.main()\nexcept SystemExit:\n    pass\n"
                    "raise SystemExit(0 if callable(c.main) else 1)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
