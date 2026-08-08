# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
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



class TestLessLesskeyDetection(unittest.TestCase):
    """Ctrl+O expand pager: detect --lesskey-content support (less's help
    misprints the option as --lesskey-context, so detection must probe the real
    option, not parse help text)."""

    def setUp(self):
        # Reset the module-level cache between tests.
        from agentica.cli.interactive import console_io as it

        it._LESS_LESSKEY_OK = None

    def test_supported_when_no_error(self):
        from agentica.cli.interactive import console_io as it

        fake = MagicMock(returncode=0, stderr="")
        with patch("agentica.cli.interactive.console_io.subprocess.run", return_value=fake):
            self.assertTrue(it._less_supports_lesskey("/usr/bin/less"))

    def test_unsupported_when_stderr_mentions_option(self):
        from agentica.cli.interactive import console_io as it

        fake = MagicMock(returncode=0, stderr="There is no lesskey-content=... option")
        with patch("agentica.cli.interactive.console_io.subprocess.run", return_value=fake):
            self.assertFalse(it._less_supports_lesskey("/usr/bin/less"))


class TestOpenInPager(unittest.TestCase):
    """Ctrl+O pager must not stop at less's binary-file confirmation."""

    def test_less_forces_opening_control_character_output(self):
        from agentica.cli.interactive import console_io as it

        run = MagicMock()
        with (
            patch("agentica.cli.interactive.console_io.shutil.which", side_effect=["/usr/bin/less"]),
            patch("agentica.cli.interactive.console_io._less_supports_lesskey", return_value=True),
            patch("agentica.cli.interactive.console_io.subprocess.run", run),
        ):
            it._open_in_pager("execute output", "result\x00with-control-byte")

        args = run.call_args.args[0]
        self.assertIn("-f", args)

    def test_legacy_less_also_forces_opening_control_character_output(self):
        from agentica.cli.interactive import console_io as it

        run = MagicMock()
        with (
            patch("agentica.cli.interactive.console_io.shutil.which", side_effect=["/usr/bin/less"]),
            patch("agentica.cli.interactive.console_io._less_supports_lesskey", return_value=False),
            patch("agentica.cli.interactive.console_io._compile_lesskey", return_value="/tmp/lesskey"),
            patch("agentica.cli.interactive.console_io.subprocess.run", run),
        ):
            it._open_in_pager("execute output", "result\x00with-control-byte")

        args = run.call_args.args[0]
        self.assertIn("-f", args)


class TestCompileLesskey(unittest.TestCase):
    """Old-less fallback: compile a lesskey file to bind Ctrl+O to quit when
    --lesskey-content is unavailable. Esc is not bound (escape-sequence
    prefix would break arrow keys)."""

    def test_returns_path_when_lesskey_compiles(self):
        from agentica.cli.interactive import console_io as it

        run_calls = []

        def fake_run(cmd, **kw):
            run_calls.append(cmd)
            # lesskey -o <out> <src>: create the compiled file to simulate success.
            if cmd[0].endswith("lesskey"):
                with open(cmd[2], "w") as fh:
                    fh.write("COMPILED")
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("agentica.cli.interactive.console_io.shutil.which", return_value="/usr/bin/lesskey"),
            patch("agentica.cli.interactive.console_io.subprocess.run", side_effect=fake_run),
        ):
            out = it._compile_lesskey("\n#command\n^O quit\n")
        self.assertTrue(out and out.endswith(".bin"))
        self.assertTrue(run_calls and run_calls[0][0].endswith("lesskey"))
        import os as _os

        _os.unlink(out)

    def test_returns_none_when_no_lesskey_binary(self):
        from agentica.cli.interactive import console_io as it

        with patch("agentica.cli.interactive.console_io.shutil.which", return_value=None):
            self.assertIsNone(it._compile_lesskey("\n#command\n^O quit\n"))


if __name__ == "__main__":
    unittest.main()
