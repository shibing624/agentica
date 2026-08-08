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



class TestSpinnerRender(unittest.TestCase):
    """Test the braille spinner renders a live marker for every phase."""

    def test_thinking_phase(self):
        from agentica.cli.interactive.stream_loop import _BRAILLE_SPINNER, _render_spinner_text

        text = _render_spinner_text(0, "thinking", "", 2.0)
        self.assertIn(_BRAILLE_SPINNER[0], text)
        self.assertIn("thinking", text)
        self.assertIn("(2s)", text)

    def test_reasoning_phase(self):
        from agentica.cli.interactive.stream_loop import _render_spinner_text

        text = _render_spinner_text(3, "reasoning", "", 1.5)
        self.assertIn("reasoning", text)
        self.assertIn("(2s)", text)  # 1.5 -> :.0f rounds to 2

    def test_tool_phase_uses_base_label(self):
        from agentica.cli.interactive.stream_loop import _render_spinner_text

        text = _render_spinner_text(0, "tool", "🔧 grep", 5.0)
        self.assertIn("🔧 grep", text)
        self.assertIn("(5s)", text)
        self.assertNotIn("thinking", text)

    def test_answering_phase(self):
        from agentica.cli.interactive.stream_loop import _render_spinner_text

        text = _render_spinner_text(0, "answering", "", 3.0)
        self.assertIn("answering", text)
        self.assertIn("(3s)", text)

    def test_idle_phase_returns_empty(self):
        from agentica.cli.interactive.stream_loop import _render_spinner_text

        self.assertEqual(_render_spinner_text(0, "idle", "", 0.0), "")

    def test_frame_advances(self):
        from agentica.cli.interactive.stream_loop import _BRAILLE_SPINNER, _render_spinner_text

        t0 = _render_spinner_text(0, "thinking", "", 0.0)
        t1 = _render_spinner_text(1, "thinking", "", 0.0)
        self.assertIn(_BRAILLE_SPINNER[0], t0)
        self.assertIn(_BRAILLE_SPINNER[1], t1)
        self.assertNotEqual(t0, t1)


if __name__ == "__main__":
    unittest.main()
