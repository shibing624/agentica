# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: BUILTIN_TOOLS display list stays in sync with the real toolset.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli.runtime import BUILTIN_TOOLS, active_tool_names
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.base import Tool
from agentica.tools.builtin import get_builtin_tools


class TestBuiltinToolsListing(unittest.TestCase):
    """The CLI shows BUILTIN_TOOLS in /config, the startup header and /tools.

    It drifted before (write_todos missing — the list said "single source of
    truth" but nobody updated it). These tests pin it to the actual factory
    output so a new builtin function fails CI until the display list catches
    up.
    """

    def test_covers_default_factory_tools(self):
        """Every function get_builtin_tools() exposes must be listed."""
        tools = get_builtin_tools(
            work_dir=".",
            include_ask_user_question=True,
            ask_user_question_callback=lambda prompt, options=None: "",
            background_process_registry=BackgroundProcessRegistry(),
        )
        factory_names = set()
        for t in tools:
            if isinstance(t, Tool) and t.functions:
                factory_names.update(t.functions.keys())

        self.assertTrue(factory_names)
        missing = factory_names - set(BUILTIN_TOOLS)
        self.assertEqual(missing, set(), f"unlisted builtin tools: {sorted(missing)}")

    def test_covers_conditionally_added_tools(self):
        """delegate / memory / todos are added outside the factory (deep.py,
        agent/base.py) — the display list must carry them too."""
        for name in ("write_todos", "delegate", "save_memory", "search_memory", "wait"):
            self.assertIn(name, BUILTIN_TOOLS)

    def test_active_tool_names_reads_live_agent(self):
        """active_tool_names reflects the agent's actual tools, sorted."""
        file_tool = next(
            t for t in get_builtin_tools(work_dir=".") if type(t).__name__ == "BuiltinFileTool"
        )
        agent = type("A", (), {"tools": [file_tool]})()
        self.assertEqual(
            active_tool_names(agent),
            ["apply_patch", "glob", "grep", "read_file", "write_file"],
        )

    def test_active_tool_names_handles_no_agent(self):
        self.assertEqual(active_tool_names(None), [])
        self.assertEqual(active_tool_names(type("A", (), {"tools": None})()), [])


if __name__ == "__main__":
    unittest.main()
