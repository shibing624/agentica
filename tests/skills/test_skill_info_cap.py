# -*- coding: utf-8 -*-
"""get_skill_info caps a huge SKILL.md and points to the rest by path + line."""
import os
import tempfile
import unittest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.skills.skill_registry import reset_skill_registry
from agentica.tools.skill_tool import SKILL_INFO_MAX_CHARS, SkillTool


class TestSkillInfoCap(unittest.TestCase):
    def _tool(self, body: str):
        reset_skill_registry()
        self._tmp = tempfile.TemporaryDirectory()
        d = os.path.join(self._tmp.name, "big")
        os.makedirs(d)
        with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: big\ndescription: a big skill\n---\n\n" + body)
        tool = SkillTool(custom_skill_dirs=[d])
        tool._ensure_initialized()
        return tool, os.path.join(d, "SKILL.md")

    def tearDown(self):
        reset_skill_registry()
        self._tmp.cleanup()

    def test_small_skill_is_returned_whole(self):
        tool, _ = self._tool("short body\n")
        out = tool.get_skill_info("big")
        self.assertIn("short body", out)
        self.assertNotIn("SKILL.md continues", out)

    def test_big_skill_is_capped_and_offset_points_at_the_next_line(self):
        lines = [f"line {i:05d} " + "x" * 60 for i in range(1000)]
        tool, path = self._tool("\n".join(lines) + "\n")
        out = tool.get_skill_info("big")
        self.assertLess(len(out), SKILL_INFO_MAX_CHARS + 1000)
        self.assertIn("SKILL.md continues", out)
        offset = int(out.split("offset=")[1].split(")")[0])
        shown_last = [ln for ln in out.split("--- Instructions ---\n")[1].splitlines() if ln.startswith("line ")][-1]
        with open(path, encoding="utf-8") as f:
            file_lines = f.read().splitlines()
        # read_file offsets are 0-based: file_lines[offset] is the first line not shown.
        self.assertEqual(file_lines[offset - 1], shown_last)
        self.assertTrue(file_lines[offset].startswith("line "))


if __name__ == "__main__":
    unittest.main()
