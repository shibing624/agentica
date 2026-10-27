# -*- coding: utf-8 -*-
"""Files the model read that changed on disk afterwards are flagged once."""
import asyncio
import os
import tempfile
import unittest

from agentica.agent import Agent
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.tools.builtin.file_tool import EXTERNAL_CHANGE_TRACK_LIMIT, BuiltinFileTool


def _bump(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


class TestTakeExternalChanges(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = BuiltinFileTool(work_dir=self.tmp)
        self.path = os.path.realpath(os.path.join(self.tmp, "a.py"))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("x = 1\n")

    def test_external_edit_is_reported_once(self):
        asyncio.run(self.tool.read_file("a.py"))
        self.assertEqual(self.tool.take_external_changes(), [])
        _bump(self.path, "x = 2  # edited elsewhere\n")
        self.assertEqual(self.tool.take_external_changes(), [self.path])
        self.assertEqual(self.tool.take_external_changes(), [])

    def test_own_writes_and_patches_are_not_external(self):
        asyncio.run(self.tool.read_file("a.py"))
        asyncio.run(self.tool.write_file("a.py", "x = 3\n"))
        self.assertEqual(self.tool.take_external_changes(), [])
        asyncio.run(self.tool.apply_patch(
            "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 3\n+x = 4\n*** End Patch"
        ))
        self.assertEqual(self.tool.take_external_changes(), [])

    def test_deleted_file_is_reported_and_dropped(self):
        asyncio.run(self.tool.read_file("a.py"))
        os.remove(self.path)
        self.assertEqual(self.tool.take_external_changes(), [f"{self.path} (deleted)"])
        self.assertEqual(self.tool.take_external_changes(), [])

    def test_unread_files_are_not_tracked(self):
        _bump(self.path, "x = 9\n")
        self.assertEqual(self.tool.take_external_changes(), [])

    def test_tracking_is_capped_oldest_first(self):
        for i in range(EXTERNAL_CHANGE_TRACK_LIMIT + 5):
            with open(os.path.join(self.tmp, f"f{i}.txt"), "w") as f:
                f.write("v")
            asyncio.run(self.tool.read_file(f"f{i}.txt"))
        self.assertEqual(len(self.tool._seen_stamps), EXTERNAL_CHANGE_TRACK_LIMIT)
        self.assertNotIn(os.path.realpath(os.path.join(self.tmp, "f0.txt")), self.tool._seen_stamps)


class TestAgentInjection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = BuiltinFileTool(work_dir=self.tmp)
        self.agent = Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"), tools=[self.tool])
        self.path = os.path.realpath(os.path.join(self.tmp, "a.py"))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        asyncio.run(self.tool.read_file("a.py"))

    def test_post_tool_hook_appends_an_injected_note(self):
        hook = self.agent._build_post_tool_hook()
        self.assertIsNotNone(hook, "a file tool alone enables the hook")
        messages = [Message(role="user", content="go")]
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), 1, "nothing changed, nothing injected")
        _bump(self.path, "x = 2\n")
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[-1]._injected)
        self.assertIn(self.path, messages[-1].content)
        self.assertIn("[File change notice]", messages[-1].content)

    def test_next_human_turn_gets_the_note_before_the_question(self):
        _bump(self.path, "x = 2\n")
        _, user_messages, _ = asyncio.run(self.agent.get_messages_for_run(message="what is x?"))
        content = user_messages[-1].content
        self.assertTrue(content.startswith("[File change notice]"))
        self.assertTrue(content.endswith("what is x?"))
        _, again, _ = asyncio.run(self.agent.get_messages_for_run(message="and now?"))
        self.assertEqual(again[-1].content, "and now?")


if __name__ == "__main__":
    unittest.main()
