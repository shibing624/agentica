# -*- coding: utf-8 -*-
"""JSONL history tools (cross compact_boundary)."""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.manager import CompressionManager
from agentica.memory.session_log import SessionLog
from agentica.tools.builtin.context_tool import BuiltinContextTool


class TestContextTool(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _agent(self, with_log=True):
        slog = SessionLog("ctx-tool", base_dir=self.base) if with_log else None
        if slog is not None:
            slog.append("user", "we decided the token budget is 300k")
            slog.append("assistant", "ok, using compact_token_limit=300000")
            slog.append_compact_boundary("", window_id=1)
            slog.append("user", "new window question")
        cm = CompressionManager()
        agent = type(
            "A",
            (),
            {
                "_session_log": slog,
                "tool_config": type("T", (), {"compression_manager": cm})(),
            },
        )()
        tool = BuiltinContextTool()
        tool.set_agent(agent)
        return tool, cm, slog

    def test_only_search_and_read_are_registered(self):
        tool, _, _ = self._agent(with_log=False)
        self.assertEqual(
            set(tool.functions),
            {"search_session", "read_session_item"},
        )

    def test_search_sees_rows_before_the_boundary(self):
        tool, _, slog = self._agent()
        out = asyncio.run(tool.search_session("token budget"))
        self.assertIn("300k", out)
        item_id = slog.search_entries("token budget")[0]["uuid"]
        body = asyncio.run(tool.read_session_item(item_id))
        self.assertIn("we decided the token budget is 300k", body)

    def test_search_without_a_log_says_so(self):
        tool, _, _ = self._agent(with_log=False)
        self.assertIn("No session log", asyncio.run(tool.search_session("x")))

    def test_gongdanhao_query_finds_ticket(self):
        tool, _, slog = self._agent()
        slog.append("user", "工单 ZX-41827 是这次唯一的追踪号")
        out = asyncio.run(tool.search_session("工单号"))
        self.assertIn("ZX-41827", out)
        self.assertIn("item_id=", out)

    def test_search_schema_matches_codex_search_contents(self):
        tool, _, _ = self._agent(with_log=False)
        fn = tool.functions["search_session"]
        fn.process_entrypoint()
        props = fn.parameters["properties"]
        q = props["query"]["description"]
        self.assertIn("substring", q)
        self.assertIn("工单号", q)
        self.assertIn("ranked", q)
        self.assertIn("limit", props)

    def test_read_schema_matches_codex_read_item(self):
        tool, _, slog = self._agent()
        fn = tool.functions["read_session_item"]
        fn.process_entrypoint()
        props = fn.parameters["properties"]
        self.assertIn("item_id", props)
        self.assertIn("offset_chars", props)
        self.assertIn("limit_chars", props)
        item_id = slog.search_entries("300k")[0]["uuid"]
        sliced = asyncio.run(
            tool.read_session_item(item_id, offset_chars=0, limit_chars=12)
        )
        self.assertIn("item_id=", sliced)
        self.assertIn("…[truncated]", sliced)

    def test_notes_file_is_next_to_the_jsonl(self):
        _, _, slog = self._agent()
        from agentica.compression.new_window import notes_path_for
        self.assertEqual(
            Path(notes_path_for(slog)).name,
            "ctx-tool.notes.md",
        )


if __name__ == "__main__":
    unittest.main()
