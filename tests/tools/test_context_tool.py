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
        # The tool holds its agent weakly (so Agent -> Model -> functions ->
        # tool does not pin Agent alive). Production keeps the Agent in scope;
        # this fixture must too, or the fake is collected mid-test.
        self._agents: list = []

    def tearDown(self):
        self._agents.clear()
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
        self._agents.append(agent)
        return tool, cm, slog

    def test_only_search_is_registered(self):
        tool, _, _ = self._agent(with_log=False)
        self.assertEqual(set(tool.functions), {"search_session"})

    def test_search_sees_rows_before_the_boundary(self):
        tool, _, slog = self._agent()
        out = asyncio.run(tool.search_session("token budget"))
        self.assertIn("300k", out)
        self.assertIn("we decided the token budget is 300k", out)

    def test_search_without_a_log_says_so(self):
        tool, _, _ = self._agent(with_log=False)
        self.assertIn("No session log", asyncio.run(tool.search_session("x")))

    def test_gongdanhao_query_finds_ticket(self):
        tool, _, slog = self._agent()
        slog.append("user", "工单 ZX-41827 是这次唯一的追踪号")
        out = asyncio.run(tool.search_session("工单号"))
        self.assertIn("ZX-41827", out)
        self.assertNotIn("item_id=", out)

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
        self.assertIn("role", props)
        self.assertEqual(props["role"]["enum"], ["user", "assistant", "tool"])
        self.assertNotIn("query", fn.parameters.get("required") or [])
        self.assertIn("user questions", q.lower())

    def test_empty_query_returns_user_question_index(self):
        tool, _, slog = self._agent()
        slog.append("user", "所以现在敲、compact就是写note.md文件吗？")
        out = asyncio.run(tool.search_session(query=""))
        self.assertIn("Recent user questions", out)
        self.assertIn("compact就是写note.md", out)
        self.assertIn("we decided the token budget is 300k", out)
        self.assertNotIn("<context_window>", out)

    def test_keyword_miss_still_includes_user_questions(self):
        tool, _, slog = self._agent()
        slog.append("user", "那当前的note.md在哪里？")
        out = asyncio.run(tool.search_session("前面问了啥"))
        self.assertIn("No session-log matches", out)
        self.assertIn("Recent user questions", out)
        self.assertIn("那当前的note.md在哪里？", out)

    def test_keyword_hit_also_includes_user_questions(self):
        tool, _, slog = self._agent()
        out = asyncio.run(tool.search_session("token budget"))
        self.assertIn("hit(s) for", out)
        self.assertIn("300k", out)
        self.assertIn("Recent user questions", out)
        self.assertIn("we decided the token budget is 300k", out)

    def test_search_session_hits_notes_file(self):
        tool, _, slog = self._agent()
        from agentica.compression.new_window import notes_path_for
        Path(notes_path_for(slog)).write_text(
            "Constraint: do not rewrite auth. Ticket HV-9917.\n",
            encoding="utf-8",
        )
        out = asyncio.run(tool.search_session("HV-9917"))
        self.assertIn("notes:", out)
        self.assertIn("HV-9917", out)

    def test_notes_file_is_next_to_the_jsonl(self):
        _, _, slog = self._agent()
        from agentica.compression.new_window import notes_path_for
        self.assertEqual(
            Path(notes_path_for(slog)).name,
            "ctx-tool.notes.md",
        )


if __name__ == "__main__":
    unittest.main()
