# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.memory.session_log import SessionLog
from agentica.memory.session_search import (
    USER_QUESTION_BUDGET_CHARS,
    list_user_questions,
    search_terms,
    strip_window_preamble,
)


class TestSearchTerms(unittest.TestCase):
    def test_chinese_bigrams(self):
        terms = search_terms("工单号")
        self.assertIn("工单", terms)
        self.assertIn("单号", terms)
        self.assertIn("工单号", terms)

    def test_drops_stopwords(self):
        terms = search_terms("what dump 什么 ZX-41827")
        self.assertIn("zx-41827", terms)
        self.assertNotIn("what", terms)
        self.assertNotIn("dump", terms)


class TestSearchEntriesRank(unittest.TestCase):
    def test_gongdanhao_hits_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            for i in range(12):
                log.append("user", f"[pad {i:04d}] Unrelated monsoon dump Kalmora")
            log.append("user", "工单 ZX-41827 是这次唯一的追踪号")
            hits = log.search_entries("工单号", limit=5)
        self.assertTrue(hits)
        self.assertIn("ZX-41827", hits[0]["snippet"])

    def test_later_decision_outranks_earlier_same_score_tie_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "缓存用 Redis")
            for i in range(8):
                log.append("user", f"[pad {i:04d}] Unrelated monsoon")
            log.append("user", "改口：缓存改用 KeyDB，不要再用 Redis")
            hits = log.search_entries("KeyDB 缓存", limit=5)
        self.assertTrue(hits)
        self.assertIn("KeyDB", hits[0]["snippet"])

    def test_token_budget_phrase_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "we decided the token budget is 300k")
            hits = log.search_entries("token budget")
        self.assertTrue(hits)
        self.assertIn("300k", hits[0]["snippet"])

    def _compacted_log(self, tmp: str) -> SessionLog:
        log = SessionLog("s", base_dir=tmp)
        log.append(
            "user",
            "所以现在敲、compact就是写note.md文件吗？ 本质也是llm summary吧，"
            "我没特别get到比之前强很大吗?",
        )
        log.append("assistant", "/compact is an empty-window cut, not a note write.")
        log.append("user", "那当前的note.md在哪里？")
        log.append("assistant", "notes_path_for next to the jsonl.")
        log.append_compact_boundary("", window_id=1)
        log.append(
            "user",
            "<context_window>\nCurrent context window 1.\n"
            "You have 512000 tokens left in this context window.\n"
            "</context_window>\n\n",
        )
        log.append("user", "h")
        log.append("user", "hi")
        log.append("user", "前面问了啥")
        return log

    def test_empty_query_is_not_a_keyword_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._compacted_log(tmp)
            self.assertEqual(log.search_entries("", limit=8), [])

    def test_list_user_questions_newest_first_skips_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._compacted_log(tmp)
            hits = log.list_user_questions(limit=8)
        texts = [h["snippet"] for h in hits]
        self.assertTrue(all(h["type"] == "user" for h in hits))
        self.assertIn("前面问了啥", texts)
        self.assertIn("那当前的note.md在哪里？", texts)
        self.assertTrue(any("compact就是写note.md" in t for t in texts))
        self.assertFalse(any("<context_window>" in t for t in texts))
        self.assertLess(texts.index("前面问了啥"), texts.index("那当前的note.md在哪里？"))

    def test_list_user_questions_caps_and_truncates(self):
        rows = [
            {"uuid": str(i), "type": "user", "content": "Q" * 400}
            for i in range(40)
        ]
        hits = list_user_questions(rows, limit=20)
        self.assertLessEqual(len(hits), 20)
        self.assertLessEqual(
            sum(len(h["snippet"]) for h in hits),
            USER_QUESTION_BUDGET_CHARS + 1,
        )
        self.assertIn(" … ", hits[0]["snippet"])

    def test_long_question_keeps_its_closing_ask(self):
        """The ask is at the end; the index must not show only the paste."""
        rows = [{
            "uuid": "1",
            "type": "user",
            "content": ("贴一段无关日志 " * 100) + "前面问了啥，咋办？",
        }]
        snippet = list_user_questions(rows)[0]["snippet"]
        self.assertIn("前面问了啥，咋办？", snippet)
        self.assertIn("贴一段无关日志", snippet)

    def test_role_filters_keyword_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "工单 ZX-41827 是这次唯一的追踪号")
            log.append("assistant", "记下了 ZX-41827")
            user_hits = log.search_entries("ZX-41827", role="user")
            toolish = log.search_entries("ZX-41827", role="assistant")
        self.assertEqual(len(user_hits), 1)
        self.assertEqual(user_hits[0]["type"], "user")
        self.assertEqual(toolish[0]["type"], "assistant")

    def test_search_does_not_hit_dropped_span_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "工单 ZX-41827 是这次唯一的追踪号")
            log.append("assistant", "记下了")
            log.append("user", "现在怎么办？")
            log.append_compact_boundary("", window_id=1)
            log.append(
                "user",
                "<context_window>\nCurrent context window 1.\n"
                "</context_window>\n\n"
                "<dropped_span>\n# Dropped span\n"
                "- user: 工单 ZX-41827 是这次唯一的追踪号\n"
                "- assistant: 记下了\n</dropped_span>\n\n"
                "现在怎么办？",
            )
            ticket = log.search_entries("ZX-41827")
            nxt = log.search_entries("现在怎么办")
            questions = [h["snippet"] for h in log.list_user_questions()]
        self.assertEqual(len(ticket), 1)
        self.assertNotIn("dropped_span", ticket[0]["snippet"])
        # The pending question sits on both sides of the boundary: the original
        # pre-boundary row and the post-boundary row that carries it inside the
        # window preamble. Only the second survives — one hit, not two.
        self.assertEqual(len(nxt), 1)
        self.assertNotIn("<context_window>", nxt[0]["snippet"])
        self.assertTrue(any("现在怎么办" in t for t in questions))
        self.assertFalse(any("<context_window>" in t for t in questions))
        self.assertFalse(any("dropped_span" in t for t in questions))
        self.assertEqual(
            [t for t in questions if "现在怎么办" in t].__len__(), 1,
            "the pending question must be listed once, not twice",
        )

    def test_chrome_only_preamble_is_not_a_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append(
                "user",
                "<context_window>\nCurrent context window 1.\n"
                "</context_window>\n\n"
                "New context window started without a conversation summary. "
                "Continue from session notes and search_session.",
            )
            log.append("user", "hi")
            self.assertEqual(log.search_entries("context window"), [])
            self.assertEqual(
                [h["snippet"] for h in log.list_user_questions()],
                ["hi"],
            )

    def test_strip_window_preamble_keeps_tail_prose(self):
        folded = (
            "<context_window>\nCurrent context window 1.\n"
            "</context_window>\n\n"
            "<dropped_span>\n- user: old\n</dropped_span>\n\n"
            "现在怎么办？"
        )
        self.assertEqual(strip_window_preamble(folded), "现在怎么办？")
        self.assertEqual(strip_window_preamble("plain"), "plain")

    def test_tail_relog_does_not_duplicate_the_pending_question(self):
        """A mid-turn compact re-appends the tail the runner already flushed.

        The write is required — ``load()`` replays only post-boundary rows — so
        the pre-boundary copy must be dropped from search/index instead. Both
        copies used to be scored, listing the pending question twice.
        """
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "旧问题 ZX-41827")
            log.append("assistant", "旧回答")
            # an in-turn flush wrote this turn's row before the window filled
            log.append("user", "现在怎么办？")
            log.append("assistant", "先查 JSONL")
            log.append("tool", "hit evict.py", tool_name="grep", tool_call_id="c1")
            log.append_compact_boundary("", window_id=1)
            log.append(
                "user",
                "<context_window>\nCurrent context window 1.\n"
                "</context_window>\n\n现在怎么办？",
            )
            log.append("assistant", "先查 JSONL")
            log.append("tool", "hit evict.py", tool_name="grep", tool_call_id="c1")

            questions = [h["snippet"] for h in log.list_user_questions()]
            pending = log.search_entries("现在怎么办")
            assistant = log.search_entries("先查 JSONL")
            tool_hit = log.search_entries("hit evict.py")
            old = log.search_entries("ZX-41827")
            prior = log.search_entries("旧回答")
        self.assertEqual(
            len([q for q in questions if "现在怎么办" in q]), 1,
            "the re-logged pending question must be listed once",
        )
        self.assertEqual(len(pending), 1)
        self.assertEqual(len(assistant), 1)
        self.assertEqual(len(tool_hit), 1)
        self.assertEqual(len(old), 1)
        self.assertEqual(len(prior), 1)

    def test_zero_hit_keyword_stays_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s", base_dir=tmp)
            log.append("user", "那当前的note.md在哪里？")
            self.assertEqual(log.search_entries("为什么", limit=5), [])
            self.assertEqual(log.search_entries("ZX-99999", limit=5), [])
            self.assertIn("note.md", log.list_user_questions()[0]["snippet"])


if __name__ == "__main__":
    unittest.main()
