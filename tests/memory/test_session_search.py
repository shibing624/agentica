# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.memory.session_log import SessionLog
from agentica.memory.session_search import search_terms


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


if __name__ == "__main__":
    unittest.main()
