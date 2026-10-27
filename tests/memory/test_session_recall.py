# -*- coding: utf-8 -*-
"""Cross-session index, intent tracking and <session_recall>."""
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent import Agent
from agentica.agent.config import WorkspaceMemoryConfig
from agentica.agent.hooks import IntentTrackHooks
from agentica.memory.intents import parse_update, process_pending_intents
from agentica.memory.session_index import SessionIndex, index_terms
from agentica.memory.session_log import SessionLog
from agentica.memory.session_recall import (
    RECALL_MAX_CHARS,
    SESSION_RECALL_OPEN,
    build_session_recall,
    match_strength,
    surfaced_markers,
)
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.tools.builtin.context_tool import BuiltinContextTool

PROJ = "/work/agentica"
OTHER = "/work/VPetMac"


def _model() -> OpenAIChat:
    return OpenAIChat(id="gpt-4o", api_key="fake_openai_key")


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.index = SessionIndex(os.path.join(self.root, "session_index.db"), projects_dir=self.root)

    def tearDown(self):
        self.index.close()
        self._tmp.cleanup()

    def log(self, sid: str, work_dir: str = PROJ) -> SessionLog:
        return SessionLog(sid, base_dir=os.path.join(self.root, sid[:4]), work_dir=work_dir,
                          search_index=self.index)


class TestSessionIndexTurns(_Base):
    def test_dual_write_skips_automated_and_preamble_only_rows(self):
        log = self.log("s1")
        log.append("user", "部署 gateway 到 9091 端口")
        log.append("user", "job finished", automated=True)
        log.append("user", "<context_window>\nfuel\n</context_window>")
        self.assertEqual(self.index.count_turns(), 1)

    def test_cjk_bigram_search_and_project_filter(self):
        self.log("s1").append("user", "查一下工单 ZX-41827 的状态")
        self.log("s2", OTHER).append("user", "工单系统接入")
        hits = self.index.search_turns("工单号", work_dir=PROJ)
        self.assertEqual([h["session_id"] for h in hits], ["s1"])
        self.assertEqual(len(self.index.search_turns("工单号")), 2)

    def test_exclude_session_and_recent_questions(self):
        a, b, c = self.log("sa"), self.log("sb"), self.log("sc")
        for i in range(4):
            a.append("user", f"a question {i}")
        time.sleep(0.01)
        b.append("user", "b question")
        c.append("user", "current question")
        rows = self.index.recent_sessions_questions(PROJ, "sc", sessions=2, per_session=3)
        self.assertEqual([r["content"] for r in rows],
                         ["b question", "a question 1", "a question 2", "a question 3"])

    def test_backfill_is_incremental_and_dedups_dual_written_rows(self):
        log = SessionLog("bf", base_dir=os.path.join(self.root, "proj"), work_dir=PROJ)
        log.append("user", "first question")
        self.assertEqual(self.index.backfill(), 1)
        self.assertEqual(self.index.backfill(), 0)
        log.append("user", "second question")
        self.assertEqual(self.index.backfill(), 1)
        live = self.log("bf2")
        live.append("user", "dual written")
        self.index.backfill()
        self.assertEqual(self.index.count_turns(), 3)

    def test_index_terms_has_tokens_not_whole_phrase(self):
        terms = index_terms("fix the gateway port").split()
        self.assertIn("gateway", terms)
        self.assertNotIn("fix the gateway port", " ".join(terms).split("  "))


class TestIntents(_Base):
    def test_parse_update_validates_ids_status_and_turns(self):
        text = json.dumps({
            "new": [{"ask": "push agentica", "status": "weird", "turns": [1, 9], "artifacts": ["a"] * 9},
                    {"ask": ""}],
            "update": [{"id": 5, "status": "done", "next_step": "x"}, {"id": 99, "status": "done"}],
        })
        parsed = parse_update(f"noise {text} trailing", {5}, n_turns=2)
        self.assertEqual(len(parsed["new"]), 1)
        self.assertEqual(parsed["new"][0]["status"], "open")
        self.assertEqual(parsed["new"][0]["turns"], [1])
        self.assertEqual(len(parsed["new"][0]["artifacts"]), 5)
        self.assertEqual(parsed["update"], [{"id": 5, "outcome": None, "status": "done", "next_step": "", "turns": []}])
        self.assertIsNone(parse_update("not json", set(), 1))

    def test_process_pending_creates_then_closes_an_item(self):
        log = self.log("s1")
        log.append("user", "改完 recall.py 先别 push")
        self.index.add_pending("s1", PROJ, "改完 recall.py 先别 push", "已提交 476a212b，未 push，要 push 说一声")
        new = json.dumps({"new": [{"ask": "agentica recall.py 改动推送", "outcome": "commit 476a212b，未 push",
                                   "status": "open", "next_step": "git push", "artifacts": ["476a212b"],
                                   "turns": [1]}], "update": []})
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=new))):
            self.assertTrue(asyncio.run(process_pending_intents(self.index, _model(), "s1", 5)))
        items = self.index.open_intents(PROJ, 0, 5)
        self.assertEqual(len(items), 1)
        self.assertEqual(len(items[0]["uuids"]), 1)

        self.index.add_pending("s2", PROJ, "git push", "已推送 476a212b")
        close = json.dumps({"new": [], "update": [{"id": items[0]["id"], "status": "done",
                                                   "outcome": "已推送 476a212b", "turns": [1]}]})
        mock = AsyncMock(return_value=ModelResponse(content=close))
        with patch.object(OpenAIChat, "response", new=mock):
            asyncio.run(process_pending_intents(self.index, _model(), "s2", 5))
        prompt = mock.call_args.kwargs["messages"][1].content
        self.assertIn(f"#{items[0]['id']}: agentica recall.py", prompt)
        self.assertEqual(self.index.open_intents(PROJ, 0, 5), [])
        self.assertEqual(self.index.get_intent(items[0]["id"])["status"], "done")

    def test_failed_call_releases_turns_for_retry(self):
        self.index.add_pending("s1", PROJ, "deploy it", "")
        with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=ConnectionError("down"))):
            self.assertFalse(asyncio.run(process_pending_intents(self.index, _model(), "s1", 5)))
        self.assertEqual(self.index.stale_pending_sessions(PROJ, None), ["s1"])


class TestSessionRecall(_Base):
    def _seed(self):
        prev = self.log("prev-session")
        prev.append("user", "把 session_index.py 的改动 commit 一下")
        self.index.add_intent(work_dir=PROJ, session_id="prev-session", ask="session_index 改动推送",
                              outcome="commit 1a2b3c4，未 push", next_step="git push")
        self.index.add_intent(work_dir=OTHER, session_id="vpet", ask="VPetMac TalkConfig 改动推送",
                              outcome="未 commit", next_step="commit + push")
        self.index.add_intent(work_dir=PROJ, session_id="prev-session", ask="旧任务", status="open",
                              now=time.time() - 30 * 86400)

    def recall(self, query, first_turn=True, context=(), session_id="new-session"):
        return asyncio.run(build_session_recall(
            self.index, None, query=query, work_dir=PROJ, session_id=session_id,
            first_turn=first_turn, context=list(context), timeout=1,
        ))

    def test_first_turn_shows_open_items_and_last_asks_of_this_project_only(self):
        self._seed()
        block = self.recall("git push")
        self.assertTrue(block.startswith(SESSION_RECALL_OPEN))
        self.assertIn("session_index 改动推送", block)
        self.assertIn("Next: git push", block)
        self.assertIn("commit 一下", block)
        self.assertNotIn("VPetMac", block)
        self.assertNotIn("旧任务", block)

    def test_other_project_only_when_named(self):
        self._seed()
        block = self.recall("VPetMac 的 TalkConfig 推了吗", first_turn=False)
        self.assertIn("VPetMac TalkConfig", block)
        self.assertIn("/work/VPetMac · ", block)

    def test_later_turn_without_overlap_recalls_nothing(self):
        self._seed()
        self.assertIsNone(self.recall("17 乘以 23 等于多少", first_turn=False))

    def test_lines_already_in_context_are_not_repeated(self):
        self._seed()
        first = self.recall("git push")
        ctx = [Message(role="user", content=f"{first}\n\ngit push")]
        self.assertTrue(surfaced_markers(ctx))
        again = self.recall("session_index 推送", first_turn=False, context=ctx)
        self.assertIsNone(again)

    def test_stale_open_item_is_labeled_and_has_no_next_step(self):
        self._seed()
        block = self.recall("旧任务", first_turn=False)
        self.assertIn("[stale", block)
        self.assertNotIn("Open items", block)

    def test_match_strength_weights(self):
        self.assertGreaterEqual(match_strength("VPetMac TalkConfig 改动", "VPetMac TalkConfig"), 3)
        self.assertGreaterEqual(match_strength("commit 3f9366a4", "3f9366a4"), 3)
        self.assertLess(match_strength("代码改动推送完成", "改动 推送"), 3)

    def test_budget_is_respected_with_whole_lines(self):
        for i in range(40):
            self.index.add_intent(work_dir=PROJ, session_id="p", ask=f"gateway 任务 {i} " + "x" * 200,
                                  next_step="push")
        block = self.recall("gateway push")
        self.assertLessEqual(len(block), RECALL_MAX_CHARS + 200)
        self.assertTrue(block.endswith("</session_recall>"))

    def test_first_turn_drains_stale_batches_from_an_exited_session(self):
        self.index.add_pending("old", PROJ, "release v1.5.1", "tagged, upload pending")
        reply = json.dumps({"new": [{"ask": "release v1.5.1", "outcome": "tag 已打", "status": "open",
                                     "next_step": "twine upload", "turns": [1]}], "update": []})
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=reply))):
            block = asyncio.run(build_session_recall(
                self.index, _model(), query="继续", work_dir=PROJ, session_id="new",
                first_turn=True, context=[], timeout=5,
            ))
        self.assertIn("twine upload", block)
        self.assertEqual(self.index.stale_pending_sessions(PROJ, "new"), [])


class TestAgentWiring(unittest.TestCase):
    def test_second_session_first_turn_gets_open_item_and_tool_reads_evidence(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as proj:
            cfg = WorkspaceMemoryConfig(recall_sessions=True, track_intents=True, intent_every_n_turns=1,
                                        extract_timeout=5)
            a = Agent(model=_model(), session_id="sess-aaaa", work_dir=proj, user_id="u1",
                      long_term_memory_config=cfg, enable_long_term_memory=True, workspace=home)
            self.assertTrue(any(isinstance(h, IntentTrackHooks)
                                for h in a._default_run_hooks._hooks_list))
            tracker = json.dumps({"new": [{"ask": "发布 v9.9", "outcome": "tag 已打", "status": "open",
                                           "next_step": "twine upload", "turns": [1]}], "update": []})
            replies = [ModelResponse(content="tag 已打，upload 待确认"), ModelResponse(content=tracker)]
            with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=replies)):
                a.run_sync("给 v9.9 打 tag 然后发布")
                asyncio.run(a.flush_pending())
            index = a._session_log.search_index
            self.assertEqual(len(index.open_intents(a._session_log.cwd, 0, 5)), 1)

            b = Agent(model=_model(), session_id="sess-bbbb", work_dir=proj, user_id="u1",
                      long_term_memory_config=WorkspaceMemoryConfig(recall_sessions=True),
                      enable_long_term_memory=True, workspace=home)
            captured = {}

            async def answer(model_self, messages, **kw):
                captured["user"] = [m.content for m in messages if m.role == "user"][-1]
                return ModelResponse(content="ok")

            with patch.object(OpenAIChat, "response", new=answer):
                b.run_sync("继续上次的")
            self.assertIn(SESSION_RECALL_OPEN, captured["user"])
            self.assertIn("twine upload", captured["user"])
            self.assertTrue(captured["user"].rstrip().endswith("继续上次的"))
            self.assertNotIn(SESSION_RECALL_OPEN, Path(b._session_log.path).read_text(encoding="utf-8"))

            tool = BuiltinContextTool()
            tool.set_agent(b)
            listing = asyncio.run(tool.search_session(scope="project"))
            self.assertIn("给 v9.9 打 tag", listing)
            uuid8 = listing.split("[q:")[1][:8]
            full = asyncio.run(tool.search_session(uuid=uuid8))
            self.assertIn("session sess-aaaa", full)
            other = asyncio.run(tool.search_session(session_id="sess-aa", query="v9.9"))
            self.assertIn("给 v9.9 打 tag", other)
            found = asyncio.run(tool.search_session(scope="project", query="v9.9 发布"))
            self.assertIn("[intent:", found)


if __name__ == "__main__":
    unittest.main()
