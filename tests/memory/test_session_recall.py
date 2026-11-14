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
from agentica.memory.recall import RecallBreaker, build_recall
from agentica.memory.session_recall import (
    RECALL_MAX_CHARS,
    RELATED_LIMIT,
    SESSION_RECALL_OPEN,
    draft_session_recall,
    surfaced_markers,
    selector_line,
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


class TestSessionRecallFormatting(_Base):
    def test_selector_line_includes_open_item_next_step(self):
        intent_id = self.index.add_intent(
            work_dir=PROJ,
            session_id="sess-next",
            ask="发布包",
            outcome="tag 已打",
            status="open",
            next_step="twine upload",
        )
        item = self.index.open_intents(PROJ, 0, 5)[0]

        line = selector_line(item, PROJ)

        self.assertIn(f"{intent_id}:", line)
        self.assertIn("Next: twine upload", line)


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
        self.mine = self.index.add_intent(work_dir=PROJ, session_id="prev-session", ask="session_index 改动推送",
                                          outcome="commit 1a2b3c4，未 push", next_step="git push")
        self.vpet = self.index.add_intent(work_dir=OTHER, session_id="vpet", ask="VPetMac TalkConfig 改动推送",
                                          outcome="未 commit", next_step="commit + push")
        self.old = self.index.add_intent(work_dir=PROJ, session_id="prev-session", ask="旧任务", status="open",
                                         now=time.time() - 30 * 86400)

    def recall(self, query, picked_work=(), first_turn=True, context=(), session_id="new-session"):
        """(block, selector request messages or None when no call was made)."""
        draft = asyncio.run(draft_session_recall(
            self.index, None, work_dir=PROJ, session_id=session_id,
            first_turn=first_turn, context=list(context), timeout=1,
        ))
        reply = json.dumps({"files": [], "work": list(picked_work)})
        mock = AsyncMock(return_value=ModelResponse(content=reply))
        with patch.object(OpenAIChat, "response", new=mock):
            _, picked = asyncio.run(build_recall(
                None, _model(), query, list(context), 5, 5.0, RecallBreaker(), session_id,
                work=draft.selector_items(), work_limit=RELATED_LIMIT, work_dir=PROJ,
            ))
        request = mock.await_args.kwargs["messages"] if mock.await_count else None
        return draft.render(picked), request

    def test_first_turn_shows_open_items_and_last_asks_of_this_project_only(self):
        self._seed()
        block, _ = self.recall("git push")
        self.assertTrue(block.startswith(SESSION_RECALL_OPEN))
        self.assertIn("session_index 改动推送", block)
        self.assertIn("Next: git push", block)
        self.assertIn("commit 一下", block)
        self.assertNotIn("VPetMac", block)
        self.assertNotIn("旧任务", block)

    def test_selector_sees_every_project_labeled_and_its_pick_is_rendered(self):
        self._seed()
        block, request = self.recall("TalkConfig 那个改了吗", picked_work=[self.vpet], first_turn=False)
        offered = request[1].content
        self.assertIn(f"{self.vpet}: [/work/VPetMac] [open]", offered)
        self.assertIn(f"{self.mine}: [this project] [open]", offered)
        self.assertIn("Earlier work (current project: /work/agentica)", offered)
        self.assertIn('"work"', request[0].content)
        self.assertIn(f"[intent:{self.vpet}]", block)
        self.assertIn("/work/VPetMac · VPetMac TalkConfig", block)
        self.assertNotIn("session_index 改动推送", block)

    def test_later_turn_recalls_only_what_the_selector_picked(self):
        self._seed()
        block, request = self.recall("17 乘以 23 等于多少", first_turn=False)
        self.assertIsNotNone(request)
        self.assertIsNone(block)

    def test_lines_already_in_context_are_not_offered_again(self):
        self._seed()
        first, _ = self.recall("git push")
        ctx = [Message(role="user", content=f"{first}\n\ngit push")]
        self.assertIn(f"intent:{self.mine}", surfaced_markers(ctx))
        again, request = self.recall("session_index 推送", picked_work=[self.mine], first_turn=False, context=ctx)
        self.assertNotIn(f"{self.mine}: ", request[1].content)
        self.assertIsNone(again)

    def test_stale_open_item_is_labeled_and_has_no_next_step(self):
        self._seed()
        block, _ = self.recall("旧任务", picked_work=[self.old], first_turn=False)
        self.assertIn("[stale", block)
        self.assertNotIn("Open items", block)

    def test_selector_failure_keeps_first_turn_sections(self):
        self._seed()
        draft = asyncio.run(draft_session_recall(
            self.index, None, work_dir=PROJ, session_id="new", first_turn=True, context=[], timeout=1,
        ))
        with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=ConnectionError("down"))):
            _, picked = asyncio.run(build_recall(
                None, _model(), "git push", [], 5, 5.0, RecallBreaker(), "new",
                work=draft.selector_items(), work_limit=RELATED_LIMIT, work_dir=PROJ,
            ))
        block = draft.render(picked)
        self.assertEqual(picked, [])
        self.assertIn("session_index 改动推送", block)

    def test_budget_is_respected_with_whole_lines(self):
        ids = [self.index.add_intent(work_dir=PROJ, session_id="p", ask=f"gateway 任务 {i} " + "x" * 200,
                                     next_step="push") for i in range(40)]
        block, _ = self.recall("gateway push", picked_work=ids[-RELATED_LIMIT:])
        self.assertLessEqual(len(block), RECALL_MAX_CHARS + 200)
        self.assertTrue(block.endswith("</session_recall>"))

    def test_first_turn_drains_stale_batches_from_an_exited_session(self):
        self.index.add_pending("old", PROJ, "release v1.5.1", "tagged, upload pending")
        reply = json.dumps({"new": [{"ask": "release v1.5.1", "outcome": "tag 已打", "status": "open",
                                     "next_step": "twine upload", "turns": [1]}], "update": []})
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=ModelResponse(content=reply))):
            draft = asyncio.run(draft_session_recall(
                self.index, _model(), work_dir=PROJ, session_id="new",
                first_turn=True, context=[], timeout=5,
            ))
        self.assertIn("twine upload", draft.render([]))
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
            stored = [
                m.content for m in b.working_memory.get_messages_from_last_n_runs()
                if m.role == "user" and isinstance(m.content, str)
            ]
            self.assertTrue(stored)
            self.assertNotIn(SESSION_RECALL_OPEN, stored[-1])

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

    def test_one_selector_call_picks_memory_and_cross_project_work(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as proj:
            cfg = WorkspaceMemoryConfig(recall_sessions=True, recall_memories=True)
            a = Agent(model=_model(), session_id="sess-cccc", work_dir=proj, user_id="u1",
                      long_term_memory_config=cfg, enable_long_term_memory=True, workspace=home)
            asyncio.run(a.workspace.write_memory_entry(
                title="prefer uv", content="Use uv, never pip.", memory_type="user"))
            index = a._session_log.search_index
            other = index.add_intent(work_dir="/work/VPetMac", session_id="vpet",
                                     ask="VPetMac 打包签名失败排查", outcome="证书过期，已续签")
            calls, captured = [], {}

            async def answer(model_self, messages, **kw):
                calls.append(messages)
                if len(calls) == 1:
                    return ModelResponse(content=json.dumps(
                        {"files": ["user_prefer_uv.md"], "work": [other]}))
                captured["user"] = [m.content for m in messages if m.role == "user"][-1]
                return ModelResponse(content="ok")

            with patch.object(OpenAIChat, "response", new=answer):
                a.run_sync("这边打包也报签名错误")
            self.assertEqual(len(calls), 2, "one selector call, then the main call")
            self.assertIn(f"{other}: [/work/VPetMac]", calls[0][1].content)
            self.assertIn(f"[intent:{other}]", captured["user"])
            self.assertIn("证书过期", captured["user"])
            self.assertIn("Use uv, never pip.", captured["user"])
            self.assertTrue(captured["user"].rstrip().endswith("这边打包也报签名错误"))


if __name__ == "__main__":
    unittest.main()
