# -*- coding: utf-8 -*-
"""New-window cut: empty summary, notes excerpt, no consecutive user roles."""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.new_window import (
    dropped_span_excerpt,
    notes_excerpt,
    notes_path_for,
    start_new_context_window,
)
from agentica.model.message import Message


class TestNotesPath(unittest.TestCase):
    def test_sits_next_to_the_jsonl(self):
        slog = type("S", (), {"path": Path("/tmp/sess/abc.jsonl")})()
        self.assertEqual(notes_path_for(slog), "/tmp/sess/abc.notes.md")

    def test_missing_log_is_none(self):
        self.assertIsNone(notes_path_for(None))
        self.assertIsNone(notes_path_for(type("S", (), {"path": None})()))


class TestNotesExcerpt(unittest.TestCase):
    def test_missing_or_empty_file_is_none(self):
        self.assertIsNone(notes_excerpt(None))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            self.assertIsNone(notes_excerpt(path))
            Path(path).write_text("  \n", encoding="utf-8")
            self.assertIsNone(notes_excerpt(path))

    def test_injects_body_so_the_first_hop_is_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text("Goal: finish auth\n", encoding="utf-8")
            excerpt = notes_excerpt(path)
        self.assertIn("Goal: finish auth", excerpt)
        self.assertIn(path, excerpt)

    def test_long_file_keeps_both_ends(self):
        """Over the cap, keep the tail too — the newest state is written last."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text(
                "Goal: finish auth\n" + ("padding\n" * 700) + "Pending: 订正行号\n",
                encoding="utf-8",
            )
            excerpt = notes_excerpt(path)
        self.assertIn("Goal: finish auth", excerpt)
        self.assertIn("Pending: 订正行号", excerpt)


class TestDroppedSpanExcerpt(unittest.TestCase):
    def test_long_digest_keeps_the_newest_turns(self):
        text = (
            "# Dropped span\n"
            + ("- assistant: 旧结论\n" * 800)
            + "- assistant: 现在怎么办？\n"
        )
        excerpt = dropped_span_excerpt(text)
        self.assertIn("现在怎么办？", excerpt)


class TestStartNewContextWindow(unittest.TestCase):
    def test_auto_compact_keeps_system_and_pending_question(self):
        msgs = [
            Message(role="system", content="you are helpful"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ]
        start_new_context_window(msgs, window_id=1, tokens_left=9000)
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "you are helpful")
        self.assertEqual(msgs[-1].role, "user")
        self.assertIn("<context_window>", msgs[-1].content)
        self.assertIn("current question", msgs[-1].content)
        self.assertNotIn("old answer", " ".join(str(m.content) for m in msgs))
        roles = [m.role for m in msgs]
        self.assertNotIn(("user", "user"), list(zip(roles, roles[1:])))

    def test_dropping_the_trailing_turn_leaves_a_blank_page(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="current question"),
        ]
        start_new_context_window(
            msgs, window_id=2, tokens_left=8000, keep_trailing_turn=False,
        )
        joined = " ".join(str(m.content) for m in msgs)
        self.assertNotIn("current question", joined)
        self.assertIn("New context window started", joined)
        self.assertEqual([m.role for m in msgs], ["system", "user"])

    def test_notes_are_folded_into_the_same_user_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = str(Path(tmp) / "s.notes.md")
            Path(notes).write_text("Constraint: do not rewrite auth\n", encoding="utf-8")
            msgs = [Message(role="user", content="keep going")]
            start_new_context_window(
                msgs, window_id=1, tokens_left=1000, notes_path=notes,
            )
        self.assertEqual(len(msgs), 1)
        self.assertIn("Constraint: do not rewrite auth", msgs[0].content)
        self.assertIn("<session_notes", msgs[0].content)
        self.assertNotIn("<dropped_span", msgs[0].content)
        self.assertIn("keep going", msgs[0].content)

    def test_dropped_span_is_not_wrapped_as_session_notes(self):
        msgs = [Message(role="user", content="keep going")]
        start_new_context_window(
            msgs,
            window_id=1,
            tokens_left=1000,
            notes_path="/tmp/s.notes.md",
            dropped_span="# Dropped span\n- user: 工单 ZX-41827",
        )
        self.assertIn("<dropped_span", msgs[0].content)
        self.assertIn("ZX-41827", msgs[0].content)
        self.assertNotIn("<session_notes", msgs[0].content)


class TestRunnerNewWindow(unittest.TestCase):
    def _agent(self, session_id=None):
        """An agent that can write notes, unless ``session_id`` is None.

        Without a session_id there is no notes path, so the runner has nothing
        to ask the model to write — see the two postpone tests below.
        """
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
        if session_id is None:
            return Agent(model=model, enable_session_log=False)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return Agent(
            model=model,
            session_id=session_id,
            session_base_dir=self._tmp.name,
        )

    def test_empty_notes_postpone_compact_once(self):
        from agentica.runner import Runner

        # With a session_id the notes file has a path, so asking for it is
        # actionable and the cut is postponed once.
        agent = self._agent(session_id="postpone-once")
        cm = agent.tool_config.compression_manager
        agent.model.functions = {"write_file": object(), "read_file": object()}
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="current question"),
        ]
        postponed = Runner._claim_notes_fallback(
            msgs, agent, agent.model, cm,
            context_tokens=960, working_window=1000,
        )
        self.assertTrue(postponed)
        self.assertTrue(cm.fallback_claimed)
        self.assertEqual(cm.compact_token_floor, 1000)
        self.assertEqual(cm.window_id, 0)
        self.assertIn("about to reset", msgs[-1].content)
        self.assertIn("current question", msgs[-1].content)
        agent.model.context_window = 1000
        self.assertFalse(cm.should_auto_compact(
            msgs, agent.model, context_tokens=960,
        ))
        self.assertTrue(cm.should_auto_compact(
            msgs, agent.model, context_tokens=1000,
        ))

        asyncio.run(cm.auto_compact(msgs, model=agent.model, force=True))
        self.assertEqual(cm.window_id, 1)
        self.assertFalse(cm.fallback_claimed)
        self.assertIsNone(cm.compact_token_floor)

    def test_no_session_id_does_not_postpone_for_an_unwritable_file(self):
        """No notes path => nothing to ask for => do not postpone the cut.

        This is the SDK shape: tools exist but there is no session_id, so
        ``notes_path_for`` returns None. The runner used to postpone anyway,
        telling the model to write a file it could not name — and the delay
        bought nothing.
        """
        from agentica.runner import Runner

        agent = self._agent(session_id=None)
        cm = agent.tool_config.compression_manager
        agent.model.functions = {"write_file": object(), "read_file": object()}
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="current question"),
        ]
        self.assertIsNone(notes_path_for(agent._session_log))
        postponed = Runner._claim_notes_fallback(
            msgs, agent, agent.model, cm,
            context_tokens=960, working_window=1000,
        )
        self.assertFalse(postponed)
        self.assertFalse(cm.fallback_claimed)
        self.assertIsNone(cm.compact_token_floor)
        self.assertEqual(len(msgs), 2, "no nudge should be folded in")
        self.assertNotIn("about to reset", msgs[-1].content)

    def test_reminder_folds_once_into_the_last_user_message(self):
        from agentica.runner import Runner

        agent = self._agent()
        cm = agent.tool_config.compression_manager
        msgs = [Message(role="user", content="hello")]
        Runner._maybe_inject_token_budget_reminder(
            msgs, agent, cm, context_tokens=800, working_window=1000,
        )
        self.assertTrue(cm.reminder_claimed)
        self.assertIn("<context_window>", msgs[0].content)
        self.assertIn("hello", msgs[0].content)
        before = msgs[0].content
        Runner._maybe_inject_token_budget_reminder(
            msgs, agent, cm, context_tokens=900, working_window=1000,
        )
        self.assertEqual(msgs[0].content, before)


if __name__ == "__main__":
    unittest.main()
