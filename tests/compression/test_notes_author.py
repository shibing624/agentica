# -*- coding: utf-8 -*-
"""The harness writes the session notes itself, from the window it is cutting."""
import asyncio
import os
import tempfile
import unittest
import weakref
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.manager import CompressionManager, apply_idle_compact
from agentica.compression.notes_author import author_session_notes, write_notes_atomically
from agentica.memory.session_log import SessionLog
from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
from agentica.model.response import ModelResponse


class _Scribe(OpenAIChat):
    """A real OpenAIChat whose stream is scripted; records each request."""

    def __init__(self, reply="- Goal: ship ZX-41827\n- Port fixed at 9091", fail=None, **kw):
        super().__init__(id="gpt-4o", api_key="fake_openai_key", **kw)
        self._reply = reply
        self._fail = fail
        self.requests = []
        self.run_tools_seen = []

    async def response_stream(self, messages):
        self.requests.append(list(messages))
        self.run_tools_seen.append(self.run_tools)
        if self._fail is not None:
            raise self._fail
        for piece in (self._reply[: len(self._reply) // 2], self._reply[len(self._reply) // 2:]):
            yield ModelResponse(content=piece)


def _window():
    return [
        Message(role="system", content="You are DeepAgent."),
        Message(role="user", content="工单 ZX-41827：端口固定 9091，不要改 compress.py"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        ),
        Message(role="tool", tool_call_id="c1", content="PORT=9091"),
    ]


class TestAuthorSessionNotes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = str(Path(self._tmp.name) / "s.notes.md")

    def test_request_is_the_window_plus_one_instruction_so_the_cache_prefix_holds(self):
        model = _Scribe()
        window = _window()
        asyncio.run(author_session_notes(window, model, self.path))
        (request,) = model.requests
        self.assertEqual(len(request), len(window) + 1)
        for sent, original in zip(request, window):
            self.assertIs(sent, original, "the cached prefix must be sent as-is")
        self.assertEqual(request[-1].role, "user")
        self.assertIn("Write the session notes file", request[-1].content)
        self.assertEqual(len(window), 4, "the caller's message list is not mutated")

    def test_tools_are_not_run_during_the_pass_and_the_setting_comes_back(self):
        model = _Scribe()
        self.assertTrue(model.run_tools)
        asyncio.run(author_session_notes(_window(), model, self.path))
        self.assertEqual(model.run_tools_seen, [False])
        self.assertTrue(model.run_tools)

    def test_the_setting_comes_back_when_the_request_fails(self):
        model = _Scribe(fail=RuntimeError("gateway 502"))
        self.assertIsNone(asyncio.run(author_session_notes(_window(), model, self.path)))
        self.assertTrue(model.run_tools)

    def test_reply_becomes_the_whole_file(self):
        Path(self.path).write_text("- stale line\n", encoding="utf-8")
        text = asyncio.run(author_session_notes(_window(), _Scribe(), self.path))
        self.assertEqual(text, "- Goal: ship ZX-41827\n- Port fixed at 9091")
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), text + "\n")

    def test_existing_notes_are_handed_over_for_revision(self):
        Path(self.path).write_text("- Constraint: keep the public API\n", encoding="utf-8")
        model = _Scribe()
        asyncio.run(author_session_notes(_window(), model, self.path))
        instruction = model.requests[0][-1].content
        self.assertIn("<current_notes>", instruction)
        self.assertIn("Constraint: keep the public API", instruction)

    def test_first_notes_carry_no_revision_block(self):
        model = _Scribe()
        asyncio.run(author_session_notes(_window(), model, self.path))
        self.assertNotIn("<current_notes>", model.requests[0][-1].content)

    def test_failed_or_empty_reply_leaves_the_file_alone(self):
        Path(self.path).write_text("- keep me\n", encoding="utf-8")
        for model in (_Scribe(fail=RuntimeError("429")), _Scribe(reply="   ")):
            self.assertIsNone(asyncio.run(author_session_notes(_window(), model, self.path)))
            self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "- keep me\n")

    def test_no_notes_path_sends_nothing(self):
        model = _Scribe()
        self.assertIsNone(asyncio.run(author_session_notes(_window(), model, None)))
        self.assertEqual(model.requests, [])

    def test_atomic_write_leaves_no_temp_file(self):
        write_notes_atomically(self.path, "- a\n")
        write_notes_atomically(self.path, "- b\n")
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "- b\n")
        self.assertEqual(sorted(p.name for p in Path(self._tmp.name).iterdir()), ["s.notes.md"])


class TestCutWritesNotes(unittest.TestCase):
    def _setup(self, session_id, model):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        slog = SessionLog(session_id, base_dir=tmp.name)
        agent = type("A", (), {"_session_log": slog})()
        self._agent = agent
        model._agent_ref = weakref.ref(agent)
        return Path(slog.path).with_name(f"{session_id}.notes.md")

    def test_auto_cut_injects_the_notes_it_just_wrote(self):
        model = _Scribe(context_window=128_000)
        notes = self._setup("cut-writes", model)
        cm = CompressionManager()
        msgs = _window() + [Message(role="user", content="继续")]
        asyncio.run(cm.auto_compact(msgs, model=model, force=True, notes_request=msgs))
        self.assertTrue(cm.last_notes_authored)
        self.assertEqual(notes.read_text(encoding="utf-8"), "- Goal: ship ZX-41827\n- Port fixed at 9091\n")
        joined = "\n".join(str(m.content) for m in msgs)
        self.assertIn("<session_notes", joined)
        self.assertIn("Port fixed at 9091", joined)
        self.assertIn("<dropped_span", joined, "the skim still rides along")

    def test_failed_authoring_still_cuts_with_the_digest(self):
        model = _Scribe(fail=RuntimeError("gateway 502"), context_window=128_000)
        notes = self._setup("cut-fails", model)
        cm = CompressionManager()
        msgs = _window() + [Message(role="user", content="继续")]
        self.assertTrue(asyncio.run(cm.auto_compact(msgs, model=model, force=True, notes_request=msgs)))
        self.assertFalse(cm.last_notes_authored)
        self.assertFalse(notes.exists())
        self.assertEqual(cm.window_id, 1)
        self.assertIn("ZX-41827", "\n".join(str(m.content) for m in msgs))

    def test_cuts_that_cannot_use_the_request_do_not_send_it(self):
        """Reactive (prompt_too_long) and pre-fallback cuts call without the flag."""
        model = _Scribe(context_window=128_000)
        self._setup("cut-reactive", model)
        asyncio.run(CompressionManager().auto_compact(_window(), model=model, force=True))
        self.assertEqual(model.requests, [])

    def test_idle_compact_writes_notes(self):
        from agentica.agent import Agent

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = _Scribe(context_window=128_000)
        agent = Agent(
            model=model, session_id="idle-notes", session_base_dir=tmp.name,
            add_history_to_context=True,
        )
        turn = [
            Message(role="user", content="工单 ZX-41827"),
            Message(role="assistant", content="记下了"),
        ]
        agent.working_memory.add_messages(turn)
        agent.working_memory.collapse_runs(turn)
        self.assertTrue(asyncio.run(apply_idle_compact(agent)))
        notes = Path(agent._session_log.path).with_name("idle-notes.notes.md")
        self.assertIn("ZX-41827", notes.read_text(encoding="utf-8"))
        (request,) = model.requests
        # Working memory holds no system prompt; without it the request
        # would share no prefix with the turn before and miss the cache.
        self.assertEqual(request[0].role, "system")
        self.assertEqual(request[0].content, asyncio.run(agent.get_system_message()).content)
        self.assertEqual([(m.role, m.content) for m in request[1:3]],
                         [("user", "工单 ZX-41827"), ("assistant", "记下了")])
        self.assertIn("Write the session notes file", request[3].content)


if __name__ == "__main__":
    unittest.main()
